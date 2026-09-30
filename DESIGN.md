# imhm design

A persistent hash map and set meant to stand in for immutable-chunkmap in
the graphix compiler wherever key order is not needed. Clones are O(1). An
update copies the path from the root to the key and shares the rest. A
map that no other version shares is updated in place.

## Why not chunkmap, why not imbl

Measured on 2026-09-29 against chunkmap at graphix's `CHUNK = 16` with
the `pool` feature on. Keys are shaped like graphix's: `BindId` → u64,
names → `CompactString`, and `ModPath` → `ArcStr` paths with shared
prefixes. The benchmark is `examples/bench.rs`.

- With string keys, chunkmap lookups cost one string compare per step of
  the ordered search. A hash map does about one compare per lookup. The
  node layout changes nothing here: imbl's `OrdMap` (a B+tree) was as
  slow as chunkmap.
- imbl's `HashMap` has the lookup speed but doesn't expose its nodes.
  `graphix-compiler/src/shared_map.rs` serializes the environment while
  preserving sharing, and it walks and rebuilds nodes to do so. imhm
  provides that node API (see below).
- The parallel compile join (`graphix-compiler/src/tracked.rs`) replays
  each written key with insert/remove and doesn't use chunkmap's `union`.
  Any map type works there.

## Structure

A hash array mapped trie (HAMT). Each level consumes 5 bits of the hash,
low bits first. Levels 0 to 12 exist; level 12 uses the last 4 bits.

- **Node:** `triomphe::Arc<Node>`. The node holds `len` (pairs in the
  subtree), `depth`, a 32-bit `bitmap`, and `Vec<Slot>` with one slot
  per set bit, in bit order.
- **Slot:** one of
  - an entry `(hash, key, val)`,
  - a collision (two or more distinct keys whose full hashes are
    equal, behind an `Arc`),
  - a child node.
- **Canonical shape:** below the root, a node is never a lone entry or
  collision. Removal collapses such a node into its parent. So a key set
  has exactly one shape for a given hasher, whatever the insert and
  remove history. Iteration follows the shape, so it's deterministic
  when the hasher is. The one exception is the order of pairs inside a
  collision.
- **Hash mix:** the hasher's output passes through
  `h * φ64 ^ (h*φ64 >> 32)`. It spreads entropy into the bits the trie
  reads first, so a weak hasher can't make lopsided trees. It's a
  bijection, so it creates no collisions. It's also invertible: the
  tests use that to put keys at chosen places in the tree (deep chains,
  full-hash collisions).
- **Default hasher:** `FxBuildHasher`, fixed seed. The shape must be the
  same in the process that encodes an image and the one that decodes
  it, which rules out `RandomState` for serialized maps.
- **Stored hash:** each entry keeps its mixed hash. Pushing an entry
  down a level needs no rehash, and a lookup compares hashes before
  keys.

## Node API

`Map::root()` → `NodeRef`, which provides `identity`, `keep`, `depth`,
`subtree_len`, and `slots` (yielding `SlotRef::{Entry, Collision, Node}`).
Rebuilding goes through `NodeHandle::create(hasher, depth, slots)` and
then `Map::from_root`.

chunkmap's `NodeHandle::create` is `unsafe` and trusts the caller.
imhm's is safe: it recomputes every key's hash and rejects any node the
map couldn't have built:
- wrong fragment order or position,
- slots whose prefixes above the node disagree,
- a child that isn't exactly one level down,
- a bad collision,
- a lone entry below the root,
- a root not at depth 0.

A node's position in any map is determined by its contents, so a node
shared by several maps is checked once, when it's created.

## Performance (ns/op, best of 5; imhm / imbl / chunk16)

| keys, N | in-place insert | get | snapshot+insert | snapshot+remove |
|---|---|---|---|---|
| u64, 10k | 50 / 28 / 118 | 13 / 4 / 24 | 776 / 761 / 801 | 709 / 746 / 634 |
| CompactString, 10k | 70 / 46 / 341 | 25 / 12 / 154 | 875 / 845 / 1201 | 837 / 849 / 954 |
| ArcStr path, 10k | 56 / 38 / 361 | 17 / 8 / 130 | 825 / 838 / 1266 | 766 / 821 / 1013 |
| ArcStr path, 100k | 81 / 57 / 548 | 29 / 19 / 242 | 1174 / 1183 / 1906 | 1144 / 1189 / 1535 |
| ArcStr path, 1M | 214 / 118 / 1320 | 129 / 68 / 1014 | 1654 / 1934 / 3426 | 1673 / 1802 / 2310 |

Against chunk16, string-key lookups are 5–8× faster, in-place inserts
4–7× faster, and snapshot updates up to 2× faster. Against imbl, snapshot
updates are even, and lookups and in-place inserts are 1.5–3× slower.

## Open decisions

Where the gap to imbl comes from, measured with `perf` on a 10k-entry
u64 lookup loop:

1. **Two dependent loads per level.** One reads the `Arc<Node>` header,
   then another reads the slot in the `Vec`'s separate buffer. Most of
   the samples sat on those two loads. A single-allocation node
   (`triomphe::HeaderSlice`) removes one of them. I built that variant
   (kept in the session scratchpad, not in the tree) with safe code only.
   Changing a node's slot count then means cloning its other slots, even
   when nothing else holds the node:
   - lookups got 10–20% faster,
   - in-place builds got 2–3× slower,
   - snapshot updates improved for u64/`ArcStr` keys but worsened for
     `CompactString`.

   Moving the slots out of a node that nothing else holds would fix the
   build cost. It needs a small `unsafe` core: `ManuallyDrop` slots,
   `ptr::read` when the node is unique, and a `Drop` that uses
   `Arc::into_unique` so the last owner drops the slots without a race.
2. **Depth.** imbl 7's bottom nodes are flat SIMD buckets of up to 32
   entries, which saves about a level. A canonical variant is possible:
   bucket leaves sorted by hash, split once they exceed 32. It changes
   the node API that `shared_map.rs` would port to.

Neither is needed to beat chunkmap. Both are worth doing only if env
lookups show up in a compile profile.
