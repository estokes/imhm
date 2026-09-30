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

- **Node** (`src/node.rs`): one reference-counted allocation. It holds
  a header (count, `len` = pairs in the subtree, `depth`, a 32-bit
  `bitmap`, `cap`) followed by room for `cap` ≤ 32 slots, one per set
  bit, in bit order. The slot count is `bitmap.count_ones()`, so it
  can't disagree with the bitmap.
  - A lookup makes one dependent memory load per level.
  - A node no other version holds gains or loses a slot in place. When
    it's full, it moves to an allocation twice the size (up to 32).
  - A copy of a shared node is exactly its slot count.
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
- **Hasher:** any `BuildHasher`, per map; nohash for integer keys and
  ahash for strings both work, and both are in the model tests. With
  nohash the mix above matters: sequential ids otherwise leave the high
  bits empty.
  - The default is `FxBuildHasher`.
  - A map that `shared_map.rs` serializes must hash the same way where
    the image is decoded. That means fixed seeds (ahash `with_seeds`,
    never `RandomState::new`). It also means the same hasher version and
    build: ahash's output depends on both its version and whether the
    build uses the CPU's AES instructions.
  - A mismatch fails safely, because `NodeHandle::create` recomputes
    every hash and rejects the node. But an image written by one binary
    may not load in another, so the image format should record which
    hasher it used.
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

## Unsafe code

All of it is in `src/node.rs`, behind a safe API: `new`, `slots`,
`make_mut`, `insert_slot`, `remove_slot`, and `NodeMut`.
- **Reference counting:** the same protocol as `std::sync::Arc`
  (relaxed increment; release decrement, then an acquire fence before
  the last owner drops the slots and frees the memory).
- **Changing a node in place:** only through `&mut` while the count is
  1.
- **Building:** a drop guard counts the slots written, so if cloning a
  key panics partway through a copy, what was written is dropped and the
  empty node frees itself.
- **Moving slots:** in-place insert and remove shift slots with
  `ptr::copy`; growing moves them to the larger allocation and frees the
  old one without dropping them. No user code runs between a move and
  the bitmap update that records it.

The earlier `triomphe::HeaderSlice` version of this code passed the full
test suite under Miri. `node.rs` has not been run under Miri yet.

## Performance (ns/op, best of 5; imhm / imbl / chunk16)

| keys, N | in-place insert | get | snapshot+insert | snapshot+remove |
|---|---|---|---|---|
| u64, 10k | 53 / 28 / 127 | 13 / 4 / 23 | 790 / 756 / 824 | 755 / 749 / 645 |
| CompactString, 10k | 71 / 48 / 387 | 28 / 13 / 167 | 973 / 860 / 1242 | 853 / 831 / 955 |
| ArcStr path, 10k | 56 / 40 / 385 | 17 / 8 / 145 | 874 / 869 / 1276 | 831 / 819 / 1023 |
| u64, 100k | 69 / 48 / 202 | 18 / 9 / 66 | 1091 / 1115 / 1234 | 1056 / 1123 / 909 |
| ArcStr path, 100k | 77 / 62 / 578 | 24 / 17 / 255 | 1192 / 1201 / 1913 | 1202 / 1206 / 1531 |
| ArcStr path, 1M | 155 / 116 / 1286 | 103 / 67 / 1002 | 1666 / 2068 / 3945 | 1846 / 2160 / 2716 |

Against chunk16:
- string-key lookups are 5–10× faster,
- in-place inserts are 2–8× faster,
- snapshot updates are even to 2× faster, except u64 removes, which are
  about 15% slower.

Against imbl:
- snapshot updates are even, and faster at 1M,
- lookups are about 1.5× slower (3× for u64 keys at 10k),
- in-place inserts are about 1.2–1.8× slower.

Compared with the earlier `Vec` node, which made two dependent loads per
level: lookups are 15–35% faster at 100k and up, and in-place inserts
are as fast or faster.

## Open decisions

**Depth.** The remaining lookup gap to imbl is mostly depth. imbl 7's
bottom nodes are flat SIMD buckets of up to 32 entries, which saves
about a level. A canonical variant is possible: a subtree with at most
B entries is a leaf bucket, sorted by hash, and a larger one is a trie
node. That rule depends only on the key set, so the shape stays
canonical. It would change the node API that `shared_map.rs` ports to,
so it's worth doing only if environment lookups show up in a compile
profile.
