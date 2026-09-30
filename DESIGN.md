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

A hash trie with two kinds of node. Inner nodes branch on 5 bits of the
hash per level, **high bits first**, so a tree's order is its hashes'
numeric order. Inner nodes exist at levels 0 to 12; level 12 uses the
last 4 bits.

- **Canonical rule.** For the pairs under a slot:
  - one pair sits inline in the parent's slot as an entry;
  - 2 to `LEAF` (32) pairs, or any number that all share one full hash,
    form a **leaf**: a flat node of entries sorted by hash;
  - more than that forms an **inner** node.

  The rule depends only on the key set, so a key set has exactly one
  shape for a given hasher, whatever the insert and remove history. The
  one exception is the order among keys with the same full hash.
  Iteration is in hash order, so it's deterministic when the hasher is.
  Keys with the same full hash sit side by side in a leaf, so there's no
  separate collision type.
- **Inner node:** a header (`len` = pairs in the subtree, `depth`, a
  32-bit `bitmap`) and one slot per set bit, in bit order. A slot is an
  inline entry `(hash, key, val)` or a child node.
- **Leaf:** a header (`depth`, and `tags`: the low byte of each of the
  first 32 hashes) and its entries. A lookup compares all 32 tags in one
  SIMD instruction (SSE2 on x86_64, 8 at a time with integer arithmetic
  elsewhere). That usually leaves one candidate, confirmed by its full
  hash and key. So the lookup has no data-dependent branch until the
  final key compare.
- **Changes.** An insert that fills a leaf past `LEAF` rebuilds it as
  the canonical subtree of its entries. A removal that brings an inner
  node to `LEAF` or fewer pairs, or down to a single leaf, gathers its
  entries into a leaf. Both move the entries when no other version
  holds the node.
- **Storage** (`src/node.rs`): both node kinds are one reference-counted
  allocation holding a header and room for `cap` items. A lookup makes
  one dependent memory load per level. A node no other version holds
  gains or loses an item in place, moving to an allocation twice the
  size when full. A copy of a shared node is exactly its item count.
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
  - A mismatch fails safely, because `NodeHandle::inner`/`leaf` recompute
    every hash and rejects the node. But an image written by one binary
    may not load in another, so the image format should record which
    hasher it used.
- **Stored hash:** each entry keeps its mixed hash. Pushing an entry
  down a level needs no rehash, and a lookup compares hashes before
  keys.

## Node API

`Map::root()` → `NodeRef`, which provides `identity`, `keep`, `depth`,
`subtree_len`, and `contents`: either `Contents::Inner` (slots,
`SlotRef::{Entry, Node}`) or `Contents::Leaf` (pairs). Rebuilding goes
through `NodeHandle::inner(hasher, depth, slots)` and
`NodeHandle::leaf(hasher, depth, pairs)`, then `Map::from_root`.

chunkmap's `NodeHandle::create` is `unsafe` and trusts the caller.
imhm's constructors are safe: they recompute every key's hash and
reject any node the map couldn't have built:
- wrong fragment or hash order,
- a prefix above the node that its contents don't share,
- a child that isn't exactly one level down,
- a leaf of one pair below the root,
- a leaf of more than `LEAF` pairs that don't all share one hash,
- a repeated key,
- an inner node of `LEAF` or fewer pairs, or of one leaf alone,
- a root not at depth 0.

A node's position in any map is determined by its contents, so a node
shared by several maps is checked once, when it's created.

## Unsafe code

`src/node.rs`, behind a safe API: `Raw::{new, items, make_mut, insert,
remove, take_items}` and `RawMut`.
- **Reference counting:** the same protocol as `std::sync::Arc`
  (relaxed increment; release decrement, then an acquire fence before
  the last owner drops the items and frees the memory).
- **Changing a node in place:** only through `&mut` while the count is
  1.
- **Building:** a drop guard counts the items written, so if cloning a
  key panics partway through a copy, what was written is dropped and the
  empty node frees itself.
- **Moving items:** in-place insert and remove shift items with
  `ptr::copy`; growing moves them to the larger allocation and frees the
  old one without dropping them; `take_items` moves them into a `Vec`
  and leaves the node empty. No user code runs between a move and the
  count update that records it.

`lib.rs` has one more `unsafe` block: the SSE2 tag compare. The tests
include threads sharing, changing and dropping clones of one map.

## Performance

Measured with `perf stat` pinned to one performance core (CPU 2), per
operation, as the difference between two round counts so map building
drops out; the harness is `examples/getcount.rs`. Instructions / cycles;
"prev" is the version without leaves (`main`, 9e1235d).

| operation | keys, N | imhm leaves | imhm prev | imbl | chunk16 |
|---|---|---|---|---|---|
| lookup | u64, 10k | 110 / 34 | 105 / 51 | 79 / 19 | 110 / 68 |
| lookup | u64, 1M | 140 / 284 | 132 / 323 | 104 / 207 | 149 / 612 |
| lookup | ArcStr, 10k | 185 / 52 | 149 / 67 | 152 / 40 | 770 / 483 |
| lookup | ArcStr, 1M | 213 / 345 | 177 / 413 | 172 / 345 | 1136 / 3646 |
| in-place insert | u64, 10k | 533 / 221 | 505 / 204 | 280 / 120 | 1702 / 525 |
| in-place insert | ArcStr, 1M | 894 / 735 | 735 / 718 | 567 / 511 | 5772 / 5348 |
| snapshot+insert | u64, 10k | 4070 / 3026 | 4320 / 3550 | 4980 / 3418 | 7005 / 3460 |
| snapshot+insert | ArcStr, 1M | 8631 / 6490 | 7034 / 6216 | 9161 / 6741 | 14716 / 8997 |

Leaves against prev:
- **Lookups** take 15–33% fewer cycles at 10k and 1M and about the same
  at 1k and 100k, because branch misses fall several-fold below 1M. But
  they cost 5–24% more instructions.
- **In-place inserts** cost 6–27% more instructions and up to 25% more
  cycles.
- **Snapshot inserts** are slightly cheaper at 10k–100k and cost more
  instructions at 1M.

Against chunk16, string-key lookups take 4–7× fewer instructions and
3–11× fewer cycles. imbl still leads on lookups and in-place inserts.

## Open decisions

**Keep the leaves?** They buy lookup cycles with instructions, and make
in-place builds dearer; `main` has the version without them. The leaf
lookup's fixed cost (tag compare, candidate mask, bounds check) is about
15–35 instructions more than an inline entry's hash-and-key compare.
