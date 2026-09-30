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
    form a **leaf**;
  - more than that forms an **inner** node.

  The rule depends only on the key set, so a key set has exactly one
  shape for a given hasher, whatever the insert and remove history. The
  one exception is the order among keys with the same full hash.
  Iteration is in hash order, so it's deterministic when the hasher is.
  Keys with the same full hash share a leaf, so there's no separate
  collision type.
- **Inner node:** a header (`len` = pairs in the subtree, `depth`, and a
  32-bit `bitmap` of the slots in use) and 32 slots, one per fragment. A
  slot is empty, an inline entry `(hash, key, val)`, or a child node. A
  lookup indexes the slot straight from the hash: no popcount, and the
  bitmap isn't read. An inner node holds more than 32 pairs, so its
  slots are mostly full; a compressed node would be nearly as big.
- **Leaf:** a header and the entries, in the order they were added.
  - `tags`: one byte per entry for the first 32, the hash's byte just
    below the leaf's prefix (0 is raised to 1). Unused lanes are 0, so
    they never match. Tags sort the same way their hashes do.
  - `order`: the entries' indices in hash order. Iteration, splitting
    and the node API follow it. Unused lanes are `0xFF`.
  - **Lookup:** compare the 32 tags with the key's tag, a few SIMD
    instructions, then check the one or two candidates by hash and key.
  - **Insert:** push the entry at the end. Its place in `order` is the
    number of smaller tags, plus a full-hash compare for equal tags, and
    `order` shifts by one lane from there. **Remove:** move the last
    entry into the hole and shift `order` back. All of these are
    fixed-width 32-byte operations with no data-dependent branch.
  - A new leaf of two has room for 8; a leaf in place grows by doubling.
    A copy for another version is exactly its size.
- **Lane operations** (`src/lanes.rs`): equal, less-than, insert-at and
  remove-at on 32 bytes. SSE2 on x86_64 and NEON on aarch64, both in
  their architecture's baseline and on stable Rust; plain Rust elsewhere.
  Each architecture keeps its compare results in the mask layout it
  makes most cheaply.
- **Changes.** An insert that fills a leaf past `LEAF` rebuilds it as
  the canonical subtree of its entries. A removal that brings an inner
  node to `LEAF` or fewer pairs, or down to a single leaf, gathers its
  entries into a leaf. Both move the entries when no other version
  holds the node.
- **Storage** (`src/node.rs`): both node kinds are one reference-counted
  allocation holding a header and room for `cap` items.
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
  keys, unless the key is small and has no destructor, when comparing
  it costs about the same.

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

## Panics

If a key's or value's `Clone` or `Drop` panics during an operation, the
map stays whole: its invariants hold, it holds what it held before or
after the operation, other versions are untouched, and nothing leaks.
- **Copies before moves.** A change copies each shared node it needs
  before it moves anything. A removal that will collapse a subtree
  first copies every shared node in it (`unshare`), so the collapse
  only moves entries.
- **Drops last.** A removed key, and a key made redundant by an insert,
  is dropped only once the tree is whole again, and before the return
  value exists: a destructor that panics while a function returns leaks
  that value (rust-lang/rust#47949).

## Unsafe code

`src/node.rs`, behind a safe API: `Raw::{new, items, make_mut, push,
swap_remove, take_items}` and `RawMut`.
- **Reference counting:** the same protocol as `std::sync::Arc`
  (relaxed increment; release decrement, then an acquire fence before
  the last owner drops the items and frees the memory).
- **Changing a node in place:** only through `&mut` while the count is
  1.
- **Building:** a drop guard counts the items written, so if cloning a
  key panics partway through a copy, what was written is dropped and the
  empty node frees itself.
- **Dropping:** a guard frees the allocation even if an item's drop
  panics. Capacity is checked to fit the header's `u32` before
  allocating.
- **Moving items:** `swap_remove` moves the last item into the hole;
  growing moves the items to a larger allocation and frees the old one
  without dropping them; `take_items` moves them into a `Vec` and
  leaves the node empty. No user code runs between a move and the count
  update that records it.

`src/lanes.rs` has the SSE2 and NEON intrinsics, each on the 32 bytes
of one array. `lib.rs` has one unchecked index, of the entry a leaf
lookup just found. The tests include threads sharing, changing and dropping clones
of one map. The whole suite passes under Miri on x86_64; the lane and
leaf tests pass under Miri for aarch64, RISC-V and s390x (big-endian),
and the whole suite passes natively on an Apple M2.

## Testing

`src/test.rs` has hand-built cases: hashers that place keys at chosen
points of the tree (deep chains, full-hash collisions), shape,
sharing, the node API, threads, and the lane operations against plain
references.

`src/test/model.rs` is the model checker:
- **What varies:**
  - Types: integers, random `ArcStr` and `CompactString` (empty,
    inline, heap, any Unicode), pairs, maps as values, and `Tracked`,
    which counts itself so every clone is shown dropped exactly once.
  - Hashers: Fx, ahash, nohash, and three hostile ones: `Deep` shares
    all but 12 bits (deep chains), `Few<6>` and `Few<3>` allow 64 or 8
    distinct hashes, and `Few<0>` gives every key one hash.
  - Sizes: 1, 2, the edges of a leaf, up to 20k, and 500k in three runs.
- **Each run:** up to 8 versions, each with a `HashMap` of what it
  should hold. The run grows, churns and shrinks, with every mutating
  call made both in place and on old versions. The touched version's
  invariants are checked every `size/32` steps and all versions every
  `size/2`. At the end, each version must equal the same key set built
  in a shuffled order (same shape, same iteration order) and the same
  tree rebuilt through the node API. Then the last version is emptied
  in hash order, which passes through every kind of collapse.
- **Also:** a set model, and a panic model: keys and values whose
  `Clone` or `Drop` panics at a random point in any operation. After
  each panic the map must pass the invariant check and hold exactly
  what it held before or after the operation, other versions must be
  untouched, and nothing may leak.
- **Seeds:** random per run; a failure prints `IMHM_SEED` to replay it.
- **Speed:** full size in release (2–4 minutes on 16 threads), a
  tenth in debug, a hundredth under Miri, which skips the 500k runs.

To check that the suite finds bugs, ten plausible ones were planted
one at a time, and it caught all ten. For example: a stale tag or
`order` lane, a lost `len` update, a missing collapse, a shared node
changed in place, a split that ignores `order`.

Miri reports leaks as well as undefined behavior. `scripts/valgrind.sh`
runs every test natively under memcheck, at a tenth of full size, with
definite and indirect leaks as errors.

## Performance

ns per operation, wall clock: `examples/getcount.rs`, driven by
`scripts/wall.sh`, pinned to one performance core (CPU 2) at
`SCHED_FIFO` 50. Each number is the difference between the fastest of 7
runs at two round counts, so building the map drops out; the builds
take turns. Keys are u64 or `ArcStr` paths; the hasher is Fx.
- **lookup:** every key once.
- **in-place insert:** build the map from empty in one version.
- **snapshot+insert:** 100 new keys, each into a new version of the last.

"start" is imhm as of `0c3d573`: compressed inner nodes and sorted
leaves.

| operation | keys | N | imhm | start | imbl | chunk16 |
|---|---|---|---|---|---|---|
| lookup | u64 | 1k | 3.0 | 4.3 | 3.5 | 4.7 |
| lookup | u64 | 10k | 4.2 | 6.8 | 4.2 | 14.2 |
| lookup | u64 | 100k | 9.3 | 15.0 | 9.1 | 53.7 |
| lookup | u64 | 1M | 42.0 | 59.8 | 48.7 | 129.6 |
| lookup | str | 1k | 6.1 | 8.2 | 9.9 | 39.9 |
| lookup | str | 10k | 8.0 | 11.2 | 8.8 | 105.1 |
| lookup | str | 100k | 14.1 | 20.2 | 15.0 | 184.0 |
| lookup | str | 1M | 56.5 | 73.5 | 75.7 | 800.0 |
| in-place insert | u64 | 1k | 18.9 | 34.5 | 28.7 | 69.4 |
| in-place insert | u64 | 10k | 23.2 | 51.7 | 25.3 | 120.1 |
| in-place insert | u64 | 100k | 45.7 | 75.9 | 42.5 | 190.1 |
| in-place insert | u64 | 1M | 63.3 | 133.8 | 83.0 | 296.4 |
| in-place insert | str | 1k | 25.8 | 41.8 | 39.8 | 202.0 |
| in-place insert | str | 10k | 31.5 | 62.4 | 35.1 | 331.9 |
| in-place insert | str | 100k | 59.1 | 91.0 | 59.1 | 506.4 |
| in-place insert | str | 1M | 90.3 | 159.6 | 110.3 | 1175.2 |
| snapshot+insert | u64 | 1k | 420 | 461 | 543 | 492 |
| snapshot+insert | u64 | 10k | 638 | 637 | 735 | 741 |
| snapshot+insert | u64 | 100k | 988 | 941 | 1021 | 1007 |
| snapshot+insert | u64 | 1M | 1092 | 1122 | 1286 | 1340 |
| snapshot+insert | str | 1k | 642 | 639 | 643 | 775 |
| snapshot+insert | str | 10k | 806 | 768 | 834 | 1114 |
| snapshot+insert | str | 100k | 1082 | 1032 | 1104 | 1466 |
| snapshot+insert | str | 1M | 1409 | 1403 | 1465 | 1910 |

Against imbl, imhm is:
- **Lookups:** 14–38% faster at 1k and 1M, and level or up to 2%
  slower at 10k and 100k.
- **In-place inserts:** 18–35% faster at 1k and 1M, 8–10% faster at
  10k, and level to 8% slower at 100k.
- **Snapshot inserts:** 0–23% faster at every size.

Against "start", snapshot inserts are between 9% faster and 5% slower.

What closed the gap with imbl, in order of effect:
- **Inner slots indexed by fragment.** The popcount and the bitmap load
  were a dependent chain of about 10 cycles per level.
- **Leaves in insertion order**, with `order` and tags that sort like
  hashes. A sorted leaf shifted entries and tags on every insert, and
  the variable-length shifts mispredicted.
- **The fragment kept in a register**, not loaded from each node's
  header.
- **Small fixed costs:** no bounds checks where an invariant holds, no
  full-hash compare for small keys, tag 0 for unused lanes so a lookup
  needs no lane mask, inlined fast paths for `make_mut` and `push`, and
  room for 8 in a new leaf.

### Target CPU (x86_64)

Lookups no longer use popcount, so a baseline x86-64 build loses little:

| operation | keys, N | native | x86-64-v2 | x86-64 |
|---|---|---|---|---|
| lookup | u64, 10k | 4.2 | 4.5 | 4.6 |
| lookup | u64, 1M | 42.2 | 42.4 | 42.3 |
| lookup | str, 10k | 8.0 | 8.0 | 8.1 |
| lookup | str, 1M | 56.4 | 57.1 | 57.0 |
| in-place insert | u64, 10k | 22.9 | 23.0 | 23.9 |
| in-place insert | u64, 1M | 64.2 | 65.5 | 66.9 |
| in-place insert | str, 10k | 31.9 | 32.5 | 33.0 |
| in-place insert | str, 1M | 92.4 | 92.7 | 95.1 |

AVX2 compares all 32 tags in one instruction but measured no faster
than SSE2's two, and choosing it at run time made a baseline build
slower, so only SSE2 is used.

### Apple M2

Same harness, not pinned (macOS has no affinity), fastest of 5.

| operation | keys | N | imhm | imbl | chunk16 |
|---|---|---|---|---|---|
| lookup | u64 | 1k | 4.3 | 3.5 | 5.7 |
| lookup | u64 | 10k | 6.3 | 4.8 | 20.7 |
| lookup | u64 | 100k | 9.0 | 6.9 | 53.6 |
| lookup | u64 | 1M | 37.6 | 37.1 | 100.2 |
| lookup | str | 1k | 8.4 | 8.4 | 57.1 |
| lookup | str | 10k | 10.5 | 10.0 | 130.5 |
| lookup | str | 100k | 13.9 | 13.2 | 210.0 |
| lookup | str | 1M | 52.3 | 59.1 | 592.4 |
| in-place insert | u64 | 1k | 22.7 | 25.1 | 71.0 |
| in-place insert | u64 | 10k | 26.8 | 19.7 | 118.8 |
| in-place insert | u64 | 100k | 47.6 | 36.2 | 177.2 |
| in-place insert | u64 | 1M | 80.2 | 86.4 | 287.5 |
| in-place insert | str | 1k | 28.3 | 33.2 | 179.6 |
| in-place insert | str | 10k | 35.1 | 27.2 | 284.2 |
| in-place insert | str | 100k | 60.0 | 47.8 | 413.9 |
| in-place insert | str | 1M | 108.9 | 118.8 | 875.1 |
| snapshot+insert | u64 | 1k | 259 | 369 | 281 |
| snapshot+insert | u64 | 10k | 320 | 430 | 427 |
| snapshot+insert | u64 | 100k | 528 | 659 | 575 |
| snapshot+insert | u64 | 1M | 692 | 909 | 792 |
| snapshot+insert | str | 1k | 303 | 373 | 421 |
| snapshot+insert | str | 10k | 370 | 459 | 655 |
| snapshot+insert | str | 100k | 555 | 669 | 905 |
| snapshot+insert | str | 1M | 780 | 980 | 1112 |

The M2 runs close to 5 instructions a cycle, so instruction count is
what it pays for. A u64 lookup there is about 100 instructions to x86's
82: LLVM expands the NEON narrowing shift and bit select into shifts,
ANDs and ORs, and 64-bit constants take 4 instructions each. imbl
compares 16 tags per lookup, imhm 32.

### Memory

Heap bytes, from `examples/mem.rs`: built in place, then each of 100
snapshot inserts kept.

| | per pair, 10k | per pair, 1M | per snapshot insert, 10k | per snapshot insert, 1M |
|---|---|---|---|---|
| imhm | 45.5 | 50.5 | 2486 | 4246 |
| imbl | 47.5 | 103.5 | 2059 | 3432 |
| chunk16 | 29.8 | 27.2 | 1102 | 1508 |

(u64 keys; `ArcStr` keys are within 2%, the strings themselves shared.)
A snapshot insert copies each inner node on its path, all 32 slots.

## Open work

- **M2 lookups at 10k–100k** trail imbl by up to 30%. Hand-written NEON
  that LLVM leaves alone, or 16-lane leaf groups as imbl has, might
  close it.
- **Snapshot memory** is 2.5–3× chunk16's per persistent insert,
  because inner nodes are copied whole.
