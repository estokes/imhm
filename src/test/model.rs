//! Model checking: random operations of every kind on maps of many key
//! and value types, sizes and hashers, some of them hostile, each
//! version checked against a `HashMap` of what it should hold.
//!
//! Seeds are random; a failure prints the one to replay with
//! `IMHM_SEED=<seed> cargo test`.

use super::{Rng, check, rebuild, unmix};
use crate::*;
use arcstr::ArcStr;
use compact_str::CompactString;
use rustc_hash::FxHasher;
use std::{
    any::Any,
    cell::{Cell, RefCell},
    collections::{HashMap, HashSet, hash_map::RandomState},
    fmt::Debug,
    hash::Hasher,
    panic::{self, AssertUnwindSafe},
};

/// How far a run goes: full in release, a tenth in debug, a hundredth
/// under Miri.
fn budget(n: usize) -> usize {
    if cfg!(miri) {
        (n / 100).max(1)
    } else if cfg!(debug_assertions) {
        (n / 10).max(1)
    } else {
        n
    }
}

fn seed() -> u64 {
    std::env::var("IMHM_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| RandomState::new().hash_one(0u8))
}

/// Prints the seed of a run that panics.
struct Replay(u64);

impl Drop for Replay {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!("replay with IMHM_SEED={}", self.0)
        }
    }
}

trait Random: Clone + Eq + Debug {
    fn random(r: &mut Rng) -> Self;
}

impl Random for u64 {
    fn random(r: &mut Rng) -> Self {
        r.next()
    }
}

impl Random for i32 {
    fn random(r: &mut Rng) -> Self {
        r.next() as i32
    }
}

impl Random for usize {
    fn random(r: &mut Rng) -> Self {
        r.next() as usize
    }
}

/// Mostly short, sometimes past `CompactString`'s inline 24 bytes, and
/// sometimes empty; mostly ASCII, sometimes any char.
fn string(r: &mut Rng) -> String {
    let len = match r.below(8) {
        0 => 0,
        1 => r.below(80) as usize,
        _ => r.below(12) as usize,
    };
    (0..len)
        .map(|_| match r.below(6) {
            0 => char::from_u32(r.below(0x11_0000) as u32).unwrap_or('?'),
            _ => (b'a' + r.below(26) as u8) as char,
        })
        .collect()
}

impl Random for ArcStr {
    fn random(r: &mut Rng) -> Self {
        ArcStr::from(string(r))
    }
}

impl Random for CompactString {
    fn random(r: &mut Rng) -> Self {
        CompactString::from(string(r))
    }
}

impl<A: Random, B: Random> Random for (A, B) {
    fn random(r: &mut Rng) -> Self {
        (A::random(r), B::random(r))
    }
}

impl Random for Map<u64, u64> {
    fn random(r: &mut Rng) -> Self {
        (0..r.below(8)).map(|_| (r.below(16), r.next())).collect()
    }
}

thread_local! {
    /// `Tracked` values alive on this thread.
    static LIVE: Cell<isize> = const { Cell::new(0) };
}

/// Counts itself alive, so a run can check that every clone was
/// dropped exactly once. Its heap payload makes a double drop a double
/// free.
#[derive(Debug, PartialEq, Eq, Hash)]
struct Tracked(Box<u64>);

impl Tracked {
    fn new(id: u64) -> Self {
        LIVE.with(|l| l.set(l.get() + 1));
        Self(Box::new(id))
    }
}

impl Clone for Tracked {
    fn clone(&self) -> Self {
        Self::new(*self.0)
    }
}

impl Drop for Tracked {
    fn drop(&mut self) {
        LIVE.with(|l| l.set(l.get() - 1));
    }
}

impl Random for Tracked {
    fn random(r: &mut Rng) -> Self {
        Self::new(r.next())
    }
}

/// Fx, then only the low `B` bits of the trie hash vary: every key
/// shares the high `64 - B`, so the tree is a deep chain to leaves full
/// of near misses.
#[derive(Clone, Copy, Default)]
struct Deep<const B: u32>;

/// Fx, then only the high `B` bits of the trie hash vary: at most `2^B`
/// distinct hashes, so leaves fill with keys of one full hash. `Few<0>`
/// hashes every key the same.
#[derive(Clone, Copy, Default)]
struct Few<const B: u32>;

struct Squash<const B: u32, const HIGH: bool>(FxHasher);

impl<const B: u32, const HIGH: bool> Hasher for Squash<B, HIGH> {
    fn finish(&self) -> u64 {
        let h = self.0.finish();
        let trie = match (HIGH, B) {
            (_, 0) => 0,
            (true, b) => h >> (64 - b) << (64 - b),
            (false, b) => h & (u64::MAX >> (64 - b)),
        };
        unmix(trie)
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes)
    }
}

impl<const B: u32> BuildHasher for Deep<B> {
    type Hasher = Squash<B, false>;

    fn build_hasher(&self) -> Self::Hasher {
        Squash(FxHasher::default())
    }
}

impl<const B: u32> BuildHasher for Few<B> {
    type Hasher = Squash<B, true>;

    fn build_hasher(&self) -> Self::Hasher {
        Squash(FxHasher::default())
    }
}

/// A tree's shape: the nodes, their depths and bitmaps, and the hashes
/// in each leaf as a set, which is all the canonical rule fixes.
fn shape<K, V>(m: &Map<K, V, impl BuildHasher>) -> Vec<(u8, u8, u64)> {
    fn walk<K, V>(n: &Node<K, V>, out: &mut Vec<(u8, u8, u64)>) {
        match n {
            Node::Leaf(l) => {
                let d = l.header().depth;
                out.push((0, d, l.items().len() as u64));
                let mut hashes: Vec<u64> = l.items().iter().map(|e| e.hash).collect();
                hashes.sort();
                out.extend(hashes.into_iter().map(|h| (1, d, h)));
            }
            Node::Inner(n) => {
                let h = n.header();
                out.push((2, h.depth, h.bitmap as u64));
                for slot in slots(n) {
                    match slot {
                        Slot::Empty => (),
                        Slot::Entry(e) => out.push((3, h.depth, e.hash)),
                        Slot::Node(c) => walk(c, out),
                    }
                }
            }
        }
    }
    let mut out = Vec::new();
    if let Some(r) = &m.root {
        walk(r, &mut out)
    }
    out
}

/// What a version should hold. Its hasher is fixed, so the keys it
/// picks and the order it lists them in replay with the seed.
type Model<K, V> = HashMap<K, V, FxBuildHasher>;

/// Everything a version must agree with its model on.
fn agree<K, V, S>(m: &Map<K, V, S>, model: &Model<K, V>)
where
    K: Hash + Eq + Debug,
    V: Eq + Debug,
    S: BuildHasher,
{
    check(m);
    assert_eq!(m.len(), model.len());
    assert_eq!(m.is_empty(), model.is_empty());
    assert_eq!(m.is_empty(), m.root.is_none());
    for (k, v) in model {
        assert_eq!(m.get(k), Some(v), "{k:?}");
    }
    let mut seen = 0;
    for (k, v) in m {
        assert_eq!(model.get(k), Some(v), "{k:?}");
        seen += 1;
    }
    assert_eq!(seen, model.len());
}

/// The same key set built in another order, and the same tree rebuilt
/// through the node API, are the same map with the same shape and
/// iteration order.
fn canonical<K, V, S>(r: &mut Rng, m: &Map<K, V, S>, model: &Model<K, V>)
where
    K: Hash + Eq + Clone + Debug,
    V: Eq + Clone + Debug,
    S: BuildHasher + Clone,
{
    let mut pairs: Vec<(K, V)> =
        model.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    for i in (1..pairs.len()).rev() {
        pairs.swap(i, r.below(i as u64 + 1) as usize);
    }
    let mut other = Map::with_hasher(m.hasher.clone());
    other.extend(pairs);
    assert!(other == *m);
    assert_eq!(shape(&other), shape(m));
    let order = |m: &Map<K, V, S>| -> Vec<u64> {
        m.iter().map(|(k, _)| hash_of(&m.hasher, k)).collect()
    };
    assert_eq!(order(&other), order(m));
    let mut memo = HashMap::new();
    let root = m.root().map(|n| rebuild(&m.hasher, n, &mut memo));
    let rebuilt = Map::from_root(root, m.hasher.clone()).unwrap();
    agree(&rebuilt, model);
    assert_eq!(shape(&rebuilt), shape(m));
}

/// Random operations on up to 8 versions of a map whose keys come from
/// a pool of `2 * size`, so about half are present. The run grows the
/// map, churns it, then shrinks it, partly in hash order so subtrees
/// empty one after another, checking every version against its model
/// as it goes.
fn hammer<K, V, S>(hasher: S, size: usize, seed: u64)
where
    K: Random + Hash,
    V: Random,
    S: BuildHasher + Clone,
{
    let live = LIVE.with(Cell::get);
    {
        let r = &mut Rng(seed);
        let pool: Vec<K> = (0..2 * size).map(|_| K::random(r)).collect();
        let key = |r: &mut Rng| pool[r.below(pool.len() as u64) as usize].clone();
        let mut versions =
            vec![(Map::<K, V, S>::with_hasher(hasher.clone()), Model::<K, V>::default())];
        let steps = 6 * size + 50;
        // A fork copies its model, so a large map forks less often.
        let fork_every = (size / 2000).max(1) as u64;
        for step in 0..steps {
            let phase = step * 3 / steps;
            let i = match r.below(4) {
                0 => r.below(versions.len() as u64) as usize,
                _ => versions.len() - 1,
            };
            let op = match (phase, r.below(100)) {
                (0, 30..=44) | (0, 55..=62) => r.below(30),
                (2, 0..=29) => 30 + r.below(15),
                (_, op) => op,
            };
            let op = match op {
                45..=62 | 72..=76 | 79..=81 if r.below(fork_every) != 0 => op % 45,
                op => op,
            };
            if let 82..=83 = op
                && versions.len() > 1
            {
                versions.remove(i);
                continue;
            }
            let (m, model) = &mut versions[i];
            let k = match (phase, model.is_empty(), r.below(3)) {
                (2, false, 0) => {
                    let n = r.below(model.len() as u64) as usize;
                    model.keys().nth(n).expect("in range").clone()
                }
                (2, false, 1) => m.iter().next().expect("not empty").0.clone(),
                _ => key(r),
            };
            let v = V::random(r);
            let mut fork = None;
            match op {
                0..=29 => assert_eq!(
                    m.insert_cow(k.clone(), v.clone()),
                    model.insert(k.clone(), v)
                ),
                30..=44 => assert_eq!(m.remove_cow(&k), model.remove(&k)),
                45..=54 => {
                    let (m2, prev) = m.insert(k.clone(), v.clone());
                    let mut model2 = model.clone();
                    assert_eq!(prev, model2.insert(k.clone(), v));
                    fork = Some((m2, model2));
                }
                55..=62 => {
                    let (m2, prev) = m.remove(&k);
                    let mut model2 = model.clone();
                    assert_eq!(prev, model2.remove(&k));
                    fork = Some((m2, model2));
                }
                63..=67 => {
                    let x = m.get_mut_cow(&k);
                    assert_eq!(x.as_deref(), model.get(&k));
                    if let Some(x) = x {
                        *x = v.clone();
                        model.insert(k.clone(), v);
                    }
                }
                68..=71 => {
                    let got = m.get_or_insert_cow(k.clone(), || v.clone()).clone();
                    assert_eq!(&got, model.entry(k.clone()).or_insert(v));
                }
                72..=74 => {
                    let batch: Vec<(K, V)> =
                        (0..r.below(40)).map(|_| (key(r), V::random(r))).collect();
                    let mut model2 = model.clone();
                    model2.extend(batch.iter().cloned());
                    fork = Some((m.insert_many(batch), model2));
                }
                75..=76 => {
                    let batch: Vec<K> = (0..r.below(40)).map(|_| key(r)).collect();
                    let mut model2 = model.clone();
                    for k in &batch {
                        model2.remove(k);
                    }
                    fork = Some((m.remove_many(batch), model2));
                }
                77..=78 => {
                    let batch: Vec<(K, V)> =
                        (0..r.below(40)).map(|_| (key(r), V::random(r))).collect();
                    model.extend(batch.iter().cloned());
                    m.extend(batch);
                }
                79..=81 => fork = Some((m.clone(), model.clone())),
                _ => {
                    assert_eq!(m.get(&k), model.get(&k));
                    assert_eq!(m.contains_key(&k), model.contains_key(&k));
                    assert_eq!(m.get_key(&k), model.get_key_value(&k).map(|(k, _)| k));
                    assert_eq!(m.get_full(&k), model.get_key_value(&k));
                }
            }
            if let Some((m, model)) = &versions.get(i) {
                assert_eq!(m.get(&k), model.get(&k));
                assert_eq!(m.len(), model.len());
                if step % (size / 32 + 1) == 0 {
                    check(m)
                }
            }
            if let Some(f) = fork {
                versions.push(f);
                if versions.len() > 8 {
                    versions.remove(r.below(8) as usize);
                }
            }
            if step % (size / 2 + 50) == 0 {
                for (m, model) in &versions {
                    agree(m, model)
                }
                let (a, b) =
                    (r.below(versions.len() as u64) as usize, versions.len() - 1);
                let same = versions[a].1 == versions[b].1;
                assert_eq!(versions[a].0 == versions[b].0, same);
            }
        }
        for (m, model) in &versions {
            agree(m, model);
            canonical(r, m, model);
        }
        let (m, model) = versions.last_mut().expect("a version");
        let keys: Vec<K> = m.iter().map(|(k, _)| k.clone()).collect();
        for (j, k) in keys.iter().enumerate() {
            assert_eq!(m.remove_cow(k), model.remove(k));
            if j % (size / 32 + 1) == 0 {
                check(m)
            }
        }
        agree(m, model);
    }
    assert_eq!(LIVE.with(Cell::get), live, "every Tracked is dropped once");
}

/// Sizes around the edges of a leaf, then larger, each with its own
/// seed drawn from the run's.
fn run<K, V, S>(hasher: S, sizes: &[usize])
where
    K: Random + Hash,
    V: Random,
    S: BuildHasher + Clone,
{
    let seed = seed();
    let _replay = Replay(seed);
    let r = &mut Rng(seed);
    for &n in sizes {
        let n = if n <= 2 * LEAF { n } else { budget(n) };
        hammer::<K, V, S>(hasher.clone(), n, r.next());
    }
}

const SIZES: &[usize] = &[1, 2, 31, 32, 33, 64, 1_000, 20_000];
/// For hashers that collide a lot, whose leaves are scanned linearly.
const SMALL: &[usize] = &[1, 2, 32, 33, 64, 500];

macro_rules! hammer {
    ($($name:ident: $k:ty, $v:ty;)*) => {
        $(mod $name {
            use super::*;

            #[test]
            fn fx() {
                run::<$k, $v, _>(FxBuildHasher, SIZES)
            }

            #[test]
            fn ahash() {
                run::<$k, $v, _>(::ahash::RandomState::with_seeds(1, 2, 3, 4), SIZES)
            }

            #[test]
            fn deep() {
                run::<$k, $v, _>(Deep::<12>, SIZES)
            }

            #[test]
            fn few() {
                run::<$k, $v, _>(Few::<6>, SMALL)
            }

            #[test]
            fn fewer() {
                run::<$k, $v, _>(Few::<3>, SMALL)
            }

            #[test]
            fn one_hash() {
                run::<$k, $v, _>(Few::<0>, SMALL)
            }
        })*
    };
}

hammer! {
    u64_u64: u64, u64;
    i32_usize: i32, usize;
    usize_i32: usize, i32;
    arcstr_u64: ArcStr, u64;
    compact_arcstr: CompactString, ArcStr;
    pair_u64: (u64, ArcStr), u64;
    u64_map: u64, Map<u64, u64>;
    tracked_tracked: Tracked, Tracked;
}

#[test]
fn nohash_u64() {
    run::<u64, u64, _>(nohash::BuildNoHashHasher::<u64>::default(), SIZES)
}

#[test]
fn nohash_i32() {
    run::<i32, u64, _>(nohash::BuildNoHashHasher::<i32>::default(), SIZES)
}

#[test]
#[cfg_attr(miri, ignore = "scale, not soundness; the small runs cover it")]
fn large_u64() {
    run::<u64, u64, _>(FxBuildHasher, &[500_000])
}

#[test]
#[cfg_attr(miri, ignore = "scale, not soundness; the small runs cover it")]
fn large_arcstr() {
    run::<ArcStr, u64, _>(::ahash::RandomState::with_seeds(1, 2, 3, 4), &[500_000])
}

#[test]
#[cfg_attr(miri, ignore = "scale, not soundness; the small runs cover it")]
fn large_tracked_deep() {
    run::<Tracked, Tracked, _>(Deep::<16>, &[200_000])
}

/// Random operations on a set and several of its versions, against
/// `HashSet`s.
fn hammer_set<K: Random + Hash, S: BuildHasher + Clone>(
    hasher: S,
    size: usize,
    seed: u64,
) {
    let r = &mut Rng(seed);
    let pool: Vec<K> = (0..2 * size).map(|_| K::random(r)).collect();
    let key = |r: &mut Rng| pool[r.below(pool.len() as u64) as usize].clone();
    let mut versions = vec![(Set::with_hasher(hasher), HashSet::<K>::new())];
    for step in 0..6 * size + 50 {
        let i = r.below(versions.len() as u64) as usize;
        let (s, model) = &mut versions[i];
        let k = key(r);
        match r.below(8) {
            0..=2 => assert_eq!(s.insert_cow(k.clone()), !model.insert(k.clone())),
            3 | 4 => assert_eq!(s.remove_cow(&k), model.remove(&k)),
            5 => {
                let (s2, present) = s.insert(k.clone());
                let mut model2 = model.clone();
                assert_eq!(present, !model2.insert(k.clone()));
                versions.push((s2, model2));
            }
            6 => {
                let (s2, present) = s.remove(&k);
                let mut model2 = model.clone();
                assert_eq!(present, model2.remove(&k));
                versions.push((s2, model2));
            }
            _ => {
                assert_eq!(s.contains(&k), model.contains(&k));
                assert_eq!(s.get(&k), model.get(&k));
            }
        }
        if versions.len() > 8 {
            versions.remove(r.below(8) as usize);
        }
        if step % (size / 2 + 50) == 0 {
            for (s, model) in &versions {
                assert_eq!(s.len(), model.len());
                assert_eq!(s.iter().collect::<HashSet<_>>(), model.iter().collect());
                check(&s.0);
            }
        }
    }
}

#[test]
fn sets() {
    let seed = seed();
    let _replay = Replay(seed);
    let r = &mut Rng(seed);
    for &n in &[1, 33, 1_000, 20_000] {
        let n = budget(n);
        hammer_set::<u64, _>(FxBuildHasher, n, r.next());
        hammer_set::<ArcStr, _>(Deep::<12>, n, r.next());
        hammer_set::<Tracked, _>(Few::<6>, n.min(500), r.next());
    }
}

thread_local! {
    /// Clones and drops of `Fragile` left before one panics, when
    /// positive.
    static FUSE: Cell<u64> = const { Cell::new(0) };
    /// Another holder of a map's nodes, dropped by the next `Fragile`
    /// clone: as if another thread dropped its version while the map
    /// copies a node they shared.
    static VANISH: RefCell<Option<Box<dyn Any>>> = const { RefCell::new(None) };
}

/// Burns one unit of the fuse, panicking on the last, unless already
/// unwinding: a second panic would abort.
fn burn() {
    let fuse = FUSE.with(Cell::get);
    if fuse > 0 && !std::thread::panicking() {
        FUSE.with(|f| f.set(fuse - 1));
        if fuse == 1 {
            panic!("fuse")
        }
    }
}

/// A key or value whose clone or drop panics when the fuse runs out.
#[derive(Debug, PartialEq, Eq, Hash)]
struct Fragile(Tracked);

impl Fragile {
    fn id(&self) -> u64 {
        *self.0.0
    }
}

impl Clone for Fragile {
    fn clone(&self) -> Self {
        burn();
        drop(VANISH.with(|v| v.borrow_mut().take()));
        Fragile(self.0.clone())
    }
}

impl Drop for Fragile {
    fn drop(&mut self) {
        burn()
    }
}

/// The pairs of `m` by id, touching no clone or drop.
fn ids<S>(m: &Map<Fragile, Fragile, S>) -> HashMap<u64, u64> {
    m.iter().map(|(k, v)| (k.id(), v.id())).collect()
}

/// Random operations on several versions of a map whose keys and
/// values panic in clone or drop at random points. A panicking operation
/// leaves its map whole, holding what it held before or after, and
/// every other version untouched.
fn fragile<S: BuildHasher + Clone + 'static>(hasher: S, size: usize, seed: u64) {
    let live = LIVE.with(Cell::get);
    {
        let r = &mut Rng(seed);
        let mut versions = vec![(Map::with_hasher(hasher), HashMap::new())];
        let mut panics = 0;
        for step in 0..8 * size + 50 {
            let i = r.below(versions.len() as u64) as usize;
            let (k, v) = (r.below(2 * size as u64), r.next());
            let (key, val) = (Fragile(Tracked::new(k)), Fragile(Tracked::new(v)));
            let op = r.below(7);
            let before: HashMap<u64, u64> = versions[i].1.clone();
            let mut after = before.clone();
            match op {
                0 | 2 => drop(after.insert(k, v)),
                1 | 3 => drop(after.remove(&k)),
                4 => {
                    if let Some(x) = after.get_mut(&k) {
                        *x = v
                    }
                }
                _ => drop(after.entry(k).or_insert(v)),
            }
            if r.below(3) == 0 {
                let other: Box<dyn Any> = Box::new(versions[i].0.clone());
                VANISH.with(|v| *v.borrow_mut() = Some(other));
            }
            let m = &mut versions[i].0;
            if r.below(2) == 0 {
                FUSE.with(|f| f.set(1 + r.below(20)));
            }
            // Whatever a closure still owns is dropped only as it returns;
            // a destructor that panics then would leak the return value
            // (rust-lang/rust#47949), so each arm drops its own.
            let result = panic::catch_unwind(AssertUnwindSafe(|| match op {
                0 => {
                    drop(m.insert_cow(key, val));
                    None
                }
                1 => {
                    drop(m.remove_cow(&key));
                    None
                }
                2 => {
                    let (fork, prev) = m.insert(key, val);
                    drop(prev);
                    Some(fork)
                }
                3 => {
                    let (fork, prev) = m.remove(&key);
                    drop((prev, key, val));
                    Some(fork)
                }
                4 => {
                    if let Some(x) = m.get_mut_cow(&key) {
                        *x = val
                    }
                    None
                }
                _ => {
                    m.get_or_insert_cow(key, || val);
                    None
                }
            }));
            FUSE.with(|f| f.set(0));
            drop(VANISH.with(|v| v.borrow_mut().take()));
            let now = ids(m);
            match result {
                Ok(fork) => {
                    if let Some(fork) = fork {
                        assert_eq!(now, before, "a new version leaves the old one");
                        check(&fork);
                        assert_eq!(ids(&fork), after);
                        versions.push((fork, after));
                    } else {
                        assert_eq!(now, after);
                        versions[i].1 = after;
                    }
                }
                Err(_) => {
                    panics += 1;
                    assert!(now == before || now == after, "whole before or after");
                    check(m);
                    versions[i].1 = now;
                }
            }
            if versions.len() > 8 {
                versions.remove(r.below(8) as usize);
            }
            if step % (size / 4 + 10) == 0 {
                for (m, model) in &versions {
                    check(m);
                    assert_eq!(&ids(m), model);
                }
            }
        }
        assert!(panics > 0);
    }
    assert_eq!(LIVE.with(Cell::get), live, "every Tracked is dropped once");
}

#[test]
fn panics() {
    let seed = seed();
    let _replay = Replay(seed);
    let r = &mut Rng(seed);
    let hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        if info.payload().downcast_ref::<&str>() != Some(&"fuse") {
            hook(info)
        }
    }));
    for &n in &[33, 200, budget(2_000)] {
        fragile(FxBuildHasher, n, r.next());
        fragile(Deep::<12>, n, r.next());
        fragile(Few::<3>, n, r.next());
    }
}
