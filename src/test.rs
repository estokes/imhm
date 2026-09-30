use super::*;
use std::{
    collections::{HashMap, HashSet, hash_map::RandomState},
    hash::Hasher,
};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let z = (self.0 ^ (self.0 >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const MUL_INV: u64 = {
    let mut inv = MUL;
    let mut i = 0;
    while i < 5 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(MUL.wrapping_mul(inv)));
        i += 1;
    }
    inv
};

fn unmix(y: u64) -> u64 {
    (y ^ (y >> 32)).wrapping_mul(MUL_INV)
}

/// Places a u64 key at trie hash `key >> SHIFT`: keys equal but for
/// the low `SHIFT` bits collide.
#[derive(Clone, Copy, Default)]
struct Placed<const SHIFT: u32>;

struct PlacedHasher<const SHIFT: u32>(u64);

impl<const SHIFT: u32> Hasher for PlacedHasher<SHIFT> {
    fn finish(&self) -> u64 {
        unmix(self.0 >> SHIFT)
    }

    fn write(&mut self, _: &[u8]) {
        unimplemented!("Placed hashes u64 keys only")
    }

    fn write_u64(&mut self, x: u64) {
        self.0 = x
    }
}

impl<const SHIFT: u32> BuildHasher for Placed<SHIFT> {
    type Hasher = PlacedHasher<SHIFT>;

    fn build_hasher(&self) -> PlacedHasher<SHIFT> {
        PlacedHasher(0)
    }
}

/// Keys in groups whose trie hashes share all but their last 4 bits.
/// Keys equal but for their low `SHIFT` bits share a hash, so with
/// `SHIFT` > 0 a group outgrows a leaf and chains to the deepest level.
fn placed_key<const SHIFT: u32>(rng: &mut Rng) -> u64 {
    let trie = rng.below(2) << (63 - SHIFT) | rng.below(4) << 20 | rng.below(16);
    trie << SHIFT | rng.below(1 << SHIFT)
}

fn check<K: Hash + Eq, V, S: BuildHasher>(m: &Map<K, V, S>) {
    if let Some(r) = &m.root {
        assert_eq!(r.depth(), 0);
        assert!(r.len() > 0);
        check_node(&m.hasher, r);
    }
    assert_eq!(m.iter().count(), m.len());
    let hashes: Vec<u64> = m.iter().map(|(k, _)| hash_of(&m.hasher, k)).collect();
    assert!(hashes.is_sorted(), "iteration is in hash order");
}

fn check_node<K: Hash + Eq, V, S: BuildHasher>(s: &S, n: &Node<K, V>) -> usize {
    let d = n.depth();
    match n {
        Node::Leaf(l) => {
            let es = l.items();
            assert!(d <= MAX_DEPTH + 1);
            assert!(es.len() >= if d > 0 { 2 } else { 1 });
            let first = es[0].hash;
            assert!(es.len() <= LEAF || es.iter().all(|e| e.hash == first));
            for (i, e) in es.iter().enumerate() {
                assert_eq!(e.hash, hash_of(s, &e.key));
                assert_eq!(prefix(e.hash, d), prefix(first, d));
                assert!(i == 0 || es[i - 1].hash <= e.hash);
                assert!(es[..i].iter().all(|p| p.key != e.key));
                if i < LEAF {
                    assert_eq!(l.header().tags[i], e.hash as u8)
                }
            }
            es.len()
        }
        Node::Inner(inner) => {
            let (h, slots) = (inner.header(), inner.items());
            assert!(d <= MAX_DEPTH);
            assert_eq!(slots.len(), h.bitmap.count_ones() as usize);
            assert!(!matches!(slots, [Slot::Node(Node::Leaf(_))]));
            let p = prefix(slots[0].hash(), d);
            let mut bits = h.bitmap;
            let mut len = 0;
            for slot in slots {
                let hash = slot.hash();
                assert_eq!(prefix(hash, d), p);
                assert_eq!(bit(hash, d), 1 << bits.trailing_zeros());
                bits &= bits - 1;
                len += match slot {
                    Slot::Entry(e) => {
                        assert_eq!(e.hash, hash_of(s, &e.key));
                        1
                    }
                    Slot::Node(c) => {
                        assert_eq!(c.depth(), d + 1);
                        check_node(s, c)
                    }
                };
            }
            assert_eq!(len, h.len);
            assert!(len > LEAF);
            len
        }
    }
}

fn depth_of<K, V>(n: &Node<K, V>) -> u8 {
    match n {
        Node::Leaf(l) => l.header().depth,
        Node::Inner(inner) => inner
            .items()
            .iter()
            .map(|s| match s {
                Slot::Node(c) => depth_of(c),
                Slot::Entry(_) => inner.header().depth,
            })
            .max()
            .unwrap_or(inner.header().depth),
    }
}

fn assert_matches<S: BuildHasher>(m: &Map<u64, u64, S>, model: &HashMap<u64, u64>) {
    check(m);
    assert_eq!(m.len(), model.len());
    for (k, v) in model {
        assert_eq!(m.get(k), Some(v));
    }
    let seen: HashSet<u64> = m.iter().map(|(k, _)| *k).collect();
    assert_eq!(seen.len(), model.len());
}

/// Random operations against a std model, keeping snapshots that later
/// operations must not disturb.
/// `n`, or a fraction of it under Miri.
fn scale(n: usize) -> usize {
    if cfg!(miri) { (n / 100).max(1) } else { n }
}

fn model_test<S: BuildHasher + Clone>(
    hasher: S,
    seed: u64,
    steps: usize,
    mut key: impl FnMut(&mut Rng) -> u64,
) -> Map<u64, u64, S> {
    let mut rng = Rng(seed);
    let mut m = Map::with_hasher(hasher);
    let mut model = HashMap::new();
    let mut snaps = Vec::new();
    for step in 0..steps {
        let k = key(&mut rng);
        let v = rng.next();
        match rng.below(10) {
            0..=2 => assert_eq!(m.insert_cow(k, v), model.insert(k, v)),
            3 | 4 => assert_eq!(m.remove_cow(&k), model.remove(&k)),
            5 => {
                let (m2, prev) = m.insert(k, v);
                assert_eq!(prev, model.insert(k, v));
                m = m2;
            }
            6 => {
                let (m2, prev) = m.remove(&k);
                assert_eq!(prev, model.remove(&k));
                m = m2;
            }
            7 => {
                let x = m.get_mut_cow(&k);
                assert_eq!(x.is_some(), model.contains_key(&k));
                if let Some(x) = x {
                    *x = v;
                    model.insert(k, v);
                }
            }
            8 => {
                *m.get_or_default_cow(k) ^= v;
                *model.entry(k).or_default() ^= v;
            }
            _ => {
                if step % 7 == 0 {
                    snaps.push((m.clone(), model.clone()))
                }
            }
        }
        assert_eq!(m.get(&k), model.get(&k));
        assert_eq!(m.len(), model.len());
        if step % 97 == 0 {
            assert_matches(&m, &model)
        }
    }
    assert_matches(&m, &model);
    for (s, sm) in &snaps {
        assert_matches(s, sm)
    }
    m
}

#[test]
fn mix_inverts() {
    let mut rng = Rng(1);
    for _ in 0..10_000 {
        let x = rng.next();
        assert_eq!(mix(unmix(x)), x);
        assert_eq!(unmix(mix(x)), x);
    }
}

#[test]
fn tag_matchers_agree() {
    let mut rng = Rng(2);
    for _ in 0..scale(10_000) {
        let t = rng.next() as u8;
        let tags: [u8; LEAF] = array::from_fn(|_| t ^ rng.below(3) as u8);
        let exact = (0..LEAF).filter(|&i| tags[i] == t).fold(0u32, |m, i| m | 1 << i);
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert_eq!(tag_matches(&tags, t), exact);
        let swar = swar_matches(&tags, t);
        let extra = swar & !exact;
        assert_eq!(swar & exact, exact);
        assert_eq!(extra & 0x0101_0101, 0);
        assert_eq!(extra & !(swar << 1), 0);
    }
}

#[test]
fn model_fx() {
    for seed in 0..scale(20) as u64 {
        model_test(FxBuildHasher, seed, scale(5_000), |r| r.below(700));
    }
    model_test(FxBuildHasher, 99, scale(100_000), |r| r.below(20_000));
}

#[test]
fn model_random_state() {
    for seed in 0..scale(5) as u64 {
        model_test(RandomState::new(), seed, scale(5_000), |r| r.below(700));
    }
}

#[test]
fn model_nohash() {
    for seed in 0..scale(10) as u64 {
        let h = nohash::BuildNoHashHasher::<u64>::default();
        model_test(h.clone(), seed, scale(5_000), |r| r.below(700));
        model_test(h, seed, scale(5_000), |r| r.below(700) << 32);
    }
}

#[test]
fn model_ahash() {
    for seed in 0..scale(10) as u64 {
        let h = ahash::RandomState::with_seeds(seed, 1, 2, 3);
        model_test(h, seed, scale(5_000), |r| r.below(700));
    }
}

/// Every value is a clone of one token, so a leak or a double drop of
/// a value shows in its count.
#[test]
fn drops_balance() {
    let token = std::sync::Arc::new(());
    {
        let mut rng = Rng(5);
        let mut m: Map<u64, std::sync::Arc<()>, Placed<2>> = Map::default();
        let mut snaps = Vec::new();
        for step in 0..scale(20_000) {
            let k = placed_key::<2>(&mut rng);
            match rng.below(5) {
                0 | 1 => {
                    m.insert_cow(k, token.clone());
                }
                2 => {
                    m.remove_cow(&k);
                }
                3 => m = m.insert(k, token.clone()).0,
                _ => m = m.remove(&k).0,
            }
            if step % 50 == 0 {
                snaps.push(m.clone());
                if snaps.len() > 10 {
                    snaps.remove(0);
                }
            }
        }
        for s in &snaps {
            check(s)
        }
    }
    assert_eq!(std::sync::Arc::strong_count(&token), 1);
}

#[test]
fn model_deep() {
    let depths: Vec<u8> = (0..scale(20) as u64)
        .map(|seed| {
            let m = model_test(Placed::<0>, seed, 5_000, placed_key::<0>);
            depth_of(m.root.as_ref().unwrap())
        })
        .collect();
    assert!(depths.iter().all(|d| *d >= 8), "{depths:?}");
}

#[test]
fn model_collisions() {
    let depths: Vec<u8> = (0..scale(20) as u64)
        .map(|seed| {
            let m = model_test(Placed::<2>, seed, 5_000, placed_key::<2>);
            depth_of(m.root.as_ref().unwrap())
        })
        .collect();
    assert!(depths.contains(&(MAX_DEPTH + 1)), "{depths:?}");
}

/// A leaf of one hash outgrows [`LEAF`]; a key of another hash splits
/// it into an inner chain, and removing that key collapses it back.
#[test]
fn one_hash_leaf() {
    let mut m: Map<u64, u64, Placed<6>> = Map::default();
    for k in 0..40 {
        m.insert_cow(k, k);
    }
    check(&m);
    assert!(matches!(m.root, Some(Node::Leaf(_))));
    m.insert_cow(1 << 6, 0);
    check(&m);
    assert_eq!(depth_of(m.root.as_ref().unwrap()), MAX_DEPTH + 1);
    assert_eq!(m.remove_cow(&(1 << 6)), Some(0));
    check(&m);
    assert!(matches!(m.root, Some(Node::Leaf(_))));
    for k in 0..40 {
        assert_eq!(m.get(&k), Some(&k));
    }
}

/// The shape depends on the key set alone, so any history gives the
/// same iteration order.
#[test]
fn shape_is_canonical() {
    let mut rng = Rng(7);
    let keys: Vec<u64> = (0..3000).map(|_| placed_key::<0>(&mut rng)).collect();
    let forward: Map<u64, u64, Placed<0>> = keys.iter().map(|k| (*k, 0)).collect();
    let mut scrambled: Map<u64, u64, Placed<0>> = Map::default();
    for k in keys.iter().rev() {
        scrambled.insert_cow(*k, 0);
        scrambled.insert_cow(k ^ (1 << 40), 0);
    }
    for k in keys.iter().rev() {
        if !keys.contains(&(k ^ (1 << 40))) {
            scrambled.remove_cow(&(k ^ (1 << 40)));
        }
    }
    check(&scrambled);
    let a: Vec<u64> = forward.iter().map(|(k, _)| *k).collect();
    let b: Vec<u64> = scrambled.iter().map(|(k, _)| *k).collect();
    assert_eq!(a, b);
}

#[test]
fn removing_everything_empties() {
    let mut m: Map<u64, u64> = (0..1000).map(|i| (i, i)).collect();
    let snap = m.clone();
    for i in 0..1000 {
        assert_eq!(m.remove_cow(&i), Some(i));
    }
    assert!(m.is_empty() && m.root.is_none());
    assert_eq!(snap.len(), 1000);
    check(&snap);
}

/// A miss leaves the tree shared with its clone.
#[test]
fn miss_copies_nothing() {
    let m: Map<u64, u64> = (0..1000).map(|i| (i, i)).collect();
    let mut c = m.clone();
    assert_eq!(c.remove_cow(&5000), None);
    assert!(c.get_mut_cow(&5000).is_none());
    assert!(m.root.as_ref().unwrap().ptr_eq(c.root.as_ref().unwrap()));
}

#[test]
fn update_copies_one_path() {
    let m: Map<u64, u64> = (0..scale(100_000) as u64).map(|i| (i, i)).collect();
    let (m2, _) = m.insert(7, 0);
    fn nodes(n: &Node<u64, u64>, ids: &mut HashSet<usize>) {
        if ids.insert(n.addr())
            && let Node::Inner(inner) = n
        {
            for s in inner.items() {
                if let Slot::Node(c) = s {
                    nodes(c, ids)
                }
            }
        }
    }
    let (mut a, mut both) = (HashSet::new(), HashSet::new());
    nodes(m.root.as_ref().unwrap(), &mut a);
    nodes(m.root.as_ref().unwrap(), &mut both);
    nodes(m2.root.as_ref().unwrap(), &mut both);
    let depth = depth_of(m2.root.as_ref().unwrap()) as usize;
    assert!(both.len() - a.len() <= depth + 1);
}

fn rebuild<K, V, S>(
    s: &S,
    n: NodeRef<'_, K, V>,
    memo: &mut HashMap<usize, NodeHandle<K, V>>,
) -> NodeHandle<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher,
{
    if let Some(h) = memo.get(&n.identity()) {
        return h.clone();
    }
    let h = match n.contents() {
        Contents::Leaf(pairs) => {
            NodeHandle::leaf(s, n.depth(), pairs.map(|(k, v)| (k.clone(), v.clone())))
        }
        Contents::Inner(slots) => {
            let slots: Vec<NewSlot<K, V>> = slots
                .map(|slot| match slot {
                    SlotRef::Entry(k, v) => NewSlot::Entry(k.clone(), v.clone()),
                    SlotRef::Node(c) => NewSlot::Node(rebuild(s, c, memo)),
                })
                .collect();
            NodeHandle::inner(s, n.depth(), slots)
        }
    }
    .unwrap();
    memo.insert(n.identity(), h.clone());
    h
}

/// A forest rebuilt through the node API equals the original and
/// shares what it shared.
#[test]
fn structural_round_trip() {
    let mut rng = Rng(3);
    let m0: Map<u64, u64, Placed<2>> =
        (0..2000).map(|_| (placed_key::<2>(&mut rng), rng.next())).collect();
    let (m1, _) = m0.insert(placed_key::<2>(&mut rng), 1);
    let m2 = m1.remove_many(m0.iter().map(|(k, _)| *k).take(3).collect::<Vec<_>>());
    let mut memo = HashMap::new();
    let rebuilt: Vec<Map<u64, u64, Placed<2>>> = [&m0, &m1, &m2]
        .iter()
        .map(|m| {
            let root = m.root().map(|r| rebuild(&Placed::<2>, r, &mut memo));
            Map::from_root(root, Placed).unwrap()
        })
        .collect();
    let distinct = memo.len();
    let mut m0_only = HashMap::new();
    rebuild(&Placed::<2>, m0.root().unwrap(), &mut m0_only);
    assert!(distinct < m0_only.len() + 3 * (MAX_DEPTH as usize + 2));
    for (orig, new) in [&m0, &m1, &m2].into_iter().zip(&rebuilt) {
        check(new);
        assert_eq!(orig, new);
        let a: Vec<_> = orig.iter().collect();
        let b: Vec<_> = new.iter().collect();
        assert_eq!(a, b);
    }
    for k in [1, 1 << 30, u64::MAX] {
        let (m3, prev) = rebuilt[1].insert(k, 1);
        assert_eq!(prev, m1.get(&k).copied());
        assert_eq!(m3.get(&k), Some(&1));
        check(&m3);
    }
}

#[test]
fn create_rejects() {
    let s = Placed::<2>;
    let k = |top: u64, low: u64| (top << 59 | low) << 2;
    let leaf =
        |depth, ks: &[u64]| NodeHandle::leaf(&s, depth, ks.iter().map(|k| (*k, 0)));
    assert!(leaf(0, &[k(0, 1), k(0, 2)]).is_ok());
    assert!(leaf(0, &[k(0, 2), k(0, 1)]).is_err(), "out of hash order");
    assert!(leaf(0, &[]).is_err(), "empty");
    assert!(leaf(1, &[k(0, 1)]).is_err(), "one pair below the root");
    assert!(leaf(0, &[k(0, 1), k(0, 1)]).is_err(), "a key twice");
    assert!(leaf(0, &[k(0, 1), k(0, 1) | 1]).is_ok(), "two keys of one hash");
    assert!(leaf(1, &[k(0, 1), k(1, 1)]).is_err(), "prefixes differ");
    assert!(leaf(MAX_DEPTH + 2, &[k(0, 1), k(0, 1) | 1]).is_err(), "too deep");
    let many: Vec<u64> = (0..LEAF as u64 + 1).map(|i| k(0, i)).collect();
    assert!(leaf(0, &many).is_err(), "too many hashes for a leaf");
    let a: Vec<u64> = (0..17).map(|i| k(0, i)).collect();
    let b: Vec<u64> = (0..17).map(|i| k(1, i)).collect();
    let (la, lb) = (leaf(1, &a).unwrap(), leaf(1, &b).unwrap());
    let node = |h: &NodeHandle<u64, u64>| NewSlot::Node(h.clone());
    let inner = |depth, slots| NodeHandle::inner(&s, depth, slots);
    assert!(inner(0, vec![node(&la), node(&lb)]).is_ok());
    assert!(inner(0, vec![node(&la)]).is_err(), "too few pairs for an inner node");
    assert!(inner(0, vec![node(&lb), node(&la)]).is_err(), "out of fragment order");
    assert!(inner(1, vec![node(&la), node(&lb)]).is_err(), "child depth");
    let entry = |k: u64| NewSlot::Entry(k, 0);
    let dup = vec![entry(k(0, 99)), node(&la), node(&lb)];
    assert!(inner(0, dup).is_err(), "one fragment twice");
    assert!(inner(0, vec![node(&la), node(&lb), entry(k(2, 0))]).is_ok());
    assert!(inner(MAX_DEPTH + 1, vec![node(&la), node(&lb)]).is_err(), "too deep");
    assert!(Map::from_root(Some(la), s).is_err(), "root not at depth 0");
    let s6 = Placed::<6>;
    let one_hash = NodeHandle::leaf(&s6, 1, (0..40u64).map(|k| (k, 0))).unwrap();
    let lone = NodeHandle::inner(&s6, 0, [NewSlot::Node(one_hash)]);
    assert!(lone.is_err(), "a lone leaf of one hash is the node");
}

#[test]
fn set() {
    let mut s: Set<u64> = Set::new();
    assert!(!s.insert_cow(1));
    assert!(s.insert_cow(1));
    let (s2, present) = s.insert(2);
    assert!(!present);
    let snap = s2.clone();
    let (s3, present) = s2.remove(&1);
    assert!(present);
    assert!(!s3.remove(&1).1);
    assert!(s3.contains(&2) && !s3.contains(&1));
    assert_eq!(snap.iter().copied().collect::<HashSet<_>>(), HashSet::from([1, 2]));
    let big: Set<u64> = (0..5000).collect();
    let small = big.remove_many(1000..5000);
    assert_eq!(small, (0..1000).collect());
    check(&small.0);
}

#[test]
fn string_keys_borrow() {
    let mut m: Map<String, usize> = Map::new();
    for i in 0..500 {
        m.insert_cow(format!("name{i}"), i);
    }
    assert_eq!(m.get("name42"), Some(&42));
    assert_eq!(m.get_key("name7").map(String::as_str), Some("name7"));
    assert_eq!(m.remove_cow("name42"), Some(42));
    assert!(!m.contains_key("name42"));
    check(&m);
}

/// Clones of one map, changed, snapshotted and dropped on several
/// threads at once: shared nodes must be neither leaked nor freed early.
#[test]
fn threads_share() {
    let token = std::sync::Arc::new(());
    let base: Map<u64, std::sync::Arc<()>> =
        (0..scale(2_000) as u64).map(|k| (k, token.clone())).collect();
    std::thread::scope(|s| {
        for t in 0..4 {
            let (mut m, token) = (base.clone(), &token);
            s.spawn(move || {
                let mut rng = Rng(t);
                for _ in 0..scale(2_000) {
                    let k = rng.below(3_000);
                    match rng.below(3) {
                        0 => {
                            m.insert_cow(k, token.clone());
                        }
                        1 => {
                            m.remove_cow(&k);
                        }
                        _ => {
                            let snap = m.clone();
                            m.insert_cow(k + 10_000, token.clone());
                            drop(snap)
                        }
                    }
                }
                check(&m);
            });
        }
    });
    drop(base);
    assert_eq!(std::sync::Arc::strong_count(&token), 1);
}
