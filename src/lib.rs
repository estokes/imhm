//! Persistent hash maps and sets: hash tries whose nodes are shared
//! between versions. A clone is O(1); an update copies the path from
//! the root to the key and shares everything else, and a map no other
//! version shares is updated in place.
//!
//! Inner nodes branch on 5 hash bits per level, high bits first; a
//! subtree of at most [`LEAF`] pairs is a flat leaf sorted by hash. A
//! key set has one tree shape for a given hasher, and iteration follows
//! it, in hash order, so with a deterministic hasher (the default) the
//! order is the same in every process. [`Map::root`] and
//! [`NodeHandle`] expose the tree to a codec that must reproduce its
//! sharing.

use anyhow::{Result, ensure};
use node::{Raw, RawMut};
use rustc_hash::FxBuildHasher;
use std::{
    array,
    borrow::Borrow,
    fmt,
    hash::{BuildHasher, Hash},
    iter::FusedIterator,
    mem, slice,
};

mod node;
#[cfg(test)]
mod test;

const BITS: u32 = 5;
/// The deepest inner node. Its fragment is the hash's last 4 bits, so
/// two distinct hashes part at or above it, and below it are only
/// leaves of one hash.
const MAX_DEPTH: u8 = 12;
/// The most pairs a leaf holds, unless all of them share one hash.
pub const LEAF: usize = 32;
const MUL: u64 = 0x9e37_79b9_7f4a_7c15;

/// Spreads the hasher's entropy over every bit the trie reads. A
/// bijection, so distinct hashes stay distinct.
#[inline]
fn mix(h: u64) -> u64 {
    let h = h.wrapping_mul(MUL);
    h ^ (h >> 32)
}

fn hash_of<Q: Hash + ?Sized>(s: &impl BuildHasher, q: &Q) -> u64 {
    mix(s.hash_one(q))
}

/// The bit of `hash`'s fragment at `depth`, high bits first, so a
/// trie's order is its hashes' order.
#[inline]
fn bit(hash: u64, depth: u8) -> u32 {
    1 << ((hash << (BITS * depth as u32)) >> 59)
}

#[inline]
fn index(bitmap: u32, bit: u32) -> usize {
    (bitmap & (bit - 1)).count_ones() as usize
}

/// The fragments above `depth`, which every hash under a node there
/// shares.
#[inline]
fn prefix(hash: u64, depth: u8) -> u64 {
    match BITS * depth as u32 {
        0 => 0,
        n @ 1..64 => hash & !(u64::MAX >> n),
        _ => hash,
    }
}

/// Bit `i` set where `tags[i] == t`.
#[cfg(target_arch = "x86_64")]
#[inline]
fn tag_matches(tags: &[u8; LEAF], t: u8) -> u32 {
    use std::arch::x86_64::{
        _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_set1_epi8,
    };
    // SAFETY: SSE2 is in the x86_64 baseline, and the two unaligned
    // loads read the 32 bytes of `tags`.
    unsafe {
        let needle = _mm_set1_epi8(t as i8);
        let lo = _mm_loadu_si128(tags.as_ptr().cast());
        let hi = _mm_loadu_si128(tags.as_ptr().add(16).cast());
        let lo = _mm_movemask_epi8(_mm_cmpeq_epi8(lo, needle)) as u32;
        let hi = _mm_movemask_epi8(_mm_cmpeq_epi8(hi, needle)) as u32;
        lo | hi << 16
    }
}

/// Bit `i` set where `tags[i] == t`. Each compare lane is masked to
/// its bit's weight, and three pairwise adds sum each run of 8 lanes
/// into one mask byte.
#[cfg(target_arch = "aarch64")]
#[inline]
fn tag_matches(tags: &[u8; LEAF], t: u8) -> u32 {
    use std::arch::aarch64::{
        vandq_u8, vceqq_u8, vdupq_n_u8, vgetq_lane_u32, vld1q_u8, vpaddq_u8,
        vreinterpretq_u32_u8,
    };
    const WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    // SAFETY: NEON is in the aarch64 baseline, and the loads read the
    // 16 bytes of `WEIGHTS` and the 32 bytes of `tags`.
    unsafe {
        let w = vld1q_u8(WEIGHTS.as_ptr());
        let needle = vdupq_n_u8(t);
        let lo = vandq_u8(vceqq_u8(vld1q_u8(tags.as_ptr()), needle), w);
        let hi = vandq_u8(vceqq_u8(vld1q_u8(tags.as_ptr().add(16)), needle), w);
        let s = vpaddq_u8(lo, hi);
        let s = vpaddq_u8(s, s);
        let s = vpaddq_u8(s, s);
        vgetq_lane_u32(vreinterpretq_u32_u8(s), 0)
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn tag_matches(tags: &[u8; LEAF], t: u8) -> u32 {
    swar_matches(tags, t)
}

/// Bit `i` set where `tags[i] == t`, and possibly where bit `i - 1` is
/// set and in the same 8; a candidate is confirmed by its full hash.
#[cfg(any(test, not(any(target_arch = "x86_64", target_arch = "aarch64"))))]
fn swar_matches(tags: &[u8; LEAF], t: u8) -> u32 {
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    let mut m = 0;
    for (w, chunk) in tags.as_chunks::<8>().0.iter().enumerate() {
        let x = u64::from_le_bytes(*chunk) ^ (LO * t as u64);
        let zero = x.wrapping_sub(LO) & !x & HI;
        m |= (((zero >> 7).wrapping_mul(0x0102_0408_1020_4080) >> 56) as u32) << (8 * w);
    }
    m
}

#[derive(Clone)]
struct Entry<K, V> {
    hash: u64,
    key: K,
    val: V,
}

#[derive(Clone, Copy)]
struct InnerHead {
    len: usize,
    bitmap: u32,
    depth: u8,
}

/// `tags[i]` is the low byte of entry `i`'s hash, for the first
/// [`LEAF`] entries.
#[derive(Clone, Copy)]
struct LeafHead {
    depth: u8,
    tags: [u8; LEAF],
}

/// Slots are in fragment order, one per bit of `bitmap`. It holds more
/// than [`LEAF`] pairs of at least two hashes.
type Inner<K, V> = Raw<InnerHead, Slot<K, V>>;

/// Entries in hash order: at least two below the root, and at most
/// [`LEAF`] unless all share one hash.
type Leaf<K, V> = Raw<LeafHead, Entry<K, V>>;

enum Node<K, V> {
    Inner(Inner<K, V>),
    Leaf(Leaf<K, V>),
}

impl<K, V> Clone for Node<K, V> {
    fn clone(&self) -> Self {
        match self {
            Node::Inner(n) => Node::Inner(n.clone()),
            Node::Leaf(n) => Node::Leaf(n.clone()),
        }
    }
}

/// A subtree of one pair is an entry in its parent's slot.
#[derive(Clone)]
enum Slot<K, V> {
    Entry(Entry<K, V>),
    Node(Node<K, V>),
}

impl<K, V> Node<K, V> {
    fn len(&self) -> usize {
        match self {
            Node::Inner(n) => n.header().len,
            Node::Leaf(n) => n.items().len(),
        }
    }

    fn depth(&self) -> u8 {
        match self {
            Node::Inner(n) => n.header().depth,
            Node::Leaf(n) => n.header().depth,
        }
    }

    /// A hash from the subtree. Every hash in it shares the fragments
    /// above its depth, so any one places it.
    fn hash(&self) -> u64 {
        match self {
            Node::Inner(n) => n.items()[0].hash(),
            Node::Leaf(n) => n.items()[0].hash,
        }
    }

    fn addr(&self) -> usize {
        match self {
            Node::Inner(n) => n.addr(),
            Node::Leaf(n) => n.addr(),
        }
    }

    fn ptr_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Node::Inner(a), Node::Inner(b)) => a.ptr_eq(b),
            (Node::Leaf(a), Node::Leaf(b)) => a.ptr_eq(b),
            _ => false,
        }
    }
}

impl<K, V> Slot<K, V> {
    fn hash(&self) -> u64 {
        match self {
            Slot::Entry(e) => e.hash,
            Slot::Node(n) => n.hash(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Slot::Entry(_) => 1,
            Slot::Node(n) => n.len(),
        }
    }
}

fn leaf<K, V>(depth: u8, entries: Vec<Entry<K, V>>) -> Leaf<K, V> {
    let mut tags = [0; LEAF];
    for (t, e) in tags.iter_mut().zip(&entries) {
        *t = e.hash as u8
    }
    let cap = entries.len();
    Raw::new(LeafHead { depth, tags }, cap, entries)
}

fn retag<K, V>(m: &mut RawMut<'_, LeafHead, Entry<K, V>>) {
    let (h, entries) = m.parts();
    for (t, e) in h.tags.iter_mut().zip(entries.iter()) {
        *t = e.hash as u8
    }
}

/// The leaf at `depth` of `a` and `b`, `a` already in the map.
fn pair<K, V>(depth: u8, a: Entry<K, V>, b: Entry<K, V>) -> Leaf<K, V> {
    let mut tags = [0; LEAF];
    let (a, b) = if b.hash < a.hash { (b, a) } else { (a, b) };
    tags[0] = a.hash as u8;
    tags[1] = b.hash as u8;
    Raw::new(LeafHead { depth, tags }, 2, [a, b])
}

/// The subtree at `depth` of `entries`, at least two, in hash order.
fn build<K, V>(depth: u8, entries: Vec<Entry<K, V>>) -> Node<K, V> {
    let n = entries.len();
    if n <= LEAF || entries[0].hash == entries[n - 1].hash {
        return Node::Leaf(leaf(depth, entries));
    }
    let mut slots = Vec::new();
    let mut bitmap = 0;
    let mut rest = entries.into_iter().peekable();
    while let Some(first) = rest.next() {
        let b = bit(first.hash, depth);
        let mut group = vec![first];
        while let Some(e) = rest.next_if(|e| bit(e.hash, depth) == b) {
            group.push(e)
        }
        bitmap |= b;
        slots.push(match <[_; 1]>::try_from(group) {
            Ok([e]) => Slot::Entry(e),
            Err(group) => Slot::Node(build(depth + 1, group)),
        });
    }
    let cap = slots.len();
    Node::Inner(Raw::new(InnerHead { len: n, bitmap, depth }, cap, slots))
}

/// Every entry under `node`, in hash order, moved out of the nodes no
/// other version holds, which are left empty.
fn drain_into<K: Clone, V: Clone>(node: &mut Node<K, V>, out: &mut Vec<Entry<K, V>>) {
    match node {
        Node::Leaf(l) => out.extend(l.take_items()),
        Node::Inner(n) => {
            for slot in n.take_items() {
                match slot {
                    Slot::Entry(e) => out.push(e),
                    Slot::Node(mut c) => drain_into(&mut c, out),
                }
            }
        }
    }
}

/// Makes an inner node that now holds at most [`LEAF`] pairs, or only
/// one leaf, a leaf.
fn settle<K: Clone, V: Clone>(node: &mut Node<K, V>) {
    let Node::Inner(n) = node else { return };
    if n.header().len > LEAF && !matches!(n.items(), [Slot::Node(Node::Leaf(_))]) {
        return;
    }
    let depth = n.header().depth;
    let mut entries = Vec::with_capacity(n.header().len);
    drain_into(node, &mut entries);
    *node = build(depth, entries);
}

#[inline]
fn leaf_index<K, V, Q>(l: &Leaf<K, V>, hash: u64, q: &Q) -> Option<usize>
where
    K: Borrow<Q>,
    Q: Eq + ?Sized,
{
    let entries = l.items();
    let n = entries.len();
    if n > LEAF {
        return (entries[0].hash == hash)
            .then(|| entries.iter().position(|e| e.key.borrow() == q))
            .flatten();
    }
    let mut m = tag_matches(&l.header().tags, hash as u8) & (u64::MAX >> (64 - n)) as u32;
    while m != 0 {
        let i = m.trailing_zeros() as usize;
        let e = &entries[i];
        if e.hash == hash && e.key.borrow() == q {
            return Some(i);
        }
        m &= m - 1;
    }
    None
}

#[inline]
fn find<'a, K, V, Q>(mut n: &'a Node<K, V>, hash: u64, q: &Q) -> Option<(&'a K, &'a V)>
where
    K: Borrow<Q>,
    Q: Eq + ?Sized,
{
    loop {
        match n {
            Node::Leaf(l) => {
                let e = &l.items()[leaf_index(l, hash, q)?];
                return Some((&e.key, &e.val));
            }
            Node::Inner(inner) => {
                let h = inner.header();
                let bit = bit(hash, h.depth);
                if h.bitmap & bit == 0 {
                    return None;
                }
                match &inner.items()[index(h.bitmap, bit)] {
                    Slot::Entry(e) => {
                        return (e.hash == hash && e.key.borrow() == q)
                            .then_some((&e.key, &e.val));
                    }
                    Slot::Node(child) => n = child,
                }
            }
        }
    }
}

fn insert<K: Eq + Clone, V: Clone>(
    node: &mut Node<K, V>,
    hash: u64,
    key: K,
    val: V,
) -> Option<V> {
    match node {
        Node::Inner(inner) => {
            let h = *inner.header();
            let bit = bit(hash, h.depth);
            let i = index(h.bitmap, bit);
            if h.bitmap & bit == 0 {
                inner.insert(i, Slot::Entry(Entry { hash, key, val }));
                let mut m = inner.make_mut();
                m.header().bitmap |= bit;
                m.header().len += 1;
                return None;
            }
            let mut m = inner.make_mut();
            let (h, slots) = m.parts();
            match &mut slots[i] {
                Slot::Entry(e) if e.hash == hash && e.key == key => {
                    return Some(mem::replace(&mut e.val, val));
                }
                Slot::Node(child) => {
                    let prev = insert(child, hash, key, val);
                    if prev.is_none() {
                        h.len += 1
                    }
                    return prev;
                }
                Slot::Entry(_) => h.len += 1,
            }
            let depth = h.depth + 1;
            let Slot::Entry(old) = m.remove(i) else {
                unreachable!("an entry, matched above")
            };
            let new = Entry { hash, key, val };
            m.insert(i, Slot::Node(Node::Leaf(pair(depth, old, new))));
            None
        }
        Node::Leaf(l) => {
            if let Some(i) = leaf_index(l, hash, &key) {
                return Some(mem::replace(&mut l.make_mut().items()[i].val, val));
            }
            let entries = l.items();
            let n = entries.len();
            let at = entries.partition_point(|e| e.hash <= hash);
            let new = Entry { hash, key, val };
            if n < LEAF || (entries[0].hash == hash && entries[n - 1].hash == hash) {
                l.insert(at, new);
                retag(&mut l.make_mut());
            } else {
                let depth = l.header().depth;
                let mut entries = l.take_items();
                entries.insert(at, new);
                *node = build(depth, entries);
            }
            None
        }
    }
}

/// Remove `q`, which must be present: a miss would copy the path for
/// nothing.
fn remove<K, V, Q>(node: &mut Node<K, V>, hash: u64, q: &Q) -> Option<V>
where
    K: Borrow<Q> + Clone,
    V: Clone,
    Q: Eq + ?Sized,
{
    match node {
        Node::Leaf(l) => {
            let i = leaf_index(l, hash, q)?;
            let e = l.remove(i);
            retag(&mut l.make_mut());
            Some(e.val)
        }
        Node::Inner(inner) => {
            let h = *inner.header();
            let bit = bit(hash, h.depth);
            if h.bitmap & bit == 0 {
                return None;
            }
            let i = index(h.bitmap, bit);
            let mut m = inner.make_mut();
            let v = match &mut m.items()[i] {
                Slot::Entry(e) if e.hash == hash && e.key.borrow() == q => {
                    let Slot::Entry(e) = m.remove(i) else {
                        unreachable!("an entry, matched above")
                    };
                    m.header().bitmap &= !bit;
                    e.val
                }
                Slot::Entry(_) => return None,
                Slot::Node(child) => {
                    let v = remove(child, hash, q)?;
                    match child {
                        Node::Leaf(l) if l.items().len() == 1 => {
                            let e = l.remove(0);
                            m.items()[i] = Slot::Entry(e);
                        }
                        _ => settle(child),
                    }
                    v
                }
            };
            m.header().len -= 1;
            Some(v)
        }
    }
}

/// The value of `q`, which must be present: a miss would copy the path
/// for nothing.
fn get_mut<'a, K, V, Q>(node: &'a mut Node<K, V>, hash: u64, q: &Q) -> Option<&'a mut V>
where
    K: Borrow<Q> + Clone,
    V: Clone,
    Q: Eq + ?Sized,
{
    match node {
        Node::Leaf(l) => {
            let i = leaf_index(l, hash, q)?;
            Some(&mut l.make_mut().into_items()[i].val)
        }
        Node::Inner(inner) => {
            let h = *inner.header();
            let bit = bit(hash, h.depth);
            if h.bitmap & bit == 0 {
                return None;
            }
            match &mut inner.make_mut().into_items()[index(h.bitmap, bit)] {
                Slot::Entry(e) => {
                    (e.hash == hash && e.key.borrow() == q).then_some(&mut e.val)
                }
                Slot::Node(child) => get_mut(child, hash, q),
            }
        }
    }
}

/// A persistent hash map.
pub struct Map<K, V, S = FxBuildHasher> {
    root: Option<Node<K, V>>,
    hasher: S,
}

impl<K, V, S: Clone> Clone for Map<K, V, S> {
    fn clone(&self) -> Self {
        Self { root: self.root.clone(), hasher: self.hasher.clone() }
    }
}

impl<K, V, S: Default> Default for Map<K, V, S> {
    fn default() -> Self {
        Self { root: None, hasher: S::default() }
    }
}

impl<K, V> Map<K, V> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<K, V, S> Map<K, V, S> {
    pub fn with_hasher(hasher: S) -> Self {
        Self { root: None, hasher }
    }

    pub fn hasher(&self) -> &S {
        &self.hasher
    }

    pub fn len(&self) -> usize {
        self.root.as_ref().map_or(0, Node::len)
    }

    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// The pairs in hash order.
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter::new(self.root.as_ref())
    }

    /// The root of the map's tree, `None` when empty.
    pub fn root(&self) -> Option<NodeRef<'_, K, V>> {
        self.root.as_ref().map(NodeRef)
    }

    /// The map over the tree rooted at `root`, a depth 0 node built
    /// with a hasher that hashes like `hasher`.
    pub fn from_root(root: Option<NodeHandle<K, V>>, hasher: S) -> Result<Self> {
        if let Some(r) = &root {
            ensure!(r.0.depth() == 0, "a root is at depth 0, not {}", r.0.depth());
        }
        Ok(Self { root: root.map(|r| r.0), hasher })
    }
}

impl<K, V, S> Map<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    pub fn get_full<Q>(&self, q: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        find(self.root.as_ref()?, hash_of(&self.hasher, q), q)
    }

    pub fn get<Q>(&self, q: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get_full(q).map(|(_, v)| v)
    }

    pub fn get_key<Q>(&self, q: &Q) -> Option<&K>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get_full(q).map(|(k, _)| k)
    }

    pub fn contains_key<Q>(&self, q: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get_full(q).is_some()
    }
}

impl<K, V, S> Map<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher,
{
    /// Insert in place, copying only the parts of the tree shared with
    /// another version. Returns the previous value; the previous key
    /// stays.
    pub fn insert_cow(&mut self, key: K, val: V) -> Option<V> {
        let hash = hash_of(&self.hasher, &key);
        match &mut self.root {
            Some(root) => insert(root, hash, key, val),
            None => {
                self.root = Some(Node::Leaf(leaf(0, vec![Entry { hash, key, val }])));
                None
            }
        }
    }

    /// Remove in place, copying only the parts of the tree shared with
    /// another version.
    pub fn remove_cow<Q>(&mut self, q: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = hash_of(&self.hasher, q);
        let root = self.root.as_mut()?;
        find(root, hash, q)?;
        let v = remove(root, hash, q)?;
        match root.len() {
            0 => self.root = None,
            _ => settle(root),
        }
        Some(v)
    }

    /// The value of `q` to change in place, copying only the parts of
    /// the tree shared with another version.
    pub fn get_mut_cow<Q>(&mut self, q: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = hash_of(&self.hasher, q);
        let root = self.root.as_mut()?;
        find(root, hash, q)?;
        get_mut(root, hash, q)
    }

    /// The value of `key`, first inserting `f()` if it is absent.
    pub fn get_or_insert_cow(&mut self, key: K, f: impl FnOnce() -> V) -> &mut V {
        if !self.contains_key(&key) {
            self.insert_cow(key.clone(), f());
        }
        self.get_mut_cow(&key).expect("present after insert")
    }

    pub fn get_or_default_cow(&mut self, key: K) -> &mut V
    where
        V: Default,
    {
        self.get_or_insert_cow(key, V::default)
    }
}

impl<K, V, S> Map<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Clone,
{
    /// A new map with `key` bound to `val`, and the previous value.
    pub fn insert(&self, key: K, val: V) -> (Self, Option<V>) {
        let mut m = self.clone();
        let prev = m.insert_cow(key, val);
        (m, prev)
    }

    /// A new map without `q`, and its value.
    pub fn remove<Q>(&self, q: &Q) -> (Self, Option<V>)
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let mut m = self.clone();
        let prev = m.remove_cow(q);
        (m, prev)
    }

    pub fn insert_many(&self, kvs: impl IntoIterator<Item = (K, V)>) -> Self {
        let mut m = self.clone();
        m.extend(kvs);
        m
    }

    pub fn remove_many<Q>(&self, qs: impl IntoIterator<Item = Q>) -> Self
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let mut m = self.clone();
        for q in qs {
            m.remove_cow(&q);
        }
        m
    }
}

impl<K, V, S> Extend<(K, V)> for Map<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher,
{
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, kvs: I) {
        for (k, v) in kvs {
            self.insert_cow(k, v);
        }
    }
}

impl<K, V, S> FromIterator<(K, V)> for Map<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default,
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(kvs: I) -> Self {
        let mut m = Self::default();
        m.extend(kvs);
        m
    }
}

impl<K, V, S> PartialEq for Map<K, V, S>
where
    K: Hash + Eq,
    V: PartialEq,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        match (&self.root, &other.root) {
            (Some(a), Some(b)) if a.ptr_eq(b) => true,
            _ => {
                self.len() == other.len()
                    && self.iter().all(|(k, v)| other.get(k) == Some(v))
            }
        }
    }
}

impl<K: Hash + Eq, V: Eq, S: BuildHasher> Eq for Map<K, V, S> {}

impl<K: fmt::Debug, V: fmt::Debug, S> fmt::Debug for Map<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<'a, K, V, S> IntoIterator for &'a Map<K, V, S> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

/// A walk of a tree in hash order, with a frame per inner level.
pub struct Iter<'a, K, V> {
    stack: [slice::Iter<'a, Slot<K, V>>; MAX_DEPTH as usize + 1],
    top: usize,
    leaf: slice::Iter<'a, Entry<K, V>>,
    remaining: usize,
}

impl<'a, K, V> Iter<'a, K, V> {
    fn new(root: Option<&'a Node<K, V>>) -> Self {
        let mut stack = array::from_fn(|_| [].iter());
        let mut leaf = [].iter();
        let mut top = 0;
        match root {
            None => (),
            Some(Node::Leaf(l)) => leaf = l.items().iter(),
            Some(Node::Inner(n)) => {
                stack[0] = n.items().iter();
                top = 1;
            }
        }
        let remaining = root.map_or(0, Node::len);
        Self { stack, top, leaf, remaining }
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        loop {
            if let Some(e) = self.leaf.next() {
                self.remaining -= 1;
                return Some((&e.key, &e.val));
            }
            match self.stack[..self.top].last_mut()?.next() {
                None => self.top -= 1,
                Some(Slot::Entry(e)) => {
                    self.remaining -= 1;
                    return Some((&e.key, &e.val));
                }
                Some(Slot::Node(Node::Leaf(l))) => self.leaf = l.items().iter(),
                Some(Slot::Node(Node::Inner(n))) => {
                    self.stack[self.top] = n.items().iter();
                    self.top += 1;
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {}
impl<K, V> FusedIterator for Iter<'_, K, V> {}

/// A borrowed node of a map's tree, for a codec that must reproduce
/// the tree's sharing. Two views with the same [`identity`] are the
/// same node; a [`NodeHandle`] from [`keep`] pins that identity for as
/// long as it is held.
///
/// [`identity`]: NodeRef::identity
/// [`keep`]: NodeRef::keep
pub struct NodeRef<'a, K, V>(&'a Node<K, V>);

impl<K, V> Clone for NodeRef<'_, K, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K, V> Copy for NodeRef<'_, K, V> {}

/// What a node holds.
pub enum Contents<'a, K, V> {
    /// Slots in fragment order.
    Inner(Slots<'a, K, V>),
    /// Pairs in hash order.
    Leaf(Pairs<'a, K, V>),
}

/// A slot of an inner node: a lone pair, or a subtree.
pub enum SlotRef<'a, K, V> {
    Entry(&'a K, &'a V),
    Node(NodeRef<'a, K, V>),
}

pub struct Slots<'a, K, V>(slice::Iter<'a, Slot<K, V>>);

impl<'a, K, V> Iterator for Slots<'a, K, V> {
    type Item = SlotRef<'a, K, V>;

    fn next(&mut self) -> Option<SlotRef<'a, K, V>> {
        Some(match self.0.next()? {
            Slot::Entry(e) => SlotRef::Entry(&e.key, &e.val),
            Slot::Node(n) => SlotRef::Node(NodeRef(n)),
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K, V> ExactSizeIterator for Slots<'_, K, V> {}

pub struct Pairs<'a, K, V>(slice::Iter<'a, Entry<K, V>>);

impl<'a, K, V> Iterator for Pairs<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        self.0.next().map(|e| (&e.key, &e.val))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K, V> ExactSizeIterator for Pairs<'_, K, V> {}

impl<'a, K, V> NodeRef<'a, K, V> {
    /// The node's allocation address: equal for two views of one node,
    /// distinct for two nodes that are both alive.
    pub fn identity(&self) -> usize {
        self.0.addr()
    }

    pub fn keep(&self) -> NodeHandle<K, V> {
        NodeHandle(self.0.clone())
    }

    /// The node's level; the root is at 0.
    pub fn depth(&self) -> u8 {
        self.0.depth()
    }

    /// The pairs in the node's subtree.
    pub fn subtree_len(&self) -> usize {
        self.0.len()
    }

    pub fn contents(&self) -> Contents<'a, K, V> {
        let node: &'a Node<K, V> = self.0;
        match node {
            Node::Inner(n) => Contents::Inner(Slots(n.items().iter())),
            Node::Leaf(n) => Contents::Leaf(Pairs(n.items().iter())),
        }
    }
}

/// An owned node, built by [`inner`](NodeHandle::inner) or
/// [`leaf`](NodeHandle::leaf), or kept from a [`NodeRef`]. A map is
/// assembled from handles with [`Map::from_root`].
pub struct NodeHandle<K, V>(Node<K, V>);

impl<K, V> Clone for NodeHandle<K, V> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<K, V> fmt::Debug for NodeHandle<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = self.view();
        write!(
            f,
            "NodeHandle({:#x}, depth {}, {} pairs)",
            v.identity(),
            v.depth(),
            v.subtree_len()
        )
    }
}

/// A slot of an inner node to [create](NodeHandle::inner).
pub enum NewSlot<K, V> {
    Entry(K, V),
    Node(NodeHandle<K, V>),
}

impl<K, V> NodeHandle<K, V> {
    pub fn view(&self) -> NodeRef<'_, K, V> {
        NodeRef(&self.0)
    }

    /// The inner node at `depth` holding `slots`, exactly as a
    /// [`NodeRef`] reported them from a map using a hasher that hashes
    /// like `hasher`. Fails on any node that map could not have built.
    pub fn inner<S: BuildHasher>(
        hasher: &S,
        depth: u8,
        slots: impl IntoIterator<Item = NewSlot<K, V>>,
    ) -> Result<Self>
    where
        K: Hash + Eq,
    {
        ensure!(depth <= MAX_DEPTH, "depth {depth} is below the deepest inner level");
        let mut out = Vec::new();
        let mut bitmap = 0;
        let mut len = 0;
        let mut first_prefix = None;
        for s in slots {
            let slot = match s {
                NewSlot::Entry(key, val) => {
                    Slot::Entry(Entry { hash: hash_of(hasher, &key), key, val })
                }
                NewSlot::Node(child) => {
                    ensure!(child.0.depth() == depth + 1, "a child is one level down");
                    Slot::Node(child.0)
                }
            };
            let (hash, bit) = (slot.hash(), bit(slot.hash(), depth));
            let p = prefix(hash, depth);
            ensure!(*first_prefix.get_or_insert(p) == p, "a node's slots share a prefix");
            ensure!(bitmap < bit, "slots are in fragment order, one per fragment");
            bitmap |= bit;
            len += slot.len();
            out.push(slot);
        }
        ensure!(len > LEAF, "an inner node holds more than {LEAF} pairs");
        ensure!(
            !matches!(out[..], [Slot::Node(Node::Leaf(_))]),
            "a lone leaf of one hash is the node itself"
        );
        let cap = out.len();
        Ok(Self(Node::Inner(Raw::new(InnerHead { len, bitmap, depth }, cap, out))))
    }

    /// The leaf at `depth` holding `pairs`, as [`inner`](Self::inner)
    /// does for an inner node.
    pub fn leaf<S: BuildHasher>(
        hasher: &S,
        depth: u8,
        pairs: impl IntoIterator<Item = (K, V)>,
    ) -> Result<Self>
    where
        K: Hash + Eq,
    {
        ensure!(depth <= MAX_DEPTH + 1, "depth {depth} is below the deepest level");
        let entries: Vec<Entry<K, V>> = pairs
            .into_iter()
            .map(|(key, val)| Entry { hash: hash_of(hasher, &key), key, val })
            .collect();
        let n = entries.len();
        let least = if depth > 0 { 2 } else { 1 };
        ensure!(n >= least, "a leaf below the root holds two pairs");
        let (first, last) = (entries[0].hash, entries[n - 1].hash);
        ensure!(n <= LEAF || first == last, "a leaf of many hashes holds at most {LEAF}");
        for (i, e) in entries.iter().enumerate() {
            ensure!(
                prefix(e.hash, depth) == prefix(first, depth),
                "a leaf shares a prefix"
            );
            ensure!(i == 0 || entries[i - 1].hash <= e.hash, "a leaf is in hash order");
            let mut same = entries[..i].iter().rev().take_while(|p| p.hash == e.hash);
            ensure!(same.all(|p| p.key != e.key), "a leaf's keys are distinct");
        }
        Ok(Self(Node::Leaf(leaf(depth, entries))))
    }
}

/// A persistent hash set.
pub struct Set<K, S = FxBuildHasher>(Map<K, (), S>);

impl<K, S: Clone> Clone for Set<K, S> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<K, S: Default> Default for Set<K, S> {
    fn default() -> Self {
        Self(Map::default())
    }
}

impl<K> Set<K> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<K, S> Set<K, S> {
    pub fn with_hasher(hasher: S) -> Self {
        Self(Map::with_hasher(hasher))
    }

    pub fn hasher(&self) -> &S {
        self.0.hasher()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The members in hash order.
    pub fn iter(&self) -> SetIter<'_, K> {
        SetIter(self.0.iter())
    }

    /// See [`Map::root`].
    pub fn root(&self) -> Option<NodeRef<'_, K, ()>> {
        self.0.root()
    }

    /// See [`Map::from_root`].
    pub fn from_root(root: Option<NodeHandle<K, ()>>, hasher: S) -> Result<Self> {
        Ok(Self(Map::from_root(root, hasher)?))
    }
}

impl<K: Hash + Eq, S: BuildHasher> Set<K, S> {
    pub fn get<Q>(&self, q: &Q) -> Option<&K>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.get_key(q)
    }

    pub fn contains<Q>(&self, q: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.contains_key(q)
    }
}

impl<K: Hash + Eq + Clone, S: BuildHasher> Set<K, S> {
    /// Insert in place, copying only the parts of the tree shared with
    /// another version. True if `key` was already a member, in which
    /// case nothing is copied.
    pub fn insert_cow(&mut self, key: K) -> bool {
        self.contains(&key) || self.0.insert_cow(key, ()).is_some()
    }

    /// Remove in place, copying only the parts of the tree shared with
    /// another version. True if `q` was a member.
    pub fn remove_cow<Q>(&mut self, q: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.remove_cow(q).is_some()
    }
}

impl<K: Hash + Eq + Clone, S: BuildHasher + Clone> Set<K, S> {
    /// A new set with `key`, and whether it was already a member.
    pub fn insert(&self, key: K) -> (Self, bool) {
        let mut s = self.clone();
        let present = s.insert_cow(key);
        (s, present)
    }

    /// A new set without `q`, and whether it was a member.
    pub fn remove<Q>(&self, q: &Q) -> (Self, bool)
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let mut s = self.clone();
        let present = s.remove_cow(q);
        (s, present)
    }

    pub fn insert_many(&self, ks: impl IntoIterator<Item = K>) -> Self {
        let mut s = self.clone();
        s.extend(ks);
        s
    }

    pub fn remove_many<Q>(&self, qs: impl IntoIterator<Item = Q>) -> Self
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        Self(self.0.remove_many(qs))
    }
}

impl<K: Hash + Eq + Clone, S: BuildHasher> Extend<K> for Set<K, S> {
    fn extend<I: IntoIterator<Item = K>>(&mut self, ks: I) {
        for k in ks {
            self.insert_cow(k);
        }
    }
}

impl<K: Hash + Eq + Clone, S: BuildHasher + Default> FromIterator<K> for Set<K, S> {
    fn from_iter<I: IntoIterator<Item = K>>(ks: I) -> Self {
        let mut s = Self::default();
        s.extend(ks);
        s
    }
}

impl<K: Hash + Eq, S: BuildHasher> PartialEq for Set<K, S> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<K: Hash + Eq, S: BuildHasher> Eq for Set<K, S> {}

impl<K: fmt::Debug, S> fmt::Debug for Set<K, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<'a, K, S> IntoIterator for &'a Set<K, S> {
    type Item = &'a K;
    type IntoIter = SetIter<'a, K>;

    fn into_iter(self) -> SetIter<'a, K> {
        self.iter()
    }
}

pub struct SetIter<'a, K>(Iter<'a, K, ()>);

impl<'a, K> Iterator for SetIter<'a, K> {
    type Item = &'a K;

    fn next(&mut self) -> Option<&'a K> {
        self.0.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K> ExactSizeIterator for SetIter<'_, K> {}
impl<K> FusedIterator for SetIter<'_, K> {}
