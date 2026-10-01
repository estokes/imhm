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
use node::Raw;
pub use rustc_hash::FxBuildHasher;
use std::{
    array,
    borrow::Borrow,
    fmt,
    hash::{BuildHasher, Hash},
    iter::FusedIterator,
    mem, slice, vec,
};

mod lanes;
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
/// The room a new leaf of two has.
const PAIR_CAP: usize = 8;
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

/// `hash`'s fragment at `depth`, high bits first, so a trie's order is
/// its hashes' order.
#[inline]
fn frag(hash: u64, depth: u8) -> usize {
    ((hash << (BITS * depth as u32)) >> 59) as usize
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

#[derive(Clone)]
struct Entry<K, V> {
    hash: u64,
    key: K,
    val: V,
}

/// `bitmap` has the bits of the fragments whose slots are not empty.
#[derive(Clone, Copy)]
struct InnerHead {
    len: usize,
    bitmap: u32,
    depth: u8,
}

/// For the first [`LEAF`] entries: `tags[i]` is the [`tag`] of entry
/// `i`'s hash, and `order` lists their indices in hash order. Past the
/// entries, the tags are 0 and the order [`lanes::GONE`].
#[derive(Clone, Copy)]
struct LeafHead {
    depth: u8,
    tags: [u8; LEAF],
    order: [u8; LEAF],
}

/// A slot per fragment, holding more than [`LEAF`] pairs of at least two
/// hashes.
type Inner<K, V> = Raw<InnerHead, Slot<K, V>>;

fn slots<K, V>(n: &Inner<K, V>) -> &[Slot<K, V>; 32] {
    n.items().try_into().expect("a slot per fragment")
}

/// At least two entries below the root, and at most [`LEAF`] unless all
/// share one hash.
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
    Empty,
    Entry(Entry<K, V>),
    Node(Node<K, V>),
}

fn empty_slots<K, V>() -> [Slot<K, V>; 32] {
    array::from_fn(|_| Slot::Empty)
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
            Node::Inner(n) => {
                slots(n)[n.header().bitmap.trailing_zeros() as usize].hash()
            }
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
    /// A hash from a slot not empty.
    fn hash(&self) -> u64 {
        match self {
            Slot::Empty => unreachable!("an empty slot has no hash"),
            Slot::Entry(e) => e.hash,
            Slot::Node(n) => n.hash(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Slot::Empty => 0,
            Slot::Entry(_) => 1,
            Slot::Node(n) => n.len(),
        }
    }
}

/// The byte of `hash` just below the fragments above `depth`, or 1 for
/// 0, so a leaf's tags are in the order of its hashes and none is 0.
#[inline]
fn tag(hash: u64, depth: u8) -> u8 {
    top_tag(hash.checked_shl(BITS * depth as u32).unwrap_or(0))
}

/// The tag of a hash whose fragments above the leaf are shifted out.
#[inline]
fn top_tag(rest: u64) -> u8 {
    ((rest >> 56) as u8).max(1)
}

/// The header of a leaf at `depth` of `entries`, in hash order.
fn leaf_head<'a, K: 'a, V: 'a>(
    depth: u8,
    entries: impl IntoIterator<Item = &'a Entry<K, V>>,
) -> LeafHead {
    let (mut tags, mut order) = ([0; LEAF], [lanes::GONE; LEAF]);
    for (i, e) in entries.into_iter().take(LEAF).enumerate() {
        tags[i] = tag(e.hash, depth);
        order[i] = i as u8;
    }
    LeafHead { depth, tags, order }
}

/// The leaf at `depth` of `entries`, in hash order.
fn leaf<K, V>(depth: u8, entries: Vec<Entry<K, V>>) -> Leaf<K, V> {
    let cap = entries.len();
    Raw::new(leaf_head(depth, &entries), cap, entries)
}

/// The leaf at `depth` of `a` and `b`.
fn pair<K, V>(depth: u8, a: Entry<K, V>, b: Entry<K, V>) -> Leaf<K, V> {
    let (a, b) = if b.hash < a.hash { (b, a) } else { (a, b) };
    Raw::new(leaf_head(depth, [&a, &b]), PAIR_CAP, [a, b])
}

/// A leaf's entries in hash order, moved out if no other version holds
/// it, which is left empty.
fn sorted_items<K: Clone, V: Clone>(l: &mut Leaf<K, V>) -> Vec<Entry<K, V>> {
    let order = l.header().order;
    let mut v = l.take_items();
    let mut placed = 0u64;
    for k in 0..v.len().min(LEAF) {
        let mut j = k;
        while placed & 1 << j == 0 {
            placed |= 1 << j;
            let from = order[j] as usize;
            if from == k {
                break;
            }
            v.swap(j, from);
            j = from;
        }
    }
    v
}

/// The subtree at `depth` of the next `n` of `entries`, at least two,
/// in hash order.
fn build<K, V>(
    depth: u8,
    entries: &mut vec::Drain<'_, Entry<K, V>>,
    n: usize,
) -> Node<K, V> {
    let s = &entries.as_slice()[..n];
    if n <= LEAF || s[0].hash == s[n - 1].hash {
        let h = leaf_head(depth, s);
        return Node::Leaf(Raw::new(h, n, entries.by_ref().take(n)));
    }
    let mut slots = empty_slots();
    let mut bitmap = 0;
    let mut left = n;
    while left > 0 {
        let s = &entries.as_slice()[..left];
        let f = frag(s[0].hash, depth);
        let k = s.iter().take_while(|e| frag(e.hash, depth) == f).count();
        slots[f] = match k {
            1 => Slot::Entry(entries.next().expect("a group's entry")),
            k => Slot::Node(build(depth + 1, entries, k)),
        };
        bitmap |= 1 << f;
        left -= k;
    }
    Node::Inner(Raw::new(InnerHead { len: n, bitmap, depth }, 32, slots))
}

/// Every entry under `node`, in hash order, moved out of the nodes no
/// other version holds, which are left empty.
fn drain_into<K: Clone, V: Clone>(
    node: &mut Node<K, V>,
    out: &mut Vec<Entry<K, V>>,
    g: &mut Grave<K, V>,
) {
    match node {
        Node::Leaf(l) => out.extend(sorted_items(l)),
        Node::Inner(n) => {
            for slot in n.take_items() {
                match slot {
                    Slot::Empty => (),
                    Slot::Entry(e) => out.push(e),
                    Slot::Node(mut c) => {
                        drain_into(&mut c, out, g);
                        g.bury(c)
                    }
                }
            }
        }
    }
}

/// Nodes a change replaced whose last handle it held. Dropping one drops
/// its entries, running user code that may panic, so a change finishes
/// its grave only once the tree is whole again.
struct Grave<K, V> {
    leaves: Vec<Leaf<K, V>>,
    inners: Vec<Inner<K, V>>,
}

impl<K, V> Grave<K, V> {
    fn new() -> Self {
        Self { leaves: Vec::new(), inners: Vec::new() }
    }

    fn bury(&mut self, n: Node<K, V>) {
        match n {
            Node::Leaf(l) => l.release(&mut self.leaves),
            Node::Inner(n) => n.release(&mut self.inners),
        }
    }

    /// Drops what was buried; the tree must be whole again.
    #[inline]
    fn finish(self) {
        if self.leaves.capacity() == 0 && self.inners.capacity() == 0 {
            mem::forget(self)
        } else {
            drop(self)
        }
    }
}

/// Makes an inner node that now holds at most [`LEAF`] pairs, or only
/// one leaf, a leaf.
fn settle<K: Clone, V: Clone>(node: &mut Node<K, V>, g: &mut Grave<K, V>) {
    let Node::Inner(n) = node else { return };
    let h = *n.header();
    if h.len > LEAF && !lone_leaf(n) {
        return;
    }
    let (depth, len) = (h.depth, h.len);
    let mut entries = Vec::with_capacity(len);
    drain_into(node, &mut entries, g);
    g.bury(mem::replace(node, build(depth, &mut entries.drain(..), len)));
}

/// Whether the node's only slot is a leaf, whose entries share one hash.
fn lone_leaf<K, V>(n: &Inner<K, V>) -> bool {
    let bitmap = n.header().bitmap;
    bitmap.count_ones() == 1
        && matches!(slots(n)[bitmap.trailing_zeros() as usize], Slot::Node(Node::Leaf(_)))
}

/// Whether comparing keys costs about as much as comparing hashes, so a
/// tag match goes straight to the key.
#[inline]
const fn cheap_eq<K>() -> bool {
    !mem::needs_drop::<K>() && mem::size_of::<K>() <= 16
}

/// The index of `q`, whose hash is `hash` with tag `t` here.
#[inline]
fn leaf_index<K, V, Q>(l: &Leaf<K, V>, t: u8, hash: u64, q: &Q) -> Option<usize>
where
    K: Borrow<Q>,
    Q: Eq + ?Sized,
{
    let entries = l.items();
    leaf_match(entries, same_tag(l, t), hash, q)
}

/// The entries tagged `t`, which is not 0.
#[inline]
fn same_tag<K, V>(l: &Leaf<K, V>, t: u8) -> lanes::Mask {
    lanes::eq(&l.header().tags, t)
}

/// The index of `q` among `entries`, `same` holding those tagged like
/// it.
#[inline]
fn leaf_match<K, V, Q>(
    entries: &[Entry<K, V>],
    same: lanes::Mask,
    hash: u64,
    q: &Q,
) -> Option<usize>
where
    K: Borrow<Q>,
    Q: Eq + ?Sized,
{
    for i in same {
        let e = &entries[i];
        if (cheap_eq::<K>() || e.hash == hash) && e.key.borrow() == q {
            return Some(i);
        }
    }
    if entries.len() > LEAF {
        return untagged_index(entries, hash, q);
    }
    None
}

/// The index of `q` among the entries past the first [`LEAF`] of a leaf
/// whose entries all share one hash.
#[cold]
fn untagged_index<K, V, Q>(entries: &[Entry<K, V>], hash: u64, q: &Q) -> Option<usize>
where
    K: Borrow<Q>,
    Q: Eq + ?Sized,
{
    (entries[0].hash == hash)
        .then(|| entries[LEAF..].iter().position(|e| e.key.borrow() == q))
        .flatten()
        .map(|i| i + LEAF)
}

/// `n` is a root: children are one level below their parent, so the
/// fragment at each level is the next 5 bits of `hash`.
#[inline]
fn find<'a, K, V, Q>(mut n: &'a Node<K, V>, hash: u64, q: &Q) -> Option<(&'a K, &'a V)>
where
    K: Borrow<Q>,
    Q: Eq + ?Sized,
{
    let mut rest = hash;
    loop {
        match n {
            Node::Leaf(l) => {
                let i = leaf_index(l, top_tag(rest), hash, q)?;
                // SAFETY: `leaf_index` returns an index of an entry.
                let e = unsafe { l.items().get_unchecked(i) };
                return Some((&e.key, &e.val));
            }
            Node::Inner(inner) => {
                let f = (rest >> 59) as usize;
                rest <<= BITS;
                match &slots(inner)[f] {
                    Slot::Empty => return None,
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
    g: &mut Grave<K, V>,
) -> Option<V> {
    match node {
        Node::Inner(inner) => {
            let mut m = inner.make_mut(&mut g.inners);
            let (h, slots) = m.parts();
            let f = frag(hash, h.depth);
            let slot = &mut slots[f];
            let prev = match slot {
                Slot::Empty => {
                    *slot = Slot::Entry(Entry { hash, key, val });
                    h.bitmap |= 1 << f;
                    None
                }
                Slot::Entry(e) if e.hash == hash && e.key == key => {
                    return without(key, mem::replace(&mut e.val, val));
                }
                Slot::Entry(_) => {
                    let Slot::Entry(old) = mem::replace(slot, Slot::Empty) else {
                        unreachable!("an entry, matched above")
                    };
                    let new = Entry { hash, key, val };
                    *slot = Slot::Node(Node::Leaf(pair(h.depth + 1, old, new)));
                    None
                }
                Slot::Node(child) => insert(child, hash, key, val, g),
            };
            if prev.is_none() {
                h.len += 1
            }
            prev
        }
        Node::Leaf(l) => {
            let t = tag(hash, l.header().depth);
            let same = same_tag(l, t);
            if let Some(i) = leaf_match(l.items(), same, hash, &key) {
                let slot = &mut l.make_mut(&mut g.leaves).into_items()[i].val;
                return without(key, mem::replace(slot, val));
            }
            let (entries, h) = (l.items(), l.header());
            let n = entries.len();
            let new = Entry { hash, key, val };
            if n < LEAF {
                let unused = LEAF - n;
                let mut rank = lanes::below(&h.tags, t).count() - unused;
                for i in same {
                    rank += (entries[i].hash <= hash) as usize;
                }
                l.push(new, &mut g.leaves);
                let mut m = l.make_mut(&mut g.leaves);
                let h = m.header();
                h.tags[n] = t;
                lanes::insert(&mut h.order, rank, n as u8);
            } else if [0, LEAF - 1].map(|k| entries[h.order[k] as usize].hash)
                == [hash; 2]
            {
                l.push(new, &mut g.leaves);
            } else {
                let depth = h.depth;
                let mut entries = sorted_items(l);
                let at = entries.partition_point(|e| e.hash <= hash);
                entries.insert(at, new);
                g.bury(mem::replace(node, build(depth, &mut entries.drain(..), n + 1)));
            }
            None
        }
    }
}

/// `val`, once `key` is dropped: a destructor that panics while a
/// function returns leaks the return value (rust-lang/rust#47949).
fn without<K, V>(key: K, val: V) -> Option<V> {
    drop(key);
    Some(val)
}

/// Makes every node under `node` one no other version holds, copying
/// those shared, so taking the subtree apart runs no user code.
fn unshare<K: Clone, V: Clone>(node: &mut Node<K, V>, g: &mut Grave<K, V>) {
    match node {
        Node::Leaf(l) => drop(l.make_mut(&mut g.leaves)),
        Node::Inner(n) => {
            for slot in n.make_mut(&mut g.inners).into_items() {
                if let Slot::Node(c) = slot {
                    unshare(c, g)
                }
            }
        }
    }
}

/// Whether removing the key in slot `f` collapses `n` into a leaf.
fn collapses<K, V>(n: &Inner<K, V>, f: usize) -> bool {
    let (h, s) = (n.header(), slots(n));
    let other = (h.bitmap & !(1 << f)).trailing_zeros() as usize;
    h.len - 1 <= LEAF
        || (h.bitmap.count_ones() == 2
            && matches!(s[f], Slot::Entry(_))
            && matches!(s.get(other), Some(Slot::Node(Node::Leaf(_)))))
}

/// Remove `q`, which must be present: a miss would copy the path for
/// nothing. The entry is returned whole, so its key is dropped only once
/// the tree is whole again. Every copy is made before anything moves,
/// so a panicking clone leaves the map as it was.
fn remove<K, V, Q>(
    node: &mut Node<K, V>,
    hash: u64,
    q: &Q,
    g: &mut Grave<K, V>,
) -> Option<Entry<K, V>>
where
    K: Borrow<Q> + Clone,
    V: Clone,
    Q: Eq + ?Sized,
{
    match node {
        Node::Leaf(l) => {
            let i = leaf_index(l, tag(hash, l.header().depth), hash, q)?;
            let n = l.items().len();
            let e = l.swap_remove(i, &mut g.leaves);
            if n <= LEAF {
                let mut m = l.make_mut(&mut g.leaves);
                let h = m.header();
                let rank = lanes::eq(&h.order, i as u8).first();
                lanes::remove(&mut h.order, rank.expect("each entry has a rank"));
                h.tags[i] = h.tags[n - 1];
                h.tags[n - 1] = 0;
                if i != n - 1 {
                    let moved = lanes::eq(&h.order, (n - 1) as u8).first();
                    h.order[moved.expect("each entry has a rank")] = i as u8;
                }
            }
            Some(e)
        }
        Node::Inner(inner) => {
            let f = frag(hash, inner.header().depth);
            if matches!(slots(inner)[f], Slot::Empty) {
                return None;
            }
            if collapses(inner, f) {
                unshare(node, g);
            }
            let Node::Inner(inner) = node else { unreachable!("still inner") };
            let mut m = inner.make_mut(&mut g.inners);
            let (h, slots) = m.parts();
            let slot = &mut slots[f];
            let v = match slot {
                Slot::Entry(e) if e.hash == hash && e.key.borrow() == q => {
                    let Slot::Entry(e) = mem::replace(slot, Slot::Empty) else {
                        unreachable!("an entry, matched above")
                    };
                    h.bitmap &= !(1 << f);
                    e
                }
                Slot::Empty | Slot::Entry(_) => return None,
                Slot::Node(child) => {
                    let e = remove(child, hash, q, g)?;
                    match child {
                        Node::Leaf(l) if l.items().len() == 1 => {
                            let last = Slot::Entry(l.swap_remove(0, &mut g.leaves));
                            if let Slot::Node(old) = mem::replace(slot, last) {
                                g.bury(old)
                            }
                        }
                        _ => settle(child, g),
                    }
                    e
                }
            };
            h.len -= 1;
            Some(v)
        }
    }
}

/// The value of `q`, which must be present: a miss would copy the path
/// for nothing.
fn get_mut<'a, K, V, Q>(
    node: &'a mut Node<K, V>,
    hash: u64,
    q: &Q,
    g: &mut Grave<K, V>,
) -> Option<&'a mut V>
where
    K: Borrow<Q> + Clone,
    V: Clone,
    Q: Eq + ?Sized,
{
    match node {
        Node::Leaf(l) => {
            let i = leaf_index(l, tag(hash, l.header().depth), hash, q)?;
            Some(&mut l.make_mut(&mut g.leaves).into_items()[i].val)
        }
        Node::Inner(inner) => {
            let f = frag(hash, inner.header().depth);
            match &mut inner.make_mut(&mut g.inners).into_items()[f] {
                Slot::Empty => None,
                Slot::Entry(e) => {
                    (e.hash == hash && e.key.borrow() == q).then_some(&mut e.val)
                }
                Slot::Node(child) => get_mut(child, hash, q, g),
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

    /// Whether the two share their whole tree, so they hold the same
    /// pairs without comparing any; false says nothing.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        match (&self.root, &other.root) {
            (Some(a), Some(b)) => a.ptr_eq(b),
            (a, b) => a.is_none() && b.is_none(),
        }
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
            Some(root) => {
                let mut g = Grave::new();
                let prev = insert(root, hash, key, val, &mut g);
                g.finish();
                prev
            }
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
        let mut g = Grave::new();
        let e = remove(root, hash, q, &mut g)?;
        match root.len() {
            0 => g.bury(self.root.take().expect("a root")),
            _ => settle(root, &mut g),
        }
        g.finish();
        without(e.key, e.val)
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
        let mut g = Grave::new();
        let v = get_mut(root, hash, q, &mut g);
        g.finish();
        v
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
        self.len() == other.len() && self.iter().all(|(k, v)| other.get(k) == Some(v))
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

/// A leaf's entries in hash order.
struct Ordered<'a, K, V> {
    entries: &'a [Entry<K, V>],
    order: slice::Iter<'a, u8>,
    rest: slice::Iter<'a, Entry<K, V>>,
}

impl<'a, K, V> Ordered<'a, K, V> {
    fn new(l: &'a Leaf<K, V>) -> Self {
        let entries = l.items();
        let tagged = entries.len().min(LEAF);
        let order = l.header().order[..tagged].iter();
        Self { entries, order, rest: entries[tagged..].iter() }
    }

    fn empty() -> Self {
        Self { entries: &[], order: [].iter(), rest: [].iter() }
    }
}

impl<'a, K, V> Iterator for Ordered<'a, K, V> {
    type Item = &'a Entry<K, V>;

    #[inline]
    fn next(&mut self) -> Option<&'a Entry<K, V>> {
        match self.order.next() {
            Some(&i) => Some(&self.entries[i as usize]),
            None => self.rest.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.order.len() + self.rest.len();
        (n, Some(n))
    }
}

/// An inner node's slots that are not empty, in fragment order.
struct Occupied<'a, K, V> {
    slots: &'a [Slot<K, V>],
    bits: u32,
}

impl<'a, K, V> Occupied<'a, K, V> {
    fn new(n: &'a Inner<K, V>) -> Self {
        Self { slots: n.items(), bits: n.header().bitmap }
    }
}

impl<K, V> Clone for Occupied<'_, K, V> {
    fn clone(&self) -> Self {
        Self { slots: self.slots, bits: self.bits }
    }
}

impl<'a, K, V> Iterator for Occupied<'a, K, V> {
    type Item = &'a Slot<K, V>;

    #[inline]
    fn next(&mut self) -> Option<&'a Slot<K, V>> {
        let f = self.bits.trailing_zeros() as usize;
        self.bits &= self.bits.wrapping_sub(1);
        self.slots.get(f)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.bits.count_ones() as usize;
        (n, Some(n))
    }
}

/// A walk of a tree in hash order, with a frame per inner level.
pub struct Iter<'a, K, V> {
    stack: [Occupied<'a, K, V>; MAX_DEPTH as usize + 1],
    top: usize,
    leaf: Ordered<'a, K, V>,
    remaining: usize,
}

impl<'a, K, V> Iter<'a, K, V> {
    fn new(root: Option<&'a Node<K, V>>) -> Self {
        let mut stack = array::from_fn(|_| Occupied { slots: &[], bits: 0 });
        let mut leaf = Ordered::empty();
        let mut top = 0;
        match root {
            None => (),
            Some(Node::Leaf(l)) => leaf = Ordered::new(l),
            Some(Node::Inner(n)) => {
                stack[0] = Occupied::new(n);
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
                Some(Slot::Empty) => unreachable!("occupied slots are not empty"),
                Some(Slot::Entry(e)) => {
                    self.remaining -= 1;
                    return Some((&e.key, &e.val));
                }
                Some(Slot::Node(Node::Leaf(l))) => self.leaf = Ordered::new(l),
                Some(Slot::Node(Node::Inner(n))) => {
                    self.stack[self.top] = Occupied::new(n);
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

pub struct Slots<'a, K, V>(Occupied<'a, K, V>);

impl<'a, K, V> Iterator for Slots<'a, K, V> {
    type Item = SlotRef<'a, K, V>;

    fn next(&mut self) -> Option<SlotRef<'a, K, V>> {
        Some(match self.0.next()? {
            Slot::Empty => unreachable!("occupied slots are not empty"),
            Slot::Entry(e) => SlotRef::Entry(&e.key, &e.val),
            Slot::Node(n) => SlotRef::Node(NodeRef(n)),
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K, V> ExactSizeIterator for Slots<'_, K, V> {}

pub struct Pairs<'a, K, V>(Ordered<'a, K, V>);

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
            Node::Inner(n) => Contents::Inner(Slots(Occupied::new(n))),
            Node::Leaf(n) => Contents::Leaf(Pairs(Ordered::new(n))),
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
        let mut out = empty_slots();
        let mut bitmap = 0u32;
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
            let (hash, f) = (slot.hash(), frag(slot.hash(), depth));
            let p = prefix(hash, depth);
            ensure!(*first_prefix.get_or_insert(p) == p, "a node's slots share a prefix");
            ensure!(bitmap < 1 << f, "slots are in fragment order, one per fragment");
            bitmap |= 1 << f;
            len += slot.len();
            out[f] = slot;
        }
        let n = Raw::new(InnerHead { len, bitmap, depth }, 32, out);
        ensure!(len > LEAF, "an inner node holds more than {LEAF} pairs");
        ensure!(!lone_leaf(&n), "a lone leaf of one hash is the node itself");
        Ok(Self(Node::Inner(n)))
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

    /// Whether the two share their whole tree, as [`Map::ptr_eq`].
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.0.ptr_eq(&other.0)
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
