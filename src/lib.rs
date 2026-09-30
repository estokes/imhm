//! Persistent hash maps and sets: hash array mapped tries whose nodes
//! are shared between versions. A clone is O(1); an update copies the
//! path from the root to the key and shares everything else, and a map
//! no other version shares is updated in place.
//!
//! A key set has one tree shape for a given hasher, and iteration
//! follows the shape, so with a deterministic hasher (the default) the
//! order is the same in every process. [`Map::root`] and
//! [`NodeHandle::create`] expose the tree to a codec that must
//! reproduce its sharing.

use anyhow::{Result, ensure};
use rustc_hash::FxBuildHasher;
use std::{
    array,
    borrow::Borrow,
    fmt,
    hash::{BuildHasher, Hash},
    iter::FusedIterator,
    mem, slice,
};
use triomphe::Arc;

#[cfg(test)]
mod test;

const BITS: u32 = 5;
/// The deepest level. Its fragment is the hash's last 4 bits, so two
/// distinct hashes part at or above it.
const MAX_DEPTH: u8 = 12;
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

#[inline]
fn bit(hash: u64, depth: u8) -> u32 {
    1 << ((hash >> (BITS * depth as u32)) & 0x1f)
}

#[inline]
fn index(bitmap: u32, bit: u32) -> usize {
    (bitmap & (bit - 1)).count_ones() as usize
}

/// The fragments above `depth`, which every hash under a node there
/// shares.
#[inline]
fn prefix(hash: u64, depth: u8) -> u64 {
    hash & ((1 << (BITS * depth as u32)) - 1)
}

#[derive(Clone)]
struct Entry<K, V> {
    hash: u64,
    key: K,
    val: V,
}

/// Two or more distinct keys with one hash.
#[derive(Clone)]
struct Collision<K, V> {
    hash: u64,
    pairs: Vec<(K, V)>,
}

#[derive(Clone)]
enum Slot<K, V> {
    Entry(Entry<K, V>),
    Collision(Arc<Collision<K, V>>),
    Node(Arc<Node<K, V>>),
}

impl<K, V> Slot<K, V> {
    /// A hash from the slot. Every hash under a child node shares its
    /// fragments down to the child's depth, so any one places it.
    fn hash(&self) -> u64 {
        match self {
            Slot::Entry(e) => e.hash,
            Slot::Collision(c) => c.hash,
            Slot::Node(n) => n.slots[0].hash(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Slot::Entry(_) => 1,
            Slot::Collision(c) => c.pairs.len(),
            Slot::Node(n) => n.len,
        }
    }
}

/// Slots are in fragment order, one per bit of `bitmap`. Below the
/// root a node is never a lone entry or collision: that lives in its
/// parent's slot instead, so a key set has one shape.
#[derive(Clone)]
struct Node<K, V> {
    len: usize,
    depth: u8,
    bitmap: u32,
    slots: Vec<Slot<K, V>>,
}

/// The node at `depth` over `a` and `b`, whose hashes differ.
fn join<K, V>(depth: u8, a: Slot<K, V>, b: Slot<K, V>) -> Arc<Node<K, V>> {
    let (ba, bb) = (bit(a.hash(), depth), bit(b.hash(), depth));
    let len = a.len() + b.len();
    let slots = if ba == bb {
        vec![Slot::Node(join(depth + 1, a, b))]
    } else if ba < bb {
        vec![a, b]
    } else {
        vec![b, a]
    };
    Arc::new(Node { len, depth, bitmap: ba | bb, slots })
}

fn find<'a, K, V, Q>(mut n: &'a Node<K, V>, hash: u64, q: &Q) -> Option<(&'a K, &'a V)>
where
    K: Borrow<Q>,
    Q: Eq + ?Sized,
{
    loop {
        let bit = bit(hash, n.depth);
        if n.bitmap & bit == 0 {
            return None;
        }
        match &n.slots[index(n.bitmap, bit)] {
            Slot::Entry(e) => {
                return (e.hash == hash && e.key.borrow() == q)
                    .then_some((&e.key, &e.val));
            }
            Slot::Collision(c) if c.hash == hash => {
                return c
                    .pairs
                    .iter()
                    .find(|(k, _)| k.borrow() == q)
                    .map(|(k, v)| (k, v));
            }
            Slot::Collision(_) => return None,
            Slot::Node(child) => n = child,
        }
    }
}

fn insert<K: Eq + Clone, V: Clone>(
    node: &mut Arc<Node<K, V>>,
    hash: u64,
    key: K,
    val: V,
) -> Option<V> {
    let n = Arc::make_mut(node);
    let bit = bit(hash, n.depth);
    let i = index(n.bitmap, bit);
    if n.bitmap & bit == 0 {
        n.bitmap |= bit;
        n.slots.insert(i, Slot::Entry(Entry { hash, key, val }));
    } else {
        match &mut n.slots[i] {
            Slot::Entry(e) if e.hash == hash && e.key == key => {
                return Some(mem::replace(&mut e.val, val));
            }
            Slot::Collision(c) if c.hash == hash => {
                let c = Arc::make_mut(c);
                match c.pairs.iter_mut().find(|(k, _)| *k == key) {
                    Some((_, v)) => return Some(mem::replace(v, val)),
                    None => c.pairs.push((key, val)),
                }
            }
            Slot::Node(child) => {
                let prev = insert(child, hash, key, val);
                if prev.is_none() {
                    n.len += 1
                }
                return prev;
            }
            Slot::Entry(_) | Slot::Collision(_) => {
                let slot = match n.slots.remove(i) {
                    Slot::Entry(e) if e.hash == hash => {
                        let pairs = vec![(e.key, e.val), (key, val)];
                        Slot::Collision(Arc::new(Collision { hash, pairs }))
                    }
                    old => {
                        let new = Slot::Entry(Entry { hash, key, val });
                        Slot::Node(join(n.depth + 1, old, new))
                    }
                };
                n.slots.insert(i, slot);
            }
        }
    }
    n.len += 1;
    None
}

/// Remove `q`, which must be present: a miss would copy the path for
/// nothing.
fn remove<K, V, Q>(node: &mut Arc<Node<K, V>>, hash: u64, q: &Q) -> Option<(K, V)>
where
    K: Borrow<Q> + Clone,
    V: Clone,
    Q: Eq + ?Sized,
{
    let n = Arc::make_mut(node);
    let bit = bit(hash, n.depth);
    if n.bitmap & bit == 0 {
        return None;
    }
    let i = index(n.bitmap, bit);
    let kv = match &mut n.slots[i] {
        Slot::Entry(e) if e.hash == hash && e.key.borrow() == q => {
            n.bitmap &= !bit;
            let Slot::Entry(e) = n.slots.remove(i) else { unreachable!() };
            (e.key, e.val)
        }
        Slot::Collision(c) if c.hash == hash => {
            let c = Arc::make_mut(c);
            let j = c.pairs.iter().position(|(k, _)| k.borrow() == q)?;
            let kv = c.pairs.swap_remove(j);
            if let [_] = c.pairs[..] {
                let (key, val) = c.pairs.pop()?;
                n.slots[i] = Slot::Entry(Entry { hash, key, val });
            }
            kv
        }
        Slot::Node(child) => {
            let kv = remove(child, hash, q)?;
            if let [Slot::Entry(_) | Slot::Collision(_)] = child.slots[..] {
                n.slots[i] = Arc::make_mut(child).slots.pop()?;
            }
            kv
        }
        Slot::Entry(_) | Slot::Collision(_) => return None,
    };
    n.len -= 1;
    Some(kv)
}

/// The value of `q`, which must be present: a miss would copy the path
/// for nothing.
fn get_mut<'a, K, V, Q>(
    node: &'a mut Arc<Node<K, V>>,
    hash: u64,
    q: &Q,
) -> Option<&'a mut V>
where
    K: Borrow<Q> + Clone,
    V: Clone,
    Q: Eq + ?Sized,
{
    let n = Arc::make_mut(node);
    let bit = bit(hash, n.depth);
    if n.bitmap & bit == 0 {
        return None;
    }
    match &mut n.slots[index(n.bitmap, bit)] {
        Slot::Entry(e) => (e.hash == hash && e.key.borrow() == q).then_some(&mut e.val),
        Slot::Collision(c) if c.hash == hash => Arc::make_mut(c)
            .pairs
            .iter_mut()
            .find(|(k, _)| k.borrow() == q)
            .map(|(_, v)| v),
        Slot::Collision(_) => None,
        Slot::Node(child) => get_mut(child, hash, q),
    }
}

/// A persistent hash map.
pub struct Map<K, V, S = FxBuildHasher> {
    root: Option<Arc<Node<K, V>>>,
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
        self.root.as_ref().map_or(0, |r| r.len)
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
            ensure!(r.0.depth == 0, "a root is at depth 0, not {}", r.0.depth);
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
                let slots = vec![Slot::Entry(Entry { hash, key, val })];
                let root = Node { len: 1, depth: 0, bitmap: bit(hash, 0), slots };
                self.root = Some(Arc::new(root));
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
        let (_, v) = remove(root, hash, q)?;
        if root.slots.is_empty() {
            self.root = None
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
            (Some(a), Some(b)) if Arc::ptr_eq(a, b) => true,
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

/// A walk of a tree in slot order, with a frame per level.
pub struct Iter<'a, K, V> {
    stack: [slice::Iter<'a, Slot<K, V>>; MAX_DEPTH as usize + 1],
    top: usize,
    collision: slice::Iter<'a, (K, V)>,
    remaining: usize,
}

impl<'a, K, V> Iter<'a, K, V> {
    fn new(root: Option<&'a Arc<Node<K, V>>>) -> Self {
        let mut stack = array::from_fn(|_| [].iter());
        let (top, remaining) = match root {
            None => (0, 0),
            Some(r) => {
                stack[0] = r.slots.iter();
                (1, r.len)
            }
        };
        Self { stack, top, collision: [].iter(), remaining }
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        loop {
            if let Some((k, v)) = self.collision.next() {
                self.remaining -= 1;
                return Some((k, v));
            }
            match self.stack[..self.top].last_mut()?.next() {
                None => self.top -= 1,
                Some(Slot::Entry(e)) => {
                    self.remaining -= 1;
                    return Some((&e.key, &e.val));
                }
                Some(Slot::Collision(c)) => self.collision = c.pairs.iter(),
                Some(Slot::Node(n)) => {
                    self.stack[self.top] = n.slots.iter();
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
pub struct NodeRef<'a, K, V>(&'a Arc<Node<K, V>>);

impl<K, V> Clone for NodeRef<'_, K, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K, V> Copy for NodeRef<'_, K, V> {}

/// A slot of a node, in fragment order.
pub enum SlotRef<'a, K, V> {
    Entry(&'a K, &'a V),
    /// Two or more keys with one hash.
    Collision(&'a [(K, V)]),
    Node(NodeRef<'a, K, V>),
}

impl<'a, K, V> NodeRef<'a, K, V> {
    /// The node's allocation address: equal for two views of one node,
    /// distinct for two nodes that are both alive.
    pub fn identity(&self) -> usize {
        Arc::as_ptr(self.0) as usize
    }

    pub fn keep(&self) -> NodeHandle<K, V> {
        NodeHandle(self.0.clone())
    }

    /// The node's level; the root is at 0.
    pub fn depth(&self) -> u8 {
        self.0.depth
    }

    /// The pairs in the node's subtree.
    pub fn subtree_len(&self) -> usize {
        self.0.len
    }

    pub fn slots(
        &self,
    ) -> impl ExactSizeIterator<Item = SlotRef<'a, K, V>> + use<'a, K, V> {
        let node: &'a Node<K, V> = self.0;
        node.slots.iter().map(|s| match s {
            Slot::Entry(e) => SlotRef::Entry(&e.key, &e.val),
            Slot::Collision(c) => SlotRef::Collision(&c.pairs),
            Slot::Node(n) => SlotRef::Node(NodeRef(n)),
        })
    }
}

/// An owned node, built by [`create`](NodeHandle::create) or kept from
/// a [`NodeRef`]. A map is assembled from handles with
/// [`Map::from_root`].
pub struct NodeHandle<K, V>(Arc<Node<K, V>>);

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

/// A slot of a node to [`create`](NodeHandle::create).
pub enum NewSlot<K, V> {
    Entry(K, V),
    Collision(Vec<(K, V)>),
    Node(NodeHandle<K, V>),
}

impl<K, V> NodeHandle<K, V> {
    pub fn view(&self) -> NodeRef<'_, K, V> {
        NodeRef(&self.0)
    }

    /// The node at `depth` holding `slots`, exactly as a [`NodeRef`]
    /// reported them from a map using a hasher that hashes like
    /// `hasher`. Fails on any node that map could not have built.
    pub fn create<S: BuildHasher>(
        hasher: &S,
        depth: u8,
        slots: impl IntoIterator<Item = NewSlot<K, V>>,
    ) -> Result<Self>
    where
        K: Hash + Eq,
    {
        ensure!(depth <= MAX_DEPTH, "depth {depth} is below the deepest level");
        let mut out = Vec::new();
        let mut bitmap = 0;
        let mut len = 0;
        let mut first_prefix = None;
        for s in slots {
            let slot = match s {
                NewSlot::Entry(key, val) => {
                    Slot::Entry(Entry { hash: hash_of(hasher, &key), key, val })
                }
                NewSlot::Collision(pairs) => {
                    ensure!(pairs.len() >= 2, "a collision holds at least two pairs");
                    let hash = hash_of(hasher, &pairs[0].0);
                    for (i, (k, _)) in pairs.iter().enumerate() {
                        ensure!(
                            hash_of(hasher, k) == hash,
                            "a collision's keys share a hash"
                        );
                        ensure!(
                            pairs[..i].iter().all(|(prev, _)| prev != k),
                            "a collision's keys are distinct"
                        );
                    }
                    Slot::Collision(Arc::new(Collision { hash, pairs }))
                }
                NewSlot::Node(child) => {
                    ensure!(child.0.depth == depth + 1, "a child is one level down");
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
        ensure!(!out.is_empty(), "a node holds a slot");
        ensure!(
            depth == 0 || !matches!(out[..], [Slot::Entry(_) | Slot::Collision(_)]),
            "below the root a lone entry or collision lives in its parent"
        );
        Ok(Self(Arc::new(Node { len, depth, bitmap, slots: out })))
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
