use std::{
    array,
    borrow::Borrow,
    hash::{BuildHasher, Hash, Hasher, RandomState},
    mem,
    sync::Arc,
};

#[derive(Clone)]
enum Bucket<K: Eq + Clone, V: Clone> {
    Empty,
    Single(K, V, u64),
    Multi(Vec<(K, V, u64)>),
}

impl<K: Eq + Clone, V: Clone> Default for Bucket<K, V> {
    fn default() -> Self {
        Self::Empty
    }
}

#[derive(Clone)]
struct ChunkInner<K: Eq + Clone, V: Clone, const C: usize> {
    buckets: [Bucket<K, V>; C],
    len: usize,
}

impl<K: Eq + Clone, V: Clone, const C: usize> Default for ChunkInner<K, V, C> {
    fn default() -> Self {
        Self { buckets: array::from_fn(|_| Bucket::Empty), len: 0 }
    }
}

macro_rules! get {
    ($bucket:expr, $q:expr) => {
        match $bucket {
            Bucket::Empty => None,
            Bucket::Single(k, v, _) => {
                if (*k).borrow() == $q {
                    Some(v)
                } else {
                    None
                }
            }
            Bucket::Multi(b) => {
                for (k, v, _) in b {
                    if (*k).borrow() == $q {
                        return Some(v);
                    }
                }
                None
            }
        }
    };
}

impl<K: Eq + Clone, V: Clone, const C: usize> ChunkInner<K, V, C> {
    fn get_mut<Q: Eq>(&mut self, q: &Q, hv: u64) -> Option<&mut V>
    where
        K: Borrow<Q>,
    {
        get!(&mut self.buckets[hv as usize % C], q)
    }

    fn get<Q: Eq>(&self, q: &Q, hv: u64) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        get!(&self.buckets[hv as usize % C], q)
    }

    fn insert(&mut self, k: K, v: V, hv: u64) -> Option<V> {
        let pos = hv as usize % C;
        match mem::take(&mut self.buckets[pos]) {
            Bucket::Empty => {
                self.buckets[pos] = Bucket::Single(k, v, hv);
                self.len += 1;
                None
            }
            Bucket::Single(pre_k, pre_v, _) if pre_k == k => {
                self.buckets[pos] = Bucket::Single(k, v, hv);
                Some(pre_v)
            }
            Bucket::Single(pre_k, pre_v, pre_hv) => {
                self.buckets[pos] =
                    Bucket::Multi(vec![(pre_k, pre_v, pre_hv), (k, v, hv)]);
                self.len += 1;
                None
            }
            Bucket::Multi(mut ents) => {
                for i in 0..ents.len() {
                    if &ents[i].0 == &k {
                        let (_, pre_v, _) = mem::replace(&mut ents[i], (k, v, hv));
                        self.buckets[pos] = Bucket::Multi(ents);
                        return Some(pre_v);
                    }
                }
                ents.push((k, v, hv));
                self.buckets[pos] = Bucket::Multi(ents);
                self.len += 1;
                None
            }
        }
    }

    fn remove<Q: Eq>(&mut self, q: &Q, hv: u64) -> Option<V>
    where
        K: Borrow<Q>,
    {
        let pos = hv as usize % C;
        match mem::take(&mut self.buckets[pos]) {
            Bucket::Empty => None,
            Bucket::Single(k, v, _) if k.borrow() == q => {
                self.len -= 1;
                Some(v)
            }
            Bucket::Single(k, v, hv) => {
                self.buckets[pos] = Bucket::Single(k, v, hv);
                None
            }
            Bucket::Multi(mut ents) => {
                let i = ents.iter().enumerate().find_map(|(i, (k, _, _))| {
                    if k.borrow() == q { Some(i) } else { None }
                });
                let res = match i {
                    None => None,
                    Some(i) => {
                        let (_, v, _) = ents.swap_remove(i);
                        self.len -= 1;
                        Some(v)
                    }
                };
                if ents.len() == 1 {
                    let (k, v, hv) = ents.pop().unwrap();
                    self.buckets[pos] = Bucket::Single(k, v, hv);
                } else {
                    self.buckets[pos] = Bucket::Multi(ents);
                }
                res
            }
        }
    }

    fn drain(&mut self) -> impl Iterator<Item = (K, V, u64)> {
        self.buckets.iter_mut().flat_map(|b| match mem::take(b) {
            Bucket::Empty => vec![],
            Bucket::Single(k, v, hv) => vec![(k, v, hv)],
            Bucket::Multi(b) => b,
        })
    }
}

#[derive(Clone)]
struct Chunk<K: Eq + Clone, V: Clone, const C: usize>(Arc<ChunkInner<K, V, C>>);

impl<K: Eq + Clone, V: Clone, const C: usize> Default for Chunk<K, V, C> {
    fn default() -> Self {
        Self(Arc::new(ChunkInner::default()))
    }
}

impl<K: Eq + Clone, V: Clone, const C: usize> Chunk<K, V, C> {
    fn get<Q: Eq>(&self, q: &Q, hv: u64) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        self.0.get(q, hv)
    }

    fn get_mut_cow<'a, Q: Eq>(&'a mut self, q: &Q, hv: u64) -> Option<&'a mut V>
    where
        K: Borrow<Q>,
    {
        Arc::make_mut(&mut self.0).get_mut(q, hv)
    }

    fn insert_cow(&mut self, k: K, v: V, hv: u64) -> Option<V> {
        Arc::make_mut(&mut self.0).insert(k, v, hv)
    }

    fn remove_cow<Q: Eq>(&mut self, q: &Q, hv: u64) -> Option<V>
    where
        K: Borrow<Q>,
    {
        Arc::make_mut(&mut self.0).remove(q, hv)
    }

    fn drain(&mut self) -> impl Iterator<Item = (K, V, u64)> {
        Arc::make_mut(&mut self.0).drain()
    }

    fn len(&self) -> usize {
        self.0.len
    }
}

#[derive(Clone)]
enum Root<K: Hash + Eq + Clone, V: Clone, const C: usize> {
    Empty,
    Flat(Chunk<K, V, C>),
    Multi(Arc<[Chunk<K, V, C>]>),
}

impl<K: Hash + Eq + Clone, V: Clone, const C: usize> Default for Root<K, V, C> {
    fn default() -> Self {
        Self::Empty
    }
}

#[derive(Clone, Default)]
pub struct IHMap<
    K: Hash + Eq + Clone,
    V: Clone,
    S: Clone + BuildHasher = RandomState,
    const C: usize = 32,
> {
    s: S,
    len: usize,
    cap: usize,
    root: Root<K, V, C>,
}

fn hash_key<Q: Hash + Eq>(s: &impl BuildHasher, q: &Q) -> u64 {
    let mut h = s.build_hasher();
    q.hash(&mut h);
    h.finish()
}

impl<K, V, S, const C: usize> IHMap<K, V, S, C>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: Clone + BuildHasher,
{
    pub fn get<Q: Hash + Eq>(&self, q: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        match &self.root {
            Root::Empty => None,
            Root::Flat(c) => c.get(q, hash_key(&self.s, q)),
            Root::Multi(chunks) => {
                let hv = hash_key(&self.s, q);
                let ch = hv as usize % chunks.len();
                chunks[ch].get(q, hv)
            }
        }
    }

    pub fn get_mut_cow<'a, Q: Hash + Eq>(&'a mut self, q: &Q) -> Option<&'a mut V>
    where
        K: Borrow<Q>,
    {
        match &mut self.root {
            Root::Empty => None,
            Root::Flat(c) => c.get_mut_cow(q, hash_key(&self.s, q)),
            Root::Multi(chunks) => {
                let hv = hash_key(&self.s, q);
                let chunks = Arc::make_mut(chunks);
                let ch = hv as usize % chunks.len();
                chunks[ch].get_mut_cow(q, hv)
            }
        }
    }

    fn grow(&mut self) {
        let (mut new_chunks, mut old_chunks) = match mem::take(&mut self.root) {
            Root::Empty => unreachable!(),
            Root::Flat(c) => {
                let new_chunks: Vec<Chunk<K, V, C>> =
                    (0..2).into_iter().map(|_| Chunk::default()).collect();
                (new_chunks, Arc::from_iter([c]))
            }
            Root::Multi(old) => {
                let new_chunks: Vec<Chunk<K, V, C>> =
                    (0..old.len() << 1).into_iter().map(|_| Chunk::default()).collect();
                (new_chunks, old)
            }
        };
        let mut mut_new_chunks: Vec<&mut ChunkInner<K, V, C>> =
            new_chunks.iter_mut().map(|c| Arc::make_mut(&mut c.0)).collect();
        let len = mut_new_chunks.len();
        let mut cap = 0;
        let iter = Arc::make_mut(&mut old_chunks).iter_mut().flat_map(|c| {
            cap += C;
            c.drain()
        });
        for (k, v, hv) in iter {
            let pos = hv as usize % len;
            mut_new_chunks[pos].insert(k, v, hv);
        }
        self.cap = cap;
        self.root = Root::Multi(Arc::from(new_chunks))
    }

    fn check_grow(&mut self) {
        if self.len >= self.cap >> 1 {
            self.grow()
        }
    }

    pub fn insert_cow(&mut self, k: K, v: V) -> Option<V> {
        let hv = hash_key(&self.s, &k);
        let res = match &mut self.root {
            Root::Empty => {
                let mut c = ChunkInner::default();
                c.insert(k, v, hv);
                self.root = Root::Flat(Chunk(Arc::new(c)));
                self.len += 1;
                self.cap = C;
                None
            }
            Root::Flat(c) => {
                let res = c.insert_cow(k, v, hv);
                self.len = c.len();
                res
            }
            Root::Multi(chunks) => {
                let i = hv as usize % chunks.len();
                let chunks = Arc::make_mut(chunks);
                self.len -= chunks[i].len();
                let res = chunks[i].insert_cow(k, v, hv);
                self.len += chunks[i].len();
                res
            }
        };
        self.check_grow();
        res
    }

    pub fn insert(&self, k: K, v: V) -> (Self, Option<V>) {
        let mut t = self.clone();
        let res = t.insert_cow(k, v);
        (t, res)
    }

    pub fn remove_cow<Q: Hash + Eq>(&mut self, q: &Q) -> Option<V>
    where
        K: Borrow<Q>,
    {
        match &mut self.root {
            Root::Empty => None,
            Root::Flat(c) => {
                let res = c.remove_cow(q, hash_key(&self.s, q));
                self.len = c.len();
                res
            }
            Root::Multi(chunks) => {
                let hv = hash_key(&self.s, q);
                let i = hv as usize % chunks.len();
                let chunks = Arc::make_mut(chunks);
                self.len -= chunks[i].len();
                let res = chunks[i].remove_cow(q, hv);
                self.len += chunks[i].len();
                res
            }
        }
    }

    pub fn remove<Q: Hash + Eq>(&self, q: &Q) -> (Self, Option<V>)
    where
        K: Borrow<Q>,
    {
        let mut t = self.clone();
        let res = t.remove_cow(q);
        (t, res)
    }
}
