//! Heap bytes per pair: `mem <u64|str> <n>` builds a map of `n` keys in
//! place with each of imhm, imbl and chunk16, then snapshots it and
//! inserts 100 keys one at a time, keeping every version.
use arcstr::ArcStr;
use compact_str::format_compact;
use rustc_hash::FxBuildHasher;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hash::Hash,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards to the system allocator, counting live bytes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        LIVE.fetch_add(l.size(), Relaxed);
        unsafe { System.alloc(l) }
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

type Imbl<K, V> =
    imbl::GenericHashMap<K, V, FxBuildHasher, imbl::shared_ptr::DefaultSharedPtr>;
type Chunk16<K, V> = immutable_chunkmap::map::Map<K, V, 16>;

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x ^ (x >> 31)
}

/// Bytes per pair of the map `build` makes, and of the 100 versions
/// `snap` makes from it.
fn measure<M>(
    n: usize,
    build: impl FnOnce() -> M,
    snap: impl FnOnce(&M) -> Vec<M>,
) -> (f64, f64) {
    let before = LIVE.load(Relaxed);
    let m = build();
    let built = LIVE.load(Relaxed) - before;
    let versions = snap(&m);
    let snapped = LIVE.load(Relaxed) - before - built;
    drop(versions);
    (built as f64 / n as f64, snapped as f64 / 100.)
}

fn run<K: Hash + Eq + Ord + Clone>(ks: &[K], extra: &[K]) {
    let n = ks.len();
    let (b, s) = measure(
        n,
        || {
            let mut m = imhm::Map::<K, u64>::new();
            ks.iter().for_each(|k| _ = m.insert_cow(k.clone(), 1));
            m
        },
        |m| {
            let mut v = vec![m.clone()];
            extra.iter().for_each(|k| v.push(v[v.len() - 1].insert(k.clone(), 0).0));
            v
        },
    );
    println!("imhm     {b:7.1} bytes/pair built, {s:8.0} bytes/snapshot insert");
    let (b, s) = measure(
        n,
        || {
            let mut m = Imbl::<K, u64>::default();
            ks.iter().for_each(|k| _ = m.insert(k.clone(), 1));
            m
        },
        |m| {
            let mut v = vec![m.clone()];
            extra.iter().for_each(|k| v.push(v[v.len() - 1].update(k.clone(), 0)));
            v
        },
    );
    println!("imbl     {b:7.1} bytes/pair built, {s:8.0} bytes/snapshot insert");
    let (b, s) = measure(
        n,
        || {
            let mut m = Chunk16::<K, u64>::new();
            ks.iter().for_each(|k| _ = m.insert_cow(k.clone(), 1));
            m
        },
        |m| {
            let mut v = vec![m.clone()];
            extra.iter().for_each(|k| v.push(v[v.len() - 1].insert(k.clone(), 0).0));
            v
        },
    );
    println!("chunk16  {b:7.1} bytes/pair built, {s:8.0} bytes/snapshot insert");
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: u64 = a[2].parse().unwrap();
    match a[1].as_str() {
        "u64" => {
            let ks: Vec<u64> = (0..n + 100).map(mix).collect();
            run(&ks[..n as usize], &ks[n as usize..])
        }
        _ => {
            let ks: Vec<ArcStr> = (0..n + 100)
                .map(|i| {
                    let x = mix(i);
                    let s = format_compact!(
                        "/std/pkg{}/module/name_{:08x}",
                        x % 4,
                        x as u32 >> 2
                    );
                    ArcStr::from(s.as_str())
                })
                .collect();
            run(&ks[..n as usize], &ks[n as usize..])
        }
    }
}
