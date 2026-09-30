//! Work for `perf stat`: `getcount <imhm|imbl|chunk16> <u64|str> <n> <rounds> <op>`
//! builds a map of `n` keys, then `rounds` times does `op`: `get` looks
//! each key up; `build` builds the map again in place; `snap` inserts 100
//! new keys, each into a fresh snapshot.
use arcstr::ArcStr;
use compact_str::format_compact;
use rustc_hash::FxBuildHasher;
use std::{hash::Hash, hint::black_box};

type Imbl<K, V> =
    imbl::GenericHashMap<K, V, FxBuildHasher, imbl::shared_ptr::DefaultSharedPtr>;
type Chunk16<K, V> = immutable_chunkmap::map::Map<K, V, 16>;

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x ^ (x >> 31)
}

fn run<K: Hash + Eq + Ord + Clone>(
    kind: &str,
    ks: &[K],
    extra: &[K],
    rounds: usize,
    op: &str,
) -> u64 {
    let mut s = 0u64;
    match kind {
        "imhm" => {
            let m: imhm::Map<K, u64> = ks.iter().map(|k| (k.clone(), 1)).collect();
            for _ in 0..rounds {
                match op {
                    "get" => {
                        for k in ks {
                            s += *m.get(black_box(k)).unwrap()
                        }
                    }
                    "build" => {
                        let mut t = imhm::Map::<K, u64>::new();
                        for k in ks {
                            t.insert_cow(k.clone(), 1);
                        }
                        s += t.len() as u64
                    }
                    _ => {
                        let mut t = m.clone();
                        for k in extra {
                            t = t.insert(k.clone(), 0).0
                        }
                        s += t.len() as u64
                    }
                }
            }
        }
        "imbl" => {
            let m: Imbl<K, u64> = ks.iter().map(|k| (k.clone(), 1)).collect();
            for _ in 0..rounds {
                match op {
                    "get" => {
                        for k in ks {
                            s += *m.get(black_box(k)).unwrap()
                        }
                    }
                    "build" => {
                        let mut t = Imbl::<K, u64>::default();
                        for k in ks {
                            t.insert(k.clone(), 1);
                        }
                        s += t.len() as u64
                    }
                    _ => {
                        let mut t = m.clone();
                        for k in extra {
                            t = t.update(k.clone(), 0)
                        }
                        s += t.len() as u64
                    }
                }
            }
        }
        _ => {
            let m: Chunk16<K, u64> = ks.iter().map(|k| (k.clone(), 1)).collect();
            for _ in 0..rounds {
                match op {
                    "get" => {
                        for k in ks {
                            s += *m.get(black_box(k)).unwrap()
                        }
                    }
                    "build" => {
                        let mut t = Chunk16::<K, u64>::new();
                        for k in ks {
                            t.insert_cow(k.clone(), 1);
                        }
                        s += t.len() as u64
                    }
                    _ => {
                        let mut t = m.clone();
                        for k in extra {
                            t = t.insert(k.clone(), 0).0
                        }
                        s += t.len() as u64
                    }
                }
            }
        }
    }
    s
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (n, rounds, op): (u64, usize, &str) =
        (a[3].parse().unwrap(), a[4].parse().unwrap(), &a[5]);
    let s = match a[2].as_str() {
        "u64" => {
            let ks: Vec<u64> = (0..n + 100).map(mix).collect();
            run(&a[1], &ks[..n as usize], &ks[n as usize..], rounds, op)
        }
        _ => {
            let ks: Vec<ArcStr> = (0..n + 100)
                .map(|i| {
                    let x = mix(i);
                    ArcStr::from(
                        format_compact!(
                            "/std/pkg{}/module/name_{:08x}",
                            x % 4,
                            x as u32 >> 2
                        )
                        .as_str(),
                    )
                })
                .collect();
            run(&a[1], &ks[..n as usize], &ks[n as usize..], rounds, op)
        }
    };
    println!("{s}");
}
