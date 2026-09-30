//! imhm against imbl's HashMap and immutable-chunkmap at graphix's
//! chunk size, on key shapes the graphix compiler uses.

use arcstr::ArcStr;
use compact_str::{CompactString, format_compact};
use immutable_chunkmap::map::Map as CMap;
use rustc_hash::FxBuildHasher;
use std::{hint::black_box, time::Instant};

type Imhm<K, V> = imhm::Map<K, V>;
type Imbl<K, V> =
    imbl::GenericHashMap<K, V, FxBuildHasher, imbl::shared_ptr::DefaultSharedPtr>;
type Chunk16<K, V> = CMap<K, V, 16>;

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x ^ (x >> 31)
}

/// Best of 5, in ns per op.
fn time<R>(ops: usize, mut f: impl FnMut() -> R) -> f64 {
    (0..5)
        .map(|_| {
            let t = Instant::now();
            black_box(f());
            t.elapsed().as_nanos() as f64 / ops as f64
        })
        .fold(f64::MAX, f64::min)
}

macro_rules! run {
    ($kty:ty, $label:expr, $n:expr, $mk:expr) => {{
        let n: usize = $n;
        let m = 2000usize;
        let all: Vec<$kty> = (0..(n + m) as u64).map(|i| $mk(mix(i))).collect();
        let (ks, extra) = all.split_at(n);
        let r = m.min(n);
        let mut h: Imhm<$kty, u64> = Imhm::default();
        let mut i: Imbl<$kty, u64> = Imbl::default();
        let mut c: Chunk16<$kty, u64> = Chunk16::new();
        let build = [
            time(n, || {
                let mut t: Imhm<$kty, u64> = Imhm::default();
                for k in ks {
                    t.insert_cow(k.clone(), 0);
                }
                t
            }),
            time(n, || {
                let mut t: Imbl<$kty, u64> = Imbl::default();
                for k in ks {
                    t.insert(k.clone(), 0);
                }
                t
            }),
            time(n, || {
                let mut t: Chunk16<$kty, u64> = Chunk16::new();
                for k in ks {
                    t.insert_cow(k.clone(), 0);
                }
                t
            }),
        ];
        for k in ks {
            h.insert_cow(k.clone(), 1);
            i.insert(k.clone(), 1);
            c.insert_cow(k.clone(), 1);
        }
        let get = [
            time(n, || ks.iter().map(|k| *h.get(k).unwrap()).sum::<u64>()),
            time(n, || ks.iter().map(|k| *i.get(k).unwrap()).sum::<u64>()),
            time(n, || ks.iter().map(|k| *c.get(k).unwrap()).sum::<u64>()),
        ];
        let ins = [
            time(m, || {
                let mut cur = h.clone();
                for k in extra {
                    cur = cur.insert(k.clone(), 0).0;
                }
                cur
            }),
            time(m, || {
                let mut cur = i.clone();
                for k in extra {
                    cur = cur.update(k.clone(), 0);
                }
                cur
            }),
            time(m, || {
                let mut cur = c.clone();
                for k in extra {
                    cur = cur.insert(k.clone(), 0).0;
                }
                cur
            }),
        ];
        let rm = [
            time(r, || {
                let mut cur = h.clone();
                for k in &ks[..r] {
                    cur = cur.remove(k).0;
                }
                cur
            }),
            time(r, || {
                let mut cur = i.clone();
                for k in &ks[..r] {
                    cur = cur.without(k);
                }
                cur
            }),
            time(r, || {
                let mut cur = c.clone();
                for k in &ks[..r] {
                    cur = cur.remove(k).0;
                }
                cur
            }),
        ];
        let row = |xs: [f64; 3]| format!("{:>6.0} {:>6.0} {:>6.0}", xs[0], xs[1], xs[2]);
        println!(
            "{:<14} {:>7} | {} | {} | {} | {}",
            $label,
            n,
            row(build),
            row(get),
            row(ins),
            row(rm)
        );
    }};
}

fn main() {
    println!(
        "ns/op, best of 5; columns imhm imbl chunk16\n{:<14} {:>7} | {:^20} | {:^20} | {:^20} | {:^20}",
        "keys", "N", "insert_cow (build)", "get", "snapshot+insert", "snapshot+remove"
    );
    for n in [100usize, 1_000, 10_000, 100_000, 1_000_000] {
        run!(u64, "u64", n, |x: u64| x);
        run!(CompactString, "CompactString", n, |x: u64| format_compact!(
            "name_{:08x}",
            x as u32
        ));
        run!(ArcStr, "ArcStr path", n, |x: u64| ArcStr::from(
            format_compact!("/std/pkg{}/module/name_{:08x}", x % 4, x as u32 >> 2)
                .as_str()
        ));
    }
}
