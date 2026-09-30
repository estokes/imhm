//! Operations on a leaf's tags or order: 32 bytes, a lane per entry.
//! SSE2 on x86_64 and NEON on aarch64, both in the baseline of their
//! architecture; scalar elsewhere.

use crate::LEAF;

pub(crate) type Lanes = [u8; LEAF];

/// What [`remove`] puts in the last lane.
pub(crate) const GONE: u8 = 0xff;

/// A set of lanes, a bit per lane: bit `i` for lane `i`, except on
/// aarch64, whose compares narrow most cheaply to lane `j` at bit `4j`
/// and lane `j + 16` at bit `4j + 2`, for `j` below 16.
#[derive(Clone, Copy)]
pub(crate) struct Mask(Bits);

#[cfg(target_arch = "aarch64")]
type Bits = u64;
#[cfg(not(target_arch = "aarch64"))]
type Bits = u32;

/// The lane of bit `b`.
#[cfg(target_arch = "aarch64")]
#[inline]
fn lane(b: u32) -> usize {
    (b / 4 + 16 * (b >> 1 & 1)) as usize
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn lane(b: u32) -> usize {
    b as usize
}

impl Mask {
    #[inline]
    pub(crate) fn count(self) -> usize {
        self.0.count_ones() as usize
    }

    #[inline]
    pub(crate) fn first(self) -> Option<usize> {
        (self.0 != 0).then(|| lane(self.0.trailing_zeros()))
    }

    /// Bit `i` set for lane `i`.
    #[cfg(test)]
    pub(crate) fn bits(self) -> u32 {
        self.fold(0, |m, i| m | 1 << i)
    }
}

impl Iterator for Mask {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        let i = self.first()?;
        self.0 &= self.0 - 1;
        Some(i)
    }
}

#[cfg(target_arch = "x86_64")]
mod imp {
    use super::{Lanes, Mask};
    use std::arch::x86_64::{
        __m128i, _mm_and_si128, _mm_andnot_si128, _mm_cmpeq_epi8, _mm_cmplt_epi8,
        _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128, _mm_set1_epi8, _mm_setr_epi8,
        _mm_slli_si128, _mm_srli_si128, _mm_storeu_si128, _mm_xor_si128,
    };

    // SAFETY, for every block here: SSE2 is in the x86_64 baseline, and
    // each unaligned load or store touches only the 32 bytes of `x`.

    #[inline]
    fn load(x: &Lanes) -> [__m128i; 2] {
        unsafe {
            [
                _mm_loadu_si128(x.as_ptr().cast()),
                _mm_loadu_si128(x.as_ptr().add(16).cast()),
            ]
        }
    }

    #[inline]
    fn store(x: &mut Lanes, [lo, hi]: [__m128i; 2]) {
        unsafe {
            _mm_storeu_si128(x.as_mut_ptr().cast(), lo);
            _mm_storeu_si128(x.as_mut_ptr().add(16).cast(), hi);
        }
    }

    #[inline]
    fn mask([lo, hi]: [__m128i; 2]) -> Mask {
        unsafe {
            Mask(_mm_movemask_epi8(lo) as u32 | (_mm_movemask_epi8(hi) as u32) << 16)
        }
    }

    /// Each lane's index, for comparing with `i`.
    #[inline]
    fn index() -> [__m128i; 2] {
        unsafe {
            let lo = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            [lo, _mm_or_si128(lo, _mm_set1_epi8(16))]
        }
    }

    #[inline]
    pub(crate) fn eq(x: &Lanes, t: u8) -> Mask {
        let [lo, hi] = load(x);
        unsafe {
            let t = _mm_set1_epi8(t as i8);
            mask([_mm_cmpeq_epi8(lo, t), _mm_cmpeq_epi8(hi, t)])
        }
    }

    /// Unsigned bytes compare as signed once their top bits flip.
    #[inline]
    pub(crate) fn below(x: &Lanes, t: u8) -> Mask {
        let [lo, hi] = load(x);
        unsafe {
            let flip = _mm_set1_epi8(i8::MIN);
            let t = _mm_set1_epi8((t ^ 0x80) as i8);
            let lo = _mm_cmplt_epi8(_mm_xor_si128(lo, flip), t);
            let hi = _mm_cmplt_epi8(_mm_xor_si128(hi, flip), t);
            mask([lo, hi])
        }
    }

    #[inline]
    pub(crate) fn insert(x: &mut Lanes, i: usize, v: u8) {
        let [lo, hi] = load(x);
        let [ilo, ihi] = index();
        unsafe {
            let up = [
                _mm_slli_si128(lo, 1),
                _mm_or_si128(_mm_slli_si128(hi, 1), _mm_srli_si128(lo, 15)),
            ];
            let (at, v) = (_mm_set1_epi8(i as i8), _mm_set1_epi8(v as i8));
            let pick = |x, up, idx| {
                let (before, here) = (_mm_cmplt_epi8(idx, at), _mm_cmpeq_epi8(idx, at));
                let kept = _mm_or_si128(_mm_and_si128(before, x), _mm_and_si128(here, v));
                _mm_or_si128(kept, _mm_andnot_si128(_mm_or_si128(before, here), up))
            };
            store(x, [pick(lo, up[0], ilo), pick(hi, up[1], ihi)]);
        }
    }

    #[inline]
    pub(crate) fn remove(x: &mut Lanes, i: usize) {
        let [lo, hi] = load(x);
        let [ilo, ihi] = index();
        unsafe {
            let gone = _mm_slli_si128(_mm_set1_epi8(super::GONE as i8), 15);
            let down = [
                _mm_or_si128(_mm_srli_si128(lo, 1), _mm_slli_si128(hi, 15)),
                _mm_or_si128(_mm_srli_si128(hi, 1), gone),
            ];
            let at = _mm_set1_epi8(i as i8);
            let pick = |x, down, idx| {
                let before = _mm_cmplt_epi8(idx, at);
                _mm_or_si128(_mm_and_si128(before, x), _mm_andnot_si128(before, down))
            };
            store(x, [pick(lo, down[0], ilo), pick(hi, down[1], ihi)]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod imp {
    use super::{Lanes, Mask};
    use std::arch::aarch64::{
        uint8x16_t, vbsl_u8, vbslq_u8, vceqq_u8, vcltq_u8, vdup_n_u8, vdupq_n_u8,
        vextq_u8, vget_lane_u64, vld1q_u8, vreinterpret_u64_u8, vreinterpretq_u16_u8,
        vshrn_n_u16, vst1q_u8,
    };

    // SAFETY, for every block here: NEON is in the aarch64 baseline, and
    // each load or store touches only the bytes of `x` or a constant.

    #[inline]
    fn load(x: &Lanes) -> [uint8x16_t; 2] {
        unsafe { [vld1q_u8(x.as_ptr()), vld1q_u8(x.as_ptr().add(16))] }
    }

    #[inline]
    fn store(x: &mut Lanes, [lo, hi]: [uint8x16_t; 2]) {
        unsafe {
            vst1q_u8(x.as_mut_ptr(), lo);
            vst1q_u8(x.as_mut_ptr().add(16), hi);
        }
    }

    /// Narrowing each pair of all-ones or zero lanes by 4 bits leaves a
    /// nibble per lane; each nibble keeps two bits of a low lane and two
    /// of the high lane 16 on, then one of each.
    #[inline]
    fn mask([lo, hi]: [uint8x16_t; 2]) -> Mask {
        unsafe {
            let lo = vshrn_n_u16(vreinterpretq_u16_u8(lo), 4);
            let hi = vshrn_n_u16(vreinterpretq_u16_u8(hi), 4);
            let both = vbsl_u8(vdup_n_u8(0x33), lo, hi);
            Mask(vget_lane_u64(vreinterpret_u64_u8(both), 0) & 0x5555_5555_5555_5555)
        }
    }

    /// Each lane's index, for comparing with `i`.
    #[inline]
    fn index() -> [uint8x16_t; 2] {
        const INDEX: [u8; 32] = {
            let mut a = [0; 32];
            let mut i = 0;
            while i < 32 {
                a[i] = i as u8;
                i += 1;
            }
            a
        };
        load(&INDEX)
    }

    #[inline]
    pub(crate) fn eq(x: &Lanes, t: u8) -> Mask {
        let [lo, hi] = load(x);
        unsafe {
            let t = vdupq_n_u8(t);
            mask([vceqq_u8(lo, t), vceqq_u8(hi, t)])
        }
    }

    #[inline]
    pub(crate) fn below(x: &Lanes, t: u8) -> Mask {
        let [lo, hi] = load(x);
        unsafe {
            let t = vdupq_n_u8(t);
            mask([vcltq_u8(lo, t), vcltq_u8(hi, t)])
        }
    }

    #[inline]
    pub(crate) fn insert(x: &mut Lanes, i: usize, v: u8) {
        let [lo, hi] = load(x);
        let [ilo, ihi] = index();
        unsafe {
            let up = [vextq_u8(vdupq_n_u8(0), lo, 15), vextq_u8(lo, hi, 15)];
            let (at, v) = (vdupq_n_u8(i as u8), vdupq_n_u8(v));
            let pick = |x, up, idx| {
                let moved = vbslq_u8(vceqq_u8(idx, at), v, up);
                vbslq_u8(vcltq_u8(idx, at), x, moved)
            };
            store(x, [pick(lo, up[0], ilo), pick(hi, up[1], ihi)]);
        }
    }

    #[inline]
    pub(crate) fn remove(x: &mut Lanes, i: usize) {
        let [lo, hi] = load(x);
        let [ilo, ihi] = index();
        unsafe {
            let down = [vextq_u8(lo, hi, 1), vextq_u8(hi, vdupq_n_u8(super::GONE), 1)];
            let at = vdupq_n_u8(i as u8);
            let pick = |x, down, idx| vbslq_u8(vcltq_u8(idx, at), x, down);
            store(x, [pick(lo, down[0], ilo), pick(hi, down[1], ihi)]);
        }
    }
}

#[cfg(any(test, not(any(target_arch = "x86_64", target_arch = "aarch64"))))]
mod scalar {
    use super::Lanes;
    use crate::LEAF;

    /// A byte of `x ^ t` is zero exactly where its top bit stays clear
    /// both in it and in its low bits plus 0x7f, which cannot carry out.
    pub(crate) fn eq(x: &Lanes, t: u8) -> u32 {
        const LOW: u64 = 0x7f7f_7f7f_7f7f_7f7f;
        let mut m = 0;
        for (w, chunk) in x.as_chunks::<8>().0.iter().enumerate() {
            let x = u64::from_le_bytes(*chunk) ^ (0x0101_0101_0101_0101 * t as u64);
            let zero = !(((x & LOW) + LOW) | x | LOW);
            m |= (((zero >> 7).wrapping_mul(0x0102_0408_1020_4080) >> 56) as u32)
                << (8 * w);
        }
        m
    }

    pub(crate) fn below(x: &Lanes, t: u8) -> u32 {
        x.iter().enumerate().fold(0, |m, (i, &x)| m | ((x < t) as u32) << i)
    }

    pub(crate) fn insert(x: &mut Lanes, i: usize, v: u8) {
        x.copy_within(i..LEAF - 1, i + 1);
        x[i] = v;
    }

    pub(crate) fn remove(x: &mut Lanes, i: usize) {
        x.copy_within(i + 1.., i);
        x[LEAF - 1] = super::GONE;
    }
}

#[cfg(test)]
pub(crate) use scalar::{
    below as scalar_below, eq as scalar_eq, insert as scalar_insert,
    remove as scalar_remove,
};

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod imp {
    use super::{Lanes, Mask, scalar};

    pub(crate) fn eq(x: &Lanes, t: u8) -> Mask {
        Mask(scalar::eq(x, t))
    }

    pub(crate) fn below(x: &Lanes, t: u8) -> Mask {
        Mask(scalar::below(x, t))
    }

    pub(crate) use scalar::{insert, remove};
}

/// The lanes where `x[i] == t`.
pub(crate) use imp::eq;

/// The lanes where `x[i] < t`.
pub(crate) use imp::below;

/// `x` with `v` at `i`, below [`LEAF`], the lanes from `i` moved up one
/// and the last dropped.
pub(crate) use imp::insert;

/// `x` without lane `i`, the lanes after it moved down one and [`GONE`]
/// last.
pub(crate) use imp::remove;
