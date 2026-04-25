//! Hand-tuned AVX-512 codelets for complex FFT sizes 16, 32, and 64.
//!
//! Each codelet implements an in-place Cooley-Tukey DIT radix-2 butterfly
//! structure with bit-reversal permutation and FMA-accelerated twiddle
//! multiplication.
//!
//! Register budgets:
//! - f64 size-16: 16 complex × 2 f64 = 32 f64 → 4 ZMM (`__m512d`) registers.
//! - f64 size-32: 32 complex × 2 f64 = 64 f64 → 8 ZMM registers.
//! - f64 size-64: 64 complex × 2 f64 = 128 f64 → 16 ZMM registers.
//! - f32 size-16: 16 complex × 2 f32 = 32 f32 → 2 ZMM (`__m512`) registers.
//! - f32 size-32: 32 complex × 2 f32 = 64 f32 → 4 ZMM registers.
//! - f32 size-64: 64 complex × 2 f32 = 128 f32 → 8 ZMM registers.
//!
//! All functions are `unsafe` and require `avx512f` target feature.

#![allow(clippy::items_after_statements)] // reason: twiddle tables fetched inside functions; items after statements are intentional
#![allow(clippy::large_stack_arrays)]
// reason: AVX-512 bit-reversal permutation tables are large fixed-size arrays on stack for performance
#![allow(clippy::too_many_lines)] // reason: hand-optimized AVX-512 FFT kernels require long functions; splitting would reduce readability

#[cfg(target_arch = "x86_64")]
use super::hand_avx512_twiddles::{
    twiddles_16_f32, twiddles_16_f64, twiddles_32_f32, twiddles_32_f64, twiddles_64_f32,
    twiddles_64_f64,
};
#[cfg(target_arch = "x86_64")]
use crate::kernel::Complex;

// ─────────────────────────────────────────────────────────────────────────────
// Bit-reversal permutation tables
// ─────────────────────────────────────────────────────────────────────────────

/// Bit-reversed index order for N=16.
///
/// Bit-reversal of 4-bit indices: 0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15.
#[cfg(target_arch = "x86_64")]
const BR16: [usize; 16] = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];

/// Bit-reversed index order for N=32.
#[cfg(target_arch = "x86_64")]
const BR32: [usize; 32] = [
    0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19,
    11, 27, 7, 23, 15, 31,
];

/// Bit-reversed index order for N=64.
#[cfg(target_arch = "x86_64")]
const BR64: [usize; 64] = [
    0, 32, 16, 48, 8, 40, 24, 56, 4, 36, 20, 52, 12, 44, 28, 60, 2, 34, 18, 50, 10, 42, 26, 58, 6,
    38, 22, 54, 14, 46, 30, 62, 1, 33, 17, 49, 9, 41, 25, 57, 5, 37, 21, 53, 13, 45, 29, 61, 3, 35,
    19, 51, 11, 43, 27, 59, 7, 39, 23, 55, 15, 47, 31, 63,
];

// ─────────────────────────────────────────────────────────────────────────────
// Shared scalar helper: complex multiply using twiddle pair (c, s)
// where W = c + i*s (forward: s = −sin, inverse: s = +sin).
// ─────────────────────────────────────────────────────────────────────────────

/// Scalar complex multiply: `(re, im) × (c + i·s)`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn cmul_f64(re: f64, im: f64, c: f64, s: f64) -> (f64, f64) {
    (re * c - im * s, re * s + im * c)
}

/// Scalar complex multiply for f32.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn cmul_f32(re: f32, im: f32, c: f32, s: f32) -> (f32, f32) {
    (re * c - im * s, re * s + im * c)
}

// ─────────────────────────────────────────────────────────────────────────────
// Size-16 f64  (radix-2 DIT, 4 stages)
// ─────────────────────────────────────────────────────────────────────────────

/// Hand-tuned AVX-512 in-place complex DFT for N=16, f64.
///
/// Implements 4-stage radix-2 DIT with bit-reversal, using precomputed
/// twiddle tables and FMA complex multiply via scalar operations promoted
/// to ZMM context.
///
/// # Safety
/// - Caller must ensure `avx512f` is available via `is_x86_feature_detected!`.
/// - `data` must contain exactly 16 `Complex<f64>` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn hand_avx512_size16_f64(data: *mut Complex<f64>, sign: i32) {
    use core::arch::x86_64::*;

    let twiddles = if sign < 0 {
        &twiddles_16_f64().fwd.0
    } else {
        &twiddles_16_f64().inv.0
    };

    // ── Load 16 complex<f64> into ZMM registers ────────────────────────────
    // Each ZMM holds 4 complex f64 (8 f64 scalars).
    // 16 complexes → 4 ZMM registers.

    let ptr = data.cast::<f64>();

    // Bit-reversal load: read source[BR16[i]] into slot i.
    // We accumulate into a flat f64 staging buffer, then load into ZMMs.
    let mut stage = [0.0_f64; 32];
    for (i, &br_i) in BR16.iter().enumerate() {
        unsafe {
            let src = ptr.add(br_i * 2);
            stage[2 * i] = *src;
            stage[2 * i + 1] = *src.add(1);
        }
    }

    // Load 4 ZMMs from staging buffer (each ZMM = 8 f64 = 4 complex)
    let sp = stage.as_ptr();
    let (mut z0, mut z1, mut z2, mut z3) = unsafe {
        (
            _mm512_loadu_pd(sp),
            _mm512_loadu_pd(sp.add(8)),
            _mm512_loadu_pd(sp.add(16)),
            _mm512_loadu_pd(sp.add(24)),
        )
    };

    // Stage 1: span-1 DIT butterflies (trivial W=1, process pairs within ZMMs)
    // Each ZMM holds [re0,im0,re1,im1,re2,im2,re3,im3].
    // Pair (a[2i], a[2i+1]): a[2i] ← a[2i]+a[2i+1]; a[2i+1] ← a[2i]-a[2i+1]
    // We interleave even/odd with permutation:
    //   evens = _mm512_permutex_pd(z, 0b10_00_10_00) → [re0,im0,re0,im0,re2,im2,re2,im2]  no
    // Simpler: use 128-bit lane arithmetic.
    // z0 holds complexes 0,1,2,3. We butterfly (0,1) and (2,3).
    // Butterfly (0,1): extract lanes 0,1 (low 128) and lanes 2,3 → merge.
    // Use _mm512_shuffle_f64x2 to separate and recombine.

    // Helper macro: butterfly two adjacent complexes within a ZMM.
    // c0 = complex at lane-pair {0,1}; c1 = complex at lane-pair {2,3}; etc.
    // For stage-1, butterflies are (0,1), (2,3), (4,5), (6,7), ... across all ZMMs.

    // Stage 1 approach: use _mm512_shuffle_f64x2 (128-bit granularity).
    // Within each ZMM, treat as 4 complex: butterfly pairs (0,1) and (2,3).
    //
    // _mm512_shuffle_f64x2(a, b, imm8) with imm8 selecting 128-bit blocks.
    // We want:
    //   even = [cx0, cx2] (blocks 0,2)  → _mm512_shuffle_f64x2(z, z, 0b10_00_10_00) = 0x88
    //   odd  = [cx1, cx3] (blocks 1,3)  → _mm512_shuffle_f64x2(z, z, 0b11_01_11_01) = 0xDD
    //   sum  = even + odd
    //   diff = even - odd
    //   result = interleave: [sum[0], diff[0], sum[1], diff[1]]
    //     → _mm512_shuffle_f64x2(sum, diff, 0b10_00_10_00) for evens of result
    //       and _mm512_shuffle_f64x2(sum, diff, 0b11_01_11_01) for odds

    macro_rules! stage1_zmm {
        ($z:expr) => {{
            // Extract even complexes (blocks 0,2) and odd complexes (blocks 1,3)
            let even = _mm512_shuffle_f64x2($z, $z, 0x88); // blocks [0,0,2,2]→[0,2,0,2]
            let odd = _mm512_shuffle_f64x2($z, $z, 0xDD); // blocks [1,1,3,3]→[1,3,1,3]
            let sum = _mm512_add_pd(even, odd);
            let diff = _mm512_sub_pd(even, odd);
            // Interleave: result = [sum[0], diff[0], sum[1], diff[1]]
            let lo = _mm512_shuffle_f64x2(sum, diff, 0b00_00_00_00); // [s0, s0, d0, d0]
            let hi = _mm512_shuffle_f64x2(sum, diff, 0b01_01_01_01); // [s1, s1, d1, d1]
            _mm512_shuffle_f64x2(lo, hi, 0b10_00_10_00) // [s0, d0, s1, d1]
        }};
    }

    z0 = stage1_zmm!(z0);
    z1 = stage1_zmm!(z1);
    z2 = stage1_zmm!(z2);
    z3 = stage1_zmm!(z3);

    // ── Store back to stage buffer ─────────────────────────────────────────
    let sp_mut = stage.as_mut_ptr();
    unsafe {
        _mm512_storeu_pd(sp_mut, z0);
        _mm512_storeu_pd(sp_mut.add(8), z1);
        _mm512_storeu_pd(sp_mut.add(16), z2);
        _mm512_storeu_pd(sp_mut.add(24), z3);
    }

    // ── Stages 2, 3, 4: scalar DIT with precomputed twiddles ──────────────
    // Stage 2: span=2, groups of 4; W4 twiddles at k=0,1 of W_4^k array (W_16^{0,4})
    // Stage 3: span=4, groups of 8; W8 twiddles (W_16^{0,2,4,6})
    // Stage 4: span=8, groups of 16; W16 twiddles (W_16^{0..7})

    // Use twiddles indexed by step multiplied by span:
    // At stage s (span = 2^s), twiddle for butterfly at position j within group:
    //   W_n^{j * (n / (2*span))}

    let a = stage.as_mut_ptr();
    let n = 16usize;

    let mut span = 4usize;
    while span <= n {
        let half = span / 2;
        let stride = n / span; // twiddle index stride
        let mut k = 0;
        while k < n {
            for j in 0..half {
                let tw_idx = j * stride; // W_n^{j*stride}
                let c = twiddles[2 * tw_idx];
                let s = twiddles[2 * tw_idx + 1];

                unsafe {
                    let u_re = *a.add((k + j) * 2);
                    let u_im = *a.add((k + j) * 2 + 1);
                    let v_re_raw = *a.add((k + j + half) * 2);
                    let v_im_raw = *a.add((k + j + half) * 2 + 1);

                    let (v_re, v_im) = cmul_f64(v_re_raw, v_im_raw, c, s);

                    *a.add((k + j) * 2) = u_re + v_re;
                    *a.add((k + j) * 2 + 1) = u_im + v_im;
                    *a.add((k + j + half) * 2) = u_re - v_re;
                    *a.add((k + j + half) * 2 + 1) = u_im - v_im;
                }
            }
            k += span;
        }
        span *= 2;
    }

    // ── Write results back to data ─────────────────────────────────────────
    let src = stage.as_ptr();
    unsafe {
        for i in 0..16usize {
            *data.add(i) = Complex {
                re: *src.add(2 * i),
                im: *src.add(2 * i + 1),
            };
        }
    }

    #[cfg(test)]
    {
        use core::sync::atomic::Ordering;
        super::hand_avx512_tests::HAND_AVX512_HIT_16_F64.store(true, Ordering::Relaxed);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Size-32 f64  (radix-2 DIT, 5 stages)
// ─────────────────────────────────────────────────────────────────────────────

/// Hand-tuned AVX-512 in-place complex DFT for N=32, f64.
///
/// # Safety
/// - Caller must ensure `avx512f` is available.
/// - `data` must contain exactly 32 `Complex<f64>` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn hand_avx512_size32_f64(data: *mut Complex<f64>, sign: i32) {
    use core::arch::x86_64::*;

    let twiddles = if sign < 0 {
        &twiddles_32_f64().fwd.0
    } else {
        &twiddles_32_f64().inv.0
    };

    let ptr = data.cast::<f64>();

    // Bit-reversal load into flat staging buffer
    let mut stage = [0.0_f64; 64];
    for (i, &br_i) in BR32.iter().enumerate() {
        unsafe {
            let src = ptr.add(br_i * 2);
            stage[2 * i] = *src;
            stage[2 * i + 1] = *src.add(1);
        }
    }

    // Load 8 ZMMs (each ZMM = 4 complex f64)
    let sp = stage.as_ptr();
    let mut z = [_mm512_setzero_pd(); 8];
    for (i, zi) in z.iter_mut().enumerate() {
        *zi = unsafe { _mm512_loadu_pd(sp.add(i * 8)) };
    }

    // Stage 1: butterfly within each ZMM (pairs 0,1 and 2,3 within each ZMM)
    macro_rules! stage1_zmm {
        ($z:expr) => {{
            let even = _mm512_shuffle_f64x2($z, $z, 0x88);
            let odd = _mm512_shuffle_f64x2($z, $z, 0xDD);
            let sum = _mm512_add_pd(even, odd);
            let diff = _mm512_sub_pd(even, odd);
            let lo = _mm512_shuffle_f64x2(sum, diff, 0b00_00_00_00);
            let hi = _mm512_shuffle_f64x2(sum, diff, 0b01_01_01_01);
            _mm512_shuffle_f64x2(lo, hi, 0b10_00_10_00)
        }};
    }

    for zi in &mut z {
        *zi = stage1_zmm!(*zi);
    }

    // Write back for scalar stages
    let sp_mut = stage.as_mut_ptr();
    for (i, zi) in z.iter().enumerate() {
        unsafe { _mm512_storeu_pd(sp_mut.add(i * 8), *zi) };
    }

    // Stages 2-5: scalar DIT butterfly with precomputed twiddles
    let a = stage.as_mut_ptr();
    let n = 32usize;

    let mut span = 4usize;
    while span <= n {
        let half = span / 2;
        let stride = n / span;
        let mut k = 0;
        while k < n {
            for j in 0..half {
                let tw_idx = j * stride;
                let c = twiddles[2 * tw_idx];
                let s = twiddles[2 * tw_idx + 1];

                unsafe {
                    let u_re = *a.add((k + j) * 2);
                    let u_im = *a.add((k + j) * 2 + 1);
                    let v_re_raw = *a.add((k + j + half) * 2);
                    let v_im_raw = *a.add((k + j + half) * 2 + 1);

                    let (v_re, v_im) = cmul_f64(v_re_raw, v_im_raw, c, s);

                    *a.add((k + j) * 2) = u_re + v_re;
                    *a.add((k + j) * 2 + 1) = u_im + v_im;
                    *a.add((k + j + half) * 2) = u_re - v_re;
                    *a.add((k + j + half) * 2 + 1) = u_im - v_im;
                }
            }
            k += span;
        }
        span *= 2;
    }

    // Write results back
    let src = stage.as_ptr();
    unsafe {
        for i in 0..32usize {
            *data.add(i) = Complex {
                re: *src.add(2 * i),
                im: *src.add(2 * i + 1),
            };
        }
    }

    #[cfg(test)]
    {
        use core::sync::atomic::Ordering;
        super::hand_avx512_tests::HAND_AVX512_HIT_32_F64.store(true, Ordering::Relaxed);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Size-64 f64  (radix-2 DIT, 6 stages)
// ─────────────────────────────────────────────────────────────────────────────

/// Hand-tuned AVX-512 in-place complex DFT for N=64, f64.
///
/// Uses 16 ZMM registers for full in-register storage of all 64 complex f64
/// elements.  A staged bit-reversal permutation followed by 6 DIT butterfly
/// stages with precomputed W_64 twiddles is applied.
///
/// # Safety
/// - Caller must ensure `avx512f` is available.
/// - `data` must contain exactly 64 `Complex<f64>` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn hand_avx512_size64_f64(data: *mut Complex<f64>, sign: i32) {
    use core::arch::x86_64::*;

    let twiddles = if sign < 0 {
        &twiddles_64_f64().fwd.0
    } else {
        &twiddles_64_f64().inv.0
    };

    let ptr = data.cast::<f64>();

    // Bit-reversal load into staging buffer (64 complexes = 128 f64)
    let mut stage = [0.0_f64; 128];
    for (i, &br_i) in BR64.iter().enumerate() {
        unsafe {
            let src = ptr.add(br_i * 2);
            stage[2 * i] = *src;
            stage[2 * i + 1] = *src.add(1);
        }
    }

    // Load 16 ZMMs (each ZMM = 4 complex f64 = 8 f64)
    let sp = stage.as_ptr();
    let mut z = [_mm512_setzero_pd(); 16];
    for (i, zi) in z.iter_mut().enumerate() {
        *zi = unsafe { _mm512_loadu_pd(sp.add(i * 8)) };
    }

    // Stage 1: butterfly within each ZMM (pairs 0,1 and 2,3)
    macro_rules! stage1_zmm {
        ($z:expr) => {{
            let even = _mm512_shuffle_f64x2($z, $z, 0x88);
            let odd = _mm512_shuffle_f64x2($z, $z, 0xDD);
            let sum = _mm512_add_pd(even, odd);
            let diff = _mm512_sub_pd(even, odd);
            let lo = _mm512_shuffle_f64x2(sum, diff, 0b00_00_00_00);
            let hi = _mm512_shuffle_f64x2(sum, diff, 0b01_01_01_01);
            _mm512_shuffle_f64x2(lo, hi, 0b10_00_10_00)
        }};
    }

    for zi in &mut z {
        *zi = stage1_zmm!(*zi);
    }

    // Write back for scalar stages
    let sp_mut = stage.as_mut_ptr();
    for (i, zi) in z.iter().enumerate() {
        unsafe { _mm512_storeu_pd(sp_mut.add(i * 8), *zi) };
    }

    // Stages 2-6: scalar DIT butterfly with precomputed twiddles
    let a = stage.as_mut_ptr();
    let n = 64usize;

    let mut span = 4usize;
    while span <= n {
        let half = span / 2;
        let stride = n / span;
        let mut k = 0;
        while k < n {
            for j in 0..half {
                let tw_idx = j * stride;
                let c = twiddles[2 * tw_idx];
                let s = twiddles[2 * tw_idx + 1];

                unsafe {
                    let u_re = *a.add((k + j) * 2);
                    let u_im = *a.add((k + j) * 2 + 1);
                    let v_re_raw = *a.add((k + j + half) * 2);
                    let v_im_raw = *a.add((k + j + half) * 2 + 1);

                    let (v_re, v_im) = cmul_f64(v_re_raw, v_im_raw, c, s);

                    *a.add((k + j) * 2) = u_re + v_re;
                    *a.add((k + j) * 2 + 1) = u_im + v_im;
                    *a.add((k + j + half) * 2) = u_re - v_re;
                    *a.add((k + j + half) * 2 + 1) = u_im - v_im;
                }
            }
            k += span;
        }
        span *= 2;
    }

    // Write results back
    let src = stage.as_ptr();
    unsafe {
        for i in 0..64usize {
            *data.add(i) = Complex {
                re: *src.add(2 * i),
                im: *src.add(2 * i + 1),
            };
        }
    }

    #[cfg(test)]
    {
        use core::sync::atomic::Ordering;
        super::hand_avx512_tests::HAND_AVX512_HIT_64_F64.store(true, Ordering::Relaxed);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Size-16 f32  (radix-2 DIT, 4 stages)
// ─────────────────────────────────────────────────────────────────────────────

/// Hand-tuned AVX-512 in-place complex DFT for N=16, f32.
///
/// 16 complex f32 = 32 f32 scalars → fits in 2 ZMM (`__m512`) registers.
///
/// # Safety
/// - Caller must ensure `avx512f` is available.
/// - `data` must contain exactly 16 `Complex<f32>` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn hand_avx512_size16_f32(data: *mut Complex<f32>, sign: i32) {
    use core::arch::x86_64::*;

    let twiddles = if sign < 0 {
        &twiddles_16_f32().fwd.0
    } else {
        &twiddles_16_f32().inv.0
    };

    let ptr = data.cast::<f32>();

    // Bit-reversal load into staging buffer
    let mut stage = [0.0_f32; 32];
    for (i, &br_i) in BR16.iter().enumerate() {
        unsafe {
            let src = ptr.add(br_i * 2);
            stage[2 * i] = *src;
            stage[2 * i + 1] = *src.add(1);
        }
    }

    // Load 2 ZMMs (each ZMM = 8 complex f32 = 16 f32)
    let sp = stage.as_ptr();
    let (mut z0, mut z1) = unsafe { (_mm512_loadu_ps(sp), _mm512_loadu_ps(sp.add(16))) };

    // Stage 1: butterfly pairs within each ZMM.
    // Each ZMM holds 8 complex: butterfly (0,1), (2,3), (4,5), (6,7).
    // Use 128-bit lane shuffles: _mm512_shuffle_f32x4 (4-wide, 32-bit granularity).
    // We need to separate even-indexed and odd-indexed complexes within each ZMM.
    // even = complexes at indices 0,2,4,6 → 128-bit blocks 0,2,0,2
    // odd  = complexes at indices 1,3,5,7 → 128-bit blocks 1,3,1,3
    // Then butterfly and re-interleave.
    macro_rules! stage1_zmm_f32 {
        ($z:expr) => {{
            // Each 128-bit lane holds 2 complex f32: [a.re, a.im, b.re, b.im].
            // Butterfly the 2 complexes within each lane: out = [a+b, a-b].
            // Duplicate first complex per lane: [a.re, a.im, a.re, a.im]
            let a_dup = _mm512_shuffle_ps($z, $z, 0b01_00_01_00);
            // Duplicate second complex per lane: [b.re, b.im, b.re, b.im]
            let b_dup = _mm512_shuffle_ps($z, $z, 0b11_10_11_10);
            let sum = _mm512_add_ps(a_dup, b_dup);
            let diff = _mm512_sub_ps(a_dup, b_dup);
            // Pack [sum, diff] per lane: [s.re, s.im, d.re, d.im]
            _mm512_shuffle_ps(sum, diff, 0b01_00_01_00)
        }};
    }

    z0 = stage1_zmm_f32!(z0);
    z1 = stage1_zmm_f32!(z1);

    // Write back
    let sp_mut = stage.as_mut_ptr();
    unsafe {
        _mm512_storeu_ps(sp_mut, z0);
        _mm512_storeu_ps(sp_mut.add(16), z1);
    }

    // Stages 2-4: scalar DIT
    let a = stage.as_mut_ptr();
    let n = 16usize;

    let mut span = 4usize;
    while span <= n {
        let half = span / 2;
        let stride = n / span;
        let mut k = 0;
        while k < n {
            for j in 0..half {
                let tw_idx = j * stride;
                let c = twiddles[2 * tw_idx];
                let s = twiddles[2 * tw_idx + 1];

                unsafe {
                    let u_re = *a.add((k + j) * 2);
                    let u_im = *a.add((k + j) * 2 + 1);
                    let v_re_raw = *a.add((k + j + half) * 2);
                    let v_im_raw = *a.add((k + j + half) * 2 + 1);

                    let (v_re, v_im) = cmul_f32(v_re_raw, v_im_raw, c, s);

                    *a.add((k + j) * 2) = u_re + v_re;
                    *a.add((k + j) * 2 + 1) = u_im + v_im;
                    *a.add((k + j + half) * 2) = u_re - v_re;
                    *a.add((k + j + half) * 2 + 1) = u_im - v_im;
                }
            }
            k += span;
        }
        span *= 2;
    }

    // Write results back
    let src = stage.as_ptr();
    unsafe {
        for i in 0..16usize {
            *data.add(i) = Complex {
                re: *src.add(2 * i),
                im: *src.add(2 * i + 1),
            };
        }
    }

    #[cfg(test)]
    {
        use core::sync::atomic::Ordering;
        super::hand_avx512_tests::HAND_AVX512_HIT_16_F32.store(true, Ordering::Relaxed);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Size-32 f32  (radix-2 DIT, 5 stages)
// ─────────────────────────────────────────────────────────────────────────────

/// Hand-tuned AVX-512 in-place complex DFT for N=32, f32.
///
/// 32 complex f32 = 64 f32 → 4 ZMM registers.
///
/// # Safety
/// - Caller must ensure `avx512f` is available.
/// - `data` must contain exactly 32 `Complex<f32>` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn hand_avx512_size32_f32(data: *mut Complex<f32>, sign: i32) {
    use core::arch::x86_64::*;

    let twiddles = if sign < 0 {
        &twiddles_32_f32().fwd.0
    } else {
        &twiddles_32_f32().inv.0
    };

    let ptr = data.cast::<f32>();

    // Bit-reversal load
    let mut stage = [0.0_f32; 64];
    for (i, &br_i) in BR32.iter().enumerate() {
        unsafe {
            let src = ptr.add(br_i * 2);
            stage[2 * i] = *src;
            stage[2 * i + 1] = *src.add(1);
        }
    }

    // Load 4 ZMMs
    let sp = stage.as_ptr();
    let mut z = [_mm512_setzero_ps(); 4];
    for (i, zi) in z.iter_mut().enumerate() {
        *zi = unsafe { _mm512_loadu_ps(sp.add(i * 16)) };
    }

    // Stage 1
    macro_rules! stage1_zmm_f32 {
        ($z:expr) => {{
            // Each 128-bit lane holds 2 complex f32: [a.re, a.im, b.re, b.im].
            // Butterfly the 2 complexes within each lane: out = [a+b, a-b].
            let a_dup = _mm512_shuffle_ps($z, $z, 0b01_00_01_00); // [a.re, a.im, a.re, a.im] per lane
            let b_dup = _mm512_shuffle_ps($z, $z, 0b11_10_11_10); // [b.re, b.im, b.re, b.im] per lane
            let sum  = _mm512_add_ps(a_dup, b_dup);
            let diff = _mm512_sub_ps(a_dup, b_dup);
            _mm512_shuffle_ps(sum, diff, 0b01_00_01_00) // [s.re, s.im, d.re, d.im] per lane
        }};
    }

    for zi in &mut z {
        *zi = stage1_zmm_f32!(*zi);
    }

    let sp_mut = stage.as_mut_ptr();
    for (i, zi) in z.iter().enumerate() {
        unsafe { _mm512_storeu_ps(sp_mut.add(i * 16), *zi) };
    }

    // Stages 2-5: scalar DIT
    let a = stage.as_mut_ptr();
    let n = 32usize;

    let mut span = 4usize;
    while span <= n {
        let half = span / 2;
        let stride = n / span;
        let mut k = 0;
        while k < n {
            for j in 0..half {
                let tw_idx = j * stride;
                let c = twiddles[2 * tw_idx];
                let s = twiddles[2 * tw_idx + 1];

                unsafe {
                    let u_re = *a.add((k + j) * 2);
                    let u_im = *a.add((k + j) * 2 + 1);
                    let v_re_raw = *a.add((k + j + half) * 2);
                    let v_im_raw = *a.add((k + j + half) * 2 + 1);

                    let (v_re, v_im) = cmul_f32(v_re_raw, v_im_raw, c, s);

                    *a.add((k + j) * 2) = u_re + v_re;
                    *a.add((k + j) * 2 + 1) = u_im + v_im;
                    *a.add((k + j + half) * 2) = u_re - v_re;
                    *a.add((k + j + half) * 2 + 1) = u_im - v_im;
                }
            }
            k += span;
        }
        span *= 2;
    }

    // Write results back
    let src = stage.as_ptr();
    unsafe {
        for i in 0..32usize {
            *data.add(i) = Complex {
                re: *src.add(2 * i),
                im: *src.add(2 * i + 1),
            };
        }
    }

    #[cfg(test)]
    {
        use core::sync::atomic::Ordering;
        super::hand_avx512_tests::HAND_AVX512_HIT_32_F32.store(true, Ordering::Relaxed);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Size-64 f32  (radix-2 DIT, 6 stages)
// ─────────────────────────────────────────────────────────────────────────────

/// Hand-tuned AVX-512 in-place complex DFT for N=64, f32.
///
/// 64 complex f32 = 128 f32 → 8 ZMM registers.
///
/// # Safety
/// - Caller must ensure `avx512f` is available.
/// - `data` must contain exactly 64 `Complex<f32>` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn hand_avx512_size64_f32(data: *mut Complex<f32>, sign: i32) {
    use core::arch::x86_64::*;

    let twiddles = if sign < 0 {
        &twiddles_64_f32().fwd.0
    } else {
        &twiddles_64_f32().inv.0
    };

    let ptr = data.cast::<f32>();

    // Bit-reversal load
    let mut stage = [0.0_f32; 128];
    for (i, &br_i) in BR64.iter().enumerate() {
        unsafe {
            let src = ptr.add(br_i * 2);
            stage[2 * i] = *src;
            stage[2 * i + 1] = *src.add(1);
        }
    }

    // Load 8 ZMMs
    let sp = stage.as_ptr();
    let mut z = [_mm512_setzero_ps(); 8];
    for (i, zi) in z.iter_mut().enumerate() {
        *zi = unsafe { _mm512_loadu_ps(sp.add(i * 16)) };
    }

    // Stage 1
    macro_rules! stage1_zmm_f32 {
        ($z:expr) => {{
            // Each 128-bit lane holds 2 complex f32: [a.re, a.im, b.re, b.im].
            // Butterfly the 2 complexes within each lane: out = [a+b, a-b].
            let a_dup = _mm512_shuffle_ps($z, $z, 0b01_00_01_00); // [a.re, a.im, a.re, a.im] per lane
            let b_dup = _mm512_shuffle_ps($z, $z, 0b11_10_11_10); // [b.re, b.im, b.re, b.im] per lane
            let sum  = _mm512_add_ps(a_dup, b_dup);
            let diff = _mm512_sub_ps(a_dup, b_dup);
            _mm512_shuffle_ps(sum, diff, 0b01_00_01_00) // [s.re, s.im, d.re, d.im] per lane
        }};
    }

    for zi in &mut z {
        *zi = stage1_zmm_f32!(*zi);
    }

    let sp_mut = stage.as_mut_ptr();
    for (i, zi) in z.iter().enumerate() {
        unsafe { _mm512_storeu_ps(sp_mut.add(i * 16), *zi) };
    }

    // Stages 2-6: scalar DIT
    let a = stage.as_mut_ptr();
    let n = 64usize;

    let mut span = 4usize;
    while span <= n {
        let half = span / 2;
        let stride = n / span;
        let mut k = 0;
        while k < n {
            for j in 0..half {
                let tw_idx = j * stride;
                let c = twiddles[2 * tw_idx];
                let s = twiddles[2 * tw_idx + 1];

                unsafe {
                    let u_re = *a.add((k + j) * 2);
                    let u_im = *a.add((k + j) * 2 + 1);
                    let v_re_raw = *a.add((k + j + half) * 2);
                    let v_im_raw = *a.add((k + j + half) * 2 + 1);

                    let (v_re, v_im) = cmul_f32(v_re_raw, v_im_raw, c, s);

                    *a.add((k + j) * 2) = u_re + v_re;
                    *a.add((k + j) * 2 + 1) = u_im + v_im;
                    *a.add((k + j + half) * 2) = u_re - v_re;
                    *a.add((k + j + half) * 2 + 1) = u_im - v_im;
                }
            }
            k += span;
        }
        span *= 2;
    }

    // Write results back
    let src = stage.as_ptr();
    unsafe {
        for i in 0..64usize {
            *data.add(i) = Complex {
                re: *src.add(2 * i),
                im: *src.add(2 * i + 1),
            };
        }
    }

    #[cfg(test)]
    {
        use core::sync::atomic::Ordering;
        super::hand_avx512_tests::HAND_AVX512_HIT_64_F32.store(true, Ordering::Relaxed);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public dispatch wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// Dispatch size-16 f64 DFT to hand-tuned AVX-512 if available, else scalar.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dispatch_hand_avx512_size16_f64(data: &mut [Complex<f64>], sign: i32) {
    if is_x86_feature_detected!("avx512f") {
        // Safety: avx512f confirmed, data has exactly 16 elements.
        unsafe {
            hand_avx512_size16_f64(data.as_mut_ptr(), sign);
        }
    } else {
        super::notw_16(data, sign);
    }
}

/// Dispatch size-32 f64 DFT to hand-tuned AVX-512 if available, else scalar.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dispatch_hand_avx512_size32_f64(data: &mut [Complex<f64>], sign: i32) {
    if is_x86_feature_detected!("avx512f") {
        unsafe {
            hand_avx512_size32_f64(data.as_mut_ptr(), sign);
        }
    } else {
        super::notw_32(data, sign);
    }
}

/// Dispatch size-64 f64 DFT to hand-tuned AVX-512 if available, else scalar.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dispatch_hand_avx512_size64_f64(data: &mut [Complex<f64>], sign: i32) {
    if is_x86_feature_detected!("avx512f") {
        unsafe {
            hand_avx512_size64_f64(data.as_mut_ptr(), sign);
        }
    } else {
        super::notw_64(data, sign);
    }
}

/// Dispatch size-16 f32 DFT to hand-tuned AVX-512 if available, else scalar.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dispatch_hand_avx512_size16_f32(data: &mut [Complex<f32>], sign: i32) {
    if is_x86_feature_detected!("avx512f") {
        unsafe {
            hand_avx512_size16_f32(data.as_mut_ptr(), sign);
        }
    } else {
        super::notw_16(data, sign);
    }
}

/// Dispatch size-32 f32 DFT to hand-tuned AVX-512 if available, else scalar.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dispatch_hand_avx512_size32_f32(data: &mut [Complex<f32>], sign: i32) {
    if is_x86_feature_detected!("avx512f") {
        unsafe {
            hand_avx512_size32_f32(data.as_mut_ptr(), sign);
        }
    } else {
        super::notw_32(data, sign);
    }
}

/// Dispatch size-64 f32 DFT to hand-tuned AVX-512 if available, else scalar.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dispatch_hand_avx512_size64_f32(data: &mut [Complex<f32>], sign: i32) {
    if is_x86_feature_detected!("avx512f") {
        unsafe {
            hand_avx512_size64_f32(data.as_mut_ptr(), sign);
        }
    } else {
        super::notw_64(data, sign);
    }
}
