//! AoS-layout pointwise complex multiplication with SIMD acceleration.
//!
//! Provides `complex_mul_aos_f64` and `complex_mul_aos_f32`, which compute
//! `dst[i] = a[i] * b[i]` for AoS (interleaved re/im) `Complex<T>` slices,
//! dispatching to the best available SIMD backend at runtime.
//!
//! # SIMD dispatch order
//!
//! - x86_64: AVX2+FMA → SSE2 → scalar
//! - aarch64: NEON → scalar
//! - other: scalar

use super::Complex;

// ============================================================================
// Public AoS dispatcher — f64
// ============================================================================

/// Compute `dst[i] = a[i] * b[i]` for AoS-layout `Complex<f64>` slices.
///
/// Dispatches to the best SIMD backend available at runtime:
/// AVX2+FMA → SSE2 → NEON → scalar fallback.
///
/// # Panics
///
/// Panics if `dst`, `a`, and `b` do not all have the same length.
pub fn complex_mul_aos_f64(dst: &mut [Complex<f64>], a: &[Complex<f64>], b: &[Complex<f64>]) {
    assert_eq!(
        dst.len(),
        a.len(),
        "complex_mul_aos_f64: dst/a length mismatch"
    );
    assert_eq!(
        dst.len(),
        b.len(),
        "complex_mul_aos_f64: dst/b length mismatch"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2 and FMA features confirmed above.
            return unsafe { complex_mul_aos_avx2_fma_f64(dst, a, b) };
        }
        if is_x86_feature_detected!("sse2") {
            // SAFETY: SSE2 feature confirmed above.
            return unsafe { complex_mul_aos_sse2_f64(dst, a, b) };
        }
        // No SIMD available on this x86_64 CPU (extremely rare).
        return complex_mul_aos_scalar_f64(dst, a, b);
    }

    #[cfg(target_arch = "aarch64")]
    // NEON is mandatory on aarch64 — dispatch unconditionally.
    // SAFETY: NEON is always present on aarch64.
    unsafe {
        complex_mul_aos_neon_f64(dst, a, b)
    };

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    complex_mul_aos_scalar_f64(dst, a, b);
}

// ============================================================================
// Public AoS dispatcher — f32
// ============================================================================

/// Compute `dst[i] = a[i] * b[i]` for AoS-layout `Complex<f32>` slices.
///
/// Dispatches to the best SIMD backend available at runtime:
/// AVX2+FMA → SSE2 → NEON → scalar fallback.
///
/// # Panics
///
/// Panics if `dst`, `a`, and `b` do not all have the same length.
pub fn complex_mul_aos_f32(dst: &mut [Complex<f32>], a: &[Complex<f32>], b: &[Complex<f32>]) {
    assert_eq!(
        dst.len(),
        a.len(),
        "complex_mul_aos_f32: dst/a length mismatch"
    );
    assert_eq!(
        dst.len(),
        b.len(),
        "complex_mul_aos_f32: dst/b length mismatch"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2 and FMA features confirmed above.
            return unsafe { complex_mul_aos_avx2_fma_f32(dst, a, b) };
        }
        if is_x86_feature_detected!("sse2") {
            // SAFETY: SSE2 feature confirmed above.
            return unsafe { complex_mul_aos_sse2_f32(dst, a, b) };
        }
        return complex_mul_aos_scalar_f32(dst, a, b);
    }

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is always present on aarch64.
    unsafe {
        complex_mul_aos_neon_f32(dst, a, b)
    };

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    complex_mul_aos_scalar_f32(dst, a, b);
}

// ============================================================================
// Scalar fallbacks (pub(crate) for use in tests and as direct fallback)
// ============================================================================

/// Scalar fallback: `dst[i] = a[i] * b[i]`.
#[inline]
#[allow(dead_code)] // reason: scalar fallback for non-x86 platforms; selected at runtime when AVX2/NEON unavailable
pub fn complex_mul_aos_scalar_f64(
    dst: &mut [Complex<f64>],
    a: &[Complex<f64>],
    b: &[Complex<f64>],
) {
    for ((d, &ai), &bi) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
        *d = ai * bi;
    }
}

/// Scalar fallback: `dst[i] = a[i] * b[i]`.
#[inline]
#[allow(dead_code)] // reason: scalar fallback for non-x86 platforms; selected at runtime when AVX2/NEON unavailable
pub fn complex_mul_aos_scalar_f32(
    dst: &mut [Complex<f32>],
    a: &[Complex<f32>],
    b: &[Complex<f32>],
) {
    for ((d, &ai), &bi) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
        *d = ai * bi;
    }
}

// ============================================================================
// AVX2+FMA backend — f64 (2 complex<f64> per __m256d lane)
// ============================================================================

/// # Safety
///
/// Caller must ensure the CPU supports `avx2` and `fma` features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn complex_mul_aos_avx2_fma_f64(
    dst: &mut [Complex<f64>],
    a: &[Complex<f64>],
    b: &[Complex<f64>],
) {
    use core::arch::x86_64::*;

    let len = dst.len();
    let chunks = len / 2; // 2 complex<f64> per __m256d

    let dst_ptr = dst.as_mut_ptr() as *mut f64;
    let a_ptr = a.as_ptr() as *const f64;
    let b_ptr = b.as_ptr() as *const f64;

    for i in 0..chunks {
        // Load 2 complex numbers from `a`: [a_re0, a_im0, a_re1, a_im1]
        let va = unsafe { _mm256_loadu_pd(a_ptr.add(i * 4)) };
        // Load 2 complex numbers from `b`: [b_re0, b_im0, b_re1, b_im1]
        let vb = unsafe { _mm256_loadu_pd(b_ptr.add(i * 4)) };

        // Duplicate real parts of `a`: [a_re0, a_re0, a_re1, a_re1]
        let a_re = _mm256_permute_pd(va, 0b0000);
        // Duplicate imag parts of `a`: [a_im0, a_im0, a_im1, a_im1]
        let a_im = _mm256_permute_pd(va, 0b1111);

        // Swap re/im in `b`: [b_im0, b_re0, b_im1, b_re1]
        let b_swap = _mm256_permute_pd(vb, 0b0101);

        // prod1 = [a_re0*b_re0, a_re0*b_im0, a_re1*b_re1, a_re1*b_im1]
        let prod1 = _mm256_mul_pd(a_re, vb);
        // prod2 = [a_im0*b_im0, a_im0*b_re0, a_im1*b_im1, a_im1*b_re1]
        let prod2 = _mm256_mul_pd(a_im, b_swap);

        // addsub(p, q) = [p0-q0, p1+q1, p2-q2, p3+q3]
        // = [a_re0*b_re0 - a_im0*b_im0, a_re0*b_im0 + a_im0*b_re0, ...]  ✓
        let result = _mm256_addsub_pd(prod1, prod2);

        unsafe { _mm256_storeu_pd(dst_ptr.add(i * 4), result) };
    }

    // Handle tail elements with scalar
    let tail_start = chunks * 2;
    for i in tail_start..len {
        dst[i] = a[i] * b[i];
    }
}

// ============================================================================
// SSE2 backend — f64 (1 complex<f64> per __m128d lane)
// ============================================================================

/// # Safety
///
/// Caller must ensure the CPU supports `sse2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn complex_mul_aos_sse2_f64(
    dst: &mut [Complex<f64>],
    a: &[Complex<f64>],
    b: &[Complex<f64>],
) {
    use core::arch::x86_64::*;

    let len = dst.len();
    let dst_ptr = dst.as_mut_ptr() as *mut f64;
    let a_ptr = a.as_ptr() as *const f64;
    let b_ptr = b.as_ptr() as *const f64;

    for i in 0..len {
        // Load 1 complex: [re, im]
        let va = unsafe { _mm_loadu_pd(a_ptr.add(i * 2)) };
        let vb = unsafe { _mm_loadu_pd(b_ptr.add(i * 2)) };

        // a_re = [a_re, a_re], a_im = [a_im, a_im]
        let a_re = _mm_unpacklo_pd(va, va);
        let a_im = _mm_unpackhi_pd(va, va);

        // b_swap = [b_im, b_re]
        let b_swap = _mm_shuffle_pd(vb, vb, 0b01);

        // prod1 = [a_re*b_re, a_re*b_im]
        let prod1 = _mm_mul_pd(a_re, vb);
        // prod2 = [a_im*b_im, a_im*b_re]
        let prod2 = _mm_mul_pd(a_im, b_swap);

        // Negate prod2 low element: sign = [-0.0, 0.0]
        let sign = _mm_set_pd(0.0_f64, -0.0_f64);
        let prod2_signed = _mm_xor_pd(prod2, sign);

        // result = [a_re*b_re - a_im*b_im, a_re*b_im + a_im*b_re]  ✓
        let result = _mm_add_pd(prod1, prod2_signed);

        unsafe { _mm_storeu_pd(dst_ptr.add(i * 2), result) };
    }
}

// ============================================================================
// NEON backend — f64 (1 complex<f64> per float64x2_t lane)
// ============================================================================

/// # Safety
///
/// Caller must ensure NEON is available (always true on aarch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn complex_mul_aos_neon_f64(
    dst: &mut [Complex<f64>],
    a: &[Complex<f64>],
    b: &[Complex<f64>],
) {
    use core::arch::aarch64::*;

    let len = dst.len();
    let dst_ptr = dst.as_mut_ptr() as *mut f64;
    let a_ptr = a.as_ptr() as *const f64;
    let b_ptr = b.as_ptr() as *const f64;

    for i in 0..len {
        unsafe {
            // Load [re, im]
            let va = vld1q_f64(a_ptr.add(i * 2));
            let vb = vld1q_f64(b_ptr.add(i * 2));

            // a_re = [a_re, a_re], a_im = [a_im, a_im]
            let a_re = vdupq_lane_f64(vget_low_f64(va), 0);
            let a_im = vdupq_lane_f64(vget_high_f64(va), 0);

            // b_swap = [b_im, b_re]
            let b_swap = vextq_f64(vb, vb, 1);

            // prod1 = [a_re*b_re, a_re*b_im]
            let prod1 = vmulq_f64(a_re, vb);
            // prod2 = [a_im*b_im, a_im*b_re]
            let prod2 = vmulq_f64(a_im, b_swap);

            // result = prod1 + prod2 * [-1.0, 1.0]
            // = [a_re*b_re - a_im*b_im, a_re*b_im + a_im*b_re]  ✓
            let sign = vld1q_f64([(-1.0_f64), 1.0_f64].as_ptr());
            let result = vfmaq_f64(prod1, prod2, sign);

            vst1q_f64(dst_ptr.add(i * 2), result);
        }
    }
}

// ============================================================================
// AVX2+FMA backend — f32 (4 complex<f32> per __m256 lane)
// ============================================================================

/// # Safety
///
/// Caller must ensure the CPU supports `avx2` and `fma`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn complex_mul_aos_avx2_fma_f32(
    dst: &mut [Complex<f32>],
    a: &[Complex<f32>],
    b: &[Complex<f32>],
) {
    use core::arch::x86_64::*;

    let len = dst.len();
    let chunks = len / 4; // 4 complex<f32> per __m256

    let dst_ptr = dst.as_mut_ptr() as *mut f32;
    let a_ptr = a.as_ptr() as *const f32;
    let b_ptr = b.as_ptr() as *const f32;

    for i in 0..chunks {
        // Load 4 complex: [re0,im0, re1,im1, re2,im2, re3,im3]
        let va = unsafe { _mm256_loadu_ps(a_ptr.add(i * 8)) };
        let vb = unsafe { _mm256_loadu_ps(b_ptr.add(i * 8)) };

        // Duplicate real parts of a: [re0,re0, re1,re1, re2,re2, re3,re3]
        let a_re = _mm256_moveldup_ps(va);
        // Duplicate imag parts of a: [im0,im0, im1,im1, im2,im2, im3,im3]
        let a_im = _mm256_movehdup_ps(va);

        // Swap re/im in b: [im0,re0, im1,re1, im2,re2, im3,re3]
        let b_swap = _mm256_permute_ps(vb, 0b10_11_00_01);

        // prod1 = [re0*b_re0, re0*b_im0, ...]
        let prod1 = _mm256_mul_ps(a_re, vb);
        // prod2 = [im0*b_im0, im0*b_re0, ...]
        let prod2 = _mm256_mul_ps(a_im, b_swap);

        // addsub(p, q) = [p0-q0, p1+q1, ...]  ✓
        let result = _mm256_addsub_ps(prod1, prod2);

        unsafe { _mm256_storeu_ps(dst_ptr.add(i * 8), result) };
    }

    // Handle tail elements with scalar
    let tail_start = chunks * 4;
    for i in tail_start..len {
        dst[i] = a[i] * b[i];
    }
}

// ============================================================================
// SSE2 backend — f32 (2 complex<f32> per __m128 lane)
// ============================================================================

/// # Safety
///
/// Caller must ensure the CPU supports `sse2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn complex_mul_aos_sse2_f32(
    dst: &mut [Complex<f32>],
    a: &[Complex<f32>],
    b: &[Complex<f32>],
) {
    use core::arch::x86_64::*;

    let len = dst.len();
    let chunks = len / 2; // 2 complex<f32> per __m128

    let dst_ptr = dst.as_mut_ptr() as *mut f32;
    let a_ptr = a.as_ptr() as *const f32;
    let b_ptr = b.as_ptr() as *const f32;

    for i in 0..chunks {
        // Load 2 complex: [re0, im0, re1, im1]
        let va = unsafe { _mm_loadu_ps(a_ptr.add(i * 4)) };
        let vb = unsafe { _mm_loadu_ps(b_ptr.add(i * 4)) };

        // Duplicate real parts: [re0, re0, re1, re1]
        let a_re = unsafe { _mm_moveldup_ps(va) };
        // Duplicate imag parts: [im0, im0, im1, im1]
        let a_im = unsafe { _mm_movehdup_ps(va) };

        // Swap re/im in b: [im0, re0, im1, re1]
        let b_swap = _mm_shuffle_ps(vb, vb, 0b10_11_00_01);

        // prod1 = [re0*b_re0, re0*b_im0, re1*b_re1, re1*b_im1]
        let prod1 = _mm_mul_ps(a_re, vb);
        // prod2 = [im0*b_im0, im0*b_re0, im1*b_im1, im1*b_re1]
        let prod2 = _mm_mul_ps(a_im, b_swap);

        // addsub: [p0-q0, p1+q1, p2-q2, p3+q3]  ✓
        let result = unsafe { _mm_addsub_ps(prod1, prod2) };

        unsafe { _mm_storeu_ps(dst_ptr.add(i * 4), result) };
    }

    // Handle tail elements with scalar
    let tail_start = chunks * 2;
    for i in tail_start..len {
        dst[i] = a[i] * b[i];
    }
}

// ============================================================================
// NEON backend — f32 (2 complex<f32> per float32x4_t lane)
// ============================================================================

/// # Safety
///
/// Caller must ensure NEON is available (always true on aarch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn complex_mul_aos_neon_f32(
    dst: &mut [Complex<f32>],
    a: &[Complex<f32>],
    b: &[Complex<f32>],
) {
    use core::arch::aarch64::*;

    let len = dst.len();
    let chunks = len / 2; // 2 complex<f32> per float32x4_t

    let dst_ptr = dst.as_mut_ptr() as *mut f32;
    let a_ptr = a.as_ptr() as *const f32;
    let b_ptr = b.as_ptr() as *const f32;

    for i in 0..chunks {
        unsafe {
            // Load [re0, im0, re1, im1]
            let va = vld1q_f32(a_ptr.add(i * 4));
            let vb = vld1q_f32(b_ptr.add(i * 4));

            // Duplicate real parts: [re0, re0, re1, re1]
            let a_re = vtrn1q_f32(va, va);
            // Duplicate imag parts: [im0, im0, im1, im1]
            let a_im = vtrn2q_f32(va, va);

            // Swap re/im in b: [im0, re0, im1, re1]
            let b_swap = vrev64q_f32(vb);

            // prod1 = [re0*b_re0, re0*b_im0, re1*b_re1, re1*b_im1]
            let prod1 = vmulq_f32(a_re, vb);
            // prod2 = [im0*b_im0, im0*b_re0, im1*b_im1, im1*b_re1]
            let prod2 = vmulq_f32(a_im, b_swap);

            // result = prod1 + prod2 * [-1, 1, -1, 1]
            // = [re0*b_re0 - im0*b_im0, re0*b_im0 + im0*b_re0, ...]  ✓
            let sign = vld1q_f32([(-1.0_f32), 1.0_f32, (-1.0_f32), 1.0_f32].as_ptr());
            let result = vfmaq_f32(prod1, prod2, sign);

            vst1q_f32(dst_ptr.add(i * 4), result);
        }
    }

    // Handle tail elements with scalar
    let tail_start = chunks * 2;
    for i in tail_start..len {
        dst[i] = a[i] * b[i];
    }
}

// ============================================================================
// Generic TypeId-dispatched wrapper for use in generic code
// ============================================================================

/// Compute `dst[i] = a[i] * b[i]` for AoS-layout `Complex<T>` slices.
///
/// Dispatches to the concrete f64/f32 SIMD implementation based on `TypeId`.
/// Falls back to scalar multiplication for other float types.
pub fn complex_mul_aos<T: super::Float>(
    dst: &mut [Complex<T>],
    a: &[Complex<T>],
    b: &[Complex<T>],
) {
    use core::any::TypeId;

    assert_eq!(dst.len(), a.len(), "complex_mul_aos: dst/a length mismatch");
    assert_eq!(dst.len(), b.len(), "complex_mul_aos: dst/b length mismatch");

    let tid = TypeId::of::<T>();

    if tid == TypeId::of::<f64>() {
        // SAFETY: TypeId confirms T == f64; reinterpreting pointers is valid.
        let dst_f64 = unsafe {
            core::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut Complex<f64>, dst.len())
        };
        let a_f64 =
            unsafe { core::slice::from_raw_parts(a.as_ptr() as *const Complex<f64>, a.len()) };
        let b_f64 =
            unsafe { core::slice::from_raw_parts(b.as_ptr() as *const Complex<f64>, b.len()) };
        complex_mul_aos_f64(dst_f64, a_f64, b_f64);
    } else if tid == TypeId::of::<f32>() {
        // SAFETY: TypeId confirms T == f32.
        let dst_f32 = unsafe {
            core::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut Complex<f32>, dst.len())
        };
        let a_f32 =
            unsafe { core::slice::from_raw_parts(a.as_ptr() as *const Complex<f32>, a.len()) };
        let b_f32 =
            unsafe { core::slice::from_raw_parts(b.as_ptr() as *const Complex<f32>, b.len()) };
        complex_mul_aos_f32(dst_f32, a_f32, b_f32);
    } else {
        // Scalar fallback for other float types (e.g., f128).
        for ((d, &ai), &bi) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
            *d = ai * bi;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn max_rel_err_f64(dst: &[Complex<f64>], ref_: &[Complex<f64>]) -> f64 {
        dst.iter()
            .zip(ref_.iter())
            .map(|(a, b)| {
                let diff_re = (a.re - b.re).abs();
                let diff_im = (a.im - b.im).abs();
                let norm = (b.re * b.re + b.im * b.im).sqrt().max(1e-30);
                (diff_re + diff_im) / norm
            })
            .fold(0.0_f64, f64::max)
    }

    fn max_rel_err_f32(dst: &[Complex<f32>], ref_: &[Complex<f32>]) -> f32 {
        dst.iter()
            .zip(ref_.iter())
            .map(|(a, b)| {
                let diff_re = (a.re - b.re).abs();
                let diff_im = (a.im - b.im).abs();
                let norm = (b.re * b.re + b.im * b.im).sqrt().max(1e-10_f32);
                (diff_re + diff_im) / norm
            })
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn simd_vs_scalar_f64_small() {
        let n = 17;
        let a: Vec<Complex<f64>> = (0..n)
            .map(|k| Complex::new((k as f64).sin(), (k as f64).cos()))
            .collect();
        let b: Vec<Complex<f64>> = (0..n)
            .map(|k| Complex::new((k as f64 * 0.7).cos(), (k as f64 * 0.3).sin()))
            .collect();

        let mut ref_dst = vec![Complex::new(0.0_f64, 0.0); n];
        complex_mul_aos_scalar_f64(&mut ref_dst, &a, &b);

        let mut simd_dst = vec![Complex::new(0.0_f64, 0.0); n];
        complex_mul_aos_f64(&mut simd_dst, &a, &b);

        let err = max_rel_err_f64(&simd_dst, &ref_dst);
        assert!(
            err < 1e-14,
            "f64 SIMD vs scalar max relative error={err} (must be < 1e-14)"
        );
    }

    #[test]
    fn simd_vs_scalar_f64_large() {
        let n = 1009;
        let a: Vec<Complex<f64>> = (0..n)
            .map(|k| Complex::new((k as f64).sin(), (k as f64).cos()))
            .collect();
        let b: Vec<Complex<f64>> = (0..n)
            .map(|k| Complex::new((k as f64 * 0.5).cos(), (k as f64 * 0.2).sin()))
            .collect();

        let mut ref_dst = vec![Complex::new(0.0_f64, 0.0); n];
        complex_mul_aos_scalar_f64(&mut ref_dst, &a, &b);

        let mut simd_dst = vec![Complex::new(0.0_f64, 0.0); n];
        complex_mul_aos_f64(&mut simd_dst, &a, &b);

        let err = max_rel_err_f64(&simd_dst, &ref_dst);
        assert!(
            err < 1e-14,
            "f64 SIMD vs scalar max relative error={err} (must be < 1e-14)"
        );
    }

    #[test]
    fn simd_vs_scalar_f32_small() {
        let n = 17;
        let a: Vec<Complex<f32>> = (0..n)
            .map(|k| Complex::new((k as f32).sin(), (k as f32).cos()))
            .collect();
        let b: Vec<Complex<f32>> = (0..n)
            .map(|k| Complex::new((k as f32 * 0.7).cos(), (k as f32 * 0.3).sin()))
            .collect();

        let mut ref_dst = vec![Complex::new(0.0_f32, 0.0); n];
        complex_mul_aos_scalar_f32(&mut ref_dst, &a, &b);

        let mut simd_dst = vec![Complex::new(0.0_f32, 0.0); n];
        complex_mul_aos_f32(&mut simd_dst, &a, &b);

        let err = max_rel_err_f32(&simd_dst, &ref_dst);
        assert!(
            err < 1e-5,
            "f32 SIMD vs scalar max relative error={err} (must be < 1e-5)"
        );
    }

    #[test]
    fn simd_vs_scalar_f32_large() {
        let n = 1009;
        let a: Vec<Complex<f32>> = (0..n)
            .map(|k| Complex::new((k as f32).sin(), (k as f32).cos()))
            .collect();
        let b: Vec<Complex<f32>> = (0..n)
            .map(|k| Complex::new((k as f32 * 0.5).cos(), (k as f32 * 0.2).sin()))
            .collect();

        let mut ref_dst = vec![Complex::new(0.0_f32, 0.0); n];
        complex_mul_aos_scalar_f32(&mut ref_dst, &a, &b);

        let mut simd_dst = vec![Complex::new(0.0_f32, 0.0); n];
        complex_mul_aos_f32(&mut simd_dst, &a, &b);

        let err = max_rel_err_f32(&simd_dst, &ref_dst);
        assert!(
            err < 1e-5,
            "f32 SIMD vs scalar max relative error={err} (must be < 1e-5)"
        );
    }

    #[test]
    fn generic_dispatcher_f64() {
        let n = 97;
        let a: Vec<Complex<f64>> = (0..n)
            .map(|k| Complex::new(k as f64, -(k as f64)))
            .collect();
        let b: Vec<Complex<f64>> = (0..n)
            .map(|k| Complex::new((k as f64).sin(), (k as f64).cos()))
            .collect();

        let mut ref_dst = vec![Complex::new(0.0_f64, 0.0); n];
        complex_mul_aos_scalar_f64(&mut ref_dst, &a, &b);

        let mut gen_dst = vec![Complex::new(0.0_f64, 0.0); n];
        complex_mul_aos(&mut gen_dst, &a, &b);

        let err = max_rel_err_f64(&gen_dst, &ref_dst);
        assert!(err < 1e-14, "generic f64 dispatch max relative error={err}");
    }

    #[test]
    fn generic_dispatcher_f32() {
        let n = 97;
        let a: Vec<Complex<f32>> = (0..n)
            .map(|k| Complex::new(k as f32, -(k as f32)))
            .collect();
        let b: Vec<Complex<f32>> = (0..n)
            .map(|k| Complex::new((k as f32).sin(), (k as f32).cos()))
            .collect();

        let mut ref_dst = vec![Complex::new(0.0_f32, 0.0); n];
        complex_mul_aos_scalar_f32(&mut ref_dst, &a, &b);

        let mut gen_dst = vec![Complex::new(0.0_f32, 0.0); n];
        complex_mul_aos(&mut gen_dst, &a, &b);

        let err = max_rel_err_f32(&gen_dst, &ref_dst);
        assert!(err < 1e-5, "generic f32 dispatch max relative error={err}");
    }
}
