//! AVX2 SIMD implementation for x86_64.
//!
//! Provides 256-bit SIMD operations using AVX2 intrinsics with FMA (Fused Multiply-Add).
//! AVX2 extends AVX with integer SIMD operations and is typically paired with FMA3.
//!
//! - f64: 4 lanes (256-bit = 4 × 64-bit)
//! - f32: 8 lanes (256-bit = 8 × 32-bit)
//!
//! The key advantage over AVX is the availability of FMA instructions which can
//! compute `a * b + c` in a single operation with better precision and performance.

use super::traits::{SimdComplex, SimdVector};
use core::arch::x86_64::*;

/// Check if AVX2 and FMA are available at runtime.
#[inline]
pub fn has_avx2_fma() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

/// AVX2 f64 vector type with FMA support (4 lanes).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Avx2F64(pub __m256d);

/// AVX2 f32 vector type with FMA support (8 lanes).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Avx2F32(pub __m256);

// Safety: AVX2 vectors are POD types that can be safely sent between threads
unsafe impl Send for Avx2F64 {}
unsafe impl Sync for Avx2F64 {}
unsafe impl Send for Avx2F32 {}
unsafe impl Sync for Avx2F32 {}

impl SimdVector for Avx2F64 {
    type Scalar = f64;
    const LANES: usize = 4;

    #[inline]
    fn splat(value: f64) -> Self {
        unsafe { Self(_mm256_set1_pd(value)) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::load_aligned`] for the full safety contract.
    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        unsafe { Self(_mm256_load_pd(ptr)) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::load_unaligned`] for the full safety contract.
    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        unsafe { Self(_mm256_loadu_pd(ptr)) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::store_aligned`] for the full safety contract.
    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        unsafe { _mm256_store_pd(ptr, self.0) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::store_unaligned`] for the full safety contract.
    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        unsafe { _mm256_storeu_pd(ptr, self.0) }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { Self(_mm256_add_pd(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { Self(_mm256_sub_pd(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { Self(_mm256_mul_pd(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { Self(_mm256_div_pd(self.0, other.0)) }
    }
}

impl Avx2F64 {
    /// Create a vector from four f64 values: [a, b, c, d]
    #[inline]
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        unsafe { Self(_mm256_set_pd(d, c, b, a)) }
    }

    /// Extract element at index (0-3).
    #[inline]
    pub fn extract(self, idx: usize) -> f64 {
        debug_assert!(idx < 4);
        let mut arr = [0.0_f64; 4];
        unsafe { self.store_unaligned(arr.as_mut_ptr()) };
        arr[idx]
    }

    /// Fused multiply-add: self * a + b
    /// Computes (self * a) + b with a single rounding.
    #[inline]
    pub fn fmadd(self, a: Self, b: Self) -> Self {
        unsafe { Self(_mm256_fmadd_pd(self.0, a.0, b.0)) }
    }

    /// Fused multiply-subtract: self * a - b
    /// Computes (self * a) - b with a single rounding.
    #[inline]
    pub fn fmsub(self, a: Self, b: Self) -> Self {
        unsafe { Self(_mm256_fmsub_pd(self.0, a.0, b.0)) }
    }

    /// Fused negative multiply-add: -(self * a) + b
    /// Computes b - (self * a) with a single rounding.
    #[inline]
    pub fn fnmadd(self, a: Self, b: Self) -> Self {
        unsafe { Self(_mm256_fnmadd_pd(self.0, a.0, b.0)) }
    }

    /// Fused negative multiply-subtract: -(self * a) - b
    /// Computes -(self * a) - b with a single rounding.
    #[inline]
    pub fn fnmsub(self, a: Self, b: Self) -> Self {
        unsafe { Self(_mm256_fnmsub_pd(self.0, a.0, b.0)) }
    }

    /// Negate all elements.
    #[inline]
    pub fn negate(self) -> Self {
        unsafe {
            let sign_mask = _mm256_set1_pd(-0.0);
            Self(_mm256_xor_pd(self.0, sign_mask))
        }
    }

    /// Shuffle within 128-bit lanes.
    #[inline]
    pub fn shuffle_within_lanes<const MASK: i32>(self) -> Self {
        unsafe { Self(_mm256_permute_pd(self.0, MASK)) }
    }

    /// Swap 128-bit lanes.
    #[inline]
    pub fn swap_lanes(self) -> Self {
        unsafe { Self(_mm256_permute2f128_pd(self.0, self.0, 0x01)) }
    }

    /// Interleave low elements within 128-bit lanes.
    #[inline]
    pub fn unpack_lo(self, other: Self) -> Self {
        unsafe { Self(_mm256_unpacklo_pd(self.0, other.0)) }
    }

    /// Interleave high elements within 128-bit lanes.
    #[inline]
    pub fn unpack_hi(self, other: Self) -> Self {
        unsafe { Self(_mm256_unpackhi_pd(self.0, other.0)) }
    }

    /// Blend elements with compile-time mask.
    #[inline]
    pub fn blend<const MASK: i32>(self, other: Self) -> Self {
        unsafe { Self(_mm256_blend_pd(self.0, other.0, MASK)) }
    }

    /// Get the low 128-bit lane as SSE vector.
    #[inline]
    pub fn low_128(self) -> super::sse2::Sse2F64 {
        unsafe { super::sse2::Sse2F64(_mm256_castpd256_pd128(self.0)) }
    }

    /// Get the high 128-bit lane as SSE vector.
    #[inline]
    pub fn high_128(self) -> super::sse2::Sse2F64 {
        unsafe { super::sse2::Sse2F64(_mm256_extractf128_pd(self.0, 1)) }
    }
}

impl SimdComplex for Avx2F64 {
    /// Complex multiply for 2 interleaved complex numbers using FMA.
    ///
    /// Format: [re0, im0, re1, im1]
    /// Uses FMA for better precision and performance.
    #[inline]
    fn cmul(self, other: Self) -> Self {
        unsafe {
            // self = [a, b, c, d] = [re0, im0, re1, im1]
            // other = [e, f, g, h] = [re0', im0', re1', im1']

            // Shuffle to get real and imaginary parts in position
            // a_re = [a, a, c, c] (real parts duplicated)
            let a_re = _mm256_permute_pd(self.0, 0b0000);
            // a_im = [b, b, d, d] (imag parts duplicated)
            let a_im = _mm256_permute_pd(self.0, 0b1111);

            // b_flip = [f, e, h, g] (swap real and imaginary of other)
            let b_flip = _mm256_permute_pd(other.0, 0b0101);

            // prod1 = a_re * other = [a*e, a*f, c*g, c*h]
            let prod1 = _mm256_mul_pd(a_re, other.0);

            // Use FMA: result = a_im * b_flip +/- prod1
            // We want [a*e - b*f, a*f + b*e, c*g - d*h, c*h + d*g]
            // addsub: [prod1[0] - a_im*b_flip[0], prod1[1] + a_im*b_flip[1], ...]
            // But _mm256_addsub does alternating sub/add on even/odd elements
            // addsub(a, b) = [a0-b0, a1+b1, a2-b2, a3+b3]
            // So addsub(prod1, a_im*b_flip) = [a*e - b*f, a*f + b*e, c*g - d*h, c*h + d*g] ✓

            // Use fmaddsub: prod1 +/- (a_im * b_flip)
            // fmaddsub(a, b, c) = a*b +/- c where it alternates +/- per element
            // _mm256_fmaddsub_pd(a, b, c) = a*b -/+ c (sub on even, add on odd)
            let prod2 = _mm256_mul_pd(a_im, b_flip);
            Self(_mm256_addsub_pd(prod1, prod2))
        }
    }

    /// Complex multiply with conjugate using FMA.
    ///
    /// Computes self * conj(other).
    #[inline]
    fn cmul_conj(self, other: Self) -> Self {
        unsafe {
            // For conj multiply: (a+bi) * (e-fi) = (ae+bf) + (be-af)i
            // Negate the imaginary parts of other first: [e, -f, g, -h]
            let sign_mask = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
            let other_conj = _mm256_xor_pd(other.0, sign_mask);

            let a_re = _mm256_permute_pd(self.0, 0b0000);
            let a_im = _mm256_permute_pd(self.0, 0b1111);
            let b_flip = _mm256_permute_pd(other_conj, 0b0101);

            let prod1 = _mm256_mul_pd(a_re, other_conj);
            let prod2 = _mm256_mul_pd(a_im, b_flip);
            Self(_mm256_addsub_pd(prod1, prod2))
        }
    }
}

impl SimdVector for Avx2F32 {
    type Scalar = f32;
    const LANES: usize = 8;

    #[inline]
    fn splat(value: f32) -> Self {
        unsafe { Self(_mm256_set1_ps(value)) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::load_aligned`] for the full safety contract.
    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        unsafe { Self(_mm256_load_ps(ptr)) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::load_unaligned`] for the full safety contract.
    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        unsafe { Self(_mm256_loadu_ps(ptr)) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::store_aligned`] for the full safety contract.
    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        unsafe { _mm256_store_ps(ptr, self.0) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::store_unaligned`] for the full safety contract.
    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        unsafe { _mm256_storeu_ps(ptr, self.0) }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { Self(_mm256_add_ps(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { Self(_mm256_sub_ps(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { Self(_mm256_mul_ps(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { Self(_mm256_div_ps(self.0, other.0)) }
    }
}

impl Avx2F32 {
    /// Create a vector from eight f32 values.
    #[inline]
    pub fn new(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> Self {
        unsafe { Self(_mm256_set_ps(h, g, f, e, d, c, b, a)) }
    }

    /// Fused multiply-add: self * a + b
    #[inline]
    pub fn fmadd(self, a: Self, b: Self) -> Self {
        unsafe { Self(_mm256_fmadd_ps(self.0, a.0, b.0)) }
    }

    /// Fused multiply-subtract: self * a - b
    #[inline]
    pub fn fmsub(self, a: Self, b: Self) -> Self {
        unsafe { Self(_mm256_fmsub_ps(self.0, a.0, b.0)) }
    }

    /// Fused negative multiply-add: -(self * a) + b
    #[inline]
    pub fn fnmadd(self, a: Self, b: Self) -> Self {
        unsafe { Self(_mm256_fnmadd_ps(self.0, a.0, b.0)) }
    }

    /// Fused negative multiply-subtract: -(self * a) - b
    #[inline]
    pub fn fnmsub(self, a: Self, b: Self) -> Self {
        unsafe { Self(_mm256_fnmsub_ps(self.0, a.0, b.0)) }
    }

    /// Negate all elements.
    #[inline]
    pub fn negate(self) -> Self {
        unsafe {
            let sign_mask = _mm256_set1_ps(-0.0);
            Self(_mm256_xor_ps(self.0, sign_mask))
        }
    }

    /// Interleave low elements within 128-bit lanes.
    #[inline]
    pub fn unpack_lo(self, other: Self) -> Self {
        unsafe { Self(_mm256_unpacklo_ps(self.0, other.0)) }
    }

    /// Interleave high elements within 128-bit lanes.
    #[inline]
    pub fn unpack_hi(self, other: Self) -> Self {
        unsafe { Self(_mm256_unpackhi_ps(self.0, other.0)) }
    }

    /// Swap 128-bit lanes.
    #[inline]
    pub fn swap_lanes(self) -> Self {
        unsafe { Self(_mm256_permute2f128_ps(self.0, self.0, 0x01)) }
    }

    /// Get the low 128-bit lane as SSE vector.
    #[inline]
    pub fn low_128(self) -> super::sse2::Sse2F32 {
        unsafe { super::sse2::Sse2F32(_mm256_castps256_ps128(self.0)) }
    }

    /// Get the high 128-bit lane as SSE vector.
    #[inline]
    pub fn high_128(self) -> super::sse2::Sse2F32 {
        unsafe { super::sse2::Sse2F32(_mm256_extractf128_ps(self.0, 1)) }
    }
}

impl SimdComplex for Avx2F32 {
    /// Complex multiply for 4 interleaved complex numbers using FMA.
    ///
    /// Format: [re0, im0, re1, im1, re2, im2, re3, im3]
    #[inline]
    fn cmul(self, other: Self) -> Self {
        unsafe {
            // Shuffle to duplicate real and imaginary parts
            // moveldup: [a, a, c, c, e, e, g, g]
            let a_re = _mm256_moveldup_ps(self.0);
            // movehdup: [b, b, d, d, f, f, h, h]
            let a_im = _mm256_movehdup_ps(self.0);

            // Swap pairs in other: [im0, re0, im1, re1, ...]
            let b_flip = _mm256_permute_ps(other.0, 0b10_11_00_01);

            // prod1 = a_re * other
            let prod1 = _mm256_mul_ps(a_re, other.0);
            // prod2 = a_im * b_flip
            let prod2 = _mm256_mul_ps(a_im, b_flip);

            // addsub: alternating sub/add
            Self(_mm256_addsub_ps(prod1, prod2))
        }
    }

    /// Complex multiply with conjugate.
    #[inline]
    fn cmul_conj(self, other: Self) -> Self {
        unsafe {
            // Negate imaginary parts of other
            let sign_mask = _mm256_set_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
            let other_conj = _mm256_xor_ps(other.0, sign_mask);

            let a_re = _mm256_moveldup_ps(self.0);
            let a_im = _mm256_movehdup_ps(self.0);
            let b_flip = _mm256_permute_ps(other_conj, 0b10_11_00_01);

            let prod1 = _mm256_mul_ps(a_re, other_conj);
            let prod2 = _mm256_mul_ps(a_im, b_flip);

            Self(_mm256_addsub_ps(prod1, prod2))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx2_fma_f64_basic() {
        if !has_avx2_fma() {
            return;
        }

        let a = Avx2F64::splat(2.0);
        let b = Avx2F64::splat(3.0);
        let c = a.add(b);
        assert_eq!(c.extract(0), 5.0);
        assert_eq!(c.extract(1), 5.0);
        assert_eq!(c.extract(2), 5.0);
        assert_eq!(c.extract(3), 5.0);
    }

    #[test]
    fn test_avx2_fma_f64_fmadd() {
        if !has_avx2_fma() {
            return;
        }

        // a * b + c = 2 * 3 + 4 = 10
        let a = Avx2F64::splat(2.0);
        let b = Avx2F64::splat(3.0);
        let c = Avx2F64::splat(4.0);
        let result = a.fmadd(b, c);
        assert_eq!(result.extract(0), 10.0);
        assert_eq!(result.extract(1), 10.0);
    }

    #[test]
    fn test_avx2_fma_f64_fmsub() {
        if !has_avx2_fma() {
            return;
        }

        // a * b - c = 2 * 3 - 4 = 2
        let a = Avx2F64::splat(2.0);
        let b = Avx2F64::splat(3.0);
        let c = Avx2F64::splat(4.0);
        let result = a.fmsub(b, c);
        assert_eq!(result.extract(0), 2.0);
        assert_eq!(result.extract(1), 2.0);
    }

    #[test]
    fn test_avx2_fma_f64_new() {
        if !has_avx2_fma() {
            return;
        }

        let v = Avx2F64::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.extract(0), 1.0);
        assert_eq!(v.extract(1), 2.0);
        assert_eq!(v.extract(2), 3.0);
        assert_eq!(v.extract(3), 4.0);
    }

    #[test]
    fn test_avx2_fma_f64_load_store() {
        if !has_avx2_fma() {
            return;
        }

        let data = [1.0_f64, 2.0, 3.0, 4.0];
        let v = unsafe { Avx2F64::load_unaligned(data.as_ptr()) };

        let mut out = [0.0_f64; 4];
        unsafe { v.store_unaligned(out.as_mut_ptr()) };

        assert_eq!(data, out);
    }

    #[test]
    fn test_avx2_fma_f64_cmul() {
        if !has_avx2_fma() {
            return;
        }

        // (3 + 4i) * (1 + 2i) = (3*1 - 4*2) + (3*2 + 4*1)i = -5 + 10i
        let a = Avx2F64::new(3.0, 4.0, 1.0, 0.0);
        let b = Avx2F64::new(1.0, 2.0, 1.0, 0.0);
        let c = a.cmul(b);

        let tol = 1e-10;
        assert!((c.extract(0) - (-5.0)).abs() < tol);
        assert!((c.extract(1) - 10.0).abs() < tol);
    }

    #[test]
    fn test_avx2_fma_f32_basic() {
        if !has_avx2_fma() {
            return;
        }

        let a = Avx2F32::splat(2.0);
        let b = Avx2F32::splat(3.0);
        let c = a.mul(b);

        let mut out = [0.0_f32; 8];
        unsafe { c.store_unaligned(out.as_mut_ptr()) };

        for val in &out {
            assert_eq!(*val, 6.0);
        }
    }

    #[test]
    fn test_avx2_fma_f32_fmadd() {
        if !has_avx2_fma() {
            return;
        }

        // a * b + c = 2 * 3 + 4 = 10
        let a = Avx2F32::splat(2.0);
        let b = Avx2F32::splat(3.0);
        let c = Avx2F32::splat(4.0);
        let result = a.fmadd(b, c);

        let mut out = [0.0_f32; 8];
        unsafe { result.store_unaligned(out.as_mut_ptr()) };

        for val in &out {
            assert_eq!(*val, 10.0);
        }
    }

    #[test]
    fn test_avx2_fma_f32_cmul() {
        if !has_avx2_fma() {
            return;
        }

        // (3 + 4i) * (1 + 2i) = -5 + 10i
        let a = Avx2F32::new(3.0, 4.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0);
        let b = Avx2F32::new(1.0, 2.0, 1.0, 0.0, 1.0, -1.0, 1.0, 0.0);
        let c = a.cmul(b);

        let mut out = [0.0_f32; 8];
        unsafe { c.store_unaligned(out.as_mut_ptr()) };

        let tol = 1e-5;
        // First complex: (3+4i) * (1+2i) = -5 + 10i
        assert!((out[0] - (-5.0)).abs() < tol);
        assert!((out[1] - 10.0).abs() < tol);
        // Second complex: (1+0i) * (1+0i) = 1 + 0i
        assert!((out[2] - 1.0).abs() < tol);
        assert!((out[3] - 0.0).abs() < tol);
    }
}
