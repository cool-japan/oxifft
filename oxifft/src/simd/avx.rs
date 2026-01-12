//! AVX SIMD implementation for x86_64.
//!
//! Provides 256-bit SIMD operations using AVX intrinsics.
//! - f64: 4 lanes (256-bit = 4 × 64-bit)
//! - f32: 8 lanes (256-bit = 8 × 32-bit)

use super::traits::{SimdComplex, SimdVector};
use core::arch::x86_64::*;

/// AVX f64 vector type (4 lanes).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct AvxF64(pub __m256d);

/// AVX f32 vector type (8 lanes).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct AvxF32(pub __m256);

// Safety: AVX vectors are POD types that can be safely sent between threads
unsafe impl Send for AvxF64 {}
unsafe impl Sync for AvxF64 {}
unsafe impl Send for AvxF32 {}
unsafe impl Sync for AvxF32 {}

impl SimdVector for AvxF64 {
    type Scalar = f64;
    const LANES: usize = 4;

    #[inline]
    fn splat(value: f64) -> Self {
        unsafe { Self(_mm256_set1_pd(value)) }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        unsafe { Self(_mm256_load_pd(ptr)) }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        unsafe { Self(_mm256_loadu_pd(ptr)) }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        unsafe { _mm256_store_pd(ptr, self.0) }
    }

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

#[allow(dead_code)]
impl AvxF64 {
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

    /// Negate all elements.
    #[inline]
    pub fn negate(self) -> Self {
        unsafe {
            let sign_mask = _mm256_set1_pd(-0.0);
            Self(_mm256_xor_pd(self.0, sign_mask))
        }
    }

    /// Permute elements within 128-bit lanes.
    /// Each 128-bit lane is permuted independently.
    #[inline]
    pub fn shuffle_within_lanes<const MASK: i32>(self) -> Self {
        unsafe { Self(_mm256_shuffle_pd(self.0, self.0, MASK)) }
    }

    /// Swap 128-bit lanes: [a, b, c, d] -> [c, d, a, b]
    #[inline]
    pub fn swap_lanes(self) -> Self {
        unsafe { Self(_mm256_permute2f128_pd(self.0, self.0, 0x01)) }
    }

    /// Interleave low elements from two vectors.
    #[inline]
    pub fn unpack_lo(self, other: Self) -> Self {
        unsafe { Self(_mm256_unpacklo_pd(self.0, other.0)) }
    }

    /// Interleave high elements from two vectors.
    #[inline]
    pub fn unpack_hi(self, other: Self) -> Self {
        unsafe { Self(_mm256_unpackhi_pd(self.0, other.0)) }
    }

    /// Blend elements from two vectors based on mask.
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

impl SimdComplex for AvxF64 {
    /// Complex multiply for 2 interleaved complex numbers.
    ///
    /// Format: [re0, im0, re1, im1]
    /// Computes: [(re0*re0' - im0*im0', re0*im0' + im0*re0'), (re1*re1' - im1*im1', re1*im1' + im1*re1')]
    #[inline]
    fn cmul(self, other: Self) -> Self {
        unsafe {
            // Duplicate real parts: [re0, re0, re1, re1]
            let a_re = _mm256_unpacklo_pd(self.0, self.0);
            // Duplicate imag parts: [im0, im0, im1, im1]
            let a_im = _mm256_unpackhi_pd(self.0, self.0);

            // Swap pairs in b: [im0, re0, im1, re1]
            let b_swap = _mm256_shuffle_pd(other.0, other.0, 0b0101);

            // prod1 = a_re * b = [re0*re0', re0*im0', re1*re1', re1*im1']
            let prod1 = _mm256_mul_pd(a_re, other.0);
            // prod2 = a_im * b_swap = [im0*im0', im0*re0', im1*im1', im1*re1']
            let prod2 = _mm256_mul_pd(a_im, b_swap);

            // Combine with addsub: [re*re - im*im, re*im + im*re, ...]
            Self(_mm256_addsub_pd(prod1, prod2))
        }
    }

    /// Complex conjugate multiply.
    #[inline]
    fn cmul_conj(self, other: Self) -> Self {
        unsafe {
            let a_re = _mm256_unpacklo_pd(self.0, self.0);
            let a_im = _mm256_unpackhi_pd(self.0, self.0);
            let b_swap = _mm256_shuffle_pd(other.0, other.0, 0b0101);

            let prod1 = _mm256_mul_pd(a_re, other.0);
            let prod2 = _mm256_mul_pd(a_im, b_swap);

            // For conjugate: swap add/sub pattern
            // [re*re + im*im, -re*im + im*re]
            let sign = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
            let prod1_signed = _mm256_xor_pd(prod1, sign);
            Self(_mm256_add_pd(prod1_signed, prod2))
        }
    }
}

impl SimdVector for AvxF32 {
    type Scalar = f32;
    const LANES: usize = 8;

    #[inline]
    fn splat(value: f32) -> Self {
        unsafe { Self(_mm256_set1_ps(value)) }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        unsafe { Self(_mm256_load_ps(ptr)) }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        unsafe { Self(_mm256_loadu_ps(ptr)) }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        unsafe { _mm256_store_ps(ptr, self.0) }
    }

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

#[allow(dead_code)]
impl AvxF32 {
    /// Create a vector from eight f32 values.
    #[inline]
    pub fn new(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> Self {
        unsafe { Self(_mm256_set_ps(h, g, f, e, d, c, b, a)) }
    }

    /// Negate all elements.
    #[inline]
    pub fn negate(self) -> Self {
        unsafe {
            let sign_mask = _mm256_set1_ps(-0.0);
            Self(_mm256_xor_ps(self.0, sign_mask))
        }
    }

    /// Interleave low elements.
    #[inline]
    pub fn unpack_lo(self, other: Self) -> Self {
        unsafe { Self(_mm256_unpacklo_ps(self.0, other.0)) }
    }

    /// Interleave high elements.
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

impl SimdComplex for AvxF32 {
    /// Complex multiply for 4 interleaved complex numbers.
    ///
    /// Format: [re0, im0, re1, im1, re2, im2, re3, im3]
    #[inline]
    fn cmul(self, other: Self) -> Self {
        unsafe {
            // Duplicate real parts: [re0, re0, re1, re1, re2, re2, re3, re3]
            let a_re = _mm256_shuffle_ps(self.0, self.0, 0b1010_0000);
            // Duplicate imag parts: [im0, im0, im1, im1, im2, im2, im3, im3]
            let a_im = _mm256_shuffle_ps(self.0, self.0, 0b1111_0101);

            // Swap pairs in b: [im0, re0, im1, re1, im2, re2, im3, re3]
            let b_swap = _mm256_shuffle_ps(other.0, other.0, 0b1011_0001);

            let prod1 = _mm256_mul_ps(a_re, other.0);
            let prod2 = _mm256_mul_ps(a_im, b_swap);

            // addsub pattern: [re*re - im*im, re*im + im*re, ...]
            Self(_mm256_addsub_ps(prod1, prod2))
        }
    }

    /// Complex conjugate multiply.
    #[inline]
    fn cmul_conj(self, other: Self) -> Self {
        unsafe {
            let a_re = _mm256_shuffle_ps(self.0, self.0, 0b1010_0000);
            let a_im = _mm256_shuffle_ps(self.0, self.0, 0b1111_0101);
            let b_swap = _mm256_shuffle_ps(other.0, other.0, 0b1011_0001);

            let prod1 = _mm256_mul_ps(a_re, other.0);
            let prod2 = _mm256_mul_ps(a_im, b_swap);

            // For conjugate: [re*re + im*im, -re*im + im*re]
            let sign = _mm256_set_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
            let prod1_signed = _mm256_xor_ps(prod1, sign);
            Self(_mm256_add_ps(prod1_signed, prod2))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_avx() -> bool {
        is_x86_feature_detected!("avx")
    }

    #[test]
    fn test_avx_f64_basic() {
        if !has_avx() {
            return;
        }

        let a = AvxF64::splat(2.0);
        let b = AvxF64::splat(3.0);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5.0);
        assert_eq!(sum.extract(3), 5.0);

        let diff = a.sub(b);
        assert_eq!(diff.extract(0), -1.0);

        let prod = a.mul(b);
        assert_eq!(prod.extract(0), 6.0);
    }

    #[test]
    fn test_avx_f64_new() {
        if !has_avx() {
            return;
        }

        let v = AvxF64::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.extract(0), 1.0);
        assert_eq!(v.extract(1), 2.0);
        assert_eq!(v.extract(2), 3.0);
        assert_eq!(v.extract(3), 4.0);
    }

    #[test]
    fn test_avx_f64_cmul() {
        if !has_avx() {
            return;
        }

        // Two complex: (1+2i), (3+4i)
        // (1+2i)*(5+6i) = (1*5-2*6) + (1*6+2*5)i = -7 + 16i
        // (3+4i)*(7+8i) = (3*7-4*8) + (3*8+4*7)i = -11 + 52i
        let a = AvxF64::new(1.0, 2.0, 3.0, 4.0);
        let b = AvxF64::new(5.0, 6.0, 7.0, 8.0);
        let c = a.cmul(b);
        assert!((c.extract(0) - (-7.0)).abs() < 1e-10);
        assert!((c.extract(1) - 16.0).abs() < 1e-10);
        assert!((c.extract(2) - (-11.0)).abs() < 1e-10);
        assert!((c.extract(3) - 52.0).abs() < 1e-10);
    }

    #[test]
    fn test_avx_f64_load_store() {
        if !has_avx() {
            return;
        }

        let data = [1.0_f64, 2.0, 3.0, 4.0];
        unsafe {
            let v = AvxF64::load_unaligned(data.as_ptr());
            assert_eq!(v.extract(0), 1.0);
            assert_eq!(v.extract(3), 4.0);

            let mut out = [0.0_f64; 4];
            v.store_unaligned(out.as_mut_ptr());
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_avx_f32_basic() {
        if !has_avx() {
            return;
        }

        let a = AvxF32::splat(2.0);
        let b = AvxF32::splat(3.0);

        let sum = a.add(b);
        let mut out = [0.0_f32; 8];
        unsafe { sum.store_unaligned(out.as_mut_ptr()) };
        assert_eq!(out, [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_avx_f32_cmul() {
        if !has_avx() {
            return;
        }

        // Four complex numbers
        let a = AvxF32::new(1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0);
        let b = AvxF32::new(5.0, 6.0, 7.0, 8.0, 1.0, 0.0, 0.0, 1.0);
        let c = a.cmul(b);
        let mut out = [0.0_f32; 8];
        unsafe { c.store_unaligned(out.as_mut_ptr()) };
        // (1+2i)*(5+6i) = -7 + 16i
        assert!((out[0] - (-7.0)).abs() < 1e-5);
        assert!((out[1] - 16.0).abs() < 1e-5);
        // (3+4i)*(7+8i) = -11 + 52i
        assert!((out[2] - (-11.0)).abs() < 1e-5);
        assert!((out[3] - 52.0).abs() < 1e-5);
        // (1+0i)*(1+0i) = 1+0i
        assert!((out[4] - 1.0).abs() < 1e-5);
        assert!((out[5] - 0.0).abs() < 1e-5);
        // (0+1i)*(0+1i) = -1+0i
        assert!((out[6] - (-1.0)).abs() < 1e-5);
        assert!((out[7] - 0.0).abs() < 1e-5);
    }
}
