//! SSE2 SIMD implementation for x86_64.
//!
//! Provides 128-bit SIMD operations using SSE2 intrinsics.
//! - f64: 2 lanes (128-bit = 2 × 64-bit)
//! - f32: 4 lanes (128-bit = 4 × 32-bit)

use super::traits::{SimdComplex, SimdVector};
use core::arch::x86_64::*;

/// SSE2 f64 vector type (2 lanes).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Sse2F64(pub __m128d);

/// SSE2 f32 vector type (4 lanes).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Sse2F32(pub __m128);

// Safety: SSE2 vectors are POD types that can be safely sent between threads
unsafe impl Send for Sse2F64 {}
unsafe impl Sync for Sse2F64 {}
unsafe impl Send for Sse2F32 {}
unsafe impl Sync for Sse2F32 {}

impl SimdVector for Sse2F64 {
    type Scalar = f64;
    const LANES: usize = 2;

    #[inline]
    fn splat(value: f64) -> Self {
        unsafe { Self(_mm_set1_pd(value)) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::load_aligned`] for the full safety contract.
    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        unsafe { Self(_mm_load_pd(ptr)) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::load_unaligned`] for the full safety contract.
    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        unsafe { Self(_mm_loadu_pd(ptr)) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::store_aligned`] for the full safety contract.
    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        unsafe { _mm_store_pd(ptr, self.0) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::store_unaligned`] for the full safety contract.
    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        unsafe { _mm_storeu_pd(ptr, self.0) }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { Self(_mm_add_pd(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { Self(_mm_sub_pd(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { Self(_mm_mul_pd(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { Self(_mm_div_pd(self.0, other.0)) }
    }
}

impl Sse2F64 {
    /// Create a vector from two f64 values: [a, b]
    #[inline]
    pub fn new(a: f64, b: f64) -> Self {
        unsafe { Self(_mm_set_pd(b, a)) }
    }

    /// Extract the low element (index 0).
    #[inline]
    pub fn low(self) -> f64 {
        unsafe { _mm_cvtsd_f64(self.0) }
    }

    /// Extract the high element (index 1).
    #[inline]
    pub fn high(self) -> f64 {
        unsafe { _mm_cvtsd_f64(_mm_unpackhi_pd(self.0, self.0)) }
    }

    /// Swap low and high elements: [a, b] -> [b, a]
    #[inline]
    pub fn swap(self) -> Self {
        unsafe { Self(_mm_shuffle_pd(self.0, self.0, 0b01)) }
    }

    /// Duplicate low element: [a, b] -> [a, a]
    #[inline]
    pub fn dup_low(self) -> Self {
        unsafe { Self(_mm_unpacklo_pd(self.0, self.0)) }
    }

    /// Duplicate high element: [a, b] -> [b, b]
    #[inline]
    pub fn dup_high(self) -> Self {
        unsafe { Self(_mm_unpackhi_pd(self.0, self.0)) }
    }

    /// Negate high element: [a, b] -> [a, -b]
    #[inline]
    pub fn negate_high(self) -> Self {
        unsafe {
            let sign_mask = _mm_set_pd(-0.0, 0.0);
            Self(_mm_xor_pd(self.0, sign_mask))
        }
    }

    /// Negate low element: [a, b] -> [-a, b]
    #[inline]
    pub fn negate_low(self) -> Self {
        unsafe {
            let sign_mask = _mm_set_pd(0.0, -0.0);
            Self(_mm_xor_pd(self.0, sign_mask))
        }
    }

    /// Negate both elements: [a, b] -> [-a, -b]
    #[inline]
    pub fn negate(self) -> Self {
        unsafe {
            let sign_mask = _mm_set1_pd(-0.0);
            Self(_mm_xor_pd(self.0, sign_mask))
        }
    }

    /// Horizontal add: returns [a+b, a+b]
    #[inline]
    pub fn hadd(self) -> Self {
        unsafe {
            let swapped = _mm_shuffle_pd(self.0, self.0, 0b01);
            Self(_mm_add_pd(self.0, swapped))
        }
    }

    /// Horizontal subtract: returns [a-b, a-b]
    #[inline]
    pub fn hsub(self) -> Self {
        unsafe {
            let swapped = _mm_shuffle_pd(self.0, self.0, 0b01);
            Self(_mm_sub_pd(self.0, swapped))
        }
    }

    /// Interleave low elements from two vectors: [a0, a1], [b0, b1] -> [a0, b0]
    #[inline]
    pub fn unpack_lo(self, other: Self) -> Self {
        unsafe { Self(_mm_unpacklo_pd(self.0, other.0)) }
    }

    /// Interleave high elements from two vectors: [a0, a1], [b0, b1] -> [a1, b1]
    #[inline]
    pub fn unpack_hi(self, other: Self) -> Self {
        unsafe { Self(_mm_unpackhi_pd(self.0, other.0)) }
    }
}

impl SimdComplex for Sse2F64 {
    /// Complex multiply for interleaved format: [re, im] * [re, im]
    ///
    /// For a = [a_re, a_im], b = [b_re, b_im]:
    /// Result = [a_re*b_re - a_im*b_im, a_re*b_im + a_im*b_re]
    #[inline]
    fn cmul(self, other: Self) -> Self {
        unsafe {
            // a = [a_re, a_im], b = [b_re, b_im]
            let a_re_re = _mm_unpacklo_pd(self.0, self.0); // [a_re, a_re]
            let a_im_im = _mm_unpackhi_pd(self.0, self.0); // [a_im, a_im]

            let b_im_re = _mm_shuffle_pd(other.0, other.0, 0b01); // [b_im, b_re]

            // real: a_re * b_re - a_im * b_im
            // imag: a_re * b_im + a_im * b_re
            let prod1 = _mm_mul_pd(a_re_re, other.0); // [a_re*b_re, a_re*b_im]
            let prod2 = _mm_mul_pd(a_im_im, b_im_re); // [a_im*b_im, a_im*b_re]

            // Combine: [a_re*b_re - a_im*b_im, a_re*b_im + a_im*b_re]
            let sign = _mm_set_pd(0.0, -0.0); // Negate low for subtraction
            let prod2_signed = _mm_xor_pd(prod2, sign);
            Self(_mm_add_pd(prod1, prod2_signed))
        }
    }

    /// Complex conjugate multiply: [re, im] * conj([re, im])
    ///
    /// For a = [a_re, a_im], b = [b_re, b_im]:
    /// Result = [a_re*b_re + a_im*b_im, -a_re*b_im + a_im*b_re]
    #[inline]
    fn cmul_conj(self, other: Self) -> Self {
        unsafe {
            let a_re_re = _mm_unpacklo_pd(self.0, self.0); // [a_re, a_re]
            let a_im_im = _mm_unpackhi_pd(self.0, self.0); // [a_im, a_im]

            let b_im_re = _mm_shuffle_pd(other.0, other.0, 0b01); // [b_im, b_re]

            let prod1 = _mm_mul_pd(a_re_re, other.0); // [a_re*b_re, a_re*b_im]
            let prod2 = _mm_mul_pd(a_im_im, b_im_re); // [a_im*b_im, a_im*b_re]

            // For conjugate: [a_re*b_re + a_im*b_im, -a_re*b_im + a_im*b_re]
            let sign = _mm_set_pd(-0.0, 0.0); // Negate high for conjugate
            let prod1_signed = _mm_xor_pd(prod1, sign);
            Self(_mm_add_pd(prod1_signed, prod2))
        }
    }
}

impl SimdVector for Sse2F32 {
    type Scalar = f32;
    const LANES: usize = 4;

    #[inline]
    fn splat(value: f32) -> Self {
        unsafe { Self(_mm_set1_ps(value)) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::load_aligned`] for the full safety contract.
    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        unsafe { Self(_mm_load_ps(ptr)) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::load_unaligned`] for the full safety contract.
    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        unsafe { Self(_mm_loadu_ps(ptr)) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::store_aligned`] for the full safety contract.
    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        unsafe { _mm_store_ps(ptr, self.0) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::store_unaligned`] for the full safety contract.
    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        unsafe { _mm_storeu_ps(ptr, self.0) }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { Self(_mm_add_ps(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { Self(_mm_sub_ps(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { Self(_mm_mul_ps(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { Self(_mm_div_ps(self.0, other.0)) }
    }
}

impl Sse2F32 {
    /// Create a vector from four f32 values: [a, b, c, d]
    #[inline]
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        unsafe { Self(_mm_set_ps(d, c, b, a)) }
    }

    /// Negate all elements.
    #[inline]
    pub fn negate(self) -> Self {
        unsafe {
            let sign_mask = _mm_set1_ps(-0.0);
            Self(_mm_xor_ps(self.0, sign_mask))
        }
    }

    /// Shuffle elements.
    #[inline]
    pub fn shuffle<const MASK: i32>(self, other: Self) -> Self {
        unsafe { Self(_mm_shuffle_ps(self.0, other.0, MASK)) }
    }

    /// Move high to low: [a, b, c, d] -> [c, d, c, d]
    #[inline]
    pub fn move_hl(self, other: Self) -> Self {
        unsafe { Self(_mm_movehl_ps(self.0, other.0)) }
    }

    /// Move low to high: [a, b, c, d] -> [a, b, a, b]
    #[inline]
    pub fn move_lh(self, other: Self) -> Self {
        unsafe { Self(_mm_movelh_ps(self.0, other.0)) }
    }

    /// Interleave low elements.
    #[inline]
    pub fn unpack_lo(self, other: Self) -> Self {
        unsafe { Self(_mm_unpacklo_ps(self.0, other.0)) }
    }

    /// Interleave high elements.
    #[inline]
    pub fn unpack_hi(self, other: Self) -> Self {
        unsafe { Self(_mm_unpackhi_ps(self.0, other.0)) }
    }
}

impl SimdComplex for Sse2F32 {
    /// Complex multiply for 2 interleaved complex numbers.
    ///
    /// Format: [re0, im0, re1, im1]
    /// Computes: [(re0*re0' - im0*im0', re0*im0' + im0*re0'), ...]
    #[inline]
    fn cmul(self, other: Self) -> Self {
        unsafe {
            // a = [a0_re, a0_im, a1_re, a1_im]
            // b = [b0_re, b0_im, b1_re, b1_im]

            // Duplicate real parts: [a0_re, a0_re, a1_re, a1_re]
            let a_re = _mm_shuffle_ps(self.0, self.0, 0b1010_0000);
            // Duplicate imag parts: [a0_im, a0_im, a1_im, a1_im]
            let a_im = _mm_shuffle_ps(self.0, self.0, 0b1111_0101);

            // Swap pairs in b: [b0_im, b0_re, b1_im, b1_re]
            let b_swap = _mm_shuffle_ps(other.0, other.0, 0b1011_0001);

            // prod1 = a_re * b = [a0_re*b0_re, a0_re*b0_im, a1_re*b1_re, a1_re*b1_im]
            let prod1 = _mm_mul_ps(a_re, other.0);
            // prod2 = a_im * b_swap = [a0_im*b0_im, a0_im*b0_re, a1_im*b1_im, a1_im*b1_re]
            let prod2 = _mm_mul_ps(a_im, b_swap);

            // Need: [re*re - im*im, re*im + im*re]
            // = prod1 + prod2 with sign adjustment on first of each pair
            let sign = _mm_set_ps(0.0, -0.0, 0.0, -0.0);
            let prod2_signed = _mm_xor_ps(prod2, sign);
            Self(_mm_add_ps(prod1, prod2_signed))
        }
    }

    /// Complex conjugate multiply.
    #[inline]
    fn cmul_conj(self, other: Self) -> Self {
        unsafe {
            let a_re = _mm_shuffle_ps(self.0, self.0, 0b1010_0000);
            let a_im = _mm_shuffle_ps(self.0, self.0, 0b1111_0101);
            let b_swap = _mm_shuffle_ps(other.0, other.0, 0b1011_0001);

            let prod1 = _mm_mul_ps(a_re, other.0);
            let prod2 = _mm_mul_ps(a_im, b_swap);

            // For conjugate: [re*re + im*im, -re*im + im*re]
            let sign = _mm_set_ps(-0.0, 0.0, -0.0, 0.0);
            let prod1_signed = _mm_xor_ps(prod1, sign);
            Self(_mm_add_ps(prod1_signed, prod2))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse2_f64_basic() {
        let a = Sse2F64::splat(2.0);
        let b = Sse2F64::splat(3.0);

        let sum = a.add(b);
        assert_eq!(sum.low(), 5.0);
        assert_eq!(sum.high(), 5.0);

        let diff = a.sub(b);
        assert_eq!(diff.low(), -1.0);
        assert_eq!(diff.high(), -1.0);

        let prod = a.mul(b);
        assert_eq!(prod.low(), 6.0);
        assert_eq!(prod.high(), 6.0);
    }

    #[test]
    fn test_sse2_f64_new() {
        let v = Sse2F64::new(1.0, 2.0);
        assert_eq!(v.low(), 1.0);
        assert_eq!(v.high(), 2.0);
    }

    #[test]
    fn test_sse2_f64_swap() {
        let v = Sse2F64::new(1.0, 2.0);
        let swapped = v.swap();
        assert_eq!(swapped.low(), 2.0);
        assert_eq!(swapped.high(), 1.0);
    }

    #[test]
    fn test_sse2_f64_negate() {
        let v = Sse2F64::new(1.0, 2.0);

        let neg = v.negate();
        assert_eq!(neg.low(), -1.0);
        assert_eq!(neg.high(), -2.0);

        let neg_low = v.negate_low();
        assert_eq!(neg_low.low(), -1.0);
        assert_eq!(neg_low.high(), 2.0);

        let neg_high = v.negate_high();
        assert_eq!(neg_high.low(), 1.0);
        assert_eq!(neg_high.high(), -2.0);
    }

    #[test]
    fn test_sse2_f64_cmul() {
        // (1 + 2i) * (3 + 4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
        let a = Sse2F64::new(1.0, 2.0);
        let b = Sse2F64::new(3.0, 4.0);
        let c = a.cmul(b);
        assert!((c.low() - (-5.0)).abs() < 1e-10);
        assert!((c.high() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_sse2_f64_cmul_conj() {
        // (1 + 2i) * conj(3 + 4i) = (1 + 2i) * (3 - 4i)
        // = (1*3 + 2*4) + (-1*4 + 2*3)i = 11 + 2i
        let a = Sse2F64::new(1.0, 2.0);
        let b = Sse2F64::new(3.0, 4.0);
        let c = a.cmul_conj(b);
        assert!((c.low() - 11.0).abs() < 1e-10);
        assert!((c.high() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sse2_f64_load_store() {
        let data = [1.0_f64, 2.0];
        let v = unsafe { Sse2F64::load_unaligned(data.as_ptr()) };
        assert_eq!(v.low(), 1.0);
        assert_eq!(v.high(), 2.0);

        let mut out = [0.0_f64; 2];
        unsafe { v.store_unaligned(out.as_mut_ptr()) };
        assert_eq!(out, [1.0, 2.0]);
    }

    #[test]
    fn test_sse2_f32_basic() {
        let a = Sse2F32::splat(2.0);
        let b = Sse2F32::splat(3.0);

        let sum = a.add(b);
        let mut out = [0.0_f32; 4];
        unsafe { sum.store_unaligned(out.as_mut_ptr()) };
        assert_eq!(out, [5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_sse2_f32_new() {
        let v = Sse2F32::new(1.0, 2.0, 3.0, 4.0);
        let mut out = [0.0_f32; 4];
        unsafe { v.store_unaligned(out.as_mut_ptr()) };
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sse2_f32_cmul() {
        // Two complex: (1+2i), (3+4i) * (5+6i), (7+8i)
        // (1+2i)*(5+6i) = (1*5-2*6) + (1*6+2*5)i = -7 + 16i
        // (3+4i)*(7+8i) = (3*7-4*8) + (3*8+4*7)i = -11 + 52i
        let a = Sse2F32::new(1.0, 2.0, 3.0, 4.0);
        let b = Sse2F32::new(5.0, 6.0, 7.0, 8.0);
        let c = a.cmul(b);
        let mut out = [0.0_f32; 4];
        unsafe { c.store_unaligned(out.as_mut_ptr()) };
        assert!((out[0] - (-7.0)).abs() < 1e-5);
        assert!((out[1] - 16.0).abs() < 1e-5);
        assert!((out[2] - (-11.0)).abs() < 1e-5);
        assert!((out[3] - 52.0).abs() < 1e-5);
    }
}
