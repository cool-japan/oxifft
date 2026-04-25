//! AVX-512 SIMD implementation for x86_64.
//!
//! Provides 512-bit SIMD operations using AVX-512 intrinsics.
//! - f64: 8 lanes (512-bit = 8 × 64-bit)
//! - f32: 16 lanes (512-bit = 16 × 32-bit)
//!
//! AVX-512 provides the widest SIMD registers available on x86_64 and includes
//! many new instructions like masked operations and fused operations.

use super::traits::{SimdComplex, SimdVector};
use core::arch::x86_64::*;

/// Check if AVX-512F (foundation) is available at runtime.
#[inline]
pub fn has_avx512f() -> bool {
    is_x86_feature_detected!("avx512f")
}

/// AVX-512 f64 vector type (8 lanes).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Avx512F64(pub __m512d);

/// AVX-512 f32 vector type (16 lanes).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Avx512F32(pub __m512);

// Safety: AVX-512 vectors are POD types that can be safely sent between threads
unsafe impl Send for Avx512F64 {}
unsafe impl Sync for Avx512F64 {}
unsafe impl Send for Avx512F32 {}
unsafe impl Sync for Avx512F32 {}

impl SimdVector for Avx512F64 {
    type Scalar = f64;
    const LANES: usize = 8;

    #[inline]
    fn splat(value: f64) -> Self {
        unsafe { Self(_mm512_set1_pd(value)) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::load_aligned`] for the full safety contract.
    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        unsafe { Self(_mm512_load_pd(ptr)) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::load_unaligned`] for the full safety contract.
    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        unsafe { Self(_mm512_loadu_pd(ptr)) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::store_aligned`] for the full safety contract.
    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        unsafe { _mm512_store_pd(ptr, self.0) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::store_unaligned`] for the full safety contract.
    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        unsafe { _mm512_storeu_pd(ptr, self.0) }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { Self(_mm512_add_pd(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { Self(_mm512_sub_pd(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { Self(_mm512_mul_pd(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { Self(_mm512_div_pd(self.0, other.0)) }
    }
}

impl Avx512F64 {
    /// Create a vector from eight f64 values.
    #[inline]
    pub fn new(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> Self {
        unsafe { Self(_mm512_set_pd(h, g, f, e, d, c, b, a)) }
    }

    /// Extract element at index (0-7).
    #[inline]
    pub fn extract(self, idx: usize) -> f64 {
        debug_assert!(idx < 8);
        let mut arr = [0.0_f64; 8];
        unsafe { self.store_unaligned(arr.as_mut_ptr()) };
        arr[idx]
    }

    /// Fused multiply-add: self * a + b
    #[inline]
    pub fn fmadd(self, a: Self, b: Self) -> Self {
        unsafe { Self(_mm512_fmadd_pd(self.0, a.0, b.0)) }
    }

    /// Fused multiply-subtract: self * a - b
    #[inline]
    pub fn fmsub(self, a: Self, b: Self) -> Self {
        unsafe { Self(_mm512_fmsub_pd(self.0, a.0, b.0)) }
    }

    /// Fused negative multiply-add: -(self * a) + b
    #[inline]
    pub fn fnmadd(self, a: Self, b: Self) -> Self {
        unsafe { Self(_mm512_fnmadd_pd(self.0, a.0, b.0)) }
    }

    /// Fused negative multiply-subtract: -(self * a) - b
    #[inline]
    pub fn fnmsub(self, a: Self, b: Self) -> Self {
        unsafe { Self(_mm512_fnmsub_pd(self.0, a.0, b.0)) }
    }

    /// Negate all elements.
    #[inline]
    pub fn negate(self) -> Self {
        unsafe {
            let sign_mask = _mm512_set1_pd(-0.0);
            Self(_mm512_xor_pd(self.0, sign_mask))
        }
    }

    /// Get the low 256-bit lane as AVX vector.
    #[inline]
    pub fn low_256(self) -> super::avx::AvxF64 {
        unsafe { super::avx::AvxF64(_mm512_castpd512_pd256(self.0)) }
    }

    /// Get the high 256-bit lane as AVX vector.
    #[inline]
    pub fn high_256(self) -> super::avx::AvxF64 {
        unsafe { super::avx::AvxF64(_mm512_extractf64x4_pd(self.0, 1)) }
    }

    /// Permute elements within 128-bit lanes.
    #[inline]
    pub fn shuffle_within_lanes<const MASK: i32>(self) -> Self {
        unsafe { Self(_mm512_permute_pd(self.0, MASK)) }
    }

    /// Interleave low elements within 128-bit lanes.
    #[inline]
    pub fn unpack_lo(self, other: Self) -> Self {
        unsafe { Self(_mm512_unpacklo_pd(self.0, other.0)) }
    }

    /// Interleave high elements within 128-bit lanes.
    #[inline]
    pub fn unpack_hi(self, other: Self) -> Self {
        unsafe { Self(_mm512_unpackhi_pd(self.0, other.0)) }
    }
}

impl SimdComplex for Avx512F64 {
    /// Complex multiply for 4 interleaved complex numbers.
    ///
    /// Format: [re0, im0, re1, im1, re2, im2, re3, im3]
    #[inline]
    fn cmul(self, other: Self) -> Self {
        unsafe {
            // Shuffle to duplicate real and imaginary parts
            // a_re = [a, a, c, c, e, e, g, g]
            let a_re = _mm512_permute_pd(self.0, 0b0000_0000);
            // a_im = [b, b, d, d, f, f, h, h]
            let a_im = _mm512_permute_pd(self.0, 0b1111_1111);

            // b_flip = [im0, re0, im1, re1, ...] swap pairs
            let b_flip = _mm512_permute_pd(other.0, 0b0101_0101);

            // Use fmaddsub: alternating add/subtract for complex multiply
            // result = a_re * other -/+ (a_im * b_flip)
            Self(_mm512_fmaddsub_pd(
                a_re,
                other.0,
                _mm512_mul_pd(a_im, b_flip),
            ))
        }
    }

    /// Complex multiply with conjugate.
    #[inline]
    fn cmul_conj(self, other: Self) -> Self {
        unsafe {
            // Negate imaginary parts of other: [e, -f, g, -h, ...]
            let sign_mask = _mm512_set_pd(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
            let other_conj = _mm512_xor_pd(other.0, sign_mask);

            let a_re = _mm512_permute_pd(self.0, 0b0000_0000);
            let a_im = _mm512_permute_pd(self.0, 0b1111_1111);
            let b_flip = _mm512_permute_pd(other_conj, 0b0101_0101);

            Self(_mm512_fmaddsub_pd(
                a_re,
                other_conj,
                _mm512_mul_pd(a_im, b_flip),
            ))
        }
    }
}

impl SimdVector for Avx512F32 {
    type Scalar = f32;
    const LANES: usize = 16;

    #[inline]
    fn splat(value: f32) -> Self {
        unsafe { Self(_mm512_set1_ps(value)) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::load_aligned`] for the full safety contract.
    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        unsafe { Self(_mm512_load_ps(ptr)) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::load_unaligned`] for the full safety contract.
    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        unsafe { Self(_mm512_loadu_ps(ptr)) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::store_aligned`] for the full safety contract.
    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        unsafe { _mm512_store_ps(ptr, self.0) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::store_unaligned`] for the full safety contract.
    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        unsafe { _mm512_storeu_ps(ptr, self.0) }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { Self(_mm512_add_ps(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { Self(_mm512_sub_ps(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { Self(_mm512_mul_ps(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { Self(_mm512_div_ps(self.0, other.0)) }
    }
}

impl Avx512F32 {
    /// Fused multiply-add: self * a + b
    #[inline]
    pub fn fmadd(self, a: Self, b: Self) -> Self {
        unsafe { Self(_mm512_fmadd_ps(self.0, a.0, b.0)) }
    }

    /// Fused multiply-subtract: self * a - b
    #[inline]
    pub fn fmsub(self, a: Self, b: Self) -> Self {
        unsafe { Self(_mm512_fmsub_ps(self.0, a.0, b.0)) }
    }

    /// Fused negative multiply-add: -(self * a) + b
    #[inline]
    pub fn fnmadd(self, a: Self, b: Self) -> Self {
        unsafe { Self(_mm512_fnmadd_ps(self.0, a.0, b.0)) }
    }

    /// Fused negative multiply-subtract: -(self * a) - b
    #[inline]
    pub fn fnmsub(self, a: Self, b: Self) -> Self {
        unsafe { Self(_mm512_fnmsub_ps(self.0, a.0, b.0)) }
    }

    /// Negate all elements.
    #[inline]
    pub fn negate(self) -> Self {
        unsafe {
            let sign_mask = _mm512_set1_ps(-0.0);
            Self(_mm512_xor_ps(self.0, sign_mask))
        }
    }

    /// Get the low 256-bit lane as AVX vector.
    #[inline]
    pub fn low_256(self) -> super::avx::AvxF32 {
        unsafe { super::avx::AvxF32(_mm512_castps512_ps256(self.0)) }
    }

    /// Get the high 256-bit lane as AVX vector.
    #[inline]
    pub fn high_256(self) -> super::avx::AvxF32 {
        unsafe { super::avx::AvxF32(_mm512_extractf32x8_ps(self.0, 1)) }
    }

    /// Interleave low elements within 128-bit lanes.
    #[inline]
    pub fn unpack_lo(self, other: Self) -> Self {
        unsafe { Self(_mm512_unpacklo_ps(self.0, other.0)) }
    }

    /// Interleave high elements within 128-bit lanes.
    #[inline]
    pub fn unpack_hi(self, other: Self) -> Self {
        unsafe { Self(_mm512_unpackhi_ps(self.0, other.0)) }
    }
}

impl SimdComplex for Avx512F32 {
    /// Complex multiply for 8 interleaved complex numbers.
    ///
    /// Format: [re0, im0, re1, im1, ..., re7, im7]
    #[inline]
    fn cmul(self, other: Self) -> Self {
        unsafe {
            // Duplicate real and imaginary parts
            // moveldup: [a, a, c, c, ...]
            let a_re = _mm512_moveldup_ps(self.0);
            // movehdup: [b, b, d, d, ...]
            let a_im = _mm512_movehdup_ps(self.0);

            // Swap pairs in other: [im0, re0, im1, re1, ...]
            let b_flip = _mm512_permute_ps(other.0, 0b10_11_00_01);

            // Use fmaddsub for complex multiply
            Self(_mm512_fmaddsub_ps(
                a_re,
                other.0,
                _mm512_mul_ps(a_im, b_flip),
            ))
        }
    }

    /// Complex multiply with conjugate.
    #[inline]
    fn cmul_conj(self, other: Self) -> Self {
        unsafe {
            // Negate imaginary parts of other
            let sign_mask = _mm512_set_ps(
                -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0,
                0.0,
            );
            let other_conj = _mm512_xor_ps(other.0, sign_mask);

            let a_re = _mm512_moveldup_ps(self.0);
            let a_im = _mm512_movehdup_ps(self.0);
            let b_flip = _mm512_permute_ps(other_conj, 0b10_11_00_01);

            Self(_mm512_fmaddsub_ps(
                a_re,
                other_conj,
                _mm512_mul_ps(a_im, b_flip),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx512_f64_basic() {
        if !has_avx512f() {
            return;
        }

        let a = Avx512F64::splat(2.0);
        let b = Avx512F64::splat(3.0);
        let c = a.add(b);

        for i in 0..8 {
            assert_eq!(c.extract(i), 5.0);
        }
    }

    #[test]
    fn test_avx512_f64_new() {
        if !has_avx512f() {
            return;
        }

        let v = Avx512F64::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        for i in 0..8 {
            assert_eq!(v.extract(i), (i + 1) as f64);
        }
    }

    #[test]
    fn test_avx512_f64_fmadd() {
        if !has_avx512f() {
            return;
        }

        // a * b + c = 2 * 3 + 4 = 10
        let a = Avx512F64::splat(2.0);
        let b = Avx512F64::splat(3.0);
        let c = Avx512F64::splat(4.0);
        let result = a.fmadd(b, c);

        for i in 0..8 {
            assert_eq!(result.extract(i), 10.0);
        }
    }

    #[test]
    fn test_avx512_f64_load_store() {
        if !has_avx512f() {
            return;
        }

        let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v = unsafe { Avx512F64::load_unaligned(data.as_ptr()) };

        let mut out = [0.0_f64; 8];
        unsafe { v.store_unaligned(out.as_mut_ptr()) };

        assert_eq!(data, out);
    }

    #[test]
    fn test_avx512_f64_cmul() {
        if !has_avx512f() {
            return;
        }

        // (3 + 4i) * (1 + 2i) = (3*1 - 4*2) + (3*2 + 4*1)i = -5 + 10i
        let a = Avx512F64::new(3.0, 4.0, 1.0, 0.0, 1.0, 1.0, 2.0, 0.0);
        let b = Avx512F64::new(1.0, 2.0, 1.0, 0.0, 1.0, -1.0, 1.0, 0.0);
        let c = a.cmul(b);

        let tol = 1e-10;
        assert!((c.extract(0) - (-5.0)).abs() < tol);
        assert!((c.extract(1) - 10.0).abs() < tol);
    }

    #[test]
    fn test_avx512_f32_basic() {
        if !has_avx512f() {
            return;
        }

        let a = Avx512F32::splat(2.0);
        let b = Avx512F32::splat(3.0);
        let c = a.mul(b);

        let mut out = [0.0_f32; 16];
        unsafe { c.store_unaligned(out.as_mut_ptr()) };

        for val in &out {
            assert_eq!(*val, 6.0);
        }
    }

    #[test]
    fn test_avx512_f32_fmadd() {
        if !has_avx512f() {
            return;
        }

        // a * b + c = 2 * 3 + 4 = 10
        let a = Avx512F32::splat(2.0);
        let b = Avx512F32::splat(3.0);
        let c = Avx512F32::splat(4.0);
        let result = a.fmadd(b, c);

        let mut out = [0.0_f32; 16];
        unsafe { result.store_unaligned(out.as_mut_ptr()) };

        for val in &out {
            assert_eq!(*val, 10.0);
        }
    }

    #[test]
    fn test_avx512_f32_cmul() {
        if !has_avx512f() {
            return;
        }

        // (3 + 4i) * (1 + 2i) = -5 + 10i
        let mut input_a = [0.0_f32; 16];
        let mut input_b = [0.0_f32; 16];

        // Set first complex: 3+4i * 1+2i = -5+10i
        input_a[0] = 3.0;
        input_a[1] = 4.0;
        input_b[0] = 1.0;
        input_b[1] = 2.0;

        // Set second complex: 1+0i * 1+0i = 1+0i
        input_a[2] = 1.0;
        input_a[3] = 0.0;
        input_b[2] = 1.0;
        input_b[3] = 0.0;

        let a = unsafe { Avx512F32::load_unaligned(input_a.as_ptr()) };
        let b = unsafe { Avx512F32::load_unaligned(input_b.as_ptr()) };
        let c = a.cmul(b);

        let mut out = [0.0_f32; 16];
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
