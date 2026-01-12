//! ARM NEON SIMD implementation for aarch64.
//!
//! Provides 128-bit SIMD operations using NEON intrinsics.
//! - f64: 2 lanes (128-bit = 2 × 64-bit)
//! - f32: 4 lanes (128-bit = 4 × 32-bit)
//!
//! NEON is always available on aarch64, so no runtime detection is needed.

use super::traits::{SimdComplex, SimdVector};
use core::arch::aarch64::*;

/// NEON f64 vector type (2 lanes).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct NeonF64(pub float64x2_t);

/// NEON f32 vector type (4 lanes).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct NeonF32(pub float32x4_t);

// Safety: NEON vectors are POD types that can be safely sent between threads
unsafe impl Send for NeonF64 {}
unsafe impl Sync for NeonF64 {}
unsafe impl Send for NeonF32 {}
unsafe impl Sync for NeonF32 {}

impl SimdVector for NeonF64 {
    type Scalar = f64;
    const LANES: usize = 2;

    #[inline]
    fn splat(value: f64) -> Self {
        unsafe { Self(vdupq_n_f64(value)) }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        // NEON doesn't distinguish aligned/unaligned loads
        unsafe { Self(vld1q_f64(ptr)) }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        unsafe { Self(vld1q_f64(ptr)) }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        unsafe { vst1q_f64(ptr, self.0) }
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        unsafe { vst1q_f64(ptr, self.0) }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { Self(vaddq_f64(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { Self(vsubq_f64(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { Self(vmulq_f64(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { Self(vdivq_f64(self.0, other.0)) }
    }
}

#[allow(dead_code)]
impl NeonF64 {
    /// Create a vector from two f64 values: [a, b]
    #[inline]
    pub fn new(a: f64, b: f64) -> Self {
        let arr = [a, b];
        unsafe { Self(vld1q_f64(arr.as_ptr())) }
    }

    /// Extract element at index (0-1).
    #[inline]
    pub fn extract(self, idx: usize) -> f64 {
        debug_assert!(idx < 2);
        let mut arr = [0.0_f64; 2];
        unsafe { self.store_unaligned(arr.as_mut_ptr()) };
        arr[idx]
    }

    /// Fused multiply-add: self * a + b
    #[inline]
    pub fn fmadd(self, a: Self, b: Self) -> Self {
        unsafe { Self(vfmaq_f64(b.0, self.0, a.0)) }
    }

    /// Fused multiply-subtract: self * a - b
    #[inline]
    pub fn fmsub(self, a: Self, b: Self) -> Self {
        unsafe { Self(vfmsq_f64(Self::splat(0.0).sub(b).0, self.0, a.0)) }
    }

    /// Negate all elements.
    #[inline]
    pub fn negate(self) -> Self {
        unsafe { Self(vnegq_f64(self.0)) }
    }

    /// Get the low element.
    #[inline]
    pub fn low(self) -> f64 {
        unsafe { vgetq_lane_f64(self.0, 0) }
    }

    /// Get the high element.
    #[inline]
    pub fn high(self) -> f64 {
        unsafe { vgetq_lane_f64(self.0, 1) }
    }

    /// Swap lanes: [a, b] -> [b, a]
    #[inline]
    pub fn swap(self) -> Self {
        unsafe { Self(vextq_f64(self.0, self.0, 1)) }
    }

    /// Interleave low elements of self and other.
    #[inline]
    pub fn zip_lo(self, other: Self) -> Self {
        unsafe { Self(vzip1q_f64(self.0, other.0)) }
    }

    /// Interleave high elements of self and other.
    #[inline]
    pub fn zip_hi(self, other: Self) -> Self {
        unsafe { Self(vzip2q_f64(self.0, other.0)) }
    }
}

impl SimdComplex for NeonF64 {
    /// Complex multiply for 1 interleaved complex number.
    ///
    /// Format: [re, im]
    #[inline]
    fn cmul(self, other: Self) -> Self {
        unsafe {
            // self = [a, b] = re + im*i
            // other = [e, f] = re' + im'*i
            // Result: (a*e - b*f) + (a*f + b*e)*i

            // Duplicate real and imaginary parts
            let a_re = vdupq_lane_f64(vget_low_f64(self.0), 0); // [a, a]
            let a_im = vdupq_lane_f64(vget_high_f64(self.0), 0); // [b, b]

            // Swap other: [f, e]
            let b_flip = vextq_f64(other.0, other.0, 1);

            // prod1 = [a*e, a*f]
            let prod1 = vmulq_f64(a_re, other.0);
            // prod2 = [b*f, b*e]
            let prod2 = vmulq_f64(a_im, b_flip);

            // Result: [a*e - b*f, a*f + b*e]
            // NEON doesn't have addsub, so we need to manually negate and add
            let sign = Self::new(-1.0, 1.0);
            Self(vfmaq_f64(prod1, prod2, sign.0))
        }
    }

    /// Complex multiply with conjugate.
    #[inline]
    fn cmul_conj(self, other: Self) -> Self {
        unsafe {
            // Conjugate other: [e, -f]
            let neg_mask = Self::new(1.0, -1.0);
            let other_conj = vmulq_f64(other.0, neg_mask.0);

            let a_re = vdupq_lane_f64(vget_low_f64(self.0), 0);
            let a_im = vdupq_lane_f64(vget_high_f64(self.0), 0);
            let b_flip = vextq_f64(other_conj, other_conj, 1);

            let prod1 = vmulq_f64(a_re, other_conj);
            let prod2 = vmulq_f64(a_im, b_flip);

            let sign = Self::new(-1.0, 1.0);
            Self(vfmaq_f64(prod1, prod2, sign.0))
        }
    }
}

impl SimdVector for NeonF32 {
    type Scalar = f32;
    const LANES: usize = 4;

    #[inline]
    fn splat(value: f32) -> Self {
        unsafe { Self(vdupq_n_f32(value)) }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        unsafe { Self(vld1q_f32(ptr)) }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        unsafe { Self(vld1q_f32(ptr)) }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        unsafe { vst1q_f32(ptr, self.0) }
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        unsafe { vst1q_f32(ptr, self.0) }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { Self(vaddq_f32(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { Self(vsubq_f32(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { Self(vmulq_f32(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { Self(vdivq_f32(self.0, other.0)) }
    }
}

#[allow(dead_code)]
impl NeonF32 {
    /// Create a vector from four f32 values.
    #[inline]
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        let arr = [a, b, c, d];
        unsafe { Self(vld1q_f32(arr.as_ptr())) }
    }

    /// Extract element at index (0-3).
    #[inline]
    pub fn extract(self, idx: usize) -> f32 {
        debug_assert!(idx < 4);
        let mut arr = [0.0_f32; 4];
        unsafe { self.store_unaligned(arr.as_mut_ptr()) };
        arr[idx]
    }

    /// Fused multiply-add: self * a + b
    #[inline]
    pub fn fmadd(self, a: Self, b: Self) -> Self {
        unsafe { Self(vfmaq_f32(b.0, self.0, a.0)) }
    }

    /// Negate all elements.
    #[inline]
    pub fn negate(self) -> Self {
        unsafe { Self(vnegq_f32(self.0)) }
    }

    /// Interleave low elements.
    #[inline]
    pub fn zip_lo(self, other: Self) -> Self {
        unsafe { Self(vzip1q_f32(self.0, other.0)) }
    }

    /// Interleave high elements.
    #[inline]
    pub fn zip_hi(self, other: Self) -> Self {
        unsafe { Self(vzip2q_f32(self.0, other.0)) }
    }

    /// Get low half as 2-lane vector.
    #[inline]
    pub fn low_half(self) -> float32x2_t {
        unsafe { vget_low_f32(self.0) }
    }

    /// Get high half as 2-lane vector.
    #[inline]
    pub fn high_half(self) -> float32x2_t {
        unsafe { vget_high_f32(self.0) }
    }
}

impl SimdComplex for NeonF32 {
    /// Complex multiply for 2 interleaved complex numbers.
    ///
    /// Format: [re0, im0, re1, im1]
    #[inline]
    fn cmul(self, other: Self) -> Self {
        unsafe {
            // Use NEON's complex multiply instruction if available (ARMv8.3+)
            // Otherwise, manually compute

            // self = [a, b, c, d] = (a+bi), (c+di)
            // other = [e, f, g, h] = (e+fi), (g+hi)

            // Duplicate real parts: [a, a, c, c]
            let a_re = vtrn1q_f32(self.0, self.0);
            // Duplicate imag parts: [b, b, d, d]
            let a_im = vtrn2q_f32(self.0, self.0);

            // Swap pairs in other: [f, e, h, g]
            let b_flip = vrev64q_f32(other.0);

            // prod1 = a_re * other = [a*e, a*f, c*g, c*h]
            let prod1 = vmulq_f32(a_re, other.0);
            // prod2 = a_im * b_flip = [b*f, b*e, d*h, d*g]
            let prod2 = vmulq_f32(a_im, b_flip);

            // Result: [a*e - b*f, a*f + b*e, c*g - d*h, c*h + d*g]
            let sign = Self::new(-1.0, 1.0, -1.0, 1.0);
            Self(vfmaq_f32(prod1, prod2, sign.0))
        }
    }

    /// Complex multiply with conjugate.
    #[inline]
    fn cmul_conj(self, other: Self) -> Self {
        unsafe {
            // Conjugate other: [e, -f, g, -h]
            let neg_mask = Self::new(1.0, -1.0, 1.0, -1.0);
            let other_conj = vmulq_f32(other.0, neg_mask.0);

            let a_re = vtrn1q_f32(self.0, self.0);
            let a_im = vtrn2q_f32(self.0, self.0);
            let b_flip = vrev64q_f32(other_conj);

            let prod1 = vmulq_f32(a_re, other_conj);
            let prod2 = vmulq_f32(a_im, b_flip);

            let sign = Self::new(-1.0, 1.0, -1.0, 1.0);
            Self(vfmaq_f32(prod1, prod2, sign.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_f64_basic() {
        let a = NeonF64::splat(2.0);
        let b = NeonF64::splat(3.0);
        let c = a.add(b);

        assert_eq!(c.extract(0), 5.0);
        assert_eq!(c.extract(1), 5.0);
    }

    #[test]
    fn test_neon_f64_new() {
        let v = NeonF64::new(1.0, 2.0);
        assert_eq!(v.extract(0), 1.0);
        assert_eq!(v.extract(1), 2.0);
    }

    #[test]
    fn test_neon_f64_fmadd() {
        // a * b + c = 2 * 3 + 4 = 10
        let a = NeonF64::splat(2.0);
        let b = NeonF64::splat(3.0);
        let c = NeonF64::splat(4.0);
        let result = a.fmadd(b, c);

        assert_eq!(result.extract(0), 10.0);
        assert_eq!(result.extract(1), 10.0);
    }

    #[test]
    fn test_neon_f64_load_store() {
        let data = [1.0_f64, 2.0];
        let v = unsafe { NeonF64::load_unaligned(data.as_ptr()) };

        let mut out = [0.0_f64; 2];
        unsafe { v.store_unaligned(out.as_mut_ptr()) };

        assert_eq!(data, out);
    }

    #[test]
    fn test_neon_f64_cmul() {
        // (3 + 4i) * (1 + 2i) = (3*1 - 4*2) + (3*2 + 4*1)i = -5 + 10i
        let a = NeonF64::new(3.0, 4.0);
        let b = NeonF64::new(1.0, 2.0);
        let c = a.cmul(b);

        let tol = 1e-10;
        assert!((c.extract(0) - (-5.0)).abs() < tol);
        assert!((c.extract(1) - 10.0).abs() < tol);
    }

    #[test]
    fn test_neon_f32_basic() {
        let a = NeonF32::splat(2.0);
        let b = NeonF32::splat(3.0);
        let c = a.mul(b);

        for i in 0..4 {
            assert_eq!(c.extract(i), 6.0);
        }
    }

    #[test]
    fn test_neon_f32_new() {
        let v = NeonF32::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.extract(0), 1.0);
        assert_eq!(v.extract(1), 2.0);
        assert_eq!(v.extract(2), 3.0);
        assert_eq!(v.extract(3), 4.0);
    }

    #[test]
    fn test_neon_f32_fmadd() {
        // a * b + c = 2 * 3 + 4 = 10
        let a = NeonF32::splat(2.0);
        let b = NeonF32::splat(3.0);
        let c = NeonF32::splat(4.0);
        let result = a.fmadd(b, c);

        for i in 0..4 {
            assert_eq!(result.extract(i), 10.0);
        }
    }

    #[test]
    fn test_neon_f32_cmul() {
        // (3 + 4i) * (1 + 2i) = -5 + 10i
        let a = NeonF32::new(3.0, 4.0, 1.0, 0.0);
        let b = NeonF32::new(1.0, 2.0, 1.0, 0.0);
        let c = a.cmul(b);

        let tol = 1e-5;
        // First complex: (3+4i) * (1+2i) = -5 + 10i
        assert!((c.extract(0) - (-5.0)).abs() < tol);
        assert!((c.extract(1) - 10.0).abs() < tol);
        // Second complex: (1+0i) * (1+0i) = 1 + 0i
        assert!((c.extract(2) - 1.0).abs() < tol);
        assert!((c.extract(3) - 0.0).abs() < tol);
    }
}
