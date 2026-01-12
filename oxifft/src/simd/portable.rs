//! Portable SIMD using std::simd (nightly feature).
//!
//! This module provides SIMD implementations using the portable_simd feature,
//! which is available on nightly Rust. When stable, this will become the
//! preferred cross-platform SIMD implementation.

use core::simd::{f32x4, f32x8, f64x2, f64x4, Simd};

use super::traits::{SimdComplex, SimdVector};

/// Portable SIMD f64x2 (2 lanes).
#[derive(Copy, Clone, Debug)]
pub struct PortableF64x2(pub f64x2);

impl SimdVector for PortableF64x2 {
    type Scalar = f64;
    const LANES: usize = 2;

    #[inline]
    fn splat(value: f64) -> Self {
        Self(f64x2::splat(value))
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        // SAFETY: Caller must ensure ptr is aligned and points to valid data
        unsafe { Self(Simd::from_array(*(ptr as *const [f64; 2]))) }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        // SAFETY: Caller must ensure ptr points to valid data
        unsafe {
            Self(Simd::from_array(core::ptr::read_unaligned(
                ptr as *const [f64; 2],
            )))
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        // SAFETY: Caller must ensure ptr is aligned and points to valid writable memory
        unsafe {
            *(ptr as *mut [f64; 2]) = self.0.to_array();
        }
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        // SAFETY: Caller must ensure ptr points to valid writable memory
        unsafe {
            core::ptr::write_unaligned(ptr as *mut [f64; 2], self.0.to_array());
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        Self(self.0 / other.0)
    }
}

impl SimdComplex for PortableF64x2 {
    #[inline]
    fn cmul(self, other: Self) -> Self {
        // For [re, im] format:
        // result_re = a_re * b_re - a_im * b_im
        // result_im = a_re * b_im + a_im * b_re
        let arr_a = self.0.to_array();
        let arr_b = other.0.to_array();

        let a_re = arr_a[0];
        let a_im = arr_a[1];
        let b_re = arr_b[0];
        let b_im = arr_b[1];

        Self(Simd::from_array([
            a_re * b_re - a_im * b_im,
            a_re * b_im + a_im * b_re,
        ]))
    }

    #[inline]
    fn cmul_conj(self, other: Self) -> Self {
        // Multiply by conjugate of other: [b_re, -b_im]
        let arr_a = self.0.to_array();
        let arr_b = other.0.to_array();

        let a_re = arr_a[0];
        let a_im = arr_a[1];
        let b_re = arr_b[0];
        let b_im = arr_b[1];

        Self(Simd::from_array([
            a_re * b_re + a_im * b_im,
            a_im * b_re - a_re * b_im,
        ]))
    }
}

/// Portable SIMD f64x4 (4 lanes).
#[derive(Copy, Clone, Debug)]
pub struct PortableF64x4(pub f64x4);

impl SimdVector for PortableF64x4 {
    type Scalar = f64;
    const LANES: usize = 4;

    #[inline]
    fn splat(value: f64) -> Self {
        Self(f64x4::splat(value))
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        // SAFETY: Caller must ensure ptr is aligned and points to valid data
        unsafe { Self(Simd::from_array(*(ptr as *const [f64; 4]))) }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        // SAFETY: Caller must ensure ptr points to valid data
        unsafe {
            Self(Simd::from_array(core::ptr::read_unaligned(
                ptr as *const [f64; 4],
            )))
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        // SAFETY: Caller must ensure ptr is aligned and points to valid writable memory
        unsafe {
            *(ptr as *mut [f64; 4]) = self.0.to_array();
        }
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        // SAFETY: Caller must ensure ptr points to valid writable memory
        unsafe {
            core::ptr::write_unaligned(ptr as *mut [f64; 4], self.0.to_array());
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        Self(self.0 / other.0)
    }
}

impl SimdComplex for PortableF64x4 {
    #[inline]
    fn cmul(self, other: Self) -> Self {
        // For interleaved complex: [re0, im0, re1, im1]
        let arr_a = self.0.to_array();
        let arr_b = other.0.to_array();

        // Complex 0
        let a0_re = arr_a[0];
        let a0_im = arr_a[1];
        let b0_re = arr_b[0];
        let b0_im = arr_b[1];

        // Complex 1
        let a1_re = arr_a[2];
        let a1_im = arr_a[3];
        let b1_re = arr_b[2];
        let b1_im = arr_b[3];

        Self(Simd::from_array([
            a0_re * b0_re - a0_im * b0_im,
            a0_re * b0_im + a0_im * b0_re,
            a1_re * b1_re - a1_im * b1_im,
            a1_re * b1_im + a1_im * b1_re,
        ]))
    }

    #[inline]
    fn cmul_conj(self, other: Self) -> Self {
        let arr_a = self.0.to_array();
        let arr_b = other.0.to_array();

        let a0_re = arr_a[0];
        let a0_im = arr_a[1];
        let b0_re = arr_b[0];
        let b0_im = arr_b[1];

        let a1_re = arr_a[2];
        let a1_im = arr_a[3];
        let b1_re = arr_b[2];
        let b1_im = arr_b[3];

        Self(Simd::from_array([
            a0_re * b0_re + a0_im * b0_im,
            a0_im * b0_re - a0_re * b0_im,
            a1_re * b1_re + a1_im * b1_im,
            a1_im * b1_re - a1_re * b1_im,
        ]))
    }
}

/// Portable SIMD f32x4 (4 lanes).
#[derive(Copy, Clone, Debug)]
pub struct PortableF32x4(pub f32x4);

impl SimdVector for PortableF32x4 {
    type Scalar = f32;
    const LANES: usize = 4;

    #[inline]
    fn splat(value: f32) -> Self {
        Self(f32x4::splat(value))
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        // SAFETY: Caller must ensure ptr is aligned and points to valid data
        unsafe { Self(Simd::from_array(*(ptr as *const [f32; 4]))) }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        // SAFETY: Caller must ensure ptr points to valid data
        unsafe {
            Self(Simd::from_array(core::ptr::read_unaligned(
                ptr as *const [f32; 4],
            )))
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        // SAFETY: Caller must ensure ptr is aligned and points to valid writable memory
        unsafe {
            *(ptr as *mut [f32; 4]) = self.0.to_array();
        }
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        // SAFETY: Caller must ensure ptr points to valid writable memory
        unsafe {
            core::ptr::write_unaligned(ptr as *mut [f32; 4], self.0.to_array());
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        Self(self.0 / other.0)
    }
}

impl SimdComplex for PortableF32x4 {
    #[inline]
    fn cmul(self, other: Self) -> Self {
        let arr_a = self.0.to_array();
        let arr_b = other.0.to_array();

        // Complex 0
        let a0_re = arr_a[0];
        let a0_im = arr_a[1];
        let b0_re = arr_b[0];
        let b0_im = arr_b[1];

        // Complex 1
        let a1_re = arr_a[2];
        let a1_im = arr_a[3];
        let b1_re = arr_b[2];
        let b1_im = arr_b[3];

        Self(Simd::from_array([
            a0_re * b0_re - a0_im * b0_im,
            a0_re * b0_im + a0_im * b0_re,
            a1_re * b1_re - a1_im * b1_im,
            a1_re * b1_im + a1_im * b1_re,
        ]))
    }

    #[inline]
    fn cmul_conj(self, other: Self) -> Self {
        let arr_a = self.0.to_array();
        let arr_b = other.0.to_array();

        let a0_re = arr_a[0];
        let a0_im = arr_a[1];
        let b0_re = arr_b[0];
        let b0_im = arr_b[1];

        let a1_re = arr_a[2];
        let a1_im = arr_a[3];
        let b1_re = arr_b[2];
        let b1_im = arr_b[3];

        Self(Simd::from_array([
            a0_re * b0_re + a0_im * b0_im,
            a0_im * b0_re - a0_re * b0_im,
            a1_re * b1_re + a1_im * b1_im,
            a1_im * b1_re - a1_re * b1_im,
        ]))
    }
}

/// Portable SIMD f32x8 (8 lanes).
#[derive(Copy, Clone, Debug)]
pub struct PortableF32x8(pub f32x8);

impl SimdVector for PortableF32x8 {
    type Scalar = f32;
    const LANES: usize = 8;

    #[inline]
    fn splat(value: f32) -> Self {
        Self(f32x8::splat(value))
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        // SAFETY: Caller must ensure ptr is aligned and points to valid data
        unsafe { Self(Simd::from_array(*(ptr as *const [f32; 8]))) }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        // SAFETY: Caller must ensure ptr points to valid data
        unsafe {
            Self(Simd::from_array(core::ptr::read_unaligned(
                ptr as *const [f32; 8],
            )))
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        // SAFETY: Caller must ensure ptr is aligned and points to valid writable memory
        unsafe {
            *(ptr as *mut [f32; 8]) = self.0.to_array();
        }
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        // SAFETY: Caller must ensure ptr points to valid writable memory
        unsafe {
            core::ptr::write_unaligned(ptr as *mut [f32; 8], self.0.to_array());
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        Self(self.0 / other.0)
    }
}

impl SimdComplex for PortableF32x8 {
    #[inline]
    fn cmul(self, other: Self) -> Self {
        let arr_a = self.0.to_array();
        let arr_b = other.0.to_array();

        let mut result = [0.0f32; 8];

        // 4 complex numbers
        for i in 0..4 {
            let idx = i * 2;
            let a_re = arr_a[idx];
            let a_im = arr_a[idx + 1];
            let b_re = arr_b[idx];
            let b_im = arr_b[idx + 1];

            result[idx] = a_re * b_re - a_im * b_im;
            result[idx + 1] = a_re * b_im + a_im * b_re;
        }

        Self(Simd::from_array(result))
    }

    #[inline]
    fn cmul_conj(self, other: Self) -> Self {
        let arr_a = self.0.to_array();
        let arr_b = other.0.to_array();

        let mut result = [0.0f32; 8];

        for i in 0..4 {
            let idx = i * 2;
            let a_re = arr_a[idx];
            let a_im = arr_a[idx + 1];
            let b_re = arr_b[idx];
            let b_im = arr_b[idx + 1];

            result[idx] = a_re * b_re + a_im * b_im;
            result[idx + 1] = a_im * b_re - a_re * b_im;
        }

        Self(Simd::from_array(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portable_f64x2_basic() {
        let a = PortableF64x2::splat(2.0);
        let b = PortableF64x2::splat(3.0);

        let sum = a.add(b);
        let arr = sum.0.to_array();
        assert!((arr[0] - 5.0).abs() < 1e-10);
        assert!((arr[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_portable_f64x2_cmul() {
        // (1 + 2i) * (3 + 4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
        let a = PortableF64x2(Simd::from_array([1.0, 2.0]));
        let b = PortableF64x2(Simd::from_array([3.0, 4.0]));

        let c = a.cmul(b);
        let arr = c.0.to_array();
        assert!((arr[0] - (-5.0)).abs() < 1e-10);
        assert!((arr[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_portable_f64x4_cmul() {
        // Two complex multiplications
        let a = PortableF64x4(Simd::from_array([1.0, 2.0, 3.0, 4.0]));
        let b = PortableF64x4(Simd::from_array([1.0, 0.0, 0.0, 1.0]));

        let c = a.cmul(b);
        let arr = c.0.to_array();

        // (1+2i) * (1+0i) = 1+2i
        assert!((arr[0] - 1.0).abs() < 1e-10);
        assert!((arr[1] - 2.0).abs() < 1e-10);

        // (3+4i) * (0+1i) = -4+3i
        assert!((arr[2] - (-4.0)).abs() < 1e-10);
        assert!((arr[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_portable_f32x4_load_store() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let v = unsafe { PortableF32x4::load_unaligned(data.as_ptr()) };

        let mut output = [0.0f32; 4];
        unsafe {
            v.store_unaligned(output.as_mut_ptr());
        }

        assert_eq!(data, output);
    }
}
