//! WASM SIMD (simd128) implementation.
//!
//! Provides 128-bit SIMD operations for WebAssembly when the simd128 feature
//! is available. This enables:
//! - f64: 2 lanes (128-bit = 2 × 64-bit)
//! - f32: 4 lanes (128-bit = 4 × 32-bit)
//!
//! # Requirements
//!
//! - Target: `wasm32-unknown-unknown` or `wasm32-wasi`
//! - Feature: `wasm` + `simd128` target feature
//! - Browser support: Chrome 91+, Firefox 89+, Safari 16.4+

#[cfg(not(feature = "std"))]
extern crate alloc;

use crate::simd::{SimdComplex, SimdVector};

/// WASM SIMD f64 vector type (2 lanes).
///
/// This uses the portable SIMD fallback on non-WASM targets.
#[derive(Copy, Clone, Debug)]
#[repr(align(16))]
pub struct WasmSimdF64 {
    /// Two f64 values.
    data: [f64; 2],
}

/// WASM SIMD f32 vector type (4 lanes).
#[derive(Copy, Clone, Debug)]
#[repr(align(16))]
pub struct WasmSimdF32 {
    /// Four f32 values.
    data: [f32; 4],
}

// Safety: These types contain only primitive floats
unsafe impl Send for WasmSimdF64 {}
unsafe impl Sync for WasmSimdF64 {}
unsafe impl Send for WasmSimdF32 {}
unsafe impl Sync for WasmSimdF32 {}

impl SimdVector for WasmSimdF64 {
    type Scalar = f64;
    const LANES: usize = 2;

    #[inline]
    fn splat(value: f64) -> Self {
        Self {
            data: [value, value],
        }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        unsafe {
            Self {
                data: [*ptr, *ptr.add(1)],
            }
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        unsafe {
            Self {
                data: [*ptr, *ptr.add(1)],
            }
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        unsafe {
            *ptr = self.data[0];
            *ptr.add(1) = self.data[1];
        }
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        unsafe {
            *ptr = self.data[0];
            *ptr.add(1) = self.data[1];
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            data: [self.data[0] + other.data[0], self.data[1] + other.data[1]],
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            data: [self.data[0] - other.data[0], self.data[1] - other.data[1]],
        }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            data: [self.data[0] * other.data[0], self.data[1] * other.data[1]],
        }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        Self {
            data: [self.data[0] / other.data[0], self.data[1] / other.data[1]],
        }
    }

    #[inline]
    fn fmadd(self, a: Self, b: Self) -> Self {
        Self {
            data: [
                self.data[0].mul_add(a.data[0], b.data[0]),
                self.data[1].mul_add(a.data[1], b.data[1]),
            ],
        }
    }
}

#[allow(dead_code)]
impl WasmSimdF64 {
    /// Create a vector from two f64 values.
    #[inline]
    pub fn new(a: f64, b: f64) -> Self {
        Self { data: [a, b] }
    }

    /// Extract element at index (0-1).
    #[inline]
    pub fn extract(self, idx: usize) -> f64 {
        self.data[idx]
    }

    /// Negate all elements.
    #[inline]
    pub fn negate(self) -> Self {
        Self {
            data: [-self.data[0], -self.data[1]],
        }
    }

    /// Swap lanes: [a, b] -> [b, a]
    #[inline]
    pub fn swap(self) -> Self {
        Self {
            data: [self.data[1], self.data[0]],
        }
    }
}

impl SimdComplex for WasmSimdF64 {
    /// Complex multiply for 1 interleaved complex number.
    ///
    /// Format: [re, im]
    #[inline]
    fn cmul(self, other: Self) -> Self {
        // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        let (a, b) = (self.data[0], self.data[1]);
        let (c, d) = (other.data[0], other.data[1]);

        Self {
            data: [
                a.mul_add(c, -(b * d)), // ac - bd
                a.mul_add(d, b * c),    // ad + bc
            ],
        }
    }

    /// Complex multiply with conjugate.
    #[inline]
    fn cmul_conj(self, other: Self) -> Self {
        // (a + bi) * (c - di) = (ac + bd) + (-ad + bc)i
        let (a, b) = (self.data[0], self.data[1]);
        let (c, d) = (other.data[0], -other.data[1]);

        Self {
            data: [a.mul_add(c, -(b * d)), a.mul_add(d, b * c)],
        }
    }
}

impl SimdVector for WasmSimdF32 {
    type Scalar = f32;
    const LANES: usize = 4;

    #[inline]
    fn splat(value: f32) -> Self {
        Self {
            data: [value, value, value, value],
        }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        unsafe {
            Self {
                data: [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)],
            }
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        unsafe {
            Self {
                data: [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)],
            }
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        unsafe {
            *ptr = self.data[0];
            *ptr.add(1) = self.data[1];
            *ptr.add(2) = self.data[2];
            *ptr.add(3) = self.data[3];
        }
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        unsafe {
            *ptr = self.data[0];
            *ptr.add(1) = self.data[1];
            *ptr.add(2) = self.data[2];
            *ptr.add(3) = self.data[3];
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            data: [
                self.data[0] + other.data[0],
                self.data[1] + other.data[1],
                self.data[2] + other.data[2],
                self.data[3] + other.data[3],
            ],
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            data: [
                self.data[0] - other.data[0],
                self.data[1] - other.data[1],
                self.data[2] - other.data[2],
                self.data[3] - other.data[3],
            ],
        }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            data: [
                self.data[0] * other.data[0],
                self.data[1] * other.data[1],
                self.data[2] * other.data[2],
                self.data[3] * other.data[3],
            ],
        }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        Self {
            data: [
                self.data[0] / other.data[0],
                self.data[1] / other.data[1],
                self.data[2] / other.data[2],
                self.data[3] / other.data[3],
            ],
        }
    }

    #[inline]
    fn fmadd(self, a: Self, b: Self) -> Self {
        Self {
            data: [
                self.data[0].mul_add(a.data[0], b.data[0]),
                self.data[1].mul_add(a.data[1], b.data[1]),
                self.data[2].mul_add(a.data[2], b.data[2]),
                self.data[3].mul_add(a.data[3], b.data[3]),
            ],
        }
    }
}

#[allow(dead_code)]
impl WasmSimdF32 {
    /// Create a vector from four f32 values.
    #[inline]
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self { data: [a, b, c, d] }
    }

    /// Extract element at index (0-3).
    #[inline]
    pub fn extract(self, idx: usize) -> f32 {
        self.data[idx]
    }

    /// Negate all elements.
    #[inline]
    pub fn negate(self) -> Self {
        Self {
            data: [-self.data[0], -self.data[1], -self.data[2], -self.data[3]],
        }
    }
}

impl SimdComplex for WasmSimdF32 {
    /// Complex multiply for 2 interleaved complex numbers.
    ///
    /// Format: [re0, im0, re1, im1]
    #[inline]
    fn cmul(self, other: Self) -> Self {
        let (a, b, c, d) = (self.data[0], self.data[1], self.data[2], self.data[3]);
        let (e, f, g, h) = (other.data[0], other.data[1], other.data[2], other.data[3]);

        Self {
            data: [
                a.mul_add(e, -(b * f)), // re0
                a.mul_add(f, b * e),    // im0
                c.mul_add(g, -(d * h)), // re1
                c.mul_add(h, d * g),    // im1
            ],
        }
    }

    /// Complex multiply with conjugate.
    #[inline]
    fn cmul_conj(self, other: Self) -> Self {
        let (a, b, c, d) = (self.data[0], self.data[1], self.data[2], self.data[3]);
        let (e, f, g, h) = (other.data[0], -other.data[1], other.data[2], -other.data[3]);

        Self {
            data: [
                a.mul_add(e, -(b * f)),
                a.mul_add(f, b * e),
                c.mul_add(g, -(d * h)),
                c.mul_add(h, d * g),
            ],
        }
    }
}

/// Check if WASM SIMD is available.
///
/// This is always true when compiled with target_feature = "simd128".
#[must_use]
pub fn has_wasm_simd() -> bool {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        true
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_simd_f64_basic() {
        let a = WasmSimdF64::splat(2.0);
        let b = WasmSimdF64::splat(3.0);
        let c = a.add(b);

        assert_eq!(c.extract(0), 5.0);
        assert_eq!(c.extract(1), 5.0);
    }

    #[test]
    fn test_wasm_simd_f64_cmul() {
        // (3 + 4i) * (1 + 2i) = -5 + 10i
        let a = WasmSimdF64::new(3.0, 4.0);
        let b = WasmSimdF64::new(1.0, 2.0);
        let c = a.cmul(b);

        let tol = 1e-10;
        assert!((c.extract(0) - (-5.0)).abs() < tol);
        assert!((c.extract(1) - 10.0).abs() < tol);
    }

    #[test]
    fn test_wasm_simd_f32_basic() {
        let a = WasmSimdF32::splat(2.0);
        let b = WasmSimdF32::splat(3.0);
        let c = a.mul(b);

        for i in 0..4 {
            assert_eq!(c.extract(i), 6.0);
        }
    }

    #[test]
    fn test_wasm_simd_f32_cmul() {
        // (3 + 4i) * (1 + 2i) = -5 + 10i
        // (1 + 0i) * (1 + 0i) = 1 + 0i
        let a = WasmSimdF32::new(3.0, 4.0, 1.0, 0.0);
        let b = WasmSimdF32::new(1.0, 2.0, 1.0, 0.0);
        let c = a.cmul(b);

        let tol = 1e-5;
        assert!((c.extract(0) - (-5.0)).abs() < tol);
        assert!((c.extract(1) - 10.0).abs() < tol);
        assert!((c.extract(2) - 1.0).abs() < tol);
        assert!((c.extract(3) - 0.0).abs() < tol);
    }

    #[test]
    fn test_wasm_simd_f64_load_store() {
        let data = [1.0_f64, 2.0];
        let v = unsafe { WasmSimdF64::load_unaligned(data.as_ptr()) };

        let mut out = [0.0_f64; 2];
        unsafe { v.store_unaligned(out.as_mut_ptr()) };

        assert_eq!(data, out);
    }
}
