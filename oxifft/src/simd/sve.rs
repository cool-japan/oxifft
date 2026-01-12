//! ARM SVE (Scalable Vector Extension) SIMD implementation.
//!
//! SVE provides scalable vectors with lengths from 128 to 2048 bits,
//! determined at runtime. This enables portable high-performance code
//! across different SVE implementations.
//!
//! # Vector Lengths
//!
//! | Platform | Vector Length | f64 Lanes | f32 Lanes |
//! |----------|---------------|-----------|-----------|
//! | AWS Graviton3 | 256-bit | 4 | 8 |
//! | Fujitsu A64FX | 512-bit | 8 | 16 |
//! | Apple M4 (?) | 128-bit | 2 | 4 |
//!
//! # Requirements
//!
//! - ARM64 architecture with SVE extension
//! - Feature flag: `sve`
//! - Nightly Rust for SVE intrinsics (when using actual SVE)
//!
//! # Current Implementation
//!
//! This module provides SVE-compatible wrappers that use the portable SIMD
//! backend internally, providing a migration path for when Rust's SVE
//! intrinsics stabilize.

use super::traits::{SimdComplex, SimdVector};

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Get the runtime SVE vector length in bytes.
///
/// Returns 0 if SVE is not available.
#[cfg(all(target_arch = "aarch64", feature = "sve"))]
pub fn sve_vector_length_bytes() -> usize {
    // In actual SVE, this would call svcntb() intrinsic
    // For now, we detect via HWCAP or return a default
    detect_sve_length()
}

#[cfg(not(all(target_arch = "aarch64", feature = "sve")))]
pub fn sve_vector_length_bytes() -> usize {
    0
}

/// Detect SVE vector length.
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn detect_sve_length() -> usize {
    // Read from /proc/sys/abi/sve_default_vector_length or use getauxval
    // For safety, return 0 and let caller fallback to NEON
    0
}

#[cfg(all(target_arch = "aarch64", not(target_os = "linux")))]
fn detect_sve_length() -> usize {
    0
}

/// Get number of f64 lanes for current SVE implementation.
pub fn sve_f64_lanes() -> usize {
    let bytes = sve_vector_length_bytes();
    if bytes == 0 {
        0
    } else {
        bytes / 8 // 8 bytes per f64
    }
}

/// Get number of f32 lanes for current SVE implementation.
pub fn sve_f32_lanes() -> usize {
    let bytes = sve_vector_length_bytes();
    if bytes == 0 {
        0
    } else {
        bytes / 4 // 4 bytes per f32
    }
}

/// Check if SVE is available.
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "sve"))]
pub fn has_sve() -> bool {
    // Check HWCAP for SVE support
    // HWCAP_SVE = 1 << 22
    #[allow(unsafe_code)]
    unsafe {
        const AT_HWCAP: libc::c_ulong = 16;
        const HWCAP_SVE: u64 = 1 << 22;
        let hwcap = libc::getauxval(AT_HWCAP);
        (hwcap & HWCAP_SVE) != 0
    }
}

#[cfg(not(all(target_arch = "aarch64", target_os = "linux", feature = "sve")))]
pub fn has_sve() -> bool {
    false
}

// Note: Variable-length SVE types would require actual SVE intrinsics.
// For now, we provide fixed-size Sve256 types that match common deployments
// like AWS Graviton3 (256-bit SVE vectors).

/// Fixed-size SVE simulation for 256-bit (4 lanes f64).
///
/// This type provides a concrete implementation for the most common
/// SVE deployment (AWS Graviton3, 256-bit vectors).
#[derive(Copy, Clone, Debug)]
#[repr(align(32))]
pub struct Sve256F64 {
    /// Four f64 values representing a 256-bit SVE vector.
    data: [f64; 4],
}

/// Fixed-size SVE simulation for 256-bit (8 lanes f32).
#[derive(Copy, Clone, Debug)]
#[repr(align(32))]
pub struct Sve256F32 {
    /// Eight f32 values representing a 256-bit SVE vector.
    data: [f32; 8],
}

// Safety: These types contain only primitive floats
unsafe impl Send for Sve256F64 {}
unsafe impl Sync for Sve256F64 {}
unsafe impl Send for Sve256F32 {}
unsafe impl Sync for Sve256F32 {}

impl SimdVector for Sve256F64 {
    type Scalar = f64;
    const LANES: usize = 4;

    #[inline]
    fn splat(value: f64) -> Self {
        Self {
            data: [value, value, value, value],
        }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        // SVE supports both aligned and unaligned loads efficiently
        unsafe {
            Self {
                data: [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)],
            }
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        unsafe {
            Self {
                data: [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)],
            }
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        unsafe {
            *ptr = self.data[0];
            *ptr.add(1) = self.data[1];
            *ptr.add(2) = self.data[2];
            *ptr.add(3) = self.data[3];
        }
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
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
        // Fused multiply-add: self * a + b
        Self {
            data: [
                self.data[0].mul_add(a.data[0], b.data[0]),
                self.data[1].mul_add(a.data[1], b.data[1]),
                self.data[2].mul_add(a.data[2], b.data[2]),
                self.data[3].mul_add(a.data[3], b.data[3]),
            ],
        }
    }

    #[inline]
    fn fmsub(self, a: Self, b: Self) -> Self {
        // Fused multiply-subtract: self * a - b
        Self {
            data: [
                self.data[0].mul_add(a.data[0], -b.data[0]),
                self.data[1].mul_add(a.data[1], -b.data[1]),
                self.data[2].mul_add(a.data[2], -b.data[2]),
                self.data[3].mul_add(a.data[3], -b.data[3]),
            ],
        }
    }
}

#[allow(dead_code)]
impl Sve256F64 {
    /// Create a vector from four f64 values.
    #[inline]
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self { data: [a, b, c, d] }
    }

    /// Extract element at index (0-3).
    #[inline]
    pub fn extract(self, idx: usize) -> f64 {
        self.data[idx]
    }

    /// Negate all elements.
    #[inline]
    pub fn negate(self) -> Self {
        Self {
            data: [-self.data[0], -self.data[1], -self.data[2], -self.data[3]],
        }
    }

    /// Get low pair: `[data[0], data[1]]`
    #[inline]
    pub fn low_pair(self) -> (f64, f64) {
        (self.data[0], self.data[1])
    }

    /// Get high pair: `[data[2], data[3]]`
    #[inline]
    pub fn high_pair(self) -> (f64, f64) {
        (self.data[2], self.data[3])
    }

    /// Interleave low elements of two vectors.
    #[inline]
    pub fn zip_lo(self, other: Self) -> Self {
        Self {
            data: [self.data[0], other.data[0], self.data[1], other.data[1]],
        }
    }

    /// Interleave high elements of two vectors.
    #[inline]
    pub fn zip_hi(self, other: Self) -> Self {
        Self {
            data: [self.data[2], other.data[2], self.data[3], other.data[3]],
        }
    }

    /// Predicated add: only add where mask is true.
    ///
    /// This simulates SVE's predicated operations.
    #[inline]
    pub fn add_predicated(self, other: Self, mask: [bool; 4]) -> Self {
        Self {
            data: [
                if mask[0] {
                    self.data[0] + other.data[0]
                } else {
                    self.data[0]
                },
                if mask[1] {
                    self.data[1] + other.data[1]
                } else {
                    self.data[1]
                },
                if mask[2] {
                    self.data[2] + other.data[2]
                } else {
                    self.data[2]
                },
                if mask[3] {
                    self.data[3] + other.data[3]
                } else {
                    self.data[3]
                },
            ],
        }
    }

    /// Predicated load: only load where mask is true.
    ///
    /// # Safety
    /// Pointer must be valid for elements where mask is true.
    #[inline]
    pub unsafe fn load_predicated(ptr: *const f64, mask: [bool; 4]) -> Self {
        unsafe {
            Self {
                data: [
                    if mask[0] { *ptr } else { 0.0 },
                    if mask[1] { *ptr.add(1) } else { 0.0 },
                    if mask[2] { *ptr.add(2) } else { 0.0 },
                    if mask[3] { *ptr.add(3) } else { 0.0 },
                ],
            }
        }
    }

    /// Predicated store: only store where mask is true.
    ///
    /// # Safety
    /// Pointer must be valid for elements where mask is true.
    #[inline]
    pub unsafe fn store_predicated(self, ptr: *mut f64, mask: [bool; 4]) {
        unsafe {
            if mask[0] {
                *ptr = self.data[0];
            }
            if mask[1] {
                *ptr.add(1) = self.data[1];
            }
            if mask[2] {
                *ptr.add(2) = self.data[2];
            }
            if mask[3] {
                *ptr.add(3) = self.data[3];
            }
        }
    }
}

impl SimdComplex for Sve256F64 {
    /// Complex multiply for 2 interleaved complex numbers.
    ///
    /// Format: [re0, im0, re1, im1]
    #[inline]
    fn cmul(self, other: Self) -> Self {
        // self = [a, b, c, d] = (a+bi), (c+di)
        // other = [e, f, g, h] = (e+fi), (g+hi)
        // Result: [(a*e - b*f), (a*f + b*e), (c*g - d*h), (c*h + d*g)]

        let (a, b, c, d) = (self.data[0], self.data[1], self.data[2], self.data[3]);
        let (e, f, g, h) = (other.data[0], other.data[1], other.data[2], other.data[3]);

        Self {
            data: [
                a.mul_add(e, -(b * f)), // a*e - b*f
                a.mul_add(f, b * e),    // a*f + b*e
                c.mul_add(g, -(d * h)), // c*g - d*h
                c.mul_add(h, d * g),    // c*h + d*g
            ],
        }
    }

    /// Complex multiply with conjugate.
    ///
    /// Format: [re0, im0, re1, im1]
    #[inline]
    fn cmul_conj(self, other: Self) -> Self {
        // Conjugate other: [e, -f, g, -h]
        // Then multiply

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

impl SimdVector for Sve256F32 {
    type Scalar = f32;
    const LANES: usize = 8;

    #[inline]
    fn splat(value: f32) -> Self {
        Self { data: [value; 8] }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        let mut data = [0.0_f32; 8];
        unsafe {
            for i in 0..8 {
                data[i] = *ptr.add(i);
            }
        }
        Self { data }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        let mut data = [0.0_f32; 8];
        unsafe {
            for i in 0..8 {
                data[i] = *ptr.add(i);
            }
        }
        Self { data }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        unsafe {
            for i in 0..8 {
                *ptr.add(i) = self.data[i];
            }
        }
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        unsafe {
            for i in 0..8 {
                *ptr.add(i) = self.data[i];
            }
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        let mut data = [0.0_f32; 8];
        for i in 0..8 {
            data[i] = self.data[i] + other.data[i];
        }
        Self { data }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        let mut data = [0.0_f32; 8];
        for i in 0..8 {
            data[i] = self.data[i] - other.data[i];
        }
        Self { data }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        let mut data = [0.0_f32; 8];
        for i in 0..8 {
            data[i] = self.data[i] * other.data[i];
        }
        Self { data }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        let mut data = [0.0_f32; 8];
        for i in 0..8 {
            data[i] = self.data[i] / other.data[i];
        }
        Self { data }
    }

    #[inline]
    fn fmadd(self, a: Self, b: Self) -> Self {
        let mut data = [0.0_f32; 8];
        for i in 0..8 {
            data[i] = self.data[i].mul_add(a.data[i], b.data[i]);
        }
        Self { data }
    }
}

#[allow(dead_code)]
impl Sve256F32 {
    /// Create a vector from eight f32 values.
    #[inline]
    pub fn new(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> Self {
        Self {
            data: [a, b, c, d, e, f, g, h],
        }
    }

    /// Extract element at index (0-7).
    #[inline]
    pub fn extract(self, idx: usize) -> f32 {
        self.data[idx]
    }

    /// Negate all elements.
    #[inline]
    pub fn negate(self) -> Self {
        let mut data = [0.0_f32; 8];
        for i in 0..8 {
            data[i] = -self.data[i];
        }
        Self { data }
    }
}

impl SimdComplex for Sve256F32 {
    /// Complex multiply for 4 interleaved complex numbers.
    ///
    /// Format: [re0, im0, re1, im1, re2, im2, re3, im3]
    #[inline]
    fn cmul(self, other: Self) -> Self {
        let mut result = [0.0_f32; 8];

        // Process 4 complex numbers
        for i in 0..4 {
            let re_idx = i * 2;
            let im_idx = i * 2 + 1;

            let a = self.data[re_idx];
            let b = self.data[im_idx];
            let e = other.data[re_idx];
            let f = other.data[im_idx];

            result[re_idx] = a.mul_add(e, -(b * f));
            result[im_idx] = a.mul_add(f, b * e);
        }

        Self { data: result }
    }

    /// Complex multiply with conjugate.
    #[inline]
    fn cmul_conj(self, other: Self) -> Self {
        let mut result = [0.0_f32; 8];

        for i in 0..4 {
            let re_idx = i * 2;
            let im_idx = i * 2 + 1;

            let a = self.data[re_idx];
            let b = self.data[im_idx];
            let e = other.data[re_idx];
            let f = -other.data[im_idx]; // Conjugate

            result[re_idx] = a.mul_add(e, -(b * f));
            result[im_idx] = a.mul_add(f, b * e);
        }

        Self { data: result }
    }
}

/// SVE predicate type for masked operations.
///
/// Represents a vector of boolean predicates for SVE's predicated instructions.
#[derive(Copy, Clone, Debug)]
pub struct SvePredicate<const N: usize> {
    /// Boolean array representing active lanes.
    pub active: [bool; N],
}

impl<const N: usize> SvePredicate<N> {
    /// Create an all-true predicate (equivalent to svptrue).
    #[inline]
    pub fn all_true() -> Self {
        Self { active: [true; N] }
    }

    /// Create an all-false predicate.
    #[inline]
    pub fn all_false() -> Self {
        Self { active: [false; N] }
    }

    /// Create a predicate with first `count` elements true (equivalent to svwhilelt).
    #[inline]
    pub fn while_lt(count: usize) -> Self {
        let mut active = [false; N];
        for item in active.iter_mut().take(count.min(N)) {
            *item = true;
        }
        Self { active }
    }

    /// Count active lanes.
    #[inline]
    pub fn count_active(&self) -> usize {
        self.active.iter().filter(|&&x| x).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sve256_f64_basic() {
        let a = Sve256F64::splat(2.0);
        let b = Sve256F64::splat(3.0);
        let c = a.add(b);

        assert_eq!(c.extract(0), 5.0);
        assert_eq!(c.extract(1), 5.0);
        assert_eq!(c.extract(2), 5.0);
        assert_eq!(c.extract(3), 5.0);
    }

    #[test]
    fn test_sve256_f64_new() {
        let v = Sve256F64::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.extract(0), 1.0);
        assert_eq!(v.extract(1), 2.0);
        assert_eq!(v.extract(2), 3.0);
        assert_eq!(v.extract(3), 4.0);
    }

    #[test]
    fn test_sve256_f64_fmadd() {
        // a * b + c = 2 * 3 + 4 = 10
        let a = Sve256F64::splat(2.0);
        let b = Sve256F64::splat(3.0);
        let c = Sve256F64::splat(4.0);
        let result = a.fmadd(b, c);

        for i in 0..4 {
            assert_eq!(result.extract(i), 10.0);
        }
    }

    #[test]
    fn test_sve256_f64_load_store() {
        let data = [1.0_f64, 2.0, 3.0, 4.0];
        let v = unsafe { Sve256F64::load_unaligned(data.as_ptr()) };

        let mut out = [0.0_f64; 4];
        unsafe { v.store_unaligned(out.as_mut_ptr()) };

        assert_eq!(data, out);
    }

    #[test]
    fn test_sve256_f64_cmul() {
        // Two complex multiplications:
        // (3 + 4i) * (1 + 2i) = (3*1 - 4*2) + (3*2 + 4*1)i = -5 + 10i
        // (1 + 0i) * (1 + 0i) = 1 + 0i
        let a = Sve256F64::new(3.0, 4.0, 1.0, 0.0);
        let b = Sve256F64::new(1.0, 2.0, 1.0, 0.0);
        let c = a.cmul(b);

        let tol = 1e-10;
        // First complex: (3+4i) * (1+2i) = -5 + 10i
        assert!((c.extract(0) - (-5.0)).abs() < tol);
        assert!((c.extract(1) - 10.0).abs() < tol);
        // Second complex: (1+0i) * (1+0i) = 1 + 0i
        assert!((c.extract(2) - 1.0).abs() < tol);
        assert!((c.extract(3) - 0.0).abs() < tol);
    }

    #[test]
    fn test_sve256_f64_predicated() {
        let a = Sve256F64::new(1.0, 2.0, 3.0, 4.0);
        let b = Sve256F64::new(10.0, 20.0, 30.0, 40.0);
        let mask = [true, false, true, false];

        let result = a.add_predicated(b, mask);

        assert_eq!(result.extract(0), 11.0); // 1 + 10
        assert_eq!(result.extract(1), 2.0); // unchanged
        assert_eq!(result.extract(2), 33.0); // 3 + 30
        assert_eq!(result.extract(3), 4.0); // unchanged
    }

    #[test]
    fn test_sve256_f32_basic() {
        let a = Sve256F32::splat(2.0);
        let b = Sve256F32::splat(3.0);
        let c = a.mul(b);

        for i in 0..8 {
            assert_eq!(c.extract(i), 6.0);
        }
    }

    #[test]
    fn test_sve256_f32_cmul() {
        // Complex multiplication: (3 + 4i) * (1 + 2i) = -5 + 10i
        let a = Sve256F32::new(3.0, 4.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0);
        let b = Sve256F32::new(1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0);
        let c = a.cmul(b);

        let tol = 1e-5;
        // First complex: (3+4i) * (1+2i) = -5 + 10i
        assert!((c.extract(0) - (-5.0)).abs() < tol);
        assert!((c.extract(1) - 10.0).abs() < tol);
    }

    #[test]
    fn test_sve_predicate() {
        let pred: SvePredicate<4> = SvePredicate::all_true();
        assert_eq!(pred.count_active(), 4);

        let pred2: SvePredicate<4> = SvePredicate::while_lt(2);
        assert_eq!(pred2.count_active(), 2);
        assert!(pred2.active[0]);
        assert!(pred2.active[1]);
        assert!(!pred2.active[2]);
        assert!(!pred2.active[3]);
    }
}
