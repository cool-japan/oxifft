//! WASM SIMD (simd128) implementation.
//!
//! Provides 128-bit SIMD operations for WebAssembly when the `simd128` target
//! feature is available. Falls back to a scalar implementation otherwise.
//!
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

// ─── SIMD128 branch ──────────────────────────────────────────────────────────

#[cfg(target_feature = "simd128")]
mod simd_impl {
    use crate::simd::{SimdComplex, SimdVector};
    use core::arch::wasm32::{
        f32x4_add, f32x4_div, f32x4_extract_lane, f32x4_mul, f32x4_neg, f32x4_replace_lane,
        f32x4_splat, f32x4_sub, f64x2_add, f64x2_div, f64x2_extract_lane, f64x2_mul, f64x2_neg,
        f64x2_replace_lane, f64x2_splat, f64x2_sub, v128, v128_load, v128_store,
    };

    /// WASM SIMD f64 vector type (2 lanes).
    ///
    /// Uses the native `v128` type from `core::arch::wasm32`.
    /// Requires compilation with `-C target-feature=+simd128`.
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub struct WasmSimdF64 {
        inner: v128,
    }

    impl core::fmt::Debug for WasmSimdF64 {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            let lane0 = f64x2_extract_lane::<0>(self.inner);
            let lane1 = f64x2_extract_lane::<1>(self.inner);
            f.debug_struct("WasmSimdF64")
                .field("lane0", &lane0)
                .field("lane1", &lane1)
                .finish()
        }
    }

    // Safety: v128 contains only primitive floats; no thread-local state.
    unsafe impl Send for WasmSimdF64 {}
    unsafe impl Sync for WasmSimdF64 {}

    impl SimdVector for WasmSimdF64 {
        type Scalar = f64;
        const LANES: usize = 2;

        #[inline]
        fn splat(value: f64) -> Self {
            Self {
                inner: f64x2_splat(value),
            }
        }

        /// # Safety
        ///
        /// `ptr` must be valid and point to at least 2 consecutive `f64` values.
        /// The pointer must be aligned to 16 bytes.
        #[inline]
        unsafe fn load_aligned(ptr: *const f64) -> Self {
            // SAFETY: caller guarantees ptr is valid, 16-byte aligned, 2 f64s readable
            unsafe {
                Self {
                    inner: v128_load(ptr.cast()),
                }
            }
        }

        /// # Safety
        ///
        /// `ptr` must be valid and point to at least 2 consecutive `f64` values.
        /// No alignment requirement.
        #[inline]
        unsafe fn load_unaligned(ptr: *const f64) -> Self {
            // SAFETY: caller guarantees ptr is valid and 2 f64s are readable
            unsafe {
                Self {
                    inner: v128_load(ptr.cast()),
                }
            }
        }

        /// # Safety
        ///
        /// `ptr` must be valid, writable, and point to at least 2 consecutive `f64` values.
        /// The pointer must be aligned to 16 bytes.
        #[inline]
        unsafe fn store_aligned(self, ptr: *mut f64) {
            // SAFETY: caller guarantees ptr is valid, 16-byte aligned, 2 f64s writable
            unsafe {
                v128_store(ptr.cast(), self.inner);
            }
        }

        /// # Safety
        ///
        /// `ptr` must be valid, writable, and point to at least 2 consecutive `f64` values.
        /// No alignment requirement.
        #[inline]
        unsafe fn store_unaligned(self, ptr: *mut f64) {
            // SAFETY: caller guarantees ptr is valid and 2 f64s are writable
            unsafe {
                v128_store(ptr.cast(), self.inner);
            }
        }

        #[inline]
        fn add(self, other: Self) -> Self {
            Self {
                inner: f64x2_add(self.inner, other.inner),
            }
        }

        #[inline]
        fn sub(self, other: Self) -> Self {
            Self {
                inner: f64x2_sub(self.inner, other.inner),
            }
        }

        #[inline]
        fn mul(self, other: Self) -> Self {
            Self {
                inner: f64x2_mul(self.inner, other.inner),
            }
        }

        #[inline]
        fn div(self, other: Self) -> Self {
            Self {
                inner: f64x2_div(self.inner, other.inner),
            }
        }

        /// Fused multiply-add: `self * a + b`.
        ///
        /// WASM SIMD128 has no native FMA instruction; implemented as
        /// separate multiply and add (two instructions).
        #[inline]
        fn fmadd(self, a: Self, b: Self) -> Self {
            Self {
                inner: f64x2_add(f64x2_mul(self.inner, a.inner), b.inner),
            }
        }
    }

    impl WasmSimdF64 {
        /// Create a vector from two `f64` values.
        #[inline]
        pub fn new(a: f64, b: f64) -> Self {
            let inner = f64x2_splat(a);
            let inner = f64x2_replace_lane::<1>(inner, b);
            Self { inner }
        }

        /// Extract element at index 0 or 1.
        ///
        /// # Panics
        ///
        /// Panics (in debug) if `idx > 1`. In release, the index is masked to 1 bit,
        /// so only indices 0 and 1 are meaningful.
        #[inline]
        pub fn extract(self, idx: usize) -> f64 {
            match idx & 1 {
                0 => f64x2_extract_lane::<0>(self.inner),
                _ => f64x2_extract_lane::<1>(self.inner),
            }
        }

        /// Negate all elements.
        #[inline]
        pub fn negate(self) -> Self {
            Self {
                inner: f64x2_neg(self.inner),
            }
        }

        /// Swap lanes: `[a, b]` → `[b, a]`.
        #[inline]
        pub fn swap(self) -> Self {
            let lane0 = f64x2_extract_lane::<0>(self.inner);
            let lane1 = f64x2_extract_lane::<1>(self.inner);
            // splat(lane1) = [b, b]; replace_lane::<1> writes lane0 into slot 1 → [b, a].
            Self {
                inner: f64x2_replace_lane::<1>(f64x2_splat(lane1), lane0),
            }
        }
    }

    impl SimdComplex for WasmSimdF64 {
        /// Complex multiply for 1 interleaved complex number.
        ///
        /// Format: `[re, im]`.
        /// `(a + bi) * (c + di) = (ac - bd) + (ad + bc)i`
        #[inline]
        fn cmul(self, other: Self) -> Self {
            let a = f64x2_extract_lane::<0>(self.inner);
            let b = f64x2_extract_lane::<1>(self.inner);
            let c = f64x2_extract_lane::<0>(other.inner);
            let d = f64x2_extract_lane::<1>(other.inner);
            // ac - bd
            let re = f64x2_splat(a * c - b * d);
            // ad + bc
            let im_val = a * d + b * c;
            let inner = f64x2_replace_lane::<1>(re, im_val);
            Self { inner }
        }

        /// Complex multiply with conjugate.
        ///
        /// `(a + bi) * conj(c + di) = (a + bi) * (c - di) = (ac + bd) + (bc - ad)i`
        #[inline]
        fn cmul_conj(self, other: Self) -> Self {
            let a = f64x2_extract_lane::<0>(self.inner);
            let b = f64x2_extract_lane::<1>(self.inner);
            let c = f64x2_extract_lane::<0>(other.inner);
            let d = -f64x2_extract_lane::<1>(other.inner);
            let re = f64x2_splat(a * c - b * d);
            let im_val = a * d + b * c;
            let inner = f64x2_replace_lane::<1>(re, im_val);
            Self { inner }
        }
    }

    // ─── f32x4 ───────────────────────────────────────────────────────────────

    /// WASM SIMD f32 vector type (4 lanes).
    ///
    /// Uses the native `v128` type from `core::arch::wasm32`.
    /// Requires compilation with `-C target-feature=+simd128`.
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub struct WasmSimdF32 {
        inner: v128,
    }

    impl core::fmt::Debug for WasmSimdF32 {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            let lane0 = f32x4_extract_lane::<0>(self.inner);
            let lane1 = f32x4_extract_lane::<1>(self.inner);
            let lane2 = f32x4_extract_lane::<2>(self.inner);
            let lane3 = f32x4_extract_lane::<3>(self.inner);
            f.debug_struct("WasmSimdF32")
                .field("lane0", &lane0)
                .field("lane1", &lane1)
                .field("lane2", &lane2)
                .field("lane3", &lane3)
                .finish()
        }
    }

    // Safety: v128 contains only primitive floats; no thread-local state.
    unsafe impl Send for WasmSimdF32 {}
    unsafe impl Sync for WasmSimdF32 {}

    impl SimdVector for WasmSimdF32 {
        type Scalar = f32;
        const LANES: usize = 4;

        #[inline]
        fn splat(value: f32) -> Self {
            Self {
                inner: f32x4_splat(value),
            }
        }

        /// # Safety
        ///
        /// `ptr` must be valid and point to at least 4 consecutive `f32` values.
        /// The pointer must be aligned to 16 bytes.
        #[inline]
        unsafe fn load_aligned(ptr: *const f32) -> Self {
            // SAFETY: caller guarantees ptr is valid, 16-byte aligned, 4 f32s readable
            unsafe {
                Self {
                    inner: v128_load(ptr.cast()),
                }
            }
        }

        /// # Safety
        ///
        /// `ptr` must be valid and point to at least 4 consecutive `f32` values.
        /// No alignment requirement.
        #[inline]
        unsafe fn load_unaligned(ptr: *const f32) -> Self {
            // SAFETY: caller guarantees ptr is valid and 4 f32s are readable
            unsafe {
                Self {
                    inner: v128_load(ptr.cast()),
                }
            }
        }

        /// # Safety
        ///
        /// `ptr` must be valid, writable, and point to at least 4 consecutive `f32` values.
        /// The pointer must be aligned to 16 bytes.
        #[inline]
        unsafe fn store_aligned(self, ptr: *mut f32) {
            // SAFETY: caller guarantees ptr is valid, 16-byte aligned, 4 f32s writable
            unsafe {
                v128_store(ptr.cast(), self.inner);
            }
        }

        /// # Safety
        ///
        /// `ptr` must be valid, writable, and point to at least 4 consecutive `f32` values.
        /// No alignment requirement.
        #[inline]
        unsafe fn store_unaligned(self, ptr: *mut f32) {
            // SAFETY: caller guarantees ptr is valid and 4 f32s are writable
            unsafe {
                v128_store(ptr.cast(), self.inner);
            }
        }

        #[inline]
        fn add(self, other: Self) -> Self {
            Self {
                inner: f32x4_add(self.inner, other.inner),
            }
        }

        #[inline]
        fn sub(self, other: Self) -> Self {
            Self {
                inner: f32x4_sub(self.inner, other.inner),
            }
        }

        #[inline]
        fn mul(self, other: Self) -> Self {
            Self {
                inner: f32x4_mul(self.inner, other.inner),
            }
        }

        #[inline]
        fn div(self, other: Self) -> Self {
            Self {
                inner: f32x4_div(self.inner, other.inner),
            }
        }

        /// Fused multiply-add: `self * a + b`.
        ///
        /// WASM SIMD128 has no native FMA instruction; implemented as
        /// separate multiply and add (two instructions).
        #[inline]
        fn fmadd(self, a: Self, b: Self) -> Self {
            Self {
                inner: f32x4_add(f32x4_mul(self.inner, a.inner), b.inner),
            }
        }
    }

    impl WasmSimdF32 {
        /// Create a vector from four `f32` values.
        #[inline]
        pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
            let inner = f32x4_splat(a);
            let inner = f32x4_replace_lane::<1>(inner, b);
            let inner = f32x4_replace_lane::<2>(inner, c);
            let inner = f32x4_replace_lane::<3>(inner, d);
            Self { inner }
        }

        /// Extract element at index 0–3.
        ///
        /// The index is masked to 2 bits; only values 0–3 are meaningful.
        #[inline]
        pub fn extract(self, idx: usize) -> f32 {
            match idx & 3 {
                0 => f32x4_extract_lane::<0>(self.inner),
                1 => f32x4_extract_lane::<1>(self.inner),
                2 => f32x4_extract_lane::<2>(self.inner),
                _ => f32x4_extract_lane::<3>(self.inner),
            }
        }

        /// Negate all elements.
        #[inline]
        pub fn negate(self) -> Self {
            Self {
                inner: f32x4_neg(self.inner),
            }
        }
    }

    impl SimdComplex for WasmSimdF32 {
        /// Complex multiply for 2 interleaved complex numbers.
        ///
        /// Format: `[re0, im0, re1, im1]`.
        #[inline]
        fn cmul(self, other: Self) -> Self {
            let a = f32x4_extract_lane::<0>(self.inner);
            let b = f32x4_extract_lane::<1>(self.inner);
            let c = f32x4_extract_lane::<2>(self.inner);
            let d = f32x4_extract_lane::<3>(self.inner);
            let e = f32x4_extract_lane::<0>(other.inner);
            let f = f32x4_extract_lane::<1>(other.inner);
            let g = f32x4_extract_lane::<2>(other.inner);
            let h = f32x4_extract_lane::<3>(other.inner);

            let re0 = a * e - b * f;
            let im0 = a * f + b * e;
            let re1 = c * g - d * h;
            let im1 = c * h + d * g;

            let inner = f32x4_splat(re0);
            let inner = f32x4_replace_lane::<1>(inner, im0);
            let inner = f32x4_replace_lane::<2>(inner, re1);
            let inner = f32x4_replace_lane::<3>(inner, im1);
            Self { inner }
        }

        /// Complex multiply with conjugate.
        ///
        /// `self * conj(other)` where conjugate negates the imaginary lanes.
        #[inline]
        fn cmul_conj(self, other: Self) -> Self {
            let a = f32x4_extract_lane::<0>(self.inner);
            let b = f32x4_extract_lane::<1>(self.inner);
            let c = f32x4_extract_lane::<2>(self.inner);
            let d = f32x4_extract_lane::<3>(self.inner);
            let e = f32x4_extract_lane::<0>(other.inner);
            let f = -f32x4_extract_lane::<1>(other.inner); // conjugate
            let g = f32x4_extract_lane::<2>(other.inner);
            let h = -f32x4_extract_lane::<3>(other.inner); // conjugate

            let re0 = a * e - b * f;
            let im0 = a * f + b * e;
            let re1 = c * g - d * h;
            let im1 = c * h + d * g;

            let inner = f32x4_splat(re0);
            let inner = f32x4_replace_lane::<1>(inner, im0);
            let inner = f32x4_replace_lane::<2>(inner, re1);
            let inner = f32x4_replace_lane::<3>(inner, im1);
            Self { inner }
        }
    }
} // end simd_impl (simd128 branch)

// ─── Scalar fallback branch ──────────────────────────────────────────────────

#[cfg(not(target_feature = "simd128"))]
mod simd_impl {
    use crate::simd::{SimdComplex, SimdVector};

    /// WASM SIMD f64 vector type (2 lanes) — scalar fallback.
    ///
    /// Used when the `simd128` target feature is not enabled.
    #[derive(Copy, Clone, Debug)]
    #[repr(align(16))]
    pub struct WasmSimdF64 {
        data: [f64; 2],
    }

    /// WASM SIMD f32 vector type (4 lanes) — scalar fallback.
    ///
    /// Used when the `simd128` target feature is not enabled.
    #[derive(Copy, Clone, Debug)]
    #[repr(align(16))]
    pub struct WasmSimdF32 {
        data: [f32; 4],
    }

    // Safety: These types contain only primitive floats; no thread-local state.
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

        /// # Safety
        ///
        /// `ptr` must be valid and point to at least 2 consecutive `f64` values.
        /// The pointer must be aligned to 16 bytes.
        #[inline]
        unsafe fn load_aligned(ptr: *const f64) -> Self {
            // SAFETY: caller guarantees ptr is valid and 2 f64s are readable
            unsafe {
                Self {
                    data: [*ptr, *ptr.add(1)],
                }
            }
        }

        /// # Safety
        ///
        /// `ptr` must be valid and point to at least 2 consecutive `f64` values.
        /// No alignment requirement.
        #[inline]
        unsafe fn load_unaligned(ptr: *const f64) -> Self {
            // SAFETY: caller guarantees ptr is valid and 2 f64s are readable
            unsafe {
                Self {
                    data: [*ptr, *ptr.add(1)],
                }
            }
        }

        /// # Safety
        ///
        /// `ptr` must be valid, writable, and point to at least 2 consecutive `f64` values.
        /// The pointer must be aligned to 16 bytes.
        #[inline]
        unsafe fn store_aligned(self, ptr: *mut f64) {
            // SAFETY: caller guarantees ptr is valid and 2 f64s are writable
            unsafe {
                *ptr = self.data[0];
                *ptr.add(1) = self.data[1];
            }
        }

        /// # Safety
        ///
        /// `ptr` must be valid, writable, and point to at least 2 consecutive `f64` values.
        /// No alignment requirement.
        #[inline]
        unsafe fn store_unaligned(self, ptr: *mut f64) {
            // SAFETY: caller guarantees ptr is valid and 2 f64s are writable
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

    impl WasmSimdF64 {
        /// Create a vector from two `f64` values.
        #[inline]
        pub fn new(a: f64, b: f64) -> Self {
            Self { data: [a, b] }
        }

        /// Extract element at index 0 or 1.
        #[inline]
        pub fn extract(self, idx: usize) -> f64 {
            self.data[idx & 1]
        }

        /// Negate all elements.
        #[inline]
        pub fn negate(self) -> Self {
            Self {
                data: [-self.data[0], -self.data[1]],
            }
        }

        /// Swap lanes: `[a, b]` → `[b, a]`.
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
        /// Format: `[re, im]`.
        /// `(a + bi) * (c + di) = (ac - bd) + (ad + bc)i`
        #[inline]
        fn cmul(self, other: Self) -> Self {
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
        ///
        /// `(a + bi) * conj(c + di) = (ac + bd) + (bc - ad)i`
        #[inline]
        fn cmul_conj(self, other: Self) -> Self {
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

        /// # Safety
        ///
        /// `ptr` must be valid and point to at least 4 consecutive `f32` values.
        /// The pointer must be aligned to 16 bytes.
        #[inline]
        unsafe fn load_aligned(ptr: *const f32) -> Self {
            // SAFETY: caller guarantees ptr is valid and 4 f32s are readable
            unsafe {
                Self {
                    data: [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)],
                }
            }
        }

        /// # Safety
        ///
        /// `ptr` must be valid and point to at least 4 consecutive `f32` values.
        /// No alignment requirement.
        #[inline]
        unsafe fn load_unaligned(ptr: *const f32) -> Self {
            // SAFETY: caller guarantees ptr is valid and 4 f32s are readable
            unsafe {
                Self {
                    data: [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)],
                }
            }
        }

        /// # Safety
        ///
        /// `ptr` must be valid, writable, and point to at least 4 consecutive `f32` values.
        /// The pointer must be aligned to 16 bytes.
        #[inline]
        unsafe fn store_aligned(self, ptr: *mut f32) {
            // SAFETY: caller guarantees ptr is valid and 4 f32s are writable
            unsafe {
                *ptr = self.data[0];
                *ptr.add(1) = self.data[1];
                *ptr.add(2) = self.data[2];
                *ptr.add(3) = self.data[3];
            }
        }

        /// # Safety
        ///
        /// `ptr` must be valid, writable, and point to at least 4 consecutive `f32` values.
        /// No alignment requirement.
        #[inline]
        unsafe fn store_unaligned(self, ptr: *mut f32) {
            // SAFETY: caller guarantees ptr is valid and 4 f32s are writable
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

    impl WasmSimdF32 {
        /// Create a vector from four `f32` values.
        #[inline]
        pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
            Self { data: [a, b, c, d] }
        }

        /// Extract element at index 0–3.
        #[inline]
        pub fn extract(self, idx: usize) -> f32 {
            self.data[idx & 3]
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
        /// Format: `[re0, im0, re1, im1]`.
        #[inline]
        fn cmul(self, other: Self) -> Self {
            let (a, b, c, d) = (self.data[0], self.data[1], self.data[2], self.data[3]);
            let (e, f, g, h) = (other.data[0], other.data[1], other.data[2], other.data[3]);
            Self {
                data: [
                    a.mul_add(e, -(b * f)), // re0: ae - bf
                    a.mul_add(f, b * e),    // im0: af + be
                    c.mul_add(g, -(d * h)), // re1: cg - dh
                    c.mul_add(h, d * g),    // im1: ch + dg
                ],
            }
        }

        /// Complex multiply with conjugate.
        ///
        /// `self * conj(other)` where conjugate negates the imaginary lanes.
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
} // end simd_impl (scalar fallback branch)

// ─── Re-export both branches uniformly ───────────────────────────────────────

pub use simd_impl::{WasmSimdF32, WasmSimdF64};

/// Check if WASM SIMD is available at compile time.
///
/// Returns `true` when compiled with `-C target-feature=+simd128`.
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
    use crate::simd::{SimdComplex, SimdVector};

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

    #[test]
    fn test_wasm_simd_f64_new_extract() {
        let v = WasmSimdF64::new(7.0, 8.0);
        assert_eq!(v.extract(0), 7.0);
        assert_eq!(v.extract(1), 8.0);
    }

    #[test]
    fn test_wasm_simd_f64_negate_swap() {
        let v = WasmSimdF64::new(1.0, 2.0);
        let neg = v.negate();
        assert_eq!(neg.extract(0), -1.0);
        assert_eq!(neg.extract(1), -2.0);

        let swapped = v.swap();
        assert_eq!(swapped.extract(0), 2.0);
        assert_eq!(swapped.extract(1), 1.0);
    }

    #[test]
    fn test_wasm_simd_f64_fmadd() {
        // (2.0 * 3.0) + 1.0 = 7.0
        let a = WasmSimdF64::splat(2.0);
        let b = WasmSimdF64::splat(3.0);
        let c = WasmSimdF64::splat(1.0);
        let result = a.fmadd(b, c);
        let tol = 1e-10;
        assert!((result.extract(0) - 7.0).abs() < tol);
        assert!((result.extract(1) - 7.0).abs() < tol);
    }

    #[test]
    fn test_wasm_simd_f32_new_extract() {
        let v = WasmSimdF32::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.extract(0), 1.0);
        assert_eq!(v.extract(1), 2.0);
        assert_eq!(v.extract(2), 3.0);
        assert_eq!(v.extract(3), 4.0);
    }

    #[test]
    fn test_wasm_simd_f32_negate() {
        let v = WasmSimdF32::new(1.0, -2.0, 3.0, -4.0);
        let neg = v.negate();
        assert_eq!(neg.extract(0), -1.0);
        assert_eq!(neg.extract(1), 2.0);
        assert_eq!(neg.extract(2), -3.0);
        assert_eq!(neg.extract(3), 4.0);
    }

    #[test]
    fn test_wasm_simd_f32_cmul_conj() {
        // (3 + 4i) * conj(1 + 2i) = (3 + 4i) * (1 - 2i) = 11 - 2i
        // re = 3*1 - 4*(-2) = 3 + 8 = 11
        // im = 3*(-2) + 4*1 = -6 + 4 = -2
        let a = WasmSimdF32::new(3.0, 4.0, 0.0, 0.0);
        let b = WasmSimdF32::new(1.0, 2.0, 1.0, 0.0);
        let c = a.cmul_conj(b);

        let tol = 1e-5;
        assert!((c.extract(0) - 11.0).abs() < tol);
        assert!((c.extract(1) - (-2.0)).abs() < tol);
    }
}
