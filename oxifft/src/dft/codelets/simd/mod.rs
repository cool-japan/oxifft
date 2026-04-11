//! SIMD-optimized codelets.
//!
//! This module provides SIMD-accelerated versions of the small DFT kernels.
//! The codelets use the architecture-specific SIMD backends when available.

// Items after statements are intentional for precomputed twiddle tables
#![allow(clippy::items_after_statements)]
// Large stack arrays are intentional for performance in fixed-size transforms
#![allow(clippy::large_stack_arrays)]

use core::any::TypeId;

use crate::kernel::{Complex, Float};
use crate::simd::{detect_simd_level, SimdLevel};

pub(crate) mod backends;
mod large_sizes;
mod small_sizes;
#[cfg(test)]
mod tests;

pub use large_sizes::*;
pub use small_sizes::*;

/// Detect if SIMD acceleration is available and beneficial.
#[inline]
pub fn simd_available() -> bool {
    let level = detect_simd_level();
    matches!(
        level,
        SimdLevel::Sse2 | SimdLevel::Avx | SimdLevel::Avx2 | SimdLevel::Avx512 | SimdLevel::Neon
    )
}

/// Size-2 DFT with automatic SIMD dispatch.
///
/// This function selects the best implementation based on available CPU features
/// and the float type. For f64, uses SIMD acceleration when available.
#[inline]
pub fn notw_2_dispatch<T: Float>(x: &mut [Complex<T>]) {
    // Check if T is f64 at runtime
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // Safety: We verified T is f64, so the memory layout is identical
        let x_f64 = unsafe {
            core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f64>>(), x.len())
        };
        notw_2_simd_f64(x_f64);
        return;
    }
    // Fallback to scalar for other types
    super::notw_2(x);
}

/// Size-4 DFT with automatic SIMD dispatch.
///
/// This function selects the best implementation based on available CPU features
/// and the float type. For f64, uses SIMD acceleration when available.
#[inline]
pub fn notw_4_dispatch<T: Float>(x: &mut [Complex<T>], sign: i32) {
    // Check if T is f64 at runtime
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // Safety: We verified T is f64, so the memory layout is identical
        let x_f64 = unsafe {
            core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f64>>(), x.len())
        };
        notw_4_simd_f64(x_f64, sign);
        return;
    }
    // Fallback to scalar for other types
    super::notw_4(x, sign);
}

/// Size-8 DFT with automatic SIMD dispatch.
///
/// This function selects the best implementation based on available CPU features
/// and the float type. For f64, uses SIMD acceleration when available.
#[inline]
pub fn notw_8_dispatch<T: Float>(x: &mut [Complex<T>], sign: i32) {
    // Check if T is f64 at runtime
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // Safety: We verified T is f64, so the memory layout is identical
        let x_f64 = unsafe {
            core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f64>>(), x.len())
        };
        notw_8_simd_f64(x_f64, sign);
        return;
    }
    // Fallback to scalar for other types
    super::notw_8(x, sign);
}

/// Size-16 DFT with automatic SIMD dispatch.
///
/// Uses scalar implementation with twiddle recurrence optimization.
#[inline]
pub fn notw_16_dispatch<T: Float>(x: &mut [Complex<T>], sign: i32) {
    // Check if T is f64 at runtime
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // Safety: We verified T is f64, so the memory layout is identical
        let x_f64 = unsafe {
            core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f64>>(), x.len())
        };
        notw_16_simd_f64(x_f64, sign);
        return;
    }
    // Fallback to scalar for other types
    super::notw_16(x, sign);
}

/// Size-32 DFT with automatic SIMD dispatch.
///
/// Uses scalar implementation with twiddle recurrence optimization.
#[inline]
pub fn notw_32_dispatch<T: Float>(x: &mut [Complex<T>], sign: i32) {
    // Check if T is f64 at runtime
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // Safety: We verified T is f64, so the memory layout is identical
        let x_f64 = unsafe {
            core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f64>>(), x.len())
        };
        notw_32_simd_f64(x_f64, sign);
        return;
    }
    // Fallback to scalar for other types
    super::notw_32(x, sign);
}

/// Size-64 DFT with automatic SIMD dispatch.
///
/// Uses scalar implementation with twiddle recurrence optimization.
#[inline]
pub fn notw_64_dispatch<T: Float>(x: &mut [Complex<T>], sign: i32) {
    // Check if T is f64 at runtime
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // Safety: We verified T is f64, so the memory layout is identical
        let x_f64 = unsafe {
            core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f64>>(), x.len())
        };
        notw_64_simd_f64(x_f64, sign);
        return;
    }
    // Fallback to scalar for other types
    super::notw_64(x, sign);
}

/// Size-128 DFT with automatic SIMD dispatch.
///
/// Uses scalar implementation with twiddle recurrence optimization.
#[inline]
pub fn notw_128_dispatch<T: Float>(x: &mut [Complex<T>], sign: i32) {
    // Check if T is f64 at runtime
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // Safety: We verified T is f64, so the memory layout is identical
        let x_f64 = unsafe {
            core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f64>>(), x.len())
        };
        notw_128_simd_f64(x_f64, sign);
        return;
    }
    // Fallback to scalar for other types
    super::notw_128(x, sign);
}

/// Size-256 DFT with automatic SIMD dispatch.
///
/// Uses scalar implementation with twiddle recurrence optimization.
#[inline]
pub fn notw_256_dispatch<T: Float>(x: &mut [Complex<T>], sign: i32) {
    // Check if T is f64 at runtime
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // Safety: We verified T is f64, so the memory layout is identical
        let x_f64 = unsafe {
            core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f64>>(), x.len())
        };
        notw_256_simd_f64(x_f64, sign);
        return;
    }
    // Fallback to scalar for other types
    super::notw_256(x, sign);
}

/// Size-512 DFT with automatic SIMD dispatch.
///
/// Uses iterative DIT with SIMD butterflies for optimal performance.
#[inline]
pub fn notw_512_dispatch<T: Float>(x: &mut [Complex<T>], sign: i32) {
    // Check if T is f64 at runtime
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // Safety: We verified T is f64, so the memory layout is identical
        let x_f64 = unsafe {
            core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f64>>(), x.len())
        };
        notw_512_simd_f64(x_f64, sign);
        return;
    }
    // Fallback to iterative DIT for other types.
    // Note: Must use execute_inplace directly to avoid infinite recursion,
    // since CooleyTukeySolver::execute dispatches back to this codelet.
    use crate::dft::problem::Sign;
    use crate::dft::solvers::CooleyTukeySolver;
    let sign_enum = if sign < 0 {
        Sign::Forward
    } else {
        Sign::Backward
    };
    CooleyTukeySolver::default().execute_dit_inplace(x, sign_enum);
}

/// Size-1024 DFT with automatic SIMD dispatch.
///
/// Uses iterative DIT with SIMD butterflies for optimal performance.
#[inline]
pub fn notw_1024_dispatch<T: Float>(x: &mut [Complex<T>], sign: i32) {
    // Check if T is f64 at runtime
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // Safety: We verified T is f64, so the memory layout is identical
        let x_f64 = unsafe {
            core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f64>>(), x.len())
        };
        notw_1024_simd_f64(x_f64, sign);
        return;
    }
    // Fallback to iterative DIT for other types.
    // Note: Must use execute_inplace directly to avoid infinite recursion,
    // since CooleyTukeySolver::execute dispatches back to this codelet.
    use crate::dft::problem::Sign;
    use crate::dft::solvers::CooleyTukeySolver;
    let sign_enum = if sign < 0 {
        Sign::Forward
    } else {
        Sign::Backward
    };
    CooleyTukeySolver::default().execute_dit_inplace(x, sign_enum);
}

/// Size-4096 DFT with automatic SIMD dispatch.
///
/// Uses iterative DIT with SIMD butterflies for optimal performance.
#[inline]
pub fn notw_4096_dispatch<T: Float>(x: &mut [Complex<T>], sign: i32) {
    // Check if T is f64 at runtime
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // Safety: We verified T is f64, so the memory layout is identical
        let x_f64 = unsafe {
            core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f64>>(), x.len())
        };
        notw_4096_simd_f64(x_f64, sign);
        return;
    }
    // Fallback to iterative DIT for other types.
    // Note: Must use execute_inplace directly to avoid infinite recursion,
    // since CooleyTukeySolver::execute dispatches back to this codelet.
    use crate::dft::problem::Sign;
    use crate::dft::solvers::CooleyTukeySolver;
    let sign_enum = if sign < 0 {
        Sign::Forward
    } else {
        Sign::Backward
    };
    CooleyTukeySolver::default().execute_dit_inplace(x, sign_enum);
}
