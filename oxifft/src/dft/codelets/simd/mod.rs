//! SIMD-optimized codelets.
//!
//! This module provides SIMD-accelerated versions of the small DFT kernels.
//! The codelets use the architecture-specific SIMD backends when available.

#![allow(clippy::items_after_statements)] // reason: precomputed twiddle table structures defined inside functions for SIMD locality
#![allow(clippy::large_stack_arrays)] // reason: fixed-size SIMD twiddle tables are stack-allocated for cache efficiency

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
/// and the float type.  For f64, uses SIMD acceleration when available.  For
/// f32, delegates to the generated codelet which provides SIMD paths on all
/// supported architectures.
#[inline]
pub fn notw_2_dispatch<T: Float>(x: &mut [Complex<T>]) {
    // Delegate to the generated dispatcher: handles both f64 (SIMD) and f32 (SIMD).
    // The size-2 butterfly is sign-independent; pass 1 as a no-op placeholder.
    super::generated_simd::generated_simd_2_dispatch(x);
}

/// Size-4 DFT with automatic SIMD dispatch.
///
/// This function selects the best implementation based on available CPU features
/// and the float type.  For f64, uses SIMD acceleration when available.  For
/// f32, delegates to the generated codelet which provides SIMD paths on all
/// supported architectures.
#[inline]
pub fn notw_4_dispatch<T: Float>(x: &mut [Complex<T>], sign: i32) {
    // Delegate to the generated dispatcher: handles both f64 (SIMD) and f32 (SIMD).
    super::generated_simd::generated_simd_4_dispatch(x, sign);
}

/// Size-8 DFT with automatic SIMD dispatch.
///
/// This function selects the best implementation based on available CPU features
/// and the float type.  For f64, uses SIMD acceleration when available.  For
/// f32, delegates to the generated codelet which provides SIMD paths on all
/// supported architectures.
#[inline]
pub fn notw_8_dispatch<T: Float>(x: &mut [Complex<T>], sign: i32) {
    // Delegate to the generated dispatcher: handles both f64 (SIMD) and f32 (SIMD).
    super::generated_simd::generated_simd_8_dispatch(x, sign);
}

/// Size-16 DFT with automatic SIMD dispatch.
///
/// On x86_64 with AVX-512F: uses hand-tuned AVX-512 codelet (f64 and f32).
/// Otherwise uses scalar optimized codelet.
#[inline]
pub fn notw_16_dispatch<T: Float>(x: &mut [Complex<T>], sign: i32) {
    // --- x86_64: try hand-tuned AVX-512 first ---
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let x_f64 = unsafe {
                core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f64>>(), x.len())
            };
            super::hand_avx512::dispatch_hand_avx512_size16_f64(x_f64, sign);
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let x_f32 = unsafe {
                core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f32>>(), x.len())
            };
            super::hand_avx512::dispatch_hand_avx512_size16_f32(x_f32, sign);
            return;
        }
    }
    // Fallback to scalar for other types / architectures
    notw_16_simd_f64_fallback(x, sign);
}

/// Internal fallback for size-16 on non-AVX-512 / non-x86_64 paths.
#[inline(always)]
fn notw_16_simd_f64_fallback<T: Float>(x: &mut [Complex<T>], sign: i32) {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let x_f64 = unsafe {
            core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f64>>(), x.len())
        };
        notw_16_simd_f64(x_f64, sign);
        return;
    }
    super::notw_16(x, sign);
}

/// Size-32 DFT with automatic SIMD dispatch.
///
/// On x86_64 with AVX-512F: uses hand-tuned AVX-512 codelet (f64 and f32).
/// Otherwise uses scalar optimized codelet.
#[inline]
pub fn notw_32_dispatch<T: Float>(x: &mut [Complex<T>], sign: i32) {
    // --- x86_64: try hand-tuned AVX-512 first ---
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let x_f64 = unsafe {
                core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f64>>(), x.len())
            };
            super::hand_avx512::dispatch_hand_avx512_size32_f64(x_f64, sign);
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let x_f32 = unsafe {
                core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f32>>(), x.len())
            };
            super::hand_avx512::dispatch_hand_avx512_size32_f32(x_f32, sign);
            return;
        }
    }
    // Fallback
    notw_32_simd_f64_fallback(x, sign);
}

/// Internal fallback for size-32 on non-AVX-512 / non-x86_64 paths.
#[inline(always)]
fn notw_32_simd_f64_fallback<T: Float>(x: &mut [Complex<T>], sign: i32) {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let x_f64 = unsafe {
            core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f64>>(), x.len())
        };
        notw_32_simd_f64(x_f64, sign);
        return;
    }
    super::notw_32(x, sign);
}

/// Size-64 DFT with automatic SIMD dispatch.
///
/// On x86_64 with AVX-512F: uses hand-tuned AVX-512 codelet (f64 and f32).
/// Otherwise uses SIMD or scalar optimized codelet.
#[inline]
pub fn notw_64_dispatch<T: Float>(x: &mut [Complex<T>], sign: i32) {
    // --- x86_64: try hand-tuned AVX-512 first ---
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let x_f64 = unsafe {
                core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f64>>(), x.len())
            };
            super::hand_avx512::dispatch_hand_avx512_size64_f64(x_f64, sign);
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let x_f32 = unsafe {
                core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f32>>(), x.len())
            };
            super::hand_avx512::dispatch_hand_avx512_size64_f32(x_f32, sign);
            return;
        }
    }
    // Fallback
    notw_64_simd_f64_fallback(x, sign);
}

/// Internal fallback for size-64 on non-AVX-512 / non-x86_64 paths.
#[inline(always)]
fn notw_64_simd_f64_fallback<T: Float>(x: &mut [Complex<T>], sign: i32) {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let x_f64 = unsafe {
            core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<Complex<f64>>(), x.len())
        };
        notw_64_simd_f64(x_f64, sign);
        return;
    }
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
