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
use crate::prelude::*;
use crate::simd::{detect_simd_level, SimdLevel};
#[cfg(target_arch = "x86_64")]
use crate::simd::{Avx2F64, SimdComplex, SimdVector, Sse2F64};

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

// ============================================================================
// SSE2 f64 SIMD codelets
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod sse2_f64 {
    use super::*;

    /// SSE2 Size-2 butterfly for f64.
    ///
    /// Processes one pair of complex numbers using SSE2.
    /// Input: x = [x0, x1] where each is Complex<f64>
    /// Output: [x0+x1, x0-x1]
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn notw_2_sse2(x: &mut [Complex<f64>]) {
        unsafe {
            debug_assert!(x.len() >= 2);

            let ptr = x.as_mut_ptr() as *mut f64;

            // Load x0 = [re0, im0] and x1 = [re1, im1]
            let v0 = Sse2F64::load_unaligned(ptr);
            let v1 = Sse2F64::load_unaligned(ptr.add(2));

            // Butterfly: (v0+v1, v0-v1)
            let (sum, diff) = Sse2F64::butterfly(v0, v1);

            // Store results
            sum.store_unaligned(ptr);
            diff.store_unaligned(ptr.add(2));
        }
    }

    /// SSE2 Size-4 DFT for f64.
    ///
    /// Uses SSE2 intrinsics to compute 4-point DFT.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn notw_4_sse2(x: &mut [Complex<f64>], sign: i32) {
        unsafe {
            debug_assert!(x.len() >= 4);

            let ptr = x.as_mut_ptr() as *mut f64;

            // Load all 4 complex numbers
            let x0 = Sse2F64::load_unaligned(ptr); // [re0, im0]
            let x1 = Sse2F64::load_unaligned(ptr.add(2)); // [re1, im1]
            let x2 = Sse2F64::load_unaligned(ptr.add(4)); // [re2, im2]
            let x3 = Sse2F64::load_unaligned(ptr.add(6)); // [re3, im3]

            // Stage 1: 2 butterflies
            let (t0, t1) = Sse2F64::butterfly(x0, x2); // t0 = x0+x2, t1 = x0-x2
            let (t2, t3) = Sse2F64::butterfly(x1, x3); // t2 = x1+x3, t3 = x1-x3

            // Apply -i or +i rotation to t3
            // For forward (sign < 0): multiply by -i means [re,im] -> [im, -re]
            // For inverse (sign >= 0): multiply by +i means [re,im] -> [-im, re]
            let t3_rot = if sign < 0 {
                // Multiply by -i: [re, im] -> [im, -re]
                t3.swap().negate_high()
            } else {
                // Multiply by +i: [re, im] -> [-im, re]
                t3.swap().negate_low()
            };

            // Stage 2: Final butterflies
            let (y0, y2) = Sse2F64::butterfly(t0, t2); // y0 = t0+t2, y2 = t0-t2
            let (y1, y3) = Sse2F64::butterfly(t1, t3_rot); // y1 = t1+t3_rot, y3 = t1-t3_rot

            // Store results in bit-reversed order
            y0.store_unaligned(ptr); // X[0]
            y1.store_unaligned(ptr.add(2)); // X[1]
            y2.store_unaligned(ptr.add(4)); // X[2]
            y3.store_unaligned(ptr.add(6)); // X[3]
        }
    }

    /// SSE2 Size-8 DFT for f64.
    ///
    /// This matches the scalar notw_8 algorithm exactly:
    /// - Stage 1: 4 radix-2 butterflies
    /// - Stage 2: Apply W8 twiddles to t3, t5, t7, then 4 more butterflies
    /// - Stage 3: Apply W4 twiddles to u3, u7, then final butterflies
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn notw_8_sse2(x: &mut [Complex<f64>], sign: i32) {
        unsafe {
            debug_assert!(x.len() >= 8);

            let ptr = x.as_mut_ptr() as *mut f64;
            let sqrt2_2 = core::f64::consts::FRAC_1_SQRT_2;

            // Load all 8 complex numbers
            let x0 = Sse2F64::load_unaligned(ptr);
            let x1 = Sse2F64::load_unaligned(ptr.add(2));
            let x2 = Sse2F64::load_unaligned(ptr.add(4));
            let x3 = Sse2F64::load_unaligned(ptr.add(6));
            let x4 = Sse2F64::load_unaligned(ptr.add(8));
            let x5 = Sse2F64::load_unaligned(ptr.add(10));
            let x6 = Sse2F64::load_unaligned(ptr.add(12));
            let x7 = Sse2F64::load_unaligned(ptr.add(14));

            // Stage 1: 4 radix-2 butterflies (no twiddles)
            let (t0, t1) = Sse2F64::butterfly(x0, x4);
            let (t2, t3) = Sse2F64::butterfly(x2, x6);
            let (t4, t5) = Sse2F64::butterfly(x1, x5);
            let (t6, t7) = Sse2F64::butterfly(x3, x7);

            // Stage 2: Apply W8 twiddles
            // t3 *= W8^2 = -i (forward) or +i (inverse)
            let t3_rot = if sign < 0 {
                t3.swap().negate_high() // [im, -re] = multiply by -i
            } else {
                t3.swap().negate_low() // [-im, re] = multiply by +i
            };

            // t5 *= W8^1 = (1-i)/sqrt(2) (forward) or (1+i)/sqrt(2) (inverse)
            let t5_rot = {
                let re = t5.low();
                let im = t5.high();
                if sign < 0 {
                    // Complex multiply: (re + i*im) * (1 - i)/sqrt(2)
                    // = (re + im)/sqrt(2) + i*(im - re)/sqrt(2)
                    Sse2F64::new((re + im) * sqrt2_2, (-re + im) * sqrt2_2)
                } else {
                    // Complex multiply: (re + i*im) * (1 + i)/sqrt(2)
                    // = (re - im)/sqrt(2) + i*(re + im)/sqrt(2)
                    Sse2F64::new((re - im) * sqrt2_2, (re + im) * sqrt2_2)
                }
            };

            // t7 *= W8^3 = (-1-i)/sqrt(2) (forward) or (-1+i)/sqrt(2) (inverse)
            let t7_rot = {
                let re = t7.low();
                let im = t7.high();
                if sign < 0 {
                    // Complex multiply: (re + i*im) * (-1 - i)/sqrt(2)
                    // = (-re + im)/sqrt(2) + i*(-re - im)/sqrt(2)
                    Sse2F64::new((-re + im) * sqrt2_2, (-re - im) * sqrt2_2)
                } else {
                    // Complex multiply: (re + i*im) * (-1 + i)/sqrt(2)
                    // = (-re - im)/sqrt(2) + i*(re - im)/sqrt(2)
                    Sse2F64::new((-re - im) * sqrt2_2, (re - im) * sqrt2_2)
                }
            };

            // 4 more butterflies
            let (u0, u1) = Sse2F64::butterfly(t0, t2);
            let (u2, u3) = Sse2F64::butterfly(t4, t6);
            let (u4, u5) = Sse2F64::butterfly(t1, t3_rot);
            let (u6, u7) = Sse2F64::butterfly(t5_rot, t7_rot);

            // Stage 3: Apply W4 twiddles and final butterflies
            // u3 *= W4^1 = -i (forward) or +i (inverse)
            let u3_rot = if sign < 0 {
                u3.swap().negate_high() // multiply by -i
            } else {
                u3.swap().negate_low() // multiply by +i
            };

            // u7 *= W4^1 = -i (forward) or +i (inverse)
            let u7_rot = if sign < 0 {
                u7.swap().negate_high() // multiply by -i
            } else {
                u7.swap().negate_low() // multiply by +i
            };

            // Final outputs
            let (y0, y4) = Sse2F64::butterfly(u0, u2);
            let (y2, y6) = Sse2F64::butterfly(u1, u3_rot);
            let (y1, y5) = Sse2F64::butterfly(u4, u6);
            let (y3, y7) = Sse2F64::butterfly(u5, u7_rot);

            // Store results
            y0.store_unaligned(ptr);
            y1.store_unaligned(ptr.add(2));
            y2.store_unaligned(ptr.add(4));
            y3.store_unaligned(ptr.add(6));
            y4.store_unaligned(ptr.add(8));
            y5.store_unaligned(ptr.add(10));
            y6.store_unaligned(ptr.add(12));
            y7.store_unaligned(ptr.add(14));
        }
    }
}

// ============================================================================
// AVX2 f64 SIMD codelets (processes 2 complex numbers per register)
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod avx2_f64 {
    use super::*;

    /// AVX2 Size-4 DFT for f64.
    ///
    /// Uses AVX2 with 256-bit registers to process pairs of complex numbers.
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn notw_4_avx2(x: &mut [Complex<f64>], sign: i32) {
        unsafe {
            debug_assert!(x.len() >= 4);

            let ptr = x.as_mut_ptr() as *mut f64;

            // Load 4 complex numbers (2 per AVX register)
            let x01 = Avx2F64::load_unaligned(ptr); // [re0, im0, re1, im1]
            let x23 = Avx2F64::load_unaligned(ptr.add(4)); // [re2, im2, re3, im3]

            // Extract individual complex numbers using permutation
            // For proper size-4 DFT we need access to x0, x1, x2, x3
            let re0 = x01.extract(0);
            let im0 = x01.extract(1);
            let re1 = x01.extract(2);
            let im1 = x01.extract(3);
            let re2 = x23.extract(0);
            let im2 = x23.extract(1);
            let re3 = x23.extract(2);
            let im3 = x23.extract(3);

            // Stage 1 butterflies
            let t0_re = re0 + re2;
            let t0_im = im0 + im2;
            let t1_re = re0 - re2;
            let t1_im = im0 - im2;
            let t2_re = re1 + re3;
            let t2_im = im1 + im3;
            let t3_re = re1 - re3;
            let t3_im = im1 - im3;

            // Apply -i or +i rotation to t3
            let (t3_rot_re, t3_rot_im) = if sign < 0 {
                (t3_im, -t3_re) // multiply by -i
            } else {
                (-t3_im, t3_re) // multiply by +i
            };

            // Stage 2 butterflies
            let y0_re = t0_re + t2_re;
            let y0_im = t0_im + t2_im;
            let y2_re = t0_re - t2_re;
            let y2_im = t0_im - t2_im;
            let y1_re = t1_re + t3_rot_re;
            let y1_im = t1_im + t3_rot_im;
            let y3_re = t1_re - t3_rot_re;
            let y3_im = t1_im - t3_rot_im;

            // Pack and store results
            let y01 = Avx2F64::new(y0_re, y0_im, y1_re, y1_im);
            let y23 = Avx2F64::new(y2_re, y2_im, y3_re, y3_im);

            y01.store_unaligned(ptr);
            y23.store_unaligned(ptr.add(4));
        }
    }
}

// ============================================================================
// NEON f64 SIMD codelets for aarch64 (Apple Silicon, etc.)
// ============================================================================
// Note: These NEON codelets were benchmarked and found to be slower than scalar
// due to memory copying overhead in even/odd deinterleaving. Kept for potential
// future optimization (e.g., in-place or streaming approaches).

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
mod neon_f64 {
    use super::*;
    use core::arch::aarch64::*;

    /// NEON Size-64 DFT for f64.
    ///
    /// Uses recursive decomposition with SIMD twiddle combination.
    /// Twiddle recurrence eliminates sin/cos calls from the inner loop.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn notw_64_neon(x: &mut [Complex<f64>], sign: i32) {
        debug_assert!(x.len() >= 64);

        // Deinterleave into even and odd arrays
        let mut even = [Complex::<f64>::zero(); 32];
        let mut odd = [Complex::<f64>::zero(); 32];

        for i in 0..32 {
            even[i] = x[2 * i];
            odd[i] = x[2 * i + 1];
        }

        // Apply size-32 DFT to each half (uses SIMD dispatch)
        super::notw_32_simd_f64(&mut even, sign);
        super::notw_32_simd_f64(&mut odd, sign);

        // Combine with NEON SIMD using twiddle recurrence
        let sign_val = f64::from(sign);
        let angle_step = sign_val * core::f64::consts::PI / 32.0;
        let w_step = Complex::cis(angle_step);

        // Sign pattern for complex multiply: [-1, 1]
        let sign_arr = [-1.0_f64, 1.0];
        let sign_pattern = unsafe { vld1q_f64(sign_arr.as_ptr()) };

        // Start with w = 1 + 0i
        let mut w = Complex::new(1.0, 0.0);

        for k in 0..32 {
            // Load even[k] and odd[k]
            let e_arr = [even[k].re, even[k].im];
            let o_arr = [odd[k].re, odd[k].im];
            let e = unsafe { vld1q_f64(e_arr.as_ptr()) };
            let o = unsafe { vld1q_f64(o_arr.as_ptr()) };

            // Create twiddle vector: [w.re, w.im]
            let tw_arr = [w.re, w.im];
            let tw = unsafe { vld1q_f64(tw_arr.as_ptr()) };
            // Flipped twiddle: [w.im, w.re]
            let tw_flip = vextq_f64(tw, tw, 1);

            // Extract and broadcast o_re and o_im
            let o_re_scalar = vgetq_lane_f64(o, 0);
            let o_im_scalar = vgetq_lane_f64(o, 1);
            let o_re = vdupq_n_f64(o_re_scalar);
            let o_im = vdupq_n_f64(o_im_scalar);

            // Complex multiply: t = odd[k] * w
            let prod1 = vmulq_f64(o_re, tw);
            let prod2 = vmulq_f64(o_im, tw_flip);
            let t = vfmaq_f64(prod1, prod2, sign_pattern);

            // Butterfly: x[k] = e + t, x[k+32] = e - t
            let out_lo = vaddq_f64(e, t);
            let out_hi = vsubq_f64(e, t);

            // Store results
            let mut lo_arr = [0.0_f64; 2];
            let mut hi_arr = [0.0_f64; 2];
            unsafe {
                vst1q_f64(lo_arr.as_mut_ptr(), out_lo);
                vst1q_f64(hi_arr.as_mut_ptr(), out_hi);
            }
            x[k] = Complex::new(lo_arr[0], lo_arr[1]);
            x[k + 32] = Complex::new(hi_arr[0], hi_arr[1]);

            // Advance twiddle using recurrence
            w = w * w_step;
        }
    }

    /// NEON Size-128 DFT for f64.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn notw_128_neon(x: &mut [Complex<f64>], sign: i32) {
        debug_assert!(x.len() >= 128);

        // Deinterleave into even and odd arrays
        let mut even = [Complex::<f64>::zero(); 64];
        let mut odd = [Complex::<f64>::zero(); 64];

        for i in 0..64 {
            even[i] = x[2 * i];
            odd[i] = x[2 * i + 1];
        }

        // Apply size-64 DFT to each half
        unsafe {
            notw_64_neon(&mut even, sign);
            notw_64_neon(&mut odd, sign);
        }

        // Combine with NEON SIMD using twiddle recurrence
        let sign_val = f64::from(sign);
        let angle_step = sign_val * core::f64::consts::PI / 64.0;
        let w_step = Complex::cis(angle_step);

        let sign_arr = [-1.0_f64, 1.0];
        let sign_pattern = unsafe { vld1q_f64(sign_arr.as_ptr()) };

        let mut w = Complex::new(1.0, 0.0);

        for k in 0..64 {
            let e_arr = [even[k].re, even[k].im];
            let o_arr = [odd[k].re, odd[k].im];
            let e = unsafe { vld1q_f64(e_arr.as_ptr()) };
            let o = unsafe { vld1q_f64(o_arr.as_ptr()) };

            let tw_arr = [w.re, w.im];
            let tw = unsafe { vld1q_f64(tw_arr.as_ptr()) };
            let tw_flip = vextq_f64(tw, tw, 1);

            let o_re_scalar = vgetq_lane_f64(o, 0);
            let o_im_scalar = vgetq_lane_f64(o, 1);
            let o_re = vdupq_n_f64(o_re_scalar);
            let o_im = vdupq_n_f64(o_im_scalar);

            let prod1 = vmulq_f64(o_re, tw);
            let prod2 = vmulq_f64(o_im, tw_flip);
            let t = vfmaq_f64(prod1, prod2, sign_pattern);

            let out_lo = vaddq_f64(e, t);
            let out_hi = vsubq_f64(e, t);

            let mut lo_arr = [0.0_f64; 2];
            let mut hi_arr = [0.0_f64; 2];
            unsafe {
                vst1q_f64(lo_arr.as_mut_ptr(), out_lo);
                vst1q_f64(hi_arr.as_mut_ptr(), out_hi);
            }
            x[k] = Complex::new(lo_arr[0], lo_arr[1]);
            x[k + 64] = Complex::new(hi_arr[0], hi_arr[1]);

            w = w * w_step;
        }
    }

    /// NEON Size-256 DFT for f64.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn notw_256_neon(x: &mut [Complex<f64>], sign: i32) {
        debug_assert!(x.len() >= 256);

        // Deinterleave into even and odd arrays
        let mut even = [Complex::<f64>::zero(); 128];
        let mut odd = [Complex::<f64>::zero(); 128];

        for i in 0..128 {
            even[i] = x[2 * i];
            odd[i] = x[2 * i + 1];
        }

        // Apply size-128 DFT to each half
        unsafe {
            notw_128_neon(&mut even, sign);
            notw_128_neon(&mut odd, sign);
        }

        // Combine with NEON SIMD using twiddle recurrence
        let sign_val = f64::from(sign);
        let angle_step = sign_val * core::f64::consts::PI / 128.0;
        let w_step = Complex::cis(angle_step);

        let sign_arr = [-1.0_f64, 1.0];
        let sign_pattern = unsafe { vld1q_f64(sign_arr.as_ptr()) };

        let mut w = Complex::new(1.0, 0.0);

        for k in 0..128 {
            let e_arr = [even[k].re, even[k].im];
            let o_arr = [odd[k].re, odd[k].im];
            let e = unsafe { vld1q_f64(e_arr.as_ptr()) };
            let o = unsafe { vld1q_f64(o_arr.as_ptr()) };

            let tw_arr = [w.re, w.im];
            let tw = unsafe { vld1q_f64(tw_arr.as_ptr()) };
            let tw_flip = vextq_f64(tw, tw, 1);

            let o_re_scalar = vgetq_lane_f64(o, 0);
            let o_im_scalar = vgetq_lane_f64(o, 1);
            let o_re = vdupq_n_f64(o_re_scalar);
            let o_im = vdupq_n_f64(o_im_scalar);

            let prod1 = vmulq_f64(o_re, tw);
            let prod2 = vmulq_f64(o_im, tw_flip);
            let t = vfmaq_f64(prod1, prod2, sign_pattern);

            let out_lo = vaddq_f64(e, t);
            let out_hi = vsubq_f64(e, t);

            let mut lo_arr = [0.0_f64; 2];
            let mut hi_arr = [0.0_f64; 2];
            unsafe {
                vst1q_f64(lo_arr.as_mut_ptr(), out_lo);
                vst1q_f64(hi_arr.as_mut_ptr(), out_hi);
            }
            x[k] = Complex::new(lo_arr[0], lo_arr[1]);
            x[k + 128] = Complex::new(hi_arr[0], hi_arr[1]);

            w = w * w_step;
        }
    }
}

// ============================================================================
// x86_64 f64 SIMD codelets for larger sizes (AVX2)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
mod x86_64_f64 {
    use super::*;

    /// AVX2 Size-64 DFT for f64.
    ///
    /// Uses recursive decomposition with SIMD twiddle combination.
    #[inline]
    pub fn notw_64_avx2(x: &mut [Complex<f64>], sign: i32) {
        // For now, fall back to scalar - AVX2 implementation can be added later
        super::super::notw_64(x, sign);
    }

    /// AVX2 Size-128 DFT for f64.
    #[inline]
    pub fn notw_128_avx2(x: &mut [Complex<f64>], sign: i32) {
        super::super::notw_128(x, sign);
    }

    /// AVX2 Size-256 DFT for f64.
    #[inline]
    pub fn notw_256_avx2(x: &mut [Complex<f64>], sign: i32) {
        super::super::notw_256(x, sign);
    }
}

// ============================================================================
// Public dispatch functions for specific float types
// ============================================================================

/// Size-2 DFT with SIMD acceleration for f64.
#[inline]
pub fn notw_2_simd_f64(x: &mut [Complex<f64>]) {
    #[cfg(target_arch = "x86_64")]
    {
        let level = detect_simd_level();
        if matches!(
            level,
            SimdLevel::Sse2 | SimdLevel::Avx | SimdLevel::Avx2 | SimdLevel::Avx512
        ) {
            // Safety: We checked SSE2 is available
            unsafe {
                sse2_f64::notw_2_sse2(x);
            }
            return;
        }
    }

    // Fallback to scalar
    super::notw_2(x);
}

/// Size-4 DFT with SIMD acceleration for f64.
#[inline]
pub fn notw_4_simd_f64(x: &mut [Complex<f64>], sign: i32) {
    #[cfg(target_arch = "x86_64")]
    {
        let level = detect_simd_level();
        if matches!(level, SimdLevel::Avx2 | SimdLevel::Avx512) {
            // Safety: We checked AVX2 is available
            unsafe {
                avx2_f64::notw_4_avx2(x, sign);
            }
            return;
        }
        if matches!(level, SimdLevel::Sse2 | SimdLevel::Avx) {
            // Safety: We checked SSE2 is available
            unsafe {
                sse2_f64::notw_4_sse2(x, sign);
            }
            return;
        }
    }

    // Fallback to scalar
    super::notw_4(x, sign);
}

/// Size-8 DFT with SIMD acceleration for f64.
#[inline]
pub fn notw_8_simd_f64(x: &mut [Complex<f64>], sign: i32) {
    #[cfg(target_arch = "x86_64")]
    {
        let level = detect_simd_level();
        if matches!(
            level,
            SimdLevel::Sse2 | SimdLevel::Avx | SimdLevel::Avx2 | SimdLevel::Avx512
        ) {
            // Safety: We checked SSE2 is available
            unsafe {
                sse2_f64::notw_8_sse2(x, sign);
            }
            return;
        }
    }

    // Fallback to scalar
    super::notw_8(x, sign);
}

/// Size-16 DFT with SIMD acceleration for f64.
///
/// Currently uses the scalar optimized codelet which has hardcoded twiddle factors.
/// Future optimization: implement full SIMD version.
#[inline]
pub fn notw_16_simd_f64(x: &mut [Complex<f64>], sign: i32) {
    // The scalar notw_16 is already highly optimized with hardcoded twiddles
    super::notw_16(x, sign);
}

/// Size-32 DFT with SIMD acceleration for f64.
///
/// Currently uses the scalar optimized codelet which has hardcoded twiddle factors.
/// Future optimization: implement full SIMD version.
#[inline]
pub fn notw_32_simd_f64(x: &mut [Complex<f64>], sign: i32) {
    // The scalar notw_32 is already highly optimized with hardcoded twiddles
    super::notw_32(x, sign);
}

/// Size-64 DFT with SIMD acceleration for f64.
///
/// Uses iterative DIT with precomputed twiddles and NEON SIMD for optimal performance.
#[inline]
pub fn notw_64_simd_f64(x: &mut [Complex<f64>], sign: i32) {
    debug_assert!(x.len() >= 64);

    // Bit-reverse permutation
    bit_reverse_permute_64(x);

    // Apply DIT butterflies with precomputed twiddles
    dit_64_precomputed(&mut x[..64], sign);
}

/// DIT butterflies for size 64 with precomputed twiddles.
#[cfg(target_arch = "aarch64")]
fn dit_64_precomputed(data: &mut [Complex<f64>], sign: i32) {
    use crate::prelude::OnceLock;
    #[cfg(not(feature = "std"))]
    #[cfg(not(feature = "std"))]
    use crate::prelude::OnceLockExt;
    use core::arch::aarch64::*;

    // Precomputed twiddles for all 6 stages
    static TWIDDLES: OnceLock<[[Complex<f64>; 32]; 6]> = OnceLock::new();
    let twiddles = TWIDDLES.get_or_init(|| {
        let mut tw = [[Complex::new(0.0, 0.0); 32]; 6];
        for s in 0..6 {
            let m = 2usize << s;
            let half_m = m / 2;
            for j in 0..half_m {
                let angle = -core::f64::consts::TAU * (j as f64) / (m as f64);
                tw[s][j] = Complex::cis(angle);
            }
        }
        tw
    });

    let ptr = data.as_mut_ptr() as *mut f64;
    let sign_arr = [-1.0_f64, 1.0];

    unsafe {
        let sign_pattern = vld1q_f64(sign_arr.as_ptr());

        let mut m = 2usize;
        for s in 0..6 {
            let half_m = m / 2;
            let tw_stage = &twiddles[s];

            if sign > 0 {
                for k in (0..64).step_by(m) {
                    for j in 0..half_m {
                        let w = Complex::new(tw_stage[j].re, -tw_stage[j].im);
                        neon_butterfly_inline(ptr, k + j, half_m, w, sign_pattern);
                    }
                }
            } else {
                for k in (0..64).step_by(m) {
                    for j in 0..half_m {
                        neon_butterfly_inline(ptr, k + j, half_m, tw_stage[j], sign_pattern);
                    }
                }
            }
            m *= 2;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn dit_64_precomputed(data: &mut [Complex<f64>], sign: i32) {
    use crate::dft::problem::Sign;
    use crate::dft::solvers::simd_butterfly::dit_butterflies_f64;
    let sign_val = if sign < 0 {
        Sign::Forward
    } else {
        Sign::Backward
    };
    dit_butterflies_f64(data, sign_val);
}

/// Fast bit-reverse permutation for size 64.
#[inline]
fn bit_reverse_permute_64(x: &mut [Complex<f64>]) {
    // Precomputed bit-reverse table for 6 bits
    const BIT_REV_64: [usize; 64] = [
        0, 32, 16, 48, 8, 40, 24, 56, 4, 36, 20, 52, 12, 44, 28, 60, 2, 34, 18, 50, 10, 42, 26, 58,
        6, 38, 22, 54, 14, 46, 30, 62, 1, 33, 17, 49, 9, 41, 25, 57, 5, 37, 21, 53, 13, 45, 29, 61,
        3, 35, 19, 51, 11, 43, 27, 59, 7, 39, 23, 55, 15, 47, 31, 63,
    ];

    for i in 0..64 {
        let j = BIT_REV_64[i];
        if i < j {
            x.swap(i, j);
        }
    }
}

/// Size-128 DFT with SIMD acceleration for f64.
///
/// Uses iterative DIT with precomputed twiddles and NEON SIMD for optimal performance.
#[inline]
pub fn notw_128_simd_f64(x: &mut [Complex<f64>], sign: i32) {
    debug_assert!(x.len() >= 128);

    // Bit-reverse permutation
    bit_reverse_permute_128(x);

    // Apply DIT butterflies with precomputed twiddles
    dit_128_precomputed(&mut x[..128], sign);
}

/// DIT butterflies for size 128 with precomputed twiddles.
#[cfg(target_arch = "aarch64")]
fn dit_128_precomputed(data: &mut [Complex<f64>], sign: i32) {
    use crate::prelude::OnceLock;
    #[cfg(not(feature = "std"))]
    #[cfg(not(feature = "std"))]
    use crate::prelude::OnceLockExt;
    use core::arch::aarch64::*;

    // Precomputed twiddles for all 7 stages
    static TWIDDLES: OnceLock<[[Complex<f64>; 64]; 7]> = OnceLock::new();
    let twiddles = TWIDDLES.get_or_init(|| {
        let mut tw = [[Complex::new(0.0, 0.0); 64]; 7];
        for s in 0..7 {
            let m = 2usize << s;
            let half_m = m / 2;
            for j in 0..half_m {
                let angle = -core::f64::consts::TAU * (j as f64) / (m as f64);
                tw[s][j] = Complex::cis(angle);
            }
        }
        tw
    });

    let ptr = data.as_mut_ptr() as *mut f64;
    let sign_arr = [-1.0_f64, 1.0];

    unsafe {
        let sign_pattern = vld1q_f64(sign_arr.as_ptr());

        let mut m = 2usize;
        for s in 0..7 {
            let half_m = m / 2;
            let tw_stage = &twiddles[s];

            // Use 4x unrolling for large stages
            if half_m >= 4 {
                if sign > 0 {
                    // Inverse transform - conjugate twiddles with 4x unrolling
                    for k in (0..128).step_by(m) {
                        let mut j = 0;
                        while j + 3 < half_m {
                            let w0 = Complex::new(tw_stage[j].re, -tw_stage[j].im);
                            let w1 = Complex::new(tw_stage[j + 1].re, -tw_stage[j + 1].im);
                            let w2 = Complex::new(tw_stage[j + 2].re, -tw_stage[j + 2].im);
                            let w3 = Complex::new(tw_stage[j + 3].re, -tw_stage[j + 3].im);
                            neon_butterfly_inline(ptr, k + j, half_m, w0, sign_pattern);
                            neon_butterfly_inline(ptr, k + j + 1, half_m, w1, sign_pattern);
                            neon_butterfly_inline(ptr, k + j + 2, half_m, w2, sign_pattern);
                            neon_butterfly_inline(ptr, k + j + 3, half_m, w3, sign_pattern);
                            j += 4;
                        }
                        while j < half_m {
                            let w = Complex::new(tw_stage[j].re, -tw_stage[j].im);
                            neon_butterfly_inline(ptr, k + j, half_m, w, sign_pattern);
                            j += 1;
                        }
                    }
                } else {
                    // Forward transform with 4x unrolling
                    for k in (0..128).step_by(m) {
                        let mut j = 0;
                        while j + 3 < half_m {
                            neon_butterfly_inline(ptr, k + j, half_m, tw_stage[j], sign_pattern);
                            neon_butterfly_inline(
                                ptr,
                                k + j + 1,
                                half_m,
                                tw_stage[j + 1],
                                sign_pattern,
                            );
                            neon_butterfly_inline(
                                ptr,
                                k + j + 2,
                                half_m,
                                tw_stage[j + 2],
                                sign_pattern,
                            );
                            neon_butterfly_inline(
                                ptr,
                                k + j + 3,
                                half_m,
                                tw_stage[j + 3],
                                sign_pattern,
                            );
                            j += 4;
                        }
                        while j < half_m {
                            neon_butterfly_inline(ptr, k + j, half_m, tw_stage[j], sign_pattern);
                            j += 1;
                        }
                    }
                }
            } else {
                // Small stages - no unrolling
                if sign > 0 {
                    for k in (0..128).step_by(m) {
                        for j in 0..half_m {
                            let w = Complex::new(tw_stage[j].re, -tw_stage[j].im);
                            neon_butterfly_inline(ptr, k + j, half_m, w, sign_pattern);
                        }
                    }
                } else {
                    for k in (0..128).step_by(m) {
                        for j in 0..half_m {
                            neon_butterfly_inline(ptr, k + j, half_m, tw_stage[j], sign_pattern);
                        }
                    }
                }
            }
            m *= 2;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn dit_128_precomputed(data: &mut [Complex<f64>], sign: i32) {
    use crate::dft::problem::Sign;
    use crate::dft::solvers::simd_butterfly::dit_butterflies_f64;
    let sign_val = if sign < 0 {
        Sign::Forward
    } else {
        Sign::Backward
    };
    dit_butterflies_f64(data, sign_val);
}

/// Fast bit-reverse permutation for size 128.
#[inline]
fn bit_reverse_permute_128(x: &mut [Complex<f64>]) {
    // Precomputed bit-reverse table for 7 bits (size 128)
    const BIT_REV_128: [usize; 128] = [
        0, 64, 32, 96, 16, 80, 48, 112, 8, 72, 40, 104, 24, 88, 56, 120, 4, 68, 36, 100, 20, 84,
        52, 116, 12, 76, 44, 108, 28, 92, 60, 124, 2, 66, 34, 98, 18, 82, 50, 114, 10, 74, 42, 106,
        26, 90, 58, 122, 6, 70, 38, 102, 22, 86, 54, 118, 14, 78, 46, 110, 30, 94, 62, 126, 1, 65,
        33, 97, 17, 81, 49, 113, 9, 73, 41, 105, 25, 89, 57, 121, 5, 69, 37, 101, 21, 85, 53, 117,
        13, 77, 45, 109, 29, 93, 61, 125, 3, 67, 35, 99, 19, 83, 51, 115, 11, 75, 43, 107, 27, 91,
        59, 123, 7, 71, 39, 103, 23, 87, 55, 119, 15, 79, 47, 111, 31, 95, 63, 127,
    ];

    for i in 0..128 {
        let j = BIT_REV_128[i];
        if i < j {
            x.swap(i, j);
        }
    }
}

/// Size-256 DFT with SIMD acceleration for f64.
///
/// Uses radix-2 DIT with precomputed twiddles and NEON SIMD for optimal performance.
#[inline]
pub fn notw_256_simd_f64(x: &mut [Complex<f64>], sign: i32) {
    debug_assert!(x.len() >= 256);

    // Bit-reverse permutation for radix-2
    bit_reverse_permute_256(x);

    // Apply radix-2 DIT butterflies with precomputed twiddles
    dit_256_precomputed(&mut x[..256], sign);
}

/// Precomputed twiddle factors as f64 pairs for direct SIMD loading.
/// Layout: [[re0, im0], [re1, im1], ...] for forward transform
/// For inverse, the imaginary part is negated.
#[cfg(target_arch = "aarch64")]
struct TwiddlesF64_256 {
    forward: [[[f64; 2]; 128]; 8],
    inverse: [[[f64; 2]; 128]; 8],
}

#[cfg(target_arch = "aarch64")]
impl TwiddlesF64_256 {
    fn new() -> Self {
        let mut forward = [[[-0.0_f64; 2]; 128]; 8];
        let mut inverse = [[[-0.0_f64; 2]; 128]; 8];
        for s in 0..8 {
            let m = 2usize << s;
            let half_m = m / 2;
            for j in 0..half_m {
                let angle = -core::f64::consts::TAU * (j as f64) / (m as f64);
                let (sin_a, cos_a) = angle.sin_cos();
                forward[s][j] = [cos_a, sin_a];
                inverse[s][j] = [cos_a, -sin_a];
            }
        }
        Self { forward, inverse }
    }
}

/// DIT butterflies for size 256 with precomputed twiddles.
#[cfg(target_arch = "aarch64")]
fn dit_256_precomputed(data: &mut [Complex<f64>], sign: i32) {
    use crate::prelude::OnceLock;
    #[cfg(not(feature = "std"))]
    #[cfg(not(feature = "std"))]
    use crate::prelude::OnceLockExt;
    use core::arch::aarch64::*;

    static TWIDDLES: OnceLock<TwiddlesF64_256> = OnceLock::new();
    let twiddles = TWIDDLES.get_or_init(TwiddlesF64_256::new);

    let ptr = data.as_mut_ptr() as *mut f64;
    let sign_arr = [-1.0_f64, 1.0];

    unsafe {
        let sign_pattern = vld1q_f64(sign_arr.as_ptr());

        // Select forward or inverse twiddles
        let tw_table = if sign > 0 {
            &twiddles.inverse
        } else {
            &twiddles.forward
        };

        // Process all 8 stages
        let mut m = 2usize;
        for s in 0..8 {
            let half_m = m / 2;
            let tw_stage = &tw_table[s];

            // For all stages >= 2, half_m >= 4
            for k in (0..256).step_by(m) {
                let mut j = 0;
                while j + 3 < half_m {
                    neon_butterfly_fast(ptr, k + j, half_m, tw_stage[j].as_ptr(), sign_pattern);
                    neon_butterfly_fast(
                        ptr,
                        k + j + 1,
                        half_m,
                        tw_stage[j + 1].as_ptr(),
                        sign_pattern,
                    );
                    neon_butterfly_fast(
                        ptr,
                        k + j + 2,
                        half_m,
                        tw_stage[j + 2].as_ptr(),
                        sign_pattern,
                    );
                    neon_butterfly_fast(
                        ptr,
                        k + j + 3,
                        half_m,
                        tw_stage[j + 3].as_ptr(),
                        sign_pattern,
                    );
                    j += 4;
                }
                while j < half_m {
                    neon_butterfly_fast(ptr, k + j, half_m, tw_stage[j].as_ptr(), sign_pattern);
                    j += 1;
                }
            }
            m *= 2;
        }
    }
}

/// Fast NEON butterfly that loads twiddle directly from memory pointer.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_butterfly_fast(
    ptr: *mut f64,
    k_j: usize,
    half_m: usize,
    tw_ptr: *const f64,
    sign_pattern: core::arch::aarch64::float64x2_t,
) {
    use core::arch::aarch64::*;

    unsafe {
        let u_ptr = ptr.add(k_j * 2);
        let v_ptr = ptr.add((k_j + half_m) * 2);
        let u = vld1q_f64(u_ptr);
        let v = vld1q_f64(v_ptr);

        // Load twiddle directly from precomputed array
        let tw = vld1q_f64(tw_ptr);
        let tw_flip = vextq_f64(tw, tw, 1);

        // Use vdupq_laneq_f64 for efficient lane broadcast
        let v_re = vdupq_laneq_f64::<0>(v);
        let v_im = vdupq_laneq_f64::<1>(v);
        let prod1 = vmulq_f64(v_re, tw);
        let prod2 = vmulq_f64(v_im, tw_flip);
        let t = vfmaq_f64(prod1, prod2, sign_pattern);
        let out_u = vaddq_f64(u, t);
        let out_v = vsubq_f64(u, t);

        vst1q_f64(u_ptr, out_u);
        vst1q_f64(v_ptr, out_v);
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn dit_256_precomputed(data: &mut [Complex<f64>], sign: i32) {
    use crate::dft::problem::Sign;
    use crate::dft::solvers::simd_butterfly::dit_butterflies_f64;
    let sign_val = if sign < 0 {
        Sign::Forward
    } else {
        Sign::Backward
    };
    dit_butterflies_f64(data, sign_val);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_butterfly_inline(
    ptr: *mut f64,
    k_j: usize,
    half_m: usize,
    w: Complex<f64>,
    sign_pattern: core::arch::aarch64::float64x2_t,
) {
    use core::arch::aarch64::*;

    unsafe {
        let u_ptr = ptr.add(k_j * 2);
        let v_ptr = ptr.add((k_j + half_m) * 2);
        let u = vld1q_f64(u_ptr);
        let v = vld1q_f64(v_ptr);

        // Load twiddle directly from memory (Complex<f64> is repr(C) with [re, im] layout)
        let tw_ptr = core::ptr::from_ref(&w) as *const f64;
        let tw = vld1q_f64(tw_ptr);
        let tw_flip = vextq_f64(tw, tw, 1);

        // Use vdupq_laneq_f64 for efficient lane broadcast (single instruction on Apple Silicon)
        let v_re = vdupq_laneq_f64::<0>(v);
        let v_im = vdupq_laneq_f64::<1>(v);
        let prod1 = vmulq_f64(v_re, tw);
        let prod2 = vmulq_f64(v_im, tw_flip);
        let t = vfmaq_f64(prod1, prod2, sign_pattern);
        let out_u = vaddq_f64(u, t);
        let out_v = vsubq_f64(u, t);

        vst1q_f64(u_ptr, out_u);
        vst1q_f64(v_ptr, out_v);
    }
}

/// Fast bit-reverse permutation for size 256.
#[inline]
fn bit_reverse_permute_256(x: &mut [Complex<f64>]) {
    // Use byte-reverse lookup table for 8 bits
    static BIT_REV_TABLE: [u8; 256] = {
        let mut table = [0u8; 256];
        let mut i = 0;
        while i < 256 {
            let mut x = i as u8;
            let mut rev = 0u8;
            let mut j = 0;
            while j < 8 {
                rev = (rev << 1) | (x & 1);
                x >>= 1;
                j += 1;
            }
            table[i] = rev;
            i += 1;
        }
        table
    };

    for i in 0..256 {
        let j = BIT_REV_TABLE[i] as usize;
        if i < j {
            x.swap(i, j);
        }
    }
}

/// Size-512 DFT with SIMD acceleration for f64.
///
/// Uses iterative DIT with precomputed twiddles and NEON SIMD for optimal performance.
#[inline]
pub fn notw_512_simd_f64(x: &mut [Complex<f64>], sign: i32) {
    debug_assert!(x.len() >= 512);

    // Bit-reverse permutation
    bit_reverse_permute_512(x);

    // Apply DIT butterflies with precomputed twiddles
    dit_512_precomputed(&mut x[..512], sign);
}

/// DIT butterflies for size 512 with precomputed twiddles.
#[cfg(target_arch = "aarch64")]
fn dit_512_precomputed(data: &mut [Complex<f64>], sign: i32) {
    use crate::prelude::OnceLock;
    #[cfg(not(feature = "std"))]
    #[cfg(not(feature = "std"))]
    use crate::prelude::OnceLockExt;
    use core::arch::aarch64::*;

    // Precomputed twiddles for all 9 stages
    static TWIDDLES: OnceLock<[[Complex<f64>; 256]; 9]> = OnceLock::new();
    let twiddles = TWIDDLES.get_or_init(|| {
        let mut tw = [[Complex::new(0.0, 0.0); 256]; 9];
        for s in 0..9 {
            let m = 2usize << s;
            let half_m = m / 2;
            for j in 0..half_m {
                let angle = -core::f64::consts::TAU * (j as f64) / (m as f64);
                tw[s][j] = Complex::cis(angle);
            }
        }
        tw
    });

    let ptr = data.as_mut_ptr() as *mut f64;
    let sign_arr = [-1.0_f64, 1.0];

    unsafe {
        let sign_pattern = vld1q_f64(sign_arr.as_ptr());

        let mut m = 2usize;
        for s in 0..9 {
            let half_m = m / 2;
            let tw_stage = &twiddles[s];

            // Use 4x unrolling for large stages
            if half_m >= 4 {
                if sign > 0 {
                    // Inverse transform - conjugate twiddles with 4x unrolling
                    for k in (0..512).step_by(m) {
                        let mut j = 0;
                        while j + 3 < half_m {
                            let w0 = Complex::new(tw_stage[j].re, -tw_stage[j].im);
                            let w1 = Complex::new(tw_stage[j + 1].re, -tw_stage[j + 1].im);
                            let w2 = Complex::new(tw_stage[j + 2].re, -tw_stage[j + 2].im);
                            let w3 = Complex::new(tw_stage[j + 3].re, -tw_stage[j + 3].im);
                            neon_butterfly_inline(ptr, k + j, half_m, w0, sign_pattern);
                            neon_butterfly_inline(ptr, k + j + 1, half_m, w1, sign_pattern);
                            neon_butterfly_inline(ptr, k + j + 2, half_m, w2, sign_pattern);
                            neon_butterfly_inline(ptr, k + j + 3, half_m, w3, sign_pattern);
                            j += 4;
                        }
                        while j < half_m {
                            let w = Complex::new(tw_stage[j].re, -tw_stage[j].im);
                            neon_butterfly_inline(ptr, k + j, half_m, w, sign_pattern);
                            j += 1;
                        }
                    }
                } else {
                    // Forward transform with 4x unrolling
                    for k in (0..512).step_by(m) {
                        let mut j = 0;
                        while j + 3 < half_m {
                            neon_butterfly_inline(ptr, k + j, half_m, tw_stage[j], sign_pattern);
                            neon_butterfly_inline(
                                ptr,
                                k + j + 1,
                                half_m,
                                tw_stage[j + 1],
                                sign_pattern,
                            );
                            neon_butterfly_inline(
                                ptr,
                                k + j + 2,
                                half_m,
                                tw_stage[j + 2],
                                sign_pattern,
                            );
                            neon_butterfly_inline(
                                ptr,
                                k + j + 3,
                                half_m,
                                tw_stage[j + 3],
                                sign_pattern,
                            );
                            j += 4;
                        }
                        while j < half_m {
                            neon_butterfly_inline(ptr, k + j, half_m, tw_stage[j], sign_pattern);
                            j += 1;
                        }
                    }
                }
            } else {
                // Small stages - no unrolling
                if sign > 0 {
                    for k in (0..512).step_by(m) {
                        for j in 0..half_m {
                            let w = Complex::new(tw_stage[j].re, -tw_stage[j].im);
                            neon_butterfly_inline(ptr, k + j, half_m, w, sign_pattern);
                        }
                    }
                } else {
                    for k in (0..512).step_by(m) {
                        for j in 0..half_m {
                            neon_butterfly_inline(ptr, k + j, half_m, tw_stage[j], sign_pattern);
                        }
                    }
                }
            }
            m *= 2;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn dit_512_precomputed(data: &mut [Complex<f64>], sign: i32) {
    use crate::dft::problem::Sign;
    use crate::dft::solvers::simd_butterfly::dit_butterflies_f64;
    let sign_val = if sign < 0 {
        Sign::Forward
    } else {
        Sign::Backward
    };
    dit_butterflies_f64(data, sign_val);
}

/// Fast bit-reverse permutation for size 512.
#[inline]
fn bit_reverse_permute_512(x: &mut [Complex<f64>]) {
    // Use byte-reverse lookup table for 9 bits
    static BIT_REV_TABLE: [u8; 256] = {
        let mut table = [0u8; 256];
        let mut i = 0;
        while i < 256 {
            let mut x = i as u8;
            let mut rev = 0u8;
            let mut j = 0;
            while j < 8 {
                rev = (rev << 1) | (x & 1);
                x >>= 1;
                j += 1;
            }
            table[i] = rev;
            i += 1;
        }
        table
    };

    for i in 0..512 {
        // For 9 bits: reverse bits 0-7 and shift, then add bit 8 at position 0
        let low = i & 0xFF;
        let high = (i >> 8) & 0x01;
        let j = high | ((BIT_REV_TABLE[low] as usize) << 1);
        if i < j {
            x.swap(i, j);
        }
    }
}

/// Size-1024 DFT with SIMD acceleration for f64.
///
/// Uses radix-2 DIT with precomputed twiddles and NEON SIMD for optimal performance.
#[inline]
pub fn notw_1024_simd_f64(x: &mut [Complex<f64>], sign: i32) {
    debug_assert!(x.len() >= 1024);

    // Bit-reverse permutation for radix-2
    bit_reverse_permute_1024(x);

    // Apply radix-2 DIT butterflies with precomputed twiddles
    dit_1024_precomputed(&mut x[..1024], sign);
}

/// Precomputed twiddle factors as f64 pairs for size 1024.
#[cfg(target_arch = "aarch64")]
struct TwiddlesF64_1024 {
    forward: [[[f64; 2]; 512]; 10],
    inverse: [[[f64; 2]; 512]; 10],
}

#[cfg(target_arch = "aarch64")]
impl TwiddlesF64_1024 {
    fn new() -> Self {
        let mut forward = [[[-0.0_f64; 2]; 512]; 10];
        let mut inverse = [[[-0.0_f64; 2]; 512]; 10];
        for s in 0..10 {
            let m = 2usize << s;
            let half_m = m / 2;
            for j in 0..half_m {
                let angle = -core::f64::consts::TAU * (j as f64) / (m as f64);
                let (sin_a, cos_a) = angle.sin_cos();
                forward[s][j] = [cos_a, sin_a];
                inverse[s][j] = [cos_a, -sin_a];
            }
        }
        Self { forward, inverse }
    }
}

/// DIT butterflies for size 1024 with precomputed twiddles.
#[cfg(target_arch = "aarch64")]
fn dit_1024_precomputed(data: &mut [Complex<f64>], sign: i32) {
    use crate::prelude::OnceLock;
    #[cfg(not(feature = "std"))]
    #[cfg(not(feature = "std"))]
    use crate::prelude::OnceLockExt;
    use core::arch::aarch64::*;

    static TWIDDLES: OnceLock<TwiddlesF64_1024> = OnceLock::new();
    let twiddles = TWIDDLES.get_or_init(TwiddlesF64_1024::new);

    let ptr = data.as_mut_ptr() as *mut f64;
    let sign_arr = [-1.0_f64, 1.0];

    unsafe {
        let sign_pattern = vld1q_f64(sign_arr.as_ptr());

        // Select forward or inverse twiddles
        let tw_table = if sign > 0 {
            &twiddles.inverse
        } else {
            &twiddles.forward
        };

        // Process all 10 stages
        let mut m = 2usize;
        for s in 0..10 {
            let half_m = m / 2;
            let tw_stage = &tw_table[s];

            // For all stages >= 2, half_m >= 4
            for k in (0..1024).step_by(m) {
                let mut j = 0;
                while j + 3 < half_m {
                    neon_butterfly_fast(ptr, k + j, half_m, tw_stage[j].as_ptr(), sign_pattern);
                    neon_butterfly_fast(
                        ptr,
                        k + j + 1,
                        half_m,
                        tw_stage[j + 1].as_ptr(),
                        sign_pattern,
                    );
                    neon_butterfly_fast(
                        ptr,
                        k + j + 2,
                        half_m,
                        tw_stage[j + 2].as_ptr(),
                        sign_pattern,
                    );
                    neon_butterfly_fast(
                        ptr,
                        k + j + 3,
                        half_m,
                        tw_stage[j + 3].as_ptr(),
                        sign_pattern,
                    );
                    j += 4;
                }
                while j < half_m {
                    neon_butterfly_fast(ptr, k + j, half_m, tw_stage[j].as_ptr(), sign_pattern);
                    j += 1;
                }
            }
            m *= 2;
        }
    }
}

/// Precomputed twiddle factors for size 1024 (x86_64).
#[cfg(target_arch = "x86_64")]
struct TwiddlesF64_1024X86 {
    forward: [[[f64; 2]; 512]; 10],
    inverse: [[[f64; 2]; 512]; 10],
}

#[cfg(target_arch = "x86_64")]
impl TwiddlesF64_1024X86 {
    fn new() -> Self {
        let mut forward = [[[-0.0_f64; 2]; 512]; 10];
        let mut inverse = [[[-0.0_f64; 2]; 512]; 10];
        for s in 0..10 {
            let m = 2usize << s;
            let half_m = m / 2;
            for j in 0..half_m {
                let angle = -core::f64::consts::TAU * (j as f64) / (m as f64);
                let (sin_a, cos_a) = angle.sin_cos();
                forward[s][j] = [cos_a, sin_a];
                inverse[s][j] = [cos_a, -sin_a];
            }
        }
        Self { forward, inverse }
    }
}

#[cfg(target_arch = "x86_64")]
fn dit_1024_precomputed(data: &mut [Complex<f64>], sign: i32) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { dit_1024_avx2(data, sign) }
    } else {
        use crate::dft::problem::Sign;
        use crate::dft::solvers::simd_butterfly::dit_butterflies_f64;
        let sign_val = if sign < 0 {
            Sign::Forward
        } else {
            Sign::Backward
        };
        dit_butterflies_f64(data, sign_val);
    }
}

/// AVX2 DIT butterflies for size 1024 with fused stages and precomputed twiddles.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dit_1024_avx2(data: &mut [Complex<f64>], sign: i32) {
    use crate::prelude::OnceLock;
    #[cfg(not(feature = "std"))]
    use crate::prelude::OnceLockExt;
    use core::arch::x86_64::*;

    static TWIDDLES: OnceLock<TwiddlesF64_1024X86> = OnceLock::new();
    let twiddles = TWIDDLES.get_or_init(TwiddlesF64_1024X86::new);

    let ptr = data.as_mut_ptr() as *mut f64;
    let sign_f = f64::from(sign);

    let tw_table = if sign > 0 {
        &twiddles.inverse
    } else {
        &twiddles.forward
    };

    // Fused stages 0-3: process 16 elements at once
    let sqrt2_2 = core::f64::consts::FRAC_1_SQRT_2;
    let w8_1 = Complex::new(sqrt2_2, sign_f * sqrt2_2);
    let w8_3 = Complex::new(-sqrt2_2, sign_f * sqrt2_2);
    let c16_1 = (core::f64::consts::PI / 8.0).cos();
    let s16_1 = (core::f64::consts::PI / 8.0).sin();
    let c16_3 = (3.0 * core::f64::consts::PI / 8.0).cos();
    let s16_3 = (3.0 * core::f64::consts::PI / 8.0).sin();
    let w16_1 = Complex::new(c16_1, sign_f * s16_1);
    let w16_2 = Complex::new(sqrt2_2, sign_f * sqrt2_2);
    let w16_3 = Complex::new(c16_3, sign_f * s16_3);
    let w16_5 = Complex::new(-c16_3, sign_f * s16_3);
    let w16_6 = Complex::new(-sqrt2_2, sign_f * sqrt2_2);
    let w16_7 = Complex::new(-c16_1, sign_f * s16_1);

    for k in (0..1024).step_by(16) {
        let mut x: [Complex<f64>; 16] = [
            data[k],
            data[k + 1],
            data[k + 2],
            data[k + 3],
            data[k + 4],
            data[k + 5],
            data[k + 6],
            data[k + 7],
            data[k + 8],
            data[k + 9],
            data[k + 10],
            data[k + 11],
            data[k + 12],
            data[k + 13],
            data[k + 14],
            data[k + 15],
        ];

        // Stage 0 (m=2)
        for i in (0..16).step_by(2) {
            let u = x[i];
            let v = x[i + 1];
            x[i] = u + v;
            x[i + 1] = u - v;
        }

        // Stage 1 (m=4)
        for i in (0..16).step_by(4) {
            let u0 = x[i];
            let u1 = x[i + 1];
            let v0 = x[i + 2];
            let v1 = x[i + 3];
            let t1 = Complex::new(-sign_f * v1.im, sign_f * v1.re);
            x[i] = u0 + v0;
            x[i + 1] = u1 + t1;
            x[i + 2] = u0 - v0;
            x[i + 3] = u1 - t1;
        }

        // Stage 2 (m=8)
        for base in [0, 8] {
            let u0 = x[base];
            let u1 = x[base + 1];
            let u2 = x[base + 2];
            let u3 = x[base + 3];
            let v0 = x[base + 4];
            let v1 = x[base + 5] * w8_1;
            let v2 = Complex::new(-sign_f * x[base + 6].im, sign_f * x[base + 6].re);
            let v3 = x[base + 7] * w8_3;
            x[base] = u0 + v0;
            x[base + 1] = u1 + v1;
            x[base + 2] = u2 + v2;
            x[base + 3] = u3 + v3;
            x[base + 4] = u0 - v0;
            x[base + 5] = u1 - v1;
            x[base + 6] = u2 - v2;
            x[base + 7] = u3 - v3;
        }

        // Stage 3 (m=16)
        let t0 = x[8];
        let t1 = x[9] * w16_1;
        let t2 = x[10] * w16_2;
        let t3 = x[11] * w16_3;
        let t4 = Complex::new(-sign_f * x[12].im, sign_f * x[12].re);
        let t5 = x[13] * w16_5;
        let t6 = x[14] * w16_6;
        let t7 = x[15] * w16_7;

        data[k] = x[0] + t0;
        data[k + 1] = x[1] + t1;
        data[k + 2] = x[2] + t2;
        data[k + 3] = x[3] + t3;
        data[k + 4] = x[4] + t4;
        data[k + 5] = x[5] + t5;
        data[k + 6] = x[6] + t6;
        data[k + 7] = x[7] + t7;
        data[k + 8] = x[0] - t0;
        data[k + 9] = x[1] - t1;
        data[k + 10] = x[2] - t2;
        data[k + 11] = x[3] - t3;
        data[k + 12] = x[4] - t4;
        data[k + 13] = x[5] - t5;
        data[k + 14] = x[6] - t6;
        data[k + 15] = x[7] - t7;
    }

    // Stages 4-9: radix-4 with precomputed twiddles
    let mut m = 32usize;
    let mut s = 4;
    while s + 1 < 10 {
        let half_m1 = m / 2;
        let m2 = m * 2;
        let half_m2 = m;

        let tw1_stage = &tw_table[s];
        let tw2_stage = &tw_table[s + 1];

        for k in (0..1024).step_by(m2) {
            let mut j = 0;

            // Process 2 radix-4 butterflies at a time using AVX256
            while j + 2 <= half_m1 {
                unsafe {
                    let tw1 = _mm256_loadu_pd(tw1_stage[j].as_ptr());
                    let tw2_a = _mm256_loadu_pd(tw2_stage[j].as_ptr());
                    let tw2_b = _mm256_loadu_pd(tw2_stage[j + half_m1].as_ptr());

                    let x0_ptr = ptr.add((k + j) * 2);
                    let x1_ptr = ptr.add((k + j + half_m1) * 2);
                    let x2_ptr = ptr.add((k + j + half_m2) * 2);
                    let x3_ptr = ptr.add((k + j + half_m2 + half_m1) * 2);

                    let x0 = _mm256_loadu_pd(x0_ptr);
                    let x1 = _mm256_loadu_pd(x1_ptr);
                    let x2 = _mm256_loadu_pd(x2_ptr);
                    let x3 = _mm256_loadu_pd(x3_ptr);

                    let tw1_re = _mm256_permute_pd(tw1, 0b0000);
                    let tw1_im = _mm256_permute_pd(tw1, 0b1111);
                    let tw2a_re = _mm256_permute_pd(tw2_a, 0b0000);
                    let tw2a_im = _mm256_permute_pd(tw2_a, 0b1111);
                    let tw2b_re = _mm256_permute_pd(tw2_b, 0b0000);
                    let tw2b_im = _mm256_permute_pd(tw2_b, 0b1111);

                    let x1_re = _mm256_permute_pd(x1, 0b0000);
                    let x1_im = _mm256_permute_pd(x1, 0b1111);
                    let t1_re = _mm256_fnmadd_pd(x1_im, tw1_im, _mm256_mul_pd(x1_re, tw1_re));
                    let t1_im = _mm256_fmadd_pd(x1_im, tw1_re, _mm256_mul_pd(x1_re, tw1_im));
                    let t1 = _mm256_blend_pd(t1_re, t1_im, 0b1010);

                    let x3_re = _mm256_permute_pd(x3, 0b0000);
                    let x3_im = _mm256_permute_pd(x3, 0b1111);
                    let t3_re = _mm256_fnmadd_pd(x3_im, tw1_im, _mm256_mul_pd(x3_re, tw1_re));
                    let t3_im = _mm256_fmadd_pd(x3_im, tw1_re, _mm256_mul_pd(x3_re, tw1_im));
                    let t3 = _mm256_blend_pd(t3_re, t3_im, 0b1010);

                    let a0 = _mm256_add_pd(x0, t1);
                    let a1 = _mm256_sub_pd(x0, t1);
                    let a2 = _mm256_add_pd(x2, t3);
                    let a3 = _mm256_sub_pd(x2, t3);

                    let a2_re = _mm256_permute_pd(a2, 0b0000);
                    let a2_im = _mm256_permute_pd(a2, 0b1111);
                    let t2a_re = _mm256_fnmadd_pd(a2_im, tw2a_im, _mm256_mul_pd(a2_re, tw2a_re));
                    let t2a_im = _mm256_fmadd_pd(a2_im, tw2a_re, _mm256_mul_pd(a2_re, tw2a_im));
                    let t2a = _mm256_blend_pd(t2a_re, t2a_im, 0b1010);

                    let a3_re = _mm256_permute_pd(a3, 0b0000);
                    let a3_im = _mm256_permute_pd(a3, 0b1111);
                    let t2b_re = _mm256_fnmadd_pd(a3_im, tw2b_im, _mm256_mul_pd(a3_re, tw2b_re));
                    let t2b_im = _mm256_fmadd_pd(a3_im, tw2b_re, _mm256_mul_pd(a3_re, tw2b_im));
                    let t2b = _mm256_blend_pd(t2b_re, t2b_im, 0b1010);

                    _mm256_storeu_pd(x0_ptr, _mm256_add_pd(a0, t2a));
                    _mm256_storeu_pd(x2_ptr, _mm256_sub_pd(a0, t2a));
                    _mm256_storeu_pd(x1_ptr, _mm256_add_pd(a1, t2b));
                    _mm256_storeu_pd(x3_ptr, _mm256_sub_pd(a1, t2b));
                }

                j += 2;
            }

            // Handle remaining butterflies
            while j < half_m1 {
                let i0 = k + j;
                let i1 = k + j + half_m1;
                let i2 = k + j + half_m2;
                let i3 = k + j + half_m2 + half_m1;

                let tw1 = tw1_stage[j];
                let tw2_a = tw2_stage[j];
                let tw2_b = tw2_stage[j + half_m1];

                let w1 = Complex::new(tw1[0], tw1[1]);
                let w2_a = Complex::new(tw2_a[0], tw2_a[1]);
                let w2_b = Complex::new(tw2_b[0], tw2_b[1]);

                let x0 = data[i0];
                let x1 = data[i1];
                let x2 = data[i2];
                let x3 = data[i3];

                let a0 = x0 + x1 * w1;
                let a1 = x0 - x1 * w1;
                let a2 = x2 + x3 * w1;
                let a3 = x2 - x3 * w1;

                data[i0] = a0 + a2 * w2_a;
                data[i2] = a0 - a2 * w2_a;
                data[i1] = a1 + a3 * w2_b;
                data[i3] = a1 - a3 * w2_b;

                j += 1;
            }
        }

        s += 2;
        m *= 4;
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn dit_1024_precomputed(data: &mut [Complex<f64>], sign: i32) {
    use crate::dft::problem::Sign;
    use crate::dft::solvers::simd_butterfly::dit_butterflies_f64;
    let sign_val = if sign < 0 {
        Sign::Forward
    } else {
        Sign::Backward
    };
    dit_butterflies_f64(data, sign_val);
}

/// Fast bit-reverse permutation for size 1024.
#[inline]
fn bit_reverse_permute_1024(x: &mut [Complex<f64>]) {
    // Use byte-reverse lookup table for 10 bits
    static BIT_REV_TABLE: [u8; 256] = {
        let mut table = [0u8; 256];
        let mut i = 0;
        while i < 256 {
            let mut x = i as u8;
            let mut rev = 0u8;
            let mut j = 0;
            while j < 8 {
                rev = (rev << 1) | (x & 1);
                x >>= 1;
                j += 1;
            }
            table[i] = rev;
            i += 1;
        }
        table
    };

    // Lookup table for reversing 2 bits: 0b00->0b00, 0b01->0b10, 0b10->0b01, 0b11->0b11
    const REV_2BITS: [usize; 4] = [0, 2, 1, 3];

    for i in 0..1024 {
        // For 10 bits: reverse bits 0-7 and shift, then add reversed bits 8-9 at positions 0-1
        let low = i & 0xFF;
        let high = (i >> 8) & 0x03;
        let j = REV_2BITS[high] | ((BIT_REV_TABLE[low] as usize) << 2);
        if i < j {
            x.swap(i, j);
        }
    }
}

/// Size-4096 DFT with SIMD acceleration for f64.
///
/// Uses radix-2 DIT with precomputed twiddles and NEON SIMD for optimal performance.
/// Note: Radix-2 is faster than radix-4 for this size due to cache effects.
#[inline]
pub fn notw_4096_simd_f64(x: &mut [Complex<f64>], sign: i32) {
    debug_assert!(x.len() >= 4096);

    // Bit-reverse permutation
    bit_reverse_permute_4096(x);

    // Apply DIT butterflies with precomputed twiddles
    dit_4096_precomputed(&mut x[..4096], sign);
}

/// Precomputed twiddle factors as f64 pairs for size 4096.
#[cfg(target_arch = "aarch64")]
struct TwiddlesF64_4096 {
    forward: Box<[[[f64; 2]; 2048]; 12]>,
    inverse: Box<[[[f64; 2]; 2048]; 12]>,
}

#[cfg(target_arch = "aarch64")]
impl TwiddlesF64_4096 {
    #[allow(clippy::large_stack_frames)]
    fn new() -> Self {
        let mut forward = Box::new([[[-0.0_f64; 2]; 2048]; 12]);
        let mut inverse = Box::new([[[-0.0_f64; 2]; 2048]; 12]);
        for s in 0..12 {
            let m = 2usize << s;
            let half_m = m / 2;
            for j in 0..half_m {
                let angle = -core::f64::consts::TAU * (j as f64) / (m as f64);
                let (sin_a, cos_a) = angle.sin_cos();
                forward[s][j] = [cos_a, sin_a];
                inverse[s][j] = [cos_a, -sin_a];
            }
        }
        Self { forward, inverse }
    }
}

/// DIT butterflies for size 4096 with precomputed twiddles.
#[cfg(target_arch = "aarch64")]
fn dit_4096_precomputed(data: &mut [Complex<f64>], sign: i32) {
    use crate::prelude::OnceLock;
    #[cfg(not(feature = "std"))]
    #[cfg(not(feature = "std"))]
    use crate::prelude::OnceLockExt;
    use core::arch::aarch64::*;

    static TWIDDLES: OnceLock<TwiddlesF64_4096> = OnceLock::new();
    let twiddles = TWIDDLES.get_or_init(TwiddlesF64_4096::new);

    let ptr = data.as_mut_ptr() as *mut f64;
    let sign_arr = [-1.0_f64, 1.0];

    unsafe {
        let sign_pattern = vld1q_f64(sign_arr.as_ptr());

        // Select forward or inverse twiddles
        let tw_table = if sign > 0 {
            &twiddles.inverse
        } else {
            &twiddles.forward
        };

        // Process all 12 stages
        let mut m = 2usize;
        for s in 0..12 {
            let half_m = m / 2;
            let tw_stage = &tw_table[s];

            // For all stages >= 2, half_m >= 4, use 8x unrolling
            for k in (0..4096).step_by(m) {
                let mut j = 0;
                while j + 7 < half_m {
                    neon_butterfly_fast(ptr, k + j, half_m, tw_stage[j].as_ptr(), sign_pattern);
                    neon_butterfly_fast(
                        ptr,
                        k + j + 1,
                        half_m,
                        tw_stage[j + 1].as_ptr(),
                        sign_pattern,
                    );
                    neon_butterfly_fast(
                        ptr,
                        k + j + 2,
                        half_m,
                        tw_stage[j + 2].as_ptr(),
                        sign_pattern,
                    );
                    neon_butterfly_fast(
                        ptr,
                        k + j + 3,
                        half_m,
                        tw_stage[j + 3].as_ptr(),
                        sign_pattern,
                    );
                    neon_butterfly_fast(
                        ptr,
                        k + j + 4,
                        half_m,
                        tw_stage[j + 4].as_ptr(),
                        sign_pattern,
                    );
                    neon_butterfly_fast(
                        ptr,
                        k + j + 5,
                        half_m,
                        tw_stage[j + 5].as_ptr(),
                        sign_pattern,
                    );
                    neon_butterfly_fast(
                        ptr,
                        k + j + 6,
                        half_m,
                        tw_stage[j + 6].as_ptr(),
                        sign_pattern,
                    );
                    neon_butterfly_fast(
                        ptr,
                        k + j + 7,
                        half_m,
                        tw_stage[j + 7].as_ptr(),
                        sign_pattern,
                    );
                    j += 8;
                }
                while j < half_m {
                    neon_butterfly_fast(ptr, k + j, half_m, tw_stage[j].as_ptr(), sign_pattern);
                    j += 1;
                }
            }
            m *= 2;
        }
    }
}

/// Precomputed twiddle factors for size 4096 (x86_64).
#[cfg(target_arch = "x86_64")]
struct TwiddlesF64_4096X86 {
    forward: Box<[[[f64; 2]; 2048]; 12]>,
    inverse: Box<[[[f64; 2]; 2048]; 12]>,
}

#[cfg(target_arch = "x86_64")]
impl TwiddlesF64_4096X86 {
    #[allow(clippy::large_stack_frames)]
    fn new() -> Self {
        let mut forward = Box::new([[[-0.0_f64; 2]; 2048]; 12]);
        let mut inverse = Box::new([[[-0.0_f64; 2]; 2048]; 12]);
        for s in 0..12 {
            let m = 2usize << s;
            let half_m = m / 2;
            for j in 0..half_m {
                let angle = -core::f64::consts::TAU * (j as f64) / (m as f64);
                let (sin_a, cos_a) = angle.sin_cos();
                forward[s][j] = [cos_a, sin_a];
                inverse[s][j] = [cos_a, -sin_a];
            }
        }
        Self { forward, inverse }
    }
}

#[cfg(target_arch = "x86_64")]
fn dit_4096_precomputed(data: &mut [Complex<f64>], sign: i32) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { dit_4096_avx2(data, sign) }
    } else {
        // SSE2 fallback
        use crate::dft::problem::Sign;
        use crate::dft::solvers::simd_butterfly::dit_butterflies_f64;
        let sign_val = if sign < 0 {
            Sign::Forward
        } else {
            Sign::Backward
        };
        dit_butterflies_f64(data, sign_val);
    }
}

/// AVX2 DIT butterflies for size 4096 with fused stages and precomputed twiddles.
/// Fuses stages 0-3 to reduce memory traffic, then uses SIMD for stages 4-11.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dit_4096_avx2(data: &mut [Complex<f64>], sign: i32) {
    use crate::prelude::OnceLock;
    #[cfg(not(feature = "std"))]
    use crate::prelude::OnceLockExt;
    use core::arch::x86_64::*;

    static TWIDDLES: OnceLock<TwiddlesF64_4096X86> = OnceLock::new();
    let twiddles = TWIDDLES.get_or_init(TwiddlesF64_4096X86::new);

    let ptr = data.as_mut_ptr() as *mut f64;
    let sign_f = f64::from(sign);

    // Select forward or inverse twiddles
    let tw_table = if sign > 0 {
        &twiddles.inverse
    } else {
        &twiddles.forward
    };

    // Fused stages 0-3: process 16 elements at once to reduce memory traffic
    let sqrt2_2 = core::f64::consts::FRAC_1_SQRT_2;
    // Stage 2 twiddles (for m=8)
    let w8_1 = Complex::new(sqrt2_2, sign_f * sqrt2_2);
    let w8_3 = Complex::new(-sqrt2_2, sign_f * sqrt2_2);
    // Stage 3 twiddles (for m=16)
    let c16_1 = (core::f64::consts::PI / 8.0).cos();
    let s16_1 = (core::f64::consts::PI / 8.0).sin();
    let c16_3 = (3.0 * core::f64::consts::PI / 8.0).cos();
    let s16_3 = (3.0 * core::f64::consts::PI / 8.0).sin();
    let w16_1 = Complex::new(c16_1, sign_f * s16_1);
    let w16_2 = Complex::new(sqrt2_2, sign_f * sqrt2_2);
    let w16_3 = Complex::new(c16_3, sign_f * s16_3);
    let w16_5 = Complex::new(-c16_3, sign_f * s16_3);
    let w16_6 = Complex::new(-sqrt2_2, sign_f * sqrt2_2);
    let w16_7 = Complex::new(-c16_1, sign_f * s16_1);

    for k in (0..4096).step_by(16) {
        // Load all 16 elements
        let mut x: [Complex<f64>; 16] = [
            data[k],
            data[k + 1],
            data[k + 2],
            data[k + 3],
            data[k + 4],
            data[k + 5],
            data[k + 6],
            data[k + 7],
            data[k + 8],
            data[k + 9],
            data[k + 10],
            data[k + 11],
            data[k + 12],
            data[k + 13],
            data[k + 14],
            data[k + 15],
        ];

        // Stage 0 (m=2): butterfly pairs
        for i in (0..16).step_by(2) {
            let u = x[i];
            let v = x[i + 1];
            x[i] = u + v;
            x[i + 1] = u - v;
        }

        // Stage 1 (m=4): butterfly pairs with ±i twiddle
        for i in (0..16).step_by(4) {
            let u0 = x[i];
            let u1 = x[i + 1];
            let v0 = x[i + 2];
            let v1 = x[i + 3];
            let t1 = Complex::new(-sign_f * v1.im, sign_f * v1.re);
            x[i] = u0 + v0;
            x[i + 1] = u1 + t1;
            x[i + 2] = u0 - v0;
            x[i + 3] = u1 - t1;
        }

        // Stage 2 (m=8): butterfly pairs with W_8 twiddles
        for base in [0, 8] {
            let u0 = x[base];
            let u1 = x[base + 1];
            let u2 = x[base + 2];
            let u3 = x[base + 3];
            let v0 = x[base + 4];
            let v1 = x[base + 5] * w8_1;
            let v2 = Complex::new(-sign_f * x[base + 6].im, sign_f * x[base + 6].re);
            let v3 = x[base + 7] * w8_3;
            x[base] = u0 + v0;
            x[base + 1] = u1 + v1;
            x[base + 2] = u2 + v2;
            x[base + 3] = u3 + v3;
            x[base + 4] = u0 - v0;
            x[base + 5] = u1 - v1;
            x[base + 6] = u2 - v2;
            x[base + 7] = u3 - v3;
        }

        // Stage 3 (m=16): butterfly pairs with W_16 twiddles
        let t0 = x[8];
        let t1 = x[9] * w16_1;
        let t2 = x[10] * w16_2;
        let t3 = x[11] * w16_3;
        let t4 = Complex::new(-sign_f * x[12].im, sign_f * x[12].re);
        let t5 = x[13] * w16_5;
        let t6 = x[14] * w16_6;
        let t7 = x[15] * w16_7;

        // Store back
        data[k] = x[0] + t0;
        data[k + 1] = x[1] + t1;
        data[k + 2] = x[2] + t2;
        data[k + 3] = x[3] + t3;
        data[k + 4] = x[4] + t4;
        data[k + 5] = x[5] + t5;
        data[k + 6] = x[6] + t6;
        data[k + 7] = x[7] + t7;
        data[k + 8] = x[0] - t0;
        data[k + 9] = x[1] - t1;
        data[k + 10] = x[2] - t2;
        data[k + 11] = x[3] - t3;
        data[k + 12] = x[4] - t4;
        data[k + 13] = x[5] - t5;
        data[k + 14] = x[6] - t6;
        data[k + 15] = x[7] - t7;
    }

    // Stages 4-11: radix-4 with precomputed twiddles (combines pairs of stages)
    // Stage 4-5: m1=32, m2=64 -> combined to m2=64
    // Stage 6-7: m1=128, m2=256 -> combined to m2=256
    // Stage 8-9: m1=512, m2=1024 -> combined to m2=1024
    // Stage 10-11: m1=2048, m2=4096 -> combined to m2=4096
    let mut m = 32usize;
    let mut s = 4;
    while s + 1 < 12 {
        let half_m1 = m / 2; // Distance for first radix-2
        let m2 = m * 2; // Combined radix-4 block size
        let half_m2 = m; // Distance for second radix-2

        let tw1_stage = &tw_table[s];
        let tw2_stage = &tw_table[s + 1];

        for k in (0..4096).step_by(m2) {
            let mut j = 0;

            // Process 2 radix-4 butterflies at a time using AVX256
            while j + 2 <= half_m1 {
                unsafe {
                    // Load twiddles
                    let tw1 = _mm256_loadu_pd(tw1_stage[j].as_ptr());
                    let tw2_a = _mm256_loadu_pd(tw2_stage[j].as_ptr());
                    let tw2_b = _mm256_loadu_pd(tw2_stage[j + half_m1].as_ptr());

                    // Compute pointers for 2 radix-4 butterflies
                    let x0_ptr = ptr.add((k + j) * 2);
                    let x1_ptr = ptr.add((k + j + half_m1) * 2);
                    let x2_ptr = ptr.add((k + j + half_m2) * 2);
                    let x3_ptr = ptr.add((k + j + half_m2 + half_m1) * 2);

                    // Load data
                    let x0 = _mm256_loadu_pd(x0_ptr);
                    let x1 = _mm256_loadu_pd(x1_ptr);
                    let x2 = _mm256_loadu_pd(x2_ptr);
                    let x3 = _mm256_loadu_pd(x3_ptr);

                    // Expand twiddles
                    let tw1_re = _mm256_permute_pd(tw1, 0b0000);
                    let tw1_im = _mm256_permute_pd(tw1, 0b1111);
                    let tw2a_re = _mm256_permute_pd(tw2_a, 0b0000);
                    let tw2a_im = _mm256_permute_pd(tw2_a, 0b1111);
                    let tw2b_re = _mm256_permute_pd(tw2_b, 0b0000);
                    let tw2b_im = _mm256_permute_pd(tw2_b, 0b1111);

                    // First radix-2: t1 = x1 * tw1, t3 = x3 * tw1
                    let x1_re = _mm256_permute_pd(x1, 0b0000);
                    let x1_im = _mm256_permute_pd(x1, 0b1111);
                    let t1_re = _mm256_fnmadd_pd(x1_im, tw1_im, _mm256_mul_pd(x1_re, tw1_re));
                    let t1_im = _mm256_fmadd_pd(x1_im, tw1_re, _mm256_mul_pd(x1_re, tw1_im));
                    let t1 = _mm256_blend_pd(t1_re, t1_im, 0b1010);

                    let x3_re = _mm256_permute_pd(x3, 0b0000);
                    let x3_im = _mm256_permute_pd(x3, 0b1111);
                    let t3_re = _mm256_fnmadd_pd(x3_im, tw1_im, _mm256_mul_pd(x3_re, tw1_re));
                    let t3_im = _mm256_fmadd_pd(x3_im, tw1_re, _mm256_mul_pd(x3_re, tw1_im));
                    let t3 = _mm256_blend_pd(t3_re, t3_im, 0b1010);

                    // Butterflies
                    let a0 = _mm256_add_pd(x0, t1);
                    let a1 = _mm256_sub_pd(x0, t1);
                    let a2 = _mm256_add_pd(x2, t3);
                    let a3 = _mm256_sub_pd(x2, t3);

                    // Second radix-2: t2a = a2 * tw2_a, t2b = a3 * tw2_b
                    let a2_re = _mm256_permute_pd(a2, 0b0000);
                    let a2_im = _mm256_permute_pd(a2, 0b1111);
                    let t2a_re = _mm256_fnmadd_pd(a2_im, tw2a_im, _mm256_mul_pd(a2_re, tw2a_re));
                    let t2a_im = _mm256_fmadd_pd(a2_im, tw2a_re, _mm256_mul_pd(a2_re, tw2a_im));
                    let t2a = _mm256_blend_pd(t2a_re, t2a_im, 0b1010);

                    let a3_re = _mm256_permute_pd(a3, 0b0000);
                    let a3_im = _mm256_permute_pd(a3, 0b1111);
                    let t2b_re = _mm256_fnmadd_pd(a3_im, tw2b_im, _mm256_mul_pd(a3_re, tw2b_re));
                    let t2b_im = _mm256_fmadd_pd(a3_im, tw2b_re, _mm256_mul_pd(a3_re, tw2b_im));
                    let t2b = _mm256_blend_pd(t2b_re, t2b_im, 0b1010);

                    // Store results
                    _mm256_storeu_pd(x0_ptr, _mm256_add_pd(a0, t2a));
                    _mm256_storeu_pd(x2_ptr, _mm256_sub_pd(a0, t2a));
                    _mm256_storeu_pd(x1_ptr, _mm256_add_pd(a1, t2b));
                    _mm256_storeu_pd(x3_ptr, _mm256_sub_pd(a1, t2b));
                }
                j += 2;
            }

            // Handle remaining butterflies
            while j < half_m1 {
                let i0 = k + j;
                let i1 = k + j + half_m1;
                let i2 = k + j + half_m2;
                let i3 = k + j + half_m2 + half_m1;

                let tw1 = tw1_stage[j];
                let tw2_a = tw2_stage[j];
                let tw2_b = tw2_stage[j + half_m1];

                let w1 = Complex::new(tw1[0], tw1[1]);
                let w2_a = Complex::new(tw2_a[0], tw2_a[1]);
                let w2_b = Complex::new(tw2_b[0], tw2_b[1]);

                let x0 = data[i0];
                let x1 = data[i1];
                let x2 = data[i2];
                let x3 = data[i3];

                // First stage
                let a0 = x0 + x1 * w1;
                let a1 = x0 - x1 * w1;
                let a2 = x2 + x3 * w1;
                let a3 = x2 - x3 * w1;

                // Second stage
                data[i0] = a0 + a2 * w2_a;
                data[i2] = a0 - a2 * w2_a;
                data[i1] = a1 + a3 * w2_b;
                data[i3] = a1 - a3 * w2_b;

                j += 1;
            }
        }

        s += 2;
        m *= 4;
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn dit_4096_precomputed(data: &mut [Complex<f64>], sign: i32) {
    use crate::dft::problem::Sign;
    use crate::dft::solvers::simd_butterfly::dit_butterflies_f64;
    let sign_val = if sign < 0 {
        Sign::Forward
    } else {
        Sign::Backward
    };
    dit_butterflies_f64(data, sign_val);
}

/// Fast bit-reverse permutation for size 4096.
#[inline]
fn bit_reverse_permute_4096(x: &mut [Complex<f64>]) {
    // Use byte-reverse lookup table for 12 bits
    static BIT_REV_TABLE: [u8; 256] = {
        let mut table = [0u8; 256];
        let mut i = 0;
        while i < 256 {
            let mut x = i as u8;
            let mut rev = 0u8;
            let mut j = 0;
            while j < 8 {
                rev = (rev << 1) | (x & 1);
                x >>= 1;
                j += 1;
            }
            table[i] = rev;
            i += 1;
        }
        table
    };

    // Lookup table for reversing 4 bits
    const REV_4BITS: [usize; 16] = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];

    for i in 0..4096 {
        // For 12 bits: reverse bits 0-7 and shift, then add reversed bits 8-11 at positions 0-3
        let low = i & 0xFF;
        let high = (i >> 8) & 0x0F;
        let j = REV_4BITS[high] | ((BIT_REV_TABLE[low] as usize) << 4);
        if i < j {
            x.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dft::codelets::{
        notw_128, notw_16, notw_2, notw_256, notw_32, notw_4, notw_64, notw_8,
    };

    fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
        (a.re - b.re).abs() < eps && (a.im - b.im).abs() < eps
    }

    #[test]
    fn test_simd_notw_2_matches_scalar() {
        let mut scalar = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let mut simd = scalar;

        notw_2(&mut scalar);
        notw_2_simd_f64(&mut simd);

        for (s, d) in scalar.iter().zip(simd.iter()) {
            assert!(
                complex_approx_eq(*s, *d, 1e-10),
                "Mismatch: scalar={s:?}, simd={d:?}"
            );
        }
    }

    #[test]
    fn test_simd_notw_4_matches_scalar_forward() {
        let mut scalar = [
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ];
        let mut simd = scalar;

        notw_4(&mut scalar, -1);
        notw_4_simd_f64(&mut simd, -1);

        for (i, (s, d)) in scalar.iter().zip(simd.iter()).enumerate() {
            assert!(
                complex_approx_eq(*s, *d, 1e-10),
                "Index {i}: scalar={s:?}, simd={d:?}"
            );
        }
    }

    #[test]
    fn test_simd_notw_4_matches_scalar_inverse() {
        let mut scalar = [
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ];
        let mut simd = scalar;

        notw_4(&mut scalar, 1);
        notw_4_simd_f64(&mut simd, 1);

        for (i, (s, d)) in scalar.iter().zip(simd.iter()).enumerate() {
            assert!(
                complex_approx_eq(*s, *d, 1e-10),
                "Index {i}: scalar={s:?}, simd={d:?}"
            );
        }
    }

    #[test]
    fn test_simd_notw_8_matches_scalar_forward() {
        let mut scalar: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut simd = scalar.clone();

        notw_8(&mut scalar, -1);
        notw_8_simd_f64(&mut simd, -1);

        for (i, (s, d)) in scalar.iter().zip(simd.iter()).enumerate() {
            assert!(
                complex_approx_eq(*s, *d, 1e-9),
                "Index {i}: scalar={s:?}, simd={d:?}"
            );
        }
    }

    #[test]
    fn test_simd_notw_8_matches_scalar_inverse() {
        let mut scalar: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut simd = scalar.clone();

        notw_8(&mut scalar, 1);
        notw_8_simd_f64(&mut simd, 1);

        for (i, (s, d)) in scalar.iter().zip(simd.iter()).enumerate() {
            assert!(
                complex_approx_eq(*s, *d, 1e-9),
                "Index {i}: scalar={s:?}, simd={d:?}"
            );
        }
    }

    #[test]
    fn test_simd_notw_4_roundtrip() {
        let original: Vec<Complex<f64>> = (0..4)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut data = original.clone();

        // Forward
        notw_4_simd_f64(&mut data, -1);
        // Inverse
        notw_4_simd_f64(&mut data, 1);
        // Normalize
        for x in &mut data {
            *x = *x / 4.0;
        }

        for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                complex_approx_eq(*o, *d, 1e-10),
                "Index {i}: original={o:?}, recovered={d:?}"
            );
        }
    }

    #[test]
    fn test_simd_notw_8_roundtrip() {
        let original: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut data = original.clone();

        // Forward
        notw_8_simd_f64(&mut data, -1);
        // Inverse
        notw_8_simd_f64(&mut data, 1);
        // Normalize
        for x in &mut data {
            *x = *x / 8.0;
        }

        for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                complex_approx_eq(*o, *d, 1e-9),
                "Index {i}: original={o:?}, recovered={d:?}"
            );
        }
    }

    #[test]
    fn test_simd_notw_16_matches_scalar() {
        let mut scalar: Vec<Complex<f64>> = (0..16)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut simd = scalar.clone();

        notw_16(&mut scalar, -1);
        notw_16_simd_f64(&mut simd, -1);

        for (i, (s, d)) in scalar.iter().zip(simd.iter()).enumerate() {
            assert!(
                complex_approx_eq(*s, *d, 1e-8),
                "Index {i}: scalar={s:?}, simd={d:?}"
            );
        }
    }

    #[test]
    fn test_simd_notw_16_roundtrip() {
        let original: Vec<Complex<f64>> = (0..16)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut data = original.clone();

        // Forward
        notw_16_simd_f64(&mut data, -1);
        // Inverse
        notw_16_simd_f64(&mut data, 1);
        // Normalize
        for x in &mut data {
            *x = *x / 16.0;
        }

        for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                complex_approx_eq(*o, *d, 1e-8),
                "Index {i}: original={o:?}, recovered={d:?}"
            );
        }
    }

    #[test]
    fn test_simd_notw_32_matches_scalar() {
        let mut scalar: Vec<Complex<f64>> = (0..32)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut simd = scalar.clone();

        notw_32(&mut scalar, -1);
        notw_32_simd_f64(&mut simd, -1);

        for (i, (s, d)) in scalar.iter().zip(simd.iter()).enumerate() {
            assert!(
                complex_approx_eq(*s, *d, 1e-8),
                "Index {i}: scalar={s:?}, simd={d:?}"
            );
        }
    }

    #[test]
    fn test_simd_notw_32_roundtrip() {
        let original: Vec<Complex<f64>> = (0..32)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut data = original.clone();

        // Forward
        notw_32_simd_f64(&mut data, -1);
        // Inverse
        notw_32_simd_f64(&mut data, 1);
        // Normalize
        for x in &mut data {
            *x = *x / 32.0;
        }

        for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                complex_approx_eq(*o, *d, 1e-8),
                "Index {i}: original={o:?}, recovered={d:?}"
            );
        }
    }

    #[test]
    fn test_simd_notw_64_matches_scalar() {
        let mut scalar: Vec<Complex<f64>> = (0..64)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut simd = scalar.clone();

        notw_64(&mut scalar, -1);
        notw_64_simd_f64(&mut simd, -1);

        for (i, (s, d)) in scalar.iter().zip(simd.iter()).enumerate() {
            assert!(
                complex_approx_eq(*s, *d, 1e-7),
                "Index {i}: scalar={s:?}, simd={d:?}"
            );
        }
    }

    #[test]
    fn test_simd_notw_64_roundtrip() {
        let original: Vec<Complex<f64>> = (0..64)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut data = original.clone();

        // Forward
        notw_64_simd_f64(&mut data, -1);
        // Inverse
        notw_64_simd_f64(&mut data, 1);
        // Normalize
        for x in &mut data {
            *x = *x / 64.0;
        }

        for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                complex_approx_eq(*o, *d, 1e-8),
                "Index {i}: original={o:?}, recovered={d:?}"
            );
        }
    }

    #[test]
    fn test_simd_notw_128_matches_scalar() {
        let mut scalar: Vec<Complex<f64>> = (0..128)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut simd = scalar.clone();

        notw_128(&mut scalar, -1);
        notw_128_simd_f64(&mut simd, -1);

        for (i, (s, d)) in scalar.iter().zip(simd.iter()).enumerate() {
            assert!(
                complex_approx_eq(*s, *d, 1e-6),
                "Index {i}: scalar={s:?}, simd={d:?}"
            );
        }
    }

    #[test]
    fn test_simd_notw_128_roundtrip() {
        let original: Vec<Complex<f64>> = (0..128)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut data = original.clone();

        // Forward
        notw_128_simd_f64(&mut data, -1);
        // Inverse
        notw_128_simd_f64(&mut data, 1);
        // Normalize
        for x in &mut data {
            *x = *x / 128.0;
        }

        for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                complex_approx_eq(*o, *d, 1e-8),
                "Index {i}: original={o:?}, recovered={d:?}"
            );
        }
    }

    #[test]
    fn test_simd_notw_256_matches_scalar() {
        let mut scalar: Vec<Complex<f64>> = (0..256)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut simd = scalar.clone();

        notw_256(&mut scalar, -1);
        notw_256_simd_f64(&mut simd, -1);

        for (i, (s, d)) in scalar.iter().zip(simd.iter()).enumerate() {
            assert!(
                complex_approx_eq(*s, *d, 1e-5),
                "Index {i}: scalar={s:?}, simd={d:?}"
            );
        }
    }

    #[test]
    fn test_simd_notw_256_roundtrip() {
        let original: Vec<Complex<f64>> = (0..256)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut data = original.clone();

        // Forward
        notw_256_simd_f64(&mut data, -1);
        // Inverse
        notw_256_simd_f64(&mut data, 1);
        // Normalize
        for x in &mut data {
            *x = *x / 256.0;
        }

        for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                complex_approx_eq(*o, *d, 1e-8),
                "Index {i}: original={o:?}, recovered={d:?}"
            );
        }
    }
}
