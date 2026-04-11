//! SIMD dispatch functions for large DFT sizes (64, 128, 256, 512, 1024, 4096).
//!
//! Contains iterative DIT implementations with precomputed twiddle factors
//! and architecture-specific SIMD butterfly operations.

// Items after statements are intentional for precomputed twiddle tables
#![allow(clippy::items_after_statements)]
// Large stack arrays are intentional for performance in fixed-size transforms
#![allow(clippy::large_stack_arrays)]

use crate::kernel::Complex;
use crate::prelude::*;

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
