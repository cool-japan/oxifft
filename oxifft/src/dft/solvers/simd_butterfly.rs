//! SIMD-accelerated butterfly operations for FFT.
//!
//! Provides vectorized butterfly implementations for power-of-2 FFT.
//! Uses runtime CPU feature detection to select optimal implementation.

// Branch structure is intentional for different unrolling strategies
#![allow(clippy::branches_sharing_code)]

use crate::dft::problem::Sign;
use crate::kernel::Complex;
use crate::prelude::*;

/// Convert a `Vec<[f64; 2]>` of exactly 65535 elements into `Box<[[f64; 2]; 65535]>`
/// without panicking. This avoids `unwrap()`/`expect()` on the `try_into()` conversion.
///
/// # Safety invariant
/// The caller must pass a Vec with exactly 65535 elements. This is enforced by
/// a debug assertion; in release builds the length is guaranteed by construction
/// (the Vec is created with `vec![[0.0_f64; 2]; 65535]` and never resized).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn vec_to_boxed_twiddles(v: Vec<[f64; 2]>) -> Box<[[f64; 2]; 65535]> {
    debug_assert_eq!(
        v.len(),
        65535,
        "twiddle vec must have exactly 65535 elements"
    );
    let boxed_slice = v.into_boxed_slice();
    // The length is guaranteed to be 65535 by construction (allocated as
    // `vec![[0.0_f64; 2]; 65535]` with no subsequent push/pop/resize).
    // We perform the conversion via raw pointer to avoid a fallible try_into
    // that would require unwrap()/expect().
    let raw = Box::into_raw(boxed_slice) as *mut [[f64; 2]; 65535];
    // SAFETY: The boxed slice has exactly 65535 elements of type [f64; 2],
    // which is layout-identical to [[f64; 2]; 65535]. The pointer was obtained
    // from Box::into_raw, so it is properly aligned and non-null.
    unsafe { Box::from_raw(raw) }
}

/// DIT butterfly stages with SIMD acceleration for f64.
///
/// Detects CPU features at runtime and uses the fastest available implementation.
#[cfg(target_arch = "x86_64")]
pub fn dit_butterflies_f64(data: &mut [Complex<f64>], sign: Sign) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // Safety: We've verified AVX2+FMA are available
        unsafe { dit_butterflies_avx2(data, sign) }
    } else if is_x86_feature_detected!("sse3") {
        // SSE3 required for _mm_addsub_pd
        // Safety: We've verified SSE3 is available
        unsafe { dit_butterflies_sse3(data, sign) }
    } else {
        // SSE2 alone doesn't provide significant benefit for complex FFT
        dit_butterflies_scalar(data, sign);
    }
}

/// DIT butterfly stages with SIMD acceleration for f64 (aarch64/Apple Silicon).
///
/// Uses NEON 128-bit SIMD for complex arithmetic.
/// NEON is always available on aarch64, so no runtime detection is needed.
#[cfg(target_arch = "aarch64")]
pub fn dit_butterflies_f64(data: &mut [Complex<f64>], sign: Sign) {
    // Safety: NEON is always available on aarch64
    unsafe { dit_butterflies_neon(data, sign) }
}

/// DIT butterfly stages for platforms without SIMD support.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn dit_butterflies_f64(data: &mut [Complex<f64>], sign: Sign) {
    dit_butterflies_scalar(data, sign);
}

/// Precomputed twiddle factors for FFT sizes up to 65536 (aarch64).
///
/// Uses sequential storage for each stage to maximize cache efficiency.
/// Stages 0-15 (m=2 to m=65536). Total storage = 2^16 - 1 = 65535 entries.
#[cfg(target_arch = "aarch64")]
pub struct PrecomputedTwiddlesNeon {
    /// Forward transform twiddles (for FFT)
    pub forward: Box<[[f64; 2]; 65535]>,
    /// Inverse transform twiddles (for IFFT)
    pub inverse: Box<[[f64; 2]; 65535]>,
    /// Offsets for each stage (stage s starts at `offset[s]`)
    pub offsets: [usize; 16],
}

#[cfg(target_arch = "aarch64")]
impl PrecomputedTwiddlesNeon {
    /// Create new precomputed twiddle factors.
    #[must_use]
    pub fn new() -> Self {
        let mut forward = vec![[0.0_f64; 2]; 65535];
        let mut inverse = vec![[0.0_f64; 2]; 65535];

        // Compute offsets: stage s has half_m = 2^s entries
        let mut offsets = [0usize; 16];
        let mut offset = 0;
        for s in 0..16 {
            offsets[s] = offset;
            let half_m = 1 << s;
            let m = half_m * 2;
            // Precompute twiddles for this stage
            for j in 0..half_m {
                let angle = -core::f64::consts::TAU * (j as f64) / (m as f64);
                let (sin_a, cos_a) = (libm::sin(angle), libm::cos(angle));
                forward[offset + j] = [cos_a, sin_a];
                inverse[offset + j] = [cos_a, -sin_a];
            }
            offset += half_m;
        }

        Self {
            forward: vec_to_boxed_twiddles(forward),
            inverse: vec_to_boxed_twiddles(inverse),
            offsets,
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl Default for PrecomputedTwiddlesNeon {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the global precomputed twiddles for NEON.
#[cfg(target_arch = "aarch64")]
pub fn get_twiddles_neon() -> &'static PrecomputedTwiddlesNeon {
    use crate::prelude::OnceLock;
    #[cfg(not(feature = "std"))]
    use crate::prelude::OnceLockExt;
    static TWIDDLES: OnceLock<PrecomputedTwiddlesNeon> = OnceLock::new();
    TWIDDLES.get_or_init(PrecomputedTwiddlesNeon::new)
}

/// NEON DIT butterfly implementation with precomputed twiddles.
///
/// Uses precomputed twiddle factors for accuracy and performance.
/// Special-cases early stages (m <= 16) for twiddle-free or simplified operations.
///
/// # Safety
/// NEON is always available on aarch64, no runtime detection needed.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::useless_let_if_seq)]
unsafe fn dit_butterflies_neon(data: &mut [Complex<f64>], sign: Sign) {
    unsafe {
        use core::arch::aarch64::*;

        let n = data.len();
        let log_n = n.trailing_zeros() as usize;
        let forward = sign == Sign::Forward;
        let sign_f = if forward { -1.0 } else { 1.0 };

        let ptr = data.as_mut_ptr() as *mut f64;

        // Sign pattern for complex multiply: [-1, 1]
        let sign_arr = [-1.0_f64, 1.0];
        let sign_pattern = vld1q_f64(sign_arr.as_ptr());

        // Get precomputed twiddles
        let twiddles = get_twiddles_neon();

        #[allow(clippy::useless_let_if_seq)]
        let mut stage = 0;
        let mut m = 2;

        // Stage 0: m=2 (no twiddle - just add/sub)
        if log_n >= 1 {
            for k in (0..n).step_by(2) {
                let u = vld1q_f64(ptr.add(k * 2));
                let v = vld1q_f64(ptr.add((k + 1) * 2));
                vst1q_f64(ptr.add(k * 2), vaddq_f64(u, v));
                vst1q_f64(ptr.add((k + 1) * 2), vsubq_f64(u, v));
            }
            stage = 1;
            m = 4;
        }

        // Stage 1: m=4 (twiddle is ±i for j=1)
        if log_n >= 2 {
            for k in (0..n).step_by(4) {
                let u0 = vld1q_f64(ptr.add(k * 2));
                let u1 = vld1q_f64(ptr.add((k + 1) * 2));
                let v0 = vld1q_f64(ptr.add((k + 2) * 2));
                let v1 = vld1q_f64(ptr.add((k + 3) * 2));

                // w0 = 1, w1 = ±i → v1 * ±i = (-sign_f * im, sign_f * re)
                let v1_swapped = vextq_f64(v1, v1, 1);
                let scale = vld1q_f64([-sign_f, sign_f].as_ptr());
                let t1 = vmulq_f64(v1_swapped, scale);

                vst1q_f64(ptr.add(k * 2), vaddq_f64(u0, v0));
                vst1q_f64(ptr.add((k + 1) * 2), vaddq_f64(u1, t1));
                vst1q_f64(ptr.add((k + 2) * 2), vsubq_f64(u0, v0));
                vst1q_f64(ptr.add((k + 3) * 2), vsubq_f64(u1, t1));
            }
            stage = 2;
            m = 8;
        }

        // Remaining stages with precomputed twiddles
        while stage < log_n {
            let half_m = m / 2;
            let tw_base = if forward {
                twiddles.forward[twiddles.offsets[stage]..].as_ptr()
            } else {
                twiddles.inverse[twiddles.offsets[stage]..].as_ptr()
            };

            for k in (0..n).step_by(m) {
                let mut j = 0;

                // Unrolled loop: process 4 butterflies at a time
                while j + 4 <= half_m {
                    for offset in 0..4 {
                        let idx = j + offset;
                        let u_idx = k + idx;
                        let v_idx = u_idx + half_m;

                        let u = vld1q_f64(ptr.add(u_idx * 2));
                        let v = vld1q_f64(ptr.add(v_idx * 2));

                        // Load precomputed twiddle
                        let tw = vld1q_f64(tw_base.add(idx) as *const f64);
                        let tw_flip = vextq_f64(tw, tw, 1);

                        // Complex multiply: t = v * tw
                        let v_re = vdupq_laneq_f64::<0>(v);
                        let v_im = vdupq_laneq_f64::<1>(v);
                        let prod1 = vmulq_f64(v_re, tw);
                        let prod2 = vmulq_f64(v_im, tw_flip);
                        let t = vfmaq_f64(prod1, prod2, sign_pattern);

                        vst1q_f64(ptr.add(u_idx * 2), vaddq_f64(u, t));
                        vst1q_f64(ptr.add(v_idx * 2), vsubq_f64(u, t));
                    }
                    j += 4;
                }

                // Handle remaining butterflies
                while j < half_m {
                    let u_idx = k + j;
                    let v_idx = u_idx + half_m;

                    let u = vld1q_f64(ptr.add(u_idx * 2));
                    let v = vld1q_f64(ptr.add(v_idx * 2));

                    let tw = vld1q_f64(tw_base.add(j) as *const f64);
                    let tw_flip = vextq_f64(tw, tw, 1);

                    let v_re = vdupq_laneq_f64::<0>(v);
                    let v_im = vdupq_laneq_f64::<1>(v);
                    let prod1 = vmulq_f64(v_re, tw);
                    let prod2 = vmulq_f64(v_im, tw_flip);
                    let t = vfmaq_f64(prod1, prod2, sign_pattern);

                    vst1q_f64(ptr.add(u_idx * 2), vaddq_f64(u, t));
                    vst1q_f64(ptr.add(v_idx * 2), vsubq_f64(u, t));
                    j += 1;
                }
            }

            stage += 1;
            m *= 2;
        }
    }
}

/// Scalar DIT butterfly implementation with twiddle recurrence.
///
/// Uses the recurrence relation w_{j+1} = w_j * w_step to avoid
/// expensive sin/cos calls for each butterfly.
#[inline]
#[allow(dead_code)] // Used as fallback on non-x86/non-aarch64 platforms and in tests
pub fn dit_butterflies_scalar(data: &mut [Complex<f64>], sign: Sign) {
    let n = data.len();
    let log_n = n.trailing_zeros() as usize;
    let sign_val = f64::from(sign.value());

    let mut m = 2;
    for _ in 0..log_n {
        let half_m = m / 2;
        let angle_step = sign_val * core::f64::consts::TAU / (m as f64);

        // Compute the twiddle step factor once per stage
        let w_step = Complex::cis(angle_step);

        for k in (0..n).step_by(m) {
            // Start with w = 1 (angle = 0)
            let mut w = Complex::new(1.0, 0.0);

            for j in 0..half_m {
                let u = data[k + j];
                let t = data[k + j + half_m] * w;
                data[k + j] = u + t;
                data[k + j + half_m] = u - t;

                // Advance twiddle using recurrence: w_{j+1} = w_j * w_step
                w = w * w_step;
            }
        }
        m *= 2;
    }
}

/// Precomputed twiddle factors for FFT sizes up to 65536.
/// Uses sequential storage for each stage to maximize cache efficiency.
/// Stages 0-15 (m=2 to m=65536). Total storage = 2^16 - 1 = 65535 entries.
#[cfg(target_arch = "x86_64")]
pub struct PrecomputedTwiddles {
    /// Forward transform twiddles (for FFT). Each entry is [cos, sin].
    pub forward: Box<[[f64; 2]; 65535]>,
    /// Inverse transform twiddles (for IFFT). Each entry is [cos, -sin].
    pub inverse: Box<[[f64; 2]; 65535]>,
    /// Offsets for each stage (stage s starts at `offset[s]`)
    pub offsets: [usize; 16],
}

#[cfg(target_arch = "x86_64")]
impl PrecomputedTwiddles {
    fn new() -> Self {
        let mut forward = vec![[0.0_f64; 2]; 65535];
        let mut inverse = vec![[0.0_f64; 2]; 65535];

        // Compute offsets: stage s has half_m = 2^s entries
        // Stage 0 (m=2): half_m=1, offset=0
        // Stage 1 (m=4): half_m=2, offset=1
        // ...
        // Stage 15 (m=65536): half_m=32768, offset=32767
        let mut offsets = [0usize; 16];
        let mut offset = 0;
        for s in 0..16 {
            offsets[s] = offset;
            let half_m = 1 << s;
            let m = half_m * 2;
            // Precompute twiddles for this stage
            for j in 0..half_m {
                let angle = -core::f64::consts::TAU * (j as f64) / (m as f64);
                let (sin_a, cos_a) = (libm::sin(angle), libm::cos(angle));
                forward[offset + j] = [cos_a, sin_a];
                inverse[offset + j] = [cos_a, -sin_a];
            }
            offset += half_m;
        }

        Self {
            forward: vec_to_boxed_twiddles(forward),
            inverse: vec_to_boxed_twiddles(inverse),
            offsets,
        }
    }
}

/// Get the global precomputed twiddles for x86_64.
/// Uses OnceLock for lazy initialization - computed once on first access.
#[cfg(target_arch = "x86_64")]
pub fn get_twiddles() -> &'static PrecomputedTwiddles {
    use crate::prelude::OnceLock;
    #[cfg(not(feature = "std"))]
    use crate::prelude::OnceLockExt;
    static TWIDDLES: OnceLock<PrecomputedTwiddles> = OnceLock::new();
    TWIDDLES.get_or_init(PrecomputedTwiddles::new)
}

/// AVX2 DIT butterfly implementation.
///
/// Processes 2 butterflies (4 complex values) per iteration using 256-bit vectors.
/// Uses FMA for efficient complex multiplication.
/// Special-cases first 2 stages for twiddle-free operations.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available on the current CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dit_butterflies_avx2(data: &mut [Complex<f64>], sign: Sign) {
    unsafe {
        use core::arch::x86_64::*;

        let n = data.len();
        let log_n = n.trailing_zeros() as usize;
        let forward = sign == Sign::Forward;
        let sign_f = if forward { -1.0 } else { 1.0 };

        // Get precomputed twiddles
        let twiddles = get_twiddles();

        let mut stage = 0;
        let mut m;

        // Fused stages 0-3 for n >= 16: process 16 elements at once
        // This reduces memory traffic by doing all early stages in registers
        if log_n >= 4 {
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

            for k in (0..n).step_by(16) {
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

                // Stage 0 (m=2): butterfly pairs (0,1), (2,3), ..., (14,15)
                for i in (0..16).step_by(2) {
                    let u = x[i];
                    let v = x[i + 1];
                    x[i] = u + v;
                    x[i + 1] = u - v;
                }

                // Stage 1 (m=4): butterfly pairs with ±i twiddle
                // (0,2), (1,3), (4,6), (5,7), (8,10), (9,11), (12,14), (13,15)
                for i in (0..16).step_by(4) {
                    let u0 = x[i];
                    let u1 = x[i + 1];
                    let v0 = x[i + 2];
                    let v1 = x[i + 3];
                    // t1 = v1 * (±i)
                    let t1 = Complex::new(-sign_f * v1.im, sign_f * v1.re);
                    x[i] = u0 + v0;
                    x[i + 1] = u1 + t1;
                    x[i + 2] = u0 - v0;
                    x[i + 3] = u1 - t1;
                }

                // Stage 2 (m=8): butterfly pairs with W_8 twiddles
                // (0,4), (1,5), (2,6), (3,7), (8,12), (9,13), (10,14), (11,15)
                for base in [0, 8] {
                    let u0 = x[base];
                    let u1 = x[base + 1];
                    let u2 = x[base + 2];
                    let u3 = x[base + 3];
                    let v0 = x[base + 4]; // w0 = 1
                    let v1 = x[base + 5] * w8_1;
                    let v2 = Complex::new(-sign_f * x[base + 6].im, sign_f * x[base + 6].re); // w2 = ±i
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
                // (0,8), (1,9), (2,10), (3,11), (4,12), (5,13), (6,14), (7,15)
                let t0 = x[8]; // w0 = 1
                let t1 = x[9] * w16_1;
                let t2 = x[10] * w16_2;
                let t3 = x[11] * w16_3;
                let t4 = Complex::new(-sign_f * x[12].im, sign_f * x[12].re); // w4 = ±i
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
            stage = 4;
            m = 32;
        } else {
            // For n < 16, use the original stage-by-stage approach
            // Stage 0: m=2, half_m=1, twiddle is always 1
            if stage < log_n {
                for k in (0..n).step_by(2) {
                    let u = data[k];
                    let v = data[k + 1];
                    data[k] = u + v;
                    data[k + 1] = u - v;
                }
                stage += 1;
            }

            // Stage 1: m=4, twiddles are 1 and ±i
            if stage < log_n {
                for k in (0..n).step_by(4) {
                    let x0 = data[k];
                    let x1 = data[k + 1];
                    let x2 = data[k + 2];
                    let x3 = data[k + 3];
                    let t3 = Complex::new(-sign_f * x3.im, sign_f * x3.re);
                    data[k] = x0 + x2;
                    data[k + 1] = x1 + t3;
                    data[k + 2] = x0 - x2;
                    data[k + 3] = x1 - t3;
                }
                stage += 1;
            }

            // Stage 2: m=8
            if stage < log_n {
                let sqrt2_2 = core::f64::consts::FRAC_1_SQRT_2;
                let w1 = Complex::new(sqrt2_2, sign_f * sqrt2_2);
                let w3 = Complex::new(-sqrt2_2, sign_f * sqrt2_2);
                for k in (0..n).step_by(8) {
                    let x0 = data[k];
                    let x1 = data[k + 1];
                    let x2 = data[k + 2];
                    let x3 = data[k + 3];
                    let t4 = data[k + 4];
                    let t5 = data[k + 5] * w1;
                    let t6 = Complex::new(-sign_f * data[k + 6].im, sign_f * data[k + 6].re);
                    let t7 = data[k + 7] * w3;
                    data[k] = x0 + t4;
                    data[k + 1] = x1 + t5;
                    data[k + 2] = x2 + t6;
                    data[k + 3] = x3 + t7;
                    data[k + 4] = x0 - t4;
                    data[k + 5] = x1 - t5;
                    data[k + 6] = x2 - t6;
                    data[k + 7] = x3 - t7;
                }
                stage += 1;
            }
            m = 16;
        }

        // SIMD stages for m >= 32 with precomputed twiddles
        let ptr = data.as_mut_ptr() as *mut f64;

        // Use radix-4 for pairs of stages when possible
        // Radix-4 combines stages s and s+1 into one, reducing twiddle multiplications
        while stage + 1 < log_n {
            // Radix-4 stage: combines radix-2 stages s and s+1
            let half_m1 = m / 2; // m1 = 2^(s+1), half_m1 = 2^s
            let m2 = m * 2; // m2 = 2^(s+2)
            let half_m2 = m; // half_m2 = 2^(s+1)

            // Twiddle base pointers for the two combined stages
            let tw1_base = if forward {
                twiddles.forward[twiddles.offsets[stage]..].as_ptr()
            } else {
                twiddles.inverse[twiddles.offsets[stage]..].as_ptr()
            };
            let tw2_base = if forward {
                twiddles.forward[twiddles.offsets[stage + 1]..].as_ptr()
            } else {
                twiddles.inverse[twiddles.offsets[stage + 1]..].as_ptr()
            };

            // Process groups of size m2
            for k in (0..n).step_by(m2) {
                let mut j = 0;

                // Process 2 radix-4 butterflies at a time using AVX256 (8 complex values)
                while j + 2 <= half_m1 {
                    // Load twiddles for first stage (2 twiddles = 4 f64s)
                    let tw1 = _mm256_loadu_pd(tw1_base.add(j) as *const f64);
                    // Load twiddles for second stage (j and j+half_m1)
                    let tw2_a = _mm256_loadu_pd(tw2_base.add(j) as *const f64);
                    let tw2_b = _mm256_loadu_pd(tw2_base.add(j + half_m1) as *const f64);

                    // Compute pointers for 2 radix-4 butterflies (8 complex values total)
                    // Butterfly 0: indices k+j, k+j+half_m1, k+j+half_m2, k+j+half_m2+half_m1
                    // Butterfly 1: indices k+j+1, k+j+1+half_m1, k+j+1+half_m2, k+j+1+half_m2+half_m1
                    let x0_ptr = ptr.add((k + j) * 2);
                    let x1_ptr = ptr.add((k + j + half_m1) * 2);
                    let x2_ptr = ptr.add((k + j + half_m2) * 2);
                    let x3_ptr = ptr.add((k + j + half_m2 + half_m1) * 2);

                    // Load 4 pairs of complex values (each pair: 2 complex from consecutive butterflies)
                    let x0 = _mm256_loadu_pd(x0_ptr); // [re0, im0, re1, im1]
                    let x1 = _mm256_loadu_pd(x1_ptr);
                    let x2 = _mm256_loadu_pd(x2_ptr);
                    let x3 = _mm256_loadu_pd(x3_ptr);

                    // Expand twiddles for parallel multiply
                    let tw1_re = _mm256_permute_pd(tw1, 0b0000);
                    let tw1_im = _mm256_permute_pd(tw1, 0b1111);
                    let tw2a_re = _mm256_permute_pd(tw2_a, 0b0000);
                    let tw2a_im = _mm256_permute_pd(tw2_a, 0b1111);
                    let tw2b_re = _mm256_permute_pd(tw2_b, 0b0000);
                    let tw2b_im = _mm256_permute_pd(tw2_b, 0b1111);

                    // First radix-2 stage: t1 = x1 * tw1, t3 = x3 * tw1
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

                    // Second radix-2 stage: t2a = a2 * tw2_a, t2b = a3 * tw2_b
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

                    // Final butterflies and store
                    _mm256_storeu_pd(x0_ptr, _mm256_add_pd(a0, t2a));
                    _mm256_storeu_pd(x2_ptr, _mm256_sub_pd(a0, t2a));
                    _mm256_storeu_pd(x1_ptr, _mm256_add_pd(a1, t2b));
                    _mm256_storeu_pd(x3_ptr, _mm256_sub_pd(a1, t2b));

                    j += 2;
                }

                // Handle remaining butterflies one at a time
                while j < half_m1 {
                    let i0 = k + j;
                    let i1 = k + j + half_m1;
                    let i2 = k + j + half_m2;
                    let i3 = k + j + half_m2 + half_m1;

                    let tw1_ptr = tw1_base.add(j) as *const f64;
                    let tw2_a_ptr = tw2_base.add(j) as *const f64;
                    let tw2_b_ptr = tw2_base.add(j + half_m1) as *const f64;

                    let w1 = Complex::new(*tw1_ptr, *tw1_ptr.add(1));
                    let w2_a = Complex::new(*tw2_a_ptr, *tw2_a_ptr.add(1));
                    let w2_b = Complex::new(*tw2_b_ptr, *tw2_b_ptr.add(1));

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

            stage += 2;
            m *= 4;
        }

        // Handle remaining single stage if log_n is odd
        while stage < log_n {
            let half_m = m / 2;
            // Get base pointer to this stage's twiddles for direct loading
            let tw_base = if forward {
                twiddles.forward[twiddles.offsets[stage]..].as_ptr()
            } else {
                twiddles.inverse[twiddles.offsets[stage]..].as_ptr()
            };

            for k in (0..n).step_by(m) {
                let mut j = 0;

                // Process 8 butterflies at a time with interleaved operations
                // This better utilizes ILP by grouping loads, computes, and stores
                while j + 8 <= half_m {
                    // Load all twiddles first
                    let tw01 = _mm256_loadu_pd(tw_base.add(j) as *const f64);
                    let tw23 = _mm256_loadu_pd(tw_base.add(j + 2) as *const f64);
                    let tw45 = _mm256_loadu_pd(tw_base.add(j + 4) as *const f64);
                    let tw67 = _mm256_loadu_pd(tw_base.add(j + 6) as *const f64);

                    // Compute all pointers
                    let u0_ptr = ptr.add((k + j) * 2);
                    let v0_ptr = ptr.add((k + j + half_m) * 2);
                    let u1_ptr = ptr.add((k + j + 2) * 2);
                    let v1_ptr = ptr.add((k + j + 2 + half_m) * 2);
                    let u2_ptr = ptr.add((k + j + 4) * 2);
                    let v2_ptr = ptr.add((k + j + 4 + half_m) * 2);
                    let u3_ptr = ptr.add((k + j + 6) * 2);
                    let v3_ptr = ptr.add((k + j + 6 + half_m) * 2);

                    // Load all data
                    let u0 = _mm256_loadu_pd(u0_ptr);
                    let v0 = _mm256_loadu_pd(v0_ptr);
                    let u1 = _mm256_loadu_pd(u1_ptr);
                    let v1 = _mm256_loadu_pd(v1_ptr);
                    let u2 = _mm256_loadu_pd(u2_ptr);
                    let v2 = _mm256_loadu_pd(v2_ptr);
                    let u3 = _mm256_loadu_pd(u3_ptr);
                    let v3 = _mm256_loadu_pd(v3_ptr);

                    // Expand twiddles
                    let tw01_re = _mm256_permute_pd(tw01, 0b0000);
                    let tw01_im = _mm256_permute_pd(tw01, 0b1111);
                    let tw23_re = _mm256_permute_pd(tw23, 0b0000);
                    let tw23_im = _mm256_permute_pd(tw23, 0b1111);
                    let tw45_re = _mm256_permute_pd(tw45, 0b0000);
                    let tw45_im = _mm256_permute_pd(tw45, 0b1111);
                    let tw67_re = _mm256_permute_pd(tw67, 0b0000);
                    let tw67_im = _mm256_permute_pd(tw67, 0b1111);

                    // Expand v components
                    let v0_re = _mm256_permute_pd(v0, 0b0000);
                    let v0_im = _mm256_permute_pd(v0, 0b1111);
                    let v1_re = _mm256_permute_pd(v1, 0b0000);
                    let v1_im = _mm256_permute_pd(v1, 0b1111);
                    let v2_re = _mm256_permute_pd(v2, 0b0000);
                    let v2_im = _mm256_permute_pd(v2, 0b1111);
                    let v3_re = _mm256_permute_pd(v3, 0b0000);
                    let v3_im = _mm256_permute_pd(v3, 0b1111);

                    // Compute t = v * tw (interleaved)
                    let t0_re = _mm256_fnmadd_pd(v0_im, tw01_im, _mm256_mul_pd(v0_re, tw01_re));
                    let t0_im = _mm256_fmadd_pd(v0_im, tw01_re, _mm256_mul_pd(v0_re, tw01_im));
                    let t1_re = _mm256_fnmadd_pd(v1_im, tw23_im, _mm256_mul_pd(v1_re, tw23_re));
                    let t1_im = _mm256_fmadd_pd(v1_im, tw23_re, _mm256_mul_pd(v1_re, tw23_im));
                    let t2_re = _mm256_fnmadd_pd(v2_im, tw45_im, _mm256_mul_pd(v2_re, tw45_re));
                    let t2_im = _mm256_fmadd_pd(v2_im, tw45_re, _mm256_mul_pd(v2_re, tw45_im));
                    let t3_re = _mm256_fnmadd_pd(v3_im, tw67_im, _mm256_mul_pd(v3_re, tw67_re));
                    let t3_im = _mm256_fmadd_pd(v3_im, tw67_re, _mm256_mul_pd(v3_re, tw67_im));

                    // Blend to get complex results
                    let t0 = _mm256_blend_pd(t0_re, t0_im, 0b1010);
                    let t1 = _mm256_blend_pd(t1_re, t1_im, 0b1010);
                    let t2 = _mm256_blend_pd(t2_re, t2_im, 0b1010);
                    let t3 = _mm256_blend_pd(t3_re, t3_im, 0b1010);

                    // Store all results
                    _mm256_storeu_pd(u0_ptr, _mm256_add_pd(u0, t0));
                    _mm256_storeu_pd(v0_ptr, _mm256_sub_pd(u0, t0));
                    _mm256_storeu_pd(u1_ptr, _mm256_add_pd(u1, t1));
                    _mm256_storeu_pd(v1_ptr, _mm256_sub_pd(u1, t1));
                    _mm256_storeu_pd(u2_ptr, _mm256_add_pd(u2, t2));
                    _mm256_storeu_pd(v2_ptr, _mm256_sub_pd(u2, t2));
                    _mm256_storeu_pd(u3_ptr, _mm256_add_pd(u3, t3));
                    _mm256_storeu_pd(v3_ptr, _mm256_sub_pd(u3, t3));

                    j += 8;
                }

                // Handle remaining butterflies (4 at a time)
                while j + 4 <= half_m {
                    let tw01 = _mm256_loadu_pd(tw_base.add(j) as *const f64);
                    let tw23 = _mm256_loadu_pd(tw_base.add(j + 2) as *const f64);

                    let u0_ptr = ptr.add((k + j) * 2);
                    let v0_ptr = ptr.add((k + j + half_m) * 2);
                    let u1_ptr = ptr.add((k + j + 2) * 2);
                    let v1_ptr = ptr.add((k + j + 2 + half_m) * 2);

                    let u0 = _mm256_loadu_pd(u0_ptr);
                    let v0 = _mm256_loadu_pd(v0_ptr);
                    let u1 = _mm256_loadu_pd(u1_ptr);
                    let v1 = _mm256_loadu_pd(v1_ptr);

                    let tw01_re = _mm256_permute_pd(tw01, 0b0000);
                    let tw01_im = _mm256_permute_pd(tw01, 0b1111);
                    let tw23_re = _mm256_permute_pd(tw23, 0b0000);
                    let tw23_im = _mm256_permute_pd(tw23, 0b1111);

                    let v0_re = _mm256_permute_pd(v0, 0b0000);
                    let v0_im = _mm256_permute_pd(v0, 0b1111);
                    let v1_re = _mm256_permute_pd(v1, 0b0000);
                    let v1_im = _mm256_permute_pd(v1, 0b1111);

                    let t0_re = _mm256_fnmadd_pd(v0_im, tw01_im, _mm256_mul_pd(v0_re, tw01_re));
                    let t0_im = _mm256_fmadd_pd(v0_im, tw01_re, _mm256_mul_pd(v0_re, tw01_im));
                    let t1_re = _mm256_fnmadd_pd(v1_im, tw23_im, _mm256_mul_pd(v1_re, tw23_re));
                    let t1_im = _mm256_fmadd_pd(v1_im, tw23_re, _mm256_mul_pd(v1_re, tw23_im));

                    let t0 = _mm256_blend_pd(t0_re, t0_im, 0b1010);
                    let t1 = _mm256_blend_pd(t1_re, t1_im, 0b1010);

                    _mm256_storeu_pd(u0_ptr, _mm256_add_pd(u0, t0));
                    _mm256_storeu_pd(v0_ptr, _mm256_sub_pd(u0, t0));
                    _mm256_storeu_pd(u1_ptr, _mm256_add_pd(u1, t1));
                    _mm256_storeu_pd(v1_ptr, _mm256_sub_pd(u1, t1));

                    j += 4;
                }

                // Handle remaining butterflies (2 at a time)
                while j + 2 <= half_m {
                    let tw = _mm256_loadu_pd(tw_base.add(j) as *const f64);
                    let tw_re = _mm256_permute_pd(tw, 0b0000);
                    let tw_im = _mm256_permute_pd(tw, 0b1111);

                    let u_ptr = ptr.add((k + j) * 2);
                    let v_ptr = ptr.add((k + j + half_m) * 2);

                    let u = _mm256_loadu_pd(u_ptr);
                    let v = _mm256_loadu_pd(v_ptr);

                    let v_re = _mm256_permute_pd(v, 0b0000);
                    let v_im = _mm256_permute_pd(v, 0b1111);

                    let t_re = _mm256_fnmadd_pd(v_im, tw_im, _mm256_mul_pd(v_re, tw_re));
                    let t_im = _mm256_fmadd_pd(v_im, tw_re, _mm256_mul_pd(v_re, tw_im));
                    let t = _mm256_blend_pd(t_re, t_im, 0b1010);

                    _mm256_storeu_pd(u_ptr, _mm256_add_pd(u, t));
                    _mm256_storeu_pd(v_ptr, _mm256_sub_pd(u, t));

                    j += 2;
                }

                // Handle remaining single butterfly (if half_m is odd, but it's always power of 2)
                if j < half_m {
                    let tw_ptr = tw_base.add(j) as *const f64;
                    let w = Complex::new(*tw_ptr, *tw_ptr.add(1));
                    let u = data[k + j];
                    let t = data[k + j + half_m] * w;
                    data[k + j] = u + t;
                    data[k + j + half_m] = u - t;
                }
            }

            m *= 2;
            stage += 1;
        }
    }
}

/// SSE3 DIT butterfly implementation.
///
/// Processes 1 butterfly (2 complex values) per iteration using 128-bit vectors.
/// Requires SSE3 for _mm_addsub_pd. Uses twiddle recurrence for efficiency.
///
/// # Safety
/// Caller must ensure SSE3 is available on the current CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse3")]
unsafe fn dit_butterflies_sse3(data: &mut [Complex<f64>], sign: Sign) {
    unsafe {
        use core::arch::x86_64::*;

        let n = data.len();
        let log_n = n.trailing_zeros() as usize;
        let sign_val = f64::from(sign.value());

        let mut m = 2;
        for _ in 0..log_n {
            let half_m = m / 2;
            let angle_step = sign_val * core::f64::consts::TAU / (m as f64);
            let w_step = Complex::cis(angle_step);

            let ptr = data.as_mut_ptr() as *mut f64;

            for k in (0..n).step_by(m) {
                let mut w = Complex::new(1.0, 0.0);

                for j in 0..half_m {
                    // Load u (1 complex = 2 f64)
                    let u_ptr = ptr.add((k + j) * 2);
                    let u = _mm_loadu_pd(u_ptr);

                    // Load v (1 complex = 2 f64)
                    let v_ptr = ptr.add((k + j + half_m) * 2);
                    let v = _mm_loadu_pd(v_ptr);

                    // Complex multiply: t = v * twiddle
                    // v = [v_re, v_im]
                    let v_re = _mm_shuffle_pd(v, v, 0b00); // [v_re, v_re]
                    let v_im = _mm_shuffle_pd(v, v, 0b11); // [v_im, v_im]

                    // t_re = v_re * tw_re - v_im * tw_im
                    // t_im = v_re * tw_im + v_im * tw_re
                    let prod1 = _mm_mul_pd(v_re, _mm_set_pd(w.im, w.re)); // [v_re*cos, v_re*sin]
                    let prod2 = _mm_mul_pd(v_im, _mm_set_pd(w.re, w.im)); // [v_im*sin, v_im*cos]

                    // addsub: [a0-b0, a1+b1]
                    // We want [v_re*cos - v_im*sin, v_re*sin + v_im*cos]
                    let t = _mm_addsub_pd(prod1, _mm_shuffle_pd(prod2, prod2, 0b01));

                    // Butterfly
                    let out_u = _mm_add_pd(u, t);
                    let out_v = _mm_sub_pd(u, t);

                    _mm_storeu_pd(u_ptr, out_u);
                    _mm_storeu_pd(v_ptr, out_v);

                    // Advance twiddle using recurrence
                    w = w * w_step;
                }
            }
            m *= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
        (a.re - b.re).abs() < eps && (a.im - b.im).abs() < eps
    }

    #[test]
    fn test_scalar_butterfly() {
        // Simple test: size 4
        let mut data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let original = data.clone();

        dit_butterflies_scalar(&mut data, Sign::Forward);

        // Verify against expected DFT output (after bit-reversal would be applied)
        // For now just verify it changed and is deterministic
        let mut data2 = original;
        dit_butterflies_scalar(&mut data2, Sign::Forward);

        for (a, b) in data.iter().zip(data2.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_simd_matches_scalar_size_8() {
        let mut data_scalar = vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(7.0, 0.0),
        ];
        let mut data_simd = data_scalar.clone();

        dit_butterflies_scalar(&mut data_scalar, Sign::Forward);
        dit_butterflies_f64(&mut data_simd, Sign::Forward);

        for (a, b) in data_scalar.iter().zip(data_simd.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10), "Mismatch: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn test_simd_matches_scalar_size_16() {
        let mut data_scalar: Vec<Complex<f64>> = (0..16)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut data_simd = data_scalar.clone();

        dit_butterflies_scalar(&mut data_scalar, Sign::Forward);
        dit_butterflies_f64(&mut data_simd, Sign::Forward);

        for (a, b) in data_scalar.iter().zip(data_simd.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9), "Mismatch: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn test_simd_matches_scalar_size_64() {
        let mut data_scalar: Vec<Complex<f64>> = (0..64)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut data_simd = data_scalar.clone();

        dit_butterflies_scalar(&mut data_scalar, Sign::Forward);
        dit_butterflies_f64(&mut data_simd, Sign::Forward);

        for (a, b) in data_scalar.iter().zip(data_simd.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9), "Mismatch: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn test_simd_matches_scalar_size_1024() {
        let mut data_scalar: Vec<Complex<f64>> = (0..1024)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut data_simd = data_scalar.clone();

        dit_butterflies_scalar(&mut data_scalar, Sign::Forward);
        dit_butterflies_f64(&mut data_simd, Sign::Forward);

        for (i, (a, b)) in data_scalar.iter().zip(data_simd.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *b, 1e-8),
                "Mismatch at {i}: {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn test_simd_backward_matches_scalar() {
        let mut data_scalar: Vec<Complex<f64>> = (0..64)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut data_simd = data_scalar.clone();

        dit_butterflies_scalar(&mut data_scalar, Sign::Backward);
        dit_butterflies_f64(&mut data_simd, Sign::Backward);

        for (a, b) in data_scalar.iter().zip(data_simd.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9), "Mismatch: {a:?} vs {b:?}");
        }
    }
}
