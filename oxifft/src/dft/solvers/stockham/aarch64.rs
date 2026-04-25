//! aarch64-specific Stockham FFT implementations.
//!
//! Contains NEON optimized implementations.

use crate::dft::problem::Sign;
use crate::dft::solvers::simd_butterfly::get_twiddles_neon;
use crate::kernel::Complex;
use crate::prelude::*;

/// Stage-fused Stockham FFT for f64 (aarch64) with full NEON SIMD.
///
/// Uses stage fusion to halve memory passes, achieving radix-4 equivalent performance.
/// Fully vectorized using NEON intrinsics for maximum throughput.
///
/// # Safety
///
/// Caller must ensure the target CPU supports the `neon` feature.
/// On aarch64, NEON is always available; this is enforced by the
/// `#[target_feature(enable = "neon")]` attribute.
/// Both `input` and `output` must have the same length, which must be a
/// power of two.
#[target_feature(enable = "neon")]
pub unsafe fn stockham_radix4_neon(
    input: &[Complex<f64>],
    output: &mut [Complex<f64>],
    sign: Sign,
) {
    unsafe {
        use core::arch::aarch64::*;

        let n = input.len();
        let log_n = n.trailing_zeros() as usize;

        // Specialized kernels for small sizes
        if n <= 4 {
            stockham_small_neon(input, output, sign);
            return;
        }

        let half_n = n / 2;
        let quarter_n = n / 4;

        // Allocate scratch buffer
        let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); n];

        // Calculate total writes for ping-pong logic
        let num_fused = log_n / 2;
        let has_final = usize::from(log_n % 2 == 1);
        let total_writes = num_fused + has_final;

        let (mut src_ptr, mut dst_ptr): (*mut f64, *mut f64) = if total_writes.is_multiple_of(2) {
            output.copy_from_slice(input);
            (
                output.as_mut_ptr() as *mut f64,
                scratch.as_mut_ptr() as *mut f64,
            )
        } else {
            scratch.copy_from_slice(input);
            (
                scratch.as_mut_ptr() as *mut f64,
                output.as_mut_ptr() as *mut f64,
            )
        };

        // Sign pattern for complex multiply: [-1, 1] for (re*im - im*re, re*im + im*re)
        let sign_pattern = vld1q_f64([-1.0_f64, 1.0].as_ptr());

        // Get precomputed twiddles
        let twiddles = get_twiddles_neon();
        let forward = sign == Sign::Forward;

        // Fused stage processing (2 stages at a time)
        let mut stage = 0;
        let mut m = 1usize;

        while stage + 1 < log_n {
            let m1 = m;
            let m2 = m * 2;
            let m4 = m * 4;

            // Get twiddle base pointers for this fused stage pair
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

            let src = src_ptr;
            let dst = dst_ptr;
            let num_groups = half_n / m2;

            // Prefetch hint distance (in elements)
            let prefetch_dist = 16;

            for g in 0..num_groups {
                let mut j = 0;
                while j + 2 <= m1 {
                    // Software prefetch using stable inline asm
                    if j + prefetch_dist < m1 {
                        let k_pf = g * m1 + j + prefetch_dist;
                        prefetch_read(src.add((k_pf) * 2) as *const i8);
                        prefetch_read(src.add((k_pf + quarter_n) * 2) as *const i8);
                    }

                    let k0 = g * m1 + j;
                    let k1 = k0 + 1;

                    // Process first element pair (j)
                    let s0_0 = k0;
                    let s1_0 = k0 + quarter_n;
                    let s2_0 = k0 + half_n;
                    let s3_0 = k0 + half_n + quarter_n;

                    let dst_base0 = g * m4 + j;

                    // Load complex values
                    let x0_0 = vld1q_f64(src.add(s0_0 * 2));
                    let x1_0 = vld1q_f64(src.add(s1_0 * 2));
                    let x2_0 = vld1q_f64(src.add(s2_0 * 2));
                    let x3_0 = vld1q_f64(src.add(s3_0 * 2));

                    // Stage s: t2 = x2 * w1[j], t3 = x3 * w1[j]
                    // Load precomputed twiddle for first stage
                    let tw1_0 = vld1q_f64(tw1_base.add(j) as *const f64);
                    let tw1_flip_0 = vextq_f64(tw1_0, tw1_0, 1);

                    let t2_0 = neon_complex_mul(x2_0, tw1_0, tw1_flip_0, sign_pattern);
                    let t3_0 = neon_complex_mul(x3_0, tw1_0, tw1_flip_0, sign_pattern);

                    let a0_0 = vaddq_f64(x0_0, t2_0);
                    let a1_0 = vsubq_f64(x0_0, t2_0);
                    let a2_0 = vaddq_f64(x1_0, t3_0);
                    let a3_0 = vsubq_f64(x1_0, t3_0);

                    // Stage s+1: b2 = a2 * w2[j], b3 = a3 * w2[j+m1]
                    // Load precomputed twiddles for second stage
                    let tw2a_0 = vld1q_f64(tw2_base.add(j) as *const f64);
                    let tw2a_flip_0 = vextq_f64(tw2a_0, tw2a_0, 1);
                    let tw2b_0 = vld1q_f64(tw2_base.add(j + m1) as *const f64);
                    let tw2b_flip_0 = vextq_f64(tw2b_0, tw2b_0, 1);

                    let b2_0 = neon_complex_mul(a2_0, tw2a_0, tw2a_flip_0, sign_pattern);
                    let b3_0 = neon_complex_mul(a3_0, tw2b_0, tw2b_flip_0, sign_pattern);

                    vst1q_f64(dst.add(dst_base0 * 2), vaddq_f64(a0_0, b2_0));
                    vst1q_f64(dst.add((dst_base0 + m1) * 2), vaddq_f64(a1_0, b3_0));
                    vst1q_f64(dst.add((dst_base0 + m2) * 2), vsubq_f64(a0_0, b2_0));
                    vst1q_f64(dst.add((dst_base0 + m2 + m1) * 2), vsubq_f64(a1_0, b3_0));

                    // Process second element pair (j+1)
                    let s0_1 = k1;
                    let s1_1 = k1 + quarter_n;
                    let s2_1 = k1 + half_n;
                    let s3_1 = k1 + half_n + quarter_n;

                    let dst_base1 = g * m4 + j + 1;

                    let x0_1 = vld1q_f64(src.add(s0_1 * 2));
                    let x1_1 = vld1q_f64(src.add(s1_1 * 2));
                    let x2_1 = vld1q_f64(src.add(s2_1 * 2));
                    let x3_1 = vld1q_f64(src.add(s3_1 * 2));

                    // Load precomputed twiddle for first stage (j+1)
                    let tw1_1 = vld1q_f64(tw1_base.add(j + 1) as *const f64);
                    let tw1_flip_1 = vextq_f64(tw1_1, tw1_1, 1);

                    let t2_1 = neon_complex_mul(x2_1, tw1_1, tw1_flip_1, sign_pattern);
                    let t3_1 = neon_complex_mul(x3_1, tw1_1, tw1_flip_1, sign_pattern);

                    let a0_1 = vaddq_f64(x0_1, t2_1);
                    let a1_1 = vsubq_f64(x0_1, t2_1);
                    let a2_1 = vaddq_f64(x1_1, t3_1);
                    let a3_1 = vsubq_f64(x1_1, t3_1);

                    // Load precomputed twiddles for second stage (j+1)
                    let tw2a_1 = vld1q_f64(tw2_base.add(j + 1) as *const f64);
                    let tw2a_flip_1 = vextq_f64(tw2a_1, tw2a_1, 1);
                    let tw2b_1 = vld1q_f64(tw2_base.add(j + 1 + m1) as *const f64);
                    let tw2b_flip_1 = vextq_f64(tw2b_1, tw2b_1, 1);

                    let b2_1 = neon_complex_mul(a2_1, tw2a_1, tw2a_flip_1, sign_pattern);
                    let b3_1 = neon_complex_mul(a3_1, tw2b_1, tw2b_flip_1, sign_pattern);

                    vst1q_f64(dst.add(dst_base1 * 2), vaddq_f64(a0_1, b2_1));
                    vst1q_f64(dst.add((dst_base1 + m1) * 2), vaddq_f64(a1_1, b3_1));
                    vst1q_f64(dst.add((dst_base1 + m2) * 2), vsubq_f64(a0_1, b2_1));
                    vst1q_f64(dst.add((dst_base1 + m2 + m1) * 2), vsubq_f64(a1_1, b3_1));

                    j += 2;
                }

                // Handle remaining odd element
                while j < m1 {
                    let k = g * m1 + j;
                    let s0 = k;
                    let s1 = k + quarter_n;
                    let s2 = k + half_n;
                    let s3 = k + half_n + quarter_n;

                    let dst_base = g * m4 + j;

                    let x0 = vld1q_f64(src.add(s0 * 2));
                    let x1 = vld1q_f64(src.add(s1 * 2));
                    let x2 = vld1q_f64(src.add(s2 * 2));
                    let x3 = vld1q_f64(src.add(s3 * 2));

                    // Load precomputed twiddle for first stage
                    let tw1 = vld1q_f64(tw1_base.add(j) as *const f64);
                    let tw1_flip = vextq_f64(tw1, tw1, 1);

                    let t2 = neon_complex_mul(x2, tw1, tw1_flip, sign_pattern);
                    let t3 = neon_complex_mul(x3, tw1, tw1_flip, sign_pattern);

                    let a0 = vaddq_f64(x0, t2);
                    let a1 = vsubq_f64(x0, t2);
                    let a2 = vaddq_f64(x1, t3);
                    let a3 = vsubq_f64(x1, t3);

                    // Load precomputed twiddles for second stage
                    let tw2a = vld1q_f64(tw2_base.add(j) as *const f64);
                    let tw2a_flip = vextq_f64(tw2a, tw2a, 1);
                    let tw2b = vld1q_f64(tw2_base.add(j + m1) as *const f64);
                    let tw2b_flip = vextq_f64(tw2b, tw2b, 1);

                    let b2 = neon_complex_mul(a2, tw2a, tw2a_flip, sign_pattern);
                    let b3 = neon_complex_mul(a3, tw2b, tw2b_flip, sign_pattern);

                    vst1q_f64(dst.add(dst_base * 2), vaddq_f64(a0, b2));
                    vst1q_f64(dst.add((dst_base + m1) * 2), vaddq_f64(a1, b3));
                    vst1q_f64(dst.add((dst_base + m2) * 2), vsubq_f64(a0, b2));
                    vst1q_f64(dst.add((dst_base + m2 + m1) * 2), vsubq_f64(a1, b3));

                    j += 1;
                }
            }

            core::mem::swap(&mut src_ptr, &mut dst_ptr);
            stage += 2;
            m *= 4;
        }

        // Handle remaining single stage if log_n is odd
        if stage < log_n {
            let m2 = m * 2;

            // Get twiddle base pointer for this stage
            let tw_base = if forward {
                twiddles.forward[twiddles.offsets[stage]..].as_ptr()
            } else {
                twiddles.inverse[twiddles.offsets[stage]..].as_ptr()
            };

            let src = src_ptr;
            let dst = dst_ptr;
            let num_groups = half_n / m;

            for g in 0..num_groups {
                let src_base = g * m;
                let dst_base = g * m2;

                for j in 0..m {
                    let src_u = src_base + j;
                    let src_v = src_u + half_n;
                    let dst_u = dst_base + j;
                    let dst_v = dst_u + m;

                    let u = vld1q_f64(src.add(src_u * 2));
                    let v = vld1q_f64(src.add(src_v * 2));

                    // Load precomputed twiddle
                    let tw = vld1q_f64(tw_base.add(j) as *const f64);
                    let tw_flip = vextq_f64(tw, tw, 1);

                    let t = neon_complex_mul(v, tw, tw_flip, sign_pattern);

                    vst1q_f64(dst.add(dst_u * 2), vaddq_f64(u, t));
                    vst1q_f64(dst.add(dst_v * 2), vsubq_f64(u, t));
                }
            }
        }
    }
}

/// NEON complex multiply helper: (a + bi) * (c + di)
///
/// # Safety
///
/// Caller must ensure the target CPU supports the `neon` feature (always true on aarch64).
/// The NEON register arguments must contain valid float64 data.
#[inline(always)]
unsafe fn neon_complex_mul(
    v: core::arch::aarch64::float64x2_t,
    tw: core::arch::aarch64::float64x2_t,
    tw_flip: core::arch::aarch64::float64x2_t,
    sign_pattern: core::arch::aarch64::float64x2_t,
) -> core::arch::aarch64::float64x2_t {
    unsafe {
        use core::arch::aarch64::*;
        let v_re = vdupq_laneq_f64::<0>(v);
        let v_im = vdupq_laneq_f64::<1>(v);
        let prod1 = vmulq_f64(v_re, tw);
        let prod2 = vmulq_f64(v_im, tw_flip);
        vfmaq_f64(prod1, prod2, sign_pattern)
    }
}

/// Specialized small-size Stockham for n <= 4 using NEON.
///
/// # Safety
///
/// Caller must ensure the target CPU supports the `neon` feature (always true on aarch64).
/// `input` and `output` must have the same length, which must be 1, 2, or 4.
#[target_feature(enable = "neon")]
unsafe fn stockham_small_neon(input: &[Complex<f64>], output: &mut [Complex<f64>], sign: Sign) {
    unsafe {
        use core::arch::aarch64::*;
        let n = input.len();
        let sign_val = f64::from(sign.value());

        match n {
            1 => {
                output[0] = input[0];
            }
            2 => {
                // Size 2: single butterfly, no twiddle
                let x0 = vld1q_f64(input.as_ptr() as *const f64);
                let x1 = vld1q_f64((input.as_ptr() as *const f64).add(2));
                vst1q_f64(output.as_mut_ptr() as *mut f64, vaddq_f64(x0, x1));
                vst1q_f64((output.as_mut_ptr() as *mut f64).add(2), vsubq_f64(x0, x1));
            }
            4 => {
                // Size 4: radix-4 butterfly
                let x0 = vld1q_f64(input.as_ptr() as *const f64);
                let x1 = vld1q_f64((input.as_ptr() as *const f64).add(2));
                let x2 = vld1q_f64((input.as_ptr() as *const f64).add(4));
                let x3 = vld1q_f64((input.as_ptr() as *const f64).add(6));

                // First stage butterflies
                let a = vaddq_f64(x0, x2);
                let b = vsubq_f64(x0, x2);
                let c = vaddq_f64(x1, x3);
                let diff = vsubq_f64(x1, x3);

                // Rotate diff by ±90°
                let swapped = vextq_f64(diff, diff, 1);
                let d = if sign_val < 0.0 {
                    vmulq_f64(swapped, vld1q_f64([1.0, -1.0].as_ptr()))
                } else {
                    vmulq_f64(swapped, vld1q_f64([-1.0, 1.0].as_ptr()))
                };

                // Output
                vst1q_f64(output.as_mut_ptr() as *mut f64, vaddq_f64(a, c));
                vst1q_f64((output.as_mut_ptr() as *mut f64).add(2), vaddq_f64(b, d));
                vst1q_f64((output.as_mut_ptr() as *mut f64).add(4), vsubq_f64(a, c));
                vst1q_f64((output.as_mut_ptr() as *mut f64).add(6), vsubq_f64(b, d));
            }
            _ => {
                output.copy_from_slice(input);
            }
        }
    }
}

/// Software prefetch for aarch64 using stable inline assembly.
/// Uses prfm (prefetch memory) with pldl1keep (prefetch for load, L1 cache, keep in cache).
///
/// # Safety
///
/// `addr` must be a valid address (does not need to point to readable memory —
/// a prefetch to an unmapped address is architecturally defined to be a no-op
/// on aarch64, but passing a completely arbitrary integer cast as a pointer may
/// violate Rust's provenance rules). Prefer passing addresses derived from valid
/// slice pointers.
#[inline(always)]
unsafe fn prefetch_read(addr: *const i8) {
    // MIRI does not support inline asm; prefetch is a hint with no functional
    // effect, so skip it entirely under MIRI to allow correctness testing.
    #[cfg(not(miri))]
    {
        use core::arch::asm;
        unsafe {
            asm!(
                "prfm pldl1keep, [{0}]",
                in(reg) addr,
                options(readonly, nostack, preserves_flags)
            );
        }
    }
    #[cfg(miri)]
    let _ = addr;
}
