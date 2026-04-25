//! NEON codelet emitters (aarch64).
//!
//! f64 variant: 128-bit, 2×f64 = 1 complex per `float64x2_t` register.
//! f32 variant: 128-bit, 4×f32 = uses `float32x2_t` (2 lanes = 1 complex) for clean
//! structural correspondence with the f64 path; `vld1_f32` / `vadd_f32` family.

use proc_macro2::TokenStream;
use quote::quote;

/// NEON size-2 butterfly on f64 data.
pub(super) fn gen_neon_size_2() -> TokenStream {
    quote! {
        /// Size-2 butterfly using NEON intrinsics for f64 data.
        ///
        /// # Safety
        /// - Must be called on aarch64 (NEON is always available).
        /// - `data` must contain at least 4 f64 elements.
        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        unsafe fn codelet_simd_2_neon_f64(data: &mut [f64], _sign: i32) {
            use core::arch::aarch64::*;

            let ptr = data.as_mut_ptr();
            let a = vld1q_f64(ptr);
            let b = vld1q_f64(ptr.add(2));
            let sum  = vaddq_f64(a, b);
            let diff = vsubq_f64(a, b);
            vst1q_f64(ptr, sum);
            vst1q_f64(ptr.add(2), diff);
        }
    }
}

/// NEON size-4 radix-4 butterfly on f64 data.
pub(super) fn gen_neon_size_4() -> TokenStream {
    quote! {
        /// Size-4 radix-4 FFT using NEON intrinsics for f64 data.
        ///
        /// # Safety
        /// - Must be called on aarch64.
        /// - `data` must contain at least 8 f64 elements.
        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        unsafe fn codelet_simd_4_neon_f64(data: &mut [f64], sign: i32) {
            use core::arch::aarch64::*;

            let ptr = data.as_mut_ptr();

            let x0 = vld1q_f64(ptr);
            let x1 = vld1q_f64(ptr.add(2));
            let x2 = vld1q_f64(ptr.add(4));
            let x3 = vld1q_f64(ptr.add(6));

            // Stage 1
            let t0 = vaddq_f64(x0, x2);
            let t1 = vsubq_f64(x0, x2);
            let t2 = vaddq_f64(x1, x3);
            let t3 = vsubq_f64(x1, x3);

            // Rotate t3 by ±i using ext (swap) + negate
            let t3_swapped = vextq_f64(t3, t3, 1); // [im3, re3]
            let t3_rot = if sign < 0 {
                // Forward: [im, -re]
                let neg_mask = vld1q_f64([1.0_f64, -1.0_f64].as_ptr());
                vmulq_f64(t3_swapped, neg_mask)
            } else {
                // Inverse: [-im, re]
                let neg_mask = vld1q_f64([-1.0_f64, 1.0_f64].as_ptr());
                vmulq_f64(t3_swapped, neg_mask)
            };

            // Stage 2
            let out0 = vaddq_f64(t0, t2);
            let out1 = vaddq_f64(t1, t3_rot);
            let out2 = vsubq_f64(t0, t2);
            let out3 = vsubq_f64(t1, t3_rot);

            vst1q_f64(ptr,       out0);
            vst1q_f64(ptr.add(2), out1);
            vst1q_f64(ptr.add(4), out2);
            vst1q_f64(ptr.add(6), out3);
        }
    }
}

/// NEON size-8 radix-2 DIT butterfly on f64 data.
pub(super) fn gen_neon_size_8() -> TokenStream {
    quote! {
        /// Size-8 FFT using NEON intrinsics for f64 data (radix-2 DIT).
        ///
        /// # Safety
        /// - Must be called on aarch64.
        /// - `data` must contain at least 16 f64 elements.
        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[allow(clippy::too_many_lines)]
        unsafe fn codelet_simd_8_neon_f64(data: &mut [f64], sign: i32) {
            use core::arch::aarch64::*;

            let ptr = data.as_mut_ptr();
            let inv_sqrt2 = vdupq_n_f64(0.707_106_781_186_547_6_f64);
            let fwd = sign < 0;

            // Helper: rotate complex by ±i
            let rotate_pm_i = |v: float64x2_t, is_fwd: bool| -> float64x2_t {
                let swapped = vextq_f64(v, v, 1);
                if is_fwd {
                    let mask = vld1q_f64([1.0_f64, -1.0_f64].as_ptr());
                    vmulq_f64(swapped, mask)
                } else {
                    let mask = vld1q_f64([-1.0_f64, 1.0_f64].as_ptr());
                    vmulq_f64(swapped, mask)
                }
            };

            // Bit-reversal load
            let mut a = [vdupq_n_f64(0.0); 8];
            a[0] = vld1q_f64(ptr);            // x[0]
            a[1] = vld1q_f64(ptr.add(8));     // x[4]
            a[2] = vld1q_f64(ptr.add(4));     // x[2]
            a[3] = vld1q_f64(ptr.add(12));    // x[6]
            a[4] = vld1q_f64(ptr.add(2));     // x[1]
            a[5] = vld1q_f64(ptr.add(10));    // x[5]
            a[6] = vld1q_f64(ptr.add(6));     // x[3]
            a[7] = vld1q_f64(ptr.add(14));    // x[7]

            // Stage 1: span-1 butterflies
            for i in (0..8usize).step_by(2) {
                let t = a[i + 1];
                a[i + 1] = vsubq_f64(a[i], t);
                a[i]     = vaddq_f64(a[i], t);
            }

            // Stage 2: span-2 butterflies with W4 twiddles
            for group in (0..8usize).step_by(4) {
                let t = a[group + 2];
                a[group + 2] = vsubq_f64(a[group], t);
                a[group]     = vaddq_f64(a[group], t);

                let t = a[group + 3];
                let t_tw = rotate_pm_i(t, fwd);
                a[group + 3] = vsubq_f64(a[group + 1], t_tw);
                a[group + 1] = vaddq_f64(a[group + 1], t_tw);
            }

            // Stage 3: span-4 with W8 twiddles
            // k=0: trivial
            let t = a[4];
            a[4] = vsubq_f64(a[0], t);
            a[0] = vaddq_f64(a[0], t);

            // k=1: W8^1 twiddle using FMA
            {
                let v = a[5];
                let swapped = vextq_f64(v, v, 1);
                let t_tw = if fwd {
                    let sum = vaddq_f64(v, swapped);
                    let diff_sr = vsubq_f64(swapped, v);
                    // [sum[0], diff_sr[0]] = [re+im, im-re]
                    let combined = vzip1q_f64(sum, diff_sr);
                    vmulq_f64(combined, inv_sqrt2)
                } else {
                    let diff = vsubq_f64(v, swapped);
                    let sum = vaddq_f64(v, swapped);
                    // [diff[0], sum[0]] = [re-im, re+im]
                    // But we need [re-im, im+re]
                    let combined = vzip1q_f64(diff, sum);
                    // combined = [re-im, re+im] but we need [re-im, im+re]
                    // Note sum = [re+im, im+re], so sum[1] = im+re
                    let combined = vzip1q_f64(diff, vextq_f64(sum, sum, 1));
                    vmulq_f64(combined, inv_sqrt2)
                };
                a[5] = vsubq_f64(a[1], t_tw);
                a[1] = vaddq_f64(a[1], t_tw);
            }

            // k=2: ∓i
            {
                let t = a[6];
                let t_tw = rotate_pm_i(t, fwd);
                a[6] = vsubq_f64(a[2], t_tw);
                a[2] = vaddq_f64(a[2], t_tw);
            }

            // k=3: W8^3 twiddle
            {
                let v = a[7];
                let swapped = vextq_f64(v, v, 1);
                let t_tw = if fwd {
                    // [(-re+im)/√2, (-im-re)/√2]
                    let diff = vsubq_f64(swapped, v); // [im-re, re-im]
                    let neg_sum = vnegq_f64(vaddq_f64(v, swapped)); // [-(re+im), -(im+re)]
                    let combined = vzip1q_f64(diff, neg_sum);
                    vmulq_f64(combined, inv_sqrt2)
                } else {
                    // [(-re-im)/√2, (-im+re)/√2]
                    let neg_sum = vnegq_f64(vaddq_f64(v, swapped));
                    let diff = vsubq_f64(swapped, v);
                    // [-(re+im), re-im]  — need [neg_sum[0], diff[1]]
                    let combined = vzip1q_f64(neg_sum, vextq_f64(diff, diff, 1));
                    vmulq_f64(combined, inv_sqrt2)
                };
                a[7] = vsubq_f64(a[3], t_tw);
                a[3] = vaddq_f64(a[3], t_tw);
            }

            // Store in natural order
            for i in 0..8usize {
                vst1q_f64(ptr.add(i * 2), a[i]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NEON f32 emitters
// ---------------------------------------------------------------------------

/// NEON size-2 butterfly on f32 data.
///
/// Uses `float32x2_t` (2 lanes = 1 complex) for clean structural match with the f64 path.
pub(super) fn gen_neon_size_2_f32() -> TokenStream {
    quote! {
        /// Size-2 butterfly using NEON intrinsics for f32 data.
        ///
        /// # Safety
        /// - Must be called on aarch64 (NEON is always available).
        /// - `data` must contain at least 4 f32 elements.
        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        unsafe fn codelet_simd_2_neon_f32(data: &mut [f32], _sign: i32) {
            use core::arch::aarch64::*;

            let ptr = data.as_mut_ptr();
            // 1 complex = 2 f32 lanes in a float32x2_t
            let a = vld1_f32(ptr);
            let b = vld1_f32(ptr.add(2));
            let sum  = vadd_f32(a, b);
            let diff = vsub_f32(a, b);
            vst1_f32(ptr, sum);
            vst1_f32(ptr.add(2), diff);
        }
    }
}

/// NEON size-4 radix-4 butterfly on f32 data.
///
/// Uses `float32x2_t` — 1 complex per register, mirrors f64 structural layout.
pub(super) fn gen_neon_size_4_f32() -> TokenStream {
    quote! {
        /// Size-4 radix-4 FFT using NEON intrinsics for f32 data.
        ///
        /// # Safety
        /// - Must be called on aarch64.
        /// - `data` must contain at least 8 f32 elements.
        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        unsafe fn codelet_simd_4_neon_f32(data: &mut [f32], sign: i32) {
            use core::arch::aarch64::*;

            let ptr = data.as_mut_ptr();

            // 1 complex = 2 f32 lanes: [re, im]
            let x0 = vld1_f32(ptr);
            let x1 = vld1_f32(ptr.add(2));
            let x2 = vld1_f32(ptr.add(4));
            let x3 = vld1_f32(ptr.add(6));

            // Stage 1: pair-wise butterflies
            let t0 = vadd_f32(x0, x2);
            let t1 = vsub_f32(x0, x2);
            let t2 = vadd_f32(x1, x3);
            let t3 = vsub_f32(x1, x3);

            // Rotate t3 by ±i: swap lanes then negate one
            // ext(v, v, 1) swaps lanes: [re, im] -> [im, re]
            let t3_swapped = vext_f32(t3, t3, 1); // [im3, re3]
            let t3_rot = if sign < 0 {
                // Forward: [im, -re]
                let neg_mask = vld1_f32([1.0_f32, -1.0_f32].as_ptr());
                vmul_f32(t3_swapped, neg_mask)
            } else {
                // Inverse: [-im, re]
                let neg_mask = vld1_f32([-1.0_f32, 1.0_f32].as_ptr());
                vmul_f32(t3_swapped, neg_mask)
            };

            // Stage 2: final butterflies
            let out0 = vadd_f32(t0, t2);
            let out1 = vadd_f32(t1, t3_rot);
            let out2 = vsub_f32(t0, t2);
            let out3 = vsub_f32(t1, t3_rot);

            vst1_f32(ptr,       out0);
            vst1_f32(ptr.add(2), out1);
            vst1_f32(ptr.add(4), out2);
            vst1_f32(ptr.add(6), out3);
        }
    }
}

/// NEON size-8 radix-2 DIT butterfly on f32 data.
///
/// Uses `float32x2_t` — 1 complex per register, mirrors the f64 path exactly.
pub(super) fn gen_neon_size_8_f32() -> TokenStream {
    quote! {
        /// Size-8 FFT using NEON intrinsics for f32 data (radix-2 DIT).
        ///
        /// # Safety
        /// - Must be called on aarch64.
        /// - `data` must contain at least 16 f32 elements.
        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[allow(clippy::too_many_lines)]
        unsafe fn codelet_simd_8_neon_f32(data: &mut [f32], sign: i32) {
            use core::arch::aarch64::*;

            let ptr = data.as_mut_ptr();
            let inv_sqrt2 = vdup_n_f32(0.707_106_8_f32);
            let fwd = sign < 0;

            // Helper: rotate complex by ±i using float32x2_t
            let rotate_pm_i = |v: float32x2_t, is_fwd: bool| -> float32x2_t {
                let swapped = vext_f32(v, v, 1); // [im, re]
                if is_fwd {
                    let mask = vld1_f32([1.0_f32, -1.0_f32].as_ptr());
                    vmul_f32(swapped, mask)
                } else {
                    let mask = vld1_f32([-1.0_f32, 1.0_f32].as_ptr());
                    vmul_f32(swapped, mask)
                }
            };

            // Bit-reversal load (1 complex per register = 2 f32 lanes)
            let mut a = [vdup_n_f32(0.0); 8];
            a[0] = vld1_f32(ptr);            // x[0]
            a[1] = vld1_f32(ptr.add(8));     // x[4]
            a[2] = vld1_f32(ptr.add(4));     // x[2]
            a[3] = vld1_f32(ptr.add(12));    // x[6]
            a[4] = vld1_f32(ptr.add(2));     // x[1]
            a[5] = vld1_f32(ptr.add(10));    // x[5]
            a[6] = vld1_f32(ptr.add(6));     // x[3]
            a[7] = vld1_f32(ptr.add(14));    // x[7]

            // Stage 1: span-1 butterflies
            for i in (0..8usize).step_by(2) {
                let t = a[i + 1];
                a[i + 1] = vsub_f32(a[i], t);
                a[i]     = vadd_f32(a[i], t);
            }

            // Stage 2: span-2 butterflies with W4 twiddles
            for group in (0..8usize).step_by(4) {
                let t = a[group + 2];
                a[group + 2] = vsub_f32(a[group], t);
                a[group]     = vadd_f32(a[group], t);

                let t = a[group + 3];
                let t_tw = rotate_pm_i(t, fwd);
                a[group + 3] = vsub_f32(a[group + 1], t_tw);
                a[group + 1] = vadd_f32(a[group + 1], t_tw);
            }

            // Stage 3: span-4 with W8 twiddles
            // k=0: trivial
            let t = a[4];
            a[4] = vsub_f32(a[0], t);
            a[0] = vadd_f32(a[0], t);

            // k=1: W8^1 twiddle
            {
                let v = a[5];
                let swapped = vext_f32(v, v, 1); // [im, re]
                let t_tw = if fwd {
                    // [(re+im)/√2, (im-re)/√2]
                    let sum  = vadd_f32(v, swapped); // [re+im, im+re]
                    let diff = vsub_f32(swapped, v); // [im-re, re-im]
                    // need [sum[0], diff[0]] = [re+im, im-re]
                    let combined = vzip1_f32(sum, diff);
                    vmul_f32(combined, inv_sqrt2)
                } else {
                    // [(re-im)/√2, (im+re)/√2]
                    let diff = vsub_f32(v, swapped); // [re-im, im-re]
                    let sum  = vadd_f32(v, swapped); // [re+im, im+re]
                    // need [diff[0], sum[0]] = [re-im, re+im]
                    // but we need [re-im, im+re]: diff[0], swapped sum lane 1
                    let combined = vzip1_f32(diff, vext_f32(sum, sum, 1));
                    vmul_f32(combined, inv_sqrt2)
                };
                a[5] = vsub_f32(a[1], t_tw);
                a[1] = vadd_f32(a[1], t_tw);
            }

            // k=2: ∓i
            {
                let t = a[6];
                let t_tw = rotate_pm_i(t, fwd);
                a[6] = vsub_f32(a[2], t_tw);
                a[2] = vadd_f32(a[2], t_tw);
            }

            // k=3: W8^3 twiddle
            {
                let v = a[7];
                let swapped = vext_f32(v, v, 1); // [im, re]
                let t_tw = if fwd {
                    // [(-re+im)/√2, (-im-re)/√2]
                    let diff = vsub_f32(swapped, v); // [im-re, re-im]
                    let neg_sum = vneg_f32(vadd_f32(v, swapped)); // [-(re+im), -(im+re)]
                    let combined = vzip1_f32(diff, neg_sum);
                    vmul_f32(combined, inv_sqrt2)
                } else {
                    // [(-re-im)/√2, (-im+re)/√2]
                    let neg_sum = vneg_f32(vadd_f32(v, swapped)); // [-(re+im), -(im+re)]
                    let diff = vsub_f32(swapped, v); // [im-re, re-im]
                    // need [neg_sum[0], diff[1]] = [-(re+im), re-im]
                    let combined = vzip1_f32(neg_sum, vext_f32(diff, diff, 1));
                    vmul_f32(combined, inv_sqrt2)
                };
                a[7] = vsub_f32(a[3], t_tw);
                a[3] = vadd_f32(a[3], t_tw);
            }

            // Store in natural order
            for i in 0..8usize {
                vst1_f32(ptr.add(i * 2), a[i]);
            }
        }
    }
}
