//! SSE2 f32 multi-transform SIMD emitters.
//!
//! Implements "true SIMD" multi-transform codelets for SSE2 f32 with V=4 transforms.
//!
//! # Data layout (`SoA` — Struct of Arrays)
//!
//! For V=4 transforms of size N, the `SoA` layout separates real and imaginary parts:
//! ```text
//! re[0..4]  = real  parts of element 0 from all 4 transforms
//! im[0..4]  = imag  parts of element 0 from all 4 transforms
//! re[4..8]  = real  parts of element 1 from all 4 transforms
//! im[4..8]  = imag  parts of element 1 from all 4 transforms
//! ...
//! ```
//!
//! A single `_mm_loadu_ps(&re[0])` loads all 4 real parts as one XMM register.
//! This means each SIMD operation processes all V transforms simultaneously.
//!
//! # Generated function signature
//!
//! ```rust,ignore
//! pub unsafe fn notw_2_v4_sse2_f32_soa(
//!     re_in: *const f32, im_in: *const f32,
//!     re_out: *mut f32, im_out: *mut f32,
//! )
//! ```
//!
//! The non-`SoA` (`AoS`) wrapper retains the existing interleaved signature.

use proc_macro2::TokenStream;
use quote::quote;

/// Emit the SSE2 f32, V=4, size-2 `SoA` multi-transform SIMD function.
///
/// Processes 4 transforms of size 2 simultaneously using SSE2 `__m128` vectors.
/// Each SIMD register holds element values from all 4 transforms.
///
/// # `SoA` layout
/// Input:  `re_in[0..4]` = re of element 0 for transforms 0..4,
///         `re_in[4..8]` = re of element 1 for transforms 0..4, etc.
/// Output: same layout.
///
/// # Algorithm (radix-2 butterfly applied to all V lanes simultaneously)
/// ```text
/// re_out[0] = re_in[0] + re_in[4]  (element 0 of output = in[0] + in[1])
/// im_out[0] = im_in[0] + im_in[4]
/// re_out[4] = re_in[0] - re_in[4]  (element 1 of output = in[0] - in[1])
/// im_out[4] = im_in[0] - im_in[4]
/// ```
///
/// All 4 transforms are processed in one pass via SIMD vector operations.
pub(super) fn gen_sse2_f32_v4_size2_soa() -> TokenStream {
    quote! {
        /// SSE2 f32 V=4 size-2 multi-transform (SoA layout).
        ///
        /// Processes 4 DFTs of size 2 simultaneously using SSE2 intrinsics.
        ///
        /// # Data layout
        /// `re_in[0..4]` = real parts of element 0 for all 4 transforms.
        /// `re_in[4..8]` = real parts of element 1 for all 4 transforms.
        /// `im_in[0..4]` / `im_in[4..8]` — corresponding imaginary parts.
        ///
        /// # Safety
        /// - `re_in` and `im_in` must be valid for at least 8 `f32` reads.
        /// - `re_out` and `im_out` must be valid for at least 8 `f32` writes.
        /// - SSE2 must be available (guaranteed on all `x86_64` targets).
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        #[target_feature(enable = "sse2")]
        pub unsafe fn notw_2_v4_sse2_f32_soa(
            re_in: *const f32,
            im_in: *const f32,
            re_out: *mut f32,
            im_out: *mut f32,
        ) {
            use core::arch::x86_64::*;

            // Load element 0 real/imag from all 4 transforms (4 f32 each)
            let re0 = _mm_loadu_ps(re_in);          // [re0_t0, re0_t1, re0_t2, re0_t3]
            let im0 = _mm_loadu_ps(im_in);          // [im0_t0, im0_t1, im0_t2, im0_t3]
            // Load element 1 real/imag from all 4 transforms
            let re1 = _mm_loadu_ps(re_in.add(4));   // [re1_t0, re1_t1, re1_t2, re1_t3]
            let im1 = _mm_loadu_ps(im_in.add(4));   // [im1_t0, im1_t1, im1_t2, im1_t3]

            // Radix-2 butterfly: out[0] = in[0] + in[1], out[1] = in[0] - in[1]
            _mm_storeu_ps(re_out,       _mm_add_ps(re0, re1)); // X[0].re for all 4 transforms
            _mm_storeu_ps(im_out,       _mm_add_ps(im0, im1)); // X[0].im for all 4 transforms
            _mm_storeu_ps(re_out.add(4), _mm_sub_ps(re0, re1)); // X[1].re for all 4 transforms
            _mm_storeu_ps(im_out.add(4), _mm_sub_ps(im0, im1)); // X[1].im for all 4 transforms
        }
    }
}

/// Emit the SSE2 f32, V=4, size-4 `SoA` multi-transform SIMD function.
///
/// Processes 4 transforms of size 4 simultaneously using SSE2 `__m128` vectors.
///
/// # `SoA` layout
/// Input: `re_in[k*4..(k+1)*4]` = real parts of element `k` for transforms 0..4.
///        `im_in[k*4..(k+1)*4]` = imag parts of element `k` for transforms 0..4.
///
/// # Algorithm (radix-4 DIF butterfly applied to all V lanes simultaneously)
///
/// Stage 1 (pair butterfly):
/// ```text
/// t0 = x[0] + x[2],  t1 = x[0] - x[2]
/// t2 = x[1] + x[3],  t3 = x[1] - x[3]
/// ```
///
/// Forward twiddle on t3 (multiply by -i): `t3_rot = (t3.im, -t3.re)`
///
/// Stage 2 (final butterfly):
/// ```text
/// X[0] = t0 + t2
/// X[1] = t1 + t3_rot
/// X[2] = t0 - t2
/// X[3] = t1 - t3_rot
/// ```
///
/// All 4 transforms execute the same butterfly operations via SIMD.
pub(super) fn gen_sse2_f32_v4_size4_soa() -> TokenStream {
    quote! {
        /// SSE2 f32 V=4 size-4 multi-transform (SoA layout).
        ///
        /// Processes 4 DFTs of size 4 simultaneously using SSE2 intrinsics.
        ///
        /// # Data layout
        /// `re_in[k*4..(k+1)*4]` = real parts of element `k` for transforms 0..4.
        /// `im_in[k*4..(k+1)*4]` = imag parts of element `k` for transforms 0..4.
        /// Output uses the same layout.
        ///
        /// # Safety
        /// - `re_in` and `im_in` must be valid for at least 16 `f32` reads.
        /// - `re_out` and `im_out` must be valid for at least 16 `f32` writes.
        /// - SSE2 must be available (guaranteed on all `x86_64` targets).
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        #[target_feature(enable = "sse2")]
        pub unsafe fn notw_4_v4_sse2_f32_soa(
            re_in: *const f32,
            im_in: *const f32,
            re_out: *mut f32,
            im_out: *mut f32,
        ) {
            use core::arch::x86_64::*;

            // Load element 0..3 real/imag parts for all 4 transforms
            // re_k = [re_k_t0, re_k_t1, re_k_t2, re_k_t3]
            let re0 = _mm_loadu_ps(re_in);
            let im0 = _mm_loadu_ps(im_in);
            let re1 = _mm_loadu_ps(re_in.add(4));
            let im1 = _mm_loadu_ps(im_in.add(4));
            let re2 = _mm_loadu_ps(re_in.add(8));
            let im2 = _mm_loadu_ps(im_in.add(8));
            let re3 = _mm_loadu_ps(re_in.add(12));
            let im3 = _mm_loadu_ps(im_in.add(12));

            // Stage 1: pair butterflies (0 ± 2) and (1 ± 3)
            let t0_re = _mm_add_ps(re0, re2); // x[0].re + x[2].re  for all 4 transforms
            let t0_im = _mm_add_ps(im0, im2);
            let t1_re = _mm_sub_ps(re0, re2); // x[0].re - x[2].re
            let t1_im = _mm_sub_ps(im0, im2);
            let t2_re = _mm_add_ps(re1, re3); // x[1].re + x[3].re
            let t2_im = _mm_add_ps(im1, im3);
            let t3_re = _mm_sub_ps(re1, re3); // x[1].re - x[3].re
            let t3_im = _mm_sub_ps(im1, im3);

            // Forward twiddle on t3: multiply by -i → (t3.im, -t3.re)
            // Negate t3_re to form t3rot_im = -t3_re
            let neg_mask = _mm_set1_ps(-0.0_f32);
            let t3rot_re = t3_im;                          // t3.im unchanged
            let t3rot_im = _mm_xor_ps(t3_re, neg_mask);   // -t3.re

            // Stage 2: final cross-butterflies
            // X[0] = t0 + t2
            _mm_storeu_ps(re_out,       _mm_add_ps(t0_re, t2_re));
            _mm_storeu_ps(im_out,       _mm_add_ps(t0_im, t2_im));
            // X[1] = t1 + t3_rot
            _mm_storeu_ps(re_out.add(4),  _mm_add_ps(t1_re, t3rot_re));
            _mm_storeu_ps(im_out.add(4),  _mm_add_ps(t1_im, t3rot_im));
            // X[2] = t0 - t2
            _mm_storeu_ps(re_out.add(8),  _mm_sub_ps(t0_re, t2_re));
            _mm_storeu_ps(im_out.add(8),  _mm_sub_ps(t0_im, t2_im));
            // X[3] = t1 - t3_rot
            _mm_storeu_ps(re_out.add(12), _mm_sub_ps(t1_re, t3rot_re));
            _mm_storeu_ps(im_out.add(12), _mm_sub_ps(t1_im, t3rot_im));
        }
    }
}
