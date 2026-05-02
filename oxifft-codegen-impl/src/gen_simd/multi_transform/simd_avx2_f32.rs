//! AVX2 f32 multi-transform SIMD emitters.
//!
//! Implements "true SIMD" multi-transform codelets for AVX2 f32 with V=8 transforms.
//!
//! # Data layout (`SoA` — Struct of Arrays)
//!
//! For V=8 transforms of size N, the `SoA` layout separates real and imaginary parts:
//! ```text
//! re[0..8]   = real  parts of element 0 from all 8 transforms
//! im[0..8]   = imag  parts of element 0 from all 8 transforms
//! re[8..16]  = real  parts of element 1 from all 8 transforms
//! im[8..16]  = imag  parts of element 1 from all 8 transforms
//! ...
//! ```
//!
//! A single `_mm256_loadu_ps(&re[0])` loads all 8 real parts into one YMM register.
//! Each SIMD operation thus processes all 8 transforms simultaneously.
//!
//! # Generated function signature
//!
//! ```rust,ignore
//! pub unsafe fn notw_2_v8_avx2_f32_soa(
//!     re_in: *const f32, im_in: *const f32,
//!     re_out: *mut f32, im_out: *mut f32,
//! )
//! ```

use proc_macro2::TokenStream;
use quote::quote;

/// Emit the AVX2 f32, V=8, size-2 `SoA` multi-transform SIMD function.
///
/// Processes 8 transforms of size 2 simultaneously using AVX2 `__m256` vectors.
/// Each YMM register holds element values from all 8 transforms.
///
/// # `SoA` layout
/// `re_in[0..8]`  = real parts of element 0 for transforms 0..8
/// `re_in[8..16]` = real parts of element 1 for transforms 0..8
/// etc.
pub(super) fn gen_avx2_f32_v8_size2_soa() -> TokenStream {
    quote! {
        /// AVX2 f32 V=8 size-2 multi-transform (SoA layout).
        ///
        /// Processes 8 DFTs of size 2 simultaneously using AVX2 `__m256` intrinsics.
        ///
        /// # Data layout
        /// `re_in[0..8]`  = real parts of element 0 for all 8 transforms.
        /// `re_in[8..16]` = real parts of element 1 for all 8 transforms.
        /// `im_in[0..8]` / `im_in[8..16]` — corresponding imaginary parts.
        /// Output uses the same SoA layout.
        ///
        /// # Safety
        /// - `re_in` and `im_in` must be valid for at least 16 `f32` reads.
        /// - `re_out` and `im_out` must be valid for at least 16 `f32` writes.
        /// - AVX2 must be available on the executing CPU.
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        #[target_feature(enable = "avx2")]
        pub unsafe fn notw_2_v8_avx2_f32_soa(
            re_in: *const f32,
            im_in: *const f32,
            re_out: *mut f32,
            im_out: *mut f32,
        ) {
            use core::arch::x86_64::*;

            // Load element 0 real/imag from all 8 transforms (8 f32 each = 256 bits)
            let re0 = _mm256_loadu_ps(re_in);          // [re0_t0..re0_t7]
            let im0 = _mm256_loadu_ps(im_in);          // [im0_t0..im0_t7]
            // Load element 1 real/imag from all 8 transforms
            let re1 = _mm256_loadu_ps(re_in.add(8));   // [re1_t0..re1_t7]
            let im1 = _mm256_loadu_ps(im_in.add(8));   // [im1_t0..im1_t7]

            // Radix-2 butterfly applied simultaneously to all 8 transforms:
            // X[0] = x[0] + x[1],  X[1] = x[0] - x[1]
            _mm256_storeu_ps(re_out,        _mm256_add_ps(re0, re1)); // X[0].re
            _mm256_storeu_ps(im_out,        _mm256_add_ps(im0, im1)); // X[0].im
            _mm256_storeu_ps(re_out.add(8),  _mm256_sub_ps(re0, re1)); // X[1].re
            _mm256_storeu_ps(im_out.add(8),  _mm256_sub_ps(im0, im1)); // X[1].im
        }
    }
}

/// Emit the AVX2 f32, V=8, size-4 `SoA` multi-transform SIMD function.
///
/// Processes 8 transforms of size 4 simultaneously using AVX2 `__m256` vectors.
///
/// # `SoA` layout
/// `re_in[k*8..(k+1)*8]` = real parts of element `k` for transforms 0..8
/// `im_in[k*8..(k+1)*8]` = imag parts of element `k` for transforms 0..8
///
/// # Algorithm (radix-4 DIF butterfly)
///
/// Stage 1 — pair butterflies:
/// ```text
/// t0 = x[0] + x[2],  t1 = x[0] - x[2]
/// t2 = x[1] + x[3],  t3 = x[1] - x[3]
/// ```
/// Forward twiddle on t3: `t3_rot = (t3.im, -t3.re)` (multiply by -i)
///
/// Stage 2 — final butterfly:
/// ```text
/// X[0] = t0 + t2,  X[1] = t1 + t3_rot
/// X[2] = t0 - t2,  X[3] = t1 - t3_rot
/// ```
pub(super) fn gen_avx2_f32_v8_size4_soa() -> TokenStream {
    quote! {
        /// AVX2 f32 V=8 size-4 multi-transform (SoA layout).
        ///
        /// Processes 8 DFTs of size 4 simultaneously using AVX2 `__m256` intrinsics.
        ///
        /// # Data layout
        /// `re_in[k*8..(k+1)*8]` = real parts of element `k` for transforms 0..8.
        /// `im_in[k*8..(k+1)*8]` = imag parts of element `k` for transforms 0..8.
        /// Output uses the same SoA layout.
        ///
        /// # Safety
        /// - `re_in` and `im_in` must be valid for at least 32 `f32` reads.
        /// - `re_out` and `im_out` must be valid for at least 32 `f32` writes.
        /// - AVX2 must be available on the executing CPU.
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        #[target_feature(enable = "avx2")]
        pub unsafe fn notw_4_v8_avx2_f32_soa(
            re_in: *const f32,
            im_in: *const f32,
            re_out: *mut f32,
            im_out: *mut f32,
        ) {
            use core::arch::x86_64::*;

            // Load element 0..3 real/imag parts for all 8 transforms (8 f32 = 256 bits each)
            let re0 = _mm256_loadu_ps(re_in);
            let im0 = _mm256_loadu_ps(im_in);
            let re1 = _mm256_loadu_ps(re_in.add(8));
            let im1 = _mm256_loadu_ps(im_in.add(8));
            let re2 = _mm256_loadu_ps(re_in.add(16));
            let im2 = _mm256_loadu_ps(im_in.add(16));
            let re3 = _mm256_loadu_ps(re_in.add(24));
            let im3 = _mm256_loadu_ps(im_in.add(24));

            // Stage 1: pair butterflies (0 ± 2) and (1 ± 3) for all 8 transforms
            let t0_re = _mm256_add_ps(re0, re2);
            let t0_im = _mm256_add_ps(im0, im2);
            let t1_re = _mm256_sub_ps(re0, re2);
            let t1_im = _mm256_sub_ps(im0, im2);
            let t2_re = _mm256_add_ps(re1, re3);
            let t2_im = _mm256_add_ps(im1, im3);
            let t3_re = _mm256_sub_ps(re1, re3);
            let t3_im = _mm256_sub_ps(im1, im3);

            // Forward twiddle on t3: multiply by -i → (t3.im, -t3.re)
            let neg_mask = _mm256_set1_ps(-0.0_f32);
            let t3rot_re = t3_im;                               // t3.im unchanged
            let t3rot_im = _mm256_xor_ps(t3_re, neg_mask);     // -t3.re

            // Stage 2: final cross-butterflies for all 8 transforms simultaneously
            // X[0] = t0 + t2
            _mm256_storeu_ps(re_out,        _mm256_add_ps(t0_re, t2_re));
            _mm256_storeu_ps(im_out,        _mm256_add_ps(t0_im, t2_im));
            // X[1] = t1 + t3_rot
            _mm256_storeu_ps(re_out.add(8),  _mm256_add_ps(t1_re, t3rot_re));
            _mm256_storeu_ps(im_out.add(8),  _mm256_add_ps(t1_im, t3rot_im));
            // X[2] = t0 - t2
            _mm256_storeu_ps(re_out.add(16), _mm256_sub_ps(t0_re, t2_re));
            _mm256_storeu_ps(im_out.add(16), _mm256_sub_ps(t0_im, t2_im));
            // X[3] = t1 - t3_rot
            _mm256_storeu_ps(re_out.add(24), _mm256_sub_ps(t1_re, t3rot_re));
            _mm256_storeu_ps(im_out.add(24), _mm256_sub_ps(t1_im, t3rot_im));
        }
    }
}

/// Emit the AVX2 f32, V=8, size-8 `SoA` multi-transform SIMD function.
///
/// Processes 8 transforms of size 8 simultaneously using AVX2 `__m256` vectors.
///
/// # `SoA` layout
/// `re_in[k*8..(k+1)*8]` = real parts of element `k` for transforms 0..8
/// `im_in[k*8..(k+1)*8]` = imag parts of element `k` for transforms 0..8
///
/// # Algorithm (radix-8 DIF butterfly, three stages)
///
/// Stage 1 — halve: upper ± lower half for k=0..3
/// Stage 2 — length-4 DIF on each half with W8 twiddles
/// Stage 3 — length-2 butterfly on each pair
#[allow(clippy::too_many_lines)]
pub(super) fn gen_avx2_f32_v8_size8_soa() -> TokenStream {
    quote! {
        /// AVX2 f32 V=8 size-8 multi-transform (SoA layout).
        ///
        /// Processes 8 DFTs of size 8 simultaneously using AVX2 `__m256` intrinsics.
        ///
        /// # Data layout
        /// `re_in[k*8..(k+1)*8]`  = real parts of element `k` for transforms 0..8.
        /// `im_in[k*8..(k+1)*8]`  = imag parts of element `k` for transforms 0..8.
        /// Output uses the same SoA layout.
        ///
        /// # Safety
        /// - `re_in` and `im_in` must be valid for at least 64 `f32` reads.
        /// - `re_out` and `im_out` must be valid for at least 64 `f32` writes.
        /// - AVX2 must be available on the executing CPU.
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        #[target_feature(enable = "avx2")]
        #[allow(clippy::too_many_lines)]
        pub unsafe fn notw_8_v8_avx2_f32_soa(
            re_in: *const f32,
            im_in: *const f32,
            re_out: *mut f32,
            im_out: *mut f32,
        ) {
            use core::arch::x86_64::*;

            // 1/√2 broadcast to all 8 lanes
            let inv_sqrt2 = _mm256_set1_ps(0.707_106_77_f32);
            let neg_mask  = _mm256_set1_ps(-0.0_f32);

            // Helper closure: negate all 8 lanes (xor with -0.0)
            // (closures in unsafe fn are fine; we expand inline below)

            // Load elements 0..7 for all 8 transforms simultaneously
            // Stride between elements in SoA is 8 f32 = one YMM
            let re0 = _mm256_loadu_ps(re_in);
            let im0 = _mm256_loadu_ps(im_in);
            let re1 = _mm256_loadu_ps(re_in.add(8));
            let im1 = _mm256_loadu_ps(im_in.add(8));
            let re2 = _mm256_loadu_ps(re_in.add(16));
            let im2 = _mm256_loadu_ps(im_in.add(16));
            let re3 = _mm256_loadu_ps(re_in.add(24));
            let im3 = _mm256_loadu_ps(im_in.add(24));
            let re4 = _mm256_loadu_ps(re_in.add(32));
            let im4 = _mm256_loadu_ps(im_in.add(32));
            let re5 = _mm256_loadu_ps(re_in.add(40));
            let im5 = _mm256_loadu_ps(im_in.add(40));
            let re6 = _mm256_loadu_ps(re_in.add(48));
            let im6 = _mm256_loadu_ps(im_in.add(48));
            let re7 = _mm256_loadu_ps(re_in.add(56));
            let im7 = _mm256_loadu_ps(im_in.add(56));

            // ── Stage 1: upper ± lower half ──────────────────────────────────────
            // a_k = x[k] + x[k+4], b_k = x[k] - x[k+4]  for k = 0..4
            let a0r = _mm256_add_ps(re0, re4);
            let a0i = _mm256_add_ps(im0, im4);
            let a1r = _mm256_add_ps(re1, re5);
            let a1i = _mm256_add_ps(im1, im5);
            let a2r = _mm256_add_ps(re2, re6);
            let a2i = _mm256_add_ps(im2, im6);
            let a3r = _mm256_add_ps(re3, re7);
            let a3i = _mm256_add_ps(im3, im7);

            let b0r = _mm256_sub_ps(re0, re4);
            let b0i = _mm256_sub_ps(im0, im4);
            let b1r = _mm256_sub_ps(re1, re5);
            let b1i = _mm256_sub_ps(im1, im5);
            let b2r = _mm256_sub_ps(re2, re6);
            let b2i = _mm256_sub_ps(im2, im6);
            let b3r = _mm256_sub_ps(re3, re7);
            let b3i = _mm256_sub_ps(im3, im7);

            // ── Apply W8 twiddles to b1, b2, b3 (forward DFT) ───────────────────
            // b1 *= W8^1 = (1-i)/√2  → ((b1r+b1i)/√2, (-b1r+b1i)/√2)
            let b1tr = _mm256_mul_ps(_mm256_add_ps(b1r, b1i), inv_sqrt2);
            let b1ti = _mm256_mul_ps(_mm256_sub_ps(b1i, b1r), inv_sqrt2);

            // b2 *= W8^2 = -i  → (b2.im, -b2.re)
            let b2tr = b2i;
            let b2ti = _mm256_xor_ps(b2r, neg_mask); // -b2r

            // b3 *= W8^3 = (-1-i)/√2  → ((-b3r+b3i)/√2, (-b3r-b3i)/√2)
            let b3tr = _mm256_mul_ps(_mm256_sub_ps(b3i, b3r), inv_sqrt2);
            let neg_b3r_plus_b3i_neg = _mm256_xor_ps(_mm256_add_ps(b3r, b3i), neg_mask);
            let b3ti = _mm256_mul_ps(neg_b3r_plus_b3i_neg, inv_sqrt2); // -(b3r+b3i)/√2

            // ── Stage 2: length-4 DIF on a-group ────────────────────────────────
            // c0 = a0 + a2, c2 = a0 - a2
            let c0r = _mm256_add_ps(a0r, a2r);
            let c0i = _mm256_add_ps(a0i, a2i);
            let c2r = _mm256_sub_ps(a0r, a2r);
            let c2i = _mm256_sub_ps(a0i, a2i);
            // c1 = a1 + a3
            let c1r = _mm256_add_ps(a1r, a3r);
            let c1i = _mm256_add_ps(a1i, a3i);
            // c3 = (a1 - a3) * (-i)  → (a1-a3).im, -(a1-a3).re
            let d3r = _mm256_sub_ps(a1r, a3r);
            let d3i = _mm256_sub_ps(a1i, a3i);
            let c3r = d3i;
            let c3i = _mm256_xor_ps(d3r, neg_mask); // -d3r

            // ── Stage 2: length-4 DIF on b-group ────────────────────────────────
            let e0r = _mm256_add_ps(b0r, b2tr);
            let e0i = _mm256_add_ps(b0i, b2ti);
            let e2r = _mm256_sub_ps(b0r, b2tr);
            let e2i = _mm256_sub_ps(b0i, b2ti);
            let e1r = _mm256_add_ps(b1tr, b3tr);
            let e1i = _mm256_add_ps(b1ti, b3ti);
            // e3 = (b1t - b3t) * (-i)
            let f3r = _mm256_sub_ps(b1tr, b3tr);
            let f3i = _mm256_sub_ps(b1ti, b3ti);
            let e3r = f3i;
            let e3i = _mm256_xor_ps(f3r, neg_mask); // -f3r

            // ── Stage 3: length-2 butterfly on each pair ─────────────────────────
            // Output element ordering: 0,4,2,6 from a-group, 1,5,3,7 from b-group
            _mm256_storeu_ps(re_out,        _mm256_add_ps(c0r, c1r)); // X[0]
            _mm256_storeu_ps(im_out,        _mm256_add_ps(c0i, c1i));
            _mm256_storeu_ps(re_out.add(8),  _mm256_add_ps(e0r, e1r)); // X[1]
            _mm256_storeu_ps(im_out.add(8),  _mm256_add_ps(e0i, e1i));
            _mm256_storeu_ps(re_out.add(16), _mm256_add_ps(c2r, c3r)); // X[2]
            _mm256_storeu_ps(im_out.add(16), _mm256_add_ps(c2i, c3i));
            _mm256_storeu_ps(re_out.add(24), _mm256_add_ps(e2r, e3r)); // X[3]
            _mm256_storeu_ps(im_out.add(24), _mm256_add_ps(e2i, e3i));
            _mm256_storeu_ps(re_out.add(32), _mm256_sub_ps(c0r, c1r)); // X[4]
            _mm256_storeu_ps(im_out.add(32), _mm256_sub_ps(c0i, c1i));
            _mm256_storeu_ps(re_out.add(40), _mm256_sub_ps(e0r, e1r)); // X[5]
            _mm256_storeu_ps(im_out.add(40), _mm256_sub_ps(e0i, e1i));
            _mm256_storeu_ps(re_out.add(48), _mm256_sub_ps(c2r, c3r)); // X[6]
            _mm256_storeu_ps(im_out.add(48), _mm256_sub_ps(c2i, c3i));
            _mm256_storeu_ps(re_out.add(56), _mm256_sub_ps(e2r, e3r)); // X[7]
            _mm256_storeu_ps(im_out.add(56), _mm256_sub_ps(e2i, e3i));
        }
    }
}
