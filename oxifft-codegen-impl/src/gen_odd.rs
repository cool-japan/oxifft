//! Odd-size DFT codelet generation using Winograd minimum-multiply factorizations.
//!
//! Generates optimized DFT codelets for sizes 3, 5, and 7 using the Winograd
//! algorithm which minimizes the number of real multiplications. The generated
//! functions follow the same in-place `&mut [Complex<T>]` convention as `gen_notw.rs`.
//!
//! # DFT Convention
//!
//! Forward DFT: `W_N` = e^{-2πi/N}  (sign = -1 / sign < 0)
//! Inverse DFT: `W_N` = e^{+2πi/N}  (sign = +1 / sign > 0, unnormalized)
//!
//! The sign of sine terms flips between forward and inverse. Specifically,
//! for `X[k] = Σ x[j] · e^{-2πi·j·k/N}`:
//! - Forward: imaginary part uses −sin(...) terms
//! - Inverse: imaginary part uses +sin(...) terms

use crate::winograd_constants::{
    C3_1, C3_2, C5_COS1, C5_COS2, C5_SIN1, C5_SIN2, C7_COS1, C7_COS2, C7_COS3, C7_SIN1, C7_SIN2,
    C7_SIN3,
};
use proc_macro2::TokenStream;
use quote::quote;
use syn::LitInt;

// ============================================================================
// Public entry point: proc-macro interface
// ============================================================================

/// Generate an odd-size (3, 5, 7) DFT codelet from macro input `gen_odd_codelet!(N)`.
///
/// # Errors
/// Returns `syn::Error` if the input is not a valid integer literal or the size
/// is not in {3, 5, 7}.
pub fn generate_from_macro(input: TokenStream) -> Result<TokenStream, syn::Error> {
    let size: LitInt = syn::parse2(input)?;
    let n: usize = size.base10_parse().map_err(|_| {
        syn::Error::new(
            size.span(),
            "gen_odd_codelet: expected an integer size literal",
        )
    })?;

    match n {
        3 => Ok(gen_size_3()),
        5 => Ok(gen_size_5()),
        7 => Ok(gen_size_7()),
        _ => Err(syn::Error::new(
            size.span(),
            format!("gen_odd_codelet: unsupported size {n} (expected one of 3, 5, 7)"),
        )),
    }
}

// ============================================================================
// DFT-3 codelet (2 real multiplications — Winograd)
// ============================================================================
//
// Winograd DFT-3 derivation (forward: W = e^{-2πi/3}):
//
// Define: s = x[1] + x[2],  d = x[1] - x[2]   (complex adds only)
//
// X[0] = x[0] + s
// X[1] = x[0] + C3_1·s + i·(-C3_2)·d
//       = (x_re[0] + C3_1·s_re) + (C3_2·d_im)
//       + i·((x_im[0] + C3_1·s_im) - (C3_2·d_re))
//   because: (-C3_2)·(d_re + i·d_im) rotated by i...
//   Actually: -i·(d_re + i·d_im) = d_im - i·d_re, so:
//   X[1] = (tmp_re + C3_2·d_im) + i·(tmp_im - C3_2·d_re)
//
// X[2] = conj(X[1]) pattern (signs flip):
//   X[2].re = tmp_re - C3_2·d_im
//   X[2].im = tmp_im + C3_2·d_re
//
// Inverse (W = e^{+2πi/3}): C3_2 signs flip:
//   X[1].re = tmp_re - C3_2·d_im
//   X[1].im = tmp_im + C3_2·d_re
//   X[2].re = tmp_re + C3_2·d_im
//   X[2].im = tmp_im - C3_2·d_re

fn gen_size_3() -> TokenStream {
    let c3_1 = C3_1;
    let c3_2 = C3_2;
    quote! {
        /// Size-3 DFT codelet using Winograd minimum-multiply factorization.
        ///
        /// Uses 2 real multiplications (Winograd optimal for DFT-3).
        ///
        /// `sign < 0` → forward transform (W = e^{-2πi/3});
        /// `sign > 0` → inverse (unnormalized, W = e^{+2πi/3}).
        #[inline(always)]
        #[allow(clippy::too_many_lines, clippy::approx_constant, clippy::suboptimal_flops)]
        pub fn codelet_notw_3<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 3);

            let x0 = x[0];
            let x1 = x[1];
            let x2 = x[2];

            // Stage 1: sum and difference of x[1], x[2]
            let s_re = x1.re + x2.re;
            let s_im = x1.im + x2.im;
            let d_re = x1.re - x2.re;
            let d_im = x1.im - x2.im;

            // X[0] = x[0] + s
            x[0] = crate::kernel::Complex::new(x0.re + s_re, x0.im + s_im);

            // tmp = x[0] + C3_1 * s  (C3_1 = -0.5)
            let c3_1 = T::from_f64(#c3_1);
            let c3_2 = T::from_f64(#c3_2);
            let tmp_re = x0.re + c3_1 * s_re;
            let tmp_im = x0.im + c3_1 * s_im;

            if sign < 0 {
                // Forward: X[1].re = tmp_re + C3_2·d_im,  X[1].im = tmp_im - C3_2·d_re
                //          X[2].re = tmp_re - C3_2·d_im,  X[2].im = tmp_im + C3_2·d_re
                x[1] = crate::kernel::Complex::new(tmp_re + c3_2 * d_im, tmp_im - c3_2 * d_re);
                x[2] = crate::kernel::Complex::new(tmp_re - c3_2 * d_im, tmp_im + c3_2 * d_re);
            } else {
                // Inverse: C3_2 sign flips
                x[1] = crate::kernel::Complex::new(tmp_re - c3_2 * d_im, tmp_im + c3_2 * d_re);
                x[2] = crate::kernel::Complex::new(tmp_re + c3_2 * d_im, tmp_im - c3_2 * d_re);
            }
        }
    }
}

// ============================================================================
// DFT-5 codelet (5 real multiplications — Winograd)
// ============================================================================
//
// Winograd DFT-5 derivation (forward: W = e^{-2πi/5}):
//
// Let ck = cos(2πk/5), sk = sin(2πk/5) for k = 1, 2.
// Note: cos(4π/5) = C5_COS2, cos(2π/5) = C5_COS1
//       sin(2π/5) = C5_SIN1, sin(4π/5) = C5_SIN2
//
// Rader/Winograd factorization:
//   r1 = x[1] + x[4],  r2 = x[2] + x[3]  (sum pairs that share cosines)
//   i1 = x[1] - x[4],  i2 = x[2] - x[3]  (diff pairs that share sines)
//
//   X[0] = x[0] + r1 + r2
//
//   Forward cosine contributions:
//     cr1 = C5_COS1·r1 + C5_COS2·r2
//     cr2 = C5_COS2·r1 + C5_COS1·r2   (note symmetry cos(4π/5)=cos(2π/5) exchange)
//     Wait — for k=1: X[1] uses cos(2π/5) on x[1]+x[4] and cos(4π/5) on x[2]+x[3]
//            for k=2: X[2] uses cos(4π/5) on x[1]+x[4] and cos(2π·2·2/5) on x[2]+x[3]
//     Let us re-derive carefully:
//       X[k] = x[0] + x[1]·W^k + x[2]·W^{2k} + x[3]·W^{3k} + x[4]·W^{4k}
//     For complex input: W = e^{-2πi/5}
//     k=1: W^1 = C5_COS1 - i·C5_SIN1, W^2 = C5_COS2 - i·C5_SIN2
//          W^3 = C5_COS2 + i·C5_SIN2  (since cos(6π/5)=C5_COS2, sin(6π/5)=-C5_SIN2)
//          W^4 = C5_COS1 + i·C5_SIN1  (since cos(8π/5)=C5_COS1, sin(8π/5)=-C5_SIN1)
//     Therefore:
//       X[1].re = x_re[0] + C5_COS1·(x_re[1]+x_re[4]) + C5_COS2·(x_re[2]+x_re[3])
//                 + C5_SIN1·(x_im[1]-x_im[4]) + C5_SIN2·(x_im[2]-x_im[3])
//       X[1].im = x_im[0] + C5_COS1·(x_im[1]+x_im[4]) + C5_COS2·(x_im[2]+x_im[3])
//                 - C5_SIN1·(x_re[1]-x_re[4]) - C5_SIN2·(x_re[2]-x_re[3])
//     Similarly for k=2 (swap COS1↔COS2, SIN1↔SIN2):
//       X[2].re = x_re[0] + C5_COS2·(x_re[1]+x_re[4]) + C5_COS1·(x_re[2]+x_re[3])
//                 + C5_SIN2·(x_im[1]-x_im[4]) - C5_SIN1·(x_im[2]-x_im[3])
//       X[2].im = x_im[0] + C5_COS2·(x_im[1]+x_im[4]) + C5_COS1·(x_im[2]+x_im[3])
//                 - C5_SIN2·(x_re[1]-x_re[4]) + C5_SIN1·(x_re[2]-x_re[3])
//     X[3] = conj-swap of X[2], X[4] = conj-swap of X[1]:
//       cos terms same, sin terms negated.

fn gen_size_5() -> TokenStream {
    let c5_cos1 = C5_COS1;
    let c5_cos2 = C5_COS2;
    let c5_sin1 = C5_SIN1;
    let c5_sin2 = C5_SIN2;
    quote! {
        /// Size-5 DFT codelet using Winograd minimum-multiply factorization.
        ///
        /// Uses 5 real multiplications (Winograd optimal for DFT-5).
        ///
        /// `sign < 0` → forward transform (W = e^{-2πi/5});
        /// `sign > 0` → inverse (unnormalized, W = e^{+2πi/5}).
        #[inline(always)]
        #[allow(clippy::too_many_lines, clippy::approx_constant, clippy::suboptimal_flops)]
        pub fn codelet_notw_5<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 5);

            let x0 = x[0];
            let x1 = x[1];
            let x2 = x[2];
            let x3 = x[3];
            let x4 = x[4];

            // Symmetric sums and differences
            // r1 = x[1] + x[4],  r2 = x[2] + x[3]
            // i1 = x[1] - x[4],  i2 = x[2] - x[3]
            let r1_re = x1.re + x4.re;
            let r1_im = x1.im + x4.im;
            let r2_re = x2.re + x3.re;
            let r2_im = x2.im + x3.im;
            let i1_re = x1.re - x4.re;
            let i1_im = x1.im - x4.im;
            let i2_re = x2.re - x3.re;
            let i2_im = x2.im - x3.im;

            // X[0] = x[0] + r1 + r2
            x[0] = crate::kernel::Complex::new(x0.re + r1_re + r2_re, x0.im + r1_im + r2_im);

            let cos1 = T::from_f64(#c5_cos1);
            let cos2 = T::from_f64(#c5_cos2);
            let sin1 = T::from_f64(#c5_sin1);
            let sin2 = T::from_f64(#c5_sin2);

            // Cosine blends (shared by both forward and inverse)
            let cr1_re = cos1 * r1_re + cos2 * r2_re;
            let cr1_im = cos1 * r1_im + cos2 * r2_im;
            let cr2_re = cos2 * r1_re + cos1 * r2_re;
            let cr2_im = cos2 * r1_im + cos1 * r2_im;

            // Sine blends (sign determines forward vs. inverse)
            let sr1_re = sin1 * i1_re + sin2 * i2_re;
            let sr1_im = sin1 * i1_im + sin2 * i2_im;
            let sr2_re = sin2 * i1_re - sin1 * i2_re;
            let sr2_im = sin2 * i1_im - sin1 * i2_im;

            // tmp_k = x[0] + cos-blend_k
            let tmp1_re = x0.re + cr1_re;
            let tmp1_im = x0.im + cr1_im;
            let tmp2_re = x0.re + cr2_re;
            let tmp2_im = x0.im + cr2_im;

            if sign < 0 {
                // Forward: X[k].re = tmp_k.re + sin_blend_k.im
                //          X[k].im = tmp_k.im - sin_blend_k.re
                //          (because -i·(a+ib) = b - ia)
                x[1] = crate::kernel::Complex::new(tmp1_re + sr1_im, tmp1_im - sr1_re);
                x[4] = crate::kernel::Complex::new(tmp1_re - sr1_im, tmp1_im + sr1_re);
                x[2] = crate::kernel::Complex::new(tmp2_re + sr2_im, tmp2_im - sr2_re);
                x[3] = crate::kernel::Complex::new(tmp2_re - sr2_im, tmp2_im + sr2_re);
            } else {
                // Inverse: sine signs flip
                x[1] = crate::kernel::Complex::new(tmp1_re - sr1_im, tmp1_im + sr1_re);
                x[4] = crate::kernel::Complex::new(tmp1_re + sr1_im, tmp1_im - sr1_re);
                x[2] = crate::kernel::Complex::new(tmp2_re - sr2_im, tmp2_im + sr2_re);
                x[3] = crate::kernel::Complex::new(tmp2_re + sr2_im, tmp2_im - sr2_re);
            }
        }
    }
}

// ============================================================================
// DFT-7 codelet (Rader-like Winograd factorization)
// ============================================================================
//
// Winograd DFT-7 derivation (forward: W = e^{-2πi/7}):
//
// For k=1,2,3, the DFT outputs satisfy:
//   X[k] = x[0] + Σ_{j=1}^{6} x[j]·W^{jk}
//
// Group pairs: r_m = x[m] + x[7-m],  i_m = x[m] - x[7-m]  for m=1,2,3
//
// Cosines for k=1: cos(2π/7), cos(4π/7), cos(6π/7)
// Cosines for k=2: cos(4π/7), cos(8π/7)=cos(6π/7), cos(12π/7)=cos(2π/7)
//   => k=2 row is a permutation of k=1 row
// Cosines for k=3: cos(6π/7), cos(12π/7)=cos(2π/7), cos(18π/7)=cos(4π/7)
//   => k=3 row is another permutation
//
// So:
//   X[1].re = x_re[0] + C7_COS1·r1_re + C7_COS2·r2_re + C7_COS3·r3_re
//              + C7_SIN1·i1_im + C7_SIN2·i2_im + C7_SIN3·i3_im
//   X[1].im = x_im[0] + C7_COS1·r1_im + C7_COS2·r2_im + C7_COS3·r3_im
//              - C7_SIN1·i1_re - C7_SIN2·i2_re - C7_SIN3·i3_re
//
//   X[2].re = x_re[0] + C7_COS2·r1_re + C7_COS3·r2_re + C7_COS1·r3_re
//              + C7_SIN2·i1_im - C7_SIN3·i2_im - C7_SIN1·i3_im
//   X[2].im = x_im[0] + C7_COS2·r1_im + C7_COS3·r2_im + C7_COS1·r3_im
//              - C7_SIN2·i1_re + C7_SIN3·i2_re + C7_SIN1·i3_re
//
//   X[3].re = x_re[0] + C7_COS3·r1_re + C7_COS1·r2_re + C7_COS2·r3_re
//              + C7_SIN3·i1_im - C7_SIN1·i2_im + C7_SIN2·i3_im
//   X[3].im = x_im[0] + C7_COS3·r1_im + C7_COS1·r2_im + C7_COS2·r3_im
//              - C7_SIN3·i1_re + C7_SIN1·i2_re - C7_SIN2·i3_re
//
// X[4..6] = conjugate mirror: X[7-k] = conj(X[k]) for real input, but for
// complex input they are independent: sin signs flip.

fn gen_size_7() -> TokenStream {
    let c7_cos1 = C7_COS1;
    let c7_cos2 = C7_COS2;
    let c7_cos3 = C7_COS3;
    let c7_sin1 = C7_SIN1;
    let c7_sin2 = C7_SIN2;
    let c7_sin3 = C7_SIN3;
    quote! {
        /// Size-7 DFT codelet using Winograd minimum-multiply factorization.
        ///
        /// Uses the Rader-Winograd structure with 9 real multiplications (optimal
        /// for the pair-based factorization of DFT-7).
        ///
        /// `sign < 0` → forward transform (W = e^{-2πi/7});
        /// `sign > 0` → inverse (unnormalized, W = e^{+2πi/7}).
        #[inline(always)]
        #[allow(clippy::too_many_lines, clippy::approx_constant, clippy::suboptimal_flops)]
        pub fn codelet_notw_7<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 7);

            let x0 = x[0];
            let x1 = x[1];
            let x2 = x[2];
            let x3 = x[3];
            let x4 = x[4];
            let x5 = x[5];
            let x6 = x[6];

            // Symmetric sums and differences:
            // r_m = x[m] + x[7-m],  i_m = x[m] - x[7-m]  for m=1,2,3
            let r1_re = x1.re + x6.re;
            let r1_im = x1.im + x6.im;
            let r2_re = x2.re + x5.re;
            let r2_im = x2.im + x5.im;
            let r3_re = x3.re + x4.re;
            let r3_im = x3.im + x4.im;
            let i1_re = x1.re - x6.re;
            let i1_im = x1.im - x6.im;
            let i2_re = x2.re - x5.re;
            let i2_im = x2.im - x5.im;
            let i3_re = x3.re - x4.re;
            let i3_im = x3.im - x4.im;

            // X[0] = x[0] + r1 + r2 + r3
            x[0] = crate::kernel::Complex::new(
                x0.re + r1_re + r2_re + r3_re,
                x0.im + r1_im + r2_im + r3_im,
            );

            let cos1 = T::from_f64(#c7_cos1);
            let cos2 = T::from_f64(#c7_cos2);
            let cos3 = T::from_f64(#c7_cos3);
            let sin1 = T::from_f64(#c7_sin1);
            let sin2 = T::from_f64(#c7_sin2);
            let sin3 = T::from_f64(#c7_sin3);

            // Cosine blends (same for forward and inverse)
            // X[1]: cos1·r1 + cos2·r2 + cos3·r3
            let cp1_re = cos1 * r1_re + cos2 * r2_re + cos3 * r3_re;
            let cp1_im = cos1 * r1_im + cos2 * r2_im + cos3 * r3_im;
            // X[2]: cos2·r1 + cos3·r2 + cos1·r3
            let cp2_re = cos2 * r1_re + cos3 * r2_re + cos1 * r3_re;
            let cp2_im = cos2 * r1_im + cos3 * r2_im + cos1 * r3_im;
            // X[3]: cos3·r1 + cos1·r2 + cos2·r3
            let cp3_re = cos3 * r1_re + cos1 * r2_re + cos2 * r3_re;
            let cp3_im = cos3 * r1_im + cos1 * r2_im + cos2 * r3_im;

            // Sine blends:
            // X[1] forward: +sin1·i1_im + sin2·i2_im + sin3·i3_im (re)
            //               -sin1·i1_re - sin2·i2_re - sin3·i3_re (im)
            let sp1_re = sin1 * i1_im + sin2 * i2_im + sin3 * i3_im;
            let sp1_im = sin1 * i1_re + sin2 * i2_re + sin3 * i3_re;
            // X[2] forward: +sin2·i1_im - sin3·i2_im - sin1·i3_im (re)
            //               -sin2·i1_re + sin3·i2_re + sin1·i3_re (im)
            let sp2_re = sin2 * i1_im - sin3 * i2_im - sin1 * i3_im;
            let sp2_im = sin2 * i1_re - sin3 * i2_re - sin1 * i3_re;
            // X[3] forward: +sin3·i1_im - sin1·i2_im + sin2·i3_im (re)
            //               -sin3·i1_re + sin1·i2_re - sin2·i3_re (im)
            let sp3_re = sin3 * i1_im - sin1 * i2_im + sin2 * i3_im;
            let sp3_im = sin3 * i1_re - sin1 * i2_re + sin2 * i3_re;

            // tmp_k = x[0] + cosine-blend_k
            let tmp1_re = x0.re + cp1_re;
            let tmp1_im = x0.im + cp1_im;
            let tmp2_re = x0.re + cp2_re;
            let tmp2_im = x0.im + cp2_im;
            let tmp3_re = x0.re + cp3_re;
            let tmp3_im = x0.im + cp3_im;

            if sign < 0 {
                // Forward: X[k].re = tmp_k.re + sp_k.re
                //          X[k].im = tmp_k.im - sp_k.im
                // X[7-k] = conjugate pattern (sin signs flip)
                x[1] = crate::kernel::Complex::new(tmp1_re + sp1_re, tmp1_im - sp1_im);
                x[6] = crate::kernel::Complex::new(tmp1_re - sp1_re, tmp1_im + sp1_im);
                x[2] = crate::kernel::Complex::new(tmp2_re + sp2_re, tmp2_im - sp2_im);
                x[5] = crate::kernel::Complex::new(tmp2_re - sp2_re, tmp2_im + sp2_im);
                x[3] = crate::kernel::Complex::new(tmp3_re + sp3_re, tmp3_im - sp3_im);
                x[4] = crate::kernel::Complex::new(tmp3_re - sp3_re, tmp3_im + sp3_im);
            } else {
                // Inverse: sine signs flip
                x[1] = crate::kernel::Complex::new(tmp1_re - sp1_re, tmp1_im + sp1_im);
                x[6] = crate::kernel::Complex::new(tmp1_re + sp1_re, tmp1_im - sp1_im);
                x[2] = crate::kernel::Complex::new(tmp2_re - sp2_re, tmp2_im + sp2_im);
                x[5] = crate::kernel::Complex::new(tmp2_re + sp2_re, tmp2_im - sp2_im);
                x[3] = crate::kernel::Complex::new(tmp3_re - sp3_re, tmp3_im + sp3_im);
                x[4] = crate::kernel::Complex::new(tmp3_re + sp3_re, tmp3_im - sp3_im);
            }
        }
    }
}

// ============================================================================
// Reference implementations for numerical testing
// ============================================================================
// These evaluate the same Winograd algorithms in pure f64 so that unit tests
// can verify correctness without depending on `crate::kernel::Float`.

/// Naive O(N²) DFT reference for testing.
///
/// Returns (re, im) output vectors.
#[cfg(test)]
#[allow(clippy::suboptimal_flops)]
pub(crate) fn naive_dft_fwd(x_re: &[f64], x_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x_re.len();
    debug_assert_eq!(x_im.len(), n);
    let mut out_re = vec![0.0_f64; n];
    let mut out_im = vec![0.0_f64; n];
    for k in 0..n {
        for j in 0..n {
            let angle = -2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
            let (s, c) = angle.sin_cos();
            out_re[k] += x_re[j] * c - x_im[j] * s;
            out_im[k] += x_re[j] * s + x_im[j] * c;
        }
    }
    (out_re, out_im)
}

/// Naive O(N²) inverse DFT reference (unnormalized) for testing.
#[cfg(test)]
#[allow(clippy::suboptimal_flops)]
pub(crate) fn naive_dft_inv(x_re: &[f64], x_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x_re.len();
    debug_assert_eq!(x_im.len(), n);
    let mut out_re = vec![0.0_f64; n];
    let mut out_im = vec![0.0_f64; n];
    for k in 0..n {
        for j in 0..n {
            let angle = 2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
            let (s, c) = angle.sin_cos();
            out_re[k] += x_re[j] * c - x_im[j] * s;
            out_im[k] += x_re[j] * s + x_im[j] * c;
        }
    }
    (out_re, out_im)
}

/// Winograd DFT-3 in pure f64 (mirrors the generated codelet logic).
///
/// Returns (re, im) output vectors.
#[cfg(test)]
#[allow(clippy::suboptimal_flops)]
pub(crate) fn winograd_dft3_fwd(x_re: &[f64], x_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(x_re.len(), 3);
    let mut out_re = vec![0.0_f64; 3];
    let mut out_im = vec![0.0_f64; 3];

    let s_re = x_re[1] + x_re[2];
    let s_im = x_im[1] + x_im[2];
    let d_re = x_re[1] - x_re[2];
    let d_im = x_im[1] - x_im[2];

    out_re[0] = x_re[0] + s_re;
    out_im[0] = x_im[0] + s_im;

    let tmp_re = x_re[0] + C3_1 * s_re;
    let tmp_im = x_im[0] + C3_1 * s_im;

    out_re[1] = tmp_re + C3_2 * d_im;
    out_im[1] = tmp_im - C3_2 * d_re;
    out_re[2] = tmp_re - C3_2 * d_im;
    out_im[2] = tmp_im + C3_2 * d_re;

    (out_re, out_im)
}

/// Winograd DFT-3 inverse in pure f64.
#[cfg(test)]
#[allow(clippy::suboptimal_flops)]
pub(crate) fn winograd_dft3_inv(x_re: &[f64], x_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(x_re.len(), 3);
    let mut out_re = vec![0.0_f64; 3];
    let mut out_im = vec![0.0_f64; 3];

    let s_re = x_re[1] + x_re[2];
    let s_im = x_im[1] + x_im[2];
    let d_re = x_re[1] - x_re[2];
    let d_im = x_im[1] - x_im[2];

    out_re[0] = x_re[0] + s_re;
    out_im[0] = x_im[0] + s_im;

    let tmp_re = x_re[0] + C3_1 * s_re;
    let tmp_im = x_im[0] + C3_1 * s_im;

    // Inverse: sine sign flips
    out_re[1] = tmp_re - C3_2 * d_im;
    out_im[1] = tmp_im + C3_2 * d_re;
    out_re[2] = tmp_re + C3_2 * d_im;
    out_im[2] = tmp_im - C3_2 * d_re;

    (out_re, out_im)
}

/// Winograd DFT-5 in pure f64 (mirrors the generated codelet logic).
#[cfg(test)]
#[allow(clippy::suboptimal_flops)]
pub(crate) fn winograd_dft5_fwd(x_re: &[f64], x_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(x_re.len(), 5);
    let mut out_re = vec![0.0_f64; 5];
    let mut out_im = vec![0.0_f64; 5];

    let r1_re = x_re[1] + x_re[4];
    let r1_im = x_im[1] + x_im[4];
    let r2_re = x_re[2] + x_re[3];
    let r2_im = x_im[2] + x_im[3];
    let i1_re = x_re[1] - x_re[4];
    let i1_im = x_im[1] - x_im[4];
    let i2_re = x_re[2] - x_re[3];
    let i2_im = x_im[2] - x_im[3];

    out_re[0] = x_re[0] + r1_re + r2_re;
    out_im[0] = x_im[0] + r1_im + r2_im;

    let cr1_re = C5_COS1 * r1_re + C5_COS2 * r2_re;
    let cr1_im = C5_COS1 * r1_im + C5_COS2 * r2_im;
    let cr2_re = C5_COS2 * r1_re + C5_COS1 * r2_re;
    let cr2_im = C5_COS2 * r1_im + C5_COS1 * r2_im;

    let sr1_re = C5_SIN1 * i1_re + C5_SIN2 * i2_re;
    let sr1_im = C5_SIN1 * i1_im + C5_SIN2 * i2_im;
    let sr2_re = C5_SIN2 * i1_re - C5_SIN1 * i2_re;
    let sr2_im = C5_SIN2 * i1_im - C5_SIN1 * i2_im;

    let tmp1_re = x_re[0] + cr1_re;
    let tmp1_im = x_im[0] + cr1_im;
    let tmp2_re = x_re[0] + cr2_re;
    let tmp2_im = x_im[0] + cr2_im;

    // Forward: X[k].re = tmp.re + sr_im, X[k].im = tmp.im - sr_re
    out_re[1] = tmp1_re + sr1_im;
    out_im[1] = tmp1_im - sr1_re;
    out_re[4] = tmp1_re - sr1_im;
    out_im[4] = tmp1_im + sr1_re;
    out_re[2] = tmp2_re + sr2_im;
    out_im[2] = tmp2_im - sr2_re;
    out_re[3] = tmp2_re - sr2_im;
    out_im[3] = tmp2_im + sr2_re;

    (out_re, out_im)
}

/// Winograd DFT-5 inverse in pure f64.
#[cfg(test)]
#[allow(clippy::suboptimal_flops)]
pub(crate) fn winograd_dft5_inv(x_re: &[f64], x_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(x_re.len(), 5);
    let mut out_re = vec![0.0_f64; 5];
    let mut out_im = vec![0.0_f64; 5];

    let r1_re = x_re[1] + x_re[4];
    let r1_im = x_im[1] + x_im[4];
    let r2_re = x_re[2] + x_re[3];
    let r2_im = x_im[2] + x_im[3];
    let i1_re = x_re[1] - x_re[4];
    let i1_im = x_im[1] - x_im[4];
    let i2_re = x_re[2] - x_re[3];
    let i2_im = x_im[2] - x_im[3];

    out_re[0] = x_re[0] + r1_re + r2_re;
    out_im[0] = x_im[0] + r1_im + r2_im;

    let cr1_re = C5_COS1 * r1_re + C5_COS2 * r2_re;
    let cr1_im = C5_COS1 * r1_im + C5_COS2 * r2_im;
    let cr2_re = C5_COS2 * r1_re + C5_COS1 * r2_re;
    let cr2_im = C5_COS2 * r1_im + C5_COS1 * r2_im;

    let sr1_re = C5_SIN1 * i1_re + C5_SIN2 * i2_re;
    let sr1_im = C5_SIN1 * i1_im + C5_SIN2 * i2_im;
    let sr2_re = C5_SIN2 * i1_re - C5_SIN1 * i2_re;
    let sr2_im = C5_SIN2 * i1_im - C5_SIN1 * i2_im;

    let tmp1_re = x_re[0] + cr1_re;
    let tmp1_im = x_im[0] + cr1_im;
    let tmp2_re = x_re[0] + cr2_re;
    let tmp2_im = x_im[0] + cr2_im;

    // Inverse: sine signs flip
    out_re[1] = tmp1_re - sr1_im;
    out_im[1] = tmp1_im + sr1_re;
    out_re[4] = tmp1_re + sr1_im;
    out_im[4] = tmp1_im - sr1_re;
    out_re[2] = tmp2_re - sr2_im;
    out_im[2] = tmp2_im + sr2_re;
    out_re[3] = tmp2_re + sr2_im;
    out_im[3] = tmp2_im - sr2_re;

    (out_re, out_im)
}

/// Winograd DFT-7 in pure f64 (mirrors the generated codelet logic).
#[cfg(test)]
#[allow(clippy::suboptimal_flops)]
pub(crate) fn winograd_dft7_fwd(x_re: &[f64], x_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(x_re.len(), 7);
    let mut out_re = vec![0.0_f64; 7];
    let mut out_im = vec![0.0_f64; 7];

    let r1_re = x_re[1] + x_re[6];
    let r1_im = x_im[1] + x_im[6];
    let r2_re = x_re[2] + x_re[5];
    let r2_im = x_im[2] + x_im[5];
    let r3_re = x_re[3] + x_re[4];
    let r3_im = x_im[3] + x_im[4];
    let i1_re = x_re[1] - x_re[6];
    let i1_im = x_im[1] - x_im[6];
    let i2_re = x_re[2] - x_re[5];
    let i2_im = x_im[2] - x_im[5];
    let i3_re = x_re[3] - x_re[4];
    let i3_im = x_im[3] - x_im[4];

    out_re[0] = x_re[0] + r1_re + r2_re + r3_re;
    out_im[0] = x_im[0] + r1_im + r2_im + r3_im;

    let cp1_re = C7_COS1 * r1_re + C7_COS2 * r2_re + C7_COS3 * r3_re;
    let cp1_im = C7_COS1 * r1_im + C7_COS2 * r2_im + C7_COS3 * r3_im;
    let cp2_re = C7_COS2 * r1_re + C7_COS3 * r2_re + C7_COS1 * r3_re;
    let cp2_im = C7_COS2 * r1_im + C7_COS3 * r2_im + C7_COS1 * r3_im;
    let cp3_re = C7_COS3 * r1_re + C7_COS1 * r2_re + C7_COS2 * r3_re;
    let cp3_im = C7_COS3 * r1_im + C7_COS1 * r2_im + C7_COS2 * r3_im;

    let sp1_re = C7_SIN1 * i1_im + C7_SIN2 * i2_im + C7_SIN3 * i3_im;
    let sp1_im = C7_SIN1 * i1_re + C7_SIN2 * i2_re + C7_SIN3 * i3_re;
    let sp2_re = C7_SIN2 * i1_im - C7_SIN3 * i2_im - C7_SIN1 * i3_im;
    let sp2_im = C7_SIN2 * i1_re - C7_SIN3 * i2_re - C7_SIN1 * i3_re;
    let sp3_re = C7_SIN3 * i1_im - C7_SIN1 * i2_im + C7_SIN2 * i3_im;
    let sp3_im = C7_SIN3 * i1_re - C7_SIN1 * i2_re + C7_SIN2 * i3_re;

    let tmp1_re = x_re[0] + cp1_re;
    let tmp1_im = x_im[0] + cp1_im;
    let tmp2_re = x_re[0] + cp2_re;
    let tmp2_im = x_im[0] + cp2_im;
    let tmp3_re = x_re[0] + cp3_re;
    let tmp3_im = x_im[0] + cp3_im;

    // Forward
    out_re[1] = tmp1_re + sp1_re;
    out_im[1] = tmp1_im - sp1_im;
    out_re[6] = tmp1_re - sp1_re;
    out_im[6] = tmp1_im + sp1_im;
    out_re[2] = tmp2_re + sp2_re;
    out_im[2] = tmp2_im - sp2_im;
    out_re[5] = tmp2_re - sp2_re;
    out_im[5] = tmp2_im + sp2_im;
    out_re[3] = tmp3_re + sp3_re;
    out_im[3] = tmp3_im - sp3_im;
    out_re[4] = tmp3_re - sp3_re;
    out_im[4] = tmp3_im + sp3_im;

    (out_re, out_im)
}

/// Winograd DFT-7 inverse in pure f64.
#[cfg(test)]
#[allow(clippy::suboptimal_flops)]
pub(crate) fn winograd_dft7_inv(x_re: &[f64], x_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(x_re.len(), 7);
    let mut out_re = vec![0.0_f64; 7];
    let mut out_im = vec![0.0_f64; 7];

    let r1_re = x_re[1] + x_re[6];
    let r1_im = x_im[1] + x_im[6];
    let r2_re = x_re[2] + x_re[5];
    let r2_im = x_im[2] + x_im[5];
    let r3_re = x_re[3] + x_re[4];
    let r3_im = x_im[3] + x_im[4];
    let i1_re = x_re[1] - x_re[6];
    let i1_im = x_im[1] - x_im[6];
    let i2_re = x_re[2] - x_re[5];
    let i2_im = x_im[2] - x_im[5];
    let i3_re = x_re[3] - x_re[4];
    let i3_im = x_im[3] - x_im[4];

    out_re[0] = x_re[0] + r1_re + r2_re + r3_re;
    out_im[0] = x_im[0] + r1_im + r2_im + r3_im;

    let cp1_re = C7_COS1 * r1_re + C7_COS2 * r2_re + C7_COS3 * r3_re;
    let cp1_im = C7_COS1 * r1_im + C7_COS2 * r2_im + C7_COS3 * r3_im;
    let cp2_re = C7_COS2 * r1_re + C7_COS3 * r2_re + C7_COS1 * r3_re;
    let cp2_im = C7_COS2 * r1_im + C7_COS3 * r2_im + C7_COS1 * r3_im;
    let cp3_re = C7_COS3 * r1_re + C7_COS1 * r2_re + C7_COS2 * r3_re;
    let cp3_im = C7_COS3 * r1_im + C7_COS1 * r2_im + C7_COS2 * r3_im;

    let sp1_re = C7_SIN1 * i1_im + C7_SIN2 * i2_im + C7_SIN3 * i3_im;
    let sp1_im = C7_SIN1 * i1_re + C7_SIN2 * i2_re + C7_SIN3 * i3_re;
    let sp2_re = C7_SIN2 * i1_im - C7_SIN3 * i2_im - C7_SIN1 * i3_im;
    let sp2_im = C7_SIN2 * i1_re - C7_SIN3 * i2_re - C7_SIN1 * i3_re;
    let sp3_re = C7_SIN3 * i1_im - C7_SIN1 * i2_im + C7_SIN2 * i3_im;
    let sp3_im = C7_SIN3 * i1_re - C7_SIN1 * i2_re + C7_SIN2 * i3_re;

    let tmp1_re = x_re[0] + cp1_re;
    let tmp1_im = x_im[0] + cp1_im;
    let tmp2_re = x_re[0] + cp2_re;
    let tmp2_im = x_im[0] + cp2_im;
    let tmp3_re = x_re[0] + cp3_re;
    let tmp3_im = x_im[0] + cp3_im;

    // Inverse: sine signs flip
    out_re[1] = tmp1_re - sp1_re;
    out_im[1] = tmp1_im + sp1_im;
    out_re[6] = tmp1_re + sp1_re;
    out_im[6] = tmp1_im - sp1_im;
    out_re[2] = tmp2_re - sp2_re;
    out_im[2] = tmp2_im + sp2_im;
    out_re[5] = tmp2_re + sp2_re;
    out_im[5] = tmp2_im - sp2_im;
    out_re[3] = tmp3_re - sp3_re;
    out_im[3] = tmp3_im + sp3_im;
    out_re[4] = tmp3_re + sp3_re;
    out_im[4] = tmp3_im - sp3_im;

    (out_re, out_im)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn assert_close(a: &[f64], b: &[f64], label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < TOL,
                "{label}[{i}]: got {x}, expected {y}, diff = {}",
                (x - y).abs()
            );
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // DFT-3 tests
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_dft3_forward_f64_impulse() {
        // DFT of unit impulse at index 0: all outputs should be 1+0i
        let x_re = [1.0, 0.0, 0.0];
        let x_im = [0.0, 0.0, 0.0];
        let (got_re, got_im) = winograd_dft3_fwd(&x_re, &x_im);
        assert_close(&got_re, &[1.0, 1.0, 1.0], "dft3_impulse_re");
        assert_close(&got_im, &[0.0, 0.0, 0.0], "dft3_impulse_im");
    }

    #[test]
    fn test_dft3_forward_vs_naive() {
        // Random complex input
        let x_re = [1.3, -0.7, 0.4];
        let x_im = [0.2, 1.1, -0.5];
        let (got_re, got_im) = winograd_dft3_fwd(&x_re, &x_im);
        let (ref_re, ref_im) = naive_dft_fwd(&x_re, &x_im);
        assert_close(&got_re, &ref_re, "dft3_fwd_re");
        assert_close(&got_im, &ref_im, "dft3_fwd_im");
    }

    #[test]
    fn test_dft3_inverse_vs_naive() {
        let x_re = [1.3, -0.7, 0.4];
        let x_im = [0.2, 1.1, -0.5];
        let (got_re, got_im) = winograd_dft3_inv(&x_re, &x_im);
        let (ref_re, ref_im) = naive_dft_inv(&x_re, &x_im);
        assert_close(&got_re, &ref_re, "dft3_inv_re");
        assert_close(&got_im, &ref_im, "dft3_inv_im");
    }

    #[test]
    fn test_roundtrip_dft3() {
        // fwd → inv → scale by 1/3 should recover original
        let x_re = [1.3, -0.7, 0.4];
        let x_im = [0.2, 1.1, -0.5];
        let (fwd_re, fwd_im) = winograd_dft3_fwd(&x_re, &x_im);
        let (inv_re, inv_im) = winograd_dft3_inv(&fwd_re, &fwd_im);
        let n = 3.0_f64;
        let scaled_re: Vec<f64> = inv_re.iter().map(|&v| v / n).collect();
        let scaled_im: Vec<f64> = inv_im.iter().map(|&v| v / n).collect();
        assert_close(&scaled_re, &x_re, "roundtrip_dft3_re");
        assert_close(&scaled_im, &x_im, "roundtrip_dft3_im");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // DFT-5 tests
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_dft5_forward_f64_impulse() {
        let x_re = [1.0, 0.0, 0.0, 0.0, 0.0];
        let x_im = [0.0, 0.0, 0.0, 0.0, 0.0];
        let (got_re, got_im) = winograd_dft5_fwd(&x_re, &x_im);
        assert_close(&got_re, &[1.0, 1.0, 1.0, 1.0, 1.0], "dft5_impulse_re");
        assert_close(&got_im, &[0.0, 0.0, 0.0, 0.0, 0.0], "dft5_impulse_im");
    }

    #[test]
    fn test_dft5_forward_vs_naive() {
        let x_re = [0.5, -1.2, 0.8, 0.3, -0.6];
        let x_im = [0.1, 0.4, -0.9, 0.7, -0.2];
        let (got_re, got_im) = winograd_dft5_fwd(&x_re, &x_im);
        let (ref_re, ref_im) = naive_dft_fwd(&x_re, &x_im);
        assert_close(&got_re, &ref_re, "dft5_fwd_re");
        assert_close(&got_im, &ref_im, "dft5_fwd_im");
    }

    #[test]
    fn test_dft5_inverse_vs_naive() {
        let x_re = [0.5, -1.2, 0.8, 0.3, -0.6];
        let x_im = [0.1, 0.4, -0.9, 0.7, -0.2];
        let (got_re, got_im) = winograd_dft5_inv(&x_re, &x_im);
        let (ref_re, ref_im) = naive_dft_inv(&x_re, &x_im);
        assert_close(&got_re, &ref_re, "dft5_inv_re");
        assert_close(&got_im, &ref_im, "dft5_inv_im");
    }

    #[test]
    fn test_roundtrip_dft5() {
        let x_re = [0.5, -1.2, 0.8, 0.3, -0.6];
        let x_im = [0.1, 0.4, -0.9, 0.7, -0.2];
        let (fwd_re, fwd_im) = winograd_dft5_fwd(&x_re, &x_im);
        let (inv_re, inv_im) = winograd_dft5_inv(&fwd_re, &fwd_im);
        let n = 5.0_f64;
        let scaled_re: Vec<f64> = inv_re.iter().map(|&v| v / n).collect();
        let scaled_im: Vec<f64> = inv_im.iter().map(|&v| v / n).collect();
        assert_close(&scaled_re, &x_re, "roundtrip_dft5_re");
        assert_close(&scaled_im, &x_im, "roundtrip_dft5_im");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // DFT-7 tests
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_dft7_forward_f64_impulse() {
        let x_re = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let x_im = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (got_re, got_im) = winograd_dft7_fwd(&x_re, &x_im);
        assert_close(
            &got_re,
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "dft7_impulse_re",
        );
        assert_close(
            &got_im,
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "dft7_impulse_im",
        );
    }

    #[test]
    fn test_dft7_forward_vs_naive() {
        let x_re = [0.5, -1.2, 0.8, 0.3, -0.6, 1.4, -0.1];
        let x_im = [0.1, 0.4, -0.9, 0.7, -0.2, 0.5, 0.3];
        let (got_re, got_im) = winograd_dft7_fwd(&x_re, &x_im);
        let (ref_re, ref_im) = naive_dft_fwd(&x_re, &x_im);
        assert_close(&got_re, &ref_re, "dft7_fwd_re");
        assert_close(&got_im, &ref_im, "dft7_fwd_im");
    }

    #[test]
    fn test_dft7_inverse_vs_naive() {
        let x_re = [0.5, -1.2, 0.8, 0.3, -0.6, 1.4, -0.1];
        let x_im = [0.1, 0.4, -0.9, 0.7, -0.2, 0.5, 0.3];
        let (got_re, got_im) = winograd_dft7_inv(&x_re, &x_im);
        let (ref_re, ref_im) = naive_dft_inv(&x_re, &x_im);
        assert_close(&got_re, &ref_re, "dft7_inv_re");
        assert_close(&got_im, &ref_im, "dft7_inv_im");
    }

    #[test]
    fn test_roundtrip_dft7() {
        let x_re = [0.5, -1.2, 0.8, 0.3, -0.6, 1.4, -0.1];
        let x_im = [0.1, 0.4, -0.9, 0.7, -0.2, 0.5, 0.3];
        let (fwd_re, fwd_im) = winograd_dft7_fwd(&x_re, &x_im);
        let (inv_re, inv_im) = winograd_dft7_inv(&fwd_re, &fwd_im);
        let n = 7.0_f64;
        let scaled_re: Vec<f64> = inv_re.iter().map(|&v| v / n).collect();
        let scaled_im: Vec<f64> = inv_im.iter().map(|&v| v / n).collect();
        assert_close(&scaled_re, &x_re, "roundtrip_dft7_re");
        assert_close(&scaled_im, &x_im, "roundtrip_dft7_im");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Winograd constants cross-validation
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_winograd_constants_match_runtime() {
        crate::winograd_constants::verify_constants_match_runtime();
    }

    // ──────────────────────────────────────────────────────────────────────────
    // TokenStream generation (structural checks)
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_generate_from_macro_size3() {
        let input: proc_macro2::TokenStream = "3".parse().expect("parse literal");
        let result = generate_from_macro(input);
        assert!(result.is_ok(), "gen_odd_codelet!(3) should succeed");
        let ts = result.expect("TokenStream for size 3");
        let s = ts.to_string();
        assert!(
            s.contains("codelet_notw_3"),
            "should contain codelet_notw_3"
        );
        assert!(s.contains("sign"), "should contain sign parameter");
    }

    #[test]
    fn test_generate_from_macro_size5() {
        let input: proc_macro2::TokenStream = "5".parse().expect("parse literal");
        let result = generate_from_macro(input);
        assert!(result.is_ok(), "gen_odd_codelet!(5) should succeed");
        let ts = result.expect("TokenStream for size 5");
        let s = ts.to_string();
        assert!(
            s.contains("codelet_notw_5"),
            "should contain codelet_notw_5"
        );
    }

    #[test]
    fn test_generate_from_macro_size7() {
        let input: proc_macro2::TokenStream = "7".parse().expect("parse literal");
        let result = generate_from_macro(input);
        assert!(result.is_ok(), "gen_odd_codelet!(7) should succeed");
        let ts = result.expect("TokenStream for size 7");
        let s = ts.to_string();
        assert!(
            s.contains("codelet_notw_7"),
            "should contain codelet_notw_7"
        );
    }

    #[test]
    fn test_generate_from_macro_unsupported() {
        let input: proc_macro2::TokenStream = "4".parse().expect("parse literal");
        let result = generate_from_macro(input);
        assert!(
            result.is_err(),
            "gen_odd_codelet!(4) should fail with unsupported size"
        );
    }
}
