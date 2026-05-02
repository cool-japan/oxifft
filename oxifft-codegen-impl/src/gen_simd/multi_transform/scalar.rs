//! Scalar butterfly emitters for multi-transform codelets.
//!
//! These emit the per-transform butterfly body used when no SIMD is available
//! or the ISA/size/precision combination has no SIMD implementation.

use proc_macro2::TokenStream;
use quote::quote;

use super::Precision;

/// Emit the scalar butterfly body for a single transform of `size`.
///
/// The body references `input`, `output`, `istride`, `ostride`, `base_in`,
/// `base_out` from the surrounding scope.
pub(super) fn gen_scalar_butterfly(size: usize, precision: Precision) -> TokenStream {
    match size {
        2 => gen_scalar_butterfly_size2(precision),
        4 => gen_scalar_butterfly_size4(precision),
        8 => gen_scalar_butterfly_size8(precision),
        _ => unreachable!("size already validated to be 2, 4, or 8"),
    }
}

/// Emit a size-2 radix-2 butterfly body.
///
/// `out[0] = x[0] + x[1]`, `out[1] = x[0] - x[1]`.
fn gen_scalar_butterfly_size2(precision: Precision) -> TokenStream {
    let ty_tokens: TokenStream = precision.type_str().parse().expect("valid type token");
    quote! {
        let x0_re = *input.add(base_in);
        let x0_im = *input.add(base_in + 1);
        let x1_re = *input.add(base_in + istride);
        let x1_im = *input.add(base_in + istride + 1);

        let out0_re: #ty_tokens = x0_re + x1_re;
        let out0_im: #ty_tokens = x0_im + x1_im;
        let out1_re: #ty_tokens = x0_re - x1_re;
        let out1_im: #ty_tokens = x0_im - x1_im;

        *output.add(base_out)               = out0_re;
        *output.add(base_out + 1)           = out0_im;
        *output.add(base_out + ostride)     = out1_re;
        *output.add(base_out + ostride + 1) = out1_im;
    }
}

/// Emit a size-4 radix-4 DIF butterfly body (forward sign).
///
/// Stages: t0=x0+x2, t1=x0-x2, t2=x1+x3, t3=x1-x3.
/// Forward rotation: `t3 *= -i` → `(im, -re)`.
fn gen_scalar_butterfly_size4(precision: Precision) -> TokenStream {
    let ty_tokens: TokenStream = precision.type_str().parse().expect("valid type token");
    quote! {
        let stride2 = istride * 2;
        let stride3 = istride * 3;
        let ostride2 = ostride * 2;
        let ostride3 = ostride * 3;

        let x0_re = *input.add(base_in);
        let x0_im = *input.add(base_in + 1);
        let x1_re = *input.add(base_in + istride);
        let x1_im = *input.add(base_in + istride + 1);
        let x2_re = *input.add(base_in + stride2);
        let x2_im = *input.add(base_in + stride2 + 1);
        let x3_re = *input.add(base_in + stride3);
        let x3_im = *input.add(base_in + stride3 + 1);

        let t0_re: #ty_tokens = x0_re + x2_re;
        let t0_im: #ty_tokens = x0_im + x2_im;
        let t1_re: #ty_tokens = x0_re - x2_re;
        let t1_im: #ty_tokens = x0_im - x2_im;
        let t2_re: #ty_tokens = x1_re + x3_re;
        let t2_im: #ty_tokens = x1_im + x3_im;
        let t3_re: #ty_tokens = x1_re - x3_re;
        let t3_im: #ty_tokens = x1_im - x3_im;

        // Forward: multiply t3 by -i: (re, im) -> (im, -re)
        let t3rot_re: #ty_tokens =  t3_im;
        let t3rot_im: #ty_tokens = -t3_re;

        *output.add(base_out)                = t0_re + t2_re;
        *output.add(base_out + 1)            = t0_im + t2_im;
        *output.add(base_out + ostride)      = t1_re + t3rot_re;
        *output.add(base_out + ostride + 1)  = t1_im + t3rot_im;
        *output.add(base_out + ostride2)     = t0_re - t2_re;
        *output.add(base_out + ostride2 + 1) = t0_im - t2_im;
        *output.add(base_out + ostride3)     = t1_re - t3rot_re;
        *output.add(base_out + ostride3 + 1) = t1_im - t3rot_im;
    }
}

/// Emit a size-8 radix-8 DIF butterfly body (forward sign, three stages).
///
/// Stage 1: upper ± lower half (k=0..3).
/// Stage 2: length-4 DIF on the a-group and b-group with W8 twiddles.
/// Stage 3: length-2 butterfly on each pair.
#[allow(clippy::too_many_lines)]
fn gen_scalar_butterfly_size8(precision: Precision) -> TokenStream {
    let ty_tokens: TokenStream = precision.type_str().parse().expect("valid type token");
    let inv_sqrt2_lit: TokenStream = if precision == Precision::F32 {
        "core::f32::consts::FRAC_1_SQRT_2"
            .parse()
            .expect("valid literal")
    } else {
        "core::f64::consts::FRAC_1_SQRT_2"
            .parse()
            .expect("valid literal")
    };
    quote! {
        let s1 = istride;
        let s2 = istride * 2;
        let s3 = istride * 3;
        let s4 = istride * 4;
        let s5 = istride * 5;
        let s6 = istride * 6;
        let s7 = istride * 7;

        // Load 8 complexes
        let (x0r, x0i) = (*input.add(base_in),        *input.add(base_in + 1));
        let (x1r, x1i) = (*input.add(base_in + s1),   *input.add(base_in + s1 + 1));
        let (x2r, x2i) = (*input.add(base_in + s2),   *input.add(base_in + s2 + 1));
        let (x3r, x3i) = (*input.add(base_in + s3),   *input.add(base_in + s3 + 1));
        let (x4r, x4i) = (*input.add(base_in + s4),   *input.add(base_in + s4 + 1));
        let (x5r, x5i) = (*input.add(base_in + s5),   *input.add(base_in + s5 + 1));
        let (x6r, x6i) = (*input.add(base_in + s6),   *input.add(base_in + s6 + 1));
        let (x7r, x7i) = (*input.add(base_in + s7),   *input.add(base_in + s7 + 1));

        let inv_sqrt2: #ty_tokens = #inv_sqrt2_lit;

        // Stage 1: upper ± lower half
        let (a0r, a0i): (#ty_tokens, #ty_tokens) = (x0r + x4r, x0i + x4i);
        let (a1r, a1i): (#ty_tokens, #ty_tokens) = (x1r + x5r, x1i + x5i);
        let (a2r, a2i): (#ty_tokens, #ty_tokens) = (x2r + x6r, x2i + x6i);
        let (a3r, a3i): (#ty_tokens, #ty_tokens) = (x3r + x7r, x3i + x7i);
        let (b0r, b0i): (#ty_tokens, #ty_tokens) = (x0r - x4r, x0i - x4i);
        let (b1r, b1i): (#ty_tokens, #ty_tokens) = (x1r - x5r, x1i - x5i);
        let (b2r, b2i): (#ty_tokens, #ty_tokens) = (x2r - x6r, x2i - x6i);
        let (b3r, b3i): (#ty_tokens, #ty_tokens) = (x3r - x7r, x3i - x7i);

        // Apply W8 twiddles to b1, b2, b3 (forward DFT):
        // b1 *= W8^1 = (1-i)/√2  → ((b1r+b1i)/√2, (-b1r+b1i)/√2)
        // b2 *= W8^2 = -i        → (b2i, -b2r)
        // b3 *= W8^3 = (-1-i)/√2 → ((-b3r+b3i)/√2, (-b3r-b3i)/√2)
        let b1tr: #ty_tokens = ( b1r + b1i) * inv_sqrt2;
        let b1ti: #ty_tokens = (-b1r + b1i) * inv_sqrt2;
        let b2tr: #ty_tokens =  b2i;
        let b2ti: #ty_tokens = -b2r;
        let b3tr: #ty_tokens = (-b3r + b3i) * inv_sqrt2;
        let b3ti: #ty_tokens = (-b3r - b3i) * inv_sqrt2;

        // Stage 2: length-4 DIF on a-group and b-group
        let (c0r, c0i): (#ty_tokens, #ty_tokens) = (a0r + a2r, a0i + a2i);
        let (c1r, c1i): (#ty_tokens, #ty_tokens) = (a1r + a3r, a1i + a3i);
        let (c2r, c2i): (#ty_tokens, #ty_tokens) = (a0r - a2r, a0i - a2i);
        let d3r: #ty_tokens = a1r - a3r;
        let d3i: #ty_tokens = a1i - a3i;
        let c3r: #ty_tokens =  d3i; // twiddle: *(-i) → (im, -re)
        let c3i: #ty_tokens = -d3r;

        let (e0r, e0i): (#ty_tokens, #ty_tokens) = (b0r + b2tr, b0i + b2ti);
        let (e1r, e1i): (#ty_tokens, #ty_tokens) = (b1tr + b3tr, b1ti + b3ti);
        let (e2r, e2i): (#ty_tokens, #ty_tokens) = (b0r - b2tr, b0i - b2ti);
        let f3r: #ty_tokens = b1tr - b3tr;
        let f3i: #ty_tokens = b1ti - b3ti;
        let e3r: #ty_tokens =  f3i; // twiddle: *(-i) → (im, -re)
        let e3i: #ty_tokens = -f3r;

        // Stage 3: length-2 butterfly on each pair
        let os1 = ostride;
        let os2 = ostride * 2;
        let os3 = ostride * 3;
        let os4 = ostride * 4;
        let os5 = ostride * 5;
        let os6 = ostride * 6;
        let os7 = ostride * 7;

        *output.add(base_out)           = c0r + c1r;
        *output.add(base_out + 1)       = c0i + c1i;
        *output.add(base_out + os4)     = c0r - c1r;
        *output.add(base_out + os4 + 1) = c0i - c1i;

        *output.add(base_out + os2)     = c2r + c3r;
        *output.add(base_out + os2 + 1) = c2i + c3i;
        *output.add(base_out + os6)     = c2r - c3r;
        *output.add(base_out + os6 + 1) = c2i - c3i;

        *output.add(base_out + os1)     = e0r + e1r;
        *output.add(base_out + os1 + 1) = e0i + e1i;
        *output.add(base_out + os5)     = e0r - e1r;
        *output.add(base_out + os5 + 1) = e0i - e1i;

        *output.add(base_out + os3)     = e2r + e3r;
        *output.add(base_out + os3 + 1) = e2i + e3i;
        *output.add(base_out + os7)     = e2r - e3r;
        *output.add(base_out + os7 + 1) = e2i - e3i;
    }
}
