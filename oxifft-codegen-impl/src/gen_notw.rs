//! Non-twiddle codelet generation.
//!
//! Generates optimized base-case FFT kernels using symbolic computation,
//! common subexpression elimination, and strength reduction.

use crate::symbolic::emit_body_from_symbolic;
use proc_macro2::TokenStream;
use quote::quote;
use syn::LitInt;

/// Generate a non-twiddle codelet for the given size.
///
/// # Errors
/// Returns a `syn::Error` when the input token stream does not parse as a valid
/// size literal, or when the size is not in the supported set {2, 4, 8, 16, 32, 64}.
pub fn generate(input: TokenStream) -> Result<TokenStream, syn::Error> {
    let size: LitInt = syn::parse2(input)?;
    let n: usize = size.base10_parse().map_err(|_| {
        syn::Error::new(
            size.span(),
            "gen_notw_codelet: expected an integer size literal",
        )
    })?;

    match n {
        2 => Ok(gen_size_2()),
        4 => Ok(gen_size_4()),
        8 => Ok(gen_size_8()),
        16 => Ok(gen_size_16()),
        32 => Ok(gen_size_32()),
        64 => Ok(gen_size_64()),
        _ => Err(syn::Error::new(
            size.span(),
            format!("gen_notw_codelet: unsupported size {n} (expected one of 2, 4, 8, 16, 32, 64)"),
        )),
    }
}

fn gen_size_2() -> TokenStream {
    quote! {
        /// Size-2 DFT codelet (butterfly).
        #[inline(always)]
        pub fn codelet_notw_2<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            _sign: i32,
        ) {
            debug_assert!(x.len() >= 2);
            let a = x[0];
            let b = x[1];
            x[0] = a + b;
            x[1] = a - b;
        }
    }
}

fn gen_size_4() -> TokenStream {
    quote! {
        /// Size-4 DFT codelet.
        #[inline(always)]
        pub fn codelet_notw_4<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 4);

            let x0 = x[0];
            let x1 = x[1];
            let x2 = x[2];
            let x3 = x[3];

            // Stage 1
            let t0 = x0 + x2;
            let t1 = x0 - x2;
            let t2 = x1 + x3;
            let t3 = x1 - x3;

            // Apply rotation
            let t3_rot = if sign < 0 {
                crate::kernel::Complex::new(t3.im, -t3.re)
            } else {
                crate::kernel::Complex::new(-t3.im, t3.re)
            };

            // Stage 2
            x[0] = t0 + t2;
            x[1] = t1 + t3_rot;
            x[2] = t0 - t2;
            x[3] = t1 - t3_rot;
        }
    }
}

fn gen_size_8() -> TokenStream {
    // Size-8 DFT using radix-2 DIT with explicit butterfly stages.
    // All constants pre-computed via T::from_f64() to avoid trait method ambiguity.
    quote! {
        /// Size-8 DFT codelet using radix-2 DIT decomposition.
        ///
        /// Inputs are taken in natural order, output is in natural order.
        #[inline(always)]
        pub fn codelet_notw_8<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 8);

            // 1/sqrt(2) ≈ 0.7071067811865476
            let c2 = T::from_f64(0.707_106_781_186_547_6_f64);

            // Bit-reversal permutation for DIT (3-bit reversal)
            // Natural:   0 1 2 3 4 5 6 7
            // Bit-rev:   0 4 2 6 1 5 3 7
            let mut a = [crate::kernel::Complex::<T>::zero(); 8];
            a[0] = x[0]; a[1] = x[4];
            a[2] = x[2]; a[3] = x[6];
            a[4] = x[1]; a[5] = x[5];
            a[6] = x[3]; a[7] = x[7];

            // DIT Stage 1: 4 butterflies, span 1 (W2^0 = 1)
            for i in (0..8usize).step_by(2) {
                let t = a[i + 1];
                a[i + 1] = a[i] - t;
                a[i]     = a[i] + t;
            }

            // DIT Stage 2: 2 groups of 2 butterflies, span 2
            // W4^0 = 1, W4^1 = -i (forward) or +i (inverse)
            for group in (0..8usize).step_by(4) {
                // k=0: W4^0 = 1
                let t = a[group + 2];
                a[group + 2] = a[group] - t;
                a[group]     = a[group] + t;

                // k=1: W4^1
                let t = a[group + 3];
                let t_tw = if sign < 0 {
                    crate::kernel::Complex::new(t.im, -t.re)
                } else {
                    crate::kernel::Complex::new(-t.im, t.re)
                };
                a[group + 3] = a[group + 1] - t_tw;
                a[group + 1] = a[group + 1] + t_tw;
            }

            // DIT Stage 3: 1 group of 4 butterflies, span 4
            // W8^k for k in 0..4
            // k=0: W8^0 = 1
            let t = a[4];
            a[4] = a[0] - t;
            a[0] = a[0] + t;

            // k=1: W8^1 = (1-i)/sqrt(2) forward, (1+i)/sqrt(2) inverse
            let t = a[5];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new((t.re + t.im) * c2, (t.im - t.re) * c2)
            } else {
                crate::kernel::Complex::new((t.re - t.im) * c2, (t.im + t.re) * c2)
            };
            a[5] = a[1] - t_tw;
            a[1] = a[1] + t_tw;

            // k=2: W8^2 = -i (forward) or +i (inverse)
            let t = a[6];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new(t.im, -t.re)
            } else {
                crate::kernel::Complex::new(-t.im, t.re)
            };
            a[6] = a[2] - t_tw;
            a[2] = a[2] + t_tw;

            // k=3: W8^3 = (-1-i)/sqrt(2) forward, (-1+i)/sqrt(2) inverse
            let t = a[7];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new((-t.re + t.im) * c2, (-t.im - t.re) * c2)
            } else {
                crate::kernel::Complex::new((-t.re - t.im) * c2, (-t.im + t.re) * c2)
            };
            a[7] = a[3] - t_tw;
            a[3] = a[3] + t_tw;

            // Write back in natural order
            for i in 0..8usize {
                x[i] = a[i];
            }
        }
    }
}

fn gen_size_16() -> TokenStream {
    // Size-16 DFT codelet generated via the symbolic optimization pipeline.
    // Forward and inverse bodies are emitted from the optimized symbolic DAG,
    // then dispatched at runtime by `sign`.
    let fwd = emit_body_from_symbolic(16, true);
    let inv = emit_body_from_symbolic(16, false);
    quote! {
        /// Size-16 DFT codelet — generated via symbolic CSE/constant-folding pipeline.
        ///
        /// `sign < 0` → forward transform; `sign > 0` → inverse (un-normalized).
        #[inline(always)]
        #[allow(clippy::too_many_lines, clippy::approx_constant, clippy::suboptimal_flops)]
        pub fn codelet_notw_16<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 16);
            if sign < 0 {
                #fwd
            } else {
                #inv
            }
        }
    }
}

fn gen_size_32() -> TokenStream {
    // Size-32 DFT codelet generated via the symbolic optimization pipeline.
    let fwd = emit_body_from_symbolic(32, true);
    let inv = emit_body_from_symbolic(32, false);
    quote! {
        /// Size-32 DFT codelet — generated via symbolic CSE/constant-folding pipeline.
        ///
        /// `sign < 0` → forward transform; `sign > 0` → inverse (un-normalized).
        #[inline(always)]
        #[allow(clippy::too_many_lines, clippy::approx_constant, clippy::suboptimal_flops)]
        pub fn codelet_notw_32<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 32);
            if sign < 0 {
                #fwd
            } else {
                #inv
            }
        }
    }
}

fn gen_size_64() -> TokenStream {
    // Size-64 DFT codelet generated via the symbolic optimization pipeline.
    let fwd = emit_body_from_symbolic(64, true);
    let inv = emit_body_from_symbolic(64, false);
    quote! {
        /// Size-64 DFT codelet — generated via symbolic CSE/constant-folding pipeline.
        ///
        /// `sign < 0` → forward transform; `sign > 0` → inverse (un-normalized).
        #[inline(always)]
        #[allow(clippy::too_many_lines, clippy::approx_constant, clippy::suboptimal_flops)]
        pub fn codelet_notw_64<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 64);
            if sign < 0 {
                #fwd
            } else {
                #inv
            }
        }
    }
}
