//! Twiddle-factor codelet generation.
//!
//! Generates codelets that apply twiddle factors during multi-radix FFT computation.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, LitInt};

/// Generate a twiddle codelet for the given radix.
pub fn generate(input: TokenStream) -> TokenStream {
    let radix = parse_macro_input!(input as LitInt);
    let r: usize = radix.base10_parse().expect("Invalid radix literal");

    match r {
        2 => gen_twiddle_2(),
        4 => gen_twiddle_4(),
        8 => gen_twiddle_8(),
        _ => panic!("Unsupported twiddle radix: {r}"),
    }
}

fn gen_twiddle_2() -> TokenStream {
    let expanded = quote! {
        /// Radix-2 twiddle codelet.
        ///
        /// Applies a single twiddle factor and computes a 2-point butterfly.
        #[inline(always)]
        pub fn codelet_twiddle_2<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            twiddle: crate::kernel::Complex<T>,
        ) {
            debug_assert!(x.len() >= 2);
            let a = x[0];
            let b = x[1] * twiddle;
            x[0] = a + b;
            x[1] = a - b;
        }
    };
    TokenStream::from(expanded)
}

fn gen_twiddle_4() -> TokenStream {
    let expanded = quote! {
        /// Radix-4 twiddle codelet.
        ///
        /// Applies twiddle factors w1, w2, w3 to inputs x[1], x[2], x[3]
        /// and computes a 4-point FFT.
        #[inline(always)]
        pub fn codelet_twiddle_4<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            tw1: crate::kernel::Complex<T>,
            tw2: crate::kernel::Complex<T>,
            tw3: crate::kernel::Complex<T>,
            sign: i32,
        ) {
            debug_assert!(x.len() >= 4);

            let x0 = x[0];
            let x1 = x[1] * tw1;
            let x2 = x[2] * tw2;
            let x3 = x[3] * tw3;

            let t0 = x0 + x2;
            let t1 = x0 - x2;
            let t2 = x1 + x3;
            let t3 = x1 - x3;

            let t3_rot = if sign < 0 {
                crate::kernel::Complex::new(t3.im, -t3.re)
            } else {
                crate::kernel::Complex::new(-t3.im, t3.re)
            };

            x[0] = t0 + t2;
            x[1] = t1 + t3_rot;
            x[2] = t0 - t2;
            x[3] = t1 - t3_rot;
        }
    };
    TokenStream::from(expanded)
}

fn gen_twiddle_8() -> TokenStream {
    let expanded = quote! {
        /// Radix-8 twiddle codelet.
        ///
        /// Applies 7 twiddle factors to inputs x[1]..x[7] and computes 8-point FFT.
        #[inline(always)]
        pub fn codelet_twiddle_8<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            twiddles: &[crate::kernel::Complex<T>; 7],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 8);

            // Apply twiddles
            let x0 = x[0];
            let x1 = x[1] * twiddles[0];
            let x2 = x[2] * twiddles[1];
            let x3 = x[3] * twiddles[2];
            let x4 = x[4] * twiddles[3];
            let x5 = x[5] * twiddles[4];
            let x6 = x[6] * twiddles[5];
            let x7 = x[7] * twiddles[6];

            // Stage 1: Size-2 butterflies
            let t0 = x0 + x4; let t1 = x0 - x4;
            let t2 = x2 + x6; let t3 = x2 - x6;
            let t4 = x1 + x5; let t5 = x1 - x5;
            let t6 = x3 + x7; let t7 = x3 - x7;

            // Rotation for t3, t7
            let t3_rot = if sign < 0 {
                crate::kernel::Complex::new(t3.im, -t3.re)
            } else {
                crate::kernel::Complex::new(-t3.im, t3.re)
            };

            let t7_rot = if sign < 0 {
                crate::kernel::Complex::new(t7.im, -t7.re)
            } else {
                crate::kernel::Complex::new(-t7.im, t7.re)
            };

            // Stage 2
            let u0 = t0 + t2; let u1 = t0 - t2;
            let u2 = t4 + t6; let u3 = t4 - t6;
            let u4 = t1 + t3_rot; let u5 = t1 - t3_rot;
            let u6 = t5 + t7_rot; let u7 = t5 - t7_rot;

            // W8 twiddles
            let sqrt2_inv = T::ONE / T::TWO.sqrt();
            let w8_re = sqrt2_inv;
            let w8_im = if sign < 0 { -sqrt2_inv } else { sqrt2_inv };

            let u3_rot = if sign < 0 {
                crate::kernel::Complex::new(u3.im, -u3.re)
            } else {
                crate::kernel::Complex::new(-u3.im, u3.re)
            };

            let u6_tw = crate::kernel::Complex::new(
                u6.re * w8_re - u6.im * w8_im,
                u6.re * w8_im + u6.im * w8_re,
            );
            let u7_tw = crate::kernel::Complex::new(
                u7.re * (-w8_im) - u7.im * w8_re,
                u7.re * w8_re + u7.im * (-w8_im),
            );

            // Stage 3: Final outputs
            x[0] = u0 + u2;
            x[4] = u0 - u2;
            x[2] = u1 + u3_rot;
            x[6] = u1 - u3_rot;
            x[1] = u4 + u6_tw;
            x[5] = u4 - u6_tw;
            x[3] = u5 + u7_tw;
            x[7] = u5 - u7_tw;
        }

        /// Radix-8 twiddle codelet with inline twiddle computation.
        ///
        /// This version computes twiddles from angle step, useful when twiddles
        /// are not precomputed.
        #[inline(always)]
        pub fn codelet_twiddle_8_inline<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            angle_step: T,
            sign: i32,
        ) {
            debug_assert!(x.len() >= 8);

            // Compute twiddles inline
            let tw1 = crate::kernel::Complex::new((angle_step).cos(), (angle_step).sin());
            let tw2 = crate::kernel::Complex::new((angle_step * T::TWO).cos(), (angle_step * T::TWO).sin());
            let tw3 = crate::kernel::Complex::new((angle_step * T::from_usize(3)).cos(), (angle_step * T::from_usize(3)).sin());
            let tw4 = crate::kernel::Complex::new((angle_step * T::from_usize(4)).cos(), (angle_step * T::from_usize(4)).sin());
            let tw5 = crate::kernel::Complex::new((angle_step * T::from_usize(5)).cos(), (angle_step * T::from_usize(5)).sin());
            let tw6 = crate::kernel::Complex::new((angle_step * T::from_usize(6)).cos(), (angle_step * T::from_usize(6)).sin());
            let tw7 = crate::kernel::Complex::new((angle_step * T::from_usize(7)).cos(), (angle_step * T::from_usize(7)).sin());

            let twiddles = [tw1, tw2, tw3, tw4, tw5, tw6, tw7];
            codelet_twiddle_8(x, &twiddles, sign);
        }
    };
    TokenStream::from(expanded)
}
