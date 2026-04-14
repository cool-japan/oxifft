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
        16 => gen_twiddle_16(),
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

#[allow(clippy::too_many_lines)]
fn gen_twiddle_16() -> TokenStream {
    let expanded = quote! {
        /// Radix-16 twiddle codelet.
        ///
        /// Applies 15 twiddle factors to inputs x[1]..x[15] and computes a 16-point FFT
        /// using a radix-2 DIT (decimation-in-time) butterfly structure.
        ///
        /// # Arguments
        /// * `x`        - Input/output slice of at least 16 complex values
        /// * `twiddles` - Array of 15 precomputed twiddle factors for positions 1..=15
        /// * `sign`     - Transform direction: -1 for forward, +1 for inverse
        #[inline(always)]
        pub fn codelet_twiddle_16<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            twiddles: &[crate::kernel::Complex<T>; 15],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 16);

            // Step 1: Apply external twiddle factors to positions 1..=15
            let x0  = x[0];
            let x1  = x[1]  * twiddles[0];
            let x2  = x[2]  * twiddles[1];
            let x3  = x[3]  * twiddles[2];
            let x4  = x[4]  * twiddles[3];
            let x5  = x[5]  * twiddles[4];
            let x6  = x[6]  * twiddles[5];
            let x7  = x[7]  * twiddles[6];
            let x8  = x[8]  * twiddles[7];
            let x9  = x[9]  * twiddles[8];
            let x10 = x[10] * twiddles[9];
            let x11 = x[11] * twiddles[10];
            let x12 = x[12] * twiddles[11];
            let x13 = x[13] * twiddles[12];
            let x14 = x[14] * twiddles[13];
            let x15 = x[15] * twiddles[14];

            // Step 2: Compute 16-point DFT using radix-2 DIT.
            // Place twiddle-applied values in bit-reversed order, then apply 4 DIT stages.
            //
            // Bit-reversal permutation for 16 (4-bit reversal):
            //   0->0, 1->8, 2->4, 3->12, 4->2, 5->10, 6->6, 7->14,
            //   8->1, 9->9, 10->5, 11->13, 12->3, 13->11, 14->7, 15->15
            let mut a = [crate::kernel::Complex::<T>::zero(); 16];
            a[0]  = x0;
            a[1]  = x8;
            a[2]  = x4;
            a[3]  = x12;
            a[4]  = x2;
            a[5]  = x10;
            a[6]  = x6;
            a[7]  = x14;
            a[8]  = x1;
            a[9]  = x9;
            a[10] = x5;
            a[11] = x13;
            a[12] = x3;
            a[13] = x11;
            a[14] = x7;
            a[15] = x15;

            // DIT Stage 1: 8 butterflies, span 1 (W2^0 = 1, no twiddle)
            for i in (0..16usize).step_by(2) {
                let t = a[i + 1];
                a[i + 1] = a[i] - t;
                a[i]     = a[i] + t;
            }

            // DIT Stage 2: 4 groups of 2 butterflies, span 2
            // W4^0 = 1,  W4^1 = -i (forward) or +i (inverse)
            for group in (0..16usize).step_by(4) {
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

            // DIT Stage 3: 2 groups of 4 butterflies, span 4
            // W8^k for k in 0..4
            // c2 = 1/sqrt(2) ≈ 0.7071067811865476
            let c2 = T::from_f64(0.707_106_781_186_547_6_f64);
            for group in (0..16usize).step_by(8) {
                // k=0: W8^0 = 1
                let t = a[group + 4];
                a[group + 4] = a[group] - t;
                a[group]     = a[group] + t;

                // k=1: W8^1 = (1-i)/sqrt(2) forward, (1+i)/sqrt(2) inverse
                let t = a[group + 5];
                let t_tw = if sign < 0 {
                    crate::kernel::Complex::new((t.re + t.im) * c2, (t.im - t.re) * c2)
                } else {
                    crate::kernel::Complex::new((t.re - t.im) * c2, (t.im + t.re) * c2)
                };
                a[group + 5] = a[group + 1] - t_tw;
                a[group + 1] = a[group + 1] + t_tw;

                // k=2: W8^2 = -i forward, +i inverse
                let t = a[group + 6];
                let t_tw = if sign < 0 {
                    crate::kernel::Complex::new(t.im, -t.re)
                } else {
                    crate::kernel::Complex::new(-t.im, t.re)
                };
                a[group + 6] = a[group + 2] - t_tw;
                a[group + 2] = a[group + 2] + t_tw;

                // k=3: W8^3 = (-1-i)/sqrt(2) forward, (-1+i)/sqrt(2) inverse
                let t = a[group + 7];
                let t_tw = if sign < 0 {
                    crate::kernel::Complex::new((-t.re + t.im) * c2, (-t.im - t.re) * c2)
                } else {
                    crate::kernel::Complex::new((-t.re - t.im) * c2, (-t.im + t.re) * c2)
                };
                a[group + 7] = a[group + 3] - t_tw;
                a[group + 3] = a[group + 3] + t_tw;
            }

            // DIT Stage 4: 1 group of 8 butterflies, span 8
            // W16^k for k in 0..8
            // Constants: cos(π/8), sin(π/8), 1/sqrt(2), cos(3π/8)=sin(π/8), sin(3π/8)=cos(π/8)
            let c1 = T::from_f64(0.923_879_532_511_286_7_f64); // cos(π/8)
            let s1 = T::from_f64(0.382_683_432_365_089_8_f64); // sin(π/8)

            // k=0: W16^0 = 1
            let t = a[8];
            a[8] = a[0] - t;
            a[0] = a[0] + t;

            // k=1: W16^1 = cos(π/8) - i*sin(π/8) forward, cos(π/8) + i*sin(π/8) inverse
            let t = a[9];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new(t.re * c1 + t.im * s1, t.im * c1 - t.re * s1)
            } else {
                crate::kernel::Complex::new(t.re * c1 - t.im * s1, t.im * c1 + t.re * s1)
            };
            a[9] = a[1] - t_tw;
            a[1] = a[1] + t_tw;

            // k=2: W16^2 = (1-i)/sqrt(2) forward, (1+i)/sqrt(2) inverse
            let t = a[10];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new((t.re + t.im) * c2, (t.im - t.re) * c2)
            } else {
                crate::kernel::Complex::new((t.re - t.im) * c2, (t.im + t.re) * c2)
            };
            a[10] = a[2] - t_tw;
            a[2] = a[2] + t_tw;

            // k=3: W16^3 = cos(3π/8) - i*sin(3π/8) forward
            //             = sin(π/8) - i*cos(π/8) forward  (since cos(3π/8)=sin(π/8))
            let c3 = s1; // cos(3π/8) = sin(π/8)
            let s3 = c1; // sin(3π/8) = cos(π/8)
            let t = a[11];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new(t.re * c3 + t.im * s3, t.im * c3 - t.re * s3)
            } else {
                crate::kernel::Complex::new(t.re * c3 - t.im * s3, t.im * c3 + t.re * s3)
            };
            a[11] = a[3] - t_tw;
            a[3] = a[3] + t_tw;

            // k=4: W16^4 = -i forward, +i inverse
            let t = a[12];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new(t.im, -t.re)
            } else {
                crate::kernel::Complex::new(-t.im, t.re)
            };
            a[12] = a[4] - t_tw;
            a[4] = a[4] + t_tw;

            // k=5: W16^5 = cos(5π/8) - i*sin(5π/8) = -sin(π/8) - i*cos(π/8) forward
            let t = a[13];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new(-t.re * s1 + t.im * c1, -t.im * s1 - t.re * c1)
            } else {
                crate::kernel::Complex::new(-t.re * s1 - t.im * c1, -t.im * s1 + t.re * c1)
            };
            a[13] = a[5] - t_tw;
            a[5] = a[5] + t_tw;

            // k=6: W16^6 = (-1-i)/sqrt(2) forward, (-1+i)/sqrt(2) inverse
            let t = a[14];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new((-t.re + t.im) * c2, (-t.im - t.re) * c2)
            } else {
                crate::kernel::Complex::new((-t.re - t.im) * c2, (-t.im + t.re) * c2)
            };
            a[14] = a[6] - t_tw;
            a[6] = a[6] + t_tw;

            // k=7: W16^7 = cos(7π/8) - i*sin(7π/8) = -cos(π/8) - i*sin(π/8) forward
            let t = a[15];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new(-t.re * c1 + t.im * s1, -t.im * c1 - t.re * s1)
            } else {
                crate::kernel::Complex::new(-t.re * c1 - t.im * s1, -t.im * c1 + t.re * s1)
            };
            a[15] = a[7] - t_tw;
            a[7] = a[7] + t_tw;

            // Write back in natural order (DIT produces natural order after bit-reversal input)
            for i in 0..16usize {
                x[i] = a[i];
            }
        }
    };
    TokenStream::from(expanded)
}

#[allow(clippy::too_many_lines)]
fn gen_twiddle_8() -> TokenStream {
    let expanded = quote! {
        /// Radix-8 twiddle codelet.
        ///
        /// Applies 7 external twiddle factors to inputs x[1]..x[7], then computes
        /// an 8-point FFT using a radix-2 DIT butterfly structure.
        ///
        /// # Arguments
        /// * `x`        - Input/output slice of at least 8 complex values
        /// * `twiddles` - Array of 7 precomputed twiddle factors for positions 1..=7
        /// * `sign`     - Transform direction: -1 for forward, +1 for inverse
        #[inline(always)]
        pub fn codelet_twiddle_8<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            twiddles: &[crate::kernel::Complex<T>; 7],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 8);

            // Step 1: Apply external twiddle factors to positions 1..=7
            let x0 = x[0];
            let x1 = x[1] * twiddles[0];
            let x2 = x[2] * twiddles[1];
            let x3 = x[3] * twiddles[2];
            let x4 = x[4] * twiddles[3];
            let x5 = x[5] * twiddles[4];
            let x6 = x[6] * twiddles[5];
            let x7 = x[7] * twiddles[6];

            // Step 2: Compute 8-point DFT using radix-2 DIT.
            // Place twiddle-applied values in bit-reversed order, then apply 3 DIT stages.
            // Bit-reversal for 8 (3-bit): 0→0, 1→4, 2→2, 3→6, 4→1, 5→5, 6→3, 7→7
            let mut a = [crate::kernel::Complex::<T>::zero(); 8];
            a[0] = x0; a[1] = x4;
            a[2] = x2; a[3] = x6;
            a[4] = x1; a[5] = x5;
            a[6] = x3; a[7] = x7;

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
            // W8^k for k in 0..4. c2 = 1/sqrt(2) ≈ 0.7071067811865476
            let c2 = T::from_f64(0.707_106_781_186_547_6_f64);

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

            // Compute twiddles inline via fully-qualified Float trait methods to avoid ambiguity
            let tw1 = crate::kernel::Complex::new(
                <T as crate::kernel::Float>::cos(angle_step),
                <T as crate::kernel::Float>::sin(angle_step),
            );
            let tw2 = crate::kernel::Complex::new(
                <T as crate::kernel::Float>::cos(angle_step * T::from_usize(2)),
                <T as crate::kernel::Float>::sin(angle_step * T::from_usize(2)),
            );
            let tw3 = crate::kernel::Complex::new(
                <T as crate::kernel::Float>::cos(angle_step * T::from_usize(3)),
                <T as crate::kernel::Float>::sin(angle_step * T::from_usize(3)),
            );
            let tw4 = crate::kernel::Complex::new(
                <T as crate::kernel::Float>::cos(angle_step * T::from_usize(4)),
                <T as crate::kernel::Float>::sin(angle_step * T::from_usize(4)),
            );
            let tw5 = crate::kernel::Complex::new(
                <T as crate::kernel::Float>::cos(angle_step * T::from_usize(5)),
                <T as crate::kernel::Float>::sin(angle_step * T::from_usize(5)),
            );
            let tw6 = crate::kernel::Complex::new(
                <T as crate::kernel::Float>::cos(angle_step * T::from_usize(6)),
                <T as crate::kernel::Float>::sin(angle_step * T::from_usize(6)),
            );
            let tw7 = crate::kernel::Complex::new(
                <T as crate::kernel::Float>::cos(angle_step * T::from_usize(7)),
                <T as crate::kernel::Float>::sin(angle_step * T::from_usize(7)),
            );

            let twiddles = [tw1, tw2, tw3, tw4, tw5, tw6, tw7];
            codelet_twiddle_8(x, &twiddles, sign);
        }
    };
    TokenStream::from(expanded)
}
