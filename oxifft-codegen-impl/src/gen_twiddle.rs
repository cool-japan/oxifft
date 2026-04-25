//! Twiddle-factor codelet generation.
//!
//! Generates codelets that apply twiddle factors during multi-radix FFT computation.

use proc_macro2::TokenStream;
use quote::quote;
use syn::LitInt;

/// Generate a twiddle codelet for the given radix.
///
/// # Errors
/// Returns a `syn::Error` when the input does not parse as a valid radix literal,
/// or when the radix is not in the supported set {2, 4, 8, 16}.
pub fn generate(input: TokenStream) -> Result<TokenStream, syn::Error> {
    let radix: LitInt = syn::parse2(input)?;
    let r: usize = radix.base10_parse().map_err(|_| {
        syn::Error::new(
            radix.span(),
            "gen_twiddle_codelet: expected an integer radix literal",
        )
    })?;

    match r {
        2 => Ok(gen_twiddle_2()),
        4 => Ok(gen_twiddle_4()),
        8 => Ok(gen_twiddle_8()),
        16 => Ok(gen_twiddle_16()),
        _ => Err(syn::Error::new(
            radix.span(),
            format!("gen_twiddle_codelet: unsupported radix {r} (expected one of 2, 4, 8, 16)"),
        )),
    }
}

fn gen_twiddle_2() -> TokenStream {
    quote! {
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
    }
}

fn gen_twiddle_4() -> TokenStream {
    quote! {
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
    }
}

#[allow(clippy::too_many_lines)]
fn gen_twiddle_16() -> TokenStream {
    quote! {
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
    }
}

#[allow(clippy::too_many_lines)]
fn gen_twiddle_8() -> TokenStream {
    quote! {
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
    }
}

/// Generate a split-radix twiddle codelet for the given size.
///
/// If no size is specified (empty input), generates the generic runtime-parameterized
/// split-radix twiddle codelet. If a size is given (8 or 16), generates a specialized
/// unrolled version for that size.
///
/// # Errors
/// Returns a `syn::Error` when the input does not parse as a valid size literal,
/// or when the size is not in the supported set {8, 16} (or empty for the generic variant).
pub fn generate_split_radix(input: TokenStream) -> Result<TokenStream, syn::Error> {
    if input.is_empty() {
        return Ok(gen_split_radix_twiddle());
    }
    let size: LitInt = syn::parse2(input)?;
    let n: usize = size.base10_parse().map_err(|_| {
        syn::Error::new(
            size.span(),
            "gen_split_radix_twiddle_codelet: expected an integer size literal",
        )
    })?;
    match n {
        8 => Ok(gen_split_radix_twiddle_8()),
        16 => Ok(gen_split_radix_twiddle_16()),
        _ => Err(syn::Error::new(
            size.span(),
            format!("gen_split_radix_twiddle_codelet: unsupported size {n} (use 8 or 16, or empty for generic)"),
        )),
    }
}

/// Generate the generic split-radix twiddle codelet (L-shaped butterfly).
///
/// The split-radix FFT decomposes an N-point DFT into:
/// - One N/2-point DFT of even-indexed elements
/// - Two N/4-point DFTs of odd-indexed elements (with twiddle factors `W_N^k` and `W_N^{3k`})
///
/// This codelet performs the combining step (L-shaped butterfly):
///   For k = 0..N/4-1:
///     t1 = `W_N^k` * O1[k],  t2 = `W_N^{3k`} * O3[k]
///     p = t1 + t2,  m = t1 - t2
///     X[k]       = E[k]     + p
///     X[k+N/4]   = E[k+N/4] - j*(t1-t2)  (forward)
///     X[k+N/2]   = E[k]     - p
///     X[k+3N/4]  = E[k+N/4] + j*(t1-t2)  (forward)
fn gen_split_radix_twiddle() -> TokenStream {
    let expanded = quote! {
        /// Split-radix twiddle codelet (L-shaped butterfly).
        ///
        /// Combines the results of an N/2-point even DFT with two N/4-point odd DFTs
        /// using split-radix decomposition. This reduces the total number of real
        /// multiplications compared to standard radix-2: approximately 4N log2(N) - 6N + 8
        /// vs. 5N log2(N) for radix-2.
        ///
        /// # Data Layout
        ///
        /// On input, `data[0..n]` is organized as:
        /// - `data[0..n/2]`: N/2-point DFT of even-indexed elements (E)
        /// - `data[n/2..3n/4]`: N/4-point DFT of odd-1 elements (O1, indices 1,5,9,...)
        /// - `data[3n/4..n]`: N/4-point DFT of odd-3 elements (O3, indices 3,7,11,...)
        ///
        /// On output, `data[0..n]` contains the combined N-point DFT result.
        ///
        /// # Arguments
        /// * `data`     - Input/output slice of at least `n` complex values
        /// * `n`        - Transform size (must be divisible by 4 and >= 4)
        /// * `twiddles` - Twiddle factors W_N^k for k = 0..n/4-1
        /// * `twiddles3`- Twiddle factors W_N^{3k} for k = 0..n/4-1
        /// * `sign`     - Transform direction: -1 for forward, +1 for inverse
        #[inline]
        pub fn codelet_split_radix_twiddle<T: crate::kernel::Float>(
            data: &mut [crate::kernel::Complex<T>],
            n: usize,
            twiddles: &[crate::kernel::Complex<T>],
            twiddles3: &[crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(n >= 4 && n % 4 == 0, "n must be >= 4 and divisible by 4");
            debug_assert!(data.len() >= n, "data slice too short for split-radix");

            let n2 = n >> 1;   // N/2
            let n4 = n >> 2;   // N/4

            debug_assert!(twiddles.len() >= n4, "twiddles slice too short");
            debug_assert!(twiddles3.len() >= n4, "twiddles3 slice too short");

            for k in 0..n4 {
                // Read the three sub-DFT results
                let e_k    = data[k];          // E[k] from even DFT
                let e_k_q  = data[k + n4];     // E[k + N/4] from even DFT
                let o1_k   = data[n2 + k];     // O1[k] from first odd DFT
                let o3_k   = data[n2 + n4 + k]; // O3[k] from second odd DFT

                // Apply twiddle factors to odd sub-DFT results
                let t1 = o1_k * twiddles[k];    // W_N^k * O1[k]
                let t2 = o3_k * twiddles3[k];   // W_N^{3k} * O3[k]

                // Split-radix butterfly computation
                let p = t1 + t2;    // Sum of twiddled odd results
                let m = t1 - t2;    // Difference of twiddled odd results

                // Rotate difference by -j (forward) or +j (inverse)
                // -j * (a + bi) = (b, -a);   +j * (a + bi) = (-b, a)
                let m_rot = if sign < 0 {
                    crate::kernel::Complex::new(m.im, -m.re)
                } else {
                    crate::kernel::Complex::new(-m.im, m.re)
                };

                // Write combined results to all four quarters
                data[k]          = e_k   + p;       // X[k]
                data[k + n4]     = e_k_q + m_rot;   // X[k + N/4]
                data[k + n2]     = e_k   - p;       // X[k + N/2]
                data[k + n2 + n4] = e_k_q - m_rot;  // X[k + 3N/4]
            }
        }

        /// Split-radix twiddle codelet with inline twiddle computation.
        ///
        /// Computes W_N^k and W_N^{3k} twiddle factors on the fly from the
        /// base angle step, useful when twiddle tables are not precomputed.
        ///
        /// # Arguments
        /// * `data` - Input/output slice of at least `n` complex values
        /// * `n`    - Transform size (must be divisible by 4 and >= 4)
        /// * `sign` - Transform direction: -1 for forward, +1 for inverse
        #[inline]
        pub fn codelet_split_radix_twiddle_inline<T: crate::kernel::Float>(
            data: &mut [crate::kernel::Complex<T>],
            n: usize,
            sign: i32,
        ) {
            debug_assert!(n >= 4 && n % 4 == 0, "n must be >= 4 and divisible by 4");
            debug_assert!(data.len() >= n, "data slice too short for split-radix");

            let n2 = n >> 1;
            let n4 = n >> 2;

            // Base angle: -2π/N (forward) or +2π/N (inverse)
            let base_angle = if sign < 0 {
                -2.0_f64 * core::f64::consts::PI / (n as f64)
            } else {
                2.0_f64 * core::f64::consts::PI / (n as f64)
            };

            for k in 0..n4 {
                let angle_k = base_angle * (k as f64);
                let angle_3k = base_angle * (3 * k) as f64;

                let tw = crate::kernel::Complex::new(
                    T::from_f64(angle_k.cos()),
                    T::from_f64(angle_k.sin()),
                );
                let tw3 = crate::kernel::Complex::new(
                    T::from_f64(angle_3k.cos()),
                    T::from_f64(angle_3k.sin()),
                );

                let e_k    = data[k];
                let e_k_q  = data[k + n4];
                let o1_k   = data[n2 + k];
                let o3_k   = data[n2 + n4 + k];

                let t1 = o1_k * tw;
                let t2 = o3_k * tw3;

                let p = t1 + t2;
                let m = t1 - t2;

                let m_rot = if sign < 0 {
                    crate::kernel::Complex::new(m.im, -m.re)
                } else {
                    crate::kernel::Complex::new(-m.im, m.re)
                };

                data[k]          = e_k   + p;
                data[k + n4]     = e_k_q + m_rot;
                data[k + n2]     = e_k   - p;
                data[k + n2 + n4] = e_k_q - m_rot;
            }
        }
    };
    expanded
}

/// Generate a specialized 8-point split-radix twiddle codelet (fully unrolled).
///
/// N=8: N/2=4 even, N/4=2 odd-1, N/4=2 odd-3.
/// Unrolls the L-shaped butterfly for k=0,1.
fn gen_split_radix_twiddle_8() -> TokenStream {
    let expanded = quote! {
        /// Split-radix twiddle codelet for N=8 (fully unrolled).
        ///
        /// Combines a 4-point even DFT with two 2-point odd DFTs using
        /// the L-shaped butterfly. All 2 iterations fully unrolled.
        ///
        /// # Data Layout
        /// - `data[0..4]`: 4-point even DFT result (E)
        /// - `data[4..6]`: 2-point odd-1 DFT result (O1)
        /// - `data[6..8]`: 2-point odd-3 DFT result (O3)
        ///
        /// # Arguments
        /// * `data`     - Input/output slice of at least 8 complex values
        /// * `twiddles` - `[W_8^0, W_8^1]` twiddle factors
        /// * `twiddles3`- `[W_8^0, W_8^3]` twiddle factors
        /// * `sign`     - Transform direction: -1 for forward, +1 for inverse
        #[inline(always)]
        pub fn codelet_split_radix_twiddle_8<T: crate::kernel::Float>(
            data: &mut [crate::kernel::Complex<T>],
            twiddles: &[crate::kernel::Complex<T>; 2],
            twiddles3: &[crate::kernel::Complex<T>; 2],
            sign: i32,
        ) {
            debug_assert!(data.len() >= 8);

            // k=0: E[0], E[2], O1[0], O3[0]
            let e0   = data[0];
            let e0_q = data[2];   // E[0 + N/4] = E[2]
            let o1_0 = data[4];
            let o3_0 = data[6];

            let t1_0 = o1_0 * twiddles[0];    // W_8^0 * O1[0]
            let t2_0 = o3_0 * twiddles3[0];   // W_8^0 * O3[0]

            let p0 = t1_0 + t2_0;
            let m0 = t1_0 - t2_0;
            let m0_rot = if sign < 0 {
                crate::kernel::Complex::new(m0.im, -m0.re)
            } else {
                crate::kernel::Complex::new(-m0.im, m0.re)
            };

            // k=1: E[1], E[3], O1[1], O3[1]
            let e1   = data[1];
            let e1_q = data[3];   // E[1 + N/4] = E[3]
            let o1_1 = data[5];
            let o3_1 = data[7];

            let t1_1 = o1_1 * twiddles[1];    // W_8^1 * O1[1]
            let t2_1 = o3_1 * twiddles3[1];   // W_8^3 * O3[1]

            let p1 = t1_1 + t2_1;
            let m1 = t1_1 - t2_1;
            let m1_rot = if sign < 0 {
                crate::kernel::Complex::new(m1.im, -m1.re)
            } else {
                crate::kernel::Complex::new(-m1.im, m1.re)
            };

            // Write all 8 outputs (4 pairs from 2 butterfly iterations)
            data[0] = e0   + p0;       // X[0]
            data[2] = e0_q + m0_rot;   // X[0 + N/4] = X[2]
            data[4] = e0   - p0;       // X[0 + N/2] = X[4]
            data[6] = e0_q - m0_rot;   // X[0 + 3N/4] = X[6]

            data[1] = e1   + p1;       // X[1]
            data[3] = e1_q + m1_rot;   // X[1 + N/4] = X[3]
            data[5] = e1   - p1;       // X[1 + N/2] = X[5]
            data[7] = e1_q - m1_rot;   // X[1 + 3N/4] = X[7]
        }
    };
    expanded
}

/// Generate a specialized 16-point split-radix twiddle codelet (fully unrolled).
///
/// N=16: N/2=8 even, N/4=4 odd-1, N/4=4 odd-3.
/// Unrolls the L-shaped butterfly for k=0,1,2,3.
#[allow(clippy::too_many_lines)]
fn gen_split_radix_twiddle_16() -> TokenStream {
    let expanded = quote! {
        /// Split-radix twiddle codelet for N=16 (fully unrolled).
        ///
        /// Combines an 8-point even DFT with two 4-point odd DFTs using
        /// the L-shaped butterfly. All 4 iterations fully unrolled.
        ///
        /// # Data Layout
        /// - `data[0..8]`:   8-point even DFT result (E)
        /// - `data[8..12]`:  4-point odd-1 DFT result (O1)
        /// - `data[12..16]`: 4-point odd-3 DFT result (O3)
        ///
        /// # Arguments
        /// * `data`     - Input/output slice of at least 16 complex values
        /// * `twiddles` - `[W_16^0, W_16^1, W_16^2, W_16^3]` twiddle factors
        /// * `twiddles3`- `[W_16^0, W_16^3, W_16^6, W_16^9]` twiddle factors
        /// * `sign`     - Transform direction: -1 for forward, +1 for inverse
        #[inline(always)]
        pub fn codelet_split_radix_twiddle_16<T: crate::kernel::Float>(
            data: &mut [crate::kernel::Complex<T>],
            twiddles: &[crate::kernel::Complex<T>; 4],
            twiddles3: &[crate::kernel::Complex<T>; 4],
            sign: i32,
        ) {
            debug_assert!(data.len() >= 16);

            // k=0: E[0], E[4], O1[0], O3[0]
            let e0   = data[0];
            let e0_q = data[4];
            let o1_0 = data[8];
            let o3_0 = data[12];

            let t1_0 = o1_0 * twiddles[0];
            let t2_0 = o3_0 * twiddles3[0];
            let p0 = t1_0 + t2_0;
            let m0 = t1_0 - t2_0;
            let m0_rot = if sign < 0 {
                crate::kernel::Complex::new(m0.im, -m0.re)
            } else {
                crate::kernel::Complex::new(-m0.im, m0.re)
            };

            // k=1: E[1], E[5], O1[1], O3[1]
            let e1   = data[1];
            let e1_q = data[5];
            let o1_1 = data[9];
            let o3_1 = data[13];

            let t1_1 = o1_1 * twiddles[1];
            let t2_1 = o3_1 * twiddles3[1];
            let p1 = t1_1 + t2_1;
            let m1 = t1_1 - t2_1;
            let m1_rot = if sign < 0 {
                crate::kernel::Complex::new(m1.im, -m1.re)
            } else {
                crate::kernel::Complex::new(-m1.im, m1.re)
            };

            // k=2: E[2], E[6], O1[2], O3[2]
            let e2   = data[2];
            let e2_q = data[6];
            let o1_2 = data[10];
            let o3_2 = data[14];

            let t1_2 = o1_2 * twiddles[2];
            let t2_2 = o3_2 * twiddles3[2];
            let p2 = t1_2 + t2_2;
            let m2 = t1_2 - t2_2;
            let m2_rot = if sign < 0 {
                crate::kernel::Complex::new(m2.im, -m2.re)
            } else {
                crate::kernel::Complex::new(-m2.im, m2.re)
            };

            // k=3: E[3], E[7], O1[3], O3[3]
            let e3   = data[3];
            let e3_q = data[7];
            let o1_3 = data[11];
            let o3_3 = data[15];

            let t1_3 = o1_3 * twiddles[3];
            let t2_3 = o3_3 * twiddles3[3];
            let p3 = t1_3 + t2_3;
            let m3 = t1_3 - t2_3;
            let m3_rot = if sign < 0 {
                crate::kernel::Complex::new(m3.im, -m3.re)
            } else {
                crate::kernel::Complex::new(-m3.im, m3.re)
            };

            // Write all 16 outputs
            // Quarter 0: X[k]
            data[0]  = e0 + p0;
            data[1]  = e1 + p1;
            data[2]  = e2 + p2;
            data[3]  = e3 + p3;

            // Quarter 1: X[k + N/4]
            data[4]  = e0_q + m0_rot;
            data[5]  = e1_q + m1_rot;
            data[6]  = e2_q + m2_rot;
            data[7]  = e3_q + m3_rot;

            // Quarter 2: X[k + N/2]
            data[8]  = e0 - p0;
            data[9]  = e1 - p1;
            data[10] = e2 - p2;
            data[11] = e3 - p3;

            // Quarter 3: X[k + 3N/4]
            data[12] = e0_q - m0_rot;
            data[13] = e1_q - m1_rot;
            data[14] = e2_q - m2_rot;
            data[15] = e3_q - m3_rot;
        }
    };
    expanded
}
