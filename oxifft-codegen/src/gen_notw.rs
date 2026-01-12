//! Non-twiddle codelet generation.
//!
//! Generates optimized base-case FFT kernels using symbolic computation,
//! common subexpression elimination, and strength reduction.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, LitInt};

/// Generate a non-twiddle codelet for the given size.
pub fn generate(input: TokenStream) -> TokenStream {
    let size = parse_macro_input!(input as LitInt);
    let n: usize = size.base10_parse().expect("Invalid size literal");

    match n {
        2 => gen_size_2(),
        4 => gen_size_4(),
        8 => gen_size_8(),
        16 => gen_size_16(),
        32 => gen_size_32(),
        64 => gen_size_64(),
        _ => panic!("Unsupported codelet size: {n}. Use 2, 4, 8, 16, 32, or 64."),
    }
}

fn gen_size_2() -> TokenStream {
    let expanded = quote! {
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
    };
    TokenStream::from(expanded)
}

fn gen_size_4() -> TokenStream {
    let expanded = quote! {
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
    };
    TokenStream::from(expanded)
}

fn gen_size_8() -> TokenStream {
    let expanded = quote! {
        /// Size-8 DFT codelet.
        #[inline(always)]
        pub fn codelet_notw_8<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 8);

            // Load inputs
            let x0 = x[0]; let x1 = x[1]; let x2 = x[2]; let x3 = x[3];
            let x4 = x[4]; let x5 = x[5]; let x6 = x[6]; let x7 = x[7];

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

            // W8 = (1-i)/sqrt(2) or (1+i)/sqrt(2)
            let sqrt2_inv = T::ONE / T::TWO.sqrt();
            let w8_re = sqrt2_inv;
            let w8_im = if sign < 0 { -sqrt2_inv } else { sqrt2_inv };

            // Apply W8 twiddles
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
    };
    TokenStream::from(expanded)
}

fn gen_size_16() -> TokenStream {
    // Size-16 DFT using 2x size-8 + twiddles
    // Decimation-in-frequency: split into even/odd, apply size-8, combine with twiddles
    let expanded = quote! {
        /// Size-16 DFT codelet using radix-2 DIF decomposition.
        #[inline(always)]
        pub fn codelet_notw_16<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 16);

            let pi = T::PI;
            let sign_t = if sign < 0 { T::NEG_ONE } else { T::ONE };

            // First stage: combine pairs with distance 8
            for k in 0..8 {
                let angle = sign_t * T::TWO * pi * T::from_usize(k) / T::from_usize(16);
                let tw = crate::kernel::Complex::new(angle.cos(), angle.sin());
                let u = x[k];
                let v = x[k + 8];
                x[k] = u + v;
                x[k + 8] = (u - v) * tw;
            }

            // Two independent size-8 FFTs
            fft8_inplace(&mut x[0..8], sign);
            fft8_inplace(&mut x[8..16], sign);

            // Bit-reversal permutation for size-16 DIF
            let indices: [usize; 16] = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];
            let mut temp = [crate::kernel::Complex::<T>::zero(); 16];
            for (i, &idx) in indices.iter().enumerate() {
                temp[i] = x[idx];
            }
            x[..16].copy_from_slice(&temp);
        }

        /// Inline size-8 FFT helper for size-16 codelet.
        #[inline(always)]
        fn fft8_inplace<T: crate::kernel::Float>(x: &mut [crate::kernel::Complex<T>], sign: i32) {
            let x0 = x[0]; let x1 = x[1]; let x2 = x[2]; let x3 = x[3];
            let x4 = x[4]; let x5 = x[5]; let x6 = x[6]; let x7 = x[7];

            let t0 = x0 + x4; let t1 = x0 - x4;
            let t2 = x2 + x6; let t3 = x2 - x6;
            let t4 = x1 + x5; let t5 = x1 - x5;
            let t6 = x3 + x7; let t7 = x3 - x7;

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

            let u0 = t0 + t2; let u1 = t0 - t2;
            let u2 = t4 + t6; let u3 = t4 - t6;
            let u4 = t1 + t3_rot; let u5 = t1 - t3_rot;
            let u6 = t5 + t7_rot; let u7 = t5 - t7_rot;

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

            x[0] = u0 + u2; x[4] = u0 - u2;
            x[2] = u1 + u3_rot; x[6] = u1 - u3_rot;
            x[1] = u4 + u6_tw; x[5] = u4 - u6_tw;
            x[3] = u5 + u7_tw; x[7] = u5 - u7_tw;
        }
    };
    TokenStream::from(expanded)
}

fn gen_size_32() -> TokenStream {
    // Size-32 DFT using radix-2 decomposition
    let expanded = quote! {
        /// Size-32 DFT codelet using radix-2 DIF decomposition.
        #[inline(always)]
        pub fn codelet_notw_32<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 32);

            let pi = T::PI;
            let sign_t = if sign < 0 { T::NEG_ONE } else { T::ONE };

            // Stage 1: combine pairs with distance 16
            for k in 0..16 {
                let angle = sign_t * T::TWO * pi * T::from_usize(k) / T::from_usize(32);
                let tw = crate::kernel::Complex::new(angle.cos(), angle.sin());

                let u = x[k];
                let v = x[k + 16];

                x[k] = u + v;
                x[k + 16] = (u - v) * tw;
            }

            // Stage 2: combine pairs with distance 8 in each half
            for base in [0, 16] {
                for k in 0..8 {
                    let angle = sign_t * T::TWO * pi * T::from_usize(k) / T::from_usize(16);
                    let tw = crate::kernel::Complex::new(angle.cos(), angle.sin());

                    let u = x[base + k];
                    let v = x[base + k + 8];

                    x[base + k] = u + v;
                    x[base + k + 8] = (u - v) * tw;
                }
            }

            // Stage 3: combine pairs with distance 4 in each quarter
            for base in [0, 8, 16, 24] {
                for k in 0..4 {
                    let angle = sign_t * T::TWO * pi * T::from_usize(k) / T::from_usize(8);
                    let tw = crate::kernel::Complex::new(angle.cos(), angle.sin());

                    let u = x[base + k];
                    let v = x[base + k + 4];

                    x[base + k] = u + v;
                    x[base + k + 4] = (u - v) * tw;
                }
            }

            // Stage 4: combine pairs with distance 2 in each eighth
            for base in (0..32).step_by(4) {
                for k in 0..2 {
                    let angle = sign_t * T::TWO * pi * T::from_usize(k) / T::from_usize(4);
                    let tw = crate::kernel::Complex::new(angle.cos(), angle.sin());

                    let u = x[base + k];
                    let v = x[base + k + 2];

                    x[base + k] = u + v;
                    x[base + k + 2] = (u - v) * tw;
                }
            }

            // Stage 5: final butterflies
            for base in (0..32).step_by(2) {
                let u = x[base];
                let v = x[base + 1];
                x[base] = u + v;
                x[base + 1] = u - v;
            }

            // Bit-reversal permutation
            let indices: [usize; 32] = [
                0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30,
                1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31
            ];

            let mut temp = [crate::kernel::Complex::<T>::zero(); 32];
            for (i, &idx) in indices.iter().enumerate() {
                temp[i] = x[idx];
            }
            x[..32].copy_from_slice(&temp);
        }
    };
    TokenStream::from(expanded)
}

fn gen_size_64() -> TokenStream {
    // Size-64 DFT using radix-2 decomposition
    let expanded = quote! {
        /// Size-64 DFT codelet using radix-2 DIF decomposition.
        #[inline(always)]
        pub fn codelet_notw_64<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 64);

            let pi = T::PI;
            let sign_t = if sign < 0 { T::NEG_ONE } else { T::ONE };

            // Stage 1: combine pairs with distance 32
            for k in 0..32 {
                let angle = sign_t * T::TWO * pi * T::from_usize(k) / T::from_usize(64);
                let tw = crate::kernel::Complex::new(angle.cos(), angle.sin());

                let u = x[k];
                let v = x[k + 32];

                x[k] = u + v;
                x[k + 32] = (u - v) * tw;
            }

            // Stage 2: combine pairs with distance 16 in each half
            for base in [0, 32] {
                for k in 0..16 {
                    let angle = sign_t * T::TWO * pi * T::from_usize(k) / T::from_usize(32);
                    let tw = crate::kernel::Complex::new(angle.cos(), angle.sin());

                    let u = x[base + k];
                    let v = x[base + k + 16];

                    x[base + k] = u + v;
                    x[base + k + 16] = (u - v) * tw;
                }
            }

            // Stage 3: combine pairs with distance 8
            for base in (0..64).step_by(16) {
                for k in 0..8 {
                    let angle = sign_t * T::TWO * pi * T::from_usize(k) / T::from_usize(16);
                    let tw = crate::kernel::Complex::new(angle.cos(), angle.sin());

                    let u = x[base + k];
                    let v = x[base + k + 8];

                    x[base + k] = u + v;
                    x[base + k + 8] = (u - v) * tw;
                }
            }

            // Stage 4: combine pairs with distance 4
            for base in (0..64).step_by(8) {
                for k in 0..4 {
                    let angle = sign_t * T::TWO * pi * T::from_usize(k) / T::from_usize(8);
                    let tw = crate::kernel::Complex::new(angle.cos(), angle.sin());

                    let u = x[base + k];
                    let v = x[base + k + 4];

                    x[base + k] = u + v;
                    x[base + k + 4] = (u - v) * tw;
                }
            }

            // Stage 5: combine pairs with distance 2
            for base in (0..64).step_by(4) {
                for k in 0..2 {
                    let angle = sign_t * T::TWO * pi * T::from_usize(k) / T::from_usize(4);
                    let tw = crate::kernel::Complex::new(angle.cos(), angle.sin());

                    let u = x[base + k];
                    let v = x[base + k + 2];

                    x[base + k] = u + v;
                    x[base + k + 2] = (u - v) * tw;
                }
            }

            // Stage 6: final butterflies
            for base in (0..64).step_by(2) {
                let u = x[base];
                let v = x[base + 1];
                x[base] = u + v;
                x[base + 1] = u - v;
            }

            // Bit-reversal permutation
            // Generate bit-reversal indices for size 64
            let mut temp = [crate::kernel::Complex::<T>::zero(); 64];
            for i in 0..64 {
                let mut j = i;
                let mut rev = 0usize;
                for _ in 0..6 {
                    rev = (rev << 1) | (j & 1);
                    j >>= 1;
                }
                temp[i] = x[rev];
            }
            x[..64].copy_from_slice(&temp);
        }
    };
    TokenStream::from(expanded)
}
