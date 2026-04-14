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
    // Size-8 DFT using radix-2 DIT with explicit butterfly stages.
    // All constants pre-computed via T::from_f64() to avoid trait method ambiguity.
    let expanded = quote! {
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
    };
    TokenStream::from(expanded)
}

#[allow(clippy::too_many_lines)]
fn gen_size_16() -> TokenStream {
    // Size-16 DFT using radix-2 DIT with explicit butterfly stages.
    // All constants pre-computed via T::from_f64() to avoid trait method ambiguity.
    let expanded = quote! {
        /// Size-16 DFT codelet using radix-2 DIT decomposition.
        ///
        /// Inputs are taken in natural order, output is in natural order.
        #[inline(always)]
        pub fn codelet_notw_16<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 16);

            // cos(π/8), sin(π/8), 1/sqrt(2)
            let c1 = T::from_f64(0.923_879_532_511_286_7_f64); // cos(π/8)
            let s1 = T::from_f64(0.382_683_432_365_089_8_f64); // sin(π/8)
            let c2 = T::from_f64(0.707_106_781_186_547_6_f64); // 1/sqrt(2) = cos(π/4)
            let c3 = s1;  // cos(3π/8) = sin(π/8)
            let s3 = c1;  // sin(3π/8) = cos(π/8)

            // Bit-reversal permutation for DIT (4-bit reversal)
            let mut a = [crate::kernel::Complex::<T>::zero(); 16];
            a[0]  = x[0];  a[1]  = x[8];
            a[2]  = x[4];  a[3]  = x[12];
            a[4]  = x[2];  a[5]  = x[10];
            a[6]  = x[6];  a[7]  = x[14];
            a[8]  = x[1];  a[9]  = x[9];
            a[10] = x[5];  a[11] = x[13];
            a[12] = x[3];  a[13] = x[11];
            a[14] = x[7];  a[15] = x[15];

            // DIT Stage 1: 8 butterflies, span 1 (W2^0 = 1, no twiddle needed)
            for i in (0..16usize).step_by(2) {
                let t = a[i + 1];
                a[i + 1] = a[i] - t;
                a[i]     = a[i] + t;
            }

            // DIT Stage 2: 4 groups of 2 butterflies, span 2
            // W4^0 = 1, W4^1 = -i (forward) or +i (inverse)
            for group in (0..16usize).step_by(4) {
                let t = a[group + 2];
                a[group + 2] = a[group] - t;
                a[group]     = a[group] + t;

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
            for group in (0..16usize).step_by(8) {
                // k=0: W8^0 = 1
                let t = a[group + 4];
                a[group + 4] = a[group] - t;
                a[group]     = a[group] + t;

                // k=1: W8^1
                let t = a[group + 5];
                let t_tw = if sign < 0 {
                    crate::kernel::Complex::new((t.re + t.im) * c2, (t.im - t.re) * c2)
                } else {
                    crate::kernel::Complex::new((t.re - t.im) * c2, (t.im + t.re) * c2)
                };
                a[group + 5] = a[group + 1] - t_tw;
                a[group + 1] = a[group + 1] + t_tw;

                // k=2: W8^2 = -i (forward) or +i (inverse)
                let t = a[group + 6];
                let t_tw = if sign < 0 {
                    crate::kernel::Complex::new(t.im, -t.re)
                } else {
                    crate::kernel::Complex::new(-t.im, t.re)
                };
                a[group + 6] = a[group + 2] - t_tw;
                a[group + 2] = a[group + 2] + t_tw;

                // k=3: W8^3
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
            // k=0: W16^0 = 1
            let t = a[8];
            a[8] = a[0] - t;
            a[0] = a[0] + t;

            // k=1: W16^1 = cos(π/8) - i*sin(π/8) (forward)
            let t = a[9];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new(t.re * c1 + t.im * s1, t.im * c1 - t.re * s1)
            } else {
                crate::kernel::Complex::new(t.re * c1 - t.im * s1, t.im * c1 + t.re * s1)
            };
            a[9] = a[1] - t_tw;
            a[1] = a[1] + t_tw;

            // k=2: W16^2 = (1-i)/sqrt(2) (forward)
            let t = a[10];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new((t.re + t.im) * c2, (t.im - t.re) * c2)
            } else {
                crate::kernel::Complex::new((t.re - t.im) * c2, (t.im + t.re) * c2)
            };
            a[10] = a[2] - t_tw;
            a[2] = a[2] + t_tw;

            // k=3: W16^3 = cos(3π/8) - i*sin(3π/8) (forward)
            let t = a[11];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new(t.re * c3 + t.im * s3, t.im * c3 - t.re * s3)
            } else {
                crate::kernel::Complex::new(t.re * c3 - t.im * s3, t.im * c3 + t.re * s3)
            };
            a[11] = a[3] - t_tw;
            a[3] = a[3] + t_tw;

            // k=4: W16^4 = -i (forward) or +i (inverse)
            let t = a[12];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new(t.im, -t.re)
            } else {
                crate::kernel::Complex::new(-t.im, t.re)
            };
            a[12] = a[4] - t_tw;
            a[4] = a[4] + t_tw;

            // k=5: W16^5 = -sin(π/8) - i*cos(π/8) (forward)
            let t = a[13];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new(-t.re * s1 + t.im * c1, -t.im * s1 - t.re * c1)
            } else {
                crate::kernel::Complex::new(-t.re * s1 - t.im * c1, -t.im * s1 + t.re * c1)
            };
            a[13] = a[5] - t_tw;
            a[5] = a[5] + t_tw;

            // k=6: W16^6 = (-1-i)/sqrt(2) (forward)
            let t = a[14];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new((-t.re + t.im) * c2, (-t.im - t.re) * c2)
            } else {
                crate::kernel::Complex::new((-t.re - t.im) * c2, (-t.im + t.re) * c2)
            };
            a[14] = a[6] - t_tw;
            a[6] = a[6] + t_tw;

            // k=7: W16^7 = -cos(π/8) - i*sin(π/8) (forward)
            let t = a[15];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new(-t.re * c1 + t.im * s1, -t.im * c1 - t.re * s1)
            } else {
                crate::kernel::Complex::new(-t.re * c1 - t.im * s1, -t.im * c1 + t.re * s1)
            };
            a[15] = a[7] - t_tw;
            a[7] = a[7] + t_tw;

            // Write back in natural order
            for i in 0..16usize {
                x[i] = a[i];
            }
        }
    };
    TokenStream::from(expanded)
}

fn gen_size_32() -> TokenStream {
    // Size-32 DFT using radix-2 DIF (decimation-in-frequency) decomposition.
    // Twiddle factors computed via T::from_f64() to avoid trait method ambiguity.
    let expanded = quote! {
        /// Size-32 DFT codelet using radix-2 DIF decomposition.
        #[inline(always)]
        pub fn codelet_notw_32<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 32);

            // Helper: compute W_N^k = e^(sign * 2πi*k/N)
            // sign<0 → forward (e^{-2πi k/N}), sign>0 → inverse (e^{+2πi k/N})
            let twiddle = |k: usize, n: usize| -> crate::kernel::Complex<T> {
                let angle = if sign < 0 {
                    -2.0_f64 * core::f64::consts::PI * (k as f64) / (n as f64)
                } else {
                    2.0_f64 * core::f64::consts::PI * (k as f64) / (n as f64)
                };
                crate::kernel::Complex::new(T::from_f64(angle.cos()), T::from_f64(angle.sin()))
            };

            // Stage 1: combine pairs with distance 16
            for k in 0..16usize {
                let tw = twiddle(k, 32);
                let u = x[k];
                let v = x[k + 16];
                x[k]      = u + v;
                x[k + 16] = (u - v) * tw;
            }

            // Stage 2: combine pairs with distance 8 in each half
            for base in [0usize, 16] {
                for k in 0..8usize {
                    let tw = twiddle(k, 16);
                    let u = x[base + k];
                    let v = x[base + k + 8];
                    x[base + k]     = u + v;
                    x[base + k + 8] = (u - v) * tw;
                }
            }

            // Stage 3: combine pairs with distance 4 in each quarter
            for base in [0usize, 8, 16, 24] {
                for k in 0..4usize {
                    let tw = twiddle(k, 8);
                    let u = x[base + k];
                    let v = x[base + k + 4];
                    x[base + k]     = u + v;
                    x[base + k + 4] = (u - v) * tw;
                }
            }

            // Stage 4: combine pairs with distance 2 in each eighth
            for base in (0..32usize).step_by(4) {
                for k in 0..2usize {
                    let tw = twiddle(k, 4);
                    let u = x[base + k];
                    let v = x[base + k + 2];
                    x[base + k]     = u + v;
                    x[base + k + 2] = (u - v) * tw;
                }
            }

            // Stage 5: final butterflies (W2^0 = 1, W2^1 = -1)
            for base in (0..32usize).step_by(2) {
                let u = x[base];
                let v = x[base + 1];
                x[base]     = u + v;
                x[base + 1] = u - v;
            }

            // Bit-reversal permutation for DIF output
            let indices: [usize; 32] = [
                0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26,  6, 22, 14, 30,
                1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27,  7, 23, 15, 31
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
    // Size-64 DFT using radix-2 DIF (decimation-in-frequency) decomposition.
    // Twiddle factors computed via T::from_f64() to avoid trait method ambiguity.
    let expanded = quote! {
        /// Size-64 DFT codelet using radix-2 DIF decomposition.
        #[inline(always)]
        pub fn codelet_notw_64<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 64);

            // Helper: compute W_N^k = e^(sign * 2πi*k/N)
            let twiddle = |k: usize, n: usize| -> crate::kernel::Complex<T> {
                let angle = if sign < 0 {
                    -2.0_f64 * core::f64::consts::PI * (k as f64) / (n as f64)
                } else {
                    2.0_f64 * core::f64::consts::PI * (k as f64) / (n as f64)
                };
                crate::kernel::Complex::new(T::from_f64(angle.cos()), T::from_f64(angle.sin()))
            };

            // Stage 1: combine pairs with distance 32
            for k in 0..32usize {
                let tw = twiddle(k, 64);
                let u = x[k];
                let v = x[k + 32];
                x[k]      = u + v;
                x[k + 32] = (u - v) * tw;
            }

            // Stage 2: combine pairs with distance 16 in each half
            for base in [0usize, 32] {
                for k in 0..16usize {
                    let tw = twiddle(k, 32);
                    let u = x[base + k];
                    let v = x[base + k + 16];
                    x[base + k]      = u + v;
                    x[base + k + 16] = (u - v) * tw;
                }
            }

            // Stage 3: combine pairs with distance 8
            for base in (0..64usize).step_by(16) {
                for k in 0..8usize {
                    let tw = twiddle(k, 16);
                    let u = x[base + k];
                    let v = x[base + k + 8];
                    x[base + k]     = u + v;
                    x[base + k + 8] = (u - v) * tw;
                }
            }

            // Stage 4: combine pairs with distance 4
            for base in (0..64usize).step_by(8) {
                for k in 0..4usize {
                    let tw = twiddle(k, 8);
                    let u = x[base + k];
                    let v = x[base + k + 4];
                    x[base + k]     = u + v;
                    x[base + k + 4] = (u - v) * tw;
                }
            }

            // Stage 5: combine pairs with distance 2
            for base in (0..64usize).step_by(4) {
                for k in 0..2usize {
                    let tw = twiddle(k, 4);
                    let u = x[base + k];
                    let v = x[base + k + 2];
                    x[base + k]     = u + v;
                    x[base + k + 2] = (u - v) * tw;
                }
            }

            // Stage 6: final butterflies (W2^0 = 1, W2^1 = -1)
            for base in (0..64usize).step_by(2) {
                let u = x[base];
                let v = x[base + 1];
                x[base]     = u + v;
                x[base + 1] = u - v;
            }

            // Bit-reversal permutation for DIF output (6-bit reversal)
            let mut temp = [crate::kernel::Complex::<T>::zero(); 64];
            for i in 0..64usize {
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
