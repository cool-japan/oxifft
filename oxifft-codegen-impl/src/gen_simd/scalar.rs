//! Scalar fallback codelet emitters (generic over any Float type).

use proc_macro2::TokenStream;
use quote::quote;

/// Scalar size-2 butterfly (generic over any Float type).
pub(super) fn gen_scalar_size_2() -> TokenStream {
    quote! {
        /// Size-2 SIMD codelet scalar fallback.
        #[inline(always)]
        fn codelet_simd_2_scalar<T: crate::kernel::Float>(
            data: &mut [crate::kernel::Complex<T>],
            _sign: i32,
        ) {
            let a = data[0];
            let b = data[1];
            data[0] = a + b;
            data[1] = a - b;
        }
    }
}

/// Scalar size-4 radix-4 butterfly (generic).
pub(super) fn gen_scalar_size_4() -> TokenStream {
    quote! {
        /// Size-4 SIMD codelet scalar fallback.
        #[inline(always)]
        fn codelet_simd_4_scalar<T: crate::kernel::Float>(
            data: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            let x0 = data[0];
            let x1 = data[1];
            let x2 = data[2];
            let x3 = data[3];

            // Stage 1: pair-wise butterflies
            let t0 = x0 + x2;
            let t1 = x0 - x2;
            let t2 = x1 + x3;
            let t3 = x1 - x3;

            // Rotate t3 by ±i depending on direction
            let t3_rot = if sign < 0 {
                // Forward: multiply by -i -> (im, -re)
                crate::kernel::Complex::new(t3.im, -t3.re)
            } else {
                // Inverse: multiply by +i -> (-im, re)
                crate::kernel::Complex::new(-t3.im, t3.re)
            };

            // Stage 2: final butterflies
            data[0] = t0 + t2;
            data[1] = t1 + t3_rot;
            data[2] = t0 - t2;
            data[3] = t1 - t3_rot;
        }
    }
}

/// Scalar size-8 radix-2 DIT butterfly (generic).
pub(super) fn gen_scalar_size_8() -> TokenStream {
    quote! {
        /// Size-8 SIMD codelet scalar fallback.
        #[inline(always)]
        fn codelet_simd_8_scalar<T: crate::kernel::Float>(
            data: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            let c2 = T::from_f64(0.707_106_781_186_547_6_f64);

            // Bit-reversal permutation (3-bit)
            let mut a = [crate::kernel::Complex::<T>::zero(); 8];
            a[0] = data[0]; a[1] = data[4];
            a[2] = data[2]; a[3] = data[6];
            a[4] = data[1]; a[5] = data[5];
            a[6] = data[3]; a[7] = data[7];

            // Stage 1: span-1 butterflies
            for i in (0..8usize).step_by(2) {
                let t = a[i + 1];
                a[i + 1] = a[i] - t;
                a[i]     = a[i] + t;
            }

            // Stage 2: span-2 butterflies with W4 twiddles
            for group in (0..8usize).step_by(4) {
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

            // Stage 3: span-4 butterflies with W8 twiddles
            // k=0: W8^0 = 1
            let t = a[4];
            a[4] = a[0] - t;
            a[0] = a[0] + t;

            // k=1: W8^1
            let t = a[5];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new((t.re + t.im) * c2, (t.im - t.re) * c2)
            } else {
                crate::kernel::Complex::new((t.re - t.im) * c2, (t.im + t.re) * c2)
            };
            a[5] = a[1] - t_tw;
            a[1] = a[1] + t_tw;

            // k=2: W8^2 = ∓i
            let t = a[6];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new(t.im, -t.re)
            } else {
                crate::kernel::Complex::new(-t.im, t.re)
            };
            a[6] = a[2] - t_tw;
            a[2] = a[2] + t_tw;

            // k=3: W8^3
            let t = a[7];
            let t_tw = if sign < 0 {
                crate::kernel::Complex::new((-t.re + t.im) * c2, (-t.im - t.re) * c2)
            } else {
                crate::kernel::Complex::new((-t.re - t.im) * c2, (-t.im + t.re) * c2)
            };
            a[7] = a[3] - t_tw;
            a[3] = a[3] + t_tw;

            for i in 0..8usize {
                data[i] = a[i];
            }
        }
    }
}

/// Scalar size-16 radix-2 DIT butterfly (generic, 4 stages).
///
/// Used as fallback for size-16 when AVX-512F is not available.
#[allow(clippy::too_many_lines)]
pub(super) fn gen_scalar_size_16() -> TokenStream {
    quote! {
        /// Size-16 SIMD codelet scalar fallback.
        #[inline(always)]
        #[allow(clippy::too_many_lines)]
        fn codelet_simd_16_scalar<T: crate::kernel::Float>(
            data: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            use crate::kernel::Complex;

            let sign_f = if sign < 0 { T::from_f64(-1.0) } else { T::from_f64(1.0) };
            let two_pi = T::from_f64(2.0 * core::f64::consts::PI);

            // Bit-reversal permutation for N=16 (4-bit reversal)
            // Mapping: 0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15
            let mut a = [Complex::<T>::zero(); 16];
            a[0]  = data[0];   a[1]  = data[8];
            a[2]  = data[4];   a[3]  = data[12];
            a[4]  = data[2];   a[5]  = data[10];
            a[6]  = data[6];   a[7]  = data[14];
            a[8]  = data[1];   a[9]  = data[9];
            a[10] = data[5];   a[11] = data[13];
            a[12] = data[3];   a[13] = data[11];
            a[14] = data[7];   a[15] = data[15];

            // Stage 1: span-1
            for i in (0..16usize).step_by(2) {
                let t = a[i + 1];
                a[i + 1] = a[i] - t;
                a[i]     = a[i] + t;
            }

            // Stage 2: span-2, W4 twiddles
            for group in (0..16usize).step_by(4) {
                let t = a[group + 2];
                a[group + 2] = a[group] - t;
                a[group]     = a[group] + t;

                let t = a[group + 3];
                let t_tw = if sign < 0 {
                    Complex::new(t.im, -t.re)
                } else {
                    Complex::new(-t.im, t.re)
                };
                a[group + 3] = a[group + 1] - t_tw;
                a[group + 1] = a[group + 1] + t_tw;
            }

            // Stage 3: span-4, W8 twiddles (k=0..3 repeated for two groups)
            let c2 = T::from_f64(0.707_106_781_186_547_6_f64);
            for group in (0..16usize).step_by(8) {
                // k=0: trivial
                let t = a[group + 4];
                a[group + 4] = a[group] - t;
                a[group]     = a[group] + t;

                // k=1: W8^1
                let t = a[group + 5];
                let t_tw = if sign < 0 {
                    Complex::new((t.re + t.im) * c2, (t.im - t.re) * c2)
                } else {
                    Complex::new((t.re - t.im) * c2, (t.im + t.re) * c2)
                };
                a[group + 5] = a[group + 1] - t_tw;
                a[group + 1] = a[group + 1] + t_tw;

                // k=2: ∓i
                let t = a[group + 6];
                let t_tw = if sign < 0 {
                    Complex::new(t.im, -t.re)
                } else {
                    Complex::new(-t.im, t.re)
                };
                a[group + 6] = a[group + 2] - t_tw;
                a[group + 2] = a[group + 2] + t_tw;

                // k=3: W8^3
                let t = a[group + 7];
                let t_tw = if sign < 0 {
                    Complex::new((-t.re + t.im) * c2, (-t.im - t.re) * c2)
                } else {
                    Complex::new((-t.re - t.im) * c2, (-t.im + t.re) * c2)
                };
                a[group + 7] = a[group + 3] - t_tw;
                a[group + 3] = a[group + 3] + t_tw;
            }

            // Stage 4: span-8, W16^k twiddles (k=0..7)
            for k in 0..8usize {
                let angle = sign_f * two_pi * T::from_f64(k as f64) / T::from_f64(16.0);
                let tw = Complex::new(
                    crate::kernel::Float::cos(angle),
                    crate::kernel::Float::sin(angle),
                );
                let t_tw = a[k + 8] * tw;
                a[k + 8] = a[k] - t_tw;
                a[k]     = a[k] + t_tw;
            }

            for i in 0..16usize {
                data[i] = a[i];
            }
        }
    }
}
