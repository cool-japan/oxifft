//! Compile-time FFT with const generics.
//!
//! This module provides FFT implementations where the size is known at compile time,
//! enabling aggressive optimizations such as:
//! - No heap allocation (stack-based arrays)
//! - Inlined twiddle factors
//! - Full loop unrolling by LLVM
//! - ~20-30% faster for small fixed sizes
//!
//! # Example
//!
//! ```ignore
//! use oxifft::const_fft::{fft_fixed, ifft_fixed};
//! use oxifft::Complex;
//!
//! // All sizes known at compile time
//! let mut input: [Complex<f64>; 8] = [Complex::new(1.0, 0.0); 8];
//! let output = fft_fixed(&input);
//! let recovered = ifft_fixed(&output);
//! ```

#[cfg(not(feature = "std"))]
extern crate alloc;

mod radix2;
mod twiddle;

pub use radix2::{fft_fixed, fft_fixed_inplace, ifft_fixed, ifft_fixed_inplace};
pub use twiddle::{const_cos, const_sin, twiddle_factor};

use crate::kernel::Complex;

/// Trait for compile-time FFT operations.
///
/// Implementations are provided for power-of-2 sizes up to 1024.
pub trait ConstFft<const N: usize> {
    /// Perform forward FFT on a fixed-size array.
    fn fft(input: &[Complex<f64>; N]) -> [Complex<f64>; N];

    /// Perform inverse FFT on a fixed-size array.
    fn ifft(input: &[Complex<f64>; N]) -> [Complex<f64>; N];

    /// Perform in-place forward FFT.
    fn fft_inplace(data: &mut [Complex<f64>; N]);

    /// Perform in-place inverse FFT.
    fn ifft_inplace(data: &mut [Complex<f64>; N]);
}

/// Marker struct for const FFT implementations.
pub struct ConstFftImpl;

// Implement ConstFft for power-of-2 sizes
macro_rules! impl_const_fft {
    ($n:expr) => {
        impl ConstFft<$n> for ConstFftImpl {
            #[inline]
            fn fft(input: &[Complex<f64>; $n]) -> [Complex<f64>; $n] {
                radix2::fft_fixed(input)
            }

            #[inline]
            fn ifft(input: &[Complex<f64>; $n]) -> [Complex<f64>; $n] {
                radix2::ifft_fixed(input)
            }

            #[inline]
            fn fft_inplace(data: &mut [Complex<f64>; $n]) {
                radix2::fft_fixed_inplace(data);
            }

            #[inline]
            fn ifft_inplace(data: &mut [Complex<f64>; $n]) {
                radix2::ifft_fixed_inplace(data);
            }
        }
    };
}

impl_const_fft!(2);
impl_const_fft!(4);
impl_const_fft!(8);
impl_const_fft!(16);
impl_const_fft!(32);
impl_const_fft!(64);
impl_const_fft!(128);
impl_const_fft!(256);
impl_const_fft!(512);
impl_const_fft!(1024);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_fft_impulse_8() {
        // Impulse at position 0 should give all ones
        let mut input = [Complex::<f64>::zero(); 8];
        input[0] = Complex::new(1.0, 0.0);

        let output = fft_fixed(&input);

        // All frequency bins should be 1+0i
        for c in &output {
            assert!((c.re - 1.0).abs() < 1e-10);
            assert!(c.im.abs() < 1e-10);
        }
    }

    #[test]
    fn test_const_fft_roundtrip_8() {
        let input: [Complex<f64>; 8] = [
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(7.0, 0.0),
            Complex::new(8.0, 0.0),
        ];

        let spectrum = fft_fixed(&input);
        let recovered = ifft_fixed(&spectrum);

        for i in 0..8 {
            assert!(
                (recovered[i].re - input[i].re).abs() < 1e-10,
                "Real part mismatch at {}: {} vs {}",
                i,
                recovered[i].re,
                input[i].re
            );
            assert!(
                (recovered[i].im - input[i].im).abs() < 1e-10,
                "Imag part mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_const_fft_dc_4() {
        // Constant signal should have DC at bin 0
        let input = [Complex::new(3.0, 0.0); 4];
        let output = fft_fixed(&input);

        // DC = sum = 12
        assert!((output[0].re - 12.0).abs() < 1e-10);
        assert!(output[0].im.abs() < 1e-10);

        // Other bins should be 0
        for i in 1..4 {
            assert!(output[i].re.abs() < 1e-10);
            assert!(output[i].im.abs() < 1e-10);
        }
    }

    #[test]
    fn test_const_fft_inplace_16() {
        let original: [Complex<f64>; 16] =
            core::array::from_fn(|i| Complex::new((i as f64 / 5.0).sin(), 0.0));

        let mut data = original;
        fft_fixed_inplace(&mut data);
        ifft_fixed_inplace(&mut data);

        for i in 0..16 {
            assert!(
                (data[i].re - original[i].re).abs() < 1e-10,
                "Mismatch at {}: {} vs {}",
                i,
                data[i].re,
                original[i].re
            );
        }
    }

    #[test]
    fn test_const_fft_trait_64() {
        let input: [Complex<f64>; 64] =
            core::array::from_fn(|i| Complex::new(if i == 0 { 1.0 } else { 0.0 }, 0.0));

        let output = <ConstFftImpl as ConstFft<64>>::fft(&input);

        // Impulse should give flat spectrum
        for c in &output {
            assert!((c.re - 1.0).abs() < 1e-10);
            assert!(c.im.abs() < 1e-10);
        }
    }
}
