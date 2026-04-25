//! Stockham Auto-Sort FFT implementation.
//!
//! The Stockham algorithm eliminates bit-reversal permutation by using
//! two buffers (ping-pong). Each stage reads with fixed stride n/2 and
//! writes with a pattern that progressively sorts the data.
//!
//! This is significantly faster for large sizes (512+) where bit-reversal
//! becomes a bottleneck due to non-sequential memory access.

mod generic;

pub use generic::{stockham_radix4_scalar, stockham_scalar};

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "x86_64")]
mod x86_64;

use crate::dft::problem::Sign;
use crate::kernel::{Complex, Float};
#[allow(unused_imports)]
// reason: prelude glob re-exports are selectively used per feature gate (std vs no_std)
use crate::prelude::*;

// Small size functions are used internally by the scalar fallback path

/// Stockham FFT solver for power-of-2 sizes.
///
/// Uses ping-pong buffers to avoid bit-reversal permutation.
/// Optimal for sizes >= 512 where bit-reversal overhead is significant.
pub struct StockhamSolver<T: Float> {
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> Default for StockhamSolver<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> StockhamSolver<T> {
    /// Create a new Stockham solver.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }

    /// Solver name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        "dft-stockham"
    }

    /// Check if size is a power of 2.
    #[must_use]
    pub fn applicable(n: usize) -> bool {
        n.is_power_of_two()
    }

    /// Execute Stockham FFT (out-of-place).
    ///
    /// The output is in natural order (no bit-reversal needed).
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        let n = input.len();
        debug_assert_eq!(n, output.len());
        debug_assert!(Self::applicable(n), "Size must be power of 2");

        if n <= 1 {
            if n == 1 {
                output[0] = input[0];
            }
            return;
        }

        // For f64, use SIMD-optimized version
        if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f64>() {
            // Safety: We've verified T is f64
            let input_f64: &[Complex<f64>] =
                unsafe { &*(core::ptr::from_ref::<[Complex<T>]>(input) as *const [Complex<f64>]) };
            let output_f64: &mut [Complex<f64>] = unsafe {
                &mut *(core::ptr::from_mut::<[Complex<T>]>(output) as *mut [Complex<f64>])
            };
            stockham_f64(input_f64, output_f64, sign);
            return;
        }

        // Generic implementation for other types
        generic::stockham_generic(input, output, sign);
    }
}

/// SIMD-optimized Stockham FFT for f64 (radix-4 with radix-2 fallback).
///
/// Uses radix-4 stage fusion for better performance (halves memory passes).
/// Falls back to radix-2 for final stage when log_n is odd.
/// Uses AVX-512 (if available), NEON on aarch64, AVX2 on x86_64, or scalar fallback.
pub fn stockham_f64(input: &[Complex<f64>], output: &mut [Complex<f64>], sign: Sign) {
    let n = input.len();

    if n <= 1 {
        if n == 1 {
            output[0] = input[0];
        }
        return;
    }

    // Use architecture-specific radix-4 implementation
    #[cfg(target_arch = "aarch64")]
    {
        // Safety: NEON is always available on aarch64
        unsafe { aarch64::stockham_radix4_neon(input, output, sign) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Prefer AVX-512 when available (4x f64 per register for complex)
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") {
            unsafe { x86_64::stockham_radix4_avx512(input, output, sign) }
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { x86_64::stockham_radix4_avx2(input, output, sign) }
        } else {
            generic::stockham_radix4_scalar(input, output, sign);
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        generic::stockham_radix4_scalar(input, output, sign);
    }
}

#[cfg(test)]
#[allow(clippy::cast_lossless, clippy::cast_precision_loss)] // reason: test helpers use usize as f64 casts for FFT angle computation; precision loss is intentional for test data
mod tests {
    use super::*;

    fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
        (a.re - b.re).abs() < eps && (a.im - b.im).abs() < eps
    }

    /// Reference DFT for testing
    fn reference_dft(input: &[Complex<f64>], sign: Sign) -> Vec<Complex<f64>> {
        let n = input.len();
        let sign_val = f64::from(sign.value());
        let mut output = vec![Complex::zero(); n];

        for k in 0..n {
            let mut sum = Complex::zero();
            for (j, &x) in input.iter().enumerate() {
                let angle =
                    sign_val * core::f64::consts::TAU * (j as f64) * (k as f64) / (n as f64);
                let w = Complex::cis(angle);
                sum = sum + x * w;
            }
            output[k] = sum;
        }
        output
    }

    #[test]
    fn test_stockham_size_4() {
        let input: Vec<Complex<f64>> = (0..4)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut output = vec![Complex::zero(); 4];

        stockham_f64(&input, &mut output, Sign::Forward);
        let reference = reference_dft(&input, Sign::Forward);

        for (i, (&out, &ref_val)) in output.iter().zip(reference.iter()).enumerate() {
            assert!(
                complex_approx_eq(out, ref_val, 1e-10),
                "Mismatch at index {i}: {out:?} vs {ref_val:?}"
            );
        }
    }

    #[test]
    fn test_stockham_size_8() {
        let input: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut output = vec![Complex::zero(); 8];

        stockham_f64(&input, &mut output, Sign::Forward);
        let reference = reference_dft(&input, Sign::Forward);

        for (i, (&out, &ref_val)) in output.iter().zip(reference.iter()).enumerate() {
            assert!(
                complex_approx_eq(out, ref_val, 1e-10),
                "Mismatch at index {i}: {out:?} vs {ref_val:?}"
            );
        }
    }

    #[test]
    fn test_stockham_size_16() {
        let input: Vec<Complex<f64>> = (0..16)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut output = vec![Complex::zero(); 16];

        stockham_f64(&input, &mut output, Sign::Forward);
        let reference = reference_dft(&input, Sign::Forward);

        for (i, (&out, &ref_val)) in output.iter().zip(reference.iter()).enumerate() {
            assert!(
                complex_approx_eq(out, ref_val, 1e-10),
                "Mismatch at index {i}: {out:?} vs {ref_val:?}"
            );
        }
    }

    #[test]
    fn test_stockham_size_64() {
        let input: Vec<Complex<f64>> = (0..64)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut output = vec![Complex::zero(); 64];

        stockham_f64(&input, &mut output, Sign::Forward);
        let reference = reference_dft(&input, Sign::Forward);

        for (i, (&out, &ref_val)) in output.iter().zip(reference.iter()).enumerate() {
            assert!(
                complex_approx_eq(out, ref_val, 1e-10),
                "Mismatch at index {i}: {out:?} vs {ref_val:?}"
            );
        }
    }

    #[test]
    fn test_stockham_size_256() {
        let input: Vec<Complex<f64>> = (0..256)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut output = vec![Complex::zero(); 256];

        stockham_f64(&input, &mut output, Sign::Forward);
        let reference = reference_dft(&input, Sign::Forward);

        for (i, (&out, &ref_val)) in output.iter().zip(reference.iter()).enumerate() {
            assert!(
                complex_approx_eq(out, ref_val, 1e-9),
                "Mismatch at index {i}: {out:?} vs {ref_val:?}"
            );
        }
    }

    #[test]
    fn test_stockham_size_1024() {
        let input: Vec<Complex<f64>> = (0..1024)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut output = vec![Complex::zero(); 1024];

        stockham_f64(&input, &mut output, Sign::Forward);
        let reference = reference_dft(&input, Sign::Forward);

        for (i, (&out, &ref_val)) in output.iter().zip(reference.iter()).enumerate() {
            assert!(
                complex_approx_eq(out, ref_val, 1e-8),
                "Mismatch at index {i}: {out:?} vs {ref_val:?}"
            );
        }
    }

    #[test]
    fn test_stockham_inverse() {
        let input: Vec<Complex<f64>> = (0..64)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut forward = vec![Complex::zero(); 64];
        let mut inverse = vec![Complex::zero(); 64];

        stockham_f64(&input, &mut forward, Sign::Forward);
        stockham_f64(&forward, &mut inverse, Sign::Backward);

        // Scale by 1/N
        let n = input.len() as f64;
        for x in &mut inverse {
            *x = *x / n;
        }

        for (i, (&out, &original)) in inverse.iter().zip(input.iter()).enumerate() {
            assert!(
                complex_approx_eq(out, original, 1e-10),
                "Mismatch at index {i}: {out:?} vs {original:?}"
            );
        }
    }

    #[test]
    fn test_stockham_solver() {
        let solver = StockhamSolver::<f64>::new();
        let input: Vec<Complex<f64>> = (0..16)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut output = vec![Complex::zero(); 16];

        solver.execute(&input, &mut output, Sign::Forward);
        let reference = reference_dft(&input, Sign::Forward);

        for (i, (&out, &ref_val)) in output.iter().zip(reference.iter()).enumerate() {
            assert!(
                complex_approx_eq(out, ref_val, 1e-10),
                "Mismatch at index {i}: {out:?} vs {ref_val:?}"
            );
        }
    }
}
