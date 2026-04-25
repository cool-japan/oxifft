//! Pruned FFT implementation for partial input/output computation.
//!
//! Pruned FFT optimizes computation when:
//! - Only a subset of inputs are non-zero (input pruning)
//! - Only a subset of outputs are needed (output pruning)
//!
//! # Complexity
//!
//! For output-pruned FFT computing M outputs from N inputs:
//! - Standard FFT: O(N log N)
//! - Pruned FFT: O(N log M) when M << N
//!
//! # Example
//!
//! ```ignore
//! use oxifft::pruned::{fft_pruned_output, fft_pruned_input, PrunedPlan};
//! use oxifft::Complex;
//!
//! let n = 1024;
//! let input: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); n];
//!
//! // Output pruning: only compute specific frequencies
//! let desired_indices = vec![10, 20, 30, 100, 200];
//! let output = fft_pruned_output(&input, &desired_indices);
//!
//! // Input pruning: when most inputs are zero
//! let nonzero_inputs = vec![
//!     (0, Complex::new(1.0, 0.0)),
//!     (100, Complex::new(0.5, 0.5)),
//! ];
//! let output = fft_pruned_input(&nonzero_inputs, n);
//! ```

mod input_pruned;
mod output_pruned;
mod plan;

pub use input_pruned::fft_pruned_input;
pub use output_pruned::fft_pruned_output;
pub use plan::{PrunedPlan, PruningMode};

use crate::kernel::{Complex, Float};
use crate::prelude::*;

/// Compute DFT for a single frequency using the Goertzel algorithm.
///
/// This is efficient when only one or a few frequencies are needed.
/// Complexity: O(N) per frequency.
///
/// # Arguments
///
/// * `input` - Input signal
/// * `freq_idx` - Desired frequency index (0 to N-1)
///
/// # Returns
///
/// Complex value at the specified frequency.
pub fn goertzel<T: Float>(input: &[Complex<T>], freq_idx: usize) -> Complex<T> {
    let n = input.len();
    if n == 0 {
        return Complex::<T>::zero();
    }

    // Goertzel algorithm
    // W = e^(-2*pi*i*k/N)
    // Uses recurrence: s[n] = x[n] + 2*cos(2*pi*k/N)*s[n-1] - s[n-2]
    // After N iterations, compute s[N] = 0 + coeff*s[N-1] - s[N-2]
    // Result: X[k] = s[N] - W*s[N-1] = cos*s1 - s0 + j*sin*s1

    let two_pi = <T as Float>::PI + <T as Float>::PI;
    let omega = two_pi * T::from_usize(freq_idx) / T::from_usize(n);
    let (sin_omega, cos_omega) = Float::sin_cos(omega);
    let coeff = cos_omega + cos_omega; // 2 * cos(omega)

    let mut s0 = T::ZERO;
    let mut s1 = T::ZERO;

    // Process real part of input
    for sample in input.iter() {
        let s2 = sample.re + coeff * s1 - s0;
        s0 = s1;
        s1 = s2;
    }

    // Correct Goertzel output: X = cos*s1 - s0 + j*sin*s1
    let re = cos_omega * s1 - s0;
    let im = sin_omega * s1;

    // For complex input, process imaginary part
    // The contribution is j * Goertzel(im) = j*(re_im + j*im_im) = -im_im + j*re_im
    s0 = T::ZERO;
    s1 = T::ZERO;

    for sample in input.iter() {
        let s2 = sample.im + coeff * s1 - s0;
        s0 = s1;
        s1 = s2;
    }

    // Goertzel output for imaginary input
    let re_im = cos_omega * s1 - s0;
    let im_im = sin_omega * s1;

    // Contribution to DFT: j * (re_im + j*im_im) = -im_im + j*re_im
    let re_from_im = T::ZERO - im_im;
    let im_from_im = re_im;

    // Combine real and imaginary contributions
    Complex::new(re + re_from_im, im + im_from_im)
}

/// Compute multiple DFT frequencies using Goertzel algorithm.
///
/// More efficient than full FFT when computing fewer than log₂(N) frequencies.
///
/// # Arguments
///
/// * `input` - Input signal
/// * `freq_indices` - Desired frequency indices
///
/// # Returns
///
/// Vector of complex values at the specified frequencies.
pub fn goertzel_multi<T: Float>(input: &[Complex<T>], freq_indices: &[usize]) -> Vec<Complex<T>> {
    freq_indices.iter().map(|&k| goertzel(input, k)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Direction, Flags, Plan};

    #[test]
    fn test_goertzel_single_frequency() {
        let n = 256;
        let freq = 10;

        // Create signal with single frequency
        let two_pi = core::f64::consts::PI * 2.0;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let angle = two_pi * (freq as f64) * f64::from(i) / f64::from(n);
                Complex::new(angle.cos(), angle.sin())
            })
            .collect();

        // Goertzel should detect the frequency
        let result = goertzel(&input, freq);
        let magnitude = (result.re * result.re + result.im * result.im).sqrt();

        // Magnitude should be close to N for a unit-amplitude signal
        assert!(
            magnitude > f64::from(n) * 0.9,
            "Expected magnitude ~{n}, got {magnitude}"
        );
    }

    #[test]
    fn test_goertzel_vs_fft() {
        let n = 64;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64) / (n as f64), 0.0))
            .collect();

        // Full FFT
        let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
        let mut fft_output = vec![Complex::new(0.0_f64, 0.0); n];
        plan.execute(&input, &mut fft_output);

        // Goertzel for a few frequencies
        for freq in [0, 5, 10, 20, 31] {
            let goertzel_result = goertzel(&input, freq);
            let fft_result = fft_output[freq];

            let diff_re = (goertzel_result.re - fft_result.re).abs();
            let diff_im = (goertzel_result.im - fft_result.im).abs();

            assert!(
                diff_re < 1e-10,
                "Real mismatch at freq {}: {} vs {}",
                freq,
                goertzel_result.re,
                fft_result.re
            );
            assert!(
                diff_im < 1e-10,
                "Imag mismatch at freq {}: {} vs {}",
                freq,
                goertzel_result.im,
                fft_result.im
            );
        }
    }

    #[test]
    fn test_goertzel_multi() {
        let n = 128;
        let input: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); n];

        let freq_indices = vec![0, 5, 10, 50];
        let results = goertzel_multi(&input, &freq_indices);

        assert_eq!(results.len(), 4);
        // DC component (freq 0) should have magnitude N for constant signal
        let dc_mag = (results[0].re * results[0].re + results[0].im * results[0].im).sqrt();
        assert!((dc_mag - n as f64).abs() < 1e-10);
    }
}
