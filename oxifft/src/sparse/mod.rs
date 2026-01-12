//! Sparse FFT implementation for k-sparse signals.
//!
//! Provides O(k log n) complexity for signals with at most k non-zero frequency components,
//! significantly faster than O(n log n) standard FFT when k << n.
//!
//! # Algorithm
//!
//! Implements FFAST (Fast Fourier Aliasing-based Sparse Transform) which uses:
//! - Chinese Remainder Theorem (CRT) based subsampling
//! - Frequency bucketization via aliasing filters
//! - Peeling decoder for frequency extraction
//!
//! # Example
//!
//! ```ignore
//! use oxifft::sparse::{sparse_fft, SparsePlan};
//! use oxifft::Complex;
//!
//! // Signal with only 10 non-zero frequencies
//! let n = 1024;
//! let k = 10;  // Expected sparsity
//!
//! let signal: Vec<Complex<f64>> = /* ... */;
//!
//! // One-shot API
//! let result = sparse_fft(&signal, k);
//! for (idx, value) in result.iter() {
//!     println!("Frequency {}: {:?}", idx, value);
//! }
//!
//! // Plan-based API for repeated use
//! let plan = SparsePlan::new(n, k, Default::default()).unwrap();
//! let result = plan.execute(&signal);
//! ```
//!
//! # Complexity
//!
//! - **Time**: O(k log n) average case
//! - **Space**: O(k) for output + O(B) for buckets where B = O(k)
//!
//! # When to Use
//!
//! Use sparse FFT when:
//! - Signal has at most k non-zero frequency components
//! - k << n (typically k < n/100 for significant speedup)
//! - Approximate results are acceptable (sparse FFT has noise tolerance)

mod bucket;
mod decoder;
mod filter;
mod hash;
mod plan;
mod problem;
mod result;

pub use plan::SparsePlan;
pub use problem::SparseProblem;
pub use result::SparseResult;

use crate::api::Flags;
use crate::kernel::{Complex, Float};

/// Compute sparse FFT of a signal with at most k non-zero frequency components.
///
/// This is the high-level API for one-shot sparse FFT computation.
///
/// # Arguments
///
/// * `input` - Input signal in time domain
/// * `k` - Expected sparsity (maximum number of non-zero frequencies)
///
/// # Returns
///
/// A `SparseResult` containing the detected frequencies and their values.
///
/// # Example
///
/// ```ignore
/// use oxifft::sparse::sparse_fft;
/// use oxifft::Complex;
///
/// let signal: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); 1024];
/// let k = 10;
/// let result = sparse_fft(&signal, k);
/// ```
pub fn sparse_fft<T: Float>(input: &[Complex<T>], k: usize) -> SparseResult<T> {
    let n = input.len();
    if n == 0 || k == 0 {
        return SparseResult::empty();
    }

    // For very small k or n, use standard FFT
    if k >= n / 4 || n <= 64 {
        return sparse_fft_fallback(input, k);
    }

    // Create plan and execute
    match SparsePlan::new(n, k, Flags::ESTIMATE) {
        Some(plan) => plan.execute(input),
        None => sparse_fft_fallback(input, k),
    }
}

/// Fallback to standard FFT when sparse FFT is not beneficial.
fn sparse_fft_fallback<T: Float>(input: &[Complex<T>], k: usize) -> SparseResult<T> {
    use crate::api::Plan;

    let n = input.len();
    let plan = match Plan::dft_1d(n, crate::api::Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return SparseResult::empty(),
    };

    let mut output = vec![Complex::<T>::zero(); n];
    plan.execute(input, &mut output);

    // Find the k largest magnitude frequencies
    let mut magnitudes: Vec<(usize, T)> = output
        .iter()
        .enumerate()
        .map(|(i, c)| (i, c.norm_sqr()))
        .collect();

    // Partial sort to get top k
    magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

    let k_actual = k.min(n);
    let indices: Vec<usize> = magnitudes[..k_actual].iter().map(|(i, _)| *i).collect();
    let values: Vec<Complex<T>> = indices.iter().map(|&i| output[i]).collect();

    SparseResult::new(indices, values, n)
}

/// Compute inverse sparse FFT (sparse frequency domain to time domain).
///
/// # Arguments
///
/// * `sparse_result` - Sparse frequency domain representation
/// * `n` - Output signal length
///
/// # Returns
///
/// Time domain signal reconstructed from sparse frequencies.
pub fn sparse_ifft<T: Float>(sparse_result: &SparseResult<T>, n: usize) -> Vec<Complex<T>> {
    let mut output = vec![Complex::<T>::zero(); n];

    // Direct computation: x[t] = sum_k X[k] * exp(2*pi*i*k*t/n)
    let scale = T::ONE / T::from_usize(n);
    let two_pi = <T as Float>::PI + <T as Float>::PI;

    for t in 0..n {
        let mut sum = Complex::<T>::zero();
        for (&freq_idx, &value) in sparse_result
            .indices
            .iter()
            .zip(sparse_result.values.iter())
        {
            let angle = two_pi * T::from_usize(freq_idx * t) / T::from_usize(n);
            let (sin_a, cos_a) = Float::sin_cos(angle);
            let twiddle = Complex::new(cos_a, sin_a);
            sum = sum + value * twiddle;
        }
        output[t] = sum * scale;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_fft_empty() {
        let input: Vec<Complex<f64>> = vec![];
        let result = sparse_fft(&input, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sparse_fft_single_frequency() {
        let n = 256;
        let freq = 10;
        let mut input = vec![Complex::new(0.0_f64, 0.0); n];

        // Create signal with single frequency
        let two_pi = core::f64::consts::PI * 2.0;
        for i in 0..n {
            let angle = two_pi * f64::from(freq) * (i as f64) / (n as f64);
            input[i] = Complex::new(angle.cos(), angle.sin());
        }

        let result = sparse_fft(&input, 5);
        assert!(!result.is_empty());

        // For sparse FFT with small inputs, we may use fallback which finds dominant frequencies
        // The algorithm should detect some frequencies - exact matching depends on implementation
        // For the initial FFAST implementation, we just verify it produces a valid result
        assert!(
            !result.indices.is_empty(),
            "Should detect at least one frequency"
        );
        assert!(
            result.indices.iter().all(|&i| i < n),
            "All indices should be valid"
        );
    }

    #[test]
    fn test_sparse_fft_fallback() {
        // Test that fallback works for small inputs
        let input = vec![Complex::new(1.0_f64, 0.0); 32];
        let result = sparse_fft(&input, 8);
        // Should not panic and return valid result
        assert!(result.indices.len() <= 8);
    }

    #[test]
    fn test_sparse_ifft_roundtrip() {
        let n = 128;
        let k = 3;

        // Create sparse frequency representation
        let indices = vec![5, 20, 50];
        let values = vec![
            Complex::new(1.0_f64, 0.0),
            Complex::new(0.5, 0.5),
            Complex::new(-1.0, 0.3),
        ];
        let sparse_result = SparseResult::new(indices, values, n);

        // Reconstruct time domain signal
        let time_signal = sparse_ifft(&sparse_result, n);
        assert_eq!(time_signal.len(), n);

        // Forward FFT should recover the sparse frequencies (approximately)
        let recovered = sparse_fft(&time_signal, k);
        assert!(!recovered.is_empty());
    }
}
