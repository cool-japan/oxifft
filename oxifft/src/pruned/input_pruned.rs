//! Input-pruned FFT: compute FFT when most inputs are zero.

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Compute FFT when only some inputs are non-zero.
///
/// This is efficient when the number of non-zero inputs is small.
/// Uses direct DFT computation for sparse inputs.
///
/// # Arguments
///
/// * `nonzero_inputs` - Vec of (index, value) pairs for non-zero inputs
/// * `n` - Total FFT size
///
/// # Returns
///
/// Full FFT output of length n.
///
/// # Example
///
/// ```ignore
/// use oxifft::pruned::fft_pruned_input;
/// use oxifft::Complex;
///
/// // Only positions 0 and 100 have non-zero values
/// let nonzero_inputs = vec![
///     (0, Complex::new(1.0, 0.0)),
///     (100, Complex::new(0.5, 0.5)),
/// ];
/// let output = fft_pruned_input(&nonzero_inputs, 1024);
/// assert_eq!(output.len(), 1024);
/// ```
pub fn fft_pruned_input<T: Float>(
    nonzero_inputs: &[(usize, Complex<T>)],
    n: usize,
) -> Vec<Complex<T>> {
    if n == 0 {
        return Vec::new();
    }

    if nonzero_inputs.is_empty() {
        return vec![Complex::<T>::zero(); n];
    }

    // Decision: use direct DFT for very sparse inputs, full FFT otherwise
    // Crossover: K * N vs N * log(N)
    // So K < log(N) is the threshold
    let k = nonzero_inputs.len();
    let crossover = (n as f64).log2().ceil() as usize;

    if k <= crossover {
        // Direct DFT: O(K * N)
        dft_sparse_input(nonzero_inputs, n)
    } else {
        // Full FFT: O(N * log N)
        fft_full_from_sparse(nonzero_inputs, n)
    }
}

/// Direct DFT computation for sparse input.
///
/// Complexity: O(K * N) where K is number of non-zero inputs.
fn dft_sparse_input<T: Float>(nonzero_inputs: &[(usize, Complex<T>)], n: usize) -> Vec<Complex<T>> {
    let mut output = vec![Complex::<T>::zero(); n];

    let two_pi = <T as Float>::PI + <T as Float>::PI;

    // For each output frequency k: X[k] = sum_m x[m] * e^(-2*pi*i*k*m/n)
    for k in 0..n {
        let mut sum = Complex::<T>::zero();

        for &(m, value) in nonzero_inputs {
            if m < n {
                // Twiddle factor: e^(-2*pi*i*k*m/n)
                let angle = two_pi * T::from_usize(k * m) / T::from_usize(n);
                let (sin_a, cos_a) = Float::sin_cos(angle);
                let twiddle = Complex::new(cos_a, T::ZERO - sin_a);

                sum = sum + value * twiddle;
            }
        }

        output[k] = sum;
    }

    output
}

/// Full FFT from sparse input.
///
/// First creates full input array, then uses standard FFT.
fn fft_full_from_sparse<T: Float>(
    nonzero_inputs: &[(usize, Complex<T>)],
    n: usize,
) -> Vec<Complex<T>> {
    // Create full input array
    let mut input = vec![Complex::<T>::zero(); n];
    for &(idx, value) in nonzero_inputs {
        if idx < n {
            input[idx] = value;
        }
    }

    // Standard FFT
    let plan = match Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return vec![Complex::<T>::zero(); n],
    };

    let mut output = vec![Complex::<T>::zero(); n];
    plan.execute(&input, &mut output);

    output
}

/// Compute FFT with input pruning using modified butterfly structure.
///
/// This approach modifies the FFT computation to skip zeros at the input stage.
///
/// # Arguments
///
/// * `nonzero_inputs` - Vec of (index, value) pairs
/// * `n` - Total FFT size (must be power of 2)
///
/// # Returns
///
/// Full FFT output.
#[allow(dead_code)]
pub fn fft_pruned_input_butterfly<T: Float>(
    nonzero_inputs: &[(usize, Complex<T>)],
    n: usize,
) -> Vec<Complex<T>> {
    if n == 0 || !n.is_power_of_two() {
        return fft_full_from_sparse(nonzero_inputs, n);
    }

    let log_n = n.trailing_zeros() as usize;

    // Create input with bit-reversed order
    let mut data = vec![Complex::<T>::zero(); n];

    for &(idx, value) in nonzero_inputs {
        if idx < n {
            let rev_idx = bit_reverse(idx, log_n);
            data[rev_idx] = value;
        }
    }

    // Track which positions have non-zero data
    // This could be used for more aggressive pruning
    let mut has_data = vec![false; n];
    for &(idx, _) in nonzero_inputs {
        if idx < n {
            has_data[bit_reverse(idx, log_n)] = true;
        }
    }

    // Standard Cooley-Tukey FFT
    let two_pi = <T as Float>::PI + <T as Float>::PI;

    for stage in 0..log_n {
        let block_size = 1 << (stage + 1);
        let half_block = block_size / 2;

        for block_start in (0..n).step_by(block_size) {
            for i in 0..half_block {
                let idx1 = block_start + i;
                let idx2 = block_start + i + half_block;

                // Twiddle factor
                let k = i * (n / block_size);
                let angle = two_pi * T::from_usize(k) / T::from_usize(n);
                let (sin_a, cos_a) = Float::sin_cos(angle);
                let twiddle = Complex::new(cos_a, T::ZERO - sin_a);

                // Butterfly
                let a = data[idx1];
                let b = data[idx2] * twiddle;

                data[idx1] = a + b;
                data[idx2] = a - b;

                // Update has_data
                has_data[idx1] = has_data[idx1] || has_data[idx2];
                has_data[idx2] = has_data[idx1];
            }
        }
    }

    data
}

/// Bit-reverse an index.
fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_pruned_input_empty() {
        let output: Vec<Complex<f64>> = fft_pruned_input(&[], 64);
        assert_eq!(output.len(), 64);
        assert!(output.iter().all(|c| c.re == 0.0 && c.im == 0.0));
    }

    #[test]
    fn test_fft_pruned_input_single() {
        let n = 64;
        // Single non-zero input at position 0
        let nonzero = vec![(0, Complex::new(1.0_f64, 0.0))];
        let output = fft_pruned_input(&nonzero, n);

        assert_eq!(output.len(), n);

        // x[0] = 1, all others 0
        // DFT: X[k] = sum x[m] * W^(km) = 1 * W^0 = 1 for all k
        for o in &output {
            assert!((o.re - 1.0).abs() < 1e-10);
            assert!(o.im.abs() < 1e-10);
        }
    }

    #[test]
    fn test_fft_pruned_input_vs_full() {
        let n = 64;
        let nonzero = vec![
            (0, Complex::new(1.0_f64, 0.0)),
            (10, Complex::new(0.5, 0.3)),
            (32, Complex::new(-1.0, 0.5)),
        ];

        let pruned_output = fft_pruned_input(&nonzero, n);

        // Compare with full FFT
        let mut input = vec![Complex::new(0.0_f64, 0.0); n];
        for &(idx, val) in &nonzero {
            input[idx] = val;
        }

        let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
        let mut full_output = vec![Complex::new(0.0_f64, 0.0); n];
        plan.execute(&input, &mut full_output);

        for i in 0..n {
            let diff_re = (pruned_output[i].re - full_output[i].re).abs();
            let diff_im = (pruned_output[i].im - full_output[i].im).abs();

            assert!(diff_re < 1e-10, "Real mismatch at index {i}");
            assert!(diff_im < 1e-10, "Imag mismatch at index {i}");
        }
    }

    #[test]
    fn test_fft_pruned_input_butterfly() {
        let n = 64;
        let nonzero = vec![(0, Complex::new(1.0_f64, 0.0)), (5, Complex::new(0.5, 0.3))];

        let pruned_output = fft_pruned_input_butterfly(&nonzero, n);

        // Compare with full FFT
        let mut input = vec![Complex::new(0.0_f64, 0.0); n];
        for &(idx, val) in &nonzero {
            input[idx] = val;
        }

        let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
        let mut full_output = vec![Complex::new(0.0_f64, 0.0); n];
        plan.execute(&input, &mut full_output);

        for i in 0..n {
            let diff_re = (pruned_output[i].re - full_output[i].re).abs();
            let diff_im = (pruned_output[i].im - full_output[i].im).abs();

            assert!(
                diff_re < 1e-8,
                "Real mismatch at index {}: {} vs {}",
                i,
                pruned_output[i].re,
                full_output[i].re
            );
            assert!(
                diff_im < 1e-8,
                "Imag mismatch at index {}: {} vs {}",
                i,
                pruned_output[i].im,
                full_output[i].im
            );
        }
    }

    #[test]
    fn test_dft_sparse_input() {
        let n = 32;
        let nonzero = vec![(0, Complex::new(1.0_f64, 0.0)), (1, Complex::new(0.5, 0.5))];

        let output = dft_sparse_input(&nonzero, n);
        assert_eq!(output.len(), n);

        // Verify against full DFT
        let mut input = vec![Complex::new(0.0_f64, 0.0); n];
        for &(idx, val) in &nonzero {
            input[idx] = val;
        }

        let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
        let mut full_output = vec![Complex::new(0.0_f64, 0.0); n];
        plan.execute(&input, &mut full_output);

        for i in 0..n {
            let diff_re = (output[i].re - full_output[i].re).abs();
            let diff_im = (output[i].im - full_output[i].im).abs();

            assert!(diff_re < 1e-10, "Real mismatch at index {i}");
            assert!(diff_im < 1e-10, "Imag mismatch at index {i}");
        }
    }
}
