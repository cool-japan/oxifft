//! Output-pruned FFT: compute only selected output frequencies.

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Compute FFT for only selected output frequencies.
///
/// This is efficient when you need fewer than log₂(N) outputs.
/// For more outputs, consider using full FFT and selecting results.
///
/// # Arguments
///
/// * `input` - Input signal
/// * `output_indices` - Indices of desired output frequencies
///
/// # Returns
///
/// Vector of complex values at the specified frequencies,
/// in the same order as `output_indices`.
///
/// # Example
///
/// ```ignore
/// use oxifft::pruned::fft_pruned_output;
/// use oxifft::Complex;
///
/// let input: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); 1024];
/// let indices = vec![10, 20, 30];
/// let output = fft_pruned_output(&input, &indices);
/// assert_eq!(output.len(), 3);
/// ```
pub fn fft_pruned_output<T: Float>(
    input: &[Complex<T>],
    output_indices: &[usize],
) -> Vec<Complex<T>> {
    let n = input.len();

    if n == 0 || output_indices.is_empty() {
        return vec![Complex::<T>::zero(); output_indices.len()];
    }

    // Decision: use Goertzel for few outputs, full FFT for many
    // Crossover point is approximately log2(N)
    let crossover = (n as f64).log2().ceil() as usize;

    if output_indices.len() <= crossover {
        // Use Goertzel algorithm for each frequency
        super::goertzel_multi(input, output_indices)
    } else {
        // Use full FFT and select outputs
        fft_and_select(input, output_indices)
    }
}

/// Compute full FFT and select specific outputs.
fn fft_and_select<T: Float>(input: &[Complex<T>], output_indices: &[usize]) -> Vec<Complex<T>> {
    let n = input.len();

    let plan = match Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return vec![Complex::<T>::zero(); output_indices.len()],
    };

    let mut full_output = vec![Complex::<T>::zero(); n];
    plan.execute(input, &mut full_output);

    // Select only requested indices
    output_indices
        .iter()
        .map(|&idx| {
            if idx < n {
                full_output[idx]
            } else {
                Complex::<T>::zero()
            }
        })
        .collect()
}

/// Compute FFT with output pruning using butterfly skipping.
///
/// This is a more sophisticated approach that actually skips
/// unnecessary butterfly computations.
///
/// # Arguments
///
/// * `input` - Input signal (must be power of 2)
/// * `output_indices` - Indices of desired outputs
///
/// # Returns
///
/// Vector of complex values at the specified frequencies.
#[allow(dead_code)]
pub fn fft_pruned_output_butterfly<T: Float>(
    input: &[Complex<T>],
    output_indices: &[usize],
) -> Vec<Complex<T>> {
    let n = input.len();

    if n == 0 || output_indices.is_empty() {
        return vec![Complex::<T>::zero(); output_indices.len()];
    }

    // Check if n is power of 2
    if !n.is_power_of_two() {
        return fft_and_select(input, output_indices);
    }

    let log_n = n.trailing_zeros() as usize;

    // Build a mask of which outputs we need
    let mut needed = vec![false; n];
    for &idx in output_indices {
        if idx < n {
            needed[idx] = true;
        }
    }

    // Propagate needed flags backwards through butterfly stages
    // For each stage, if output k is needed, both inputs to its butterfly are needed
    let mut stage_needed = needed.clone();

    for stage in (0..log_n).rev() {
        let block_size = 1 << (stage + 1);
        let half_block = block_size / 2;

        for block_start in (0..n).step_by(block_size) {
            for i in 0..half_block {
                let idx1 = block_start + i;
                let idx2 = block_start + i + half_block;

                if idx1 < n && idx2 < n {
                    let needs_either = stage_needed[idx1] || stage_needed[idx2];
                    stage_needed[idx1] = needs_either;
                    stage_needed[idx2] = needs_either;
                }
            }
        }
    }

    // Bit-reverse permutation
    let mut data: Vec<Complex<T>> = (0..n)
        .map(|i| {
            let rev = bit_reverse(i, log_n);
            if rev < input.len() {
                input[rev]
            } else {
                Complex::<T>::zero()
            }
        })
        .collect();

    // Perform pruned FFT with butterfly skipping
    let two_pi = <T as Float>::PI + <T as Float>::PI;

    for stage in 0..log_n {
        let block_size = 1 << (stage + 1);
        let half_block = block_size / 2;

        for block_start in (0..n).step_by(block_size) {
            for i in 0..half_block {
                let idx1 = block_start + i;
                let idx2 = block_start + i + half_block;

                // Skip butterfly if not needed
                if !stage_needed[idx1] && !stage_needed[idx2] {
                    continue;
                }

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
            }
        }
    }

    // Extract only requested outputs
    output_indices
        .iter()
        .map(|&idx| {
            if idx < n {
                data[idx]
            } else {
                Complex::<T>::zero()
            }
        })
        .collect()
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
    fn test_fft_pruned_output_empty() {
        let input: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); 64];
        let output = fft_pruned_output(&input, &[]);
        assert!(output.is_empty());
    }

    #[test]
    fn test_fft_pruned_output_single() {
        let n = 64;
        let input: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); n];

        // DC component should be N for constant signal
        let output = fft_pruned_output(&input, &[0]);
        assert_eq!(output.len(), 1);
        assert!((output[0].re - n as f64).abs() < 1e-10);
    }

    #[test]
    fn test_fft_pruned_output_multiple() {
        let n = 64;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64) / (n as f64), 0.0))
            .collect();

        let indices = vec![0, 5, 10, 20, 31];
        let pruned_output = fft_pruned_output(&input, &indices);

        // Compare with full FFT
        let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
        let mut full_output = vec![Complex::new(0.0_f64, 0.0); n];
        plan.execute(&input, &mut full_output);

        for (i, &idx) in indices.iter().enumerate() {
            let diff_re = (pruned_output[i].re - full_output[idx].re).abs();
            let diff_im = (pruned_output[i].im - full_output[idx].im).abs();

            assert!(diff_re < 1e-10, "Real mismatch at index {idx}");
            assert!(diff_im < 1e-10, "Imag mismatch at index {idx}");
        }
    }

    #[test]
    fn test_fft_pruned_output_butterfly() {
        let n = 64;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        let indices = vec![0, 5, 10];
        let pruned_output = fft_pruned_output_butterfly(&input, &indices);

        // Compare with full FFT
        let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
        let mut full_output = vec![Complex::new(0.0_f64, 0.0); n];
        plan.execute(&input, &mut full_output);

        for (i, &idx) in indices.iter().enumerate() {
            let diff_re = (pruned_output[i].re - full_output[idx].re).abs();
            let diff_im = (pruned_output[i].im - full_output[idx].im).abs();

            assert!(
                diff_re < 1e-8,
                "Real mismatch at index {}: {} vs {}",
                idx,
                pruned_output[i].re,
                full_output[idx].re
            );
            assert!(
                diff_im < 1e-8,
                "Imag mismatch at index {}: {} vs {}",
                idx,
                pruned_output[i].im,
                full_output[idx].im
            );
        }
    }

    #[test]
    fn test_bit_reverse() {
        assert_eq!(bit_reverse(0, 3), 0);
        assert_eq!(bit_reverse(1, 3), 4);
        assert_eq!(bit_reverse(2, 3), 2);
        assert_eq!(bit_reverse(3, 3), 6);
        assert_eq!(bit_reverse(4, 3), 1);
    }
}
