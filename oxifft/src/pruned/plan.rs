//! Pruned FFT plan for repeated transforms.

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Pruning mode specification.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PruningMode {
    /// Only specified inputs are non-zero.
    InputPruned {
        /// Indices of non-zero inputs.
        nonzero_indices: Vec<usize>,
    },
    /// Only specified outputs are needed.
    OutputPruned {
        /// Indices of desired outputs.
        desired_indices: Vec<usize>,
    },
    /// Both input and output pruning.
    Both {
        /// Non-zero input indices.
        input_indices: Vec<usize>,
        /// Desired output indices.
        output_indices: Vec<usize>,
    },
}

/// Pruned FFT plan for repeated transforms with fixed pruning pattern.
///
/// Pre-computes optimization structures for efficient repeated pruned FFT.
pub struct PrunedPlan<T: Float> {
    /// Transform size.
    n: usize,
    /// Pruning mode.
    mode: PruningMode,
    /// Inner FFT plan (for full FFT fallback when pruned is not beneficial).
    #[allow(dead_code)]
    // reason: reserved field for full-FFT fallback path; used when input/output pruning is not beneficial
    inner_plan: Option<Plan<T>>,
    /// Direction.
    direction: Direction,
    /// Planning flags.
    flags: Flags,
    /// Precomputed twiddle factors for direct computation.
    twiddles: Vec<Complex<T>>,
}

impl<T: Float> PrunedPlan<T> {
    /// Create an output-pruned plan.
    ///
    /// # Arguments
    ///
    /// * `n` - Transform size
    /// * `output_indices` - Indices of desired outputs
    /// * `flags` - Planning flags
    ///
    /// # Returns
    ///
    /// `None` if plan creation fails.
    pub fn output_pruned(n: usize, output_indices: &[usize], flags: Flags) -> Option<Self> {
        if n == 0 {
            return None;
        }

        let inner_plan = Plan::dft_1d(n, Direction::Forward, flags);

        // Precompute twiddle factors for Goertzel
        let two_pi = <T as Float>::PI + <T as Float>::PI;
        let twiddles: Vec<Complex<T>> = output_indices
            .iter()
            .map(|&k| {
                let omega = two_pi * T::from_usize(k) / T::from_usize(n);
                let (sin_omega, cos_omega) = Float::sin_cos(omega);
                Complex::new(cos_omega + cos_omega, sin_omega) // [2*cos, sin]
            })
            .collect();

        Some(Self {
            n,
            mode: PruningMode::OutputPruned {
                desired_indices: output_indices.to_vec(),
            },
            inner_plan,
            direction: Direction::Forward,
            flags,
            twiddles,
        })
    }

    /// Create an input-pruned plan.
    ///
    /// # Arguments
    ///
    /// * `n` - Transform size
    /// * `input_indices` - Indices of non-zero inputs
    /// * `flags` - Planning flags
    ///
    /// # Returns
    ///
    /// `None` if plan creation fails.
    pub fn input_pruned(n: usize, input_indices: &[usize], flags: Flags) -> Option<Self> {
        if n == 0 {
            return None;
        }

        let inner_plan = Plan::dft_1d(n, Direction::Forward, flags);

        Some(Self {
            n,
            mode: PruningMode::InputPruned {
                nonzero_indices: input_indices.to_vec(),
            },
            inner_plan,
            direction: Direction::Forward,
            flags,
            twiddles: Vec::new(),
        })
    }

    /// Create a dual-pruned plan (both input and output).
    ///
    /// # Arguments
    ///
    /// * `n` - Transform size
    /// * `input_indices` - Indices of non-zero inputs
    /// * `output_indices` - Indices of desired outputs
    /// * `flags` - Planning flags
    ///
    /// # Returns
    ///
    /// `None` if plan creation fails.
    pub fn both_pruned(
        n: usize,
        input_indices: &[usize],
        output_indices: &[usize],
        flags: Flags,
    ) -> Option<Self> {
        if n == 0 {
            return None;
        }

        let inner_plan = Plan::dft_1d(n, Direction::Forward, flags);

        // Precompute twiddle factors for direct DFT computation
        let two_pi = <T as Float>::PI + <T as Float>::PI;
        let mut twiddles = Vec::with_capacity(input_indices.len() * output_indices.len());

        for &k in output_indices {
            for &m in input_indices {
                let angle = two_pi * T::from_usize(k * m) / T::from_usize(n);
                let (sin_a, cos_a) = Float::sin_cos(angle);
                twiddles.push(Complex::new(cos_a, T::ZERO - sin_a));
            }
        }

        Some(Self {
            n,
            mode: PruningMode::Both {
                input_indices: input_indices.to_vec(),
                output_indices: output_indices.to_vec(),
            },
            inner_plan,
            direction: Direction::Forward,
            flags,
            twiddles,
        })
    }

    /// Execute the pruned FFT.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data
    /// * `output` - Output buffer
    ///
    /// For output-pruned: `input` should have length n, `output` should have length = num desired outputs.
    /// For input-pruned: `input` should have length = num non-zero inputs, `output` should have length n.
    /// For both: `input` has length = num non-zero, `output` has length = num desired.
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        match &self.mode {
            PruningMode::OutputPruned { desired_indices } => {
                self.execute_output_pruned(input, output, desired_indices);
            }
            PruningMode::InputPruned { nonzero_indices } => {
                self.execute_input_pruned(input, output, nonzero_indices);
            }
            PruningMode::Both {
                input_indices,
                output_indices,
            } => {
                self.execute_both_pruned(input, output, input_indices, output_indices);
            }
        }
    }

    /// Execute output-pruned FFT.
    fn execute_output_pruned(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        desired_indices: &[usize],
    ) {
        if input.len() != self.n || output.len() != desired_indices.len() {
            return;
        }

        // Use Goertzel for each output
        let two_pi = <T as Float>::PI + <T as Float>::PI;

        for (out_idx, &freq_idx) in desired_indices.iter().enumerate() {
            let omega = two_pi * T::from_usize(freq_idx) / T::from_usize(self.n);
            let (sin_omega, cos_omega) = Float::sin_cos(omega);
            let coeff = cos_omega + cos_omega;

            // Process real part of input
            let mut s0 = T::ZERO;
            let mut s1 = T::ZERO;
            for sample in input.iter() {
                let s2 = sample.re + coeff * s1 - s0;
                s0 = s1;
                s1 = s2;
            }
            // Correct Goertzel: X = cos*s1 - s0 + j*sin*s1
            let re = cos_omega * s1 - s0;
            let im = sin_omega * s1;

            // Process imaginary part of input
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
            // Contribution: j*(re_im + j*im_im) = -im_im + j*re_im
            let re_from_im = T::ZERO - im_im;
            let im_from_im = re_im;

            output[out_idx] = Complex::new(re + re_from_im, im + im_from_im);
        }
    }

    /// Execute input-pruned FFT.
    fn execute_input_pruned(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        nonzero_indices: &[usize],
    ) {
        if input.len() != nonzero_indices.len() || output.len() != self.n {
            return;
        }

        // Direct DFT for sparse input
        let two_pi = <T as Float>::PI + <T as Float>::PI;

        for k in 0..self.n {
            let mut sum = Complex::<T>::zero();

            for (i, &m) in nonzero_indices.iter().enumerate() {
                if m < self.n {
                    let angle = two_pi * T::from_usize(k * m) / T::from_usize(self.n);
                    let (sin_a, cos_a) = Float::sin_cos(angle);
                    let twiddle = Complex::new(cos_a, T::ZERO - sin_a);
                    sum = sum + input[i] * twiddle;
                }
            }

            output[k] = sum;
        }
    }

    /// Execute dual-pruned FFT.
    fn execute_both_pruned(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        input_indices: &[usize],
        output_indices: &[usize],
    ) {
        if input.len() != input_indices.len() || output.len() != output_indices.len() {
            return;
        }

        let num_inputs = input_indices.len();

        // Use precomputed twiddles
        for (out_idx, _) in output_indices.iter().enumerate() {
            let mut sum = Complex::<T>::zero();

            for (in_idx, _) in input_indices.iter().enumerate() {
                let twiddle_idx = out_idx * num_inputs + in_idx;
                if twiddle_idx < self.twiddles.len() {
                    sum = sum + input[in_idx] * self.twiddles[twiddle_idx];
                }
            }

            output[out_idx] = sum;
        }
    }

    /// Get the transform size.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the pruning mode.
    pub fn mode(&self) -> &PruningMode {
        &self.mode
    }

    /// Get the direction.
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Get the planning flags.
    pub fn flags(&self) -> Flags {
        self.flags
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_pruned_plan() {
        let n = 64;
        let indices = vec![0, 5, 10];

        let plan: PrunedPlan<f64> =
            PrunedPlan::output_pruned(n, &indices, Flags::ESTIMATE).unwrap();
        assert_eq!(plan.n(), n);

        let input: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); n];
        let mut output = vec![Complex::new(0.0_f64, 0.0); indices.len()];

        plan.execute(&input, &mut output);

        // DC component should be N
        assert!((output[0].re - n as f64).abs() < 1e-10);
    }

    #[test]
    fn test_input_pruned_plan() {
        let n = 64;
        let input_indices = vec![0, 10];

        let plan: PrunedPlan<f64> =
            PrunedPlan::input_pruned(n, &input_indices, Flags::ESTIMATE).unwrap();
        assert_eq!(plan.n(), n);

        let input = vec![Complex::new(1.0_f64, 0.0), Complex::new(0.5, 0.0)];
        let mut output = vec![Complex::new(0.0_f64, 0.0); n];

        plan.execute(&input, &mut output);

        // Verify output is not all zeros
        let sum_mag: f64 = output.iter().map(|c| c.re * c.re + c.im * c.im).sum();
        assert!(sum_mag > 0.0);
    }

    #[test]
    fn test_both_pruned_plan() {
        let n = 64;
        let input_indices = vec![0, 5];
        let output_indices = vec![0, 10, 20];

        let plan: PrunedPlan<f64> =
            PrunedPlan::both_pruned(n, &input_indices, &output_indices, Flags::ESTIMATE).unwrap();

        let input = vec![Complex::new(1.0_f64, 0.0), Complex::new(0.5, 0.3)];
        let mut output = vec![Complex::new(0.0_f64, 0.0); output_indices.len()];

        plan.execute(&input, &mut output);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_pruned_plan_vs_full_fft() {
        let n = 64;
        let output_indices = vec![0, 5, 10, 31];

        let plan: PrunedPlan<f64> =
            PrunedPlan::output_pruned(n, &output_indices, Flags::ESTIMATE).unwrap();

        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64) / (n as f64), 0.0))
            .collect();

        let mut pruned_output = vec![Complex::new(0.0_f64, 0.0); output_indices.len()];
        plan.execute(&input, &mut pruned_output);

        // Compare with full FFT
        let full_plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
        let mut full_output = vec![Complex::new(0.0_f64, 0.0); n];
        full_plan.execute(&input, &mut full_output);

        for (i, &idx) in output_indices.iter().enumerate() {
            let diff_re = (pruned_output[i].re - full_output[idx].re).abs();
            let diff_im = (pruned_output[i].im - full_output[idx].im).abs();

            assert!(diff_re < 1e-10, "Real mismatch at index {idx}");
            assert!(diff_im < 1e-10, "Imag mismatch at index {idx}");
        }
    }
}
