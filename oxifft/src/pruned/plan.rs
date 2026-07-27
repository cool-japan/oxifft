//! Pruned FFT plan for repeated transforms.

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Ceiling of log2(n); returns at least 1 so it is safe as a cost/threshold.
#[inline]
fn log2_ceil(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    (usize::BITS - (n - 1).leading_zeros()) as usize
}

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
    /// Inner full-size FFT plan, used as the fallback when pruning is not
    /// beneficial (few indices pruned away, i.e. the requested set is close to
    /// the whole transform).  Consulted by every `execute_*` method via the
    /// per-mode benefit heuristic (`log2_ceil`-based crossover): when the direct
    /// pruned evaluation would cost more than a full FFT, this plan runs instead.
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

        // Benefit heuristic: per-output Goertzel costs O(M·N); once M reaches
        // ~log2(N) a full FFT (O(N log N)) is cheaper, so dispatch to the inner
        // full-FFT plan and select the requested bins.  This is the "fallback
        // when pruning is not beneficial" path.
        let m = desired_indices.len();
        if m >= log2_ceil(self.n) {
            if let Some(inner) = self.inner_plan.as_ref() {
                let mut full = vec![Complex::<T>::zero(); self.n];
                inner.execute(input, &mut full);
                for (out_idx, &freq_idx) in desired_indices.iter().enumerate() {
                    output[out_idx] = if freq_idx < self.n {
                        full[freq_idx]
                    } else {
                        Complex::<T>::zero()
                    };
                }
                return;
            }
        }

        // Few outputs requested: Goertzel per output.
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

        // Benefit heuristic: the direct sparse-input DFT costs O(K·N).  Once the
        // number of non-zero inputs K reaches ~log2(N), scattering into a full
        // buffer and running the inner full-size FFT (O(N log N)) is cheaper.
        let k_nonzero = nonzero_indices.len();
        if k_nonzero >= log2_ceil(self.n) {
            if let Some(inner) = self.inner_plan.as_ref() {
                let mut full_in = vec![Complex::<T>::zero(); self.n];
                for (i, &m) in nonzero_indices.iter().enumerate() {
                    if m < self.n {
                        full_in[m] = input[i];
                    }
                }
                inner.execute(&full_in, output);
                return;
            }
        }

        // Few non-zero inputs: direct DFT.
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

        // Benefit heuristic: the dual-pruned direct evaluation costs O(K·M).
        // When that exceeds the full-FFT cost O(N log N), scatter the sparse
        // input, run the inner full-size FFT, and select the requested outputs.
        let k_nonzero = input_indices.len();
        let m_out = output_indices.len();
        let full_cost = self.n.saturating_mul(log2_ceil(self.n));
        if k_nonzero.saturating_mul(m_out) >= full_cost {
            if let Some(inner) = self.inner_plan.as_ref() {
                let mut full_in = vec![Complex::<T>::zero(); self.n];
                for (i, &m) in input_indices.iter().enumerate() {
                    if m < self.n {
                        full_in[m] = input[i];
                    }
                }
                let mut full_out = vec![Complex::<T>::zero(); self.n];
                inner.execute(&full_in, &mut full_out);
                for (out_idx, &freq_idx) in output_indices.iter().enumerate() {
                    output[out_idx] = if freq_idx < self.n {
                        full_out[freq_idx]
                    } else {
                        Complex::<T>::zero()
                    };
                }
                return;
            }
        }

        let num_inputs = input_indices.len();

        // Few input×output pairs: direct evaluation with precomputed twiddles.
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
    fn test_log2_ceil() {
        assert_eq!(log2_ceil(1), 1);
        assert_eq!(log2_ceil(2), 1);
        assert_eq!(log2_ceil(3), 2);
        assert_eq!(log2_ceil(64), 6);
        assert_eq!(log2_ceil(1024), 10);
    }

    /// Reference full FFT of an input vector.
    fn full_fft(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let n = input.len();
        let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
        let mut out = vec![Complex::new(0.0_f64, 0.0); n];
        plan.execute(input, &mut out);
        out
    }

    /// Output-pruned plan: large M (M ≥ log2(N)) must route through the inner
    /// full-FFT fallback and still match the full FFT exactly.
    #[test]
    fn test_output_pruned_inner_plan_route() {
        let n = 64;
        // M = 40 ≥ log2(64) = 6 → inner_plan fallback route.
        let indices: Vec<usize> = (0..40).collect();
        let plan: PrunedPlan<f64> =
            PrunedPlan::output_pruned(n, &indices, Flags::ESTIMATE).unwrap();

        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64 * 0.2).sin(), (i as f64 * 0.1).cos()))
            .collect();
        let mut out = vec![Complex::new(0.0_f64, 0.0); indices.len()];
        plan.execute(&input, &mut out);

        let full = full_fft(&input);
        for (i, &idx) in indices.iter().enumerate() {
            assert!((out[i].re - full[idx].re).abs() < 1e-9, "re idx {idx}");
            assert!((out[i].im - full[idx].im).abs() < 1e-9, "im idx {idx}");
        }
    }

    /// Output-pruned plan: small M (M < log2(N)) uses the Goertzel route and
    /// must also match the full FFT.
    #[test]
    fn test_output_pruned_goertzel_route() {
        let n = 256;
        let indices = vec![3usize, 17, 40]; // M = 3 < log2(256) = 8
        let plan: PrunedPlan<f64> =
            PrunedPlan::output_pruned(n, &indices, Flags::ESTIMATE).unwrap();

        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64 * 0.05).cos(), 0.0))
            .collect();
        let mut out = vec![Complex::new(0.0_f64, 0.0); indices.len()];
        plan.execute(&input, &mut out);

        let full = full_fft(&input);
        for (i, &idx) in indices.iter().enumerate() {
            assert!((out[i].re - full[idx].re).abs() < 1e-9);
            assert!((out[i].im - full[idx].im).abs() < 1e-9);
        }
    }

    /// Input-pruned plan: many non-zero inputs (K ≥ log2(N)) must route through
    /// the inner full-FFT fallback (scatter + FFT) and match the full FFT.
    #[test]
    fn test_input_pruned_inner_plan_route() {
        let n = 64;
        // K = 20 ≥ log2(64) = 6 → inner_plan fallback route.
        // Deduplicate indices (i*3 % 64 is unique for i<20 here, but be safe).
        let mut seen = std::collections::BTreeSet::new();
        let input_indices: Vec<usize> = (0..20)
            .map(|i| i * 3 % n)
            .filter(|x| seen.insert(*x))
            .collect();

        let plan: PrunedPlan<f64> =
            PrunedPlan::input_pruned(n, &input_indices, Flags::ESTIMATE).unwrap();

        let values: Vec<Complex<f64>> = (0..input_indices.len())
            .map(|i| Complex::new((i as f64 + 1.0) * 0.5, (i as f64) * 0.25))
            .collect();
        let mut out = vec![Complex::new(0.0_f64, 0.0); n];
        plan.execute(&values, &mut out);

        // Reference: scatter into a dense buffer and run a full FFT.
        let mut dense = vec![Complex::new(0.0_f64, 0.0); n];
        for (i, &idx) in input_indices.iter().enumerate() {
            dense[idx] = values[i];
        }
        let full = full_fft(&dense);
        for i in 0..n {
            assert!((out[i].re - full[i].re).abs() < 1e-9, "re bin {i}");
            assert!((out[i].im - full[i].im).abs() < 1e-9, "im bin {i}");
        }
    }

    /// Both-pruned plan: a large K·M product must route through the inner
    /// full-FFT fallback and match the full FFT at the requested outputs.
    #[test]
    fn test_both_pruned_inner_plan_route() {
        let n = 64;
        // K·M = 16·16 = 256 ≥ N·log2(N) = 384? No — pick larger to force route.
        let input_indices: Vec<usize> = (0..40).map(|i| i % n).collect();
        let output_indices: Vec<usize> = (0..40).collect();
        // K·M = 1600 ≥ 64·6 = 384 → inner_plan route.
        let plan: PrunedPlan<f64> =
            PrunedPlan::both_pruned(n, &input_indices, &output_indices, Flags::ESTIMATE).unwrap();

        let values: Vec<Complex<f64>> = (0..input_indices.len())
            .map(|i| Complex::new((i as f64) * 0.1, (i as f64) * 0.05))
            .collect();
        let mut out = vec![Complex::new(0.0_f64, 0.0); output_indices.len()];
        plan.execute(&values, &mut out);

        let mut dense = vec![Complex::new(0.0_f64, 0.0); n];
        for (i, &idx) in input_indices.iter().enumerate() {
            dense[idx] = dense[idx] + values[i];
        }
        let full = full_fft(&dense);
        for (i, &idx) in output_indices.iter().enumerate() {
            assert!((out[i].re - full[idx].re).abs() < 1e-9, "re out {idx}");
            assert!((out[i].im - full[idx].im).abs() < 1e-9, "im out {idx}");
        }
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
