//! Sparse FFT plan.

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};

use super::bucket::BucketArray;
use super::decoder::PeelingDecoder;
use super::filter::{create_optimal_filter, AliasingFilter};
use super::hash::{generate_coprime_factors, FrequencyHash};
use super::problem::SparseProblem;
use super::result::SparseResult;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Sparse FFT plan for k-sparse signals.
///
/// Pre-computes subsampling factors, filters, and internal FFT plans
/// for efficient repeated sparse FFT computation.
pub struct SparsePlan<T: Float> {
    /// Signal length.
    n: usize,
    /// Expected sparsity.
    k: usize,
    /// Number of buckets per stage.
    num_buckets: usize,
    /// Number of stages (hash functions).
    num_stages: usize,
    /// Subsampling factors (coprime).
    subsample_factors: Vec<usize>,
    /// Hash functions for each stage.
    hash_functions: Vec<FrequencyHash>,
    /// Aliasing filter.
    filter: AliasingFilter<T>,
    /// Internal FFT plans for bucket computation.
    bucket_plans: Vec<Plan<T>>,
    /// Detection threshold.
    threshold: T,
    /// Planning flags.
    flags: Flags,
}

impl<T: Float> SparsePlan<T> {
    /// Create a new sparse FFT plan.
    ///
    /// # Arguments
    ///
    /// * `n` - Signal length
    /// * `k` - Expected sparsity (max non-zero frequencies)
    /// * `flags` - Planning flags
    ///
    /// # Returns
    ///
    /// `None` if plan creation fails (e.g., invalid parameters).
    pub fn new(n: usize, k: usize, flags: Flags) -> Option<Self> {
        if n == 0 || k == 0 || k > n {
            return None;
        }

        let problem: SparseProblem<T> = SparseProblem::new(n, k, Direction::Forward);

        // Calculate optimal parameters
        let num_buckets = problem.optimal_buckets();
        let num_stages = problem.optimal_repetitions().max(2);

        // Generate coprime subsampling factors
        let subsample_factors = generate_coprime_factors(num_buckets, n / 2);
        let num_stages = num_stages.min(subsample_factors.len());

        // Create hash functions for each stage
        let hash_functions: Vec<_> = (0..num_stages)
            .map(|i| {
                let bucket_count = if i < subsample_factors.len() {
                    subsample_factors[i]
                } else {
                    num_buckets
                };
                FrequencyHash::new(bucket_count, n)
            })
            .collect();

        // Create aliasing filter
        let filter = create_optimal_filter(n, k, num_buckets);

        // Create internal FFT plans for each stage
        let bucket_plans: Vec<_> = hash_functions
            .iter()
            .filter_map(|h| Plan::dft_1d(h.num_buckets(), Direction::Forward, flags))
            .collect();

        if bucket_plans.len() != num_stages {
            return None; // Failed to create some plans
        }

        // Set detection threshold based on expected signal magnitude
        let threshold = T::from_f64(1e-10);

        Some(Self {
            n,
            k,
            num_buckets,
            num_stages,
            subsample_factors,
            hash_functions,
            filter,
            bucket_plans,
            threshold,
            flags,
        })
    }

    /// Execute the sparse FFT.
    ///
    /// # Arguments
    ///
    /// * `input` - Input signal (length n)
    ///
    /// # Returns
    ///
    /// `SparseResult` containing detected frequencies and values.
    pub fn execute(&self, input: &[Complex<T>]) -> SparseResult<T> {
        if input.len() != self.n {
            return SparseResult::empty();
        }

        // Stage 1: Subsampling and bucket FFTs
        let mut bucket_stages = self.compute_bucket_stages(input);

        // Stage 2: Singleton detection and peeling
        let mut decoder = PeelingDecoder::new(self.n, self.k, self.threshold);
        decoder.decode(&mut bucket_stages)
    }

    /// Compute bucket stages from input signal.
    fn compute_bucket_stages(&self, input: &[Complex<T>]) -> Vec<BucketArray<T>> {
        let mut stages = Vec::with_capacity(self.num_stages);

        for stage_idx in 0..self.num_stages {
            let hash = &self.hash_functions[stage_idx];
            let bucket_count = hash.num_buckets();

            // Subsample the input
            let subsample_factor = if stage_idx < self.subsample_factors.len() {
                self.subsample_factors[stage_idx]
            } else {
                1
            };

            let subsampled = self.subsample(input, subsample_factor, bucket_count);

            // Apply filter (in frequency domain)
            let filtered = self.apply_filter(&subsampled);

            // FFT to get bucket values
            let bucket_fft = self.bucket_fft(&filtered, stage_idx);

            // Create bucket array
            let mut buckets = BucketArray::new(bucket_count, subsample_factor, self.n);
            buckets.fill_from_fft(&bucket_fft);

            stages.push(buckets);
        }

        // Analyze singletons across stages
        if stages.len() >= 2 {
            let (first, rest) = stages.split_at_mut(1);
            if !rest.is_empty() {
                first[0].analyze_singletons(&rest[0], self.threshold);
            }
        }

        stages
    }

    /// Subsample the input signal.
    fn subsample(&self, input: &[Complex<T>], factor: usize, output_len: usize) -> Vec<Complex<T>> {
        let mut output = vec![Complex::<T>::zero(); output_len];

        // Sum elements that alias to each output bin
        for (i, &val) in input.iter().enumerate() {
            let out_idx = i % output_len;
            output[out_idx] = output[out_idx] + val;
        }

        // Apply phase correction based on subsampling factor
        let two_pi = <T as Float>::PI + <T as Float>::PI;
        for (i, out) in output.iter_mut().enumerate() {
            let phase = two_pi * T::from_usize(i * factor) / T::from_usize(self.n);
            let (sin_p, cos_p) = Float::sin_cos(phase);
            let twiddle = Complex::new(cos_p, -sin_p);
            *out = *out * twiddle;
        }

        output
    }

    /// Apply aliasing filter.
    fn apply_filter(&self, signal: &[Complex<T>]) -> Vec<Complex<T>> {
        // For simplicity, apply filter element-wise with periodic extension
        let filter_len = self.filter.coeffs.len();

        signal
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let filter_idx = i % filter_len;
                val * self.filter.coeffs[filter_idx]
            })
            .collect()
    }

    /// Compute bucket FFT for a stage.
    fn bucket_fft(&self, input: &[Complex<T>], stage_idx: usize) -> Vec<Complex<T>> {
        let bucket_count = self.hash_functions[stage_idx].num_buckets();

        // Ensure input matches expected size
        let input_adjusted: Vec<Complex<T>> = if input.len() >= bucket_count {
            input[..bucket_count].to_vec()
        } else {
            let mut adjusted = input.to_vec();
            adjusted.resize(bucket_count, Complex::<T>::zero());
            adjusted
        };

        let mut output = vec![Complex::<T>::zero(); bucket_count];

        if stage_idx < self.bucket_plans.len() {
            self.bucket_plans[stage_idx].execute(&input_adjusted, &mut output);
        }

        output
    }

    /// Get signal length.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get expected sparsity.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Get number of buckets.
    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    /// Get number of stages.
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    /// Get the planning flags.
    pub fn flags(&self) -> Flags {
        self.flags
    }

    /// Set the detection threshold.
    pub fn set_threshold(&mut self, threshold: T) {
        self.threshold = threshold;
    }

    /// Get the detection threshold.
    pub fn threshold(&self) -> T {
        self.threshold
    }

    /// Estimate the computational complexity.
    ///
    /// Returns estimated number of operations.
    pub fn estimated_ops(&self) -> usize {
        // Sparse FFT: O(k log n) operations
        // More precisely: O(B log B * num_stages + k * num_stages)
        let b = self.num_buckets;
        let log_b = (b as f64).log2().ceil() as usize;

        // Bucket FFTs
        let bucket_fft_ops = self.num_stages * b * log_b;

        // Subsampling
        let subsample_ops = self.n;

        // Decoding
        let decode_ops = self.k * self.num_stages;

        bucket_fft_ops + subsample_ops + decode_ops
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_plan_creation() {
        let plan: Option<SparsePlan<f64>> = SparsePlan::new(1024, 10, Flags::ESTIMATE);
        assert!(plan.is_some());

        let plan = plan.unwrap();
        assert_eq!(plan.n(), 1024);
        assert_eq!(plan.k(), 10);
        assert!(plan.num_stages() >= 2);
    }

    #[test]
    fn test_sparse_plan_invalid() {
        // Zero length
        assert!(SparsePlan::<f64>::new(0, 10, Flags::ESTIMATE).is_none());

        // Zero sparsity
        assert!(SparsePlan::<f64>::new(1024, 0, Flags::ESTIMATE).is_none());

        // k > n
        assert!(SparsePlan::<f64>::new(10, 100, Flags::ESTIMATE).is_none());
    }

    #[test]
    fn test_sparse_plan_execute() {
        let n = 256;
        let k = 5;

        let plan = SparsePlan::<f64>::new(n, k, Flags::ESTIMATE).unwrap();

        // Create a simple sparse signal
        let mut input = vec![Complex::new(0.0_f64, 0.0); n];

        // Add a few frequency components
        let two_pi = core::f64::consts::PI * 2.0;
        for i in 0..n {
            let t = i as f64 / n as f64;
            // Frequency at bin 10
            input[i].re += (two_pi * 10.0 * t).cos();
            input[i].im += (two_pi * 10.0 * t).sin();
        }

        let result = plan.execute(&input);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_estimated_ops() {
        let plan = SparsePlan::<f64>::new(1024, 10, Flags::ESTIMATE).unwrap();

        let ops = plan.estimated_ops();
        // Should be much less than O(n log n) = 10240
        assert!(ops < 5000);
    }

    #[test]
    fn test_threshold() {
        let mut plan = SparsePlan::<f64>::new(256, 5, Flags::ESTIMATE).unwrap();

        plan.set_threshold(0.001);
        assert_eq!(plan.threshold(), 0.001);
    }
}
