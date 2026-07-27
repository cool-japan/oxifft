//! Sparse FFT plan.
//!
//! Implements an aliasing-based sparse FFT with a *certified* fast path and a
//! guaranteed-correct dense fallback.  See [`super`] for the algorithm and its
//! accuracy guarantees.

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};
use crate::prelude::*;

use super::decoder::{certified_peel, AliasStage};
use super::problem::SparseProblem;
use super::result::SparseResult;

/// Select the bucket counts (one per aliasing stage) for a signal of length
/// `n` given a target bucket count.
///
/// Each stage's bucket count must **divide `n`** (so the downsampling is exact)
/// and must be at most `n / 2` (so the downsampling factor `L = n / B` is at
/// least 2 and the phase-shift measurement is well defined).  We choose the
/// smallest few such divisors that are `>= target`; using several distinct
/// bucket counts lets the peeling decoder separate frequencies that happen to
/// collide under one bucketization but not another.
///
/// Returns an empty vector when `n` has no suitable divisor (e.g. `n` prime, or
/// `target > n / 2`), in which case the plan uses the dense path exclusively.
fn select_stage_sizes(n: usize, target: usize) -> Vec<usize> {
    const MAX_STAGES: usize = 3;

    if n < 4 {
        return Vec::new();
    }
    let max_b = n / 2;
    let target = target.clamp(2, max_b.max(2));

    // Enumerate all divisors of n in [target, n/2].
    let mut divisors: Vec<usize> = Vec::new();
    let mut i = 1usize;
    while i * i <= n {
        if n.is_multiple_of(i) {
            let j = n / i;
            if (target..=max_b).contains(&i) {
                divisors.push(i);
            }
            if j != i && (target..=max_b).contains(&j) {
                divisors.push(j);
            }
        }
        i += 1;
    }

    divisors.sort_unstable();
    divisors.dedup();
    divisors.truncate(MAX_STAGES);
    divisors
}

/// Sparse FFT plan for k-sparse signals.
///
/// Pre-computes the aliasing-stage bucket counts and their internal FFT plans
/// for efficient repeated sparse FFT computation.  Every call is guaranteed to
/// return a correct result: the certified fast path runs first, and if it
/// cannot certify its solution the plan falls back to a full FFT plus top-`k`
/// selection.
pub struct SparsePlan<T: Float> {
    /// Signal length.
    n: usize,
    /// Expected sparsity.
    k: usize,
    /// Largest stage bucket count (0 when the fast path is unavailable).
    num_buckets: usize,
    /// Bucket count of each aliasing stage (each divides `n`, ascending).
    stage_bucket_sizes: Vec<usize>,
    /// Internal FFT plans, one per stage bucket count.
    bucket_plans: Vec<Plan<T>>,
    /// Full-size FFT plan used by the dense fallback.
    full_plan: Plan<T>,
    /// Absolute detection threshold: energy below this is treated as noise.
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
    /// `None` if `n == 0`, `k == 0`, `k > n`, or a full-size FFT plan for `n`
    /// cannot be created.
    pub fn new(n: usize, k: usize, flags: Flags) -> Option<Self> {
        if n == 0 || k == 0 || k > n {
            return None;
        }

        // The dense fallback must always be available.
        let full_plan = Plan::dft_1d(n, Direction::Forward, flags)?;

        let problem: SparseProblem<T> = SparseProblem::new(n, k, Direction::Forward);
        let target = problem.optimal_buckets();
        let candidate_sizes = select_stage_sizes(n, target);

        // Build a bucket FFT plan for each candidate stage, dropping any stage
        // whose plan cannot be created so the two vectors stay aligned.
        let mut stage_bucket_sizes = Vec::with_capacity(candidate_sizes.len());
        let mut bucket_plans = Vec::with_capacity(candidate_sizes.len());
        for &b in &candidate_sizes {
            if let Some(plan) = Plan::dft_1d(b, Direction::Forward, flags) {
                stage_bucket_sizes.push(b);
                bucket_plans.push(plan);
            }
        }

        let num_buckets = stage_bucket_sizes.iter().copied().max().unwrap_or(0);
        let threshold = T::from_f64(1e-10);

        Some(Self {
            n,
            k,
            num_buckets,
            stage_bucket_sizes,
            bucket_plans,
            full_plan,
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
    /// `SparseResult` containing detected frequencies and values.  The fast
    /// path is used only when it can certify its answer; otherwise the result
    /// comes from a full FFT plus top-`k` selection, so the returned pairs are
    /// always the correct top-`k` bins.
    pub fn execute(&self, input: &[Complex<T>]) -> SparseResult<T> {
        if input.len() != self.n {
            return SparseResult::empty();
        }

        if let Some(result) = self.try_fast(input) {
            return result;
        }
        self.dense_topk(input)
    }

    /// Attempt the certified fast path.  Returns `None` when the fast path is
    /// unavailable or cannot certify its solution.
    fn try_fast(&self, input: &[Complex<T>]) -> Option<SparseResult<T>> {
        if self.stage_bucket_sizes.is_empty() {
            return None;
        }

        // Build one aliasing stage per bucket count.
        let mut stages: Vec<AliasStage<T>> = Vec::with_capacity(self.stage_bucket_sizes.len());
        for (si, &b_count) in self.stage_bucket_sizes.iter().enumerate() {
            let l = self.n / b_count;

            // Downsample by L at offsets 0 and 1 (each reads only `b_count`
            // samples — this is what makes the fast path sub-linear).
            let z0_in: Vec<Complex<T>> = (0..b_count).map(|j| input[(j * l) % self.n]).collect();
            let z1_in: Vec<Complex<T>> =
                (0..b_count).map(|j| input[(j * l + 1) % self.n]).collect();

            let mut z0 = vec![Complex::<T>::zero(); b_count];
            let mut z1 = vec![Complex::<T>::zero(); b_count];
            self.bucket_plans[si].execute(&z0_in, &mut z0);
            self.bucket_plans[si].execute(&z1_in, &mut z1);

            // Convert to coefficient space (multiply by L) so a singleton
            // bucket holds X[f] directly.
            let l_scale = T::from_usize(l);
            for v in z0.iter_mut() {
                *v = *v * l_scale;
            }
            for v in z1.iter_mut() {
                *v = *v * l_scale;
            }

            stages.push(AliasStage {
                b_count,
                coeff0: z0,
                coeff1: z1,
            });
        }

        let abs_threshold = self.threshold.to_f64().unwrap_or(0.0);
        let recovered = certified_peel(&mut stages, self.n, self.n, abs_threshold)?;

        Some(Self::top_k(recovered, self.k, self.n))
    }

    /// Dense fallback: full FFT then top-`k` bins by magnitude.
    fn dense_topk(&self, input: &[Complex<T>]) -> SparseResult<T> {
        let mut output = vec![Complex::<T>::zero(); self.n];
        self.full_plan.execute(input, &mut output);

        let mut magnitudes: Vec<(usize, T)> = output
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.norm_sqr()))
            .collect();
        magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        let k_actual = self.k.min(self.n);
        let indices: Vec<usize> = magnitudes[..k_actual].iter().map(|(i, _)| *i).collect();
        let values: Vec<Complex<T>> = indices.iter().map(|&i| output[i]).collect();
        SparseResult::new(indices, values, self.n)
    }

    /// Sort recovered `(freq, value)` pairs by descending magnitude and keep
    /// the top `k`.
    fn top_k(mut pairs: Vec<(usize, Complex<T>)>, k: usize, n: usize) -> SparseResult<T> {
        pairs.sort_by(|a, b| {
            b.1.norm_sqr()
                .partial_cmp(&a.1.norm_sqr())
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        let k_actual = k.min(pairs.len());
        let indices: Vec<usize> = pairs[..k_actual].iter().map(|(i, _)| *i).collect();
        let values: Vec<Complex<T>> = pairs[..k_actual].iter().map(|(_, v)| *v).collect();
        SparseResult::new(indices, values, n)
    }

    /// Get signal length.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get expected sparsity.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Get the largest aliasing-stage bucket count (0 when only the dense path
    /// is available).
    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    /// Get the number of aliasing stages.
    pub fn num_stages(&self) -> usize {
        self.stage_bucket_sizes.len()
    }

    /// Get the planning flags.
    pub fn flags(&self) -> Flags {
        self.flags
    }

    /// Set the detection threshold (absolute magnitude floor).
    pub fn set_threshold(&mut self, threshold: T) {
        self.threshold = threshold;
    }

    /// Get the detection threshold.
    pub fn threshold(&self) -> T {
        self.threshold
    }

    /// Estimate the computational complexity of the fast path, in operations.
    ///
    /// When the fast path is available this reflects the O(k log n)-style cost
    /// of the aliasing stages; otherwise it reports the dense O(n log n) cost.
    pub fn estimated_ops(&self) -> usize {
        let log_n = libm::ceil(libm::log2(self.n.max(2) as f64)) as usize;

        if self.stage_bucket_sizes.is_empty() {
            // Dense path only.
            return self.n.saturating_mul(log_n.max(1));
        }

        // Bucket FFTs (two offset measurements per stage).
        let mut bucket_fft_ops = 0usize;
        for &b in &self.stage_bucket_sizes {
            let log_b = libm::ceil(libm::log2(b.max(2) as f64)) as usize;
            bucket_fft_ops += 2 * b * log_b.max(1);
        }

        // Downsampling reads plus peeling.
        let sample_ops: usize = self.stage_bucket_sizes.iter().map(|&b| 2 * b).sum();
        let decode_ops = self.k.saturating_mul(self.stage_bucket_sizes.len());

        bucket_fft_ops + sample_ops + decode_ops
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_plan_creation() {
        let plan: Option<SparsePlan<f64>> = SparsePlan::new(1024, 10, Flags::ESTIMATE);
        assert!(plan.is_some());

        let plan = plan.expect("plan creation should succeed for valid params");
        assert_eq!(plan.n(), 1024);
        assert_eq!(plan.k(), 10);
        assert!(plan.num_stages() >= 1);
    }

    #[test]
    fn test_sparse_plan_invalid() {
        assert!(SparsePlan::<f64>::new(0, 10, Flags::ESTIMATE).is_none());
        assert!(SparsePlan::<f64>::new(1024, 0, Flags::ESTIMATE).is_none());
        assert!(SparsePlan::<f64>::new(10, 100, Flags::ESTIMATE).is_none());
    }

    /// Plan-based execution must recover a planted single tone exactly.
    #[test]
    fn test_sparse_plan_execute_single_tone() {
        let n = 256;
        let freq = 37;
        let plan =
            SparsePlan::<f64>::new(n, 4, Flags::ESTIMATE).expect("plan creation should succeed");

        let two_pi = core::f64::consts::PI * 2.0;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|t| {
                let angle = two_pi * (freq as f64) * (t as f64) / (n as f64);
                Complex::new(angle.cos(), angle.sin())
            })
            .collect();

        let result = plan.execute(&input);
        assert_eq!(result.indices.len(), 1, "exactly one tone");
        assert_eq!(result.indices[0], freq);
        // X[freq] = n for a unit-amplitude exponential.
        assert!((result.values[0].re - n as f64).abs() < 1e-6);
        assert!(result.values[0].im.abs() < 1e-6);
    }

    #[test]
    fn test_estimated_ops() {
        let plan = SparsePlan::<f64>::new(1024, 10, Flags::ESTIMATE)
            .expect("plan creation should succeed");
        let ops = plan.estimated_ops();
        // Fast path must be far cheaper than dense O(n log n) = 10240.
        assert!(ops < 5000, "estimated_ops = {ops}");
    }

    #[test]
    fn test_threshold() {
        let mut plan =
            SparsePlan::<f64>::new(256, 5, Flags::ESTIMATE).expect("plan creation should succeed");
        plan.set_threshold(0.001);
        assert_eq!(plan.threshold(), 0.001);
    }

    #[test]
    fn test_select_stage_sizes_power_of_two() {
        // n = 1024, target = 30 -> smallest divisors >= 30 and <= 512.
        let sizes = select_stage_sizes(1024, 30);
        assert_eq!(sizes, vec![32, 64, 128]);
        for &b in &sizes {
            assert_eq!(1024 % b, 0);
            assert!(b <= 512);
        }
    }

    #[test]
    fn test_select_stage_sizes_prime_is_empty() {
        // A prime n has no divisor in [target, n/2].
        let sizes = select_stage_sizes(257, 16);
        assert!(sizes.is_empty());
    }

    /// When no fast stage exists the plan still returns correct dense results.
    #[test]
    fn test_prime_length_falls_back_to_dense() {
        let n = 257; // prime
        let freq = 40;
        let plan =
            SparsePlan::<f64>::new(n, 3, Flags::ESTIMATE).expect("plan creation should succeed");
        assert_eq!(plan.num_stages(), 0, "prime length has no fast stage");

        let two_pi = core::f64::consts::PI * 2.0;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|t| {
                let angle = two_pi * (freq as f64) * (t as f64) / (n as f64);
                Complex::new(angle.cos(), angle.sin())
            })
            .collect();

        let result = plan.execute(&input);
        // Dense top-k still recovers the dominant bin.
        assert!(result.indices.contains(&freq));
    }
}
