//! Sparse FFT problem definition.

use crate::api::Direction;
use crate::kernel::Float;

/// Sparse FFT problem specification.
///
/// Defines a sparse FFT computation with expected sparsity k.
#[derive(Debug, Clone)]
pub struct SparseProblem<T: Float> {
    /// Signal length.
    pub n: usize,
    /// Expected sparsity (max non-zero frequencies).
    pub k: usize,
    /// Transform direction.
    pub direction: Direction,
    /// Noise tolerance factor (0.0 = exact, higher = more tolerant).
    pub noise_tolerance: T,
}

impl<T: Float> SparseProblem<T> {
    /// Create a new sparse FFT problem.
    ///
    /// # Arguments
    ///
    /// * `n` - Signal length
    /// * `k` - Expected sparsity
    /// * `direction` - Transform direction
    pub fn new(n: usize, k: usize, direction: Direction) -> Self {
        Self {
            n,
            k,
            direction,
            noise_tolerance: T::ZERO,
        }
    }

    /// Set noise tolerance factor.
    ///
    /// Higher values make the algorithm more tolerant to noise but may
    /// produce less accurate results.
    pub fn with_noise_tolerance(mut self, tolerance: T) -> Self {
        self.noise_tolerance = tolerance;
        self
    }

    /// Check if the problem is valid.
    pub fn is_valid(&self) -> bool {
        self.n > 0 && self.k > 0 && self.k <= self.n
    }

    /// Check if sparse FFT is beneficial for this problem.
    ///
    /// Sparse FFT is generally beneficial when k << n.
    /// Returns false if standard FFT would be faster.
    pub fn is_sparse_beneficial(&self) -> bool {
        // Heuristic: sparse FFT is beneficial when k < n/16
        // This accounts for the overhead of bucketization and decoding
        self.k < self.n / 16 && self.n >= 128
    }

    /// Calculate optimal number of buckets.
    ///
    /// The number of buckets B should be O(k) for optimal complexity.
    pub fn optimal_buckets(&self) -> usize {
        // B = c * k where c is a small constant (typically 2-4)
        let c = 3;
        (c * self.k).max(16).min(self.n)
    }

    /// Calculate optimal number of repetitions for robustness.
    ///
    /// More repetitions increase accuracy but also runtime.
    pub fn optimal_repetitions(&self) -> usize {
        // O(log(n/k)) repetitions for high probability success
        let ratio = (self.n as f64) / (self.k as f64);
        let reps = (ratio.ln() / 2.0_f64.ln()).ceil() as usize;
        reps.max(1).min(10)
    }

    /// Get subsampling factors based on Chinese Remainder Theorem.
    ///
    /// Returns a set of coprime factors for CRT-based subsampling.
    pub fn crt_factors(&self) -> Vec<usize> {
        // Choose small coprime numbers whose product covers n
        // These are used for aliasing-based frequency bucketization
        let b = self.optimal_buckets();

        // Find small primes for subsampling
        let mut factors = Vec::new();
        let small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];

        let mut product = 1usize;
        for &p in &small_primes {
            if product >= b {
                break;
            }
            factors.push(p);
            product *= p;
        }

        // Ensure we have at least 2 factors
        if factors.len() < 2 {
            factors = vec![2, 3];
        }

        factors
    }
}

impl<T: Float> Default for SparseProblem<T> {
    fn default() -> Self {
        Self {
            n: 0,
            k: 0,
            direction: Direction::Forward,
            noise_tolerance: T::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_problem_creation() {
        let problem: SparseProblem<f64> = SparseProblem::new(1024, 10, Direction::Forward);
        assert_eq!(problem.n, 1024);
        assert_eq!(problem.k, 10);
        assert!(problem.is_valid());
    }

    #[test]
    fn test_sparse_beneficial() {
        // k << n should be beneficial
        let problem1: SparseProblem<f64> = SparseProblem::new(1024, 10, Direction::Forward);
        assert!(problem1.is_sparse_beneficial());

        // k close to n should not be beneficial
        let problem2: SparseProblem<f64> = SparseProblem::new(1024, 512, Direction::Forward);
        assert!(!problem2.is_sparse_beneficial());

        // Small n should not be beneficial
        let problem3: SparseProblem<f64> = SparseProblem::new(64, 4, Direction::Forward);
        assert!(!problem3.is_sparse_beneficial());
    }

    #[test]
    fn test_optimal_buckets() {
        let problem: SparseProblem<f64> = SparseProblem::new(1024, 10, Direction::Forward);
        let buckets = problem.optimal_buckets();
        // Should be O(k) = O(30) for c=3
        assert!(buckets >= 16);
        assert!(buckets <= 1024);
    }

    #[test]
    fn test_crt_factors() {
        let problem: SparseProblem<f64> = SparseProblem::new(1024, 10, Direction::Forward);
        let factors = problem.crt_factors();
        assert!(factors.len() >= 2);

        // All factors should be coprime
        for i in 0..factors.len() {
            for j in i + 1..factors.len() {
                assert_eq!(gcd(factors[i], factors[j]), 1);
            }
        }
    }

    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
}
