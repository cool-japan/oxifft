//! Peeling decoder for sparse FFT.
//!
//! Implements the peeling-based decoding algorithm that iteratively
//! extracts frequency components from bucketized observations.

// Allow dead code - infrastructure for future algorithm enhancements
#![allow(dead_code)]

use crate::kernel::{Complex, Float};

use super::bucket::{Bucket, BucketArray};
use super::result::SparseResult;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Peeling decoder state.
#[derive(Debug, Clone)]
pub struct PeelingDecoder<T: Float> {
    /// Signal length.
    n: usize,
    /// Expected sparsity.
    k: usize,
    /// Detected frequencies and values.
    detected: Vec<(usize, Complex<T>)>,
    /// Magnitude threshold for detection.
    threshold: T,
    /// Maximum iterations.
    max_iterations: usize,
}

impl<T: Float> PeelingDecoder<T> {
    /// Create a new peeling decoder.
    ///
    /// # Arguments
    ///
    /// * `n` - Signal length
    /// * `k` - Expected sparsity
    /// * `threshold` - Magnitude threshold for frequency detection
    pub fn new(n: usize, k: usize, threshold: T) -> Self {
        Self {
            n,
            k,
            detected: Vec::with_capacity(k),
            threshold,
            max_iterations: k * 3, // Allow some extra iterations
        }
    }

    /// Run the peeling decoder on bucket observations.
    ///
    /// # Arguments
    ///
    /// * `bucket_stages` - Multiple stages of bucket observations
    ///
    /// # Returns
    ///
    /// Sparse result containing detected frequencies.
    pub fn decode(&mut self, bucket_stages: &mut [BucketArray<T>]) -> SparseResult<T> {
        self.detected.clear();

        if bucket_stages.is_empty() {
            return SparseResult::empty();
        }

        // Iterative peeling algorithm
        for _iter in 0..self.max_iterations {
            if self.detected.len() >= self.k {
                break;
            }

            // Find singleton buckets across all stages
            let singletons = self.find_singletons(bucket_stages);

            if singletons.is_empty() {
                // No more singletons found, check for remaining multitons
                break;
            }

            // Peel off detected frequencies
            for (freq, value) in singletons {
                if !self.is_already_detected(freq) {
                    self.detected.push((freq, value));

                    // Remove contribution from all bucket stages
                    self.peel_frequency(bucket_stages, freq, value);
                }
            }
        }

        // Handle any remaining multitons with heuristics
        self.resolve_multitons(bucket_stages);

        // Build result
        let indices: Vec<usize> = self.detected.iter().map(|(i, _)| *i).collect();
        let values: Vec<Complex<T>> = self.detected.iter().map(|(_, v)| *v).collect();

        SparseResult::new(indices, values, self.n)
    }

    /// Find singleton buckets across all stages.
    fn find_singletons(&self, bucket_stages: &[BucketArray<T>]) -> Vec<(usize, Complex<T>)> {
        let mut singletons = Vec::new();

        // For each stage, look for buckets with single dominant frequency
        for (stage_idx, stage) in bucket_stages.iter().enumerate() {
            for bucket_idx in 0..stage.len() {
                if let Some(bucket) = stage.get(bucket_idx) {
                    if let Some((freq, value)) =
                        self.check_singleton(bucket, bucket_stages, stage_idx)
                    {
                        // Verify against other stages
                        if self.verify_singleton(freq, value, bucket_stages) {
                            singletons.push((freq, value));
                        }
                    }
                }
            }
        }

        // Deduplicate
        singletons.sort_by_key(|(f, _)| *f);
        singletons.dedup_by_key(|(f, _)| *f);

        singletons
    }

    /// Check if a bucket contains a singleton frequency.
    fn check_singleton(
        &self,
        bucket: &Bucket<T>,
        bucket_stages: &[BucketArray<T>],
        stage_idx: usize,
    ) -> Option<(usize, Complex<T>)> {
        // Skip if below threshold
        if bucket.value.norm_sqr() < self.threshold * self.threshold {
            return None;
        }

        // If we have explicit singleton info, use it
        if let Some(freq) = bucket.detected_freq {
            return Some((freq, bucket.value));
        }

        // Otherwise, try to identify using CRT across stages
        if bucket_stages.len() > 1 {
            let other_stage_idx = (stage_idx + 1) % bucket_stages.len();
            let other_stage = &bucket_stages[other_stage_idx];

            // Find corresponding bucket in other stage
            let bucket_idx = bucket.index % other_stage.len();
            if let Some(other_bucket) = other_stage.get(bucket_idx) {
                // Use phase information to estimate frequency
                return self.estimate_frequency_crt(bucket, other_bucket, bucket_stages, stage_idx);
            }
        }

        None
    }

    /// Estimate frequency using Chinese Remainder Theorem.
    fn estimate_frequency_crt(
        &self,
        bucket1: &Bucket<T>,
        bucket2: &Bucket<T>,
        bucket_stages: &[BucketArray<T>],
        stage_idx: usize,
    ) -> Option<(usize, Complex<T>)> {
        let val1 = bucket1.value;
        let val2 = bucket2.value;

        // Check if magnitudes are similar (indicating same frequency)
        let mag1 = val1.norm_sqr();
        let mag2 = val2.norm_sqr();

        if mag2 < self.threshold * self.threshold {
            return None;
        }

        let ratio = mag1 / mag2;
        if ratio < T::from_f64(0.25) || ratio > T::from_f64(4.0) {
            return None; // Magnitudes too different
        }

        // Get bucket sizes for CRT
        let b1 = bucket_stages[stage_idx].len();
        let other_idx = (stage_idx + 1) % bucket_stages.len();
        let b2 = bucket_stages[other_idx].len();

        // Frequency satisfies: freq ≡ bucket1.index (mod b1) and freq ≡ bucket2.index (mod b2)
        let idx1 = bucket1.index % b1;
        let idx2 = bucket2.index % b2;

        // Solve using CRT (simple search for small n)
        for candidate in 0..self.n {
            if candidate % b1 == idx1 && candidate % b2 == idx2 {
                return Some((candidate, val1));
            }
        }

        None
    }

    /// Verify singleton detection against other stages.
    fn verify_singleton(
        &self,
        freq: usize,
        expected_value: Complex<T>,
        bucket_stages: &[BucketArray<T>],
    ) -> bool {
        let expected_mag = expected_value.norm_sqr();

        for stage in bucket_stages {
            let bucket_idx = freq % stage.len();
            if let Some(bucket) = stage.get(bucket_idx) {
                let bucket_mag = bucket.value.norm_sqr();

                // Check if magnitude is consistent (allowing for some noise)
                let ratio = bucket_mag / (expected_mag + T::from_f64(1e-10));
                if ratio < T::from_f64(0.1) || ratio > T::from_f64(10.0) {
                    return false;
                }
            }
        }

        true
    }

    /// Check if a frequency has already been detected.
    fn is_already_detected(&self, freq: usize) -> bool {
        self.detected.iter().any(|(f, _)| *f == freq)
    }

    /// Remove frequency contribution from all bucket stages.
    fn peel_frequency(&self, bucket_stages: &mut [BucketArray<T>], freq: usize, value: Complex<T>) {
        for stage in bucket_stages.iter_mut() {
            let bucket_idx = freq % stage.len();
            if let Some(bucket) = stage.get_mut(bucket_idx) {
                // Subtract the frequency's contribution
                bucket.value = bucket.value - value;
                if bucket.count > 0 {
                    bucket.count -= 1;
                }
            }
        }
    }

    /// Try to resolve remaining multiton buckets.
    fn resolve_multitons(&mut self, bucket_stages: &[BucketArray<T>]) {
        // Simple heuristic: look at remaining significant buckets
        // and try to extract the largest remaining frequency

        for stage in bucket_stages {
            for bucket_idx in 0..stage.len() {
                if self.detected.len() >= self.k {
                    return;
                }

                if let Some(bucket) = stage.get(bucket_idx) {
                    if bucket.value.norm_sqr() > self.threshold * self.threshold {
                        // For multitons, estimate the dominant frequency
                        // This is a heuristic and may not be perfectly accurate
                        let freq = bucket_idx; // Simple estimate

                        if freq < self.n && !self.is_already_detected(freq) {
                            self.detected.push((freq, bucket.value));
                        }
                    }
                }
            }
        }
    }

    /// Get the number of detected frequencies.
    pub fn num_detected(&self) -> usize {
        self.detected.len()
    }

    /// Clear detected frequencies.
    pub fn reset(&mut self) {
        self.detected.clear();
    }
}

/// Simple singleton detection for a single bucket observation.
///
/// Uses magnitude-based detection with threshold.
pub fn detect_singleton<T: Float>(bucket_value: Complex<T>, threshold: T) -> Option<Complex<T>> {
    if bucket_value.norm_sqr() >= threshold * threshold {
        Some(bucket_value)
    } else {
        None
    }
}

/// Collision detection between two bucket observations.
///
/// Returns true if the buckets likely contain multiple aliased frequencies.
pub fn is_collision<T: Float>(val1: Complex<T>, val2: Complex<T>, threshold: T) -> bool {
    let mag1 = val1.norm_sqr();
    let mag2 = val2.norm_sqr();

    // If magnitudes are very different, likely collision
    if mag1 < threshold * threshold || mag2 < threshold * threshold {
        return false;
    }

    let ratio = mag1 / mag2;
    ratio < T::from_f64(0.3) || ratio > T::from_f64(3.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let decoder: PeelingDecoder<f64> = PeelingDecoder::new(1024, 10, 0.001);
        assert_eq!(decoder.n, 1024);
        assert_eq!(decoder.k, 10);
        assert_eq!(decoder.num_detected(), 0);
    }

    #[test]
    fn test_singleton_detection() {
        let val = Complex::new(1.0_f64, 0.5);
        let threshold = 0.1;

        assert!(detect_singleton(val, threshold).is_some());
        assert!(detect_singleton(Complex::new(0.01_f64, 0.01), threshold).is_none());
    }

    #[test]
    fn test_collision_detection() {
        let val1 = Complex::new(1.0_f64, 0.0);
        let val2 = Complex::new(0.9, 0.1);
        let threshold = 0.1;

        // Similar magnitudes - not collision
        assert!(!is_collision(val1, val2, threshold));

        // Very different magnitudes - collision
        let val3 = Complex::new(0.1_f64, 0.0);
        assert!(is_collision(val1, val3, threshold));
    }

    #[test]
    fn test_decoder_empty_input() {
        let mut decoder: PeelingDecoder<f64> = PeelingDecoder::new(64, 5, 0.001);
        let mut stages: Vec<BucketArray<f64>> = Vec::new();

        let result = decoder.decode(&mut stages);
        assert!(result.is_empty());
    }
}
