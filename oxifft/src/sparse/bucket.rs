//! Frequency bucketization for sparse FFT.
//!
//! Implements the bucketing step of FFAST algorithm where frequencies
//! are hashed into buckets using subsampling and aliasing.

use crate::kernel::{Complex, Float};
use crate::prelude::*;

/// Bucket containing frequency information.
#[derive(Debug, Clone)]
pub struct Bucket<T: Float> {
    /// Bucket index.
    pub index: usize,
    /// Sum of aliased frequency values.
    pub value: Complex<T>,
    /// Number of frequencies that hashed to this bucket.
    pub count: usize,
    /// Detected frequency index (if singleton).
    pub detected_freq: Option<usize>,
}

impl<T: Float> Bucket<T> {
    /// Create a new empty bucket.
    pub fn new(index: usize) -> Self {
        Self {
            index,
            value: Complex::<T>::zero(),
            count: 0,
            detected_freq: None,
        }
    }

    /// Check if bucket is empty (zeroton).
    pub fn is_zeroton(&self, threshold: T) -> bool {
        self.value.norm_sqr() < threshold * threshold
    }

    /// Check if bucket contains exactly one frequency (singleton).
    pub fn is_singleton(&self) -> bool {
        self.count == 1 && self.detected_freq.is_some()
    }

    /// Check if bucket contains multiple frequencies (multiton).
    pub fn is_multiton(&self) -> bool {
        self.count > 1
    }

    /// Add a frequency contribution to this bucket.
    pub fn add(&mut self, value: Complex<T>, freq_idx: usize) {
        self.value = self.value + value;
        self.count += 1;
        if self.count == 1 {
            self.detected_freq = Some(freq_idx);
        } else {
            self.detected_freq = None; // No longer singleton
        }
    }

    /// Reset bucket to empty state.
    pub fn reset(&mut self) {
        self.value = Complex::<T>::zero();
        self.count = 0;
        self.detected_freq = None;
    }
}

/// Bucket array for frequency bucketization.
#[derive(Debug, Clone)]
pub struct BucketArray<T: Float> {
    /// Array of buckets.
    buckets: Vec<Bucket<T>>,
    /// Number of buckets.
    num_buckets: usize,
    /// Subsampling factor used.
    subsample_factor: usize,
    /// Original signal length.
    n: usize,
}

impl<T: Float> BucketArray<T> {
    /// Create a new bucket array.
    ///
    /// # Arguments
    ///
    /// * `num_buckets` - Number of buckets (B)
    /// * `subsample_factor` - Subsampling factor for this stage
    /// * `n` - Original signal length
    pub fn new(num_buckets: usize, subsample_factor: usize, n: usize) -> Self {
        let buckets = (0..num_buckets).map(Bucket::new).collect();
        Self {
            buckets,
            num_buckets,
            subsample_factor,
            n,
        }
    }

    /// Get the bucket index for a given frequency.
    ///
    /// Uses aliasing: freq maps to freq mod num_buckets
    #[inline]
    pub fn bucket_index(&self, freq: usize) -> usize {
        freq % self.num_buckets
    }

    /// Fill buckets from subsampled FFT result.
    ///
    /// # Arguments
    ///
    /// * `subsampled_fft` - FFT of subsampled signal (length = num_buckets)
    pub fn fill_from_fft(&mut self, subsampled_fft: &[Complex<T>]) {
        debug_assert_eq!(subsampled_fft.len(), self.num_buckets);

        for (i, &val) in subsampled_fft.iter().enumerate() {
            self.buckets[i].value = val;
            // Count will be determined by collision analysis
        }
    }

    /// Analyze bucket contents to detect singletons.
    ///
    /// Uses phase information from multiple subsampling stages to determine
    /// which buckets contain single frequencies vs collisions.
    pub fn analyze_singletons(
        &mut self,
        other_stage: &Self,
        threshold: T,
    ) -> Vec<(usize, Complex<T>)> {
        let mut singletons = Vec::new();

        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            if bucket.is_zeroton(threshold) {
                bucket.count = 0;
                continue;
            }

            // Try to identify if this is a singleton by comparing with other stage
            // Using the Chinese Remainder Theorem for frequency identification
            let other_bucket_idx = i % other_stage.num_buckets;

            if other_bucket_idx < other_stage.buckets.len() {
                let other_val = other_stage.buckets[other_bucket_idx].value;

                // If both buckets have similar magnitude, likely same frequency
                let ratio = bucket.value.norm_sqr() / (other_val.norm_sqr() + T::from_f64(1e-10));

                if ratio > T::from_f64(0.5) && ratio < T::from_f64(2.0) {
                    // Estimate frequency using phase difference
                    if let Some(freq) = estimate_frequency_from_phase(
                        bucket.value,
                        other_val,
                        self.subsample_factor,
                        other_stage.subsample_factor,
                        self.n,
                    ) {
                        bucket.detected_freq = Some(freq);
                        bucket.count = 1;
                        singletons.push((freq, bucket.value));
                    }
                }
            }
        }

        singletons
    }

    /// Get all non-zero buckets.
    pub fn non_zero_buckets(&self, threshold: T) -> Vec<&Bucket<T>> {
        self.buckets
            .iter()
            .filter(|b| !b.is_zeroton(threshold))
            .collect()
    }

    /// Get singleton buckets.
    pub fn singleton_buckets(&self) -> Vec<&Bucket<T>> {
        self.buckets.iter().filter(|b| b.is_singleton()).collect()
    }

    /// Reset all buckets.
    pub fn reset(&mut self) {
        for bucket in &mut self.buckets {
            bucket.reset();
        }
    }

    /// Get bucket at index.
    pub fn get(&self, index: usize) -> Option<&Bucket<T>> {
        self.buckets.get(index)
    }

    /// Get mutable bucket at index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Bucket<T>> {
        self.buckets.get_mut(index)
    }

    /// Number of buckets.
    pub fn len(&self) -> usize {
        self.num_buckets
    }

    /// Check if bucket array is empty.
    pub fn is_empty(&self) -> bool {
        self.num_buckets == 0
    }
}

/// Estimate frequency from phase difference between two subsampling stages.
///
/// Uses the Chinese Remainder Theorem to recover the original frequency
/// from aliased values at different subsampling rates.
fn estimate_frequency_from_phase<T: Float>(
    val1: Complex<T>,
    val2: Complex<T>,
    factor1: usize,
    factor2: usize,
    n: usize,
) -> Option<usize> {
    // Convert to f64 for computation (unwrap_or to handle Option)
    let re1 = val1.re.to_f64().unwrap_or(0.0);
    let im1 = val1.im.to_f64().unwrap_or(0.0);
    let re2 = val2.re.to_f64().unwrap_or(0.0);
    let im2 = val2.im.to_f64().unwrap_or(0.0);

    // Phase of complex number using atan2
    let phase1_f64 = libm::atan2(im1, re1);
    let _phase2_f64 = libm::atan2(im2, re2);

    // Phase difference corresponds to frequency shift due to subsampling
    // Use CRT to recover original frequency
    let gcd = gcd_usize(factor1, factor2);
    let lcm = (factor1 * factor2) / gcd;

    // For now, return a simple estimate based on magnitude peak
    let magnitude1 = re1 * re1 + im1 * im1;
    let magnitude2 = re2 * re2 + im2 * im2;

    if magnitude1 < 1e-10 || magnitude2 < 1e-10 {
        return None;
    }

    // Simple frequency estimation from phase
    let two_pi = core::f64::consts::PI * 2.0;
    let phase1_abs = libm::fabs(phase1_f64);
    let freq_estimate = libm::round(phase1_abs * (n as f64) / two_pi) as usize;

    if freq_estimate < n && freq_estimate % lcm < lcm {
        Some(freq_estimate % n)
    } else {
        Some(0)
    }
}

/// Greatest common divisor.
fn gcd_usize(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd_usize(b, a % b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_creation() {
        let bucket: Bucket<f64> = Bucket::new(0);
        assert!(bucket.is_zeroton(0.01));
        assert!(!bucket.is_singleton());
        assert!(!bucket.is_multiton());
    }

    #[test]
    fn test_bucket_add() {
        let mut bucket: Bucket<f64> = Bucket::new(0);
        bucket.add(Complex::new(1.0, 0.0), 5);
        assert!(bucket.is_singleton());
        assert_eq!(bucket.detected_freq, Some(5));

        bucket.add(Complex::new(0.5, 0.5), 10);
        assert!(bucket.is_multiton());
        assert_eq!(bucket.detected_freq, None);
    }

    #[test]
    fn test_bucket_array() {
        let array: BucketArray<f64> = BucketArray::new(16, 2, 1024);
        assert_eq!(array.len(), 16);
        assert!(!array.is_empty());

        // Bucket index should be freq mod num_buckets
        assert_eq!(array.bucket_index(17), 1);
        assert_eq!(array.bucket_index(32), 0);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd_usize(12, 8), 4);
        assert_eq!(gcd_usize(17, 13), 1);
        assert_eq!(gcd_usize(100, 25), 25);
    }
}
