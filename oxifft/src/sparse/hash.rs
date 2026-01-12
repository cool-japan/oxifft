//! Frequency hashing utilities for sparse FFT.
//!
//! Implements hash functions used to map frequencies to buckets
//! in the FFAST algorithm.

// Allow dead code - infrastructure for future algorithm enhancements
#![allow(dead_code)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Hash function for frequency bucketization.
#[derive(Debug, Clone, Copy)]
pub struct FrequencyHash {
    /// Number of buckets.
    num_buckets: usize,
    /// Signal length.
    n: usize,
    /// Permutation offset (random shift).
    offset: usize,
    /// Permutation scale (multiplicative).
    scale: usize,
}

impl FrequencyHash {
    /// Create a new frequency hash function.
    ///
    /// # Arguments
    ///
    /// * `num_buckets` - Number of buckets (B)
    /// * `n` - Signal length
    pub fn new(num_buckets: usize, n: usize) -> Self {
        Self {
            num_buckets,
            n,
            offset: 0,
            scale: 1,
        }
    }

    /// Create with random permutation parameters.
    ///
    /// # Arguments
    ///
    /// * `num_buckets` - Number of buckets
    /// * `n` - Signal length
    /// * `seed` - Random seed for permutation
    pub fn with_permutation(num_buckets: usize, n: usize, seed: u64) -> Self {
        // Simple pseudo-random permutation using linear congruential generator
        let offset = ((seed * 1103515245 + 12345) % (n as u64)) as usize;
        let scale = Self::find_coprime(n, seed);

        Self {
            num_buckets,
            n,
            offset,
            scale,
        }
    }

    /// Find a coprime for multiplicative permutation.
    fn find_coprime(n: usize, seed: u64) -> usize {
        // Start from a seed-based value and find nearest coprime
        let start = ((seed * 48271) % (n as u64)) as usize;
        let start = start.max(1);

        for candidate in start..n {
            if gcd(candidate, n) == 1 {
                return candidate;
            }
        }

        // Fallback
        1
    }

    /// Hash a frequency index to a bucket.
    #[inline]
    pub fn hash(&self, freq: usize) -> usize {
        // Apply permutation: (scale * freq + offset) mod n, then mod num_buckets
        let permuted = (self.scale.wrapping_mul(freq).wrapping_add(self.offset)) % self.n;
        permuted % self.num_buckets
    }

    /// Inverse hash: find candidate frequencies that map to a bucket.
    ///
    /// Returns all frequencies in [0, n) that hash to the given bucket.
    pub fn inverse_hash(&self, bucket: usize) -> Vec<usize> {
        let mut candidates = Vec::new();

        for freq in 0..self.n {
            if self.hash(freq) == bucket {
                candidates.push(freq);
            }
        }

        candidates
    }

    /// Get collision probability estimate.
    ///
    /// Returns expected number of collisions for k sparse frequencies.
    pub fn collision_probability(&self, k: usize) -> f64 {
        // Birthday paradox approximation
        let b = self.num_buckets as f64;
        let k_f = k as f64;

        // Expected collisions ≈ k^2 / (2B)
        (k_f * k_f) / (2.0 * b)
    }

    /// Number of buckets.
    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    /// Signal length.
    pub fn signal_length(&self) -> usize {
        self.n
    }
}

/// Multi-hash using Chinese Remainder Theorem.
///
/// Uses multiple coprime bucket counts to uniquely identify frequencies.
/// Kept for future algorithm enhancements.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CrtHash {
    /// Individual hash functions with coprime bucket counts.
    hashes: Vec<FrequencyHash>,
    /// Product of all bucket counts.
    product: usize,
    /// Signal length.
    n: usize,
}

#[allow(dead_code)]
impl CrtHash {
    /// Create a CRT-based multi-hash.
    ///
    /// # Arguments
    ///
    /// * `bucket_counts` - Array of coprime bucket counts
    /// * `n` - Signal length
    pub fn new(bucket_counts: &[usize], n: usize) -> Self {
        let hashes: Vec<_> = bucket_counts
            .iter()
            .map(|&b| FrequencyHash::new(b, n))
            .collect();

        let product: usize = bucket_counts.iter().product();

        Self { hashes, product, n }
    }

    /// Create from coprime factors.
    ///
    /// # Arguments
    ///
    /// * `factors` - Coprime factors for bucket counts
    /// * `n` - Signal length
    pub fn from_factors(factors: &[usize], n: usize) -> Self {
        Self::new(factors, n)
    }

    /// Hash a frequency to a tuple of bucket indices.
    pub fn hash(&self, freq: usize) -> Vec<usize> {
        self.hashes.iter().map(|h| h.hash(freq)).collect()
    }

    /// Recover frequency from bucket indices using CRT.
    ///
    /// # Arguments
    ///
    /// * `bucket_indices` - Tuple of bucket indices from each hash
    ///
    /// # Returns
    ///
    /// Recovered frequency if valid, None otherwise.
    pub fn recover_frequency(&self, bucket_indices: &[usize]) -> Option<usize> {
        if bucket_indices.len() != self.hashes.len() {
            return None;
        }

        // Apply CRT to solve:
        // x ≡ bucket_indices[i] (mod bucket_counts[i]) for all i

        let mut result = 0usize;

        for (i, &idx) in bucket_indices.iter().enumerate() {
            let ni = self.hashes[i].num_buckets();
            let mi = self.product / ni;

            // Find multiplicative inverse of mi mod ni
            if let Some(yi) = mod_inverse(mi % ni, ni) {
                result = result.wrapping_add(idx.wrapping_mul(mi).wrapping_mul(yi));
            } else {
                return None;
            }
        }

        let freq = result % self.product;
        if freq < self.n {
            Some(freq)
        } else {
            None
        }
    }

    /// Number of hash functions.
    pub fn num_hashes(&self) -> usize {
        self.hashes.len()
    }

    /// Get individual hash functions.
    pub fn hashes(&self) -> &[FrequencyHash] {
        &self.hashes
    }
}

/// Calculate GCD using Euclidean algorithm.
#[allow(dead_code)]
fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Calculate modular multiplicative inverse using extended Euclidean algorithm.
#[allow(dead_code)]
fn mod_inverse(a: usize, m: usize) -> Option<usize> {
    if m == 0 {
        return None;
    }

    let (gcd, x, _) = extended_gcd(a as i64, m as i64);

    if gcd != 1 {
        return None; // No inverse exists
    }

    // Make sure result is positive
    let inv = ((x % m as i64) + m as i64) % m as i64;
    Some(inv as usize)
}

/// Extended Euclidean algorithm.
///
/// Returns (gcd, x, y) such that a*x + b*y = gcd.
#[allow(dead_code)]
fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if a == 0 {
        (b, 0, 1)
    } else {
        let (gcd, x, y) = extended_gcd(b % a, a);
        (gcd, y - (b / a) * x, x)
    }
}

/// Generate coprime factors for CRT.
///
/// Returns a set of small coprime numbers whose product is at least target.
pub fn generate_coprime_factors(target: usize, max_factor: usize) -> Vec<usize> {
    let small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];

    let mut factors = Vec::new();
    let mut product = 1usize;

    for &p in &small_primes {
        if product >= target {
            break;
        }
        if p <= max_factor {
            factors.push(p);
            product *= p;
        }
    }

    factors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_hash() {
        let hash = FrequencyHash::new(16, 1024);

        // Simple modulo hash
        assert_eq!(hash.hash(0), 0);
        assert_eq!(hash.hash(16), 0);
        assert_eq!(hash.hash(17), 1);
    }

    #[test]
    fn test_hash_with_permutation() {
        let hash = FrequencyHash::with_permutation(16, 1024, 12345);

        // Should produce valid bucket indices
        for freq in 0..100 {
            let bucket = hash.hash(freq);
            assert!(bucket < 16);
        }
    }

    #[test]
    fn test_inverse_hash() {
        let hash = FrequencyHash::new(8, 64);

        let candidates = hash.inverse_hash(0);
        // Should include 0, 8, 16, 24, 32, 40, 48, 56
        assert!(candidates.contains(&0));
        assert!(candidates.contains(&8));
        assert!(candidates.contains(&16));
    }

    #[test]
    fn test_collision_probability() {
        let hash = FrequencyHash::new(100, 1000);

        let prob = hash.collision_probability(10);
        // k^2/(2B) = 100/200 = 0.5
        assert!((prob - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_crt_hash() {
        let crt = CrtHash::new(&[3, 5, 7], 1024);

        // Test round-trip: hash and recover
        for freq in 0..100 {
            let buckets = crt.hash(freq);
            if let Some(recovered) = crt.recover_frequency(&buckets) {
                // Recovered frequency should match original mod product
                assert_eq!(recovered % 105, freq % 105);
            }
        }
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(17, 13), 1);
        assert_eq!(gcd(100, 35), 5);
    }

    #[test]
    fn test_mod_inverse() {
        // 3 * 7 = 21 ≡ 1 (mod 10)
        assert_eq!(mod_inverse(3, 10), Some(7));

        // No inverse when gcd != 1
        assert_eq!(mod_inverse(4, 8), None);
    }

    #[test]
    fn test_generate_coprime_factors() {
        let factors = generate_coprime_factors(100, 50);
        assert!(factors.len() >= 2);

        // Product should be at least 100
        let product: usize = factors.iter().product();
        assert!(product >= 100);

        // All factors should be coprime
        for i in 0..factors.len() {
            for j in i + 1..factors.len() {
                assert_eq!(gcd(factors[i], factors[j]), 1);
            }
        }
    }
}
