//! Twiddle factor computation and caching.
//!
//! Twiddle factors are the complex exponentials W_n^k = e^(-2πik/n)
//! used in FFT butterfly operations.

use super::{Complex, Float};
use crate::prelude::*;

/// Cache for twiddle factors.
///
/// Stores precomputed twiddle factors keyed by (n, k).
pub struct TwiddleCache<T: Float> {
    cache: HashMap<(usize, usize), Vec<Complex<T>>>,
}

impl<T: Float> Default for TwiddleCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> TwiddleCache<T> {
    /// Create a new empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Get or compute twiddle factors for size n and radix k.
    ///
    /// Returns W_n^0, W_n^1, ..., W_n^(k-1) for the given parameters.
    pub fn get(&mut self, n: usize, k: usize) -> &[Complex<T>] {
        self.cache
            .entry((n, k))
            .or_insert_with(|| compute_twiddles(n, k))
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

/// Compute twiddle factors W_n^j for j = 0..k.
///
/// W_n^j = e^(-2πij/n) = cos(2πj/n) - i*sin(2πj/n)
#[must_use]
pub fn compute_twiddles<T: Float>(n: usize, k: usize) -> Vec<Complex<T>> {
    let mut result = Vec::with_capacity(k);
    let theta_base = -T::TWO_PI / T::from_usize(n);

    for j in 0..k {
        let theta = theta_base * T::from_usize(j);
        result.push(Complex::cis(theta));
    }

    result
}

/// Compute a single twiddle factor W_n^k.
#[allow(dead_code)]
#[inline]
#[must_use]
pub fn twiddle<T: Float>(n: usize, k: usize) -> Complex<T> {
    let theta = -T::TWO_PI * T::from_usize(k) / T::from_usize(n);
    Complex::cis(theta)
}

/// Compute twiddle factor for inverse transform: W_n^(-k).
#[allow(dead_code)]
#[inline]
#[must_use]
pub fn twiddle_inverse<T: Float>(n: usize, k: usize) -> Complex<T> {
    let theta = T::TWO_PI * T::from_usize(k) / T::from_usize(n);
    Complex::cis(theta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twiddle_w4() {
        // W_4^0 = 1
        let w0: Complex<f64> = twiddle(4, 0);
        assert!((w0.re - 1.0).abs() < 1e-10);
        assert!(w0.im.abs() < 1e-10);

        // W_4^1 = -i
        let w1: Complex<f64> = twiddle(4, 1);
        assert!(w1.re.abs() < 1e-10);
        assert!((w1.im - (-1.0)).abs() < 1e-10);

        // W_4^2 = -1
        let w2: Complex<f64> = twiddle(4, 2);
        assert!((w2.re - (-1.0)).abs() < 1e-10);
        assert!(w2.im.abs() < 1e-10);
    }

    #[test]
    fn test_compute_twiddles() {
        let tw: Vec<Complex<f64>> = compute_twiddles(8, 4);
        assert_eq!(tw.len(), 4);

        // W_8^0 = 1
        assert!((tw[0].re - 1.0).abs() < 1e-10);
        assert!(tw[0].im.abs() < 1e-10);
    }
}
