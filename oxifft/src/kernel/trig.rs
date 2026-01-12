//! Trigonometric table generation.
//!
//! Provides precomputed sine/cosine tables for efficient FFT computation.

use super::Float;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Precomputed trigonometric table.
pub struct TrigTable<T: Float> {
    /// Sine values.
    pub sin: Vec<T>,
    /// Cosine values.
    pub cos: Vec<T>,
    /// Table size.
    pub size: usize,
}

impl<T: Float> TrigTable<T> {
    /// Create a new trigonometric table for n points.
    ///
    /// Precomputes sin(2πk/n) and cos(2πk/n) for k = 0..n.
    #[must_use]
    pub fn new(n: usize) -> Self {
        let mut sin = Vec::with_capacity(n);
        let mut cos = Vec::with_capacity(n);

        let theta_step = T::TWO_PI / T::from_usize(n);

        for k in 0..n {
            let theta = theta_step * T::from_usize(k);
            let (s, c) = Float::sin_cos(theta);
            sin.push(s);
            cos.push(c);
        }

        Self { sin, cos, size: n }
    }

    /// Get sin(2πk/n) with wrapping.
    #[inline]
    #[must_use]
    pub fn sin(&self, k: usize) -> T {
        self.sin[k % self.size]
    }

    /// Get cos(2πk/n) with wrapping.
    #[inline]
    #[must_use]
    pub fn cos(&self, k: usize) -> T {
        self.cos[k % self.size]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trig_table() {
        let table: TrigTable<f64> = TrigTable::new(4);

        // sin(0) = 0, cos(0) = 1
        assert!(table.sin(0).abs() < 1e-10);
        assert!((table.cos(0) - 1.0).abs() < 1e-10);

        // sin(π/2) = 1, cos(π/2) = 0
        assert!((table.sin(1) - 1.0).abs() < 1e-10);
        assert!(table.cos(1).abs() < 1e-10);
    }
}
