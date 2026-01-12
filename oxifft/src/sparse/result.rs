//! Sparse FFT result type.

use crate::kernel::{Complex, Float};

/// Result of sparse FFT computation.
///
/// Contains the indices and values of detected non-zero frequency components.
#[derive(Debug, Clone)]
pub struct SparseResult<T: Float> {
    /// Frequency indices with non-zero coefficients.
    pub indices: Vec<usize>,
    /// Complex values at the detected frequencies.
    pub values: Vec<Complex<T>>,
    /// Original signal length.
    pub n: usize,
}

impl<T: Float> SparseResult<T> {
    /// Create a new sparse result.
    pub fn new(indices: Vec<usize>, values: Vec<Complex<T>>, n: usize) -> Self {
        debug_assert_eq!(indices.len(), values.len());
        Self { indices, values, n }
    }

    /// Create an empty sparse result.
    pub fn empty() -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            n: 0,
        }
    }

    /// Check if the result is empty.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Get the number of detected frequencies.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Get the sparsity ratio (k/n).
    pub fn sparsity_ratio(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.indices.len() as f64 / self.n as f64
        }
    }

    /// Iterator over (index, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &Complex<T>)> {
        self.indices.iter().copied().zip(self.values.iter())
    }

    /// Get value at a specific frequency index (returns zero if not present).
    pub fn get(&self, freq_idx: usize) -> Complex<T> {
        for (i, &idx) in self.indices.iter().enumerate() {
            if idx == freq_idx {
                return self.values[i];
            }
        }
        Complex::<T>::zero()
    }

    /// Convert to dense representation.
    pub fn to_dense(&self) -> Vec<Complex<T>> {
        let mut dense = vec![Complex::<T>::zero(); self.n];
        for (i, &idx) in self.indices.iter().enumerate() {
            if idx < self.n {
                dense[idx] = self.values[i];
            }
        }
        dense
    }

    /// Get indices sorted by magnitude (descending).
    pub fn sorted_by_magnitude(&self) -> Vec<(usize, Complex<T>)> {
        let mut pairs: Vec<_> = self
            .indices
            .iter()
            .copied()
            .zip(self.values.iter().copied())
            .collect();
        pairs.sort_by(|a, b| {
            b.1.norm_sqr()
                .partial_cmp(&a.1.norm_sqr())
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        pairs
    }

    /// Filter frequencies by minimum magnitude threshold.
    pub fn filter_by_magnitude(&self, threshold: T) -> Self {
        let threshold_sqr = threshold * threshold;
        let filtered: Vec<_> = self
            .indices
            .iter()
            .copied()
            .zip(self.values.iter().copied())
            .filter(|(_, v)| v.norm_sqr() >= threshold_sqr)
            .collect();

        Self {
            indices: filtered.iter().map(|(i, _)| *i).collect(),
            values: filtered.iter().map(|(_, v)| *v).collect(),
            n: self.n,
        }
    }

    /// Merge two sparse results, keeping the larger magnitude for duplicates.
    pub fn merge(&self, other: &Self) -> Self {
        use alloc::collections::BTreeMap;

        let mut map: BTreeMap<usize, Complex<T>> = BTreeMap::new();

        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            map.insert(idx, val);
        }

        for (&idx, &val) in other.indices.iter().zip(other.values.iter()) {
            map.entry(idx)
                .and_modify(|existing: &mut Complex<T>| {
                    if val.norm_sqr() > existing.norm_sqr() {
                        *existing = val;
                    }
                })
                .or_insert(val);
        }

        let indices: Vec<usize> = map.keys().copied().collect();
        let values: Vec<Complex<T>> = map.values().copied().collect();

        Self {
            indices,
            values,
            n: self.n.max(other.n),
        }
    }
}

impl<T: Float> Default for SparseResult<T> {
    fn default() -> Self {
        Self::empty()
    }
}

extern crate alloc;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_result_empty() {
        let result: SparseResult<f64> = SparseResult::empty();
        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_sparse_result_to_dense() {
        let indices = vec![1, 5, 10];
        let values = vec![
            Complex::new(1.0_f64, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        ];
        let result = SparseResult::new(indices, values, 16);

        let dense = result.to_dense();
        assert_eq!(dense.len(), 16);
        assert_eq!(dense[1].re, 1.0);
        assert_eq!(dense[5].re, 2.0);
        assert_eq!(dense[10].re, 3.0);
        assert_eq!(dense[0].re, 0.0);
    }

    #[test]
    fn test_sparse_result_get() {
        let indices = vec![1, 5];
        let values = vec![Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        let result = SparseResult::new(indices, values, 16);

        assert_eq!(result.get(1).re, 1.0);
        assert_eq!(result.get(5).re, 3.0);
        assert_eq!(result.get(0).re, 0.0); // Not present
    }

    #[test]
    fn test_sparse_result_filter() {
        let indices = vec![1, 2, 3];
        let values = vec![
            Complex::new(0.1_f64, 0.0), // magnitude 0.1
            Complex::new(1.0, 0.0),     // magnitude 1.0
            Complex::new(10.0, 0.0),    // magnitude 10.0
        ];
        let result = SparseResult::new(indices, values, 16);

        let filtered = result.filter_by_magnitude(0.5);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.indices.contains(&2));
        assert!(filtered.indices.contains(&3));
    }
}
