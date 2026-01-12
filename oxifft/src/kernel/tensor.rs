//! N-dimensional tensor representation with strides.
//!
//! This module provides the core data structures for representing
//! multi-dimensional FFT problems with arbitrary memory layouts.

use crate::prelude::*;

/// Dimension specification with separate input and output strides.
///
/// This allows representing transforms where input and output have
/// different memory layouts (e.g., different strides, in-place vs out-of-place).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IoDim {
    /// Size of this dimension.
    pub n: usize,
    /// Input stride (can be negative for reversed access).
    pub is: isize,
    /// Output stride (can be negative for reversed access).
    pub os: isize,
}

impl IoDim {
    /// Create a new dimension specification.
    #[inline]
    #[must_use]
    pub const fn new(n: usize, is: isize, os: isize) -> Self {
        Self { n, is, os }
    }

    /// Create a contiguous dimension (stride = 1 for both input and output).
    #[inline]
    #[must_use]
    pub const fn contiguous(n: usize) -> Self {
        Self::new(n, 1, 1)
    }

    /// Check if this dimension is contiguous.
    #[inline]
    #[must_use]
    pub const fn is_contiguous(&self) -> bool {
        self.is == 1 && self.os == 1
    }

    /// Check if input and output have the same stride.
    #[inline]
    #[must_use]
    pub const fn is_inplace_compatible(&self) -> bool {
        self.is == self.os
    }
}

/// N-dimensional tensor representation.
///
/// A tensor describes the shape and memory layout of FFT data.
/// It consists of a vector of dimensions, each with its own size and strides.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Tensor {
    /// The dimensions of the tensor.
    pub dims: Vec<IoDim>,
}

impl Tensor {
    /// Create a new tensor with the given dimensions.
    #[inline]
    #[must_use]
    pub fn new(dims: Vec<IoDim>) -> Self {
        Self { dims }
    }

    /// Create an empty (rank-0) tensor.
    #[inline]
    #[must_use]
    pub fn empty() -> Self {
        Self { dims: Vec::new() }
    }

    /// Create a 1D contiguous tensor.
    #[inline]
    #[must_use]
    pub fn rank1(n: usize) -> Self {
        Self::new(vec![IoDim::contiguous(n)])
    }

    /// Create a 2D row-major contiguous tensor.
    #[inline]
    #[must_use]
    pub fn rank2(n0: usize, n1: usize) -> Self {
        Self::new(vec![
            IoDim::new(n0, n1 as isize, n1 as isize),
            IoDim::contiguous(n1),
        ])
    }

    /// Create a 3D row-major contiguous tensor.
    #[inline]
    #[must_use]
    pub fn rank3(n0: usize, n1: usize, n2: usize) -> Self {
        let stride0 = (n1 * n2) as isize;
        let stride1 = n2 as isize;
        Self::new(vec![
            IoDim::new(n0, stride0, stride0),
            IoDim::new(n1, stride1, stride1),
            IoDim::contiguous(n2),
        ])
    }

    /// Get the rank (number of dimensions).
    #[inline]
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Check if the tensor is empty (rank 0).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }

    /// Get the total number of elements.
    #[inline]
    #[must_use]
    pub fn total_size(&self) -> usize {
        self.dims.iter().map(|d| d.n).product()
    }

    /// Check if all dimensions are contiguous.
    #[inline]
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        if self.dims.is_empty() {
            return true;
        }

        // Check if strides match row-major layout
        let mut expected_stride = 1isize;
        for dim in self.dims.iter().rev() {
            if dim.is != expected_stride || dim.os != expected_stride {
                return false;
            }
            expected_stride *= dim.n as isize;
        }
        true
    }

    /// Check if input and output have identical strides (in-place compatible).
    #[inline]
    #[must_use]
    pub fn is_inplace_compatible(&self) -> bool {
        self.dims.iter().all(|d| d.is_inplace_compatible())
    }

    /// Split the tensor at the given axis.
    ///
    /// Returns (outer dimensions, inner dimensions).
    #[must_use]
    pub fn split(&self, axis: usize) -> (Self, Self) {
        let outer = Self::new(self.dims[..axis].to_vec());
        let inner = Self::new(self.dims[axis..].to_vec());
        (outer, inner)
    }

    /// Get the first dimension, if any.
    #[inline]
    #[must_use]
    pub fn first(&self) -> Option<&IoDim> {
        self.dims.first()
    }

    /// Get the last dimension, if any.
    #[inline]
    #[must_use]
    pub fn last(&self) -> Option<&IoDim> {
        self.dims.last()
    }

    /// Remove and return the first dimension.
    #[must_use]
    pub fn pop_front(&self) -> Option<(IoDim, Self)> {
        if self.dims.is_empty() {
            None
        } else {
            let first = self.dims[0].clone();
            let rest = Self::new(self.dims[1..].to_vec());
            Some((first, rest))
        }
    }

    /// Append a dimension.
    pub fn push(&mut self, dim: IoDim) {
        self.dims.push(dim);
    }

    /// Prepend a dimension.
    pub fn push_front(&mut self, dim: IoDim) {
        self.dims.insert(0, dim);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_rank1() {
        let t = Tensor::rank1(256);
        assert_eq!(t.rank(), 1);
        assert_eq!(t.total_size(), 256);
        assert!(t.is_contiguous());
    }

    #[test]
    fn test_tensor_rank2() {
        let t = Tensor::rank2(64, 64);
        assert_eq!(t.rank(), 2);
        assert_eq!(t.total_size(), 4096);
        assert!(t.is_contiguous());
    }

    #[test]
    fn test_tensor_split() {
        let t = Tensor::rank3(4, 8, 16);
        let (outer, inner) = t.split(1);
        assert_eq!(outer.rank(), 1);
        assert_eq!(inner.rank(), 2);
    }
}
