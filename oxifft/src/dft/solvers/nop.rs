//! No-op solver for size-1 transforms.
//!
//! A size-1 DFT is trivial: the output equals the input.
//! This solver handles this base case efficiently.

use crate::kernel::{Complex, Float};

/// No-op solver for trivial (size-1) transforms.
///
/// For a size-1 DFT, X\[0\] = x\[0\] (just copy the single element).
/// This is the base case for recursive FFT algorithms.
pub struct NopSolver<T: Float> {
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> Default for NopSolver<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> NopSolver<T> {
    /// Create a new no-op solver.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }

    /// Solver name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        "dft-nop"
    }

    /// Check if this solver can handle the given size.
    #[must_use]
    pub fn applicable(n: usize) -> bool {
        n <= 1
    }

    /// Execute the no-op transform.
    ///
    /// For size 0: does nothing
    /// For size 1: copies input\[0\] to output\[0\]
    #[inline]
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        match input.len() {
            0 => {}
            1 => output[0] = input[0],
            _ => panic!("NopSolver only handles size 0 or 1"),
        }
    }

    /// Execute in-place (no-op for size 0 or 1).
    #[inline]
    pub fn execute_inplace(&self, _data: &mut [Complex<T>]) {
        // Nothing to do - data is already in place
    }

    /// Execute on raw pointers.
    ///
    /// # Safety
    /// - For n=1: both pointers must be valid
    /// - Pointers may be equal (in-place)
    #[inline]
    pub unsafe fn execute_ptr(&self, input: *const Complex<T>, output: *mut Complex<T>, n: usize) {
        if n == 1 && input != output as *const _ {
            unsafe { *output = *input };
        }
    }
}

/// Convenience function for size-1 DFT.
#[inline]
pub fn dft_nop<T: Float>(input: &[Complex<T>], output: &mut [Complex<T>]) {
    NopSolver::new().execute(input, output);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nop_size_0() {
        let input: [Complex<f64>; 0] = [];
        let mut output: [Complex<f64>; 0] = [];

        NopSolver::new().execute(&input, &mut output);
        // No assertion needed - just shouldn't panic
    }

    #[test]
    fn test_nop_size_1() {
        let input = [Complex::new(3.0_f64, 4.0)];
        let mut output = [Complex::zero()];

        NopSolver::new().execute(&input, &mut output);

        assert!((output[0].re - 3.0).abs() < 1e-10);
        assert!((output[0].im - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_applicable() {
        assert!(NopSolver::<f64>::applicable(0));
        assert!(NopSolver::<f64>::applicable(1));
        assert!(!NopSolver::<f64>::applicable(2));
        assert!(!NopSolver::<f64>::applicable(100));
    }
}
