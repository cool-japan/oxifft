//! Batch (vector rank >= 1) DFT solver.
//!
//! This solver handles batched transforms where the same 1D FFT is applied
//! to multiple arrays (e.g., rows of a matrix, or multiple independent signals).

use crate::dft::problem::Sign;
use crate::kernel::{Complex, Float, IoDim};
use crate::prelude::*;

/// Solver for batched transforms.
///
/// Executes multiple 1D FFTs with configurable strides.
/// This is useful for:
/// - Processing multiple independent signals in a single buffer
/// - Applying FFT to rows/columns of a 2D array with custom layout
/// - Batch transforms with non-contiguous memory access patterns
pub struct VrankGeq1Solver<T: Float> {
    /// Transform size (length of each 1D FFT)
    n: usize,
    /// Number of batches
    howmany: usize,
    /// Input stride between consecutive elements within a single FFT
    istride: isize,
    /// Output stride between consecutive elements within a single FFT
    ostride: isize,
    /// Input distance between starts of consecutive batches
    idist: isize,
    /// Output distance between starts of consecutive batches
    odist: isize,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> Default for VrankGeq1Solver<T> {
    fn default() -> Self {
        Self::new_contiguous(1, 1)
    }
}

impl<T: Float> VrankGeq1Solver<T> {
    /// Create a new vector rank >= 1 solver with full stride control.
    ///
    /// # Arguments
    /// * `n` - Transform size (length of each 1D FFT)
    /// * `howmany` - Number of batches to process
    /// * `istride` - Input stride between consecutive elements
    /// * `ostride` - Output stride between consecutive elements
    /// * `idist` - Input distance between batch starts
    /// * `odist` - Output distance between batch starts
    #[must_use]
    pub fn new(
        n: usize,
        howmany: usize,
        istride: isize,
        ostride: isize,
        idist: isize,
        odist: isize,
    ) -> Self {
        Self {
            n,
            howmany,
            istride,
            ostride,
            idist,
            odist,
            _marker: core::marker::PhantomData,
        }
    }

    /// Create a solver for contiguous batched data.
    ///
    /// Each batch is contiguous in memory (stride=1), and batches are
    /// stored consecutively (dist=n).
    #[must_use]
    pub fn new_contiguous(n: usize, howmany: usize) -> Self {
        Self::new(n, howmany, 1, 1, n as isize, n as isize)
    }

    /// Create a solver from IoDim specifications.
    ///
    /// # Arguments
    /// * `transform_dim` - Dimension of the transform (n, istride, ostride)
    /// * `batch_dim` - Batch dimension (howmany, idist, odist)
    #[must_use]
    pub fn from_dims(transform_dim: &IoDim, batch_dim: &IoDim) -> Self {
        Self::new(
            transform_dim.n,
            batch_dim.n,
            transform_dim.is,
            transform_dim.os,
            batch_dim.is,
            batch_dim.os,
        )
    }

    /// Solver name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        "dft-vrank-geq1"
    }

    /// Get the transform size.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the number of batches.
    #[must_use]
    pub fn howmany(&self) -> usize {
        self.howmany
    }

    /// Execute batched FFT using the provided 1D solver function.
    ///
    /// This is the core batch execution method that delegates to any 1D FFT
    /// implementation provided as a closure.
    ///
    /// # Arguments
    /// * `input` - Input buffer (must have at least n * howmany accessible elements)
    /// * `output` - Output buffer
    /// * `fft_1d` - Closure that executes a single 1D FFT
    pub fn execute_with<F>(&self, input: &[Complex<T>], output: &mut [Complex<T>], fft_1d: F)
    where
        F: Fn(&[Complex<T>], &mut [Complex<T>]),
    {
        if self.n == 0 || self.howmany == 0 {
            return;
        }

        // Optimized path for contiguous data
        if self.istride == 1
            && self.ostride == 1
            && self.idist == self.n as isize
            && self.odist == self.n as isize
        {
            for batch in 0..self.howmany {
                let start = batch * self.n;
                let end = start + self.n;
                fft_1d(&input[start..end], &mut output[start..end]);
            }
            return;
        }

        // General strided path
        let mut in_buf = vec![Complex::zero(); self.n];
        let mut out_buf = vec![Complex::zero(); self.n];

        for batch in 0..self.howmany {
            let in_base = (batch as isize * self.idist) as usize;
            let out_base = (batch as isize * self.odist) as usize;

            // Gather input with stride
            for i in 0..self.n {
                let idx = in_base as isize + i as isize * self.istride;
                in_buf[i] = input[idx as usize];
            }

            // Execute 1D FFT
            fft_1d(&in_buf, &mut out_buf);

            // Scatter output with stride
            for i in 0..self.n {
                let idx = out_base as isize + i as isize * self.ostride;
                output[idx as usize] = out_buf[i];
            }
        }
    }

    /// Execute batched FFT in-place using the provided 1D solver function.
    pub fn execute_inplace_with<F>(&self, data: &mut [Complex<T>], fft_1d: F)
    where
        F: Fn(&mut [Complex<T>]),
    {
        if self.n == 0 || self.howmany == 0 {
            return;
        }

        // Optimized path for contiguous data with same input/output strides
        if self.istride == 1
            && self.ostride == 1
            && self.idist == self.odist
            && self.idist == self.n as isize
        {
            for batch in 0..self.howmany {
                let start = batch * self.n;
                let end = start + self.n;
                fft_1d(&mut data[start..end]);
            }
            return;
        }

        // General strided path for in-place
        // When strides differ or aren't contiguous, we need a temp buffer
        let mut buf = vec![Complex::zero(); self.n];

        for batch in 0..self.howmany {
            let base = (batch as isize * self.idist) as usize;

            // Gather with input stride
            for i in 0..self.n {
                let idx = base as isize + i as isize * self.istride;
                buf[i] = data[idx as usize];
            }

            // Execute in-place 1D FFT on buffer
            fft_1d(&mut buf);

            // Scatter with output stride
            let out_base = (batch as isize * self.odist) as usize;
            for i in 0..self.n {
                let idx = out_base as isize + i as isize * self.ostride;
                data[idx as usize] = buf[i];
            }
        }
    }

    /// Execute batched FFT using Cooley-Tukey algorithm (for power-of-2 sizes).
    ///
    /// This is a convenience method that uses the optimal solver for power-of-2 sizes.
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        use crate::dft::solvers::{
            BluesteinSolver, CooleyTukeySolver, CtVariant, DirectSolver, GenericSolver, NopSolver,
        };

        if self.n <= 1 {
            self.execute_with(input, output, |i, o| NopSolver::new().execute(i, o));
        } else if CooleyTukeySolver::<T>::applicable(self.n) {
            let solver = CooleyTukeySolver::new(CtVariant::Dit);
            self.execute_with(input, output, |i, o| solver.execute(i, o, sign));
        } else if self.n <= 16 {
            let solver = DirectSolver::new();
            self.execute_with(input, output, |i, o| solver.execute(i, o, sign));
        } else if GenericSolver::<T>::applicable(self.n) {
            let solver = GenericSolver::new(self.n);
            self.execute_with(input, output, |i, o| solver.execute(i, o, sign));
        } else {
            let solver = BluesteinSolver::new(self.n);
            self.execute_with(input, output, |i, o| solver.execute(i, o, sign));
        }
    }

    /// Execute batched FFT in-place.
    pub fn execute_inplace(&self, data: &mut [Complex<T>], sign: Sign) {
        use crate::dft::solvers::{
            BluesteinSolver, CooleyTukeySolver, CtVariant, DirectSolver, GenericSolver, NopSolver,
        };

        if self.n <= 1 {
            self.execute_inplace_with(data, |d| NopSolver::new().execute_inplace(d));
        } else if CooleyTukeySolver::<T>::applicable(self.n) {
            let solver = CooleyTukeySolver::new(CtVariant::Dit);
            self.execute_inplace_with(data, |d| solver.execute_inplace(d, sign));
        } else if self.n <= 16 {
            let solver = DirectSolver::new();
            self.execute_inplace_with(data, |d| solver.execute_inplace(d, sign));
        } else if GenericSolver::<T>::applicable(self.n) {
            let solver = GenericSolver::new(self.n);
            self.execute_inplace_with(data, |d| solver.execute_inplace(d, sign));
        } else {
            let solver = BluesteinSolver::new(self.n);
            self.execute_inplace_with(data, |d| solver.execute_inplace(d, sign));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
        approx_eq(a.re, b.re, eps) && approx_eq(a.im, b.im, eps)
    }

    #[test]
    fn test_batch_contiguous_power_of_2() {
        let n = 4;
        let howmany = 3;
        let solver = VrankGeq1Solver::<f64>::new_contiguous(n, howmany);

        // Create input: 3 batches of 4 elements each
        let input: Vec<Complex<f64>> = (0..(n * howmany))
            .map(|i| Complex::new(i as f64, 0.0))
            .collect();
        let mut output = vec![Complex::zero(); n * howmany];

        solver.execute(&input, &mut output, Sign::Forward);

        // Each batch should have DC = sum of its elements
        // Batch 0: 0,1,2,3 -> DC = 6
        // Batch 1: 4,5,6,7 -> DC = 22
        // Batch 2: 8,9,10,11 -> DC = 38
        assert!(complex_approx_eq(output[0], Complex::new(6.0, 0.0), 1e-10));
        assert!(complex_approx_eq(output[4], Complex::new(22.0, 0.0), 1e-10));
        assert!(complex_approx_eq(output[8], Complex::new(38.0, 0.0), 1e-10));
    }

    #[test]
    fn test_batch_roundtrip() {
        let n = 8;
        let howmany = 4;
        let solver = VrankGeq1Solver::<f64>::new_contiguous(n, howmany);

        let original: Vec<Complex<f64>> = (0..(n * howmany))
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut transformed = vec![Complex::zero(); n * howmany];
        let mut recovered = vec![Complex::zero(); n * howmany];

        // Forward
        solver.execute(&original, &mut transformed, Sign::Forward);

        // Backward
        solver.execute(&transformed, &mut recovered, Sign::Backward);

        // Normalize and compare
        let scale = n as f64;
        for (a, b) in original.iter().zip(recovered.iter()) {
            let normalized = Complex::new(b.re / scale, b.im / scale);
            assert!(complex_approx_eq(*a, normalized, 1e-10));
        }
    }

    #[test]
    fn test_batch_strided_column_access() {
        // Simulate column FFT on a 4x4 matrix
        // Matrix stored row-major, we FFT each column
        let rows = 4;
        let cols = 4;

        // istride = cols (to access elements in same column)
        // idist = 1 (columns are adjacent at start)
        let solver = VrankGeq1Solver::<f64>::new(
            rows,          // n = number of rows
            cols,          // howmany = number of columns
            cols as isize, // istride = cols (skip to next row)
            cols as isize, // ostride = cols
            1,             // idist = 1 (next column)
            1,             // odist = 1
        );

        // Row-major matrix:
        // [0  1  2  3 ]
        // [4  5  6  7 ]
        // [8  9  10 11]
        // [12 13 14 15]
        let input: Vec<Complex<f64>> = (0..16).map(|i| Complex::new(f64::from(i), 0.0)).collect();
        let mut output = vec![Complex::zero(); 16];

        solver.execute(&input, &mut output, Sign::Forward);

        // Column 0: [0, 4, 8, 12] -> DC = 24
        assert!(complex_approx_eq(output[0], Complex::new(24.0, 0.0), 1e-10));

        // Column 1: [1, 5, 9, 13] -> DC = 28
        assert!(complex_approx_eq(output[1], Complex::new(28.0, 0.0), 1e-10));

        // Column 2: [2, 6, 10, 14] -> DC = 32
        assert!(complex_approx_eq(output[2], Complex::new(32.0, 0.0), 1e-10));

        // Column 3: [3, 7, 11, 15] -> DC = 36
        assert!(complex_approx_eq(output[3], Complex::new(36.0, 0.0), 1e-10));
    }

    #[test]
    fn test_batch_inplace() {
        let n = 8;
        let howmany = 3;
        let solver = VrankGeq1Solver::<f64>::new_contiguous(n, howmany);

        let input: Vec<Complex<f64>> = (0..(n * howmany))
            .map(|i| Complex::new(i as f64, 0.0))
            .collect();

        // Out-of-place
        let mut out_of_place = vec![Complex::zero(); n * howmany];
        solver.execute(&input, &mut out_of_place, Sign::Forward);

        // In-place
        let mut in_place = input;
        solver.execute_inplace(&mut in_place, Sign::Forward);

        for (a, b) in out_of_place.iter().zip(in_place.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_batch_non_power_of_2() {
        let n = 5; // Prime size (uses Bluestein)
        let howmany = 2;
        let solver = VrankGeq1Solver::<f64>::new_contiguous(n, howmany);

        let original: Vec<Complex<f64>> = (0..(n * howmany))
            .map(|i| Complex::new((i as f64).sin(), 0.0))
            .collect();
        let mut transformed = vec![Complex::zero(); n * howmany];
        let mut recovered = vec![Complex::zero(); n * howmany];

        solver.execute(&original, &mut transformed, Sign::Forward);
        solver.execute(&transformed, &mut recovered, Sign::Backward);

        let scale = n as f64;
        for (a, b) in original.iter().zip(recovered.iter()) {
            let normalized = Complex::new(b.re / scale, b.im / scale);
            assert!(complex_approx_eq(*a, normalized, 1e-9));
        }
    }

    #[test]
    fn test_batch_from_dims() {
        let transform_dim = IoDim::new(8, 1, 1); // n=8, stride=1
        let batch_dim = IoDim::new(4, 8, 8); // howmany=4, dist=8

        let solver = VrankGeq1Solver::<f64>::from_dims(&transform_dim, &batch_dim);

        assert_eq!(solver.n(), 8);
        assert_eq!(solver.howmany(), 4);
    }
}
