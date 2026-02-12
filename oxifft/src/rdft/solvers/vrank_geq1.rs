//! Batched RDFT solver.
//!
//! This solver handles batched real-to-complex and complex-to-real transforms
//! where the same 1D RDFT is applied to multiple arrays.

use crate::kernel::{Complex, Float, IoDim};
use crate::prelude::*;
use crate::rdft::solvers::{C2rSolver, R2cSolver};

/// Batched RDFT solver.
///
/// Executes multiple 1D real FFTs with configurable strides.
/// Supports both R2C (real-to-complex) and C2R (complex-to-real) transforms.
pub struct RdftVrankGeq1Solver<T: Float> {
    /// Transform size (length of each real FFT)
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

impl<T: Float> Default for RdftVrankGeq1Solver<T> {
    fn default() -> Self {
        Self::new_contiguous(1, 1)
    }
}

impl<T: Float> RdftVrankGeq1Solver<T> {
    /// Create a new batched RDFT solver with full stride control.
    ///
    /// # Arguments
    /// * `n` - Transform size (length of each real FFT)
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
    /// For R2C: Input batches are contiguous real arrays of size n,
    /// output batches are contiguous complex arrays of size n/2+1.
    ///
    /// For C2R: Input batches are contiguous complex arrays of size n/2+1,
    /// output batches are contiguous real arrays of size n.
    #[must_use]
    pub fn new_contiguous(n: usize, howmany: usize) -> Self {
        Self::new(n, howmany, 1, 1, n as isize, n as isize)
    }

    /// Create a solver from IoDim specifications.
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

    /// Returns the solver name identifier (`"rdft-vrank-geq1"`).
    #[must_use]
    pub fn name(&self) -> &'static str {
        "rdft-vrank-geq1"
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

    /// Execute batched R2C (real-to-complex) FFT.
    ///
    /// For each batch, takes n real values and produces n/2+1 complex values.
    ///
    /// # Arguments
    /// * `input` - Input buffer of real values
    /// * `output` - Output buffer of complex values
    pub fn execute_r2c(&self, input: &[T], output: &mut [Complex<T>]) {
        if self.n == 0 || self.howmany == 0 {
            return;
        }

        let out_len = self.n / 2 + 1;

        // Optimized path for contiguous data
        if self.istride == 1
            && self.ostride == 1
            && self.idist == self.n as isize
            && self.odist == out_len as isize
        {
            let solver = R2cSolver::new(self.n);
            for batch in 0..self.howmany {
                let in_start = batch * self.n;
                let out_start = batch * out_len;
                solver.execute(
                    &input[in_start..in_start + self.n],
                    &mut output[out_start..out_start + out_len],
                );
            }
            return;
        }

        // General strided path
        let mut in_buf = vec![T::ZERO; self.n];
        let mut out_buf = vec![Complex::zero(); out_len];
        let solver = R2cSolver::new(self.n);

        for batch in 0..self.howmany {
            let in_base = (batch as isize * self.idist) as usize;
            let out_base = (batch as isize * self.odist) as usize;

            // Gather input with stride
            for i in 0..self.n {
                let idx = in_base as isize + i as isize * self.istride;
                in_buf[i] = input[idx as usize];
            }

            // Execute R2C FFT
            solver.execute(&in_buf, &mut out_buf);

            // Scatter output with stride
            for i in 0..out_len {
                let idx = out_base as isize + i as isize * self.ostride;
                output[idx as usize] = out_buf[i];
            }
        }
    }

    /// Execute batched C2R (complex-to-real) FFT with normalization.
    ///
    /// For each batch, takes n/2+1 complex values and produces n real values.
    /// Output is normalized by 1/n.
    ///
    /// # Arguments
    /// * `input` - Input buffer of complex values
    /// * `output` - Output buffer of real values
    pub fn execute_c2r(&self, input: &[Complex<T>], output: &mut [T]) {
        if self.n == 0 || self.howmany == 0 {
            return;
        }

        let in_len = self.n / 2 + 1;

        // Optimized path for contiguous data
        if self.istride == 1
            && self.ostride == 1
            && self.idist == in_len as isize
            && self.odist == self.n as isize
        {
            let solver = C2rSolver::new(self.n);
            for batch in 0..self.howmany {
                let in_start = batch * in_len;
                let out_start = batch * self.n;
                solver.execute_normalized(
                    &input[in_start..in_start + in_len],
                    &mut output[out_start..out_start + self.n],
                );
            }
            return;
        }

        // General strided path
        let mut in_buf = vec![Complex::zero(); in_len];
        let mut out_buf = vec![T::ZERO; self.n];
        let solver = C2rSolver::new(self.n);

        for batch in 0..self.howmany {
            let in_base = (batch as isize * self.idist) as usize;
            let out_base = (batch as isize * self.odist) as usize;

            // Gather input with stride
            for i in 0..in_len {
                let idx = in_base as isize + i as isize * self.istride;
                in_buf[i] = input[idx as usize];
            }

            // Execute C2R FFT
            solver.execute_normalized(&in_buf, &mut out_buf);

            // Scatter output with stride
            for i in 0..self.n {
                let idx = out_base as isize + i as isize * self.ostride;
                output[idx as usize] = out_buf[i];
            }
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
    fn test_batch_r2c_contiguous() {
        let n = 8;
        let howmany = 3;
        let out_len = n / 2 + 1;
        // For R2C, idist = n (real elements), odist = out_len (complex elements)
        let solver =
            RdftVrankGeq1Solver::<f64>::new(n, howmany, 1, 1, n as isize, out_len as isize);

        // Create input: 3 batches of 8 real elements each
        let input: Vec<f64> = (0..(n * howmany)).map(|i| i as f64).collect();
        let mut output = vec![Complex::zero(); out_len * howmany];

        solver.execute_r2c(&input, &mut output);

        // Each batch DC should equal sum of its elements
        // Batch 0: 0+1+2+3+4+5+6+7 = 28
        // Batch 1: 8+9+10+11+12+13+14+15 = 92
        // Batch 2: 16+17+18+19+20+21+22+23 = 156
        assert!(complex_approx_eq(output[0], Complex::new(28.0, 0.0), 1e-10));
        assert!(complex_approx_eq(
            output[out_len],
            Complex::new(92.0, 0.0),
            1e-10
        ));
        assert!(complex_approx_eq(
            output[2 * out_len],
            Complex::new(156.0, 0.0),
            1e-10
        ));
    }

    #[test]
    fn test_batch_r2c_c2r_roundtrip() {
        let n = 8;
        let howmany = 4;
        let out_len = n / 2 + 1;
        let solver = RdftVrankGeq1Solver::<f64>::new(
            n,
            howmany,
            1,
            1,
            n as isize,
            out_len as isize, // Input contiguous, output by complex count
        );
        let solver_back = RdftVrankGeq1Solver::<f64>::new(
            n,
            howmany,
            1,
            1,
            out_len as isize,
            n as isize, // Input by complex count, output contiguous
        );

        let original: Vec<f64> = (0..(n * howmany)).map(|i| (i as f64).sin()).collect();
        let mut transformed = vec![Complex::zero(); out_len * howmany];
        let mut recovered = vec![0.0_f64; n * howmany];

        solver.execute_r2c(&original, &mut transformed);
        solver_back.execute_c2r(&transformed, &mut recovered);

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(approx_eq(*a, *b, 1e-10), "got {b}, expected {a}");
        }
    }

    #[test]
    fn test_batch_from_dims() {
        let transform_dim = IoDim::new(16, 1, 1);
        let batch_dim = IoDim::new(4, 16, 16);

        let solver = RdftVrankGeq1Solver::<f64>::from_dims(&transform_dim, &batch_dim);

        assert_eq!(solver.n(), 16);
        assert_eq!(solver.howmany(), 4);
    }
}
