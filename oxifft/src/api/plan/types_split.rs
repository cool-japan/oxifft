//! Split-complex FFT plan types (1D, 2D, 3D, ND).
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#![allow(clippy::items_after_statements)] // reason: SplitRS-generated code places type defs and constants after use statements

use crate::api::{Direction, Flags};
use crate::dft::problem::Sign;
use crate::kernel::{Complex, Float};
use crate::prelude::*;

use super::types::{Plan, Plan2D};

/// A plan for split-complex format (separate real and imaginary arrays).
///
/// Split-complex format stores the real parts in one array and the imaginary
/// parts in another, rather than interleaving them:
///
/// - Interleaved: `[re0, im0, re1, im1, re2, im2, ...]`
/// - Split: `real = [re0, re1, re2, ...]`, `imag = [im0, im1, im2, ...]`
///
/// This format can be more efficient for SIMD processing and is used by
/// some numerical libraries.
///
/// # Example
///
/// ```ignore
/// use oxifft::{SplitPlan, Direction, Flags};
///
/// let n = 256;
/// let plan = SplitPlan::<f64>::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
///
/// let mut in_real = vec![0.0; n];
/// let mut in_imag = vec![0.0; n];
/// let mut out_real = vec![0.0; n];
/// let mut out_imag = vec![0.0; n];
///
/// // Initialize input...
/// in_real[0] = 1.0;
///
/// plan.execute(&in_real, &in_imag, &mut out_real, &mut out_imag);
/// ```
pub struct SplitPlan<T: Float> {
    /// Underlying complex plan
    plan: Plan<T>,
}
impl<T: Float> SplitPlan<T> {
    /// Create a 1D DFT plan for split-complex format.
    ///
    /// # Arguments
    /// * `n` - Transform size
    /// * `direction` - Forward or Backward
    /// * `flags` - Planning flags
    #[must_use]
    pub fn dft_1d(n: usize, direction: Direction, flags: Flags) -> Option<Self> {
        let plan = Plan::dft_1d(n, direction, flags)?;
        Some(Self { plan })
    }
    /// Get the transform size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.plan.size()
    }
    /// Get the transform direction.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.plan.direction()
    }
    /// Execute the transform on split-complex input/output.
    ///
    /// # Arguments
    /// * `in_real` - Input real parts
    /// * `in_imag` - Input imaginary parts
    /// * `out_real` - Output real parts
    /// * `out_imag` - Output imaginary parts
    ///
    /// # Panics
    /// Panics if any buffer size doesn't match the plan size.
    pub fn execute(&self, in_real: &[T], in_imag: &[T], out_real: &mut [T], out_imag: &mut [T]) {
        let n = self.plan.size();
        assert_eq!(in_real.len(), n, "Input real size must match plan size");
        assert_eq!(
            in_imag.len(),
            n,
            "Input imaginary size must match plan size"
        );
        assert_eq!(out_real.len(), n, "Output real size must match plan size");
        assert_eq!(
            out_imag.len(),
            n,
            "Output imaginary size must match plan size"
        );
        let input: Vec<Complex<T>> = in_real
            .iter()
            .zip(in_imag.iter())
            .map(|(&re, &im)| Complex::new(re, im))
            .collect();
        let mut output = vec![Complex::<T>::zero(); n];
        self.plan.execute(&input, &mut output);
        for (i, c) in output.iter().enumerate() {
            out_real[i] = c.re;
            out_imag[i] = c.im;
        }
    }
    /// Execute the transform in-place on split-complex data.
    ///
    /// # Arguments
    /// * `real` - Real parts (in-place)
    /// * `imag` - Imaginary parts (in-place)
    ///
    /// # Panics
    /// Panics if any buffer size doesn't match the plan size.
    pub fn execute_inplace(&self, real: &mut [T], imag: &mut [T]) {
        let n = self.plan.size();
        assert_eq!(real.len(), n, "Real size must match plan size");
        assert_eq!(imag.len(), n, "Imaginary size must match plan size");
        let mut data: Vec<Complex<T>> = real
            .iter()
            .zip(imag.iter())
            .map(|(&re, &im)| Complex::new(re, im))
            .collect();
        self.plan.execute_inplace(&mut data);
        for (i, c) in data.iter().enumerate() {
            real[i] = c.re;
            imag[i] = c.im;
        }
    }
}
/// A multi-dimensional plan for split-complex format.
pub struct SplitPlan2D<T: Float> {
    /// Underlying 2D complex plan
    plan: Plan2D<T>,
}
impl<T: Float> SplitPlan2D<T> {
    /// Create a 2D DFT plan for split-complex format.
    ///
    /// # Arguments
    /// * `n0` - Number of rows
    /// * `n1` - Number of columns
    /// * `direction` - Forward or Backward
    /// * `flags` - Planning flags
    #[must_use]
    pub fn new(n0: usize, n1: usize, direction: Direction, flags: Flags) -> Option<Self> {
        let plan = Plan2D::new(n0, n1, direction, flags)?;
        Some(Self { plan })
    }
    /// Get the number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.plan.rows()
    }
    /// Get the number of columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.plan.cols()
    }
    /// Get the total size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.plan.size()
    }
    /// Get the transform direction.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.plan.direction()
    }
    /// Execute the 2D transform on split-complex input/output.
    ///
    /// Data is row-major order.
    ///
    /// # Panics
    /// Panics if any buffer size doesn't match n0 × n1.
    pub fn execute(&self, in_real: &[T], in_imag: &[T], out_real: &mut [T], out_imag: &mut [T]) {
        let n = self.size();
        assert_eq!(in_real.len(), n, "Input real size must match n0 × n1");
        assert_eq!(in_imag.len(), n, "Input imaginary size must match n0 × n1");
        assert_eq!(out_real.len(), n, "Output real size must match n0 × n1");
        assert_eq!(
            out_imag.len(),
            n,
            "Output imaginary size must match n0 × n1"
        );
        let input: Vec<Complex<T>> = in_real
            .iter()
            .zip(in_imag.iter())
            .map(|(&re, &im)| Complex::new(re, im))
            .collect();
        let mut output = vec![Complex::<T>::zero(); n];
        self.plan.execute(&input, &mut output);
        for (i, c) in output.iter().enumerate() {
            out_real[i] = c.re;
            out_imag[i] = c.im;
        }
    }
    /// Execute in-place on split-complex data.
    pub fn execute_inplace(&self, real: &mut [T], imag: &mut [T]) {
        let n = self.size();
        assert_eq!(real.len(), n, "Real size must match n0 × n1");
        assert_eq!(imag.len(), n, "Imaginary size must match n0 × n1");
        let mut data: Vec<Complex<T>> = real
            .iter()
            .zip(imag.iter())
            .map(|(&re, &im)| Complex::new(re, im))
            .collect();
        self.plan.execute_inplace(&mut data);
        for (i, c) in data.iter().enumerate() {
            real[i] = c.re;
            imag[i] = c.im;
        }
    }
}
/// A plan for 3D split-complex transforms.
pub struct SplitPlan3D<T: Float> {
    n0: usize,
    n1: usize,
    n2: usize,
    direction: Direction,
    _marker: core::marker::PhantomData<T>,
}
impl<T: Float> SplitPlan3D<T> {
    /// Create a 3D split-complex plan.
    #[must_use]
    pub fn new(
        n0: usize,
        n1: usize,
        n2: usize,
        direction: Direction,
        _flags: Flags,
    ) -> Option<Self> {
        if n0 == 0 || n1 == 0 || n2 == 0 {
            return None;
        }
        Some(Self {
            n0,
            n1,
            n2,
            direction,
            _marker: core::marker::PhantomData,
        })
    }
    /// Execute the 3D split-complex transform.
    pub fn execute(&self, in_real: &[T], in_imag: &[T], out_real: &mut [T], out_imag: &mut [T]) {
        let total = self.n0 * self.n1 * self.n2;
        assert_eq!(in_real.len(), total);
        assert_eq!(in_imag.len(), total);
        assert_eq!(out_real.len(), total);
        assert_eq!(out_imag.len(), total);
        let mut data: Vec<Complex<T>> = in_real
            .iter()
            .zip(in_imag.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        let sign = match self.direction {
            Direction::Forward => Sign::Forward,
            Direction::Backward => Sign::Backward,
        };
        use crate::dft::solvers::GenericSolver;
        let solver_n2 = GenericSolver::new(self.n2);
        let mut row = vec![Complex::zero(); self.n2];
        let mut row_out = vec![Complex::zero(); self.n2];
        for i in 0..(self.n0 * self.n1) {
            let start = i * self.n2;
            row.copy_from_slice(&data[start..start + self.n2]);
            solver_n2.execute(&row, &mut row_out, sign);
            data[start..start + self.n2].copy_from_slice(&row_out);
        }
        let solver_n1 = GenericSolver::new(self.n1);
        let mut col = vec![Complex::zero(); self.n1];
        let mut col_out = vec![Complex::zero(); self.n1];
        for i in 0..self.n0 {
            for k in 0..self.n2 {
                for j in 0..self.n1 {
                    col[j] = data[i * self.n1 * self.n2 + j * self.n2 + k];
                }
                solver_n1.execute(&col, &mut col_out, sign);
                for j in 0..self.n1 {
                    data[i * self.n1 * self.n2 + j * self.n2 + k] = col_out[j];
                }
            }
        }
        let solver_n0 = GenericSolver::new(self.n0);
        let mut depth = vec![Complex::zero(); self.n0];
        let mut depth_out = vec![Complex::zero(); self.n0];
        for j in 0..self.n1 {
            for k in 0..self.n2 {
                for i in 0..self.n0 {
                    depth[i] = data[i * self.n1 * self.n2 + j * self.n2 + k];
                }
                solver_n0.execute(&depth, &mut depth_out, sign);
                for i in 0..self.n0 {
                    data[i * self.n1 * self.n2 + j * self.n2 + k] = depth_out[i];
                }
            }
        }
        if self.direction == Direction::Backward {
            let scale = T::one() / T::from_usize(total);
            for c in &mut data {
                *c = *c * scale;
            }
        }
        for (i, c) in data.iter().enumerate() {
            out_real[i] = c.re;
            out_imag[i] = c.im;
        }
    }
    /// Execute in-place 3D split-complex transform.
    pub fn execute_inplace(&self, real: &mut [T], imag: &mut [T]) {
        let total = self.n0 * self.n1 * self.n2;
        assert_eq!(real.len(), total);
        assert_eq!(imag.len(), total);
        let mut out_real = vec![T::ZERO; total];
        let mut out_imag = vec![T::ZERO; total];
        self.execute(real, imag, &mut out_real, &mut out_imag);
        real.copy_from_slice(&out_real);
        imag.copy_from_slice(&out_imag);
    }
    /// Get direction (crate-internal).
    #[must_use]
    pub(crate) fn direction(&self) -> Direction {
        self.direction
    }
    /// Get first dimension (crate-internal).
    #[must_use]
    pub(crate) fn dim0(&self) -> usize {
        self.n0
    }
    /// Get second dimension (crate-internal).
    #[must_use]
    pub(crate) fn dim1(&self) -> usize {
        self.n1
    }
    /// Get third dimension (crate-internal).
    #[must_use]
    pub(crate) fn dim2(&self) -> usize {
        self.n2
    }
}
/// A plan for N-dimensional split-complex transforms.
pub struct SplitPlanND<T: Float> {
    dims: Vec<usize>,
    direction: Direction,
    _marker: core::marker::PhantomData<T>,
}
impl<T: Float> SplitPlanND<T> {
    /// Create an N-dimensional split-complex plan.
    #[must_use]
    pub fn new(dims: &[usize], direction: Direction, _flags: Flags) -> Option<Self> {
        if dims.is_empty() || dims.contains(&0) {
            return None;
        }
        Some(Self {
            dims: dims.to_vec(),
            direction,
            _marker: core::marker::PhantomData,
        })
    }
    /// Execute the N-dimensional split-complex transform.
    pub fn execute(&self, in_real: &[T], in_imag: &[T], out_real: &mut [T], out_imag: &mut [T]) {
        let total: usize = self.dims.iter().product();
        assert_eq!(in_real.len(), total);
        assert_eq!(in_imag.len(), total);
        assert_eq!(out_real.len(), total);
        assert_eq!(out_imag.len(), total);
        let mut data: Vec<Complex<T>> = in_real
            .iter()
            .zip(in_imag.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        let sign = match self.direction {
            Direction::Forward => Sign::Forward,
            Direction::Backward => Sign::Backward,
        };
        use crate::dft::solvers::GenericSolver;
        for (dim_idx, &dim_size) in self.dims.iter().enumerate().rev() {
            let solver = GenericSolver::new(dim_size);
            let mut buf = vec![Complex::zero(); dim_size];
            let mut buf_out = vec![Complex::zero(); dim_size];
            let inner_stride: usize = self.dims[dim_idx + 1..].iter().product();
            let inner_stride = inner_stride.max(1);
            let outer_count: usize = self.dims[..dim_idx].iter().product();
            let outer_count = outer_count.max(1);
            for outer in 0..outer_count {
                for inner in 0..inner_stride {
                    for k in 0..dim_size {
                        let idx = outer * (dim_size * inner_stride) + k * inner_stride + inner;
                        buf[k] = data[idx];
                    }
                    solver.execute(&buf, &mut buf_out, sign);
                    for k in 0..dim_size {
                        let idx = outer * (dim_size * inner_stride) + k * inner_stride + inner;
                        data[idx] = buf_out[k];
                    }
                }
            }
        }
        if self.direction == Direction::Backward {
            let scale = T::one() / T::from_usize(total);
            for c in &mut data {
                *c = *c * scale;
            }
        }
        for (i, c) in data.iter().enumerate() {
            out_real[i] = c.re;
            out_imag[i] = c.im;
        }
    }
    /// Execute in-place N-dimensional split-complex transform.
    pub fn execute_inplace(&self, real: &mut [T], imag: &mut [T]) {
        let total: usize = self.dims.iter().product();
        assert_eq!(real.len(), total);
        assert_eq!(imag.len(), total);
        let mut out_real = vec![T::ZERO; total];
        let mut out_imag = vec![T::ZERO; total];
        self.execute(real, imag, &mut out_real, &mut out_imag);
        real.copy_from_slice(&out_real);
        imag.copy_from_slice(&out_imag);
    }
    /// Get dims (crate-internal).
    #[must_use]
    pub(crate) fn dims(&self) -> &[usize] {
        &self.dims
    }
    /// Get direction (crate-internal).
    #[must_use]
    pub(crate) fn direction(&self) -> Direction {
        self.direction
    }
}
