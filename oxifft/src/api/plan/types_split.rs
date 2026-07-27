//! Split-complex FFT plan types (1D, 2D, 3D, ND).
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#![allow(clippy::items_after_statements)] // reason: SplitRS-generated code places type defs and constants after use statements

use crate::api::{Direction, Flags};
use crate::kernel::{Complex, Float};
use crate::prelude::*;

use super::types::{Plan, Plan2D, Plan3D};
use super::types_nd::PlanND;

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
/// ```
/// use oxifft::{SplitPlan, Direction, Flags};
///
/// let n = 256;
/// let plan = SplitPlan::<f64>::dft_1d(n, Direction::Forward, Flags::ESTIMATE)
///     .expect("256 is a supported size");
///
/// let mut in_real = vec![0.0; n];
/// let in_imag = vec![0.0; n];
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
    /// Underlying 3D complex plan (flag-aware, unnormalized).
    plan: Plan3D<T>,
}
impl<T: Float> SplitPlan3D<T> {
    /// Create a 3D split-complex plan.
    ///
    /// `flags` are forwarded to the internal [`Plan3D`], so
    /// `MEASURE`/`PATIENT`/wisdom influence per-axis algorithm selection (this
    /// used to ignore the flags and always use the scalar `GenericSolver`).
    ///
    /// Like [`SplitPlan`] and [`SplitPlan2D`], `execute` is **unnormalized** in
    /// both directions; the `1/N` inverse normalization is applied by the
    /// [`ifft3d_split`](crate::ifft3d_split) convenience wrapper.
    #[must_use]
    pub fn new(
        n0: usize,
        n1: usize,
        n2: usize,
        direction: Direction,
        flags: Flags,
    ) -> Option<Self> {
        let plan = Plan3D::new(n0, n1, n2, direction, flags)?;
        Some(Self { plan })
    }
    /// Execute the 3D split-complex transform (unnormalized).
    pub fn execute(&self, in_real: &[T], in_imag: &[T], out_real: &mut [T], out_imag: &mut [T]) {
        let total = self.plan.size();
        assert_eq!(in_real.len(), total);
        assert_eq!(in_imag.len(), total);
        assert_eq!(out_real.len(), total);
        assert_eq!(out_imag.len(), total);
        let input: Vec<Complex<T>> = in_real
            .iter()
            .zip(in_imag.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        let mut output = vec![Complex::<T>::zero(); total];
        self.plan.execute(&input, &mut output);
        for (i, c) in output.iter().enumerate() {
            out_real[i] = c.re;
            out_imag[i] = c.im;
        }
    }
    /// Execute in-place 3D split-complex transform (unnormalized).
    pub fn execute_inplace(&self, real: &mut [T], imag: &mut [T]) {
        let total = self.plan.size();
        assert_eq!(real.len(), total);
        assert_eq!(imag.len(), total);
        let mut data: Vec<Complex<T>> = real
            .iter()
            .zip(imag.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        self.plan.execute_inplace(&mut data);
        for (i, c) in data.iter().enumerate() {
            real[i] = c.re;
            imag[i] = c.im;
        }
    }
    /// Get direction (crate-internal).
    #[must_use]
    pub(crate) fn direction(&self) -> Direction {
        self.plan.direction()
    }
    /// Get first dimension (crate-internal).
    #[must_use]
    pub(crate) fn dim0(&self) -> usize {
        self.plan.dim0()
    }
    /// Get second dimension (crate-internal).
    #[must_use]
    pub(crate) fn dim1(&self) -> usize {
        self.plan.dim1()
    }
    /// Get third dimension (crate-internal).
    #[must_use]
    pub(crate) fn dim2(&self) -> usize {
        self.plan.dim2()
    }
}
/// A plan for N-dimensional split-complex transforms.
pub struct SplitPlanND<T: Float> {
    /// Underlying N-D complex plan (flag-aware, unnormalized).
    plan: PlanND<T>,
}
impl<T: Float> SplitPlanND<T> {
    /// Create an N-dimensional split-complex plan.
    ///
    /// `flags` are forwarded to the internal [`PlanND`], so
    /// `MEASURE`/`PATIENT`/wisdom influence per-axis algorithm selection (this
    /// used to ignore the flags and always use the scalar `GenericSolver`).
    ///
    /// Like the lower-rank split plans, `execute` is **unnormalized**; the
    /// `1/N` inverse normalization is applied by the
    /// [`ifft_nd_split`](crate::ifft_nd_split) convenience wrapper.
    #[must_use]
    pub fn new(dims: &[usize], direction: Direction, flags: Flags) -> Option<Self> {
        if dims.is_empty() || dims.contains(&0) {
            return None;
        }
        let plan = PlanND::new(dims, direction, flags)?;
        Some(Self { plan })
    }
    /// Execute the N-dimensional split-complex transform (unnormalized).
    pub fn execute(&self, in_real: &[T], in_imag: &[T], out_real: &mut [T], out_imag: &mut [T]) {
        let total = self.plan.size();
        assert_eq!(in_real.len(), total);
        assert_eq!(in_imag.len(), total);
        assert_eq!(out_real.len(), total);
        assert_eq!(out_imag.len(), total);
        let input: Vec<Complex<T>> = in_real
            .iter()
            .zip(in_imag.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        let mut output = vec![Complex::<T>::zero(); total];
        self.plan.execute(&input, &mut output);
        for (i, c) in output.iter().enumerate() {
            out_real[i] = c.re;
            out_imag[i] = c.im;
        }
    }
    /// Execute in-place N-dimensional split-complex transform (unnormalized).
    pub fn execute_inplace(&self, real: &mut [T], imag: &mut [T]) {
        let total = self.plan.size();
        assert_eq!(real.len(), total);
        assert_eq!(imag.len(), total);
        let mut data: Vec<Complex<T>> = real
            .iter()
            .zip(imag.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        self.plan.execute_inplace(&mut data);
        for (i, c) in data.iter().enumerate() {
            real[i] = c.re;
            imag[i] = c.im;
        }
    }
    /// Get dims (crate-internal).
    #[must_use]
    pub(crate) fn dims(&self) -> &[usize] {
        self.plan.dims()
    }
    /// Get direction (crate-internal).
    #[must_use]
    pub(crate) fn direction(&self) -> Direction {
        self.plan.direction()
    }
}
