//! Real FFT plan types (1D, 2D, 3D).
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#![allow(clippy::items_after_statements)] // reason: SplitRS-generated code places type defs and constants after use statements

use crate::api::Flags;
use crate::dft::problem::Sign;
use crate::kernel::{Complex, Float};
use crate::prelude::*;

use super::types::RealPlanKind;

/// A plan for executing real FFT transforms.
///
/// Real FFTs are more efficient than complex FFTs for real-valued input,
/// producing only the non-redundant half of the spectrum.
pub struct RealPlan<T: Float> {
    /// Transform size (number of real values)
    n: usize,
    /// Transform kind
    kind: RealPlanKind,
    _marker: core::marker::PhantomData<T>,
}
impl<T: Float> RealPlan<T> {
    /// Create a 1D real-to-complex FFT plan.
    ///
    /// # Arguments
    /// * `n` - Transform size (number of real input values)
    /// * `flags` - Planning flags
    ///
    /// # Returns
    /// A plan that transforms n real values to n/2+1 complex values.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxifft::{Complex, Flags, RealPlan};
    ///
    /// let plan = RealPlan::<f64>::r2c_1d(8, Flags::ESTIMATE)
    ///     .expect("plan construction failed");
    /// // DC bin = sum of all real inputs
    /// let input = vec![1.0_f64; 8];
    /// let mut output = vec![Complex::<f64>::zero(); plan.complex_size()];
    /// plan.execute_r2c(&input, &mut output);
    /// // For all-ones input, DC bin = 8
    /// assert!((output[0].re - 8.0_f64).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn r2c_1d(n: usize, _flags: Flags) -> Option<Self> {
        if n == 0 {
            return None;
        }
        Some(Self {
            n,
            kind: RealPlanKind::R2C,
            _marker: core::marker::PhantomData,
        })
    }
    /// Create a 1D complex-to-real FFT plan.
    ///
    /// # Arguments
    /// * `n` - Transform size (number of real output values)
    /// * `flags` - Planning flags
    ///
    /// # Returns
    /// A plan that transforms n/2+1 complex values to n real values.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxifft::{Complex, Flags, RealPlan};
    ///
    /// // Round-trip: r2c followed by c2r recovers the original signal
    /// let n = 8;
    /// let r2c = RealPlan::<f64>::r2c_1d(n, Flags::ESTIMATE).unwrap();
    /// let c2r = RealPlan::<f64>::c2r_1d(n, Flags::ESTIMATE).unwrap();
    ///
    /// let input = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// let mut spectrum = vec![Complex::<f64>::zero(); r2c.complex_size()];
    /// r2c.execute_r2c(&input, &mut spectrum);
    ///
    /// let mut recovered = vec![0.0_f64; n];
    /// c2r.execute_c2r(&spectrum, &mut recovered);
    /// // execute_c2r normalizes by 1/n automatically
    /// assert!((recovered[0] - 1.0_f64).abs() < 1e-9);
    /// assert!((recovered[3] - 4.0_f64).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn c2r_1d(n: usize, _flags: Flags) -> Option<Self> {
        if n == 0 {
            return None;
        }
        Some(Self {
            n,
            kind: RealPlanKind::C2R,
            _marker: core::marker::PhantomData,
        })
    }
    /// Get the transform size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.n
    }
    /// Get the complex buffer size (n/2 + 1).
    #[must_use]
    pub fn complex_size(&self) -> usize {
        self.n / 2 + 1
    }
    /// Get the transform kind.
    #[must_use]
    pub fn kind(&self) -> RealPlanKind {
        self.kind
    }
    /// Execute the R2C plan.
    ///
    /// # Panics
    /// Panics if the plan is not R2C or buffer sizes don't match.
    pub fn execute_r2c(&self, input: &[T], output: &mut [Complex<T>]) {
        use crate::rdft::solvers::R2cSolver;
        assert_eq!(self.kind, RealPlanKind::R2C, "Plan must be R2C");
        assert_eq!(input.len(), self.n, "Input size must match plan size");
        assert_eq!(
            output.len(),
            self.complex_size(),
            "Output size must be n/2+1"
        );
        R2cSolver::new(self.n).execute(input, output);
    }
    /// Execute the C2R plan with normalization.
    ///
    /// Output is normalized by 1/n.
    ///
    /// # Panics
    /// Panics if the plan is not C2R or buffer sizes don't match.
    pub fn execute_c2r(&self, input: &[Complex<T>], output: &mut [T]) {
        use crate::rdft::solvers::C2rSolver;
        assert_eq!(self.kind, RealPlanKind::C2R, "Plan must be C2R");
        assert_eq!(input.len(), self.complex_size(), "Input size must be n/2+1");
        assert_eq!(output.len(), self.n, "Output size must match plan size");
        C2rSolver::new(self.n).execute_normalized(input, output);
    }
    /// Execute the C2R plan without normalization.
    ///
    /// # Panics
    /// Panics if the plan is not C2R or buffer sizes don't match.
    pub fn execute_c2r_unnormalized(&self, input: &[Complex<T>], output: &mut [T]) {
        use crate::rdft::solvers::C2rSolver;
        assert_eq!(self.kind, RealPlanKind::C2R, "Plan must be C2R");
        assert_eq!(input.len(), self.complex_size(), "Input size must be n/2+1");
        assert_eq!(output.len(), self.n, "Output size must match plan size");
        C2rSolver::new(self.n).execute(input, output);
    }
}
/// A plan for 2D real-to-complex and complex-to-real transforms.
///
/// For R2C: Takes n0×n1 real values and produces n0×(n1/2+1) complex values.
/// For C2R: Takes n0×(n1/2+1) complex values and produces n0×n1 real values.
pub struct RealPlan2D<T: Float> {
    n0: usize,
    n1: usize,
    kind: RealPlanKind,
    _marker: core::marker::PhantomData<T>,
}
impl<T: Float> RealPlan2D<T> {
    /// Create a 2D R2C plan.
    #[must_use]
    pub fn r2c(n0: usize, n1: usize, _flags: Flags) -> Option<Self> {
        if n0 == 0 || n1 == 0 {
            return None;
        }
        Some(Self {
            n0,
            n1,
            kind: RealPlanKind::R2C,
            _marker: core::marker::PhantomData,
        })
    }
    /// Create a 2D C2R plan.
    #[must_use]
    pub fn c2r(n0: usize, n1: usize, _flags: Flags) -> Option<Self> {
        if n0 == 0 || n1 == 0 {
            return None;
        }
        Some(Self {
            n0,
            n1,
            kind: RealPlanKind::C2R,
            _marker: core::marker::PhantomData,
        })
    }
    /// Execute the 2D real plan.
    ///
    /// For R2C: input is n0×n1 real, output is n0×(n1/2+1) complex.
    /// For C2R: input is n0×(n1/2+1) complex, output is n0×n1 real.
    pub fn execute_r2c(&self, input: &[T], output: &mut [Complex<T>]) {
        assert_eq!(self.kind, RealPlanKind::R2C);
        let expected_in = self.n0 * self.n1;
        let expected_out = self.n0 * (self.n1 / 2 + 1);
        assert_eq!(input.len(), expected_in);
        assert_eq!(output.len(), expected_out);
        use crate::rdft::solvers::R2cSolver;
        let out_cols = self.n1 / 2 + 1;
        let r2c_solver = R2cSolver::new(self.n1);
        let mut temp = vec![Complex::zero(); self.n0 * out_cols];
        for row in 0..self.n0 {
            let in_start = row * self.n1;
            let out_start = row * out_cols;
            r2c_solver.execute(
                &input[in_start..in_start + self.n1],
                &mut temp[out_start..out_start + out_cols],
            );
        }
        use crate::dft::solvers::GenericSolver;
        let col_solver = GenericSolver::new(self.n0);
        let mut col_in = vec![Complex::zero(); self.n0];
        let mut col_out = vec![Complex::zero(); self.n0];
        for col in 0..out_cols {
            for row in 0..self.n0 {
                col_in[row] = temp[row * out_cols + col];
            }
            col_solver.execute(&col_in, &mut col_out, Sign::Forward);
            for row in 0..self.n0 {
                output[row * out_cols + col] = col_out[row];
            }
        }
    }
    /// Execute C2R transform.
    pub fn execute_c2r(&self, input: &[Complex<T>], output: &mut [T]) {
        assert_eq!(self.kind, RealPlanKind::C2R);
        let expected_in = self.n0 * (self.n1 / 2 + 1);
        let expected_out = self.n0 * self.n1;
        assert_eq!(input.len(), expected_in);
        assert_eq!(output.len(), expected_out);
        let out_cols = self.n1 / 2 + 1;
        use crate::dft::solvers::GenericSolver;
        let col_solver = GenericSolver::new(self.n0);
        let mut temp = vec![Complex::zero(); self.n0 * out_cols];
        let mut col_in = vec![Complex::zero(); self.n0];
        let mut col_out = vec![Complex::zero(); self.n0];
        for col in 0..out_cols {
            for row in 0..self.n0 {
                col_in[row] = input[row * out_cols + col];
            }
            col_solver.execute(&col_in, &mut col_out, Sign::Backward);
            for row in 0..self.n0 {
                temp[row * out_cols + col] = col_out[row];
            }
        }
        use crate::rdft::solvers::C2rSolver;
        let c2r_solver = C2rSolver::new(self.n1);
        for row in 0..self.n0 {
            let in_start = row * out_cols;
            let out_start = row * self.n1;
            c2r_solver.execute(
                &temp[in_start..in_start + out_cols],
                &mut output[out_start..out_start + self.n1],
            );
        }
    }
    /// Get row count (crate-internal).
    #[must_use]
    pub(crate) fn rows(&self) -> usize {
        self.n0
    }
    /// Get column count (crate-internal).
    #[must_use]
    pub(crate) fn cols(&self) -> usize {
        self.n1
    }
    /// Get kind (crate-internal).
    #[must_use]
    pub(crate) fn plan_kind(&self) -> RealPlanKind {
        self.kind
    }
}
/// A plan for 3D real-to-complex and complex-to-real transforms.
pub struct RealPlan3D<T: Float> {
    n0: usize,
    n1: usize,
    n2: usize,
    kind: RealPlanKind,
    _marker: core::marker::PhantomData<T>,
}
impl<T: Float> RealPlan3D<T> {
    /// Create a 3D R2C plan.
    #[must_use]
    pub fn r2c(n0: usize, n1: usize, n2: usize, _flags: Flags) -> Option<Self> {
        if n0 == 0 || n1 == 0 || n2 == 0 {
            return None;
        }
        Some(Self {
            n0,
            n1,
            n2,
            kind: RealPlanKind::R2C,
            _marker: core::marker::PhantomData,
        })
    }
    /// Create a 3D C2R plan.
    #[must_use]
    pub fn c2r(n0: usize, n1: usize, n2: usize, _flags: Flags) -> Option<Self> {
        if n0 == 0 || n1 == 0 || n2 == 0 {
            return None;
        }
        Some(Self {
            n0,
            n1,
            n2,
            kind: RealPlanKind::C2R,
            _marker: core::marker::PhantomData,
        })
    }
    /// Execute R2C transform.
    pub fn execute_r2c(&self, input: &[T], output: &mut [Complex<T>]) {
        assert_eq!(self.kind, RealPlanKind::R2C);
        let expected_in = self.n0 * self.n1 * self.n2;
        let expected_out = self.n0 * self.n1 * (self.n2 / 2 + 1);
        assert_eq!(input.len(), expected_in);
        assert_eq!(output.len(), expected_out);
        let out_last = self.n2 / 2 + 1;
        let slice_in_size = self.n1 * self.n2;
        let slice_out_size = self.n1 * out_last;
        let plan_2d = RealPlan2D::<T>::r2c(self.n1, self.n2, Flags::ESTIMATE)
            .expect("Failed to create internal 2D R2C plan");
        let mut temp = vec![Complex::zero(); self.n0 * slice_out_size];
        for i in 0..self.n0 {
            let in_start = i * slice_in_size;
            let out_start = i * slice_out_size;
            plan_2d.execute_r2c(
                &input[in_start..in_start + slice_in_size],
                &mut temp[out_start..out_start + slice_out_size],
            );
        }
        use crate::dft::solvers::GenericSolver;
        let n0_solver = GenericSolver::new(self.n0);
        let mut col_in = vec![Complex::zero(); self.n0];
        let mut col_out = vec![Complex::zero(); self.n0];
        for j in 0..self.n1 {
            for k in 0..out_last {
                for i in 0..self.n0 {
                    col_in[i] = temp[i * slice_out_size + j * out_last + k];
                }
                n0_solver.execute(&col_in, &mut col_out, Sign::Forward);
                for i in 0..self.n0 {
                    output[i * slice_out_size + j * out_last + k] = col_out[i];
                }
            }
        }
    }
    /// Execute C2R transform.
    pub fn execute_c2r(&self, input: &[Complex<T>], output: &mut [T]) {
        assert_eq!(self.kind, RealPlanKind::C2R);
        let expected_in = self.n0 * self.n1 * (self.n2 / 2 + 1);
        let expected_out = self.n0 * self.n1 * self.n2;
        assert_eq!(input.len(), expected_in);
        assert_eq!(output.len(), expected_out);
        let out_last = self.n2 / 2 + 1;
        let slice_in_size = self.n1 * out_last;
        let slice_out_size = self.n1 * self.n2;
        use crate::dft::solvers::GenericSolver;
        let n0_solver = GenericSolver::new(self.n0);
        let mut temp = vec![Complex::zero(); self.n0 * slice_in_size];
        let mut col_in = vec![Complex::zero(); self.n0];
        let mut col_out = vec![Complex::zero(); self.n0];
        for j in 0..self.n1 {
            for k in 0..out_last {
                for i in 0..self.n0 {
                    col_in[i] = input[i * slice_in_size + j * out_last + k];
                }
                n0_solver.execute(&col_in, &mut col_out, Sign::Backward);
                for i in 0..self.n0 {
                    temp[i * slice_in_size + j * out_last + k] = col_out[i];
                }
            }
        }
        let plan_2d = RealPlan2D::<T>::c2r(self.n1, self.n2, Flags::ESTIMATE)
            .expect("Failed to create internal 2D C2R plan");
        for i in 0..self.n0 {
            let in_start = i * slice_in_size;
            let out_start = i * slice_out_size;
            plan_2d.execute_c2r(
                &temp[in_start..in_start + slice_in_size],
                &mut output[out_start..out_start + slice_out_size],
            );
        }
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
    /// Get kind (crate-internal).
    #[must_use]
    pub(crate) fn plan_kind(&self) -> RealPlanKind {
        self.kind
    }
}
