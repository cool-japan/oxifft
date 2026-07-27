//! Real FFT plan types (1D, 2D, 3D).
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#![allow(clippy::items_after_statements)] // reason: SplitRS-generated code places type defs and constants after use statements

use crate::api::{Direction, Flags};
use crate::dft::problem::Sign;
use crate::kernel::{Complex, Float};
use crate::prelude::*;

use super::types::{Plan, RealPlanKind};

/// Apply a flag-aware complex FFT of length `n` along the outermost axis.
///
/// The source is treated as `n` rows of `stride` columns (row-major:
/// `src[row * stride + col]`); each of the `stride` columns is an independent
/// length-`n` complex FFT written to the same position in `dst`. Column
/// selection honours the caller's planning `flags` via [`Plan::dft_1d`] (so
/// `MEASURE`/`PATIENT`/wisdom influence the inter-dimension transform, matching
/// the flag-aware last-axis R2C/C2R solvers), falling back to the
/// always-applicable [`GenericSolver`](crate::dft::solvers::GenericSolver) only
/// if the planner cannot build a plan for `n`.
///
/// [`Plan::execute`] uses the same unnormalized sign convention as
/// `GenericSolver::execute`, so this is a drop-in replacement that never
/// changes numerical results — only which algorithm runs.
fn transform_outer_columns<T: Float>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    n: usize,
    stride: usize,
    direction: Direction,
    flags: Flags,
) {
    use crate::dft::solvers::GenericSolver;
    let plan = Plan::<T>::dft_1d(n, direction, flags);
    let sign = match direction {
        Direction::Forward => Sign::Forward,
        Direction::Backward => Sign::Backward,
    };
    let fallback = if plan.is_none() {
        Some(GenericSolver::<T>::new(n))
    } else {
        None
    };
    let mut col_in = vec![Complex::zero(); n];
    let mut col_out = vec![Complex::zero(); n];
    for col in 0..stride {
        for row in 0..n {
            col_in[row] = src[row * stride + col];
        }
        if let Some(ref p) = plan {
            p.execute(&col_in, &mut col_out);
        } else if let Some(ref g) = fallback {
            g.execute(&col_in, &mut col_out, sign);
        }
        for row in 0..n {
            dst[row * stride + col] = col_out[row];
        }
    }
}

/// A plan for executing real FFT transforms.
///
/// Real FFTs are more efficient than complex FFTs for real-valued input,
/// producing only the non-redundant half of the spectrum.
pub struct RealPlan<T: Float> {
    /// Transform size (number of real values)
    n: usize,
    /// Transform kind
    kind: RealPlanKind,
    /// Planning flags, forwarded to the internal complex sub-transform so that
    /// `Flags::MEASURE`/`PATIENT`/`EXHAUSTIVE` (and wisdom caching) apply.
    flags: Flags,
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
    pub fn r2c_1d(n: usize, flags: Flags) -> Option<Self> {
        if n == 0 {
            return None;
        }
        Some(Self {
            n,
            kind: RealPlanKind::R2C,
            flags,
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
    pub fn c2r_1d(n: usize, flags: Flags) -> Option<Self> {
        if n == 0 {
            return None;
        }
        Some(Self {
            n,
            kind: RealPlanKind::C2R,
            flags,
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
        R2cSolver::new_with_flags(self.n, self.flags).execute(input, output);
    }
    /// Execute the C2R plan **with** normalization.
    ///
    /// The output is divided by `n`, so `r2c` followed by `c2r` recovers the
    /// original signal exactly: `execute_c2r(execute_r2c(x)) == x`.
    ///
    /// # Normalization convention
    /// OxiFFT's [`execute_c2r`](Self::execute_c2r) is **normalized** across every
    /// dimensionality (`RealPlan`, [`RealPlan2D`], [`RealPlan3D`], `RealPlanND`),
    /// so a round trip is the identity. This deliberately differs from FFTW,
    /// whose `c2r` transforms are unnormalized (the caller must divide by `N`).
    /// Use [`execute_c2r_unnormalized`](Self::execute_c2r_unnormalized) for the
    /// raw, FFTW-style result.
    ///
    /// # Panics
    /// Panics if the plan is not C2R or buffer sizes don't match.
    pub fn execute_c2r(&self, input: &[Complex<T>], output: &mut [T]) {
        use crate::rdft::solvers::C2rSolver;
        assert_eq!(self.kind, RealPlanKind::C2R, "Plan must be C2R");
        assert_eq!(input.len(), self.complex_size(), "Input size must be n/2+1");
        assert_eq!(output.len(), self.n, "Output size must match plan size");
        C2rSolver::new_with_flags(self.n, self.flags).execute_normalized(input, output);
    }
    /// Execute the C2R plan **without** normalization (FFTW convention).
    ///
    /// The result is scaled by `n` relative to [`execute_c2r`](Self::execute_c2r).
    ///
    /// # Panics
    /// Panics if the plan is not C2R or buffer sizes don't match.
    pub fn execute_c2r_unnormalized(&self, input: &[Complex<T>], output: &mut [T]) {
        use crate::rdft::solvers::C2rSolver;
        assert_eq!(self.kind, RealPlanKind::C2R, "Plan must be C2R");
        assert_eq!(input.len(), self.complex_size(), "Input size must be n/2+1");
        assert_eq!(output.len(), self.n, "Output size must match plan size");
        C2rSolver::new_with_flags(self.n, self.flags).execute(input, output);
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
    /// Planning flags, forwarded to the internal 1D real sub-transforms.
    flags: Flags,
    _marker: core::marker::PhantomData<T>,
}
impl<T: Float> RealPlan2D<T> {
    /// Create a 2D R2C plan.
    #[must_use]
    pub fn r2c(n0: usize, n1: usize, flags: Flags) -> Option<Self> {
        if n0 == 0 || n1 == 0 {
            return None;
        }
        Some(Self {
            n0,
            n1,
            kind: RealPlanKind::R2C,
            flags,
            _marker: core::marker::PhantomData,
        })
    }
    /// Create a 2D C2R plan.
    #[must_use]
    pub fn c2r(n0: usize, n1: usize, flags: Flags) -> Option<Self> {
        if n0 == 0 || n1 == 0 {
            return None;
        }
        Some(Self {
            n0,
            n1,
            kind: RealPlanKind::C2R,
            flags,
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
        let r2c_solver = R2cSolver::new_with_flags(self.n1, self.flags);
        let mut temp = vec![Complex::zero(); self.n0 * out_cols];
        for row in 0..self.n0 {
            let in_start = row * self.n1;
            let out_start = row * out_cols;
            r2c_solver.execute(
                &input[in_start..in_start + self.n1],
                &mut temp[out_start..out_start + out_cols],
            );
        }
        transform_outer_columns(
            &temp,
            output,
            self.n0,
            out_cols,
            Direction::Forward,
            self.flags,
        );
    }
    /// Execute the 2D C2R transform **with** normalization.
    ///
    /// The output is divided by `n0 * n1`, so a 2D `r2c` -> `c2r` round trip is
    /// the identity. See [`RealPlan::execute_c2r`] for the crate-wide
    /// normalization convention (it deliberately differs from FFTW). Use
    /// [`execute_c2r_unnormalized`](Self::execute_c2r_unnormalized) for the raw,
    /// FFTW-style result.
    pub fn execute_c2r(&self, input: &[Complex<T>], output: &mut [T]) {
        self.c2r_into_raw(input, output);
        let scale = T::one() / T::from_usize(self.n0 * self.n1);
        for x in output.iter_mut() {
            *x = *x * scale;
        }
    }
    /// Execute the 2D C2R transform **without** normalization (FFTW convention).
    ///
    /// The result is scaled by `n0 * n1` relative to
    /// [`execute_c2r`](Self::execute_c2r).
    pub fn execute_c2r_unnormalized(&self, input: &[Complex<T>], output: &mut [T]) {
        self.c2r_into_raw(input, output);
    }
    /// Raw (unnormalized) 2D C2R pipeline, shared by the normalized and
    /// unnormalized public entry points and reused by [`RealPlan3D`].
    pub(crate) fn c2r_into_raw(&self, input: &[Complex<T>], output: &mut [T]) {
        assert_eq!(self.kind, RealPlanKind::C2R);
        let expected_in = self.n0 * (self.n1 / 2 + 1);
        let expected_out = self.n0 * self.n1;
        assert_eq!(input.len(), expected_in);
        assert_eq!(output.len(), expected_out);
        let out_cols = self.n1 / 2 + 1;
        let mut temp = vec![Complex::zero(); self.n0 * out_cols];
        transform_outer_columns(
            input,
            &mut temp,
            self.n0,
            out_cols,
            Direction::Backward,
            self.flags,
        );
        use crate::rdft::solvers::C2rSolver;
        let c2r_solver = C2rSolver::new_with_flags(self.n1, self.flags);
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
    /// Planning flags, forwarded to the internal 2D/1D real sub-transforms.
    flags: Flags,
    _marker: core::marker::PhantomData<T>,
}
impl<T: Float> RealPlan3D<T> {
    /// Create a 3D R2C plan.
    #[must_use]
    pub fn r2c(n0: usize, n1: usize, n2: usize, flags: Flags) -> Option<Self> {
        if n0 == 0 || n1 == 0 || n2 == 0 {
            return None;
        }
        Some(Self {
            n0,
            n1,
            n2,
            kind: RealPlanKind::R2C,
            flags,
            _marker: core::marker::PhantomData,
        })
    }
    /// Create a 3D C2R plan.
    #[must_use]
    pub fn c2r(n0: usize, n1: usize, n2: usize, flags: Flags) -> Option<Self> {
        if n0 == 0 || n1 == 0 || n2 == 0 {
            return None;
        }
        Some(Self {
            n0,
            n1,
            n2,
            kind: RealPlanKind::C2R,
            flags,
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
        let plan_2d = RealPlan2D::<T>::r2c(self.n1, self.n2, self.flags)
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
        // The (j, k) index pair enumerates `n1 * out_last == slice_out_size`
        // contiguous columns, each an independent length-`n0` FFT.
        transform_outer_columns(
            &temp,
            output,
            self.n0,
            slice_out_size,
            Direction::Forward,
            self.flags,
        );
    }
    /// Execute the 3D C2R transform **with** normalization.
    ///
    /// The output is divided by `n0 * n1 * n2`, so a 3D `r2c` -> `c2r` round trip
    /// is the identity. See [`RealPlan::execute_c2r`] for the crate-wide
    /// normalization convention (it deliberately differs from FFTW). Use
    /// [`execute_c2r_unnormalized`](Self::execute_c2r_unnormalized) for the raw,
    /// FFTW-style result.
    pub fn execute_c2r(&self, input: &[Complex<T>], output: &mut [T]) {
        self.c2r_into_raw(input, output);
        let scale = T::one() / T::from_usize(self.n0 * self.n1 * self.n2);
        for x in output.iter_mut() {
            *x = *x * scale;
        }
    }
    /// Execute the 3D C2R transform **without** normalization (FFTW convention).
    ///
    /// The result is scaled by `n0 * n1 * n2` relative to
    /// [`execute_c2r`](Self::execute_c2r).
    pub fn execute_c2r_unnormalized(&self, input: &[Complex<T>], output: &mut [T]) {
        self.c2r_into_raw(input, output);
    }
    /// Raw (unnormalized) 3D C2R pipeline, shared by the normalized and
    /// unnormalized entry points and reused by `RealPlanND`.
    pub(crate) fn c2r_into_raw(&self, input: &[Complex<T>], output: &mut [T]) {
        assert_eq!(self.kind, RealPlanKind::C2R);
        let expected_in = self.n0 * self.n1 * (self.n2 / 2 + 1);
        let expected_out = self.n0 * self.n1 * self.n2;
        assert_eq!(input.len(), expected_in);
        assert_eq!(output.len(), expected_out);
        let out_last = self.n2 / 2 + 1;
        let slice_in_size = self.n1 * out_last;
        let slice_out_size = self.n1 * self.n2;
        let mut temp = vec![Complex::zero(); self.n0 * slice_in_size];
        // Each (j, k) column (there are `n1 * out_last == slice_in_size` of
        // them) is an independent length-`n0` inverse FFT.
        transform_outer_columns(
            input,
            &mut temp,
            self.n0,
            slice_in_size,
            Direction::Backward,
            self.flags,
        );
        // Apply the 2D C2R along the (n1, n2) planes using the *raw*
        // (unnormalized) pipeline; the single `1/(n0·n1·n2)` normalization for
        // the whole 3D transform is applied once by the public `execute_c2r`.
        let plan_2d = RealPlan2D::<T>::c2r(self.n1, self.n2, self.flags)
            .expect("Failed to create internal 2D C2R plan");
        for i in 0..self.n0 {
            let in_start = i * slice_in_size;
            let out_start = i * slice_out_size;
            plan_2d.c2r_into_raw(
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
