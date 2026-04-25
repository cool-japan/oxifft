//! N-dimensional FFT plan types.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#![allow(clippy::items_after_statements)] // reason: SplitRS-generated code places type defs and constants after use statements

use crate::api::{Direction, Flags};
use crate::dft::problem::Sign;
use crate::kernel::{Complex, Float};
use crate::prelude::*;

use super::types::{Plan, RealPlanKind};
use super::types_real::{RealPlan2D, RealPlan3D};

/// A plan for executing N-dimensional FFT transforms.
///
/// Generalizes Plan2D and Plan3D to arbitrary dimensions using
/// successive 1D FFTs along each dimension.
pub struct PlanND<T: Float> {
    /// Dimensions in row-major order (slowest varying first).
    dims: Vec<usize>,
    /// Total size (product of all dimensions).
    total_size: usize,
    /// Strides for each dimension.
    strides: Vec<usize>,
    /// Transform direction.
    direction: Direction,
    /// 1D plans for each dimension.
    plans: Vec<Plan<T>>,
}
impl<T: Float> PlanND<T> {
    /// Create an N-dimensional complex-to-complex DFT plan.
    ///
    /// # Arguments
    /// * `dims` - Array of dimension sizes in row-major order (slowest varying first)
    /// * `direction` - Forward or Backward transform
    /// * `flags` - Planning flags
    ///
    /// # Returns
    /// A plan for row-major N-dimensional data.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxifft::{Complex, Direction, Flags, PlanND};
    ///
    /// // 4×4 2D FFT via PlanND
    /// let plan = PlanND::<f64>::new(&[4, 4], Direction::Forward, Flags::ESTIMATE)
    ///     .expect("plan construction failed");
    /// assert_eq!(plan.size(), 16);
    /// // All-ones input: DC bin = total element count = 16
    /// let input = vec![Complex::<f64>::new(1.0, 0.0); 16];
    /// let mut output = vec![Complex::<f64>::zero(); 16];
    /// plan.execute(&input, &mut output);
    /// assert!((output[0].re - 16.0_f64).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn new(dims: &[usize], direction: Direction, flags: Flags) -> Option<Self> {
        if dims.is_empty() {
            return None;
        }
        let mut total_size: usize = 1;
        for &d in dims {
            total_size = total_size.checked_mul(d)?;
        }
        let mut strides = vec![1; dims.len()];
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        let mut plans = Vec::with_capacity(dims.len());
        for &dim in dims {
            plans.push(Plan::dft_1d(dim, direction, flags)?);
        }
        Some(Self {
            dims: dims.to_vec(),
            total_size,
            strides,
            direction,
            plans,
        })
    }
    /// Get the number of dimensions.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }
    /// Get the dimensions.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
    /// Get the total size (product of all dimensions).
    #[must_use]
    pub fn size(&self) -> usize {
        self.total_size
    }
    /// Get the transform direction.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.direction
    }
    /// Execute the N-dimensional FFT on the given input/output buffers.
    ///
    /// Data is in row-major order (last dimension varies fastest).
    ///
    /// # Panics
    /// Panics if buffer sizes don't match the total size.
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        assert_eq!(
            input.len(),
            self.total_size,
            "Input size must match total size"
        );
        assert_eq!(
            output.len(),
            self.total_size,
            "Output size must match total size"
        );
        if self.total_size == 0 {
            return;
        }
        let mut current = input.to_vec();
        let mut next = vec![Complex::zero(); self.total_size];
        for dim_idx in (0..self.dims.len()).rev() {
            self.transform_along_dimension(&current, &mut next, dim_idx);
            core::mem::swap(&mut current, &mut next);
        }
        output.copy_from_slice(&current);
    }
    /// Execute the N-dimensional FFT in-place.
    ///
    /// # Panics
    /// Panics if buffer size doesn't match the total size.
    pub fn execute_inplace(&self, data: &mut [Complex<T>]) {
        assert_eq!(
            data.len(),
            self.total_size,
            "Data size must match total size"
        );
        if self.total_size == 0 {
            return;
        }
        let mut temp = vec![Complex::zero(); self.total_size];
        for dim_idx in (0..self.dims.len()).rev() {
            self.transform_along_dimension(data, &mut temp, dim_idx);
            data.copy_from_slice(&temp);
        }
    }
    /// Transform along a single dimension.
    ///
    /// For each "fiber" along dimension `dim_idx`, extract it, transform it,
    /// and place it back.
    fn transform_along_dimension(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        dim_idx: usize,
    ) {
        let dim_size = self.dims[dim_idx];
        let stride = self.strides[dim_idx];
        let num_fibers = self.total_size / dim_size;
        let mut fiber_in = vec![Complex::zero(); dim_size];
        let mut fiber_out = vec![Complex::zero(); dim_size];
        for fiber_idx in 0..num_fibers {
            let start_idx = self.fiber_start_index(fiber_idx, dim_idx);
            for i in 0..dim_size {
                fiber_in[i] = input[start_idx + i * stride];
            }
            self.plans[dim_idx].execute(&fiber_in, &mut fiber_out);
            for i in 0..dim_size {
                output[start_idx + i * stride] = fiber_out[i];
            }
        }
    }
    /// Compute the starting index for a fiber along dimension `dim_idx`.
    ///
    /// Given a linear fiber index, compute the base index in the flat array.
    fn fiber_start_index(&self, fiber_idx: usize, dim_idx: usize) -> usize {
        let mut idx = 0;
        let mut remaining = fiber_idx;
        for d in 0..self.dims.len() {
            if d == dim_idx {
                continue;
            }
            let below_count = self.fiber_below_count(d, dim_idx);
            let coord = remaining / below_count;
            remaining %= below_count;
            idx += coord * self.strides[d];
        }
        idx
    }
    /// Count how many fiber coordinates are "below" dimension d (excluding dim_idx).
    fn fiber_below_count(&self, d: usize, dim_idx: usize) -> usize {
        let mut count = 1;
        for i in (d + 1)..self.dims.len() {
            if i != dim_idx {
                count *= self.dims[i];
            }
        }
        count
    }
}
/// A plan for N-dimensional real-to-complex and complex-to-real transforms.
pub struct RealPlanND<T: Float> {
    dims: Vec<usize>,
    kind: RealPlanKind,
    _marker: core::marker::PhantomData<T>,
}
impl<T: Float> RealPlanND<T> {
    /// Create an N-dimensional R2C plan.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxifft::{Complex, Flags, RealPlanND};
    ///
    /// // 1D R2C transform via RealPlanND
    /// let plan = RealPlanND::<f64>::r2c(&[8], Flags::ESTIMATE)
    ///     .expect("plan construction failed");
    /// let input = vec![1.0_f64; 8];
    /// // Output size = last_dim / 2 + 1 = 5
    /// let mut output = vec![Complex::<f64>::zero(); 5];
    /// plan.execute_r2c(&input, &mut output);
    /// // DC bin = sum of all elements = 8
    /// assert!((output[0].re - 8.0_f64).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn r2c(dims: &[usize], _flags: Flags) -> Option<Self> {
        if dims.is_empty() || dims.contains(&0) {
            return None;
        }
        Some(Self {
            dims: dims.to_vec(),
            kind: RealPlanKind::R2C,
            _marker: core::marker::PhantomData,
        })
    }
    /// Create an N-dimensional C2R plan.
    #[must_use]
    pub fn c2r(dims: &[usize], _flags: Flags) -> Option<Self> {
        if dims.is_empty() || dims.contains(&0) {
            return None;
        }
        Some(Self {
            dims: dims.to_vec(),
            kind: RealPlanKind::C2R,
            _marker: core::marker::PhantomData,
        })
    }
    /// Execute R2C transform.
    pub fn execute_r2c(&self, input: &[T], output: &mut [Complex<T>]) {
        assert_eq!(self.kind, RealPlanKind::R2C);
        let expected_in: usize = self.dims.iter().product();
        let last = *self
            .dims
            .last()
            .expect("RealPlanND dimensions cannot be empty");
        let prefix: usize = self.dims[..self.dims.len() - 1].iter().product();
        let prefix = prefix.max(1);
        let expected_out = prefix * (last / 2 + 1);
        assert_eq!(input.len(), expected_in);
        assert_eq!(output.len(), expected_out);
        match self.dims.len() {
            1 => {
                use crate::rdft::solvers::R2cSolver;
                let solver = R2cSolver::new(last);
                solver.execute(input, output);
            }
            2 => {
                let plan = RealPlan2D::<T>::r2c(self.dims[0], self.dims[1], Flags::ESTIMATE)
                    .expect("Failed to create internal 2D R2C plan");
                plan.execute_r2c(input, output);
            }
            3 => {
                let plan =
                    RealPlan3D::<T>::r2c(self.dims[0], self.dims[1], self.dims[2], Flags::ESTIMATE)
                        .expect("Failed to create internal 3D R2C plan");
                plan.execute_r2c(input, output);
            }
            _ => {
                let out_last = last / 2 + 1;
                let inner_size = last;
                use crate::rdft::solvers::R2cSolver;
                let r2c_solver = R2cSolver::new(last);
                let mut temp = vec![Complex::zero(); prefix * out_last];
                for row in 0..prefix {
                    let in_start = row * inner_size;
                    let out_start = row * out_last;
                    r2c_solver.execute(
                        &input[in_start..in_start + inner_size],
                        &mut temp[out_start..out_start + out_last],
                    );
                }
                use crate::dft::solvers::GenericSolver;
                let remaining_dims = &self.dims[..self.dims.len() - 1];
                for (dim_idx, &dim_size) in remaining_dims.iter().enumerate().rev() {
                    let solver = GenericSolver::new(dim_size);
                    let mut col_in = vec![Complex::zero(); dim_size];
                    let mut col_out = vec![Complex::zero(); dim_size];
                    let inner_stride: usize = self.dims[dim_idx + 1..]
                        .iter()
                        .map(|&d| {
                            if dim_idx == self.dims.len() - 2 {
                                out_last
                            } else {
                                d
                            }
                        })
                        .product();
                    let outer_count: usize = self.dims[..dim_idx].iter().product();
                    let outer_count = outer_count.max(1);
                    for outer in 0..outer_count {
                        for inner in 0..inner_stride {
                            for k in 0..dim_size {
                                let idx =
                                    outer * (dim_size * inner_stride) + k * inner_stride + inner;
                                col_in[k] = temp[idx];
                            }
                            solver.execute(&col_in, &mut col_out, Sign::Forward);
                            for k in 0..dim_size {
                                let idx =
                                    outer * (dim_size * inner_stride) + k * inner_stride + inner;
                                temp[idx] = col_out[k];
                            }
                        }
                    }
                }
                output.copy_from_slice(&temp);
            }
        }
    }
    /// Execute C2R transform.
    pub fn execute_c2r(&self, input: &[Complex<T>], output: &mut [T]) {
        assert_eq!(self.kind, RealPlanKind::C2R);
        let expected_out: usize = self.dims.iter().product();
        let last = *self
            .dims
            .last()
            .expect("RealPlanND dimensions cannot be empty");
        let prefix: usize = self.dims[..self.dims.len() - 1].iter().product();
        let prefix = prefix.max(1);
        let expected_in = prefix * (last / 2 + 1);
        assert_eq!(input.len(), expected_in);
        assert_eq!(output.len(), expected_out);
        match self.dims.len() {
            1 => {
                use crate::rdft::solvers::C2rSolver;
                let solver = C2rSolver::new(last);
                solver.execute(input, output);
            }
            2 => {
                let plan = RealPlan2D::<T>::c2r(self.dims[0], self.dims[1], Flags::ESTIMATE)
                    .expect("Failed to create internal 2D C2R plan");
                plan.execute_c2r(input, output);
            }
            3 => {
                let plan =
                    RealPlan3D::<T>::c2r(self.dims[0], self.dims[1], self.dims[2], Flags::ESTIMATE)
                        .expect("Failed to create internal 3D C2R plan");
                plan.execute_c2r(input, output);
            }
            _ => {
                let out_last = last / 2 + 1;
                let mut temp: Vec<Complex<T>> = input.to_vec();
                use crate::dft::solvers::GenericSolver;
                let remaining_dims = &self.dims[..self.dims.len() - 1];
                for (dim_idx, &dim_size) in remaining_dims.iter().enumerate() {
                    let solver = GenericSolver::new(dim_size);
                    let mut col_in = vec![Complex::zero(); dim_size];
                    let mut col_out = vec![Complex::zero(); dim_size];
                    let inner_stride: usize = self.dims[dim_idx + 1..]
                        .iter()
                        .map(|&d| {
                            if dim_idx == self.dims.len() - 2 {
                                out_last
                            } else {
                                d
                            }
                        })
                        .product();
                    let outer_count: usize = self.dims[..dim_idx].iter().product();
                    let outer_count = outer_count.max(1);
                    for outer in 0..outer_count {
                        for inner in 0..inner_stride {
                            for k in 0..dim_size {
                                let idx =
                                    outer * (dim_size * inner_stride) + k * inner_stride + inner;
                                col_in[k] = temp[idx];
                            }
                            solver.execute(&col_in, &mut col_out, Sign::Backward);
                            for k in 0..dim_size {
                                let idx =
                                    outer * (dim_size * inner_stride) + k * inner_stride + inner;
                                temp[idx] = col_out[k];
                            }
                        }
                    }
                }
                use crate::rdft::solvers::C2rSolver;
                let c2r_solver = C2rSolver::new(last);
                for row in 0..prefix {
                    let in_start = row * out_last;
                    let out_start = row * last;
                    c2r_solver.execute(
                        &temp[in_start..in_start + out_last],
                        &mut output[out_start..out_start + last],
                    );
                }
            }
        }
    }
    /// Get dims (crate-internal).
    #[must_use]
    pub(crate) fn dims(&self) -> &[usize] {
        &self.dims
    }
    /// Get kind (crate-internal).
    #[must_use]
    pub(crate) fn plan_kind(&self) -> RealPlanKind {
        self.kind
    }
}
