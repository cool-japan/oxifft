//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#![allow(clippy::items_after_statements)]
#![allow(clippy::manual_contains)]

use crate::api::{Direction, Flags};
use crate::dft::problem::Sign;
use crate::dft::solvers::{
    BluesteinSolver, CooleyTukeySolver, CtVariant, DirectSolver, GenericSolver, NopSolver,
    StockhamSolver,
};
use crate::kernel::{Complex, Float, Tensor};
use crate::prelude::*;
use crate::rdft::solvers::R2rKind;

/// A plan for N-dimensional real-to-complex and complex-to-real transforms.
pub struct RealPlanND<T: Float> {
    dims: Vec<usize>,
    kind: RealPlanKind,
    _marker: core::marker::PhantomData<T>,
}
impl<T: Float> RealPlanND<T> {
    /// Create an N-dimensional R2C plan.
    #[must_use]
    pub fn r2c(dims: &[usize], _flags: Flags) -> Option<Self> {
        if dims.is_empty() || dims.iter().any(|&d| d == 0) {
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
        if dims.is_empty() || dims.iter().any(|&d| d == 0) {
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
}
/// A plan for executing 3D FFT transforms.
///
/// Implements layered decomposition: apply 2D FFT to each xy-plane,
/// then 1D FFT along z-axis.
pub struct Plan3D<T: Float> {
    /// Dimensions (z, y, x) in row-major order
    n0: usize,
    n1: usize,
    n2: usize,
    /// Transform direction
    direction: Direction,
    /// 2D plan for each xy-plane
    plane_plan: Plan2D<T>,
    /// 1D plan for z-axis
    z_plan: Plan<T>,
}
impl<T: Float> Plan3D<T> {
    /// Create a 3D complex-to-complex DFT plan.
    ///
    /// # Arguments
    /// * `n0` - Size along first axis (z/depth)
    /// * `n1` - Size along second axis (y/height)
    /// * `n2` - Size along third axis (x/width)
    /// * `direction` - Forward or Backward transform
    /// * `flags` - Planning flags
    ///
    /// # Returns
    /// A plan for row-major 3D data of size n0 × n1 × n2.
    #[must_use]
    pub fn new(
        n0: usize,
        n1: usize,
        n2: usize,
        direction: Direction,
        flags: Flags,
    ) -> Option<Self> {
        let plane_plan = Plan2D::new(n1, n2, direction, flags)?;
        let z_plan = Plan::dft_1d(n0, direction, flags)?;
        Some(Self {
            n0,
            n1,
            n2,
            direction,
            plane_plan,
            z_plan,
        })
    }
    /// Get the total size (n0 × n1 × n2).
    #[must_use]
    pub fn size(&self) -> usize {
        self.n0 * self.n1 * self.n2
    }
    /// Get the transform direction.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.direction
    }
    /// Execute the 3D FFT on the given input/output buffers.
    ///
    /// Data is row-major: element at (i, j, k) is at index i*n1*n2 + j*n2 + k.
    ///
    /// # Panics
    /// Panics if buffer sizes don't match n0 × n1 × n2.
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        let total = self.n0 * self.n1 * self.n2;
        assert_eq!(input.len(), total, "Input size must match n0 × n1 × n2");
        assert_eq!(output.len(), total, "Output size must match n0 × n1 × n2");
        if total == 0 {
            return;
        }
        let plane_size = self.n1 * self.n2;
        let mut temp = vec![Complex::zero(); total];
        for i in 0..self.n0 {
            let plane_start = i * plane_size;
            let plane_end = plane_start + plane_size;
            self.plane_plan.execute(
                &input[plane_start..plane_end],
                &mut temp[plane_start..plane_end],
            );
        }
        let mut z_col = vec![Complex::zero(); self.n0];
        let mut z_out = vec![Complex::zero(); self.n0];
        for j in 0..self.n1 {
            for k in 0..self.n2 {
                for i in 0..self.n0 {
                    z_col[i] = temp[i * plane_size + j * self.n2 + k];
                }
                self.z_plan.execute(&z_col, &mut z_out);
                for i in 0..self.n0 {
                    output[i * plane_size + j * self.n2 + k] = z_out[i];
                }
            }
        }
    }
    /// Execute the 3D FFT in-place.
    ///
    /// # Panics
    /// Panics if buffer size doesn't match n0 × n1 × n2.
    pub fn execute_inplace(&self, data: &mut [Complex<T>]) {
        let total = self.n0 * self.n1 * self.n2;
        assert_eq!(data.len(), total, "Data size must match n0 × n1 × n2");
        if total == 0 {
            return;
        }
        let plane_size = self.n1 * self.n2;
        for i in 0..self.n0 {
            let plane_start = i * plane_size;
            let plane_end = plane_start + plane_size;
            self.plane_plan
                .execute_inplace(&mut data[plane_start..plane_end]);
        }
        let mut z_col = vec![Complex::zero(); self.n0];
        for j in 0..self.n1 {
            for k in 0..self.n2 {
                for i in 0..self.n0 {
                    z_col[i] = data[i * plane_size + j * self.n2 + k];
                }
                self.z_plan.execute_inplace(&mut z_col);
                for i in 0..self.n0 {
                    data[i * plane_size + j * self.n2 + k] = z_col[i];
                }
            }
        }
    }
}
/// Transform kind for real FFT plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealPlanKind {
    /// Real to Complex (forward)
    R2C,
    /// Complex to Real (backward/inverse)
    C2R,
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
        if dims.is_empty() || dims.iter().any(|&d| d == 0) {
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
    /// # Example
    /// ```ignore
    /// // 4D FFT with dimensions [2, 3, 4, 5]
    /// let plan = PlanND::new(&[2, 3, 4, 5], Direction::Forward, Flags::ESTIMATE)?;
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
}
/// Algorithm selection for the plan.
#[allow(dead_code)]
enum Algorithm<T: Float> {
    /// No-op for size 0 or 1
    Nop,
    /// Direct O(n²) computation (only for very small sizes where overhead matters)
    Direct,
    /// Cooley-Tukey radix-2 FFT
    CooleyTukey(CtVariant),
    /// Stockham auto-sort FFT (avoids bit-reversal, good for large sizes)
    Stockham,
    /// Specialized composite codelets (12, 24, 36, 48, 60, 72, 96, 100)
    Composite(usize),
    /// Generic mixed-radix for composite sizes
    Generic(Box<GenericSolver<T>>),
    /// Bluestein's algorithm for arbitrary sizes (fallback for primes)
    Bluestein(Box<BluesteinSolver<T>>),
}
/// Guru interface for maximum flexibility.
///
/// Allows specifying arbitrary dimensions, strides, and batch parameters.
/// This is the most general FFT interface, supporting:
/// - Arbitrary transform dimensions (1D, 2D, 3D, ND)
/// - Arbitrary strides for input and output
/// - Batched transforms with arbitrary batch dimensions
/// - In-place or out-of-place operation
///
/// # Example
///
/// ```ignore
/// use oxifft::{Complex, Direction, Flags, IoDim, Tensor};
/// use oxifft::api::GuruPlan;
///
/// // Create a batched 1D FFT: 10 transforms of size 256
/// let dims = Tensor::new(vec![IoDim::contiguous(256)]);
/// let howmany = Tensor::new(vec![IoDim::new(10, 256, 256)]); // 10 batches, stride=256
///
/// let mut input = vec![Complex::new(0.0, 0.0); 2560];
/// let mut output = vec![Complex::new(0.0, 0.0); 2560];
///
/// let plan = GuruPlan::<f64>::dft(
///     &dims,
///     &howmany,
///     Direction::Forward,
///     Flags::ESTIMATE,
/// ).unwrap();
///
/// plan.execute(&input, &mut output);
/// ```
pub struct GuruPlan<T: Float> {
    /// Transform dimensions
    dims: Tensor,
    /// Batch dimensions (howmany)
    howmany: Tensor,
    /// Transform direction
    direction: Direction,
    /// 1D plans for each transform dimension (innermost first)
    plans: Vec<Plan<T>>,
}
impl<T: Float> GuruPlan<T> {
    /// Create a guru DFT plan with full control over dimensions and strides.
    ///
    /// # Arguments
    /// * `dims` - Transform dimensions with input/output strides
    /// * `howmany` - Batch dimensions (vector rank). Can be empty for single transform.
    /// * `direction` - Forward or Backward
    /// * `flags` - Planning flags
    ///
    /// # Returns
    /// A plan that can be executed on buffers with the specified layout.
    /// Returns `None` if any dimension has size 0.
    ///
    /// # Memory Layout
    /// - `dims` specifies the transform dimensions (innermost dimensions transform)
    /// - `howmany` specifies the batch dimensions (outermost dimensions iterate)
    /// - Strides can be different for input and output (for in-place: use same strides)
    pub fn dft(
        dims: &Tensor,
        howmany: &Tensor,
        direction: Direction,
        flags: Flags,
    ) -> Option<Self> {
        if dims.is_empty() {
            return None;
        }
        for dim in &dims.dims {
            if dim.n == 0 {
                return None;
            }
        }
        for dim in &howmany.dims {
            if dim.n == 0 {
                return None;
            }
        }
        let mut plans = Vec::with_capacity(dims.rank());
        for dim in &dims.dims {
            let plan = Plan::dft_1d(dim.n, direction, flags)?;
            plans.push(plan);
        }
        Some(Self {
            dims: dims.clone(),
            howmany: howmany.clone(),
            direction,
            plans,
        })
    }
    /// Get the transform dimensions.
    #[must_use]
    pub fn dims(&self) -> &Tensor {
        &self.dims
    }
    /// Get the batch dimensions.
    #[must_use]
    pub fn howmany(&self) -> &Tensor {
        &self.howmany
    }
    /// Get the transform direction.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.direction
    }
    /// Get the total number of elements in one transform.
    #[must_use]
    pub fn transform_size(&self) -> usize {
        self.dims.total_size()
    }
    /// Get the total number of transforms (batch count).
    #[must_use]
    pub fn batch_count(&self) -> usize {
        if self.howmany.is_empty() {
            1
        } else {
            self.howmany.total_size()
        }
    }
    /// Execute the guru plan on the given buffers.
    ///
    /// # Arguments
    /// * `input` - Input buffer with layout matching `dims` and `howmany`
    /// * `output` - Output buffer with layout matching `dims` and `howmany`
    ///
    /// # Panics
    /// Panics if buffers are too small for the specified layout.
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        if self.dims.rank() == 1 && self.howmany.is_empty() {
            let dim = &self.dims.dims[0];
            if dim.is == 1 && dim.os == 1 {
                self.plans[0].execute(input, output);
                return;
            }
        }
        self.execute_batched(input, output);
    }
    /// Execute batched transforms.
    fn execute_batched(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        let batch_count = self.batch_count();
        if batch_count == 1 {
            self.execute_single(input, output, 0, 0);
        } else {
            for batch_idx in 0..batch_count {
                let (in_offset, out_offset) = self.compute_batch_offset(batch_idx);
                self.execute_single(input, output, in_offset, out_offset);
            }
        }
    }
    /// Compute input and output offsets for a given batch index.
    fn compute_batch_offset(&self, batch_idx: usize) -> (isize, isize) {
        if self.howmany.is_empty() {
            return (0, 0);
        }
        let mut in_offset: isize = 0;
        let mut out_offset: isize = 0;
        let mut remaining = batch_idx;
        for dim in self.howmany.dims.iter().rev() {
            let idx = remaining % dim.n;
            remaining /= dim.n;
            in_offset += (idx as isize) * dim.is;
            out_offset += (idx as isize) * dim.os;
        }
        (in_offset, out_offset)
    }
    /// Execute a single transform at the given offsets.
    fn execute_single(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        in_offset: isize,
        out_offset: isize,
    ) {
        if self.dims.rank() == 1 {
            self.execute_1d(input, output, in_offset, out_offset);
        } else {
            self.execute_nd(input, output, in_offset, out_offset);
        }
    }
    /// Execute a 1D transform with arbitrary strides.
    fn execute_1d(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        in_offset: isize,
        out_offset: isize,
    ) {
        let dim = &self.dims.dims[0];
        let n = dim.n;
        let in_stride = dim.is;
        let out_stride = dim.os;
        let input_contiguous: Vec<Complex<T>>;
        let input_slice = if in_stride == 1 && in_offset >= 0 {
            &input[in_offset as usize..in_offset as usize + n]
        } else {
            input_contiguous = (0..n)
                .map(|i| {
                    let idx = in_offset + (i as isize) * in_stride;
                    input[idx as usize]
                })
                .collect();
            &input_contiguous
        };
        let mut temp = vec![Complex::<T>::zero(); n];
        self.plans[0].execute(input_slice, &mut temp);
        for i in 0..n {
            let idx = out_offset + (i as isize) * out_stride;
            output[idx as usize] = temp[i];
        }
    }
    /// Execute an N-dimensional transform using row-column decomposition.
    fn execute_nd(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        in_offset: isize,
        out_offset: isize,
    ) {
        let total_size = self.transform_size();
        let mut work = vec![Complex::<T>::zero(); total_size];
        self.gather_nd(input, &mut work, in_offset);
        for (dim_idx, plan) in self.plans.iter().enumerate() {
            self.apply_1d_along_dimension(&mut work, dim_idx, plan);
        }
        self.scatter_nd(&work, output, out_offset);
    }
    /// Gather N-dimensional data into contiguous buffer.
    fn gather_nd(&self, input: &[Complex<T>], work: &mut [Complex<T>], base_offset: isize) {
        let total = self.transform_size();
        for flat_idx in 0..total {
            let src_offset = self.compute_nd_offset(flat_idx, base_offset, true);
            work[flat_idx] = input[src_offset as usize];
        }
    }
    /// Scatter from contiguous buffer to N-dimensional output.
    fn scatter_nd(&self, work: &[Complex<T>], output: &mut [Complex<T>], base_offset: isize) {
        let total = self.transform_size();
        for flat_idx in 0..total {
            let dst_offset = self.compute_nd_offset(flat_idx, base_offset, false);
            output[dst_offset as usize] = work[flat_idx];
        }
    }
    /// Compute offset for multi-dimensional index.
    fn compute_nd_offset(&self, flat_idx: usize, base_offset: isize, is_input: bool) -> isize {
        let mut offset = base_offset;
        let mut remaining = flat_idx;
        for dim in self.dims.dims.iter().rev() {
            let idx = remaining % dim.n;
            remaining /= dim.n;
            let stride = if is_input { dim.is } else { dim.os };
            offset += (idx as isize) * stride;
        }
        offset
    }
    /// Apply 1D FFT along a specific dimension.
    fn apply_1d_along_dimension(&self, work: &mut [Complex<T>], dim_idx: usize, plan: &Plan<T>) {
        let n = self.dims.dims[dim_idx].n;
        let inner_size: usize = self.dims.dims[dim_idx + 1..].iter().map(|d| d.n).product();
        let inner_size = if inner_size == 0 { 1 } else { inner_size };
        let outer_size: usize = self.dims.dims[..dim_idx].iter().map(|d| d.n).product();
        let outer_size = if outer_size == 0 { 1 } else { outer_size };
        let stride = inner_size;
        let mut temp_in = vec![Complex::<T>::zero(); n];
        let mut temp_out = vec![Complex::<T>::zero(); n];
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base = outer * n * inner_size + inner;
                for i in 0..n {
                    temp_in[i] = work[base + i * stride];
                }
                plan.execute(&temp_in, &mut temp_out);
                for i in 0..n {
                    work[base + i * stride] = temp_out[i];
                }
            }
        }
    }
    /// Execute the plan in-place.
    ///
    /// For in-place execution, input and output strides must be identical.
    pub fn execute_inplace(&self, data: &mut [Complex<T>]) {
        if !self.dims.is_inplace_compatible() {
            panic!("In-place execution requires identical input and output strides");
        }
        let batch_count = self.batch_count();
        if batch_count == 1 {
            self.execute_inplace_single(data, 0);
        } else {
            for batch_idx in 0..batch_count {
                let (offset, _) = self.compute_batch_offset(batch_idx);
                self.execute_inplace_single(data, offset);
            }
        }
    }
    /// Execute a single in-place transform.
    fn execute_inplace_single(&self, data: &mut [Complex<T>], offset: isize) {
        let n = self.transform_size();
        let dim = &self.dims.dims[0];
        if self.dims.rank() == 1 && dim.is == 1 && offset >= 0 {
            let start = offset as usize;
            let end = start + n;
            let mut temp = vec![Complex::<T>::zero(); n];
            self.plans[0].execute(&data[start..end], &mut temp);
            data[start..end].copy_from_slice(&temp);
        } else {
            let mut work = vec![Complex::<T>::zero(); n];
            for (flat_idx, item) in work.iter_mut().enumerate().take(n) {
                let src_offset = self.compute_nd_offset(flat_idx, offset, true);
                *item = data[src_offset as usize];
            }
            let mut result = vec![Complex::<T>::zero(); n];
            if self.dims.rank() == 1 {
                self.plans[0].execute(&work, &mut result);
            } else {
                for (dim_idx, plan) in self.plans.iter().enumerate() {
                    self.apply_1d_along_dimension(&mut work, dim_idx, plan);
                }
                result = work;
            }
            for (flat_idx, &item) in result.iter().enumerate().take(n) {
                let dst_offset = self.compute_nd_offset(flat_idx, offset, false);
                data[dst_offset as usize] = item;
            }
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
        self.plan.n0
    }
    /// Get the number of columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.plan.n1
    }
    /// Get the total size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.plan.size()
    }
    /// Get the transform direction.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.plan.direction
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
}
/// A plan for executing FFT transforms.
///
/// Plans are created once and can be executed multiple times.
/// The planning process may measure different algorithms to find the fastest.
pub struct Plan<T: Float> {
    /// Transform size
    n: usize,
    /// Transform direction
    direction: Direction,
    /// Selected algorithm
    algorithm: Algorithm<T>,
}
impl<T: Float> Plan<T> {
    /// Create a 1D complex-to-complex DFT plan.
    ///
    /// # Arguments
    /// * `n` - Transform size
    /// * `direction` - Forward or Backward transform
    /// * `flags` - Planning flags (ESTIMATE, MEASURE, PATIENT, EXHAUSTIVE)
    ///
    /// # Returns
    /// A plan that can be executed on input/output buffers of size `n`.
    #[must_use]
    pub fn dft_1d(n: usize, direction: Direction, flags: Flags) -> Option<Self> {
        let algorithm = Self::select_algorithm(n, flags);
        Some(Self {
            n,
            direction,
            algorithm,
        })
    }
    /// Create a 2D complex-to-complex DFT plan.
    pub fn dft_2d(_n0: usize, _n1: usize, _direction: Direction, _flags: Flags) -> Option<Self> {
        todo!("Implement dft_2d planning")
    }
    /// Create a 3D complex-to-complex DFT plan.
    pub fn dft_3d(
        _n0: usize,
        _n1: usize,
        _n2: usize,
        _direction: Direction,
        _flags: Flags,
    ) -> Option<Self> {
        todo!("Implement dft_3d planning")
    }
    /// Create a 1D real-to-complex FFT plan.
    pub fn r2c_1d(_n: usize, _flags: Flags) -> Option<Self> {
        todo!("Implement r2c_1d planning")
    }
    /// Create a 1D complex-to-real FFT plan.
    pub fn c2r_1d(_n: usize, _flags: Flags) -> Option<Self> {
        todo!("Implement c2r_1d planning")
    }
    /// Select the best algorithm for the given size.
    fn select_algorithm(n: usize, _flags: Flags) -> Algorithm<T> {
        use crate::dft::codelets::has_composite_codelet;

        if n <= 1 {
            Algorithm::Nop
        } else if CooleyTukeySolver::<T>::applicable(n) {
            // Use DIT with SIMD-accelerated butterflies for all power-of-2 sizes
            // Note: Stockham needs optimization before it can compete with DIT+codelets
            Algorithm::CooleyTukey(CtVariant::Dit)
        } else if has_composite_codelet(n) {
            // Use specialized composite codelets for common sizes (12, 24, 36, 48, 60, 72, 96, 100)
            Algorithm::Composite(n)
        } else if n <= 16 {
            // For small non-power-of-2 sizes without codelets, use direct O(n²)
            Algorithm::Direct
        } else if GenericSolver::<T>::applicable(n) {
            Algorithm::Generic(Box::new(GenericSolver::new(n)))
        } else {
            Algorithm::Bluestein(Box::new(BluesteinSolver::new(n)))
        }
    }
    /// Get the transform size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.n
    }
    /// Get the transform direction.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.direction
    }
    /// Execute the plan on the given input/output buffers.
    ///
    /// # Panics
    /// Panics if input or output buffer sizes don't match the plan size.
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        use crate::dft::codelets::execute_composite_codelet;

        assert_eq!(input.len(), self.n, "Input size must match plan size");
        assert_eq!(output.len(), self.n, "Output size must match plan size");
        let sign = match self.direction {
            Direction::Forward => Sign::Forward,
            Direction::Backward => Sign::Backward,
        };
        match &self.algorithm {
            Algorithm::Nop => {
                NopSolver::new().execute(input, output);
            }
            Algorithm::Direct => {
                DirectSolver::new().execute(input, output, sign);
            }
            Algorithm::CooleyTukey(variant) => {
                CooleyTukeySolver::new(*variant).execute(input, output, sign);
            }
            Algorithm::Stockham => {
                StockhamSolver::new().execute(input, output, sign);
            }
            Algorithm::Composite(n) => {
                output.copy_from_slice(input);
                let sign_int = if sign == Sign::Forward { -1 } else { 1 };
                execute_composite_codelet(output, *n, sign_int);
            }
            Algorithm::Generic(solver) => {
                solver.execute(input, output, sign);
            }
            Algorithm::Bluestein(solver) => {
                solver.execute(input, output, sign);
            }
        }
    }
    /// Execute the plan in-place.
    ///
    /// # Panics
    /// Panics if buffer size doesn't match the plan size.
    pub fn execute_inplace(&self, data: &mut [Complex<T>]) {
        use crate::dft::codelets::execute_composite_codelet;

        assert_eq!(data.len(), self.n, "Data size must match plan size");
        let sign = match self.direction {
            Direction::Forward => Sign::Forward,
            Direction::Backward => Sign::Backward,
        };
        match &self.algorithm {
            Algorithm::Nop => {
                NopSolver::new().execute_inplace(data);
            }
            Algorithm::Direct => {
                DirectSolver::new().execute_inplace(data, sign);
            }
            Algorithm::CooleyTukey(variant) => {
                CooleyTukeySolver::new(*variant).execute_inplace(data, sign);
            }
            Algorithm::Stockham => {
                // Stockham is out-of-place, use temp buffer and copy back
                let input = data.to_vec();
                StockhamSolver::new().execute(&input, data, sign);
            }
            Algorithm::Composite(n) => {
                let sign_int = if sign == Sign::Forward { -1 } else { 1 };
                execute_composite_codelet(data, *n, sign_int);
            }
            Algorithm::Generic(solver) => {
                solver.execute_inplace(data, sign);
            }
            Algorithm::Bluestein(solver) => {
                solver.execute_inplace(data, sign);
            }
        }
    }
}
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
        self.plan.n
    }
    /// Get the transform direction.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.plan.direction
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
        let n = self.plan.n;
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
        let n = self.plan.n;
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
/// A plan for executing 2D FFT transforms.
///
/// Implements row-column decomposition: apply 1D FFT to all rows,
/// then to all columns.
pub struct Plan2D<T: Float> {
    /// Number of rows
    n0: usize,
    /// Number of columns
    n1: usize,
    /// Transform direction
    direction: Direction,
    /// 1D plan for rows (size n1)
    row_plan: Plan<T>,
    /// 1D plan for columns (size n0)
    col_plan: Plan<T>,
}
impl<T: Float> Plan2D<T> {
    /// Create a 2D complex-to-complex DFT plan.
    ///
    /// # Arguments
    /// * `n0` - Number of rows
    /// * `n1` - Number of columns
    /// * `direction` - Forward or Backward transform
    /// * `flags` - Planning flags
    ///
    /// # Returns
    /// A plan that can be executed on row-major input/output buffers of size n0 × n1.
    #[must_use]
    pub fn new(n0: usize, n1: usize, direction: Direction, flags: Flags) -> Option<Self> {
        let row_plan = Plan::dft_1d(n1, direction, flags)?;
        let col_plan = Plan::dft_1d(n0, direction, flags)?;
        Some(Self {
            n0,
            n1,
            direction,
            row_plan,
            col_plan,
        })
    }
    /// Get the number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.n0
    }
    /// Get the number of columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.n1
    }
    /// Get the total size (n0 × n1).
    #[must_use]
    pub fn size(&self) -> usize {
        self.n0 * self.n1
    }
    /// Get the transform direction.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.direction
    }
    /// Execute the 2D FFT on the given input/output buffers.
    ///
    /// Input and output are row-major: element at (i, j) is at index i*n1 + j.
    ///
    /// # Panics
    /// Panics if buffer sizes don't match n0 × n1.
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        let total = self.n0 * self.n1;
        assert_eq!(input.len(), total, "Input size must match n0 × n1");
        assert_eq!(output.len(), total, "Output size must match n0 × n1");
        if total == 0 {
            return;
        }
        let mut temp = vec![Complex::zero(); total];
        for i in 0..self.n0 {
            let row_start = i * self.n1;
            let row_end = row_start + self.n1;
            self.row_plan
                .execute(&input[row_start..row_end], &mut temp[row_start..row_end]);
        }
        let mut col_in = vec![Complex::zero(); self.n0];
        let mut col_out = vec![Complex::zero(); self.n0];
        for j in 0..self.n1 {
            for i in 0..self.n0 {
                col_in[i] = temp[i * self.n1 + j];
            }
            self.col_plan.execute(&col_in, &mut col_out);
            for i in 0..self.n0 {
                output[i * self.n1 + j] = col_out[i];
            }
        }
    }
    /// Execute the 2D FFT in-place.
    ///
    /// # Panics
    /// Panics if buffer size doesn't match n0 × n1.
    pub fn execute_inplace(&self, data: &mut [Complex<T>]) {
        let total = self.n0 * self.n1;
        assert_eq!(data.len(), total, "Data size must match n0 × n1");
        if total == 0 {
            return;
        }
        for i in 0..self.n0 {
            let row_start = i * self.n1;
            let row_end = row_start + self.n1;
            self.row_plan.execute_inplace(&mut data[row_start..row_end]);
        }
        let mut col = vec![Complex::zero(); self.n0];
        for j in 0..self.n1 {
            for i in 0..self.n0 {
                col[i] = data[i * self.n1 + j];
            }
            self.col_plan.execute_inplace(&mut col);
            for i in 0..self.n0 {
                data[i * self.n1 + j] = col[i];
            }
        }
    }
}
/// A plan for executing real-to-real transforms (DCT/DST/DHT).
///
/// Real-to-real transforms map real input to real output, and include:
/// - DCT (Discrete Cosine Transform) types I-IV
/// - DST (Discrete Sine Transform) types I-IV
/// - DHT (Discrete Hartley Transform)
pub struct R2rPlan<T: Float> {
    /// Transform size
    n: usize,
    /// Transform kind
    kind: R2rKind,
    _marker: core::marker::PhantomData<T>,
}
impl<T: Float> R2rPlan<T> {
    /// Create a 1D real-to-real transform plan.
    ///
    /// # Arguments
    /// * `n` - Transform size
    /// * `kind` - Type of transform (DCT, DST, or DHT variant)
    /// * `flags` - Planning flags
    ///
    /// # Returns
    /// A plan that transforms n real values to n real values.
    #[must_use]
    pub fn r2r_1d(n: usize, kind: R2rKind, _flags: Flags) -> Option<Self> {
        if n == 0 {
            return None;
        }
        Some(Self {
            n,
            kind,
            _marker: core::marker::PhantomData,
        })
    }
    /// Create a DCT-I (REDFT00) plan.
    #[must_use]
    pub fn dct1(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Redft00, flags)
    }
    /// Create a DCT-II (REDFT10) plan - the "standard" DCT.
    #[must_use]
    pub fn dct2(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Redft10, flags)
    }
    /// Create a DCT-III (REDFT01) plan - the inverse of DCT-II.
    #[must_use]
    pub fn dct3(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Redft01, flags)
    }
    /// Create a DCT-IV (REDFT11) plan.
    #[must_use]
    pub fn dct4(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Redft11, flags)
    }
    /// Create a DST-I (RODFT00) plan.
    #[must_use]
    pub fn dst1(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Rodft00, flags)
    }
    /// Create a DST-II (RODFT10) plan.
    #[must_use]
    pub fn dst2(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Rodft10, flags)
    }
    /// Create a DST-III (RODFT01) plan - the inverse of DST-II.
    #[must_use]
    pub fn dst3(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Rodft01, flags)
    }
    /// Create a DST-IV (RODFT11) plan.
    #[must_use]
    pub fn dst4(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Rodft11, flags)
    }
    /// Create a DHT (Discrete Hartley Transform) plan.
    #[must_use]
    pub fn dht(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Dht, flags)
    }
    /// Get the transform size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.n
    }
    /// Get the transform kind.
    #[must_use]
    pub fn kind(&self) -> R2rKind {
        self.kind
    }
    /// Execute the plan.
    ///
    /// # Panics
    /// Panics if buffer sizes don't match the plan size.
    pub fn execute(&self, input: &[T], output: &mut [T]) {
        use crate::rdft::solvers::R2rSolver;
        assert_eq!(input.len(), self.n, "Input size must match plan size");
        assert_eq!(output.len(), self.n, "Output size must match plan size");
        let solver = R2rSolver::new(self.kind);
        solver.execute(input, output);
    }
    /// Execute the plan in-place.
    ///
    /// # Panics
    /// Panics if buffer size doesn't match the plan size.
    pub fn execute_inplace(&self, data: &mut [T]) {
        use crate::rdft::solvers::R2rSolver;
        assert_eq!(data.len(), self.n, "Data size must match plan size");
        let input = data.to_vec();
        let solver = R2rSolver::new(self.kind);
        solver.execute(&input, data);
    }
}
