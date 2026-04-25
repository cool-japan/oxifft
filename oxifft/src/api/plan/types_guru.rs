//! Guru FFT plan type with full control over dimensions and strides.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#![allow(clippy::items_after_statements)] // reason: SplitRS-generated code places type defs and constants after use statements

use crate::api::{Direction, Flags};
use crate::kernel::{Complex, Float, Tensor};
use crate::prelude::*;

use super::types::Plan;

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
    ///
    /// # Examples
    ///
    /// ```
    /// use oxifft::{Complex, Direction, Flags, GuruPlan, Tensor};
    ///
    /// // Simple 1D forward FFT via the guru interface
    /// let dims = Tensor::rank1(8);
    /// let howmany = Tensor::empty();
    /// let plan = GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE)
    ///     .expect("plan construction failed");
    ///
    /// let input = vec![Complex::<f64>::new(1.0, 0.0); 8];
    /// let mut output = vec![Complex::<f64>::zero(); 8];
    /// plan.execute(&input, &mut output);
    /// // DC bin = sum of all ones = 8
    /// assert!((output[0].re - 8.0_f64).abs() < 1e-9);
    /// ```
    #[must_use]
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
    ///
    /// Panics if buffers are too small for the specified layout.
    ///
    /// Panics (in debug builds with an earlier diagnostic; in release builds
    /// via an out-of-bounds slice index) if the computed element offset is
    /// negative. This can happen when [`crate::IoDim`] strides are negative
    /// and the batch index is large enough that `base_offset + i * stride < 0`.
    /// Always ensure that for every element `i` in `0..n`, the expression
    /// `in_offset + i * is` and `out_offset + i * os` evaluate to a
    /// non-negative value.
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
                    debug_assert!(
                        idx >= 0,
                        "input offset {idx} is negative (in_offset={in_offset}, i={i}, stride={in_stride}); \
                         negative effective offsets produce out-of-bounds access"
                    );
                    input[idx as usize]
                })
                .collect();
            &input_contiguous
        };
        let mut temp = vec![Complex::<T>::zero(); n];
        self.plans[0].execute(input_slice, &mut temp);
        for i in 0..n {
            let idx = out_offset + (i as isize) * out_stride;
            debug_assert!(
                idx >= 0,
                "output offset {idx} is negative (out_offset={out_offset}, i={i}, stride={out_stride}); \
                 negative effective offsets produce out-of-bounds access"
            );
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
            debug_assert!(
                src_offset >= 0,
                "input offset {src_offset} is negative (flat_idx={flat_idx}, base_offset={base_offset}); \
                 ensure IoDim strides produce non-negative offsets for all elements"
            );
            work[flat_idx] = input[src_offset as usize];
        }
    }
    /// Scatter from contiguous buffer to N-dimensional output.
    fn scatter_nd(&self, work: &[Complex<T>], output: &mut [Complex<T>], base_offset: isize) {
        let total = self.transform_size();
        for flat_idx in 0..total {
            let dst_offset = self.compute_nd_offset(flat_idx, base_offset, false);
            debug_assert!(
                dst_offset >= 0,
                "output offset {dst_offset} is negative (flat_idx={flat_idx}, base_offset={base_offset}); \
                 ensure IoDim strides produce non-negative offsets for all elements"
            );
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
        assert!(
            self.dims.is_inplace_compatible(),
            "In-place execution requires identical input and output strides"
        );
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
                debug_assert!(
                    src_offset >= 0,
                    "in-place offset {src_offset} is negative (flat_idx={flat_idx}, base_offset={offset}); \
                     ensure IoDim strides produce non-negative offsets for all elements"
                );
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
                debug_assert!(
                    dst_offset >= 0,
                    "in-place dst offset {dst_offset} is negative (flat_idx={flat_idx}, base_offset={offset}); \
                     ensure IoDim strides produce non-negative offsets for all elements"
                );
                data[dst_offset as usize] = item;
            }
        }
    }
}
