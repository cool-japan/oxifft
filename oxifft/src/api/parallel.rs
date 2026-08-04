//! Parallel FFT execution support.
//!
//! This module provides multi-threaded versions of FFT plans that utilize
//! the [`ThreadPool`](crate::threading::ThreadPool) trait for parallel execution.

use crate::kernel::{Complex, Float};
use crate::prelude::*;
use crate::threading::ThreadPool;

use super::plan::Plan;
use super::types::{Direction, Flags};

/// `Send`/`Sync` wrapper carrying a raw pointer across the [`ThreadPool`]
/// boundary.
///
/// The payload is a **pointer**, not a `usize`. An earlier version stored
/// `ptr as usize` and cast back with `usize as *mut T`; that round trip is an
/// integer-to-pointer cast, which strips the pointer's provenance and leaves the
/// reconstructed pointer with only whatever provenance the compiler can guess.
/// Miri flags every such site (`warning: integer-to-pointer cast`) precisely
/// because the resulting accesses are no longer provably in-bounds of the
/// original allocation under Stacked/Tree Borrows. Storing `*mut u8` and casting
/// pointer-to-pointer keeps the original provenance intact, so the derived
/// `add`/`from_raw_parts` accesses stay attached to the buffer they came from.
///
/// SAFETY: The caller must ensure the pointer is valid for the accesses it
/// performs and that no data races occur. See [`IndexClaims`] for the guard that
/// keeps a misbehaving pool from producing overlapping accesses.
#[derive(Clone, Copy)]
struct RawPtr(*mut u8);

impl RawPtr {
    fn from_ptr<T>(ptr: *const T) -> Self {
        Self(ptr.cast::<u8>().cast_mut())
    }

    fn from_mut_ptr<T>(ptr: *mut T) -> Self {
        Self(ptr.cast::<u8>())
    }

    fn as_ptr<T>(self) -> *const T {
        self.0.cast_const().cast::<T>()
    }

    fn as_mut_ptr<T>(self) -> *mut T {
        self.0.cast::<T>()
    }
}

unsafe impl Send for RawPtr {}
unsafe impl Sync for RawPtr {}

/// One-shot claim table guarding raw-pointer index arithmetic against a
/// misbehaving [`ThreadPool`].
///
/// [`ThreadPool::parallel_for`] is a **safe** trait method, so a third-party
/// implementation may legally call the closure with an index outside
/// `0..count`, or call it twice (even concurrently) for the same index. Every
/// raw-pointer partitioning site in this module derives a disjoint sub-buffer
/// from the index alone, so either misbehaviour would create out-of-bounds
/// writes or `&mut` aliasing without a single `unsafe` block in the user's
/// code.
///
/// `IndexClaims::claim` closes that hole: it returns `true` at most once per
/// index and never for an out-of-range index, so a broken pool yields
/// incomplete results instead of undefined behaviour.
struct IndexClaims {
    claimed: Vec<core::sync::atomic::AtomicBool>,
}

impl IndexClaims {
    fn new(count: usize) -> Self {
        Self {
            claimed: (0..count)
                .map(|_| core::sync::atomic::AtomicBool::new(false))
                .collect(),
        }
    }

    /// Returns `true` exactly once for each index in `0..count`, and `false`
    /// for repeated or out-of-range indices.
    fn claim(&self, index: usize) -> bool {
        match self.claimed.get(index) {
            Some(flag) => !flag.swap(true, core::sync::atomic::Ordering::AcqRel),
            None => false,
        }
    }
}

/// Multiply two extents, panicking with a clear message instead of wrapping.
///
/// The wrapped product would otherwise be compared against the caller's buffer
/// length by an `assert_eq!`, letting an undersized buffer pass the guard and
/// then be indexed with the *unwrapped* extents through raw pointers.
#[inline]
fn checked_extent(a: usize, b: usize, what: &str) -> usize {
    match a.checked_mul(b) {
        Some(total) => total,
        None => panic!("{what} overflows usize"),
    }
}

/// Product of all dimensions, panicking instead of wrapping. See
/// [`checked_extent`] for why the wrapped value must never be used.
#[inline]
fn checked_dims_product(dims: &[usize]) -> usize {
    let mut total: usize = 1;
    for &d in dims {
        total = match total.checked_mul(d) {
            Some(t) => t,
            None => panic!("product of dimensions overflows usize"),
        };
    }
    total
}

/// A parallel plan for executing 2D FFT transforms.
///
/// Uses a thread pool to parallelize row and column transforms.
pub struct ParallelPlan2D<T: Float, P: ThreadPool> {
    /// Number of rows
    n0: usize,
    /// Number of columns
    n1: usize,
    /// Total element count (`n0 * n1`), computed once with checked arithmetic.
    total_size: usize,
    /// Transform direction
    direction: Direction,
    /// 1D plan for rows (size n1)
    row_plan: Plan<T>,
    /// 1D plan for columns (size n0)
    col_plan: Plan<T>,
    /// Thread pool for parallel execution
    pool: P,
}

impl<T: Float, P: ThreadPool> ParallelPlan2D<T, P> {
    /// Create a parallel 2D complex-to-complex DFT plan.
    ///
    /// # Arguments
    /// * `n0` - Number of rows
    /// * `n1` - Number of columns
    /// * `direction` - Forward or Backward transform
    /// * `flags` - Planning flags
    /// * `pool` - Thread pool for parallel execution
    ///
    /// # Returns
    /// A plan that can be executed on row-major input/output buffers of size n0 x n1.
    ///
    /// Returns `None` when either 1D plan cannot be built, or when `n0 * n1`
    /// overflows `usize` (a wrapped total would make the buffer-length asserts
    /// in [`Self::execute`] accept an undersized buffer).
    #[must_use]
    pub fn new(n0: usize, n1: usize, direction: Direction, flags: Flags, pool: P) -> Option<Self> {
        let total_size = n0.checked_mul(n1)?;
        let row_plan = Plan::dft_1d(n1, direction, flags)?;
        let col_plan = Plan::dft_1d(n0, direction, flags)?;

        Some(Self {
            n0,
            n1,
            total_size,
            direction,
            row_plan,
            col_plan,
            pool,
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

    /// Get the total size (n0 x n1).
    #[must_use]
    pub fn size(&self) -> usize {
        self.total_size
    }

    /// Get the transform direction.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Get a reference to the thread pool.
    #[must_use]
    pub fn pool(&self) -> &P {
        &self.pool
    }

    /// Execute the 2D FFT in parallel on the given input/output buffers.
    ///
    /// Input and output are row-major: element at (i, j) is at index i*n1 + j.
    ///
    /// # Panics
    /// Panics if buffer sizes don't match n0 x n1.
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        let total = self.total_size;
        assert_eq!(input.len(), total, "Input size must match n0 x n1");
        assert_eq!(output.len(), total, "Output size must match n0 x n1");

        if total == 0 {
            return;
        }

        // Use intermediate buffer for row transforms
        let mut temp = vec![Complex::<T>::zero(); total];

        // Step 1: Apply 1D FFT to each row in parallel
        // SAFETY: Each thread accesses disjoint row slices. `row_claims`
        // enforces that even if `pool` calls the closure with a repeated or
        // out-of-range index (see `IndexClaims`).
        let temp_ptr = RawPtr::from_mut_ptr(temp.as_mut_ptr());
        let input_ptr = RawPtr::from_ptr(input.as_ptr());
        let n1 = self.n1;
        let row_plan = &self.row_plan;
        let row_claims = IndexClaims::new(self.n0);
        let row_claims_ref = &row_claims;
        self.pool.parallel_for(self.n0, move |i| {
            if !row_claims_ref.claim(i) {
                return;
            }
            let row_start = i * n1;
            unsafe {
                let in_ptr: *const Complex<T> = input_ptr.as_ptr();
                let out_ptr: *mut Complex<T> = temp_ptr.as_mut_ptr();
                let in_slice = core::slice::from_raw_parts(in_ptr.add(row_start), n1);
                let out_slice = core::slice::from_raw_parts_mut(out_ptr.add(row_start), n1);
                row_plan.execute(in_slice, out_slice);
            }
        });

        // Step 2: Apply 1D FFT to each column in parallel
        // Each thread handles a subset of columns
        let output_ptr = RawPtr::from_mut_ptr(output.as_mut_ptr());
        let temp_ptr = RawPtr::from_ptr(temp.as_ptr());
        let n0 = self.n0;
        let col_plan = &self.col_plan;
        let col_claims = IndexClaims::new(self.n1);
        let col_claims_ref = &col_claims;
        self.pool.parallel_for(self.n1, move |j| {
            if !col_claims_ref.claim(j) {
                return;
            }
            // Each thread needs its own column buffers
            let mut col_in = vec![Complex::<T>::zero(); n0];
            let mut col_out = vec![Complex::<T>::zero(); n0];

            // Extract column j
            for i in 0..n0 {
                unsafe {
                    let temp_p: *const Complex<T> = temp_ptr.as_ptr();
                    col_in[i] = *temp_p.add(i * n1 + j);
                }
            }

            // Transform column
            col_plan.execute(&col_in, &mut col_out);

            // Place back into output
            for i in 0..n0 {
                unsafe {
                    let out_p: *mut Complex<T> = output_ptr.as_mut_ptr();
                    *out_p.add(i * n1 + j) = col_out[i];
                }
            }
        });
    }

    /// Execute the 2D FFT in-place in parallel.
    ///
    /// # Panics
    /// Panics if buffer size doesn't match n0 x n1.
    pub fn execute_inplace(&self, data: &mut [Complex<T>]) {
        let total = self.total_size;
        assert_eq!(data.len(), total, "Data size must match n0 x n1");

        if total == 0 {
            return;
        }

        // Step 1: Apply 1D FFT to each row in parallel
        let data_ptr = RawPtr::from_mut_ptr(data.as_mut_ptr());
        let n1 = self.n1;
        let n0 = self.n0;
        let row_plan = &self.row_plan;
        let row_claims = IndexClaims::new(self.n0);
        let row_claims_ref = &row_claims;
        self.pool.parallel_for(self.n0, move |i| {
            if !row_claims_ref.claim(i) {
                return;
            }
            let row_start = i * n1;
            unsafe {
                let ptr: *mut Complex<T> = data_ptr.as_mut_ptr();
                let row_slice = core::slice::from_raw_parts_mut(ptr.add(row_start), n1);
                row_plan.execute_inplace(row_slice);
            }
        });

        // Step 2: Apply 1D FFT to each column in parallel
        let col_plan = &self.col_plan;
        let col_claims = IndexClaims::new(self.n1);
        let col_claims_ref = &col_claims;
        self.pool.parallel_for(self.n1, move |j| {
            if !col_claims_ref.claim(j) {
                return;
            }
            let mut col = vec![Complex::<T>::zero(); n0];

            // Extract column j
            for i in 0..n0 {
                unsafe {
                    let ptr: *const Complex<T> = data_ptr.as_ptr();
                    col[i] = *ptr.add(i * n1 + j);
                }
            }

            // Transform column in-place
            col_plan.execute_inplace(&mut col);

            // Place back into data
            for i in 0..n0 {
                unsafe {
                    let ptr: *mut Complex<T> = data_ptr.as_mut_ptr();
                    *ptr.add(i * n1 + j) = col[i];
                }
            }
        });
    }
}

/// A parallel plan for executing N-dimensional FFT transforms.
///
/// Uses a thread pool to parallelize fiber transforms along each dimension.
pub struct ParallelPlanND<T: Float, P: ThreadPool> {
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
    /// Thread pool for parallel execution.
    pool: P,
}

impl<T: Float, P: ThreadPool> ParallelPlanND<T, P> {
    /// Create a parallel N-dimensional complex-to-complex DFT plan.
    ///
    /// # Arguments
    /// * `dims` - Array of dimension sizes in row-major order (slowest varying first)
    /// * `direction` - Forward or Backward transform
    /// * `flags` - Planning flags
    /// * `pool` - Thread pool for parallel execution
    ///
    /// # Returns
    /// A plan for row-major N-dimensional data.
    #[must_use]
    pub fn new(dims: &[usize], direction: Direction, flags: Flags, pool: P) -> Option<Self> {
        if dims.is_empty() {
            return None;
        }

        // Calculate total size and strides
        let mut total_size: usize = 1;
        for &d in dims {
            total_size = total_size.checked_mul(d)?;
        }

        // Strides: stride[i] = product of dims[i+1..]
        let mut strides = vec![1; dims.len()];
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        // Create 1D plans for each dimension
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
            pool,
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

    /// Get a reference to the thread pool.
    #[must_use]
    pub fn pool(&self) -> &P {
        &self.pool
    }

    /// Execute the N-dimensional FFT in parallel on the given input/output buffers.
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

        // Work with alternating buffers
        let mut current = input.to_vec();
        let mut next = vec![Complex::<T>::zero(); self.total_size];

        // Apply 1D FFT along each dimension, starting from the last (fastest varying)
        for dim_idx in (0..self.dims.len()).rev() {
            self.transform_along_dimension_parallel(&current, &mut next, dim_idx);
            core::mem::swap(&mut current, &mut next);
        }

        output.copy_from_slice(&current);
    }

    /// Execute the N-dimensional FFT in-place in parallel.
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

        // Use temporary buffer
        let mut temp = vec![Complex::<T>::zero(); self.total_size];

        // Apply 1D FFT along each dimension, starting from the last (fastest varying)
        for dim_idx in (0..self.dims.len()).rev() {
            self.transform_along_dimension_parallel(data, &mut temp, dim_idx);
            data.copy_from_slice(&temp);
        }
    }

    /// Transform along a single dimension in parallel.
    ///
    /// For each "fiber" along dimension `dim_idx`, extract it, transform it,
    /// and place it back.
    fn transform_along_dimension_parallel(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        dim_idx: usize,
    ) {
        let dim_size = self.dims[dim_idx];
        let stride = self.strides[dim_idx];

        // Number of fibers to transform
        let num_fibers = self.total_size / dim_size;

        // Precompute fiber start indices to avoid referencing self in the closure
        let fiber_starts: Vec<usize> = (0..num_fibers)
            .map(|fiber_idx| self.fiber_start_index(fiber_idx, dim_idx))
            .collect();

        // Process fibers in parallel
        let input_ptr = RawPtr::from_ptr(input.as_ptr());
        let output_ptr = RawPtr::from_mut_ptr(output.as_mut_ptr());
        let fiber_starts_ptr = RawPtr::from_ptr(fiber_starts.as_ptr());
        let plan = &self.plans[dim_idx];

        let fiber_claims = IndexClaims::new(num_fibers);
        let fiber_claims_ref = &fiber_claims;

        self.pool.parallel_for(num_fibers, move |fiber_idx| {
            if !fiber_claims_ref.claim(fiber_idx) {
                return;
            }
            // Get precomputed starting index for this fiber
            let start_idx = unsafe {
                let ptr: *const usize = fiber_starts_ptr.as_ptr();
                *ptr.add(fiber_idx)
            };

            // Each thread needs its own fiber buffers
            let mut fiber_in = vec![Complex::<T>::zero(); dim_size];
            let mut fiber_out = vec![Complex::<T>::zero(); dim_size];

            // Extract fiber
            for i in 0..dim_size {
                unsafe {
                    let in_p: *const Complex<T> = input_ptr.as_ptr();
                    fiber_in[i] = *in_p.add(start_idx + i * stride);
                }
            }

            // Transform
            plan.execute(&fiber_in, &mut fiber_out);

            // Place back
            for i in 0..dim_size {
                unsafe {
                    let out_p: *mut Complex<T> = output_ptr.as_mut_ptr();
                    *out_p.add(start_idx + i * stride) = fiber_out[i];
                }
            }
        });
    }

    /// Compute the starting index for a fiber along dimension `dim_idx`.
    ///
    /// Given a linear fiber index, compute the base index in the flat array.
    fn fiber_start_index(&self, fiber_idx: usize, dim_idx: usize) -> usize {
        // fiber_idx enumerates all combinations of indices except dim_idx
        // We need to compute the corresponding flat index with dim_idx = 0

        let mut idx = 0;
        let mut remaining = fiber_idx;

        // Process dimensions in order, skipping dim_idx
        for d in 0..self.dims.len() {
            if d == dim_idx {
                continue;
            }

            // How many elements are "below" this dimension in the fiber enumeration?
            let below_count = self.fiber_below_count(d, dim_idx);

            // Extract this dimension's coordinate
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

/// Convenience function for parallel 2D forward FFT.
///
/// Input is row-major with n0 rows and n1 columns.
pub fn fft2d_parallel<T: Float, P: ThreadPool + Clone>(
    input: &[Complex<T>],
    n0: usize,
    n1: usize,
    pool: &P,
) -> Vec<Complex<T>> {
    let total = checked_extent(n0, n1, "n0 * n1");
    assert_eq!(input.len(), total, "Input size must match n0 x n1");
    let mut output = vec![Complex::<T>::zero(); total];

    if let Some(plan) =
        ParallelPlan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE, pool.clone())
    {
        plan.execute(input, &mut output);
    }

    output
}

/// Convenience function for parallel 2D inverse FFT with normalization.
///
/// Normalizes by 1/(n0 x n1).
pub fn ifft2d_parallel<T: Float, P: ThreadPool + Clone>(
    input: &[Complex<T>],
    n0: usize,
    n1: usize,
    pool: &P,
) -> Vec<Complex<T>> {
    let total = checked_extent(n0, n1, "n0 * n1");
    assert_eq!(input.len(), total, "Input size must match n0 x n1");
    let mut output = vec![Complex::<T>::zero(); total];

    if let Some(plan) =
        ParallelPlan2D::new(n0, n1, Direction::Backward, Flags::ESTIMATE, pool.clone())
    {
        plan.execute(input, &mut output);

        // Normalize
        let scale = T::from_usize(total);
        for x in &mut output {
            *x = *x / scale;
        }
    }

    output
}

/// Convenience function for parallel N-dimensional forward FFT.
///
/// Input is in row-major order.
pub fn fft_nd_parallel<T: Float, P: ThreadPool + Clone>(
    input: &[Complex<T>],
    dims: &[usize],
    pool: &P,
) -> Vec<Complex<T>> {
    let total = checked_dims_product(dims);
    assert_eq!(
        input.len(),
        total,
        "Input size must match product of dimensions"
    );

    let mut output = vec![Complex::<T>::zero(); total];

    if let Some(plan) = ParallelPlanND::new(dims, Direction::Forward, Flags::ESTIMATE, pool.clone())
    {
        plan.execute(input, &mut output);
    }

    output
}

/// Convenience function for parallel N-dimensional inverse FFT with normalization.
///
/// Normalizes by 1/(product of dimensions).
pub fn ifft_nd_parallel<T: Float, P: ThreadPool + Clone>(
    input: &[Complex<T>],
    dims: &[usize],
    pool: &P,
) -> Vec<Complex<T>> {
    let total = checked_dims_product(dims);
    assert_eq!(
        input.len(),
        total,
        "Input size must match product of dimensions"
    );

    let mut output = vec![Complex::<T>::zero(); total];

    if let Some(plan) =
        ParallelPlanND::new(dims, Direction::Backward, Flags::ESTIMATE, pool.clone())
    {
        plan.execute(input, &mut output);

        // Normalize
        let scale = T::from_usize(total);
        for x in &mut output {
            *x = *x / scale;
        }
    }

    output
}

/// Convenience function for parallel batched 1D forward FFT.
///
/// Performs `howmany` independent 1D FFTs, each of size `n`, in parallel.
/// Input and output are contiguous: batch `i` starts at index `i * n`.
pub fn fft_batch_parallel<T: Float, P: ThreadPool>(
    input: &[Complex<T>],
    n: usize,
    howmany: usize,
    pool: &P,
) -> Vec<Complex<T>> {
    let total = checked_extent(n, howmany, "n * howmany");
    assert_eq!(input.len(), total, "Input size must match n * howmany");

    let mut output = vec![Complex::<T>::zero(); total];

    // Create 1D plan
    if let Some(plan) = Plan::<T>::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
        let input_ptr = RawPtr::from_ptr(input.as_ptr());
        let output_ptr = RawPtr::from_mut_ptr(output.as_mut_ptr());

        // Process each batch in parallel
        let claims = IndexClaims::new(howmany);
        let claims_ref = &claims;
        pool.parallel_for(howmany, move |batch_idx| {
            if !claims_ref.claim(batch_idx) {
                return;
            }
            let offset = batch_idx * n;
            unsafe {
                let in_p: *const Complex<T> = input_ptr.as_ptr();
                let out_p: *mut Complex<T> = output_ptr.as_mut_ptr();
                let in_slice = core::slice::from_raw_parts(in_p.add(offset), n);
                let out_slice = core::slice::from_raw_parts_mut(out_p.add(offset), n);
                plan.execute(in_slice, out_slice);
            }
        });
    }

    output
}

/// Convenience function for parallel batched 1D inverse FFT with normalization.
///
/// Performs `howmany` independent 1D inverse FFTs, each of size `n`, in parallel.
/// Normalizes each output by 1/n.
pub fn ifft_batch_parallel<T: Float, P: ThreadPool>(
    input: &[Complex<T>],
    n: usize,
    howmany: usize,
    pool: &P,
) -> Vec<Complex<T>> {
    let total = checked_extent(n, howmany, "n * howmany");
    assert_eq!(input.len(), total, "Input size must match n * howmany");

    let mut output = vec![Complex::<T>::zero(); total];

    // Create 1D plan
    if let Some(plan) = Plan::<T>::dft_1d(n, Direction::Backward, Flags::ESTIMATE) {
        let input_ptr = RawPtr::from_ptr(input.as_ptr());
        let output_ptr = RawPtr::from_mut_ptr(output.as_mut_ptr());

        // Process each batch in parallel
        let claims = IndexClaims::new(howmany);
        let claims_ref = &claims;
        pool.parallel_for(howmany, move |batch_idx| {
            if !claims_ref.claim(batch_idx) {
                return;
            }
            let offset = batch_idx * n;
            unsafe {
                let in_p: *const Complex<T> = input_ptr.as_ptr();
                let out_p: *mut Complex<T> = output_ptr.as_mut_ptr();
                let in_slice = core::slice::from_raw_parts(in_p.add(offset), n);
                let out_slice = core::slice::from_raw_parts_mut(out_p.add(offset), n);
                plan.execute(in_slice, out_slice);
            }
        });

        // Normalize
        let scale = T::from_usize(n);
        for x in &mut output {
            *x = *x / scale;
        }
    }

    output
}

/// Convenience function for parallel batched 1D Real-to-Complex FFT.
///
/// Performs `howmany` independent R2C FFTs, each of size `n`, in parallel.
/// Input batches are contiguous (size n), output batches are contiguous (size n/2+1).
pub fn rfft_batch_parallel<T: Float, P: ThreadPool>(
    input: &[T],
    n: usize,
    howmany: usize,
    pool: &P,
) -> Vec<Complex<T>> {
    use crate::rdft::solvers::R2cSolver;

    let total = checked_extent(n, howmany, "n * howmany");
    assert_eq!(input.len(), total, "Input size must match n * howmany");

    let out_len = n / 2 + 1;
    let mut output =
        vec![Complex::<T>::zero(); checked_extent(out_len, howmany, "(n/2+1) * howmany")];

    let solver = R2cSolver::new(n);
    let input_ptr = RawPtr::from_ptr(input.as_ptr());
    let output_ptr = RawPtr::from_mut_ptr(output.as_mut_ptr());

    // Process each batch in parallel
    let claims = IndexClaims::new(howmany);
    let claims_ref = &claims;
    pool.parallel_for(howmany, move |batch_idx| {
        if !claims_ref.claim(batch_idx) {
            return;
        }
        let in_offset = batch_idx * n;
        let out_offset = batch_idx * out_len;
        unsafe {
            let in_p: *const T = input_ptr.as_ptr();
            let out_p: *mut Complex<T> = output_ptr.as_mut_ptr();
            let in_slice = core::slice::from_raw_parts(in_p.add(in_offset), n);
            let out_slice = core::slice::from_raw_parts_mut(out_p.add(out_offset), out_len);
            solver.execute(in_slice, out_slice);
        }
    });

    output
}

/// Convenience function for parallel batched 1D Complex-to-Real FFT with normalization.
///
/// Performs `howmany` independent C2R FFTs, each producing `n` real values, in parallel.
/// Input batches have size n/2+1, output batches have size n.
pub fn irfft_batch_parallel<T: Float, P: ThreadPool>(
    input: &[Complex<T>],
    n: usize,
    howmany: usize,
    pool: &P,
) -> Vec<T> {
    use crate::rdft::solvers::C2rSolver;

    let in_len = n / 2 + 1;
    assert_eq!(
        input.len(),
        checked_extent(in_len, howmany, "(n/2+1) * howmany"),
        "Input size must match (n/2+1) * howmany"
    );

    let mut output = vec![T::ZERO; checked_extent(n, howmany, "n * howmany")];

    let solver = C2rSolver::new(n);
    let input_ptr = RawPtr::from_ptr(input.as_ptr());
    let output_ptr = RawPtr::from_mut_ptr(output.as_mut_ptr());

    // Process each batch in parallel
    let claims = IndexClaims::new(howmany);
    let claims_ref = &claims;
    pool.parallel_for(howmany, move |batch_idx| {
        if !claims_ref.claim(batch_idx) {
            return;
        }
        let in_offset = batch_idx * in_len;
        let out_offset = batch_idx * n;
        unsafe {
            let in_p: *const Complex<T> = input_ptr.as_ptr();
            let out_p: *mut T = output_ptr.as_mut_ptr();
            let in_slice = core::slice::from_raw_parts(in_p.add(in_offset), in_len);
            let out_slice = core::slice::from_raw_parts_mut(out_p.add(out_offset), n);
            solver.execute_normalized(in_slice, out_slice);
        }
    });

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::plan::{fft2d, fft_batch, fft_nd, rfft_batch};
    use crate::threading::SerialPool;

    fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
        (a.re - b.re).abs() < eps && (a.im - b.im).abs() < eps
    }

    #[test]
    fn test_parallel_plan2d_matches_serial() {
        let n0 = 8;
        let n1 = 8;
        let total = n0 * n1;

        let input: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        // Serial version
        let output_serial = fft2d(&input, n0, n1);

        // Parallel version with SerialPool (should produce same result)
        let pool = SerialPool::new();
        let output_parallel = fft2d_parallel(&input, n0, n1, &pool);

        for (a, b) in output_serial.iter().zip(output_parallel.iter()) {
            assert!(
                complex_approx_eq(*a, *b, 1e-10),
                "Mismatch: serial={a:?}, parallel={b:?}"
            );
        }
    }

    #[test]
    fn test_parallel_plan2d_roundtrip() {
        let n0 = 8;
        let n1 = 8;
        let total = n0 * n1;

        let original: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        let pool = SerialPool::new();
        let transformed = fft2d_parallel(&original, n0, n1, &pool);
        let recovered = ifft2d_parallel(&transformed, n0, n1, &pool);

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(
                complex_approx_eq(*a, *b, 1e-10),
                "got {b:?}, expected {a:?}"
            );
        }
    }

    #[test]
    fn test_parallel_plan2d_inplace() {
        let n0 = 8;
        let n1 = 8;
        let total = n0 * n1;

        let original: Vec<Complex<f64>> = (0..total).map(|i| Complex::new(i as f64, 0.0)).collect();

        let pool = SerialPool::new();

        // Out-of-place
        let mut out_of_place = vec![Complex::<f64>::zero(); total];
        let plan = ParallelPlan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE, pool).unwrap();
        plan.execute(&original, &mut out_of_place);

        // In-place
        let pool2 = SerialPool::new();
        let plan2 =
            ParallelPlan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE, pool2).unwrap();
        let mut in_place = original;
        plan2.execute_inplace(&mut in_place);

        for (a, b) in out_of_place.iter().zip(in_place.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_parallel_plannd_matches_serial() {
        let dims = [4, 4, 4];
        let total: usize = dims.iter().product();

        let input: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        // Serial version
        let output_serial = fft_nd(&input, &dims);

        // Parallel version with SerialPool
        let pool = SerialPool::new();
        let output_parallel = fft_nd_parallel(&input, &dims, &pool);

        for (a, b) in output_serial.iter().zip(output_parallel.iter()) {
            assert!(
                complex_approx_eq(*a, *b, 1e-10),
                "Mismatch: serial={a:?}, parallel={b:?}"
            );
        }
    }

    #[test]
    fn test_parallel_plannd_roundtrip() {
        let dims = [4, 4, 4];
        let total: usize = dims.iter().product();

        let original: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        let pool = SerialPool::new();
        let transformed = fft_nd_parallel(&original, &dims, &pool);
        let recovered = ifft_nd_parallel(&transformed, &dims, &pool);

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(
                complex_approx_eq(*a, *b, 1e-10),
                "got {b:?}, expected {a:?}"
            );
        }
    }

    #[test]
    fn test_parallel_plannd_inplace() {
        let dims = [4, 4, 4];
        let total: usize = dims.iter().product();

        let original: Vec<Complex<f64>> = (0..total).map(|i| Complex::new(i as f64, 0.0)).collect();

        let pool = SerialPool::new();

        // Out-of-place
        let mut out_of_place = vec![Complex::<f64>::zero(); total];
        let plan = ParallelPlanND::new(&dims, Direction::Forward, Flags::ESTIMATE, pool).unwrap();
        plan.execute(&original, &mut out_of_place);

        // In-place
        let pool2 = SerialPool::new();
        let plan2 = ParallelPlanND::new(&dims, Direction::Forward, Flags::ESTIMATE, pool2).unwrap();
        let mut in_place = original;
        plan2.execute_inplace(&mut in_place);

        for (a, b) in out_of_place.iter().zip(in_place.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_parallel_plan2d_non_power_of_2() {
        let n0 = 5;
        let n1 = 7;
        let total = n0 * n1;

        let original: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        let pool = SerialPool::new();
        let transformed = fft2d_parallel(&original, n0, n1, &pool);
        let recovered = ifft2d_parallel(&transformed, n0, n1, &pool);

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9), "got {b:?}, expected {a:?}");
        }
    }

    // Batch parallel FFT tests

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_fft_batch_parallel_matches_serial() {
        let n = 16;
        let howmany = 4;
        let total = n * howmany;

        let input: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        // Serial version
        let output_serial = fft_batch(&input, n, howmany);

        // Parallel version with SerialPool
        let pool = SerialPool::new();
        let output_parallel = fft_batch_parallel(&input, n, howmany, &pool);

        for (a, b) in output_serial.iter().zip(output_parallel.iter()) {
            assert!(
                complex_approx_eq(*a, *b, 1e-10),
                "Mismatch: serial={a:?}, parallel={b:?}"
            );
        }
    }

    #[test]
    fn test_fft_batch_parallel_roundtrip() {
        let n = 16;
        let howmany = 4;
        let total = n * howmany;

        let original: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        let pool = SerialPool::new();
        let transformed = fft_batch_parallel(&original, n, howmany, &pool);
        let recovered = ifft_batch_parallel(&transformed, n, howmany, &pool);

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(
                complex_approx_eq(*a, *b, 1e-10),
                "got {b:?}, expected {a:?}"
            );
        }
    }

    #[test]
    fn test_rfft_batch_parallel_matches_serial() {
        let n = 16;
        let howmany = 4;

        let input: Vec<f64> = (0..(n * howmany)).map(|i| (i as f64).sin()).collect();

        // Serial version
        let output_serial = rfft_batch(&input, n, howmany);

        // Parallel version with SerialPool
        let pool = SerialPool::new();
        let output_parallel = rfft_batch_parallel(&input, n, howmany, &pool);

        for (a, b) in output_serial.iter().zip(output_parallel.iter()) {
            assert!(
                complex_approx_eq(*a, *b, 1e-10),
                "Mismatch: serial={a:?}, parallel={b:?}"
            );
        }
    }

    #[test]
    fn test_rfft_batch_parallel_roundtrip() {
        let n = 16;
        let howmany = 4;

        let original: Vec<f64> = (0..(n * howmany)).map(|i| (i as f64).sin()).collect();

        let pool = SerialPool::new();
        let freq = rfft_batch_parallel(&original, n, howmany, &pool);
        let recovered = irfft_batch_parallel(&freq, n, howmany, &pool);

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(approx_eq(*a, *b, 1e-10), "got {b}, expected {a}");
        }
    }
}

// ============================================================================
// Regression tests: hostile `ThreadPool` impls and extent overflow
// ============================================================================

#[cfg(test)]
mod hardening_tests {
    use super::*;
    use crate::threading::SerialPool;

    /// A 100% safe `ThreadPool` implementation that violates the "call `f`
    /// exactly once per index in `0..count`" contract.
    ///
    /// Every raw-pointer partitioning site in this module derives its
    /// sub-buffer from the index alone, so before `IndexClaims` existed this
    /// pool made safe user code produce out-of-bounds raw-pointer writes
    /// (`f(count * 4)` → `row_start` past the end of the buffer) and `&mut`
    /// aliasing (`f(0)` repeatedly) without a single `unsafe` block.
    struct HostilePool;

    impl ThreadPool for HostilePool {
        fn parallel_for<F>(&self, count: usize, f: F)
        where
            F: Fn(usize) + Send + Sync,
        {
            // Out-of-range indices first...
            if count > 0 {
                f(count);
                f(count * 4 + 1);
                f(usize::MAX);
            }
            // ...then every valid index, each one twice.
            for i in 0..count {
                f(i);
                f(i);
            }
        }

        fn num_threads(&self) -> usize {
            1
        }

        fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
        where
            A: FnOnce() -> RA + Send,
            B: FnOnce() -> RB + Send,
            RA: Send,
            RB: Send,
        {
            (a(), b())
        }
    }

    fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
        (a.re - b.re).abs() < eps && (a.im - b.im).abs() < eps
    }

    fn make_input(total: usize) -> Vec<Complex<f64>> {
        (0..total)
            .map(|i| Complex::new((i as f64 * 0.11).sin(), (i as f64 * 0.07).cos()))
            .collect()
    }

    #[test]
    fn parallel_plan2d_survives_a_contract_breaking_pool() {
        let (n0, n1) = (8_usize, 8_usize);
        let input = make_input(n0 * n1);

        let mut expected = vec![Complex::<f64>::zero(); n0 * n1];
        ParallelPlan2D::new(
            n0,
            n1,
            Direction::Forward,
            Flags::ESTIMATE,
            SerialPool::new(),
        )
        .expect("serial plan")
        .execute(&input, &mut expected);

        let mut got = vec![Complex::<f64>::zero(); n0 * n1];
        ParallelPlan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE, HostilePool)
            .expect("hostile plan")
            .execute(&input, &mut got);

        // Out-of-range indices are ignored and each row/column is transformed
        // exactly once, so the result still matches the serial reference.
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *b, 1e-10),
                "element {i}: serial={a:?} hostile={b:?}"
            );
        }
    }

    #[test]
    fn parallel_plan2d_inplace_survives_a_contract_breaking_pool() {
        let (n0, n1) = (8_usize, 8_usize);
        let input = make_input(n0 * n1);

        let mut expected = input.clone();
        ParallelPlan2D::new(
            n0,
            n1,
            Direction::Forward,
            Flags::ESTIMATE,
            SerialPool::new(),
        )
        .expect("serial plan")
        .execute_inplace(&mut expected);

        let mut got = input;
        ParallelPlan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE, HostilePool)
            .expect("hostile plan")
            .execute_inplace(&mut got);

        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *b, 1e-10),
                "element {i}: serial={a:?} hostile={b:?}"
            );
        }
    }

    #[test]
    fn parallel_plan_nd_survives_a_contract_breaking_pool() {
        let dims = [4_usize, 4, 2];
        let total: usize = dims.iter().product();
        let input = make_input(total);

        let mut expected = vec![Complex::<f64>::zero(); total];
        ParallelPlanND::new(
            &dims,
            Direction::Forward,
            Flags::ESTIMATE,
            SerialPool::new(),
        )
        .expect("serial plan")
        .execute(&input, &mut expected);

        let mut got = vec![Complex::<f64>::zero(); total];
        ParallelPlanND::new(&dims, Direction::Forward, Flags::ESTIMATE, HostilePool)
            .expect("hostile plan")
            .execute(&input, &mut got);

        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *b, 1e-10),
                "element {i}: serial={a:?} hostile={b:?}"
            );
        }
    }

    #[test]
    fn batch_helpers_survive_a_contract_breaking_pool() {
        let (n, howmany) = (8_usize, 4_usize);
        let input = make_input(n * howmany);

        let expected = fft_batch_parallel(&input, n, howmany, &SerialPool::new());
        let got = fft_batch_parallel(&input, n, howmany, &HostilePool);
        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *b, 1e-10),
                "element {i}: serial={a:?} hostile={b:?}"
            );
        }

        let real_input: Vec<f64> = (0..(n * howmany)).map(|i| (i as f64).sin()).collect();
        let expected_r = rfft_batch_parallel(&real_input, n, howmany, &SerialPool::new());
        let got_r = rfft_batch_parallel(&real_input, n, howmany, &HostilePool);
        for (i, (a, b)) in expected_r.iter().zip(got_r.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *b, 1e-10),
                "bin {i}: serial={a:?} hostile={b:?}"
            );
        }

        let expected_c = irfft_batch_parallel(&expected_r, n, howmany, &SerialPool::new());
        let got_c = irfft_batch_parallel(&expected_r, n, howmany, &HostilePool);
        for (i, (a, b)) in expected_c.iter().zip(got_c.iter()).enumerate() {
            assert!((a - b).abs() < 1e-10, "sample {i}: serial={a} hostile={b}");
        }
    }

    #[test]
    fn parallel_plan2d_rejects_overflowing_extents() {
        // `n0 * n1` wraps to 0 here.  The old code then let a zero-length (or
        // any wrapped-length) buffer pass `assert_eq!(input.len(), total)` and
        // afterwards indexed with the *unwrapped* `n0`/`n1` through raw
        // pointers.
        let n0 = 1_usize << 63;
        let n1 = 2_usize;
        assert!(n0.wrapping_mul(n1) == 0, "test premise: the product wraps");
        assert!(
            ParallelPlan2D::<f64, SerialPool>::new(
                n0,
                n1,
                Direction::Forward,
                Flags::ESTIMATE,
                SerialPool::new(),
            )
            .is_none(),
            "an overflowing n0 * n1 must not produce a plan"
        );
    }

    #[test]
    #[should_panic(expected = "overflows usize")]
    fn fft2d_parallel_panics_on_overflowing_extents() {
        let pool = SerialPool::new();
        let empty: Vec<Complex<f64>> = Vec::new();
        let _ = fft2d_parallel(&empty, 1_usize << 63, 2, &pool);
    }

    #[test]
    #[should_panic(expected = "overflows usize")]
    fn fft_nd_parallel_panics_on_overflowing_extents() {
        let pool = SerialPool::new();
        let empty: Vec<Complex<f64>> = Vec::new();
        let _ = fft_nd_parallel(&empty, &[1_usize << 62, 4, 4], &pool);
    }

    #[test]
    #[should_panic(expected = "overflows usize")]
    fn fft_batch_parallel_panics_on_overflowing_extents() {
        let pool = SerialPool::new();
        let empty: Vec<Complex<f64>> = Vec::new();
        let _ = fft_batch_parallel(&empty, 1_usize << 63, 4, &pool);
    }
}
