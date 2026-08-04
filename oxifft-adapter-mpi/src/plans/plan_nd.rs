//! N-dimensional distributed FFT plan.

use mpi::topology::Communicator;

use oxifft::api::Direction;
use oxifft::kernel::{Complex, Float};

use crate::distribution::{Distribution, LocalPartition};
use crate::error::MpiError;
use crate::pool::{MpiFloat, MpiPool};
use crate::transpose::distributed_transpose;
use crate::MpiFlags;

/// N-dimensional distributed FFT plan.
///
/// Generalizes the distributed FFT to arbitrary dimensions using slab decomposition.
/// The first dimension is distributed across processes.
///
/// # Transposed layouts
///
/// The trailing axes are treated as one flat `stride = product(dims[1..])`
/// dimension, so the transposed distribution is `[local_stride][n0]` — the N-D
/// generalisation of the 2-D `[local_n1][n0]` layout.
/// [`MpiFlags::transposed_out`](crate::MpiFlags::transposed_out) emits that
/// layout (skipping the final transpose) and
/// [`MpiFlags::transposed_in`](crate::MpiFlags::transposed_in) consumes it, so a
/// `transposed_out` forward followed by a `transposed_in` inverse costs one
/// `alltoallv` per transform instead of two.
pub struct MpiPlanND<'p, T: Float, C: Communicator> {
    /// Global dimensions.
    dims: Vec<usize>,
    /// Local number of hyperplanes in first dimension.
    local_n0: usize,
    /// Global starting index in first dimension.
    local_0_start: usize,
    /// Transform direction.
    direction: Direction,
    /// Planning flags.
    flags: MpiFlags,
    /// Borrow of the MPI pool; the borrow checker guarantees the pool outlives
    /// the plan.
    pool: &'p MpiPool<C>,
    /// Local plans for each dimension.
    local_plans: Vec<oxifft::api::Plan<T>>,
    /// Scratch buffer holding the transposed `[local_stride][n0]` layout used by
    /// the single-`alltoallv` distributed FFT along dimension 0.
    scratch: Vec<Complex<T>>,
}

impl<'p, T: Float + MpiFloat, C: Communicator> MpiPlanND<'p, T, C> {
    /// Create a new N-D distributed FFT plan.
    ///
    /// # Arguments
    /// * `dims` - Dimension sizes (first dimension is distributed)
    /// * `direction` - Transform direction
    /// * `flags` - Planning flags
    /// * `pool` - MPI pool
    ///
    /// # Errors
    ///
    /// Returns `MpiError::InvalidDimension` if `dims` is empty or any
    /// element is zero.
    pub fn new(
        dims: &[usize],
        direction: Direction,
        flags: MpiFlags,
        pool: &'p MpiPool<C>,
    ) -> Result<Self, MpiError> {
        if dims.is_empty() {
            return Err(MpiError::InvalidDimension {
                dim: 0,
                size: 0,
                message: "Cannot create plan with zero dimensions".to_string(),
            });
        }

        for (i, &size) in dims.iter().enumerate() {
            if size == 0 {
                return Err(MpiError::InvalidDimension {
                    dim: i,
                    size,
                    message: "Dimension size cannot be zero".to_string(),
                });
            }
        }

        let partition = pool.local_partition(dims[0]);
        let local_n0 = partition.local_n;
        let local_0_start = partition.local_start;

        // Calculate local allocation
        let remaining_product: usize = dims[1..].iter().product();
        let local_size = local_n0 * remaining_product;

        // Create local 1D plans for each dimension
        let mut local_plans = Vec::with_capacity(dims.len());
        for (i, &n) in dims.iter().enumerate() {
            let plan = oxifft::api::Plan::dft_1d(n, direction, flags.base).ok_or_else(|| {
                MpiError::FftError {
                    message: format!("Failed to create plan for dimension {i} (size {n})"),
                }
            })?;
            local_plans.push(plan);
        }

        // Scratch buffer for the distributed transpose. The FFT along the
        // distributed dimension gathers every `stride` fiber into the transposed
        // `[local_stride][n0]` layout, so the scratch must hold the larger of the
        // normal (`local_n0 * stride`) and transposed (`n0 * local_stride`)
        // footprints.
        let trans_partition = LocalPartition::new(remaining_product, pool.size(), pool.rank());
        let scratch_size = local_size.max(dims[0] * trans_partition.local_n);
        let scratch = vec![Complex::<T>::zero(); scratch_size];

        Ok(Self {
            dims: dims.to_vec(),
            local_n0,
            local_0_start,
            direction,
            flags,
            pool,
            local_plans,
            scratch,
        })
    }

    /// Get global dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Data-distribution strategy used by this plan (always [`Distribution::Slab`]).
    pub fn distribution(&self) -> Distribution {
        Distribution::Slab
    }

    /// Get number of dimensions.
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Get local partition info.
    pub fn local_info(&self) -> (usize, usize) {
        (self.local_n0, self.local_0_start)
    }

    /// Get the transform direction.
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Get the planning flags.
    pub fn flags(&self) -> MpiFlags {
        self.flags
    }

    /// Local element counts this plan reads and writes, `(input, output)`.
    ///
    /// `transposed_in` / `transposed_out` each swap the corresponding side to the
    /// `[local_stride][n0]` footprint, so the two can differ; `execute_inplace`
    /// requires a buffer at least as large as their maximum.
    pub fn local_footprints(&self) -> (usize, usize) {
        let stride: usize = self.dims[1..].iter().product();
        let local_stride = LocalPartition::new(stride, self.pool.size(), self.pool.rank()).local_n;
        let normal = self.local_n0 * stride;
        let transposed = self.dims[0] * local_stride;
        (
            if self.flags.transposed_in {
                transposed
            } else {
                normal
            },
            if self.flags.transposed_out {
                transposed
            } else {
                normal
            },
        )
    }

    /// Execute the distributed N-D FFT in-place.
    ///
    /// Input layout is the `[local_n0][stride]` slab, or the `[local_stride][n0]`
    /// transposed distribution when `transposed_in` is set; the output layout is
    /// chosen the same way by `transposed_out`.
    ///
    /// # Errors
    /// Returns `MpiError::SizeMismatch` if data buffer is too small.
    pub fn execute_inplace(&mut self, data: &mut [Complex<T>]) -> Result<(), MpiError> {
        let ndim = self.dims.len();

        let (in_size, out_size) = self.local_footprints();
        let required = in_size.max(out_size);
        if data.len() < required {
            return Err(MpiError::SizeMismatch {
                expected: required,
                actual: data.len(),
            });
        }

        let pool = self.pool;

        if self.flags.transposed_in {
            return self.execute_transposed_in(data);
        }

        // Step 1: FFTs along all local dimensions (dims[1..]).
        // For a 1D problem this range is empty; the single (distributed)
        // dimension is handled entirely by `distributed_fft_dim0` below.
        for d in (1..ndim).rev() {
            self.fft_along_dimension(data, d)?;
        }

        // Step 2: Distributed FFT along dimension 0 via a single alltoallv-based
        // transpose over all `stride` fibers. For a 1D problem `stride == 1`, so
        // this degenerates to gathering the single distributed axis onto rank 0,
        // transforming it, and scattering it back.
        self.distributed_fft_dim0(data, pool)?;

        Ok(())
    }

    /// Transposed-input pipeline: `data` already holds the `[local_stride][n0]`
    /// distribution that `transposed_out` produces, carrying *untransformed*
    /// values.
    ///
    /// The distributed `n0` axis is contiguous and complete on this rank in that
    /// layout, so it is transformed first with no communication at all; one
    /// `alltoallv` then restores the `[local_n0][stride]` slab for the remaining
    /// (purely local) axes. A second `alltoallv` runs only when `transposed_out`
    /// is also requested.
    fn execute_transposed_in(&mut self, data: &mut [Complex<T>]) -> Result<(), MpiError> {
        let ndim = self.dims.len();
        let n0 = self.dims[0];
        let stride: usize = self.dims[1..].iter().product();
        let pool = self.pool;

        let trans_partition = LocalPartition::new(stride, pool.size(), pool.rank());
        let local_stride = trans_partition.local_n;
        let transposed_size = n0 * local_stride;
        let normal_size = self.local_n0 * stride;

        // Step 1: local 1-D FFTs along the contiguous n0 fibers.
        {
            let plan_n0 = &self.local_plans[0];
            let mut fiber_in = vec![Complex::<T>::zero(); n0];
            let mut fiber_out = vec![Complex::<T>::zero(); n0];
            for col in 0..local_stride {
                let base = col * n0;
                fiber_in.copy_from_slice(&data[base..base + n0]);
                plan_n0.execute(&fiber_in, &mut fiber_out);
                data[base..base + n0].copy_from_slice(&fiber_out);
            }
        }

        // Step 2: transpose `[local_stride][n0]` -> `[local_n0][stride]`.
        distributed_transpose(
            pool,
            &data[..transposed_size],
            &mut self.scratch,
            stride,
            n0,
            local_stride,
            trans_partition.local_start,
        )?;
        data[..normal_size].copy_from_slice(&self.scratch[..normal_size]);

        // Step 3: FFTs along every local dimension. Empty for a 1-D problem.
        for d in (1..ndim).rev() {
            self.fft_along_dimension(data, d)?;
        }

        // Step 4: emit in the requested output distribution.
        if self.flags.transposed_out {
            distributed_transpose(
                pool,
                &data[..normal_size],
                &mut self.scratch,
                n0,
                stride,
                self.local_n0,
                self.local_0_start,
            )?;
            data[..transposed_size].copy_from_slice(&self.scratch[..transposed_size]);
        }

        Ok(())
    }

    /// Execute FFT along a local dimension (not dimension 0).
    #[allow(clippy::needless_pass_by_ref_mut)] // reason: &mut self needed for future multi-rank transpose plans; API consistency
    fn fft_along_dimension(&mut self, data: &mut [Complex<T>], dim: usize) -> Result<(), MpiError> {
        let n_dim = self.dims[dim];
        let plan = &self.local_plans[dim];

        // Calculate strides
        let inner_product: usize = self.dims[dim + 1..].iter().product();
        let outer_product: usize = self.local_n0 * self.dims[1..dim].iter().product::<usize>();

        let mut buffer = vec![Complex::<T>::zero(); n_dim];
        let mut output = vec![Complex::<T>::zero(); n_dim];

        for outer in 0..outer_product {
            for inner in 0..inner_product {
                // Gather elements along this dimension
                for i in 0..n_dim {
                    let idx = outer * self.dims[dim..].iter().product::<usize>()
                        + i * inner_product
                        + inner;
                    buffer[i] = data[idx];
                }

                // FFT
                plan.execute(&buffer, &mut output);

                // Scatter back
                for i in 0..n_dim {
                    let idx = outer * self.dims[dim..].iter().product::<usize>()
                        + i * inner_product
                        + inner;
                    data[idx] = output[i];
                }
            }
        }

        Ok(())
    }

    /// Distributed FFT along dimension 0.
    ///
    /// The first dimension is the distributed one, so every length-`n0` "fiber"
    /// (there are `stride == product(dims[1..])` of them) straddles all ranks and
    /// must be gathered before it can be transformed. Rather than issuing one
    /// blocking `all_gather` per fiber — `O(stride)` collectives, each replicating
    /// the whole fiber onto every rank — this packs *all* `stride` fibers into a
    /// single `alltoallv`-based transpose, transforms them locally, then transposes
    /// back with a second `alltoallv`.
    ///
    /// The trailing `stride` fibers are treated as one flat second dimension, so
    /// the classic `transpose -> local 1-D FFT along n0 -> transpose-back` pipeline
    /// (identical in spirit to the batched transpose driving
    /// [`MpiPlan3D`](crate::MpiPlan3D)) applies
    /// directly: exactly two collectives regardless of `stride`, and each rank only
    /// ever materialises its own `1/P` share of the transposed data.
    fn distributed_fft_dim0(
        &mut self,
        data: &mut [Complex<T>],
        pool: &MpiPool<C>,
    ) -> Result<(), MpiError> {
        let n0 = self.dims[0];
        let stride: usize = self.dims[1..].iter().product();
        let local_n0 = self.local_n0;
        let local_0_start = self.local_0_start;

        // Partition of the flat `stride` dimension owned by this rank after the
        // forward transpose.
        let trans_partition = LocalPartition::new(stride, pool.size(), pool.rank());
        let local_stride = trans_partition.local_n;
        let local_stride_start = trans_partition.local_start;
        let transposed_len = n0 * local_stride;

        // Step 1: transpose `[local_n0][stride]` -> `[local_stride][n0]`, packing
        // every fiber into a single alltoallv.
        distributed_transpose(
            pool,
            data,
            &mut self.scratch,
            n0,
            stride,
            local_n0,
            local_0_start,
        )?;

        // Step 2: local 1-D FFTs along the (now contiguous) n0 axis. `self.scratch`
        // and `self.local_plans` are disjoint fields, so a split borrow keeps the
        // plan reference and the mutable scratch slice live simultaneously.
        let plan_n0 = &self.local_plans[0];
        let scratch = &mut self.scratch;
        let mut fiber_in = vec![Complex::<T>::zero(); n0];
        let mut fiber_out = vec![Complex::<T>::zero(); n0];
        for col in 0..local_stride {
            let base = col * n0;
            fiber_in.copy_from_slice(&scratch[base..base + n0]);
            plan_n0.execute(&fiber_in, &mut fiber_out);
            scratch[base..base + n0].copy_from_slice(&fiber_out);
        }

        // Step 3: transpose back `[local_stride][n0]` -> `[local_n0][stride]` with a
        // second alltoallv, restoring the original slab layout in `data` — unless
        // the caller asked for transposed output, in which case the transposed
        // layout already in `scratch` *is* the requested result and the second
        // collective is skipped entirely.
        if self.flags.transposed_out {
            data[..transposed_len].copy_from_slice(&self.scratch[..transposed_len]);
        } else {
            distributed_transpose(
                pool,
                &self.scratch[..transposed_len],
                data,
                stride,
                n0,
                local_stride,
                local_stride_start,
            )?;
        }

        Ok(())
    }

    /// Execute the distributed FFT out-of-place.
    ///
    /// # Errors
    /// Returns `MpiError::SizeMismatch` if input buffer is too small.
    pub fn execute(
        &mut self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
    ) -> Result<(), MpiError> {
        let (in_size, out_size) = self.local_footprints();

        if input.len() < in_size {
            return Err(MpiError::SizeMismatch {
                expected: in_size,
                actual: input.len(),
            });
        }
        if output.len() < in_size.max(out_size) {
            return Err(MpiError::SizeMismatch {
                expected: in_size.max(out_size),
                actual: output.len(),
            });
        }

        output[..in_size].copy_from_slice(&input[..in_size]);
        self.execute_inplace(output)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_nd_dimensions() {
        // Test dimension calculations without MPI
        let dims = [16, 8, 4];
        let remaining: usize = dims[1..].iter().product();
        assert_eq!(remaining, 32);
    }
}
