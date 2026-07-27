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

        if flags.transposed_in {
            return Err(MpiError::FftError {
                message: "MpiFlags::transposed_in is not yet implemented for the N-D slab plan"
                    .to_string(),
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

    /// Execute the distributed N-D FFT in-place.
    ///
    /// For dimensions 1-3, delegates to optimized implementations.
    /// For higher dimensions, uses a general row-major traversal.
    ///
    /// # Errors
    /// Returns `MpiError::SizeMismatch` if data buffer is too small.
    pub fn execute_inplace(&mut self, data: &mut [Complex<T>]) -> Result<(), MpiError> {
        let ndim = self.dims.len();

        // Calculate expected local size
        let remaining_product: usize = self.dims[1..].iter().product();
        let expected_size = self.local_n0 * remaining_product;

        if data.len() < expected_size {
            return Err(MpiError::SizeMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }

        let pool = self.pool;

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
        // second alltoallv, restoring the original slab layout in `data`.
        distributed_transpose(
            pool,
            &self.scratch[..transposed_len],
            data,
            stride,
            n0,
            local_stride,
            local_stride_start,
        )?;

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
        let remaining_product: usize = self.dims[1..].iter().product();
        let expected_size = self.local_n0 * remaining_product;

        if input.len() < expected_size {
            return Err(MpiError::SizeMismatch {
                expected: expected_size,
                actual: input.len(),
            });
        }
        if output.len() < expected_size {
            return Err(MpiError::SizeMismatch {
                expected: expected_size,
                actual: output.len(),
            });
        }

        output[..expected_size].copy_from_slice(&input[..expected_size]);
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
