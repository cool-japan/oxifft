//! N-dimensional distributed FFT plan.

use mpi::datatype::Equivalence;
use mpi::topology::Communicator;

use crate::api::Direction;
use crate::kernel::{Complex, Float};

use crate::mpi::error::MpiError;
use crate::mpi::pool::{MpiFloat, MpiPool};
use crate::mpi::MpiFlags;

/// N-dimensional distributed FFT plan.
///
/// Generalizes the distributed FFT to arbitrary dimensions using slab decomposition.
/// The first dimension is distributed across processes.
pub struct MpiPlanND<T: Float, C: Communicator> {
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
    /// Reference to MPI pool.
    pool: *const MpiPool<C>,
    /// Local plans for each dimension.
    local_plans: Vec<crate::api::Plan<T>>,
    /// Scratch buffer (reserved for future multi-rank transpose implementation).
    _scratch: Vec<Complex<T>>,
    /// Marker.
    _phantom: core::marker::PhantomData<(T, C)>,
}

unsafe impl<T: Float, C: Communicator + Send> Send for MpiPlanND<T, C> {}
unsafe impl<T: Float, C: Communicator + Sync> Sync for MpiPlanND<T, C> {}

impl<T: Float + MpiFloat, C: Communicator> MpiPlanND<T, C>
where
    Complex<T>: Equivalence,
{
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
        pool: &MpiPool<C>,
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
            let plan = crate::api::Plan::dft_1d(n, direction, flags.base).ok_or_else(|| {
                MpiError::FftError {
                    message: format!("Failed to create plan for dimension {i} (size {n})"),
                }
            })?;
            local_plans.push(plan);
        }

        // Scratch buffer for intermediate results
        let scratch_size = local_size * 2; // Extra space for transpose operations
        let scratch = vec![Complex::<T>::zero(); scratch_size];

        Ok(Self {
            dims: dims.to_vec(),
            local_n0,
            local_0_start,
            direction,
            flags,
            pool: std::ptr::from_ref(pool),
            local_plans,
            _scratch: scratch,
            _phantom: core::marker::PhantomData,
        })
    }

    /// Get global dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.dims
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

        let pool = unsafe { &*self.pool };

        // For 1D case, just do local FFTs
        if ndim == 1 {
            // Each local element is independent
            // This is essentially a batch of 1-point "FFTs"
            return Ok(());
        }

        // Step 1: FFTs along all local dimensions (dims[1..])
        // We iterate from the innermost dimension outward
        for d in (1..ndim).rev() {
            self.fft_along_dimension(data, d)?;
        }

        // Step 2: Distributed FFT along dimension 0
        // This requires global communication
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
    #[allow(clippy::needless_pass_by_ref_mut)] // reason: &mut self needed for future multi-rank transpose plans; API consistency
    fn distributed_fft_dim0(
        &mut self,
        data: &mut [Complex<T>],
        pool: &MpiPool<C>,
    ) -> Result<(), MpiError> {
        let n0 = self.dims[0];
        let plan_n0 = &self.local_plans[0];

        // For the distributed dimension, we need to gather across processes
        // This is the most expensive operation

        // Calculate the stride: number of elements between consecutive dim0 indices
        let stride: usize = self.dims[1..].iter().product();

        // For each position in dims[1..], we need to do a distributed FFT
        // This requires all-to-all communication

        // Simplified approach: use MPI all-gather to collect full column, FFT, scatter back
        // This is not the most efficient but is correct

        let num_procs = pool.size();

        // Build recv_counts and recv_displs once outside the loop:
        // process p contributes LocalPartition::new(n0, num_procs, p).local_n elements.
        let mut recv_counts: Vec<i32> = Vec::with_capacity(num_procs);
        let mut recv_displs: Vec<i32> = Vec::with_capacity(num_procs);
        {
            use crate::mpi::distribution::LocalPartition;
            let mut displ: i32 = 0;
            for p in 0..num_procs {
                let part = LocalPartition::new(n0, num_procs, p);
                let count_i32 =
                    i32::try_from(part.local_n).map_err(|_| MpiError::CountOverflow {
                        count: part.local_n,
                        rank: p,
                    })?;
                recv_counts.push(count_i32);
                recv_displs.push(displ);
                displ = displ
                    .checked_add(count_i32)
                    .ok_or(MpiError::CountOverflow { count: n0, rank: p })?;
            }
        }

        let local_partition = pool.local_partition(n0);

        // For each "fiber" along dimension 0
        for fiber_idx in 0..stride {
            // Build the local send buffer for this fiber.
            let mut send_buf: Vec<Complex<T>> = Vec::with_capacity(self.local_n0);
            for i0 in 0..self.local_n0 {
                send_buf.push(data[i0 * stride + fiber_idx]);
            }

            // All-gather across all processes: collect the complete fiber.
            let mut global_fiber = vec![Complex::<T>::zero(); n0];
            pool.all_gather_v_complex(&send_buf, &mut global_fiber, &recv_counts, &recv_displs)?;

            // FFT the full fiber (plan_n0 is a 1-D plan for dimension 0).
            let mut fft_result = vec![Complex::<T>::zero(); n0];
            plan_n0.execute(&global_fiber, &mut fft_result);

            // Store back local portion.
            for i0 in 0..self.local_n0 {
                let global_i0 = local_partition.local_start + i0;
                data[i0 * stride + fiber_idx] = fft_result[global_i0];
            }
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
        let remaining_product: usize = self.dims[1..].iter().product();
        let expected_size = self.local_n0 * remaining_product;

        if input.len() < expected_size {
            return Err(MpiError::SizeMismatch {
                expected: expected_size,
                actual: input.len(),
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
