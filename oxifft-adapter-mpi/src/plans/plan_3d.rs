//! 3D distributed FFT plan.

use mpi::topology::Communicator;

use oxifft::api::{Direction, Plan};
use oxifft::kernel::{Complex, Float};

use crate::distribution::{Distribution, LocalPartition};
use crate::error::MpiError;
use crate::pool::{MpiFloat, MpiPool};
use crate::transpose::distributed_transpose_batched;
use crate::MpiFlags;

/// 3D distributed FFT plan.
///
/// Uses slab decomposition: distributes the first dimension across processes.
/// Algorithm:
/// 1. Local 2D FFTs on (n1, n2) planes
/// 2. Distributed transpose to distribute n1
/// 3. Local 1D FFTs along n0 dimension
/// 4. Optional: Transpose back
pub struct MpiPlan3D<'p, T: Float, C: Communicator> {
    /// Global dimensions.
    dims: [usize; 3],
    /// Local number of planes owned by this process.
    local_n0: usize,
    /// Global starting plane for this process.
    local_0_start: usize,
    /// Transform direction.
    direction: Direction,
    /// Planning flags.
    flags: MpiFlags,
    /// Borrow of the MPI pool; the borrow checker guarantees the pool outlives
    /// the plan.
    pool: &'p MpiPool<C>,
    /// Local plan for n2 dimension.
    plan_n2: Plan<T>,
    /// Local plan for n1 dimension.
    plan_n1: Plan<T>,
    /// Local plan for n0 dimension.
    plan_n0: Plan<T>,
    /// Scratch buffer.
    scratch: Vec<Complex<T>>,
}

impl<'p, T: Float + MpiFloat, C: Communicator> MpiPlan3D<'p, T, C> {
    /// Create a new 3D distributed FFT plan.
    ///
    /// # Arguments
    /// * `n0` - First dimension (distributed)
    /// * `n1` - Second dimension
    /// * `n2` - Third dimension
    /// * `direction` - Transform direction
    /// * `flags` - Planning flags
    /// * `pool` - MPI pool
    ///
    /// # Errors
    ///
    /// Returns `MpiError::InvalidDimension` if any dimension is zero.
    pub fn new(
        n0: usize,
        n1: usize,
        n2: usize,
        direction: Direction,
        flags: MpiFlags,
        pool: &'p MpiPool<C>,
    ) -> Result<Self, MpiError> {
        if n0 == 0 || n1 == 0 || n2 == 0 {
            return Err(MpiError::InvalidDimension {
                dim: if n0 == 0 {
                    0
                } else if n1 == 0 {
                    1
                } else {
                    2
                },
                size: if n0 == 0 {
                    n0
                } else if n1 == 0 {
                    n1
                } else {
                    n2
                },
                message: "Dimension size cannot be zero".to_string(),
            });
        }

        if flags.transposed_in {
            return Err(MpiError::FftError {
                message: "MpiFlags::transposed_in is not yet implemented for the 3D slab plan"
                    .to_string(),
            });
        }

        let partition = pool.local_partition(n0);
        let local_n0 = partition.local_n;
        let local_0_start = partition.local_start;

        // Calculate scratch size
        let transposed_partition = LocalPartition::new(n1, pool.size(), pool.rank());
        let normal_size = local_n0 * n1 * n2;
        let transposed_size = n0 * transposed_partition.local_n * n2;
        let scratch_size = normal_size.max(transposed_size);

        // Create local 1D plans
        let plan_n2 =
            Plan::dft_1d(n2, direction, flags.base).ok_or_else(|| MpiError::FftError {
                message: format!("Failed to create n2 plan for size {n2}"),
            })?;

        let plan_n1 =
            Plan::dft_1d(n1, direction, flags.base).ok_or_else(|| MpiError::FftError {
                message: format!("Failed to create n1 plan for size {n1}"),
            })?;

        let plan_n0 =
            Plan::dft_1d(n0, direction, flags.base).ok_or_else(|| MpiError::FftError {
                message: format!("Failed to create n0 plan for size {n0}"),
            })?;

        let scratch = vec![Complex::<T>::zero(); scratch_size];

        Ok(Self {
            dims: [n0, n1, n2],
            local_n0,
            local_0_start,
            direction,
            flags,
            pool,
            plan_n2,
            plan_n1,
            plan_n0,
            scratch,
        })
    }

    /// Get global dimensions.
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Data-distribution strategy used by this plan (always [`Distribution::Slab`]).
    pub fn distribution(&self) -> Distribution {
        Distribution::Slab
    }

    /// Get the transform direction.
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Get local dimensions.
    pub fn local_dims(&self) -> (usize, usize, usize, usize) {
        (
            self.local_n0,
            self.local_0_start,
            self.dims[1],
            self.dims[2],
        )
    }

    /// Execute the distributed FFT in-place.
    ///
    /// Input layout: `data[i0 * n1 * n2 + i1 * n2 + i2]` where `i0` is local.
    ///
    /// # Errors
    /// Returns `MpiError::SizeMismatch` if data buffer is too small.
    pub fn execute_inplace(&mut self, data: &mut [Complex<T>]) -> Result<(), MpiError> {
        let n0 = self.dims[0];
        let n1 = self.dims[1];
        let n2 = self.dims[2];

        let pool = self.pool;

        // The transposed-output path writes `local_n1 * n0 * n2` elements into
        // `data`, which can exceed the `local_n0 * n1 * n2` input footprint.
        // Validate the worst case up front to convert a would-be slice-index
        // panic into an error.
        let transposed_partition = LocalPartition::new(n1, pool.size(), pool.rank());
        let local_n1 = transposed_partition.local_n;
        let in_size = self.local_n0 * n1 * n2;
        let out_size = if self.flags.transposed_out {
            local_n1 * n0 * n2
        } else {
            in_size
        };
        let required = in_size.max(out_size);
        if data.len() < required {
            return Err(MpiError::SizeMismatch {
                expected: required,
                actual: data.len(),
            });
        }

        // Step 1: Local FFTs along n2 (innermost, always local)
        let mut buffer_n2 = vec![Complex::<T>::zero(); n2];
        for i0 in 0..self.local_n0 {
            for i1 in 0..n1 {
                let offset = i0 * n1 * n2 + i1 * n2;
                buffer_n2.copy_from_slice(&data[offset..offset + n2]);
                self.plan_n2
                    .execute(&buffer_n2, &mut data[offset..offset + n2]);
            }
        }

        // Step 2: Local FFTs along n1
        let mut buffer_n1 = vec![Complex::<T>::zero(); n1];
        for i0 in 0..self.local_n0 {
            for i2 in 0..n2 {
                // Gather along n1 dimension
                for i1 in 0..n1 {
                    buffer_n1[i1] = data[i0 * n1 * n2 + i1 * n2 + i2];
                }
                // FFT
                let mut output_n1 = vec![Complex::<T>::zero(); n1];
                self.plan_n1.execute(&buffer_n1, &mut output_n1);
                // Scatter back
                for i1 in 0..n1 {
                    data[i0 * n1 * n2 + i1 * n2 + i2] = output_n1[i1];
                }
            }
        }

        // Step 3: Distributed transpose (distribute n1, gather n0), batched over
        // the local n2 dimension so all n2 planes go through a single alltoallv
        // collective instead of one per plane.
        // `data` is already in `[local_n0][n1][n2]` layout and the batched
        // transpose writes directly into scratch as `[local_n1][n0][n2]`.
        // (`transposed_partition`/`local_n1` were computed above for buffer sizing.)
        distributed_transpose_batched(pool, data, &mut self.scratch, n0, n1, self.local_n0, n2)?;

        // Step 4: Local FFTs along n0 (now fully local after transpose)
        let mut buffer_n0 = vec![Complex::<T>::zero(); n0];
        for i1_local in 0..local_n1 {
            for i2 in 0..n2 {
                // Gather along n0
                for i0 in 0..n0 {
                    buffer_n0[i0] = self.scratch[i1_local * n0 * n2 + i0 * n2 + i2];
                }
                // FFT
                let mut output_n0 = vec![Complex::<T>::zero(); n0];
                self.plan_n0.execute(&buffer_n0, &mut output_n0);
                // Scatter back
                for i0 in 0..n0 {
                    self.scratch[i1_local * n0 * n2 + i0 * n2 + i2] = output_n0[i0];
                }
            }
        }

        // Step 5: Transpose back (unless TRANSPOSED_OUT), again batched over n2:
        // scratch `[local_n1][n0][n2]` -> data `[local_n0][n1][n2]` in one
        // alltoallv collective.
        if !self.flags.transposed_out {
            distributed_transpose_batched(pool, &self.scratch, data, n1, n0, local_n1, n2)?;
        } else {
            // Copy transposed result to output
            let transposed_size = local_n1 * n0 * n2;
            data[..transposed_size].copy_from_slice(&self.scratch[..transposed_size]);
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
        let expected_size = self.local_n0 * self.dims[1] * self.dims[2];
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
    fn test_3d_partition() {
        use crate::distribution::LocalPartition;

        let p = LocalPartition::new(32, 4, 0);
        assert_eq!(p.local_n, 8);
        assert_eq!(p.local_start, 0);
    }
}
