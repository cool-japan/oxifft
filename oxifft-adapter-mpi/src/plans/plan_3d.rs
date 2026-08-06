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
///
/// # Transposed layouts
///
/// [`MpiFlags::transposed_out`](crate::MpiFlags::transposed_out) leaves the
/// result distributed over `n1` in `[local_n1][n0][n2]` order (step 4 skipped),
/// and [`MpiFlags::transposed_in`](crate::MpiFlags::transposed_in) declares that
/// the *input* already arrives in that distribution. With `transposed_in` the
/// pipeline is mirrored — `n0` is transformed first (it is complete locally),
/// then a single batched transpose restores the slab for the `n1`/`n2`
/// transforms — so a `transposed_out` forward followed by a `transposed_in`
/// inverse costs one `alltoallv` per transform instead of two.
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

    /// Local element counts this plan reads and writes, `(input, output)`.
    ///
    /// `transposed_in` / `transposed_out` each swap the corresponding side to the
    /// `[local_n1][n0][n2]` footprint, so the two can differ; `execute_inplace`
    /// requires a buffer at least as large as their maximum.
    pub fn local_footprints(&self) -> (usize, usize) {
        let [n0, n1, n2] = self.dims;
        let local_n1 = LocalPartition::new(n1, self.pool.size(), self.pool.rank()).local_n;
        let normal = self.local_n0 * n1 * n2;
        let transposed = local_n1 * n0 * n2;
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

    /// Execute the distributed FFT in-place.
    ///
    /// Input layout: `data[i0 * n1 * n2 + i1 * n2 + i2]` where `i0` is local, or
    /// `data[i1_local * n0 * n2 + i0 * n2 + i2]` when `transposed_in` is set.
    ///
    /// # Errors
    /// Returns `MpiError::SizeMismatch` if data buffer is too small.
    pub fn execute_inplace(&mut self, data: &mut [Complex<T>]) -> Result<(), MpiError> {
        let n0 = self.dims[0];
        let n1 = self.dims[1];
        let n2 = self.dims[2];

        let pool = self.pool;

        // A transposed side reads/writes `local_n1 * n0 * n2` elements, which can
        // exceed the `local_n0 * n1 * n2` slab footprint. Validate the worst case
        // up front to convert a would-be slice-index panic into an error.
        let transposed_partition = LocalPartition::new(n1, pool.size(), pool.rank());
        let local_n1 = transposed_partition.local_n;
        let (in_size, out_size) = self.local_footprints();
        let required = in_size.max(out_size);
        if data.len() < required {
            return Err(MpiError::SizeMismatch {
                expected: required,
                actual: data.len(),
            });
        }

        if self.flags.transposed_in {
            return self.execute_transposed_in(data, local_n1);
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

    /// Transposed-input pipeline: `data` already holds the `[local_n1][n0][n2]`
    /// distribution that `transposed_out` produces, carrying *untransformed*
    /// values.
    ///
    /// `n0` is complete on this rank in that layout, so it is transformed first
    /// (stride `n2` within each `i1` plane); one batched transpose then restores
    /// the `[local_n0][n1][n2]` slab for the `n1` and `n2` transforms. A second
    /// batched transpose runs only when `transposed_out` is also requested.
    fn execute_transposed_in(
        &mut self,
        data: &mut [Complex<T>],
        local_n1: usize,
    ) -> Result<(), MpiError> {
        let [n0, n1, n2] = self.dims;
        let pool = self.pool;
        let transposed_size = local_n1 * n0 * n2;
        let normal_size = self.local_n0 * n1 * n2;

        // Step 1: local FFTs along n0 (complete on this rank; stride n2).
        let mut fiber_in = vec![Complex::<T>::zero(); n0];
        let mut fiber_out = vec![Complex::<T>::zero(); n0];
        for i1_local in 0..local_n1 {
            let plane = i1_local * n0 * n2;
            for i2 in 0..n2 {
                for i0 in 0..n0 {
                    fiber_in[i0] = data[plane + i0 * n2 + i2];
                }
                self.plan_n0.execute(&fiber_in, &mut fiber_out);
                for i0 in 0..n0 {
                    data[plane + i0 * n2 + i2] = fiber_out[i0];
                }
            }
        }

        // Step 2: batched transpose `[local_n1][n0][n2]` -> `[local_n0][n1][n2]`.
        distributed_transpose_batched(
            pool,
            &data[..transposed_size],
            &mut self.scratch,
            n1,
            n0,
            local_n1,
            n2,
        )?;

        // Step 3: local FFTs along n1, then along n2, on the restored slab.
        let mut buffer_n1 = vec![Complex::<T>::zero(); n1];
        let mut output_n1 = vec![Complex::<T>::zero(); n1];
        for i0 in 0..self.local_n0 {
            for i2 in 0..n2 {
                for i1 in 0..n1 {
                    buffer_n1[i1] = self.scratch[i0 * n1 * n2 + i1 * n2 + i2];
                }
                self.plan_n1.execute(&buffer_n1, &mut output_n1);
                for i1 in 0..n1 {
                    self.scratch[i0 * n1 * n2 + i1 * n2 + i2] = output_n1[i1];
                }
            }
        }

        let mut buffer_n2 = vec![Complex::<T>::zero(); n2];
        for i0 in 0..self.local_n0 {
            for i1 in 0..n1 {
                let offset = i0 * n1 * n2 + i1 * n2;
                buffer_n2.copy_from_slice(&self.scratch[offset..offset + n2]);
                self.plan_n2
                    .execute(&buffer_n2, &mut self.scratch[offset..offset + n2]);
            }
        }

        // Step 4: emit in the requested output distribution.
        if self.flags.transposed_out {
            distributed_transpose_batched(
                pool,
                &self.scratch[..normal_size],
                data,
                n0,
                n1,
                self.local_n0,
                n2,
            )?;
        } else {
            data[..normal_size].copy_from_slice(&self.scratch[..normal_size]);
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
    fn test_3d_partition() {
        use crate::distribution::LocalPartition;

        let p = LocalPartition::new(32, 4, 0);
        assert_eq!(p.local_n, 8);
        assert_eq!(p.local_start, 0);
    }
}
