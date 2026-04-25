//! 3D distributed FFT plan.

use mpi::datatype::Equivalence;
use mpi::topology::Communicator;

use crate::api::{Direction, Plan};
use crate::kernel::{Complex, Float};

use crate::mpi::distribution::LocalPartition;
use crate::mpi::error::MpiError;
use crate::mpi::pool::{MpiFloat, MpiPool};
use crate::mpi::transpose::distributed_transpose;
use crate::mpi::MpiFlags;

/// 3D distributed FFT plan.
///
/// Uses slab decomposition: distributes the first dimension across processes.
/// Algorithm:
/// 1. Local 2D FFTs on (n1, n2) planes
/// 2. Distributed transpose to distribute n1
/// 3. Local 1D FFTs along n0 dimension
/// 4. Optional: Transpose back
pub struct MpiPlan3D<T: Float, C: Communicator> {
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
    /// Reference to MPI pool.
    pool: *const MpiPool<C>,
    /// Local plan for n2 dimension.
    plan_n2: Plan<T>,
    /// Local plan for n1 dimension.
    plan_n1: Plan<T>,
    /// Local plan for n0 dimension.
    plan_n0: Plan<T>,
    /// Scratch buffer.
    scratch: Vec<Complex<T>>,
    /// Marker.
    _phantom: core::marker::PhantomData<(T, C)>,
}

unsafe impl<T: Float, C: Communicator + Send> Send for MpiPlan3D<T, C> {}
unsafe impl<T: Float, C: Communicator + Sync> Sync for MpiPlan3D<T, C> {}

impl<T: Float + MpiFloat, C: Communicator> MpiPlan3D<T, C>
where
    Complex<T>: Equivalence,
{
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
        pool: &MpiPool<C>,
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
            pool: std::ptr::from_ref(pool),
            plan_n2,
            plan_n1,
            plan_n0,
            scratch,
            _phantom: core::marker::PhantomData,
        })
    }

    /// Get global dimensions.
    pub fn dims(&self) -> [usize; 3] {
        self.dims
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

        let expected_size = self.local_n0 * n1 * n2;
        if data.len() < expected_size {
            return Err(MpiError::SizeMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }

        let pool = unsafe { &*self.pool };

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

        // Step 3: Distributed transpose (distribute n1, gather n0)
        // This is complex for 3D, we transpose the (n0, n1) plane while keeping n2 local
        // For simplicity, we'll do a 2D transpose for each n2 slice
        let transposed_partition = LocalPartition::new(n1, pool.size(), pool.rank());
        let local_n1 = transposed_partition.local_n;

        // Transpose each n2 slice
        for i2 in 0..n2 {
            // Extract slice
            let mut slice_in = vec![Complex::<T>::zero(); self.local_n0 * n1];
            for i0 in 0..self.local_n0 {
                for i1 in 0..n1 {
                    slice_in[i0 * n1 + i1] = data[i0 * n1 * n2 + i1 * n2 + i2];
                }
            }

            // Transpose
            let mut slice_out = vec![Complex::<T>::zero(); n0 * local_n1];
            distributed_transpose(
                pool,
                &slice_in,
                &mut slice_out,
                n0,
                n1,
                self.local_n0,
                self.local_0_start,
            )?;

            // Store in scratch
            for i1_local in 0..local_n1 {
                for i0 in 0..n0 {
                    self.scratch[i1_local * n0 * n2 + i0 * n2 + i2] = slice_out[i1_local * n0 + i0];
                }
            }
        }

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

        // Step 5: Transpose back (unless TRANSPOSED_OUT)
        if !self.flags.transposed_out {
            for i2 in 0..n2 {
                // Extract transposed slice
                let mut slice_in = vec![Complex::<T>::zero(); local_n1 * n0];
                for i1_local in 0..local_n1 {
                    for i0 in 0..n0 {
                        slice_in[i1_local * n0 + i0] =
                            self.scratch[i1_local * n0 * n2 + i0 * n2 + i2];
                    }
                }

                // Transpose back
                let mut slice_out = vec![Complex::<T>::zero(); self.local_n0 * n1];
                distributed_transpose(
                    pool,
                    &slice_in,
                    &mut slice_out,
                    n1,
                    n0,
                    local_n1,
                    transposed_partition.local_start,
                )?;

                // Store back
                for i0 in 0..self.local_n0 {
                    for i1 in 0..n1 {
                        data[i0 * n1 * n2 + i1 * n2 + i2] = slice_out[i0 * n1 + i1];
                    }
                }
            }
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

        output[..expected_size].copy_from_slice(&input[..expected_size]);
        self.execute_inplace(output)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_3d_partition() {
        use crate::mpi::distribution::LocalPartition;

        let p = LocalPartition::new(32, 4, 0);
        assert_eq!(p.local_n, 8);
        assert_eq!(p.local_start, 0);
    }
}
