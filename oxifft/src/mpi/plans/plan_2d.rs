//! 2D distributed FFT plan.

use mpi::datatype::Equivalence;
use mpi::topology::Communicator;

use crate::api::{Direction, Plan};
use crate::kernel::{Complex, Float};

use crate::mpi::distribution::LocalPartition;
use crate::mpi::error::MpiError;
use crate::mpi::pool::{MpiFloat, MpiPool};
use crate::mpi::transpose::distributed_transpose;
use crate::mpi::MpiFlags;

/// 2D distributed FFT plan.
///
/// Implements the classic four-step algorithm:
/// 1. Local row-wise FFTs
/// 2. Distributed transpose
/// 3. Local column-wise FFTs (now rows after transpose)
/// 4. Optional: Distributed transpose back (unless TRANSPOSED_OUT)
pub struct MpiPlan2D<T: Float, C: Communicator> {
    /// Global number of rows.
    n0: usize,
    /// Global number of columns.
    n1: usize,
    /// Local number of rows owned by this process.
    local_n0: usize,
    /// Global starting row for this process.
    local_0_start: usize,
    /// Transform direction.
    direction: Direction,
    /// Planning flags.
    flags: MpiFlags,
    /// Reference to MPI pool.
    pool: *const MpiPool<C>,
    /// Local plan for row transforms (size n1).
    row_plan: Plan<T>,
    /// Local plan for column transforms (size n0, after transpose).
    col_plan: Plan<T>,
    /// Scratch buffer for transpose.
    scratch: Vec<Complex<T>>,
    /// Marker for T and C types.
    _phantom: core::marker::PhantomData<(T, C)>,
}

// Safety: The raw pointer to MpiPool is only used during execute, and
// the plan lifetime should not exceed the pool lifetime.
unsafe impl<T: Float, C: Communicator + Send> Send for MpiPlan2D<T, C> {}
unsafe impl<T: Float, C: Communicator + Sync> Sync for MpiPlan2D<T, C> {}

impl<T: Float + MpiFloat, C: Communicator> MpiPlan2D<T, C>
where
    Complex<T>: Equivalence,
{
    /// Create a new 2D distributed FFT plan.
    ///
    /// # Arguments
    /// * `n0` - Number of rows (distributed across processes)
    /// * `n1` - Number of columns (local to each process)
    /// * `direction` - Transform direction
    /// * `flags` - Planning flags
    /// * `pool` - MPI pool
    ///
    /// # Errors
    /// Returns error if dimensions are invalid or insufficient processes.
    pub fn new(
        n0: usize,
        n1: usize,
        direction: Direction,
        flags: MpiFlags,
        pool: &MpiPool<C>,
    ) -> Result<Self, MpiError> {
        if n0 == 0 || n1 == 0 {
            return Err(MpiError::InvalidDimension {
                dim: usize::from(n0 != 0),
                size: if n0 == 0 { n0 } else { n1 },
                message: "Dimension size cannot be zero".to_string(),
            });
        }

        // Calculate local partition
        let partition = pool.local_partition(n0);
        let local_n0 = partition.local_n;
        let local_0_start = partition.local_start;

        // Calculate transposed partition for scratch buffer
        let transposed_partition = LocalPartition::new(n1, pool.size(), pool.rank());
        let scratch_size = (local_n0 * n1).max(n0 * transposed_partition.local_n);

        // Create local 1D plans
        let row_plan =
            Plan::dft_1d(n1, direction, flags.base).ok_or_else(|| MpiError::FftError {
                message: format!("Failed to create row plan for size {n1}"),
            })?;

        let col_plan =
            Plan::dft_1d(n0, direction, flags.base).ok_or_else(|| MpiError::FftError {
                message: format!("Failed to create column plan for size {n0}"),
            })?;

        let scratch = vec![Complex::<T>::zero(); scratch_size];

        Ok(Self {
            n0,
            n1,
            local_n0,
            local_0_start,
            direction,
            flags,
            pool: std::ptr::from_ref(pool),
            row_plan,
            col_plan,
            scratch,
            _phantom: core::marker::PhantomData,
        })
    }

    /// Get global dimensions.
    pub fn dims(&self) -> (usize, usize) {
        (self.n0, self.n1)
    }

    /// Get local dimensions and start.
    pub fn local_dims(&self) -> (usize, usize, usize) {
        (self.local_n0, self.local_0_start, self.n1)
    }

    /// Get the transform direction.
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Execute the distributed FFT in-place.
    ///
    /// Input layout: `data[row * n1 + col]` where `row` is local (0..local_n0).
    /// Output layout depends on `transposed_out` flag.
    ///
    /// # Errors
    /// Returns `MpiError::SizeMismatch` if data buffer is too small.
    pub fn execute_inplace(&mut self, data: &mut [Complex<T>]) -> Result<(), MpiError> {
        let expected_size = self.local_n0 * self.n1;
        if data.len() < expected_size {
            return Err(MpiError::SizeMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }

        // Safety: we only access pool during execution
        let pool = unsafe { &*self.pool };

        // Step 1: Local row FFTs
        let mut row_buffer = vec![Complex::<T>::zero(); self.n1];
        for row in 0..self.local_n0 {
            let row_start = row * self.n1;
            row_buffer.copy_from_slice(&data[row_start..row_start + self.n1]);
            self.row_plan
                .execute(&row_buffer, &mut data[row_start..row_start + self.n1]);
        }

        // Step 2: Distributed transpose
        distributed_transpose(
            pool,
            data,
            &mut self.scratch,
            self.n0,
            self.n1,
            self.local_n0,
            self.local_0_start,
        )?;

        // Step 3: Local column FFTs (now stored as rows after transpose)
        // After transpose: scratch[local_col * n0 + global_row]
        // We need to FFT along the n0 dimension (columns of original, now contiguous)
        let transposed_partition = LocalPartition::new(self.n1, pool.size(), pool.rank());
        let local_n1 = transposed_partition.local_n;

        let mut col_buffer = vec![Complex::<T>::zero(); self.n0];
        for col in 0..local_n1 {
            // Extract column (now a row in transposed layout)
            let col_start = col * self.n0;
            col_buffer.copy_from_slice(&self.scratch[col_start..col_start + self.n0]);
            self.col_plan.execute(
                &col_buffer,
                &mut self.scratch[col_start..col_start + self.n0],
            );
        }

        // Step 4: Transpose back (unless TRANSPOSED_OUT)
        if !self.flags.transposed_out {
            // Transpose back: from column-distributed to row-distributed
            // This is the reverse transpose: n1 x n0 -> n0 x n1
            let temp = self.scratch.clone();
            distributed_transpose(
                pool,
                &temp,
                data,
                self.n1,
                self.n0,
                local_n1,
                transposed_partition.local_start,
            )?;
        } else {
            // Output in transposed layout
            let transposed_size = local_n1 * self.n0;
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
        let expected_size = self.local_n0 * self.n1;
        if input.len() < expected_size {
            return Err(MpiError::SizeMismatch {
                expected: expected_size,
                actual: input.len(),
            });
        }

        // Copy input to output and execute in-place
        output[..expected_size].copy_from_slice(&input[..expected_size]);
        self.execute_inplace(output)
    }
}

#[cfg(test)]
mod tests {
    // MPI tests require MPI runtime, so we only test non-MPI parts here

    #[test]
    fn test_local_partition() {
        use crate::mpi::distribution::LocalPartition;

        // Test partition calculation
        let p = LocalPartition::new(16, 4, 0);
        assert_eq!(p.local_n, 4);
        assert_eq!(p.local_start, 0);
    }
}
