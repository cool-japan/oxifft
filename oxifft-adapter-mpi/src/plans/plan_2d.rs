//! 2D distributed FFT plan.

use mpi::topology::Communicator;

use oxifft::api::{Direction, Plan};
use oxifft::kernel::{Complex, Float};

use crate::distribution::{Distribution, LocalPartition};
use crate::error::MpiError;
use crate::pool::{MpiFloat, MpiPool};
use crate::transpose::distributed_transpose;
use crate::MpiFlags;

/// 2D distributed FFT plan.
///
/// Implements the classic four-step algorithm:
/// 1. Local row-wise FFTs
/// 2. Distributed transpose
/// 3. Local column-wise FFTs (now rows after transpose)
/// 4. Optional: Distributed transpose back (unless TRANSPOSED_OUT)
///
/// # Transposed layouts
///
/// [`MpiFlags::transposed_out`](crate::MpiFlags::transposed_out) leaves the
/// result distributed over `n1` in `[local_n1][n0]` order (step 4 skipped), and
/// [`MpiFlags::transposed_in`](crate::MpiFlags::transposed_in) declares that the
/// *input* already arrives in that same `[local_n1][n0]` distribution. With
/// `transposed_in` the pipeline is mirrored — the `n0` axis (contiguous and
/// complete locally) is transformed first, then one transpose restores the slab
/// layout for the `n1` transforms — so a `transposed_out` forward followed by a
/// `transposed_in` inverse costs one `alltoallv` per transform instead of two.
pub struct MpiPlan2D<'p, T: Float, C: Communicator> {
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
    /// Borrow of the MPI pool; the borrow checker guarantees the pool outlives
    /// the plan.
    pool: &'p MpiPool<C>,
    /// Local plan for row transforms (size n1).
    row_plan: Plan<T>,
    /// Local plan for column transforms (size n0, after transpose).
    col_plan: Plan<T>,
    /// Scratch buffer for transpose.
    scratch: Vec<Complex<T>>,
}

impl<'p, T: Float + MpiFloat, C: Communicator> MpiPlan2D<'p, T, C> {
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
        pool: &'p MpiPool<C>,
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
            pool,
            row_plan,
            col_plan,
            scratch,
        })
    }

    /// Data-distribution strategy used by this plan (always [`Distribution::Slab`]).
    pub fn distribution(&self) -> Distribution {
        Distribution::Slab
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

    /// Local element counts this plan reads and writes, `(input, output)`.
    ///
    /// `transposed_in` / `transposed_out` each swap the corresponding side to the
    /// `[local_n1][n0]` footprint, so the two can differ; `execute_inplace`
    /// requires a buffer at least as large as their maximum.
    pub fn local_footprints(&self) -> (usize, usize) {
        let local_n1 = LocalPartition::new(self.n1, self.pool.size(), self.pool.rank()).local_n;
        let normal = self.local_n0 * self.n1;
        let transposed = local_n1 * self.n0;
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
    /// Input layout: `data[row * n1 + col]` where `row` is local (0..local_n0),
    /// or `data[local_col * n0 + row]` when `transposed_in` is set.
    /// Output layout depends on the `transposed_out` flag.
    ///
    /// # Errors
    /// Returns `MpiError::SizeMismatch` if data buffer is too small.
    pub fn execute_inplace(&mut self, data: &mut [Complex<T>]) -> Result<(), MpiError> {
        let pool = self.pool;

        // A transposed side writes/reads `local_n1 * n0` elements, which can
        // exceed the `local_n0 * n1` slab footprint. Validate the true worst case
        // up front so an undersized buffer becomes an error rather than a
        // slice-index panic later.
        let transposed_partition = LocalPartition::new(self.n1, pool.size(), pool.rank());
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
            return self.execute_transposed_in(data, transposed_partition);
        }

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
        // (`transposed_partition`/`local_n1` were computed above for buffer sizing.)
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

    /// Transposed-input pipeline: the caller's `data` already holds the
    /// `[local_n1][n0]` distribution that `transposed_out` produces, carrying
    /// *untransformed* values.
    ///
    /// Mirrors the normal pipeline: the `n0` axis is contiguous and complete on
    /// this rank, so it is transformed first; a single distributed transpose then
    /// restores the `[local_n0][n1]` slab for the `n1` transforms. When
    /// `transposed_out` is also set a second transpose puts the result back into
    /// the transposed distribution.
    fn execute_transposed_in(
        &mut self,
        data: &mut [Complex<T>],
        transposed_partition: LocalPartition,
    ) -> Result<(), MpiError> {
        let pool = self.pool;
        let (n0, n1) = (self.n0, self.n1);
        let local_n1 = transposed_partition.local_n;
        let transposed_size = local_n1 * n0;
        let normal_size = self.local_n0 * n1;

        // Step 1: local FFTs along n0, which is contiguous in this layout.
        let mut fiber = vec![Complex::<T>::zero(); n0];
        for col in 0..local_n1 {
            let base = col * n0;
            fiber.copy_from_slice(&data[base..base + n0]);
            self.col_plan.execute(&fiber, &mut data[base..base + n0]);
        }

        // Step 2: distributed transpose `[local_n1][n0]` -> `[local_n0][n1]`.
        distributed_transpose(
            pool,
            &data[..transposed_size],
            &mut self.scratch,
            n1,
            n0,
            local_n1,
            transposed_partition.local_start,
        )?;

        // Step 3: local FFTs along n1 (contiguous rows of the restored slab).
        let mut row_buffer = vec![Complex::<T>::zero(); n1];
        for row in 0..self.local_n0 {
            let base = row * n1;
            row_buffer.copy_from_slice(&self.scratch[base..base + n1]);
            self.row_plan
                .execute(&row_buffer, &mut self.scratch[base..base + n1]);
        }

        // Step 4: emit in the requested output distribution.
        if self.flags.transposed_out {
            distributed_transpose(
                pool,
                &self.scratch[..normal_size],
                data,
                n0,
                n1,
                self.local_n0,
                self.local_0_start,
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

        // Copy input to output and execute in-place. `execute_inplace` repeats the
        // worst-case size validation on `output`.
        output[..in_size].copy_from_slice(&input[..in_size]);
        self.execute_inplace(output)
    }
}

#[cfg(test)]
mod tests {
    // MPI tests require MPI runtime, so we only test non-MPI parts here

    #[test]
    fn test_local_partition() {
        use crate::distribution::LocalPartition;

        // Test partition calculation
        let p = LocalPartition::new(16, 4, 0);
        assert_eq!(p.local_n, 4);
        assert_eq!(p.local_start, 0);
    }
}
