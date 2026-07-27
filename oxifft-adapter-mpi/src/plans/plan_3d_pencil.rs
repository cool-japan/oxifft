//! 3D distributed FFT using pencil decomposition.
//!
//! Pencil decomposition distributes the FFT across a 2D process grid
//! `P = P_row × P_col`, enabling scaling to more MPI ranks than slab decomposition.
//!
//! # Algorithm
//!
//! Given an `n0 x n1 x n2` 3D FFT on `P = P_row x P_col` ranks:
//!
//! **Forward:**
//! 1. Local Z-FFT (n2 fully local)
//! 2. Row-comm alltoallv: `[local_n0][local_n1][n2]` to `[local_n0][n1][local_n2']`
//!    where `local_n2' = part(n2, P_col, col_rank)`
//! 3. Local Y-FFT (n1 now fully local)
//! 4. Col-comm alltoallv: `[local_n0][n1][local_n2']` to `[n0][local_n1'][local_n2']`
//!    where `local_n1' = part(n1, P_row, row_rank)`
//! 5. Local X-FFT (n0 now fully local)
//!
//! **Inverse:**
//! Reverse: X-IFFT, col-comm transpose, Y-IFFT, row-comm transpose, Z-IFFT.
//!
//! # Data Layout
//!
//! On entry to `execute_inplace`, each rank owns a block of size
//! `local_n0 x local_n1 x n2`, stored in row-major order:
//! `data[i0 * local_n1 * n2 + i1 * n2 + i2]`
//! where `i0 in [0, local_n0)`, `i1 in [0, local_n1)`, `i2 in [0, n2)`.
//!
//! For a single-rank case (`P = 1`), the layout is `[n0][n1][n2]` and all
//! three FFT passes are applied locally with no MPI communication.
//!
//! # Multi-rank status
//!
//! Both single-rank and multi-rank execution are fully implemented.
//! For multi-rank plans, `execute_inplace` performs the full Z→Y→X pencil FFT
//! with two distributed alltoallv transposes through the row and column
//! sub-communicators. The output layout after a multi-rank transform is
//! `[local_n2_col][local_n1_row][n0]` (row-major), which differs from the
//! input layout `[local_n0][local_n1][n2]`.

use mpi::topology::{Color, Communicator, SimpleCommunicator};

use oxifft::api::{Direction, Flags, Plan};
use oxifft::kernel::{Complex, Float};

use crate::distribution::{Distribution, LocalPartition};
use crate::error::MpiError;
use crate::pool::{MpiFloat, MpiPool};
use crate::transpose::distributed_transpose;

/// 2D process grid configuration for pencil decomposition.
///
/// The total process count must satisfy `n_rows * n_cols == comm.size()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PencilGrid {
    /// Number of rows in the 2D process grid (P_row).
    pub n_rows: usize,
    /// Number of columns in the 2D process grid (P_col).
    pub n_cols: usize,
}

impl PencilGrid {
    /// Create a new pencil grid.
    ///
    /// Both `n_rows` and `n_cols` must be non-zero; [`row_rank`](Self::row_rank)
    /// and [`col_rank`](Self::col_rank) divide by `n_cols`, so a zero grid is a
    /// divide-by-zero footgun. This is enforced with a `debug_assert!`; callers
    /// that build a plan go through [`PencilPlan3D::new`], which additionally
    /// rejects any grid whose `total_procs()` does not match the pool size.
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        debug_assert!(
            n_rows > 0 && n_cols > 0,
            "PencilGrid dimensions must be non-zero (got {n_rows}x{n_cols})"
        );
        Self { n_rows, n_cols }
    }

    /// Total number of processes required (`n_rows * n_cols`).
    pub fn total_procs(&self) -> usize {
        self.n_rows * self.n_cols
    }

    /// Row rank for a given global rank (0-indexed row in the 2D grid).
    ///
    /// Returns 0 for a degenerate zero-column grid instead of panicking on a
    /// divide-by-zero.
    pub fn row_rank(&self, global_rank: usize) -> usize {
        if self.n_cols == 0 {
            return 0;
        }
        global_rank / self.n_cols
    }

    /// Column rank for a given global rank (0-indexed column in the 2D grid).
    ///
    /// Returns 0 for a degenerate zero-column grid instead of panicking on a
    /// divide-by-zero.
    pub fn col_rank(&self, global_rank: usize) -> usize {
        if self.n_cols == 0 {
            return 0;
        }
        global_rank % self.n_cols
    }
}

/// 3D distributed FFT plan using pencil decomposition.
///
/// Uses a 2D process grid `P_row x P_col` to distribute data across more MPI
/// ranks than slab decomposition. When `P = 1` (single process), pencil
/// degenerates to three sequential local 1D FFTs with no communication.
///
/// # Type parameters
///
/// - `T`: Float type (`f32` or `f64`)
/// - `C`: MPI communicator type
pub struct PencilPlan3D<'p, T: Float, C: Communicator> {
    /// Global dimensions `[n0, n1, n2]`.
    dims: [usize; 3],
    /// 2D process grid configuration.
    grid: PencilGrid,
    /// Row rank (position in n0 axis), 0..grid.n_rows.
    row_rank: usize,
    /// Column rank (position in n1 axis), 0..grid.n_cols.
    col_rank: usize,
    /// Local n0 slice owned by this process.
    local_n0: usize,
    /// Global starting index in n0 for this process.
    local_0_start: usize,
    /// Local n1 slice owned by this process.
    local_n1: usize,
    /// Global starting index in n1 for this process.
    local_1_start: usize,
    /// Transform direction (baked into the 1D plans; stored for multi-rank impl).
    direction: Direction,
    /// Local 1D FFT plan along X (n0).
    plan_x: Plan<T>,
    /// Local 1D FFT plan along Y (n1).
    plan_y: Plan<T>,
    /// Local 1D FFT plan along Z (n2).
    plan_z: Plan<T>,
    /// Pre-allocated scratch buffer for alltoall / transpose operations.
    ///
    /// Sized to hold `max(local_n0 * local_n1 * n2, local_n0 * n1 * local_n2_col,
    /// n0 * local_n1_row * local_n2_col)` elements.
    scratch: Vec<Complex<T>>,
    /// Number of complex elements the caller must allocate for the data buffer
    /// (the max over the input and all intermediate/output layouts).
    alloc_local: usize,
    /// Borrow of the global MPI pool; the borrow checker guarantees the pool
    /// outlives the plan. The row/column sub-communicators below are the ones
    /// actually used for communication.
    _pool: &'p MpiPool<C>,
    /// Row sub-communicator pool: all procs with the same row_rank, varying col_rank.
    /// `None` for single-rank plans.
    row_pool: Option<MpiPool<SimpleCommunicator>>,
    /// Column sub-communicator pool: all procs with the same col_rank, varying row_rank.
    /// `None` for single-rank plans.
    col_pool: Option<MpiPool<SimpleCommunicator>>,
}

impl<'p, T: Float + MpiFloat, C: Communicator> PencilPlan3D<'p, T, C> {
    /// Create a new 3D pencil-decomposition distributed FFT plan.
    ///
    /// # Arguments
    ///
    /// * `n0` - First global dimension (distributed across `grid.n_rows`)
    /// * `n1` - Second global dimension (distributed across `grid.n_cols`)
    /// * `n2` - Third global dimension (always local)
    /// * `grid` - 2D process grid -- must satisfy `grid.total_procs() == pool.size()`
    /// * `direction` - Transform direction (`Direction::Forward` or `Direction::Backward`)
    /// * `flags` - FFT planning flags (e.g., `Flags::ESTIMATE`)
    /// * `pool` - MPI pool wrapping the global communicator (must outlive this plan)
    ///
    /// # Errors
    ///
    /// - `MpiError::InvalidDimension` -- if any dimension is zero
    /// - `MpiError::InsufficientProcesses` -- if `grid.total_procs() != pool.size()`
    /// - `MpiError::CommunicationError` -- if sub-communicator creation fails
    /// - `MpiError::FftError` -- if any local 1D plan cannot be created
    pub fn new(
        n0: usize,
        n1: usize,
        n2: usize,
        grid: PencilGrid,
        direction: Direction,
        flags: Flags,
        pool: &'p MpiPool<C>,
    ) -> Result<Self, MpiError> {
        let dims = [n0, n1, n2];
        for (i, &d) in dims.iter().enumerate() {
            if d == 0 {
                return Err(MpiError::InvalidDimension {
                    dim: i,
                    size: d,
                    message: "Dimension size cannot be zero".to_string(),
                });
            }
        }

        if grid.total_procs() != pool.size() {
            return Err(MpiError::InsufficientProcesses {
                required: grid.total_procs(),
                available: pool.size(),
            });
        }

        let global_rank = pool.rank();
        let row_rank = grid.row_rank(global_rank);
        let col_rank = grid.col_rank(global_rank);

        let part0 = LocalPartition::new(n0, grid.n_rows, row_rank);
        let part1 = LocalPartition::new(n1, grid.n_cols, col_rank);
        let local_n0 = part0.local_n;
        let local_0_start = part0.local_start;
        let local_n1 = part1.local_n;
        let local_1_start = part1.local_start;

        let plan_x = Plan::dft_1d(n0, direction, flags).ok_or_else(|| MpiError::FftError {
            message: format!("Failed to create X (n0={n0}) plan"),
        })?;
        let plan_y = Plan::dft_1d(n1, direction, flags).ok_or_else(|| MpiError::FftError {
            message: format!("Failed to create Y (n1={n1}) plan"),
        })?;
        let plan_z = Plan::dft_1d(n2, direction, flags).ok_or_else(|| MpiError::FftError {
            message: format!("Failed to create Z (n2={n2}) plan"),
        })?;

        // Size the buffer for the largest of the three layouts the data buffer
        // passes through: the input `[local_n0][local_n1][n2]`, the post-row
        // layout `[local_n0][local_n2_col][n1]`, and the final post-col layout
        // `[local_n2_col][local_n1_row][n0]`. For non-divisible sizes an
        // intermediate/output layout can exceed the input footprint, so this is
        // the buffer the caller must allocate (see [`Self::alloc_local`]). The
        // scratch buffer is sized identically.
        let local_n2_col = LocalPartition::new(n2, grid.n_cols, col_rank).local_n;
        let local_n1_row = LocalPartition::new(n1, grid.n_rows, row_rank).local_n;
        let alloc_local = (local_n0 * local_n1 * n2)
            .max(local_n0 * n1 * local_n2_col)
            .max(n0 * local_n1_row * local_n2_col);
        let scratch = vec![Complex::<T>::zero(); alloc_local];

        // Create row and column sub-communicator pools (only if P > 1).
        let (row_pool, col_pool) = if pool.size() == 1 {
            (None, None)
        } else {
            let row_comm = pool
                .comm()
                .split_by_color(Color::with_value(row_rank as i32))
                .ok_or_else(|| MpiError::CommunicationError {
                    message: format!("Failed to create row sub-comm (row_rank={row_rank})"),
                })?;
            let col_comm = pool
                .comm()
                .split_by_color(Color::with_value(col_rank as i32))
                .ok_or_else(|| MpiError::CommunicationError {
                    message: format!("Failed to create col sub-comm (col_rank={col_rank})"),
                })?;
            (Some(MpiPool::new(row_comm)), Some(MpiPool::new(col_comm)))
        };

        Ok(Self {
            dims,
            grid,
            row_rank,
            col_rank,
            local_n0,
            local_0_start,
            local_n1,
            local_1_start,
            direction,
            plan_x,
            plan_y,
            plan_z,
            scratch,
            alloc_local,
            _pool: pool,
            row_pool,
            col_pool,
        })
    }

    /// Get global dimensions `[n0, n1, n2]`.
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Data-distribution strategy used by this plan (always [`Distribution::Pencil`]).
    pub fn distribution(&self) -> Distribution {
        Distribution::Pencil
    }

    /// Get the 2D process grid.
    pub fn grid(&self) -> PencilGrid {
        self.grid
    }

    /// Get the transform direction.
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Row rank in the 2D process grid (position along n0 axis).
    pub fn row_rank(&self) -> usize {
        self.row_rank
    }

    /// Column rank in the 2D process grid (position along n1 axis).
    pub fn col_rank(&self) -> usize {
        self.col_rank
    }

    /// Get local dimensions: `(local_n0, local_0_start, local_n1, local_1_start, n2)`.
    pub fn local_dims(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.local_n0,
            self.local_0_start,
            self.local_n1,
            self.local_1_start,
            self.dims[2],
        )
    }

    /// Returns a reference to the column sub-communicator pool, if present.
    ///
    /// `None` for single-rank plans; set for multi-rank pencil decomposition.
    pub fn col_pool(&self) -> Option<&MpiPool<SimpleCommunicator>> {
        self.col_pool.as_ref()
    }

    /// Number of complex elements the caller must allocate for the `data`
    /// buffer passed to [`execute_inplace`](Self::execute_inplace) / the output
    /// of [`execute`](Self::execute).
    ///
    /// Analogous to FFTW-MPI's `alloc_local`: for non-divisible sizes an
    /// intermediate or the final transposed layout can exceed the input
    /// footprint (`local_n0 * local_n1 * n2`), so the buffer must be sized to
    /// this value, not merely to the input.
    pub fn alloc_local(&self) -> usize {
        self.alloc_local
    }

    /// Execute the distributed 3D FFT in-place.
    ///
    /// Input layout: `data[i0 * local_n1 * n2 + i1 * n2 + i2]`
    /// where `i0 in [0, local_n0)`, `i1 in [0, local_n1)`, `i2 in [0, n2)`.
    ///
    /// For single-rank plans (`P = 1`), all data is local and no MPI
    /// communication is performed; the output keeps the `[n0][n1][n2]` layout.
    ///
    /// For multi-rank plans, this performs the full Z->Y->X pencil FFT with two
    /// distributed alltoallv transposes through the row and column
    /// sub-communicators (see `Self::execute_multirank`). The output is in the
    /// transposed layout `data[i2 * local_n1_row * n0 + i1 * n0 + i0]`, i.e.
    /// `[local_n2_col][local_n1_row][n0]` where `local_n2_col` partitions `n2`
    /// across `grid.n_cols` and `local_n1_row` partitions `n1` across
    /// `grid.n_rows`.
    ///
    /// # Errors
    ///
    /// - `MpiError::SizeMismatch` -- if `data.len() < self.alloc_local()`
    /// - `MpiError::CommunicationError` -- if a sub-communicator pool is missing
    /// - `MpiError::CountOverflow` -- if any element count exceeds `i32::MAX`
    pub fn execute_inplace(&mut self, data: &mut [Complex<T>]) -> Result<(), MpiError> {
        let [n0, n1, n2] = self.dims;
        // The buffer must hold every intermediate/output layout, not just the
        // input footprint (see `alloc_local`); for non-divisible sizes those
        // layouts can be larger.
        if data.len() < self.alloc_local {
            return Err(MpiError::SizeMismatch {
                expected: self.alloc_local,
                actual: data.len(),
            });
        }

        if self.row_pool.is_none() {
            // Single-rank: three sequential local 1D FFTs, no MPI.
            pure::fft_3d_zyx_with_plans(data, n0, n1, n2, &self.plan_x, &self.plan_y, &self.plan_z);
            Ok(())
        } else {
            self.execute_multirank(data, n0, n1, n2)
        }
    }

    /// Execute the multi-rank 3D pencil FFT (called when `row_pool.is_some()`).
    ///
    /// Algorithm (Z→Y→X):
    /// 1. Z-FFT: local, on the `[local_n0, local_n1, n2]` slab.
    /// 2. Row-comm alltoallv: redistributes n2 across col-ranks so each rank gets
    ///    `[local_n0, local_n2_col, n1]` (via `distributed_transpose` on each i0 slab).
    /// 3. Y-FFT: local, FFT over the contiguous n1 dimension.
    /// 4. Col-comm alltoallv: redistributes n1 across row-ranks so each rank gets
    ///    `[local_n2_col, local_n1_row, n0]` (via `distributed_transpose` on each i2 slab).
    /// 5. X-FFT: local, FFT over the contiguous n0 dimension.
    ///
    /// # Errors
    ///
    /// - `MpiError::CommunicationError` -- if sub-communicator pools are missing
    /// - `MpiError::CountOverflow` -- if element counts exceed `i32::MAX`
    /// - `MpiError::SizeMismatch` -- propagated from `distributed_transpose`
    fn execute_multirank(
        &mut self,
        data: &mut [Complex<T>],
        n0: usize,
        n1: usize,
        n2: usize,
    ) -> Result<(), MpiError> {
        let row_pool = self
            .row_pool
            .as_ref()
            .ok_or_else(|| MpiError::CommunicationError {
                message: "row_pool is None in multi-rank path".to_string(),
            })?;
        let col_pool = self
            .col_pool
            .as_ref()
            .ok_or_else(|| MpiError::CommunicationError {
                message: "col_pool is None in multi-rank path".to_string(),
            })?;

        let local_n0 = self.local_n0;
        let local_n1 = self.local_n1;
        let local_0_start = self.local_0_start;
        let local_1_start = self.local_1_start;

        // Local n2 slice for this col-rank (col-comm size = grid.n_cols = row_pool.size()).
        let local_n2_col = LocalPartition::new(n2, row_pool.size(), self.col_rank).local_n;
        // Local n1 slice for this row-rank after col transpose (col-comm size = grid.n_rows = col_pool.size()).
        let local_n1_row = LocalPartition::new(n1, col_pool.size(), self.row_rank).local_n;

        // ── Step 1: Z-FFT ────────────────────────────────────────────────────────
        // data layout: [local_n0][local_n1][n2]
        // FFT along n2 for each (i0, i1) row — contiguous access.
        {
            let mut tmp_in = vec![Complex::<T>::zero(); n2];
            let mut tmp_out = vec![Complex::<T>::zero(); n2];
            for i0 in 0..local_n0 {
                for i1 in 0..local_n1 {
                    let off = i0 * local_n1 * n2 + i1 * n2;
                    tmp_in.copy_from_slice(&data[off..off + n2]);
                    self.plan_z.execute(&tmp_in, &mut tmp_out);
                    data[off..off + n2].copy_from_slice(&tmp_out);
                }
            }
        }

        // ── Step 2: Row-comm alltoallv (redistribute n2) ─────────────────────────
        // For each i0, call distributed_transpose on the [local_n1, n2] slab.
        // Produces: [local_n0][local_n2_col][n1]  (stored as local_n2_col * n1 per i0)
        let after_row_size = local_n0 * local_n2_col * n1;
        // Use scratch for output of this stage.
        self.scratch[..after_row_size].fill(Complex::<T>::zero());
        {
            let slab_in_size = local_n1 * n2;
            let slab_out_size = local_n2_col * n1; // distributed_transpose output: [local_n2_col, n1]
                                                   // We need mutable borrow of both data and scratch — use split at boundary.
                                                   // Since scratch is a separate field, we can borrow both simultaneously.
            let scratch = &mut self.scratch;
            for i0 in 0..local_n0 {
                let in_off = i0 * slab_in_size;
                let out_off = i0 * slab_out_size;
                let input_slab = &data[in_off..in_off + slab_in_size];
                let output_slab = &mut scratch[out_off..out_off + slab_out_size];
                // distributed_transpose(row_pool, input, output, n0=n1, n1=n2, local_n0=local_n1, local_1_start)
                distributed_transpose(
                    row_pool,
                    input_slab,
                    output_slab,
                    n1,
                    n2,
                    local_n1,
                    local_1_start,
                )?;
            }
        }
        // Copy result back into data for the Y-FFT pass.
        // data layout after copy: [local_n0][local_n2_col][n1]
        data[..after_row_size].copy_from_slice(&self.scratch[..after_row_size]);

        // ── Step 3: Y-FFT ────────────────────────────────────────────────────────
        // data layout: [local_n0][local_n2_col][n1]
        // FFT along n1 for each (i0, i2) — contiguous access (n1 is innermost).
        {
            let mut tmp_in = vec![Complex::<T>::zero(); n1];
            let mut tmp_out = vec![Complex::<T>::zero(); n1];
            for i0 in 0..local_n0 {
                for i2 in 0..local_n2_col {
                    let off = i0 * local_n2_col * n1 + i2 * n1;
                    tmp_in.copy_from_slice(&data[off..off + n1]);
                    self.plan_y.execute(&tmp_in, &mut tmp_out);
                    data[off..off + n1].copy_from_slice(&tmp_out);
                }
            }
        }

        // ── Step 4: Col-comm alltoallv (redistribute n1) ─────────────────────────
        // For each i2 (local n2 index), call distributed_transpose on the [local_n0, n1] slab.
        // But data is currently [local_n0][local_n2_col][n1].
        // We need to extract [local_n0, n1] for each i2, but they are not contiguous.
        // Build a contiguous slab first, then transpose.
        // Output per-i2 slab: [local_n1_row, n0] (distributed_transpose output layout)
        // Full output: [local_n2_col][local_n1_row][n0]
        let after_col_size = local_n2_col * local_n1_row * n0;
        self.scratch[..after_col_size].fill(Complex::<T>::zero());
        {
            let mut slab_in = vec![Complex::<T>::zero(); local_n0 * n1];
            let mut slab_out = vec![Complex::<T>::zero(); local_n1_row * n0];
            for i2 in 0..local_n2_col {
                // Extract [local_n0, n1] slab for this i2: gather strided elements.
                for i0 in 0..local_n0 {
                    let src_off = i0 * local_n2_col * n1 + i2 * n1;
                    let dst_off = i0 * n1;
                    slab_in[dst_off..dst_off + n1].copy_from_slice(&data[src_off..src_off + n1]);
                }
                // distributed_transpose(col_pool, slab_in, slab_out, n0=n0, n1=n1, local_n0, local_0_start)
                distributed_transpose(
                    col_pool,
                    &slab_in,
                    &mut slab_out,
                    n0,
                    n1,
                    local_n0,
                    local_0_start,
                )?;
                // slab_out layout: [local_n1_row, n0] stored as output[j * n0 + global_i0]
                // Store into scratch: scratch[i2 * local_n1_row * n0 ..]
                let dst_off = i2 * local_n1_row * n0;
                self.scratch[dst_off..dst_off + local_n1_row * n0].copy_from_slice(&slab_out);
            }
        }
        // Copy back; final layout: [local_n2_col][local_n1_row][n0]
        data[..after_col_size].copy_from_slice(&self.scratch[..after_col_size]);

        // ── Step 5: X-FFT ────────────────────────────────────────────────────────
        // data layout: [local_n2_col][local_n1_row][n0]
        // FFT along n0 for each (i2, i1) — contiguous access (n0 is innermost).
        {
            let mut tmp_in = vec![Complex::<T>::zero(); n0];
            let mut tmp_out = vec![Complex::<T>::zero(); n0];
            for i2 in 0..local_n2_col {
                for i1 in 0..local_n1_row {
                    let off = i2 * local_n1_row * n0 + i1 * n0;
                    tmp_in.copy_from_slice(&data[off..off + n0]);
                    self.plan_x.execute(&tmp_in, &mut tmp_out);
                    data[off..off + n0].copy_from_slice(&tmp_out);
                }
            }
        }

        Ok(())
    }

    /// Execute the distributed 3D FFT out-of-place.
    ///
    /// Copies `input` to `output` then calls `execute_inplace`.
    ///
    /// # Errors
    ///
    /// - `MpiError::SizeMismatch` -- if either buffer is too small
    /// - `MpiError::FftError` -- propagated from `execute_inplace`
    pub fn execute(
        &mut self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
    ) -> Result<(), MpiError> {
        // `input` holds the input footprint; `output` must hold every
        // intermediate/output layout (see `alloc_local`).
        let input_len = self.local_n0 * self.local_n1 * self.dims[2];
        if input.len() < input_len {
            return Err(MpiError::SizeMismatch {
                expected: input_len,
                actual: input.len(),
            });
        }
        if output.len() < self.alloc_local {
            return Err(MpiError::SizeMismatch {
                expected: self.alloc_local,
                actual: output.len(),
            });
        }
        output[..input_len].copy_from_slice(&input[..input_len]);
        self.execute_inplace(output)
    }
}

/// Pure-Rust helpers for pencil decomposition that do not require MPI.
///
/// These functions implement the single-rank degenerate case and are used
/// both by `PencilPlan3D::execute_inplace` (P=1 path) and by unit tests.
pub mod pure {
    use super::*;

    /// Apply a 3D FFT via three sequential 1D passes in Z -> Y -> X order,
    /// using pre-created `Plan` instances.
    ///
    /// Input/output layout: row-major `[n0][n1][n2]`.
    ///
    /// This is the single-rank pencil kernel; `PencilPlan3D::execute_inplace`
    /// delegates to it when `P = 1`.
    pub(super) fn fft_3d_zyx_with_plans<T: Float>(
        data: &mut [Complex<T>],
        n0: usize,
        n1: usize,
        n2: usize,
        plan_x: &Plan<T>,
        plan_y: &Plan<T>,
        plan_z: &Plan<T>,
    ) {
        // Z-pass: stride-1 access for every (i0, i1) row.
        {
            let mut tmp = vec![Complex::<T>::zero(); n2];
            for i0 in 0..n0 {
                for i1 in 0..n1 {
                    let off = i0 * n1 * n2 + i1 * n2;
                    tmp.copy_from_slice(&data[off..off + n2]);
                    plan_z.execute(&tmp.clone(), &mut data[off..off + n2]);
                }
            }
        }

        // Y-pass: gather n1 values for each (i0, i2) pair, FFT, scatter back.
        {
            let mut col_in = vec![Complex::<T>::zero(); n1];
            let mut col_out = vec![Complex::<T>::zero(); n1];
            for i0 in 0..n0 {
                for i2 in 0..n2 {
                    for i1 in 0..n1 {
                        col_in[i1] = data[i0 * n1 * n2 + i1 * n2 + i2];
                    }
                    plan_y.execute(&col_in, &mut col_out);
                    for i1 in 0..n1 {
                        data[i0 * n1 * n2 + i1 * n2 + i2] = col_out[i1];
                    }
                }
            }
        }

        // X-pass: gather n0 values for each (i1, i2) pair, FFT, scatter back.
        {
            let mut row_in = vec![Complex::<T>::zero(); n0];
            let mut row_out = vec![Complex::<T>::zero(); n0];
            for i1 in 0..n1 {
                for i2 in 0..n2 {
                    for i0 in 0..n0 {
                        row_in[i0] = data[i0 * n1 * n2 + i1 * n2 + i2];
                    }
                    plan_x.execute(&row_in, &mut row_out);
                    for i0 in 0..n0 {
                        data[i0 * n1 * n2 + i1 * n2 + i2] = row_out[i0];
                    }
                }
            }
        }
    }

    /// Apply a 3D FFT via three sequential 1D passes in Z -> Y -> X order.
    ///
    /// Creates the required 1D plans internally. Suitable for standalone use
    /// and unit tests.
    ///
    /// Input/output layout: row-major `[n0][n1][n2]`.
    ///
    /// # Errors
    ///
    /// - `MpiError::InvalidDimension` -- if any dimension is zero
    /// - `MpiError::FftError` -- if any 1D plan cannot be created
    #[cfg(test)]
    pub fn fft_3d_zyx<T: Float>(
        data: &mut [Complex<T>],
        n0: usize,
        n1: usize,
        n2: usize,
        direction: Direction,
    ) -> Result<(), MpiError> {
        for (i, &d) in [n0, n1, n2].iter().enumerate() {
            if d == 0 {
                return Err(MpiError::InvalidDimension {
                    dim: i,
                    size: d,
                    message: "Dimension cannot be zero".to_string(),
                });
            }
        }

        let flags = Flags::ESTIMATE;

        let plan_z = Plan::dft_1d(n2, direction, flags).ok_or_else(|| MpiError::FftError {
            message: format!("Failed to create Z plan for size {n2}"),
        })?;
        let plan_y = Plan::dft_1d(n1, direction, flags).ok_or_else(|| MpiError::FftError {
            message: format!("Failed to create Y plan for size {n1}"),
        })?;
        let plan_x = Plan::dft_1d(n0, direction, flags).ok_or_else(|| MpiError::FftError {
            message: format!("Failed to create X plan for size {n0}"),
        })?;

        fft_3d_zyx_with_plans(data, n0, n1, n2, &plan_x, &plan_y, &plan_z);
        Ok(())
    }

    /// Compute the max absolute error between two complex slices.
    #[cfg(test)]
    pub fn max_abs_error<T: Float>(a: &[Complex<T>], b: &[Complex<T>]) -> T {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let diff = *x - *y;
                Float::sqrt(diff.re * diff.re + diff.im * diff.im)
            })
            .fold(T::zero(), |acc, v| if v > acc { v } else { acc })
    }
}

#[cfg(test)]
mod tests {
    use super::pure::{fft_3d_zyx, max_abs_error};
    use oxifft::api::Direction;
    use oxifft::kernel::Complex;

    fn make_test_input_f64(n0: usize, n1: usize, n2: usize) -> Vec<Complex<f64>> {
        let n = n0 * n1 * n2;
        (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                Complex {
                    re: (2.0 * core::f64::consts::PI * t * 3.0).cos(),
                    im: (2.0 * core::f64::consts::PI * t * 5.0).sin(),
                }
            })
            .collect()
    }

    // ----- PencilGrid unit tests -----

    #[test]
    fn pencil_grid_basic() {
        use super::PencilGrid;
        let g = PencilGrid::new(2, 4);
        assert_eq!(g.total_procs(), 8);
        assert_eq!(g.row_rank(0), 0);
        assert_eq!(g.row_rank(4), 1);
        assert_eq!(g.col_rank(0), 0);
        assert_eq!(g.col_rank(3), 3);
        assert_eq!(g.col_rank(4), 0);
        assert_eq!(g.col_rank(7), 3);
    }

    #[test]
    fn pencil_grid_single_proc() {
        use super::PencilGrid;
        let g = PencilGrid::new(1, 1);
        assert_eq!(g.total_procs(), 1);
        assert_eq!(g.row_rank(0), 0);
        assert_eq!(g.col_rank(0), 0);
    }

    #[test]
    fn pencil_grid_zero_cols_no_panic() {
        use super::PencilGrid;
        // Regression: a degenerate zero-column grid (constructed directly to
        // bypass `new`'s debug_assert) must not divide-by-zero panic.
        let g = PencilGrid {
            n_rows: 2,
            n_cols: 0,
        };
        assert_eq!(g.row_rank(5), 0);
        assert_eq!(g.col_rank(5), 0);
    }

    // ----- pure::fft_3d_zyx correctness tests -----

    /// For a unit impulse at (0,0,0), the DFT equals all-ones.
    #[test]
    fn pencil_pure_fft_4x4x4_impulse_gives_ones() {
        let n0 = 4;
        let n1 = 4;
        let n2 = 4;
        let n = n0 * n1 * n2;

        let mut data = vec![Complex::<f64>::zero(); n];
        data[0] = Complex { re: 1.0, im: 0.0 };

        fft_3d_zyx(&mut data, n0, n1, n2, Direction::Forward)
            .expect("fft_3d_zyx 4x4x4 should succeed");

        for (i, &v) in data.iter().enumerate() {
            assert!(
                (v.re - 1.0).abs() < 1e-10,
                "coeff[{i}].re = {:.2e} (expected 1.0)",
                v.re
            );
            assert!(
                v.im.abs() < 1e-10,
                "coeff[{i}].im = {:.2e} (expected 0.0)",
                v.im
            );
        }
    }

    /// For a unit impulse at (0,0,0), the DFT equals all-ones (8x8x8).
    #[test]
    fn pencil_pure_fft_8x8x8_impulse_gives_ones() {
        let n0 = 8;
        let n1 = 8;
        let n2 = 8;
        let n = n0 * n1 * n2;

        let mut data = vec![Complex::<f64>::zero(); n];
        data[0] = Complex { re: 1.0, im: 0.0 };

        fft_3d_zyx(&mut data, n0, n1, n2, Direction::Forward)
            .expect("fft_3d_zyx 8x8x8 should succeed");

        for (i, &v) in data.iter().enumerate() {
            assert!(
                (v.re - 1.0).abs() < 1e-10,
                "coeff[{i}].re = {:.2e} (expected 1.0)",
                v.re
            );
            assert!(
                v.im.abs() < 1e-10,
                "coeff[{i}].im = {:.2e} (expected 0.0)",
                v.im
            );
        }
    }

    /// Forward then inverse (normalized) should recover the original input -- 4x4x4.
    #[test]
    fn pencil_pure_roundtrip_4x4x4() {
        let n0 = 4;
        let n1 = 4;
        let n2 = 4;
        let n = n0 * n1 * n2;
        let scale = n as f64;

        let original = make_test_input_f64(n0, n1, n2);
        let mut data = original.clone();

        fft_3d_zyx(&mut data, n0, n1, n2, Direction::Forward).expect("forward fft should succeed");
        fft_3d_zyx(&mut data, n0, n1, n2, Direction::Backward).expect("inverse fft should succeed");

        for v in data.iter_mut() {
            v.re /= scale;
            v.im /= scale;
        }

        let err = max_abs_error(&original, &data);
        assert!(
            err < 1e-10,
            "roundtrip error {err:.2e} exceeds 1e-10 for 4x4x4"
        );
    }

    /// Forward then inverse (normalized) should recover the original input -- 8x8x8.
    #[test]
    fn pencil_pure_roundtrip_8x8x8() {
        let n0 = 8;
        let n1 = 8;
        let n2 = 8;
        let n = n0 * n1 * n2;
        let scale = n as f64;

        let original = make_test_input_f64(n0, n1, n2);
        let mut data = original.clone();

        fft_3d_zyx(&mut data, n0, n1, n2, Direction::Forward).expect("forward fft should succeed");
        fft_3d_zyx(&mut data, n0, n1, n2, Direction::Backward).expect("inverse fft should succeed");

        for v in data.iter_mut() {
            v.re /= scale;
            v.im /= scale;
        }

        let err = max_abs_error(&original, &data);
        assert!(
            err < 1e-10,
            "roundtrip error {err:.2e} exceeds 1e-10 for 8x8x8"
        );
    }

    /// FFT(a*x + b*y) = a*FFT(x) + b*FFT(y) -- linearity property.
    #[test]
    fn pencil_pure_linearity_8x8x8() {
        let n0 = 8;
        let n1 = 8;
        let n2 = 8;
        let n = n0 * n1 * n2;

        let x = make_test_input_f64(n0, n1, n2);
        let y: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let t = (i + 7) as f64 / n as f64;
                Complex {
                    re: (2.0 * core::f64::consts::PI * t).cos(),
                    im: 0.0,
                }
            })
            .collect();

        let a = Complex::<f64> { re: 2.0, im: -1.0 };
        let b = Complex::<f64> { re: -0.5, im: 3.0 };

        // FFT(a*x + b*y)
        let mut combined: Vec<Complex<f64>> = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| a * xi + b * yi)
            .collect();
        fft_3d_zyx(&mut combined, n0, n1, n2, Direction::Forward)
            .expect("combined fft should succeed");

        // a*FFT(x) + b*FFT(y)
        let mut fx = x;
        let mut fy = y;
        fft_3d_zyx(&mut fx, n0, n1, n2, Direction::Forward).expect("fx fft should succeed");
        fft_3d_zyx(&mut fy, n0, n1, n2, Direction::Forward).expect("fy fft should succeed");
        let linear: Vec<Complex<f64>> = fx
            .iter()
            .zip(fy.iter())
            .map(|(&xi, &yi)| a * xi + b * yi)
            .collect();

        let err = max_abs_error(&combined, &linear);
        assert!(
            err < 1e-8,
            "linearity error {err:.2e} exceeds 1e-8 for 8x8x8"
        );
    }

    /// Zero-dimension inputs should return an error.
    #[test]
    fn pencil_pure_zero_dim_error() {
        let mut data: Vec<Complex<f64>> = Vec::new();
        let result = fft_3d_zyx(&mut data, 0, 4, 4, Direction::Forward);
        assert!(result.is_err(), "expected error for zero n0");
    }

    // Real MPI test for the single-rank (P=1) pencil path.
    //
    // It is `#[ignore]`d because a plain `cargo test` runs test functions on
    // spawned threads, whereas MPI is initialised `MPI_THREAD_SINGLE`. This test
    // therefore only exercises the P=1 code path (no cross-rank collectives, so
    // no thread/collective deadlock). *Multi-rank* correctness is covered by the
    // `examples/mpi_integration.rs` binary (its MPI calls run on the process
    // main thread), driven by `scripts/run_mpi_tests.sh`.
    //
    // Run this one with, e.g.:
    //   cargo test -p oxifft-adapter-mpi --no-run
    //   mpirun -n 1 <test-binary> --ignored --exact \
    //       plans::plan_3d_pencil::tests::mpi_required::pencil_mpi_single_rank
    mod mpi_required {
        use super::super::pure::{fft_3d_zyx, max_abs_error};
        use super::super::{PencilGrid, PencilPlan3D};
        use crate::pool::MpiPool;
        use oxifft::api::{Direction, Flags};
        use oxifft::kernel::Complex;

        fn sample(idx: usize) -> Complex<f64> {
            let t = idx as f64;
            Complex {
                re: (0.1 * t).sin() + 0.3,
                im: (0.07 * t + 1.0).cos() - 0.2,
            }
        }

        /// Forward pencil FFT (P=1) vs. the serial `fft_3d_zyx` reference.
        ///
        /// Run under `mpirun -n 1`. Under a larger `-n` this returns early (the
        /// multi-rank path is validated by `examples/mpi_integration.rs`).
        #[test]
        #[ignore = "Requires MPI runtime: run under `mpirun -n 1 <bin> --ignored`"]
        fn pencil_mpi_single_rank() {
            let Some(universe) = mpi::initialize() else {
                return;
            };
            let world = universe.world();
            let pool = MpiPool::new(world);
            if pool.size() != 1 {
                // Multi-rank collectives from libtest's spawned thread would
                // deadlock; see examples/mpi_integration.rs for real coverage.
                return;
            }

            for &(n0, n1, n2) in &[(8usize, 8usize, 8usize), (6, 5, 4)] {
                let grid = PencilGrid::new(1, 1);
                let mut reference: Vec<Complex<f64>> = (0..n0 * n1 * n2).map(sample).collect();
                fft_3d_zyx(&mut reference, n0, n1, n2, Direction::Forward)
                    .expect("reference fft_3d_zyx");

                let mut data: Vec<Complex<f64>> = (0..n0 * n1 * n2).map(sample).collect();
                let mut plan =
                    PencilPlan3D::new(n0, n1, n2, grid, Direction::Forward, Flags::ESTIMATE, &pool)
                        .expect("PencilPlan3D::new");
                plan.execute_inplace(&mut data).expect("execute_inplace");

                let err = max_abs_error(&data, &reference);
                assert!(err < 1e-8, "pencil P=1 {n0}x{n1}x{n2} err {err:.2e}");
            }
        }
    }
}
