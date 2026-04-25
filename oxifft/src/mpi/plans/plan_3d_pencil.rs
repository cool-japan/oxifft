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
//! The multi-rank alltoallv passes are not yet implemented. Calling
//! `execute_inplace` on a multi-rank plan returns
//! `MpiError::FftError { message: "multi-rank pencil execution not yet implemented" }`.
//! The type, constructor, buffer pre-allocation, and sub-communicator creation
//! are fully implemented. Single-rank execution is fully functional.

use mpi::datatype::Equivalence;
use mpi::topology::{Color, Communicator, SimpleCommunicator};

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};

use crate::mpi::distribution::LocalPartition;
use crate::mpi::error::MpiError;
use crate::mpi::pool::{MpiFloat, MpiPool};

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
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self { n_rows, n_cols }
    }

    /// Total number of processes required (`n_rows * n_cols`).
    pub fn total_procs(&self) -> usize {
        self.n_rows * self.n_cols
    }

    /// Row rank for a given global rank (0-indexed row in the 2D grid).
    pub fn row_rank(&self, global_rank: usize) -> usize {
        global_rank / self.n_cols
    }

    /// Column rank for a given global rank (0-indexed column in the 2D grid).
    pub fn col_rank(&self, global_rank: usize) -> usize {
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
pub struct PencilPlan3D<T: Float, C: Communicator> {
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
    _scratch: Vec<Complex<T>>,
    /// Raw pointer to the global MPI pool (must outlive this plan; reserved for multi-rank impl).
    _pool: *const MpiPool<C>,
    /// Row sub-communicator pool: all procs with the same row_rank, varying col_rank.
    /// `None` for single-rank plans.
    row_pool: Option<MpiPool<SimpleCommunicator>>,
    /// Column sub-communicator pool: all procs with the same col_rank, varying row_rank.
    /// `None` for single-rank plans.
    col_pool: Option<MpiPool<SimpleCommunicator>>,
    /// Marker for type parameters.
    _phantom: core::marker::PhantomData<(T, C)>,
}

// SAFETY: PencilPlan3D stores a raw pointer to an MpiPool that must outlive the plan.
// The caller ensures the pool outlives the plan (same contract as MpiPlan3D).
unsafe impl<T: Float, C: Communicator + Send> Send for PencilPlan3D<T, C> {}
unsafe impl<T: Float, C: Communicator + Sync> Sync for PencilPlan3D<T, C> {}

impl<T: Float + MpiFloat, C: Communicator> PencilPlan3D<T, C>
where
    Complex<T>: Equivalence,
{
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
        pool: &MpiPool<C>,
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

        // Pre-allocate scratch buffer sized for all intermediate layouts.
        let local_n2_col = LocalPartition::new(n2, grid.n_cols, col_rank).local_n;
        let local_n1_row = LocalPartition::new(n1, grid.n_rows, row_rank).local_n;
        let scratch_size = (local_n0 * local_n1 * n2)
            .max(local_n0 * n1 * local_n2_col)
            .max(n0 * local_n1_row * local_n2_col);
        let scratch = vec![Complex::<T>::zero(); scratch_size];

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
            _scratch: scratch,
            _pool: core::ptr::from_ref(pool),
            row_pool,
            col_pool,
            _phantom: core::marker::PhantomData,
        })
    }

    /// Get global dimensions `[n0, n1, n2]`.
    pub fn dims(&self) -> [usize; 3] {
        self.dims
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

    /// Execute the distributed 3D FFT in-place.
    ///
    /// Input/output layout: `data[i0 * local_n1 * n2 + i1 * n2 + i2]`
    /// where `i0 in [0, local_n0)`, `i1 in [0, local_n1)`, `i2 in [0, n2)`.
    ///
    /// For single-rank plans (`P = 1`), all data is local and no MPI
    /// communication is performed. For multi-rank plans, this currently returns
    /// `MpiError::FftError` with a "not yet implemented" message.
    ///
    /// # Errors
    ///
    /// - `MpiError::SizeMismatch` -- if `data.len() < local_n0 * local_n1 * n2`
    /// - `MpiError::FftError` -- if multi-rank execution is attempted (NYI)
    pub fn execute_inplace(&mut self, data: &mut [Complex<T>]) -> Result<(), MpiError> {
        let [n0, n1, n2] = self.dims;
        let expected = self.local_n0 * self.local_n1 * n2;
        if data.len() < expected {
            return Err(MpiError::SizeMismatch {
                expected,
                actual: data.len(),
            });
        }

        if self.row_pool.is_none() {
            // Single-rank: three sequential local 1D FFTs, no MPI.
            pure::fft_3d_zyx_with_plans(data, n0, n1, n2, &self.plan_x, &self.plan_y, &self.plan_z);
            Ok(())
        } else {
            // Multi-rank: two alltoallv transposes + three local FFT passes -- NYI.
            Err(MpiError::FftError {
                message: "multi-rank pencil execution not yet implemented".to_string(),
            })
        }
    }

    /// Execute the distributed 3D FFT out-of-place.
    ///
    /// Copies `input` to `output` then calls `execute_inplace`.
    ///
    /// # Errors
    ///
    /// - `MpiError::SizeMismatch` -- if either buffer is too small
    /// - `MpiError::FftError` -- if multi-rank execution is attempted (NYI)
    pub fn execute(
        &mut self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
    ) -> Result<(), MpiError> {
        let expected = self.local_n0 * self.local_n1 * self.dims[2];
        if input.len() < expected {
            return Err(MpiError::SizeMismatch {
                expected,
                actual: input.len(),
            });
        }
        if output.len() < expected {
            return Err(MpiError::SizeMismatch {
                expected,
                actual: output.len(),
            });
        }
        output[..expected].copy_from_slice(&input[..expected]);
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
    use crate::api::Direction;
    use crate::kernel::Complex;

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

    // Multi-rank tests require `mpirun -n N cargo test --features mpi`.
    // They are marked `#[ignore]` so they are skipped in standard `cargo test`.
    #[cfg(feature = "mpi")]
    mod mpi_required {
        /// Placeholder: construction test for P=1 via MPI runtime.
        ///
        /// Run with: `mpirun -n 1 cargo test --features mpi pencil_mpi_construction_p1`
        #[test]
        #[ignore = "Requires MPI runtime: mpirun -n 1 cargo test --features mpi"]
        fn pencil_mpi_construction_p1() {
            // When run with mpirun:
            //   let universe = mpi::initialize().unwrap();
            //   let world = universe.world();
            //   let pool = MpiPool::new(world.duplicate());
            //   let grid = PencilGrid::new(1, 1);
            //   let mut plan = PencilPlan3D::<f64, _>::new(
            //       8, 8, 8, grid, Direction::Forward, Flags::ESTIMATE, &pool,
            //   ).expect("plan creation should succeed");
            //   let mut data = vec![Complex::<f64>::zero(); 8 * 8 * 8];
            //   assert!(plan.execute_inplace(&mut data).is_ok());
        }

        /// Placeholder: 4-rank 8x8x8 test -- requires multi-rank NYI to be implemented.
        ///
        /// Run with: `mpirun -n 4 cargo test --features mpi pencil_mpi_4rank_8x8x8`
        #[test]
        #[ignore = "Requires MPI runtime with 4 ranks: mpirun -n 4 cargo test --features mpi"]
        fn pencil_mpi_4rank_8x8x8() {
            // When multi-rank execute_inplace is implemented:
            // 1. Initialize MPI, create 2x2 PencilPlan3D
            // 2. Distribute 8x8x8 data across 4 ranks
            // 3. Run forward FFT
            // 4. Gather results and compare with pure::fft_3d_zyx (tolerance 1e-10)
        }
    }
}
