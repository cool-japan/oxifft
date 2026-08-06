//! Distributed **real** FFT plans (slab-decomposed r2c / c2r).
//!
//! These mirror FFTW-MPI's distributed real transforms. Like the complex slab
//! plans ([`MpiPlan2D`](crate::MpiPlan2D) / [`MpiPlan3D`](crate::MpiPlan3D)) the
//! first dimension is distributed across processes; unlike them the *last*
//! dimension is transformed with a real-to-complex (r2c) / complex-to-real (c2r)
//! transform, so it is stored in the FFTW half-complex layout: a real last
//! dimension of size `n` produces `n/2 + 1` complex coefficients (Hermitian
//! symmetry makes the remaining `n - (n/2 + 1)` coefficients redundant).
//!
//! # Pipeline
//!
//! For r2c the transform runs "inner axes first, distributed axis last":
//!
//! 1. **Local** r2c along the trailing axes (a 1-D r2c for 2-D, a 2-D r2c per
//!    plane for 3-D), turning each local slab `[local_n0][…][n_last]` (real) into
//!    `[local_n0][…][n_last/2 + 1]` (complex).
//! 2. A **distributed transpose** (single `alltoallv`) redistributes so the first
//!    dimension becomes local, a local 1-D complex FFT runs along it, and a second
//!    transpose restores the slab layout (skipped when `transposed_out` is set).
//!
//! c2r runs the exact inverse and, matching the OxiFFT 0.4.0 core convention,
//! **normalizes** the result by `1 / product(dims)` so that an r2c -> c2r round
//! trip is the identity. (FFTW leaves c2r unnormalized; the core crate documents
//! this deliberate divergence, and this adapter is consistent with the core.)

use mpi::topology::Communicator;

use oxifft::api::{Direction, Plan, RealPlan, RealPlan2D};
use oxifft::kernel::{Complex, Float};

use crate::distribution::{Distribution, LocalPartition};
use crate::error::MpiError;
use crate::pool::{MpiFloat, MpiPool};
use crate::transpose::{distributed_transpose, distributed_transpose_batched};
use crate::MpiFlags;

/// Which real transform a plan performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RealKind {
    /// Real-to-complex (forward).
    R2c,
    /// Complex-to-real (inverse).
    C2r,
}

/// Reject `transposed_in` on an **r2c** plan.
///
/// `transposed_in` describes the distribution of the *half-complex* array, which
/// for r2c is the output, not the input: an r2c plan's input is the real-space
/// slab `[local_n0][…][n_last]`, and there is no transposed real-space layout for
/// it to be in. This matches FFTW-MPI, where `FFTW_MPI_TRANSPOSED_IN` is
/// meaningful for c2r and `FFTW_MPI_TRANSPOSED_OUT` for r2c. The round-trip
/// idiom is therefore `r2c` + `transposed_out` followed by `c2r` +
/// `transposed_in`, which skips one `alltoallv` in each direction.
fn reject_transposed_in_r2c(flags: MpiFlags, plan: &str) -> Result<(), MpiError> {
    if flags.transposed_in {
        return Err(MpiError::FftError {
            message: format!(
                "MpiFlags::transposed_in is not meaningful for {plan}: an r2c plan's input is the \
                 real-space slab, which is never transposed. Use transposed_out on the r2c plan \
                 and transposed_in on the matching c2r plan."
            ),
        });
    }
    Ok(())
}

// ===========================================================================
// 2D real plan
// ===========================================================================

/// 2D distributed real FFT plan (slab decomposition over the first dimension).
///
/// * r2c: `n0 x n1` real values distributed as `[local_n0][n1]` produce
///   `n0 x (n1/2 + 1)` complex values distributed as `[local_n0][n1/2 + 1]`.
/// * c2r: the inverse, normalized by `1 / (n0 * n1)`.
pub struct MpiRealPlan2D<'p, T: Float, C: Communicator> {
    /// Global number of rows (distributed).
    n0: usize,
    /// Global number of real columns.
    n1: usize,
    /// Half-complex column count, `n1 / 2 + 1`.
    n1c: usize,
    /// Local number of rows owned by this process.
    local_n0: usize,
    /// Global starting row for this process.
    local_0_start: usize,
    /// Transform kind (r2c or c2r).
    kind: RealKind,
    /// Planning flags.
    flags: MpiFlags,
    /// Borrow of the MPI pool.
    pool: &'p MpiPool<C>,
    /// Local 1-D real transform along the last (n1) axis.
    line_plan: RealPlan<T>,
    /// Local 1-D complex transform along the distributed (n0) axis.
    n0_plan: Plan<T>,
    /// Buffer holding the `[local_n0][n1c]` half-complex slab.
    work: Vec<Complex<T>>,
    /// Buffer holding the transposed `[local_n1c][n0]` layout.
    scratch: Vec<Complex<T>>,
}

impl<'p, T: Float + MpiFloat, C: Communicator> MpiRealPlan2D<'p, T, C> {
    /// Create a 2D real-to-complex distributed plan.
    ///
    /// # Errors
    /// Returns [`MpiError::InvalidDimension`] if a dimension is zero,
    /// [`MpiError::FftError`] if `transposed_in` is set (an r2c plan's input is
    /// the real-space slab, which is never transposed — set `transposed_out`
    /// here and `transposed_in` on the matching c2r plan instead) or a local
    /// sub-plan cannot be built.
    pub fn r2c(
        n0: usize,
        n1: usize,
        flags: MpiFlags,
        pool: &'p MpiPool<C>,
    ) -> Result<Self, MpiError> {
        Self::new(n0, n1, RealKind::R2c, flags, pool)
    }

    /// Create a 2D complex-to-real distributed plan (normalized inverse).
    ///
    /// Set `transposed_in` on `flags` to consume the `[local_n1c][n0]`
    /// half-complex distribution emitted by a 2D r2c plan built with
    /// `transposed_out`; that pairing skips one `alltoallv` in each direction.
    ///
    /// # Errors
    /// Returns [`MpiError::InvalidDimension`] if a dimension is zero,
    /// [`MpiError::FftError`] if `transposed_out` is set (a c2r plan's output is
    /// the real-space slab, which is never transposed) or a local sub-plan
    /// cannot be built.
    pub fn c2r(
        n0: usize,
        n1: usize,
        flags: MpiFlags,
        pool: &'p MpiPool<C>,
    ) -> Result<Self, MpiError> {
        Self::new(n0, n1, RealKind::C2r, flags, pool)
    }

    fn new(
        n0: usize,
        n1: usize,
        kind: RealKind,
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
        if kind == RealKind::R2c {
            reject_transposed_in_r2c(flags, "the 2D r2c slab plan")?;
        }
        if kind == RealKind::C2r && flags.transposed_out {
            return Err(MpiError::FftError {
                message: "MpiFlags::transposed_out is not supported for the 2D c2r slab plan"
                    .to_string(),
            });
        }

        let n1c = n1 / 2 + 1;
        let partition = pool.local_partition(n0);
        let local_n0 = partition.local_n;
        let local_0_start = partition.local_start;
        let tpart = LocalPartition::new(n1c, pool.size(), pool.rank());
        let local_n1c = tpart.local_n;

        let (direction, line_plan) = match kind {
            RealKind::R2c => (
                Direction::Forward,
                RealPlan::<T>::r2c_1d(n1, flags.base).ok_or_else(|| MpiError::FftError {
                    message: format!("Failed to create r2c line plan for size {n1}"),
                })?,
            ),
            RealKind::C2r => (
                Direction::Backward,
                RealPlan::<T>::c2r_1d(n1, flags.base).ok_or_else(|| MpiError::FftError {
                    message: format!("Failed to create c2r line plan for size {n1}"),
                })?,
            ),
        };
        let n0_plan =
            Plan::<T>::dft_1d(n0, direction, flags.base).ok_or_else(|| MpiError::FftError {
                message: format!("Failed to create n0 plan for size {n0}"),
            })?;

        Ok(Self {
            n0,
            n1,
            n1c,
            local_n0,
            local_0_start,
            kind,
            flags,
            pool,
            line_plan,
            n0_plan,
            work: vec![Complex::<T>::zero(); local_n0 * n1c],
            scratch: vec![Complex::<T>::zero(); n0 * local_n1c],
        })
    }

    /// Global dimensions `(n0, n1)`.
    pub fn dims(&self) -> (usize, usize) {
        (self.n0, self.n1)
    }

    /// Local partition info `(local_n0, local_0_start)`.
    pub fn local_info(&self) -> (usize, usize) {
        (self.local_n0, self.local_0_start)
    }

    /// Data-distribution strategy (always [`Distribution::Slab`]).
    pub fn distribution(&self) -> Distribution {
        Distribution::Slab
    }

    /// Execute the distributed real-to-complex transform.
    ///
    /// `input` is the local real slab `[local_n0][n1]`; `output` is the local
    /// half-complex slab `[local_n0][n1/2 + 1]` (or `[local_n1c][n0]` when
    /// `transposed_out` is set).
    ///
    /// # Errors
    /// Returns [`MpiError::FftError`] if this is not an r2c plan,
    /// [`MpiError::SizeMismatch`] if a buffer is too small, or any transpose
    /// error.
    pub fn execute_r2c(&mut self, input: &[T], output: &mut [Complex<T>]) -> Result<(), MpiError> {
        if self.kind != RealKind::R2c {
            return Err(MpiError::FftError {
                message: "execute_r2c called on a c2r plan".to_string(),
            });
        }
        let (n0, n1, n1c) = (self.n0, self.n1, self.n1c);
        let (local_n0, local_0_start) = (self.local_n0, self.local_0_start);
        let pool = self.pool;
        let tpart = LocalPartition::new(n1c, pool.size(), pool.rank());
        let local_n1c = tpart.local_n;

        if input.len() < local_n0 * n1 {
            return Err(MpiError::SizeMismatch {
                expected: local_n0 * n1,
                actual: input.len(),
            });
        }
        let out_size = if self.flags.transposed_out {
            local_n1c * n0
        } else {
            local_n0 * n1c
        };
        if output.len() < out_size {
            return Err(MpiError::SizeMismatch {
                expected: out_size,
                actual: output.len(),
            });
        }

        // Step 1: local r2c along n1 -> work[local_n0][n1c].
        for i0 in 0..local_n0 {
            self.line_plan.execute_r2c(
                &input[i0 * n1..i0 * n1 + n1],
                &mut self.work[i0 * n1c..i0 * n1c + n1c],
            );
        }

        // Step 2: distributed complex FFT along n0 (transpose -> local FFT).
        distributed_transpose(
            pool,
            &self.work,
            &mut self.scratch,
            n0,
            n1c,
            local_n0,
            local_0_start,
        )?;
        self.fft_columns_n0(local_n1c);

        // Step 3: restore slab layout (or emit transposed output directly).
        if self.flags.transposed_out {
            output[..local_n1c * n0].copy_from_slice(&self.scratch[..local_n1c * n0]);
        } else {
            distributed_transpose(
                pool,
                &self.scratch[..n0 * local_n1c],
                output,
                n1c,
                n0,
                local_n1c,
                tpart.local_start,
            )?;
        }
        Ok(())
    }

    /// Execute the distributed complex-to-real transform (**normalized**).
    ///
    /// `input` is the local half-complex slab `[local_n0][n1/2 + 1]` — or, when
    /// `transposed_in` is set, the transposed half-complex distribution
    /// `[local_n1c][n0]` produced by an r2c plan with `transposed_out`. `output`
    /// is the local real slab `[local_n0][n1]`, normalized by `1 / (n0 * n1)` so
    /// that r2c -> c2r is the identity.
    ///
    /// # Errors
    /// Returns [`MpiError::FftError`] if this is not a c2r plan,
    /// [`MpiError::SizeMismatch`] if a buffer is too small, or any transpose
    /// error.
    pub fn execute_c2r(&mut self, input: &[Complex<T>], output: &mut [T]) -> Result<(), MpiError> {
        if self.kind != RealKind::C2r {
            return Err(MpiError::FftError {
                message: "execute_c2r called on an r2c plan".to_string(),
            });
        }
        let (n0, n1, n1c) = (self.n0, self.n1, self.n1c);
        let (local_n0, local_0_start) = (self.local_n0, self.local_0_start);
        let pool = self.pool;
        let tpart = LocalPartition::new(n1c, pool.size(), pool.rank());
        let local_n1c = tpart.local_n;

        let in_size = if self.flags.transposed_in {
            local_n1c * n0
        } else {
            local_n0 * n1c
        };
        if input.len() < in_size {
            return Err(MpiError::SizeMismatch {
                expected: in_size,
                actual: input.len(),
            });
        }
        if output.len() < local_n0 * n1 {
            return Err(MpiError::SizeMismatch {
                expected: local_n0 * n1,
                actual: output.len(),
            });
        }

        // Step 1: distributed complex inverse FFT along n0. With `transposed_in`
        // the caller already handed us the `[local_n1c][n0]` distribution the
        // forward transpose would have produced, so that collective is skipped and
        // the input is simply adopted as the transposed working set.
        if self.flags.transposed_in {
            self.scratch[..in_size].copy_from_slice(&input[..in_size]);
        } else {
            distributed_transpose(
                pool,
                &input[..local_n0 * n1c],
                &mut self.scratch,
                n0,
                n1c,
                local_n0,
                local_0_start,
            )?;
        }
        self.fft_columns_n0(local_n1c);
        distributed_transpose(
            pool,
            &self.scratch[..n0 * local_n1c],
            &mut self.work,
            n1c,
            n0,
            local_n1c,
            tpart.local_start,
        )?;

        // Step 2: local unnormalized c2r along n1 -> output[local_n0][n1].
        for i0 in 0..local_n0 {
            self.line_plan.execute_c2r_unnormalized(
                &self.work[i0 * n1c..i0 * n1c + n1c],
                &mut output[i0 * n1..i0 * n1 + n1],
            );
        }

        // Step 3: single 1/(n0*n1) normalization for the whole 2D transform.
        let scale = T::ONE / T::from_usize(n0 * n1);
        for x in output[..local_n0 * n1].iter_mut() {
            *x = *x * scale;
        }
        Ok(())
    }

    /// Apply the local length-`n0` complex FFT to each of `local_n1c` contiguous
    /// columns held in `self.scratch` as `[local_n1c][n0]`.
    fn fft_columns_n0(&mut self, local_n1c: usize) {
        let n0 = self.n0;
        let mut fiber_in = vec![Complex::<T>::zero(); n0];
        let mut fiber_out = vec![Complex::<T>::zero(); n0];
        for col in 0..local_n1c {
            let base = col * n0;
            fiber_in.copy_from_slice(&self.scratch[base..base + n0]);
            self.n0_plan.execute(&fiber_in, &mut fiber_out);
            self.scratch[base..base + n0].copy_from_slice(&fiber_out);
        }
    }
}

// ===========================================================================
// 3D real plan
// ===========================================================================

/// 3D distributed real FFT plan (slab decomposition over the first dimension).
///
/// * r2c: `n0 x n1 x n2` real values distributed as `[local_n0][n1][n2]` produce
///   `n0 x n1 x (n2/2 + 1)` complex values distributed as
///   `[local_n0][n1][n2/2 + 1]`.
/// * c2r: the inverse, normalized by `1 / (n0 * n1 * n2)`.
pub struct MpiRealPlan3D<'p, T: Float, C: Communicator> {
    /// Global dimensions.
    dims: [usize; 3],
    /// Half-complex last-dimension count, `n2 / 2 + 1`.
    n2c: usize,
    /// Local number of planes owned by this process.
    local_n0: usize,
    /// Global starting plane for this process.
    local_0_start: usize,
    /// Local number of `n1` rows owned after the distributed transpose.
    local_n1: usize,
    /// Transform kind (r2c or c2r).
    kind: RealKind,
    /// Planning flags.
    flags: MpiFlags,
    /// Borrow of the MPI pool.
    pool: &'p MpiPool<C>,
    /// Local 2-D real transform along the trailing `(n1, n2)` plane.
    plane_plan: RealPlan2D<T>,
    /// Local 1-D complex transform along the distributed (n0) axis.
    n0_plan: Plan<T>,
    /// Buffer holding the `[local_n0][n1][n2c]` half-complex slab.
    work: Vec<Complex<T>>,
    /// Buffer holding the transposed `[local_n1][n0][n2c]` layout.
    scratch: Vec<Complex<T>>,
}

impl<'p, T: Float + MpiFloat, C: Communicator> MpiRealPlan3D<'p, T, C> {
    /// Create a 3D real-to-complex distributed plan.
    ///
    /// # Errors
    /// Returns [`MpiError::InvalidDimension`] if a dimension is zero,
    /// [`MpiError::FftError`] if `transposed_in` is set (an r2c plan's input is
    /// the real-space slab, which is never transposed — set `transposed_out`
    /// here and `transposed_in` on the matching c2r plan instead) or a local
    /// sub-plan cannot be built.
    pub fn r2c(
        n0: usize,
        n1: usize,
        n2: usize,
        flags: MpiFlags,
        pool: &'p MpiPool<C>,
    ) -> Result<Self, MpiError> {
        Self::new(n0, n1, n2, RealKind::R2c, flags, pool)
    }

    /// Create a 3D complex-to-real distributed plan (normalized inverse).
    ///
    /// Set `transposed_in` on `flags` to consume the `[local_n1][n0][n2c]`
    /// half-complex distribution emitted by a 3D r2c plan built with
    /// `transposed_out`; that pairing skips one `alltoallv` in each direction.
    ///
    /// # Errors
    /// Returns [`MpiError::InvalidDimension`] if a dimension is zero,
    /// [`MpiError::FftError`] if `transposed_out` is set (a c2r plan's output is
    /// the real-space slab, which is never transposed) or a local sub-plan
    /// cannot be built.
    pub fn c2r(
        n0: usize,
        n1: usize,
        n2: usize,
        flags: MpiFlags,
        pool: &'p MpiPool<C>,
    ) -> Result<Self, MpiError> {
        Self::new(n0, n1, n2, RealKind::C2r, flags, pool)
    }

    fn new(
        n0: usize,
        n1: usize,
        n2: usize,
        kind: RealKind,
        flags: MpiFlags,
        pool: &'p MpiPool<C>,
    ) -> Result<Self, MpiError> {
        if n0 == 0 || n1 == 0 || n2 == 0 {
            let (dim, size) = if n0 == 0 {
                (0, n0)
            } else if n1 == 0 {
                (1, n1)
            } else {
                (2, n2)
            };
            return Err(MpiError::InvalidDimension {
                dim,
                size,
                message: "Dimension size cannot be zero".to_string(),
            });
        }
        if kind == RealKind::R2c {
            reject_transposed_in_r2c(flags, "the 3D r2c slab plan")?;
        }
        if kind == RealKind::C2r && flags.transposed_out {
            return Err(MpiError::FftError {
                message: "MpiFlags::transposed_out is not supported for the 3D c2r slab plan"
                    .to_string(),
            });
        }

        let n2c = n2 / 2 + 1;
        let partition = pool.local_partition(n0);
        let local_n0 = partition.local_n;
        let local_0_start = partition.local_start;
        let local_n1 = LocalPartition::new(n1, pool.size(), pool.rank()).local_n;

        let (direction, plane_plan) = match kind {
            RealKind::R2c => (
                Direction::Forward,
                RealPlan2D::<T>::r2c(n1, n2, flags.base).ok_or_else(|| MpiError::FftError {
                    message: format!("Failed to create r2c plane plan for {n1}x{n2}"),
                })?,
            ),
            RealKind::C2r => (
                Direction::Backward,
                RealPlan2D::<T>::c2r(n1, n2, flags.base).ok_or_else(|| MpiError::FftError {
                    message: format!("Failed to create c2r plane plan for {n1}x{n2}"),
                })?,
            ),
        };
        let n0_plan =
            Plan::<T>::dft_1d(n0, direction, flags.base).ok_or_else(|| MpiError::FftError {
                message: format!("Failed to create n0 plan for size {n0}"),
            })?;

        Ok(Self {
            dims: [n0, n1, n2],
            n2c,
            local_n0,
            local_0_start,
            local_n1,
            kind,
            flags,
            pool,
            plane_plan,
            n0_plan,
            work: vec![Complex::<T>::zero(); local_n0 * n1 * n2c],
            scratch: vec![Complex::<T>::zero(); n0 * local_n1 * n2c],
        })
    }

    /// Global dimensions `[n0, n1, n2]`.
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Local partition info `(local_n0, local_0_start)`.
    pub fn local_info(&self) -> (usize, usize) {
        (self.local_n0, self.local_0_start)
    }

    /// Data-distribution strategy (always [`Distribution::Slab`]).
    pub fn distribution(&self) -> Distribution {
        Distribution::Slab
    }

    /// Execute the distributed real-to-complex transform.
    ///
    /// `input` is the local real slab `[local_n0][n1][n2]`; `output` is the local
    /// half-complex slab `[local_n0][n1][n2/2 + 1]` (or `[local_n1][n0][n2/2 + 1]`
    /// when `transposed_out` is set).
    ///
    /// # Errors
    /// Returns [`MpiError::FftError`] if this is not an r2c plan,
    /// [`MpiError::SizeMismatch`] if a buffer is too small, or any transpose
    /// error.
    pub fn execute_r2c(&mut self, input: &[T], output: &mut [Complex<T>]) -> Result<(), MpiError> {
        if self.kind != RealKind::R2c {
            return Err(MpiError::FftError {
                message: "execute_r2c called on a c2r plan".to_string(),
            });
        }
        let [n0, n1, n2] = self.dims;
        let n2c = self.n2c;
        let (local_n0, local_n1) = (self.local_n0, self.local_n1);
        let pool = self.pool;

        if input.len() < local_n0 * n1 * n2 {
            return Err(MpiError::SizeMismatch {
                expected: local_n0 * n1 * n2,
                actual: input.len(),
            });
        }
        let out_size = if self.flags.transposed_out {
            local_n1 * n0 * n2c
        } else {
            local_n0 * n1 * n2c
        };
        if output.len() < out_size {
            return Err(MpiError::SizeMismatch {
                expected: out_size,
                actual: output.len(),
            });
        }

        // Step 1: local 2-D r2c on each (n1, n2) plane -> work[local_n0][n1][n2c].
        let plane_in = n1 * n2;
        let plane_out = n1 * n2c;
        for i0 in 0..local_n0 {
            self.plane_plan.execute_r2c(
                &input[i0 * plane_in..i0 * plane_in + plane_in],
                &mut self.work[i0 * plane_out..i0 * plane_out + plane_out],
            );
        }

        // Step 2: distributed complex FFT along n0 (batched transpose over n2c).
        distributed_transpose_batched(pool, &self.work, &mut self.scratch, n0, n1, local_n0, n2c)?;
        self.fft_fibers_n0(local_n1);

        // Step 3: restore slab layout (or emit transposed output directly).
        if self.flags.transposed_out {
            output[..local_n1 * n0 * n2c].copy_from_slice(&self.scratch[..local_n1 * n0 * n2c]);
        } else {
            distributed_transpose_batched(
                pool,
                &self.scratch[..n0 * local_n1 * n2c],
                output,
                n1,
                n0,
                local_n1,
                n2c,
            )?;
        }
        Ok(())
    }

    /// Execute the distributed complex-to-real transform (**normalized**).
    ///
    /// `input` is the local half-complex slab `[local_n0][n1][n2/2 + 1]` — or,
    /// when `transposed_in` is set, the transposed half-complex distribution
    /// `[local_n1][n0][n2/2 + 1]` produced by an r2c plan with `transposed_out`.
    /// `output` is the local real slab `[local_n0][n1][n2]`, normalized by
    /// `1 / (n0 * n1 * n2)` so that r2c -> c2r is the identity.
    ///
    /// # Errors
    /// Returns [`MpiError::FftError`] if this is not a c2r plan,
    /// [`MpiError::SizeMismatch`] if a buffer is too small, or any transpose
    /// error.
    pub fn execute_c2r(&mut self, input: &[Complex<T>], output: &mut [T]) -> Result<(), MpiError> {
        if self.kind != RealKind::C2r {
            return Err(MpiError::FftError {
                message: "execute_c2r called on an r2c plan".to_string(),
            });
        }
        let [n0, n1, n2] = self.dims;
        let n2c = self.n2c;
        let (local_n0, local_n1) = (self.local_n0, self.local_n1);
        let pool = self.pool;

        let in_size = if self.flags.transposed_in {
            local_n1 * n0 * n2c
        } else {
            local_n0 * n1 * n2c
        };
        if input.len() < in_size {
            return Err(MpiError::SizeMismatch {
                expected: in_size,
                actual: input.len(),
            });
        }
        if output.len() < local_n0 * n1 * n2 {
            return Err(MpiError::SizeMismatch {
                expected: local_n0 * n1 * n2,
                actual: output.len(),
            });
        }

        // Step 1: distributed complex inverse FFT along n0. With `transposed_in`
        // the caller already handed us the `[local_n1][n0][n2c]` distribution the
        // forward transpose would have produced, so that collective is skipped.
        if self.flags.transposed_in {
            self.scratch[..in_size].copy_from_slice(&input[..in_size]);
        } else {
            distributed_transpose_batched(
                pool,
                &input[..local_n0 * n1 * n2c],
                &mut self.scratch,
                n0,
                n1,
                local_n0,
                n2c,
            )?;
        }
        self.fft_fibers_n0(local_n1);
        distributed_transpose_batched(
            pool,
            &self.scratch[..n0 * local_n1 * n2c],
            &mut self.work,
            n1,
            n0,
            local_n1,
            n2c,
        )?;

        // Step 2: local unnormalized 2-D c2r on each plane -> output[local_n0][n1][n2].
        let plane_in = n1 * n2c;
        let plane_out = n1 * n2;
        for i0 in 0..local_n0 {
            self.plane_plan.execute_c2r_unnormalized(
                &self.work[i0 * plane_in..i0 * plane_in + plane_in],
                &mut output[i0 * plane_out..i0 * plane_out + plane_out],
            );
        }

        // Step 3: single 1/(n0*n1*n2) normalization for the whole 3D transform.
        let scale = T::ONE / T::from_usize(n0 * n1 * n2);
        for x in output[..local_n0 * plane_out].iter_mut() {
            *x = *x * scale;
        }
        Ok(())
    }

    /// Apply the local length-`n0` complex FFT to every `(local_n1, n2c)` fiber
    /// held in `self.scratch` as `[local_n1][n0][n2c]` (n0 axis has stride n2c).
    fn fft_fibers_n0(&mut self, local_n1: usize) {
        let n0 = self.dims[0];
        let n2c = self.n2c;
        let mut fiber_in = vec![Complex::<T>::zero(); n0];
        let mut fiber_out = vec![Complex::<T>::zero(); n0];
        for i1 in 0..local_n1 {
            for i2c in 0..n2c {
                let plane = i1 * n0 * n2c;
                for i0 in 0..n0 {
                    fiber_in[i0] = self.scratch[plane + i0 * n2c + i2c];
                }
                self.n0_plan.execute(&fiber_in, &mut fiber_out);
                for i0 in 0..n0 {
                    self.scratch[plane + i0 * n2c + i2c] = fiber_out[i0];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_half_complex_size() {
        // FFTW r2c convention: last dim n -> n/2 + 1 complex coefficients.
        assert_eq!(8 / 2 + 1, 5);
        assert_eq!(9 / 2 + 1, 5); // odd last dim
        assert_eq!(5 / 2 + 1, 3);
    }
}
