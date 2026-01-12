//! MPI distributed FFT support.
//!
//! This module provides distributed FFT computation using MPI, similar to FFTW-MPI.
//! It enables computing large-scale FFTs across multiple processes.
//!
//! # Features
//!
//! - 2D, 3D, and N-D distributed FFTs
//! - Slab decomposition (row-major distribution)
//! - Efficient all-to-all transpose operations
//! - Compatible with FFTW-MPI data layouts
//!
//! # Example
//!
//! ```ignore
//! use oxifft::mpi::{MpiPool, MpiPlan2D, MpiFlags};
//! use mpi::topology::Communicator;
//!
//! // Initialize MPI
//! let universe = mpi::initialize().unwrap();
//! let world = universe.world();
//!
//! // Create MPI pool
//! let pool = MpiPool::new(world.duplicate());
//!
//! // Get local allocation size
//! let (local_n0, local_0_start, alloc_local) = local_size_2d(n0, n1, &pool);
//!
//! // Create plan
//! let plan = MpiPlan2D::new(n0, n1, Direction::Forward, MpiFlags::default(), &pool)?;
//!
//! // Execute
//! let mut data = vec![Complex::zero(); alloc_local];
//! plan.execute_inplace(&mut data);
//! ```

mod distribution;
mod error;
mod local_size;
mod plans;
mod pool;
mod transpose;

pub use distribution::{Distribution, LocalPartition};
pub use error::MpiError;
pub use local_size::{local_size_2d, local_size_3d, local_size_nd};
pub use plans::{MpiPlan2D, MpiPlan3D, MpiPlanND};
pub use pool::{MpiFloat, MpiPool};
pub use transpose::distributed_transpose;

use crate::api::Flags;

/// MPI-specific planning flags.
#[derive(Debug, Clone, Copy, Default)]
pub struct MpiFlags {
    /// Base FFT planning flags.
    pub base: Flags,
    /// Output data in transposed layout (avoids final transpose).
    /// Corresponds to FFTW_MPI_TRANSPOSED_OUT.
    pub transposed_out: bool,
    /// Input data is already transposed.
    /// Corresponds to FFTW_MPI_TRANSPOSED_IN.
    pub transposed_in: bool,
}

impl MpiFlags {
    /// Create new MPI flags with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the base FFT planning flags.
    pub fn with_base(mut self, flags: Flags) -> Self {
        self.base = flags;
        self
    }

    /// Enable transposed output (skips final transpose).
    pub fn transposed_out(mut self) -> Self {
        self.transposed_out = true;
        self
    }

    /// Indicate that input is already transposed.
    pub fn transposed_in(mut self) -> Self {
        self.transposed_in = true;
        self
    }

    /// Convenience: create ESTIMATE flags.
    pub fn estimate() -> Self {
        Self {
            base: Flags::ESTIMATE,
            ..Default::default()
        }
    }

    /// Convenience: create MEASURE flags.
    pub fn measure() -> Self {
        Self {
            base: Flags::MEASURE,
            ..Default::default()
        }
    }
}
