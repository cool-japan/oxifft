//! Multi-dimensional (rank >= 2) DFT solver.

use crate::kernel::Float;

/// Solver for multi-dimensional transforms.
pub struct RankGeq2Solver<T: Float> {
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> Default for RankGeq2Solver<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> RankGeq2Solver<T> {
    /// Create a new rank >= 2 solver.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }

    /// Solver name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        "dft-rank-geq2"
    }
}
