//! Multi-dimensional RDFT solver.

use crate::kernel::Float;

/// Multi-dimensional RDFT solver.
pub struct RdftRankGeq2Solver<T: Float> {
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> Default for RdftRankGeq2Solver<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> RdftRankGeq2Solver<T> {
    /// Creates a new multi-dimensional RDFT solver instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns the solver name identifier (`"rdft-rank-geq2"`).
    #[must_use]
    pub fn name(&self) -> &'static str {
        "rdft-rank-geq2"
    }
}
