//! Real-to-real plan type (`R2rPlan`).
//!
//! Extracted from `types.rs` to keep individual files under 2000 lines.

use crate::api::Flags;
use crate::kernel::Float;
use crate::rdft::solvers::R2rKind;

/// A plan for executing real-to-real transforms (DCT/DST/DHT).
///
/// Real-to-real transforms map real input to real output, and include:
/// - DCT (Discrete Cosine Transform) types I-IV
/// - DST (Discrete Sine Transform) types I-IV
/// - DHT (Discrete Hartley Transform)
pub struct R2rPlan<T: Float> {
    /// Transform size
    n: usize,
    /// Transform kind
    kind: R2rKind,
    _marker: core::marker::PhantomData<T>,
}
impl<T: Float> R2rPlan<T> {
    /// Create a 1D real-to-real transform plan.
    ///
    /// # Arguments
    /// * `n` - Transform size
    /// * `kind` - Type of transform (DCT, DST, or DHT variant)
    /// * `flags` - Planning flags
    ///
    /// # Returns
    /// A plan that transforms n real values to n real values.
    #[must_use]
    pub fn r2r_1d(n: usize, kind: R2rKind, _flags: Flags) -> Option<Self> {
        if n == 0 {
            return None;
        }
        Some(Self {
            n,
            kind,
            _marker: core::marker::PhantomData,
        })
    }
    /// Create a DCT-I (REDFT00) plan.
    #[must_use]
    pub fn dct1(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Redft00, flags)
    }
    /// Create a DCT-II (REDFT10) plan - the "standard" DCT.
    #[must_use]
    pub fn dct2(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Redft10, flags)
    }
    /// Create a DCT-III (REDFT01) plan - the inverse of DCT-II.
    #[must_use]
    pub fn dct3(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Redft01, flags)
    }
    /// Create a DCT-IV (REDFT11) plan.
    #[must_use]
    pub fn dct4(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Redft11, flags)
    }
    /// Create a DST-I (RODFT00) plan.
    #[must_use]
    pub fn dst1(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Rodft00, flags)
    }
    /// Create a DST-II (RODFT10) plan.
    #[must_use]
    pub fn dst2(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Rodft10, flags)
    }
    /// Create a DST-III (RODFT01) plan - the inverse of DST-II.
    #[must_use]
    pub fn dst3(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Rodft01, flags)
    }
    /// Create a DST-IV (RODFT11) plan.
    #[must_use]
    pub fn dst4(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Rodft11, flags)
    }
    /// Create a DHT (Discrete Hartley Transform) plan.
    #[must_use]
    pub fn dht(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Dht, flags)
    }
    /// Get the transform size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.n
    }
    /// Get the transform kind.
    #[must_use]
    pub fn kind(&self) -> R2rKind {
        self.kind
    }
    /// Execute the plan.
    ///
    /// # Panics
    /// Panics if buffer sizes don't match the plan size.
    pub fn execute(&self, input: &[T], output: &mut [T]) {
        use crate::rdft::solvers::R2rSolver;
        assert_eq!(input.len(), self.n, "Input size must match plan size");
        assert_eq!(output.len(), self.n, "Output size must match plan size");
        let solver = R2rSolver::new(self.kind);
        solver.execute(input, output);
    }
    /// Execute the plan in-place.
    ///
    /// # Panics
    /// Panics if buffer size doesn't match the plan size.
    pub fn execute_inplace(&self, data: &mut [T]) {
        use crate::rdft::solvers::R2rSolver;
        assert_eq!(data.len(), self.n, "Data size must match plan size");
        let input = data.to_vec();
        let solver = R2rSolver::new(self.kind);
        solver.execute(&input, data);
    }
}
