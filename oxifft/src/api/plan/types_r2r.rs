//! Real-to-real plan type (`R2rPlan`).
//!
//! Extracted from `types.rs` to keep individual files under 2000 lines.

use crate::api::Flags;
use crate::kernel::Float;
use crate::rdft::solvers::{R2rKind, R2rSolver};

/// A plan for executing real-to-real transforms (DCT/DST/DHT).
///
/// Real-to-real transforms map real input to real output, and include:
/// - DCT (Discrete Cosine Transform) types I-IV
/// - DST (Discrete Sine Transform) types I-IV
/// - DHT (Discrete Hartley Transform)
pub struct R2rPlan<T: Float> {
    /// Pre-built solver (twiddle tables + FFT plans cached at construction).
    solver: R2rSolver<T>,
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
    ///
    /// # Examples
    ///
    /// ```
    /// use oxifft::{Flags, R2rPlan};
    ///
    /// // Use the dct2 convenience constructor (DCT-II / REDFT10)
    /// let plan = R2rPlan::<f64>::dct2(8, Flags::ESTIMATE)
    ///     .expect("plan construction failed");
    /// let input = vec![1.0_f64; 8];
    /// let mut output = vec![0.0_f64; 8];
    /// plan.execute(&input, &mut output);
    /// // DCT-II of all-ones: first coefficient is positive (unnormalized sum)
    /// assert!(output[0] > 0.0);
    /// ```
    #[must_use]
    pub fn r2r_1d(n: usize, kind: R2rKind, _flags: Flags) -> Option<Self> {
        if n == 0 {
            return None;
        }
        Some(Self {
            solver: R2rSolver::new(kind, n),
        })
    }

    /// Create a DCT-I (REDFT00) plan.
    #[must_use]
    pub fn dct1(n: usize, flags: Flags) -> Option<Self> {
        Self::r2r_1d(n, R2rKind::Redft00, flags)
    }

    /// Create a DCT-II (REDFT10) plan - the "standard" DCT.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxifft::{Flags, R2rPlan};
    ///
    /// let plan = R2rPlan::<f64>::dct2(8, Flags::ESTIMATE)
    ///     .expect("plan construction failed");
    /// let input = vec![1.0_f64; 8];
    /// let mut output = vec![0.0_f64; 8];
    /// plan.execute(&input, &mut output);
    /// // DCT-II of all-ones: first coefficient is positive
    /// assert!(output[0] > 0.0);
    /// // All higher-frequency coefficients are zero for a constant signal
    /// assert!(output[1].abs() < 1e-10);
    /// ```
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
        self.solver.size()
    }

    /// Get the transform kind.
    #[must_use]
    pub fn kind(&self) -> R2rKind {
        self.solver.kind()
    }

    /// Execute the plan.
    ///
    /// # Panics
    /// Panics if buffer sizes don't match the plan size.
    pub fn execute(&self, input: &[T], output: &mut [T]) {
        assert_eq!(
            input.len(),
            self.solver.size(),
            "Input size must match plan size"
        );
        assert_eq!(
            output.len(),
            self.solver.size(),
            "Output size must match plan size"
        );
        self.solver.execute(input, output);
    }

    /// Execute the plan in-place.
    ///
    /// # Panics
    /// Panics if buffer size doesn't match the plan size.
    pub fn execute_inplace(&self, data: &mut [T]) {
        assert_eq!(
            data.len(),
            self.solver.size(),
            "Data size must match plan size"
        );
        let input = data.to_vec();
        self.solver.execute(&input, data);
    }
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use crate::api::Flags;

    #[test]
    // MIRI intentionally introduces floating-point non-determinism to detect
    // code that incorrectly assumes deterministic FP results. Bit-exact
    // comparison via `to_bits()` is therefore not meaningful under MIRI.
    // The same test logic is verified under native execution (no MIRI).
    fn execute_is_idempotent() {
        let plan = R2rPlan::<f64>::dct2(8, Flags::ESTIMATE).expect("plan");
        let input = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out1 = vec![0.0_f64; 8];
        let mut out2 = vec![0.0_f64; 8];
        plan.execute(&input, &mut out1);
        plan.execute(&input, &mut out2);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "execute must be bit-identical across calls"
            );
        }
    }
}
