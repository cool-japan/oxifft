//! Real-to-real plan type (`R2rPlan`).
//!
//! Extracted from `types.rs` to keep individual files under 2000 lines.

use crate::api::Flags;
use crate::kernel::Float;
use crate::prelude::*;
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
    pub fn r2r_1d(n: usize, kind: R2rKind, flags: Flags) -> Option<Self> {
        if n == 0 {
            return None;
        }
        Some(Self {
            solver: R2rSolver::new_with_flags(kind, n, flags),
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

/// A plan for 2D separable real-to-real transforms (DCT/DST/DHT).
///
/// Mirrors FFTW's `fftw_plan_r2r_2d`: the 1D transform of the requested kind is
/// applied along each axis in turn (row-column decomposition). A different kind
/// may be selected per axis via [`R2rPlan2D::new`], or the same kind for both
/// axes via [`R2rPlan2D::r2r_2d`].
///
/// Data is row-major: element `(i, j)` lives at index `i * n1 + j`.
///
/// # Examples
///
/// ```
/// use oxifft::api::R2rPlan2D;
/// use oxifft::rdft::solvers::R2rKind;
/// use oxifft::Flags;
///
/// let plan = R2rPlan2D::<f64>::r2r_2d(4, 8, R2rKind::Redft10, Flags::ESTIMATE)
///     .expect("plan construction failed");
/// let input = vec![1.0_f64; 4 * 8];
/// let mut output = vec![0.0_f64; 4 * 8];
/// plan.execute(&input, &mut output);
/// // 2D DCT-II of a constant field concentrates all energy in the DC term.
/// assert!(output[0] > 0.0);
/// assert!(output[1].abs() < 1e-9);
/// ```
pub struct R2rPlan2D<T: Float> {
    n0: usize,
    n1: usize,
    kind0: R2rKind,
    kind1: R2rKind,
    /// Solver applied along axis 0 (columns, length `n0`).
    solver0: R2rSolver<T>,
    /// Solver applied along axis 1 (rows, length `n1`).
    solver1: R2rSolver<T>,
}

impl<T: Float> R2rPlan2D<T> {
    /// Create a 2D real-to-real plan with a (possibly different) kind per axis.
    ///
    /// * `kind0` is applied along axis 0 (length `n0`).
    /// * `kind1` is applied along axis 1 (length `n1`).
    #[must_use]
    pub fn new(n0: usize, n1: usize, kind0: R2rKind, kind1: R2rKind, flags: Flags) -> Option<Self> {
        if n0 == 0 || n1 == 0 {
            return None;
        }
        Some(Self {
            n0,
            n1,
            kind0,
            kind1,
            solver0: R2rSolver::new_with_flags(kind0, n0, flags),
            solver1: R2rSolver::new_with_flags(kind1, n1, flags),
        })
    }

    /// Create a 2D real-to-real plan using the same `kind` along both axes.
    #[must_use]
    pub fn r2r_2d(n0: usize, n1: usize, kind: R2rKind, flags: Flags) -> Option<Self> {
        Self::new(n0, n1, kind, kind, flags)
    }

    /// Number of rows (axis-0 length).
    #[must_use]
    pub fn rows(&self) -> usize {
        self.n0
    }

    /// Number of columns (axis-1 length).
    #[must_use]
    pub fn cols(&self) -> usize {
        self.n1
    }

    /// Total element count (`n0 * n1`).
    #[must_use]
    pub fn size(&self) -> usize {
        self.n0 * self.n1
    }

    /// Transform kind applied along axis 0.
    #[must_use]
    pub fn kind0(&self) -> R2rKind {
        self.kind0
    }

    /// Transform kind applied along axis 1.
    #[must_use]
    pub fn kind1(&self) -> R2rKind {
        self.kind1
    }

    /// Execute the 2D transform.
    ///
    /// # Panics
    /// Panics if buffer sizes don't match `n0 * n1`.
    pub fn execute(&self, input: &[T], output: &mut [T]) {
        let total = self.n0 * self.n1;
        assert_eq!(input.len(), total, "Input size must match n0 * n1");
        assert_eq!(output.len(), total, "Output size must match n0 * n1");

        // Stage 1: transform along axis 1 (each contiguous row of length n1).
        let mut row_in = vec![T::ZERO; self.n1];
        let mut row_out = vec![T::ZERO; self.n1];
        for r in 0..self.n0 {
            let base = r * self.n1;
            row_in.copy_from_slice(&input[base..base + self.n1]);
            self.solver1.execute(&row_in, &mut row_out);
            output[base..base + self.n1].copy_from_slice(&row_out);
        }

        // Stage 2: transform along axis 0 (each column of length n0, stride n1).
        let mut col_in = vec![T::ZERO; self.n0];
        let mut col_out = vec![T::ZERO; self.n0];
        for c in 0..self.n1 {
            for r in 0..self.n0 {
                col_in[r] = output[r * self.n1 + c];
            }
            self.solver0.execute(&col_in, &mut col_out);
            for r in 0..self.n0 {
                output[r * self.n1 + c] = col_out[r];
            }
        }
    }

    /// Execute the 2D transform in-place.
    ///
    /// # Panics
    /// Panics if the buffer size doesn't match `n0 * n1`.
    pub fn execute_inplace(&self, data: &mut [T]) {
        assert_eq!(
            data.len(),
            self.n0 * self.n1,
            "Data size must match n0 * n1"
        );
        let input = data.to_vec();
        self.execute(&input, data);
    }
}

/// A plan for 3D separable real-to-real transforms (DCT/DST/DHT).
///
/// Mirrors FFTW's `fftw_plan_r2r_3d`: the 1D transform of the requested kind is
/// applied along each of the three axes in turn (row-column-tube). A different
/// kind may be selected per axis via [`R2rPlan3D::new`], or the same kind for
/// all axes via [`R2rPlan3D::r2r_3d`].
///
/// Data is row-major: element `(i, j, k)` lives at index
/// `i * n1 * n2 + j * n2 + k`.
pub struct R2rPlan3D<T: Float> {
    n0: usize,
    n1: usize,
    n2: usize,
    kind0: R2rKind,
    kind1: R2rKind,
    kind2: R2rKind,
    solver0: R2rSolver<T>,
    solver1: R2rSolver<T>,
    solver2: R2rSolver<T>,
}

impl<T: Float> R2rPlan3D<T> {
    /// Create a 3D real-to-real plan with a (possibly different) kind per axis.
    #[must_use]
    pub fn new(
        n0: usize,
        n1: usize,
        n2: usize,
        kind0: R2rKind,
        kind1: R2rKind,
        kind2: R2rKind,
        flags: Flags,
    ) -> Option<Self> {
        if n0 == 0 || n1 == 0 || n2 == 0 {
            return None;
        }
        Some(Self {
            n0,
            n1,
            n2,
            kind0,
            kind1,
            kind2,
            solver0: R2rSolver::new_with_flags(kind0, n0, flags),
            solver1: R2rSolver::new_with_flags(kind1, n1, flags),
            solver2: R2rSolver::new_with_flags(kind2, n2, flags),
        })
    }

    /// Create a 3D real-to-real plan using the same `kind` along all axes.
    #[must_use]
    pub fn r2r_3d(n0: usize, n1: usize, n2: usize, kind: R2rKind, flags: Flags) -> Option<Self> {
        Self::new(n0, n1, n2, kind, kind, kind, flags)
    }

    /// Total element count (`n0 * n1 * n2`).
    #[must_use]
    pub fn size(&self) -> usize {
        self.n0 * self.n1 * self.n2
    }

    /// Axis-0 length.
    #[must_use]
    pub fn dim0(&self) -> usize {
        self.n0
    }

    /// Axis-1 length.
    #[must_use]
    pub fn dim1(&self) -> usize {
        self.n1
    }

    /// Axis-2 length.
    #[must_use]
    pub fn dim2(&self) -> usize {
        self.n2
    }

    /// Transform kind applied along axis 0.
    #[must_use]
    pub fn kind0(&self) -> R2rKind {
        self.kind0
    }

    /// Transform kind applied along axis 1.
    #[must_use]
    pub fn kind1(&self) -> R2rKind {
        self.kind1
    }

    /// Transform kind applied along axis 2.
    #[must_use]
    pub fn kind2(&self) -> R2rKind {
        self.kind2
    }

    /// Execute the 3D transform.
    ///
    /// # Panics
    /// Panics if buffer sizes don't match `n0 * n1 * n2`.
    pub fn execute(&self, input: &[T], output: &mut [T]) {
        let total = self.n0 * self.n1 * self.n2;
        assert_eq!(input.len(), total, "Input size must match n0 * n1 * n2");
        assert_eq!(output.len(), total, "Output size must match n0 * n1 * n2");
        output.copy_from_slice(input);

        let plane = self.n1 * self.n2;

        // Stage 1: axis 2 (contiguous tubes of length n2).
        let mut buf2_in = vec![T::ZERO; self.n2];
        let mut buf2_out = vec![T::ZERO; self.n2];
        for base in (0..total).step_by(self.n2) {
            buf2_in.copy_from_slice(&output[base..base + self.n2]);
            self.solver2.execute(&buf2_in, &mut buf2_out);
            output[base..base + self.n2].copy_from_slice(&buf2_out);
        }

        // Stage 2: axis 1 (length n1, stride n2).
        let mut buf1_in = vec![T::ZERO; self.n1];
        let mut buf1_out = vec![T::ZERO; self.n1];
        for i in 0..self.n0 {
            for k in 0..self.n2 {
                for j in 0..self.n1 {
                    buf1_in[j] = output[i * plane + j * self.n2 + k];
                }
                self.solver1.execute(&buf1_in, &mut buf1_out);
                for j in 0..self.n1 {
                    output[i * plane + j * self.n2 + k] = buf1_out[j];
                }
            }
        }

        // Stage 3: axis 0 (length n0, stride n1 * n2).
        let mut buf0_in = vec![T::ZERO; self.n0];
        let mut buf0_out = vec![T::ZERO; self.n0];
        for j in 0..self.n1 {
            for k in 0..self.n2 {
                for i in 0..self.n0 {
                    buf0_in[i] = output[i * plane + j * self.n2 + k];
                }
                self.solver0.execute(&buf0_in, &mut buf0_out);
                for i in 0..self.n0 {
                    output[i * plane + j * self.n2 + k] = buf0_out[i];
                }
            }
        }
    }

    /// Execute the 3D transform in-place.
    ///
    /// # Panics
    /// Panics if the buffer size doesn't match `n0 * n1 * n2`.
    pub fn execute_inplace(&self, data: &mut [T]) {
        assert_eq!(
            data.len(),
            self.n0 * self.n1 * self.n2,
            "Data size must match n0 * n1 * n2"
        );
        let input = data.to_vec();
        self.execute(&input, data);
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
