//! RDFT problem definition.

use core::hash::{Hash, Hasher};

use crate::kernel::{Complex, Float, Problem, ProblemKind, Tensor};

/// Kind of real DFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RdftKind {
    /// Real to Complex (forward real FFT)
    R2C,
    /// Complex to Real (inverse real FFT)
    C2R,
    /// Real to Half-Complex
    R2HC,
    /// Half-Complex to Real
    HC2R,
    /// Real to Real (DCT/DST type)
    R2R,
}

/// Real DFT problem.
#[derive(Debug, Clone)]
pub struct RdftProblem<T: Float> {
    /// Transform dimensions.
    pub sz: Tensor,
    /// Batch dimensions.
    pub vecsz: Tensor,
    /// Real input/output buffer.
    pub real_buf: *mut T,
    /// Complex input/output buffer (for R2C/C2R).
    pub complex_buf: *mut Complex<T>,
    /// Transform kind.
    pub kind: RdftKind,
}

// Safety: Pointers are only dereferenced during execution
unsafe impl<T: Float> Send for RdftProblem<T> {}
unsafe impl<T: Float> Sync for RdftProblem<T> {}

impl<T: Float> RdftProblem<T> {
    /// Create a 1D R2C problem.
    #[must_use]
    pub fn new_r2c_1d(n: usize, real_input: *mut T, complex_output: *mut Complex<T>) -> Self {
        Self {
            sz: Tensor::rank1(n),
            vecsz: Tensor::empty(),
            real_buf: real_input,
            complex_buf: complex_output,
            kind: RdftKind::R2C,
        }
    }

    /// Create a 1D C2R problem.
    #[must_use]
    pub fn new_c2r_1d(n: usize, complex_input: *mut Complex<T>, real_output: *mut T) -> Self {
        Self {
            sz: Tensor::rank1(n),
            vecsz: Tensor::empty(),
            real_buf: real_output,
            complex_buf: complex_input,
            kind: RdftKind::C2R,
        }
    }

    /// Get transform size.
    #[must_use]
    pub fn transform_size(&self) -> usize {
        self.sz.total_size()
    }

    /// Get complex buffer size for R2C/C2R.
    /// For real FFT of size n, complex output has n/2 + 1 elements.
    #[must_use]
    pub fn complex_size(&self) -> usize {
        self.transform_size() / 2 + 1
    }
}

impl<T: Float> Hash for RdftProblem<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.sz.hash(state);
        self.vecsz.hash(state);
        self.kind.hash(state);
    }
}

impl<T: Float> Problem for RdftProblem<T> {
    fn kind(&self) -> ProblemKind {
        ProblemKind::Rdft
    }

    fn zero(&self) {
        // Zero appropriate buffer based on kind
    }

    fn total_size(&self) -> usize {
        self.sz.total_size() * self.vecsz.total_size().max(1)
    }

    fn is_inplace(&self) -> bool {
        self.real_buf as *const () == self.complex_buf as *const ()
    }
}
