//! DFT problem definition.

use core::hash::{Hash, Hasher};

use crate::kernel::{Complex, Float, Problem, ProblemKind, Tensor};

/// Transform sign/direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sign {
    /// Forward transform (exponent = -1)
    Forward = -1,
    /// Backward/inverse transform (exponent = +1)
    Backward = 1,
}

impl Sign {
    /// Get the numeric value.
    #[must_use]
    pub const fn value(self) -> i32 {
        self as i32
    }
}

/// Complex DFT problem.
#[derive(Debug, Clone)]
pub struct DftProblem<T: Float> {
    /// Transform dimensions with strides.
    pub sz: Tensor,
    /// Batch/vector dimensions.
    pub vecsz: Tensor,
    /// Input buffer pointer.
    pub input: *mut Complex<T>,
    /// Output buffer pointer.
    pub output: *mut Complex<T>,
    /// Transform direction.
    pub sign: Sign,
}

// Safety: Pointers are only dereferenced during execution
unsafe impl<T: Float> Send for DftProblem<T> {}
unsafe impl<T: Float> Sync for DftProblem<T> {}

impl<T: Float> DftProblem<T> {
    /// Create a new 1D DFT problem.
    #[must_use]
    pub fn new_1d(n: usize, input: *mut Complex<T>, output: *mut Complex<T>, sign: Sign) -> Self {
        Self {
            sz: Tensor::rank1(n),
            vecsz: Tensor::empty(),
            input,
            output,
            sign,
        }
    }

    /// Create a 2D DFT problem.
    #[must_use]
    pub fn new_2d(
        n0: usize,
        n1: usize,
        input: *mut Complex<T>,
        output: *mut Complex<T>,
        sign: Sign,
    ) -> Self {
        Self {
            sz: Tensor::rank2(n0, n1),
            vecsz: Tensor::empty(),
            input,
            output,
            sign,
        }
    }

    /// Check if this is an in-place transform.
    #[must_use]
    pub fn is_inplace(&self) -> bool {
        self.input == self.output
    }

    /// Get the transform size (product of all dimensions).
    #[must_use]
    pub fn transform_size(&self) -> usize {
        self.sz.total_size()
    }

    /// Get the batch size (product of vector dimensions).
    #[must_use]
    pub fn batch_size(&self) -> usize {
        if self.vecsz.is_empty() {
            1
        } else {
            self.vecsz.total_size()
        }
    }
}

impl<T: Float> Hash for DftProblem<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.sz.hash(state);
        self.vecsz.hash(state);
        self.sign.hash(state);
        self.is_inplace().hash(state);
    }
}

impl<T: Float> Problem for DftProblem<T> {
    fn kind(&self) -> ProblemKind {
        ProblemKind::Dft
    }

    fn zero(&self) {
        // Zero the output buffer
        let size = self.sz.total_size() * self.vecsz.total_size().max(1);
        unsafe {
            for i in 0..size {
                *self.output.add(i) = Complex::zero();
            }
        }
    }

    fn total_size(&self) -> usize {
        self.transform_size() * self.batch_size()
    }

    fn is_inplace(&self) -> bool {
        self.input == self.output
    }
}
