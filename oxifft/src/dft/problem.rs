//! DFT problem definition.

use core::hash::{Hash, Hasher};

use crate::kernel::{Complex, Float, Problem, ProblemKind, Tensor};

/// Transform sign/direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
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
///
/// This is a low-level, FFTW-style problem descriptor built directly on raw
/// buffer pointers. It is **unsafe to construct**: the pointer fields are
/// `pub(crate)` and the only public constructors ([`Self::new_1d`],
/// [`Self::new_2d`]) are `unsafe fn`, so safe code cannot fabricate a
/// `DftProblem` with dangling, aliasing, or undersized buffers. The mutating
/// operations ([`Self::zero`] via the [`Problem`] trait, and the `solve` method
/// of [`crate::dft::DftPlan`]) rely on the construction-time contract for their
/// soundness.
///
/// For that reasoning to hold, the *size* fields must be immutable from safe
/// code too: if `sz` were publicly writable, a caller holding a legitimately
/// constructed descriptor could enlarge it after the fact and make `solve`
/// build a slice far longer than the buffer the `unsafe` constructor was
/// promised. They are therefore `pub(crate)` and exposed read-only through
/// [`Self::sz`] / [`Self::vecsz`]. As belt-and-braces, the element count
/// recorded at construction time (`buffer_len`) — not the possibly-derived
/// tensor product — is what bounds every raw-pointer access.
#[derive(Debug, Clone)]
pub struct DftProblem<T: Float> {
    /// Transform dimensions with strides.
    pub(crate) sz: Tensor,
    /// Batch/vector dimensions.
    pub(crate) vecsz: Tensor,
    /// Number of `Complex<T>` elements the constructor's caller promised each
    /// buffer is valid for. This is the authoritative bound for all
    /// raw-pointer access; it can never be changed after construction.
    pub(crate) buffer_len: usize,
    /// Input buffer pointer.
    pub(crate) input: *mut Complex<T>,
    /// Output buffer pointer.
    pub(crate) output: *mut Complex<T>,
    /// Transform direction.
    pub sign: Sign,
}

// SAFETY: `DftProblem` carries raw pointers, so it is `Send`/`Sync` only under a
// contract. That contract is established by the `unsafe` constructors
// (`new_1d`/`new_2d`): the caller guarantees the pointers are valid for the
// transform's element count and that any cross-thread or aliased use (this type
// is `Clone`) does not create a data race. Given that contract, moving/sharing
// the descriptor across threads is sound.
unsafe impl<T: Float> Send for DftProblem<T> {}
unsafe impl<T: Float> Sync for DftProblem<T> {}

impl<T: Float> DftProblem<T> {
    /// Create a new 1D DFT problem from raw input/output buffer pointers.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - `input` is valid for reads of `n` `Complex<T>` values, and `output` is
    ///   valid for writes of `n` `Complex<T>` values, for as long as the
    ///   returned problem (or any clone of it) is used;
    /// - `input` and `output` either do not overlap, or are identical (in-place);
    /// - both pointers are correctly aligned for `Complex<T>`;
    /// - because `DftProblem` is `Clone + Send + Sync`, the caller does not use
    ///   the descriptor (or a clone) from multiple threads in a way that races
    ///   on the referenced buffers.
    #[must_use]
    pub unsafe fn new_1d(
        n: usize,
        input: *mut Complex<T>,
        output: *mut Complex<T>,
        sign: Sign,
    ) -> Self {
        Self {
            sz: Tensor::rank1(n),
            vecsz: Tensor::empty(),
            buffer_len: n,
            input,
            output,
            sign,
        }
    }

    /// Create a 2D DFT problem from raw input/output buffer pointers.
    ///
    /// # Safety
    /// Same contract as [`Self::new_1d`], with the element count taken to be
    /// `n0 * n1`. If that product overflows `usize` the problem is degenerate
    /// (no buffer can be that long); it is recorded as empty, so `solve`/`zero`
    /// become no-ops rather than reading a wrapped length.
    #[must_use]
    pub unsafe fn new_2d(
        n0: usize,
        n1: usize,
        input: *mut Complex<T>,
        output: *mut Complex<T>,
        sign: Sign,
    ) -> Self {
        Self {
            sz: Tensor::rank2(n0, n1),
            vecsz: Tensor::empty(),
            buffer_len: n0.checked_mul(n1).unwrap_or(0),
            input,
            output,
            sign,
        }
    }

    /// Transform dimensions with strides.
    ///
    /// Read-only: see the type-level documentation for why safe code must not
    /// be able to change the descriptor's extent after construction.
    #[must_use]
    pub const fn sz(&self) -> &Tensor {
        &self.sz
    }

    /// Batch/vector dimensions.
    ///
    /// Read-only for the same reason as [`Self::sz`].
    #[must_use]
    pub const fn vecsz(&self) -> &Tensor {
        &self.vecsz
    }

    /// Number of `Complex<T>` elements each buffer is guaranteed valid for.
    #[must_use]
    pub const fn buffer_len(&self) -> usize {
        self.buffer_len
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
        // Zero the output buffer.  The bound is the construction-time element
        // count, never a product recomputed from the tensors: that product is
        // what the `unsafe` constructor's caller actually vouched for, and it
        // cannot overflow or be enlarged afterwards.
        let size = self.buffer_len;
        unsafe {
            for i in 0..size {
                *self.output.add(i) = Complex::zero();
            }
        }
    }

    fn total_size(&self) -> usize {
        // Saturating: a descriptor whose extents overflow describes no buffer
        // that can exist, and this is a plain reporting accessor — it must not
        // become a panic (debug) / wrapping surprise (release).
        self.transform_size().saturating_mul(self.batch_size())
    }

    fn is_inplace(&self) -> bool {
        self.input == self.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dft::DftPlan;
    use crate::kernel::{OpCount, Plan};

    #[test]
    fn dft_problem_solve_via_unsafe_ctor() {
        let mut input = vec![Complex::new(1.0_f64, 0.0); 4];
        let mut output = vec![Complex::zero(); 4];

        // SAFETY: `input`/`output` are valid, non-overlapping, aligned length-4
        // buffers used only from this thread — the `new_1d` contract holds.
        let problem = unsafe {
            DftProblem::new_1d(4, input.as_mut_ptr(), output.as_mut_ptr(), Sign::Forward)
        };
        let plan = DftPlan::<f64>::new("test-dft", OpCount::zero());
        plan.solve(&problem);

        // 4-point DFT of [1,1,1,1] = [4,0,0,0].
        assert!((output[0].re - 4.0).abs() < 1e-10);
        assert!(output[0].im.abs() < 1e-10);
        for bin in &output[1..] {
            assert!(bin.re.abs() < 1e-10 && bin.im.abs() < 1e-10);
        }
    }

    /// Regression: the size tensors must not be reachable (let alone writable)
    /// from safe code, and every raw-pointer access must be bounded by the
    /// element count the `unsafe` constructor was given.
    ///
    /// Previously `sz` was a `pub` field, so a caller holding a legitimately
    /// constructed 8-element problem could write
    /// `problem.sz = Tensor::rank1(1 << 20);` in entirely safe code and then
    /// call the safe `DftPlan::solve`, which built an 8-element buffer into a
    /// 1 Mi-element slice via `from_raw_parts_mut`.
    #[test]
    fn buffer_len_records_the_construction_time_element_count() {
        let mut input = vec![Complex::new(1.0_f64, 0.0); 8];
        let mut output = vec![Complex::zero(); 8];

        // SAFETY: valid, non-overlapping, aligned length-8 buffers used only here.
        let problem = unsafe {
            DftProblem::new_1d(8, input.as_mut_ptr(), output.as_mut_ptr(), Sign::Forward)
        };
        assert_eq!(problem.buffer_len(), 8);
        assert_eq!(problem.sz().total_size(), 8);
        assert!(problem.vecsz().is_empty());
        // `sz` is read-only: `problem.sz = ...` no longer compiles.
    }

    /// A 2D problem whose `n0 * n1` overflows cannot describe any real buffer;
    /// it must be recorded as empty so `solve`/`zero` stay no-ops instead of
    /// writing through a wrapped length.
    #[test]
    fn overflowing_2d_extent_is_recorded_as_empty() {
        let mut buf = vec![Complex::new(1.0_f64, 2.0); 4];
        // SAFETY: the extents are degenerate, which is exactly what is under
        // test; `buffer_len` becomes 0 so no element is ever accessed.
        let problem = unsafe {
            DftProblem::new_2d(
                1_usize << 63,
                2,
                buf.as_mut_ptr(),
                buf.as_mut_ptr(),
                Sign::Forward,
            )
        };
        assert_eq!(problem.buffer_len(), 0);

        let plan = DftPlan::<f64>::new("test-dft", OpCount::zero());
        plan.solve(&problem);
        problem.zero();

        // Nothing was touched.
        assert!(buf.iter().all(|c| c.re == 1.0 && c.im == 2.0));
    }

    #[test]
    fn dft_problem_zero_clears_output() {
        let mut input = vec![Complex::new(2.0_f64, -3.0); 2];
        let mut output = vec![Complex::new(9.0_f64, 9.0); 2];

        // SAFETY: valid length-2 buffers used only here.
        let problem = unsafe {
            DftProblem::new_1d(2, input.as_mut_ptr(), output.as_mut_ptr(), Sign::Forward)
        };
        problem.zero();
        assert!(output.iter().all(|c| c.re == 0.0 && c.im == 0.0));
    }
}
