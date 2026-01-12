//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::api::{Direction, Flags};
use crate::kernel::{Complex, Float};
use crate::prelude::*;

use super::types::{
    Plan, Plan2D, PlanND, RealPlan2D, RealPlan3D, RealPlanND, SplitPlan, SplitPlan2D, SplitPlan3D,
    SplitPlanND,
};

/// Convenience function for N-dimensional forward FFT.
///
/// Input is in row-major order.
pub fn fft_nd<T: Float>(input: &[Complex<T>], dims: &[usize]) -> Vec<Complex<T>> {
    let total: usize = dims.iter().product();
    assert_eq!(
        input.len(),
        total,
        "Input size must match product of dimensions"
    );
    let mut output = vec![Complex::zero(); total];
    if let Some(plan) = PlanND::new(dims, Direction::Forward, Flags::ESTIMATE) {
        plan.execute(input, &mut output);
    }
    output
}
/// Convenience function for N-dimensional inverse FFT with normalization.
///
/// Normalizes by 1/(product of dimensions).
pub fn ifft_nd<T: Float>(input: &[Complex<T>], dims: &[usize]) -> Vec<Complex<T>> {
    let total: usize = dims.iter().product();
    assert_eq!(
        input.len(),
        total,
        "Input size must match product of dimensions"
    );
    let mut output = vec![Complex::zero(); total];
    if let Some(plan) = PlanND::new(dims, Direction::Backward, Flags::ESTIMATE) {
        plan.execute(input, &mut output);
        let scale = T::from_usize(total);
        for x in &mut output {
            *x = *x / scale;
        }
    }
    output
}
/// Convenience function for 2D forward FFT.
///
/// Input is row-major with n0 rows and n1 columns.
pub fn fft2d<T: Float>(input: &[Complex<T>], n0: usize, n1: usize) -> Vec<Complex<T>> {
    assert_eq!(input.len(), n0 * n1, "Input size must match n0 × n1");
    let mut output = vec![Complex::zero(); n0 * n1];
    if let Some(plan) = Plan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE) {
        plan.execute(input, &mut output);
    }
    output
}
/// Convenience function for 2D inverse FFT with normalization.
///
/// Normalizes by 1/(n0 × n1).
pub fn ifft2d<T: Float>(input: &[Complex<T>], n0: usize, n1: usize) -> Vec<Complex<T>> {
    assert_eq!(input.len(), n0 * n1, "Input size must match n0 × n1");
    let mut output = vec![Complex::zero(); n0 * n1];
    if let Some(plan) = Plan2D::new(n0, n1, Direction::Backward, Flags::ESTIMATE) {
        plan.execute(input, &mut output);
        let scale = T::from_usize(n0 * n1);
        for x in &mut output {
            *x = *x / scale;
        }
    }
    output
}
/// Convenience function for 1D forward FFT.
///
/// Creates a plan, executes it, and returns the result.
pub fn fft<T: Float>(input: &[Complex<T>]) -> Vec<Complex<T>> {
    let n = input.len();
    let mut output = vec![Complex::zero(); n];
    if let Some(plan) = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
        plan.execute(input, &mut output);
    }
    output
}
/// Convenience function for 1D inverse FFT with normalization.
///
/// Creates a plan, executes it, and normalizes by 1/N.
pub fn ifft<T: Float>(input: &[Complex<T>]) -> Vec<Complex<T>> {
    let n = input.len();
    let mut output = vec![Complex::zero(); n];
    if let Some(plan) = Plan::dft_1d(n, Direction::Backward, Flags::ESTIMATE) {
        plan.execute(input, &mut output);
        let scale = T::from_usize(n);
        for x in &mut output {
            *x = *x / scale;
        }
    }
    output
}
/// Convenience function for 1D Real-to-Complex FFT.
///
/// Takes N real values and produces N/2+1 complex values.
/// The output satisfies conjugate symmetry: X\[k\] = X\[N-k\]*.
pub fn rfft<T: Float>(input: &[T]) -> Vec<Complex<T>> {
    use crate::rdft::solvers::R2cSolver;
    let n = input.len();
    let mut output = vec![Complex::zero(); n / 2 + 1];
    R2cSolver::new(n).execute(input, &mut output);
    output
}
/// Convenience function for 1D Complex-to-Real FFT with normalization.
///
/// Takes N/2+1 complex values (conjugate symmetric) and produces N real values.
/// This is the inverse of rfft.
pub fn irfft<T: Float>(input: &[Complex<T>], n: usize) -> Vec<T> {
    use crate::rdft::solvers::C2rSolver;
    let mut output = vec![T::ZERO; n];
    C2rSolver::new(n).execute_normalized(input, &mut output);
    output
}
/// Convenience function for split-complex FFT.
///
/// Computes the forward FFT of split-complex input.
pub fn fft_split<T: Float>(in_real: &[T], in_imag: &[T]) -> (Vec<T>, Vec<T>) {
    let n = in_real.len();
    assert_eq!(
        in_imag.len(),
        n,
        "Real and imaginary arrays must have same length"
    );
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let plan = SplitPlan::dft_1d(n, Direction::Forward, Flags::ESTIMATE)
        .expect("Failed to create split-complex FFT plan");
    let mut out_real = vec![T::zero(); n];
    let mut out_imag = vec![T::zero(); n];
    plan.execute(in_real, in_imag, &mut out_real, &mut out_imag);
    (out_real, out_imag)
}
/// Convenience function for split-complex IFFT.
///
/// Computes the inverse FFT of split-complex input, with normalization.
pub fn ifft_split<T: Float>(in_real: &[T], in_imag: &[T]) -> (Vec<T>, Vec<T>) {
    let n = in_real.len();
    assert_eq!(
        in_imag.len(),
        n,
        "Real and imaginary arrays must have same length"
    );
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let plan = SplitPlan::dft_1d(n, Direction::Backward, Flags::ESTIMATE)
        .expect("Failed to create split-complex IFFT plan");
    let mut out_real = vec![T::zero(); n];
    let mut out_imag = vec![T::zero(); n];
    plan.execute(in_real, in_imag, &mut out_real, &mut out_imag);
    let scale = T::one() / T::from_usize(n);
    for r in &mut out_real {
        *r = *r * scale;
    }
    for i in &mut out_imag {
        *i = *i * scale;
    }
    (out_real, out_imag)
}
/// Convenience function for batched 1D forward FFT.
///
/// Performs `howmany` independent 1D FFTs, each of size `n`.
/// Input and output are contiguous: batch `i` starts at index `i * n`.
pub fn fft_batch<T: Float>(input: &[Complex<T>], n: usize, howmany: usize) -> Vec<Complex<T>> {
    use crate::dft::problem::Sign;
    use crate::dft::solvers::VrankGeq1Solver;
    assert_eq!(
        input.len(),
        n * howmany,
        "Input size must match n × howmany"
    );
    let mut output = vec![Complex::zero(); n * howmany];
    let solver = VrankGeq1Solver::new_contiguous(n, howmany);
    solver.execute(input, &mut output, Sign::Forward);
    output
}
/// Convenience function for batched 1D inverse FFT with normalization.
///
/// Performs `howmany` independent 1D inverse FFTs, each of size `n`.
/// Normalizes each output by 1/n.
pub fn ifft_batch<T: Float>(input: &[Complex<T>], n: usize, howmany: usize) -> Vec<Complex<T>> {
    use crate::dft::problem::Sign;
    use crate::dft::solvers::VrankGeq1Solver;
    assert_eq!(
        input.len(),
        n * howmany,
        "Input size must match n × howmany"
    );
    let mut output = vec![Complex::zero(); n * howmany];
    let solver = VrankGeq1Solver::new_contiguous(n, howmany);
    solver.execute(input, &mut output, Sign::Backward);
    let scale = T::from_usize(n);
    for x in &mut output {
        *x = *x / scale;
    }
    output
}
/// Convenience function for batched 1D Real-to-Complex FFT.
///
/// Performs `howmany` independent R2C FFTs, each of size `n`.
/// Input batches are contiguous (size n), output batches are contiguous (size n/2+1).
pub fn rfft_batch<T: Float>(input: &[T], n: usize, howmany: usize) -> Vec<Complex<T>> {
    use crate::rdft::solvers::RdftVrankGeq1Solver;
    assert_eq!(
        input.len(),
        n * howmany,
        "Input size must match n × howmany"
    );
    let out_len = n / 2 + 1;
    let mut output = vec![Complex::zero(); out_len * howmany];
    let solver = RdftVrankGeq1Solver::new(n, howmany, 1, 1, n as isize, out_len as isize);
    solver.execute_r2c(input, &mut output);
    output
}
/// Convenience function for batched 1D Complex-to-Real FFT with normalization.
///
/// Performs `howmany` independent C2R FFTs, each producing `n` real values.
/// Input batches have size n/2+1, output batches have size n.
pub fn irfft_batch<T: Float>(input: &[Complex<T>], n: usize, howmany: usize) -> Vec<T> {
    use crate::rdft::solvers::RdftVrankGeq1Solver;
    let in_len = n / 2 + 1;
    assert_eq!(
        input.len(),
        in_len * howmany,
        "Input size must match (n/2+1) × howmany"
    );
    let mut output = vec![T::ZERO; n * howmany];
    let solver = RdftVrankGeq1Solver::new(n, howmany, 1, 1, in_len as isize, n as isize);
    solver.execute_c2r(input, &mut output);
    output
}
/// Convenience function for 2D Real-to-Complex FFT.
///
/// Takes n0×n1 real values and produces n0×(n1/2+1) complex values.
pub fn rfft2d<T: Float>(input: &[T], n0: usize, n1: usize) -> Vec<Complex<T>> {
    let expected_in = n0 * n1;
    assert_eq!(input.len(), expected_in, "Input size must match n0 × n1");
    let out_len = n0 * (n1 / 2 + 1);
    let mut output = vec![Complex::zero(); out_len];
    let plan = RealPlan2D::r2c(n0, n1, Flags::ESTIMATE).expect("Failed to create plan");
    plan.execute_r2c(input, &mut output);
    output
}
/// Convenience function for 2D Complex-to-Real FFT with normalization.
///
/// Takes n0×(n1/2+1) complex values and produces n0×n1 real values.
pub fn irfft2d<T: Float>(input: &[Complex<T>], n0: usize, n1: usize) -> Vec<T> {
    let expected_in = n0 * (n1 / 2 + 1);
    assert_eq!(
        input.len(),
        expected_in,
        "Input size must match n0 × (n1/2+1)"
    );
    let out_len = n0 * n1;
    let mut output = vec![T::ZERO; out_len];
    let plan = RealPlan2D::c2r(n0, n1, Flags::ESTIMATE).expect("Failed to create plan");
    plan.execute_c2r(input, &mut output);
    let scale = T::one() / T::from_usize(out_len);
    for x in &mut output {
        *x = *x * scale;
    }
    output
}
/// Convenience function for 3D Real-to-Complex FFT.
///
/// Takes n0×n1×n2 real values and produces n0×n1×(n2/2+1) complex values.
pub fn rfft3d<T: Float>(input: &[T], n0: usize, n1: usize, n2: usize) -> Vec<Complex<T>> {
    let expected_in = n0 * n1 * n2;
    assert_eq!(
        input.len(),
        expected_in,
        "Input size must match n0 × n1 × n2"
    );
    let out_len = n0 * n1 * (n2 / 2 + 1);
    let mut output = vec![Complex::zero(); out_len];
    let plan = RealPlan3D::r2c(n0, n1, n2, Flags::ESTIMATE).expect("Failed to create plan");
    plan.execute_r2c(input, &mut output);
    output
}
/// Convenience function for 3D Complex-to-Real FFT with normalization.
///
/// Takes n0×n1×(n2/2+1) complex values and produces n0×n1×n2 real values.
pub fn irfft3d<T: Float>(input: &[Complex<T>], n0: usize, n1: usize, n2: usize) -> Vec<T> {
    let expected_in = n0 * n1 * (n2 / 2 + 1);
    assert_eq!(
        input.len(),
        expected_in,
        "Input size must match n0 × n1 × (n2/2+1)"
    );
    let out_len = n0 * n1 * n2;
    let mut output = vec![T::ZERO; out_len];
    let plan = RealPlan3D::c2r(n0, n1, n2, Flags::ESTIMATE).expect("Failed to create plan");
    plan.execute_c2r(input, &mut output);
    let scale = T::one() / T::from_usize(out_len);
    for x in &mut output {
        *x = *x * scale;
    }
    output
}
/// Convenience function for N-dimensional Real-to-Complex FFT.
///
/// Takes product(dims) real values and produces prefix×(last/2+1) complex values.
pub fn rfft_nd<T: Float>(input: &[T], dims: &[usize]) -> Vec<Complex<T>> {
    assert!(!dims.is_empty(), "Dimensions cannot be empty");
    let expected_in: usize = dims.iter().product();
    assert_eq!(
        input.len(),
        expected_in,
        "Input size must match product of dims"
    );
    let last = *dims
        .last()
        .expect("Dimensions cannot be empty (checked above)");
    let prefix: usize = dims[..dims.len() - 1].iter().product();
    let out_len = prefix.max(1) * (last / 2 + 1);
    let mut output = vec![Complex::zero(); out_len];
    let plan = RealPlanND::r2c(dims, Flags::ESTIMATE).expect("Failed to create plan");
    plan.execute_r2c(input, &mut output);
    output
}
/// Convenience function for N-dimensional Complex-to-Real FFT with normalization.
///
/// Takes prefix×(last/2+1) complex values and produces product(dims) real values.
pub fn irfft_nd<T: Float>(input: &[Complex<T>], dims: &[usize]) -> Vec<T> {
    assert!(!dims.is_empty(), "Dimensions cannot be empty");
    let last = *dims
        .last()
        .expect("Dimensions cannot be empty (checked above)");
    let prefix: usize = dims[..dims.len() - 1].iter().product();
    let expected_in = prefix.max(1) * (last / 2 + 1);
    assert_eq!(
        input.len(),
        expected_in,
        "Input size must match prefix × (last/2+1)"
    );
    let out_len: usize = dims.iter().product();
    let mut output = vec![T::ZERO; out_len];
    let plan = RealPlanND::c2r(dims, Flags::ESTIMATE).expect("Failed to create plan");
    plan.execute_c2r(input, &mut output);
    let scale = T::one() / T::from_usize(out_len);
    for x in &mut output {
        *x = *x * scale;
    }
    output
}
/// Convenience function for 2D split-complex forward FFT.
pub fn fft2d_split<T: Float>(
    in_real: &[T],
    in_imag: &[T],
    n0: usize,
    n1: usize,
) -> (Vec<T>, Vec<T>) {
    let total = n0 * n1;
    assert_eq!(in_real.len(), total);
    assert_eq!(in_imag.len(), total);
    let mut out_real = vec![T::ZERO; total];
    let mut out_imag = vec![T::ZERO; total];
    let plan = SplitPlan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE)
        .expect("Failed to create plan");
    plan.execute(in_real, in_imag, &mut out_real, &mut out_imag);
    (out_real, out_imag)
}
/// Convenience function for 2D split-complex inverse FFT with normalization.
pub fn ifft2d_split<T: Float>(
    in_real: &[T],
    in_imag: &[T],
    n0: usize,
    n1: usize,
) -> (Vec<T>, Vec<T>) {
    let total = n0 * n1;
    assert_eq!(in_real.len(), total);
    assert_eq!(in_imag.len(), total);
    let mut out_real = vec![T::ZERO; total];
    let mut out_imag = vec![T::ZERO; total];
    let plan = SplitPlan2D::new(n0, n1, Direction::Backward, Flags::ESTIMATE)
        .expect("Failed to create plan");
    plan.execute(in_real, in_imag, &mut out_real, &mut out_imag);
    (out_real, out_imag)
}
/// Convenience function for 3D split-complex forward FFT.
pub fn fft3d_split<T: Float>(
    in_real: &[T],
    in_imag: &[T],
    n0: usize,
    n1: usize,
    n2: usize,
) -> (Vec<T>, Vec<T>) {
    let total = n0 * n1 * n2;
    assert_eq!(in_real.len(), total);
    assert_eq!(in_imag.len(), total);
    let mut out_real = vec![T::ZERO; total];
    let mut out_imag = vec![T::ZERO; total];
    let plan = SplitPlan3D::new(n0, n1, n2, Direction::Forward, Flags::ESTIMATE)
        .expect("Failed to create plan");
    plan.execute(in_real, in_imag, &mut out_real, &mut out_imag);
    (out_real, out_imag)
}
/// Convenience function for 3D split-complex inverse FFT with normalization.
pub fn ifft3d_split<T: Float>(
    in_real: &[T],
    in_imag: &[T],
    n0: usize,
    n1: usize,
    n2: usize,
) -> (Vec<T>, Vec<T>) {
    let total = n0 * n1 * n2;
    assert_eq!(in_real.len(), total);
    assert_eq!(in_imag.len(), total);
    let mut out_real = vec![T::ZERO; total];
    let mut out_imag = vec![T::ZERO; total];
    let plan = SplitPlan3D::new(n0, n1, n2, Direction::Backward, Flags::ESTIMATE)
        .expect("Failed to create plan");
    plan.execute(in_real, in_imag, &mut out_real, &mut out_imag);
    (out_real, out_imag)
}
/// Convenience function for N-dimensional split-complex forward FFT.
pub fn fft_nd_split<T: Float>(in_real: &[T], in_imag: &[T], dims: &[usize]) -> (Vec<T>, Vec<T>) {
    let total: usize = dims.iter().product();
    assert_eq!(in_real.len(), total);
    assert_eq!(in_imag.len(), total);
    let mut out_real = vec![T::ZERO; total];
    let mut out_imag = vec![T::ZERO; total];
    let plan =
        SplitPlanND::new(dims, Direction::Forward, Flags::ESTIMATE).expect("Failed to create plan");
    plan.execute(in_real, in_imag, &mut out_real, &mut out_imag);
    (out_real, out_imag)
}
/// Convenience function for N-dimensional split-complex inverse FFT with normalization.
pub fn ifft_nd_split<T: Float>(in_real: &[T], in_imag: &[T], dims: &[usize]) -> (Vec<T>, Vec<T>) {
    let total: usize = dims.iter().product();
    assert_eq!(in_real.len(), total);
    assert_eq!(in_imag.len(), total);
    let mut out_real = vec![T::ZERO; total];
    let mut out_imag = vec![T::ZERO; total];
    let plan = SplitPlanND::new(dims, Direction::Backward, Flags::ESTIMATE)
        .expect("Failed to create plan");
    plan.execute(in_real, in_imag, &mut out_real, &mut out_imag);
    (out_real, out_imag)
}
