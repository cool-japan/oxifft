//! Direct O(n²) DFT solver.
//!
//! Reference implementation for correctness testing.
//! This is the naive DFT computation: X\[k\] = Σ x\[n\] * W_N^(nk)
//! where W_N = e^(-2πi/N) for forward transform.

use crate::kernel::{Complex, Float};
use crate::prelude::*;

use super::super::problem::Sign;

/// Direct DFT solver using O(n²) algorithm.
///
/// This solver computes the DFT directly from the definition:
/// ```text
/// X[k] = Σ_{n=0}^{N-1} x[n] * e^(-2πink/N)  (forward)
/// x[n] = (1/N) Σ_{k=0}^{N-1} X[k] * e^(2πink/N)  (backward)
/// ```
///
/// Time complexity: O(n²)
/// Space complexity: O(1) additional
///
/// This solver is primarily used as a reference for testing other implementations.
pub struct DirectSolver<T: Float> {
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> Default for DirectSolver<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> DirectSolver<T> {
    /// Create a new direct solver.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }

    /// Solver name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        "dft-direct"
    }

    /// Execute the direct DFT computation.
    ///
    /// # Arguments
    /// * `input` - Input array of complex values
    /// * `output` - Output array (must be same length as input)
    /// * `sign` - Transform direction (Forward or Backward)
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        let n = input.len();
        debug_assert_eq!(n, output.len(), "Input and output must have same length");

        if n == 0 {
            return;
        }

        let sign_val = T::from_isize(sign.value() as isize);
        let two_pi_over_n = T::TWO_PI / T::from_usize(n);

        for k in 0..n {
            let mut sum = Complex::zero();
            let k_t = T::from_usize(k);

            for (j, &x_j) in input.iter().enumerate() {
                // W_N^(jk) = e^(sign * 2πi * j * k / N)
                let angle = sign_val * two_pi_over_n * T::from_usize(j) * k_t;
                let twiddle = Complex::cis(angle);
                sum = sum + x_j * twiddle;
            }

            output[k] = sum;
        }
    }

    /// Execute in-place direct DFT computation.
    ///
    /// Uses O(n) additional memory for temporary storage.
    pub fn execute_inplace(&self, data: &mut [Complex<T>], sign: Sign) {
        let n = data.len();
        if n == 0 {
            return;
        }

        // Need temporary storage for in-place
        let input: Vec<Complex<T>> = data.to_vec();
        self.execute(&input, data, sign);
    }

    /// Execute DFT on raw pointers.
    ///
    /// # Safety
    /// - `input` must point to `n` valid `Complex<T>` values
    /// - `output` must point to `n` valid `Complex<T>` values
    /// - Input and output may overlap only if they are identical (in-place)
    pub unsafe fn execute_ptr(
        &self,
        input: *const Complex<T>,
        output: *mut Complex<T>,
        n: usize,
        sign: Sign,
    ) {
        if n == 0 {
            return;
        }

        let sign_val = T::from_isize(sign.value() as isize);
        let two_pi_over_n = T::TWO_PI / T::from_usize(n);

        // Check for in-place
        if input as *const () == output as *const () {
            // In-place: need temporary storage
            let input_copy: Vec<Complex<T>> = (0..n).map(|i| unsafe { *input.add(i) }).collect();

            for k in 0..n {
                let mut sum = Complex::zero();
                let k_t = T::from_usize(k);

                for (j, &x_j) in input_copy.iter().enumerate() {
                    let angle = sign_val * two_pi_over_n * T::from_usize(j) * k_t;
                    let twiddle = Complex::cis(angle);
                    sum = sum + x_j * twiddle;
                }

                unsafe { *output.add(k) = sum };
            }
        } else {
            // Out-of-place: direct computation
            for k in 0..n {
                let mut sum = Complex::zero();
                let k_t = T::from_usize(k);

                for j in 0..n {
                    let x_j = unsafe { *input.add(j) };
                    let angle = sign_val * two_pi_over_n * T::from_usize(j) * k_t;
                    let twiddle = Complex::cis(angle);
                    sum = sum + x_j * twiddle;
                }

                unsafe { *output.add(k) = sum };
            }
        }
    }
}

/// Convenience function for forward DFT.
pub fn dft_direct<T: Float>(input: &[Complex<T>], output: &mut [Complex<T>]) {
    DirectSolver::new().execute(input, output, Sign::Forward);
}

/// Convenience function for inverse DFT (without normalization).
pub fn idft_direct<T: Float>(input: &[Complex<T>], output: &mut [Complex<T>]) {
    DirectSolver::new().execute(input, output, Sign::Backward);
}

/// Convenience function for inverse DFT with normalization (1/N factor).
pub fn idft_direct_normalized<T: Float>(input: &[Complex<T>], output: &mut [Complex<T>]) {
    DirectSolver::new().execute(input, output, Sign::Backward);

    let n = T::from_usize(output.len());
    for x in output.iter_mut() {
        *x = *x / n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
        approx_eq(a.re, b.re, eps) && approx_eq(a.im, b.im, eps)
    }

    #[test]
    fn test_dft_size_1() {
        let input = [Complex::new(3.0_f64, 4.0)];
        let mut output = [Complex::zero()];

        dft_direct(&input, &mut output);

        // DFT of size 1 is identity
        assert!(complex_approx_eq(output[0], input[0], 1e-10));
    }

    #[test]
    fn test_dft_size_2() {
        let input = [Complex::new(1.0_f64, 0.0), Complex::new(2.0, 0.0)];
        let mut output = [Complex::zero(); 2];

        dft_direct(&input, &mut output);

        // X[0] = x[0] + x[1] = 3
        // X[1] = x[0] - x[1] = -1
        assert!(complex_approx_eq(output[0], Complex::new(3.0, 0.0), 1e-10));
        assert!(complex_approx_eq(output[1], Complex::new(-1.0, 0.0), 1e-10));
    }

    #[test]
    fn test_dft_size_4() {
        let input = [
            Complex::new(1.0_f64, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut output = [Complex::zero(); 4];

        dft_direct(&input, &mut output);

        // X[0] = sum = 10
        assert!(complex_approx_eq(output[0], Complex::new(10.0, 0.0), 1e-10));

        // X[2] = x[0] - x[1] + x[2] - x[3] = 1 - 2 + 3 - 4 = -2
        assert!(complex_approx_eq(output[2], Complex::new(-2.0, 0.0), 1e-10));
    }

    #[test]
    fn test_inverse_recovers_input() {
        let original = [
            Complex::new(1.0_f64, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ];
        let mut transformed = [Complex::zero(); 4];
        let mut recovered = [Complex::zero(); 4];

        // Forward transform
        dft_direct(&original, &mut transformed);

        // Inverse transform with normalization
        idft_direct_normalized(&transformed, &mut recovered);

        // Should recover original
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_parseval_theorem() {
        // Parseval: sum(|x|²) = (1/N) * sum(|X|²)
        let input = [
            Complex::new(1.0_f64, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ];
        let mut output = [Complex::zero(); 4];

        dft_direct(&input, &mut output);

        let time_energy: f64 = input.iter().map(|x| x.norm_sqr()).sum();
        let freq_energy: f64 = output.iter().map(|x| x.norm_sqr()).sum();

        let n = input.len() as f64;
        assert!(approx_eq(time_energy, freq_energy / n, 1e-10));
    }

    #[test]
    fn test_linearity() {
        let x = [
            Complex::new(1.0_f64, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let y = [
            Complex::new(5.0_f64, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(7.0, 0.0),
            Complex::new(8.0, 0.0),
        ];
        let a = 2.0;

        // Compute DFT(a*x + y)
        let ax_plus_y: Vec<_> = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| xi * a + yi)
            .collect();
        let mut dft_combined = [Complex::zero(); 4];
        dft_direct(&ax_plus_y, &mut dft_combined);

        // Compute a*DFT(x) + DFT(y)
        let mut dft_x = [Complex::zero(); 4];
        let mut dft_y = [Complex::zero(); 4];
        dft_direct(&x, &mut dft_x);
        dft_direct(&y, &mut dft_y);

        for i in 0..4 {
            let expected = dft_x[i] * a + dft_y[i];
            assert!(complex_approx_eq(dft_combined[i], expected, 1e-10));
        }
    }

    #[test]
    fn test_inplace() {
        let original = [
            Complex::new(1.0_f64, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ];

        // Out-of-place reference
        let mut out_of_place = [Complex::zero(); 4];
        dft_direct(&original, &mut out_of_place);

        // In-place
        let mut in_place = original;
        DirectSolver::new().execute_inplace(&mut in_place, Sign::Forward);

        for (a, b) in out_of_place.iter().zip(in_place.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }
}
