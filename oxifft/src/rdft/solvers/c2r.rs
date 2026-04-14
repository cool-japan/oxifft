//! Complex-to-Real FFT solver.
//!
//! Implements complex-to-real FFT (inverse of R2C) using the reverse of the packing algorithm:
//! 1. Unpack N/2+1 complex values into N/2 complex values using symmetry
//! 2. Compute N/2-point inverse complex FFT
//! 3. Interleave real and imaginary parts to get N real outputs
//!
//! This is the inverse operation of R2C and produces real output from
//! conjugate-symmetric complex input.

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};
use crate::prelude::*;
use crate::rdft::codelets::{hc2r_4, hc2r_8};

/// Complex-to-Real FFT solver.
///
/// For N/2+1 complex inputs satisfying conjugate symmetry, produces N real outputs.
/// This is the inverse of R2C (without normalization).
pub struct C2rSolver<T: Float> {
    /// Transform size (number of real outputs)
    n: usize,
    /// Precomputed twiddle factors for packing
    twiddles: Vec<Complex<T>>,
}

impl<T: Float> Default for C2rSolver<T> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<T: Float> C2rSolver<T> {
    /// Create a new C2R solver for the given output size.
    #[must_use]
    pub fn new(n: usize) -> Self {
        if n == 0 {
            return Self {
                n: 0,
                twiddles: Vec::new(),
            };
        }

        // Precompute twiddle factors for packing
        // W_N^k = e^(+2πik/N) (positive for inverse)
        let mut twiddles = Vec::with_capacity(n / 2);
        for k in 0..n / 2 {
            let angle = <T as Float>::TWO_PI * T::from_usize(k) / T::from_usize(n);
            twiddles.push(Complex::cis(angle));
        }

        Self { n, twiddles }
    }

    /// Solver name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        "rdft-c2r"
    }

    /// Get the transform size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.n
    }

    /// Get the input complex size (N/2 + 1).
    #[must_use]
    pub fn input_size(&self) -> usize {
        self.n / 2 + 1
    }

    /// Execute the C2R FFT.
    ///
    /// # Arguments
    /// * `input` - Complex input of size N/2+1 (conjugate symmetric)
    /// * `output` - Real output of size N
    ///
    /// # Note
    /// The output is NOT normalized. Divide by N to get the true inverse.
    ///
    /// # Panics
    /// Panics if input size is not N/2+1 or output size is not N.
    pub fn execute(&self, input: &[Complex<T>], output: &mut [T]) {
        let n = self.n;

        assert_eq!(input.len(), n / 2 + 1, "Input size must be N/2+1");
        assert_eq!(output.len(), n, "Output size must be N");

        if n == 0 {
            return;
        }

        if n == 1 {
            output[0] = input[0].re;
            return;
        }

        if n == 2 {
            // X[0] and X[1] are both real for real input
            // x[0] = (X[0] + X[1]) / 2, x[1] = (X[0] - X[1]) / 2
            // But we don't normalize, so:
            output[0] = input[0].re + input[1].re;
            output[1] = input[0].re - input[1].re;
            return;
        }

        if n == 4 {
            hc2r_4(input, output);
            return;
        }

        if n == 8 {
            hc2r_8(input, output);
            return;
        }

        let half_n = n / 2;

        // Step 1: Pack the complex input into Z[k] for N/2-point IFFT
        // This is the reverse of the R2C unpacking
        //
        // From R2C: X[k] = 0.5 * (Z[k] + Z*[N/2-k]) - 0.5i * (Z[k] - Z*[N/2-k]) * W_N^k
        // Inverting: Z[k] = X[k] + X*[N/2-k] + i * (X[k] - X*[N/2-k]) * W_N^(-k)
        //
        // Z[0] has special handling from DC and Nyquist

        let mut z = vec![Complex::zero(); half_n];

        // Z[0] from X[0] and X[N/2]
        // X[0] = Re(Z[0]) + Im(Z[0]), X[N/2] = Re(Z[0]) - Im(Z[0])
        // So: Re(Z[0]) = (X[0] + X[N/2]) / 2, Im(Z[0]) = (X[0] - X[N/2]) / 2
        // Without normalization:
        z[0] = Complex::new(
            input[0].re + input[half_n].re,
            input[0].re - input[half_n].re,
        );

        // Middle components
        for k in 1..half_n {
            let xk = input[k];
            // X[N/2-k] is conjugate of X[N-(N/2-k)] = X[N/2+k]
            // But we only have X[0..N/2], so we use X*[N/2-k]
            let xn_k = input[half_n - k].conj();

            // Sum and difference
            let sum = xk + xn_k;
            let diff = xk - xn_k;

            // Get inverse twiddle factor W_N^(-k) = W_N^k conjugate
            let w = self.twiddles[k];

            // Z[k] = sum + i * diff * W
            // i * diff = (-Im(diff), Re(diff))
            let i_diff = Complex::new(-diff.im, diff.re);
            let term = i_diff * w;

            z[k] = sum + term;
        }

        // Step 2: Compute N/2-point inverse complex FFT
        let mut z_ifft = vec![Complex::zero(); half_n];
        if let Some(plan) = Plan::dft_1d(half_n, Direction::Backward, Flags::ESTIMATE) {
            plan.execute(&z, &mut z_ifft);
        }

        // Step 3: Unpack z_ifft into real output
        // z[k] = x[2k] + i*x[2k+1], so:
        // x[2k] = Re(z[k]), x[2k+1] = Im(z[k])
        for k in 0..half_n {
            output[2 * k] = z_ifft[k].re;
            output[2 * k + 1] = z_ifft[k].im;
        }
    }

    /// Execute C2R FFT with normalization.
    ///
    /// The output is divided by N to give the true inverse of R2C.
    pub fn execute_normalized(&self, input: &[Complex<T>], output: &mut [T]) {
        self.execute(input, output);

        let n = T::from_usize(self.n);
        for x in output.iter_mut() {
            *x = *x / n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdft::solvers::R2cSolver;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_c2r_size_2() {
        // Test that C2R(R2C(x)) = x (with normalization)
        let original = [1.0_f64, 2.0];
        let mut freq = vec![Complex::zero(); 2];
        let mut recovered = [0.0_f64; 2];

        R2cSolver::new(2).execute(&original, &mut freq);
        C2rSolver::new(2).execute_normalized(&freq, &mut recovered);

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(approx_eq(*a, *b, 1e-10), "got {b}, expected {a}");
        }
    }

    #[test]
    fn test_c2r_size_4() {
        let original = [1.0_f64, 2.0, 3.0, 4.0];
        let mut freq = vec![Complex::zero(); 3];
        let mut recovered = [0.0_f64; 4];

        R2cSolver::new(4).execute(&original, &mut freq);
        C2rSolver::new(4).execute_normalized(&freq, &mut recovered);

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(approx_eq(*a, *b, 1e-10), "got {b}, expected {a}");
        }
    }

    #[test]
    fn test_c2r_size_8() {
        let original: Vec<f64> = (0..8).map(f64::from).collect();
        let mut freq = vec![Complex::zero(); 5];
        let mut recovered = vec![0.0_f64; 8];

        R2cSolver::new(8).execute(&original, &mut freq);
        C2rSolver::new(8).execute_normalized(&freq, &mut recovered);

        for (i, (a, b)) in original.iter().zip(recovered.iter()).enumerate() {
            assert!(
                approx_eq(*a, *b, 1e-9),
                "Mismatch at {i}: got {b}, expected {a}"
            );
        }
    }

    #[test]
    fn test_c2r_size_16() {
        let original: Vec<f64> = (0..16).map(|i| f64::from(i).sin()).collect();
        let mut freq = vec![Complex::zero(); 9];
        let mut recovered = vec![0.0_f64; 16];

        R2cSolver::new(16).execute(&original, &mut freq);
        C2rSolver::new(16).execute_normalized(&freq, &mut recovered);

        for (i, (a, b)) in original.iter().zip(recovered.iter()).enumerate() {
            assert!(
                approx_eq(*a, *b, 1e-9),
                "Mismatch at {i}: got {b}, expected {a}"
            );
        }
    }

    #[test]
    fn test_c2r_roundtrip_random() {
        // Test with more varied input
        let original: Vec<f64> = (0..32)
            .map(|i| (f64::from(i) * 0.7).sin() + (f64::from(i) * 1.3).cos())
            .collect();
        let mut freq = vec![Complex::zero(); 17];
        let mut recovered = vec![0.0_f64; 32];

        R2cSolver::new(32).execute(&original, &mut freq);
        C2rSolver::new(32).execute_normalized(&freq, &mut recovered);

        for (i, (a, b)) in original.iter().zip(recovered.iter()).enumerate() {
            assert!(
                approx_eq(*a, *b, 1e-9),
                "Mismatch at {i}: got {b}, expected {a}"
            );
        }
    }
}
