//! Real-to-Complex FFT solver.
//!
//! Implements real-to-complex FFT using the "packing" algorithm:
//! 1. Pack N real values into N/2 complex values: z[k] = x[2k] + i*x[2k+1]
//! 2. Compute N/2-point complex FFT on z
//! 3. Unpack using symmetry relations to get N/2+1 complex outputs
//!
//! This approach requires only one complex FFT of half the size,
//! achieving approximately 2x speedup over naive complex FFT.

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};
use crate::prelude::*;
use crate::rdft::codelets::{r2hc_4, r2hc_8};

/// Real-to-Complex FFT solver.
///
/// For a real input of size N, produces N/2+1 complex outputs.
/// The output satisfies conjugate symmetry: X\[k\] = X\[N-k\]*.
pub struct R2cSolver<T: Float> {
    /// Transform size (number of real inputs)
    n: usize,
    /// Precomputed twiddle factors for unpacking
    twiddles: Vec<Complex<T>>,
}

impl<T: Float> Default for R2cSolver<T> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<T: Float> R2cSolver<T> {
    /// Create a new R2C solver for the given size.
    #[must_use]
    pub fn new(n: usize) -> Self {
        if n == 0 {
            return Self {
                n: 0,
                twiddles: Vec::new(),
            };
        }

        // Precompute twiddle factors for unpacking
        // W_N^k = e^(-2πik/N) for k = 0..N/2
        let mut twiddles = Vec::with_capacity(n / 2);
        for k in 0..n / 2 {
            let angle = -<T as Float>::TWO_PI * T::from_usize(k) / T::from_usize(n);
            twiddles.push(Complex::cis(angle));
        }

        Self { n, twiddles }
    }

    /// Solver name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        "rdft-r2c"
    }

    /// Get the transform size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.n
    }

    /// Get the output complex size (N/2 + 1).
    #[must_use]
    pub fn output_size(&self) -> usize {
        self.n / 2 + 1
    }

    /// Execute the R2C FFT.
    ///
    /// # Arguments
    /// * `input` - Real input of size N
    /// * `output` - Complex output of size N/2+1
    ///
    /// # Panics
    /// Panics if input size is not N or output size is not N/2+1.
    pub fn execute(&self, input: &[T], output: &mut [Complex<T>]) {
        let n = self.n;

        assert_eq!(input.len(), n, "Input size must be N");
        assert_eq!(output.len(), n / 2 + 1, "Output size must be N/2+1");

        if n == 0 {
            return;
        }

        if n == 1 {
            output[0] = Complex::new(input[0], T::ZERO);
            return;
        }

        if n == 2 {
            output[0] = Complex::new(input[0] + input[1], T::ZERO);
            output[1] = Complex::new(input[0] - input[1], T::ZERO);
            return;
        }

        if n == 4 {
            r2hc_4(input, output);
            return;
        }

        if n == 8 {
            r2hc_8(input, output);
            return;
        }

        // Step 1: Pack N real values into N/2 complex values
        // z[k] = x[2k] + i*x[2k+1]
        let half_n = n / 2;
        let mut z = vec![Complex::zero(); half_n];
        for k in 0..half_n {
            z[k] = Complex::new(input[2 * k], input[2 * k + 1]);
        }

        // Step 2: Compute N/2-point complex FFT
        let mut z_fft = vec![Complex::zero(); half_n];
        if let Some(plan) = Plan::dft_1d(half_n, Direction::Forward, Flags::ESTIMATE) {
            plan.execute(&z, &mut z_fft);
        }

        // Step 3: Unpack using the symmetry relations
        // Z[k] = FFT(z)[k]
        // X[k] = 0.5 * (Z[k] + Z*[N/2-k]) - 0.5i * (Z[k] - Z*[N/2-k]) * W_N^k
        //
        // Simplifying:
        // Let A = Z[k], B = Z*[N/2-k]
        // X[k] = 0.5 * (A + B) - 0.5i * (A - B) * W_N^k

        // DC component: X[0]
        // X[0] = Re(Z[0]) + Im(Z[0])
        output[0] = Complex::new(z_fft[0].re + z_fft[0].im, T::ZERO);

        // Nyquist component: X[N/2]
        // X[N/2] = Re(Z[0]) - Im(Z[0])
        output[half_n] = Complex::new(z_fft[0].re - z_fft[0].im, T::ZERO);

        // Middle components
        let half = T::from_usize(2).recip();
        for k in 1..half_n {
            let zk = z_fft[k];
            let zn_k = z_fft[half_n - k].conj();

            // A + B
            let sum = zk + zn_k;
            // A - B
            let diff = zk - zn_k;

            // Get twiddle factor W_N^k
            let w = self.twiddles[k];

            // X[k] = 0.5 * (A + B) - 0.5i * (A - B) * W
            // -i * (A - B) = (Im(A-B), -Re(A-B))
            let i_diff = Complex::new(diff.im, -diff.re);
            let term = i_diff * w;

            output[k] = (sum + term) * half;
        }
    }

    /// Execute R2C FFT in-place.
    ///
    /// The input buffer is reinterpreted: first N real values are input,
    /// and first N/2+1 complex values will be output.
    ///
    /// # Safety
    /// The buffer must have enough space for max(N, 2*(N/2+1)) elements
    /// when viewed as T.
    pub fn execute_inplace(&self, data: &mut [T]) {
        let n = self.n;
        assert!(data.len() >= n, "Buffer too small for input");

        if n <= 2 {
            if n == 0 {
                return;
            }
            if n == 1 {
                // Output is just the single value (real only)
                return;
            }
            // n == 2
            let x0 = data[0];
            let x1 = data[1];
            data[0] = x0 + x1; // Re(X[0])
            data[1] = T::ZERO; // Im(X[0])
                               // X[1] would be at data[2], data[3] but we only have 2 elements
            return;
        }

        // For in-place, we need temporary storage
        let input: Vec<T> = data[..n].to_vec();
        let mut output = vec![Complex::zero(); n / 2 + 1];
        self.execute(&input, &mut output);

        // Copy output back (reinterpreting complex as pairs of reals)
        for (i, c) in output.iter().enumerate() {
            data[2 * i] = c.re;
            data[2 * i + 1] = c.im;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::fft;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
        approx_eq(a.re, b.re, eps) && approx_eq(a.im, b.im, eps)
    }

    #[test]
    fn test_r2c_size_2() {
        let input = [1.0_f64, 2.0];
        let mut output = vec![Complex::zero(); 2];

        R2cSolver::new(2).execute(&input, &mut output);

        // X[0] = x[0] + x[1] = 3
        // X[1] = x[0] - x[1] = -1
        assert!(complex_approx_eq(output[0], Complex::new(3.0, 0.0), 1e-10));
        assert!(complex_approx_eq(output[1], Complex::new(-1.0, 0.0), 1e-10));
    }

    #[test]
    fn test_r2c_size_4() {
        let input = [1.0_f64, 2.0, 3.0, 4.0];
        let mut output = vec![Complex::zero(); 3];

        R2cSolver::new(4).execute(&input, &mut output);

        // Compare with full complex FFT
        let complex_input: Vec<Complex<f64>> =
            input.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let full_fft = fft(&complex_input);

        // X[0] should match
        assert!(complex_approx_eq(output[0], full_fft[0], 1e-10));
        // X[1] should match
        assert!(complex_approx_eq(output[1], full_fft[1], 1e-10));
        // X[2] should match
        assert!(complex_approx_eq(output[2], full_fft[2], 1e-10));
    }

    #[test]
    fn test_r2c_size_8() {
        let input: Vec<f64> = (0..8).map(f64::from).collect();
        let mut output = vec![Complex::zero(); 5];

        R2cSolver::new(8).execute(&input, &mut output);

        // Compare with full complex FFT
        let complex_input: Vec<Complex<f64>> =
            input.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let full_fft = fft(&complex_input);

        for k in 0..5 {
            assert!(
                complex_approx_eq(output[k], full_fft[k], 1e-9),
                "Mismatch at k={}: got {:?}, expected {:?}",
                k,
                output[k],
                full_fft[k]
            );
        }
    }

    #[test]
    fn test_r2c_size_16() {
        let input: Vec<f64> = (0..16).map(|i| f64::from(i).sin()).collect();
        let mut output = vec![Complex::zero(); 9];

        R2cSolver::new(16).execute(&input, &mut output);

        // Compare with full complex FFT
        let complex_input: Vec<Complex<f64>> =
            input.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let full_fft = fft(&complex_input);

        for k in 0..9 {
            assert!(
                complex_approx_eq(output[k], full_fft[k], 1e-9),
                "Mismatch at k={}: got {:?}, expected {:?}",
                k,
                output[k],
                full_fft[k]
            );
        }
    }

    #[test]
    fn test_r2c_dc_and_nyquist_are_real() {
        let input: Vec<f64> = (0..8).map(|i| f64::from(i).sin() + 0.5).collect();
        let mut output = vec![Complex::zero(); 5];

        R2cSolver::new(8).execute(&input, &mut output);

        // DC (X[0]) and Nyquist (X[N/2]) should be purely real for real input
        assert!(
            approx_eq(output[0].im, 0.0, 1e-10),
            "DC component should be real"
        );
        assert!(
            approx_eq(output[4].im, 0.0, 1e-10),
            "Nyquist component should be real"
        );
    }
}
