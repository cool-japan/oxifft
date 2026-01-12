//! Bluestein's Chirp-Z algorithm for arbitrary sizes.
//!
//! This algorithm converts a DFT of any size N to a convolution of size M,
//! where M is the next power of 2 >= 2N-1. The convolution is then computed
//! using FFT.
//!
//! Time complexity: O(N log N) for any N (not just powers of 2)
//! Space complexity: O(M) where M is next power of 2 >= 2N-1

use crate::dft::problem::Sign;
use crate::kernel::{Complex, Float};
use crate::prelude::*;

use super::ct::CooleyTukeySolver;

/// Bluestein (Chirp-Z) solver for arbitrary sizes.
///
/// Bluestein's algorithm uses the identity:
/// nk = (n² + k² - (k-n)²) / 2
///
/// This allows rewriting the DFT as:
/// X\[k\] = W_N^(k²/2) * Σ_{n=0}^{N-1} (x\[n\] * W_N^(n²/2)) * W_N^(-(k-n)²/2)
///
/// The summation is a convolution, which can be computed via FFT.
///
/// This solver pre-allocates work buffers to avoid per-execution allocations.
/// Uses `Mutex` for thread-safe interior mutability with `try_lock()` fallback.
pub struct BluesteinSolver<T: Float> {
    /// Original size
    n: usize,
    /// Padded size (power of 2)
    m: usize,
    /// Chirp sequence: W_N^(n²/2) for n = 0..N-1
    chirp: Vec<Complex<T>>,
    /// Conjugate chirp for convolution: W_N^(-n²/2) for n = -(N-1)..N-1
    chirp_conj_fft: Vec<Complex<T>>,
    /// Pre-allocated work buffer for y (input * chirp) - thread-safe with fallback
    #[cfg(feature = "std")]
    work_y: Mutex<Vec<Complex<T>>>,
    /// Pre-allocated work buffer for FFT(y) - thread-safe with fallback
    #[cfg(feature = "std")]
    work_y_fft: Mutex<Vec<Complex<T>>>,
    /// Pre-allocated work buffer for convolution result - thread-safe with fallback
    #[cfg(feature = "std")]
    work_conv: Mutex<Vec<Complex<T>>>,
}

impl<T: Float> BluesteinSolver<T> {
    /// Create a new Bluestein solver for the given size.
    #[must_use]
    pub fn new(n: usize) -> Self {
        if n == 0 {
            return Self {
                n: 0,
                m: 0,
                chirp: Vec::new(),
                chirp_conj_fft: Vec::new(),
                #[cfg(feature = "std")]
                work_y: Mutex::new(Vec::new()),
                #[cfg(feature = "std")]
                work_y_fft: Mutex::new(Vec::new()),
                #[cfg(feature = "std")]
                work_conv: Mutex::new(Vec::new()),
            };
        }

        // Padded size must be >= 2N-1 and a power of 2
        let m = (2 * n - 1).next_power_of_two();

        // Compute chirp sequence: W_N^(n²/2) = e^(-πi n²/N)
        let mut chirp = Vec::with_capacity(n);
        for i in 0..n {
            let i_sq = (i * i) % (2 * n); // Reduce modulo 2N for numerical stability
            let angle = -<T as Float>::PI * T::from_usize(i_sq) / T::from_usize(n);
            chirp.push(Complex::cis(angle));
        }

        // Compute conjugate chirp for convolution
        // chirp_conj[k] = W_N^(-k²/2) for k = 0..M-1
        // Need to handle wrap-around: indices M-N+1..M-1 correspond to k = -(N-1)...-1
        let mut chirp_conj = vec![Complex::zero(); m];

        // k = 0..N-1: chirp_conj[k] = conj(chirp[k])
        for i in 0..n {
            chirp_conj[i] = chirp[i].conj();
        }

        // k = -(N-1)...-1: stored at indices M-N+1..M-1
        // chirp_conj[M-k] = conj(chirp[k]) for k = 1..N-1
        for i in 1..n {
            chirp_conj[m - i] = chirp[i].conj();
        }

        // Pre-compute FFT of chirp_conj
        let mut chirp_conj_fft = vec![Complex::zero(); m];
        CooleyTukeySolver::<T>::default().execute(&chirp_conj, &mut chirp_conj_fft, Sign::Forward);

        Self {
            n,
            m,
            chirp,
            chirp_conj_fft,
            #[cfg(feature = "std")]
            work_y: Mutex::new(vec![Complex::zero(); m]),
            #[cfg(feature = "std")]
            work_y_fft: Mutex::new(vec![Complex::zero(); m]),
            #[cfg(feature = "std")]
            work_conv: Mutex::new(vec![Complex::zero(); m]),
        }
    }

    /// Solver name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        "dft-bluestein"
    }

    /// Get the transform size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.n
    }

    /// Check if this solver is applicable.
    ///
    /// Bluestein can handle any size > 0, but for powers of 2,
    /// Cooley-Tukey is more efficient.
    #[must_use]
    pub fn applicable(n: usize) -> bool {
        n > 0
    }

    /// Execute the Bluestein FFT with provided work buffers.
    fn execute_with_buffers(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        sign: Sign,
        y: &mut [Complex<T>],
        y_fft: &mut [Complex<T>],
        conv: &mut [Complex<T>],
    ) {
        let n = self.n;
        let m = self.m;
        let ct = CooleyTukeySolver::<T>::default();

        // Step 1: Multiply input by chirp
        // y[k] = x[k] * chirp[k], zero-padded to size M
        for i in 0..m {
            y[i] = Complex::zero();
        }
        match sign {
            Sign::Forward => {
                for i in 0..n {
                    y[i] = input[i] * self.chirp[i];
                }
            }
            Sign::Backward => {
                // For inverse, use conjugate chirp
                for i in 0..n {
                    y[i] = input[i] * self.chirp[i].conj();
                }
            }
        }

        // Step 2: FFT of y
        ct.execute(y, y_fft, Sign::Forward);

        // Step 3: Pointwise multiply with chirp_conj_fft
        // For inverse transform, need conjugate of chirp_conj_fft
        match sign {
            Sign::Forward => {
                for i in 0..m {
                    y_fft[i] = y_fft[i] * self.chirp_conj_fft[i];
                }
            }
            Sign::Backward => {
                for i in 0..m {
                    y_fft[i] = y_fft[i] * self.chirp_conj_fft[i].conj();
                }
            }
        }

        // Step 4: IFFT
        ct.execute(y_fft, conv, Sign::Backward);

        // Normalize the IFFT
        let m_inv = T::ONE / T::from_usize(m);

        // Step 5: Extract result and multiply by chirp
        match sign {
            Sign::Forward => {
                for i in 0..n {
                    output[i] = conv[i] * m_inv * self.chirp[i];
                }
            }
            Sign::Backward => {
                for i in 0..n {
                    output[i] = conv[i] * m_inv * self.chirp[i].conj();
                }
            }
        }
    }

    /// Execute the Bluestein FFT.
    ///
    /// Uses pre-allocated work buffers when available (single-threaded case).
    /// Falls back to fresh allocation when buffers are locked (parallel execution).
    #[cfg(feature = "std")]
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        let n = self.n;
        debug_assert_eq!(input.len(), n);
        debug_assert_eq!(output.len(), n);

        if n == 0 {
            return;
        }

        if n == 1 {
            output[0] = input[0];
            return;
        }

        let m = self.m;

        // Try to acquire all three locks. If any fails, allocate fresh buffers.
        // This provides optimal performance for single-threaded use while
        // maintaining correctness for parallel execution.
        let y_guard = self.work_y.try_lock();
        let y_fft_guard = self.work_y_fft.try_lock();
        let conv_guard = self.work_conv.try_lock();

        if let (Ok(mut y), Ok(mut y_fft), Ok(mut conv)) = (y_guard, y_fft_guard, conv_guard) {
            // Use pre-allocated buffers
            self.execute_with_buffers(input, output, sign, &mut y, &mut y_fft, &mut conv);
        } else {
            // Fallback: allocate fresh buffers (parallel execution case)
            let mut y = vec![Complex::zero(); m];
            let mut y_fft = vec![Complex::zero(); m];
            let mut conv = vec![Complex::zero(); m];
            self.execute_with_buffers(input, output, sign, &mut y, &mut y_fft, &mut conv);
        }
    }

    /// Execute the Bluestein FFT (no_std version - always allocates).
    #[cfg(not(feature = "std"))]
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        let n = self.n;
        debug_assert_eq!(input.len(), n);
        debug_assert_eq!(output.len(), n);

        if n == 0 {
            return;
        }

        if n == 1 {
            output[0] = input[0];
            return;
        }

        let m = self.m;

        // no_std: always allocate fresh buffers
        let mut y = vec![Complex::zero(); m];
        let mut y_fft = vec![Complex::zero(); m];
        let mut conv = vec![Complex::zero(); m];
        self.execute_with_buffers(input, output, sign, &mut y, &mut y_fft, &mut conv);
    }

    /// Execute in-place Bluestein FFT.
    ///
    /// Uses pre-allocated work buffers when available.
    /// Still requires a temporary copy of the input data.
    pub fn execute_inplace(&self, data: &mut [Complex<T>], sign: Sign) {
        let n = self.n;
        debug_assert_eq!(data.len(), n);

        if n <= 1 {
            return;
        }

        // Need temporary storage for in-place (copy input first)
        let input: Vec<Complex<T>> = data.to_vec();
        self.execute(&input, data, sign);
    }
}

impl<T: Float> Default for BluesteinSolver<T> {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Convenience function for forward FFT using Bluestein.
#[allow(dead_code)]
pub fn fft_bluestein<T: Float>(input: &[Complex<T>], output: &mut [Complex<T>]) {
    BluesteinSolver::new(input.len()).execute(input, output, Sign::Forward);
}

/// Convenience function for inverse FFT using Bluestein (without normalization).
#[allow(dead_code)]
pub fn ifft_bluestein<T: Float>(input: &[Complex<T>], output: &mut [Complex<T>]) {
    BluesteinSolver::new(input.len()).execute(input, output, Sign::Backward);
}

/// Convenience function for in-place forward FFT using Bluestein.
#[allow(dead_code)]
pub fn fft_bluestein_inplace<T: Float>(data: &mut [Complex<T>]) {
    BluesteinSolver::new(data.len()).execute_inplace(data, Sign::Forward);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dft::solvers::direct::DirectSolver;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
        approx_eq(a.re, b.re, eps) && approx_eq(a.im, b.im, eps)
    }

    #[test]
    fn test_bluestein_size_1() {
        let input = [Complex::new(3.0_f64, 4.0)];
        let mut output = [Complex::zero()];

        BluesteinSolver::new(1).execute(&input, &mut output, Sign::Forward);
        assert!(complex_approx_eq(output[0], input[0], 1e-10));
    }

    #[test]
    fn test_bluestein_size_5() {
        // Non-power-of-2 size
        let input: Vec<Complex<f64>> = (0..5).map(|i| Complex::new(f64::from(i), 0.0)).collect();
        let mut output_bluestein = vec![Complex::zero(); 5];
        let mut output_direct = vec![Complex::zero(); 5];

        BluesteinSolver::new(5).execute(&input, &mut output_bluestein, Sign::Forward);
        DirectSolver::new().execute(&input, &mut output_direct, Sign::Forward);

        for (a, b) in output_bluestein.iter().zip(output_direct.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_bluestein_size_7() {
        // Prime size
        let input: Vec<Complex<f64>> = (0..7)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut output_bluestein = vec![Complex::zero(); 7];
        let mut output_direct = vec![Complex::zero(); 7];

        BluesteinSolver::new(7).execute(&input, &mut output_bluestein, Sign::Forward);
        DirectSolver::new().execute(&input, &mut output_direct, Sign::Forward);

        for (a, b) in output_bluestein.iter().zip(output_direct.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_bluestein_size_12() {
        // Composite non-power-of-2
        let input: Vec<Complex<f64>> = (0..12)
            .map(|i| Complex::new(f64::from(i), f64::from(i) * 0.5))
            .collect();
        let mut output_bluestein = vec![Complex::zero(); 12];
        let mut output_direct = vec![Complex::zero(); 12];

        BluesteinSolver::new(12).execute(&input, &mut output_bluestein, Sign::Forward);
        DirectSolver::new().execute(&input, &mut output_direct, Sign::Forward);

        for (a, b) in output_bluestein.iter().zip(output_direct.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_bluestein_inverse_recovers_input() {
        let original: Vec<Complex<f64>> = (0..11)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut transformed = vec![Complex::zero(); 11];
        let mut recovered = vec![Complex::zero(); 11];

        let solver = BluesteinSolver::new(11);
        solver.execute(&original, &mut transformed, Sign::Forward);
        solver.execute(&transformed, &mut recovered, Sign::Backward);

        // Normalize
        let n = original.len() as f64;
        for x in &mut recovered {
            *x = *x / n;
        }

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_bluestein_inplace() {
        let original: Vec<Complex<f64>> = (0..9).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        // Out-of-place reference
        let mut out_of_place = vec![Complex::zero(); 9];
        let solver = BluesteinSolver::new(9);
        solver.execute(&original, &mut out_of_place, Sign::Forward);

        // In-place
        let mut in_place = original;
        solver.execute_inplace(&mut in_place, Sign::Forward);

        for (a, b) in out_of_place.iter().zip(in_place.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }
}
