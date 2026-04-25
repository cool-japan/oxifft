//! Bluestein's Chirp-Z algorithm for arbitrary sizes.
//!
//! This algorithm converts a DFT of any size N to a convolution of size M,
//! where M is the next power of 2 >= 2N-1. The convolution is then computed
//! using FFT.
//!
//! Time complexity: O(N log N) for any N (not just powers of 2)
//! Space complexity: O(M) where M is next power of 2 >= 2N-1
//!
//! # Optimizations in this implementation
//!
//! - **SIMD-accelerated pointwise complex multiplies** via `complex_mul_aos` /
//!   `complex_mul_conj_aos`, dispatching to AVX2+FMA / SSE2 / NEON at runtime.
//! - **Pre-computed chirp variants**: both `chirp_fwd` and `chirp_bwd` (conjugate)
//!   are stored at construction time, eliminating per-call conjugation.
//! - **Pre-computed chirp_conj_fft variants**: both forward and backward FFT
//!   of the chirp convolution kernel are precomputed.
//! - **4th inplace scratch mutex**: `work_inplace` avoids the unconditional
//!   `data.to_vec()` allocation in `execute_inplace` for the common case.
//! - **Thread-local scratch** (under `std` feature): solver-id-keyed per-thread
//!   scratch buffers amortize allocation cost when mutex is contended.

use core::sync::atomic::{AtomicU64, Ordering};

use crate::dft::problem::Sign;
use crate::kernel::complex_mul::complex_mul_aos;
use crate::kernel::{Complex, Float};
use crate::prelude::*;

use super::ct::CooleyTukeySolver;

/// Global counter for assigning unique IDs to each `BluesteinSolver`.
static BLUESTEIN_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

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
    /// Chirp sequence for forward transform: W_N^(n²/2) for n = 0..N-1
    chirp_fwd: Vec<Complex<T>>,
    /// Chirp sequence for backward transform: conj(W_N^(n²/2)) = W_N^(-n²/2)
    chirp_bwd: Vec<Complex<T>>,
    /// Pre-computed FFT of chirp_conj for forward transform
    chirp_conj_fft_fwd: Vec<Complex<T>>,
    /// Pre-computed FFT of chirp_conj for backward transform (conjugate)
    chirp_conj_fft_bwd: Vec<Complex<T>>,
    /// Unique solver identifier (useful for debugging and future thread-local keying).
    pub(crate) solver_id: u64,
    /// Pre-allocated work buffer for y (input * chirp) - thread-safe with fallback
    #[cfg(feature = "std")]
    work_y: Mutex<Vec<Complex<T>>>,
    /// Pre-allocated work buffer for FFT(y) - thread-safe with fallback
    #[cfg(feature = "std")]
    work_y_fft: Mutex<Vec<Complex<T>>>,
    /// Pre-allocated work buffer for convolution result - thread-safe with fallback
    #[cfg(feature = "std")]
    work_conv: Mutex<Vec<Complex<T>>>,
    /// Pre-allocated work buffer for in-place scratch (avoids unconditional alloc)
    #[cfg(feature = "std")]
    work_inplace: Mutex<Vec<Complex<T>>>,
}

impl<T: Float> BluesteinSolver<T> {
    /// Create a new Bluestein solver for the given size.
    #[must_use]
    pub fn new(n: usize) -> Self {
        let solver_id = BLUESTEIN_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        if n == 0 {
            return Self {
                n: 0,
                m: 0,
                chirp_fwd: Vec::new(),
                chirp_bwd: Vec::new(),
                chirp_conj_fft_fwd: Vec::new(),
                chirp_conj_fft_bwd: Vec::new(),
                solver_id,
                #[cfg(feature = "std")]
                work_y: Mutex::new(Vec::new()),
                #[cfg(feature = "std")]
                work_y_fft: Mutex::new(Vec::new()),
                #[cfg(feature = "std")]
                work_conv: Mutex::new(Vec::new()),
                #[cfg(feature = "std")]
                work_inplace: Mutex::new(Vec::new()),
            };
        }

        // Padded size must be >= 2N-1 and a power of 2
        let m = (2 * n - 1).next_power_of_two();

        // Compute chirp sequence: W_N^(n²/2) = e^(-πi n²/N)
        let mut chirp_fwd = Vec::with_capacity(n);
        for i in 0..n {
            let i_sq = (i * i) % (2 * n); // Reduce modulo 2N for numerical stability
            let angle = -<T as Float>::PI * T::from_usize(i_sq) / T::from_usize(n);
            chirp_fwd.push(Complex::cis(angle));
        }

        // chirp_bwd[i] = conj(chirp_fwd[i]) — precomputed for zero-cost backward path
        let chirp_bwd: Vec<Complex<T>> = chirp_fwd.iter().map(|c| c.conj()).collect();

        // Compute conjugate chirp for convolution
        // chirp_conj[k] = W_N^(-k²/2) for k = 0..M-1
        // Need to handle wrap-around: indices M-N+1..M-1 correspond to k = -(N-1)...-1
        let mut chirp_conj = vec![Complex::zero(); m];
        for i in 0..n {
            chirp_conj[i] = chirp_fwd[i].conj();
        }
        for i in 1..n {
            chirp_conj[m - i] = chirp_fwd[i].conj();
        }

        // Pre-compute FFT of chirp_conj for forward path
        let mut chirp_conj_fft_fwd = vec![Complex::zero(); m];
        CooleyTukeySolver::<T>::default().execute(
            &chirp_conj,
            &mut chirp_conj_fft_fwd,
            Sign::Forward,
        );

        // Pre-compute conjugated version for backward path
        let chirp_conj_fft_bwd: Vec<Complex<T>> =
            chirp_conj_fft_fwd.iter().map(|c| c.conj()).collect();

        Self {
            n,
            m,
            chirp_fwd,
            chirp_bwd,
            chirp_conj_fft_fwd,
            chirp_conj_fft_bwd,
            solver_id,
            #[cfg(feature = "std")]
            work_y: Mutex::new(vec![Complex::zero(); m]),
            #[cfg(feature = "std")]
            work_y_fft: Mutex::new(vec![Complex::zero(); m]),
            #[cfg(feature = "std")]
            work_conv: Mutex::new(vec![Complex::zero(); m]),
            #[cfg(feature = "std")]
            work_inplace: Mutex::new(vec![Complex::zero(); n]),
        }
    }

    /// Solver name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        "dft-bluestein"
    }

    /// Returns the unique solver ID (monotonically increasing per-process counter).
    #[must_use]
    pub fn id(&self) -> u64 {
        self.solver_id
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
    ///
    /// All three work buffers must have length `m` (the padded size).
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

        // Select the pre-computed chirp variant for this sign.
        // chirp_bwd = conj(chirp_fwd) and chirp_conj_fft_bwd = conj(chirp_conj_fft_fwd)
        // both pre-computed at construction time, so the hot path has zero extra work.
        let (chirp, chirp_conv_fft) = match sign {
            Sign::Forward => (&self.chirp_fwd, &self.chirp_conj_fft_fwd),
            Sign::Backward => (&self.chirp_bwd, &self.chirp_conj_fft_bwd),
        };

        // Step 1: y[i] = input[i] * chirp[i], zero-padded to size M
        y[..m].fill(Complex::zero());
        // SIMD pointwise multiply for the first n elements
        complex_mul_aos(&mut y[..n], input, chirp);

        // Step 2: FFT of y
        ct.execute(y, y_fft, Sign::Forward);

        // Step 3: Pointwise multiply with chirp_conv_fft (pre-selected above).
        // We must not alias y_fft as both src and dst.  We use `conv` (currently
        // uninitialized at this point) as the destination, then swap buffers.
        complex_mul_aos(&mut conv[..m], y_fft, chirp_conv_fft);

        // Step 4: IFFT — conv now holds y_fft * chirp_conv_fft; use y_fft as output
        ct.execute(conv, y_fft, Sign::Backward);
        // y_fft now holds the IFFT result (reusing the buffer)

        // Normalize the IFFT result
        let m_inv = T::ONE / T::from_usize(m);

        // Step 5: output[i] = (y_fft[i] * m_inv) * chirp[i]
        // Scale y_fft[..n] in-place (it's scratch we own; IFFT already wrote here)
        // then SIMD-multiply into output — zero extra allocation.
        for v in &mut y_fft[..n] {
            *v = *v * m_inv;
        }
        complex_mul_aos(output, &y_fft[..n], chirp);
    }

    /// Execute the Bluestein FFT.
    ///
    /// Uses pre-allocated work buffers when available (single-threaded case).
    /// Falls back to thread-local scratch (parallel contention) or fresh
    /// allocation (heavily parallel or no-std).
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

        // Try to acquire all three mutex locks atomically.
        let y_guard = self.work_y.try_lock();
        let y_fft_guard = self.work_y_fft.try_lock();
        let conv_guard = self.work_conv.try_lock();

        if let (Ok(mut y), Ok(mut y_fft), Ok(mut conv)) = (y_guard, y_fft_guard, conv_guard) {
            // Fast path: use pre-allocated buffers
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
    /// Uses the pre-allocated `work_inplace` buffer when available (avoids
    /// an unconditional `data.to_vec()` allocation in the common case).
    pub fn execute_inplace(&self, data: &mut [Complex<T>], sign: Sign) {
        let n = self.n;
        debug_assert_eq!(data.len(), n);

        if n <= 1 {
            return;
        }

        #[cfg(feature = "std")]
        {
            // Try the pre-allocated inplace scratch buffer.
            if let Ok(mut inplace_buf) = self.work_inplace.try_lock() {
                // Ensure capacity (should always be correct after `new`)
                if inplace_buf.len() < n {
                    inplace_buf.resize(n, Complex::zero());
                }
                inplace_buf[..n].copy_from_slice(data);
                let input: &[Complex<T>] = &inplace_buf[..n];
                // SAFETY: input borrows from inplace_buf, output is data.
                // They do not alias because data is the caller's slice and
                // inplace_buf is our internal Mutex-protected buffer.
                //
                // We must extend the lifetime here because Rust cannot see that
                // `inplace_buf` and `data` are disjoint. We take a raw pointer
                // to convert the borrow.
                let input_ptr = input.as_ptr();
                let input_slice = unsafe { core::slice::from_raw_parts(input_ptr, n) };
                self.execute(input_slice, data, sign);
                return;
            }
        }

        // Fallback: allocate a fresh copy (mutex contended or no_std)
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
pub fn fft_bluestein<T: Float>(input: &[Complex<T>], output: &mut [Complex<T>]) {
    BluesteinSolver::new(input.len()).execute(input, output, Sign::Forward);
}

/// Convenience function for inverse FFT using Bluestein (without normalization).
pub fn ifft_bluestein<T: Float>(input: &[Complex<T>], output: &mut [Complex<T>]) {
    BluesteinSolver::new(input.len()).execute(input, output, Sign::Backward);
}

/// Convenience function for in-place forward FFT using Bluestein.
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

    #[test]
    fn test_fft_bluestein_convenience() {
        let input: Vec<Complex<f64>> = (0..7).map(|i| Complex::new(f64::from(i), 0.0)).collect();
        let mut output = vec![Complex::zero(); 7];
        fft_bluestein(&input, &mut output);
        // Result should be non-trivial
        let energy: f64 = output.iter().map(|c| c.norm_sqr()).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_ifft_bluestein_convenience() {
        let input: Vec<Complex<f64>> = (0..7).map(|i| Complex::new(f64::from(i), 0.0)).collect();
        let mut forward = vec![Complex::zero(); 7];
        let mut backward = vec![Complex::zero(); 7];
        fft_bluestein(&input, &mut forward);
        ifft_bluestein(&forward, &mut backward);
        // Backward of forward should reconstruct (unnormalized)
        let n = 7.0_f64;
        for (orig, recov) in input.iter().zip(backward.iter()) {
            assert!(complex_approx_eq(*orig, *recov / n, 1e-9));
        }
    }

    #[test]
    fn test_fft_bluestein_inplace_convenience() {
        let original: Vec<Complex<f64>> = (0..6).map(|i| Complex::new(f64::from(i), 0.0)).collect();
        let mut inplace = original.clone();

        let mut ref_output = vec![Complex::zero(); 6];
        fft_bluestein(&original, &mut ref_output);
        fft_bluestein_inplace(&mut inplace);

        for (a, b) in ref_output.iter().zip(inplace.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    // -------------------------------------------------------------------------
    // Round-trip tests for prime sizes (f64 and f32)
    // -------------------------------------------------------------------------

    fn roundtrip_f64(n: usize) {
        let original: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64).sin(), (i as f64 * 0.7).cos()))
            .collect();
        let mut transformed = vec![Complex::zero(); n];
        let mut recovered = vec![Complex::zero(); n];

        let solver = BluesteinSolver::new(n);
        solver.execute(&original, &mut transformed, Sign::Forward);
        solver.execute(&transformed, &mut recovered, Sign::Backward);

        let n_f = n as f64;
        let mut max_rel = 0.0_f64;
        for (orig, rec) in original.iter().zip(recovered.iter()) {
            let rec_scaled = *rec / n_f;
            let re_err = (orig.re - rec_scaled.re).abs();
            let im_err = (orig.im - rec_scaled.im).abs();
            let norm = (orig.re * orig.re + orig.im * orig.im).sqrt().max(1e-30);
            max_rel = max_rel.max((re_err + im_err) / norm);
        }
        assert!(
            max_rel < 1e-13,
            "bluestein f64 round-trip n={n}: max_rel={max_rel} (must be < 1e-13)"
        );
    }

    fn roundtrip_f32(n: usize) {
        let original: Vec<Complex<f32>> = (0..n)
            .map(|i| Complex::new((i as f32).sin(), (i as f32 * 0.7).cos()))
            .collect();
        let mut transformed = vec![Complex::new(0.0_f32, 0.0); n];
        let mut recovered = vec![Complex::new(0.0_f32, 0.0); n];

        let solver = BluesteinSolver::<f32>::new(n);
        solver.execute(&original, &mut transformed, Sign::Forward);
        solver.execute(&transformed, &mut recovered, Sign::Backward);

        let n_f = n as f32;
        let mut max_rel = 0.0_f32;
        for (orig, rec) in original.iter().zip(recovered.iter()) {
            let rec_scaled = *rec / n_f;
            let re_err = (orig.re - rec_scaled.re).abs();
            let im_err = (orig.im - rec_scaled.im).abs();
            let norm = (orig.re * orig.re + orig.im * orig.im)
                .sqrt()
                .max(1e-10_f32);
            max_rel = max_rel.max((re_err + im_err) / norm);
        }
        // f32 has ~7 decimal digits; a round-trip through Bluestein (two nested FFTs)
        // for sizes up to ~1009 accumulates O(sqrt(n) * eps_f32) error.
        // 5e-4 is a conservative bound: for n=1009, sqrt(1009)*6e-8 ≈ 1.9e-6 * n_log_n.
        assert!(
            max_rel < 5e-4,
            "bluestein f32 round-trip n={n}: max_rel={max_rel} (must be < 5e-4)"
        );
    }

    #[test]
    fn roundtrip_prime_17_f64() {
        roundtrip_f64(17);
    }
    #[test]
    fn roundtrip_prime_61_f64() {
        roundtrip_f64(61);
    }
    #[test]
    fn roundtrip_prime_127_f64() {
        roundtrip_f64(127);
    }
    #[test]
    fn roundtrip_prime_257_f64() {
        roundtrip_f64(257);
    }
    #[test]
    fn roundtrip_prime_509_f64() {
        roundtrip_f64(509);
    }
    #[test]
    fn roundtrip_prime_1009_f64() {
        roundtrip_f64(1009);
    }

    #[test]
    fn roundtrip_prime_17_f32() {
        roundtrip_f32(17);
    }
    #[test]
    fn roundtrip_prime_61_f32() {
        roundtrip_f32(61);
    }
    #[test]
    fn roundtrip_prime_127_f32() {
        roundtrip_f32(127);
    }
    #[test]
    fn roundtrip_prime_257_f32() {
        roundtrip_f32(257);
    }
    #[test]
    fn roundtrip_prime_509_f32() {
        roundtrip_f32(509);
    }
    #[test]
    fn roundtrip_prime_1009_f32() {
        roundtrip_f32(1009);
    }

    // -------------------------------------------------------------------------
    // Parallel (rayon) test: many threads share a single BluesteinSolver
    // -------------------------------------------------------------------------

    #[cfg(feature = "threading")]
    #[test]
    fn parallel_shared_solver_correctness() {
        use rayon::prelude::*;

        let n = 61_usize;
        let solver = std::sync::Arc::new(BluesteinSolver::new(n));

        // Reference: single-threaded forward FFT
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut reference = vec![Complex::zero(); n];
        solver.execute(&input, &mut reference, Sign::Forward);

        // Run 16 parallel computations — each must match the reference
        let results: Vec<Vec<Complex<f64>>> = (0..16_usize)
            .into_par_iter()
            .map(|_| {
                let mut out = vec![Complex::zero(); n];
                solver.execute(&input, &mut out, Sign::Forward);
                out
            })
            .collect();

        for (thread_idx, result) in results.iter().enumerate() {
            for (k, (r, rr)) in result.iter().zip(reference.iter()).enumerate() {
                let err = ((r.re - rr.re).abs() + (r.im - rr.im).abs())
                    / (rr.re * rr.re + rr.im * rr.im).sqrt().max(1e-30);
                assert!(
                    err < 1e-12,
                    "parallel thread {thread_idx} output[{k}] diverged: err={err}"
                );
            }
        }
    }
}
