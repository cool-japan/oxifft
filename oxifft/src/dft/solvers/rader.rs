//! Rader's algorithm for prime-size DFT.
//!
//! Rader's algorithm converts a prime-size DFT into a cyclic convolution
//! of size p-1, which can then be computed using FFT if p-1 has good factors.
//!
//! For a prime p, the DFT can be rewritten using the primitive root g:
//! X[g^k] = x[0] + Σ_{j=0}^{p-2} x[g^{-j}] * W_p^{g^{k-j}}
//!
//! The summation is a cyclic convolution.
//!
//! Time complexity: O(p log p) for prime p (when p-1 has good factors)
//! Space complexity: O(p)
//!
//! # Optimizations in this implementation
//!
//! - **SIMD-accelerated pointwise multiply** of `a_fft[i] * twiddle_fft[i]` via
//!   `complex_mul_aos`, dispatching to AVX2+FMA / SSE2 / NEON at runtime.
//! - **SIMD-dispatched inner convolution FFT** of size `p-1`: the length-(p-1)
//!   cyclic convolution is computed with the fastest available engine for that
//!   size ([`ConvSolver`]) rather than unconditionally through Bluestein.  For a
//!   prime `p ≥ 5`, `p-1` is always even and therefore composite, so the
//!   convolution routes through the SIMD-accelerated radix-2 Cooley-Tukey path
//!   (power-of-two `p-1`) or the SIMD mixed-radix [`GenericSolver`] (smooth
//!   composite `p-1`), which are markedly cheaper than Bluestein's zero-padding
//!   to `next_pow2(2(p-1)-1)`.  Bluestein remains the fallback only when `p-1`
//!   has a large prime factor (so the mixed-radix recursion would itself bottom
//!   out in Bluestein anyway).
//! - **4th inplace scratch mutex**: `work_inplace` avoids an unconditional
//!   `data.to_vec()` allocation in `execute_inplace` for the common case.
//! - **Unique solver ID**: assigned at construction for potential debugging.

// `AtomicUsize` (not `AtomicU64`) so the monotonic solver-ID counter also works
// on targets without 64-bit atomics (e.g. riscv32imac). The id is session-local
// and never persisted, so the narrower range on 32-bit targets is harmless.
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::dft::problem::Sign;
use crate::kernel::complex_mul::complex_mul_aos;
use crate::kernel::{is_prime, primitive_root};
use crate::kernel::{Complex, Float};
use crate::prelude::*;

use super::bluestein::BluesteinSolver;
use super::ct::CooleyTukeySolver;
use super::generic::GenericSolver;

/// Inner FFT engine for the length-(p-1) cyclic convolution at the heart of
/// Rader's algorithm.
///
/// The convolution requires a forward and an inverse DFT of size `n = p-1`.
/// Because `p` is an odd prime (`≥ 3`), `n` is always even; for every prime
/// `p ≥ 5` this makes `n` composite, so a dedicated composite engine is both
/// applicable and much faster than Bluestein (which would pad `n` up to
/// `next_pow2(2n-1)` and run two FFTs of that larger size per convolution DFT).
///
/// All three variants share the same DFT sign convention (`Forward` uses
/// `e^{-2πi/n}`) and leave the inverse transform unnormalized, so the cyclic
/// convolution identity holds regardless of which one is selected.
enum ConvSolver<T: Float> {
    /// `n` is a power of two: SIMD-accelerated radix-2 DIT Cooley-Tukey.
    PowerOfTwo(CooleyTukeySolver<T>),
    /// `n` is composite (non power-of-two): SIMD mixed-radix recursive solver.
    Generic(Box<GenericSolver<T>>),
    /// Fallback for `n` with a large prime factor (`> GENERIC_MAX_PRIME_FACTOR`):
    /// Bluestein over the whole of `n`.
    Bluestein(Box<BluesteinSolver<T>>),
}

impl<T: Float> ConvSolver<T> {
    /// Largest prime factor for which the mixed-radix [`GenericSolver`] recursion
    /// still bottoms out in a small codelet / direct kernel rather than an inner
    /// Bluestein call.  For `n` whose largest prime factor exceeds this, a single
    /// Bluestein transform over the whole of `n` is cheaper than splitting `n`
    /// only to recurse into Bluestein on the large prime factor anyway.
    const GENERIC_MAX_PRIME_FACTOR: usize = 13;

    /// Select the fastest inner engine for a length-`n` cyclic convolution.
    fn new(n: usize) -> Self {
        if n >= 2 && n.is_power_of_two() {
            Self::PowerOfTwo(CooleyTukeySolver::default())
        } else if GenericSolver::<T>::applicable(n) && Self::generic_is_favourable(n) {
            Self::Generic(Box::new(GenericSolver::new(n)))
        } else {
            Self::Bluestein(Box::new(BluesteinSolver::new(n)))
        }
    }

    /// Returns `true` when every prime factor of `n` is small enough that the
    /// mixed-radix decomposition terminates in codelets rather than an inner
    /// Bluestein transform (see [`Self::GENERIC_MAX_PRIME_FACTOR`]).
    fn generic_is_favourable(n: usize) -> bool {
        use crate::kernel::factor;
        factor(n)
            .last()
            .is_none_or(|&(prime, _)| prime <= Self::GENERIC_MAX_PRIME_FACTOR)
    }

    /// Execute a forward/inverse DFT of size `n` (unnormalized).
    #[inline]
    fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        match self {
            Self::PowerOfTwo(s) => s.execute(input, output, sign),
            Self::Generic(s) => s.execute(input, output, sign),
            Self::Bluestein(s) => s.execute(input, output, sign),
        }
    }
}

/// Global counter for assigning unique IDs to each `RaderSolver`.
static RADER_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Rader's algorithm solver for prime sizes.
///
/// This solver uses the fact that for prime p, the non-zero indices
/// form a cyclic group under multiplication modulo p, generated by
/// a primitive root g.
///
/// Pre-allocates work buffers to avoid per-execution allocations.
/// Uses `Mutex` for thread-safe interior mutability with `try_lock()` fallback.
pub struct RaderSolver<T: Float> {
    /// Prime size
    p: usize,
    /// Primitive root
    g: usize,
    /// Powers of g: g^0, g^1, ..., g^(p-2) mod p
    g_powers: Vec<usize>,
    /// Inverse powers: g^(-0), g^(-1), ..., g^(-(p-2)) mod p (same as g^(p-1-k))
    g_inv_powers: Vec<usize>,
    /// Precomputed FFT of forward twiddle factors
    twiddle_fft_fwd: Vec<Complex<T>>,
    /// Precomputed FFT of backward twiddle factors
    twiddle_fft_bwd: Vec<Complex<T>>,
    /// Inner FFT engine for the length-(p-1) cyclic convolution.
    conv_solver: ConvSolver<T>,
    /// Unique solver identifier for debugging.
    pub(crate) solver_id: u64,
    /// Pre-allocated work buffer for reordered input
    #[cfg(feature = "std")]
    work_a: Mutex<Vec<Complex<T>>>,
    /// Pre-allocated work buffer for FFT of input
    #[cfg(feature = "std")]
    work_a_fft: Mutex<Vec<Complex<T>>>,
    /// Pre-allocated work buffer for convolution result
    #[cfg(feature = "std")]
    work_conv: Mutex<Vec<Complex<T>>>,
    /// Pre-allocated work buffer for in-place scratch (avoids unconditional alloc)
    #[cfg(feature = "std")]
    work_inplace: Mutex<Vec<Complex<T>>>,
}

impl<T: Float> RaderSolver<T> {
    /// Create a new Rader solver for prime size p.
    ///
    /// Returns `None` if p is not prime or p < 3.
    #[must_use]
    pub fn new(p: usize) -> Option<Self> {
        if p < 3 || !is_prime(p) {
            return None;
        }

        let solver_id = RADER_ID_COUNTER.fetch_add(1, Ordering::Relaxed) as u64;
        let g = primitive_root(p)?;
        let n = p - 1; // Convolution size

        // Compute powers of g
        let mut g_powers = Vec::with_capacity(n);
        let mut g_inv_powers = Vec::with_capacity(n);

        let mut power = 1usize;
        for _ in 0..n {
            g_powers.push(power);
            power = (power * g) % p;
        }

        // g^(-k) = g^(p-1-k) mod p (since g^(p-1) = 1 mod p by Fermat's little theorem)
        for k in 0..n {
            g_inv_powers.push(g_powers[(n - k) % n]);
        }

        let conv_solver = ConvSolver::new(n);

        // Precompute forward twiddle factors: W_p^(g^k) for k = 0..p-2
        // W_p = e^(-2πi/p) for forward transform
        let mut twiddles_fwd = Vec::with_capacity(n);
        for k in 0..n {
            let exp = g_powers[k];
            let angle = -<T as Float>::TWO_PI * T::from_usize(exp) / T::from_usize(p);
            twiddles_fwd.push(Complex::cis(angle));
        }

        // FFT of forward twiddles
        let mut twiddle_fft_fwd = vec![Complex::zero(); n];
        conv_solver.execute(&twiddles_fwd, &mut twiddle_fft_fwd, Sign::Forward);

        // Precompute backward twiddle factors: W_p^(-g^k) for k = 0..p-2
        // W_p^(-1) = e^(+2πi/p) for backward transform
        let mut twiddles_bwd = Vec::with_capacity(n);
        for k in 0..n {
            let exp = g_powers[k];
            let angle = <T as Float>::TWO_PI * T::from_usize(exp) / T::from_usize(p);
            twiddles_bwd.push(Complex::cis(angle));
        }

        // FFT of backward twiddles
        let mut twiddle_fft_bwd = vec![Complex::zero(); n];
        conv_solver.execute(&twiddles_bwd, &mut twiddle_fft_bwd, Sign::Forward);

        Some(Self {
            p,
            g,
            g_powers,
            g_inv_powers,
            twiddle_fft_fwd,
            twiddle_fft_bwd,
            conv_solver,
            solver_id,
            #[cfg(feature = "std")]
            work_a: Mutex::new(vec![Complex::zero(); n]),
            #[cfg(feature = "std")]
            work_a_fft: Mutex::new(vec![Complex::zero(); n]),
            #[cfg(feature = "std")]
            work_conv: Mutex::new(vec![Complex::zero(); n]),
            #[cfg(feature = "std")]
            work_inplace: Mutex::new(vec![Complex::zero(); p]),
        })
    }

    /// Solver name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        "dft-rader"
    }

    /// Get the prime size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.p
    }

    /// Get the primitive root used.
    #[must_use]
    pub fn primitive_root(&self) -> usize {
        self.g
    }

    /// Returns the unique solver ID (monotonically increasing per-process counter).
    #[must_use]
    pub fn id(&self) -> u64 {
        self.solver_id
    }

    /// Check if this solver is applicable (p is prime and >= 3).
    #[must_use]
    pub fn applicable(p: usize) -> bool {
        p >= 3 && is_prime(p)
    }

    /// Execute Rader's FFT with provided work buffers.
    fn execute_with_buffers(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        sign: Sign,
        a: &mut [Complex<T>],
        a_fft: &mut [Complex<T>],
        conv: &mut [Complex<T>],
    ) {
        let p = self.p;
        let n = p - 1;

        // Step 1: Compute X[0] = sum of all inputs
        let mut sum = Complex::zero();
        for x in input {
            sum = sum + *x;
        }

        // Step 2: Reorder input according to g^(-j)
        // This scatter/gather pattern stays scalar (indexing is indirect).
        for j in 0..n {
            a[j] = input[self.g_inv_powers[j]];
        }

        // Step 3: FFT of reordered input
        self.conv_solver.execute(a, a_fft, Sign::Forward);

        // Step 4: Pointwise multiply with appropriate twiddle FFT (SIMD-accelerated).
        let twiddle_fft = match sign {
            Sign::Forward => &self.twiddle_fft_fwd,
            Sign::Backward => &self.twiddle_fft_bwd,
        };

        // Use conv as temp storage to avoid aliasing a_fft as both src and dst.
        complex_mul_aos(&mut conv[..n], a_fft, twiddle_fft);
        // Now conv holds a_fft * twiddle_fft; use it as input to IFFT
        // and write result back to a_fft (reusing that buffer).
        a_fft[..n].copy_from_slice(&conv[..n]);

        // Step 5: IFFT to get convolution result
        self.conv_solver.execute(a_fft, conv, Sign::Backward);

        // Normalize IFFT
        let n_inv = T::ONE / T::from_usize(n);
        for x in conv.iter_mut().take(n) {
            *x = *x * n_inv;
        }

        // Step 6: Compute output
        // X[0] = sum (already computed)
        output[0] = sum;

        // X[g^k] = x[0] + conv[k] for k = 0..p-2
        for k in 0..n {
            let idx = self.g_powers[k];
            output[idx] = input[0] + conv[k];
        }
    }

    /// Execute Rader's FFT algorithm.
    ///
    /// Uses pre-allocated work buffers when available (single-threaded case).
    /// Falls back to fresh allocation when buffers are locked (parallel execution).
    #[cfg(feature = "std")]
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        let p = self.p;
        let n = p - 1;

        debug_assert_eq!(input.len(), p);
        debug_assert_eq!(output.len(), p);

        // Try to acquire all three locks. If any fails, allocate fresh buffers.
        let a_guard = self.work_a.try_lock();
        let a_fft_guard = self.work_a_fft.try_lock();
        let conv_guard = self.work_conv.try_lock();

        if let (Ok(mut a), Ok(mut a_fft), Ok(mut conv)) = (a_guard, a_fft_guard, conv_guard) {
            // Use pre-allocated buffers
            self.execute_with_buffers(input, output, sign, &mut a, &mut a_fft, &mut conv);
        } else {
            // Fallback: allocate fresh buffers (parallel execution case)
            let mut a = vec![Complex::zero(); n];
            let mut a_fft = vec![Complex::zero(); n];
            let mut conv = vec![Complex::zero(); n];
            self.execute_with_buffers(input, output, sign, &mut a, &mut a_fft, &mut conv);
        }
    }

    /// Execute Rader's FFT algorithm (no_std version - always allocates).
    #[cfg(not(feature = "std"))]
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        let p = self.p;
        let n = p - 1;

        debug_assert_eq!(input.len(), p);
        debug_assert_eq!(output.len(), p);

        // no_std: always allocate fresh buffers
        let mut a = vec![Complex::zero(); n];
        let mut a_fft = vec![Complex::zero(); n];
        let mut conv = vec![Complex::zero(); n];
        self.execute_with_buffers(input, output, sign, &mut a, &mut a_fft, &mut conv);
    }

    /// Execute Rader's FFT in-place.
    ///
    /// Uses the pre-allocated `work_inplace` buffer when available (avoids
    /// an unconditional `data.to_vec()` allocation in the common case).
    pub fn execute_inplace(&self, data: &mut [Complex<T>], sign: Sign) {
        let p = self.p;
        debug_assert_eq!(data.len(), p);

        #[cfg(feature = "std")]
        {
            if let Ok(mut inplace_buf) = self.work_inplace.try_lock() {
                if inplace_buf.len() < p {
                    inplace_buf.resize(p, Complex::zero());
                }
                inplace_buf[..p].copy_from_slice(data);
                let input_ptr = inplace_buf[..p].as_ptr();
                let input_slice = unsafe { core::slice::from_raw_parts(input_ptr, p) };
                self.execute(input_slice, data, sign);
                return;
            }
        }

        // Fallback: allocate a fresh copy (mutex contended or no_std)
        let input: Vec<Complex<T>> = data.to_vec();
        self.execute(&input, data, sign);
    }
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

    /// Regression test for the inner-convolution engine selection.
    ///
    /// Rader used to force `BluesteinSolver` for the length-(p-1) convolution
    /// even when `p-1` is a power of two or a smooth composite, where a direct
    /// Cooley-Tukey / mixed-radix transform is markedly faster.  This test
    /// locks in that the fast engines are chosen for the sizes that matter and
    /// that Bluestein remains the fallback only when `p-1` has a large prime
    /// factor (so the mixed-radix recursion would otherwise bottom out in
    /// Bluestein anyway).
    #[test]
    fn conv_solver_selects_fast_engine() {
        // p-1 is a power of two (Fermat-style primes 17, 257) -> Cooley-Tukey.
        assert!(matches!(
            ConvSolver::<f64>::new(16),
            ConvSolver::PowerOfTwo(_)
        ));
        assert!(matches!(
            ConvSolver::<f64>::new(256),
            ConvSolver::PowerOfTwo(_)
        ));
        // Smooth composite p-1 (96 = 2^5·3, 1008 = 2^4·3^2·7) -> mixed-radix.
        assert!(matches!(ConvSolver::<f64>::new(96), ConvSolver::Generic(_)));
        assert!(matches!(
            ConvSolver::<f64>::new(1008),
            ConvSolver::Generic(_)
        ));
        // p-1 with a large prime factor (358 = 2·179) -> Bluestein fallback.
        assert!(matches!(
            ConvSolver::<f64>::new(358),
            ConvSolver::Bluestein(_)
        ));
    }

    /// Correctness across all three inner-engine paths, checked against the
    /// O(n²) direct DFT so a mis-selected engine cannot pass silently.
    #[test]
    fn rader_matches_direct_across_engine_paths() {
        // p=17 -> conv 16 (Cooley-Tukey); p=97 -> conv 96 (Generic);
        // p=359 -> conv 358 (Bluestein fallback, largest prime factor 179).
        for &p in &[17_usize, 97, 359] {
            let input: Vec<Complex<f64>> = (0..p)
                .map(|i| Complex::new((i as f64 * 0.3).sin(), (i as f64 * 0.7).cos()))
                .collect();
            let mut rader_out = vec![Complex::zero(); p];
            let mut direct_out = vec![Complex::zero(); p];

            RaderSolver::new(p)
                .expect("p is prime")
                .execute(&input, &mut rader_out, Sign::Forward);
            DirectSolver::new().execute(&input, &mut direct_out, Sign::Forward);

            for (k, (a, b)) in rader_out.iter().zip(direct_out.iter()).enumerate() {
                assert!(
                    complex_approx_eq(*a, *b, 1e-8),
                    "p={p} output[{k}]: rader={a:?} direct={b:?}"
                );
            }
        }
    }

    #[test]
    fn test_rader_applicable() {
        assert!(!RaderSolver::<f64>::applicable(0));
        assert!(!RaderSolver::<f64>::applicable(1));
        assert!(!RaderSolver::<f64>::applicable(2));
        assert!(RaderSolver::<f64>::applicable(3));
        assert!(!RaderSolver::<f64>::applicable(4));
        assert!(RaderSolver::<f64>::applicable(5));
        assert!(!RaderSolver::<f64>::applicable(6));
        assert!(RaderSolver::<f64>::applicable(7));
        assert!(RaderSolver::<f64>::applicable(11));
        assert!(RaderSolver::<f64>::applicable(13));
    }

    #[test]
    fn test_rader_size_3() {
        let input: Vec<Complex<f64>> = (0..3).map(|i| Complex::new(f64::from(i), 0.0)).collect();
        let mut output_rader = vec![Complex::zero(); 3];
        let mut output_direct = vec![Complex::zero(); 3];

        RaderSolver::new(3).expect("p=3 is prime").execute(
            &input,
            &mut output_rader,
            Sign::Forward,
        );
        DirectSolver::new().execute(&input, &mut output_direct, Sign::Forward);

        for (a, b) in output_rader.iter().zip(output_direct.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_rader_size_5() {
        let input: Vec<Complex<f64>> = (0..5)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut output_rader = vec![Complex::zero(); 5];
        let mut output_direct = vec![Complex::zero(); 5];

        RaderSolver::new(5).expect("p=5 is prime").execute(
            &input,
            &mut output_rader,
            Sign::Forward,
        );
        DirectSolver::new().execute(&input, &mut output_direct, Sign::Forward);

        for (a, b) in output_rader.iter().zip(output_direct.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_rader_size_7() {
        let input: Vec<Complex<f64>> = (0..7)
            .map(|i| Complex::new(f64::from(i), f64::from(i) * 0.5))
            .collect();
        let mut output_rader = vec![Complex::zero(); 7];
        let mut output_direct = vec![Complex::zero(); 7];

        RaderSolver::new(7).expect("p=7 is prime").execute(
            &input,
            &mut output_rader,
            Sign::Forward,
        );
        DirectSolver::new().execute(&input, &mut output_direct, Sign::Forward);

        for (a, b) in output_rader.iter().zip(output_direct.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_rader_size_13() {
        let input: Vec<Complex<f64>> = (0..13)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut output_rader = vec![Complex::zero(); 13];
        let mut output_direct = vec![Complex::zero(); 13];

        RaderSolver::new(13).expect("p=13 is prime").execute(
            &input,
            &mut output_rader,
            Sign::Forward,
        );
        DirectSolver::new().execute(&input, &mut output_direct, Sign::Forward);

        for (a, b) in output_rader.iter().zip(output_direct.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-8));
        }
    }

    #[test]
    fn test_rader_inverse_recovers_input() {
        let original: Vec<Complex<f64>> = (0..11)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut transformed = vec![Complex::zero(); 11];
        let mut recovered = vec![Complex::zero(); 11];

        let solver = RaderSolver::new(11).expect("p=11 is prime");
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
    fn test_rader_inplace() {
        let original: Vec<Complex<f64>> = (0..7).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        // Out-of-place reference
        let mut out_of_place = vec![Complex::zero(); 7];
        let solver = RaderSolver::new(7).expect("p=7 is prime");
        solver.execute(&original, &mut out_of_place, Sign::Forward);

        // In-place
        let mut in_place = original;
        solver.execute_inplace(&mut in_place, Sign::Forward);

        for (a, b) in out_of_place.iter().zip(in_place.iter()) {
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

        let solver = RaderSolver::new(n).expect("n must be prime");
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
            max_rel < 1e-12,
            "rader f64 round-trip n={n}: max_rel={max_rel} (must be < 1e-12)"
        );
    }

    fn roundtrip_f32(n: usize) {
        let original: Vec<Complex<f32>> = (0..n)
            .map(|i| Complex::new((i as f32).sin(), (i as f32 * 0.7).cos()))
            .collect();
        let mut transformed = vec![Complex::new(0.0_f32, 0.0); n];
        let mut recovered = vec![Complex::new(0.0_f32, 0.0); n];

        let solver = RaderSolver::<f32>::new(n).expect("n must be prime");
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
        // f32 accumulates more rounding in Rader's nested FFT chain (Bluestein inside Rader).
        // For n=1009, two levels of FFT means error budget ≈ O(log²(n) * eps_f32).
        assert!(
            max_rel < 1e-3,
            "rader f32 round-trip n={n}: max_rel={max_rel} (must be < 1e-3)"
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
    // Parallel (rayon) test: many threads share a single RaderSolver
    // -------------------------------------------------------------------------

    #[cfg(feature = "threading")]
    #[test]
    fn parallel_shared_rader_correctness() {
        use rayon::prelude::*;

        let p = 61_usize;
        let solver = std::sync::Arc::new(RaderSolver::new(p).expect("p=61 is prime"));

        // Reference: single-threaded forward FFT
        let input: Vec<Complex<f64>> = (0..p)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut reference = vec![Complex::zero(); p];
        solver.execute(&input, &mut reference, Sign::Forward);

        // Run 16 parallel computations — each must match the reference
        let results: Vec<Vec<Complex<f64>>> = (0..16_usize)
            .into_par_iter()
            .map(|_| {
                let mut out = vec![Complex::zero(); p];
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
