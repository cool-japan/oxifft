//! Sparse FFT for k-sparse signals with a certified fast path.
//!
//! Recovers the non-zero frequency components of a signal that has at most `k`
//! significant bins.  The fast path is sub-linear in the common case, and the
//! public API is *always correct*: whenever the fast path cannot certify its
//! answer the implementation transparently falls back to a full FFT plus
//! top-`k` selection, so a silently-wrong result can never be returned.
//!
//! # Algorithm
//!
//! The fast path is an aliasing-based sparse transform:
//!
//! 1. **Aliasing stages.** For each of a few bucket counts `B` (each a divisor
//!    of `n`), the signal is downsampled by `L = n / B` at offset 0
//!    (`z0[j] = x[j·L]`) and offset 1 (`z1[j] = x[j·L + 1]`).  Each length-`B`
//!    DFT aliases every frequency `f` into bucket `b = f mod B`:
//!    `Z0[b] = (1/L)·Σ_{f≡b} X[f]` and `Z1[b] = (1/L)·Σ_{f≡b} X[f]·e^{2πi f/n}`.
//!    Building a stage reads only `2B` samples, which is what makes the fast
//!    path cheap when `k ≪ n`.
//!
//! 2. **Phase-based singleton detection.** A bucket holding exactly one
//!    frequency (a *singleton*) has `Z1[b]/Z0[b] = e^{2πi f/n}`, a pure phase,
//!    from which `f` is recovered exactly and `X[f] = L·Z0[b]`.  A bucket is
//!    accepted as a singleton only when the recovered `f` aliases back to `b`
//!    **and** `|Z1[b]| = |Z0[b]|`; collisions fail both checks.
//!
//! 3. **Peeling.** Recovered frequencies are subtracted from every stage,
//!    which may expose new singletons; this repeats until no singleton remains.
//!    Distinct bucket counts let peeling separate frequencies that collide
//!    under one bucketization but not another.
//!
//! 4. **Certification.** The recovered set is accepted only if the residual of
//!    **both** offset measurements has collapsed below a relative tolerance in
//!    every bucket.  Two distinct un-recovered frequencies cannot cancel in
//!    both measurements at once (that would require identical phase, i.e.
//!    identical frequency), so a passing certification proves the recovered set
//!    explains essentially all signal energy.  Its top-`k` is then the true
//!    top-`k`.
//!
//! If certification fails, the dense fallback runs a full FFT and returns the
//! `k` largest-magnitude bins — the exact same top-`k` contract, just computed
//! the expensive way.
//!
//! # Complexity
//!
//! - **Fast path (certified):** reads `O(B)` samples per stage and runs a few
//!   `O(B log B)` bucket FFTs with `B = O(k)`, i.e. sub-linear sampling and
//!   `O(k log k)`-scale work, plus `O(k)` peeling.
//! - **Dense fallback:** `O(n log n)`, identical to `Plan::dft_1d` + top-`k`.
//!
//! # Accuracy Guarantees
//!
//! - **Correct top-`k`, always.** Every value returned by [`sparse_fft`] /
//!   [`SparsePlan::execute`] is either a certified fast-path result or a dense
//!   top-`k` result; both are the correct `k` largest DFT bins with their exact
//!   complex values (to floating-point tolerance).  No silently-wrong output is
//!   possible.
//!
//! - **Exact recovery on clean sparse signals.** When the spectrum has at most
//!   `k` non-zero bins and every bin isolates in at least one aliasing stage,
//!   the fast path recovers each planted `(index, value)` exactly.
//!
//! - **Graceful degradation.** When frequencies collide across every stage
//!   (e.g. two bins separated by `n/2` for a power-of-two `n`), or when noise
//!   raises the residual, certification fails and the dense fallback supplies
//!   the correct top-`k`.
//!
//! # Regime split (when the fast path is actually used)
//!
//! The fast path is *attempted* when `k < n/4` and `n > 64` and `n` has a
//! divisor `B` with `O(k) ≤ B ≤ n/2` (so the downsampling is exact and
//! reduces).  It *succeeds* (certifies) when the planted frequencies isolate
//! under the chosen bucketizations and the signal is close to exactly
//! `k`-sparse.  Prime `n`, dense spectra (`k ≥ n/4`), tiny `n` (`≤ 64`),
//! heavy collisions, and noisy signals all route through the dense fallback,
//! which is why they remain correct.
//!
//! # Noise
//!
//! The fast path does not denoise: additive noise spreads energy into every
//! bucket, so certification fails and the dense fallback returns the top-`k`
//! bins of the *noisy* spectrum.  As long as the noise floor stays well below
//! the `k`-th largest planted bin (empirically, per-bin noise magnitude below
//! roughly 1 % of the smallest planted `|X[f]|`), those top-`k` bins are the
//! planted ones.  Beyond that the returned bins simply track the dense FFT's
//! top-`k`, as expected.
//!
//! # Example
//!
//! ```
//! use oxifft::sparse::{sparse_fft, SparsePlan};
//! use oxifft::api::Flags;
//!
//! let n = 1024;
//! let k = 10;
//! let signal = vec![oxifft::Complex::new(0.0_f64, 0.0); n];
//!
//! // One-shot API.
//! let result = sparse_fft(&signal, k);
//! for (idx, value) in result.iter() {
//!     let _ = (idx, value); // inspect each detected frequency
//! }
//!
//! // Plan-based API for repeated use.
//! let plan = SparsePlan::<f64>::new(n, k, Flags::ESTIMATE).expect("valid plan params");
//! let result = plan.execute(&signal);
//! assert!(result.indices.len() <= k);
//! ```

mod decoder;
mod plan;
mod problem;
mod result;

pub use plan::SparsePlan;
pub use problem::SparseProblem;
pub use result::SparseResult;

use crate::api::Flags;
use crate::kernel::{Complex, Float};
use crate::prelude::*;

/// Compute the sparse FFT of a signal with at most `k` non-zero frequency
/// components.
///
/// Returns the `k` largest-magnitude frequency bins with their exact complex
/// values.  The result is always correct: the certified fast path is used when
/// it can prove its answer, otherwise a full FFT plus top-`k` selection is used
/// (see the [module documentation](self)).
///
/// - `k < n/4` and `n > 64`: the fast path is attempted (dense fallback on
///   certification failure).
/// - `k >= n/4` or `n <= 64`: the dense path is used directly.
///
/// # Arguments
///
/// * `input` - Input signal in the time domain.
/// * `k` - Expected sparsity (maximum number of non-zero frequencies).  `k = 0`
///   returns an empty result; the result never contains more than `k` bins.
///
/// # Returns
///
/// A [`SparseResult`] with the detected frequency indices and their values.
pub fn sparse_fft<T: Float>(input: &[Complex<T>], k: usize) -> SparseResult<T> {
    let n = input.len();
    if n == 0 || k == 0 {
        return SparseResult::empty();
    }

    // For dense spectra or tiny inputs, use the dense path directly.
    if k >= n / 4 || n <= 64 {
        return sparse_fft_fallback(input, k);
    }

    match SparsePlan::new(n, k, Flags::ESTIMATE) {
        Some(plan) => plan.execute(input),
        None => sparse_fft_fallback(input, k),
    }
}

/// Dense path: full FFT then top-`k` bins by magnitude.
fn sparse_fft_fallback<T: Float>(input: &[Complex<T>], k: usize) -> SparseResult<T> {
    use crate::api::Plan;

    let n = input.len();
    let plan = match Plan::dft_1d(n, crate::api::Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return SparseResult::empty(),
    };

    let mut output = vec![Complex::<T>::zero(); n];
    plan.execute(input, &mut output);

    let mut magnitudes: Vec<(usize, T)> = output
        .iter()
        .enumerate()
        .map(|(i, c)| (i, c.norm_sqr()))
        .collect();
    magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

    let k_actual = k.min(n);
    let indices: Vec<usize> = magnitudes[..k_actual].iter().map(|(i, _)| *i).collect();
    let values: Vec<Complex<T>> = indices.iter().map(|&i| output[i]).collect();

    SparseResult::new(indices, values, n)
}

/// Compute a sparse FFT with automatic sparsity detection (adaptive `k`).
///
/// Estimates `k` from the signal via an energy-based heuristic, then returns
/// the energy-dominant bins.  Because it performs a **full FFT first**, its cost
/// is `O(n log n)` regardless of true sparsity; use it when sparsity is unknown.
///
/// # Arguments
///
/// * `input` - Input signal in the time domain.
pub fn sparse_fft_auto<T: Float>(input: &[Complex<T>]) -> SparseResult<T> {
    sparse_fft_auto_with_ratio(input, T::from_f64(0.99))
}

/// Compute a sparse FFT with automatic sparsity detection and a custom energy
/// ratio.
///
/// The `energy_ratio` parameter (in `(0, 1]`) controls how much of the total
/// spectral energy must be captured: a higher ratio keeps more bins.
///
/// # Arguments
///
/// * `input`        - Input signal in the time domain.
/// * `energy_ratio` - Fraction of total spectral energy to retain (0 < r ≤ 1).
pub fn sparse_fft_auto_with_ratio<T: Float>(
    input: &[Complex<T>],
    energy_ratio: T,
) -> SparseResult<T> {
    let n = input.len();
    if n == 0 {
        return SparseResult::empty();
    }

    use crate::api::Plan;
    let plan = match Plan::dft_1d(n, crate::api::Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return SparseResult::empty(),
    };

    let mut spectrum = vec![Complex::<T>::zero(); n];
    plan.execute(input, &mut spectrum);

    let mut mag_vec: Vec<(usize, T)> = spectrum
        .iter()
        .enumerate()
        .map(|(i, c)| (i, c.norm_sqr()))
        .collect();

    let total_energy: T = mag_vec.iter().map(|(_, m)| *m).fold(T::ZERO, |a, b| a + b);
    if total_energy <= T::ZERO {
        return SparseResult::empty();
    }

    mag_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

    let target = total_energy * energy_ratio;
    let mut accumulated = T::ZERO;
    let mut estimated_k: usize = 0;
    for &(_, mag) in &mag_vec {
        accumulated = accumulated + mag;
        estimated_k += 1;
        if accumulated >= target {
            break;
        }
    }

    let estimated_k = estimated_k.max(1).min(n);
    let top_k = &mag_vec[..estimated_k];
    let indices: Vec<usize> = top_k.iter().map(|(i, _)| *i).collect();
    let values: Vec<Complex<T>> = indices.iter().map(|&i| spectrum[i]).collect();

    SparseResult::new(indices, values, n)
}

/// Compute the inverse sparse FFT (sparse frequency domain to time domain).
///
/// # Arguments
///
/// * `sparse_result` - Sparse frequency-domain representation.
/// * `n` - Output signal length.
///
/// # Returns
///
/// Time-domain signal reconstructed from the sparse frequencies.
pub fn sparse_ifft<T: Float>(sparse_result: &SparseResult<T>, n: usize) -> Vec<Complex<T>> {
    let mut output = vec![Complex::<T>::zero(); n];

    let scale = T::ONE / T::from_usize(n);
    let two_pi = <T as Float>::PI + <T as Float>::PI;

    for t in 0..n {
        let mut sum = Complex::<T>::zero();
        for (&freq_idx, &value) in sparse_result
            .indices
            .iter()
            .zip(sparse_result.values.iter())
        {
            let angle = two_pi * T::from_usize(freq_idx * t) / T::from_usize(n);
            let (sin_a, cos_a) = Float::sin_cos(angle);
            let twiddle = Complex::new(cos_a, sin_a);
            sum = sum + value * twiddle;
        }
        output[t] = sum * scale;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Direction, Plan};
    use proptest::prelude::*;

    // ── deterministic PRNG (SplitMix64) ─────────────────────────────────────

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [-1, 1).
    fn rand_signed(state: &mut u64) -> f64 {
        (splitmix64(state) as f64 / u64::MAX as f64) * 2.0 - 1.0
    }

    // ── helpers ─────────────────────────────────────────────────────────────

    /// Plant `k` distinct random complex exponentials.  Returns the time-domain
    /// signal and the ground-truth `(freq, X[freq])` pairs, where `X[freq]` is
    /// the *unnormalized* DFT value (`amplitude · n`).
    fn plant_random_sparse(
        n: usize,
        k: usize,
        seed: u64,
    ) -> (Vec<Complex<f64>>, Vec<(usize, Complex<f64>)>) {
        let mut state = seed;
        let mut freqs: Vec<usize> = Vec::new();
        while freqs.len() < k {
            let f = (splitmix64(&mut state) as usize) % n;
            if !freqs.contains(&f) {
                freqs.push(f);
            }
        }

        let two_pi = core::f64::consts::PI * 2.0;
        let mut signal = vec![Complex::new(0.0_f64, 0.0); n];
        let mut planted = Vec::with_capacity(k);

        for &f in &freqs {
            // Amplitude magnitude in [0.5, 2.5] so every bin is well above the
            // detection floor and distinct.
            let amp = Complex::new(
                0.5 + rand_signed(&mut state).abs() * 2.0,
                rand_signed(&mut state),
            );
            planted.push((f, Complex::new(amp.re * n as f64, amp.im * n as f64)));
            for (t, s) in signal.iter_mut().enumerate() {
                let angle = two_pi * (f as f64) * (t as f64) / (n as f64);
                let w = Complex::new(angle.cos(), angle.sin());
                *s = *s + amp * w;
            }
        }

        (signal, planted)
    }

    /// Sorted copy of an index slice, for order-independent set comparison.
    fn sorted(indices: &[usize]) -> Vec<usize> {
        let mut v = indices.to_vec();
        v.sort_unstable();
        v
    }

    /// Reference top-`k` bins (indices) from a dense FFT.
    fn dense_topk_indices(signal: &[Complex<f64>], k: usize) -> Vec<usize> {
        let n = signal.len();
        let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).expect("dense plan");
        let mut out = vec![Complex::new(0.0_f64, 0.0); n];
        plan.execute(signal, &mut out);
        let mut mags: Vec<(usize, f64)> = out
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.norm_sqr()))
            .collect();
        mags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        let mut idx: Vec<usize> = mags[..k.min(n)].iter().map(|(i, _)| *i).collect();
        idx.sort_unstable();
        idx
    }

    fn assert_matches_planted(
        result: &SparseResult<f64>,
        planted: &[(usize, Complex<f64>)],
        tol: f64,
    ) {
        assert_eq!(
            result.indices.len(),
            planted.len(),
            "expected {} components, got {}",
            planted.len(),
            result.indices.len()
        );
        for &(f, xval) in planted {
            let got = result.get(f);
            let scale = xval.norm() + 1.0;
            assert!(
                (got.re - xval.re).abs() < tol * scale && (got.im - xval.im).abs() < tol * scale,
                "freq {f}: expected {xval:?}, got {got:?}"
            );
        }
    }

    // ── adversarial correctness tests (index+value vs ground truth) ──────────

    /// Random k-sparse signals at several sizes and sparsities must recover the
    /// planted `(index, value)` pairs exactly, and match the dense FFT top-`k`.
    #[test]
    fn test_random_ksparse_exact_recovery() {
        let cases = [
            (256usize, 1usize),
            (256, 2),
            (512, 2),
            (512, 5),
            (1024, 5),
            (1024, 16),
            (2048, 16),
            (4096, 16),
        ];
        for (ci, &(n, k)) in cases.iter().enumerate() {
            for rep in 0..4u64 {
                let seed = 0x1234_5678u64
                    .wrapping_add(ci as u64 * 1013)
                    .wrapping_add(rep * 7919);
                let (signal, planted) = plant_random_sparse(n, k, seed);

                let result = sparse_fft(&signal, k);

                // Value-level correctness against ground truth.
                assert_matches_planted(&result, &planted, 1e-6);

                // Index-set equality against an independent dense FFT top-k.
                let got = sorted(&result.indices);
                let expect = dense_topk_indices(&signal, k);
                assert_eq!(got, expect, "n={n} k={k} seed={seed}: index set mismatch");
            }
        }
    }

    /// Off-grid / permanent-collision case: two bins separated by n/2 collide in
    /// every power-of-two bucketization, so the fast path cannot certify and the
    /// dense fallback must still return the correct top-`k`.
    #[test]
    fn test_permanent_collision_still_correct() {
        let n = 1024;
        let two_pi = core::f64::consts::PI * 2.0;
        // 100 and 612 differ by 512 = n/2 → collide under every power-of-two B.
        let planted_freqs = [
            (100usize, Complex::new(1.0_f64, 0.5)),
            (612, Complex::new(0.8, -0.3)),
            (33, Complex::new(1.5, 0.0)),
        ];
        let mut signal = vec![Complex::new(0.0_f64, 0.0); n];
        let mut planted = Vec::new();
        for &(f, amp) in &planted_freqs {
            planted.push((f, Complex::new(amp.re * n as f64, amp.im * n as f64)));
            for (t, s) in signal.iter_mut().enumerate() {
                let angle = two_pi * (f as f64) * (t as f64) / (n as f64);
                let w = Complex::new(angle.cos(), angle.sin());
                *s = *s + amp * w;
            }
        }

        let result = sparse_fft(&signal, 3);
        assert_matches_planted(&result, &planted, 1e-6);

        let got = sorted(&result.indices);
        assert_eq!(got, dense_topk_indices(&signal, 3));
    }

    /// Noise robustness: with per-bin noise well below the smallest planted bin,
    /// the returned top-`k` must still be the planted bins (matching the dense
    /// FFT top-`k`).  Threshold documented in the module docs.
    #[test]
    fn test_noise_robust_topk() {
        let n = 1024;
        let k = 4;
        let (mut signal, planted) = plant_random_sparse(n, k, 0xC0FF_EE00);

        // Deterministic small noise: magnitude ~0.02 per sample, so per-bin
        // noise ~0.02·√n ≈ 0.64, far below the planted |X[f]| ≥ 0.5·n = 512.
        let mut state = 0xBADD_5EEDu64;
        for s in signal.iter_mut() {
            s.re += 0.02 * rand_signed(&mut state);
            s.im += 0.02 * rand_signed(&mut state);
        }

        let result = sparse_fft(&signal, k);

        // Index set must equal the planted bins and the dense FFT top-k.
        let got = sorted(&result.indices);
        let planted_idx = sorted(&planted.iter().map(|(f, _)| *f).collect::<Vec<_>>());
        assert_eq!(got, planted_idx, "noisy top-k should equal planted bins");
        assert_eq!(got, dense_topk_indices(&signal, k));
    }

    /// `k = 0` must always return an empty result.
    #[test]
    fn test_sparse_fft_k_zero() {
        let input = vec![Complex::new(1.0_f64, 0.0); 256];
        assert!(sparse_fft(&input, 0).is_empty());
    }

    #[test]
    fn test_sparse_fft_empty() {
        let input: Vec<Complex<f64>> = vec![];
        assert!(sparse_fft(&input, 0).is_empty());
    }

    /// `k = 1`: a single clean tone must be recovered exactly.
    #[test]
    fn test_sparse_fft_k_one_exact() {
        let n = 512;
        let freq = 137;
        let two_pi = core::f64::consts::PI * 2.0;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|t| {
                let angle = two_pi * (freq as f64) * (t as f64) / (n as f64);
                Complex::new(angle.cos(), angle.sin())
            })
            .collect();
        let result = sparse_fft(&input, 1);
        assert_eq!(result.indices, vec![freq]);
        assert!((result.values[0].re - n as f64).abs() < 1e-6);
        assert!(result.values[0].im.abs() < 1e-6);
    }

    /// Dense regime (`k >= n/4`) routes through the fallback and returns valid,
    /// correct top-`k` indices.
    #[test]
    fn test_sparse_fft_dense_regime() {
        let n = 128;
        let k = n / 2;
        let input = vec![Complex::new(1.0_f64, 0.0); n];
        let result = sparse_fft(&input, k);
        assert!(result.indices.len() <= k);
        // Constant signal: only bin 0 is non-zero, so it must be present.
        assert!(result.indices.contains(&0));
        for &idx in &result.indices {
            assert!(idx < n);
        }
    }

    /// Sparse inverse followed by forward must round-trip the planted bins.
    #[test]
    fn test_sparse_ifft_roundtrip() {
        let n = 256;
        let indices = vec![5, 20, 50];
        let values = vec![
            Complex::new(n as f64, 0.0),
            Complex::new(0.0, 0.5 * n as f64),
            Complex::new(-(n as f64), 0.3 * n as f64),
        ];
        let sparse_result = SparseResult::new(indices.clone(), values.clone(), n);

        let time_signal = sparse_ifft(&sparse_result, n);
        assert_eq!(time_signal.len(), n);

        let recovered = sparse_fft(&time_signal, 3);
        let got = sorted(&recovered.indices);
        assert_eq!(got, vec![5, 20, 50]);
        for (&f, &v) in indices.iter().zip(values.iter()) {
            let g = recovered.get(f);
            assert!((g.re - v.re).abs() < 1e-6 * (v.norm() + 1.0));
            assert!((g.im - v.im).abs() < 1e-6 * (v.norm() + 1.0));
        }
    }

    // ── adaptive sparsity detection ─────────────────────────────────────────

    #[test]
    fn test_sparse_fft_auto_empty() {
        let input: Vec<Complex<f64>> = vec![];
        assert!(sparse_fft_auto(&input).is_empty());
    }

    #[test]
    fn test_sparse_fft_auto_single_freq() {
        let n = 256;
        let freq = 10;
        let two_pi = core::f64::consts::PI * 2.0;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|t| {
                let angle = two_pi * (freq as f64) * (t as f64) / (n as f64);
                Complex::new(angle.cos(), angle.sin())
            })
            .collect();
        let result = sparse_fft_auto(&input);
        assert!(!result.is_empty());
        let sorted = result.sorted_by_magnitude();
        assert_eq!(sorted[0].0, freq);
    }

    #[test]
    fn test_sparse_fft_auto_custom_ratio() {
        let n = 256;
        let two_pi = core::f64::consts::PI * 2.0;
        let planted = [(10usize, 10.0_f64), (50, 1.0), (100, 0.5)];
        let mut input = vec![Complex::new(0.0_f64, 0.0); n];
        for &(freq, amp) in &planted {
            for (t, s) in input.iter_mut().enumerate() {
                let angle = two_pi * (freq as f64) * (t as f64) / (n as f64);
                s.re += amp * angle.cos();
                s.im += amp * angle.sin();
            }
        }
        let tight = sparse_fft_auto_with_ratio(&input, 0.90);
        let loose = sparse_fft_auto_with_ratio(&input, 0.99);
        assert!(tight.len() <= loose.len());
    }

    // ── property tests ──────────────────────────────────────────────────────

    proptest! {
        /// Random k-sparse signals: sparse_fft must reproduce the dense FFT
        /// top-`k` index set exactly.
        #[test]
        fn prop_sparse_matches_dense_topk(
            n_log2 in 8usize..=11usize,   // n in {256 .. 2048}
            k in 1usize..=8usize,
            seed in 0u64..=100_000u64,
        ) {
            let n = 1usize << n_log2;
            let k = k.min(n / 8).max(1);
            let (signal, _planted) = plant_random_sparse(n, k, seed);

            let result = sparse_fft(&signal, k);

            let got = sorted(&result.indices);
            let expect = dense_topk_indices(&signal, k);
            prop_assert_eq!(got, expect);
        }

        /// `k = 0` always yields an empty result.
        #[test]
        fn prop_sparse_fft_k_zero_always_empty(n_log2 in 4usize..=10usize) {
            let n = 1usize << n_log2;
            let signal = vec![Complex::new(1.0_f64, 0.0); n];
            prop_assert!(sparse_fft(&signal, 0).is_empty());
        }

        /// Result indices are always unique and in range.
        #[test]
        fn prop_sparse_fft_unique_in_range(
            n_log2 in 8usize..=10usize,
            k in 1usize..=6usize,
            seed in 0u64..=50_000u64,
        ) {
            let n = 1usize << n_log2;
            let k = k.min(n / 8).max(1);
            let (signal, _) = plant_random_sparse(n, k, seed);
            let result = sparse_fft(&signal, k);

            let mut seen = std::collections::BTreeSet::new();
            for &idx in &result.indices {
                prop_assert!(idx < n);
                prop_assert!(seen.insert(idx), "duplicate index {}", idx);
            }
        }
    }
}
