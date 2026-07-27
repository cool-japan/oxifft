//! Certified peeling decoder for the aliasing-based sparse FFT.
//!
//! The decoder consumes a set of *aliasing stages*, each of which bucketizes
//! the input spectrum by a distinct downsampling factor, and recovers the
//! individual non-zero frequency components via phase-based singleton
//! detection plus iterative peeling.
//!
//! # Why two offset measurements per stage
//!
//! Each stage is built from two downsampled copies of the signal: one taken at
//! offset 0 (`z0[j] = x[j·L]`) and one at offset 1 (`z1[j] = x[j·L + 1]`),
//! where `L = n / B` and `B` is the stage's bucket count.  Their length-`B`
//! DFTs alias every input frequency `f` into bucket `b = f mod B`:
//!
//! ```text
//! Z0[b] = (1/L) · Σ_{f ≡ b (mod B)}  X[f]
//! Z1[b] = (1/L) · Σ_{f ≡ b (mod B)}  X[f] · exp(2πi f / n)
//! ```
//!
//! For a *singleton* bucket (exactly one non-zero `X[f]` aliases into it) the
//! ratio `Z1[b] / Z0[b] = exp(2πi f / n)` is a pure phase, from which `f` is
//! recovered exactly, and `X[f] = L · Z0[b]`.  Working in *coefficient space*
//! (`L · Z0`, `L · Z1`) removes the per-stage `L` factor so that a recovered
//! frequency can be subtracted uniformly from every stage.
//!
//! # Soundness of certification
//!
//! After peeling, the decoder checks that the residual in **both** the
//! offset-0 and offset-1 measurements of every bucket has collapsed below a
//! relative tolerance.  Two distinct un-recovered frequencies cannot cancel in
//! both measurements simultaneously (that would require identical phase, i.e.
//! identical frequency), so a passing certification proves the recovered set
//! explains essentially all signal energy — its top-`k` is therefore the true
//! top-`k`.  When certification fails the caller must fall back to a dense FFT,
//! guaranteeing that no silently-wrong result is ever returned.

use crate::kernel::{Complex, Float};
use crate::prelude::*;

/// One aliasing stage, stored as bucket residuals in *coefficient space*.
///
/// `coeff0[b]` holds `L · Z0[b]` and `coeff1[b]` holds `L · Z1[b]`.  In
/// coefficient space a singleton bucket containing exactly one true frequency
/// `f` satisfies `coeff0[b] = X[f]` and `coeff1[b] = X[f] · exp(2πi f / n)`,
/// independent of the stage's `L`.  Both arrays are mutated in place as
/// frequencies are peeled off.
pub struct AliasStage<T: Float> {
    /// Number of buckets in this stage (a divisor of `n`).
    pub b_count: usize,
    /// Residual of `L · Z0`, updated in place during peeling.
    pub coeff0: Vec<Complex<T>>,
    /// Residual of `L · Z1`, updated in place during peeling.
    pub coeff1: Vec<Complex<T>>,
}

/// Magnitude of a complex value as an `f64` (used for tolerance comparisons).
#[inline]
fn abs_f64<T: Float>(c: Complex<T>) -> f64 {
    let re = c.re.to_f64().unwrap_or(0.0);
    let im = c.im.to_f64().unwrap_or(0.0);
    libm::sqrt(re * re + im * im)
}

/// Subtract a recovered frequency `f` with coefficient `coeff = X[f]` from
/// every stage's residual, in both offset measurements.
fn subtract_frequency<T: Float>(
    stages: &mut [AliasStage<T>],
    n: usize,
    f: usize,
    coeff: Complex<T>,
) {
    // Twiddle exp(+2πi f / n) applied to the offset-1 measurement.
    let theta = <T as Float>::TWO_PI * T::from_usize(f) / T::from_usize(n);
    let (sin_t, cos_t) = Float::sin_cos(theta);
    let w = Complex::new(cos_t, sin_t);
    let coeff_w = coeff * w;

    for stage in stages.iter_mut() {
        let b = f % stage.b_count;
        stage.coeff0[b] = stage.coeff0[b] - coeff;
        stage.coeff1[b] = stage.coeff1[b] - coeff_w;
    }
}

/// Run certified peeling over the aliasing stages.
///
/// Returns `Some(recovered)` — a list of `(frequency, X[frequency])` pairs —
/// only when the recovered set is *certified* to explain the whole signal
/// (all residuals below a relative tolerance).  Returns `None` when the fast
/// path cannot certify its solution, signalling the caller to fall back to a
/// dense FFT.  A completely silent signal certifies trivially as an empty set.
///
/// # Arguments
///
/// * `stages`        - Aliasing stages in coefficient space (mutated in place).
/// * `n`             - Original signal length.
/// * `max_recover`   - Hard ceiling on the number of frequencies to peel.
/// * `abs_threshold` - Absolute magnitude floor; energy below this is treated
///   as noise and ignored (maps the public `SparsePlan` threshold).
pub fn certified_peel<T: Float>(
    stages: &mut [AliasStage<T>],
    n: usize,
    max_recover: usize,
    abs_threshold: f64,
) -> Option<Vec<(usize, Complex<T>)>> {
    if stages.is_empty() || n == 0 {
        return None;
    }

    // Peak coefficient magnitude across all stages, for relative tolerances.
    let mut peak = 0.0_f64;
    for stage in stages.iter() {
        for c in stage.coeff0.iter() {
            let m = abs_f64(*c);
            if m > peak {
                peak = m;
            }
        }
    }

    // Silent (or entirely sub-threshold) signal: nothing to recover, and the
    // empty set trivially explains it.
    if peak <= abs_threshold {
        return Some(Vec::new());
    }

    // Relative tolerances derived from the signal peak.
    let occ_floor = (1e-9 * peak).max(abs_threshold);
    let phase_tol = 1e-6_f64;
    let cert_tol = 1e-6 * peak;
    let two_pi = core::f64::consts::PI * 2.0;

    let mut recovered: Vec<(usize, Complex<T>)> = Vec::new();

    // Each outer pass scans for a single certified singleton, removes it, and
    // restarts (so that newly-isolated buckets become visible).  A pass that
    // finds nothing terminates the loop.  The iteration ceiling is a safety net
    // — in practice the loop stops after `recovered.len()` successful passes.
    let iter_cap = max_recover.saturating_mul(4).saturating_add(8);
    let mut iterations = 0usize;

    'peel: loop {
        iterations += 1;
        if iterations > iter_cap || recovered.len() >= max_recover {
            break;
        }

        for si in 0..stages.len() {
            let b_count = stages[si].b_count;
            for b in 0..b_count {
                let c0 = stages[si].coeff0[b];
                let m0 = abs_f64(c0);
                if m0 <= occ_floor {
                    continue;
                }

                let c1 = stages[si].coeff1[b];

                // Estimate the candidate frequency from the phase advance
                // between the two offset measurements.
                let a0 = libm::atan2(c0.im.to_f64().unwrap_or(0.0), c0.re.to_f64().unwrap_or(0.0));
                let a1 = libm::atan2(c1.im.to_f64().unwrap_or(0.0), c1.re.to_f64().unwrap_or(0.0));
                let dtheta = (a1 - a0).rem_euclid(two_pi);
                let f_est = (libm::round(dtheta * (n as f64) / two_pi) as usize) % n;

                // Singleton validation:
                //   (a) the recovered frequency must alias back to this bucket,
                //   (b) the two measurements must have equal magnitude (a pure
                //       phase rotation — otherwise the bucket is a collision).
                if f_est % b_count != b {
                    continue;
                }
                let m1 = abs_f64(c1);
                if (m1 - m0).abs() > phase_tol * m0 {
                    continue;
                }
                if recovered.iter().any(|(ff, _)| *ff == f_est) {
                    continue;
                }

                // Accept: the residual at a singleton bucket *is* X[f_est].
                subtract_frequency(stages, n, f_est, c0);
                recovered.push((f_est, c0));
                continue 'peel;
            }
        }

        // A full pass found no new singleton.
        break;
    }

    // Certification: every residual bucket must be explained in both offset
    // measurements.  A single un-explained bucket forces the dense fallback.
    let mut residual = 0.0_f64;
    for stage in stages.iter() {
        for c in stage.coeff0.iter().chain(stage.coeff1.iter()) {
            let m = abs_f64(*c);
            if m > residual {
                residual = m;
            }
        }
    }

    if residual <= cert_tol {
        Some(recovered)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a single aliasing stage from a dense spectrum, in coefficient
    /// space, for testing the decoder in isolation.
    fn stage_from_spectrum(spectrum: &[Complex<f64>], b_count: usize) -> AliasStage<f64> {
        let n = spectrum.len();
        let two_pi = core::f64::consts::PI * 2.0;
        let mut coeff0 = vec![Complex::new(0.0, 0.0); b_count];
        let mut coeff1 = vec![Complex::new(0.0, 0.0); b_count];
        for (f, &x) in spectrum.iter().enumerate() {
            let b = f % b_count;
            let theta = two_pi * (f as f64) / (n as f64);
            let w = Complex::new(theta.cos(), theta.sin());
            coeff0[b] = coeff0[b] + x;
            coeff1[b] = coeff1[b] + x * w;
        }
        AliasStage {
            b_count,
            coeff0,
            coeff1,
        }
    }

    #[test]
    fn test_single_tone_recovered_exactly() {
        let n = 256;
        let mut spectrum = vec![Complex::new(0.0, 0.0); n];
        spectrum[37] = Complex::new(3.0, -1.5);

        let mut stages = vec![stage_from_spectrum(&spectrum, 16)];
        let recovered =
            certified_peel(&mut stages, n, n, 1e-10).expect("clean single tone must certify");

        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].0, 37);
        assert!((recovered[0].1.re - 3.0).abs() < 1e-9);
        assert!((recovered[0].1.im - (-1.5)).abs() < 1e-9);
    }

    #[test]
    fn test_multiple_isolated_tones() {
        let n = 512;
        let planted = [
            (11usize, Complex::new(1.0, 0.0)),
            (200, Complex::new(-2.0, 0.5)),
            (301, Complex::new(0.25, 0.75)),
        ];
        let mut spectrum = vec![Complex::new(0.0, 0.0); n];
        for &(f, v) in &planted {
            spectrum[f] = v;
        }

        // Two stages with distinct bucket counts help isolation.
        let mut stages = vec![
            stage_from_spectrum(&spectrum, 32),
            stage_from_spectrum(&spectrum, 64),
        ];
        let recovered =
            certified_peel(&mut stages, n, n, 1e-10).expect("isolated tones must certify");

        assert_eq!(recovered.len(), 3);
        for &(f, v) in &planted {
            let found = recovered
                .iter()
                .find(|(rf, _)| *rf == f)
                .expect("planted frequency must be recovered");
            assert!((found.1.re - v.re).abs() < 1e-9);
            assert!((found.1.im - v.im).abs() < 1e-9);
        }
    }

    #[test]
    fn test_permanent_collision_fails_certification() {
        // Two frequencies that differ by n/2 collide in every power-of-two
        // bucketization smaller than n, so a single power-of-two stage cannot
        // separate them — certification must fail (never a wrong answer).
        let n = 256;
        let mut spectrum = vec![Complex::new(0.0, 0.0); n];
        spectrum[10] = Complex::new(1.0, 0.0);
        spectrum[10 + 128] = Complex::new(0.5, 0.3);

        let mut stages = vec![
            stage_from_spectrum(&spectrum, 16),
            stage_from_spectrum(&spectrum, 32),
        ];
        // Both stages are power-of-two, and 10 ≡ 138 (mod 16) and (mod 32):
        // the pair never separates, so no certified solution exists.
        let result = certified_peel(&mut stages, n, n, 1e-10);
        assert!(result.is_none(), "an unresolved collision must not certify");
    }

    #[test]
    fn test_silent_signal_certifies_empty() {
        let n = 64;
        let spectrum = vec![Complex::new(0.0, 0.0); n];
        let mut stages = vec![stage_from_spectrum(&spectrum, 16)];
        let recovered =
            certified_peel(&mut stages, n, n, 1e-10).expect("silent signal certifies as empty");
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_empty_stages_returns_none() {
        let mut stages: Vec<AliasStage<f64>> = Vec::new();
        assert!(certified_peel(&mut stages, 64, 64, 1e-10).is_none());
    }

    #[test]
    fn test_sub_threshold_signal_ignored() {
        // A tone whose magnitude is below the absolute threshold is treated as
        // noise and yields a certified-empty result.
        let n = 64;
        let mut spectrum = vec![Complex::new(0.0, 0.0); n];
        spectrum[5] = Complex::new(1e-12, 0.0);
        let mut stages = vec![stage_from_spectrum(&spectrum, 16)];
        let recovered =
            certified_peel(&mut stages, n, n, 1e-10).expect("sub-threshold certifies empty");
        assert!(recovered.is_empty());
    }
}
