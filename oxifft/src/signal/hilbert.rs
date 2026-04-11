//! Hilbert transform and analytic signal computation.
//!
//! The Hilbert transform produces the analytic signal from a real signal.
//! The analytic signal has no negative-frequency components, enabling
//! instantaneous amplitude, phase, and frequency extraction.
//!
//! # Algorithm
//!
//! 1. Compute FFT of the real signal
//! 2. Zero out negative frequencies (n/2+1 .. n-1)
//! 3. Double positive frequencies (1 .. n/2-1)
//! 4. Keep DC (index 0) and Nyquist (index n/2) unchanged
//! 5. Compute IFFT and normalize

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};

/// Compute the analytic signal via the Hilbert transform.
///
/// Returns a complex-valued vector of the same length as `signal`.
/// The real part of the analytic signal equals the original input (up to
/// numerical precision), and the imaginary part is its Hilbert transform.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "signal")]
/// # {
/// use oxifft::signal::hilbert;
/// let n = 64;
/// let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
/// let analytic = hilbert(&signal);
/// assert_eq!(analytic.len(), n);
/// # }
/// ```
pub fn hilbert<T: Float>(signal: &[T]) -> Vec<Complex<T>> {
    if signal.is_empty() {
        return Vec::new();
    }

    let n = signal.len();

    // Convert real signal to complex.
    let a_complex: Vec<Complex<T>> = signal.iter().map(|&s| Complex::new(s, T::ZERO)).collect();

    // Forward FFT.
    let fwd_plan = match Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut spectrum = vec![Complex::<T>::zero(); n];
    fwd_plan.execute(&a_complex, &mut spectrum);

    // Apply Hilbert multiplier in-place.
    //
    // For even n:
    //   index 0        (DC):      keep
    //   indices 1..n/2-1:         multiply by 2
    //   index n/2      (Nyquist): keep
    //   indices n/2+1..n-1:       zero out
    //
    // For odd n:
    //   index 0:                  keep
    //   indices 1..(n+1)/2-1:     multiply by 2
    //   indices (n+1)/2..n-1:     zero out
    let two = T::TWO;

    if n.is_multiple_of(2) {
        let half = n / 2;
        // Positive frequencies (excluding DC and Nyquist): double.
        for s in spectrum.iter_mut().take(half).skip(1) {
            *s = Complex::new(s.re * two, s.im * two);
        }
        // Negative frequencies: zero out.
        for s in spectrum.iter_mut().skip(half + 1) {
            *s = Complex::zero();
        }
        // DC (0) and Nyquist (half) are left unchanged.
    } else {
        let pos_end = (n + 1) / 2; // exclusive upper bound for positive freqs
                                   // Positive frequencies (excluding DC): double.
        for s in spectrum.iter_mut().take(pos_end).skip(1) {
            *s = Complex::new(s.re * two, s.im * two);
        }
        // Negative frequencies: zero out.
        for s in spectrum.iter_mut().skip(pos_end) {
            *s = Complex::zero();
        }
    }

    // Inverse FFT.
    let inv_plan = match Plan::dft_1d(n, Direction::Backward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut analytic = vec![Complex::<T>::zero(); n];
    inv_plan.execute(&spectrum, &mut analytic);

    // Normalize by 1/n.
    let scale = T::ONE / T::from_usize(n);
    for a in &mut analytic {
        *a = Complex::new(a.re * scale, a.im * scale);
    }

    analytic
}

/// Compute the envelope (instantaneous amplitude) of a real signal.
///
/// Returns `|analytic_signal[i]|` for each sample, where `analytic_signal`
/// is the result of `hilbert(signal)`.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "signal")]
/// # {
/// use oxifft::signal::envelope;
/// let n = 512;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / n as f64).sin())
///     .collect();
/// let env = envelope(&signal);
/// assert_eq!(env.len(), n);
/// # }
/// ```
pub fn envelope<T: Float>(signal: &[T]) -> Vec<T> {
    hilbert(signal)
        .iter()
        .map(|c| {
            let re = c.re;
            let im = c.im;
            Float::sqrt(re * re + im * im)
        })
        .collect()
}

/// Compute the instantaneous phase of a real signal.
///
/// Returns `atan2(im, re)` for each sample of the analytic signal,
/// yielding values in the range `[-π, π]`.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "signal")]
/// # {
/// use oxifft::signal::instantaneous_phase;
/// let n = 256;
/// let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
/// let phase = instantaneous_phase(&signal);
/// assert_eq!(phase.len(), n);
/// # }
/// ```
pub fn instantaneous_phase<T: Float>(signal: &[T]) -> Vec<T> {
    hilbert(signal)
        .iter()
        .map(|c| num_traits::Float::atan2(c.im, c.re))
        .collect()
}

/// Compute the instantaneous frequency of a real signal.
///
/// Returns the phase-difference sequence (with phase unwrapping), normalized
/// to cycles per sample.  The output length is `signal.len() - 1`, or `0`
/// when the signal has fewer than 2 samples.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "signal")]
/// # {
/// use oxifft::signal::instantaneous_frequency;
/// let n = 128;
/// let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
/// let freq = instantaneous_frequency(&signal);
/// assert_eq!(freq.len(), n - 1);
/// # }
/// ```
pub fn instantaneous_frequency<T: Float>(signal: &[T]) -> Vec<T> {
    if signal.len() < 2 {
        return Vec::new();
    }

    let phase = instantaneous_phase(signal);

    // π and 2π as T.
    let pi = <T as Float>::PI;
    let two_pi = <T as Float>::TWO_PI;

    phase
        .windows(2)
        .map(|w| {
            let mut diff = w[1] - w[0];
            // Wrap diff into [-π, π].
            while diff > pi {
                diff = diff - two_pi;
            }
            while diff < -pi {
                diff = diff + two_pi;
            }
            // Convert from radians/sample to cycles/sample.
            diff / two_pi
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hilbert_real_part_matches_input() {
        // Real part of analytic signal equals original
        let n = 64;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let analytic = hilbert(&signal);
        assert_eq!(analytic.len(), n);
        for (i, (&orig, a)) in signal.iter().zip(analytic.iter()).enumerate() {
            assert!(
                (a.re - orig).abs() < 1e-10,
                "Real part mismatch at {i}: {} vs {}",
                a.re,
                orig
            );
        }
    }

    #[test]
    fn test_envelope_sine_is_constant() {
        // Envelope of a pure sine should be approximately 1.0 (ignoring edge effects)
        let n = 512;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / n as f64).sin())
            .collect();
        let env = envelope(&signal);
        // Check middle portion (avoid edge effects)
        for i in n / 4..3 * n / 4 {
            assert!((env[i] - 1.0).abs() < 0.02, "Envelope at {i}: {}", env[i]);
        }
    }

    #[test]
    fn test_hilbert_empty() {
        let empty: Vec<f64> = Vec::new();
        assert!(hilbert(&empty).is_empty());
        assert!(envelope(&empty).is_empty());
    }

    #[test]
    fn test_instantaneous_phase_sine() {
        let n = 256;
        let freq = 5.0; // cycles per n samples
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / n as f64).sin())
            .collect();
        let phase = instantaneous_phase(&signal);
        assert_eq!(phase.len(), n);
    }

    #[test]
    fn test_instantaneous_frequency_length() {
        let n = 128;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
        let freq = instantaneous_frequency(&signal);
        // Length should be n-1
        assert_eq!(freq.len(), n - 1);
    }
}
