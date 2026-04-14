//! Cepstral analysis for OxiFFT.
//!
//! The cepstrum is the result of taking the inverse Fourier transform of
//! the logarithm of the spectrum. It deconvolves signals that were combined
//! by convolution (e.g., speech source + vocal tract filter).
//!
//! # Cepstrum Types
//!
//! - **Real cepstrum**: `IFFT(log(|FFT(x)|))` — symmetric, real-valued
//! - **Complex cepstrum**: `IFFT(log(FFT(x)))` — with phase unwrapping
//! - **Minimum phase**: Reconstruct minimum-phase system from real cepstrum

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};

extern crate alloc;
use alloc::vec::Vec;

/// Unwrap a phase sequence to be continuous (no jumps larger than π).
///
/// Standard incremental phase unwrapping: for each step, the difference
/// is wrapped back into `[-π, π]` before accumulation.
pub fn unwrap_phase<T: Float>(phases: &[T]) -> Vec<T> {
    let n = phases.len();
    if n == 0 {
        return Vec::new();
    }

    let pi = <T as Float>::PI;
    let two_pi = <T as Float>::TWO_PI;

    let mut unwrapped = Vec::with_capacity(n);
    unwrapped.push(phases[0]);

    for k in 1..n {
        let mut diff = phases[k] - phases[k - 1];

        // Wrap diff into [-π, π]
        while diff > pi {
            diff = diff - two_pi;
        }
        while diff < -pi {
            diff = diff + two_pi;
        }

        let prev = unwrapped[k - 1];
        unwrapped.push(prev + diff);
    }

    unwrapped
}

/// Compute the real cepstrum of a signal.
///
/// The real cepstrum is defined as `IFFT(log(|FFT(x)|))`. Because only
/// the log-magnitude is used (phase is discarded), the result is a real,
/// even (symmetric) sequence. It is useful for pitch detection and
/// spectral envelope estimation.
///
/// # Arguments
/// * `signal` - Input real-valued signal
///
/// # Returns
/// Real cepstrum of the same length as the input. Returns an empty
/// `Vec` if the input is empty or if plan creation fails.
pub fn real_cepstrum<T: Float>(signal: &[T]) -> Vec<T> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }

    // Build complex input from real signal
    let spectrum: Vec<Complex<T>> = signal.iter().map(|&s| Complex::new(s, T::ZERO)).collect();

    let mut fft_out = vec![Complex::<T>::zero(); n];

    // Forward FFT
    let fft_plan = match Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };
    fft_plan.execute(&spectrum, &mut fft_out);

    // Compute log-magnitude spectrum (zero phase)
    let eps = T::from_f64(1e-30);
    let floor = T::from_f64(-30.0);

    let log_spectrum: Vec<Complex<T>> = fft_out
        .iter()
        .map(|c| {
            let mag = c.norm();
            let log_mag = if mag > eps {
                num_traits::Float::ln(mag)
            } else {
                floor
            };
            Complex::new(log_mag, T::ZERO)
        })
        .collect();

    // IFFT of log-magnitude spectrum
    let ifft_plan = match Plan::dft_1d(n, Direction::Backward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut cepstrum_out = vec![Complex::<T>::zero(); n];
    ifft_plan.execute(&log_spectrum, &mut cepstrum_out);

    // Normalize and extract real parts
    let scale = T::ONE / T::from_usize(n);
    cepstrum_out.iter().map(|c| c.re * scale).collect()
}

/// Compute the complex cepstrum of a signal.
///
/// The complex cepstrum is defined as `IFFT(log(FFT(x)))` where `log`
/// is the complex logarithm with phase unwrapping applied to ensure
/// continuity. Unlike the real cepstrum, it preserves phase information
/// and can be used to separate minimum-phase and maximum-phase components.
///
/// # Arguments
/// * `signal` - Input real-valued signal
///
/// # Returns
/// Complex cepstrum (real parts of the IFFT result) of the same length
/// as the input. Returns an empty `Vec` if the input is empty or if
/// plan creation fails.
pub fn complex_cepstrum<T: Float>(signal: &[T]) -> Vec<T> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }

    // Build complex input from real signal
    let input: Vec<Complex<T>> = signal.iter().map(|&s| Complex::new(s, T::ZERO)).collect();

    let mut fft_out = vec![Complex::<T>::zero(); n];

    // Forward FFT
    let fft_plan = match Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };
    fft_plan.execute(&input, &mut fft_out);

    // Compute complex log: log(z) = ln|z| + i*arg(z)
    let eps = T::from_f64(1e-30);
    let floor = T::from_f64(-30.0);

    // Build log-spectrum with unwrapped phase
    let mut log_spectrum: Vec<Complex<T>> = fft_out
        .iter()
        .map(|c| {
            let mag = c.norm();
            let log_mag = if mag > eps {
                num_traits::Float::ln(mag)
            } else {
                floor
            };
            // Raw (wrapped) phase
            let phase = c.arg();
            Complex::new(log_mag, phase)
        })
        .collect();

    // Extract phases, unwrap them, and put them back
    let raw_phases: Vec<T> = log_spectrum.iter().map(|c| c.im).collect();
    let unwrapped = unwrap_phase(&raw_phases);
    for (c, &ph) in log_spectrum.iter_mut().zip(unwrapped.iter()) {
        c.im = ph;
    }

    // IFFT of complex log-spectrum
    let ifft_plan = match Plan::dft_1d(n, Direction::Backward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut cepstrum_out = vec![Complex::<T>::zero(); n];
    ifft_plan.execute(&log_spectrum, &mut cepstrum_out);

    // Normalize and extract real parts
    let scale = T::ONE / T::from_usize(n);
    cepstrum_out.iter().map(|c| c.re * scale).collect()
}

/// Reconstruct a minimum-phase signal from a real-valued input.
///
/// The minimum-phase reconstruction works by:
/// 1. Computing the real cepstrum of the input
/// 2. Applying a causal "lifter" window to select only the causal part
/// 3. Exponentiating the result back into the time domain
///
/// The liftering rule for a signal of length `n` is:
/// - Index 0: keep as-is
/// - Indices `1..n/2`: multiply by 2
/// - Index `n/2` (only when `n` is even): keep as-is
/// - Indices `n/2+1..n`: set to zero
///
/// # Arguments
/// * `signal` - Input real-valued signal
///
/// # Returns
/// Minimum-phase version of the signal, same length as input. Returns
/// an empty `Vec` if the input is empty or if plan creation fails.
pub fn minimum_phase<T: Float>(signal: &[T]) -> Vec<T> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }

    // Step 1: compute real cepstrum
    let rcep = real_cepstrum(signal);
    if rcep.is_empty() {
        return Vec::new();
    }

    // Step 2: apply causal liftering window
    // For even n: keep [0], double [1..n/2), keep [n/2], zero [n/2+1..n]
    // For odd  n: keep [0], double [1..(n+1)/2), zero [(n+1)/2..n]
    let mut liftered: Vec<Complex<T>> = vec![Complex::zero(); n];
    liftered[0] = Complex::new(rcep[0], T::ZERO);

    let half = n / 2;
    let two = T::TWO;

    if n.is_multiple_of(2) {
        // Even length
        for k in 1..half {
            liftered[k] = Complex::new(rcep[k] * two, T::ZERO);
        }
        liftered[half] = Complex::new(rcep[half], T::ZERO);
        // indices half+1..n remain zero
    } else {
        // Odd length
        let upper = n.div_ceil(2);
        for k in 1..upper {
            liftered[k] = Complex::new(rcep[k] * two, T::ZERO);
        }
        // indices upper..n remain zero
    }

    // Step 3: FFT of liftered cepstrum → complex log-spectrum
    let fft_plan = match Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut log_spectrum = vec![Complex::<T>::zero(); n];
    fft_plan.execute(&liftered, &mut log_spectrum);

    // Step 4: exponentiate element-wise: exp(a + ib) = exp(a)*(cos(b) + i*sin(b))
    for c in &mut log_spectrum {
        let amp = num_traits::Float::exp(c.re);
        let (sin_b, cos_b) = Float::sin_cos(c.im);
        c.re = amp * cos_b;
        c.im = amp * sin_b;
    }

    // Step 5: IFFT and normalize
    let ifft_plan = match Plan::dft_1d(n, Direction::Backward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut result = vec![Complex::<T>::zero(); n];
    ifft_plan.execute(&log_spectrum, &mut result);

    let scale = T::ONE / T::from_usize(n);
    result.iter().map(|c| c.re * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_cepstrum_empty() {
        let c: Vec<f64> = real_cepstrum(&[]);
        assert!(c.is_empty());
    }

    #[test]
    fn test_real_cepstrum_length() {
        let signal: Vec<f64> = (0..64).map(|i| (f64::from(i) * 0.5).sin()).collect();
        let c = real_cepstrum(&signal);
        assert_eq!(c.len(), signal.len());
    }

    #[test]
    fn test_real_cepstrum_symmetry() {
        // Real cepstrum of a symmetric signal should be symmetric
        let n = 64;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin() + 0.5).collect();
        let c = real_cepstrum(&signal);
        // The real cepstrum is an even function: c[k] ≈ c[n-k]
        for k in 1..n / 4 {
            let diff = (c[k] - c[n - k]).abs();
            assert!(diff < 1e-8, "Asymmetry at k={k}: {} vs {}", c[k], c[n - k]);
        }
    }

    #[test]
    fn test_complex_cepstrum_length() {
        let signal: Vec<f64> = (0..128).map(|i| (f64::from(i) * 0.2).sin() + 0.1).collect();
        let c = complex_cepstrum(&signal);
        assert_eq!(c.len(), 128);
    }

    #[test]
    fn test_minimum_phase_energy() {
        // Minimum phase signal should have same spectral energy as original
        let n = 64;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin() + 0.5).collect();
        let mp = minimum_phase(&signal);
        assert_eq!(mp.len(), n);
        // Both should have similar total energy
        let orig_energy: f64 = signal.iter().map(|&x| x * x).sum();
        let mp_energy: f64 = mp.iter().map(|&x| x * x).sum();
        let ratio = mp_energy / orig_energy;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "Energy ratio {ratio} out of expected range"
        );
    }

    #[test]
    fn test_unwrap_phase_simple() {
        // Test that phase jumping across pi is unwrapped
        use std::f64::consts::PI;
        let phases = vec![0.0f64, PI * 0.9, -PI * 0.9, -PI * 0.7];
        let unwrapped = unwrap_phase(&phases);
        assert_eq!(unwrapped.len(), 4);
        // Phase should increase monotonically for this case
        for i in 1..unwrapped.len() {
            let diff = (unwrapped[i] - unwrapped[i - 1]).abs();
            assert!(diff < PI, "Jump of {diff} > pi between {} and {}", i - 1, i);
        }
    }
}
