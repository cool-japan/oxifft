//! Power spectral density estimation.
//!
//! Implements periodogram and Welch's method for PSD estimation,
//! cross-spectral density, and magnitude-squared coherence.

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};

extern crate alloc;
use alloc::vec::Vec;

/// Window function for spectral analysis.
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpectralWindow {
    /// Rectangular (no windowing)
    Rectangular,
    /// Hann window
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
}

/// Configuration for Welch's method.
#[derive(Clone, Debug)]
pub struct WelchConfig {
    /// Segment length (FFT size)
    pub segment_len: usize,
    /// Overlap between segments (in samples); must be less than `segment_len`.
    pub overlap: usize,
    /// Window function applied to each segment.
    pub window: SpectralWindow,
}

/// Generate window coefficients for the given window type and length.
///
/// For `n == 1` the function always returns `[1.0]` regardless of window type.
/// All trigonometric constants are computed via the `Float` trait to support
/// both `f32` and `f64`.
fn generate_window<T: Float>(window: SpectralWindow, n: usize) -> Vec<T> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![T::ONE];
    }

    let n_minus_1 = T::from_usize(n - 1);
    let two_pi: T = T::TWO_PI;

    match window {
        SpectralWindow::Rectangular => vec![T::ONE; n],
        SpectralWindow::Hann => (0..n)
            .map(|k| {
                let k_t = T::from_usize(k);
                let half: T = T::from_f64(0.5);
                half * (T::ONE - Float::cos(two_pi * k_t / n_minus_1))
            })
            .collect(),
        SpectralWindow::Hamming => (0..n)
            .map(|k| {
                let k_t = T::from_usize(k);
                let a0: T = T::from_f64(0.54);
                let a1: T = T::from_f64(0.46);
                a0 - a1 * Float::cos(two_pi * k_t / n_minus_1)
            })
            .collect(),
        SpectralWindow::Blackman => (0..n)
            .map(|k| {
                let k_t = T::from_usize(k);
                let a0: T = T::from_f64(0.42);
                let a1: T = T::from_f64(0.5);
                let a2: T = T::from_f64(0.08);
                let phase = two_pi * k_t / n_minus_1;
                a0 - a1 * Float::cos(phase) + a2 * Float::cos(T::TWO * phase)
            })
            .collect(),
    }
}

/// Compute the periodogram power spectral density of a signal.
///
/// Uses a Hann window internally.  Returns a one-sided spectrum of length
/// `signal.len() / 2 + 1`.  Returns an empty `Vec` when `signal` is empty.
///
/// The normalization convention is:
/// ```text
/// psd[k] = |X[k]|² / (w_sum_sq * n)
/// ```
/// where `w_sum_sq` is the sum of squared window coefficients and `n` is the
/// signal length.
pub fn periodogram<T: Float>(signal: &[T]) -> Vec<T> {
    if signal.is_empty() {
        return Vec::new();
    }

    let n = signal.len();
    let window = generate_window::<T>(SpectralWindow::Hann, n);

    // Window power (sum of squared coefficients) for normalization.
    let w_sum_sq: T = window.iter().fold(T::ZERO, |acc, &w| acc + w * w);

    // Build complex windowed signal.
    let windowed: Vec<Complex<T>> = (0..n)
        .map(|i| Complex::new(signal[i] * window[i], T::ZERO))
        .collect();

    // Create FFT plan; on failure return empty spectrum.
    let plan = match Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut spectrum = vec![Complex::<T>::zero(); n];
    plan.execute(&windowed, &mut spectrum);

    let n_t = T::from_usize(n);
    let denom = w_sum_sq * n_t;

    // One-sided PSD: length = n/2 + 1.
    (0..=n / 2)
        .map(|k| spectrum[k].norm_sqr() / denom)
        .collect()
}

/// Compute power spectral density using Welch's method.
///
/// Divides the signal into overlapping segments, applies a window to each,
/// computes the FFT, and averages the squared magnitudes.
///
/// * `config.segment_len` must be > 0.
/// * `config.overlap` must be < `config.segment_len`.
///
/// If the signal is shorter than one segment, falls back to [`periodogram`].
/// Returns a one-sided PSD of length `segment_len / 2 + 1`.
pub fn welch<T: Float>(signal: &[T], config: &WelchConfig) -> Vec<T> {
    let segment_len = config.segment_len;
    let overlap = config.overlap;

    if segment_len == 0 || overlap >= segment_len {
        return Vec::new();
    }

    if signal.len() < segment_len {
        return periodogram(signal);
    }

    let hop_size = segment_len - overlap;
    let window = generate_window::<T>(config.window, segment_len);
    let w_sum_sq: T = window.iter().fold(T::ZERO, |acc, &w| acc + w * w);

    let plan = match Plan::dft_1d(segment_len, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let num_freq = segment_len / 2 + 1;
    let mut psd_accum = vec![T::ZERO; num_freq];
    let mut num_segments: usize = 0;

    let mut start = 0usize;
    while start + segment_len <= signal.len() {
        let segment = &signal[start..start + segment_len];
        let windowed: Vec<Complex<T>> = (0..segment_len)
            .map(|i| Complex::new(segment[i] * window[i], T::ZERO))
            .collect();

        let mut spectrum = vec![Complex::<T>::zero(); segment_len];
        plan.execute(&windowed, &mut spectrum);

        for k in 0..num_freq {
            psd_accum[k] = psd_accum[k] + spectrum[k].norm_sqr();
        }

        num_segments += 1;
        start += hop_size;
    }

    if num_segments == 0 {
        return periodogram(signal);
    }

    let num_seg_t = T::from_usize(num_segments);
    let seg_len_t = T::from_usize(segment_len);
    let denom = num_seg_t * w_sum_sq * seg_len_t;

    for val in &mut psd_accum {
        *val = *val / denom;
    }

    psd_accum
}

/// Compute the cross-spectral density (CSD) between two signals using Welch's method.
///
/// For each segment the windowed FFT is computed for both `x` and `y`, then
/// `X[k] * conj(Y[k])` is accumulated and averaged.
///
/// Returns a one-sided CSD of length `segment_len / 2 + 1`.  If the signals
/// are shorter than one segment, returns an empty `Vec`.
pub fn cross_spectral_density<T: Float>(x: &[T], y: &[T], config: &WelchConfig) -> Vec<Complex<T>> {
    let segment_len = config.segment_len;
    let overlap = config.overlap;

    if segment_len == 0 || overlap >= segment_len {
        return Vec::new();
    }

    // Both signals must be long enough for at least one segment.
    let min_len = x.len().min(y.len());
    if min_len < segment_len {
        return Vec::new();
    }

    let hop_size = segment_len - overlap;
    let window = generate_window::<T>(config.window, segment_len);
    let w_sum_sq: T = window.iter().fold(T::ZERO, |acc, &w| acc + w * w);

    let plan = match Plan::dft_1d(segment_len, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let num_freq = segment_len / 2 + 1;
    let mut csd_accum = vec![Complex::<T>::zero(); num_freq];
    let mut num_segments: usize = 0;

    let mut start = 0usize;
    while start + segment_len <= min_len {
        // Windowed complex segment for x.
        let x_windowed: Vec<Complex<T>> = (0..segment_len)
            .map(|i| Complex::new(x[start + i] * window[i], T::ZERO))
            .collect();
        // Windowed complex segment for y.
        let y_windowed: Vec<Complex<T>> = (0..segment_len)
            .map(|i| Complex::new(y[start + i] * window[i], T::ZERO))
            .collect();

        let mut x_spectrum = vec![Complex::<T>::zero(); segment_len];
        let mut y_spectrum = vec![Complex::<T>::zero(); segment_len];
        plan.execute(&x_windowed, &mut x_spectrum);
        plan.execute(&y_windowed, &mut y_spectrum);

        for k in 0..num_freq {
            csd_accum[k] = csd_accum[k] + x_spectrum[k] * y_spectrum[k].conj();
        }

        num_segments += 1;
        start += hop_size;
    }

    if num_segments == 0 {
        return Vec::new();
    }

    let num_seg_t = T::from_usize(num_segments);
    let seg_len_t = T::from_usize(segment_len);
    let denom = num_seg_t * w_sum_sq * seg_len_t;

    for val in &mut csd_accum {
        *val = Complex::new(val.re / denom, val.im / denom);
    }

    csd_accum
}

/// Compute magnitude-squared coherence between two signals.
///
/// Coherence is defined as:
/// ```text
/// C[k] = |Cxy[k]|² / (Pxx[k] * Pyy[k])
/// ```
/// where `Cxy` is the cross-spectral density and `Pxx`, `Pyy` are the
/// auto-power spectral densities estimated via [`welch`].
///
/// Values are clamped to `[0, 1]` for numerical stability.  If the denominator
/// is below a small epsilon the result is set to `0`.
///
/// Returns a one-sided coherence estimate of length `segment_len / 2 + 1`.
pub fn coherence<T: Float>(x: &[T], y: &[T], config: &WelchConfig) -> Vec<T> {
    let pxx = welch(x, config);
    let pyy = welch(y, config);
    let cxy = cross_spectral_density(x, y, config);

    // If any of the above failed to produce output, return empty.
    if pxx.is_empty() || pyy.is_empty() || cxy.is_empty() {
        return Vec::new();
    }

    let num_freq = pxx.len();
    // Use a small epsilon relative to the data scale to guard against division
    // by zero when both signals are silent (or nearly so).
    let epsilon: T = T::from_f64(1e-30);

    (0..num_freq)
        .map(|k| {
            let denom = pxx[k] * pyy[k];
            if denom < epsilon {
                T::ZERO
            } else {
                let c = cxy[k].norm_sqr() / denom;
                // Clamp to [0, 1] for numerical stability.
                if c > T::ONE {
                    T::ONE
                } else if c < T::ZERO {
                    T::ZERO
                } else {
                    c
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_periodogram_empty() {
        let psd: Vec<f64> = periodogram(&[]);
        assert!(psd.is_empty());
    }

    #[test]
    fn test_periodogram_dc() {
        // DC signal: all energy at bin 0.
        let n = 256;
        let signal = vec![1.0f64; n];
        let psd = periodogram(&signal);
        assert_eq!(psd.len(), n / 2 + 1);
        // DC bin should be the largest.
        let dc = psd[0];
        for (i, &p) in psd.iter().skip(1).enumerate() {
            assert!(
                p <= dc * 1.1 + 1e-10,
                "Bin {} has more power than DC: {} > {}",
                i + 1,
                p,
                dc
            );
        }
    }

    #[test]
    fn test_periodogram_sine() {
        // Sine at frequency 10/256: energy should peak at bin 10.
        let n = 256_u32;
        let freq_bin = 10_u32;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * f64::from(freq_bin) * f64::from(i) / f64::from(n))
                    .sin()
            })
            .collect();
        let psd = periodogram(&signal);
        let (peak_idx, _) = psd
            .iter()
            .enumerate()
            .skip(1)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        // Hann window spreads energy slightly, so allow ±1 bin.
        assert!(
            (peak_idx as i64 - i64::from(freq_bin)).abs() <= 1,
            "Peak at {peak_idx} expected near {freq_bin}"
        );
    }

    #[test]
    fn test_welch_returns_correct_length() {
        let signal: Vec<f64> = (0..1024).map(|i| (f64::from(i) * 0.1).sin()).collect();
        let config = WelchConfig {
            segment_len: 256,
            overlap: 128,
            window: SpectralWindow::Hann,
        };
        let psd = welch(&signal, &config);
        assert_eq!(psd.len(), 256 / 2 + 1);
    }

    #[test]
    fn test_coherence_identical_signals() {
        // Coherence of a signal with itself should be ~1.
        let signal: Vec<f64> = (0..512).map(|i| (f64::from(i) * 0.2).sin()).collect();
        let config = WelchConfig {
            segment_len: 128,
            overlap: 64,
            window: SpectralWindow::Hann,
        };
        let coh = coherence(&signal, &signal, &config);
        // Middle bins (skip DC, take first 30) should be near 1.
        for &c in coh.iter().skip(1).take(30) {
            assert!(
                c > 0.99,
                "Coherence should be ~1 for identical signals, got {c}"
            );
        }
    }

    #[test]
    fn test_coherence_returns_correct_length() {
        let x: Vec<f64> = (0..512).map(|i| (f64::from(i) * 0.3).sin()).collect();
        let y: Vec<f64> = (0..512).map(|i| (f64::from(i) * 0.5).cos()).collect();
        let config = WelchConfig {
            segment_len: 128,
            overlap: 0,
            window: SpectralWindow::Rectangular,
        };
        let coh = coherence(&x, &y, &config);
        assert_eq!(coh.len(), 128 / 2 + 1);
    }

    #[test]
    fn test_window_rectangular() {
        let w: Vec<f64> = generate_window(SpectralWindow::Rectangular, 8);
        assert_eq!(w.len(), 8);
        for &v in &w {
            assert!((v - 1.0).abs() < 1e-12, "Expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_window_hann_endpoints() {
        let w: Vec<f64> = generate_window(SpectralWindow::Hann, 8);
        assert_eq!(w.len(), 8);
        // Hann window starts and ends near 0.
        assert!(w[0].abs() < 1e-12, "Hann[0] should be ~0, got {}", w[0]);
        assert!(w[7].abs() < 1e-12, "Hann[7] should be ~0, got {}", w[7]);
    }

    #[test]
    fn test_window_single_element() {
        let w: Vec<f64> = generate_window(SpectralWindow::Blackman, 1);
        assert_eq!(w, vec![1.0f64]);
    }

    #[test]
    fn test_welch_invalid_config_overlap() {
        // overlap >= segment_len should return empty.
        let signal = vec![1.0f64; 512];
        let config = WelchConfig {
            segment_len: 128,
            overlap: 128,
            window: SpectralWindow::Hann,
        };
        let psd = welch(&signal, &config);
        assert!(psd.is_empty());
    }

    #[test]
    fn test_cross_spectral_density_length() {
        let x: Vec<f64> = (0..512).map(|i| (f64::from(i) * 0.2).sin()).collect();
        let y: Vec<f64> = (0..512).map(|i| (f64::from(i) * 0.4).cos()).collect();
        let config = WelchConfig {
            segment_len: 128,
            overlap: 64,
            window: SpectralWindow::Hamming,
        };
        let csd = cross_spectral_density(&x, &y, &config);
        assert_eq!(csd.len(), 128 / 2 + 1);
    }
}
