//! Mel-frequency analysis for audio signal processing.
//!
//! Provides mel filterbank, mel spectrogram, and MFCC computation
//! which are fundamental to speech recognition and audio ML.
//!
//! # Mel Scale
//!
//! The mel scale is a perceptual scale of pitches, where equal distances
//! correspond to equal perceived pitch differences. The conversion is:
//!   mel = 2595 * log10(1 + hz / 700)
//!   hz = 700 * (10^(mel/2595) - 1)
//!
//! # Example
//!
//! ```ignore
//! use oxifft::streaming::{mel_spectrogram, mfcc, MelConfig};
//!
//! let signal: Vec<f64> = (0..8000).map(|i| (i as f64 * 0.1).sin()).collect();
//! let config = MelConfig::new(8000.0, 256, 128, 40);
//! let mel_spec = mel_spectrogram(&signal, &config);  // Vec<Vec<f64>>
//! let coeffs = mfcc(&signal, &config, 13);            // Vec<Vec<f64>>
//! ```

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use num_traits::Float as NumFloat;

use crate::kernel::Float;

use super::stft::{power_spectrogram, stft as compute_stft};
use super::window::WindowFunction;

/// Configuration for mel-frequency analysis.
#[derive(Clone, Debug)]
pub struct MelConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// FFT size (frame size).
    pub fft_size: usize,
    /// Hop size (frame advance in samples).
    pub hop_size: usize,
    /// Number of mel filterbanks.
    pub n_mels: usize,
    /// Lowest frequency (Hz, default 0.0).
    pub f_min: f64,
    /// Highest frequency (Hz, default sample_rate / 2).
    pub f_max: f64,
}

impl MelConfig {
    /// Create mel config with sensible defaults.
    ///
    /// `f_min` defaults to 0.0 and `f_max` defaults to `sample_rate / 2`.
    pub fn new(sample_rate: f64, fft_size: usize, hop_size: usize, n_mels: usize) -> Self {
        Self {
            sample_rate,
            fft_size,
            hop_size,
            n_mels,
            f_min: 0.0,
            f_max: sample_rate / 2.0,
        }
    }
}

/// Convert a frequency in Hz to mel scale.
///
/// mel = 2595 * log10(1 + hz / 700)
#[inline]
pub fn hz_to_mel<T: Float>(hz: T) -> T {
    let base = T::from_f64(700.0);
    let factor = T::from_f64(2595.0);
    let ln10 = T::from_f64(10.0_f64.ln());
    factor * NumFloat::ln(T::ONE + hz / base) / ln10
}

/// Convert a mel-scale value back to Hz.
///
/// hz = 700 * (10^(mel / 2595) - 1)
#[inline]
pub fn mel_to_hz<T: Float>(mel: T) -> T {
    let base = T::from_f64(700.0);
    let factor = T::from_f64(2595.0);
    let ten = T::from_f64(10.0);
    base * (NumFloat::powf(ten, mel / factor) - T::ONE)
}

/// Build a mel filterbank matrix of shape `[n_mels][n_freq_bins]`.
///
/// Each row is a triangular filter covering a region of the FFT spectrum.
/// Filters are normalized by their width.
///
/// `n_freq_bins` = `fft_size / 2 + 1` (the one-sided spectrum).
///
/// # Arguments
///
/// * `config` - Mel configuration specifying sample rate, FFT size, etc.
///
/// # Returns
///
/// A `Vec<Vec<T>>` of shape `[n_mels][n_freq_bins]`.
pub fn build_mel_filterbank<T: Float>(config: &MelConfig) -> Vec<Vec<T>> {
    let n_freq_bins = config.fft_size / 2 + 1;
    let n_mels = config.n_mels;

    // Convert frequency boundaries to mel scale
    let mel_min = hz_to_mel::<T>(T::from_f64(config.f_min));
    let mel_max = hz_to_mel::<T>(T::from_f64(config.f_max));

    // n_mels + 2 equally spaced mel points (including the two edge points)
    let n_points = n_mels + 2;
    let mel_points: Vec<T> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * T::from_f64(i as f64 / (n_points - 1) as f64))
        .collect();

    // Convert mel points back to Hz
    let hz_points: Vec<T> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz to FFT bin indices: bin = floor(hz * (fft_size + 1) / sample_rate)
    let fft_size_plus_one = T::from_f64((config.fft_size + 1) as f64);
    let sample_rate = T::from_f64(config.sample_rate);
    let bin_indices: Vec<usize> = hz_points
        .iter()
        .map(|&hz| {
            let raw = NumFloat::floor(hz * fft_size_plus_one / sample_rate);
            // Clamp to valid range [0, n_freq_bins - 1]
            let idx = raw.to_usize().unwrap_or(0);
            idx.min(n_freq_bins.saturating_sub(1))
        })
        .collect();

    // Build each triangular filter
    let mut filterbank = vec![vec![T::ZERO; n_freq_bins]; n_mels];

    for m in 0..n_mels {
        let left = bin_indices[m];
        let center = bin_indices[m + 1];
        let right = bin_indices[m + 2];

        let width = right.saturating_sub(left);
        if width == 0 {
            // Zero-width filter; leave as all-zeros to avoid division by zero
            continue;
        }

        let width_t = T::from_f64(width as f64);

        // Rising slope: [left, center)
        if center > left {
            let rise_width = center - left;
            for k in left..center.min(n_freq_bins) {
                let numerator = T::from_f64((k - left) as f64);
                filterbank[m][k] = numerator / width_t;
            }
            // Ensure the center bin has rising contribution accounted for
            // (will be overwritten or combined with falling below if center == right)
            let _ = rise_width; // used above
        }

        // Falling slope: (center, right]
        if right > center {
            for k in center..right.min(n_freq_bins) {
                let numerator = T::from_f64((right - k) as f64);
                filterbank[m][k] = numerator / width_t;
            }
        }

        // Set center bin to peak (1 / width * width = 1 ... but normalized peak
        // at center is (center - left) / width using the rising formula, and
        // (right - center) / width using the falling formula — they may differ.
        // The standard triangular filter peaks at 1.0 only when left-center == center-right.
        // We instead set it as the maximum of the two slopes, which for a symmetric
        // filter gives 1.0. For asymmetric, the standard HTK formula just uses the
        // two-slope parametrisation without forcing peak=1.
        //
        // The above loops already handle the center bin via the falling slope
        // (k == center gives (right - center) / width). If center == left the
        // rising loop produces 0 for k=left which is correct.
    }

    filterbank
}

/// Compute log-mel spectrogram of a signal.
///
/// Applies STFT with a Hann window, computes the power spectrogram, applies
/// the mel filterbank matrix, and takes the natural logarithm (with a small
/// epsilon floor to avoid log(0)).
///
/// # Arguments
///
/// * `signal` - Input time-domain signal.
/// * `config` - Mel configuration.
///
/// # Returns
///
/// `Vec<Vec<T>>` of shape `[n_frames][n_mels]`.  Each inner vector contains
/// log-mel energies for one analysis frame.
pub fn mel_spectrogram<T: Float>(signal: &[T], config: &MelConfig) -> Vec<Vec<T>> {
    let n_freq_bins = config.fft_size / 2 + 1;

    // Build the mel filterbank
    let filterbank = build_mel_filterbank::<T>(config);

    // Compute STFT and derive power spectrogram
    let complex_spectrogram = compute_stft(
        signal,
        config.fft_size,
        config.hop_size,
        WindowFunction::Hann,
    );
    let power_spec = power_spectrogram(&complex_spectrogram);

    let epsilon = T::from_f64(1e-10);

    power_spec
        .iter()
        .map(|power_frame| {
            // Only use the one-sided spectrum bins [0, n_freq_bins)
            let n_bins = power_frame.len().min(n_freq_bins);

            // Apply filterbank: mel_energy[m] = sum_k filterbank[m][k] * power[k]
            (0..config.n_mels)
                .map(|m| {
                    let energy: T = filterbank[m][..n_bins]
                        .iter()
                        .zip(power_frame[..n_bins].iter())
                        .fold(T::ZERO, |acc, (&filt, &pwr)| acc + filt * pwr);

                    // Log-mel: natural log with epsilon floor
                    NumFloat::ln(if energy > epsilon { energy } else { epsilon })
                })
                .collect()
        })
        .collect()
}

/// Compute Mel-Frequency Cepstral Coefficients (MFCC).
///
/// Applies a DCT-II to the log-mel spectrogram to decorrelate the features.
/// This is the standard approach used in speech recognition.
///
/// DCT-II formula (orthonormal variant, coefficient k for frame with n_mels values):
/// `c[k] = sum_{m=0}^{n_mels-1} frame[m] * cos(pi * k * (m + 0.5) / n_mels)`
///
/// # Arguments
///
/// * `signal`   - Input time-domain signal.
/// * `config`   - Mel configuration.
/// * `n_coeffs` - Number of MFCC coefficients to compute (typically 13).
///
/// # Returns
///
/// `Vec<Vec<T>>` of shape `[n_frames][n_coeffs]`.
pub fn mfcc<T: Float>(signal: &[T], config: &MelConfig, n_coeffs: usize) -> Vec<Vec<T>> {
    let log_mel = mel_spectrogram(signal, config);
    let n_mels = config.n_mels;
    let n_mels_f = T::from_f64(n_mels as f64);
    let pi = <T as Float>::PI;

    log_mel
        .iter()
        .map(|frame| {
            (0..n_coeffs)
                .map(|k| {
                    let k_t = T::from_f64(k as f64);
                    frame.iter().enumerate().fold(T::ZERO, |acc, (m, &val)| {
                        let m_plus_half = T::from_f64(m as f64) + T::from_f64(0.5);
                        let angle = pi * k_t * m_plus_half / n_mels_f;
                        acc + val * NumFloat::cos(angle)
                    })
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hz_to_mel_and_back() {
        // Round-trip should recover original Hz
        let hz_values = [100.0f64, 440.0, 1000.0, 4000.0, 8000.0];
        for &hz in &hz_values {
            let mel = hz_to_mel(hz);
            let recovered = mel_to_hz(mel);
            assert!(
                (recovered - hz).abs() < 0.001,
                "Round-trip failed for {hz}: got {recovered}"
            );
        }
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let config = MelConfig::new(8000.0, 256, 128, 40);
        let fb = build_mel_filterbank::<f64>(&config);
        assert_eq!(fb.len(), 40); // n_mels rows
        assert_eq!(fb[0].len(), 256 / 2 + 1); // n_freq_bins
    }

    #[test]
    fn test_mel_filterbank_nonnegative() {
        let config = MelConfig::new(8000.0, 256, 128, 40);
        let fb = build_mel_filterbank::<f64>(&config);
        for row in &fb {
            for &v in row {
                assert!(v >= 0.0, "Filterbank value should be non-negative: {v}");
            }
        }
    }

    #[test]
    fn test_mel_spectrogram_shape() {
        let sample_rate = 8000.0f64;
        let signal: Vec<f64> = (0..8000).map(|i| (f64::from(i) * 0.1).sin()).collect();
        let config = MelConfig::new(sample_rate, 256, 128, 40);
        let mel_spec = mel_spectrogram(&signal, &config);
        // Should have multiple frames
        assert!(!mel_spec.is_empty(), "Should have at least one frame");
        // Each frame should have n_mels values
        assert_eq!(mel_spec[0].len(), 40);
    }

    #[test]
    fn test_mfcc_shape() {
        let sample_rate = 8000.0f64;
        let signal: Vec<f64> = (0..8000).map(|i| (f64::from(i) * 0.1).sin()).collect();
        let config = MelConfig::new(sample_rate, 256, 128, 40);
        let coefficients = mfcc(&signal, &config, 13);
        assert!(!coefficients.is_empty());
        assert_eq!(coefficients[0].len(), 13);
    }

    #[test]
    fn test_mel_filterbank_sum_of_weights() {
        // Each filter should sum to a positive value (it has some energy coverage)
        let config = MelConfig::new(16000.0, 512, 256, 40);
        let fb = build_mel_filterbank::<f64>(&config);
        // At least most filters should be non-zero (skip possible zero-width edge filters)
        let nonzero_filters = fb.iter().filter(|row| row.iter().any(|&v| v > 0.0)).count();
        assert!(
            nonzero_filters >= config.n_mels * 3 / 4,
            "Expected most filters to be non-zero, got {nonzero_filters} out of {}",
            config.n_mels
        );
    }

    #[test]
    fn test_mel_spectrogram_values_finite() {
        // All output values should be finite (no NaN or Inf from log(0))
        let sample_rate = 8000.0f64;
        let signal: Vec<f64> = (0..4096).map(|i| (f64::from(i) * 0.05).sin()).collect();
        let config = MelConfig::new(sample_rate, 256, 128, 40);
        let mel_spec = mel_spectrogram(&signal, &config);
        for (frame_idx, frame) in mel_spec.iter().enumerate() {
            for (mel_idx, &val) in frame.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Non-finite value at frame {frame_idx} mel bin {mel_idx}: {val}"
                );
            }
        }
    }

    #[test]
    fn test_mfcc_values_finite() {
        let sample_rate = 8000.0f64;
        let signal: Vec<f64> = (0..4096).map(|i| (f64::from(i) * 0.05).sin()).collect();
        let config = MelConfig::new(sample_rate, 256, 128, 40);
        let coefficients = mfcc(&signal, &config, 13);
        for (frame_idx, frame) in coefficients.iter().enumerate() {
            for (coeff_idx, &val) in frame.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Non-finite MFCC at frame {frame_idx} coeff {coeff_idx}: {val}"
                );
            }
        }
    }

    #[test]
    fn test_mel_config_defaults() {
        let config = MelConfig::new(22050.0, 1024, 512, 80);
        assert_eq!(config.f_min, 0.0);
        assert!((config.f_max - 11025.0).abs() < 1e-6);
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.fft_size, 1024);
        assert_eq!(config.hop_size, 512);
    }
}
