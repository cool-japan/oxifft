//! FFT-based signal resampling.
//!
//! Provides spectrally-accurate resampling via frequency-domain zero-padding
//! (interpolation) or truncation (decimation). This method preserves all
//! frequency content up to the Nyquist frequency of the output signal.
//!
//! # Algorithm
//!
//! For upsampling (output_len > input_len):
//! 1. FFT of input
//! 2. Zero-pad spectrum at the Nyquist frequency
//! 3. IFFT with proper scaling
//!
//! For downsampling (output_len < input_len):
//! 1. FFT of input
//! 2. Truncate spectrum at new Nyquist (low-pass anti-alias filter)
//! 3. IFFT with proper scaling
//!
//! # Example
//!
//! ```ignore
//! use oxifft::signal::resample;
//!
//! // Upsample from 100 to 200 samples
//! let signal: Vec<f64> = (0..100).map(|i| (i as f64 * 0.3).sin()).collect();
//! let resampled = resample(&signal, 200);
//! assert_eq!(resampled.len(), 200);
//! ```

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};

extern crate alloc;
use alloc::vec::Vec;

/// Resample a real signal to a new length using spectral interpolation/decimation.
///
/// Uses FFT-based frequency-domain resampling:
/// - **Upsampling** (`new_len > signal.len()`): zero-pads the spectrum between
///   the positive and negative frequencies, preserving all existing spectral
///   content and splitting the Nyquist bin to avoid aliasing artifacts.
/// - **Downsampling** (`new_len < signal.len()`): truncates the spectrum at the
///   new Nyquist frequency, acting as an ideal low-pass anti-aliasing filter.
///
/// This is spectrally perfect for bandlimited signals.
///
/// # Arguments
/// * `signal` - Input real-valued signal.
/// * `new_len` - Desired output length.
///
/// # Returns
/// Resampled signal of length `new_len`. Returns an empty `Vec` if the input
/// is empty, `new_len` is zero, or FFT plan creation fails.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "signal")]
/// # {
/// use oxifft::signal::resample;
/// let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.3).sin()).collect();
/// let up = resample(&signal, 128);
/// assert_eq!(up.len(), 128);
/// let down = resample(&signal, 32);
/// assert_eq!(down.len(), 32);
/// # }
/// ```
pub fn resample<T: Float>(signal: &[T], new_len: usize) -> Vec<T> {
    if signal.is_empty() || new_len == 0 {
        return Vec::new();
    }

    let n = signal.len();

    if new_len == n {
        return signal.to_vec();
    }

    // Convert real signal to complex for FFT.
    let input: Vec<Complex<T>> = signal.iter().map(|&s| Complex::new(s, T::ZERO)).collect();

    // Forward FFT.
    let fwd_plan = match Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut spectrum = vec![Complex::<T>::zero(); n];
    fwd_plan.execute(&input, &mut spectrum);

    // Build the new (zero-initialized) output spectrum of length new_len.
    let mut new_spectrum = vec![Complex::<T>::zero(); new_len];

    // Amplitude scale factor: new_len / n ensures correct amplitude after IFFT.
    let amplitude_scale = T::from_usize(new_len) / T::from_usize(n);

    if new_len > n {
        // --- Upsampling: zero-pad the spectrum ---
        //
        // For a real signal of length n, the DFT has the Hermitian symmetry:
        //   spectrum[k] = conj(spectrum[n-k])
        //
        // Layout: [DC, pos_1, ..., pos_{m-1}, Nyquist*, neg_{m-1}, ..., neg_1]
        //   where m = n/2 (Nyquist* only exists when n is even).
        //
        // For even n:
        //   pos_count = n/2        (DC=1 + positives without Nyquist = n/2-1 → total n/2)
        //   neg_count = n/2 - 1   (pure negatives, no Nyquist or DC)
        //   Nyquist bin at index n/2 is split into two half-energy bins.
        //
        // For odd n:
        //   pos_count = (n+1)/2   (DC + all positives, no Nyquist bin)
        //   neg_count = n - pos_count = (n-1)/2

        let n_is_even = n.is_multiple_of(2);

        if n_is_even {
            let half = n / 2; // index of Nyquist bin in old spectrum

            // Copy DC + positive frequencies (excluding Nyquist): indices [0..half].
            new_spectrum[..half].copy_from_slice(&spectrum[..half]);

            // Copy pure negative frequencies: indices [half+1..n] in old → [new_len-half+1..new_len] in new.
            let neg_count = half - 1; // number of pure negative bins
            if neg_count > 0 {
                new_spectrum[new_len - neg_count..].copy_from_slice(&spectrum[half + 1..]);
            }

            // Split Nyquist energy into two half-amplitude bins to avoid aliasing.
            let half_nyq = Complex::new(spectrum[half].re / T::TWO, spectrum[half].im / T::TWO);
            new_spectrum[half] = half_nyq;
            new_spectrum[new_len - half] = half_nyq;
        } else {
            // Odd n: no Nyquist bin. pos_count = (n+1)/2 includes DC and all positive freqs.
            let pos_count = n.div_ceil(2);
            let neg_count = n - pos_count; // = (n-1)/2

            // Copy DC + positive frequencies: indices [0..pos_count].
            new_spectrum[..pos_count].copy_from_slice(&spectrum[..pos_count]);

            // Copy pure negative frequencies at the tail.
            if neg_count > 0 {
                new_spectrum[new_len - neg_count..].copy_from_slice(&spectrum[pos_count..]);
            }
        }
    } else {
        // --- Downsampling: truncate the spectrum ---
        //
        // Keep only frequency bins up to the new Nyquist to act as a
        // low-pass anti-aliasing filter.
        //
        // For even new_len:
        //   pos_count = new_len/2    (DC + positives excl Nyquist)
        //   neg_count = new_len/2 - 1  (pure negatives)
        //   Nyquist of old spectrum lands at new_len/2 (kept as-is)
        //
        // For odd new_len:
        //   pos_count = (new_len+1)/2
        //   neg_count = new_len - pos_count

        let new_is_even = new_len.is_multiple_of(2);

        if new_is_even {
            let new_half = new_len / 2;
            let neg_count = new_half - 1;

            // DC + positive frequencies (including Nyquist of the new signal): [0..=new_half].
            new_spectrum[..=new_half].copy_from_slice(&spectrum[..=new_half]);

            // Pure negative frequencies: take the last neg_count bins of old spectrum.
            if neg_count > 0 {
                new_spectrum[new_len - neg_count..].copy_from_slice(&spectrum[n - neg_count..]);
            }
        } else {
            let pos_count = new_len.div_ceil(2); // DC + positives, no Nyquist
            let neg_count = new_len - pos_count;

            // DC + positive frequencies.
            new_spectrum[..pos_count].copy_from_slice(&spectrum[..pos_count]);

            // Pure negative frequencies from the tail of the old spectrum.
            if neg_count > 0 {
                new_spectrum[new_len - neg_count..].copy_from_slice(&spectrum[n - neg_count..]);
            }
        }
    }

    // Apply amplitude scale to all bins.
    for c in &mut new_spectrum {
        *c = Complex::new(c.re * amplitude_scale, c.im * amplitude_scale);
    }

    // Inverse FFT of the new spectrum.
    let inv_plan = match Plan::dft_1d(new_len, Direction::Backward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut output = vec![Complex::<T>::zero(); new_len];
    inv_plan.execute(&new_spectrum, &mut output);

    // Normalize by 1/new_len (IFFT normalization) and extract real parts.
    let ifft_scale = T::ONE / T::from_usize(new_len);
    output.iter().map(|c| c.re * ifft_scale).collect()
}

/// Resample a real signal from one sample rate to another.
///
/// Convenience wrapper around [`resample`] that computes the output length
/// from the ratio of sample rates.
///
/// # Arguments
/// * `signal`    - Input real-valued signal.
/// * `orig_rate` - Original sample rate in Hz (must be positive).
/// * `new_rate`  - Target sample rate in Hz (must be positive).
///
/// # Returns
/// Resampled signal. Returns an empty `Vec` if the input is empty, either
/// rate is non-positive, or the computed output length is zero.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "signal")]
/// # {
/// use oxifft::signal::resample_to;
/// let signal: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
/// // Resample from 8 kHz to 16 kHz
/// let up = resample_to(&signal, 8000.0, 16000.0);
/// assert_eq!(up.len(), 2000);
/// # }
/// ```
pub fn resample_to<T: Float>(signal: &[T], orig_rate: f64, new_rate: f64) -> Vec<T> {
    if signal.is_empty() || orig_rate <= 0.0 || new_rate <= 0.0 {
        return Vec::new();
    }
    let new_len = (signal.len() as f64 * new_rate / orig_rate).round() as usize;
    if new_len == 0 {
        return Vec::new();
    }
    resample(signal, new_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_identity() {
        let n = 64;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let result = resample(&signal, n);
        assert_eq!(result.len(), n);
        for (a, b) in signal.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-9, "Identity mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_resample_double_length() {
        // Upsampling: result should have correct length and preserve signal content
        let n = 64;
        let freq = 5.0f64;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / n as f64).sin())
            .collect();
        let resampled = resample(&signal, 2 * n);
        assert_eq!(resampled.len(), 2 * n);
        // The resampled signal at even indices should match original (approximately)
        for i in 0..n / 4 {
            let diff = (resampled[2 * i] - signal[i]).abs();
            assert!(
                diff < 0.02,
                "Sample mismatch at {}: {} vs {}",
                i,
                resampled[2 * i],
                signal[i]
            );
        }
    }

    #[test]
    fn test_resample_half_length() {
        let n = 128;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
        let resampled = resample(&signal, n / 2);
        assert_eq!(resampled.len(), n / 2);
    }

    #[test]
    fn test_resample_empty() {
        let empty: Vec<f64> = Vec::new();
        assert!(resample(&empty, 64).is_empty());
        assert!(resample(&[1.0f64], 0).is_empty());
    }

    #[test]
    fn test_resample_to() {
        let signal: Vec<f64> = (0..1000).map(|i| (f64::from(i) * 0.1).sin()).collect();
        // Resample from 8000 Hz to 16000 Hz (double)
        let resampled = resample_to(&signal, 8000.0, 16000.0);
        assert_eq!(resampled.len(), 2000);
    }

    #[test]
    fn test_resample_energy_preservation() {
        // Upsample and check energy is approximately preserved
        let n = 128;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let up = resample(&signal, 2 * n);
        // Scale factor: upsampled signal has same amplitude, more samples
        let orig_energy: f64 = signal.iter().map(|&x| x * x).sum::<f64>();
        let up_energy: f64 = up.iter().map(|&x| x * x).sum::<f64>();
        // Upsampled has 2x the samples at same amplitude, so 2x energy
        let ratio = up_energy / orig_energy;
        assert!(
            (ratio - 2.0).abs() < 0.1,
            "Energy ratio {ratio} should be ~2.0 for 2x upsampling"
        );
    }
}
