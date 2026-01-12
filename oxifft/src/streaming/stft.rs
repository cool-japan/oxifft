//! Short-Time Fourier Transform (STFT) implementation.
//!
//! STFT analyzes how the frequency content of a signal changes over time
//! by computing FFT on overlapping windowed segments.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};

use super::window::WindowFunction;
use super::RingBuffer;

/// Streaming FFT processor for real-time STFT.
///
/// Maintains internal state for efficient frame-by-frame processing.
pub struct StreamingFft<T: Float> {
    /// FFT size.
    fft_size: usize,
    /// Hop size (frame advance).
    hop_size: usize,
    /// Window function coefficients.
    window: Vec<T>,
    /// Forward FFT plan.
    forward_plan: Option<Plan<T>>,
    /// Inverse FFT plan.
    inverse_plan: Option<Plan<T>>,
    /// Input ring buffer.
    input_buffer: RingBuffer<T>,
    /// Output ring buffer for overlap-add.
    output_buffer: Vec<T>,
    /// Output buffer write position.
    output_pos: usize,
    /// Pending output frames.
    pending_frames: Vec<Vec<Complex<T>>>,
}

impl<T: Float> StreamingFft<T> {
    /// Create a new streaming FFT processor.
    ///
    /// # Arguments
    ///
    /// * `fft_size` - Size of FFT (window size)
    /// * `hop_size` - Number of samples between frames
    /// * `window` - Window function to use
    ///
    /// # Example
    ///
    /// ```ignore
    /// use oxifft::streaming::{StreamingFft, WindowFunction};
    ///
    /// let mut processor = StreamingFft::new(256, 64, WindowFunction::Hann);
    /// ```
    pub fn new(fft_size: usize, hop_size: usize, window: WindowFunction) -> Self {
        let window_coeffs = window.generate(fft_size);
        let forward_plan = Plan::dft_1d(fft_size, Direction::Forward, Flags::ESTIMATE);
        let inverse_plan = Plan::dft_1d(fft_size, Direction::Backward, Flags::ESTIMATE);

        Self {
            fft_size,
            hop_size,
            window: window_coeffs,
            forward_plan,
            inverse_plan,
            input_buffer: RingBuffer::new(fft_size),
            output_buffer: vec![T::ZERO; fft_size * 2],
            output_pos: 0,
            pending_frames: Vec::new(),
        }
    }

    /// Feed new samples into the processor.
    ///
    /// Returns the number of complete frames available.
    pub fn feed(&mut self, samples: &[T]) -> usize {
        self.input_buffer.push_slice(samples);

        // Process any complete frames
        let mut frames_processed = 0;
        while self.input_buffer.len() >= self.fft_size {
            if let Some(frame) = self.process_frame() {
                self.pending_frames.push(frame);
                frames_processed += 1;
            }
            self.input_buffer.advance(self.hop_size);
        }

        frames_processed
    }

    /// Pop the next available output frame (spectrum).
    pub fn pop_frame(&mut self) -> Option<Vec<Complex<T>>> {
        if self.pending_frames.is_empty() {
            None
        } else {
            Some(self.pending_frames.remove(0))
        }
    }

    /// Process a single frame from the input buffer.
    fn process_frame(&self) -> Option<Vec<Complex<T>>> {
        let plan = self.forward_plan.as_ref()?;

        // Read windowed input
        let mut frame = vec![T::ZERO; self.fft_size];
        self.input_buffer.read_last(&mut frame);

        // Apply window
        let input: Vec<Complex<T>> = frame
            .iter()
            .zip(self.window.iter())
            .map(|(&s, &w)| Complex::new(s * w, T::ZERO))
            .collect();

        // Execute FFT
        let mut output = vec![Complex::<T>::zero(); self.fft_size];
        plan.execute(&input, &mut output);

        Some(output)
    }

    /// Process a single frame and return spectrum.
    ///
    /// # Arguments
    ///
    /// * `frame` - Input frame (must have length = fft_size)
    ///
    /// # Returns
    ///
    /// Complex spectrum of length fft_size.
    pub fn analyze_frame(&self, frame: &[T]) -> Vec<Complex<T>> {
        if frame.len() != self.fft_size {
            return vec![Complex::<T>::zero(); self.fft_size];
        }

        let plan = match &self.forward_plan {
            Some(p) => p,
            None => return vec![Complex::<T>::zero(); self.fft_size],
        };

        // Apply window and convert to complex
        let input: Vec<Complex<T>> = frame
            .iter()
            .zip(self.window.iter())
            .map(|(&s, &w)| Complex::new(s * w, T::ZERO))
            .collect();

        let mut output = vec![Complex::<T>::zero(); self.fft_size];
        plan.execute(&input, &mut output);
        output
    }

    /// Synthesize a frame from spectrum.
    ///
    /// # Arguments
    ///
    /// * `spectrum` - Complex spectrum (must have length = fft_size)
    ///
    /// # Returns
    ///
    /// Time-domain frame of length fft_size.
    pub fn synthesize_frame(&self, spectrum: &[Complex<T>]) -> Vec<T> {
        if spectrum.len() != self.fft_size {
            return vec![T::ZERO; self.fft_size];
        }

        let plan = match &self.inverse_plan {
            Some(p) => p,
            None => return vec![T::ZERO; self.fft_size],
        };

        let mut output = vec![Complex::<T>::zero(); self.fft_size];
        plan.execute(spectrum, &mut output);

        // Apply window and normalize
        let scale = T::ONE / T::from_usize(self.fft_size);
        output
            .iter()
            .zip(self.window.iter())
            .map(|(c, &w)| c.re * scale * w)
            .collect()
    }

    /// Get the FFT size.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Get the hop size.
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Get the window coefficients.
    pub fn window(&self) -> &[T] {
        &self.window
    }

    /// Clear internal buffers.
    pub fn clear(&mut self) {
        self.input_buffer.clear();
        self.pending_frames.clear();
        for v in &mut self.output_buffer {
            *v = T::ZERO;
        }
        self.output_pos = 0;
    }
}

/// Compute Short-Time Fourier Transform (STFT).
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `fft_size` - Size of each FFT frame
/// * `hop_size` - Number of samples between frames
/// * `window` - Window function
///
/// # Returns
///
/// Spectrogram as a `Vec` of `Vec<Complex<T>>`, one spectrum per frame.
///
/// # Example
///
/// ```ignore
/// use oxifft::streaming::{stft, WindowFunction};
///
/// let signal = vec![0.0f64; 4096];
/// let spectrogram = stft(&signal, 256, 64, WindowFunction::Hann);
/// println!("Number of frames: {}", spectrogram.len());
/// ```
pub fn stft<T: Float>(
    signal: &[T],
    fft_size: usize,
    hop_size: usize,
    window: WindowFunction,
) -> Vec<Vec<Complex<T>>> {
    if signal.len() < fft_size || fft_size == 0 || hop_size == 0 {
        return Vec::new();
    }

    let window_coeffs: Vec<T> = window.generate(fft_size);
    let plan = match Plan::dft_1d(fft_size, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let num_frames = (signal.len() - fft_size) / hop_size + 1;
    let mut spectrogram = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        let end = start + fft_size;

        // Apply window
        let input: Vec<Complex<T>> = signal[start..end]
            .iter()
            .zip(window_coeffs.iter())
            .map(|(&s, &w)| Complex::new(s * w, T::ZERO))
            .collect();

        let mut output = vec![Complex::<T>::zero(); fft_size];
        plan.execute(&input, &mut output);

        spectrogram.push(output);
    }

    spectrogram
}

/// Compute Inverse Short-Time Fourier Transform (ISTFT).
///
/// Reconstructs a time-domain signal from a spectrogram using overlap-add.
///
/// # Arguments
///
/// * `spectrogram` - Spectrogram (Vec of spectra)
/// * `hop_size` - Number of samples between frames (must match STFT)
/// * `window` - Window function (must match STFT)
///
/// # Returns
///
/// Reconstructed time-domain signal.
///
/// # Example
///
/// ```ignore
/// use oxifft::streaming::{stft, istft, WindowFunction};
///
/// let signal = vec![1.0f64; 1024];
/// let spectrogram = stft(&signal, 256, 64, WindowFunction::Hann);
/// let reconstructed = istft(&spectrogram, 64, WindowFunction::Hann);
/// ```
pub fn istft<T: Float>(
    spectrogram: &[Vec<Complex<T>>],
    hop_size: usize,
    window: WindowFunction,
) -> Vec<T> {
    if spectrogram.is_empty() || hop_size == 0 {
        return Vec::new();
    }

    let fft_size = spectrogram[0].len();
    if fft_size == 0 {
        return Vec::new();
    }

    let window_coeffs: Vec<T> = window.generate(fft_size);
    let plan = match Plan::dft_1d(fft_size, Direction::Backward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    // Calculate output length
    let num_frames = spectrogram.len();
    let output_len = fft_size + (num_frames - 1) * hop_size;

    let mut output = vec![T::ZERO; output_len];
    let mut window_sum = vec![T::ZERO; output_len];

    // Overlap-add synthesis
    let scale = T::ONE / T::from_usize(fft_size);

    for (frame_idx, spectrum) in spectrogram.iter().enumerate() {
        if spectrum.len() != fft_size {
            continue;
        }

        // Inverse FFT
        let mut frame = vec![Complex::<T>::zero(); fft_size];
        plan.execute(spectrum, &mut frame);

        // Apply window and add to output
        let start = frame_idx * hop_size;
        for i in 0..fft_size {
            let w = window_coeffs[i];
            output[start + i] = output[start + i] + frame[i].re * scale * w;
            window_sum[start + i] = window_sum[start + i] + w * w;
        }
    }

    // Normalize by window sum to achieve perfect reconstruction
    // output[n] = x[n] * window_sum[n], so we divide by window_sum[n] to recover x[n]
    let threshold = T::from_f64(1e-10);
    for i in 0..output_len {
        if window_sum[i] > threshold {
            output[i] = output[i] / window_sum[i];
        }
    }

    output
}

/// Compute magnitude spectrogram.
///
/// # Arguments
///
/// * `spectrogram` - Complex spectrogram
///
/// # Returns
///
/// Magnitude of each bin `(|X[k]|)`.
pub fn magnitude_spectrogram<T: Float>(spectrogram: &[Vec<Complex<T>>]) -> Vec<Vec<T>> {
    spectrogram
        .iter()
        .map(|frame| frame.iter().map(|c| c.norm()).collect())
        .collect()
}

/// Compute power spectrogram.
///
/// # Arguments
///
/// * `spectrogram` - Complex spectrogram
///
/// # Returns
///
/// Power of each bin `(|X[k]|²)`.
pub fn power_spectrogram<T: Float>(spectrogram: &[Vec<Complex<T>>]) -> Vec<Vec<T>> {
    spectrogram
        .iter()
        .map(|frame| frame.iter().map(|c| c.norm_sqr()).collect())
        .collect()
}

/// Compute phase spectrogram.
///
/// # Arguments
///
/// * `spectrogram` - Complex spectrogram
///
/// # Returns
///
/// Phase of each bin `(angle(X[k]))` in radians.
pub fn phase_spectrogram<T: Float>(spectrogram: &[Vec<Complex<T>>]) -> Vec<Vec<T>> {
    spectrogram
        .iter()
        .map(|frame| frame.iter().map(|c| c.im.atan2(c.re)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_fft_basic() {
        let mut processor: StreamingFft<f64> = StreamingFft::new(8, 4, WindowFunction::Hann);

        let samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let frames = processor.feed(&samples);

        assert!(frames > 0);
        assert!(processor.pop_frame().is_some());
    }

    #[test]
    fn test_streaming_fft_analyze_synthesize() {
        let processor: StreamingFft<f64> = StreamingFft::new(8, 4, WindowFunction::Hann);

        let frame = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let spectrum = processor.analyze_frame(&frame);
        let reconstructed = processor.synthesize_frame(&spectrum);

        assert_eq!(spectrum.len(), 8);
        assert_eq!(reconstructed.len(), 8);
    }

    #[test]
    fn test_stft_basic() {
        let signal: Vec<f64> = vec![0.0; 128];
        let spectrogram = stft(&signal, 32, 16, WindowFunction::Hann);

        // Number of frames = (128 - 32) / 16 + 1 = 7
        assert_eq!(spectrogram.len(), 7);
        assert_eq!(spectrogram[0].len(), 32);
    }

    #[test]
    fn test_stft_istft_roundtrip() {
        // Generate test signal
        let n = 256;
        let signal: Vec<f64> = (0..n).map(|i| (f64::from(i) / 10.0).sin()).collect();

        let fft_size = 64;
        let hop_size = 16;
        let window = WindowFunction::Hann;

        // Forward STFT
        let spectrogram = stft(&signal, fft_size, hop_size, window.clone());

        // Inverse STFT
        let reconstructed = istft(&spectrogram, hop_size, window);

        // Check middle portion (edges may have artifacts)
        let start = fft_size;
        let end = reconstructed.len().saturating_sub(fft_size);

        if end > start {
            for i in start..end.min(signal.len()) {
                let diff = (reconstructed[i] - signal[i]).abs();
                assert!(
                    diff < 0.1,
                    "Mismatch at {}: {} vs {}",
                    i,
                    reconstructed[i],
                    signal[i]
                );
            }
        }
    }

    #[test]
    fn test_magnitude_spectrogram() {
        let signal: Vec<f64> = vec![1.0; 64];
        let spectrogram = stft(&signal, 16, 8, WindowFunction::Rectangular);

        let magnitudes = magnitude_spectrogram(&spectrogram);

        assert!(!magnitudes.is_empty());
        // DC component should be large for constant signal
        assert!(magnitudes[0][0] > 0.0);
    }

    #[test]
    fn test_power_spectrogram() {
        let signal: Vec<f64> = vec![1.0; 64];
        let spectrogram = stft(&signal, 16, 8, WindowFunction::Rectangular);

        let powers = power_spectrogram(&spectrogram);

        assert!(!powers.is_empty());
        assert!(powers[0][0] >= 0.0);
    }

    #[test]
    fn test_phase_spectrogram() {
        let signal: Vec<f64> = (0..64).map(|i| f64::from(i).sin()).collect();
        let spectrogram = stft(&signal, 16, 8, WindowFunction::Hann);

        let phases = phase_spectrogram(&spectrogram);

        assert!(!phases.is_empty());
        // Phase should be in [-π, π]
        for frame in &phases {
            for &phase in frame {
                assert!(phase >= -core::f64::consts::PI - 0.01);
                assert!(phase <= core::f64::consts::PI + 0.01);
            }
        }
    }
}
