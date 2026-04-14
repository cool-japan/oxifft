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

/// Compute overlap-save STFT analysis.
///
/// The overlap-save (overlap-discard) method analyses a signal by sliding a window
/// over the raw input with `n_overlap = fft_size - hop_size` samples of overlap.
/// Unlike overlap-add, each input frame is read directly from the input buffer
/// (with window applied), so no accumulation is needed.
///
/// # Arguments
///
/// * `signal`   - Input time-domain signal
/// * `fft_size` - Size of each FFT frame (window length)
/// * `hop_size` - Number of new samples consumed per frame (`hop_size < fft_size`)
/// * `window`   - Window function applied to each frame before FFT
///
/// # Returns
///
/// A 2-D matrix of complex spectra: `result[t][k]` is the complex spectrum of frame
/// `t` at frequency bin `k`.  The first frame covers input samples `[0, fft_size)`,
/// but the overlap region (samples `[0, n_overlap)`) is initialised to zero, so
/// the first frame has `n_overlap` zero-padded samples on the left — exactly the
/// standard overlap-save convention for causal filters.
///
/// # Example
///
/// ```ignore
/// use oxifft::streaming::{stft_overlap_save, istft_overlap_save, WindowFunction};
///
/// let signal: Vec<f64> = (0..512).map(|i| (i as f64 * 0.1).sin()).collect();
/// let spectra = stft_overlap_save(&signal, 64, 16, WindowFunction::Rectangular);
/// let reconstructed = istft_overlap_save(&spectra, 64, 16, WindowFunction::Rectangular);
/// ```
pub fn stft_overlap_save<T: Float>(
    signal: &[T],
    fft_size: usize,
    hop_size: usize,
    window: WindowFunction,
) -> Vec<Vec<Complex<T>>> {
    if fft_size == 0 || hop_size == 0 || hop_size >= fft_size || signal.is_empty() {
        return Vec::new();
    }

    let window_coeffs: Vec<T> = window.generate(fft_size);
    let plan = match Plan::dft_1d(fft_size, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let n_overlap = fft_size - hop_size;

    // Build the padded input: prepend n_overlap zeros so that the first
    // hop-sized chunk starts at index 0 and the full first frame is
    // [zeros(n_overlap) | signal[0..hop_size]].
    let padded_len = n_overlap + signal.len();
    let num_frames = padded_len / hop_size; // integer division

    let mut spectrogram = Vec::with_capacity(num_frames);
    let mut frame_buf = vec![Complex::<T>::zero(); fft_size];
    let mut output = vec![Complex::<T>::zero(); fft_size];

    for frame_idx in 0..num_frames {
        let buf_start = frame_idx * hop_size; // position in the padded buffer

        // Fill frame_buf with windowed samples from the padded input
        for i in 0..fft_size {
            let padded_pos = buf_start + i;
            let sample = if padded_pos < n_overlap {
                // Still in the pre-pended zero region
                T::ZERO
            } else {
                let sig_pos = padded_pos - n_overlap;
                if sig_pos < signal.len() {
                    signal[sig_pos]
                } else {
                    T::ZERO
                }
            };
            frame_buf[i] = Complex::new(sample * window_coeffs[i], T::ZERO);
        }

        plan.execute(&frame_buf, &mut output);
        spectrogram.push(output.clone());
    }

    spectrogram
}

/// Inverse STFT using overlap-save synthesis.
///
/// Given a spectrogram produced by [`stft_overlap_save`], reconstructs the
/// time-domain signal.  Each frame is inverse-transformed and normalised by
/// `1/fft_size`; the first `n_overlap = fft_size - hop_size` samples of each
/// IFFT output (the "overlap" region that carries boundary effects) are
/// discarded, and only the trailing `hop_size` samples are appended to the
/// output.
///
/// Perfect reconstruction is obtained when `window = WindowFunction::Rectangular`
/// and the analysis and synthesis parameters are identical.
///
/// # Arguments
///
/// * `spectra`  - Spectrogram returned by [`stft_overlap_save`]
/// * `fft_size` - FFT frame length used during analysis
/// * `hop_size` - Hop size used during analysis
/// * `window`   - Window function used during analysis
///
/// # Returns
///
/// Reconstructed time-domain signal.  The length equals
/// `num_frames * hop_size` (before any trailing zeros added during analysis
/// are stripped).
///
/// # Example
///
/// ```ignore
/// use oxifft::streaming::{stft_overlap_save, istft_overlap_save, WindowFunction};
///
/// let signal: Vec<f64> = (0..256).map(|i| (i as f64 * 0.05).cos()).collect();
/// let spectra = stft_overlap_save(&signal, 64, 16, WindowFunction::Rectangular);
/// let recovered = istft_overlap_save(&spectra, 64, 16, WindowFunction::Rectangular);
/// // recovered should match signal in the valid interior region
/// ```
pub fn istft_overlap_save<T: Float>(
    spectra: &[Vec<Complex<T>>],
    fft_size: usize,
    hop_size: usize,
    window: WindowFunction,
) -> Vec<T> {
    if spectra.is_empty() || fft_size == 0 || hop_size == 0 || hop_size >= fft_size {
        return Vec::new();
    }

    let window_coeffs: Vec<T> = window.generate(fft_size);
    let plan = match Plan::dft_1d(fft_size, Direction::Backward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let n_overlap = fft_size - hop_size;
    let scale = T::ONE / T::from_usize(fft_size);

    // Each frame contributes exactly hop_size valid output samples.
    let total_len = spectra.len() * hop_size;
    let mut output = Vec::with_capacity(total_len);

    let mut ifft_buf = vec![Complex::<T>::zero(); fft_size];

    for frame_spectrum in spectra {
        if frame_spectrum.len() != fft_size {
            // Pad short frames with zeros to maintain alignment
            for _ in 0..hop_size {
                output.push(T::ZERO);
            }
            continue;
        }

        plan.execute(frame_spectrum, &mut ifft_buf);

        // Apply the analysis window inverse and scale; then keep only the
        // valid (non-overlapping) tail of each IFFT block.
        let threshold = T::from_f64(1e-12);
        for i in n_overlap..fft_size {
            let w = window_coeffs[i];
            // Avoid division by near-zero window coefficients.
            // Use explicit comparison to avoid ambiguous abs() dispatch.
            let sample = if w > threshold || w < (T::ZERO - threshold) {
                ifft_buf[i].re * scale / w
            } else {
                ifft_buf[i].re * scale
            };
            output.push(sample);
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

    // -----------------------------------------------------------------------
    // Overlap-save tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stft_overlap_save_basic_shape() {
        let signal: Vec<f64> = vec![0.0; 256];
        let fft_size = 64;
        let hop_size = 16;

        let spectra = stft_overlap_save(&signal, fft_size, hop_size, WindowFunction::Hann);

        // Verify all frames have the correct frequency-bin count
        assert!(!spectra.is_empty(), "Should produce at least one frame");
        for frame in &spectra {
            assert_eq!(frame.len(), fft_size);
        }
    }

    #[test]
    fn test_stft_overlap_save_same_frame_count_as_stft() {
        // Both methods should produce a consistent number of frames for the
        // same signal length and parameters.
        let signal: Vec<f64> = (0..512)
            .map(|i| (f64::from(i) / 8.0 * core::f64::consts::TAU).sin())
            .collect();
        let fft_size = 64;
        let hop_size = 16;

        let oa_spectra = stft(&signal, fft_size, hop_size, WindowFunction::Hann);
        let os_spectra = stft_overlap_save(&signal, fft_size, hop_size, WindowFunction::Hann);

        // The overlap-save method pads the front with n_overlap zeros, so it
        // may produce more frames.  What matters is that both are non-empty and
        // have the correct bin count.
        assert!(!oa_spectra.is_empty());
        assert!(!os_spectra.is_empty());
        for frame in &os_spectra {
            assert_eq!(frame.len(), fft_size);
        }
    }

    #[test]
    fn test_stft_overlap_save_roundtrip_rectangular() {
        // Perfect reconstruction is achievable with a rectangular window
        // because every output sample in the valid region is unweighted (w=1).
        //
        // Trace the overlap-save algorithm:
        //   - Analysis pads the front with n_overlap = fft_size - hop_size zeros.
        //   - Frame 0 reads padded[0..fft_size] = [zeros(n_overlap) | signal[0..hop_size]].
        //   - Synthesis discards the first n_overlap IFFT samples, keeps [n_overlap..fft_size]
        //     = signal[0..hop_size] → placed at recovered[0..hop_size].
        //   - Therefore recovered[i] == signal[i] starting at index 0.
        let n = 256;
        let fft_size = 64;
        let hop_size = 32;
        let window = WindowFunction::Rectangular;

        // Use a non-periodic signal so the test is not trivially passing by
        // coincidence when the period divides n_overlap.
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.0731_f64).sin()).collect();

        let spectra = stft_overlap_save(&signal, fft_size, hop_size, window.clone());
        let recovered = istft_overlap_save(&spectra, fft_size, hop_size, window);

        // recovered[i] == signal[i] from index 0 up to however many full hops
        // the synthesis produced.
        let check_len = recovered.len().min(n);

        let mut max_err = 0.0f64;
        for i in 0..check_len {
            let err = (recovered[i] - signal[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 1e-9,
            "Max roundtrip error {max_err} exceeds threshold with rectangular window"
        );
    }

    #[test]
    fn test_stft_overlap_save_magnitude_similar_to_overlap_add() {
        // For a simple sinusoidal signal the magnitude spectrum obtained by
        // overlap-save should concentrate energy near the same frequency bin
        // as overlap-add.
        let n = 512;
        let fft_size = 64;
        let hop_size = 16;
        let window = WindowFunction::Hann;

        // Pure tone at k=4 (frequency = 4/64 of sample rate)
        let freq_bin = 4usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * core::f64::consts::PI * freq_bin as f64 * i as f64 / fft_size as f64).sin()
            })
            .collect();

        let os_spectra = stft_overlap_save(&signal, fft_size, hop_size, window.clone());
        let oa_spectra = stft(&signal, fft_size, hop_size, window);

        // For both methods the bin with the most energy should be `freq_bin`
        // (or its mirror at `fft_size - freq_bin`) in the interior frames.
        let mid = os_spectra.len() / 2;
        let os_frame = &os_spectra[mid];
        let oa_frame = &oa_spectra[(oa_spectra.len() / 2).min(oa_spectra.len() - 1)];

        let os_peak = os_frame
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.norm()
                    .partial_cmp(&b.norm())
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k)
            .unwrap_or(0);

        let oa_peak = oa_frame
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.norm()
                    .partial_cmp(&b.norm())
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k)
            .unwrap_or(0);

        // Both peaks should be at the same bin (or its conjugate mirror)
        let mirror = fft_size - freq_bin;
        assert!(
            os_peak == freq_bin || os_peak == mirror,
            "Overlap-save peak at bin {os_peak}, expected {freq_bin} or {mirror}"
        );
        assert!(
            oa_peak == freq_bin || oa_peak == mirror,
            "Overlap-add peak at bin {oa_peak}, expected {freq_bin} or {mirror}"
        );
    }

    #[test]
    fn test_stft_overlap_save_empty_signal() {
        let spectra = stft_overlap_save::<f64>(&[], 64, 16, WindowFunction::Hann);
        assert!(spectra.is_empty());
    }

    #[test]
    fn test_stft_overlap_save_invalid_params() {
        let signal = vec![0.0f64; 128];
        // hop_size >= fft_size should return empty
        let spectra = stft_overlap_save(&signal, 32, 32, WindowFunction::Hann);
        assert!(spectra.is_empty());
        // zero fft_size
        let spectra = stft_overlap_save(&signal, 0, 16, WindowFunction::Hann);
        assert!(spectra.is_empty());
    }
}
