//! Example demonstrating Short-Time Fourier Transform (STFT) for streaming analysis.
//!
//! STFT is useful for analyzing time-varying signals like audio or sensor data.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::needless_range_loop)]

use oxifft::streaming::{
    istft, magnitude_spectrogram, power_spectrogram, stft, StreamingFft, WindowFunction,
};
use oxifft::Complex;

fn main() {
    println!("=== Streaming FFT (STFT) Example ===\n");

    // Generate a chirp signal (frequency increases over time)
    let sample_rate = 8000.0;
    let duration = 1.0; // seconds
    let n_samples = (sample_rate * duration) as usize;

    let mut signal = vec![0.0; n_samples];
    for i in 0..n_samples {
        let t = i as f64 / sample_rate;
        // Chirp from 100 Hz to 1000 Hz
        let freq = 900.0f64.mul_add(t, 100.0);
        signal[i] = (2.0 * std::f64::consts::PI * freq * t).sin();
    }

    println!("Signal parameters:");
    println!("  Sample rate: {sample_rate} Hz");
    println!("  Duration: {duration} seconds");
    println!("  Samples: {n_samples}");
    println!();

    // STFT parameters
    let fft_size = 256;
    let hop_size = 128;

    println!("STFT parameters:");
    println!("  FFT size: {fft_size}");
    println!("  Hop size: {hop_size}");
    println!("  Window: Hann");
    println!();

    // Method 1: Batch STFT
    println!("Method 1: Batch STFT");
    let spectrogram = stft(&signal, fft_size, hop_size, WindowFunction::Hann);

    println!(
        "Spectrogram shape: {} frames × {} bins",
        spectrogram.len(),
        spectrogram[0].len()
    );
    println!();

    // Compute magnitude spectrogram
    let mag_spec = magnitude_spectrogram(&spectrogram);

    // Find peak frequency in each frame
    println!("Peak frequencies over time:");
    let freq_resolution = sample_rate / fft_size as f64;
    for (frame_idx, frame) in mag_spec.iter().enumerate().step_by(10) {
        let (peak_bin, _) = frame[..=fft_size / 2] // only positive frequencies
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN in magnitude spectrum"))
            .unwrap_or((0, &0.0));
        let peak_freq = peak_bin as f64 * freq_resolution;
        let time = frame_idx as f64 * hop_size as f64 / sample_rate;
        println!("  t = {time:.3} s: peak at {peak_freq:.1} Hz");
    }
    println!();

    // Method 2: Streaming processing (simulating real-time)
    println!("Method 2: Streaming processing (real-time simulation)");
    let mut streaming_fft = StreamingFft::new(fft_size, hop_size, WindowFunction::Hamming);

    // Process in chunks of 512 samples
    let chunk_size = 512;
    let mut total_frames = 0;
    for chunk_start in (0..n_samples).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(n_samples);
        let chunk = &signal[chunk_start..chunk_end];

        // Feed samples into the streaming processor
        let frames_ready = streaming_fft.feed(chunk);

        // Pop and process available frames
        for _ in 0..frames_ready {
            if let Some(spec) = streaming_fft.pop_frame() {
                total_frames += 1;
                if total_frames % 5 == 0 {
                    // Find peak in this frame (positive frequencies only)
                    let magnitudes: Vec<f64> =
                        spec.iter().map(|c: &Complex<f64>| c.norm()).collect();
                    let (peak_bin, peak_mag): (usize, &f64) = magnitudes[..=fft_size / 2]
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN in magnitude"))
                        .unwrap_or((0, &0.0));
                    let peak_freq = peak_bin as f64 * freq_resolution;
                    println!("  Frame {total_frames}: peak at {peak_freq:.1} Hz (magnitude: {peak_mag:.3})");
                }
            }
        }
    }
    println!();

    // Reconstruction
    println!("Reconstruction test:");
    let reconstructed = istft(&spectrogram, hop_size, WindowFunction::Hann);
    println!("  Original signal length: {}", signal.len());
    println!("  Reconstructed length: {}", reconstructed.len());

    // Check reconstruction error
    let min_len = signal.len().min(reconstructed.len());
    let mut error = 0.0;
    for i in 0..min_len {
        let diff = signal[i] - reconstructed[i];
        error += diff * diff;
    }
    error = (error / min_len as f64).sqrt();
    println!("  RMS reconstruction error: {error:.6}");
    println!();

    // Power spectrogram
    let power_spec = power_spectrogram(&spectrogram);
    let avg_power: f64 = power_spec
        .iter()
        .flat_map(|frame| frame.iter())
        .sum::<f64>()
        / (power_spec.len() * power_spec[0].len()) as f64;
    println!("Average spectral power: {avg_power:.3}");
}
