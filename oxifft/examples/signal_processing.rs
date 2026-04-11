//! Signal processing example for `OxiFFT`.
//!
//! Demonstrates:
//! - Hilbert transform and analytic signal
//! - Envelope detection
//! - Power spectral density via Welch's method
//! - Cepstral analysis
//!
//! Run with:
//!   `cargo run --example signal_processing --features signal`

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use oxifft::signal::{
    coherence, complex_cepstrum, cross_spectral_density, envelope, hilbert,
    instantaneous_frequency, instantaneous_phase, minimum_phase, periodogram, real_cepstrum, welch,
    SpectralWindow, WelchConfig,
};

fn demo_hilbert_transform(signal: &[f64]) {
    println!("\n--- Hilbert Transform ---");
    let analytic = hilbert(signal);
    println!("Analytic signal length: {}", analytic.len());
    println!("First 5 samples (complex): {:?}", &analytic[..5]);

    println!("\n--- Envelope Detection ---");
    let env = envelope(signal);
    let max_env = env.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_env = env.iter().copied().fold(f64::INFINITY, f64::min);
    println!("Envelope range: [{min_env:.4}, {max_env:.4}]");
    println!("Expected: ~1.0 to ~1.5 (sum of amplitudes)");
}

fn demo_instantaneous_frequency(signal: &[f64], sample_rate: f64) {
    println!("\n--- Instantaneous Phase ---");
    let inst_phase = instantaneous_phase(signal);
    println!("Instantaneous phase length: {}", inst_phase.len());
    let phase_min = inst_phase[..10]
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let phase_max = inst_phase[..10]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    println!("Phase range (first 10): [{phase_min:.4}, {phase_max:.4}]");

    println!("\n--- Instantaneous Frequency ---");
    let inst_freq = instantaneous_frequency(signal);
    println!("Instantaneous frequency length: {}", inst_freq.len());
    let mid = inst_freq.len() / 2;
    let window_half = 50_usize;
    let avg_freq: f64 = inst_freq[mid - window_half..mid + window_half]
        .iter()
        .sum::<f64>()
        / (window_half * 2) as f64;
    println!("Average instantaneous frequency (mid-signal): {avg_freq:.4} cycles/sample");
    println!(
        "As Hz at {sample_rate} sample rate: {:.1} Hz",
        avg_freq * sample_rate
    );
}

fn demo_periodogram(signal: &[f64], sample_rate: f64, n: usize) {
    println!("\n--- Periodogram ---");
    let psd = periodogram(signal);
    println!("PSD length (one-sided): {}", psd.len());
    let mut indexed: Vec<(usize, f64)> = psd.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let freq_res = sample_rate / n as f64;
    println!("Top 3 frequency peaks:");
    for (bin, power) in indexed.iter().take(3) {
        let freq = *bin as f64 * freq_res;
        println!("  Bin {bin:4} = {freq:7.1} Hz  (power = {power:.4})");
    }

    println!("\n--- Welch's Method PSD ---");
    let config = WelchConfig {
        segment_len: 512,
        overlap: 256,
        window: SpectralWindow::Hann,
    };
    let welch_psd = welch(signal, &config);
    println!("Welch PSD length: {}", welch_psd.len());
    let mut welch_indexed: Vec<(usize, f64)> =
        welch_psd.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    welch_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let welch_freq_res = sample_rate / config.segment_len as f64;
    println!("Top 2 frequency peaks (Welch):");
    for (bin, power) in welch_indexed.iter().take(2) {
        let freq = *bin as f64 * welch_freq_res;
        println!("  Bin {bin:4} = {freq:7.1} Hz  (power = {power:.6})");
    }
}

fn demo_coherence(signal: &[f64]) {
    println!("\n--- Cross-Spectral Density & Coherence ---");
    let signal2: Vec<f64> = signal.iter().map(|&x| x + 0.1).collect();
    let config = WelchConfig {
        segment_len: 512,
        overlap: 256,
        window: SpectralWindow::Hann,
    };
    let csd = cross_spectral_density(signal, &signal2, &config);
    println!("CSD length: {}", csd.len());

    let coh = coherence(signal, &signal2, &config);
    let avg_coh: f64 = coh.iter().sum::<f64>() / coh.len() as f64;
    println!("Average coherence (vs same signal + DC offset): {avg_coh:.4}");
    println!("Expected: high coherence (signals are nearly identical)");
}

fn demo_cepstrum(signal: &[f64]) {
    println!("\n--- Cepstral Analysis ---");
    let rc = real_cepstrum(signal);
    println!("Real cepstrum length: {}", rc.len());
    let max_q = rc.iter().enumerate().max_by(|a, b| {
        a.1.abs()
            .partial_cmp(&b.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some((q, v)) = max_q {
        println!("Peak quefrency: {q} (value = {v:.6})");
    }

    let cc = complex_cepstrum(signal);
    println!("Complex cepstrum length: {}", cc.len());
    let cc_max = cc.iter().enumerate().max_by(|a, b| {
        a.1.abs()
            .partial_cmp(&b.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some((q, v)) = cc_max {
        println!("Complex cepstrum peak quefrency: {q} (value = {v:.6})");
    }

    let mp = minimum_phase(signal);
    let orig_energy: f64 = signal.iter().map(|&x| x * x).sum::<f64>();
    let mp_energy: f64 = mp.iter().map(|&x| x * x).sum::<f64>();
    println!("\n--- Minimum Phase ---");
    println!("Original energy: {orig_energy:.4}");
    println!("Minimum phase energy: {mp_energy:.4}");
    println!("Energy ratio (mp/orig): {:.4}", mp_energy / orig_energy);
}

fn main() {
    let sample_rate = 8000.0f64;
    let n = 2048_usize;
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = f64::from(i as u32) / sample_rate;
            0.5f64.mul_add(
                (2.0 * std::f64::consts::PI * 880.0 * t).sin(),
                (2.0 * std::f64::consts::PI * 440.0 * t).sin(),
            )
        })
        .collect();

    println!("=== OxiFFT Signal Processing Demo ===\n");
    println!("Signal: 440 Hz + 880 Hz sine, {n} samples at {sample_rate} Hz");

    demo_hilbert_transform(&signal);
    demo_instantaneous_frequency(&signal, sample_rate);
    demo_periodogram(&signal, sample_rate, n);
    demo_coherence(&signal);
    demo_cepstrum(&signal);

    println!("\n=== Done ===");
}
