//! End-to-end streaming STFT/COLA reconstruction test at 48 kHz.
//!
//! Validates that `stft()` + `istft()` round-trip on 10 seconds of white noise
//! at 48 kHz achieves SNR ≥ 60 dB in the central region, within real-time
//! (wall-clock < 10 s).

#![cfg(feature = "streaming")]

use oxifft::streaming::{istft, stft, WindowFunction};

const SAMPLE_RATE: usize = 48_000;
const DURATION_S: usize = 10;
const N_SAMPLES: usize = SAMPLE_RATE * DURATION_S; // 480_000
const WINDOW: usize = 2048;
const HOP: usize = 512; // hop = WINDOW / 4

/// Splitmix64 PRNG — no external dependency.
const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Map a `u64` PRNG output to `f32` in `[-1.0, 1.0]` without precision-loss casts.
///
/// Constructs an IEEE 754 f32 in `[1.0, 2.0)` from the top 23 mantissa bits, then
/// maps to `[-1.0, 1.0]` via subtraction, so no integer→float precision is lost.
fn u64_to_f32_bipolar(bits: u64) -> f32 {
    // Extract the top 23 bits (f32 mantissa width).
    let mantissa = ((bits >> 41) as u32) & 0x007f_ffff;
    // Form an f32 in [1.0, 2.0): exponent bias 127, sign 0.
    let f_one_to_two = f32::from_bits(0x3f80_0000_u32 | mantissa);
    // Shift to [-1.0, 1.0].
    f_one_to_two * 2.0 - 3.0
}

/// Generate deterministic white noise in `[-1.0, 1.0]` using splitmix64.
fn gen_white_noise_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            let bits = splitmix64(&mut state);
            u64_to_f32_bipolar(bits)
        })
        .collect()
}

#[test]
fn streaming_48khz_snr_gate() {
    // Generate 10 s of deterministic white noise at 48 kHz.
    let samples: Vec<f32> = gen_white_noise_f32(N_SAMPLES, 0xdead_beef_cafe_babe);

    let t_start = std::time::Instant::now();

    // Forward STFT: module-level function avoids the ring-buffer footgun
    // (StreamingFft has capacity=fft_size and silently drops old samples
    // when fed the whole signal at once).
    let spectrogram = stft(&samples, WINDOW, HOP, WindowFunction::Hann);

    // Inverse STFT: overlap-add with Σw² normalisation (built-in WOLA).
    let reconstructed: Vec<f32> = istft(&spectrogram, HOP, WindowFunction::Hann);

    let wall = t_start.elapsed();

    // The output may be shorter than N_SAMPLES if the last incomplete frame
    // was not produced; guard against that.
    let recon_len = reconstructed.len();
    let valid_end = recon_len.min(N_SAMPLES);

    // Skip first/last WINDOW samples to avoid edge-effect transients.
    let edge = WINDOW;
    assert!(
        valid_end > 2 * edge,
        "Reconstructed signal too short ({recon_len}) to measure SNR — \
         need at least {} samples",
        2 * edge + 1
    );

    let s = &samples[edge..valid_end - edge];
    let r = &reconstructed[edge..valid_end - edge];

    let sig_energy: f32 = s.iter().map(|x| x * x).sum();
    let err_energy: f32 = s
        .iter()
        .zip(r.iter())
        .map(|(orig, rec)| (orig - rec).powi(2))
        .sum();

    assert!(
        sig_energy > 0.0,
        "Signal energy is zero — noise generation broken"
    );
    assert!(
        err_energy.is_finite(),
        "Error energy is not finite — reconstruction produced NaN/Inf"
    );

    let snr_db = if err_energy == 0.0 {
        f32::INFINITY
    } else {
        10.0 * (sig_energy / err_energy).log10()
    };

    // DURATION_S = 10, which fits in f64 exactly.
    let duration_secs = 10.0_f64;
    let rtf = duration_secs / wall.as_secs_f64();

    println!("[streaming_48khz] SNR={snr_db:.2}dB wall={wall:?} RTF={rtf:.1}x");

    // Correctness gate: SNR ≥ 60 dB.
    assert!(
        snr_db >= 60.0,
        "STFT/ISTFT round-trip SNR {snr_db:.2} dB is below the 60 dB gate. \
         This indicates a bug in the WOLA normalization or window configuration."
    );

    // Performance gate: must finish in < 10 s (real-time factor > 1.0×).
    // Skipped in debug builds — MIRI, sanitizers, and parallel nextest runs
    // make wall time unreliable. Run with `--release` for meaningful timing.
    #[cfg(not(debug_assertions))]
    assert!(
        wall.as_secs_f64() < 10.0,
        "STFT/ISTFT processing took {wall:?}, exceeding the 10 s real-time gate \
         (RTF = {rtf:.2}×)"
    );
}
