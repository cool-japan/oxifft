//! Signal processing functions built on FFT.
//!
//! This module provides common signal processing operations:
//! - Hilbert transform and analytic signal
//! - Power spectral density (Welch's method)
//! - Cepstral analysis
//!
//! Requires the `signal` feature flag (depends on `std` for math functions).
//!
//! # Example
//!
//! ```ignore
//! use oxifft::signal::{hilbert, envelope, welch, WelchConfig, SpectralWindow};
//!
//! // Analytic signal
//! let signal: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.1).sin()).collect();
//! let analytic = hilbert(&signal);
//! let env = envelope(&signal);
//!
//! // Power spectral density
//! let config = WelchConfig { segment_len: 256, overlap: 128, window: SpectralWindow::Hann };
//! let psd = welch(&signal, &config);
//! ```

mod cepstrum;
mod hilbert;
mod resample;
mod spectral;

pub use cepstrum::{complex_cepstrum, minimum_phase, real_cepstrum};
pub use hilbert::{envelope, hilbert, instantaneous_frequency, instantaneous_phase};
pub use resample::{resample, resample_to};
pub use spectral::{
    coherence, cross_spectral_density, periodogram, welch, SpectralWindow, WelchConfig,
};
