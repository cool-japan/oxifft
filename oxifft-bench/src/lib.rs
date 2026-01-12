//! `OxiFFT` Benchmarks and Comparison Tests
//!
//! This crate provides benchmarks comparing `OxiFFT` against other FFT implementations:
//! - rustfft (always available)
//! - FFTW (requires `fftw-compare` feature and libfftw3 installed)
//!
//! # Features
//!
//! - `fftw-compare`: Enable FFTW comparison tests (requires libfftw3)
//!
//! # Running Benchmarks
//!
//! ```bash
//! # Basic benchmarks (rustfft comparison)
//! cargo bench -p oxifft-bench
//!
//! # With FFTW comparison (requires libfftw3)
//! cargo bench -p oxifft-bench --features fftw-compare
//! ```

pub mod utils;

#[cfg(feature = "fftw-compare")]
pub mod fftw_utils;
