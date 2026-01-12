//! FFTW utility functions for comparison tests.
//!
//! This module is only available with the `fftw-compare` feature.

use fftw::array::AlignedVec;
use fftw::plan::{C2CPlan, C2CPlan64};
use fftw::types::{c64, Sign};
use oxifft::Complex;

/// Convert from oxifft Complex to fftw c64.
#[must_use]
pub const fn to_fftw_complex(c: Complex<f64>) -> c64 {
    c64::new(c.re, c.im)
}

/// Convert from fftw c64 to oxifft Complex.
#[must_use]
pub const fn from_fftw_complex(c: c64) -> Complex<f64> {
    Complex::new(c.re, c.im)
}

/// Compute FFT using FFTW for reference.
///
/// # Panics
///
/// Panics if FFTW plan creation or execution fails.
#[must_use]
pub fn fftw_forward(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let n = input.len();
    let mut input_vec: AlignedVec<c64> = AlignedVec::new(n);
    let mut output_vec: AlignedVec<c64> = AlignedVec::new(n);

    for (i, c) in input.iter().enumerate() {
        input_vec[i] = to_fftw_complex(*c);
    }

    let mut plan: C2CPlan64 =
        C2CPlan::aligned(&[n], Sign::Forward, fftw::types::Flag::ESTIMATE).unwrap();
    plan.c2c(&mut input_vec, &mut output_vec).unwrap();

    output_vec.iter().map(|c| from_fftw_complex(*c)).collect()
}

/// Compute inverse FFT using FFTW for reference (normalized).
///
/// # Panics
///
/// Panics if FFTW plan creation or execution fails.
#[must_use]
#[allow(clippy::cast_precision_loss)] // FFT sizes fit in f64 mantissa
pub fn fftw_inverse(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let n = input.len();
    let mut input_vec: AlignedVec<c64> = AlignedVec::new(n);
    let mut output_vec: AlignedVec<c64> = AlignedVec::new(n);

    for (i, c) in input.iter().enumerate() {
        input_vec[i] = to_fftw_complex(*c);
    }

    let mut plan: C2CPlan64 =
        C2CPlan::aligned(&[n], Sign::Backward, fftw::types::Flag::ESTIMATE).unwrap();
    plan.c2c(&mut input_vec, &mut output_vec).unwrap();

    // FFTW doesn't normalize, so we do it here
    let scale = 1.0 / n as f64;
    output_vec
        .iter()
        .map(|c| {
            let scaled = c64::new(c.re * scale, c.im * scale);
            from_fftw_complex(scaled)
        })
        .collect()
}
