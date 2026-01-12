//! Utility functions for benchmarks and tests.

#![allow(clippy::doc_markdown)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::imprecise_flops)]
#![allow(clippy::ptr_as_ptr)]
#![allow(clippy::missing_const_for_fn)]

use oxifft::Complex;

/// Convert from oxifft Complex to `num_complex::Complex`.
#[must_use]
pub fn to_num_complex(c: Complex<f64>) -> num_complex::Complex<f64> {
    num_complex::Complex::new(c.re, c.im)
}

/// Convert from num_complex::Complex to oxifft Complex.
pub fn from_num_complex(c: num_complex::Complex<f64>) -> Complex<f64> {
    Complex::new(c.re, c.im)
}

/// Compare two complex numbers within tolerance (relative error).
pub fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
    let diff = ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)).sqrt();
    let mag_a = (a.re.powi(2) + a.im.powi(2)).sqrt();
    let mag_b = (b.re.powi(2) + b.im.powi(2)).sqrt();
    let mag = mag_a.max(mag_b).max(1.0);
    diff / mag < eps
}

/// Generate test input data.
pub fn generate_input(n: usize) -> Vec<Complex<f64>> {
    (0..n)
        .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
        .collect()
}

/// Compute FFT using rustfft for reference.
pub fn rustfft_forward(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    use rustfft::FftPlanner;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(input.len());
    let mut buffer: Vec<num_complex::Complex<f64>> =
        input.iter().map(|c| to_num_complex(*c)).collect();
    fft.process(&mut buffer);
    buffer.iter().map(|c| from_num_complex(*c)).collect()
}

/// Compute inverse FFT using rustfft for reference (normalized).
pub fn rustfft_inverse(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    use rustfft::FftPlanner;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(input.len());
    let mut buffer: Vec<num_complex::Complex<f64>> =
        input.iter().map(|c| to_num_complex(*c)).collect();
    fft.process(&mut buffer);
    let scale = 1.0 / input.len() as f64;
    buffer
        .iter()
        .map(|c| {
            let scaled = num_complex::Complex::new(c.re * scale, c.im * scale);
            from_num_complex(scaled)
        })
        .collect()
}
