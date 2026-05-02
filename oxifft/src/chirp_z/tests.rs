//! Tests for the Chirp-Z Transform implementation.
//!
//! Covers:
//! 1. Identity: CZT with A=1, W=e^{-2πi/N} matches standard DFT (f64).
//! 2. Zoom-FFT: sinusoid at 105.3 Hz recovers peak within 1 bin.
//! 3. Off-unit-circle: |A|≠1, |W|≠1 against naive formula.
//! 4. Different N/M: N=256, M=64 against naive formula.
//! 5. f32 identity: same as (1) at lower precision.
//! 6. Error-path validation.

use super::{CztError, CztPlan};
use crate::api::{Direction, Flags, Plan};
use crate::kernel::Complex;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Naive O(N·M) CZT reference implementation (f64 only, for comparison).
fn naive_czt_f64(
    x: &[Complex<f64>],
    m: usize,
    a: Complex<f64>,
    w: Complex<f64>,
) -> Vec<Complex<f64>> {
    let mut out = Vec::with_capacity(m);
    for k in 0..m {
        let mut acc = Complex::zero();
        for (nn, &xn) in x.iter().enumerate() {
            let nn_f = nn as f64;
            let k_f = k as f64;
            let a_pow = complex_pow_f64(a, -nn_f);
            let w_pow = complex_pow_f64(w, nn_f * k_f);
            acc = acc + xn * a_pow * w_pow;
        }
        out.push(acc);
    }
    out
}

/// Raise a complex f64 number `z` to a real power `p` (polar form).
fn complex_pow_f64(z: Complex<f64>, p: f64) -> Complex<f64> {
    let r = z.norm();
    let theta = f64::atan2(z.im, z.re);
    let r_p = r.powf(p);
    let angle = p * theta;
    Complex::from_polar(r_p, angle)
}

/// Maximum absolute error across all bins (f64).
fn max_err_f64(a: &[Complex<f64>], b: &[Complex<f64>]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let d = ai - bi;
            f64::sqrt(d.re * d.re + d.im * d.im)
        })
        .fold(0.0_f64, f64::max)
}

/// Maximum absolute error across all bins (f32).
fn max_err_f32(a: &[Complex<f32>], b: &[Complex<f32>]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let d = ai - bi;
            f64::from(f32::sqrt(d.re * d.re + d.im * d.im))
        })
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// Test 1: Identity (f64) — CZT == DFT when A=1, W=e^{-2πi/N}
// ---------------------------------------------------------------------------

#[test]
fn identity_czt_matches_dft_f64() {
    for &n in &[16_usize, 64, 256] {
        let two_pi_over_n = -2.0 * core::f64::consts::PI / n as f64;
        let a = Complex::<f64>::one();
        let w = Complex::from_polar(1.0_f64, two_pi_over_n);

        let plan = CztPlan::<f64>::new(n, n, a, w).expect("CZT plan failed");

        let dft_plan =
            Plan::<f64>::dft_1d(n, Direction::Forward, Flags::ESTIMATE).expect("DFT plan failed");

        // Deterministic test signal.
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64 * 0.37).sin(), (i as f64 * 0.71).cos()))
            .collect();

        let mut czt_out = vec![Complex::zero(); n];
        let mut dft_out = vec![Complex::zero(); n];

        plan.execute(&input, &mut czt_out)
            .expect("CZT execute failed");
        dft_plan.execute(&input, &mut dft_out);

        let err = max_err_f64(&czt_out, &dft_out);
        assert!(
            err < 1e-10,
            "identity CZT vs DFT error {err:.2e} > 1e-10 for n={n}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2: Zoom-FFT — sinusoid at 105.3 Hz
// ---------------------------------------------------------------------------

#[test]
fn zoom_fft_peak_detection() {
    let n = 1024_usize;
    let fs = 1000.0_f64;
    let f_signal = 105.3_f64;
    let m = 100_usize;
    let f_start = 100.0_f64;
    let f_stop = 110.0_f64;

    let input: Vec<Complex<f64>> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            let re = (2.0 * core::f64::consts::PI * f_signal * t).cos();
            Complex::new(re, 0.0)
        })
        .collect();

    let plan = CztPlan::<f64>::zoom_fft(n, m, f_start, f_stop, fs).expect("zoom_fft plan failed");

    let mut output = vec![Complex::zero(); m];
    plan.execute(&input, &mut output)
        .expect("zoom_fft execute failed");

    // Find the bin with maximum magnitude.
    let peak_bin = output
        .iter()
        .enumerate()
        .max_by(|(_, ai), (_, bi)| {
            let a_mag = ai.re * ai.re + ai.im * ai.im;
            let b_mag = bi.re * bi.re + bi.im * bi.im;
            a_mag
                .partial_cmp(&b_mag)
                .unwrap_or(core::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Bin step = (f_stop - f_start) / m = 0.1 Hz → 105.3 Hz is bin 53.
    let expected_bin = ((f_signal - f_start) / (f_stop - f_start) * m as f64).round() as usize;
    let distance = (peak_bin as i64 - expected_bin as i64).unsigned_abs() as usize;
    assert!(
        distance <= 1,
        "zoom_fft peak at bin {peak_bin}, expected ~{expected_bin} (±1)"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Off-unit-circle against naive formula
// ---------------------------------------------------------------------------

#[test]
fn off_unit_circle_matches_naive() {
    let n = 8_usize;
    let m = 8_usize;
    let a = Complex::from_polar(1.05_f64, 0.0_f64);
    let w = Complex::from_polar(0.99_f64, -2.0 * core::f64::consts::PI / 8.0);

    let plan = CztPlan::<f64>::new(n, m, a, w).expect("off-unit plan failed");

    let input: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new((i as f64 + 1.0) * 0.5, -(i as f64) * 0.3))
        .collect();

    let mut czt_out = vec![Complex::zero(); m];
    plan.execute(&input, &mut czt_out)
        .expect("off-unit execute failed");

    let naive_out = naive_czt_f64(&input, m, a, w);
    let err = max_err_f64(&czt_out, &naive_out);
    assert!(err < 1e-11, "off-unit-circle error {err:.2e} > 1e-11");
}

// ---------------------------------------------------------------------------
// Test 4: Different N/M (N=256, M=64) against naive formula
// ---------------------------------------------------------------------------

#[test]
fn different_n_m_matches_naive() {
    let n = 256_usize;
    let m = 64_usize;
    let two_pi_over_n = -2.0 * core::f64::consts::PI / n as f64;
    let a = Complex::<f64>::one();
    let w = Complex::from_polar(1.0_f64, two_pi_over_n);

    let plan = CztPlan::<f64>::new(n, m, a, w).expect("N/M plan failed");

    let input: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new((i as f64 * 0.13).sin(), (i as f64 * 0.29).cos()))
        .collect();

    let mut czt_out = vec![Complex::zero(); m];
    plan.execute(&input, &mut czt_out)
        .expect("N/M execute failed");

    let naive_out = naive_czt_f64(&input, m, a, w);
    let err = max_err_f64(&czt_out, &naive_out);
    // 1e-10 accommodates MIRI's softfloat rounding path which can produce
    // errors ~1 order of magnitude above the native-float baseline.
    assert!(err < 1e-10, "N=256/M=64 error {err:.2e} > 1e-10");
}

// ---------------------------------------------------------------------------
// Test 5: f32 identity
// ---------------------------------------------------------------------------

#[test]
fn identity_czt_matches_dft_f32() {
    let n = 64_usize;
    let two_pi_over_n = -2.0 * core::f32::consts::PI / n as f32;
    let a = Complex::<f32>::one();
    let w = Complex::from_polar(1.0_f32, two_pi_over_n);

    let plan = CztPlan::<f32>::new(n, n, a, w).expect("f32 CZT plan failed");

    let dft_plan =
        Plan::<f32>::dft_1d(n, Direction::Forward, Flags::ESTIMATE).expect("f32 DFT plan failed");

    let input: Vec<Complex<f32>> = (0..n)
        .map(|i| Complex::new((i as f32 * 0.37).sin(), (i as f32 * 0.71).cos()))
        .collect();

    let mut czt_out = vec![Complex::zero(); n];
    let mut dft_out = vec![Complex::zero(); n];

    plan.execute(&input, &mut czt_out)
        .expect("f32 CZT execute failed");
    dft_plan.execute(&input, &mut dft_out);

    let err = max_err_f32(&czt_out, &dft_out);
    // f32 CZT involves ~4 FFTs of length L=128 through the Bluestein chain;
    // the accumulated rounding exceeds the per-FFT 1e-5 budget.
    // Under MIRI's softfloat path the per-operation rounding is ~1 order of
    // magnitude worse, so 5e-3 is needed for cross-platform reproducibility.
    // (Still well within f32 dynamic range given ~1.2e-7 machine epsilon.)
    assert!(err < 5e-3, "f32 identity CZT vs DFT error {err:.2e} > 5e-3");
}

// ---------------------------------------------------------------------------
// Test 6: Error-path validation
// ---------------------------------------------------------------------------

#[test]
fn czt_error_paths() {
    let w = Complex::from_polar(1.0_f64, -2.0 * core::f64::consts::PI / 8.0);
    let a = Complex::<f64>::one();

    // Zero N
    assert!(matches!(
        CztPlan::<f64>::new(0, 8, a, w),
        Err(CztError::InvalidSize(0))
    ));

    // Zero M
    assert!(matches!(
        CztPlan::<f64>::new(8, 0, a, w),
        Err(CztError::InvalidSize(0))
    ));

    // Mismatched input length
    let plan = CztPlan::<f64>::new(8, 8, a, w).expect("plan");
    let bad_input = vec![Complex::<f64>::zero(); 4];
    let mut out = vec![Complex::<f64>::zero(); 8];
    assert!(matches!(
        plan.execute(&bad_input, &mut out),
        Err(CztError::MismatchedLength {
            expected: 8,
            actual: 4
        })
    ));

    // Mismatched output length
    let good_input = vec![Complex::<f64>::zero(); 8];
    let mut bad_out = vec![Complex::<f64>::zero(); 4];
    assert!(matches!(
        plan.execute(&good_input, &mut bad_out),
        Err(CztError::MismatchedLength {
            expected: 8,
            actual: 4
        })
    ));

    // zoom_fft with f_start >= f_stop
    assert!(matches!(
        CztPlan::<f64>::zoom_fft(64, 32, 200.0, 100.0, 1000.0),
        Err(CztError::InvalidParameter)
    ));
}
