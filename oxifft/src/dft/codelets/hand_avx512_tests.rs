//! Tests for hand-tuned AVX-512 codelets (sizes 16, 32, 64; f64 and f32).
//!
//! All AVX-512 paths are guarded by `is_x86_feature_detected!("avx512f")` so
//! the tests skip cleanly on non-AVX-512 hardware.

#![cfg(test)]
#![cfg(target_arch = "x86_64")]

use core::sync::atomic::{AtomicBool, Ordering};

use super::hand_avx512::{
    dispatch_hand_avx512_size16_f32, dispatch_hand_avx512_size16_f64,
    dispatch_hand_avx512_size32_f32, dispatch_hand_avx512_size32_f64,
    dispatch_hand_avx512_size64_f32, dispatch_hand_avx512_size64_f64,
};
use crate::kernel::Complex;

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher hit sentinels (set inside codelets under #[cfg(test)])
// ─────────────────────────────────────────────────────────────────────────────

/// Set to `true` inside `hand_avx512_size16_f64` on first call (test builds only).
pub static HAND_AVX512_HIT_16_F64: AtomicBool = AtomicBool::new(false);
/// Set to `true` inside `hand_avx512_size32_f64` on first call (test builds only).
pub static HAND_AVX512_HIT_32_F64: AtomicBool = AtomicBool::new(false);
/// Set to `true` inside `hand_avx512_size64_f64` on first call (test builds only).
pub static HAND_AVX512_HIT_64_F64: AtomicBool = AtomicBool::new(false);
/// Set to `true` inside `hand_avx512_size16_f32` on first call (test builds only).
pub static HAND_AVX512_HIT_16_F32: AtomicBool = AtomicBool::new(false);
/// Set to `true` inside `hand_avx512_size32_f32` on first call (test builds only).
pub static HAND_AVX512_HIT_32_F32: AtomicBool = AtomicBool::new(false);
/// Set to `true` inside `hand_avx512_size64_f32` on first call (test builds only).
pub static HAND_AVX512_HIT_64_F32: AtomicBool = AtomicBool::new(false);

// ─────────────────────────────────────────────────────────────────────────────
// Reference implementations
// ─────────────────────────────────────────────────────────────────────────────

/// Naive O(n²) DFT reference for f64.  `sign=-1` → forward, `sign=+1` → inverse.
fn naive_dft_f64(x: &[Complex<f64>], sign: i32) -> Vec<Complex<f64>> {
    let n = x.len();
    (0..n)
        .map(|k| {
            x.iter()
                .enumerate()
                .fold(Complex::new(0.0_f64, 0.0), |acc, (j, &xj)| {
                    let angle =
                        sign as f64 * 2.0 * core::f64::consts::PI * (k * j) as f64 / n as f64;
                    let w = Complex::new(angle.cos(), angle.sin());
                    acc + xj * w
                })
        })
        .collect()
}

/// Naive O(n²) DFT reference for f32.
fn naive_dft_f32(x: &[Complex<f32>], sign: i32) -> Vec<Complex<f32>> {
    let n = x.len();
    (0..n)
        .map(|k| {
            x.iter()
                .enumerate()
                .fold(Complex::new(0.0_f32, 0.0), |acc, (j, &xj)| {
                    let angle =
                        sign as f32 * 2.0 * core::f32::consts::PI * (k * j) as f32 / n as f32;
                    let w = Complex::new(angle.cos(), angle.sin());
                    acc + xj * w
                })
        })
        .collect()
}

fn approx_eq_f64_abs(a: &[Complex<f64>], b: &[Complex<f64>], tol: f64) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| (x.re - y.re).abs() < tol && (x.im - y.im).abs() < tol)
}

fn approx_eq_f32_abs(a: &[Complex<f32>], b: &[Complex<f32>], tol: f32) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| (x.re - y.re).abs() < tol && (x.im - y.im).abs() < tol)
}

/// Build a deterministic pseudo-random input vector of length `n` using f64.
fn test_input_f64(n: usize) -> Vec<Complex<f64>> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            Complex::new(t.sin() * 1.3 + 0.7, t.cos() * 0.9 - 0.4)
        })
        .collect()
}

/// Build a deterministic pseudo-random input vector of length `n` using f32.
fn test_input_f32(n: usize) -> Vec<Complex<f32>> {
    (0..n)
        .map(|i| {
            let t = i as f32;
            Complex::new(t.sin() * 1.3_f32 + 0.7_f32, t.cos() * 0.9_f32 - 0.4_f32)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Size-16 f64 tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hand_avx512_size16_f64_forward_correctness() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let input = test_input_f64(16);
    let expected = naive_dft_f64(&input, -1);
    let mut data = input;
    dispatch_hand_avx512_size16_f64(&mut data, -1);
    let tol = 1e-10_f64 * 16.0;
    assert!(
        approx_eq_f64_abs(&data, &expected, tol),
        "size-16 f64 forward: max_err={:.2e}",
        data.iter()
            .zip(expected.iter())
            .map(|(a, b)| { (a.re - b.re).abs().max((a.im - b.im).abs()) })
            .fold(0.0_f64, f64::max)
    );
}

#[test]
fn hand_avx512_size16_f64_inverse_correctness() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let input = test_input_f64(16);
    let expected = naive_dft_f64(&input, 1);
    let mut data = input;
    dispatch_hand_avx512_size16_f64(&mut data, 1);
    let tol = 1e-10_f64 * 16.0;
    assert!(
        approx_eq_f64_abs(&data, &expected, tol),
        "size-16 f64 inverse: max_err={:.2e}",
        data.iter()
            .zip(expected.iter())
            .map(|(a, b)| { (a.re - b.re).abs().max((a.im - b.im).abs()) })
            .fold(0.0_f64, f64::max)
    );
}

#[test]
fn hand_avx512_size16_f64_roundtrip() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let original = test_input_f64(16);
    let mut data = original.clone();
    dispatch_hand_avx512_size16_f64(&mut data, -1);
    dispatch_hand_avx512_size16_f64(&mut data, 1);
    let n = 16.0_f64;
    let tol = 1e-10_f64 * n;
    for (i, (got, orig)) in data.iter().zip(original.iter()).enumerate() {
        let re_err = (got.re / n - orig.re).abs();
        let im_err = (got.im / n - orig.im).abs();
        assert!(
            re_err < tol && im_err < tol,
            "size-16 f64 roundtrip idx {i}: re_err={re_err:.2e} im_err={im_err:.2e}"
        );
    }
}

#[test]
fn hand_avx512_size16_f64_parity_vs_scalar() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let input = test_input_f64(16);
    let expected = naive_dft_f64(&input, -1);
    let mut data = input;
    dispatch_hand_avx512_size16_f64(&mut data, -1);
    let tol = 1e-12_f64 * 16.0;
    assert!(
        approx_eq_f64_abs(&data, &expected, tol),
        "size-16 f64 parity vs scalar (rel 1e-12)"
    );
}

#[test]
fn hand_avx512_size16_f64_dispatcher_hit() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    HAND_AVX512_HIT_16_F64.store(false, Ordering::Relaxed);
    let mut data = test_input_f64(16);
    dispatch_hand_avx512_size16_f64(&mut data, -1);
    assert!(
        HAND_AVX512_HIT_16_F64.load(Ordering::Relaxed),
        "hand_avx512_size16_f64 was not called"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Size-32 f64 tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hand_avx512_size32_f64_forward_correctness() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let input = test_input_f64(32);
    let expected = naive_dft_f64(&input, -1);
    let mut data = input;
    dispatch_hand_avx512_size32_f64(&mut data, -1);
    let tol = 1e-10_f64 * 32.0;
    assert!(
        approx_eq_f64_abs(&data, &expected, tol),
        "size-32 f64 forward: max_err={:.2e}",
        data.iter()
            .zip(expected.iter())
            .map(|(a, b)| { (a.re - b.re).abs().max((a.im - b.im).abs()) })
            .fold(0.0_f64, f64::max)
    );
}

#[test]
fn hand_avx512_size32_f64_roundtrip() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let original = test_input_f64(32);
    let mut data = original.clone();
    dispatch_hand_avx512_size32_f64(&mut data, -1);
    dispatch_hand_avx512_size32_f64(&mut data, 1);
    let n = 32.0_f64;
    let tol = 1e-10_f64 * n;
    for (i, (got, orig)) in data.iter().zip(original.iter()).enumerate() {
        let re_err = (got.re / n - orig.re).abs();
        let im_err = (got.im / n - orig.im).abs();
        assert!(
            re_err < tol && im_err < tol,
            "size-32 f64 roundtrip idx {i}: re_err={re_err:.2e} im_err={im_err:.2e}"
        );
    }
}

#[test]
fn hand_avx512_size32_f64_parity_vs_scalar() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let input = test_input_f64(32);
    let expected = naive_dft_f64(&input, -1);
    let mut data = input;
    dispatch_hand_avx512_size32_f64(&mut data, -1);
    let tol = 1e-12_f64 * 32.0;
    assert!(
        approx_eq_f64_abs(&data, &expected, tol),
        "size-32 f64 parity vs scalar (rel 1e-12)"
    );
}

#[test]
fn hand_avx512_size32_f64_dispatcher_hit() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    HAND_AVX512_HIT_32_F64.store(false, Ordering::Relaxed);
    let mut data = test_input_f64(32);
    dispatch_hand_avx512_size32_f64(&mut data, -1);
    assert!(
        HAND_AVX512_HIT_32_F64.load(Ordering::Relaxed),
        "hand_avx512_size32_f64 was not called"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Size-64 f64 tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hand_avx512_size64_f64_forward_correctness() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let input = test_input_f64(64);
    let expected = naive_dft_f64(&input, -1);
    let mut data = input;
    dispatch_hand_avx512_size64_f64(&mut data, -1);
    let tol = 1e-10_f64 * 64.0;
    assert!(
        approx_eq_f64_abs(&data, &expected, tol),
        "size-64 f64 forward: max_err={:.2e}",
        data.iter()
            .zip(expected.iter())
            .map(|(a, b)| { (a.re - b.re).abs().max((a.im - b.im).abs()) })
            .fold(0.0_f64, f64::max)
    );
}

#[test]
fn hand_avx512_size64_f64_roundtrip() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let original = test_input_f64(64);
    let mut data = original.clone();
    dispatch_hand_avx512_size64_f64(&mut data, -1);
    dispatch_hand_avx512_size64_f64(&mut data, 1);
    let n = 64.0_f64;
    let tol = 1e-10_f64 * n;
    for (i, (got, orig)) in data.iter().zip(original.iter()).enumerate() {
        let re_err = (got.re / n - orig.re).abs();
        let im_err = (got.im / n - orig.im).abs();
        assert!(
            re_err < tol && im_err < tol,
            "size-64 f64 roundtrip idx {i}: re_err={re_err:.2e} im_err={im_err:.2e}"
        );
    }
}

#[test]
fn hand_avx512_size64_f64_parity_vs_scalar() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let input = test_input_f64(64);
    let expected = naive_dft_f64(&input, -1);
    let mut data = input;
    dispatch_hand_avx512_size64_f64(&mut data, -1);
    let tol = 1e-12_f64 * 64.0;
    assert!(
        approx_eq_f64_abs(&data, &expected, tol),
        "size-64 f64 parity vs scalar (rel 1e-12)"
    );
}

#[test]
fn hand_avx512_size64_f64_dispatcher_hit() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    HAND_AVX512_HIT_64_F64.store(false, Ordering::Relaxed);
    let mut data = test_input_f64(64);
    dispatch_hand_avx512_size64_f64(&mut data, -1);
    assert!(
        HAND_AVX512_HIT_64_F64.load(Ordering::Relaxed),
        "hand_avx512_size64_f64 was not called"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Size-16 f32 tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hand_avx512_size16_f32_forward_correctness() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let input = test_input_f32(16);
    let expected = naive_dft_f32(&input, -1);
    let mut data = input;
    dispatch_hand_avx512_size16_f32(&mut data, -1);
    let tol = 1e-5_f32 * 16.0;
    assert!(
        approx_eq_f32_abs(&data, &expected, tol),
        "size-16 f32 forward: max_err={:.2e}",
        data.iter()
            .zip(expected.iter())
            .map(|(a, b)| { (a.re - b.re).abs().max((a.im - b.im).abs()) })
            .fold(0.0_f32, f32::max)
    );
}

#[test]
fn hand_avx512_size16_f32_roundtrip() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let original = test_input_f32(16);
    let mut data = original.clone();
    dispatch_hand_avx512_size16_f32(&mut data, -1);
    dispatch_hand_avx512_size16_f32(&mut data, 1);
    let n = 16.0_f32;
    let tol = 1e-5_f32 * n;
    for (i, (got, orig)) in data.iter().zip(original.iter()).enumerate() {
        let re_err = (got.re / n - orig.re).abs();
        let im_err = (got.im / n - orig.im).abs();
        assert!(
            re_err < tol && im_err < tol,
            "size-16 f32 roundtrip idx {i}: re_err={re_err:.2e} im_err={im_err:.2e}"
        );
    }
}

#[test]
fn hand_avx512_size16_f32_dispatcher_hit() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    HAND_AVX512_HIT_16_F32.store(false, Ordering::Relaxed);
    let mut data = test_input_f32(16);
    dispatch_hand_avx512_size16_f32(&mut data, -1);
    assert!(
        HAND_AVX512_HIT_16_F32.load(Ordering::Relaxed),
        "hand_avx512_size16_f32 was not called"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Size-32 f32 tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hand_avx512_size32_f32_forward_correctness() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let input = test_input_f32(32);
    let expected = naive_dft_f32(&input, -1);
    let mut data = input;
    dispatch_hand_avx512_size32_f32(&mut data, -1);
    let tol = 1e-5_f32 * 32.0;
    assert!(
        approx_eq_f32_abs(&data, &expected, tol),
        "size-32 f32 forward: max_err={:.2e}",
        data.iter()
            .zip(expected.iter())
            .map(|(a, b)| { (a.re - b.re).abs().max((a.im - b.im).abs()) })
            .fold(0.0_f32, f32::max)
    );
}

#[test]
fn hand_avx512_size32_f32_roundtrip() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let original = test_input_f32(32);
    let mut data = original.clone();
    dispatch_hand_avx512_size32_f32(&mut data, -1);
    dispatch_hand_avx512_size32_f32(&mut data, 1);
    let n = 32.0_f32;
    let tol = 1e-5_f32 * n;
    for (i, (got, orig)) in data.iter().zip(original.iter()).enumerate() {
        let re_err = (got.re / n - orig.re).abs();
        let im_err = (got.im / n - orig.im).abs();
        assert!(
            re_err < tol && im_err < tol,
            "size-32 f32 roundtrip idx {i}: re_err={re_err:.2e} im_err={im_err:.2e}"
        );
    }
}

#[test]
fn hand_avx512_size32_f32_dispatcher_hit() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    HAND_AVX512_HIT_32_F32.store(false, Ordering::Relaxed);
    let mut data = test_input_f32(32);
    dispatch_hand_avx512_size32_f32(&mut data, -1);
    assert!(
        HAND_AVX512_HIT_32_F32.load(Ordering::Relaxed),
        "hand_avx512_size32_f32 was not called"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Size-64 f32 tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hand_avx512_size64_f32_forward_correctness() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let input = test_input_f32(64);
    let expected = naive_dft_f32(&input, -1);
    let mut data = input;
    dispatch_hand_avx512_size64_f32(&mut data, -1);
    let tol = 1e-5_f32 * 64.0;
    assert!(
        approx_eq_f32_abs(&data, &expected, tol),
        "size-64 f32 forward: max_err={:.2e}",
        data.iter()
            .zip(expected.iter())
            .map(|(a, b)| { (a.re - b.re).abs().max((a.im - b.im).abs()) })
            .fold(0.0_f32, f32::max)
    );
}

#[test]
fn hand_avx512_size64_f32_roundtrip() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let original = test_input_f32(64);
    let mut data = original.clone();
    dispatch_hand_avx512_size64_f32(&mut data, -1);
    dispatch_hand_avx512_size64_f32(&mut data, 1);
    let n = 64.0_f32;
    let tol = 1e-5_f32 * n;
    for (i, (got, orig)) in data.iter().zip(original.iter()).enumerate() {
        let re_err = (got.re / n - orig.re).abs();
        let im_err = (got.im / n - orig.im).abs();
        assert!(
            re_err < tol && im_err < tol,
            "size-64 f32 roundtrip idx {i}: re_err={re_err:.2e} im_err={im_err:.2e}"
        );
    }
}

#[test]
fn hand_avx512_size64_f32_dispatcher_hit() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    HAND_AVX512_HIT_64_F32.store(false, Ordering::Relaxed);
    let mut data = test_input_f32(64);
    dispatch_hand_avx512_size64_f32(&mut data, -1);
    assert!(
        HAND_AVX512_HIT_64_F32.load(Ordering::Relaxed),
        "hand_avx512_size64_f32 was not called"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Impulse response tests (verifies bin-by-bin correctness)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hand_avx512_size16_f64_impulse() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    // Impulse at index 0: forward DFT should give all-ones.
    let mut data: Vec<Complex<f64>> = (0..16)
        .map(|i| {
            if i == 0 {
                Complex::new(1.0, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            }
        })
        .collect();
    dispatch_hand_avx512_size16_f64(&mut data, -1);
    for (k, v) in data.iter().enumerate() {
        assert!(
            (v.re - 1.0).abs() < 1e-12 && v.im.abs() < 1e-12,
            "size-16 f64 impulse bin {k}: {v:?}"
        );
    }
}

#[test]
fn hand_avx512_size32_f64_impulse() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let mut data: Vec<Complex<f64>> = (0..32)
        .map(|i| {
            if i == 0 {
                Complex::new(1.0, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            }
        })
        .collect();
    dispatch_hand_avx512_size32_f64(&mut data, -1);
    for (k, v) in data.iter().enumerate() {
        assert!(
            (v.re - 1.0).abs() < 1e-12 && v.im.abs() < 1e-12,
            "size-32 f64 impulse bin {k}: {v:?}"
        );
    }
}

#[test]
fn hand_avx512_size64_f64_impulse() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let mut data: Vec<Complex<f64>> = (0..64)
        .map(|i| {
            if i == 0 {
                Complex::new(1.0, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            }
        })
        .collect();
    dispatch_hand_avx512_size64_f64(&mut data, -1);
    for (k, v) in data.iter().enumerate() {
        assert!(
            (v.re - 1.0).abs() < 1e-12 && v.im.abs() < 1e-12,
            "size-64 f64 impulse bin {k}: {v:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DC test (all-ones input → energy at bin 0 only)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hand_avx512_size16_f64_dc() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let mut data: Vec<Complex<f64>> = (0..16).map(|_| Complex::new(1.0, 0.0)).collect();
    dispatch_hand_avx512_size16_f64(&mut data, -1);
    assert!((data[0].re - 16.0).abs() < 1e-11, "DC bin 0: {:?}", data[0]);
    for (k, v) in data.iter().enumerate().skip(1) {
        assert!(
            v.re.abs() < 1e-11 && v.im.abs() < 1e-11,
            "DC bin {k}: {v:?}"
        );
    }
}

#[test]
fn hand_avx512_size32_f64_dc() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let mut data: Vec<Complex<f64>> = (0..32).map(|_| Complex::new(1.0, 0.0)).collect();
    dispatch_hand_avx512_size32_f64(&mut data, -1);
    assert!((data[0].re - 32.0).abs() < 1e-10, "DC bin 0: {:?}", data[0]);
    for (k, v) in data.iter().enumerate().skip(1) {
        assert!(
            v.re.abs() < 1e-10 && v.im.abs() < 1e-10,
            "DC bin {k}: {v:?}"
        );
    }
}

#[test]
fn hand_avx512_size64_f64_dc() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let mut data: Vec<Complex<f64>> = (0..64).map(|_| Complex::new(1.0, 0.0)).collect();
    dispatch_hand_avx512_size64_f64(&mut data, -1);
    assert!((data[0].re - 64.0).abs() < 1e-9, "DC bin 0: {:?}", data[0]);
    for (k, v) in data.iter().enumerate().skip(1) {
        assert!(v.re.abs() < 1e-9 && v.im.abs() < 1e-9, "DC bin {k}: {v:?}");
    }
}
