#![allow(clippy::cast_precision_loss)]
//! Regression tests for `Plan::dft_2d`, `Plan::dft_3d`, `Plan::r2c_1d`, `Plan::c2r_1d`.
//!
//! These methods previously panicked with `todo!()`. After the v0.2.0 fix they
//! now delegate to the correct dedicated plan types (`Plan2D`, `Plan3D`, `RealPlan`).
//! This file ensures they never regress back to panicking.

use oxifft::{Complex, Direction, Flags, Plan};

// ── dft_2d ──────────────────────────────────────────────────────────────────

#[test]
fn plan_dft_2d_returns_some_for_valid_dimensions() {
    let plan = Plan::<f64>::dft_2d(4, 8, Direction::Forward, Flags::ESTIMATE);
    assert!(
        plan.is_some(),
        "Plan::dft_2d should return Some for valid n0={}, n1={}",
        4,
        8
    );
}

#[test]
fn plan_dft_2d_zero_dimensions_are_nop() {
    // Zero-sized 2D plans return Some with a Nop algorithm — they execute as no-ops.
    // This mirrors Plan::dft_1d(0) which also returns Some.
    let plan = Plan::<f64>::dft_2d(0, 8, Direction::Forward, Flags::ESTIMATE);
    assert!(plan.is_some(), "Plan::dft_2d returns Some(Nop) when n0=0");

    let plan = Plan::<f64>::dft_2d(4, 0, Direction::Forward, Flags::ESTIMATE);
    assert!(plan.is_some(), "Plan::dft_2d returns Some(Nop) when n1=0");
}

#[test]
fn plan_dft_2d_roundtrip() {
    let n0 = 4;
    let n1 = 8;
    let fwd = Plan::<f64>::dft_2d(n0, n1, Direction::Forward, Flags::ESTIMATE).unwrap();
    let bwd = Plan::<f64>::dft_2d(n0, n1, Direction::Backward, Flags::ESTIMATE).unwrap();

    // Impulse at origin — forward FFT is flat, inverse should recover impulse
    let mut input: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n0 * n1];
    input[0] = Complex::new(1.0, 0.0);
    let mut spectrum = vec![Complex::new(0.0, 0.0); n0 * n1];
    let mut recovered = vec![Complex::new(0.0, 0.0); n0 * n1];

    fwd.execute(&input, &mut spectrum);
    bwd.execute(&spectrum, &mut recovered);

    // Unnormalised roundtrip: recovered[i] ≈ n * input[i]
    let scale = (n0 * n1) as f64;
    let err = (recovered[0].re - scale).abs();
    assert!(
        err < 1e-9,
        "2D roundtrip: recovered[0].re={} expected {scale}",
        recovered[0].re
    );
}

#[test]
fn plan_dft_2d_f32_works() {
    let plan = Plan::<f32>::dft_2d(8, 8, Direction::Forward, Flags::ESTIMATE);
    assert!(plan.is_some());
}

// ── dft_3d ──────────────────────────────────────────────────────────────────

#[test]
fn plan_dft_3d_returns_some_for_valid_dimensions() {
    let plan = Plan::<f64>::dft_3d(2, 4, 8, Direction::Forward, Flags::ESTIMATE);
    assert!(
        plan.is_some(),
        "Plan::dft_3d should return Some for valid dimensions"
    );
}

#[test]
fn plan_dft_3d_zero_dimensions_are_nop() {
    // Zero-sized 3D plans return Some with a Nop algorithm — they execute as no-ops.
    assert!(Plan::<f64>::dft_3d(0, 4, 8, Direction::Forward, Flags::ESTIMATE).is_some());
    assert!(Plan::<f64>::dft_3d(2, 0, 8, Direction::Forward, Flags::ESTIMATE).is_some());
    assert!(Plan::<f64>::dft_3d(2, 4, 0, Direction::Forward, Flags::ESTIMATE).is_some());
}

#[test]
fn plan_dft_3d_roundtrip() {
    let (n0, n1, n2) = (2, 4, 4);
    let fwd = Plan::<f64>::dft_3d(n0, n1, n2, Direction::Forward, Flags::ESTIMATE).unwrap();
    let bwd = Plan::<f64>::dft_3d(n0, n1, n2, Direction::Backward, Flags::ESTIMATE).unwrap();

    let mut input: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n0 * n1 * n2];
    input[0] = Complex::new(1.0, 0.0);
    let mut spectrum = vec![Complex::new(0.0, 0.0); n0 * n1 * n2];
    let mut recovered = vec![Complex::new(0.0, 0.0); n0 * n1 * n2];

    fwd.execute(&input, &mut spectrum);
    bwd.execute(&spectrum, &mut recovered);

    let scale = (n0 * n1 * n2) as f64;
    let err = (recovered[0].re - scale).abs();
    assert!(
        err < 1e-9,
        "3D roundtrip: recovered[0].re={} expected {scale}",
        recovered[0].re
    );
}

#[test]
fn plan_dft_3d_f32_works() {
    let plan = Plan::<f32>::dft_3d(4, 4, 4, Direction::Forward, Flags::ESTIMATE);
    assert!(plan.is_some());
}

// ── r2c_1d ──────────────────────────────────────────────────────────────────

#[test]
fn plan_r2c_1d_returns_some_for_valid_size() {
    let plan = Plan::<f64>::r2c_1d(64, Flags::ESTIMATE);
    assert!(plan.is_some(), "Plan::r2c_1d should return Some for n=64");
}

#[test]
fn plan_r2c_1d_returns_none_for_zero() {
    let plan = Plan::<f64>::r2c_1d(0, Flags::ESTIMATE);
    assert!(plan.is_none(), "Plan::r2c_1d should return None for n=0");
}

#[test]
fn plan_r2c_1d_output_size_is_half_plus_one() {
    let n = 64;
    let plan = Plan::<f64>::r2c_1d(n, Flags::ESTIMATE).unwrap();
    assert_eq!(plan.complex_size(), n / 2 + 1);
}

#[test]
fn plan_r2c_1d_dc_component() {
    // DC of all-ones real signal = n
    let n = 16;
    let plan = Plan::<f64>::r2c_1d(n, Flags::ESTIMATE).unwrap();
    let input = vec![1.0_f64; n];
    let mut spectrum = vec![Complex::new(0.0, 0.0); n / 2 + 1];
    plan.execute_r2c(&input, &mut spectrum);
    let err = (spectrum[0].re - n as f64).abs();
    assert!(
        err < 1e-10,
        "DC component = {:.4}, expected {n}",
        spectrum[0].re
    );
    assert!(spectrum[0].im.abs() < 1e-10);
}

// ── c2r_1d ──────────────────────────────────────────────────────────────────

#[test]
fn plan_c2r_1d_returns_some_for_valid_size() {
    let plan = Plan::<f64>::c2r_1d(64, Flags::ESTIMATE);
    assert!(plan.is_some(), "Plan::c2r_1d should return Some for n=64");
}

#[test]
fn plan_c2r_1d_returns_none_for_zero() {
    let plan = Plan::<f64>::c2r_1d(0, Flags::ESTIMATE);
    assert!(plan.is_none(), "Plan::c2r_1d should return None for n=0");
}

#[test]
fn plan_r2c_c2r_roundtrip() {
    // Full R2C → C2R roundtrip via the Plan delegation methods (power-of-2)
    let n = 32;
    let fwd = Plan::<f64>::r2c_1d(n, Flags::ESTIMATE).unwrap();
    let bwd = Plan::<f64>::c2r_1d(n, Flags::ESTIMATE).unwrap();

    let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
    let mut spectrum = vec![Complex::new(0.0, 0.0); n / 2 + 1];
    let mut recovered = vec![0.0_f64; n];

    fwd.execute_r2c(&input, &mut spectrum);
    bwd.execute_c2r(&spectrum, &mut recovered); // normalised: output ≈ input

    for (i, (a, b)) in input.iter().zip(recovered.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "r2c→c2r mismatch at i={i}: orig={a:.6}, rec={b:.6}"
        );
    }
}

// ── Debug impl smoke test ────────────────────────────────────────────────────

#[test]
fn plan_types_implement_debug() {
    let p1 = Plan::<f64>::dft_1d(16, Direction::Forward, Flags::ESTIMATE).unwrap();
    let _ = format!("{p1:?}");

    let p2 = Plan::<f64>::dft_2d(4, 4, Direction::Forward, Flags::ESTIMATE).unwrap();
    let _ = format!("{p2:?}");

    let p3 = Plan::<f64>::dft_3d(2, 4, 4, Direction::Forward, Flags::ESTIMATE).unwrap();
    let _ = format!("{p3:?}");

    let pr = Plan::<f64>::r2c_1d(16, Flags::ESTIMATE).unwrap();
    let _ = format!("{pr:?}");
}

// ── must_use check (compile-time only) ──────────────────────────────────────
// The #[must_use] attribute on plan creation methods is enforced by the compiler.
// If these methods were called without using the result, a compiler warning would fire.
// We can't write a "must compile with warning" test, but we verify the methods are
// callable and produce results that can be used.
#[test]
fn plan_creation_methods_are_callable() {
    let _ = Plan::<f64>::dft_1d(8, Direction::Forward, Flags::ESTIMATE);
    let _ = Plan::<f64>::dft_2d(4, 4, Direction::Forward, Flags::ESTIMATE);
    let _ = Plan::<f64>::dft_3d(2, 4, 4, Direction::Forward, Flags::ESTIMATE);
    let _ = Plan::<f64>::r2c_1d(8, Flags::ESTIMATE);
    let _ = Plan::<f64>::c2r_1d(8, Flags::ESTIMATE);
}
