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

// ── Planning-mode delegation (MEASURE / PATIENT / EXHAUSTIVE) ─────────────────
//
// The delegation paths must keep working under every planning mode, not just
// ESTIMATE.  MEASURE/PATIENT/EXHAUSTIVE additionally exercise the runtime
// wisdom cache + benchmarking path in the underlying `Plan::dft_1d`.

const MEASURED_MODES: [Flags; 3] = [Flags::MEASURE, Flags::PATIENT, Flags::EXHAUSTIVE];

#[test]
fn plan_dft_2d_delegates_in_all_planning_modes() {
    for flags in MEASURED_MODES {
        let plan = Plan::<f64>::dft_2d(4, 8, Direction::Forward, flags);
        assert!(plan.is_some(), "dft_2d must delegate under {flags:?}");
    }
}

#[test]
fn plan_dft_3d_delegates_in_all_planning_modes() {
    for flags in MEASURED_MODES {
        let plan = Plan::<f64>::dft_3d(2, 4, 8, Direction::Forward, flags);
        assert!(plan.is_some(), "dft_3d must delegate under {flags:?}");
    }
}

#[test]
fn plan_r2c_c2r_delegate_in_all_planning_modes() {
    for flags in MEASURED_MODES {
        assert!(
            Plan::<f64>::r2c_1d(64, flags).is_some(),
            "r2c_1d must delegate under {flags:?}"
        );
        assert!(
            Plan::<f64>::c2r_1d(64, flags).is_some(),
            "c2r_1d must delegate under {flags:?}"
        );
    }
}

#[test]
fn plan_dft_2d_measure_roundtrip_is_correct() {
    // Correctness must not depend on the planning mode: a MEASURE-planned 2D
    // transform must round-trip just like the ESTIMATE one.
    let (n0, n1) = (8, 8);
    let fwd = Plan::<f64>::dft_2d(n0, n1, Direction::Forward, Flags::MEASURE).unwrap();
    let bwd = Plan::<f64>::dft_2d(n0, n1, Direction::Backward, Flags::PATIENT).unwrap();

    let mut input = vec![Complex::new(0.0, 0.0); n0 * n1];
    input[0] = Complex::new(1.0, 0.0);
    let mut spectrum = vec![Complex::new(0.0, 0.0); n0 * n1];
    let mut recovered = vec![Complex::new(0.0, 0.0); n0 * n1];

    fwd.execute(&input, &mut spectrum);
    bwd.execute(&spectrum, &mut recovered);

    let scale = (n0 * n1) as f64;
    assert!((recovered[0].re - scale).abs() < 1e-9);
}

// ── Runtime wisdom cache: MEASURE must not re-benchmark every call ────────────

#[test]
fn measure_caches_winner_and_second_call_is_fast() {
    use std::time::Instant;

    // A size exclusive to this test so no other test pre-populates its wisdom.
    let n = 8192;

    // First MEASURE benchmarks several real candidates (ct-dit, ct-dif,
    // stockham, cache-oblivious) — comparatively slow.
    let t0 = Instant::now();
    let p1 = Plan::<f64>::dft_1d(n, Direction::Forward, Flags::MEASURE).expect("first plan");
    let first = t0.elapsed();

    // Second MEASURE must hit the cached wisdom entry and reconstruct without
    // benchmarking.  Previously the tuner re-ran on every call for zero payoff.
    let t1 = Instant::now();
    let p2 = Plan::<f64>::dft_1d(n, Direction::Forward, Flags::MEASURE).expect("second plan");
    let second = t1.elapsed();

    assert_eq!(
        p1.algorithm_name(),
        p2.algorithm_name(),
        "cached plan must match the measured winner"
    );
    assert!(
        second * 4 < first,
        "second MEASURE ({second:?}) should be far faster than the first ({first:?}); \
         re-benchmarking on every call is the bug being guarded against"
    );

    // Having measured the size, WISDOM_ONLY must now succeed for it.
    assert!(
        Plan::<f64>::dft_1d(n, Direction::Forward, Flags::WISDOM_ONLY).is_some(),
        "WISDOM_ONLY must succeed once the size has been measured"
    );
}

// ── WISDOM_ONLY failure path ─────────────────────────────────────────────────

#[test]
fn wisdom_only_fails_when_no_wisdom_exists() {
    // A prime size used by no other test: with the default (empty) build-time
    // baseline and no prior measurement, WISDOM_ONLY has nothing to use and must
    // fail rather than silently falling back to the heuristic.
    let n = 1013;
    let plan = Plan::<f64>::dft_1d(n, Direction::Forward, Flags::WISDOM_ONLY);
    assert!(
        plan.is_none(),
        "WISDOM_ONLY must return None when no wisdom exists for n={n}"
    );
}
