//! Regression tests for the unified real-transform (c2r) normalization
//! convention and the r2c/c2r round-trip robustness fixes.
//!
//! Background (fftw-api-parity audit finding, CRITICAL):
//! `RealPlan` (1D) auto-normalized c2r by `1/N`, but `RealPlan2D` / `RealPlan3D`
//! / `RealPlanND` did not, so the same spectrum produced different results
//! depending on which struct you used (e.g. `RealPlanND::c2r(&[16])` returned
//! every sample scaled by 16x). These tests pin the unified convention:
//! **every** real c2r path is normalized by `1/product(dims)`, so an
//! `r2c` -> `c2r` round trip is the identity, for even *and* odd sizes.
//!
//! Also covers the `r2c_roundtrip` cargo-fuzz crash: a near-`f32::MAX/2`
//! amplitude at `N = 2` overflowed to `-inf` during the round trip.

#![allow(clippy::cast_precision_loss)] // test math uses float casts of sizes
#![allow(clippy::suboptimal_flops)] // clarity over fused multiply-add in tests
#![allow(clippy::similar_names)] // out_1d/out_2d/out_nd mirror the API dimensions

use oxifft::{Complex, Flags, RealPlan, RealPlan2D, RealPlan3D, RealPlanND};

fn approx(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps * (1.0 + a.abs().max(b.abs()))
}

fn sample_1d(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (i as f64 * 0.7).sin() + 0.25 * (i as f64) + 0.3)
        .collect()
}

// ---------------------------------------------------------------------------
// 1D
// ---------------------------------------------------------------------------

fn roundtrip_1d(n: usize) {
    let input = sample_1d(n);
    let r2c = RealPlan::<f64>::r2c_1d(n, Flags::ESTIMATE).expect("r2c plan");
    let c2r = RealPlan::<f64>::c2r_1d(n, Flags::ESTIMATE).expect("c2r plan");
    let mut spec = vec![Complex::zero(); r2c.complex_size()];
    r2c.execute_r2c(&input, &mut spec);

    let mut recon = vec![0.0; n];
    c2r.execute_c2r(&spec, &mut recon);
    for (i, (a, b)) in input.iter().zip(recon.iter()).enumerate() {
        assert!(
            approx(*a, *b, 1e-9),
            "1D n={n} idx={i}: got {b}, expected {a}"
        );
    }

    // Unnormalized == normalized * N.
    let mut recon_un = vec![0.0; n];
    c2r.execute_c2r_unnormalized(&spec, &mut recon_un);
    for (a, b) in recon.iter().zip(recon_un.iter()) {
        assert!(
            approx(a * n as f64, *b, 1e-9),
            "1D unnormalized n={n}: {b} != {a}*{n}"
        );
    }
}

#[test]
fn roundtrip_1d_even_and_odd() {
    for n in [2usize, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 31, 32] {
        roundtrip_1d(n);
    }
}

/// Odd-size r2c must match a naive DFT (not merely round-trip): the even-only
/// pair-packing path used to silently drop the final sample for odd `N`.
#[test]
fn r2c_odd_matches_naive_dft() {
    for n in [3usize, 5, 7, 9, 15] {
        let input = sample_1d(n);
        let r2c = RealPlan::<f64>::r2c_1d(n, Flags::ESTIMATE).expect("plan");
        let mut spec = vec![Complex::zero(); n / 2 + 1];
        r2c.execute_r2c(&input, &mut spec);
        for (k, sk) in spec.iter().enumerate() {
            let mut re = 0.0;
            let mut im = 0.0;
            for (j, &x) in input.iter().enumerate() {
                let ang = -2.0 * std::f64::consts::PI * (j as f64) * (k as f64) / n as f64;
                re += x * ang.cos();
                im += x * ang.sin();
            }
            assert!(
                approx(sk.re, re, 1e-9),
                "n={n} bin {k} re: {} vs {}",
                sk.re,
                re
            );
            assert!(
                approx(sk.im, im, 1e-9),
                "n={n} bin {k} im: {} vs {}",
                sk.im,
                im
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2D / 3D / ND round trips (even and odd, including odd real axis)
// ---------------------------------------------------------------------------

fn roundtrip_2d(n0: usize, n1: usize) {
    let total = n0 * n1;
    let input: Vec<f64> = (0..total)
        .map(|i| ((i as f64) * 0.31).cos() + 0.13 * i as f64)
        .collect();
    let r2c = RealPlan2D::<f64>::r2c(n0, n1, Flags::ESTIMATE).expect("r2c");
    let c2r = RealPlan2D::<f64>::c2r(n0, n1, Flags::ESTIMATE).expect("c2r");
    let mut spec = vec![Complex::zero(); n0 * (n1 / 2 + 1)];
    r2c.execute_r2c(&input, &mut spec);
    let mut recon = vec![0.0; total];
    c2r.execute_c2r(&spec, &mut recon);
    for (i, (a, b)) in input.iter().zip(recon.iter()).enumerate() {
        assert!(approx(*a, *b, 1e-9), "2D {n0}x{n1} idx={i}: {b} != {a}");
    }
}

fn roundtrip_3d(n0: usize, n1: usize, n2: usize) {
    let total = n0 * n1 * n2;
    let input: Vec<f64> = (0..total)
        .map(|i| ((i as f64) * 0.17).sin() + 0.05 * i as f64)
        .collect();
    let r2c = RealPlan3D::<f64>::r2c(n0, n1, n2, Flags::ESTIMATE).expect("r2c");
    let c2r = RealPlan3D::<f64>::c2r(n0, n1, n2, Flags::ESTIMATE).expect("c2r");
    let mut spec = vec![Complex::zero(); n0 * n1 * (n2 / 2 + 1)];
    r2c.execute_r2c(&input, &mut spec);
    let mut recon = vec![0.0; total];
    c2r.execute_c2r(&spec, &mut recon);
    for (i, (a, b)) in input.iter().zip(recon.iter()).enumerate() {
        assert!(
            approx(*a, *b, 1e-9),
            "3D {n0}x{n1}x{n2} idx={i}: {b} != {a}"
        );
    }
}

fn roundtrip_nd(dims: &[usize]) {
    let total: usize = dims.iter().product();
    let last = *dims.last().unwrap();
    let prefix: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
    let input: Vec<f64> = (0..total)
        .map(|i| ((i as f64) * 0.23).cos() + 0.07 * i as f64)
        .collect();
    let r2c = RealPlanND::<f64>::r2c(dims, Flags::ESTIMATE).expect("r2c");
    let c2r = RealPlanND::<f64>::c2r(dims, Flags::ESTIMATE).expect("c2r");
    let mut spec = vec![Complex::zero(); prefix * (last / 2 + 1)];
    r2c.execute_r2c(&input, &mut spec);
    let mut recon = vec![0.0; total];
    c2r.execute_c2r(&spec, &mut recon);
    for (i, (a, b)) in input.iter().zip(recon.iter()).enumerate() {
        assert!(approx(*a, *b, 1e-9), "ND {dims:?} idx={i}: {b} != {a}");
    }
}

#[test]
fn roundtrip_2d_even_and_odd() {
    for (n0, n1) in [(4, 8), (3, 8), (8, 8), (5, 7), (3, 5), (6, 4)] {
        roundtrip_2d(n0, n1);
    }
}

#[test]
fn roundtrip_3d_even_and_odd() {
    for (n0, n1, n2) in [(2, 4, 8), (4, 4, 4), (3, 3, 5), (2, 3, 4)] {
        roundtrip_3d(n0, n1, n2);
    }
}

#[test]
fn roundtrip_nd_rank4_even_and_odd() {
    for dims in [vec![2, 2, 2, 4], vec![2, 3, 2, 5], vec![2, 2, 3, 4]] {
        roundtrip_nd(&dims);
    }
}

// ---------------------------------------------------------------------------
// Cross-struct normalization consistency (the core of the finding)
// ---------------------------------------------------------------------------

/// The same r2c spectrum, fed through `RealPlan`, `RealPlan2D` (n0=1) and
/// `RealPlanND` (`dims=[n]`), must produce identical, normalized output.
/// Previously `RealPlanND::c2r(&[16])` returned 16x the correct value.
#[test]
fn cross_struct_c2r_normalization_agrees() {
    for n in [8usize, 15, 16, 17] {
        let input = sample_1d(n);
        let r2c = RealPlan::<f64>::r2c_1d(n, Flags::ESTIMATE).expect("plan");
        let mut spec = vec![Complex::zero(); n / 2 + 1];
        r2c.execute_r2c(&input, &mut spec);

        let mut out_1d = vec![0.0; n];
        RealPlan::<f64>::c2r_1d(n, Flags::ESTIMATE)
            .expect("plan")
            .execute_c2r(&spec, &mut out_1d);

        let mut out_2d = vec![0.0; n];
        RealPlan2D::<f64>::c2r(1, n, Flags::ESTIMATE)
            .expect("plan")
            .execute_c2r(&spec, &mut out_2d);

        let mut out_nd = vec![0.0; n];
        RealPlanND::<f64>::c2r(&[n], Flags::ESTIMATE)
            .expect("plan")
            .execute_c2r(&spec, &mut out_nd);

        for i in 0..n {
            assert!(
                approx(out_1d[i], input[i], 1e-9),
                "1D roundtrip n={n} idx={i}"
            );
            assert!(
                approx(out_2d[i], out_1d[i], 1e-9),
                "2D vs 1D n={n} idx={i}: {} vs {}",
                out_2d[i],
                out_1d[i]
            );
            assert!(
                approx(out_nd[i], out_1d[i], 1e-9),
                "ND vs 1D n={n} idx={i}: {} vs {}",
                out_nd[i],
                out_1d[i]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// r2c_roundtrip fuzz-crash regression
// ---------------------------------------------------------------------------

/// The exact 10-byte fuzz input that crashed `r2c_roundtrip`: at `N = 2`, an
/// amplitude of `-2^127` (~ `-f32::MAX/2`) overflowed to `-inf` because the
/// c2r path summed `X[0] + X[1] = 2*x[0]` *before* dividing by `N`.
#[test]
fn c2r_no_overflow_near_f32_max_n2() {
    let x0 = f32::from_le_bytes([0, 0, 0, 0xff]); // -2^127
    let x1 = f32::from_le_bytes([0xff, 0xff, 0xff, 0]); // ~2.35e-38
    let n = 2;
    let r2c = RealPlan::<f32>::r2c_1d(n, Flags::ESTIMATE).expect("plan");
    let c2r = RealPlan::<f32>::c2r_1d(n, Flags::ESTIMATE).expect("plan");
    let mut spec = vec![Complex::<f32>::new(0.0, 0.0); n / 2 + 1];
    r2c.execute_r2c(&[x0, x1], &mut spec);
    let mut recon = vec![0.0f32; n];
    c2r.execute_c2r(&spec, &mut recon);

    assert!(
        recon.iter().all(|v| v.is_finite()),
        "c2r overflowed to non-finite: {recon:?}"
    );
    // The large but individually-representable value round-trips.
    assert!(
        (recon[0] - x0).abs() <= x0.abs() * 1e-3,
        "recon[0]={} expected ~{x0}",
        recon[0]
    );
}

/// Replays the fuzz harness logic over adversarial-but-bounded inputs (n=2, odd
/// n, larger n) and asserts the round trip stays finite and correct.
#[test]
fn r2c_roundtrip_adversarial_bounded() {
    for n in [2usize, 3, 4, 7, 8, 16, 31, 64] {
        // Deterministic pseudo-random-ish bounded amplitudes.
        let input: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 12.9898).sin() * 43758.547).fract() * 2.0e6 - 1.0e6)
            .collect();
        let r2c = RealPlan::<f32>::r2c_1d(n, Flags::ESTIMATE).expect("plan");
        let c2r = RealPlan::<f32>::c2r_1d(n, Flags::ESTIMATE).expect("plan");
        let mut spec = vec![Complex::<f32>::new(0.0, 0.0); n / 2 + 1];
        r2c.execute_r2c(&input, &mut spec);
        let mut recon = vec![0.0f32; n];
        c2r.execute_c2r(&spec, &mut recon);
        for (i, (a, b)) in input.iter().zip(recon.iter()).enumerate() {
            assert!(b.is_finite(), "n={n} idx={i} non-finite");
            let scale = a.abs().max(1.0);
            assert!(
                (b - a).abs() / scale < 5e-3,
                "n={n} idx={i}: got {b}, expected {a}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Flags routing: MEASURE must not change correctness (Task 3 smoke test)
// ---------------------------------------------------------------------------

#[test]
fn measure_flag_real_transform_still_correct() {
    let n = 12;
    let input = sample_1d(n);
    for flags in [Flags::ESTIMATE, Flags::MEASURE, Flags::PATIENT] {
        let r2c = RealPlan::<f64>::r2c_1d(n, flags).expect("plan");
        let c2r = RealPlan::<f64>::c2r_1d(n, flags).expect("plan");
        let mut spec = vec![Complex::zero(); n / 2 + 1];
        r2c.execute_r2c(&input, &mut spec);
        let mut recon = vec![0.0; n];
        c2r.execute_c2r(&spec, &mut recon);
        for (a, b) in input.iter().zip(recon.iter()) {
            assert!(approx(*a, *b, 1e-9), "flags={flags:?}: {b} != {a}");
        }
    }
}
