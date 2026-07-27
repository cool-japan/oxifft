//! Correctness tests for multi-dimensional real-to-real transforms
//! (`R2rPlan2D` / `R2rPlan3D`), closing the FFTW `fftw_plan_r2r_2d` /
//! `fftw_plan_r2r_3d` parity gap.
//!
//! Each separable transform is checked against an independent naive O(N^2)
//! tensor-product reference built directly from the crate's 1D kernel
//! definitions (DCT-II / DST-II / DHT).

#![allow(clippy::cast_precision_loss)] // test math uses float casts of indices/sizes
#![allow(clippy::suboptimal_flops)] // clarity over fused multiply-add in references
#![allow(clippy::similar_names)] // n0/n1/n2, k0/k1/k2 mirror the API
#![allow(clippy::redundant_clone)] // explicit input snapshots for in-place tests

use oxifft::api::{R2rPlan2D, R2rPlan3D};
// `R2rPlan*` uses the REDFT/RODFT kind enum (same one taken by `R2rPlan::r2r_1d`),
// which lives in `rdft::solvers` — distinct from the FFTW-named `oxifft::R2rKind`.
use oxifft::rdft::solvers::R2rKind;
use oxifft::Flags;
use std::f64::consts::PI;

/// 1D basis value `B_kind(idx, k, N)`, matching the crate's `execute_*_direct`
/// definitions in `rdft/solvers/r2r.rs`.
fn kernel(kind: R2rKind, idx: usize, k: usize, big_n: usize) -> f64 {
    let idx = idx as f64;
    let k = k as f64;
    let bn = big_n as f64;
    match kind {
        // DCT-II: cos(pi (2i+1) k / (2N))
        R2rKind::Redft10 => (PI * (2.0 * idx + 1.0) * k / (2.0 * bn)).cos(),
        // DST-II: sin(pi (2i+1) (k+1) / (2N))
        R2rKind::Rodft10 => (PI * (2.0 * idx + 1.0) * (k + 1.0) / (2.0 * bn)).sin(),
        // DHT: cas(2 pi i k / N) = cos + sin
        R2rKind::Dht => {
            let a = 2.0 * PI * idx * k / bn;
            a.cos() + a.sin()
        }
        other => panic!("kernel not defined for {other:?}"),
    }
}

fn naive_2d(x: &[f64], n0: usize, n1: usize, k0: R2rKind, k1: R2rKind) -> Vec<f64> {
    let mut y = vec![0.0; n0 * n1];
    for a0 in 0..n0 {
        for a1 in 0..n1 {
            let mut s = 0.0;
            for i0 in 0..n0 {
                let b0 = kernel(k0, i0, a0, n0);
                for i1 in 0..n1 {
                    s += x[i0 * n1 + i1] * b0 * kernel(k1, i1, a1, n1);
                }
            }
            y[a0 * n1 + a1] = s;
        }
    }
    y
}

#[allow(clippy::too_many_arguments)]
fn naive_3d(
    x: &[f64],
    n0: usize,
    n1: usize,
    n2: usize,
    k0: R2rKind,
    k1: R2rKind,
    k2: R2rKind,
) -> Vec<f64> {
    let mut y = vec![0.0; n0 * n1 * n2];
    for a0 in 0..n0 {
        for a1 in 0..n1 {
            for a2 in 0..n2 {
                let mut s = 0.0;
                for i0 in 0..n0 {
                    let b0 = kernel(k0, i0, a0, n0);
                    for i1 in 0..n1 {
                        let b01 = b0 * kernel(k1, i1, a1, n1);
                        for i2 in 0..n2 {
                            s += x[(i0 * n1 + i1) * n2 + i2] * b01 * kernel(k2, i2, a2, n2);
                        }
                    }
                }
                y[(a0 * n1 + a1) * n2 + a2] = s;
            }
        }
    }
    y
}

fn assert_close(got: &[f64], want: &[f64], tag: &str) {
    assert_eq!(got.len(), want.len(), "{tag}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-9 * (1.0 + w.abs()),
            "{tag}: idx {i}: got {g}, expected {w}"
        );
    }
}

#[test]
fn r2r_2d_matches_naive_4x8() {
    let (n0, n1) = (4, 8);
    let x: Vec<f64> = (0..n0 * n1)
        .map(|i| ((i as f64) * 0.37).sin() + 0.11 * i as f64)
        .collect();
    for kind in [R2rKind::Redft10, R2rKind::Rodft10, R2rKind::Dht] {
        let plan = R2rPlan2D::<f64>::r2r_2d(n0, n1, kind, Flags::ESTIMATE).expect("plan");
        let mut out = vec![0.0; n0 * n1];
        plan.execute(&x, &mut out);
        let want = naive_2d(&x, n0, n1, kind, kind);
        assert_close(&out, &want, &format!("2D {kind:?}"));
    }
}

#[test]
fn r2r_2d_matches_naive_8x8() {
    let (n0, n1) = (8, 8);
    let x: Vec<f64> = (0..n0 * n1)
        .map(|i| ((i as f64) * 0.13).cos() - 0.2)
        .collect();
    for kind in [R2rKind::Redft10, R2rKind::Rodft10, R2rKind::Dht] {
        let plan = R2rPlan2D::<f64>::r2r_2d(n0, n1, kind, Flags::ESTIMATE).expect("plan");
        let mut out = vec![0.0; n0 * n1];
        plan.execute(&x, &mut out);
        let want = naive_2d(&x, n0, n1, kind, kind);
        assert_close(&out, &want, &format!("2D8 {kind:?}"));
    }
}

/// Per-axis (mixed) kinds, mirroring FFTW's `fftw_plan_r2r_2d(kind0, kind1)`.
#[test]
fn r2r_2d_mixed_kinds() {
    let (n0, n1) = (4, 8);
    let x: Vec<f64> = (0..n0 * n1).map(|i| ((i as f64) * 0.29).sin()).collect();
    let plan = R2rPlan2D::<f64>::new(n0, n1, R2rKind::Redft10, R2rKind::Rodft10, Flags::ESTIMATE)
        .expect("plan");
    let mut out = vec![0.0; n0 * n1];
    plan.execute(&x, &mut out);
    let want = naive_2d(&x, n0, n1, R2rKind::Redft10, R2rKind::Rodft10);
    assert_close(&out, &want, "2D mixed DCT-II/DST-II");
}

#[test]
fn r2r_2d_inplace_matches_out_of_place() {
    let (n0, n1) = (4, 8);
    let x: Vec<f64> = (0..n0 * n1).map(|i| ((i as f64) * 0.41).cos()).collect();
    let plan = R2rPlan2D::<f64>::r2r_2d(n0, n1, R2rKind::Dht, Flags::ESTIMATE).expect("plan");
    let mut out = vec![0.0; n0 * n1];
    plan.execute(&x, &mut out);
    let mut data = x.clone();
    plan.execute_inplace(&mut data);
    assert_close(&data, &out, "2D inplace");
}

#[test]
fn r2r_3d_matches_naive_8x8x8() {
    let (n0, n1, n2) = (8, 8, 8);
    let x: Vec<f64> = (0..n0 * n1 * n2)
        .map(|i| ((i as f64) * 0.07).sin() + 0.01 * i as f64)
        .collect();
    for kind in [R2rKind::Redft10, R2rKind::Rodft10, R2rKind::Dht] {
        let plan = R2rPlan3D::<f64>::r2r_3d(n0, n1, n2, kind, Flags::ESTIMATE).expect("plan");
        let mut out = vec![0.0; n0 * n1 * n2];
        plan.execute(&x, &mut out);
        let want = naive_3d(&x, n0, n1, n2, kind, kind, kind);
        assert_close(&out, &want, &format!("3D {kind:?}"));
    }
}

#[test]
fn r2r_3d_inplace_matches_out_of_place() {
    let (n0, n1, n2) = (4, 4, 4);
    let x: Vec<f64> = (0..n0 * n1 * n2)
        .map(|i| ((i as f64) * 0.19).cos())
        .collect();
    let plan =
        R2rPlan3D::<f64>::r2r_3d(n0, n1, n2, R2rKind::Redft10, Flags::ESTIMATE).expect("plan");
    let mut out = vec![0.0; n0 * n1 * n2];
    plan.execute(&x, &mut out);
    let mut data = x.clone();
    plan.execute_inplace(&mut data);
    assert_close(&data, &out, "3D inplace");
}
