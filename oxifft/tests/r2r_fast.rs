#![allow(clippy::cast_precision_loss)]
//! Integration tests for FFT-based fast r2r transforms.
//!
//! Tests that `execute_X_fast` matches `execute_X_direct` for all sizes >= 16.

use oxifft::rdft::solvers::{R2rKind, R2rSolver};

/// Returns true if fast ≈ direct within relative tolerance `rel_tol` or absolute tolerance `abs_tol`.
fn approx_eq(fast: f64, direct: f64, rel_tol: f64, abs_tol: f64) -> bool {
    let abs_err = (fast - direct).abs();
    if abs_err < abs_tol {
        return true;
    }
    let scale = direct.abs().max(fast.abs()).max(1e-30);
    abs_err / scale < rel_tol
}

fn make_input(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect()
}

#[test]
fn test_dct2_fast_matches_direct() {
    let solver = R2rSolver::<f64>::new(R2rKind::Redft10);
    for &n in &[16usize, 32, 64, 128, 256, 512, 1024] {
        let input = make_input(n);
        let mut out_direct = vec![0.0_f64; n];
        let mut out_fast = vec![0.0_f64; n];
        solver.execute_dct2_direct(&input, &mut out_direct);
        solver.execute_dct2_fast(&input, &mut out_fast);
        for k in 0..n {
            assert!(
                approx_eq(out_fast[k], out_direct[k], 1e-8, 1e-10),
                "DCT-II size {}: mismatch at k={}: fast={} direct={}",
                n,
                k,
                out_fast[k],
                out_direct[k]
            );
        }
    }
}

#[test]
fn test_dct3_fast_matches_direct() {
    let solver = R2rSolver::<f64>::new(R2rKind::Redft01);
    for &n in &[16usize, 32, 64, 128, 256] {
        let input = make_input(n);
        let mut out_direct = vec![0.0_f64; n];
        let mut out_fast = vec![0.0_f64; n];
        solver.execute_dct3_direct(&input, &mut out_direct);
        solver.execute_dct3_fast(&input, &mut out_fast);
        for k in 0..n {
            assert!(
                approx_eq(out_fast[k], out_direct[k], 1e-8, 1e-10),
                "DCT-III size {}: mismatch at k={}: fast={} direct={}",
                n,
                k,
                out_fast[k],
                out_direct[k]
            );
        }
    }
}

#[test]
fn test_dct4_fast_matches_direct() {
    let solver = R2rSolver::<f64>::new(R2rKind::Redft11);
    for &n in &[16usize, 32, 64, 128, 256] {
        let input = make_input(n);
        let mut out_direct = vec![0.0_f64; n];
        let mut out_fast = vec![0.0_f64; n];
        solver.execute_dct4_direct(&input, &mut out_direct);
        solver.execute_dct4_fast(&input, &mut out_fast);
        for k in 0..n {
            assert!(
                approx_eq(out_fast[k], out_direct[k], 1e-8, 1e-10),
                "DCT-IV size {}: mismatch at k={}: fast={} direct={}",
                n,
                k,
                out_fast[k],
                out_direct[k]
            );
        }
    }
}

#[test]
fn test_dct1_fast_matches_direct() {
    let solver = R2rSolver::<f64>::new(R2rKind::Redft00);
    for &n in &[16usize, 32, 64, 128, 256] {
        let input = make_input(n);
        let mut out_direct = vec![0.0_f64; n];
        let mut out_fast = vec![0.0_f64; n];
        solver.execute_dct1_direct(&input, &mut out_direct);
        solver.execute_dct1_fast(&input, &mut out_fast);
        for k in 0..n {
            assert!(
                approx_eq(out_fast[k], out_direct[k], 1e-8, 1e-10),
                "DCT-I size {}: mismatch at k={}: fast={} direct={}",
                n,
                k,
                out_fast[k],
                out_direct[k]
            );
        }
    }
}

#[test]
fn test_dht_fast_matches_direct() {
    let solver = R2rSolver::<f64>::new(R2rKind::Dht);
    for &n in &[16usize, 32, 64, 128, 256] {
        let input = make_input(n);
        let mut out_direct = vec![0.0_f64; n];
        let mut out_fast = vec![0.0_f64; n];
        solver.execute_dht_direct(&input, &mut out_direct);
        solver.execute_dht_fast(&input, &mut out_fast);
        for k in 0..n {
            assert!(
                approx_eq(out_fast[k], out_direct[k], 1e-8, 1e-10),
                "DHT size {}: mismatch at k={}: fast={} direct={}",
                n,
                k,
                out_fast[k],
                out_direct[k]
            );
        }
    }
}

#[test]
fn test_dst1_fast_matches_direct() {
    let solver = R2rSolver::<f64>::new(R2rKind::Rodft00);
    for &n in &[16usize, 32, 64, 128, 256] {
        let input = make_input(n);
        let mut out_direct = vec![0.0_f64; n];
        let mut out_fast = vec![0.0_f64; n];
        solver.execute_dst1_direct(&input, &mut out_direct);
        solver.execute_dst1_fast(&input, &mut out_fast);
        for k in 0..n {
            assert!(
                approx_eq(out_fast[k], out_direct[k], 1e-8, 1e-10),
                "DST-I size {}: mismatch at k={}: fast={} direct={}",
                n,
                k,
                out_fast[k],
                out_direct[k]
            );
        }
    }
}

#[test]
fn test_dst2_fast_matches_direct() {
    let solver = R2rSolver::<f64>::new(R2rKind::Rodft10);
    for &n in &[16usize, 32, 64, 128, 256] {
        let input = make_input(n);
        let mut out_direct = vec![0.0_f64; n];
        let mut out_fast = vec![0.0_f64; n];
        solver.execute_dst2_direct(&input, &mut out_direct);
        solver.execute_dst2_fast(&input, &mut out_fast);
        for k in 0..n {
            assert!(
                approx_eq(out_fast[k], out_direct[k], 1e-8, 1e-10),
                "DST-II size {}: mismatch at k={}: fast={} direct={}",
                n,
                k,
                out_fast[k],
                out_direct[k]
            );
        }
    }
}

#[test]
fn test_dst3_fast_matches_direct() {
    let solver = R2rSolver::<f64>::new(R2rKind::Rodft01);
    for &n in &[16usize, 32, 64, 128, 256] {
        let input = make_input(n);
        let mut out_direct = vec![0.0_f64; n];
        let mut out_fast = vec![0.0_f64; n];
        solver.execute_dst3_direct(&input, &mut out_direct);
        solver.execute_dst3_fast(&input, &mut out_fast);
        for k in 0..n {
            assert!(
                approx_eq(out_fast[k], out_direct[k], 1e-8, 1e-10),
                "DST-III size {}: mismatch at k={}: fast={} direct={}",
                n,
                k,
                out_fast[k],
                out_direct[k]
            );
        }
    }
}

#[test]
fn test_dst4_fast_matches_direct() {
    let solver = R2rSolver::<f64>::new(R2rKind::Rodft11);
    for &n in &[16usize, 32, 64, 128, 256] {
        let input = make_input(n);
        let mut out_direct = vec![0.0_f64; n];
        let mut out_fast = vec![0.0_f64; n];
        solver.execute_dst4_direct(&input, &mut out_direct);
        solver.execute_dst4_fast(&input, &mut out_fast);
        for k in 0..n {
            assert!(
                approx_eq(out_fast[k], out_direct[k], 1e-8, 1e-10),
                "DST-IV size {}: mismatch at k={}: fast={} direct={}",
                n,
                k,
                out_fast[k],
                out_direct[k]
            );
        }
    }
}
