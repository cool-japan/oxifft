//! Tests for split-complex format support.
//!
//! Tests the `SplitPlan` and `SplitPlan2D` interfaces for split real/imaginary arrays.

#![allow(clippy::similar_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::redundant_clone)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::doc_markdown)]

use oxifft::{fft_split, ifft_split, Complex, Direction, Flags, Plan, SplitPlan, SplitPlan2D};

/// Check if two vectors are approximately equal.
fn vec_approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if (x - y).abs() > tol {
            eprintln!(
                "Mismatch at index {}: got {}, expected {}, diff = {}",
                i,
                x,
                y,
                (x - y).abs()
            );
            return false;
        }
    }

    true
}

// ============================================================================
// SplitPlan 1D tests
// ============================================================================

#[test]
fn test_split_plan_creation() {
    let plan = SplitPlan::<f64>::dft_1d(16, Direction::Forward, Flags::ESTIMATE);
    assert!(plan.is_some());

    let plan = plan.unwrap();
    assert_eq!(plan.size(), 16);
    assert_eq!(plan.direction(), Direction::Forward);
}

#[test]
fn test_split_plan_creation_zero_size() {
    let plan = SplitPlan::<f64>::dft_1d(0, Direction::Forward, Flags::ESTIMATE);
    assert!(plan.is_some());
}

#[test]
fn test_split_plan_matches_interleaved() {
    let n = 16;

    // Create test data
    let in_real: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let in_imag: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();

    // Interleaved format
    let interleaved: Vec<Complex<f64>> = in_real
        .iter()
        .zip(in_imag.iter())
        .map(|(&re, &im)| Complex::new(re, im))
        .collect();

    // Compute using interleaved
    let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
    let mut interleaved_out = vec![Complex::new(0.0, 0.0); n];
    plan.execute(&interleaved, &mut interleaved_out);

    // Compute using split
    let split_plan = SplitPlan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
    let mut out_real = vec![0.0; n];
    let mut out_imag = vec![0.0; n];
    split_plan.execute(&in_real, &in_imag, &mut out_real, &mut out_imag);

    // Compare
    let expected_real: Vec<f64> = interleaved_out.iter().map(|c| c.re).collect();
    let expected_imag: Vec<f64> = interleaved_out.iter().map(|c| c.im).collect();

    assert!(
        vec_approx_eq(&out_real, &expected_real, 1e-12),
        "Real parts don't match"
    );
    assert!(
        vec_approx_eq(&out_imag, &expected_imag, 1e-12),
        "Imaginary parts don't match"
    );
}

#[test]
fn test_split_plan_roundtrip() {
    let n = 32;

    let in_real: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
    let in_imag: Vec<f64> = (0..n).map(|i| (i as f64) * 0.05 + 0.5).collect();

    // Forward
    let forward = SplitPlan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
    let mut freq_real = vec![0.0; n];
    let mut freq_imag = vec![0.0; n];
    forward.execute(&in_real, &in_imag, &mut freq_real, &mut freq_imag);

    // Backward
    let backward = SplitPlan::dft_1d(n, Direction::Backward, Flags::ESTIMATE).unwrap();
    let mut recovered_real = vec![0.0; n];
    let mut recovered_imag = vec![0.0; n];
    backward.execute(
        &freq_real,
        &freq_imag,
        &mut recovered_real,
        &mut recovered_imag,
    );

    // Normalize
    let scale = 1.0 / (n as f64);
    for r in &mut recovered_real {
        *r *= scale;
    }
    for i in &mut recovered_imag {
        *i *= scale;
    }

    assert!(
        vec_approx_eq(&in_real, &recovered_real, 1e-12),
        "Real parts roundtrip failed"
    );
    assert!(
        vec_approx_eq(&in_imag, &recovered_imag, 1e-12),
        "Imaginary parts roundtrip failed"
    );
}

#[test]
fn test_split_plan_inplace() {
    let n = 16;

    let in_real: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let in_imag: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();

    // Out-of-place reference
    let plan = SplitPlan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
    let mut out_real = vec![0.0; n];
    let mut out_imag = vec![0.0; n];
    plan.execute(&in_real, &in_imag, &mut out_real, &mut out_imag);

    // In-place
    let mut ip_real = in_real.clone();
    let mut ip_imag = in_imag.clone();
    plan.execute_inplace(&mut ip_real, &mut ip_imag);

    assert!(
        vec_approx_eq(&ip_real, &out_real, 1e-12),
        "In-place real mismatch"
    );
    assert!(
        vec_approx_eq(&ip_imag, &out_imag, 1e-12),
        "In-place imaginary mismatch"
    );
}

#[test]
fn test_split_plan_various_sizes() {
    let sizes = [4, 7, 8, 15, 16, 31, 32, 64, 100, 128, 256];

    for &n in &sizes {
        let in_real: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let in_imag: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();

        // Forward then backward
        let forward = SplitPlan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
        let backward = SplitPlan::dft_1d(n, Direction::Backward, Flags::ESTIMATE).unwrap();

        let mut freq_real = vec![0.0; n];
        let mut freq_imag = vec![0.0; n];
        forward.execute(&in_real, &in_imag, &mut freq_real, &mut freq_imag);

        let mut recovered_real = vec![0.0; n];
        let mut recovered_imag = vec![0.0; n];
        backward.execute(
            &freq_real,
            &freq_imag,
            &mut recovered_real,
            &mut recovered_imag,
        );

        // Normalize
        let scale = 1.0 / (n as f64);
        for r in &mut recovered_real {
            *r *= scale;
        }
        for i in &mut recovered_imag {
            *i *= scale;
        }

        let tol = 1e-10 * (n as f64).log2().max(1.0);
        assert!(
            vec_approx_eq(&in_real, &recovered_real, tol),
            "Size {} real roundtrip failed",
            n
        );
        assert!(
            vec_approx_eq(&in_imag, &recovered_imag, tol),
            "Size {} imaginary roundtrip failed",
            n
        );
    }
}

// ============================================================================
// SplitPlan2D tests
// ============================================================================

#[test]
fn test_split_plan_2d_creation() {
    let plan = SplitPlan2D::<f64>::new(8, 8, Direction::Forward, Flags::ESTIMATE);
    assert!(plan.is_some());

    let plan = plan.unwrap();
    assert_eq!(plan.rows(), 8);
    assert_eq!(plan.cols(), 8);
    assert_eq!(plan.size(), 64);
}

#[test]
fn test_split_plan_2d_matches_interleaved() {
    use oxifft::Plan2D;

    let n0 = 8;
    let n1 = 8;
    let n = n0 * n1;

    let in_real: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let in_imag: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();

    // Interleaved format
    let interleaved: Vec<Complex<f64>> = in_real
        .iter()
        .zip(in_imag.iter())
        .map(|(&re, &im)| Complex::new(re, im))
        .collect();

    // Compute using interleaved
    let plan = Plan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE).unwrap();
    let mut interleaved_out = vec![Complex::new(0.0, 0.0); n];
    plan.execute(&interleaved, &mut interleaved_out);

    // Compute using split
    let split_plan = SplitPlan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE).unwrap();
    let mut out_real = vec![0.0; n];
    let mut out_imag = vec![0.0; n];
    split_plan.execute(&in_real, &in_imag, &mut out_real, &mut out_imag);

    // Compare
    let expected_real: Vec<f64> = interleaved_out.iter().map(|c| c.re).collect();
    let expected_imag: Vec<f64> = interleaved_out.iter().map(|c| c.im).collect();

    assert!(
        vec_approx_eq(&out_real, &expected_real, 1e-10),
        "2D real parts don't match"
    );
    assert!(
        vec_approx_eq(&out_imag, &expected_imag, 1e-10),
        "2D imaginary parts don't match"
    );
}

#[test]
fn test_split_plan_2d_roundtrip() {
    let n0 = 8;
    let n1 = 8;
    let n = n0 * n1;

    let in_real: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
    let in_imag: Vec<f64> = (0..n).map(|i| (i as f64) * 0.05).collect();

    // Forward
    let forward = SplitPlan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE).unwrap();
    let mut freq_real = vec![0.0; n];
    let mut freq_imag = vec![0.0; n];
    forward.execute(&in_real, &in_imag, &mut freq_real, &mut freq_imag);

    // Backward
    let backward = SplitPlan2D::new(n0, n1, Direction::Backward, Flags::ESTIMATE).unwrap();
    let mut recovered_real = vec![0.0; n];
    let mut recovered_imag = vec![0.0; n];
    backward.execute(
        &freq_real,
        &freq_imag,
        &mut recovered_real,
        &mut recovered_imag,
    );

    // Normalize
    let scale = 1.0 / (n as f64);
    for r in &mut recovered_real {
        *r *= scale;
    }
    for i in &mut recovered_imag {
        *i *= scale;
    }

    assert!(
        vec_approx_eq(&in_real, &recovered_real, 1e-10),
        "2D real roundtrip failed"
    );
    assert!(
        vec_approx_eq(&in_imag, &recovered_imag, 1e-10),
        "2D imaginary roundtrip failed"
    );
}

#[test]
fn test_split_plan_2d_inplace() {
    let n0 = 8;
    let n1 = 8;
    let n = n0 * n1;

    let in_real: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let in_imag: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();

    let plan = SplitPlan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE).unwrap();

    // Out-of-place reference
    let mut out_real = vec![0.0; n];
    let mut out_imag = vec![0.0; n];
    plan.execute(&in_real, &in_imag, &mut out_real, &mut out_imag);

    // In-place
    let mut ip_real = in_real.clone();
    let mut ip_imag = in_imag.clone();
    plan.execute_inplace(&mut ip_real, &mut ip_imag);

    assert!(
        vec_approx_eq(&ip_real, &out_real, 1e-10),
        "2D in-place real mismatch"
    );
    assert!(
        vec_approx_eq(&ip_imag, &out_imag, 1e-10),
        "2D in-place imaginary mismatch"
    );
}

// ============================================================================
// Convenience function tests
// ============================================================================

#[test]
fn test_fft_split_convenience() {
    let n = 16;

    let in_real: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let in_imag: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();

    let (out_real, out_imag) = fft_split(&in_real, &in_imag);

    assert_eq!(out_real.len(), n);
    assert_eq!(out_imag.len(), n);

    // Verify by comparing with plan
    let plan = SplitPlan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
    let mut expected_real = vec![0.0; n];
    let mut expected_imag = vec![0.0; n];
    plan.execute(&in_real, &in_imag, &mut expected_real, &mut expected_imag);

    assert!(vec_approx_eq(&out_real, &expected_real, 1e-12));
    assert!(vec_approx_eq(&out_imag, &expected_imag, 1e-12));
}

#[test]
fn test_ifft_split_convenience() {
    let n = 16;

    let in_real: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let in_imag: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();

    // Forward
    let (freq_real, freq_imag) = fft_split(&in_real, &in_imag);

    // Backward (normalized)
    let (recovered_real, recovered_imag) = ifft_split(&freq_real, &freq_imag);

    assert!(
        vec_approx_eq(&in_real, &recovered_real, 1e-12),
        "fft_split/ifft_split roundtrip real failed"
    );
    assert!(
        vec_approx_eq(&in_imag, &recovered_imag, 1e-12),
        "fft_split/ifft_split roundtrip imaginary failed"
    );
}

#[test]
fn test_fft_split_empty() {
    let (out_real, out_imag): (Vec<f64>, Vec<f64>) = fft_split(&[], &[]);
    assert!(out_real.is_empty());
    assert!(out_imag.is_empty());
}

#[test]
fn test_ifft_split_empty() {
    let (out_real, out_imag): (Vec<f64>, Vec<f64>) = ifft_split(&[], &[]);
    assert!(out_real.is_empty());
    assert!(out_imag.is_empty());
}

// ============================================================================
// f32 tests
// ============================================================================

#[test]
fn test_split_plan_f32() {
    let n = 16;

    let in_real: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
    let in_imag: Vec<f32> = (0..n).map(|i| (i as f32).cos()).collect();

    let plan = SplitPlan::<f32>::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
    let mut out_real = vec![0.0f32; n];
    let mut out_imag = vec![0.0f32; n];
    plan.execute(&in_real, &in_imag, &mut out_real, &mut out_imag);

    // Verify roundtrip
    let backward = SplitPlan::<f32>::dft_1d(n, Direction::Backward, Flags::ESTIMATE).unwrap();
    let mut recovered_real = vec![0.0f32; n];
    let mut recovered_imag = vec![0.0f32; n];
    backward.execute(
        &out_real,
        &out_imag,
        &mut recovered_real,
        &mut recovered_imag,
    );

    // Normalize
    let scale = 1.0 / (n as f32);
    for r in &mut recovered_real {
        *r *= scale;
    }
    for i in &mut recovered_imag {
        *i *= scale;
    }

    // f32 has lower precision
    let tol = 1e-5f32;
    for (i, ((&a, &b), (&c, &d))) in in_real
        .iter()
        .zip(in_imag.iter())
        .zip(recovered_real.iter().zip(recovered_imag.iter()))
        .enumerate()
    {
        assert!(
            (a - c).abs() < tol && (b - d).abs() < tol,
            "f32 roundtrip failed at {}: ({}, {}) vs ({}, {})",
            i,
            a,
            b,
            c,
            d
        );
    }
}
