//! Regression tests for the split-complex normalization reconciliation.
//!
//! Before the fix, `SplitPlan3D`/`SplitPlanND::execute` normalized the inverse
//! transform internally (by `1/N`) while `SplitPlan`/`SplitPlan2D::execute` did
//! not, and the `ifft2d_split` convenience wrapper normalized *nowhere* — so
//! `ifft2d_split(fft2d_split(x))` returned `N * x` instead of `x`, contradicting
//! its "with normalization" contract.
//!
//! The reconciliation makes every `SplitPlan*::execute` unnormalized (matching
//! the corresponding `Plan*`) and applies the single `1/N` inverse
//! normalization in every `ifft*_split` convenience wrapper. These tests pin the
//! round-trip identity for all ranks.

#![allow(clippy::cast_precision_loss)] // test math casts indices to float
#![allow(clippy::suboptimal_flops)] // clarity over fused multiply-add in test signals

use oxifft::{
    fft2d_split, fft3d_split, fft_nd_split, fft_split, ifft2d_split, ifft3d_split, ifft_nd_split,
    ifft_split,
};

fn signal(n: usize) -> (Vec<f64>, Vec<f64>) {
    let re: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.37).sin() + 0.5 * i as f64)
        .collect();
    let im: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.21).cos() - 0.3 * i as f64)
        .collect();
    (re, im)
}

fn assert_close(got: &[f64], want: &[f64], tag: &str) {
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-9, "{tag}[{i}]: got {g}, want {w}");
    }
}

#[test]
fn ifft_split_1d_roundtrip_identity() {
    for n in [1usize, 2, 3, 4, 5, 8, 12, 16] {
        let (re, im) = signal(n);
        let (fr, fi) = fft_split(&re, &im);
        let (rr, ri) = ifft_split(&fr, &fi);
        assert_close(&rr, &re, "1d re");
        assert_close(&ri, &im, "1d im");
    }
}

#[test]
fn ifft2d_split_roundtrip_identity() {
    // This is the exact bug: previously returned n0*n1*x.
    for (n0, n1) in [(2usize, 3usize), (4, 4), (3, 5), (2, 8)] {
        let (re, im) = signal(n0 * n1);
        let (fr, fi) = fft2d_split(&re, &im, n0, n1);
        let (rr, ri) = ifft2d_split(&fr, &fi, n0, n1);
        assert_close(&rr, &re, "2d re");
        assert_close(&ri, &im, "2d im");
    }
}

#[test]
fn ifft3d_split_roundtrip_identity() {
    for (a, b, c) in [(2usize, 2usize, 2usize), (2, 3, 4), (3, 3, 3)] {
        let (re, im) = signal(a * b * c);
        let (fr, fi) = fft3d_split(&re, &im, a, b, c);
        let (rr, ri) = ifft3d_split(&fr, &fi, a, b, c);
        assert_close(&rr, &re, "3d re");
        assert_close(&ri, &im, "3d im");
    }
}

#[test]
fn ifft_nd_split_roundtrip_identity() {
    for dims in [vec![4usize], vec![2, 3], vec![2, 2, 2], vec![2, 2, 2, 3]] {
        let total: usize = dims.iter().product();
        let (re, im) = signal(total);
        let (fr, fi) = fft_nd_split(&re, &im, &dims);
        let (rr, ri) = ifft_nd_split(&fr, &fi, &dims);
        assert_close(&rr, &re, "nd re");
        assert_close(&ri, &im, "nd im");
    }
}

/// The split-complex forward FFT must equal the equivalent interleaved forward
/// FFT (same unnormalized convention), for 3D — verifying the delegation to the
/// SIMD `Plan3D` preserves the transform, and that flags flow through.
#[test]
fn fft3d_split_matches_interleaved_plan() {
    use oxifft::{Complex, Direction, Flags, Plan3D};
    let (a, b, c) = (2usize, 3usize, 4usize);
    let total = a * b * c;
    let (re, im) = signal(total);
    let (fr, fi) = fft3d_split(&re, &im, a, b, c);

    let input: Vec<Complex<f64>> = re
        .iter()
        .zip(im.iter())
        .map(|(&r, &i)| Complex::new(r, i))
        .collect();
    let mut output = vec![Complex::new(0.0, 0.0); total];
    let plan = Plan3D::<f64>::new(a, b, c, Direction::Forward, Flags::ESTIMATE).expect("plan");
    plan.execute(&input, &mut output);

    for (i, c) in output.iter().enumerate() {
        assert!(
            (fr[i] - c.re).abs() < 1e-9,
            "re[{i}]: {} != {}",
            fr[i],
            c.re
        );
        assert!(
            (fi[i] - c.im).abs() < 1e-9,
            "im[{i}]: {} != {}",
            fi[i],
            c.im
        );
    }
}
