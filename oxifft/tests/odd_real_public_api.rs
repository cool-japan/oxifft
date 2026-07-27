//! Regression test for the adversarial numerics finding: the public real-FFT
//! *convenience functions* (`oxifft::rfft`/`irfft`, `rfft2d`/`irfft2d`,
//! `rfft_batch`/`irfft_batch`) returned O(1)-wrong results for every odd
//! transform size >= 3 (the pair-packing path silently dropped the last real
//! sample and fabricated a non-existent Nyquist bin).
//!
//! These tests exercise the *free-function* surface (not `RealPlan` directly),
//! which is exactly what the numerics validation flagged, and pin the exact
//! N=3 minimal repro from that report.

#![allow(clippy::cast_precision_loss)] // test math casts indices/sizes to float
#![allow(clippy::suboptimal_flops)] // clarity over fused multiply-add in references

use oxifft::Complex;

fn naive_dft(x: &[f64]) -> Vec<Complex<f64>> {
    let n = x.len();
    let mut out = vec![Complex::new(0.0, 0.0); n];
    for (k, o) in out.iter_mut().enumerate() {
        let mut re = 0.0;
        let mut im = 0.0;
        for (j, &xj) in x.iter().enumerate() {
            let ang = -2.0 * std::f64::consts::PI * (k as f64) * (j as f64) / (n as f64);
            re += xj * ang.cos();
            im += xj * ang.sin();
        }
        *o = Complex::new(re, im);
    }
    out
}

#[test]
fn rfft_n3_exact_minimal_repro() {
    // The exact repro from the numerics report.
    let x = [1.0_f64, 2.0, 3.0];
    let spec = oxifft::rfft(&x); // N/2+1 = 2 bins
    assert_eq!(spec.len(), 2, "rfft(N=3) must yield 2 bins");
    // DC bin must be the sum: 1+2+3 = 6.
    assert!(
        (spec[0].re - 6.0).abs() < 1e-9,
        "DC bin wrong: got {}, want 6.0",
        spec[0].re
    );
    // First AC bin must match the naive DFT bin 1: -1.5 + i*0.8660254...
    let want = naive_dft(&x)[1];
    assert!(
        (spec[1].re - want.re).abs() < 1e-9 && (spec[1].im - want.im).abs() < 1e-9,
        "AC bin wrong: got {:?}, want {:?}",
        spec[1],
        want
    );
    // Round trip must reconstruct the input exactly.
    let back = oxifft::irfft(&spec, 3);
    for (i, (&a, &b)) in x.iter().zip(back.iter()).enumerate() {
        assert!((a - b).abs() < 1e-9, "irfft(rfft(x))[{i}] = {b}, want {a}");
    }
}

#[test]
fn rfft_irfft_odd_sizes_roundtrip_and_spectrum() {
    for n in [3usize, 5, 7, 9, 11, 15, 63, 97, 101] {
        let x: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.7).sin() + 0.25 * i as f64)
            .collect();
        let spec = oxifft::rfft(&x);
        assert_eq!(spec.len(), n / 2 + 1, "n={n}: wrong spectrum length");
        let reference = naive_dft(&x);
        for (k, s) in spec.iter().enumerate() {
            assert!(
                (s.re - reference[k].re).abs() < 1e-8 && (s.im - reference[k].im).abs() < 1e-8,
                "n={n} bin {k}: got {s:?}, want {:?}",
                reference[k]
            );
        }
        let back = oxifft::irfft(&spec, n);
        for (i, (&a, &b)) in x.iter().zip(back.iter()).enumerate() {
            assert!((a - b).abs() < 1e-8, "n={n} irfft[{i}] = {b}, want {a}");
        }
    }
}

#[test]
fn rfft2d_odd_last_dim_roundtrip() {
    // Odd last dimension was corrupted before the fix.
    for (n0, n1) in [(4usize, 5usize), (3, 7), (5, 9)] {
        let total = n0 * n1;
        let x: Vec<f64> = (0..total)
            .map(|i| (i as f64 * 0.31).cos() + 0.1 * i as f64)
            .collect();
        let spec = oxifft::rfft2d(&x, n0, n1);
        let back = oxifft::irfft2d(&spec, n0, n1);
        for (i, (&a, &b)) in x.iter().zip(back.iter()).enumerate() {
            assert!((a - b).abs() < 1e-8, "rfft2d {n0}x{n1} idx={i}: {b} != {a}");
        }
    }
}

/// Naive 2D DFT reference (unnormalized forward), full complex spectrum.
fn naive_dft2d(x: &[f64], n0: usize, n1: usize) -> Vec<Complex<f64>> {
    let mut out = vec![Complex::new(0.0, 0.0); n0 * n1];
    for k0 in 0..n0 {
        for k1 in 0..n1 {
            let mut re = 0.0;
            let mut im = 0.0;
            for j0 in 0..n0 {
                for j1 in 0..n1 {
                    let ang = -2.0
                        * std::f64::consts::PI
                        * ((k0 * j0) as f64 / n0 as f64 + (k1 * j1) as f64 / n1 as f64);
                    let v = x[j0 * n1 + j1];
                    re += v * ang.cos();
                    im += v * ang.sin();
                }
            }
            out[k0 * n1 + k1] = Complex::new(re, im);
        }
    }
    out
}

/// The absolute forward spectrum of `rfft2d` must match a naive 2D DFT
/// (bins 0..n0 x 0..=n1/2). This guards the flag-aware column FFT convention.
#[test]
fn rfft2d_forward_spectrum_matches_naive() {
    for (n0, n1) in [(4usize, 5usize), (3, 6), (5, 7), (4, 8)] {
        let total = n0 * n1;
        let x: Vec<f64> = (0..total)
            .map(|i| (i as f64 * 0.29).sin() + 0.1 * i as f64)
            .collect();
        let spec = oxifft::rfft2d(&x, n0, n1);
        let out_last = n1 / 2 + 1;
        assert_eq!(spec.len(), n0 * out_last, "{n0}x{n1}: wrong out len");
        let reference = naive_dft2d(&x, n0, n1);
        for k0 in 0..n0 {
            for k1 in 0..out_last {
                let got = spec[k0 * out_last + k1];
                let want = reference[k0 * n1 + k1];
                assert!(
                    (got.re - want.re).abs() < 1e-8 && (got.im - want.im).abs() < 1e-8,
                    "{n0}x{n1} bin ({k0},{k1}): got {got:?}, want {want:?}"
                );
            }
        }
    }
}

#[test]
fn rfft_batch_odd_size_roundtrip() {
    // Batch input is flat contiguous: `howmany` signals each of length `n`.
    for n in [5usize, 9, 11] {
        let howmany = 3;
        let input: Vec<f64> = (0..n * howmany)
            .map(|i| (i as f64 * 0.23).sin() + 0.05 * i as f64)
            .collect();
        let spectra = oxifft::rfft_batch(&input, n, howmany);
        assert_eq!(spectra.len(), (n / 2 + 1) * howmany);
        // Per-batch spectrum must match a naive DFT of that batch (bins 0..=n/2).
        let out_len = n / 2 + 1;
        for b in 0..howmany {
            let seg = &input[b * n..(b + 1) * n];
            let reference = naive_dft(seg);
            for k in 0..out_len {
                let s = spectra[b * out_len + k];
                assert!(
                    (s.re - reference[k].re).abs() < 1e-8 && (s.im - reference[k].im).abs() < 1e-8,
                    "batch {b} n={n} bin {k}: got {s:?}, want {:?}",
                    reference[k]
                );
            }
        }
        // Normalized inverse round-trip is the identity.
        let recovered = oxifft::irfft_batch(&spectra, n, howmany);
        for (i, (&a, &c)) in input.iter().zip(recovered.iter()).enumerate() {
            assert!((a - c).abs() < 1e-8, "batch n={n} idx={i}: {c} != {a}");
        }
    }
}
