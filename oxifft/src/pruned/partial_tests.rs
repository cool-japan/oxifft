//! Tests for `pruned::partial` (PartialFft / PartialStrategy).

use crate::api::{Direction, Flags, Plan};
use crate::kernel::Complex;
use crate::prelude::*;

use super::partial::{PartialFft, PartialStrategy};

// ─── Helper ──────────────────────────────────────────────────────────────────

/// Run a full forward FFT and return the complete output vector.
fn full_fft(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let n = input.len();
    let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
    let mut output = vec![Complex::new(0.0_f64, 0.0); n];
    plan.execute(input, &mut output);
    output
}

/// Absolute difference between two complex numbers.
fn cdiff(a: Complex<f64>, b: Complex<f64>) -> f64 {
    let dr = a.re - b.re;
    let di = a.im - b.im;
    (dr * dr + di * di).sqrt()
}

// ─── Test 1: Goertzel single bin ─────────────────────────────────────────────

#[test]
fn test_goertzel_single_bin_vs_full_fft() {
    let n = 64;
    let input: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new((i as f64) / (n as f64), 0.0))
        .collect();

    let full = full_fft(&input);

    let bin = 5_usize;
    let pf = PartialFft::<f64>::new_sparse(n, &[bin]);
    let mut out = vec![Complex::new(0.0_f64, 0.0); 1];
    pf.execute(&input, &mut out);

    assert!(
        cdiff(out[0], full[bin]) < 1e-10,
        "bin {bin}: got {:?}, expected {:?}",
        out[0],
        full[bin]
    );
}

// ─── Test 2: Goertzel multiple bins ──────────────────────────────────────────

#[test]
fn test_goertzel_multi_bins_vs_full_fft() {
    let n = 256;
    let input: Vec<Complex<f64>> = (0..n)
        .map(|i| {
            let t = (i as f64) / (n as f64);
            Complex::new((3.0 * t).sin(), (7.0 * t).cos())
        })
        .collect();

    let full = full_fft(&input);

    // K=4 bins; 4 < log2(256)=8, so Goertzel is chosen
    let bins = [3_usize, 17, 50, 200];
    let pf = PartialFft::<f64>::new_sparse(n, &bins);
    let mut out = vec![Complex::new(0.0_f64, 0.0); bins.len()];
    pf.execute(&input, &mut out);

    for (i, &bin) in bins.iter().enumerate() {
        assert!(
            cdiff(out[i], full[bin]) < 1e-10,
            "bin {bin}: got {:?}, expected {:?}",
            out[i],
            full[bin]
        );
    }
}

// ─── Test 3: Goertzel edge cases ─────────────────────────────────────────────

#[test]
fn test_goertzel_edge_cases() {
    let n = 64_usize;
    let input: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new(1.0 + (i as f64) * 0.1, (i as f64) * 0.05))
        .collect();

    let full = full_fft(&input);

    // DC bin (k=0)
    let pf_dc = PartialFft::<f64>::new_sparse(n, &[0]);
    let mut out_dc = vec![Complex::new(0.0_f64, 0.0); 1];
    pf_dc.execute(&input, &mut out_dc);
    assert!(
        cdiff(out_dc[0], full[0]) < 1e-10,
        "DC bin mismatch: got {:?}, expected {:?}",
        out_dc[0],
        full[0]
    );

    // Nyquist bin (k = N/2)
    let nyquist = n / 2;
    let pf_ny = PartialFft::<f64>::new_sparse(n, &[nyquist]);
    let mut out_ny = vec![Complex::new(0.0_f64, 0.0); 1];
    pf_ny.execute(&input, &mut out_ny);
    assert!(
        cdiff(out_ny[0], full[nyquist]) < 1e-10,
        "Nyquist bin mismatch: got {:?}, expected {:?}",
        out_ny[0],
        full[nyquist]
    );

    // Last bin (k = N-1)
    let last = n - 1;
    let pf_last = PartialFft::<f64>::new_sparse(n, &[last]);
    let mut out_last = vec![Complex::new(0.0_f64, 0.0); 1];
    pf_last.execute(&input, &mut out_last);
    assert!(
        cdiff(out_last[0], full[last]) < 1e-10,
        "Last bin mismatch: got {:?}, expected {:?}",
        out_last[0],
        full[last]
    );
}

// ─── Test 4: OutputPruned prefix ─────────────────────────────────────────────

#[test]
fn test_output_pruned_prefix_vs_full_fft() {
    let n = 1024_usize;
    let input: Vec<Complex<f64>> = (0..n)
        .map(|i| {
            let t = (i as f64) / (n as f64);
            Complex::new(t.sin() + (5.0 * t).cos(), t * 0.1)
        })
        .collect();

    let full = full_fft(&input);

    // m=16, n=1024: log2(1024)=10, 16 > 10 so strategy is actually FullThenSlice
    // Use m=8 to get OutputPruned: 8 < 10 and 8 is power-of-two and 8 <= 512
    let m = 8_usize;
    let pf = PartialFft::<f64>::new_prefix(n, m);

    // Verify OutputPruned strategy was chosen
    assert!(
        matches!(pf.strategy(), PartialStrategy::OutputPruned { m: 8 }),
        "Expected OutputPruned, got {:?}",
        pf.strategy()
    );

    let mut out = vec![Complex::new(0.0_f64, 0.0); m];
    pf.execute(&input, &mut out);

    // Goertzel on N=1024 accumulates O(N·eps) ≈ 1e-9 error; use 2e-8 tolerance.
    // MIRI soft-float accumulates more rounding error than hardware, so we use 2e-8.
    for k in 0..m {
        assert!(
            cdiff(out[k], full[k]) < 2e-8,
            "prefix bin {k}: got {:?}, expected {:?}",
            out[k],
            full[k]
        );
    }
}

// ─── Test 5: FullThenSlice ───────────────────────────────────────────────────

#[test]
fn test_full_then_slice_vs_full_fft() {
    let n = 1024_usize;
    let input: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new((i as f64).sin(), (i as f64 * 1.3).cos()))
        .collect();

    let full = full_fft(&input);

    // m=900: 900 >= log2(1024)=10, so FullThenSlice is chosen
    let m = 900_usize;
    let pf = PartialFft::<f64>::new_prefix(n, m);

    assert!(
        matches!(pf.strategy(), PartialStrategy::FullThenSlice { .. }),
        "Expected FullThenSlice, got {:?}",
        pf.strategy()
    );

    let mut out = vec![Complex::new(0.0_f64, 0.0); m];
    pf.execute(&input, &mut out);

    for k in 0..m {
        assert!(
            cdiff(out[k], full[k]) < 1e-10,
            "slice bin {k}: got {:?}, expected {:?}",
            out[k],
            full[k]
        );
    }
}

// ─── Test 6: Strategy crossover assertions ───────────────────────────────────

#[test]
fn test_strategy_crossover_sparse_goertzel() {
    // K=1 << log2(1024)=10: must pick Goertzel
    let pf = PartialFft::<f64>::new_sparse(1024, &[5]);
    assert!(
        matches!(pf.strategy(), PartialStrategy::Goertzel { .. }),
        "Expected Goertzel for K=1, got {:?}",
        pf.strategy()
    );
}

#[test]
fn test_strategy_crossover_prefix_output_pruned() {
    // n=1024, m=8: 8 < 10, power-of-two, 8 <= 512 → OutputPruned
    let pf = PartialFft::<f64>::new_prefix(1024, 8);
    assert!(
        matches!(pf.strategy(), PartialStrategy::OutputPruned { m: 8 }),
        "Expected OutputPruned for m=8, n=1024, got {:?}",
        pf.strategy()
    );
}

#[test]
fn test_strategy_crossover_prefix_full_then_slice() {
    // n=1024, m=900: 900 >= log2(1024)=10 → FullThenSlice
    let pf = PartialFft::<f64>::new_prefix(1024, 900);
    assert!(
        matches!(pf.strategy(), PartialStrategy::FullThenSlice { .. }),
        "Expected FullThenSlice for m=900, n=1024, got {:?}",
        pf.strategy()
    );
}

// ─── Test 7: Numerical stability at edge frequencies ─────────────────────────

#[test]
fn test_numerical_stability_dc_bin() {
    // bin 0: omega = 0.0 — potential cancellation in naive implementations
    let n = 256_usize;
    let input: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); n];
    let full = full_fft(&input);

    let pf = PartialFft::<f64>::new_sparse(n, &[0]);
    let mut out = vec![Complex::new(0.0_f64, 0.0); 1];
    pf.execute(&input, &mut out);

    // DC of constant 1 signal = N
    assert!(
        (out[0].re - n as f64).abs() < 1e-10,
        "DC real: expected {}, got {}",
        n,
        out[0].re
    );
    assert!(
        out[0].im.abs() < 1e-10,
        "DC imag should be ~0, got {}",
        out[0].im
    );
    assert!(
        cdiff(out[0], full[0]) < 1e-10,
        "DC stability: got {:?}, expected {:?}",
        out[0],
        full[0]
    );
}

#[test]
fn test_numerical_stability_nyquist_bin() {
    // bin N/2: omega = pi — another potential catastrophic cancellation point
    let n = 256_usize;
    let input: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new(if i % 2 == 0 { 1.0 } else { -1.0 }, 0.0))
        .collect();

    let full = full_fft(&input);
    let nyquist = n / 2;

    let pf = PartialFft::<f64>::new_sparse(n, &[nyquist]);
    let mut out = vec![Complex::new(0.0_f64, 0.0); 1];
    pf.execute(&input, &mut out);

    // MIRI soft-float accumulates more rounding error than hardware; use 2e-9 here.
    assert!(
        cdiff(out[0], full[nyquist]) < 2e-9,
        "Nyquist stability: got {:?}, expected {:?}",
        out[0],
        full[nyquist]
    );
}
