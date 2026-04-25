//! R2C/C2R performance regression bench.
//!
//! Covers `RealPlan::r2c_1d` and `c2r_1d` across 11 sizes × f32/f64.
//! The plan is created **outside** the iter loop; only the execution step is
//! timed.  Stable `BenchmarkId` names allow future regression tracking with
//! `--save-baseline v0.3.0-rc1`.

#![allow(clippy::cast_precision_loss)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oxifft::{Complex, Flags, RealPlan};
use std::hint::black_box;
use std::time::Duration;

/// Sizes to benchmark: {16 … 16384} (all power-of-2, 11 entries).
const SIZES: &[usize] = &[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];

// ---------------------------------------------------------------------------
// R2C — f64
// ---------------------------------------------------------------------------

fn bench_r2c_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("r2c_f64");
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    for &n in SIZES {
        let plan = RealPlan::<f64>::r2c_1d(n, Flags::ESTIMATE)
            .expect("r2c_1d plan creation must succeed for non-zero n");
        let input: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let mut output: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n / 2 + 1];

        group.bench_with_input(BenchmarkId::new("r2c_f64", n), &n, |b, _| {
            b.iter(|| {
                plan.execute_r2c(black_box(&input), black_box(&mut output));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// R2C — f32
// ---------------------------------------------------------------------------

fn bench_r2c_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("r2c_f32");
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    for &n in SIZES {
        let plan = RealPlan::<f32>::r2c_1d(n, Flags::ESTIMATE)
            .expect("r2c_1d plan creation must succeed for non-zero n");
        let input: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let mut output: Vec<Complex<f32>> = vec![Complex::new(0.0_f32, 0.0_f32); n / 2 + 1];

        group.bench_with_input(BenchmarkId::new("r2c_f32", n), &n, |b, _| {
            b.iter(|| {
                plan.execute_r2c(black_box(&input), black_box(&mut output));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// C2R — f64
// ---------------------------------------------------------------------------

fn bench_c2r_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("c2r_f64");
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    for &n in SIZES {
        let plan = RealPlan::<f64>::c2r_1d(n, Flags::ESTIMATE)
            .expect("c2r_1d plan creation must succeed for non-zero n");
        let complex_len = n / 2 + 1;
        let input: Vec<Complex<f64>> = (0..complex_len)
            .map(|i| Complex::new((i as f64).cos(), (i as f64).sin()))
            .collect();
        let mut output: Vec<f64> = vec![0.0_f64; n];

        group.bench_with_input(BenchmarkId::new("c2r_f64", n), &n, |b, _| {
            b.iter(|| {
                plan.execute_c2r(black_box(&input), black_box(&mut output));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// C2R — f32
// ---------------------------------------------------------------------------

fn bench_c2r_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("c2r_f32");
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    for &n in SIZES {
        let plan = RealPlan::<f32>::c2r_1d(n, Flags::ESTIMATE)
            .expect("c2r_1d plan creation must succeed for non-zero n");
        let complex_len = n / 2 + 1;
        let input: Vec<Complex<f32>> = (0..complex_len)
            .map(|i| Complex::new((i as f32).cos(), (i as f32).sin()))
            .collect();
        let mut output: Vec<f32> = vec![0.0_f32; n];

        group.bench_with_input(BenchmarkId::new("c2r_f32", n), &n, |b, _| {
            b.iter(|| {
                plan.execute_c2r(black_box(&input), black_box(&mut output));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_r2c_f64,
    bench_r2c_f32,
    bench_c2r_f64,
    bench_c2r_f32
);

criterion_main!(benches);
