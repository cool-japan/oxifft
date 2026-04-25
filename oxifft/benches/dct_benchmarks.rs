//! DCT/DST/DHT benchmarks using the Makhoul O(N log N) fast paths.
//!
//! Benchmarks the cached-plan DCT-II, DCT-III, and DCT-IV implementations
//! at sizes 256, 1024, 4096, and 16384 for both f64 and f32.

#![allow(clippy::cast_precision_loss)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oxifft::rdft::solvers::{R2rKind, R2rSolver};

// ---------------------------------------------------------------------------
// DCT-II benchmarks
// ---------------------------------------------------------------------------

fn bench_dct2_f64(c: &mut Criterion) {
    let sizes = [256usize, 1024, 4096, 16384];
    let mut group = c.benchmark_group("dct2_f64");

    for &n in &sizes {
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut output = vec![0.0_f64; n];
        let solver = R2rSolver::<f64>::new(R2rKind::Redft10, n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| {
                solver.execute_dct2_fast(&input, &mut output);
            });
        });
    }

    group.finish();
}

fn bench_dct2_f32(c: &mut Criterion) {
    let sizes = [256usize, 1024, 4096, 16384];
    let mut group = c.benchmark_group("dct2_f32");

    for &n in &sizes {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut output = vec![0.0_f32; n];
        let solver = R2rSolver::<f32>::new(R2rKind::Redft10, n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| {
                solver.execute_dct2_fast(&input, &mut output);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// DCT-III benchmarks
// ---------------------------------------------------------------------------

fn bench_dct3_f64(c: &mut Criterion) {
    let sizes = [256usize, 1024, 4096, 16384];
    let mut group = c.benchmark_group("dct3_f64");

    for &n in &sizes {
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut output = vec![0.0_f64; n];
        let solver = R2rSolver::<f64>::new(R2rKind::Redft01, n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| {
                solver.execute_dct3_fast(&input, &mut output);
            });
        });
    }

    group.finish();
}

fn bench_dct3_f32(c: &mut Criterion) {
    let sizes = [256usize, 1024, 4096, 16384];
    let mut group = c.benchmark_group("dct3_f32");

    for &n in &sizes {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut output = vec![0.0_f32; n];
        let solver = R2rSolver::<f32>::new(R2rKind::Redft01, n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| {
                solver.execute_dct3_fast(&input, &mut output);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// DCT-IV benchmarks
// ---------------------------------------------------------------------------

fn bench_dct4_f64(c: &mut Criterion) {
    let sizes = [256usize, 1024, 4096, 16384];
    let mut group = c.benchmark_group("dct4_f64");

    for &n in &sizes {
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut output = vec![0.0_f64; n];
        let solver = R2rSolver::<f64>::new(R2rKind::Redft11, n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| {
                solver.execute_dct4_fast(&input, &mut output);
            });
        });
    }

    group.finish();
}

fn bench_dct4_f32(c: &mut Criterion) {
    let sizes = [256usize, 1024, 4096, 16384];
    let mut group = c.benchmark_group("dct4_f32");

    for &n in &sizes {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut output = vec![0.0_f32; n];
        let solver = R2rSolver::<f32>::new(R2rKind::Redft11, n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| {
                solver.execute_dct4_fast(&input, &mut output);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Plan-cached vs per-call-planned comparison (DCT-II f64)
// ---------------------------------------------------------------------------

fn bench_dct2_cached_vs_tmp_f64(c: &mut Criterion) {
    let n = 1024_usize;
    let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
    let mut output = vec![0.0_f64; n];

    let mut group = c.benchmark_group("dct2_cached_vs_tmp_n1024_f64");

    // Cached plan: solver built with n=1024; plan_fwd_n is precomputed.
    let solver_cached = R2rSolver::<f64>::new(R2rKind::Redft10, n);
    group.bench_function("cached_plan", |b| {
        b.iter(|| {
            solver_cached.execute_dct2_fast(&input, &mut output);
        });
    });

    // Per-call planning: solver built with n=0; forces execute_dct2_fast_tmp.
    let solver_tmp = R2rSolver::<f64>::new(R2rKind::Redft10, 0);
    group.bench_function("tmp_plan", |b| {
        b.iter(|| {
            solver_tmp.execute_dct2_fast(&input, &mut output);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_dct2_f64,
    bench_dct2_f32,
    bench_dct3_f64,
    bench_dct3_f32,
    bench_dct4_f64,
    bench_dct4_f32,
    bench_dct2_cached_vs_tmp_f64,
);
criterion_main!(benches);
