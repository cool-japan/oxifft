//! DCT/DST benchmark group.
//!
//! Covers DCT-I/II/III/IV and DST-I/II/III/IV across 9 sizes, demonstrating
//! O(n log n) growth for n ≥ 16 (FFT-based fast path) vs the O(n²) direct
//! path for n < 16.

#![allow(clippy::cast_precision_loss)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oxifft::reodft::{dct_i, dct_ii, dct_iii, dct_iv, dst_i, dst_ii, dst_iii, dst_iv};
use std::hint::black_box;

/// Sizes to benchmark.  8 and 16 exercise the n<16 direct O(n²) path and the
/// threshold to the O(n log n) FFT path; 32–2048 show the fast path scaling.
const SIZES: &[usize] = &[8, 16, 32, 64, 128, 256, 512, 1024, 2048];

/// Build a non-trivial real input vector of length `n`.
fn make_input(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let s = (i as f64 + 1.0).sin() * 0.5;
            let c = (i as f64 * 0.31416_f64).cos() * 0.5;
            s + c
        })
        .collect()
}

// ---------------------------------------------------------------------------
// DCT benchmarks
// ---------------------------------------------------------------------------

fn bench_dct_i(c: &mut Criterion) {
    let mut group = c.benchmark_group("dct_i");
    for &n in SIZES {
        let input = make_input(n);
        let mut output = vec![0.0_f64; n];
        group.bench_with_input(BenchmarkId::new("oxifft", n), &n, |b, _| {
            b.iter(|| {
                dct_i(black_box(&input), black_box(&mut output));
            });
        });
    }
    group.finish();
}

fn bench_dct_ii(c: &mut Criterion) {
    let mut group = c.benchmark_group("dct_ii");
    for &n in SIZES {
        let input = make_input(n);
        let mut output = vec![0.0_f64; n];
        group.bench_with_input(BenchmarkId::new("oxifft", n), &n, |b, _| {
            b.iter(|| {
                dct_ii(black_box(&input), black_box(&mut output));
            });
        });
    }
    group.finish();
}

fn bench_dct_iii(c: &mut Criterion) {
    let mut group = c.benchmark_group("dct_iii");
    for &n in SIZES {
        let input = make_input(n);
        let mut output = vec![0.0_f64; n];
        group.bench_with_input(BenchmarkId::new("oxifft", n), &n, |b, _| {
            b.iter(|| {
                dct_iii(black_box(&input), black_box(&mut output));
            });
        });
    }
    group.finish();
}

fn bench_dct_iv(c: &mut Criterion) {
    let mut group = c.benchmark_group("dct_iv");
    for &n in SIZES {
        let input = make_input(n);
        let mut output = vec![0.0_f64; n];
        group.bench_with_input(BenchmarkId::new("oxifft", n), &n, |b, _| {
            b.iter(|| {
                dct_iv(black_box(&input), black_box(&mut output));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// DST benchmarks
// ---------------------------------------------------------------------------

fn bench_dst_i(c: &mut Criterion) {
    let mut group = c.benchmark_group("dst_i");
    for &n in SIZES {
        let input = make_input(n);
        let mut output = vec![0.0_f64; n];
        group.bench_with_input(BenchmarkId::new("oxifft", n), &n, |b, _| {
            b.iter(|| {
                dst_i(black_box(&input), black_box(&mut output));
            });
        });
    }
    group.finish();
}

fn bench_dst_ii(c: &mut Criterion) {
    let mut group = c.benchmark_group("dst_ii");
    for &n in SIZES {
        let input = make_input(n);
        let mut output = vec![0.0_f64; n];
        group.bench_with_input(BenchmarkId::new("oxifft", n), &n, |b, _| {
            b.iter(|| {
                dst_ii(black_box(&input), black_box(&mut output));
            });
        });
    }
    group.finish();
}

fn bench_dst_iii(c: &mut Criterion) {
    let mut group = c.benchmark_group("dst_iii");
    for &n in SIZES {
        let input = make_input(n);
        let mut output = vec![0.0_f64; n];
        group.bench_with_input(BenchmarkId::new("oxifft", n), &n, |b, _| {
            b.iter(|| {
                dst_iii(black_box(&input), black_box(&mut output));
            });
        });
    }
    group.finish();
}

fn bench_dst_iv(c: &mut Criterion) {
    let mut group = c.benchmark_group("dst_iv");
    for &n in SIZES {
        let input = make_input(n);
        let mut output = vec![0.0_f64; n];
        group.bench_with_input(BenchmarkId::new("oxifft", n), &n, |b, _| {
            b.iter(|| {
                dst_iv(black_box(&input), black_box(&mut output));
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
    bench_dct_i,
    bench_dct_ii,
    bench_dct_iii,
    bench_dct_iv,
    bench_dst_i,
    bench_dst_ii,
    bench_dst_iii,
    bench_dst_iv
);

criterion_main!(benches);
