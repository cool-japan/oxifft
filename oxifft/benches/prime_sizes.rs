//! Prime-size FFT benchmarks using Bluestein's and Rader's algorithms.
//!
//! Measures forward/inverse f64 and forward f32 performance at sizes
//! 17, 97, 257, 1009, 4093 — all prime, exercising the Bluestein/Rader
//! code paths that were SIMD-optimized in this release.

#![allow(clippy::cast_precision_loss)] // FFT sizes fit safely in f64 mantissa

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oxifft::{Complex, Direction, Flags, Plan};

fn bench_prime_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("prime_sizes");
    group.sample_size(20);

    for &size in &[17_usize, 97, 257, 1009, 4093] {
        // --- forward f64 ---
        group.bench_with_input(BenchmarkId::new("forward_f64", size), &size, |b, &n| {
            let plan = Plan::<f64>::dft_1d(n, Direction::Forward, Flags::ESTIMATE).expect("plan");
            let data: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(i as f64, 0.0)).collect();
            let mut output = vec![Complex::<f64>::zero(); n];
            b.iter(|| {
                plan.execute(&data, &mut output);
            });
        });

        // --- inverse f64 ---
        group.bench_with_input(BenchmarkId::new("inverse_f64", size), &size, |b, &n| {
            let plan = Plan::<f64>::dft_1d(n, Direction::Backward, Flags::ESTIMATE).expect("plan");
            let data: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(i as f64, 0.0)).collect();
            let mut output = vec![Complex::<f64>::zero(); n];
            b.iter(|| {
                plan.execute(&data, &mut output);
            });
        });

        // --- forward f32 ---
        group.bench_with_input(BenchmarkId::new("forward_f32", size), &size, |b, &n| {
            let plan = Plan::<f32>::dft_1d(n, Direction::Forward, Flags::ESTIMATE).expect("plan");
            let data: Vec<Complex<f32>> = (0..n).map(|i| Complex::new(i as f32, 0.0)).collect();
            let mut output = vec![Complex::<f32>::zero(); n];
            b.iter(|| {
                plan.execute(&data, &mut output);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_prime_sizes);
criterion_main!(benches);
