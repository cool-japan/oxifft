//! 1D DFT benchmarks.

#![allow(clippy::cast_precision_loss)] // FFT sizes fit in f64 mantissa

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oxifft::{Complex, Direction, Flags, Plan};

fn benchmark_power_of_2(c: &mut Criterion) {
    let sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192];
    let mut group = c.benchmark_group("fft_power_of_2");

    for &size in &sizes {
        let input: Vec<Complex<f64>> = (0..size)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut output = vec![Complex::zero(); size];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_size| {
            let plan = Plan::dft_1d(size, Direction::Forward, Flags::ESTIMATE)
                .expect("Failed to create plan");
            b.iter(|| {
                plan.execute(&input, &mut output);
            });
        });
    }

    group.finish();
}

fn benchmark_prime_sizes(c: &mut Criterion) {
    let sizes = [17, 31, 61, 127, 251, 509];
    let mut group = c.benchmark_group("fft_prime_sizes");

    for &size in &sizes {
        let input: Vec<Complex<f64>> = (0..size)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut output = vec![Complex::zero(); size];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_size| {
            let plan = Plan::dft_1d(size, Direction::Forward, Flags::ESTIMATE)
                .expect("Failed to create plan");
            b.iter(|| {
                plan.execute(&input, &mut output);
            });
        });
    }

    group.finish();
}

fn benchmark_composite_sizes(c: &mut Criterion) {
    let sizes = [60, 120, 240, 360, 720, 1440];
    let mut group = c.benchmark_group("fft_composite_sizes");

    for &size in &sizes {
        let input: Vec<Complex<f64>> = (0..size)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut output = vec![Complex::zero(); size];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_size| {
            let plan = Plan::dft_1d(size, Direction::Forward, Flags::ESTIMATE)
                .expect("Failed to create plan");
            b.iter(|| {
                plan.execute(&input, &mut output);
            });
        });
    }

    group.finish();
}

fn benchmark_inplace(c: &mut Criterion) {
    let sizes = [128, 512, 2048];
    let mut group = c.benchmark_group("fft_inplace");

    for &size in &sizes {
        let mut data: Vec<Complex<f64>> = (0..size)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_size| {
            let plan = Plan::dft_1d(size, Direction::Forward, Flags::ESTIMATE)
                .expect("Failed to create plan");
            b.iter(|| {
                plan.execute_inplace(&mut data);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_power_of_2,
    benchmark_prime_sizes,
    benchmark_composite_sizes,
    benchmark_inplace
);
criterion_main!(benches);
