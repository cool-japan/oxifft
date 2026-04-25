//! Cooley-Tukey scaling benchmarks: forward transform for 2^10 through 2^20.
//!
//! Measures throughput in elements/second for both f64 and f32 forward
//! transforms.  Sizes are chosen to span the range where:
//!   - 1024..4096   : small (fits in L1/L2 cache)
//!   - 16384..65536 : medium (L2/L3 boundary)
//!   - 262144..1048576 : large (main-memory bound)

#![allow(clippy::cast_precision_loss)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxifft::{Complex, Direction, Flags, Plan};
use std::hint::black_box;

/// Forward CT benchmark for f64, sizes 2^10 to 2^20.
fn bench_ct_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("cooley_tukey_f64");

    for &n in &[1024usize, 4096, 16384, 65536, 262_144, 1_048_576] {
        group.throughput(Throughput::Elements(n as u64));

        // Pre-allocate outside the benchmark loop so allocation is not measured.
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut output = vec![Complex::new(0.0_f64, 0.0); n];

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &sz| {
            let plan = Plan::dft_1d(sz, Direction::Forward, Flags::ESTIMATE)
                .expect("failed to create CT plan for f64");
            b.iter(|| {
                plan.execute(&input, &mut output);
                black_box(&output);
            });
        });
    }

    group.finish();
}

/// Forward CT benchmark for f32, sizes 2^10 to 2^20.
fn bench_ct_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("cooley_tukey_f32");

    for &n in &[1024usize, 4096, 16384, 65536, 262_144, 1_048_576] {
        group.throughput(Throughput::Elements(n as u64));

        let input: Vec<Complex<f32>> = (0..n)
            .map(|i| Complex::new((i as f32).sin(), (i as f32).cos()))
            .collect();
        let mut output = vec![Complex::new(0.0_f32, 0.0); n];

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &sz| {
            let plan = Plan::dft_1d(sz, Direction::Forward, Flags::ESTIMATE)
                .expect("failed to create CT plan for f32");
            b.iter(|| {
                plan.execute(&input, &mut output);
                black_box(&output);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_ct_f64, bench_ct_f32);
criterion_main!(benches);
