//! GPU vs CPU FFT benchmarks.
//!
//! Compares `Plan::dft_1d` (CPU Cooley-Tukey) against `GpuFft::new` (GPU) for
//! transform sizes 4096, 16384, 65536, and 262144.
//!
//! **Backend notes:**
//! - Metal (macOS): real GPU backend via Metal Performance Shaders — speedup
//!   expected at larger sizes where data-transfer overhead is amortised.
//! - CUDA: CPU FFT is used as the computation engine until oxicuda-launch GPU
//!   kernel integration lands; GPU numbers will closely match CPU reference.
//!
//! Run with:
//! ```text
//! cargo bench --bench gpu_vs_cpu --features gpu
//! ```

#![cfg(any(feature = "gpu", feature = "metal", feature = "cuda"))]
#![allow(clippy::cast_precision_loss)] // i as f32 / i as f64 for bench input generation

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxifft::{Complex, Direction, Flags, GpuBackend, GpuFft, Plan};
use std::hint::black_box;

/// Sizes to bench: 4 K, 16 K, 64 K, 256 K.
const SIZES: &[usize] = &[4096, 16384, 65536, 262_144];

/// Generate a synthetic input signal of length `n`.
fn make_input_f32(n: usize) -> Vec<Complex<f32>> {
    (0..n)
        .map(|i| Complex::new((i as f32).sin(), (i as f32).cos()))
        .collect()
}

fn make_input_f64(n: usize) -> Vec<Complex<f64>> {
    (0..n)
        .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
        .collect()
}

/// Correctness guard: verify GPU output is within `tol` relative error of CPU.
fn check_gpu_vs_cpu_f32(size: usize) {
    let cpu_plan =
        Plan::dft_1d(size, Direction::Forward, Flags::ESTIMATE).expect("cpu plan creation failed");

    let mut gpu_plan =
        GpuFft::<f32>::new(size, GpuBackend::Auto).expect("gpu plan creation failed");

    let input = make_input_f32(size);

    let mut cpu_out = vec![Complex::zero(); size];
    cpu_plan.execute(&input, &mut cpu_out);

    let gpu_out = gpu_plan.forward(&input).expect("gpu forward failed");

    let max_rel_err = cpu_out
        .iter()
        .zip(gpu_out.iter())
        .filter(|(c, _)| c.norm() > 1e-10_f32)
        .map(|(c, g)| {
            let diff_re = c.re - g.re;
            let diff_im = c.im - g.im;
            diff_re.hypot(diff_im) / c.norm()
        })
        .fold(0.0_f32, f32::max);

    assert!(
        max_rel_err < 1e-3_f32,
        "GPU/CPU f32 diverge at size {size}: max_rel_err = {max_rel_err}",
    );
}

fn bench_gpu_vs_cpu(c: &mut Criterion) {
    // Correctness warmup before benchmarking.
    // This ensures any panics are caught immediately rather than mid-benchmark.
    for &size in SIZES {
        check_gpu_vs_cpu_f32(size);
    }

    // ── CPU f32 ────────────────────────────────────────────────────────────
    {
        let mut group = c.benchmark_group("gpu_vs_cpu/cpu_f32");
        group.sample_size(20);

        for &size in SIZES {
            group.throughput(Throughput::Elements(size as u64));

            // Pre-allocate input and output outside the timed loop.
            let input = make_input_f32(size);
            let mut output = vec![Complex::zero(); size];

            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
                let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE)
                    .expect("cpu plan creation failed");
                b.iter(|| {
                    plan.execute(black_box(&input), &mut output);
                    black_box(&output);
                });
            });
        }

        group.finish();
    }

    // ── GPU f32 ────────────────────────────────────────────────────────────
    {
        let mut group = c.benchmark_group("gpu_vs_cpu/gpu_f32");
        group.sample_size(20);

        for &size in SIZES {
            group.throughput(Throughput::Elements(size as u64));

            let input = make_input_f32(size);

            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
                let mut plan =
                    GpuFft::<f32>::new(n, GpuBackend::Auto).expect("gpu plan creation failed");
                b.iter(|| {
                    let out = plan.forward(black_box(&input)).expect("gpu forward failed");
                    black_box(out);
                });
            });
        }

        group.finish();
    }

    // ── CPU f64 ────────────────────────────────────────────────────────────
    {
        let mut group = c.benchmark_group("gpu_vs_cpu/cpu_f64");
        group.sample_size(20);

        for &size in SIZES {
            group.throughput(Throughput::Elements(size as u64));

            let input = make_input_f64(size);
            let mut output = vec![Complex::zero(); size];

            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
                let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE)
                    .expect("cpu plan creation failed");
                b.iter(|| {
                    plan.execute(black_box(&input), &mut output);
                    black_box(&output);
                });
            });
        }

        group.finish();
    }

    // ── GPU f64 ────────────────────────────────────────────────────────────
    {
        let mut group = c.benchmark_group("gpu_vs_cpu/gpu_f64");
        group.sample_size(20);

        for &size in SIZES {
            group.throughput(Throughput::Elements(size as u64));

            let input = make_input_f64(size);

            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
                let mut plan =
                    GpuFft::<f64>::new(n, GpuBackend::Auto).expect("gpu plan creation failed");
                b.iter(|| {
                    let out = plan.forward(black_box(&input)).expect("gpu forward failed");
                    black_box(out);
                });
            });
        }

        group.finish();
    }
}

criterion_group!(benches, bench_gpu_vs_cpu);
criterion_main!(benches);
