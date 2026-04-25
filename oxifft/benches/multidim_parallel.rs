//! Multi-dimensional parallel FFT benchmarks.
//!
//! Measures throughput of `Plan2D` and `Plan3D` at several sizes, exercising
//! rayon work-stealing parallelism on row/plane decompositions.
//!
//! Run with:
//! ```bash
//! cargo bench -p oxifft --bench multidim_parallel -- --sample-size 5
//! ```

#![allow(clippy::cast_precision_loss)] // FFT sizes fit in f64 mantissa

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxifft::{Complex, Direction, Flags, Plan2D, Plan3D};

// ── Plan2D benchmarks ────────────────────────────────────────────────────────

fn bench_plan2d_forward(c: &mut Criterion) {
    let sizes: &[(usize, usize)] = &[(256, 256), (512, 512), (1024, 1024)];

    let mut group = c.benchmark_group("plan2d_forward");
    group.sample_size(10);

    for &(rows, cols) in sizes {
        let total = rows * cols;
        let input: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new((i as f64 * 0.007).sin(), (i as f64 * 0.013).cos()))
            .collect();
        let mut output = vec![Complex::<f64>::zero(); total];

        let name = format!("{rows}x{cols}");
        group.throughput(Throughput::Elements(total as u64));
        group.bench_with_input(
            BenchmarkId::new("parallel", &name),
            &(rows, cols),
            |b, &(r, c)| {
                let plan = Plan2D::new(r, c, Direction::Forward, Flags::ESTIMATE)
                    .expect("plan creation failed");
                b.iter(|| plan.execute(&input, &mut output));
            },
        );
    }

    group.finish();
}

fn bench_plan2d_inverse(c: &mut Criterion) {
    let sizes: &[(usize, usize)] = &[(256, 256), (512, 512), (1024, 1024)];

    let mut group = c.benchmark_group("plan2d_inverse");
    group.sample_size(10);

    for &(rows, cols) in sizes {
        let total = rows * cols;
        let input: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new((i as f64 * 0.007).sin(), (i as f64 * 0.013).cos()))
            .collect();
        let mut output = vec![Complex::<f64>::zero(); total];

        let name = format!("{rows}x{cols}");
        group.throughput(Throughput::Elements(total as u64));
        group.bench_with_input(
            BenchmarkId::new("parallel", &name),
            &(rows, cols),
            |b, &(r, c)| {
                let plan = Plan2D::new(r, c, Direction::Backward, Flags::ESTIMATE)
                    .expect("plan creation failed");
                b.iter(|| plan.execute(&input, &mut output));
            },
        );
    }

    group.finish();
}

fn bench_plan2d_inplace(c: &mut Criterion) {
    let sizes: &[(usize, usize)] = &[(256, 256), (512, 512), (1024, 1024)];

    let mut group = c.benchmark_group("plan2d_inplace");
    group.sample_size(10);

    for &(rows, cols) in sizes {
        let total = rows * cols;
        let input: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new((i as f64 * 0.007).sin(), (i as f64 * 0.013).cos()))
            .collect();

        let name = format!("{rows}x{cols}");
        group.throughput(Throughput::Elements(total as u64));
        group.bench_with_input(
            BenchmarkId::new("parallel_inplace", &name),
            &(rows, cols),
            |b, &(r, c)| {
                let plan = Plan2D::new(r, c, Direction::Forward, Flags::ESTIMATE)
                    .expect("plan creation failed");
                b.iter_batched(
                    || input.clone(),
                    |mut data| plan.execute_inplace(&mut data),
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

// ── Plan3D benchmarks ────────────────────────────────────────────────────────

fn bench_plan3d_forward(c: &mut Criterion) {
    let sizes: &[(usize, usize, usize)] = &[(64, 64, 64), (128, 128, 128)];

    let mut group = c.benchmark_group("plan3d_forward");
    group.sample_size(10);

    for &(d0, d1, d2) in sizes {
        let total = d0 * d1 * d2;
        let input: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new((i as f64 * 0.005).sin(), (i as f64 * 0.011).cos()))
            .collect();
        let mut output = vec![Complex::<f64>::zero(); total];

        let name = format!("{d0}x{d1}x{d2}");
        group.throughput(Throughput::Elements(total as u64));
        group.bench_with_input(
            BenchmarkId::new("parallel", &name),
            &(d0, d1, d2),
            |b, &(n0, n1, n2)| {
                let plan = Plan3D::new(n0, n1, n2, Direction::Forward, Flags::ESTIMATE)
                    .expect("plan creation failed");
                b.iter(|| plan.execute(&input, &mut output));
            },
        );
    }

    group.finish();
}

fn bench_plan3d_inplace(c: &mut Criterion) {
    let sizes: &[(usize, usize, usize)] = &[(64, 64, 64), (128, 128, 128)];

    let mut group = c.benchmark_group("plan3d_inplace");
    group.sample_size(10);

    for &(d0, d1, d2) in sizes {
        let total = d0 * d1 * d2;
        let input: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new((i as f64 * 0.005).sin(), (i as f64 * 0.011).cos()))
            .collect();

        let name = format!("{d0}x{d1}x{d2}");
        group.throughput(Throughput::Elements(total as u64));
        group.bench_with_input(
            BenchmarkId::new("parallel_inplace", &name),
            &(d0, d1, d2),
            |b, &(n0, n1, n2)| {
                let plan = Plan3D::new(n0, n1, n2, Direction::Forward, Flags::ESTIMATE)
                    .expect("plan creation failed");
                b.iter_batched(
                    || input.clone(),
                    |mut data| plan.execute_inplace(&mut data),
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

// ── Thread-pool sweep: 1, 2, 4, num_cpus.min(8) threads for 1024×1024 ──────

fn bench_plan2d_thread_sweep(c: &mut Criterion) {
    let n0 = 1024_usize;
    let n1 = 1024_usize;
    let total = n0 * n1;

    let input: Vec<Complex<f64>> = (0..total)
        .map(|i| Complex::new((i as f64 * 0.007).sin(), (i as f64 * 0.013).cos()))
        .collect();
    let mut output = vec![Complex::<f64>::zero(); total];

    let avail_cpus = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let thread_counts = [1_usize, 2, 4, avail_cpus.min(8)];

    let mut group = c.benchmark_group("plan2d_thread_sweep_1024x1024");
    group.sample_size(10);
    group.throughput(Throughput::Elements(total as u64));

    for &nthreads in &thread_counts {
        let pool = std::sync::Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(nthreads)
                .build()
                .expect("pool build"),
        );
        let plan = Plan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE)
            .expect("plan")
            .with_rayon_pool(pool);

        group.bench_with_input(BenchmarkId::new("threads", nthreads), &nthreads, |b, _| {
            b.iter(|| plan.execute(&input, &mut output));
        });
    }

    group.finish();
}

// ── criterion boilerplate ─────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_plan2d_forward,
    bench_plan2d_inverse,
    bench_plan2d_inplace,
    bench_plan3d_forward,
    bench_plan3d_inplace,
    bench_plan2d_thread_sweep,
);
criterion_main!(benches);
