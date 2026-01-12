//! FFT performance comparison benchmarks.
//!
//! Compares `OxiFFT` against rustfft (and optionally FFTW with `fftw-compare` feature).

#![allow(clippy::cast_precision_loss)] // FFT size computations use float for math
#![allow(clippy::suboptimal_flops)] // Benchmarks prioritize clarity over micro-optimization
#![allow(clippy::assign_op_pattern)] // Clearer in benchmark context

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxifft::api::{fft2d, fft_batch, rfft, Direction, Flags, Plan};
use oxifft::Complex;
use rustfft::FftPlanner;
use std::hint::black_box;

fn generate_input(n: usize) -> Vec<Complex<f64>> {
    (0..n)
        .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
        .collect()
}

const fn to_num_complex(c: Complex<f64>) -> num_complex::Complex<f64> {
    num_complex::Complex::new(c.re, c.im)
}

fn generate_real_input(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64).sin()).collect()
}

// =============================================================================
// Power-of-2 benchmarks
// =============================================================================

fn bench_power_of_2(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![16, 64, 256, 1024, 4096, 16384, 65536];

    let mut group = c.benchmark_group("fft_power_of_2");

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        let input = generate_input(size);

        // OxiFFT benchmark - create plan once like RustFFT/FFTW do
        let oxifft_plan = Plan::dft_1d(size, Direction::Forward, Flags::ESTIMATE)
            .expect("Failed to create OxiFFT plan");
        let mut oxifft_output = vec![Complex::zero(); size];

        group.bench_with_input(BenchmarkId::new("oxifft", size), &input, |b, input| {
            b.iter(|| {
                oxifft_plan.execute(black_box(input), black_box(&mut oxifft_output));
            });
        });

        // rustfft benchmark
        let mut planner = FftPlanner::new();
        let rustfft_plan = planner.plan_fft_forward(size);
        let rustfft_input: Vec<num_complex::Complex<f64>> =
            input.iter().map(|c| to_num_complex(*c)).collect();

        group.bench_with_input(
            BenchmarkId::new("rustfft", size),
            &rustfft_input,
            |b, input| {
                b.iter(|| {
                    let mut buffer = input.clone();
                    rustfft_plan.process(black_box(&mut buffer));
                    black_box(buffer)
                });
            },
        );

        // FFTW benchmark (if feature enabled)
        #[cfg(feature = "fftw-compare")]
        {
            use fftw::array::AlignedVec;
            use fftw::plan::{C2CPlan, C2CPlan64};
            use fftw::types::{c64, Sign};

            let mut fftw_input: AlignedVec<c64> = AlignedVec::new(size);
            let mut fftw_output: AlignedVec<c64> = AlignedVec::new(size);
            for (i, c) in input.iter().enumerate() {
                fftw_input[i] = c64::new(c.re, c.im);
            }
            let mut fftw_plan: C2CPlan64 =
                C2CPlan::aligned(&[size], Sign::Forward, fftw::types::Flag::MEASURE).unwrap();

            group.bench_function(BenchmarkId::new("fftw", size), |b| {
                b.iter(|| {
                    fftw_plan
                        .c2c(black_box(&mut fftw_input), black_box(&mut fftw_output))
                        .unwrap();
                });
            });
        }
    }

    group.finish();
}

// =============================================================================
// Prime size benchmarks
// =============================================================================

fn bench_prime_sizes(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![17, 97, 257, 1009, 4093];

    let mut group = c.benchmark_group("fft_prime");

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        let input = generate_input(size);

        // OxiFFT benchmark - create plan once like RustFFT/FFTW do
        let oxifft_plan = Plan::dft_1d(size, Direction::Forward, Flags::ESTIMATE)
            .expect("Failed to create OxiFFT plan");
        let mut oxifft_output = vec![Complex::zero(); size];

        group.bench_with_input(BenchmarkId::new("oxifft", size), &input, |b, input| {
            b.iter(|| {
                oxifft_plan.execute(black_box(input), black_box(&mut oxifft_output));
            });
        });

        // rustfft benchmark
        let mut planner = FftPlanner::new();
        let rustfft_plan = planner.plan_fft_forward(size);
        let rustfft_input: Vec<num_complex::Complex<f64>> =
            input.iter().map(|c| to_num_complex(*c)).collect();

        group.bench_with_input(
            BenchmarkId::new("rustfft", size),
            &rustfft_input,
            |b, input| {
                b.iter(|| {
                    let mut buffer = input.clone();
                    rustfft_plan.process(black_box(&mut buffer));
                    black_box(buffer)
                });
            },
        );

        // FFTW benchmark (if feature enabled)
        #[cfg(feature = "fftw-compare")]
        {
            use fftw::array::AlignedVec;
            use fftw::plan::{C2CPlan, C2CPlan64};
            use fftw::types::{c64, Sign};

            let mut fftw_input: AlignedVec<c64> = AlignedVec::new(size);
            let mut fftw_output: AlignedVec<c64> = AlignedVec::new(size);
            for (i, c) in input.iter().enumerate() {
                fftw_input[i] = c64::new(c.re, c.im);
            }
            let mut fftw_plan: C2CPlan64 =
                C2CPlan::aligned(&[size], Sign::Forward, fftw::types::Flag::MEASURE).unwrap();

            group.bench_function(BenchmarkId::new("fftw", size), |b| {
                b.iter(|| {
                    fftw_plan
                        .c2c(black_box(&mut fftw_input), black_box(&mut fftw_output))
                        .unwrap();
                });
            });
        }
    }

    group.finish();
}

// =============================================================================
// Composite size benchmarks
// =============================================================================

fn bench_composite_sizes(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![12, 100, 360, 1000, 4000];

    let mut group = c.benchmark_group("fft_composite");

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        let input = generate_input(size);

        // OxiFFT benchmark - create plan once like RustFFT/FFTW do
        let oxifft_plan = Plan::dft_1d(size, Direction::Forward, Flags::ESTIMATE)
            .expect("Failed to create OxiFFT plan");
        let mut oxifft_output = vec![Complex::zero(); size];

        group.bench_with_input(BenchmarkId::new("oxifft", size), &input, |b, input| {
            b.iter(|| {
                oxifft_plan.execute(black_box(input), black_box(&mut oxifft_output));
            });
        });

        // rustfft benchmark
        let mut planner = FftPlanner::new();
        let rustfft_plan = planner.plan_fft_forward(size);
        let rustfft_input: Vec<num_complex::Complex<f64>> =
            input.iter().map(|c| to_num_complex(*c)).collect();

        group.bench_with_input(
            BenchmarkId::new("rustfft", size),
            &rustfft_input,
            |b, input| {
                b.iter(|| {
                    let mut buffer = input.clone();
                    rustfft_plan.process(black_box(&mut buffer));
                    black_box(buffer)
                });
            },
        );

        // FFTW benchmark (if feature enabled)
        #[cfg(feature = "fftw-compare")]
        {
            use fftw::array::AlignedVec;
            use fftw::plan::{C2CPlan, C2CPlan64};
            use fftw::types::{c64, Sign};

            let mut fftw_input: AlignedVec<c64> = AlignedVec::new(size);
            let mut fftw_output: AlignedVec<c64> = AlignedVec::new(size);
            for (i, c) in input.iter().enumerate() {
                fftw_input[i] = c64::new(c.re, c.im);
            }
            let mut fftw_plan: C2CPlan64 =
                C2CPlan::aligned(&[size], Sign::Forward, fftw::types::Flag::MEASURE).unwrap();

            group.bench_function(BenchmarkId::new("fftw", size), |b| {
                b.iter(|| {
                    fftw_plan
                        .c2c(black_box(&mut fftw_input), black_box(&mut fftw_output))
                        .unwrap();
                });
            });
        }
    }

    group.finish();
}

// =============================================================================
// Real FFT benchmarks
// =============================================================================

fn bench_real_fft(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![64, 256, 1024, 4096, 16384];

    let mut group = c.benchmark_group("rfft");

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        let input = generate_real_input(size);

        // OxiFFT benchmark
        group.bench_with_input(BenchmarkId::new("oxifft", size), &input, |b, input| {
            b.iter(|| {
                let result = rfft(black_box(input));
                black_box(result)
            });
        });

        // rustfft benchmark (using FftPlanner with real input padded to complex)
        let mut planner = FftPlanner::new();
        let rustfft_plan = planner.plan_fft_forward(size);
        let rustfft_input: Vec<num_complex::Complex<f64>> = input
            .iter()
            .map(|&r| num_complex::Complex::new(r, 0.0))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("rustfft", size),
            &rustfft_input,
            |b, input| {
                b.iter(|| {
                    let mut buffer = input.clone();
                    rustfft_plan.process(black_box(&mut buffer));
                    black_box(buffer)
                });
            },
        );

        // FFTW benchmark (if feature enabled)
        #[cfg(feature = "fftw-compare")]
        {
            use fftw::array::AlignedVec;
            use fftw::plan::{R2CPlan, R2CPlan64};
            use fftw::types::c64;

            let mut fftw_input: AlignedVec<f64> = AlignedVec::new(size);
            let mut fftw_output: AlignedVec<c64> = AlignedVec::new(size / 2 + 1);
            for (i, &r) in input.iter().enumerate() {
                fftw_input[i] = r;
            }
            let mut fftw_plan: R2CPlan64 =
                R2CPlan::aligned(&[size], fftw::types::Flag::MEASURE).unwrap();

            group.bench_function(BenchmarkId::new("fftw", size), |b| {
                b.iter(|| {
                    fftw_plan
                        .r2c(black_box(&mut fftw_input), black_box(&mut fftw_output))
                        .unwrap();
                });
            });
        }
    }

    group.finish();
}

// =============================================================================
// 2D FFT benchmarks
// =============================================================================

fn bench_2d_fft(c: &mut Criterion) {
    let sizes: Vec<(usize, usize)> = vec![(16, 16), (32, 32), (64, 64), (128, 128), (256, 256)];

    let mut group = c.benchmark_group("fft_2d");

    for (rows, cols) in sizes {
        let total = rows * cols;
        group.throughput(Throughput::Elements(total as u64));

        let input = generate_input(total);

        // OxiFFT benchmark
        let label = format!("{rows}x{cols}");
        group.bench_with_input(BenchmarkId::new("oxifft", &label), &input, |b, input| {
            b.iter(|| {
                let result = fft2d(black_box(input), rows, cols);
                black_box(result)
            });
        });

        // rustfft benchmark (row-column approach)
        let mut planner = FftPlanner::new();
        let row_plan = planner.plan_fft_forward(cols);
        let col_plan = planner.plan_fft_forward(rows);
        let rustfft_input: Vec<num_complex::Complex<f64>> =
            input.iter().map(|c| to_num_complex(*c)).collect();

        group.bench_with_input(
            BenchmarkId::new("rustfft", &label),
            &rustfft_input,
            |b, input| {
                b.iter(|| {
                    let mut buffer = input.clone();
                    // Row FFTs
                    for row in 0..rows {
                        let start = row * cols;
                        row_plan.process(black_box(&mut buffer[start..start + cols]));
                    }
                    // Transpose and column FFTs (simplified - just do column FFTs with striding)
                    let mut col_buffer = vec![num_complex::Complex::new(0.0, 0.0); rows];
                    for col in 0..cols {
                        for row in 0..rows {
                            col_buffer[row] = buffer[row * cols + col];
                        }
                        col_plan.process(black_box(&mut col_buffer));
                        for row in 0..rows {
                            buffer[row * cols + col] = col_buffer[row];
                        }
                    }
                    black_box(buffer)
                });
            },
        );

        // FFTW benchmark (if feature enabled)
        #[cfg(feature = "fftw-compare")]
        {
            use fftw::array::AlignedVec;
            use fftw::plan::{C2CPlan, C2CPlan64};
            use fftw::types::{c64, Sign};

            let mut fftw_input: AlignedVec<c64> = AlignedVec::new(total);
            let mut fftw_output: AlignedVec<c64> = AlignedVec::new(total);
            for (i, c) in input.iter().enumerate() {
                fftw_input[i] = c64::new(c.re, c.im);
            }
            let mut fftw_plan: C2CPlan64 =
                C2CPlan::aligned(&[rows, cols], Sign::Forward, fftw::types::Flag::MEASURE).unwrap();

            group.bench_function(BenchmarkId::new("fftw", &label), |b| {
                b.iter(|| {
                    fftw_plan
                        .c2c(black_box(&mut fftw_input), black_box(&mut fftw_output))
                        .unwrap();
                });
            });
        }
    }

    group.finish();
}

// =============================================================================
// Batch FFT benchmarks
// =============================================================================

fn bench_batch_fft(c: &mut Criterion) {
    let configs: Vec<(usize, usize)> = vec![
        (64, 100),  // 100 FFTs of size 64
        (256, 50),  // 50 FFTs of size 256
        (1024, 20), // 20 FFTs of size 1024
        (4096, 10), // 10 FFTs of size 4096
    ];

    let mut group = c.benchmark_group("fft_batch");

    for (size, howmany) in configs {
        let total = size * howmany;
        group.throughput(Throughput::Elements(total as u64));

        let input = generate_input(total);

        // OxiFFT benchmark
        let label = format!("{size}x{howmany}");
        group.bench_with_input(BenchmarkId::new("oxifft", &label), &input, |b, input| {
            b.iter(|| {
                let result = fft_batch(black_box(input), size, howmany);
                black_box(result)
            });
        });

        // rustfft benchmark (sequential FFTs)
        let mut planner = FftPlanner::new();
        let rustfft_plan = planner.plan_fft_forward(size);
        let rustfft_input: Vec<num_complex::Complex<f64>> =
            input.iter().map(|c| to_num_complex(*c)).collect();

        group.bench_with_input(
            BenchmarkId::new("rustfft", &label),
            &rustfft_input,
            |b, input| {
                b.iter(|| {
                    let mut buffer = input.clone();
                    for i in 0..howmany {
                        let start = i * size;
                        rustfft_plan.process(black_box(&mut buffer[start..start + size]));
                    }
                    black_box(buffer)
                });
            },
        );

        // FFTW benchmark (if feature enabled)
        #[cfg(feature = "fftw-compare")]
        {
            use fftw::array::AlignedVec;
            use fftw::plan::{C2CPlan, C2CPlan64};
            use fftw::types::{c64, Sign};

            let mut fftw_input: AlignedVec<c64> = AlignedVec::new(total);
            let mut fftw_output: AlignedVec<c64> = AlignedVec::new(total);
            for (i, c) in input.iter().enumerate() {
                fftw_input[i] = c64::new(c.re, c.im);
            }
            // FFTW supports batched transforms natively
            let mut fftw_plan: C2CPlan64 =
                C2CPlan::aligned(&[size], Sign::Forward, fftw::types::Flag::MEASURE).unwrap();

            group.bench_function(BenchmarkId::new("fftw", &label), |b| {
                b.iter(|| {
                    for i in 0..howmany {
                        let start = i * size;
                        let end = start + size;
                        fftw_plan
                            .c2c(
                                black_box(&mut fftw_input[start..end]),
                                black_box(&mut fftw_output[start..end]),
                            )
                            .unwrap();
                    }
                });
            });
        }
    }

    group.finish();
}

// =============================================================================
// SAR Processing Benchmarks
// =============================================================================

/// SAR Range Compression: Large 1D FFTs typical of range line processing
/// Common range sample counts: 4096, 8192, 16384, 32768
fn bench_sar_range_compression(c: &mut Criterion) {
    // Typical SAR range line sizes (power-of-2 for efficient processing)
    let sizes: Vec<usize> = vec![4096, 8192, 16384, 32768];

    let mut group = c.benchmark_group("sar_range_compression");
    group.sample_size(50); // Reduce samples for large sizes

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        let input = generate_input(size);

        // OxiFFT benchmark
        let oxifft_plan = Plan::dft_1d(size, Direction::Forward, Flags::ESTIMATE)
            .expect("Failed to create OxiFFT plan");
        let mut oxifft_output = vec![Complex::zero(); size];

        group.bench_with_input(BenchmarkId::new("oxifft", size), &input, |b, input| {
            b.iter(|| {
                oxifft_plan.execute(black_box(input), black_box(&mut oxifft_output));
            });
        });

        // rustfft benchmark
        let mut planner = FftPlanner::new();
        let rustfft_plan = planner.plan_fft_forward(size);
        let rustfft_input: Vec<num_complex::Complex<f64>> =
            input.iter().map(|c| to_num_complex(*c)).collect();

        group.bench_with_input(
            BenchmarkId::new("rustfft", size),
            &rustfft_input,
            |b, input| {
                b.iter(|| {
                    let mut buffer = input.clone();
                    rustfft_plan.process(black_box(&mut buffer));
                    black_box(buffer)
                });
            },
        );
    }

    group.finish();
}

/// SAR Azimuth Processing: Batch FFTs simulating azimuth compression
/// Processes many range lines (azimuth samples) in batch
fn bench_sar_azimuth_batch(c: &mut Criterion) {
    // Typical SAR azimuth processing configurations
    // (azimuth_fft_size, num_range_bins)
    let configs: Vec<(usize, usize)> = vec![
        (2048, 512),   // Small scene: 2048 azimuth samples, 512 range bins
        (4096, 1024),  // Medium scene: 4096 azimuth, 1024 range bins
        (8192, 2048),  // Large scene: 8192 azimuth, 2048 range bins
        (16384, 1024), // High resolution: 16384 azimuth, 1024 range bins
    ];

    let mut group = c.benchmark_group("sar_azimuth_batch");
    group.sample_size(20); // Reduce samples for very large workloads

    for (azimuth_size, num_range_bins) in configs {
        let total = azimuth_size * num_range_bins;
        group.throughput(Throughput::Elements(total as u64));

        let input = generate_input(total);

        let label = format!("{azimuth_size}x{num_range_bins}");

        // OxiFFT benchmark using batch API
        group.bench_with_input(BenchmarkId::new("oxifft", &label), &input, |b, input| {
            b.iter(|| {
                let result = fft_batch(black_box(input), azimuth_size, num_range_bins);
                black_box(result)
            });
        });

        // rustfft benchmark
        let mut planner = FftPlanner::new();
        let rustfft_plan = planner.plan_fft_forward(azimuth_size);
        let rustfft_input: Vec<num_complex::Complex<f64>> =
            input.iter().map(|c| to_num_complex(*c)).collect();

        group.bench_with_input(
            BenchmarkId::new("rustfft", &label),
            &rustfft_input,
            |b, input| {
                b.iter(|| {
                    let mut buffer = input.clone();
                    for i in 0..num_range_bins {
                        let start = i * azimuth_size;
                        rustfft_plan.process(black_box(&mut buffer[start..start + azimuth_size]));
                    }
                    black_box(buffer)
                });
            },
        );
    }

    group.finish();
}

/// SAR 2D Image Formation: Large 2D FFTs for algorithms like Range-Doppler, ω-k
fn bench_sar_2d_image_formation(c: &mut Criterion) {
    // Typical SAR image sizes (square for simplicity, real SAR may differ)
    let sizes: Vec<(usize, usize)> = vec![
        (512, 512),   // Quick test / small scene
        (1024, 1024), // Medium resolution
        (2048, 2048), // High resolution
        (4096, 4096), // Very high resolution
    ];

    let mut group = c.benchmark_group("sar_2d_image");
    group.sample_size(10); // Very large workloads

    for (rows, cols) in sizes {
        let total = rows * cols;
        group.throughput(Throughput::Elements(total as u64));

        let input = generate_input(total);

        let label = format!("{rows}x{cols}");

        // OxiFFT benchmark
        group.bench_with_input(BenchmarkId::new("oxifft", &label), &input, |b, input| {
            b.iter(|| {
                let result = fft2d(black_box(input), rows, cols);
                black_box(result)
            });
        });

        // rustfft benchmark (row-column decomposition)
        let mut planner = FftPlanner::new();
        let row_plan = planner.plan_fft_forward(cols);
        let col_plan = planner.plan_fft_forward(rows);
        let rustfft_input: Vec<num_complex::Complex<f64>> =
            input.iter().map(|c| to_num_complex(*c)).collect();

        group.bench_with_input(
            BenchmarkId::new("rustfft", &label),
            &rustfft_input,
            |b, input| {
                b.iter(|| {
                    let mut buffer = input.clone();
                    // Row FFTs
                    for row in 0..rows {
                        let start = row * cols;
                        row_plan.process(black_box(&mut buffer[start..start + cols]));
                    }
                    // Column FFTs with gather/scatter
                    let mut col_buffer = vec![num_complex::Complex::new(0.0, 0.0); rows];
                    for col in 0..cols {
                        for row in 0..rows {
                            col_buffer[row] = buffer[row * cols + col];
                        }
                        col_plan.process(black_box(&mut col_buffer));
                        for row in 0..rows {
                            buffer[row * cols + col] = col_buffer[row];
                        }
                    }
                    black_box(buffer)
                });
            },
        );
    }

    group.finish();
}

/// SAR Chirp Convolution Pipeline: Forward FFT → Multiply → Inverse FFT
/// This is the core of range/azimuth compression
fn bench_sar_chirp_convolution(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![4096, 8192, 16384];

    let mut group = c.benchmark_group("sar_chirp_conv");
    group.sample_size(50);

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        let input = generate_input(size);
        // Simulated matched filter (chirp reference in frequency domain)
        let chirp_ref: Vec<Complex<f64>> = (0..size)
            .map(|i| {
                let phase = std::f64::consts::PI * (i as f64) / (size as f64);
                Complex::new(phase.cos(), phase.sin())
            })
            .collect();

        // OxiFFT benchmark
        let fwd_plan = Plan::dft_1d(size, Direction::Forward, Flags::ESTIMATE)
            .expect("Failed to create forward plan");
        let inv_plan = Plan::dft_1d(size, Direction::Backward, Flags::ESTIMATE)
            .expect("Failed to create inverse plan");
        let mut fft_out = vec![Complex::zero(); size];
        let mut result = vec![Complex::zero(); size];

        group.bench_with_input(BenchmarkId::new("oxifft", size), &input, |b, input| {
            b.iter(|| {
                // Forward FFT
                fwd_plan.execute(black_box(input), black_box(&mut fft_out));
                // Multiply with chirp reference
                for i in 0..size {
                    fft_out[i] = Complex::new(
                        fft_out[i].re * chirp_ref[i].re - fft_out[i].im * chirp_ref[i].im,
                        fft_out[i].re * chirp_ref[i].im + fft_out[i].im * chirp_ref[i].re,
                    );
                }
                // Inverse FFT
                inv_plan.execute(black_box(&fft_out), black_box(&mut result));
            });
        });

        // rustfft benchmark
        let mut planner = FftPlanner::new();
        let rustfft_fwd = planner.plan_fft_forward(size);
        let rustfft_inv = planner.plan_fft_inverse(size);
        let rustfft_input: Vec<num_complex::Complex<f64>> =
            input.iter().map(|c| to_num_complex(*c)).collect();
        let rustfft_chirp: Vec<num_complex::Complex<f64>> =
            chirp_ref.iter().map(|c| to_num_complex(*c)).collect();

        group.bench_with_input(
            BenchmarkId::new("rustfft", size),
            &rustfft_input,
            |b, input| {
                b.iter(|| {
                    let mut buffer = input.clone();
                    // Forward FFT
                    rustfft_fwd.process(black_box(&mut buffer));
                    // Multiply with chirp reference
                    for i in 0..size {
                        buffer[i] = buffer[i] * rustfft_chirp[i];
                    }
                    // Inverse FFT
                    rustfft_inv.process(black_box(&mut buffer));
                    black_box(buffer)
                });
            },
        );
    }

    group.finish();
}

/// SAR Forward+Inverse Roundtrip: Measures complete FFT cycle performance
fn bench_sar_roundtrip(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![4096, 8192, 16384, 32768];

    let mut group = c.benchmark_group("sar_roundtrip");
    group.sample_size(50);

    for size in sizes {
        group.throughput(Throughput::Elements((size * 2) as u64)); // Count both FFT operations

        let input = generate_input(size);

        // OxiFFT benchmark
        let fwd_plan = Plan::dft_1d(size, Direction::Forward, Flags::ESTIMATE)
            .expect("Failed to create forward plan");
        let inv_plan = Plan::dft_1d(size, Direction::Backward, Flags::ESTIMATE)
            .expect("Failed to create inverse plan");
        let mut fft_out = vec![Complex::zero(); size];
        let mut result = vec![Complex::zero(); size];

        group.bench_with_input(BenchmarkId::new("oxifft", size), &input, |b, input| {
            b.iter(|| {
                fwd_plan.execute(black_box(input), black_box(&mut fft_out));
                inv_plan.execute(black_box(&fft_out), black_box(&mut result));
            });
        });

        // rustfft benchmark
        let mut planner = FftPlanner::new();
        let rustfft_fwd = planner.plan_fft_forward(size);
        let rustfft_inv = planner.plan_fft_inverse(size);
        let rustfft_input: Vec<num_complex::Complex<f64>> =
            input.iter().map(|c| to_num_complex(*c)).collect();

        group.bench_with_input(
            BenchmarkId::new("rustfft", size),
            &rustfft_input,
            |b, input| {
                b.iter(|| {
                    let mut buffer = input.clone();
                    rustfft_fwd.process(black_box(&mut buffer));
                    rustfft_inv.process(black_box(&mut buffer));
                    black_box(buffer)
                });
            },
        );
    }

    group.finish();
}

/// SAR Real-to-Complex FFT: Raw SAR data is often real-valued
fn bench_sar_real_fft(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![4096, 8192, 16384, 32768];

    let mut group = c.benchmark_group("sar_real_fft");
    group.sample_size(50);

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        let input = generate_real_input(size);

        // OxiFFT benchmark
        group.bench_with_input(BenchmarkId::new("oxifft", size), &input, |b, input| {
            b.iter(|| {
                let result = rfft(black_box(input));
                black_box(result)
            });
        });

        // rustfft benchmark (real input padded to complex)
        let mut planner = FftPlanner::new();
        let rustfft_plan = planner.plan_fft_forward(size);
        let rustfft_input: Vec<num_complex::Complex<f64>> = input
            .iter()
            .map(|&r| num_complex::Complex::new(r, 0.0))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("rustfft", size),
            &rustfft_input,
            |b, input| {
                b.iter(|| {
                    let mut buffer = input.clone();
                    rustfft_plan.process(black_box(&mut buffer));
                    black_box(buffer)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_power_of_2,
    bench_prime_sizes,
    bench_composite_sizes,
    bench_real_fft,
    bench_2d_fft,
    bench_batch_fft
);

// SAR-specific benchmark group
criterion_group!(
    sar_benches,
    bench_sar_range_compression,
    bench_sar_azimuth_batch,
    bench_sar_2d_image_formation,
    bench_sar_chirp_convolution,
    bench_sar_roundtrip,
    bench_sar_real_fft
);

criterion_main!(benches, sar_benches);
