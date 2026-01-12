//! Benchmarks for "Beyond FFTW" features.
//!
//! Benchmarks sparse FFT, pruned FFT, streaming FFT, and compile-time FFT.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::significant_drop_tightening)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxifft::Complex;

// ============================================================================
// Sparse FFT Benchmarks
// ============================================================================

#[cfg(feature = "sparse")]
fn benchmark_sparse_fft(c: &mut Criterion) {
    use oxifft::{sparse_fft, Direction, Flags, Plan};

    let mut group = c.benchmark_group("sparse_fft");

    // Compare sparse FFT vs regular FFT for sparse signals
    let sizes = [(1024, 10), (4096, 20), (16384, 50), (65536, 100)];

    for &(n, k) in &sizes {
        // Create sparse signal (k non-zero frequencies)
        let mut input: Vec<Complex<f64>> = vec![Complex::zero(); n];
        for i in 0..k {
            let freq = (i * n / k) % n;
            input[freq] = Complex::new(1.0, 0.0);
        }

        group.throughput(Throughput::Elements(n as u64));

        // Benchmark sparse FFT
        group.bench_with_input(
            BenchmarkId::new("sparse", format!("n={n}_k={k}")),
            &(n, k),
            |b, &(_n, k)| {
                b.iter(|| sparse_fft(&input, k));
            },
        );

        // Benchmark regular FFT for comparison
        group.bench_with_input(
            BenchmarkId::new("regular", format!("n={n}_k={k}")),
            &(n, k),
            |b, &(n, _k)| {
                let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE)
                    .expect("Failed to create plan");
                let mut output = vec![Complex::zero(); n];
                b.iter(|| {
                    plan.execute(&input, &mut output);
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Pruned FFT Benchmarks
// ============================================================================

#[cfg(feature = "pruned")]
fn benchmark_pruned_fft(c: &mut Criterion) {
    use oxifft::{fft_pruned_output, goertzel, Direction, Flags, Plan};

    let mut group = c.benchmark_group("pruned_fft");

    // Output-pruned FFT: compute only subset of frequencies
    let sizes = [(1024, 10), (4096, 20), (16384, 50)];

    for &(n, num_outputs) in &sizes {
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64 * 0.1).sin(), 0.0))
            .collect();

        let output_indices: Vec<usize> = (0..num_outputs).collect();

        group.throughput(Throughput::Elements(n as u64));

        // Benchmark output-pruned FFT
        group.bench_with_input(
            BenchmarkId::new("output_pruned", format!("n={n}_out={num_outputs}")),
            &n,
            |b, &_n| {
                b.iter(|| fft_pruned_output(&input, &output_indices));
            },
        );

        // Benchmark regular FFT for comparison
        group.bench_with_input(
            BenchmarkId::new("regular", format!("n={n}_out={num_outputs}")),
            &n,
            |b, &n| {
                let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE)
                    .expect("Failed to create plan");
                let mut output = vec![Complex::zero(); n];
                b.iter(|| {
                    plan.execute(&input, &mut output);
                });
            },
        );
    }

    group.finish();

    // Goertzel algorithm benchmark
    let mut goertzel_group = c.benchmark_group("goertzel");

    let sizes = [256, 1024, 4096, 16384];

    for &n in &sizes {
        // Goertzel expects Complex input
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((f64::from(i) * 0.1).sin(), 0.0))
            .collect();

        goertzel_group.throughput(Throughput::Elements(n as u64));

        // Single frequency Goertzel
        goertzel_group.bench_with_input(BenchmarkId::new("single_freq", n), &n, |b, &_n| {
            b.iter(|| {
                goertzel(&input, 10) // Frequency bin 10
            });
        });
    }

    goertzel_group.finish();
}

// ============================================================================
// Streaming FFT (STFT) Benchmarks
// ============================================================================

#[cfg(feature = "streaming")]
fn benchmark_streaming_fft(c: &mut Criterion) {
    use oxifft::{istft, stft, StreamingFft, WindowFunction};

    let mut group = c.benchmark_group("streaming_fft");

    // STFT benchmark
    let configs = [
        (4096, 256, 64),    // Short signal, small window
        (16384, 512, 128),  // Medium signal
        (65536, 1024, 256), // Long signal, large window
    ];

    for &(signal_len, fft_size, hop_size) in &configs {
        let signal: Vec<f64> = (0..signal_len)
            .map(|i| (f64::from(i) * 0.01).sin())
            .collect();

        group.throughput(Throughput::Elements(signal_len as u64));

        // STFT benchmark
        group.bench_with_input(
            BenchmarkId::new(
                "stft",
                format!("n={signal_len}_fft={fft_size}_hop={hop_size}"),
            ),
            &signal_len,
            |b, &_| {
                b.iter(|| stft(&signal, fft_size, hop_size, WindowFunction::Hann));
            },
        );

        // STFT + ISTFT roundtrip
        group.bench_with_input(
            BenchmarkId::new(
                "stft_istft",
                format!("n={signal_len}_fft={fft_size}_hop={hop_size}"),
            ),
            &signal_len,
            |b, &_| {
                b.iter(|| {
                    let spec = stft(&signal, fft_size, hop_size, WindowFunction::Hann);
                    istft(&spec, hop_size, WindowFunction::Hann)
                });
            },
        );
    }

    group.finish();

    // StreamingFft benchmark (real-time processing simulation)
    let mut streaming_group = c.benchmark_group("streaming_realtime");

    let fft_size = 512;
    let hop_size = 128;
    let chunk_size = 1024; // Simulated audio buffer size

    let chunks: Vec<Vec<f64>> = (0..10)
        .map(|c| {
            (0..chunk_size)
                .map(|i| (f64::from(c * chunk_size + i) * 0.01).sin())
                .collect()
        })
        .collect();

    streaming_group.bench_function("process_chunks", |b| {
        b.iter(|| {
            let mut processor: StreamingFft<f64> =
                StreamingFft::new(fft_size, hop_size, WindowFunction::Hann);
            for chunk in &chunks {
                processor.feed(chunk);
                while processor.pop_frame().is_some() {}
            }
        });
    });

    streaming_group.finish();
}

// ============================================================================
// Compile-time FFT Benchmarks
// ============================================================================

#[cfg(feature = "const-fft")]
fn benchmark_const_fft(c: &mut Criterion) {
    use oxifft::{fft_fixed, ifft_fixed, Direction, Flags, Plan};

    let mut group = c.benchmark_group("const_fft");

    // Compare const FFT vs dynamic FFT for small sizes
    macro_rules! bench_const_fft {
        ($($size:expr),*) => {
            $(
                {
                    let input: [Complex<f64>; $size] = core::array::from_fn(|i| {
                        Complex::new((i as f64 * 0.1).sin(), 0.0)
                    });

                    group.throughput(Throughput::Elements($size as u64));

                    // Const FFT
                    group.bench_function(
                        BenchmarkId::new("const", $size),
                        |b| {
                            b.iter(|| {
                                fft_fixed(&input)
                            });
                        },
                    );

                    // Dynamic FFT for comparison
                    let input_vec: Vec<Complex<f64>> = input.to_vec();
                    let plan = Plan::dft_1d($size, Direction::Forward, Flags::ESTIMATE)
                        .expect("Failed to create plan");
                    let mut output = vec![Complex::zero(); $size];

                    group.bench_function(
                        BenchmarkId::new("dynamic", $size),
                        |b| {
                            b.iter(|| {
                                plan.execute(&input_vec, &mut output);
                            });
                        },
                    );
                }
            )*
        };
    }

    bench_const_fft!(8, 16, 32, 64, 128, 256, 512, 1024);

    group.finish();

    // Roundtrip benchmark
    let mut roundtrip_group = c.benchmark_group("const_fft_roundtrip");

    let input_256: [Complex<f64>; 256] =
        core::array::from_fn(|i| Complex::new((i as f64 * 0.1).sin(), (i as f64 * 0.05).cos()));

    roundtrip_group.bench_function("const_256", |b| {
        b.iter(|| {
            let spectrum = fft_fixed(&input_256);
            ifft_fixed(&spectrum)
        });
    });

    roundtrip_group.finish();
}

// ============================================================================
// Window Function Benchmarks
// ============================================================================

#[cfg(feature = "streaming")]
fn benchmark_window_functions(c: &mut Criterion) {
    use oxifft::{blackman, hamming, hann, kaiser};

    let mut group = c.benchmark_group("window_functions");

    let sizes = [256, 512, 1024, 2048, 4096];

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("hann", size), &size, |b, &n| {
            b.iter(|| hann::<f64>(n));
        });

        group.bench_with_input(BenchmarkId::new("hamming", size), &size, |b, &n| {
            b.iter(|| hamming::<f64>(n));
        });

        group.bench_with_input(BenchmarkId::new("blackman", size), &size, |b, &n| {
            b.iter(|| blackman::<f64>(n));
        });

        group.bench_with_input(BenchmarkId::new("kaiser_beta5", size), &size, |b, &n| {
            b.iter(|| kaiser::<f64>(n, 5.0));
        });
    }

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

#[cfg(feature = "sparse")]
criterion_group!(sparse_benches, benchmark_sparse_fft);

#[cfg(feature = "pruned")]
criterion_group!(pruned_benches, benchmark_pruned_fft);

#[cfg(feature = "streaming")]
criterion_group!(
    streaming_benches,
    benchmark_streaming_fft,
    benchmark_window_functions
);

#[cfg(feature = "const-fft")]
criterion_group!(const_fft_benches, benchmark_const_fft);

// Main entry point - conditionally include groups based on features
#[cfg(all(
    feature = "sparse",
    feature = "pruned",
    feature = "streaming",
    feature = "const-fft"
))]
criterion_main!(
    sparse_benches,
    pruned_benches,
    streaming_benches,
    const_fft_benches
);

#[cfg(all(
    feature = "sparse",
    feature = "pruned",
    feature = "streaming",
    not(feature = "const-fft")
))]
criterion_main!(sparse_benches, pruned_benches, streaming_benches);

#[cfg(all(
    feature = "sparse",
    feature = "pruned",
    not(feature = "streaming"),
    feature = "const-fft"
))]
criterion_main!(sparse_benches, pruned_benches, const_fft_benches);

#[cfg(all(
    feature = "sparse",
    not(feature = "pruned"),
    feature = "streaming",
    feature = "const-fft"
))]
criterion_main!(sparse_benches, streaming_benches, const_fft_benches);

#[cfg(all(
    not(feature = "sparse"),
    feature = "pruned",
    feature = "streaming",
    feature = "const-fft"
))]
criterion_main!(pruned_benches, streaming_benches, const_fft_benches);

// Fallback for when no features are enabled
#[cfg(not(any(
    feature = "sparse",
    feature = "pruned",
    feature = "streaming",
    feature = "const-fft"
)))]
fn main() {
    println!(
        "No benchmark features enabled. Enable sparse, pruned, streaming, or const-fft features."
    );
}
