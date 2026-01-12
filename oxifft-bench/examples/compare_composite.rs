#![allow(clippy::cast_precision_loss)]
#![allow(clippy::uninlined_format_args)]

use oxifft::{Complex, Direction, Flags, Plan};
use rustfft::FftPlanner;
use std::time::Instant;

fn bench_oxifft(n: usize, iterations: usize) -> f64 {
    let input: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new((i as f64).sin(), (i as f64 * 0.1).cos()))
        .collect();
    let mut output: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];

    let plan: Plan<f64> =
        Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).expect("Failed to create OxiFFT plan");

    // Warmup
    for _ in 0..100 {
        plan.execute(&input, &mut output);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        plan.execute(&input, &mut output);
    }
    start.elapsed().as_nanos() as f64 / iterations as f64
}

fn bench_rustfft(n: usize, iterations: usize) -> f64 {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer: Vec<rustfft::num_complex::Complex<f64>> = (0..n)
        .map(|i| rustfft::num_complex::Complex::new((i as f64).sin(), (i as f64 * 0.1).cos()))
        .collect();

    // Warmup
    for _ in 0..100 {
        fft.process(&mut buffer);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        fft.process(&mut buffer);
    }
    start.elapsed().as_nanos() as f64 / iterations as f64
}

fn main() {
    println!("OxiFFT vs RustFFT - Composite Sizes");
    println!("====================================");
    println!(
        "{:>6} {:>12} {:>12} {:>10}",
        "Size", "OxiFFT (ns)", "RustFFT (ns)", "Ratio"
    );
    println!("{:-<6} {:-<12} {:-<12} {:-<10}", "", "", "", "");

    let sizes = [12, 15, 18, 20, 24, 30, 36, 45, 48, 50, 60, 72, 80, 96, 100];
    let iterations = 50_000;

    for &n in &sizes {
        let oxifft_ns = bench_oxifft(n, iterations);
        let rustfft_ns = bench_rustfft(n, iterations);
        let ratio = oxifft_ns / rustfft_ns;

        let ratio_str = if ratio < 1.0 {
            format!("{:.2}x faster", 1.0 / ratio)
        } else {
            format!("{:.2}x slower", ratio)
        };

        println!(
            "{:>6} {:>12.1} {:>12.1} {:>10}",
            n, oxifft_ns, rustfft_ns, ratio_str
        );
    }
}
