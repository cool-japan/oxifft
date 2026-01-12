#![allow(clippy::cast_precision_loss)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::single_match_else)]

use oxifft::{Complex, Direction, Flags, Plan};
use std::time::Instant;

fn bench_size(n: usize, iterations: usize) {
    let input: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new((i as f64).sin(), (i as f64 * 0.1).cos()))
        .collect();
    let mut output: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];

    // Create plan once
    let plan: Plan<f64> = match Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => {
            eprintln!("Failed to create plan for size {}", n);
            return;
        }
    };

    // Warmup
    for _ in 0..100 {
        plan.execute(&input, &mut output);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        plan.execute(&input, &mut output);
    }
    let elapsed = start.elapsed();

    let avg_ns = elapsed.as_nanos() as f64 / iterations as f64;
    println!("Size {:4}: {:8.1} ns/transform", n, avg_ns);
}

fn main() {
    println!("Composite codelet benchmarks:");
    println!("==============================");

    // Sizes with composite codelets
    let sizes = [12, 15, 18, 20, 24, 30, 36, 45, 48, 50, 60, 72, 80, 96, 100];
    let iterations = 100_000;

    for &n in &sizes {
        bench_size(n, iterations);
    }
}
