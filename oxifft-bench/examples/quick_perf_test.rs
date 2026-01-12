#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::uninlined_format_args)]

use oxifft::api::{Direction, Flags, Plan};
use oxifft::Complex;
use std::time::Instant;

fn main() {
    let size = 64;
    let input: Vec<Complex<f64>> = (0..size)
        .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
        .collect();

    println!("Testing FFT performance for size {}", size);
    println!("{}", "=".repeat(60));

    // Test 1: Creating plan every time (old way - slow)
    let iterations = 10000;
    let start = Instant::now();
    for _ in 0..iterations {
        let mut output = vec![Complex::zero(); size];
        if let Some(plan) = Plan::dft_1d(size, Direction::Forward, Flags::ESTIMATE) {
            plan.execute(&input, &mut output);
        }
    }
    let duration_with_planning = start.elapsed();

    // Test 2: Reusing plan (new way - fast)
    let plan =
        Plan::dft_1d(size, Direction::Forward, Flags::ESTIMATE).expect("Failed to create plan");
    let start = Instant::now();
    for _ in 0..iterations {
        let mut output = vec![Complex::zero(); size];
        plan.execute(&input, &mut output);
    }
    let duration_without_planning = start.elapsed();

    let avg_with = duration_with_planning.as_nanos() as f64 / iterations as f64;
    let avg_without = duration_without_planning.as_nanos() as f64 / iterations as f64;

    println!("\nResults ({} iterations):", iterations);
    println!(
        "  With planning overhead:    {:.2} µs/iter",
        avg_with / 1000.0
    );
    println!(
        "  Without planning overhead: {:.2} µs/iter",
        avg_without / 1000.0
    );
    println!("\n  Speedup: {:.2}x", avg_with / avg_without);
    println!("\nThis demonstrates the importance of creating plans once");
    println!("rather than on every FFT execution.");
}
