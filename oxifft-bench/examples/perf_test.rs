#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::uninlined_format_args)]

use oxifft::api::{Direction, Flags, Plan};
use oxifft::Complex;
use std::time::Instant;

fn main() {
    let sizes = [16, 64, 256, 1024, 4096];
    let iterations = 10000;

    println!("OxiFFT Performance Test (NEON SIMD optimized)");
    println!("=============================================\n");

    for &size in &sizes {
        let input: Vec<Complex<f64>> = (0..size)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        let plan =
            Plan::dft_1d(size, Direction::Forward, Flags::ESTIMATE).expect("Failed to create plan");

        let mut output = vec![Complex::zero(); size];

        // Warm up
        for _ in 0..100 {
            plan.execute(&input, &mut output);
        }

        // Measure
        let start = Instant::now();
        for _ in 0..iterations {
            plan.execute(&input, &mut output);
        }
        let duration = start.elapsed();

        let avg_ns = duration.as_nanos() as f64 / iterations as f64;
        let throughput = (size as f64) / (avg_ns / 1e9) / 1e6;

        println!(
            "Size {:5}: {:8.1} ns/iter | {:7.1} Melem/s",
            size, avg_ns, throughput
        );
    }
}
