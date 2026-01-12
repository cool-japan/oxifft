#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::uninlined_format_args)]

use oxifft::api::{Direction, Flags, Plan};
use oxifft::Complex;
use std::time::Instant;

fn main() {
    let sizes = [1024, 4096, 16384, 65536];
    let iterations = 1000;

    println!("FFT Profile Analysis");
    println!("====================\n");

    for &size in &sizes {
        let input: Vec<Complex<f64>> = (0..size)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        // Create plan (measures planning time)
        let plan_start = Instant::now();
        for _ in 0..100 {
            let _ = Plan::<f64>::dft_1d(size, Direction::Forward, Flags::ESTIMATE);
        }
        let plan_ns = plan_start.elapsed().as_nanos() as f64 / 100.0;

        let plan = Plan::<f64>::dft_1d(size, Direction::Forward, Flags::ESTIMATE)
            .expect("Failed to create plan");
        let mut output = vec![Complex::zero(); size];

        // Warm up
        for _ in 0..100 {
            plan.execute(&input, &mut output);
        }

        // Measure execution
        let exec_start = Instant::now();
        for _ in 0..iterations {
            plan.execute(&input, &mut output);
        }
        let exec_ns = exec_start.elapsed().as_nanos() as f64 / iterations as f64;

        // Compare with RustFFT
        let mut planner = rustfft::FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(size);
        let rust_data: Vec<rustfft::num_complex::Complex<f64>> = (0..size)
            .map(|i| rustfft::num_complex::Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        // Warm up
        for _ in 0..100 {
            let mut data = rust_data.clone();
            fft.process(&mut data);
        }

        let rust_start = Instant::now();
        for _ in 0..iterations {
            let mut data = rust_data.clone();
            fft.process(&mut data);
        }
        let rust_ns = rust_start.elapsed().as_nanos() as f64 / iterations as f64;

        println!("Size {}:", size);
        println!("  Plan creation:   {:8.1} ns", plan_ns);
        println!("  OxiFFT execute:  {:8.1} ns", exec_ns);
        println!("  RustFFT execute: {:8.1} ns", rust_ns);
        println!("  Ratio:           {:8.2}x", exec_ns / rust_ns);
        println!();
    }
}
