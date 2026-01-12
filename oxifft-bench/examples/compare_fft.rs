#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::uninlined_format_args)]

use oxifft::api::{Direction, Flags, Plan};
use oxifft::Complex;
use rustfft::FftPlanner;
use std::time::Instant;

fn main() {
    let sizes = [16, 64, 256, 1024, 4096, 16384, 65536];

    println!("FFT Performance Comparison: OxiFFT vs RustFFT");
    println!("==============================================\n");

    for &size in &sizes {
        // Adjust iterations based on size
        let iterations = if size >= 16384 { 1000 } else { 10000 };

        // OxiFFT
        let oxi_input: Vec<Complex<f64>> = (0..size)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let plan =
            Plan::dft_1d(size, Direction::Forward, Flags::ESTIMATE).expect("Failed to create plan");
        let mut oxi_output = vec![Complex::zero(); size];

        // Warm up
        for _ in 0..100 {
            plan.execute(&oxi_input, &mut oxi_output);
        }
        let start = Instant::now();
        for _ in 0..iterations {
            plan.execute(&oxi_input, &mut oxi_output);
        }
        let oxi_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

        // RustFFT
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(size);
        let rust_data: Vec<rustfft::num_complex::Complex<f64>> = (0..size)
            .map(|i| rustfft::num_complex::Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        // Warm up
        for _ in 0..100 {
            let mut data = rust_data.clone();
            fft.process(&mut data);
        }
        let start = Instant::now();
        for _ in 0..iterations {
            let mut data = rust_data.clone();
            fft.process(&mut data);
        }
        let rust_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

        let ratio = oxi_ns / rust_ns;
        println!(
            "Size {:5}: OxiFFT {:8.1}ns | RustFFT {:8.1}ns | Ratio: {:.2}x",
            size, oxi_ns, rust_ns, ratio
        );
    }
}
