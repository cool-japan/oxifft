#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::uninlined_format_args)]

use oxifft::Complex;
use std::time::Instant;

fn main() {
    let iterations = 100_000;

    println!("Codelet Performance Test");
    println!("========================\n");

    // Test size 64 codelet directly
    let size = 64;
    let mut data: Vec<Complex<f64>> = (0..size)
        .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
        .collect();

    // Warm up
    for _ in 0..1000 {
        oxifft::dft::codelets::simd::notw_64_dispatch(&mut data, -1);
    }

    // Measure SIMD version
    let start = Instant::now();
    for _ in 0..iterations {
        oxifft::dft::codelets::simd::notw_64_dispatch(&mut data, -1);
    }
    let simd_duration = start.elapsed();
    let simd_ns = simd_duration.as_nanos() as f64 / iterations as f64;

    // Reset data and measure scalar version
    let mut data2: Vec<Complex<f64>> = (0..size)
        .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
        .collect();

    for _ in 0..1000 {
        oxifft::dft::codelets::notw_64(&mut data2, -1);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        oxifft::dft::codelets::notw_64(&mut data2, -1);
    }
    let scalar_duration = start.elapsed();
    let scalar_ns = scalar_duration.as_nanos() as f64 / iterations as f64;

    println!("Size 64 codelet:");
    println!("  SIMD:   {:8.1} ns", simd_ns);
    println!("  Scalar: {:8.1} ns", scalar_ns);
    println!("  Ratio:  {:8.2}x", scalar_ns / simd_ns);
}
