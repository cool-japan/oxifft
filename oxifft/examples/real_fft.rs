//! Real FFT example demonstrating R2C and C2R transforms.
//!
//! For real-valued signals, the R2C transform is more efficient than
//! padding to complex and using the standard FFT.

#![allow(clippy::cast_precision_loss)] // FFT size computations use float for math

use oxifft::api::{irfft, rfft};

fn main() {
    // Create a real-valued signal
    let n = 16;
    let input: Vec<f64> = (0..n)
        .map(|k| {
            let t = 2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
            // Sum of two cosines at frequencies 1 and 3
            0.5f64.mul_add((3.0 * t).cos(), (t).cos())
        })
        .collect();

    println!("Real input signal:");
    for (i, &x) in input.iter().enumerate() {
        println!("  x[{i:2}] = {x:+.4}");
    }

    // Compute real-to-complex FFT
    // Output has n/2 + 1 complex values due to Hermitian symmetry
    let spectrum = rfft(&input);

    println!("\nR2C FFT output (complex spectrum):");
    println!("  (Note: For real input, only n/2+1 values are unique)");
    for (i, c) in spectrum.iter().enumerate() {
        let mag = c.re.hypot(c.im);
        if mag > 0.01 {
            println!(
                "  X[{:2}] = {:+.4} + {:+.4}i  (|X| = {:.4})",
                i, c.re, c.im, mag
            );
        }
    }

    // Compute inverse (complex-to-real)
    let recovered = irfft(&spectrum, n);

    println!("\nRecovered real signal:");
    for (i, &x) in recovered.iter().enumerate() {
        println!("  x[{i:2}] = {x:+.4}");
    }

    // Verify roundtrip
    let max_error: f64 = input
        .iter()
        .zip(recovered.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    println!("\nMax roundtrip error: {max_error:.2e}");
}
