//! Simple FFT example demonstrating basic usage of `OxiFFT`.
//!
//! This example shows how to compute forward and inverse FFTs.

use oxifft::api::{fft, ifft};
use oxifft::Complex;

fn main() {
    // Create a simple input signal: a sinusoid at frequency 2
    let n = 16;
    let input: Vec<Complex<f64>> = (0..n)
        .map(|k| {
            let t = 2.0 * std::f64::consts::PI * f64::from(k) / f64::from(n);
            // A sine wave at frequency 2
            Complex::new((2.0 * t).sin(), 0.0)
        })
        .collect();

    println!("Input signal (real parts):");
    for (i, c) in input.iter().enumerate() {
        println!("  x[{:2}] = {:+.4}", i, c.re);
    }

    // Compute FFT
    let spectrum = fft(&input);

    println!("\nFFT output (magnitude):");
    for (i, c) in spectrum.iter().enumerate() {
        let mag = c.re.hypot(c.im);
        if mag > 0.01 {
            println!(
                "  X[{:2}] = {:+.4} + {:+.4}i  (|X| = {:.4})",
                i, c.re, c.im, mag
            );
        }
    }

    // Compute inverse FFT to recover original signal
    let recovered = ifft(&spectrum);

    println!("\nRecovered signal (real parts):");
    for (i, c) in recovered.iter().enumerate() {
        println!("  x[{:2}] = {:+.4}", i, c.re);
    }

    // Verify roundtrip
    let max_error: f64 = input
        .iter()
        .zip(recovered.iter())
        .map(|(a, b)| (a.re - b.re).hypot(a.im - b.im))
        .fold(0.0, f64::max);

    println!("\nMax roundtrip error: {max_error:.2e}");
}
