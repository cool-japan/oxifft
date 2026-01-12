//! Batch FFT example demonstrating processing multiple transforms at once.
//!
//! Batch processing is useful when you need to transform many signals of the same size.

#![allow(clippy::cast_precision_loss)] // FFT size computations use float for math

use oxifft::api::{fft_batch, ifft_batch};
use oxifft::Complex;

fn main() {
    // Parameters
    let n = 8; // Size of each FFT
    let howmany = 4; // Number of transforms

    // Create batch input: 4 different signals, each of size 8
    let mut input: Vec<Complex<f64>> = Vec::with_capacity(n * howmany);

    for batch in 0..howmany {
        let freq = (batch + 1) as f64; // Each batch has a different frequency
        for k in 0..n {
            let t = 2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
            input.push(Complex::new((freq * t).sin(), 0.0));
        }
    }

    println!("Batch input signals:");
    for batch in 0..howmany {
        print!("  Signal {batch}: [");
        for k in 0..n {
            print!("{:+.2}", input[batch * n + k].re);
            if k < n - 1 {
                print!(", ");
            }
        }
        println!("]");
    }

    // Compute batch FFT
    let spectrum = fft_batch(&input, n, howmany);

    println!("\nBatch FFT output (magnitudes > 0.1):");
    for batch in 0..howmany {
        print!("  Spectrum {batch}: ");
        let mut found_any = false;
        for k in 0..n {
            let c = spectrum[batch * n + k];
            let mag = c.re.hypot(c.im);
            if mag > 0.1 {
                if found_any {
                    print!(", ");
                }
                print!("X[{k}]={mag:.2}");
                found_any = true;
            }
        }
        println!();
    }

    // Compute batch inverse FFT
    let recovered = ifft_batch(&spectrum, n, howmany);

    // Verify roundtrip
    let max_error: f64 = input
        .iter()
        .zip(recovered.iter())
        .map(|(a, b)| (a.re - b.re).hypot(a.im - b.im))
        .fold(0.0, f64::max);

    println!("\nMax roundtrip error: {max_error:.2e}");
}
