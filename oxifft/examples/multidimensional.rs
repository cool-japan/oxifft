//! Multi-dimensional FFT example.
//!
//! Demonstrates 2D and 3D FFT transforms using `OxiFFT`.

#![allow(clippy::cast_precision_loss)] // FFT size computations use float for math
#![allow(clippy::similar_names)] // row/col and 2d/3d names are intentionally similar

use oxifft::api::{fft2d, fft_nd, ifft2d, ifft_nd};
use oxifft::Complex;

fn main() {
    // ========================
    // 2D FFT Example
    // ========================
    println!("=== 2D FFT Example ===\n");

    let rows = 4;
    let cols = 4;

    // Create a 2D signal (stored as 1D array in row-major order)
    let input_2d: Vec<Complex<f64>> = (0..(rows * cols))
        .map(|idx| {
            let row = idx / cols;
            let col = idx % cols;
            // Simple pattern: 1.0 in center, 0.0 elsewhere
            if row == 1 && col == 1 {
                Complex::new(1.0, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            }
        })
        .collect();

    println!("2D Input ({rows}x{cols}):");
    for row in 0..rows {
        print!("  ");
        for col in 0..cols {
            print!("{:+.2} ", input_2d[row * cols + col].re);
        }
        println!();
    }

    // Compute 2D FFT
    let spectrum_2d = fft2d(&input_2d, rows, cols);

    println!("\n2D FFT Output (magnitudes):");
    for row in 0..rows {
        print!("  ");
        for col in 0..cols {
            let c = spectrum_2d[row * cols + col];
            let mag = c.re.hypot(c.im);
            print!("{mag:+.2} ");
        }
        println!();
    }

    // Inverse 2D FFT
    let recovered_2d = ifft2d(&spectrum_2d, rows, cols);

    let max_error_2d: f64 = input_2d
        .iter()
        .zip(recovered_2d.iter())
        .map(|(a, b)| (a.re - b.re).hypot(a.im - b.im))
        .fold(0.0, f64::max);

    println!("\n2D roundtrip error: {max_error_2d:.2e}");

    // ========================
    // N-Dimensional FFT Example (3D)
    // ========================
    println!("\n=== 3D FFT Example ===\n");

    let dims = [2, 2, 2]; // 2x2x2 cube
    let total = dims.iter().product::<usize>();

    // Create a 3D signal
    let input_3d: Vec<Complex<f64>> = (0..total)
        .map(|idx| Complex::new(idx as f64, 0.0))
        .collect();

    println!(
        "3D Input values: {:?}",
        input_3d.iter().map(|c| c.re).collect::<Vec<_>>()
    );

    // Compute N-D FFT
    let spectrum_nd = fft_nd(&input_3d, &dims);

    println!(
        "3D FFT output magnitudes: {:?}",
        spectrum_nd
            .iter()
            .map(|c| c.re.hypot(c.im))
            .collect::<Vec<_>>()
    );

    // Inverse N-D FFT
    let recovered_nd = ifft_nd(&spectrum_nd, &dims);

    let max_error_nd: f64 = input_3d
        .iter()
        .zip(recovered_nd.iter())
        .map(|(a, b)| (a.re - b.re).hypot(a.im - b.im))
        .fold(0.0, f64::max);

    println!("3D roundtrip error: {max_error_nd:.2e}");
}
