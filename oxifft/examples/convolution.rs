//! Example demonstrating FFT-based convolution and correlation.
//!
//! Convolution is fundamental in signal processing, image filtering,
//! and polynomial multiplication. FFT-based convolution is O(n log n)
//! instead of O(n²) for direct convolution.

#![allow(clippy::too_many_lines)]

use oxifft::conv::{
    convolve, convolve_circular, convolve_mode, correlate, polynomial_multiply, polynomial_power,
    ConvMode,
};

fn main() {
    println!("=== FFT-based Convolution and Correlation ===\n");

    // Example 1: Linear convolution
    println!("Example 1: Linear convolution");
    println!("Use case: Filtering, impulse response\n");

    let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let kernel = vec![0.5, 1.0, 0.5]; // Simple smoothing kernel

    let result = convolve(&signal, &kernel);

    println!("Signal: {signal:?}");
    println!("Kernel: {kernel:?}");
    println!("Convolution result: {result:?}");
    println!(
        "Output length: {} (input {} + kernel {} - 1)\n",
        result.len(),
        signal.len(),
        kernel.len()
    );

    // Example 2: Convolution modes
    println!("Example 2: Convolution with different modes");
    println!("Modes: Full, Same, Valid\n");

    let sig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let filt = vec![1.0, 2.0, 1.0];

    let full = convolve_mode(&sig, &filt, ConvMode::Full);
    let same = convolve_mode(&sig, &filt, ConvMode::Same);
    let valid = convolve_mode(&sig, &filt, ConvMode::Valid);

    println!("Signal length: {}", sig.len());
    println!("Filter length: {}", filt.len());
    println!("Full mode (len {}): {:?}", full.len(), full);
    println!("Same mode (len {}): {:?}", same.len(), same);
    println!("Valid mode (len {}): {:?}", valid.len(), valid);
    println!();

    // Example 3: Circular convolution
    println!("Example 3: Circular convolution");
    println!("Use case: Periodic signals, DFT property\n");

    let periodic_signal = vec![1.0, 2.0, 3.0, 4.0];
    let periodic_kernel = vec![0.25, 0.5, 0.25, 0.0];

    let circular_result = convolve_circular(&periodic_signal, &periodic_kernel);

    println!("Periodic signal: {periodic_signal:?}");
    println!("Periodic kernel: {periodic_kernel:?}");
    println!("Circular convolution: {circular_result:?}");
    println!("Output length: {} (same as input)\n", circular_result.len());

    // Example 4: Cross-correlation
    println!("Example 4: Cross-correlation");
    println!("Use case: Pattern matching, signal alignment\n");

    let template = vec![1.0, 2.0, 1.0];
    let search_signal = vec![0.0, 0.5, 1.0, 2.0, 1.0, 0.5, 0.0];

    let correlation = correlate(&search_signal, &template);

    println!("Template: {template:?}");
    println!("Search signal: {search_signal:?}");
    println!("Correlation: {correlation:?}");

    // Find peak (best match location)
    let (peak_idx, &peak_val) = correlation
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    println!("Peak at index {peak_idx}: {peak_val:.3} (best match location)\n");

    // Example 5: Polynomial multiplication
    println!("Example 5: Polynomial multiplication");
    println!("Use case: Symbolic computation, generating functions\n");

    // Polynomial p(x) = 1 + 2x + 3x²
    let p = vec![1.0, 2.0, 3.0];
    // Polynomial q(x) = 2 + x
    let q = vec![2.0, 1.0];

    let product = polynomial_multiply(&p, &q);

    println!("p(x) = 1 + 2x + 3x²");
    println!("q(x) = 2 + x");
    println!("p(x) × q(x) = 2 + 5x + 8x² + 3x³");
    println!("Coefficients: {product:?}\n");

    // Example 6: Polynomial power
    println!("Example 6: Polynomial power");
    println!("Compute p(x)^n efficiently\n");

    // Polynomial (1 + x)
    let base = vec![1.0, 1.0];

    // Compute (1 + x)³ = 1 + 3x + 3x² + x³
    let power = polynomial_power(&base, 3);

    println!("p(x) = 1 + x");
    println!("p(x)³ = 1 + 3x + 3x² + x³");
    println!("Coefficients: {power:?}");
    println!();

    // Example 7: 2D convolution (via separable filters)
    println!("Example 7: 2D convolution simulation");
    println!("Separable filter: row-wise then column-wise\n");

    // Simulate 2D image as flattened rows
    let image_width = 5;
    let image_height = 5;
    let mut image = vec![0.0; image_width * image_height];

    // Create a simple pattern (cross)
    for i in 0..image_height {
        image[i * image_width + 2] = 1.0; // Vertical line
    }
    for j in 0..image_width {
        image[2 * image_width + j] = 1.0; // Horizontal line
    }

    println!("Original 5×5 image (cross pattern):");
    for i in 0..image_height {
        for j in 0..image_width {
            print!("{:.0} ", image[i * image_width + j]);
        }
        println!();
    }
    println!();

    // 1D smoothing kernel
    let smooth_kernel = vec![0.25, 0.5, 0.25];

    // Row-wise convolution
    let mut smoothed_rows = Vec::new();
    for i in 0..image_height {
        let row: Vec<f64> = (0..image_width)
            .map(|j| image[i * image_width + j])
            .collect();
        let smoothed_row = convolve_mode(&row, &smooth_kernel, ConvMode::Same);
        smoothed_rows.extend(smoothed_row);
    }

    // Column-wise convolution
    let mut smoothed_image = vec![0.0; image_width * image_height];
    for j in 0..image_width {
        let col: Vec<f64> = (0..image_height)
            .map(|i| smoothed_rows[i * image_width + j])
            .collect();
        let smoothed_col = convolve_mode(&col, &smooth_kernel, ConvMode::Same);
        for i in 0..image_height {
            smoothed_image[i * image_width + j] = smoothed_col[i];
        }
    }

    println!("Smoothed 5×5 image (after separable 2D convolution):");
    for i in 0..image_height {
        for j in 0..image_width {
            print!("{:.2} ", smoothed_image[i * image_width + j]);
        }
        println!();
    }
}
