//! Example demonstrating Sparse FFT for k-sparse signals.
//!
//! Sparse FFT achieves O(k log n) complexity instead of O(n log n)
//! when the signal has at most k non-zero frequency components.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::needless_range_loop)]

use oxifft::{sparse_fft, Complex, Flags, SparsePlan};

fn main() {
    println!("=== Sparse FFT Example ===\n");

    // Signal parameters
    let n = 1024;
    let k = 10; // Expected sparsity

    // Create a sparse signal with only a few frequency components
    let mut signal = vec![Complex::new(0.0, 0.0); n];

    // Add a few sinusoidal components (corresponding to sparse frequency domain)
    let two_pi = 2.0 * std::f64::consts::PI;
    for i in 0..n {
        let t = i as f64 / n as f64;

        // Frequency at bin 50
        signal[i].re += (two_pi * 50.0 * t).cos();
        signal[i].im += (two_pi * 50.0 * t).sin();

        // Frequency at bin 150
        signal[i].re = 0.5f64.mul_add((two_pi * 150.0 * t).cos(), signal[i].re);
        signal[i].im = 0.5f64.mul_add((two_pi * 150.0 * t).sin(), signal[i].im);

        // Frequency at bin 300
        signal[i].re = 0.3f64.mul_add((two_pi * 300.0 * t).cos(), signal[i].re);
        signal[i].im = 0.3f64.mul_add((two_pi * 300.0 * t).sin(), signal[i].im);
    }

    println!("Signal size: {n} points");
    println!("Expected sparsity: {k} non-zero frequencies\n");

    // Method 1: One-shot API
    println!("Method 1: One-shot sparse_fft()");
    let result = sparse_fft(&signal, k);

    println!("Detected {} frequency components:", result.indices.len());
    for (idx, value) in result.indices.iter().zip(result.values.iter()) {
        let magnitude = value.norm();
        if magnitude > 0.1 {
            println!("  Frequency bin {idx}: magnitude = {magnitude:.3}");
        }
    }
    println!();

    // Method 2: Plan-based API (for repeated use)
    println!("Method 2: Plan-based SparsePlan");
    let plan = SparsePlan::new(n, k, Flags::ESTIMATE).expect("Failed to create sparse FFT plan");

    println!("Plan created:");
    println!("  Transform size: {}", plan.n());
    println!("  Sparsity: {}", plan.k());
    println!("  Number of stages: {}", plan.num_stages());
    println!("  Estimated operations: {}", plan.estimated_ops());
    println!();

    let result2 = plan.execute(&signal);
    println!(
        "Detected {} frequency components (plan-based)",
        result2.indices.len()
    );
    println!();

    // Performance comparison hint
    println!("Performance note:");
    println!(
        "  Standard FFT: O(n log n) = O({} log {}) ≈ {} operations",
        n,
        n,
        n * (n as f64).log2() as usize
    );
    println!(
        "  Sparse FFT:   O(k log n) = O({} log {}) ≈ {} operations",
        k,
        n,
        plan.estimated_ops()
    );
    println!(
        "  Speedup potential: {:.1}x",
        (n * (n as f64).log2() as usize) as f64 / plan.estimated_ops() as f64
    );
}
