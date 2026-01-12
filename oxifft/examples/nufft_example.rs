//! Example demonstrating Non-Uniform FFT (NUFFT) for irregularly sampled data.
//!
//! NUFFT is useful when data is sampled at non-uniform intervals,
//! common in astronomy, medical imaging, and compressed sensing.

use oxifft::nufft::{nufft_type1, nufft_type2, nufft_type3, Nufft, NufftOptions, NufftType};
use oxifft::Complex;

fn main() {
    println!("=== Non-Uniform FFT (NUFFT) Example ===\n");

    // Type 1: Non-uniform to uniform (analysis)
    println!("Type 1 NUFFT: Non-uniform time → Uniform frequency");
    println!("Use case: Irregularly sampled signal analysis\n");

    // Irregular sampling points (normalized to [0, 1))
    let sample_points = vec![0.05, 0.12, 0.23, 0.31, 0.45, 0.58, 0.67, 0.79, 0.88, 0.95];
    let n_samples = sample_points.len();

    // Sample a sinusoidal signal at irregular points
    let frequency = 3.0; // 3 cycles
    let values: Vec<Complex<f64>> = sample_points
        .iter()
        .map(|&t| {
            let phase = 2.0 * std::f64::consts::PI * frequency * t;
            Complex::new(phase.cos(), phase.sin())
        })
        .collect();

    println!("Irregular sampling points: {sample_points:?}");
    println!("Number of samples: {n_samples}");
    println!("True frequency: {frequency} cycles\n");

    // Compute Type 1 NUFFT
    let n_freq = 16; // Number of uniform frequency bins
    let tolerance = 1e-6;

    let spectrum =
        nufft_type1(&sample_points, &values, n_freq, tolerance).expect("Type 1 NUFFT failed");

    println!("Frequency spectrum ({} bins):", spectrum.len());
    for (k, val) in spectrum.iter().enumerate() {
        let magnitude = val.norm();
        if magnitude > 0.1 {
            println!("  Bin {k}: magnitude = {magnitude:.4}");
        }
    }
    println!();

    // Type 2: Uniform to non-uniform (synthesis)
    println!("Type 2 NUFFT: Uniform frequency → Non-uniform time");
    println!("Use case: Signal reconstruction at specific points\n");

    // Create a frequency spectrum with a single frequency
    let mut freq_spectrum = vec![Complex::new(0.0, 0.0); 16];
    freq_spectrum[3] = Complex::new(1.0, 0.0); // Frequency bin 3

    // Desired output points (non-uniform)
    let eval_points = vec![0.1, 0.25, 0.4, 0.55, 0.7, 0.85];

    let interpolated =
        nufft_type2(&freq_spectrum, &eval_points, tolerance).expect("Type 2 NUFFT failed");

    println!("Evaluation points: {eval_points:?}");
    println!("Interpolated values:");
    for (point, value) in eval_points.iter().zip(interpolated.iter()) {
        println!(
            "  t = {:.2}: ({:.4}, {:.4}i) magnitude = {:.4}",
            point,
            value.re,
            value.im,
            value.norm()
        );
    }
    println!();

    // Type 3: Non-uniform to non-uniform
    println!("Type 3 NUFFT: Non-uniform → Non-uniform");
    println!("Use case: General scattered data interpolation\n");

    let source_points = vec![0.15, 0.35, 0.55, 0.75];
    let source_values: Vec<Complex<f64>> = source_points
        .iter()
        .map(|&t| Complex::new((2.0 * std::f64::consts::PI * t).cos(), 0.0))
        .collect();

    let target_points = vec![0.2, 0.4, 0.6, 0.8];

    let result = nufft_type3(&source_points, &source_values, &target_points, tolerance)
        .expect("Type 3 NUFFT failed");

    println!("Source points: {source_points:?}");
    println!("Target points: {target_points:?}");
    println!("Transformed values:");
    for (point, value) in target_points.iter().zip(result.iter()) {
        println!("  Target {:.2}: ({:.4}, {:.4}i)", point, value.re, value.im);
    }
    println!();

    // Plan-based API for repeated use
    println!("Plan-based NUFFT (for repeated transforms):");
    let options = NufftOptions {
        tolerance: 1e-8,
        oversampling: 2.0,
        kernel_width: 6,
        threaded: true,
    };

    let plan = Nufft::with_options(NufftType::Type1, n_freq, &sample_points, &options)
        .expect("Failed to create NUFFT plan");

    println!("NUFFT plan created:");
    println!("  Non-uniform points: {}", sample_points.len());
    println!("  Uniform output size: {n_freq}");
    println!("  Tolerance: {:.0e}", 1e-8);
    println!();

    // Execute plan (can be reused for different input values)
    let spectrum2 = plan.execute(&values).expect("NUFFT execution failed");
    println!("Spectrum computed via plan: {} bins", spectrum2.len());

    // Verify consistency
    let max_diff: f64 = spectrum
        .iter()
        .zip(spectrum2.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0_f64, f64::max);
    println!("Max difference from one-shot API: {max_diff:.2e}");
}
