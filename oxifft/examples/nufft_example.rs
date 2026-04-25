//! Example demonstrating Non-Uniform FFT (NUFFT) for irregularly sampled data.
//!
//! NUFFT is useful when data is sampled at non-uniform intervals,
//! common in astronomy, medical imaging, and compressed sensing.

use oxifft::nufft::{
    nufft2d_type1, nufft_type1, nufft_type2, nufft_type3, Nufft, NufftOptions, NufftType,
};
use oxifft::Complex;

/// Demonstrate Type 3 NUFFT: non-uniform source → non-uniform target.
fn demo_type3(tolerance: f64) {
    println!("Type 3 NUFFT: Non-uniform → Non-uniform");
    println!("Use case: General scattered data interpolation\n");

    let source_points = vec![-2.0, -0.5, 0.5, 1.8_f64];
    let source_values: Vec<Complex<f64>> = source_points
        .iter()
        .map(|&t| Complex::new(t.cos(), 0.0))
        .collect();

    let target_points = vec![-1.5, 0.0, 1.0, 2.5_f64];

    let result = nufft_type3(&source_points, &source_values, &target_points, tolerance)
        .expect("Type 3 NUFFT failed");

    println!("Source points: {source_points:?}");
    println!("Target points: {target_points:?}");
    println!("Transformed values:");
    for (point, value) in target_points.iter().zip(result.iter()) {
        println!("  Target {:.2}: ({:.4}, {:.4}i)", point, value.re, value.im);
    }
    println!();
}

/// Demonstrate 2D NUFFT (new in v0.3.0): non-uniform 2D points → uniform 2D grid.
fn demo_2d_nufft() {
    println!("2D NUFFT (new in v0.3.0): Non-uniform 2D points → Uniform 2D grid");
    let points_x = vec![-2.0, -0.5, 0.8, 1.5_f64];
    let points_y = vec![-1.5, 0.3, -0.7, 2.1_f64];
    let values_2d: Vec<Complex<f64>> = points_x
        .iter()
        .zip(points_y.iter())
        .map(|(&x, &y)| Complex::new((x + y).cos(), (x - y).sin()))
        .collect();
    let options_2d = NufftOptions {
        tolerance: 1e-6,
        ..Default::default()
    };
    // Output grid size: 8x8; all points must lie in [-π, π]²
    let grid_2d = nufft2d_type1(&points_x, &points_y, &values_2d, 8, 8, &options_2d);
    match grid_2d {
        Ok(grid) => println!("2D NUFFT output grid: 8x8 = {} coefficients\n", grid.len()),
        Err(e) => println!("2D NUFFT: {e}\n"),
    }
}

fn main() {
    println!("=== Non-Uniform FFT (NUFFT) Example ===\n");

    // Irregular sampling points in [-π, π] (required by the NUFFT API)
    let sample_points = vec![-2.8, -1.9, -0.8, 0.1, 0.9, 1.6, 2.1, -0.4, 0.5, 2.5_f64];
    let n_samples = sample_points.len();
    let frequency = 3.0_f64;
    let values: Vec<Complex<f64>> = sample_points
        .iter()
        .map(|&t| {
            let phase = 2.0 * std::f64::consts::PI * frequency * t;
            Complex::new(phase.cos(), phase.sin())
        })
        .collect();

    // Type 1: Non-uniform to uniform (analysis)
    println!("Type 1 NUFFT: Non-uniform time → Uniform frequency");
    println!("Use case: Irregularly sampled signal analysis\n");
    println!("Irregular sampling points: {sample_points:?}");
    println!("Number of samples: {n_samples}");
    println!("True frequency: {frequency} cycles\n");

    let n_freq = 16;
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
    let mut freq_spectrum = vec![Complex::new(0.0, 0.0); 16];
    freq_spectrum[3] = Complex::new(1.0, 0.0);
    let eval_points = vec![-2.5, -1.5, -0.5, 0.5, 1.5, 2.5_f64];
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

    demo_type3(tolerance);
    demo_2d_nufft();

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

    // Execute plan (can be reused for different input values); points must be in [-π, π]
    let spectrum2 = plan.type1(&values).expect("NUFFT execution failed");
    println!("Spectrum computed via plan: {} bins", spectrum2.len());

    let max_diff: f64 = spectrum
        .iter()
        .zip(spectrum2.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0_f64, f64::max);
    println!("Max difference from one-shot API: {max_diff:.2e}");
}
