//! Example demonstrating automatic differentiation for FFT operations.
//!
//! This is useful for optimizing FFT-based neural network layers,
//! signal processing pipelines, and scientific computing workflows.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::items_after_statements)]

use oxifft::autodiff::{fft_jacobian, grad_fft, jvp_fft, vjp_fft, DiffFftPlan, DualComplex};
use oxifft::Complex;

fn main() {
    println!("=== Automatic Differentiation for FFT ===\n");

    // Example 1: Gradient computation (backward mode)
    println!("Example 1: Backward mode gradient");
    println!("Compute ∂L/∂x given ∂L/∂y\n");

    let n = 8;

    // Create input signal
    let input: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(i as f64, 0.0)).collect();

    // Gradient from loss w.r.t. FFT output
    let grad_output: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new(1.0 / (i + 1) as f64, 0.0))
        .collect();

    println!(
        "Input signal: {:?}",
        input.iter().map(|c| c.re).collect::<Vec<_>>()
    );
    println!(
        "Gradient w.r.t. output: {:?}",
        grad_output.iter().map(|c| c.re).collect::<Vec<_>>()
    );

    let grad_input = grad_fft(&grad_output).expect("Gradient computation failed");

    println!("\nGradient w.r.t. input:");
    for (i, g) in grad_input.iter().enumerate() {
        println!("  ∂L/∂x[{}] = ({:.4}, {:.4}i)", i, g.re, g.im);
    }
    println!();

    // Example 2: Vector-Jacobian Product (VJP)
    println!("Example 2: Vector-Jacobian Product");
    println!("Efficient gradient computation for backpropagation\n");

    let vjp_result = vjp_fft(&grad_output).expect("VJP failed");

    println!("VJP result:");
    for (i, val) in vjp_result.iter().enumerate() {
        println!("  vjp[{}] = ({:.4}, {:.4}i)", i, val.re, val.im);
    }
    println!();

    // Example 3: Jacobian-Vector Product (JVP - forward mode)
    println!("Example 3: Jacobian-Vector Product");
    println!("Forward mode differentiation\n");

    let jvp_result = jvp_fft(&input).expect("JVP failed");

    println!("JVP result (directional derivative):");
    for (i, val) in jvp_result.iter().enumerate() {
        println!("  jvp[{}] = ({:.4}, {:.4}i)", i, val.re, val.im);
    }
    println!();

    // Example 4: Full Jacobian matrix (small size only!)
    println!("Example 4: Full Jacobian matrix");
    println!("For small N, compute the entire DFT Jacobian\n");

    let n_small = 4;
    let jacobian = fft_jacobian::<f64>(n_small);

    println!("Jacobian matrix ({n_small}×{n_small}):");
    for i in 0..n_small {
        print!("  Row {i}: [");
        for j in 0..n_small {
            print!("({:.3},{:.3}i)", jacobian[i][j].re, jacobian[i][j].im);
            if j < n_small - 1 {
                print!(", ");
            }
        }
        println!("]");
    }
    println!();

    // Example 5: Plan-based differentiation
    println!("Example 5: DiffFftPlan for repeated differentiation");
    println!("Efficient when computing gradients multiple times\n");

    let plan = DiffFftPlan::<f64>::new(n).expect("Failed to create differentiable FFT plan");

    // Forward pass with dual numbers (carries derivatives)
    let dual_input: Vec<DualComplex<f64>> = input
        .iter()
        .enumerate()
        .map(|(i, &val)| {
            let value = val;
            let deriv = if i == 0 {
                Complex::new(1.0, 0.0) // Derivative w.r.t. first input
            } else {
                Complex::new(0.0, 0.0)
            };
            DualComplex { value, deriv }
        })
        .collect();

    let (fft_output, sensitivity) = plan.forward_dual(&dual_input);

    println!("FFT output:");
    for (i, val) in fft_output.iter().enumerate() {
        println!("  y[{}] = ({:.4}, {:.4}i)", i, val.re, val.im);
    }

    println!("\nSensitivity ∂y/∂x[0]:");
    for (i, val) in sensitivity.iter().enumerate() {
        println!("  ∂y[{}]/∂x[0] = ({:.4}, {:.4}i)", i, val.re, val.im);
    }
    println!();

    // Example 6: Gradient check (numerical vs analytical)
    println!("Example 6: Gradient verification");
    println!("Compare analytical gradient with numerical approximation\n");

    let epsilon = 1e-6;
    let test_input = vec![
        Complex::new(1.0, 0.5),
        Complex::new(2.0, -0.3),
        Complex::new(0.5, 1.2),
        Complex::new(-1.0, 0.8),
    ];
    let n_test = test_input.len();

    // Analytical gradient
    let grad_out_test = vec![Complex::new(1.0, 0.0); n_test];
    let analytical_grad = grad_fft(&grad_out_test).expect("Gradient failed");

    // Numerical gradient (finite difference)
    let mut numerical_grad = vec![Complex::new(0.0, 0.0); n_test];
    for i in 0..n_test {
        let mut perturbed = test_input.clone();
        perturbed[i].re += epsilon;

        // Compute FFT of perturbed input
        use oxifft::fft;
        let fft_base = fft(&test_input);
        let fft_perturbed = fft(&perturbed);

        // Finite difference for real part
        for j in 0..n_test {
            numerical_grad[i].re +=
                (fft_perturbed[j].re - fft_base[j].re) * grad_out_test[j].re / epsilon;
            numerical_grad[i].re +=
                (fft_perturbed[j].im - fft_base[j].im) * grad_out_test[j].im / epsilon;
        }
    }

    println!("Gradient comparison:");
    for i in 0..n_test {
        let diff = f64::abs(analytical_grad[i].re - numerical_grad[i].re);
        println!(
            "  Input {}: analytical = {:.6}, numerical = {:.6}, diff = {:.2e}",
            i, analytical_grad[i].re, numerical_grad[i].re, diff
        );
    }
}
