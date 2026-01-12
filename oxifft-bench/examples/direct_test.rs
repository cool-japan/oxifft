#![allow(clippy::cast_precision_loss)]
#![allow(clippy::uninlined_format_args)]

use oxifft::api::{Direction, Flags, Plan};
use oxifft::Complex;

fn main() {
    println!("Testing OxiFFT execution directly...");

    for size in &[16, 64, 256] {
        println!("\nTesting size {}", size);

        let input: Vec<Complex<f64>> = (0..*size)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        println!("  Creating plan...");
        let plan = Plan::dft_1d(*size, Direction::Forward, Flags::ESTIMATE)
            .expect("Failed to create plan");

        println!("  Executing FFT...");
        let mut output = vec![Complex::zero(); *size];
        plan.execute(&input, &mut output);

        println!("  ✓ Success! First output value: {:?}", output[0]);
    }

    println!("\nAll tests passed!");
}
