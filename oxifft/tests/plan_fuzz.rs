#![allow(clippy::cast_precision_loss)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::needless_pass_by_ref_mut)]
//! Property-based ("fuzz-style") tests for plan creation and execution.
//!
//! These tests exercise plan creation with arbitrary sizes to catch panics,
//! infinite loops, or incorrect results that deterministic unit tests miss.
//! They use proptest as a lightweight fuzz framework (no cargo-fuzz required).

use oxifft::{Direction, Flags, Plan};
use proptest::prelude::*;

// Verify plan creation never panics and always returns Some/None correctly.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn plan_1d_creation_never_panics(n in 0usize..=8192) {
        // Must never panic — only return Some or None
        let plan = Plan::<f64>::dft_1d(n, Direction::Forward, Flags::ESTIMATE);
        if n == 0 {
            // size-0 returns Some (Nop plan) — that's the current behavior
            // Just ensure no panic occurred
        }
        let _ = plan; // drop cleanly
    }

    #[test]
    fn plan_1d_f32_creation_never_panics(n in 0usize..=4096) {
        let plan = Plan::<f32>::dft_1d(n, Direction::Forward, Flags::ESTIMATE);
        let _ = plan;
    }

    #[test]
    fn plan_r2c_creation_never_panics(n in 0usize..=4096) {
        let plan = Plan::<f64>::r2c_1d(n, Flags::ESTIMATE);
        let _ = plan;
    }

    #[test]
    fn plan_c2r_creation_never_panics(n in 0usize..=4096) {
        let plan = Plan::<f64>::c2r_1d(n, Flags::ESTIMATE);
        let _ = plan;
    }

    #[test]
    fn plan_2d_creation_never_panics(
        n0 in 0usize..=64,
        n1 in 0usize..=64,
    ) {
        let plan = Plan::<f64>::dft_2d(n0, n1, Direction::Forward, Flags::ESTIMATE);
        let _ = plan;
    }

    #[test]
    fn plan_3d_creation_never_panics(
        n0 in 0usize..=16,
        n1 in 0usize..=16,
        n2 in 0usize..=16,
    ) {
        let plan = Plan::<f64>::dft_3d(n0, n1, n2, Direction::Forward, Flags::ESTIMATE);
        let _ = plan;
    }
}

// Verify roundtrip correctness: IFFT(FFT(x)) ≈ x for arbitrary sizes.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn plan_1d_roundtrip_arbitrary_size(
        n in 1usize..=512,
        // Use small rational amplitudes to avoid floating-point catastrophe
        amp in 0.1f64..=10.0,
    ) {
        use oxifft::Complex;

        let fwd = match Plan::<f64>::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
            Some(p) => p,
            None => return Ok(()), // skip if plan creation fails
        };
        let bwd = match Plan::<f64>::dft_1d(n, Direction::Backward, Flags::ESTIMATE) {
            Some(p) => p,
            None => return Ok(()),
        };

        // Build a simple input: DC + first harmonic
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(amp * (i as f64 / n as f64), 0.0))
            .collect();
        let mut spectrum = vec![Complex::new(0.0, 0.0); n];
        let mut recovered = vec![Complex::new(0.0, 0.0); n];

        fwd.execute(&input, &mut spectrum);
        bwd.execute(&spectrum, &mut recovered);

        // IFFT is unnormalised: recovered[i] ≈ n * input[i]
        let scale = n as f64;
        for (orig, rec) in input.iter().zip(recovered.iter()) {
            let err = orig.re.mul_add(scale, -rec.re).abs() + orig.im.mul_add(scale, -rec.im).abs();
            prop_assert!(
                err < 1e-8 * scale,
                "roundtrip error {err} at n={n}: orig={orig:?}, scaled_rec={:?}",
                Complex::new(rec.re, rec.im)
            );
        }
    }

    #[test]
    fn real_plan_roundtrip_power_of_two(
        // log2(n): sizes 2^1 through 2^9 = 2..=512
        log2_n in 1u32..=9u32,
    ) {
        // R2C/C2R is currently stable for power-of-2 sizes.
        // Non-power-of-2 real FFT correctness is tracked as a v0.3.0 item.
        let n = 1usize << log2_n;
        use oxifft::irfft;
        use oxifft::rfft;

        // Real input: simple sinusoidal + ramp to exercise DC and AC components
        let input: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.3_f64).sin() + i as f64 / n as f64)
            .collect();

        let spectrum = rfft(&input);
        let recovered = irfft(&spectrum, n);

        for (i, (orig, rec)) in input.iter().zip(recovered.iter()).enumerate() {
            let err = (orig - rec).abs();
            prop_assert!(
                err < 1e-9,
                "real roundtrip error {err:.2e} at n={n} i={i}: orig={orig:.6}, rec={rec:.6}"
            );
        }
    }
}

// Verify Parseval's theorem holds for arbitrary sizes.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn parseval_arbitrary_size(n in 1usize..=256) {
        use oxifft::Complex;

        let plan = match Plan::<f64>::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
            Some(p) => p,
            None => return Ok(()),
        };

        // Build random-ish input using deterministic formula
        let mut input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(
                (i as f64 * 0.7_f64).sin(),
                (i as f64 * 1.3_f64).cos(),
            ))
            .collect();
        let mut spectrum = vec![Complex::new(0.0, 0.0); n];

        fwd_plan_execute(&plan, &mut input, &mut spectrum);

        // Parseval: Σ|x[i]|² = (1/n) Σ|X[k]|²
        let energy_time: f64 = input.iter().map(|c| c.im.mul_add(c.im, c.re * c.re)).sum();
        let energy_freq: f64 = spectrum.iter().map(|c| c.im.mul_add(c.im, c.re * c.re)).sum();

        let parseval_ratio = energy_freq / (n as f64 * energy_time.max(1e-300));
        prop_assert!(
            (parseval_ratio - 1.0).abs() < 1e-9,
            "Parseval violated at n={n}: ratio={parseval_ratio}"
        );
    }
}

// Helper to work around borrow checker: plan takes &mut even though it doesn't mutate.
fn fwd_plan_execute(
    plan: &Plan<f64>,
    input: &mut Vec<oxifft::Complex<f64>>,
    output: &mut Vec<oxifft::Complex<f64>>,
) {
    plan.execute(input, output);
}
