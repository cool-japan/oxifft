//! Comprehensive tests for various FFT sizes.
//!
//! Tests correctness across a wide range of transform sizes:
//! - Power-of-2 sizes (2 to 2^16)
//! - Prime sizes
//! - Composite sizes
//! - Edge cases

#![allow(clippy::cast_precision_loss)] // FFT size computations use float for math
#![allow(clippy::unreadable_literal)] // mathematical constants more readable as-is
#![allow(clippy::needless_range_loop)] // explicit loops clearer for FFT indices

use oxifft::{Complex, Direction, Flags, Plan};

/// Generate test input data.
fn generate_input(n: usize) -> Vec<Complex<f64>> {
    (0..n)
        .map(|i| {
            let t = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            Complex::new(t.cos(), t.sin())
        })
        .collect()
}

/// Compute expected output using Direct O(n²) algorithm.
fn dft_direct(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let n = input.len();
    let mut output = vec![Complex::new(0.0, 0.0); n];

    for k in 0..n {
        for j in 0..n {
            let angle = -2.0 * std::f64::consts::PI * (k as f64) * (j as f64) / (n as f64);
            let twiddle = Complex::new(angle.cos(), angle.sin());
            output[k] += input[j] * twiddle;
        }
    }

    output
}

/// Check if two complex vectors are approximately equal.
fn approx_eq(a: &[Complex<f64>], b: &[Complex<f64>], tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x.re - y.re).hypot(x.im - y.im);
        let mag = y.re.hypot(y.im).max(1.0);
        if diff / mag > tol {
            eprintln!(
                "Mismatch at index {}: got ({}, {}), expected ({}, {}), rel_diff = {}",
                i,
                x.re,
                x.im,
                y.re,
                y.im,
                diff / mag
            );
            return false;
        }
    }

    true
}

/// Test FFT correctness for a given size.
fn test_fft_size(n: usize) {
    let input = generate_input(n);
    let expected = dft_direct(&input);

    let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
    let mut output = vec![Complex::new(0.0, 0.0); n];
    plan.execute(&input, &mut output);

    let tol = 1e-9 * (n as f64).log2().max(1.0);
    assert!(
        approx_eq(&output, &expected, tol),
        "FFT mismatch for size {n}"
    );
}

/// Test forward/inverse roundtrip for a given size.
fn test_roundtrip_size(n: usize) {
    let input = generate_input(n);

    let forward_plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
    let inverse_plan = Plan::dft_1d(n, Direction::Backward, Flags::ESTIMATE).unwrap();

    let mut freq = vec![Complex::new(0.0, 0.0); n];
    let mut recovered = vec![Complex::new(0.0, 0.0); n];

    forward_plan.execute(&input, &mut freq);
    inverse_plan.execute(&freq, &mut recovered);

    // Normalize
    let scale = 1.0 / (n as f64);
    for c in &mut recovered {
        c.re *= scale;
        c.im *= scale;
    }

    let tol = 1e-10;
    assert!(
        approx_eq(&input, &recovered, tol),
        "Roundtrip mismatch for size {n}"
    );
}

// ============================================================================
// Power-of-2 tests
// ============================================================================

#[test]
fn test_power_of_2_small() {
    for exp in 1..=6 {
        let n = 1 << exp;
        test_fft_size(n);
    }
}

#[test]
fn test_power_of_2_medium() {
    for exp in 7..=10 {
        let n = 1 << exp;
        test_fft_size(n);
    }
}

#[test]
#[ignore = "slow: stress test for large power-of-2 sizes (2048–16384), run with: cargo test -- --ignored"]
fn test_power_of_2_large() {
    for exp in 11..=14 {
        let n = 1 << exp;
        test_fft_size(n);
    }
}

#[test]
fn test_power_of_2_roundtrip() {
    for exp in 1..=12 {
        let n = 1 << exp;
        test_roundtrip_size(n);
    }
}

// ============================================================================
// Prime size tests
// ============================================================================

#[test]
fn test_small_primes() {
    let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];
    for &p in &primes {
        test_fft_size(p);
    }
}

#[test]
fn test_medium_primes() {
    let primes = [37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97];
    for &p in &primes {
        test_fft_size(p);
    }
}

#[test]
fn test_larger_primes() {
    let primes = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149];
    for &p in &primes {
        test_fft_size(p);
    }
}

#[test]
fn test_prime_roundtrips() {
    let primes = [3, 5, 7, 11, 13, 17, 23, 29, 31, 37, 41, 43, 47];
    for &p in &primes {
        test_roundtrip_size(p);
    }
}

// ============================================================================
// Composite size tests
// ============================================================================

#[test]
fn test_composite_smooth_small() {
    // Products of small primes (2, 3, 5, 7)
    let sizes = [6, 9, 10, 12, 14, 15, 18, 20, 21, 24, 25, 27, 28, 30];
    for &n in &sizes {
        test_fft_size(n);
    }
}

#[test]
fn test_composite_smooth_medium() {
    let sizes = [
        36, 42, 48, 50, 54, 56, 60, 63, 70, 72, 75, 80, 81, 84, 90, 96, 100,
    ];
    for &n in &sizes {
        test_fft_size(n);
    }
}

#[test]
fn test_composite_smooth_large() {
    let sizes = [
        120, 144, 168, 180, 200, 210, 240, 252, 280, 300, 336, 360, 420, 480, 504,
    ];
    for &n in &sizes {
        test_fft_size(n);
    }
}

#[test]
fn test_composite_with_large_prime() {
    // Products that include a larger prime factor
    let sizes = [22, 26, 34, 38, 46, 51, 58, 62, 68, 74, 82, 86, 94];
    for &n in &sizes {
        test_fft_size(n);
    }
}

#[test]
fn test_composite_roundtrips() {
    let sizes = [
        6, 12, 15, 18, 20, 24, 30, 36, 42, 48, 60, 72, 84, 96, 100, 120,
    ];
    for &n in &sizes {
        test_roundtrip_size(n);
    }
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_size_1() {
    let input: Vec<Complex<f64>> = vec![Complex::new(42.0, 17.0)];
    let plan = Plan::dft_1d(1, Direction::Forward, Flags::ESTIMATE).unwrap();
    let mut output: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); 1];
    plan.execute(&input, &mut output);

    assert!((output[0].re - 42.0_f64).abs() < 1e-10);
    assert!((output[0].im - 17.0_f64).abs() < 1e-10);
}

#[test]
fn test_size_2() {
    test_fft_size(2);
    test_roundtrip_size(2);
}

#[test]
fn test_size_3() {
    test_fft_size(3);
    test_roundtrip_size(3);
}

#[test]
fn test_powers_of_4() {
    for exp in 1..=6 {
        let n = 4usize.pow(exp);
        test_fft_size(n);
    }
}

#[test]
fn test_powers_of_8() {
    for exp in 1..=4 {
        let n = 8usize.pow(exp);
        test_fft_size(n);
    }
}

#[test]
fn test_one_less_than_power_of_2() {
    // Sizes like 255, 511, 1023 which are often edge cases
    let sizes = [3, 7, 15, 31, 63, 127, 255, 511, 1023];
    for &n in &sizes {
        test_fft_size(n);
    }
}

#[test]
fn test_one_more_than_power_of_2() {
    // Sizes like 257, 513, 1025
    let sizes = [3, 5, 9, 17, 33, 65, 129, 257, 513, 1025];
    for &n in &sizes {
        test_fft_size(n);
    }
}

// ============================================================================
// Specific algorithm selection tests
// ============================================================================

#[test]
fn test_bluestein_sizes() {
    // Sizes that should trigger Bluestein algorithm (non-smooth, non-prime)
    let sizes = [77, 91, 119, 133, 143, 161, 187, 209, 221, 247, 253, 259];
    for &n in &sizes {
        test_fft_size(n);
    }
}

#[test]
fn test_rader_eligible_primes() {
    // Medium primes suitable for Rader's algorithm
    let primes = [151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199];
    for &p in &primes {
        test_fft_size(p);
    }
}

#[test]
fn test_mixed_radix_sizes() {
    // Sizes that require mixed-radix decomposition — expressible with radices
    // from {2, 3, 4, 5, 7, 8, 16}.
    let sizes = [
        // Original smooth-7 sizes
        45,   // 3² × 5
        75,   // 3 × 5²
        105,  // 3 × 5 × 7
        135,  // 3³ × 5
        225,  // 3² × 5²
        315,  // 3² × 5 × 7
        375,  // 3 × 5³
        525,  // 3 × 5² × 7
        675,  // 3³ × 5²
        945,  // 3³ × 5 × 7
        1125, // 3² × 5³
        // Additional sizes from the e2e requirements (not covered by other algorithms)
        6,   // 3 × 2
        10,  // 5 × 2
        14,  // 7 × 2
        18,  // 3² × 2 — also has Composite codelet but verifying correctness
        20,  // 5 × 4
        28,  // 7 × 4
        40,  // 5 × 8
        42,  // 7 × 3 × 2
        56,  // 7 × 8
        80,  // 5 × 16
        84,  // 7 × 3 × 4
        112, // 7 × 16
        120, // 5 × 3 × 8
        168, // 7 × 3 × 8
        240, // 5 × 3 × 16
        // Large smooth-7 sizes
        1680, // 5 × 7 × 48 = 5 × 7 × 3 × 16
        2520, // 7 × 5 × 8 × 9 = 7 × 5 × 8 × 3 × 3
    ];
    for &n in &sizes {
        test_fft_size(n);
    }
}

/// Test that `MixedRadix` algorithm is selected for smooth-7 composite sizes
/// not covered by other specialized algorithms (Composite codelets, `WinogradPfa`, etc.).
///
/// This test validates the dispatch chain:
///   Nop → `CooleyTukey` → Winograd → `WinogradPfa` → Composite → `MixedRadix` → Direct → Generic → Bluestein
///
/// Excluded from dispatch assertions (they have higher-priority algorithms):
///   - Powers of 2: `CooleyTukey`
///   - 3, 5, 7, 9, 11, 13: Winograd
///   - 15, 21, 35: `WinogradPfa`
///   - 12, 15, 18, 20, 24, 30, 36, 45, 48, 50, 60, 72, 80, 96, 100: Composite codelet
#[test]
fn test_mixed_radix_dispatch_assertion() {
    // Sizes that should route to MixedRadix — smooth-7 composites with
    // radices from {2,3,4,5,7,8,16}, not covered by Composite/Winograd/PFA/CT.
    // Composite codelet sizes (12,15,18,20,24,30,36,45,48,50,60,72,80,96,100) are excluded.
    let mixed_radix_sizes = [
        (6, "3×2"),
        (10, "5×2"),
        (14, "7×2"),
        (28, "7×4"),
        (40, "5×8"),
        (42, "7×3×2"),
        (56, "7×8"),
        (70, "5×7×2"),
        (75, "3×5²"),
        (84, "7×3×4"),
        (98, "7²×2"),
        (105, "3×5×7"),
        (112, "7×16"),
        (120, "5×3×8"),
        (168, "7×3×8"),
        (175, "5²×7"),
        (240, "5×3×16"),
        (336, "7×3×16"),
        (490, "5×7²×2"),
        (1260, "5×7×4×9"),
    ];

    for (n, desc) in &mixed_radix_sizes {
        let plan = Plan::<f64>::dft_1d(*n, Direction::Forward, Flags::ESTIMATE)
            .unwrap_or_else(|| panic!("plan creation failed for n={n} ({desc})"));
        assert_eq!(
            plan.algorithm_name(),
            "MixedRadix",
            "n={n} ({desc}): expected MixedRadix, got {}",
            plan.algorithm_name()
        );
    }
}

/// Correctness test: forward FFT followed by inverse FFT recovers original signal,
/// specifically for sizes handled by `MixedRadix`.
#[test]
fn test_mixed_radix_roundtrip_fwd_inv() {
    let sizes = [
        6, 10, 14, 20, 28, 40, 42, 45, 56, 70, 75, 80, 84, 98, 105, 112, 120, 168, 240,
    ];

    for &n in &sizes {
        test_roundtrip_size(n);
    }
}

// ============================================================================
// f32 MixedRadix correctness tests
// ============================================================================

/// Verify f32 Plan correctness for `MixedRadix` sizes.
///
/// The executor generates twiddles in f64, converts to f32 via `T::from_f64`, and applies
/// the butterfly.  This test ensures the resulting f32 output is close to the exact DFT
/// (computed at f64 precision) within an f32-appropriate tolerance.
///
/// Tolerance: 1e-4 relative error (f32 has ~7 significant decimal digits; each stage adds
/// rounding errors on the order of eps×log2(n) ≈ 1.2e-7×17 ≈ 2e-6, well below 1e-4).
#[test]
#[allow(clippy::cast_possible_truncation)] // reason: intentional f64→f32 for comparison tolerance
fn test_mixed_radix_f32_correctness() {
    // Sizes that route to MixedRadix and span several factorizations.
    // Include 1680 to confirm precision at a large smooth-7 size.
    let sizes: &[usize] = &[6, 10, 14, 28, 42, 56, 70, 105, 120, 168, 240, 1680];

    for &n in sizes {
        // Build f32 plan (ESTIMATE so no benchmarking overhead)
        let plan_f32 =
            Plan::<f32>::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap_or_else(|| {
                panic!("f32 plan failed for n={n}");
            });

        // Input: same deterministic signal as the f64 helpers but in f32.
        let input_f32: Vec<oxifft::Complex<f32>> = (0..n)
            .map(|i| {
                let t = 2.0_f32 * std::f32::consts::PI * (i as f32) / (n as f32);
                oxifft::Complex::new(t.cos(), t.sin())
            })
            .collect();

        let mut output_f32 = vec![oxifft::Complex::<f32>::new(0.0, 0.0); n];
        plan_f32.execute(&input_f32, &mut output_f32);

        // Reference: exact DFT at f64 precision, then downcast for comparison.
        let input_f64: Vec<Complex<f64>> = input_f32
            .iter()
            .map(|c| Complex::new(f64::from(c.re), f64::from(c.im)))
            .collect();
        let ref_f64 = dft_direct(&input_f64);

        // Relative error tolerance appropriate for f32 accumulation.
        // Scale loosely with log2(n) to account for multi-stage error growth.
        let tol = 1e-4_f32 * (n as f32).log2().max(1.0);

        for (k, (got, exp_f64)) in output_f32.iter().zip(ref_f64.iter()).enumerate() {
            let got_re_f64 = f64::from(got.re);
            let got_im_f64 = f64::from(got.im);
            let diff = ((got_re_f64 - exp_f64.re).hypot(got_im_f64 - exp_f64.im)) as f32;
            let mag = (exp_f64.re.hypot(exp_f64.im) as f32).max(1.0);
            let rel = diff / mag;
            assert!(
                rel <= tol,
                "f32 MixedRadix mismatch at n={n}, k={k}: got ({}, {}), ref ({:.6}, {:.6}), rel_err={rel:.2e}, tol={tol:.2e}",
                got.re,
                got.im,
                exp_f64.re,
                exp_f64.im,
            );
        }
    }
}

/// f32 forward/inverse roundtrip for `MixedRadix` sizes.
///
/// Verifies that `ifft(fft(x)) / n ≈ x` in f32 arithmetic.
/// Tolerance is relaxed to 1e-5 to accommodate f32 precision limits.
#[test]
fn test_mixed_radix_f32_roundtrip() {
    let sizes: &[usize] = &[6, 10, 14, 28, 42, 56, 70, 105, 120, 168, 240];

    for &n in sizes {
        let fwd_plan =
            Plan::<f32>::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap_or_else(|| {
                panic!("f32 fwd plan failed for n={n}");
            });
        let inv_plan =
            Plan::<f32>::dft_1d(n, Direction::Backward, Flags::ESTIMATE).unwrap_or_else(|| {
                panic!("f32 inv plan failed for n={n}");
            });

        let input: Vec<oxifft::Complex<f32>> = (0..n)
            .map(|i| {
                let t = 2.0_f32 * std::f32::consts::PI * (i as f32) / (n as f32);
                oxifft::Complex::new(t.cos(), t.sin())
            })
            .collect();

        let mut freq = vec![oxifft::Complex::<f32>::new(0.0, 0.0); n];
        let mut recovered = vec![oxifft::Complex::<f32>::new(0.0, 0.0); n];

        fwd_plan.execute(&input, &mut freq);
        inv_plan.execute(&freq, &mut recovered);

        let scale = 1.0_f32 / (n as f32);
        for c in &mut recovered {
            c.re *= scale;
            c.im *= scale;
        }

        let tol = 1e-5_f32 * (n as f32).log2().max(1.0);
        for i in 0..n {
            let diff = (recovered[i].re - input[i].re).hypot(recovered[i].im - input[i].im);
            assert!(
                diff <= tol,
                "f32 roundtrip mismatch at n={n}, i={i}: got ({}, {}), expected ({}, {}), diff={diff:.2e}, tol={tol:.2e}",
                recovered[i].re,
                recovered[i].im,
                input[i].re,
                input[i].im,
            );
        }
    }
}

// ============================================================================
// In-place tests at various sizes
// ============================================================================

#[test]
fn test_various_sizes_roundtrip() {
    use oxifft::{fft, ifft};

    let sizes = [4, 8, 16, 32, 64, 100, 128, 256, 500, 512, 1000, 1024];
    for &n in &sizes {
        // Use random-ish input that won't be an eigenvector of FFT
        let data: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let t = (i as f64).mul_add(0.123456789, (i * i) as f64 * 0.0001);
                Complex::new(t.sin() + 0.5, t.cos() * 0.7)
            })
            .collect();
        let original = data.clone();

        // Forward FFT (returns new vector)
        let transformed = fft(&data);

        // Inverse FFT (returns new vector)
        let recovered = ifft(&transformed);

        // Verify roundtrip
        for i in 0..n {
            let diff = (recovered[i].re - original[i].re).hypot(recovered[i].im - original[i].im);
            assert!(
                diff < 1e-10,
                "Roundtrip failed for size {} at index {}: got ({}, {}), expected ({}, {})",
                n,
                i,
                recovered[i].re,
                recovered[i].im,
                original[i].re,
                original[i].im
            );
        }
    }
}

// ============================================================================
// Stress tests (can be slow, use for validation)
// ============================================================================

#[test]
#[ignore = "slow stress test - run with --ignored for thorough testing"]
fn test_all_sizes_1_to_256() {
    for n in 1..=256 {
        test_fft_size(n);
    }
}

#[test]
#[ignore = "slow stress test - run with --ignored for thorough testing"]
fn test_all_sizes_1_to_256_roundtrip() {
    for n in 1..=256 {
        test_roundtrip_size(n);
    }
}

#[test]
#[ignore = "slow stress test - run with --ignored for thorough testing"]
fn test_power_of_2_up_to_2_20() {
    for exp in 1..=20 {
        let n = 1 << exp;
        test_roundtrip_size(n);
    }
}
