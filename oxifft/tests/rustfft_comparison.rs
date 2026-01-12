//! Correctness tests comparing `OxiFFT` against rustfft.
//!
//! These tests verify that `OxiFFT` produces results matching the well-tested rustfft library
//! within floating-point tolerance.

#![allow(clippy::cast_precision_loss)] // FFT size computations use float for math
#![allow(clippy::needless_pass_by_value)] // test convenience
#![allow(clippy::similar_names)] // fwd/bwd, real/imag pairs

use oxifft::api::{fft, fft_batch, ifft};
use oxifft::Complex;
use rustfft::FftPlanner;

/// Maximum allowed relative error for comparison.
const TOLERANCE: f64 = 1e-10;

/// Convert from oxifft Complex to `num_complex::Complex` for rustfft.
const fn to_num_complex(c: Complex<f64>) -> num_complex::Complex<f64> {
    num_complex::Complex::new(c.re, c.im)
}

/// Convert from `num_complex::Complex` to oxifft Complex.
const fn from_num_complex(c: num_complex::Complex<f64>) -> Complex<f64> {
    Complex::new(c.re, c.im)
}

/// Compare two complex numbers within tolerance.
fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
    let diff = (a.re - b.re).hypot(a.im - b.im);
    let mag_a = a.re.hypot(a.im);
    let mag_b = b.re.hypot(b.im);
    let mag = mag_a.max(mag_b).max(1.0);
    diff / mag < eps
}

/// Compute FFT using rustfft for reference.
fn rustfft_forward(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(input.len());
    let mut buffer: Vec<num_complex::Complex<f64>> =
        input.iter().map(|c| to_num_complex(*c)).collect();
    fft.process(&mut buffer);
    buffer.iter().map(|c| from_num_complex(*c)).collect()
}

/// Compute inverse FFT using rustfft for reference.
fn rustfft_inverse(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(input.len());
    let mut buffer: Vec<num_complex::Complex<f64>> =
        input.iter().map(|c| to_num_complex(*c)).collect();
    fft.process(&mut buffer);
    // rustfft doesn't normalize, so we do it here
    let scale = 1.0 / input.len() as f64;
    buffer
        .iter()
        .map(|c| {
            let scaled = num_complex::Complex::new(c.re * scale, c.im * scale);
            from_num_complex(scaled)
        })
        .collect()
}

// =============================================================================
// Power-of-2 sizes
// =============================================================================

#[test]
fn test_fft_matches_rustfft_size_2() {
    let input: Vec<Complex<f64>> = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 2, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_size_4() {
    let input: Vec<Complex<f64>> = (0..4)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 4, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_size_8() {
    let input: Vec<Complex<f64>> = (0..8)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 8, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_size_16() {
    let input: Vec<Complex<f64>> = (0..16)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 16, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_size_32() {
    let input: Vec<Complex<f64>> = (0..32)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 32, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_size_64() {
    let input: Vec<Complex<f64>> = (0..64)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 64, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_size_128() {
    let input: Vec<Complex<f64>> = (0..128)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 128, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_size_256() {
    let input: Vec<Complex<f64>> = (0..256)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 256, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_size_512() {
    let input: Vec<Complex<f64>> = (0..512)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 512, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_size_1024() {
    let input: Vec<Complex<f64>> = (0..1024)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 1024, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

// =============================================================================
// Prime sizes (uses Rader's algorithm)
// =============================================================================

#[test]
fn test_fft_matches_rustfft_prime_3() {
    let input: Vec<Complex<f64>> = (0..3)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Prime 3, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_prime_5() {
    let input: Vec<Complex<f64>> = (0..5)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Prime 5, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_prime_7() {
    let input: Vec<Complex<f64>> = (0..7)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Prime 7, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_prime_11() {
    let input: Vec<Complex<f64>> = (0..11)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Prime 11, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_prime_13() {
    let input: Vec<Complex<f64>> = (0..13)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Prime 13, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_prime_17() {
    let input: Vec<Complex<f64>> = (0..17)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Prime 17, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_prime_97() {
    let input: Vec<Complex<f64>> = (0..97)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, 1e-9), // Slightly relaxed for larger primes
            "Prime 97, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

// =============================================================================
// Composite non-power-of-2 sizes (uses mixed-radix)
// =============================================================================

#[test]
fn test_fft_matches_rustfft_size_6() {
    let input: Vec<Complex<f64>> = (0..6)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 6, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_size_12() {
    let input: Vec<Complex<f64>> = (0..12)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 12, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_size_15() {
    let input: Vec<Complex<f64>> = (0..15)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 15, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_size_100() {
    let input: Vec<Complex<f64>> = (0..100)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, 1e-9),
            "Size 100, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

// =============================================================================
// Inverse FFT tests
// =============================================================================

#[test]
fn test_ifft_matches_rustfft_size_16() {
    let input: Vec<Complex<f64>> = (0..16)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = ifft(&input);
    let rustfft_result = rustfft_inverse(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "IFFT size 16, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_ifft_matches_rustfft_size_64() {
    let input: Vec<Complex<f64>> = (0..64)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = ifft(&input);
    let rustfft_result = rustfft_inverse(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "IFFT size 64, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

// =============================================================================
// Batch transform tests
// =============================================================================

#[test]
fn test_batch_fft_matches_rustfft() {
    let n = 16;
    let howmany = 4;
    let input: Vec<Complex<f64>> = (0..(n * howmany))
        .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
        .collect();

    let batch_result = fft_batch(&input, n, howmany);

    // Compare each batch against rustfft
    for batch_idx in 0..howmany {
        let start = batch_idx * n;
        let single_input: Vec<Complex<f64>> = input[start..start + n].to_vec();
        let rustfft_result = rustfft_forward(&single_input);

        for i in 0..n {
            assert!(
                complex_approx_eq(batch_result[start + i], rustfft_result[i], TOLERANCE),
                "Batch {} index {}: oxifft={:?}, rustfft={:?}",
                batch_idx,
                i,
                batch_result[start + i],
                rustfft_result[i]
            );
        }
    }
}

// =============================================================================
// Large size tests
// =============================================================================

#[test]
fn test_fft_matches_rustfft_size_4096() {
    let input: Vec<Complex<f64>> = (0..4096)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, 1e-8), // Relaxed for larger sizes
            "Size 4096, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}

#[test]
fn test_fft_matches_rustfft_size_8192() {
    let input: Vec<Complex<f64>> = (0..8192)
        .map(|i| Complex::new((f64::from(i) * 0.01).sin(), (f64::from(i) * 0.01).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let rustfft_result = rustfft_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(rustfft_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, 1e-7),
            "Size 8192, index {i}: oxifft={a:?}, rustfft={b:?}"
        );
    }
}
