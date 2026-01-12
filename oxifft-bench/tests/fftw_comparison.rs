//! Correctness tests comparing `OxiFFT` against FFTW.
//!
//! These tests verify that `OxiFFT` produces results matching the gold-standard FFTW library
//! within floating-point tolerance.
//!
//! Requires the `fftw-compare` feature and libfftw3 installed on the system.

#![allow(clippy::uninlined_format_args)] // Allow traditional format args style in tests
#![allow(clippy::cast_lossless)] // Allow `as` casts for clarity in tests

use oxifft::api::{fft, fft_batch, ifft};
use oxifft::Complex;
use oxifft_bench::fftw_utils::{fftw_forward, fftw_inverse};
use oxifft_bench::utils::{complex_approx_eq, generate_input};

/// Maximum allowed relative error for comparison.
const TOLERANCE: f64 = 1e-10;

// =============================================================================
// Power-of-2 sizes
// =============================================================================

#[test]
fn test_fft_matches_fftw_size_2() {
    let input: Vec<Complex<f64>> = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];

    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 2, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_size_4() {
    let input = generate_input(4);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 4, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_size_8() {
    let input = generate_input(8);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 8, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_size_16() {
    let input = generate_input(16);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 16, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_size_32() {
    let input = generate_input(32);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 32, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_size_64() {
    let input = generate_input(64);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 64, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_size_128() {
    let input = generate_input(128);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 128, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_size_256() {
    let input = generate_input(256);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 256, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_size_512() {
    let input = generate_input(512);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 512, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_size_1024() {
    let input = generate_input(1024);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 1024, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

// =============================================================================
// Prime sizes (uses Rader's algorithm)
// =============================================================================

#[test]
fn test_fft_matches_fftw_prime_3() {
    let input = generate_input(3);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Prime 3, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_prime_5() {
    let input = generate_input(5);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Prime 5, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_prime_7() {
    let input = generate_input(7);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Prime 7, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_prime_11() {
    let input = generate_input(11);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Prime 11, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_prime_13() {
    let input = generate_input(13);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Prime 13, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_prime_17() {
    let input = generate_input(17);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Prime 17, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_prime_97() {
    let input = generate_input(97);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, 1e-9), // Slightly relaxed for larger primes
            "Prime 97, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

// =============================================================================
// Composite non-power-of-2 sizes (uses mixed-radix)
// =============================================================================

#[test]
fn test_fft_matches_fftw_size_6() {
    let input = generate_input(6);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 6, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_size_12() {
    let input = generate_input(12);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 12, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_size_15() {
    let input = generate_input(15);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "Size 15, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_size_100() {
    let input = generate_input(100);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, 1e-9),
            "Size 100, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

// =============================================================================
// Inverse FFT tests
// =============================================================================

#[test]
fn test_ifft_matches_fftw_size_16() {
    let input = generate_input(16);
    let oxifft_result = ifft(&input);
    let fftw_result = fftw_inverse(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "IFFT size 16, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_ifft_matches_fftw_size_64() {
    let input = generate_input(64);
    let oxifft_result = ifft(&input);
    let fftw_result = fftw_inverse(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, TOLERANCE),
            "IFFT size 64, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

// =============================================================================
// Batch transform tests
// =============================================================================

#[test]
fn test_batch_fft_matches_fftw() {
    let n = 16;
    let howmany = 4;
    let input = generate_input(n * howmany);

    let batch_result = fft_batch(&input, n, howmany);

    // Compare each batch against fftw
    for batch_idx in 0..howmany {
        let start = batch_idx * n;
        let single_input: Vec<Complex<f64>> = input[start..start + n].to_vec();
        let fftw_result = fftw_forward(&single_input);

        for i in 0..n {
            assert!(
                complex_approx_eq(batch_result[start + i], fftw_result[i], TOLERANCE),
                "Batch {} index {}: oxifft={:?}, fftw={:?}",
                batch_idx,
                i,
                batch_result[start + i],
                fftw_result[i]
            );
        }
    }
}

// =============================================================================
// Large size tests
// =============================================================================

#[test]
fn test_fft_matches_fftw_size_4096() {
    let input = generate_input(4096);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, 1e-8), // Relaxed for larger sizes
            "Size 4096, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_size_8192() {
    let input: Vec<Complex<f64>> = (0..8192)
        .map(|i| Complex::new((i as f64 * 0.01).sin(), (i as f64 * 0.01).cos()))
        .collect();

    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, 1e-7),
            "Size 8192, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

// =============================================================================
// Very large sizes (matching FFTW within tolerance)
// =============================================================================

#[test]
fn test_fft_matches_fftw_size_16384() {
    let input = generate_input(16384);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, 1e-7),
            "Size 16384, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fft_matches_fftw_size_65536() {
    let input = generate_input(65536);
    let oxifft_result = fft(&input);
    let fftw_result = fftw_forward(&input);

    for (i, (a, b)) in oxifft_result.iter().zip(fftw_result.iter()).enumerate() {
        assert!(
            complex_approx_eq(*a, *b, 1e-6), // More relaxed for very large sizes
            "Size 65536, index {}: oxifft={:?}, fftw={:?}",
            i,
            a,
            b
        );
    }
}
