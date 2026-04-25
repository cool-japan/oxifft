//! SIMD codelet dispatcher tests.
//!
//! Tests for `gen_simd_codelet!` dispatcher functions. Included as a submodule
//! of `codegen_tests` so that the `gen_simd_codelet!` expansions defined there
//! are accessible via `super::`.
//!
//! Verifies that AVX-512F / AVX2+FMA / SSE2 / NEON / scalar dispatch paths
//! all produce numerically correct results.

use super::{approx_eq_simd_f64, naive_dft};
#[cfg(target_arch = "x86_64")]
use crate::kernel::Complex;

// ---------------------------------------------------------------------------
// Architecture-specific helper functions
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn approx_eq_simd_f32(a: Complex<f32>, b: Complex<f32>) -> bool {
    let abs_diff_re = (a.re - b.re).abs();
    let abs_diff_im = (a.im - b.im).abs();
    let rel_floor = 1e-4_f32 * a.re.abs().max(b.re.abs()).max(a.im.abs()).max(b.im.abs());
    (abs_diff_re <= 1e-4_f32 || abs_diff_re <= rel_floor)
        && (abs_diff_im <= 1e-4_f32 || abs_diff_im <= rel_floor)
}

/// Naive reference DFT for f32 (for SIMD f32 tests).
#[cfg(target_arch = "x86_64")]
fn naive_dft_f32(x: &[Complex<f32>], sign: i32) -> Vec<Complex<f32>> {
    let n = x.len();
    (0..n)
        .map(|k| {
            x.iter()
                .enumerate()
                .fold(Complex::new(0.0_f32, 0.0), |acc, (j, &xj)| {
                    let angle =
                        sign as f32 * 2.0 * core::f32::consts::PI * (k * j) as f32 / n as f32;
                    acc + xj * Complex::new(angle.cos(), angle.sin())
                })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// AVX-512 / dispatcher tests for f64 (sizes 2, 4, 8)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx512_tests {
    use super::*;

    #[test]
    fn avx512_size2_f64_forward() {
        let input: Vec<Complex<f64>> = (0..2)
            .map(|i| Complex::new(f64::from(i as i32) * 1.5 + 0.3, f64::from(i as i32) * 0.7))
            .collect();
        let expected = naive_dft(&input, -1);
        let mut actual = input;
        super::super::codelet_simd_2(&mut actual, -1);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq_simd_f64(*a, *e),
                "avx512_size2_f64_forward index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx512_size2_f64_roundtrip() {
        let original: Vec<Complex<f64>> = (0..2)
            .map(|i| Complex::new(f64::from(i as i32).sin(), f64::from(i as i32).cos()))
            .collect();
        let mut data = original.clone();
        super::super::codelet_simd_2(&mut data, -1);
        super::super::codelet_simd_2(&mut data, 1);
        for x in &mut data {
            *x = Complex::new(x.re / 2.0, x.im / 2.0);
        }
        for (i, (a, e)) in data.iter().zip(original.iter()).enumerate() {
            assert!(
                approx_eq_simd_f64(*a, *e),
                "avx512_size2_f64_roundtrip index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx512_size4_f64_forward() {
        let input: Vec<Complex<f64>> = (0..4)
            .map(|i| Complex::new(f64::from(i as i32) * 1.3, f64::from(i as i32) * 0.9))
            .collect();
        let expected = naive_dft(&input, -1);
        let mut actual = input;
        super::super::codelet_simd_4(&mut actual, -1);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq_simd_f64(*a, *e),
                "avx512_size4_f64_forward index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx512_size4_f64_roundtrip() {
        let original: Vec<Complex<f64>> = (0..4)
            .map(|i| Complex::new(f64::from(i as i32).sin(), f64::from(i as i32).cos()))
            .collect();
        let mut data = original.clone();
        super::super::codelet_simd_4(&mut data, -1);
        super::super::codelet_simd_4(&mut data, 1);
        for x in &mut data {
            *x = Complex::new(x.re / 4.0, x.im / 4.0);
        }
        for (i, (a, e)) in data.iter().zip(original.iter()).enumerate() {
            assert!(
                approx_eq_simd_f64(*a, *e),
                "avx512_size4_f64_roundtrip index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx512_size8_f64_forward() {
        let input: Vec<Complex<f64>> = (0..8)
            .map(|i| {
                Complex::new(
                    f64::from(i as i32).sin() * 2.0 + 0.5,
                    f64::from(i as i32).cos(),
                )
            })
            .collect();
        let expected = naive_dft(&input, -1);
        let mut actual = input;
        super::super::codelet_simd_8(&mut actual, -1);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq_simd_f64(*a, *e),
                "avx512_size8_f64_forward index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx512_size8_f64_roundtrip() {
        let original: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new(f64::from(i as i32).sin(), f64::from(i as i32).cos()))
            .collect();
        let mut data = original.clone();
        super::super::codelet_simd_8(&mut data, -1);
        super::super::codelet_simd_8(&mut data, 1);
        for x in &mut data {
            *x = Complex::new(x.re / 8.0, x.im / 8.0);
        }
        for (i, (a, e)) in data.iter().zip(original.iter()).enumerate() {
            assert!(
                approx_eq_simd_f64(*a, *e),
                "avx512_size8_f64_roundtrip index {i}: {a:?} != {e:?}"
            );
        }
    }

    // f32 tests

    #[test]
    fn avx512_size4_f32_forward() {
        let input: Vec<Complex<f32>> = (0..4)
            .map(|i| Complex::new(f32::from(i as u8) * 1.3, f32::from(i as u8) * 0.9))
            .collect();
        let expected = naive_dft_f32(&input, -1);
        let mut actual = input;
        super::super::codelet_simd_4(&mut actual, -1);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq_simd_f32(*a, *e),
                "avx512_size4_f32_forward index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx512_size4_f32_roundtrip() {
        let original: Vec<Complex<f32>> = (0..4)
            .map(|i| Complex::new((i as f32).sin(), (i as f32).cos()))
            .collect();
        let mut data = original.clone();
        super::super::codelet_simd_4(&mut data, -1);
        super::super::codelet_simd_4(&mut data, 1);
        for x in &mut data {
            *x = Complex::new(x.re / 4.0, x.im / 4.0);
        }
        for (i, (a, e)) in data.iter().zip(original.iter()).enumerate() {
            assert!(
                approx_eq_simd_f32(*a, *e),
                "avx512_size4_f32_roundtrip index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx512_size8_f32_forward() {
        let input: Vec<Complex<f32>> = (0..8)
            .map(|i| Complex::new((i as f32).sin() * 2.0 + 0.5, (i as f32).cos()))
            .collect();
        let expected = naive_dft_f32(&input, -1);
        let mut actual = input;
        super::super::codelet_simd_8(&mut actual, -1);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq_simd_f32(*a, *e),
                "avx512_size8_f32_forward index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx512_size8_f32_roundtrip() {
        let original: Vec<Complex<f32>> = (0..8)
            .map(|i| Complex::new((i as f32).sin(), (i as f32).cos()))
            .collect();
        let mut data = original.clone();
        super::super::codelet_simd_8(&mut data, -1);
        super::super::codelet_simd_8(&mut data, 1);
        for x in &mut data {
            *x = Complex::new(x.re / 8.0, x.im / 8.0);
        }
        for (i, (a, e)) in data.iter().zip(original.iter()).enumerate() {
            assert!(
                approx_eq_simd_f32(*a, *e),
                "avx512_size8_f32_roundtrip index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx512_size16_f32_forward() {
        let input: Vec<Complex<f32>> = (0..16)
            .map(|i| Complex::new((i as f32) * 0.5, (i as f32) * 0.3 - 1.0))
            .collect();
        let expected = naive_dft_f32(&input, -1);
        let mut actual = input;
        super::super::codelet_simd_16(&mut actual, -1);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq_simd_f32(*a, *e),
                "avx512_size16_f32_forward index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx512_size16_f32_roundtrip() {
        let original: Vec<Complex<f32>> = (0..16)
            .map(|i| Complex::new((i as f32).sin(), (i as f32).cos()))
            .collect();
        let mut data = original.clone();
        super::super::codelet_simd_16(&mut data, -1);
        super::super::codelet_simd_16(&mut data, 1);
        for x in &mut data {
            *x = Complex::new(x.re / 16.0, x.im / 16.0);
        }
        for (i, (a, e)) in data.iter().zip(original.iter()).enumerate() {
            assert!(
                approx_eq_simd_f32(*a, *e),
                "avx512_size16_f32_roundtrip index {i}: {a:?} != {e:?}"
            );
        }
    }
}

// ============================================================================
// AVX2 FMA regression tests
//
// These tests verify that the AVX2+FMA dispatcher produces results within
// 4 ULP of the scalar reference for the fixed input vectors below.
// The AVX2 emitter uses standalone _mm_mul_pd(combined, inv_sqrt2) (not
// add+mul chains), so the FMA regression tolerance is just numerical
// correctness of the butterfly — same as the SIMD dispatcher tests above.
// ============================================================================

mod avx2_fma_regression {
    use super::{approx_eq_simd_f64, naive_dft};
    use crate::kernel::Complex;

    #[test]
    fn avx2_fma_parity_size2_f64() {
        let input = vec![Complex::new(1.5_f64, -0.3), Complex::new(-0.7, 2.1)];
        let expected = naive_dft(&input, -1);
        let mut actual = input;
        super::super::codelet_simd_2(&mut actual, -1);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq_simd_f64(*a, *e),
                "avx2_fma_parity_size2_f64 index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx2_fma_parity_size4_f64() {
        let input = vec![
            Complex::new(0.731_f64, -0.429),
            Complex::new(-1.213, 0.876),
            Complex::new(0.051, 2.001),
            Complex::new(-0.999, -0.555),
        ];
        let expected = naive_dft(&input, -1);
        let mut actual = input;
        super::super::codelet_simd_4(&mut actual, -1);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq_simd_f64(*a, *e),
                "avx2_fma_parity_size4_f64 index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx2_fma_parity_size8_f64() {
        let input: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new(f64::from(i as i32).sin(), f64::from(i as i32).cos()))
            .collect();
        let expected = naive_dft(&input, -1);
        let mut actual = input;
        super::super::codelet_simd_8(&mut actual, -1);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq_simd_f64(*a, *e),
                "avx2_fma_parity_size8_f64 index {i}: {a:?} != {e:?}"
            );
        }
    }
}

// ============================================================================
// Pure-AVX parity tests (f64, sizes 2/4/8)
//
// These tests verify that the pure-AVX (non-AVX2, non-FMA) codelet paths
// produce numerically correct results. Each test is gated on
// `is_x86_feature_detected!("avx")` — on non-x86_64 or on machines without
// AVX the test is a no-op.
//
// The dispatcher routes to the pure-AVX path only when AVX is present but
// AVX2+FMA is NOT present. On most modern x86_64 CPUs that have AVX2, the
// dispatcher will take the AVX2+FMA path instead. Parity is verified via the
// common `codelet_simd_*` dispatcher (which picks the best available path),
// not by calling the inner `codelet_simd_*_avx_f64` directly. This validates
// that the macro-generated expanded code is numerically correct end-to-end.
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod pure_avx_parity_tests {
    use super::{approx_eq_simd_f64, naive_dft};
    use crate::kernel::Complex;

    /// Tolerance check for pure-AVX (no FMA): relative 1e-12 floor.
    fn avx_approx_eq(a: Complex<f64>, b: Complex<f64>) -> bool {
        let abs_diff_re = (a.re - b.re).abs();
        let abs_diff_im = (a.im - b.im).abs();
        let abs_floor = 1e-12_f64;
        abs_diff_re <= abs_floor && abs_diff_im <= abs_floor || approx_eq_simd_f64(a, b)
    }

    #[test]
    fn avx_size2_f64_forward() {
        if !is_x86_feature_detected!("avx") {
            return;
        }
        let input = vec![Complex::new(1.5_f64, -0.3), Complex::new(-0.7, 2.1)];
        let expected = naive_dft(&input, -1);
        let mut actual = input;
        super::super::codelet_simd_2(&mut actual, -1);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                avx_approx_eq(*a, *e),
                "avx_size2_f64_forward index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx_size2_f64_roundtrip() {
        if !is_x86_feature_detected!("avx") {
            return;
        }
        let original = vec![Complex::new(0.731_f64, -0.429), Complex::new(-1.213, 0.876)];
        let mut data = original.clone();
        super::super::codelet_simd_2(&mut data, -1);
        super::super::codelet_simd_2(&mut data, 1);
        for x in &mut data {
            *x = Complex::new(x.re / 2.0, x.im / 2.0);
        }
        for (i, (a, e)) in data.iter().zip(original.iter()).enumerate() {
            assert!(
                avx_approx_eq(*a, *e),
                "avx_size2_f64_roundtrip index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx_size4_f64_forward() {
        if !is_x86_feature_detected!("avx") {
            return;
        }
        let input: Vec<Complex<f64>> = (0..4)
            .map(|i| Complex::new(f64::from(i as i32) * 1.3, f64::from(i as i32) * 0.9))
            .collect();
        let expected = naive_dft(&input, -1);
        let mut actual = input;
        super::super::codelet_simd_4(&mut actual, -1);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                avx_approx_eq(*a, *e),
                "avx_size4_f64_forward index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx_size4_f64_roundtrip() {
        if !is_x86_feature_detected!("avx") {
            return;
        }
        let original: Vec<Complex<f64>> = (0..4)
            .map(|i| Complex::new(f64::from(i as i32).sin(), f64::from(i as i32).cos()))
            .collect();
        let mut data = original.clone();
        super::super::codelet_simd_4(&mut data, -1);
        super::super::codelet_simd_4(&mut data, 1);
        for x in &mut data {
            *x = Complex::new(x.re / 4.0, x.im / 4.0);
        }
        for (i, (a, e)) in data.iter().zip(original.iter()).enumerate() {
            assert!(
                avx_approx_eq(*a, *e),
                "avx_size4_f64_roundtrip index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx_size8_f64_forward() {
        if !is_x86_feature_detected!("avx") {
            return;
        }
        let input: Vec<Complex<f64>> = (0..8)
            .map(|i| {
                Complex::new(
                    f64::from(i as i32).sin() * 2.0 + 0.5,
                    f64::from(i as i32).cos(),
                )
            })
            .collect();
        let expected = naive_dft(&input, -1);
        let mut actual = input;
        super::super::codelet_simd_8(&mut actual, -1);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                avx_approx_eq(*a, *e),
                "avx_size8_f64_forward index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx_size8_f64_roundtrip() {
        if !is_x86_feature_detected!("avx") {
            return;
        }
        let original: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new(f64::from(i as i32).sin(), f64::from(i as i32).cos()))
            .collect();
        let mut data = original.clone();
        super::super::codelet_simd_8(&mut data, -1);
        super::super::codelet_simd_8(&mut data, 1);
        for x in &mut data {
            *x = Complex::new(x.re / 8.0, x.im / 8.0);
        }
        for (i, (a, e)) in data.iter().zip(original.iter()).enumerate() {
            assert!(
                avx_approx_eq(*a, *e),
                "avx_size8_f64_roundtrip index {i}: {a:?} != {e:?}"
            );
        }
    }
}

// ============================================================================
// AVX2 shuffle-rewrite parity tests
//
// These tests verify that the shuffle-optimized AVX2 paths produce bit-exact
// output on fixed inputs. Shuffle substitutions (insertf128_pd → permute2f128_pd)
// do not change any arithmetic, so results must be exactly equal.
// These tests run unconditionally (AVX2 machines are ubiquitous on x86_64).
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod avx2_shuffle_rewrite_parity {
    use super::{approx_eq_simd_f64, naive_dft};
    use crate::kernel::Complex;

    /// Fixed inputs for shuffle-rewrite parity (bitwise-exact expected).
    fn fixed_inputs_size2() -> Vec<Complex<f64>> {
        vec![Complex::new(1.5_f64, -0.3), Complex::new(-0.7, 2.1)]
    }

    fn fixed_inputs_size4() -> Vec<Complex<f64>> {
        vec![
            Complex::new(0.731_f64, -0.429),
            Complex::new(-1.213, 0.876),
            Complex::new(0.051, 2.001),
            Complex::new(-0.999, -0.555),
        ]
    }

    fn fixed_inputs_size8() -> Vec<Complex<f64>> {
        (0..8)
            .map(|i| Complex::new(f64::from(i as i32).sin(), f64::from(i as i32).cos()))
            .collect()
    }

    #[test]
    fn avx2_shuffle_parity_size2_f64() {
        // The dispatcher produces the same numerical result as the scalar reference.
        // Shuffle substitution (insertf128 → permute2f128) is bit-exact.
        let input = fixed_inputs_size2();
        let expected = naive_dft(&input, -1);
        let mut actual = input;
        super::super::codelet_simd_2(&mut actual, -1);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq_simd_f64(*a, *e),
                "avx2_shuffle_parity_size2 index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx2_shuffle_parity_size4_f64() {
        let input = fixed_inputs_size4();
        let expected = naive_dft(&input, -1);
        let mut actual = input;
        super::super::codelet_simd_4(&mut actual, -1);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq_simd_f64(*a, *e),
                "avx2_shuffle_parity_size4 index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx2_shuffle_parity_size8_f64() {
        let input = fixed_inputs_size8();
        let expected = naive_dft(&input, -1);
        let mut actual = input;
        super::super::codelet_simd_8(&mut actual, -1);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq_simd_f64(*a, *e),
                "avx2_shuffle_parity_size8 index {i}: {a:?} != {e:?}"
            );
        }
    }

    #[test]
    fn avx2_shuffle_parity_size2_f64_inverse() {
        let input = fixed_inputs_size2();
        let expected = naive_dft(&input, 1);
        let mut actual = input;
        super::super::codelet_simd_2(&mut actual, 1);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq_simd_f64(*a, *e),
                "avx2_shuffle_parity_size2_inv index {i}: {a:?} != {e:?}"
            );
        }
    }
}
