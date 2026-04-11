//! Tests for SIMD-optimized codelets.

use super::*;
use crate::dft::codelets::{notw_128, notw_16, notw_2, notw_256, notw_32, notw_4, notw_64, notw_8};

fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
    (a.re - b.re).abs() < eps && (a.im - b.im).abs() < eps
}

#[test]
fn test_simd_notw_2_matches_scalar() {
    let mut scalar = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
    let mut simd = scalar;

    notw_2(&mut scalar);
    notw_2_simd_f64(&mut simd);

    for (s, d) in scalar.iter().zip(simd.iter()) {
        assert!(
            complex_approx_eq(*s, *d, 1e-10),
            "Mismatch: scalar={s:?}, simd={d:?}"
        );
    }
}

#[test]
fn test_simd_notw_4_matches_scalar_forward() {
    let mut scalar = [
        Complex::new(1.0, 2.0),
        Complex::new(3.0, 4.0),
        Complex::new(5.0, 6.0),
        Complex::new(7.0, 8.0),
    ];
    let mut simd = scalar;

    notw_4(&mut scalar, -1);
    notw_4_simd_f64(&mut simd, -1);

    for (i, (s, d)) in scalar.iter().zip(simd.iter()).enumerate() {
        assert!(
            complex_approx_eq(*s, *d, 1e-10),
            "Index {i}: scalar={s:?}, simd={d:?}"
        );
    }
}

#[test]
fn test_simd_notw_4_matches_scalar_inverse() {
    let mut scalar = [
        Complex::new(1.0, 2.0),
        Complex::new(3.0, 4.0),
        Complex::new(5.0, 6.0),
        Complex::new(7.0, 8.0),
    ];
    let mut simd = scalar;

    notw_4(&mut scalar, 1);
    notw_4_simd_f64(&mut simd, 1);

    for (i, (s, d)) in scalar.iter().zip(simd.iter()).enumerate() {
        assert!(
            complex_approx_eq(*s, *d, 1e-10),
            "Index {i}: scalar={s:?}, simd={d:?}"
        );
    }
}

#[test]
fn test_simd_notw_8_matches_scalar_forward() {
    let mut scalar: Vec<Complex<f64>> = (0..8)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();
    let mut simd = scalar.clone();

    notw_8(&mut scalar, -1);
    notw_8_simd_f64(&mut simd, -1);

    for (i, (s, d)) in scalar.iter().zip(simd.iter()).enumerate() {
        assert!(
            complex_approx_eq(*s, *d, 1e-9),
            "Index {i}: scalar={s:?}, simd={d:?}"
        );
    }
}

#[test]
fn test_simd_notw_8_matches_scalar_inverse() {
    let mut scalar: Vec<Complex<f64>> = (0..8)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();
    let mut simd = scalar.clone();

    notw_8(&mut scalar, 1);
    notw_8_simd_f64(&mut simd, 1);

    for (i, (s, d)) in scalar.iter().zip(simd.iter()).enumerate() {
        assert!(
            complex_approx_eq(*s, *d, 1e-9),
            "Index {i}: scalar={s:?}, simd={d:?}"
        );
    }
}

#[test]
fn test_simd_notw_4_roundtrip() {
    let original: Vec<Complex<f64>> = (0..4)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();
    let mut data = original.clone();

    // Forward
    notw_4_simd_f64(&mut data, -1);
    // Inverse
    notw_4_simd_f64(&mut data, 1);
    // Normalize
    for x in &mut data {
        *x = *x / 4.0;
    }

    for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
        assert!(
            complex_approx_eq(*o, *d, 1e-10),
            "Index {i}: original={o:?}, recovered={d:?}"
        );
    }
}

#[test]
fn test_simd_notw_8_roundtrip() {
    let original: Vec<Complex<f64>> = (0..8)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();
    let mut data = original.clone();

    // Forward
    notw_8_simd_f64(&mut data, -1);
    // Inverse
    notw_8_simd_f64(&mut data, 1);
    // Normalize
    for x in &mut data {
        *x = *x / 8.0;
    }

    for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
        assert!(
            complex_approx_eq(*o, *d, 1e-9),
            "Index {i}: original={o:?}, recovered={d:?}"
        );
    }
}

#[test]
fn test_simd_notw_16_matches_scalar() {
    let mut scalar: Vec<Complex<f64>> = (0..16)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();
    let mut simd = scalar.clone();

    notw_16(&mut scalar, -1);
    notw_16_simd_f64(&mut simd, -1);

    for (i, (s, d)) in scalar.iter().zip(simd.iter()).enumerate() {
        assert!(
            complex_approx_eq(*s, *d, 1e-8),
            "Index {i}: scalar={s:?}, simd={d:?}"
        );
    }
}

#[test]
fn test_simd_notw_16_roundtrip() {
    let original: Vec<Complex<f64>> = (0..16)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();
    let mut data = original.clone();

    // Forward
    notw_16_simd_f64(&mut data, -1);
    // Inverse
    notw_16_simd_f64(&mut data, 1);
    // Normalize
    for x in &mut data {
        *x = *x / 16.0;
    }

    for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
        assert!(
            complex_approx_eq(*o, *d, 1e-8),
            "Index {i}: original={o:?}, recovered={d:?}"
        );
    }
}

#[test]
fn test_simd_notw_32_matches_scalar() {
    let mut scalar: Vec<Complex<f64>> = (0..32)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();
    let mut simd = scalar.clone();

    notw_32(&mut scalar, -1);
    notw_32_simd_f64(&mut simd, -1);

    for (i, (s, d)) in scalar.iter().zip(simd.iter()).enumerate() {
        assert!(
            complex_approx_eq(*s, *d, 1e-8),
            "Index {i}: scalar={s:?}, simd={d:?}"
        );
    }
}

#[test]
fn test_simd_notw_32_roundtrip() {
    let original: Vec<Complex<f64>> = (0..32)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();
    let mut data = original.clone();

    // Forward
    notw_32_simd_f64(&mut data, -1);
    // Inverse
    notw_32_simd_f64(&mut data, 1);
    // Normalize
    for x in &mut data {
        *x = *x / 32.0;
    }

    for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
        assert!(
            complex_approx_eq(*o, *d, 1e-8),
            "Index {i}: original={o:?}, recovered={d:?}"
        );
    }
}

#[test]
fn test_simd_notw_64_matches_scalar() {
    let mut scalar: Vec<Complex<f64>> = (0..64)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();
    let mut simd = scalar.clone();

    notw_64(&mut scalar, -1);
    notw_64_simd_f64(&mut simd, -1);

    for (i, (s, d)) in scalar.iter().zip(simd.iter()).enumerate() {
        assert!(
            complex_approx_eq(*s, *d, 1e-7),
            "Index {i}: scalar={s:?}, simd={d:?}"
        );
    }
}

#[test]
fn test_simd_notw_64_roundtrip() {
    let original: Vec<Complex<f64>> = (0..64)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();
    let mut data = original.clone();

    // Forward
    notw_64_simd_f64(&mut data, -1);
    // Inverse
    notw_64_simd_f64(&mut data, 1);
    // Normalize
    for x in &mut data {
        *x = *x / 64.0;
    }

    for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
        assert!(
            complex_approx_eq(*o, *d, 1e-8),
            "Index {i}: original={o:?}, recovered={d:?}"
        );
    }
}

#[test]
fn test_simd_notw_128_matches_scalar() {
    let mut scalar: Vec<Complex<f64>> = (0..128)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();
    let mut simd = scalar.clone();

    notw_128(&mut scalar, -1);
    notw_128_simd_f64(&mut simd, -1);

    for (i, (s, d)) in scalar.iter().zip(simd.iter()).enumerate() {
        assert!(
            complex_approx_eq(*s, *d, 1e-6),
            "Index {i}: scalar={s:?}, simd={d:?}"
        );
    }
}

#[test]
fn test_simd_notw_128_roundtrip() {
    let original: Vec<Complex<f64>> = (0..128)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();
    let mut data = original.clone();

    // Forward
    notw_128_simd_f64(&mut data, -1);
    // Inverse
    notw_128_simd_f64(&mut data, 1);
    // Normalize
    for x in &mut data {
        *x = *x / 128.0;
    }

    for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
        assert!(
            complex_approx_eq(*o, *d, 1e-8),
            "Index {i}: original={o:?}, recovered={d:?}"
        );
    }
}

#[test]
fn test_simd_notw_256_matches_scalar() {
    let mut scalar: Vec<Complex<f64>> = (0..256)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();
    let mut simd = scalar.clone();

    notw_256(&mut scalar, -1);
    notw_256_simd_f64(&mut simd, -1);

    for (i, (s, d)) in scalar.iter().zip(simd.iter()).enumerate() {
        assert!(
            complex_approx_eq(*s, *d, 1e-5),
            "Index {i}: scalar={s:?}, simd={d:?}"
        );
    }
}

#[test]
fn test_simd_notw_256_roundtrip() {
    let original: Vec<Complex<f64>> = (0..256)
        .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
        .collect();
    let mut data = original.clone();

    // Forward
    notw_256_simd_f64(&mut data, -1);
    // Inverse
    notw_256_simd_f64(&mut data, 1);
    // Normalize
    for x in &mut data {
        *x = *x / 256.0;
    }

    for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
        assert!(
            complex_approx_eq(*o, *d, 1e-8),
            "Index {i}: original={o:?}, recovered={d:?}"
        );
    }
}
