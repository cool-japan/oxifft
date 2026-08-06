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

// ── Regression: fixed-size dispatchers must reject wrong-length slices ──────
//
// `notw_16/32/64_dispatch` are `pub` and reachable from outside the crate
// (`oxifft::dft::codelets::simd::…` / the `dft::codelets` re-exports).  On
// x86_64 with the `avx512` feature they forward to
// `hand_avx512::dispatch_hand_avx512_size*`, which calls a raw-pointer
// codelet that unconditionally touches N `Complex<T>` elements.  Passing a
// shorter slice from entirely safe code used to read and write past the end
// of the caller's allocation; it must now be a deterministic panic.

#[test]
#[should_panic(expected = "notw_16_dispatch requires exactly 16 elements")]
fn notw_16_dispatch_rejects_short_slice() {
    let mut data = vec![Complex::<f64>::zero(); 2];
    notw_16_dispatch(&mut data, -1);
}

#[test]
#[should_panic(expected = "notw_32_dispatch requires exactly 32 elements")]
fn notw_32_dispatch_rejects_short_slice() {
    let mut data = vec![Complex::<f64>::zero(); 2];
    notw_32_dispatch(&mut data, -1);
}

#[test]
#[should_panic(expected = "notw_64_dispatch requires exactly 64 elements")]
fn notw_64_dispatch_rejects_long_slice() {
    let mut data = vec![Complex::<f32>::zero(); 65];
    notw_64_dispatch(&mut data, -1);
}

/// Direct check of the AVX-512 entry points themselves; only compiled where
/// they exist.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
#[should_panic(expected = "requires exactly 16 elements")]
fn hand_avx512_dispatch_rejects_short_slice() {
    let mut data = vec![Complex::<f64>::zero(); 2];
    crate::dft::codelets::hand_avx512::dispatch_hand_avx512_size16_f64(&mut data, -1);
}
