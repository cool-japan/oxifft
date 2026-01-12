//! Twiddle-factor codelets.
//!
//! These codelets apply twiddle factors as part of the FFT recursion.
//! Includes lazily-initialized twiddle tables for common sizes.

use crate::kernel::{Complex, Float};
use crate::prelude::OnceLock;
#[cfg(not(feature = "std"))]
use crate::prelude::OnceLockExt;

// ============================================================================
// Lazily-initialized twiddle tables for f64
// ============================================================================
// W_N^k = e^(-2πik/N) for forward FFT
// These are for the combining stage: we need W_N^k for k = 0..N/2

static TWIDDLES_64_FWD: OnceLock<[(f64, f64); 32]> = OnceLock::new();
static TWIDDLES_128_FWD: OnceLock<[(f64, f64); 64]> = OnceLock::new();
static TWIDDLES_256_FWD: OnceLock<[(f64, f64); 128]> = OnceLock::new();

/// Get precomputed twiddles for size-64 FFT
#[inline]
pub fn get_twiddles_64() -> &'static [(f64, f64); 32] {
    TWIDDLES_64_FWD.get_or_init(|| {
        let mut arr = [(0.0, 0.0); 32];
        for k in 0..32 {
            let angle = -2.0 * core::f64::consts::PI * (k as f64) / 64.0;
            arr[k] = angle.sin_cos();
            // sin_cos returns (sin, cos), we want (cos, sin)
            arr[k] = (arr[k].1, arr[k].0);
        }
        arr
    })
}

/// Get precomputed twiddles for size-128 FFT
#[inline]
pub fn get_twiddles_128() -> &'static [(f64, f64); 64] {
    TWIDDLES_128_FWD.get_or_init(|| {
        let mut arr = [(0.0, 0.0); 64];
        for k in 0..64 {
            let angle = -2.0 * core::f64::consts::PI * (k as f64) / 128.0;
            arr[k] = angle.sin_cos();
            arr[k] = (arr[k].1, arr[k].0);
        }
        arr
    })
}

/// Get precomputed twiddles for size-256 FFT
#[inline]
pub fn get_twiddles_256() -> &'static [(f64, f64); 128] {
    TWIDDLES_256_FWD.get_or_init(|| {
        let mut arr = [(0.0, 0.0); 128];
        for k in 0..128 {
            let angle = -2.0 * core::f64::consts::PI * (k as f64) / 256.0;
            arr[k] = angle.sin_cos();
            arr[k] = (arr[k].1, arr[k].0);
        }
        arr
    })
}

/// Apply twiddle factors to a sub-array.
#[inline]
pub fn apply_twiddles<T: Float>(x: &mut [Complex<T>], twiddles: &[Complex<T>], stride: usize) {
    for (i, tw) in twiddles.iter().enumerate() {
        let idx = i * stride;
        if idx < x.len() {
            x[idx] = x[idx] * *tw;
        }
    }
}

/// Radix-2 butterfly with twiddle.
#[inline]
pub fn butterfly_2<T: Float>(
    a: Complex<T>,
    b: Complex<T>,
    tw: Complex<T>,
) -> (Complex<T>, Complex<T>) {
    let btw = b * tw;
    (a + btw, a - btw)
}

/// Radix-4 butterfly with twiddles.
#[inline]
pub fn butterfly_4<T: Float>(
    x0: Complex<T>,
    x1: Complex<T>,
    x2: Complex<T>,
    x3: Complex<T>,
    tw1: Complex<T>,
    tw2: Complex<T>,
    tw3: Complex<T>,
    sign: i32,
) -> (Complex<T>, Complex<T>, Complex<T>, Complex<T>) {
    let x1tw = x1 * tw1;
    let x2tw = x2 * tw2;
    let x3tw = x3 * tw3;

    let t0 = x0 + x2tw;
    let t1 = x0 - x2tw;
    let t2 = x1tw + x3tw;
    let t3 = x1tw - x3tw;

    let t3_rot = if sign < 0 {
        Complex::new(t3.im, -t3.re)
    } else {
        Complex::new(-t3.im, t3.re)
    };

    (t0 + t2, t1 + t3_rot, t0 - t2, t1 - t3_rot)
}
