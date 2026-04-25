//! Compile-time twiddle factor computation.
//!
//! Uses Taylor series approximations for sin and cos that can be computed
//! at compile time (in const context).

#![allow(clippy::unreadable_literal)] // reason: Taylor series factorial denominators (87178291200, 6227020800) are unreadable by design

use core::f64::consts::PI;

/// Compute cosine using Taylor series.
///
/// cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
///
/// Accurate for |x| < 2π.
#[inline]
pub const fn const_cos(x: f64) -> f64 {
    // Reduce to [-π, π] for better accuracy
    let x = reduce_angle(x);

    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x4 * x4;
    let x10 = x6 * x4;
    let x12 = x6 * x6;
    let x14 = x8 * x6;

    // Taylor series coefficients: 1/n!
    // 2! = 2, 4! = 24, 6! = 720, 8! = 40320, 10! = 3628800, 12! = 479001600, 14! = 87178291200
    1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0 + x8 / 40320.0 - x10 / 3628800.0 + x12 / 479001600.0
        - x14 / 87178291200.0
}

/// Compute sine using Taylor series.
///
/// sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
///
/// Accurate for |x| < 2π.
#[inline]
pub const fn const_sin(x: f64) -> f64 {
    // Reduce to [-π, π] for better accuracy
    let x = reduce_angle(x);

    let x2 = x * x;
    let x3 = x * x2;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    let x9 = x7 * x2;
    let x11 = x9 * x2;
    let x13 = x11 * x2;

    // Taylor series coefficients: 1/n!
    // 3! = 6, 5! = 120, 7! = 5040, 9! = 362880, 11! = 39916800, 13! = 6227020800
    x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0 + x9 / 362880.0 - x11 / 39916800.0 + x13 / 6227020800.0
}

/// Reduce angle to [-π, π] range for better Taylor series accuracy.
#[inline]
const fn reduce_angle(x: f64) -> f64 {
    // Handle common cases without full reduction
    if x >= -PI && x <= PI {
        return x;
    }

    // For angles within [-2π, 2π], simple reduction
    let two_pi = 2.0 * PI;
    let mut angle = x;

    // Reduce to [-2π, 2π]
    if angle > two_pi {
        let n = (angle / two_pi) as i64;
        angle = angle - (n as f64) * two_pi;
    } else if angle < -two_pi {
        let n = (-angle / two_pi) as i64;
        angle = angle + (n as f64) * two_pi;
    }

    // Reduce to [-π, π]
    if angle > PI {
        angle = angle - two_pi;
    } else if angle < -PI {
        angle = angle + two_pi;
    }

    angle
}

/// Compute twiddle factor W_N^k = e^(-2πik/N) = cos(2πk/N) - i*sin(2πk/N).
///
/// Returns (cos, sin) tuple.
#[inline]
pub const fn twiddle_factor(k: usize, n: usize) -> (f64, f64) {
    let angle = -2.0 * PI * (k as f64) / (n as f64);
    (const_cos(angle), const_sin(angle))
}

/// Compute twiddle factor for inverse FFT: W_N^{-k} = e^(2πik/N).
///
/// Returns (cos, sin) tuple.
#[inline]
#[allow(dead_code)] // reason: public API for inverse twiddle factors, used by const-FFT callers
pub const fn twiddle_factor_inv(k: usize, n: usize) -> (f64, f64) {
    let angle = 2.0 * PI * (k as f64) / (n as f64);
    (const_cos(angle), const_sin(angle))
}

/// Precomputed twiddle factors for size N.
#[allow(dead_code)] // reason: public API type for const-FFT users; not all fields used internally
pub struct TwiddleTable<const N: usize> {
    /// Cosine values: cos(-2πk/N) for k = 0..N/2
    pub cos: [f64; N],
    /// Sine values: sin(-2πk/N) for k = 0..N/2
    pub sin: [f64; N],
}

#[allow(dead_code)] // reason: impl block for public TwiddleTable; methods unused internally but part of public API
impl<const N: usize> TwiddleTable<N> {
    /// Create a new twiddle table at runtime.
    pub fn new() -> Self {
        let mut cos = [0.0; N];
        let mut sin = [0.0; N];

        for k in 0..N {
            let angle = -2.0 * PI * (k as f64) / (N as f64);
            cos[k] = libm::cos(angle);
            sin[k] = libm::sin(angle);
        }

        Self { cos, sin }
    }

    /// Get twiddle factor (cos, sin) for index k.
    #[inline]
    pub fn get(&self, k: usize) -> (f64, f64) {
        (self.cos[k % N], self.sin[k % N])
    }
}

impl<const N: usize> Default for TwiddleTable<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_cos_0() {
        let c = const_cos(0.0);
        assert!((c - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_const_cos_pi_2() {
        let c = const_cos(PI / 2.0);
        assert!(c.abs() < 1e-10);
    }

    #[test]
    fn test_const_cos_pi() {
        let c = const_cos(PI);
        // Taylor series has limited precision at boundary π
        // Error is ~4e-6, accept 1e-4 tolerance
        assert!((c - (-1.0)).abs() < 1e-4, "cos(π) = {c}");
    }

    #[test]
    fn test_const_sin_0() {
        let s = const_sin(0.0);
        assert!(s.abs() < 1e-14);
    }

    #[test]
    fn test_const_sin_pi_2() {
        let s = const_sin(PI / 2.0);
        // Taylor series has limited precision near π/2
        assert!((s - 1.0).abs() < 1e-8, "sin(π/2) = {s}");
    }

    #[test]
    fn test_const_sin_pi() {
        let s = const_sin(PI);
        // Taylor series has limited precision at boundary π
        // Error is ~2e-5, accept 1e-4 tolerance
        assert!(s.abs() < 1e-4, "sin(π) = {s}");
    }

    #[test]
    fn test_const_sin_cos_identity() {
        // sin²(x) + cos²(x) = 1
        let angles = [
            0.0,
            PI / 6.0,
            PI / 4.0,
            PI / 3.0,
            PI / 2.0,
            PI,
            3.0 * PI / 2.0,
        ];

        for &angle in &angles {
            let c = const_cos(angle);
            let s = const_sin(angle);
            let sum = c * c + s * s;
            // Taylor series accumulates errors, especially at boundaries
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "Identity failed for angle {angle}: {sum}"
            );
        }
    }

    #[test]
    fn test_twiddle_factor_unity() {
        // W_N^0 = 1
        let (c, s) = twiddle_factor(0, 8);
        assert!((c - 1.0).abs() < 1e-10);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_twiddle_factor_w8_1() {
        // W_8^1 = e^(-2πi/8) = cos(-π/4) - i*sin(-π/4) = 1/√2 + i*(-1/√2) = (√2/2, -√2/2)
        let (c, s) = twiddle_factor(1, 8);
        let sqrt2_2 = core::f64::consts::FRAC_1_SQRT_2;

        assert!((c - sqrt2_2).abs() < 1e-10, "cos: {c} vs {sqrt2_2}");
        assert!((s - (-sqrt2_2)).abs() < 1e-10, "sin: {} vs {}", s, -sqrt2_2);
    }

    #[test]
    fn test_twiddle_factor_w4_1() {
        // W_4^1 = e^(-2πi/4) = e^(-iπ/2) = cos(-π/2) + i*sin(-π/2) = 0 + i*(-1) = -i
        // So (cos(-π/2), sin(-π/2)) = (0, -1)
        let (c, s) = twiddle_factor(1, 4);
        assert!(c.abs() < 1e-10, "cos should be 0: {c}");
        // Taylor series has limited precision
        assert!((s - (-1.0)).abs() < 1e-8, "sin should be -1: {s}");
    }

    #[test]
    fn test_twiddle_table() {
        let table = TwiddleTable::<8>::new();

        // Compare with runtime computation
        for k in 0..8 {
            let angle = -2.0 * PI * (k as f64) / 8.0;
            let expected_cos = angle.cos();
            let expected_sin = angle.sin();

            assert!(
                (table.cos[k] - expected_cos).abs() < 1e-14,
                "cos mismatch at {}: {} vs {}",
                k,
                table.cos[k],
                expected_cos
            );
            assert!(
                (table.sin[k] - expected_sin).abs() < 1e-14,
                "sin mismatch at {}: {} vs {}",
                k,
                table.sin[k],
                expected_sin
            );
        }
    }
}
