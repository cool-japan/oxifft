//! Real-valued twiddle codelets for R2C/C2R post-/pre-processing.
//!
//! When computing a real FFT via the packed-real trick (pack N reals into N/2 complex,
//! compute N/2-point complex FFT, then unpack), the "unpack" step requires twiddle
//! multiplications.  Because the original signal is real, Hermitian symmetry lets us
//! process bin-pairs (k, N/2−k) together, halving the twiddle work.
//!
//! **Post-processing** (`real_twiddle_post`): applied after the N/2-point complex FFT
//! to produce the correct N-point R2C spectrum.
//!
//! **Pre-processing** (`real_twiddle_pre`): applied to the N/2+1 complex spectrum
//! before running an inverse N/2-point complex FFT, producing packed-real output
//! that can be deinterleaved to recover the real signal.
//!
//! Size-specialized versions (N=4, 8, 16) use hardcoded twiddle constants to avoid
//! trigonometric calls entirely.
//!
//! ## Post-processing formula (R2C)
//!
//! ```text
//! DC:      X[0]   = (Z[0].re + Z[0].im, 0)
//! Nyquist: X[N/2] = (Z[0].re - Z[0].im, 0)
//! For k = 1 to N/2-1:
//!   Let A = Z[k],  B = conj(Z[N/2-k])
//!   sum  = A + B
//!   diff = A - B
//!   X[k] = (sum + (-i · diff) · W_N^k) / 2
//!   where W_N^k = exp(-2πi k/N)
//! ```
//!
//! ## Pre-processing formula (C2R)
//!
//! ```text
//! Z[0] = ((X[0].re + X[N/2].re)/2,  (X[0].re - X[N/2].re)/2)
//! For k = 1 to N/2-1:
//!   Let A = X[k],  B = conj(X[N/2-k])
//!   e = (A + B) / 2
//!   o = (A - B) / 2
//!   Z[k] = e + (i · conj(W_N^k)) · o
//! ```

use crate::kernel::{Complex, Float};

// ─── Generic real twiddle post-processing (R2C direction) ────────────────────

/// Real-valued twiddle post-processing for R2C.
///
/// Given the N/2-point complex FFT output `Z[0..N/2]` of packed-real data,
/// produces the correct N-point R2C spectrum in-place.
///
/// On entry `data[0..N/2]` holds the complex FFT output.
/// On exit  `data[0..=N/2]` holds the R2C half-spectrum.
///
/// # Panics
///
/// Panics if `data.len() < n / 2 + 1` or `n` is odd or `n < 2`.
pub fn real_twiddle_post<T: Float>(data: &mut [Complex<T>], n: usize) {
    assert!(n >= 2 && n.is_multiple_of(2), "n must be even and >= 2");
    let half = n / 2;
    assert!(data.len() > half, "data must have at least N/2+1 elements");

    let inv2 = T::ONE / T::TWO;

    // DC and Nyquist from Z[0]
    let z0 = data[0];
    data[0] = Complex::new(z0.re + z0.im, T::ZERO);
    data[half] = Complex::new(z0.re - z0.im, T::ZERO);

    // Process conjugate pairs (k, j = half-k) where k < j.
    // We read data[k] and data[j] before writing either.
    let mut k = 1;
    while k < half - k {
        let j = half - k;
        let zk = data[k];
        let zj = data[j];

        // X[k]: A = zk, B = conj(zj)
        data[k] = unpack_bin(zk, zj.conj(), k, n, inv2);

        // X[j]: A = zj, B = conj(zk)
        data[j] = unpack_bin(zj, zk.conj(), j, n, inv2);

        k += 1;
    }

    // Self-pair when half is even: k == j == half/2
    if k == half - k {
        let zk = data[k];
        data[k] = unpack_bin(zk, zk.conj(), k, n, inv2);
    }
}

/// Compute one R2C output bin: `(sum + (-i · diff) · W) / 2`.
#[inline(always)]
fn unpack_bin<T: Float>(a: Complex<T>, b: Complex<T>, k: usize, n: usize, inv2: T) -> Complex<T> {
    let sum = a + b;
    let diff = a - b;
    // -i * diff = Complex(diff.im, -diff.re)
    let idiff = Complex::new(diff.im, -diff.re);
    let angle = -T::TWO_PI * T::from_usize(k) / T::from_usize(n);
    let w = Complex::cis(angle);
    (sum + idiff * w) * inv2
}

// ─── Generic real twiddle pre-processing (C2R direction) ─────────────────────

/// Real-valued twiddle pre-processing for C2R.
///
/// Inverse of [`real_twiddle_post`]. Given the R2C half-spectrum
/// `X[0..=N/2]`, produces packed-complex data `Z[0..N/2]` suitable
/// for an inverse N/2-point complex FFT.
///
/// # Panics
///
/// Panics if `data.len() < n / 2 + 1` or `n` is odd or `n < 2`.
pub fn real_twiddle_pre<T: Float>(data: &mut [Complex<T>], n: usize) {
    assert!(n >= 2 && n.is_multiple_of(2), "n must be even and >= 2");
    let half = n / 2;
    assert!(data.len() > half, "data must have at least N/2+1 elements");

    let inv2 = T::ONE / T::TWO;

    // DC / Nyquist → Z[0]
    let dc = data[0].re;
    let nyq = data[half].re;
    data[0] = Complex::new((dc + nyq) * inv2, (dc - nyq) * inv2);

    // Process pair (k, j = half-k) where k < j
    let mut k = 1;
    while k < half - k {
        let j = half - k;
        let xk = data[k];
        let xj = data[j];

        data[k] = repack_bin(xk, xj.conj(), k, n, inv2);
        data[j] = repack_bin(xj, xk.conj(), j, n, inv2);

        k += 1;
    }

    // Self-pair
    if k == half - k {
        let xk = data[k];
        data[k] = repack_bin(xk, xk.conj(), k, n, inv2);
    }
}

/// Compute one packed-FFT bin from R2C spectrum (pre-processing).
///
/// `a = X[k]`, `b = conj(X[N/2-k])`.
/// Result: `e + (i · conj(W_N^k)) · o` where `e = (a+b)/2`, `o = (a-b)/2`.
#[inline(always)]
fn repack_bin<T: Float>(a: Complex<T>, b: Complex<T>, k: usize, n: usize, inv2: T) -> Complex<T> {
    let e = (a + b) * inv2;
    let o = (a - b) * inv2;
    // i · conj(W_N^k) = i · exp(+2πi k/N) = Complex(-sin(2πk/N), cos(2πk/N))
    let angle = T::TWO_PI * T::from_usize(k) / T::from_usize(n);
    let (sin_a, cos_a) = <T as Float>::sin_cos(angle);
    let t = Complex::new(-sin_a, cos_a);
    e + t * o
}

// ─── Size-specialized post-processing (R2C) ─────────────────────────────────

/// Real twiddle post-processing for N=4.
///
/// Only bin k=1 (self-pair since N/2−1=1). The unpack formula with W_4^1=−i
/// simplifies to X\[1\] = conj(Z\[1\]).
///
/// `data` must have length ≥ 3.
#[inline(always)]
pub fn real_twiddle_post_4<T: Float>(data: &mut [Complex<T>]) {
    debug_assert!(data.len() >= 3, "real_twiddle_post_4: need >= 3 elements");

    let z0 = data[0];
    data[0] = Complex::new(z0.re + z0.im, T::ZERO);
    data[2] = Complex::new(z0.re - z0.im, T::ZERO);

    // Self-pair k=1: X[1] = conj(Z[1])
    data[1] = data[1].conj();
}

/// Real twiddle post-processing for N=8.
///
/// Pair (k=1,j=3) with hardcoded W_8^k, plus self-pair k=2.
/// For the self-pair with W_8^2=−i, X\[2\] = conj(Z\[2\]).
///
/// `data` must have length ≥ 5.
#[inline(always)]
pub fn real_twiddle_post_8<T: Float>(data: &mut [Complex<T>]) {
    debug_assert!(data.len() >= 5, "real_twiddle_post_8: need >= 5 elements");
    let inv2 = T::ONE / T::TWO;

    // DC, Nyquist
    let z0 = data[0];
    data[0] = Complex::new(z0.re + z0.im, T::ZERO);
    data[4] = Complex::new(z0.re - z0.im, T::ZERO);

    // Twiddles
    let s = T::ONE / <T as Float>::sqrt(T::TWO); // sqrt(2)/2
    let w1 = Complex::new(s, -s); // W_8^1 = exp(-pi i/4)
    let w3 = Complex::new(-s, -s); // W_8^3 = exp(-3 pi i/4)

    // Pair (k=1, j=3)
    let z1 = data[1];
    let z3 = data[3];
    data[1] = post_bin_hardcoded(z1, z3.conj(), w1, inv2);
    data[3] = post_bin_hardcoded(z3, z1.conj(), w3, inv2);

    // Self-pair k=2: X[2] = conj(Z[2])
    data[2] = data[2].conj();
}

/// Real twiddle post-processing for N=16.
///
/// Pairs (1,7), (2,6), (3,5) with precomputed twiddles, plus self-pair k=4.
///
/// `data` must have length ≥ 9.
#[inline(always)]
pub fn real_twiddle_post_16<T: Float>(data: &mut [Complex<T>]) {
    debug_assert!(data.len() >= 9, "real_twiddle_post_16: need >= 9 elements");
    let inv2 = T::ONE / T::TWO;

    // DC, Nyquist
    let z0 = data[0];
    data[0] = Complex::new(z0.re + z0.im, T::ZERO);
    data[8] = Complex::new(z0.re - z0.im, T::ZERO);

    // W_16^k = cos(k pi/8) - i sin(k pi/8)
    let pi_8 = <T as Float>::PI / T::from_usize(8);
    let (s1, c1) = <T as Float>::sin_cos(pi_8);
    let (s2, c2) = <T as Float>::sin_cos(T::TWO * pi_8);
    let (s3, c3) = <T as Float>::sin_cos(T::from_usize(3) * pi_8);

    let w1 = Complex::new(c1, -s1);
    let w2 = Complex::new(c2, -s2);
    let w3 = Complex::new(c3, -s3);
    // W_16^(8-k) = cos((8-k)pi/8) - i sin((8-k)pi/8)
    // = -cos(k pi/8) - i sin(k pi/8)   (using cos(pi - x) = -cos x, sin(pi - x) = sin x)
    let w5 = Complex::new(-c3, -s3);
    let w6 = Complex::new(-c2, -s2);
    let w7 = Complex::new(-c1, -s1);

    // Pair (1, 7)
    let z1 = data[1];
    let z7 = data[7];
    let out_1 = post_bin_hardcoded(z1, z7.conj(), w1, inv2);
    let out_7 = post_bin_hardcoded(z7, z1.conj(), w7, inv2);

    // Pair (2, 6)
    let z2 = data[2];
    let z6 = data[6];
    let out_2 = post_bin_hardcoded(z2, z6.conj(), w2, inv2);
    let out_6 = post_bin_hardcoded(z6, z2.conj(), w6, inv2);

    // Pair (3, 5)
    let z3 = data[3];
    let z5 = data[5];
    let out_3 = post_bin_hardcoded(z3, z5.conj(), w3, inv2);
    let out_5 = post_bin_hardcoded(z5, z3.conj(), w5, inv2);

    data[1] = out_1;
    data[2] = out_2;
    data[3] = out_3;
    data[5] = out_5;
    data[6] = out_6;
    data[7] = out_7;

    // Self-pair k=4: X[4] = conj(Z[4])
    data[4] = data[4].conj();
}

/// Post-processing one bin with a precomputed twiddle.
///
/// `(sum + (-i · diff) · w) / 2` where `sum = a+b`, `diff = a-b`.
#[inline(always)]
fn post_bin_hardcoded<T: Float>(
    a: Complex<T>,
    b: Complex<T>,
    w: Complex<T>,
    inv2: T,
) -> Complex<T> {
    let sum = a + b;
    let diff = a - b;
    let idiff = Complex::new(diff.im, -diff.re); // -i * diff
    (sum + idiff * w) * inv2
}

// ─── Size-specialized pre-processing (C2R) ──────────────────────────────────

/// Real twiddle pre-processing for N=4 (C2R direction).
///
/// Inverse of [`real_twiddle_post_4`].
/// Self-pair k=1: Z\[1\] = conj(X\[1\]).
///
/// `data` must have length ≥ 3.
#[inline(always)]
pub fn real_twiddle_pre_4<T: Float>(data: &mut [Complex<T>]) {
    debug_assert!(data.len() >= 3, "real_twiddle_pre_4: need >= 3 elements");
    let inv2 = T::ONE / T::TWO;

    let dc = data[0].re;
    let nyq = data[2].re;
    data[0] = Complex::new((dc + nyq) * inv2, (dc - nyq) * inv2);

    // Self-pair k=1: Z[1] = conj(X[1])
    data[1] = data[1].conj();
}

/// Real twiddle pre-processing for N=8 (C2R direction).
///
/// Inverse of [`real_twiddle_post_8`].
///
/// `data` must have length ≥ 5.
#[inline(always)]
pub fn real_twiddle_pre_8<T: Float>(data: &mut [Complex<T>]) {
    debug_assert!(data.len() >= 5, "real_twiddle_pre_8: need >= 5 elements");
    let inv2 = T::ONE / T::TWO;

    let dc = data[0].re;
    let nyq = data[4].re;
    data[0] = Complex::new((dc + nyq) * inv2, (dc - nyq) * inv2);

    // i * conj(W_8^k) = Complex(-sin(k pi/4), cos(k pi/4))
    let s = T::ONE / <T as Float>::sqrt(T::TWO);
    let t1 = Complex::new(-s, s); // k=1: (-sin pi/4, cos pi/4)
    let t3 = Complex::new(-s, -s); // k=3: (-sin 3pi/4, cos 3pi/4)

    // Pair (1, 3)
    let x1 = data[1];
    let x3 = data[3];
    data[1] = pre_bin_hardcoded(x1, x3.conj(), t1, inv2);
    data[3] = pre_bin_hardcoded(x3, x1.conj(), t3, inv2);

    // Self-pair k=2: Z[2] = conj(X[2])
    data[2] = data[2].conj();
}

/// Real twiddle pre-processing for N=16 (C2R direction).
///
/// Inverse of [`real_twiddle_post_16`].
///
/// `data` must have length ≥ 9.
#[inline(always)]
pub fn real_twiddle_pre_16<T: Float>(data: &mut [Complex<T>]) {
    debug_assert!(data.len() >= 9, "real_twiddle_pre_16: need >= 9 elements");
    let inv2 = T::ONE / T::TWO;

    let dc = data[0].re;
    let nyq = data[8].re;
    data[0] = Complex::new((dc + nyq) * inv2, (dc - nyq) * inv2);

    // i * conj(W_16^k) = Complex(-sin(k pi/8), cos(k pi/8))
    let pi_8 = <T as Float>::PI / T::from_usize(8);
    let (s1, c1) = <T as Float>::sin_cos(pi_8);
    let (s2, c2) = <T as Float>::sin_cos(T::TWO * pi_8);
    let (s3, c3) = <T as Float>::sin_cos(T::from_usize(3) * pi_8);

    let t1 = Complex::new(-s1, c1);
    let t2 = Complex::new(-s2, c2);
    let t3 = Complex::new(-s3, c3);
    // For j = 8-k: sin(j pi/8) = sin(k pi/8), cos(j pi/8) = -cos(k pi/8)
    let t5 = Complex::new(-s3, -c3);
    let t6 = Complex::new(-s2, -c2);
    let t7 = Complex::new(-s1, -c1);

    // Pair (1, 7)
    let x1 = data[1];
    let x7 = data[7];
    let out_1 = pre_bin_hardcoded(x1, x7.conj(), t1, inv2);
    let out_7 = pre_bin_hardcoded(x7, x1.conj(), t7, inv2);

    // Pair (2, 6)
    let x2 = data[2];
    let x6 = data[6];
    let out_2 = pre_bin_hardcoded(x2, x6.conj(), t2, inv2);
    let out_6 = pre_bin_hardcoded(x6, x2.conj(), t6, inv2);

    // Pair (3, 5)
    let x3 = data[3];
    let x5 = data[5];
    let out_3 = pre_bin_hardcoded(x3, x5.conj(), t3, inv2);
    let out_5 = pre_bin_hardcoded(x5, x3.conj(), t5, inv2);

    data[1] = out_1;
    data[2] = out_2;
    data[3] = out_3;
    data[5] = out_5;
    data[6] = out_6;
    data[7] = out_7;

    // Self-pair k=4: Z[4] = conj(X[4])
    data[4] = data[4].conj();
}

/// Pre-processing one bin with a precomputed effective twiddle.
///
/// `e + t · o` where `e = (a+b)/2`, `o = (a-b)/2`, `t = i · conj(W)`.
#[inline(always)]
fn pre_bin_hardcoded<T: Float>(a: Complex<T>, b: Complex<T>, t: Complex<T>, inv2: T) -> Complex<T> {
    let e = (a + b) * inv2;
    let o = (a - b) * inv2;
    e + t * o
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    fn approx_eq_complex(a: Complex<f64>, b: Complex<f64>) -> bool {
        approx_eq(a.re, b.re) && approx_eq(a.im, b.im)
    }

    /// Naive DFT of real input — reference for verification.
    fn naive_r2c(x: &[f64]) -> Vec<Complex<f64>> {
        let n = x.len();
        (0..=n / 2)
            .map(|k| {
                x.iter().enumerate().fold(Complex::zero(), |acc, (j, &xj)| {
                    let angle = -2.0 * core::f64::consts::PI * (j * k) as f64 / n as f64;
                    let w = Complex::new(angle.cos(), angle.sin());
                    acc + Complex::new(xj, 0.0) * w
                })
            })
            .collect()
    }

    /// Pack N reals into N/2 complex, compute naive complex DFT.
    fn pack_and_fft(x: &[f64]) -> Vec<Complex<f64>> {
        let n = x.len();
        let half = n / 2;
        let z: Vec<Complex<f64>> = (0..half)
            .map(|k| Complex::new(x[2 * k], x[2 * k + 1]))
            .collect();
        (0..half)
            .map(|k| {
                z.iter().enumerate().fold(Complex::zero(), |acc, (j, &zj)| {
                    let angle = -2.0 * core::f64::consts::PI * (j * k) as f64 / half as f64;
                    let w = Complex::new(angle.cos(), angle.sin());
                    acc + zj * w
                })
            })
            .collect()
    }

    // ─── generic post matches naive R2C ───

    fn test_post_matches_r2c(n: usize) {
        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.7 - 1.5).collect();
        let expected = naive_r2c(&x);

        let mut data = pack_and_fft(&x);
        data.resize(n / 2 + 1, Complex::zero());
        real_twiddle_post(&mut data, n);

        for k in 0..=n / 2 {
            assert!(
                approx_eq_complex(data[k], expected[k]),
                "N={n}, k={k}: got {:?}, expected {:?}",
                data[k],
                expected[k]
            );
        }
    }

    #[test]
    fn test_post_matches_r2c_4() {
        test_post_matches_r2c(4);
    }

    #[test]
    fn test_post_matches_r2c_8() {
        test_post_matches_r2c(8);
    }

    #[test]
    fn test_post_matches_r2c_16() {
        test_post_matches_r2c(16);
    }

    #[test]
    fn test_post_matches_r2c_32() {
        test_post_matches_r2c(32);
    }

    #[test]
    fn test_post_matches_r2c_64() {
        test_post_matches_r2c(64);
    }

    // ─── roundtrip post + pre = identity ───

    fn test_roundtrip(n: usize) {
        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 1.3 + 0.5).collect();
        let z_fft = pack_and_fft(&x);
        let mut data = z_fft.clone();
        data.resize(n / 2 + 1, Complex::zero());

        real_twiddle_post(&mut data, n);
        real_twiddle_pre(&mut data, n);

        for k in 0..n / 2 {
            assert!(
                approx_eq_complex(data[k], z_fft[k]),
                "N={n}, k={k}: roundtrip got {:?}, expected {:?}",
                data[k],
                z_fft[k]
            );
        }
    }

    #[test]
    fn test_roundtrip_4() {
        test_roundtrip(4);
    }

    #[test]
    fn test_roundtrip_8() {
        test_roundtrip(8);
    }

    #[test]
    fn test_roundtrip_16() {
        test_roundtrip(16);
    }

    #[test]
    fn test_roundtrip_32() {
        test_roundtrip(32);
    }

    #[test]
    fn test_roundtrip_64() {
        test_roundtrip(64);
    }

    // ─── Hermitian symmetry of post output ───

    fn test_hermitian_symmetry(n: usize) {
        let x: Vec<f64> = (0..n).map(|i| ((i as f64) * 2.1).sin()).collect();

        let mut data = pack_and_fft(&x);
        data.resize(n / 2 + 1, Complex::zero());
        real_twiddle_post(&mut data, n);

        // DC and Nyquist must be purely real
        assert!(data[0].im.abs() < TOL, "N={n}: DC im={}", data[0].im);
        assert!(
            data[n / 2].im.abs() < TOL,
            "N={n}: Nyquist im={}",
            data[n / 2].im
        );

        let expected = naive_r2c(&x);
        for k in 0..=n / 2 {
            assert!(
                approx_eq_complex(data[k], expected[k]),
                "N={n}, k={k}: symmetry mismatch"
            );
        }
    }

    #[test]
    fn test_hermitian_4() {
        test_hermitian_symmetry(4);
    }

    #[test]
    fn test_hermitian_8() {
        test_hermitian_symmetry(8);
    }

    #[test]
    fn test_hermitian_16() {
        test_hermitian_symmetry(16);
    }

    #[test]
    fn test_hermitian_32() {
        test_hermitian_symmetry(32);
    }

    #[test]
    fn test_hermitian_64() {
        test_hermitian_symmetry(64);
    }

    // ─── specialized post matches generic ───

    fn test_specialized_post(n: usize, specialized_fn: fn(&mut [Complex<f64>])) {
        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.3 - 2.0).collect();
        let z_fft = pack_and_fft(&x);

        let mut generic = z_fft.clone();
        generic.resize(n / 2 + 1, Complex::zero());
        real_twiddle_post(&mut generic, n);

        let mut special = z_fft;
        special.resize(n / 2 + 1, Complex::zero());
        specialized_fn(&mut special);

        for k in 0..=n / 2 {
            assert!(
                approx_eq_complex(special[k], generic[k]),
                "N={n}, k={k}: specialized {:?} != generic {:?}",
                special[k],
                generic[k]
            );
        }
    }

    #[test]
    fn test_specialized_post_4() {
        test_specialized_post(4, real_twiddle_post_4);
    }

    #[test]
    fn test_specialized_post_8() {
        test_specialized_post(8, real_twiddle_post_8);
    }

    #[test]
    fn test_specialized_post_16() {
        test_specialized_post(16, real_twiddle_post_16);
    }

    // ─── specialized pre matches generic ───

    fn test_specialized_pre(n: usize, specialized_fn: fn(&mut [Complex<f64>])) {
        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.3 - 2.0).collect();
        let spectrum = naive_r2c(&x);

        let mut generic = spectrum.clone();
        real_twiddle_pre(&mut generic, n);

        let mut special = spectrum;
        specialized_fn(&mut special);

        for k in 0..n / 2 {
            assert!(
                approx_eq_complex(special[k], generic[k]),
                "N={n}, k={k}: specialized pre {:?} != generic {:?}",
                special[k],
                generic[k]
            );
        }
    }

    #[test]
    fn test_specialized_pre_4() {
        test_specialized_pre(4, real_twiddle_pre_4);
    }

    #[test]
    fn test_specialized_pre_8() {
        test_specialized_pre(8, real_twiddle_pre_8);
    }

    #[test]
    fn test_specialized_pre_16() {
        test_specialized_pre(16, real_twiddle_pre_16);
    }

    // ─── specialized roundtrip ───

    fn test_specialized_roundtrip(
        n: usize,
        post_fn: fn(&mut [Complex<f64>]),
        pre_fn: fn(&mut [Complex<f64>]),
    ) {
        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 1.1 - 3.0).collect();
        let z_fft = pack_and_fft(&x);
        let mut data = z_fft.clone();
        data.resize(n / 2 + 1, Complex::zero());

        post_fn(&mut data);
        pre_fn(&mut data);

        for k in 0..n / 2 {
            assert!(
                approx_eq_complex(data[k], z_fft[k]),
                "N={n}, k={k}: roundtrip got {:?}, expected {:?}",
                data[k],
                z_fft[k]
            );
        }
    }

    #[test]
    fn test_specialized_roundtrip_4() {
        test_specialized_roundtrip(4, real_twiddle_post_4, real_twiddle_pre_4);
    }

    #[test]
    fn test_specialized_roundtrip_8() {
        test_specialized_roundtrip(8, real_twiddle_post_8, real_twiddle_pre_8);
    }

    #[test]
    fn test_specialized_roundtrip_16() {
        test_specialized_roundtrip(16, real_twiddle_post_16, real_twiddle_pre_16);
    }

    // ─── f32 smoke tests ───

    #[test]
    fn test_post_f32_smoke() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n = x.len();
        let half = n / 2;

        let z: Vec<Complex<f32>> = (0..half)
            .map(|k| Complex::new(x[2 * k], x[2 * k + 1]))
            .collect();
        let z_fft: Vec<Complex<f32>> = (0..half)
            .map(|k| {
                z.iter().enumerate().fold(Complex::zero(), |acc, (j, &zj)| {
                    let angle = -2.0_f32 * core::f32::consts::PI * (j * k) as f32 / half as f32;
                    let w = Complex::new(angle.cos(), angle.sin());
                    acc + zj * w
                })
            })
            .collect();

        let mut data = z_fft;
        data.resize(half + 1, Complex::zero());
        real_twiddle_post(&mut data, n);

        assert!(data[0].im.abs() < 1e-5, "f32 DC im: {}", data[0].im);
        assert!(data[half].im.abs() < 1e-5, "f32 Nyq im: {}", data[half].im);
        let expected_dc: f32 = x.iter().sum();
        assert!(
            (data[0].re - expected_dc).abs() < 1e-4,
            "f32 DC: {} vs {}",
            data[0].re,
            expected_dc
        );
    }

    #[test]
    fn test_roundtrip_f32_smoke() {
        let x: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0];
        let n = x.len();
        let half = n / 2;

        let z: Vec<Complex<f32>> = (0..half)
            .map(|k| Complex::new(x[2 * k], x[2 * k + 1]))
            .collect();
        let z_fft: Vec<Complex<f32>> = (0..half)
            .map(|k| {
                z.iter().enumerate().fold(Complex::zero(), |acc, (j, &zj)| {
                    let angle = -2.0_f32 * core::f32::consts::PI * (j * k) as f32 / half as f32;
                    let w = Complex::new(angle.cos(), angle.sin());
                    acc + zj * w
                })
            })
            .collect();

        let mut data = z_fft.clone();
        data.resize(half + 1, Complex::zero());

        real_twiddle_post(&mut data, n);
        real_twiddle_pre(&mut data, n);

        for k in 0..half {
            assert!(
                (data[k].re - z_fft[k].re).abs() < 1e-5 && (data[k].im - z_fft[k].im).abs() < 1e-5,
                "f32 roundtrip k={k}: {:?} vs {:?}",
                data[k],
                z_fft[k]
            );
        }
    }
}
