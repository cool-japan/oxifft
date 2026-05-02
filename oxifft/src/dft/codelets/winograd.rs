//! Winograd small-prime FFT kernels.
//!
//! These kernels implement efficient DFT for prime and small composite sizes
//! using the symmetric-pair decomposition (Winograd's algorithm). The approach
//! exploits the Hermitian symmetry of real-valued DFT output and generalises to
//! complex inputs via paired sums/differences.
//!
//! Kernel convention (matches notw.rs):
//!   - In-place: `fn winograd_N<T: Float>(x: &mut [Complex<T>], sign: i32)`
//!   - sign < 0 → forward (W = e^{-2πi/N})
//!   - sign > 0 → inverse (W = e^{+2πi/N})
//!
//! Sizes 3, 5, 7, and 16 delegate to the already-correct `notw_*` implementations.
//! New implementations are provided for 9, 11, and 13.

use crate::kernel::{Complex, Float};

// Re-export the already-implemented small-prime kernels under winograd names.
// These use the same symmetric-pair Winograd reduction and are already tested.
pub use super::notw::{
    notw_16 as winograd_16, notw_3 as winograd_3, notw_5 as winograd_5, notw_7 as winograd_7,
};

// ─── DFT-9 ───────────────────────────────────────────────────────────────────
//
// N=9=3² is handled via the two-stage radix-3 DIT decomposition:
//   1. Apply DFT-3 to each of three interleaved sub-sequences (stride 3).
//   2. Multiply by twiddle factors W9^{n1·n2}.
//   3. Apply DFT-3 across the three results at each position.
//   4. Transpose the 3×3 output matrix to natural order.

/// Apply DFT-3 in-place on three elements at arbitrary indices.
fn dft3_at<T: Float>(x: &mut [Complex<T>], i0: usize, i1: usize, i2: usize, sign: i32) {
    let a = x[i0];
    let b = x[i1];
    let c_v = x[i2];
    let t_half = T::from_f64(-0.5); // cos(2π/3)
    let s = T::from_f64(0.866_025_403_784_438_6); // sin(2π/3)
    let sum = b + c_v;
    let t1 = a + sum * t_half;
    let diff = b - c_v;
    let t2_rot = if sign < 0 {
        Complex::new(diff.im * s, -diff.re * s)
    } else {
        Complex::new(-diff.im * s, diff.re * s)
    };
    x[i0] = a + sum;
    x[i1] = t1 + t2_rot;
    x[i2] = t1 - t2_rot;
}

/// Apply complex twiddle multiplication.
/// sign < 0: forward W = e^{-2πi/N} → multiply by (cos − i·sin)
/// sign ≥ 0: inverse W = e^{+2πi/N} → multiply by (cos + i·sin)
#[inline]
fn twiddle_mul<T: Float>(v: Complex<T>, c: T, s: T, sign: i32) -> Complex<T> {
    if sign < 0 {
        Complex::new(v.re * c + v.im * s, v.im * c - v.re * s)
    } else {
        Complex::new(v.re * c - v.im * s, v.im * c + v.re * s)
    }
}

/// Size-9 DFT using a two-stage radix-3 decomposition.
///
/// # Arguments
/// * `x` - in/out slice of length ≥ 9
/// * `sign` - negative for forward (e^{-2πi/N}), positive for inverse
#[inline]
pub fn winograd_9<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 9);

    // Stage 1: DFT-3 on each of three interleaved sub-sequences (stride 3)
    dft3_at(x, 0, 3, 6, sign);
    dft3_at(x, 1, 4, 7, sign);
    dft3_at(x, 2, 5, 8, sign);

    // Stage 2: apply twiddle factors W9^{n1·n2}
    // Row n1=0: all W9^0 = 1 (no-op)
    // Row n1=1: x[3] *= W9^0, x[4] *= W9^1, x[5] *= W9^2
    // Row n1=2: x[6] *= W9^0, x[7] *= W9^2, x[8] *= W9^4
    let c1 = T::from_f64(0.766_044_443_118_978); // cos(2π/9)
    let s1 = T::from_f64(0.642_787_609_686_539); // sin(2π/9)
    let c2 = T::from_f64(0.173_648_177_666_930_3); // cos(4π/9)
    let s2 = T::from_f64(0.984_807_753_012_208); // sin(4π/9)
    let c4 = T::from_f64(-0.939_692_620_785_908_4); // cos(8π/9)
    let s4 = T::from_f64(0.342_020_143_325_668_7); // sin(8π/9)

    x[4] = twiddle_mul(x[4], c1, s1, sign);
    x[5] = twiddle_mul(x[5], c2, s2, sign);
    x[7] = twiddle_mul(x[7], c2, s2, sign);
    x[8] = twiddle_mul(x[8], c4, s4, sign);

    // Stage 3: DFT-3 on each contiguous group (positions 0-2, 3-5, 6-8).
    // After stage 1+2, data is in row-major order: row n1 occupies positions n1*3 to n1*3+2.
    // The N2=3 output DFTs need to be applied column-wise (across rows at each column index k2),
    // but since all column indices are interleaved as k2=0→x[0,1,2], k2=1→x[3,4,5], k2=2→x[6,7,8]
    // we apply DFT-3 on each contiguous group of 3.
    dft3_at(x, 0, 1, 2, sign); // column k2=0: contributes to frequencies k2*N1 + k1 = k1
    dft3_at(x, 3, 4, 5, sign); // column k2=1: contributes to frequencies 3 + k1
    dft3_at(x, 6, 7, 8, sign); // column k2=2: contributes to frequencies 6 + k1

    // Stage 4: output frequency k = k1 + N1*k2 = k1 + 3*k2 is stored at position k2*3 + k1.
    // Convert from position [k2*3 + k1] to natural index [k1 + 3*k2] via transpose:
    x.swap(1, 3);
    x.swap(2, 6);
    x.swap(5, 7);
}

// ─── DFT-11 ──────────────────────────────────────────────────────────────────
//
// Symmetric-pair decomposition for N=11 (prime). Half = 5 pairs.
//
// For complex input x, forward DFT:
//   X[k] = x[0] + Σ_{j=1}^{5} [ a[j]*cos(2πjk/11) − i·b[j]*sin(2πjk/11) ]
// where a[j]=x[j]+x[11-j], b[j]=x[j]-x[11-j].
//
// Expanding in real/imag components (−i·b = b.im − i·b.re):
//   X[k].re = x[0].re + Σ_j ( a[j].re·cos + b[j].im·sin )
//   X[k].im = x[0].im + Σ_j ( a[j].im·cos − b[j].re·sin )
//
// X[11-k] uses +i·b[j]·sin instead:
//   X[11-k].re = x[0].re + Σ_j ( a[j].re·cos − b[j].im·sin )
//   X[11-k].im = x[0].im + Σ_j ( a[j].im·cos + b[j].re·sin )

/// Size-11 DFT using symmetric-pair decomposition.
///
/// # Arguments
/// * `x` - in/out slice of length ≥ 11
/// * `sign` - negative for forward, positive for inverse
#[inline]
pub fn winograd_11<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 11);

    use super::winograd_constants::{
        C11_COS1, C11_COS2, C11_COS3, C11_COS4, C11_COS5, C11_SIN1, C11_SIN2, C11_SIN3, C11_SIN4,
        C11_SIN5,
    };

    let cos = [
        T::from_f64(C11_COS1),
        T::from_f64(C11_COS2),
        T::from_f64(C11_COS3),
        T::from_f64(C11_COS4),
        T::from_f64(C11_COS5),
    ];
    let sin_t = [
        T::from_f64(C11_SIN1),
        T::from_f64(C11_SIN2),
        T::from_f64(C11_SIN3),
        T::from_f64(C11_SIN4),
        T::from_f64(C11_SIN5),
    ];

    let x0 = x[0];
    let a = [
        x[1] + x[10],
        x[2] + x[9],
        x[3] + x[8],
        x[4] + x[7],
        x[5] + x[6],
    ];
    let b = [
        x[1] - x[10],
        x[2] - x[9],
        x[3] - x[8],
        x[4] - x[7],
        x[5] - x[6],
    ];

    x[0] = x0 + a[0] + a[1] + a[2] + a[3] + a[4];

    for k in 1_usize..=5 {
        let mut fwd_re = x0.re;
        let mut fwd_im = x0.im;
        let mut bwd_re = x0.re;
        let mut bwd_im = x0.im;

        for j in 1_usize..=5 {
            let jk_mod = (j * k) % 11;
            // Map to table using symmetry: cos(2π(N-m)/N)=cos(2πm/N), sin(2π(N-m)/N)=-sin(2πm/N)
            let (c_val, s_val) = if jk_mod <= 5 {
                (cos[jk_mod - 1], sin_t[jk_mod - 1])
            } else {
                (cos[11 - jk_mod - 1], -sin_t[11 - jk_mod - 1])
            };

            let aj = a[j - 1];
            let bj = b[j - 1];

            if sign < 0 {
                // Forward: X[k]   += a*cos - i*b*sin  => re: a.re*cos + b.im*sin
                fwd_re += aj.re * c_val + bj.im * s_val;
                fwd_im += aj.im * c_val - bj.re * s_val;
                // Forward: X[11-k] += a*cos + i*b*sin => re: a.re*cos - b.im*sin
                bwd_re += aj.re * c_val - bj.im * s_val;
                bwd_im += aj.im * c_val + bj.re * s_val;
            } else {
                // Inverse signs are swapped
                fwd_re += aj.re * c_val - bj.im * s_val;
                fwd_im += aj.im * c_val + bj.re * s_val;
                bwd_re += aj.re * c_val + bj.im * s_val;
                bwd_im += aj.im * c_val - bj.re * s_val;
            }
        }

        x[k] = Complex::new(fwd_re, fwd_im);
        x[11 - k] = Complex::new(bwd_re, bwd_im);
    }
}

// ─── DFT-13 ──────────────────────────────────────────────────────────────────

/// Size-13 DFT using symmetric-pair decomposition.
///
/// # Arguments
/// * `x` - in/out slice of length ≥ 13
/// * `sign` - negative for forward, positive for inverse
#[inline]
pub fn winograd_13<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 13);

    use super::winograd_constants::{
        C13_COS1, C13_COS2, C13_COS3, C13_COS4, C13_COS5, C13_COS6, C13_SIN1, C13_SIN2, C13_SIN3,
        C13_SIN4, C13_SIN5, C13_SIN6,
    };

    let cos = [
        T::from_f64(C13_COS1),
        T::from_f64(C13_COS2),
        T::from_f64(C13_COS3),
        T::from_f64(C13_COS4),
        T::from_f64(C13_COS5),
        T::from_f64(C13_COS6),
    ];
    let sin_t = [
        T::from_f64(C13_SIN1),
        T::from_f64(C13_SIN2),
        T::from_f64(C13_SIN3),
        T::from_f64(C13_SIN4),
        T::from_f64(C13_SIN5),
        T::from_f64(C13_SIN6),
    ];

    let x0 = x[0];
    let a = [
        x[1] + x[12],
        x[2] + x[11],
        x[3] + x[10],
        x[4] + x[9],
        x[5] + x[8],
        x[6] + x[7],
    ];
    let b = [
        x[1] - x[12],
        x[2] - x[11],
        x[3] - x[10],
        x[4] - x[9],
        x[5] - x[8],
        x[6] - x[7],
    ];

    x[0] = x0 + a[0] + a[1] + a[2] + a[3] + a[4] + a[5];

    for k in 1_usize..=6 {
        let mut fwd_re = x0.re;
        let mut fwd_im = x0.im;
        let mut bwd_re = x0.re;
        let mut bwd_im = x0.im;

        for j in 1_usize..=6 {
            let jk_mod = (j * k) % 13;
            let (c_val, s_val) = if jk_mod <= 6 {
                (cos[jk_mod - 1], sin_t[jk_mod - 1])
            } else {
                (cos[13 - jk_mod - 1], -sin_t[13 - jk_mod - 1])
            };

            let aj = a[j - 1];
            let bj = b[j - 1];

            if sign < 0 {
                fwd_re += aj.re * c_val + bj.im * s_val;
                fwd_im += aj.im * c_val - bj.re * s_val;
                bwd_re += aj.re * c_val - bj.im * s_val;
                bwd_im += aj.im * c_val + bj.re * s_val;
            } else {
                fwd_re += aj.re * c_val - bj.im * s_val;
                fwd_im += aj.im * c_val + bj.re * s_val;
                bwd_re += aj.re * c_val + bj.im * s_val;
                bwd_im += aj.im * c_val - bj.re * s_val;
            }
        }

        x[k] = Complex::new(fwd_re, fwd_im);
        x[13 - k] = Complex::new(bwd_re, bwd_im);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft(x: &[Complex<f64>], sign: i32) -> Vec<Complex<f64>> {
        let n = x.len();
        let sign_f = f64::from(sign);
        (0..n)
            .map(|k| {
                x.iter()
                    .enumerate()
                    .map(|(j, xj)| {
                        let angle = sign_f * core::f64::consts::TAU * (j * k) as f64 / n as f64;
                        let (s, c) = angle.sin_cos();
                        Complex::new(xj.re * c - xj.im * s, xj.re * s + xj.im * c)
                    })
                    .fold(Complex::new(0.0, 0.0), |acc, v| acc + v)
            })
            .collect()
    }

    fn check_eq(got: &[Complex<f64>], exp: &[Complex<f64>], tol: f64, label: &str) {
        for (i, (g, e)) in got.iter().zip(exp.iter()).enumerate() {
            let diff_re = (g.re - e.re).abs();
            let diff_im = (g.im - e.im).abs();
            assert!(
                diff_re < tol && diff_im < tol,
                "{label}[{i}]: got ({}, {}), expected ({}, {}), diff ({diff_re:.2e}, {diff_im:.2e})",
                g.re, g.im, e.re, e.im
            );
        }
    }

    fn test_kernel(n: usize, kernel: impl Fn(&mut [Complex<f64>], i32)) {
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i + 1) as f64 * 0.3, (i as f64 * 0.7).sin()))
            .collect();
        let tol = 1e-11 * n as f64;

        // Forward
        let expected_fwd = naive_dft(&input, -1);
        let mut actual_fwd = input.clone();
        kernel(&mut actual_fwd, -1);
        check_eq(&actual_fwd, &expected_fwd, tol, "forward");

        // Roundtrip: fwd then bwd then scale by 1/N
        let mut rt = actual_fwd.clone();
        kernel(&mut rt, 1);
        let n_f = n as f64;
        for v in &mut rt {
            *v = Complex::new(v.re / n_f, v.im / n_f);
        }
        check_eq(&rt, &input, tol, "roundtrip");

        // Impulse: forward DFT of [1, 0, 0, ...] = all-ones
        let mut impulse = vec![Complex::new(0.0_f64, 0.0_f64); n];
        impulse[0] = Complex::new(1.0, 0.0);
        kernel(&mut impulse, -1);
        for (i, v) in impulse.iter().enumerate() {
            let diff_re = (v.re - 1.0).abs();
            let diff_im = v.im.abs();
            assert!(
                diff_re < tol && diff_im < tol,
                "impulse[{i}]: got ({}, {})",
                v.re,
                v.im
            );
        }
    }

    #[test]
    fn test_winograd_constants_verify() {
        super::super::winograd_constants::verify_constants();
    }

    #[test]
    fn test_winograd_9() {
        test_kernel(9, winograd_9);
    }

    #[test]
    fn test_winograd_11() {
        test_kernel(11, winograd_11);
    }

    #[test]
    fn test_winograd_13() {
        test_kernel(13, winograd_13);
    }

    #[test]
    fn test_winograd_3_alias() {
        test_kernel(3, winograd_3);
    }

    #[test]
    fn test_winograd_5_alias() {
        test_kernel(5, winograd_5);
    }

    #[test]
    fn test_winograd_7_alias() {
        test_kernel(7, winograd_7);
    }

    #[test]
    fn test_winograd_16_alias() {
        test_kernel(16, winograd_16);
    }
}
