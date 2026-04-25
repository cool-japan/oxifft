//! RDFT codelets — small-size Real-to-Half-Complex and Half-Complex-to-Real transforms.
//!
//! These hardcoded butterfly networks provide optimal performance for sizes 2, 4, and 8.
//! All codelets are unnormalized (HC2R produces N × original values; caller divides by N).

#[cfg(test)]
mod codegen_tests;
mod generated;
pub mod real_twiddle;

use crate::kernel::{Complex, Float};

pub(crate) use generated::{hc2r_4_gen, hc2r_8_gen, r2hc_4_gen, r2hc_8_gen};

pub use real_twiddle::{
    real_twiddle_post, real_twiddle_post_16, real_twiddle_post_4, real_twiddle_post_8,
    real_twiddle_pre, real_twiddle_pre_16, real_twiddle_pre_4, real_twiddle_pre_8,
};

// ─── R2HC Codelets ───────────────────────────────────────────────────────────

/// R2HC (real to half-complex) codelet for N=2.
///
/// Given 2 real inputs `x`, writes 2 complex outputs `y`:
/// - `y[0]` = DC  = x\[0\] + x\[1\]  (purely real)
/// - `y[1]` = Nyq = x\[0\] - x\[1\]  (purely real for N=2)
#[inline(always)]
pub fn r2hc_2<T: Float>(x: &[T], y: &mut [Complex<T>]) {
    debug_assert_eq!(x.len(), 2, "r2hc_2: input must have exactly 2 elements");
    debug_assert!(y.len() >= 2, "r2hc_2: output must have at least 2 elements");
    y[0] = Complex::new(x[0] + x[1], T::ZERO);
    y[1] = Complex::new(x[0] - x[1], T::ZERO);
}

/// R2HC (real to half-complex) codelet for N=4.
///
/// Given 4 real inputs `x`, writes 3 complex outputs `y` (k=0,1,2):
/// - `y[0]` = DC        (purely real)
/// - `y[1]` = k=1 bin   (complex)
/// - `y[2]` = Nyquist   (purely real)
#[inline(always)]
pub fn r2hc_4<T: Float>(x: &[T], y: &mut [Complex<T>]) {
    debug_assert_eq!(x.len(), 4, "r2hc_4: input must have exactly 4 elements");
    debug_assert!(y.len() >= 3, "r2hc_4: output must have at least 3 elements");

    let a = x[0] + x[2]; // even-half sum
    let b = x[0] - x[2]; // even-half diff
    let c = x[1] + x[3]; // odd-half sum
    let d = x[1] - x[3]; // odd-half diff

    y[0] = Complex::new(a + c, T::ZERO); // DC
    y[1] = Complex::new(b, -d); // k=1: (b - i*d)
    y[2] = Complex::new(a - c, T::ZERO); // Nyquist
}

/// R2HC (real to half-complex) codelet for N=8.
///
/// Given 8 real inputs `x`, writes 5 complex outputs `y` (k=0..4):
/// - `y[0]` = DC           (purely real)
/// - `y[1]` .. `y[3]`     = k=1,2,3 bins
/// - `y[4]` = Nyquist      (purely real)
///
/// Uses radix-2 DIT decomposition: even-indexed inputs form a size-4 sub-problem,
/// odd-indexed inputs form another, then combined with twiddle factors.
///
/// Twiddles W8^k = exp(-2πi*k/8):
///   W8^0=1, W8^1=(1-i)/√2, W8^2=-i, W8^3=-(1+i)/√2
#[inline(always)]
pub fn r2hc_8<T: Float>(x: &[T], y: &mut [Complex<T>]) {
    debug_assert_eq!(x.len(), 8, "r2hc_8: input must have exactly 8 elements");
    debug_assert!(y.len() >= 5, "r2hc_8: output must have at least 5 elements");

    // ── Size-4 R2HC on even-indexed samples: x[0],x[2],x[4],x[6] ──
    let ae = x[0] + x[4];
    let be = x[0] - x[4];
    let ce = x[2] + x[6];
    let de = x[2] - x[6];

    // E[0]=(ae+ce,0), E[1]=(be,-de), E[2]=(ae-ce,0), E[3]=conj(E[1])
    let e0 = Complex::new(ae + ce, T::ZERO);
    let e1 = Complex::new(be, -de);
    let e2 = Complex::new(ae - ce, T::ZERO);

    // ── Size-4 R2HC on odd-indexed samples: x[1],x[3],x[5],x[7] ──
    let ao = x[1] + x[5];
    let bo = x[1] - x[5];
    let co = x[3] + x[7];
    let do_ = x[3] - x[7];

    let o0 = Complex::new(ao + co, T::ZERO);
    let o1 = Complex::new(bo, -do_);
    let o2 = Complex::new(ao - co, T::ZERO);

    // ── Apply twiddle factors W8^k and combine ──
    // W8^0 = 1,  W8^1 = (1-i)/sqrt(2),  W8^2 = -i,  W8^3 = -(1+i)/sqrt(2)
    let sqrt2_inv = T::ONE / <T as Float>::sqrt(T::TWO);
    let w8_1 = Complex::new(sqrt2_inv, -sqrt2_inv);
    let w8_2 = Complex::new(T::ZERO, -T::ONE);
    let w8_3 = Complex::new(-sqrt2_inv, -sqrt2_inv);

    // Y[k] = E[k] + W8^k * O[k]  for k=0..3
    // Y[4] = E[0] - O[0]  (Nyquist)
    // E[3] = conj(E[1]), O[3] = conj(O[1])
    y[0] = e0 + o0;
    y[1] = e1 + w8_1 * o1;
    y[2] = e2 + w8_2 * o2;
    y[3] = e1.conj() + w8_3 * o1.conj();
    y[4] = e0 - o0;
}

// ─── HC2R Codelets ───────────────────────────────────────────────────────────

/// HC2R (half-complex to real) codelet for N=2.
///
/// Input: 2 complex values `y` (Y\[0\]=DC, Y\[1\]=Nyquist, both purely real for real signals).
/// Output: 2 unnormalized real values `x`.  Caller divides by 2 for true inverse.
///
/// This is the exact inverse butterfly of `r2hc_2`.
#[inline(always)]
pub fn hc2r_2<T: Float>(y: &[Complex<T>], x: &mut [T]) {
    debug_assert!(y.len() >= 2, "hc2r_2: input must have at least 2 elements");
    debug_assert_eq!(x.len(), 2, "hc2r_2: output must have exactly 2 elements");
    x[0] = y[0].re + y[1].re;
    x[1] = y[0].re - y[1].re;
}

/// HC2R (half-complex to real) codelet for N=4.
///
/// Input: 3 complex values `y` (Y\[0\]=DC, Y\[1\]=k=1, Y\[2\]=Nyquist; Y\[0\]/Y\[2\] purely real).
/// Output: 4 unnormalized real values `x`.  Caller divides by 4 for true inverse.
///
/// Inverse butterfly of `r2hc_4`:
/// - a = Y\[0\].re + Y\[2\].re
/// - b = Y\[0\].re - Y\[2\].re
/// - c = 2·Y\[1\].re   (factor-of-2 comes from Hermitian symmetry contribution)
/// - d = 2·Y\[1\].im
/// - x\[0\] = a + c,  x\[1\] = b - d,  x\[2\] = a - c,  x\[3\] = b + d
#[inline(always)]
pub fn hc2r_4<T: Float>(y: &[Complex<T>], x: &mut [T]) {
    debug_assert!(y.len() >= 3, "hc2r_4: input must have at least 3 elements");
    debug_assert_eq!(x.len(), 4, "hc2r_4: output must have exactly 4 elements");

    let a = y[0].re + y[2].re;
    let b = y[0].re - y[2].re;
    let c = y[1].re + y[1].re; // 2·Y[1].re
    let d = y[1].im + y[1].im; // 2·Y[1].im

    x[0] = a + c;
    x[1] = b - d;
    x[2] = a - c;
    x[3] = b + d;
}

/// HC2R (half-complex to real) codelet for N=8.
///
/// Input: 5 complex values `y` (Y\[0..4\]).
/// Output: 8 unnormalized real values `x`.  Caller divides by 8 for true inverse.
///
/// Inverts `r2hc_8` via the radix-2 DIT butterfly structure (run in reverse).
///
/// DIT forward:  Y\[k\]   = E\[k\] + W8^k * O\[k\]  for k=0..3
///               Y\[k+4\] = E\[k\] - W8^k * O\[k\]  for k=0..3
///
/// Conjugate symmetry of a real-input DFT:  Y\[8-k\] = conj(Y\[k\]), so:
///   Y\[5\] = conj(Y\[3\]),  Y\[6\] = conj(Y\[2\]),  Y\[7\] = conj(Y\[1\])
///
/// Unnormalized recovery (no /2 ensures HC2R-4 sub-transforms give 8× output):
///   E\[k\] = Y\[k\] + Y\[k+4\]
///   O\[k\] = conj(W8^k) × (Y\[k\] - Y\[k+4\])
#[inline(always)]
pub fn hc2r_8<T: Float>(y: &[Complex<T>], x: &mut [T]) {
    debug_assert!(y.len() >= 5, "hc2r_8: input must have at least 5 elements");
    debug_assert_eq!(x.len(), 8, "hc2r_8: output must have exactly 8 elements");

    // Conjugate twiddles W8^(-k) = conj(W8^k):
    //   W8^1 = (1-i)/sqrt(2)  →  conj = (1+i)/sqrt(2)
    //   W8^2 = -i              →  conj = +i
    let sqrt2_inv = T::ONE / <T as Float>::sqrt(T::TWO);
    let w8_1c = Complex::new(sqrt2_inv, sqrt2_inv); // conj(W8^1)
    let w8_2c = Complex::new(T::ZERO, T::ONE); // conj(W8^2) = +i

    // ── Step 1: Recover E[k] and O[k] from the half-complex spectrum ──
    //
    // Conjugate symmetry: Y[8-k] = conj(Y[k])
    //   Y[5] = conj(Y[3]),  Y[6] = conj(Y[2]),  Y[7] = conj(Y[1])
    //
    // Unnormalized recovery:
    //   E[k] = Y[k] + Y[k+4]
    //   O[k] = conj(W8^k) * (Y[k] - Y[k+4])

    // k=0: Y[4] is stored (purely real for real input)
    let e0 = Complex::new(y[0].re + y[4].re, T::ZERO);
    let o0 = Complex::new(y[0].re - y[4].re, T::ZERO);

    // k=1: Y[5] = conj(Y[3])
    let y5 = y[3].conj();
    let e1 = y[1] + y5;
    let o1 = w8_1c * (y[1] - y5);

    // k=2: Y[6] = conj(Y[2])
    let y6 = y[2].conj();
    let e2 = y[2] + y6;
    let o2 = w8_2c * (y[2] - y6);

    // k=3 gives E[3]=conj(E[1]) and O[3]=conj(O[1]) — not needed explicitly.

    // ── Step 2: Invert size-4 DFT on even samples via HC2R-4 butterfly ──
    // Uses only E[0], E[1], E[2] (E[3]=conj(E[1]) by conjugate symmetry).
    // Butterfly: a=E[0].re+E[2].re, b=E[0].re-E[2].re, c=2*E[1].re, d=2*E[1].im
    //   xe[0]=a+c, xe[1]=b-d, xe[2]=a-c, xe[3]=b+d
    let ae = e0.re + e2.re;
    let be = e0.re - e2.re;
    let ce = e1.re + e1.re;
    let de = e1.im + e1.im;

    let xe0 = ae + ce;
    let xe1 = be - de;
    let xe2 = ae - ce;
    let xe3 = be + de;

    // ── Step 3: Invert size-4 DFT on odd samples via HC2R-4 butterfly ──
    let ao = o0.re + o2.re;
    let bo = o0.re - o2.re;
    let co = o1.re + o1.re;
    let do_ = o1.im + o1.im;

    let xo0 = ao + co;
    let xo1 = bo - do_;
    let xo2 = ao - co;
    let xo3 = bo + do_;

    // ── Step 4: Interleave even and odd samples ──
    // x[2k] = xe[k],  x[2k+1] = xo[k]
    x[0] = xe0;
    x[1] = xo0;
    x[2] = xe1;
    x[3] = xo1;
    x[4] = xe2;
    x[5] = xo2;
    x[6] = xe3;
    x[7] = xo3;
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn approx_eq_complex(a: Complex<f64>, b: Complex<f64>, tol: f64) -> bool {
        approx_eq(a.re, b.re, tol) && approx_eq(a.im, b.im, tol)
    }

    // ─── Reference DFT for verification ───

    fn naive_dft(x: &[f64]) -> Vec<Complex<f64>> {
        let n = x.len();
        (0..n)
            .map(|k| {
                x.iter().enumerate().fold(Complex::zero(), |acc, (j, &xj)| {
                    let angle = -2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
                    let w = Complex::new(angle.cos(), angle.sin());
                    acc + Complex::new(xj, 0.0) * w
                })
            })
            .collect()
    }

    fn naive_idft_unnorm(y: &[Complex<f64>], n: usize) -> Vec<f64> {
        // Reconstruct full spectrum from half-complex (conjugate symmetry)
        let mut full = vec![Complex::zero(); n];
        full[..y.len()].copy_from_slice(y);
        for k in y.len()..n {
            full[k] = full[n - k].conj();
        }
        // IDFT unnormalized (no /N)
        (0..n)
            .map(|j| {
                full.iter().enumerate().fold(0.0_f64, |acc, (k, &yk)| {
                    let angle = 2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
                    acc + yk.re * angle.cos() - yk.im * angle.sin()
                })
            })
            .collect()
    }

    // ─── R2HC size 2 ───

    #[test]
    fn test_r2hc_2_dc_only() {
        let x = [3.0_f64, 3.0];
        let mut y = [Complex::<f64>::zero(); 2];
        r2hc_2(&x, &mut y);
        assert!(approx_eq(y[0].re, 6.0, 1e-14), "Y[0].re={}", y[0].re);
        assert!(approx_eq(y[0].im, 0.0, 1e-14));
        assert!(approx_eq(y[1].re, 0.0, 1e-14), "Y[1].re={}", y[1].re);
        assert!(approx_eq(y[1].im, 0.0, 1e-14));
    }

    #[test]
    fn test_r2hc_2_matches_dft() {
        let x = [1.0_f64, 3.0];
        let mut y = [Complex::<f64>::zero(); 2];
        r2hc_2(&x, &mut y);
        let ref_y = naive_dft(&x);
        assert!(
            approx_eq_complex(y[0], ref_y[0], 1e-12),
            "Y[0]: {:?} vs {:?}",
            y[0],
            ref_y[0]
        );
        assert!(
            approx_eq_complex(y[1], ref_y[1], 1e-12),
            "Y[1]: {:?} vs {:?}",
            y[1],
            ref_y[1]
        );
    }

    // ─── HC2R size 2 ───

    #[test]
    fn test_hc2r_2_roundtrip() {
        let original = [1.0_f64, 3.0];
        let mut y = [Complex::<f64>::zero(); 2];
        r2hc_2(&original, &mut y);
        let mut recovered = [0.0_f64; 2];
        hc2r_2(&y, &mut recovered);
        for i in 0..2 {
            assert!(
                approx_eq(recovered[i] / 2.0, original[i], 1e-14),
                "idx={i}: recovered/2={}, original={}",
                recovered[i] / 2.0,
                original[i]
            );
        }
    }

    #[test]
    fn test_hc2r_2_against_naive_idft() {
        let original = [2.5_f64, -1.0];
        let mut y = [Complex::<f64>::zero(); 2];
        r2hc_2(&original, &mut y);
        let mut hc2r_out = [0.0_f64; 2];
        hc2r_2(&y, &mut hc2r_out);
        let naive_out = naive_idft_unnorm(&y, 2);
        for i in 0..2 {
            assert!(
                approx_eq(hc2r_out[i], naive_out[i], 1e-12),
                "idx={i}: hc2r={}, naive={}",
                hc2r_out[i],
                naive_out[i]
            );
        }
    }

    // ─── R2HC size 4 ───

    #[test]
    fn test_r2hc_4_matches_dft() {
        let x = [1.0_f64, 2.0, 3.0, 4.0];
        let mut y = [Complex::<f64>::zero(); 3];
        r2hc_4(&x, &mut y);
        let ref_y = naive_dft(&x);
        // Y[0] = 10
        assert!(approx_eq(y[0].re, 10.0, 1e-12), "Y[0].re={}", y[0].re);
        assert!(approx_eq(y[0].im, 0.0, 1e-14));
        // Y[1] = -2 + 2i
        assert!(approx_eq(y[1].re, -2.0, 1e-12), "Y[1].re={}", y[1].re);
        assert!(approx_eq(y[1].im, 2.0, 1e-12), "Y[1].im={}", y[1].im);
        // Y[2] = -2
        assert!(approx_eq(y[2].re, -2.0, 1e-12), "Y[2].re={}", y[2].re);
        assert!(approx_eq(y[2].im, 0.0, 1e-14));
        for k in 0..3 {
            assert!(
                approx_eq_complex(y[k], ref_y[k], 1e-12),
                "k={k}: codelet={:?}, naive={:?}",
                y[k],
                ref_y[k]
            );
        }
    }

    #[test]
    fn test_r2hc_4_dc_signal() {
        let x = [5.0_f64, 5.0, 5.0, 5.0];
        let mut y = [Complex::<f64>::zero(); 3];
        r2hc_4(&x, &mut y);
        assert!(approx_eq(y[0].re, 20.0, 1e-14));
        assert!(approx_eq(y[0].im, 0.0, 1e-14));
        assert!(approx_eq(y[1].re, 0.0, 1e-14));
        assert!(approx_eq(y[1].im, 0.0, 1e-14));
        assert!(approx_eq(y[2].re, 0.0, 1e-14));
        assert!(approx_eq(y[2].im, 0.0, 1e-14));
    }

    // ─── HC2R size 4 ───

    #[test]
    fn test_hc2r_4_roundtrip() {
        let original = [1.0_f64, 2.0, 3.0, 4.0];
        let mut y = [Complex::<f64>::zero(); 3];
        r2hc_4(&original, &mut y);
        let mut recovered = [0.0_f64; 4];
        hc2r_4(&y, &mut recovered);
        for i in 0..4 {
            assert!(
                approx_eq(recovered[i] / 4.0, original[i], 1e-12),
                "idx={i}: recovered/4={}, original={}",
                recovered[i] / 4.0,
                original[i]
            );
        }
    }

    #[test]
    fn test_hc2r_4_against_naive_idft() {
        let original = [7.0_f64, -2.0, 3.5, 1.0];
        let mut y = [Complex::<f64>::zero(); 3];
        r2hc_4(&original, &mut y);
        let mut hc2r_out = [0.0_f64; 4];
        hc2r_4(&y, &mut hc2r_out);
        let naive_out = naive_idft_unnorm(&y, 4);
        for i in 0..4 {
            assert!(
                approx_eq(hc2r_out[i], naive_out[i], 1e-11),
                "idx={i}: hc2r={}, naive={}",
                hc2r_out[i],
                naive_out[i]
            );
        }
    }

    // ─── R2HC size 8 ───

    #[test]
    fn test_r2hc_8_dc_signal() {
        let x = [1.0_f64; 8];
        let mut y = [Complex::<f64>::zero(); 5];
        r2hc_8(&x, &mut y);
        assert!(approx_eq(y[0].re, 8.0, 1e-14), "Y[0].re={}", y[0].re);
        assert!(approx_eq(y[0].im, 0.0, 1e-14));
        for k in 1..5 {
            assert!(approx_eq(y[k].re, 0.0, 1e-12), "Y[{k}].re={}", y[k].re);
            assert!(approx_eq(y[k].im, 0.0, 1e-12), "Y[{k}].im={}", y[k].im);
        }
    }

    #[test]
    fn test_r2hc_8_matches_dft() {
        let x: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let mut y = [Complex::<f64>::zero(); 5];
        r2hc_8(&x, &mut y);
        let ref_y = naive_dft(&x);
        for k in 0..5 {
            assert!(
                approx_eq_complex(y[k], ref_y[k], 1e-11),
                "k={k}: codelet={:?}, naive={:?}",
                y[k],
                ref_y[k]
            );
        }
    }

    #[test]
    fn test_r2hc_8_matches_dft_varied_input() {
        let x = [1.5_f64, -2.3, 0.7, 4.1, -1.0, 3.3, 2.2, -0.5];
        let mut y = [Complex::<f64>::zero(); 5];
        r2hc_8(&x, &mut y);
        let ref_y = naive_dft(&x);
        for k in 0..5 {
            assert!(
                approx_eq_complex(y[k], ref_y[k], 1e-11),
                "k={k}: codelet={:?}, naive={:?}",
                y[k],
                ref_y[k]
            );
        }
    }

    #[test]
    fn test_r2hc_8_dc_and_nyquist_purely_real() {
        let x = [2.0_f64, -1.0, 0.5, 3.0, -2.5, 1.5, 0.0, -0.5];
        let mut y = [Complex::<f64>::zero(); 5];
        r2hc_8(&x, &mut y);
        assert!(
            approx_eq(y[0].im, 0.0, 1e-13),
            "DC should be real, im={}",
            y[0].im
        );
        assert!(
            approx_eq(y[4].im, 0.0, 1e-13),
            "Nyquist should be real, im={}",
            y[4].im
        );
    }

    // ─── HC2R size 8 ───

    #[test]
    fn test_hc2r_8_roundtrip() {
        let original = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut y = [Complex::<f64>::zero(); 5];
        r2hc_8(&original, &mut y);
        let mut recovered = [0.0_f64; 8];
        hc2r_8(&y, &mut recovered);
        for i in 0..8 {
            assert!(
                approx_eq(recovered[i] / 8.0, original[i], 1e-11),
                "idx={i}: recovered/8={}, original={}",
                recovered[i] / 8.0,
                original[i]
            );
        }
    }

    #[test]
    fn test_hc2r_8_roundtrip_varied() {
        let original = [1.5_f64, -2.3, 0.7, 4.1, -1.0, 3.3, 2.2, -0.5];
        let mut y = [Complex::<f64>::zero(); 5];
        r2hc_8(&original, &mut y);
        let mut recovered = [0.0_f64; 8];
        hc2r_8(&y, &mut recovered);
        for i in 0..8 {
            assert!(
                approx_eq(recovered[i] / 8.0, original[i], 1e-11),
                "idx={i}: recovered/8={}, original={}",
                recovered[i] / 8.0,
                original[i]
            );
        }
    }

    #[test]
    fn test_hc2r_8_against_naive_idft() {
        let original: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let mut y = [Complex::<f64>::zero(); 5];
        r2hc_8(&original, &mut y);
        let mut hc2r_out = [0.0_f64; 8];
        hc2r_8(&y, &mut hc2r_out);
        let naive_out = naive_idft_unnorm(&y, 8);
        for i in 0..8 {
            assert!(
                approx_eq(hc2r_out[i], naive_out[i], 1e-10),
                "idx={i}: hc2r={}, naive={}",
                hc2r_out[i],
                naive_out[i]
            );
        }
    }

    // ─── f32 smoke tests ───

    #[test]
    fn test_r2hc_2_f32() {
        let x = [1.0_f32, 3.0];
        let mut y = [Complex::<f32>::zero(); 2];
        r2hc_2(&x, &mut y);
        assert!((y[0].re - 4.0_f32).abs() < 1e-6, "Y[0].re={}", y[0].re);
        assert!((y[1].re - (-2.0_f32)).abs() < 1e-6, "Y[1].re={}", y[1].re);
    }

    #[test]
    fn test_r2hc_4_f32() {
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let mut y = [Complex::<f32>::zero(); 3];
        r2hc_4(&x, &mut y);
        assert!((y[0].re - 10.0_f32).abs() < 1e-5, "Y[0].re={}", y[0].re);
        assert!((y[1].re - (-2.0_f32)).abs() < 1e-5, "Y[1].re={}", y[1].re);
        assert!((y[1].im - 2.0_f32).abs() < 1e-5, "Y[1].im={}", y[1].im);
    }

    #[test]
    fn test_r2hc_8_f32() {
        let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut y = [Complex::<f32>::zero(); 5];
        r2hc_8(&x, &mut y);
        // DC = 0+1+2+...+7 = 28
        assert!((y[0].re - 28.0_f32).abs() < 1e-4, "Y[0].re={}", y[0].re);
        assert!(y[0].im.abs() < 1e-5, "DC should be real, im={}", y[0].im);
    }
}
