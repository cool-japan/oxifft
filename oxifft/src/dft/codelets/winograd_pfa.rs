//! Prime Factor Algorithm (PFA) composer for coprime-size DFTs.
//!
//! The PFA computes DFT-N for N = N1 × N2 where gcd(N1, N2) = 1 by mapping
//! a 1-D DFT into a 2-D DFT via Good's index mapping:
//!
//! - Input map:  n  = (n1·N2·inv(N2, N1) + n2·N1·inv(N1, N2)) mod N
//! - Output map: k  = (k1·N2              + k2·N1             ) mod N
//!
//! where inv(a, m) is the modular inverse of a modulo m.
//!
//! This enables mixed-size transforms without twiddle factors:
//!   DFT-15 = PFA(DFT-3,  DFT-5)
//!   DFT-21 = PFA(DFT-3,  DFT-7)
//!   DFT-35 = PFA(DFT-5,  DFT-7)
//!
//! The kernel functions passed in must follow the same signature as the other
//! Winograd kernels: `fn(&mut [Complex<T>], sign: i32)`.

use crate::kernel::{Complex, Float};

/// Compute the modular inverse of `a` modulo `m` via extended Euclidean algorithm.
///
/// Returns `Some(x)` where `a*x ≡ 1 (mod m)`, or `None` if gcd(a, m) != 1.
pub fn mod_inv(a: usize, m: usize) -> Option<usize> {
    if m <= 1 {
        return None;
    }
    let (mut old_r, mut r) = (a as isize, m as isize);
    let (mut old_s, mut s) = (1_isize, 0_isize);

    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
    }

    // old_r is gcd
    if old_r != 1 {
        return None;
    }

    // Bring old_s into the range [0, m)
    let result = old_s.rem_euclid(m as isize) as usize;
    Some(result)
}

/// Apply the PFA (Good's index mapping) to compute DFT-N for N = n1 × n2 with gcd(n1,n2)=1.
///
/// # Arguments
/// * `input`   — read-only source slice of length N = n1 * n2
/// * `output`  — destination slice of length N (will be overwritten)
/// * `n1`, `n2` — coprime factor pair; N1 × N2 must equal `input.len()`
/// * `kernel1` — in-place DFT kernel for size `n1` (sign convention: sign < 0 for forward)
/// * `kernel2` — in-place DFT kernel for size `n2`
/// * `sign`    — −1 for forward DFT, +1 for inverse DFT
///
/// # Panics
/// Panics if `n1 * n2 != input.len()` or gcd(n1, n2) != 1 (in debug builds).
pub fn pfa_compose<T, K1, K2>(
    input: &[Complex<T>],
    output: &mut [Complex<T>],
    n1: usize,
    n2: usize,
    kernel1: K1,
    kernel2: K2,
    sign: i32,
) where
    T: Float,
    K1: Fn(&mut [Complex<T>], i32),
    K2: Fn(&mut [Complex<T>], i32),
{
    let n = n1 * n2;
    debug_assert_eq!(input.len(), n, "input length must equal n1*n2");
    debug_assert_eq!(output.len(), n, "output length must equal n1*n2");

    // Compute modular inverses: inv(N2, N1) and inv(N1, N2)
    // These are needed for Good's input index mapping.
    // For the sizes used in the planner (3,5,7,15,21,35) these always exist.
    let inv_n2_mod_n1 =
        mod_inv(n2 % n1, n1).expect("n2 must be invertible modulo n1 (gcd must be 1)");
    let inv_n1_mod_n2 =
        mod_inv(n1 % n2, n2).expect("n1 must be invertible modulo n2 (gcd must be 1)");

    // Allocate 2-D working array (n1 rows × n2 columns)
    let mut work = vec![Complex::<T>::zero(); n];

    // Step 1: Input permutation via Good's mapping.
    // For each (n1_idx, n2_idx), the input sample at linear index
    //   n = (n1_idx * n2 * inv_n2_mod_n1 + n2_idx * n1 * inv_n1_mod_n2) mod N
    // is placed at work[n1_idx * n2 + n2_idx].
    for n1_idx in 0..n1 {
        for n2_idx in 0..n2 {
            let src_idx = (n1_idx * n2 * inv_n2_mod_n1 + n2_idx * n1 * inv_n1_mod_n2) % n;
            work[n1_idx * n2 + n2_idx] = input[src_idx];
        }
    }

    // Step 2: Apply kernel2 (size n2) to each row (n1 rows).
    for row in 0..n1 {
        let start = row * n2;
        kernel2(&mut work[start..start + n2], sign);
    }

    // Step 3: Apply kernel1 (size n1) to each column (n2 columns).
    // Columns are not contiguous; extract, transform, scatter back.
    let mut col_buf = vec![Complex::<T>::zero(); n1];
    for col in 0..n2 {
        for row in 0..n1 {
            col_buf[row] = work[row * n2 + col];
        }
        kernel1(&mut col_buf, sign);
        for row in 0..n1 {
            work[row * n2 + col] = col_buf[row];
        }
    }

    // Step 4: Output permutation — inverse of Good's output mapping.
    // Output frequency k = (k1 * n2 + k2 * n1) mod N maps from work[k1 * n2 + k2].
    for k1 in 0..n1 {
        for k2 in 0..n2 {
            let dst_idx = (k1 * n2 + k2 * n1) % n;
            output[dst_idx] = work[k1 * n2 + k2];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::winograd::{winograd_3, winograd_5, winograd_7};
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

    fn pfa_test(
        n1: usize,
        n2: usize,
        k1: impl Fn(&mut [Complex<f64>], i32),
        k2: impl Fn(&mut [Complex<f64>], i32),
    ) {
        let n = n1 * n2;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i + 1) as f64 * 0.25, (i as f64 * 0.6).cos()))
            .collect();
        let tol = 1e-10 * n as f64;

        // Forward
        let expected = naive_dft(&input, -1);
        let mut output = vec![Complex::new(0.0, 0.0); n];
        pfa_compose(&input, &mut output, n1, n2, &k1, &k2, -1);
        check_eq(&output, &expected, tol, &format!("pfa_fwd({n1}x{n2})"));

        // Roundtrip
        let mut rt = vec![Complex::new(0.0, 0.0); n];
        pfa_compose(&output, &mut rt, n1, n2, &k1, &k2, 1);
        let n_f = n as f64;
        for v in &mut rt {
            *v = Complex::new(v.re / n_f, v.im / n_f);
        }
        check_eq(&rt, &input, tol, &format!("pfa_roundtrip({n1}x{n2})"));
    }

    #[test]
    fn test_pfa_15() {
        pfa_test(3, 5, winograd_3, winograd_5);
    }

    #[test]
    fn test_pfa_21() {
        pfa_test(3, 7, winograd_3, winograd_7);
    }

    #[test]
    fn test_pfa_35() {
        pfa_test(5, 7, winograd_5, winograd_7);
    }

    #[test]
    fn test_mod_inv() {
        assert_eq!(mod_inv(2, 5), Some(3)); // 2*3=6≡1 mod 5
        assert_eq!(mod_inv(3, 5), Some(2)); // 3*2=6≡1 mod 5
        assert_eq!(mod_inv(2, 3), Some(2)); // 2*2=4≡1 mod 3
        assert_eq!(mod_inv(5, 3), Some(2)); // 5%3=2, 2*2=4≡1 mod 3
        assert_eq!(mod_inv(2, 4), None); // gcd(2,4)=2 ≠ 1
        assert_eq!(mod_inv(1, 1), None); // m=1
    }
}
