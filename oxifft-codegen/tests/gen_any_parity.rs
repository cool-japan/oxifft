//! Numerical parity tests for `gen_any_codelet!` output.
//!
//! `gen_any_codelet!(N)` classifies `N` and routes it to the appropriate emitter.
//! For the self-contained classes (direct NOTW {2,4,8,16,32,64}, Winograd-odd
//! {3,5,7}, hardcoded Rader {11,13}, and the identity {1}) the generated codelet
//! depends only on the `crate::kernel` contract and can be compiled and executed
//! here — so we do exactly that and compare against a naive O(n^2) DFT for both
//! `f32` and `f64`, forward and inverse.
//!
//! The runtime-wrapper classes (smooth-7 `MixedRadix`, runtime `RaderPrime`, and
//! `Bluestein`) additionally delegate to `::oxifft`'s `Plan::dft_1d`, which cannot
//! be linked here without a circular dependency; those are exercised from the
//! `oxifft` crate instead (see the crate-level test plan).
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::approx_constant,
    clippy::assign_op_pattern,
    clippy::suboptimal_flops,
    clippy::derive_partial_eq_without_eq,
    clippy::missing_const_for_fn
)]

use oxifft_codegen::gen_any_codelet;

// The `crate::kernel` contract, satisfied by the reference implementation.
mod kernel {
    pub use oxifft_codegen_impl::kernel_contract::{Complex, Float};
}

use kernel::Complex;

// ---------------------------------------------------------------------------
// Generate the codelets under test through the real `gen_any_codelet!` macro.
// Hardcoded classes emit `codelet_notw_{N}`; the identity emits `codelet_any_1`.
// ---------------------------------------------------------------------------

gen_any_codelet!(1); // identity -> codelet_any_1
gen_any_codelet!(2); // direct   -> codelet_notw_2
gen_any_codelet!(3); // odd      -> codelet_notw_3
gen_any_codelet!(4); // direct   -> codelet_notw_4
gen_any_codelet!(5); // odd      -> codelet_notw_5
gen_any_codelet!(7); // odd      -> codelet_notw_7
gen_any_codelet!(8); // direct   -> codelet_notw_8
gen_any_codelet!(11); // rader    -> codelet_notw_11
gen_any_codelet!(13); // rader    -> codelet_notw_13
gen_any_codelet!(16); // direct   -> codelet_notw_16
gen_any_codelet!(32); // direct   -> codelet_notw_32
gen_any_codelet!(64); // direct   -> codelet_notw_64

// ---------------------------------------------------------------------------
// Naive O(n^2) DFT reference (computed in f64).
// sign = -1 -> forward (e^{-2 pi i j k / N}); sign = +1 -> inverse (unnormalized).
// ---------------------------------------------------------------------------

fn dft_naive(input: &[Complex<f64>], sign: i32) -> Vec<Complex<f64>> {
    let n = input.len();
    let n_f = n as f64;
    (0..n)
        .map(|k| {
            let mut acc = Complex::<f64>::new(0.0, 0.0);
            for (j, &x) in input.iter().enumerate() {
                let angle = f64::from(sign) * 2.0 * core::f64::consts::PI * (j * k) as f64 / n_f;
                let (s, c) = angle.sin_cos();
                acc.re += x.re * c - x.im * s;
                acc.im += x.re * s + x.im * c;
            }
            acc
        })
        .collect()
}

fn make_input_f64(n: usize) -> Vec<Complex<f64>> {
    // Deterministic, non-trivial input.
    (0..n)
        .map(|j| {
            let jr = j as f64;
            Complex::new(
                0.5 + jr * 0.75 - (jr * 0.3).sin(),
                -0.25 + (jr * 0.9).cos() * 1.5,
            )
        })
        .collect()
}

fn to_f32(v: &[Complex<f64>]) -> Vec<Complex<f32>> {
    v.iter()
        .map(|c| Complex::new(c.re as f32, c.im as f32))
        .collect()
}

fn check_f64(got: &[Complex<f64>], expected: &[Complex<f64>], tol: f64, label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        let dr = (g.re - e.re).abs();
        let di = (g.im - e.im).abs();
        assert!(
            dr < tol && di < tol,
            "{label}[{i}]: got {g:?}, expected {e:?}, err=({dr},{di}) >= {tol}"
        );
    }
}

fn check_f32(got: &[Complex<f32>], expected: &[Complex<f64>], tol: f64, label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        let dr = (f64::from(g.re) - e.re).abs();
        let di = (f64::from(g.im) - e.im).abs();
        assert!(
            dr < tol && di < tol,
            "{label}[{i}]: got {g:?}, expected {e:?}, err=({dr},{di}) >= {tol}"
        );
    }
}

/// Run the size-`n` codelet (both precisions, both directions) and compare to the
/// naive DFT reference.
fn run_case(
    n: usize,
    codelet_f64: fn(&mut [Complex<f64>], i32),
    codelet_f32: fn(&mut [Complex<f32>], i32),
) {
    let input64 = make_input_f64(n);
    let input32 = to_f32(&input64);

    for &sign in &[-1_i32, 1] {
        let expected = dft_naive(&input64, sign);

        let mut x64 = input64.clone();
        codelet_f64(&mut x64, sign);
        check_f64(&x64, &expected, 1e-9, &format!("size {n} f64 sign {sign}"));

        let mut x32 = input32.clone();
        codelet_f32(&mut x32, sign);
        // Larger sizes accumulate more f32 rounding; 1e-2 is generous but catches
        // any structural (sign-branch) error.
        let tol = if n >= 32 { 2e-2 } else { 5e-3 };
        check_f32(&x32, &expected, tol, &format!("size {n} f32 sign {sign}"));
    }
}

macro_rules! parity_test {
    ($test:ident, $n:literal, $fn:ident) => {
        #[test]
        fn $test() {
            run_case($n, $fn::<f64>, $fn::<f32>);
        }
    };
}

parity_test!(any_2_vs_naive, 2, codelet_notw_2);
parity_test!(any_3_vs_naive, 3, codelet_notw_3);
parity_test!(any_4_vs_naive, 4, codelet_notw_4);
parity_test!(any_5_vs_naive, 5, codelet_notw_5);
parity_test!(any_7_vs_naive, 7, codelet_notw_7);
parity_test!(any_8_vs_naive, 8, codelet_notw_8);
parity_test!(any_11_vs_naive, 11, codelet_notw_11);
parity_test!(any_13_vs_naive, 13, codelet_notw_13);
parity_test!(any_16_vs_naive, 16, codelet_notw_16);
parity_test!(any_32_vs_naive, 32, codelet_notw_32);
parity_test!(any_64_vs_naive, 64, codelet_notw_64);

#[test]
fn any_1_identity() {
    // The identity codelet leaves the single element unchanged for either sign.
    for &sign in &[-1_i32, 1] {
        let mut x = vec![Complex::<f64>::new(3.25, -1.5)];
        codelet_any_1(&mut x, sign);
        assert_eq!(x[0], Complex::new(3.25, -1.5));
    }
}
