// Numerical correctness tests for gen_notw_codelet! macro output.
//
// Compares generated codelets against a naïve O(n²) DFT for sizes 2, 4, 8
// in both f32 and f64 precision.
//
// NOTE: The generated code references `crate::kernel::Float` and
// `crate::kernel::Complex<T>`.  Since adding `oxifft` as a dev-dep here would
// create a circular dependency (oxifft → oxifft-codegen), we re-export the
// required kernel types through the `kernel` module below.
//
// Clippy lints suppressed in this test file:
//   - cast_precision_loss: usize/i32→f64 casts for small DFT indices (n ≤ 8) are exact.
//   - approx_constant: generated code uses explicit f64 literals close to π/√2.
//   - assign_op_pattern: generated code uses explicit `a = a + b` for clarity.
//   - derive_partial_eq_without_eq: stub Complex<T> is PartialEq only (Float is not Eq).
//   - missing_const_for_fn: zero() cannot be const due to trait bound.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::approx_constant,
    clippy::assign_op_pattern,
    clippy::derive_partial_eq_without_eq,
    clippy::missing_const_for_fn,
    clippy::suboptimal_flops
)]

use oxifft_codegen::gen_notw_codelet;

// ============================================================================
// Minimal kernel stub — mirrors the types / impls the generated code needs.
// ============================================================================

pub mod kernel {
    use core::fmt;
    use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

    pub trait Float:
        Copy
        + Clone
        + Default
        + fmt::Debug
        + Send
        + Sync
        + PartialOrd
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
        + Neg<Output = Self>
        + num_traits::NumAssign
        + num_traits::Float
        + num_traits::FloatConst
        + 'static
    {
        const ZERO: Self;
        const ONE: Self;
        const TWO: Self;
        const PI: Self;
        const TWO_PI: Self;

        #[must_use]
        fn sin(self) -> Self;
        #[must_use]
        fn cos(self) -> Self;
        #[must_use]
        fn sin_cos(self) -> (Self, Self);
        #[must_use]
        fn sqrt(self) -> Self;
        #[must_use]
        fn abs(self) -> Self;
        #[must_use]
        fn from_usize(n: usize) -> Self;
        #[must_use]
        fn from_isize(n: isize) -> Self;
        #[must_use]
        fn from_f64(n: f64) -> Self;
    }

    impl Float for f32 {
        const ZERO: Self = 0.0;
        const ONE: Self = 1.0;
        const TWO: Self = 2.0;
        const PI: Self = core::f32::consts::PI;
        const TWO_PI: Self = core::f32::consts::TAU;

        fn sin(self) -> Self {
            num_traits::Float::sin(self)
        }
        fn cos(self) -> Self {
            num_traits::Float::cos(self)
        }
        fn sin_cos(self) -> (Self, Self) {
            num_traits::Float::sin_cos(self)
        }
        fn sqrt(self) -> Self {
            num_traits::Float::sqrt(self)
        }
        fn abs(self) -> Self {
            num_traits::Float::abs(self)
        }
        fn from_usize(n: usize) -> Self {
            n as Self
        }
        fn from_isize(n: isize) -> Self {
            n as Self
        }
        fn from_f64(n: f64) -> Self {
            n as Self
        }
    }

    impl Float for f64 {
        const ZERO: Self = 0.0;
        const ONE: Self = 1.0;
        const TWO: Self = 2.0;
        const PI: Self = core::f64::consts::PI;
        const TWO_PI: Self = core::f64::consts::TAU;

        fn sin(self) -> Self {
            num_traits::Float::sin(self)
        }
        fn cos(self) -> Self {
            num_traits::Float::cos(self)
        }
        fn sin_cos(self) -> (Self, Self) {
            num_traits::Float::sin_cos(self)
        }
        fn sqrt(self) -> Self {
            num_traits::Float::sqrt(self)
        }
        fn abs(self) -> Self {
            num_traits::Float::abs(self)
        }
        fn from_usize(n: usize) -> Self {
            n as Self
        }
        fn from_isize(n: isize) -> Self {
            n as Self
        }
        fn from_f64(n: f64) -> Self {
            n
        }
    }

    #[derive(Copy, Clone, Default, PartialEq)]
    #[repr(C)]
    pub struct Complex<T: Float> {
        pub re: T,
        pub im: T,
    }

    impl<T: Float> Complex<T> {
        #[inline]
        pub const fn new(re: T, im: T) -> Self {
            Self { re, im }
        }

        #[inline]
        pub fn zero() -> Self {
            Self::new(T::ZERO, T::ZERO)
        }
    }

    impl<T: Float> fmt::Debug for Complex<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{:?}+{:?}i", self.re, self.im)
        }
    }

    impl<T: Float> Add for Complex<T> {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self::new(self.re + rhs.re, self.im + rhs.im)
        }
    }
    impl<T: Float> AddAssign for Complex<T> {
        fn add_assign(&mut self, rhs: Self) {
            self.re += rhs.re;
            self.im += rhs.im;
        }
    }
    impl<T: Float> Sub for Complex<T> {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self::new(self.re - rhs.re, self.im - rhs.im)
        }
    }
    impl<T: Float> SubAssign for Complex<T> {
        fn sub_assign(&mut self, rhs: Self) {
            self.re -= rhs.re;
            self.im -= rhs.im;
        }
    }
    impl<T: Float> Mul for Complex<T> {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self::new(
                self.re * rhs.re - self.im * rhs.im,
                self.re * rhs.im + self.im * rhs.re,
            )
        }
    }
    impl<T: Float> MulAssign for Complex<T> {
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }
    impl<T: Float> Div for Complex<T> {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            let norm_sq = rhs.re * rhs.re + rhs.im * rhs.im;
            Self::new(
                (self.re * rhs.re + self.im * rhs.im) / norm_sq,
                (self.im * rhs.re - self.re * rhs.im) / norm_sq,
            )
        }
    }
    impl<T: Float> DivAssign for Complex<T> {
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }
    impl<T: Float> Neg for Complex<T> {
        type Output = Self;
        fn neg(self) -> Self {
            Self::new(-self.re, -self.im)
        }
    }
    impl<T: Float> Mul<T> for Complex<T> {
        type Output = Self;
        fn mul(self, rhs: T) -> Self {
            Self::new(self.re * rhs, self.im * rhs)
        }
    }
}

// ============================================================================
// Generate the codelets under test (proc-macro invocations).
// Each expands to `pub fn codelet_notw_N<T: crate::kernel::Float>`.
// ============================================================================

gen_notw_codelet!(2);
gen_notw_codelet!(4);
gen_notw_codelet!(8);

// ============================================================================
// Naïve O(n²) DFT reference.
// sign = -1 → forward (e^{-2πi·jk/N}), sign = +1 → inverse (e^{+2πi·jk/N}).
// ============================================================================

fn dft_naive_f64(input: &[kernel::Complex<f64>], sign: i32) -> Vec<kernel::Complex<f64>> {
    let n = input.len();
    let n_f = n as f64;
    (0..n)
        .map(|k| {
            input
                .iter()
                .enumerate()
                .fold(kernel::Complex::new(0.0_f64, 0.0), |acc, (j, &x)| {
                    let angle =
                        f64::from(sign) * 2.0 * core::f64::consts::PI * (j * k) as f64 / n_f;
                    let w = kernel::Complex::new(angle.cos(), angle.sin());
                    // Complex multiplication: (acc + x*w)
                    kernel::Complex::new(
                        x.im.mul_add(-w.im, x.re.mul_add(w.re, acc.re)),
                        x.im.mul_add(w.re, x.re.mul_add(w.im, acc.im)),
                    )
                })
        })
        .collect()
}

fn dft_naive_f32(input: &[kernel::Complex<f32>], sign: i32) -> Vec<kernel::Complex<f32>> {
    // Use f64 arithmetic internally for precision, convert back at the end.
    let n = input.len();
    let n_f = n as f64;
    (0..n)
        .map(|k| {
            let (re, im) =
                input
                    .iter()
                    .enumerate()
                    .fold((0.0_f64, 0.0_f64), |(acc_re, acc_im), (j, &x)| {
                        let angle =
                            f64::from(sign) * 2.0 * core::f64::consts::PI * (j * k) as f64 / n_f;
                        let (ws, wc) = angle.sin_cos();
                        let x_re = f64::from(x.re);
                        let x_im = f64::from(x.im);
                        (
                            (-x_im).mul_add(ws, x_re.mul_add(wc, acc_re)),
                            x_im.mul_add(wc, x_re.mul_add(ws, acc_im)),
                        )
                    });
            kernel::Complex::new(re as f32, im as f32)
        })
        .collect()
}

// ============================================================================
// Tolerance checkers
// ============================================================================

fn check_close_f64(
    got: &[kernel::Complex<f64>],
    expected: &[kernel::Complex<f64>],
    tol: f64,
    label: &str,
) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let err_re = (g.re - e.re).abs();
        let err_im = (g.im - e.im).abs();
        assert!(
            err_re < tol && err_im < tol,
            "{label}[{i}]: got {g:?}, expected {e:?}, err=({err_re}, {err_im}) >= tol={tol}"
        );
    }
}

fn check_close_f32(
    got: &[kernel::Complex<f32>],
    expected: &[kernel::Complex<f32>],
    tol: f32,
    label: &str,
) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let err_re = (g.re - e.re).abs();
        let err_im = (g.im - e.im).abs();
        assert!(
            err_re < tol && err_im < tol,
            "{label}[{i}]: got {g:?}, expected {e:?}, err=({err_re}, {err_im}) >= tol={tol}"
        );
    }
}

// ============================================================================
// Small deterministic LCG for reproducible pseudo-random inputs
// ============================================================================

fn lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    // Map [0, 2^53) to [-1, 1) using only the top 53 bits to stay in the f64 mantissa range.
    let mantissa_bits = *state >> 11; // 53 bits
    let scale = 1.0_f64 / (1u64 << 52) as f64; // 2^-52
    mantissa_bits as f64 * scale - 1.0
}

fn make_input_f64(n: usize, seed: u64) -> Vec<kernel::Complex<f64>> {
    let mut s = seed;
    (0..n)
        .map(|_| kernel::Complex::new(lcg_next(&mut s), lcg_next(&mut s)))
        .collect()
}

fn make_input_f32(n: usize, seed: u64) -> Vec<kernel::Complex<f32>> {
    let mut s = seed;
    (0..n)
        .map(|_| kernel::Complex::new(lcg_next(&mut s) as f32, lcg_next(&mut s) as f32))
        .collect()
}

// ============================================================================
// Tests — size-2 × f64
// ============================================================================

#[test]
fn notw_2_f64_forward_vs_naive() {
    let input = make_input_f64(2, 0xDEAD_BEEF_0001);
    let expected = dft_naive_f64(&input, -1);
    let mut got = input;
    codelet_notw_2(&mut got, -1);
    check_close_f64(&got, &expected, 1e-12, "notw_2_f64_fwd");
}

#[test]
fn notw_2_f64_inverse_vs_naive() {
    let input = make_input_f64(2, 0xDEAD_BEEF_0002);
    let expected = dft_naive_f64(&input, 1);
    let mut got = input;
    codelet_notw_2(&mut got, 1);
    check_close_f64(&got, &expected, 1e-12, "notw_2_f64_inv");
}

#[test]
fn notw_2_f64_roundtrip() {
    let original = make_input_f64(2, 0xDEAD_BEEF_0003);
    let mut data = original.clone();
    codelet_notw_2(&mut data, -1);
    codelet_notw_2(&mut data, 1);
    for x in &mut data {
        x.re /= 2.0;
        x.im /= 2.0;
    }
    check_close_f64(&data, &original, 1e-12, "notw_2_f64_roundtrip");
}

// ============================================================================
// Tests — size-2 × f32
// ============================================================================

#[test]
fn notw_2_f32_forward_vs_naive() {
    let input = make_input_f32(2, 0xCAFE_BABE_0001);
    let expected = dft_naive_f32(&input, -1);
    let mut got = input;
    codelet_notw_2(&mut got, -1);
    check_close_f32(&got, &expected, 1e-5, "notw_2_f32_fwd");
}

#[test]
fn notw_2_f32_inverse_vs_naive() {
    let input = make_input_f32(2, 0xCAFE_BABE_0002);
    let expected = dft_naive_f32(&input, 1);
    let mut got = input;
    codelet_notw_2(&mut got, 1);
    check_close_f32(&got, &expected, 1e-5, "notw_2_f32_inv");
}

#[test]
fn notw_2_f32_roundtrip() {
    let original = make_input_f32(2, 0xCAFE_BABE_0003);
    let mut data = original.clone();
    codelet_notw_2(&mut data, -1);
    codelet_notw_2(&mut data, 1);
    for x in &mut data {
        x.re /= 2.0;
        x.im /= 2.0;
    }
    check_close_f32(&data, &original, 1e-5, "notw_2_f32_roundtrip");
}

// ============================================================================
// Tests — size-4 × f64
// ============================================================================

#[test]
fn notw_4_f64_forward_vs_naive() {
    let input = make_input_f64(4, 0xDEAD_BEEF_0011);
    let expected = dft_naive_f64(&input, -1);
    let mut got = input;
    codelet_notw_4(&mut got, -1);
    check_close_f64(&got, &expected, 1e-12, "notw_4_f64_fwd");
}

#[test]
fn notw_4_f64_inverse_vs_naive() {
    let input = make_input_f64(4, 0xDEAD_BEEF_0012);
    let expected = dft_naive_f64(&input, 1);
    let mut got = input;
    codelet_notw_4(&mut got, 1);
    check_close_f64(&got, &expected, 1e-12, "notw_4_f64_inv");
}

#[test]
fn notw_4_f64_roundtrip() {
    let original = make_input_f64(4, 0xDEAD_BEEF_0013);
    let mut data = original.clone();
    codelet_notw_4(&mut data, -1);
    codelet_notw_4(&mut data, 1);
    for x in &mut data {
        x.re /= 4.0;
        x.im /= 4.0;
    }
    check_close_f64(&data, &original, 1e-12, "notw_4_f64_roundtrip");
}

// ============================================================================
// Tests — size-4 × f32
// ============================================================================

#[test]
fn notw_4_f32_forward_vs_naive() {
    let input = make_input_f32(4, 0xCAFE_BABE_0011);
    let expected = dft_naive_f32(&input, -1);
    let mut got = input;
    codelet_notw_4(&mut got, -1);
    check_close_f32(&got, &expected, 1e-5, "notw_4_f32_fwd");
}

#[test]
fn notw_4_f32_inverse_vs_naive() {
    let input = make_input_f32(4, 0xCAFE_BABE_0012);
    let expected = dft_naive_f32(&input, 1);
    let mut got = input;
    codelet_notw_4(&mut got, 1);
    check_close_f32(&got, &expected, 1e-5, "notw_4_f32_inv");
}

#[test]
fn notw_4_f32_roundtrip() {
    let original = make_input_f32(4, 0xCAFE_BABE_0013);
    let mut data = original.clone();
    codelet_notw_4(&mut data, -1);
    codelet_notw_4(&mut data, 1);
    for x in &mut data {
        x.re /= 4.0;
        x.im /= 4.0;
    }
    check_close_f32(&data, &original, 1e-5, "notw_4_f32_roundtrip");
}

// ============================================================================
// Tests — size-8 × f64
// ============================================================================

#[test]
fn notw_8_f64_forward_vs_naive() {
    let input = make_input_f64(8, 0xDEAD_BEEF_0021);
    let expected = dft_naive_f64(&input, -1);
    let mut got = input;
    codelet_notw_8(&mut got, -1);
    check_close_f64(&got, &expected, 1e-12, "notw_8_f64_fwd");
}

#[test]
fn notw_8_f64_inverse_vs_naive() {
    let input = make_input_f64(8, 0xDEAD_BEEF_0022);
    let expected = dft_naive_f64(&input, 1);
    let mut got = input;
    codelet_notw_8(&mut got, 1);
    check_close_f64(&got, &expected, 1e-12, "notw_8_f64_inv");
}

#[test]
fn notw_8_f64_roundtrip() {
    let original = make_input_f64(8, 0xDEAD_BEEF_0023);
    let mut data = original.clone();
    codelet_notw_8(&mut data, -1);
    codelet_notw_8(&mut data, 1);
    for x in &mut data {
        x.re /= 8.0;
        x.im /= 8.0;
    }
    check_close_f64(&data, &original, 1e-12, "notw_8_f64_roundtrip");
}

// ============================================================================
// Tests — size-8 × f32
// ============================================================================

#[test]
fn notw_8_f32_forward_vs_naive() {
    let input = make_input_f32(8, 0xCAFE_BABE_0021);
    let expected = dft_naive_f32(&input, -1);
    let mut got = input;
    codelet_notw_8(&mut got, -1);
    check_close_f32(&got, &expected, 1e-5, "notw_8_f32_fwd");
}

#[test]
fn notw_8_f32_inverse_vs_naive() {
    let input = make_input_f32(8, 0xCAFE_BABE_0022);
    let expected = dft_naive_f32(&input, 1);
    let mut got = input;
    codelet_notw_8(&mut got, 1);
    check_close_f32(&got, &expected, 1e-5, "notw_8_f32_inv");
}

#[test]
fn notw_8_f32_roundtrip() {
    let original = make_input_f32(8, 0xCAFE_BABE_0023);
    let mut data = original.clone();
    codelet_notw_8(&mut data, -1);
    codelet_notw_8(&mut data, 1);
    for x in &mut data {
        x.re /= 8.0;
        x.im /= 8.0;
    }
    check_close_f32(&data, &original, 1e-5, "notw_8_f32_roundtrip");
}
