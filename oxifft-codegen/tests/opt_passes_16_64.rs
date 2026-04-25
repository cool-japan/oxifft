// Numerical correctness tests for symbolically-generated codelets: sizes 16, 32, 64.
//
// These tests verify that `gen_notw_codelet!` for sizes 16/32/64 (now generated via
// the symbolic CSE/constant-folding pipeline) produces numerically correct output
// compared to a naïve O(n²) DFT reference.
//
// The `crate::kernel` module mirrors the stub in `tests/notw_small_sizes.rs` since
// we cannot add `oxifft` as a dev-dep (that would be circular).
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
// Minimal kernel stub — identical to the one in notw_small_sizes.rs
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
// Generate the codelets under test via the symbolic optimization pipeline.
// ============================================================================

gen_notw_codelet!(16);
gen_notw_codelet!(32);
gen_notw_codelet!(64);

// ============================================================================
// Naïve O(n²) DFT reference.
// sign = -1 → forward, sign = +1 → inverse.
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
                    kernel::Complex::new(
                        x.im.mul_add(-w.im, x.re.mul_add(w.re, acc.re)),
                        x.im.mul_add(w.re, x.re.mul_add(w.im, acc.im)),
                    )
                })
        })
        .collect()
}

fn dft_naive_f32(input: &[kernel::Complex<f32>], sign: i32) -> Vec<kernel::Complex<f32>> {
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
    let mantissa_bits = *state >> 11;
    let scale = 1.0_f64 / (1u64 << 52) as f64;
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
// Tests — size-16 × f64
// ============================================================================

#[test]
fn notw_16_f64_forward_vs_naive() {
    let input = make_input_f64(16, 0xDEAD_BEEF_0031);
    let expected = dft_naive_f64(&input, -1);
    let mut got = input;
    codelet_notw_16(&mut got, -1);
    check_close_f64(&got, &expected, 1e-12, "notw_16_f64_fwd");
}

#[test]
fn notw_16_f64_inverse_vs_naive() {
    let input = make_input_f64(16, 0xDEAD_BEEF_0032);
    let expected = dft_naive_f64(&input, 1);
    let mut got = input;
    codelet_notw_16(&mut got, 1);
    check_close_f64(&got, &expected, 1e-12, "notw_16_f64_inv");
}

#[test]
fn notw_16_f64_roundtrip() {
    let original = make_input_f64(16, 0xDEAD_BEEF_0033);
    let mut data = original.clone();
    codelet_notw_16(&mut data, -1);
    codelet_notw_16(&mut data, 1);
    let scale = 16.0_f64;
    for x in &mut data {
        x.re /= scale;
        x.im /= scale;
    }
    check_close_f64(&data, &original, 1e-11, "notw_16_f64_roundtrip");
}

// ============================================================================
// Tests — size-16 × f32
// ============================================================================

#[test]
fn notw_16_f32_forward_vs_naive() {
    let input = make_input_f32(16, 0xCAFE_BABE_0031);
    let expected = dft_naive_f32(&input, -1);
    let mut got = input;
    codelet_notw_16(&mut got, -1);
    check_close_f32(&got, &expected, 1e-4, "notw_16_f32_fwd");
}

#[test]
fn notw_16_f32_inverse_vs_naive() {
    let input = make_input_f32(16, 0xCAFE_BABE_0032);
    let expected = dft_naive_f32(&input, 1);
    let mut got = input;
    codelet_notw_16(&mut got, 1);
    check_close_f32(&got, &expected, 1e-4, "notw_16_f32_inv");
}

// ============================================================================
// Tests — size-32 × f64
// ============================================================================

#[test]
fn notw_32_f64_forward_vs_naive() {
    let input = make_input_f64(32, 0xDEAD_BEEF_0041);
    let expected = dft_naive_f64(&input, -1);
    let mut got = input;
    codelet_notw_32(&mut got, -1);
    check_close_f64(&got, &expected, 1e-11, "notw_32_f64_fwd");
}

#[test]
fn notw_32_f64_inverse_vs_naive() {
    let input = make_input_f64(32, 0xDEAD_BEEF_0042);
    let expected = dft_naive_f64(&input, 1);
    let mut got = input;
    codelet_notw_32(&mut got, 1);
    check_close_f64(&got, &expected, 1e-11, "notw_32_f64_inv");
}

#[test]
fn notw_32_f64_roundtrip() {
    let original = make_input_f64(32, 0xDEAD_BEEF_0043);
    let mut data = original.clone();
    codelet_notw_32(&mut data, -1);
    codelet_notw_32(&mut data, 1);
    let scale = 32.0_f64;
    for x in &mut data {
        x.re /= scale;
        x.im /= scale;
    }
    check_close_f64(&data, &original, 1e-10, "notw_32_f64_roundtrip");
}

// ============================================================================
// Tests — size-32 × f32
// ============================================================================

#[test]
fn notw_32_f32_forward_vs_naive() {
    let input = make_input_f32(32, 0xCAFE_BABE_0041);
    let expected = dft_naive_f32(&input, -1);
    let mut got = input;
    codelet_notw_32(&mut got, -1);
    check_close_f32(&got, &expected, 1e-3, "notw_32_f32_fwd");
}

// ============================================================================
// Tests — size-64 × f64
// ============================================================================

#[test]
fn notw_64_f64_forward_vs_naive() {
    let input = make_input_f64(64, 0xDEAD_BEEF_0051);
    let expected = dft_naive_f64(&input, -1);
    let mut got = input;
    codelet_notw_64(&mut got, -1);
    check_close_f64(&got, &expected, 1e-10, "notw_64_f64_fwd");
}

#[test]
fn notw_64_f64_inverse_vs_naive() {
    let input = make_input_f64(64, 0xDEAD_BEEF_0052);
    let expected = dft_naive_f64(&input, 1);
    let mut got = input;
    codelet_notw_64(&mut got, 1);
    check_close_f64(&got, &expected, 1e-10, "notw_64_f64_inv");
}

#[test]
fn notw_64_f64_roundtrip() {
    let original = make_input_f64(64, 0xDEAD_BEEF_0053);
    let mut data = original.clone();
    codelet_notw_64(&mut data, -1);
    codelet_notw_64(&mut data, 1);
    let scale = 64.0_f64;
    for x in &mut data {
        x.re /= scale;
        x.im /= scale;
    }
    check_close_f64(&data, &original, 1e-9, "notw_64_f64_roundtrip");
}

// ============================================================================
// Tests — size-64 × f32
// ============================================================================

#[test]
fn notw_64_f32_forward_vs_naive() {
    let input = make_input_f32(64, 0xCAFE_BABE_0051);
    let expected = dft_naive_f32(&input, -1);
    let mut got = input;
    codelet_notw_64(&mut got, -1);
    check_close_f32(&got, &expected, 1e-2, "notw_64_f32_fwd");
}
