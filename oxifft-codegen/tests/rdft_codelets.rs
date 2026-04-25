// Numerical correctness tests for gen_rdft_codelet! macro output.
//
// Compares generated R2HC / HC2R codelets against:
//  (a) a naive O(n²) DFT reference (for R2HC)
//  (b) roundtrip correctness (R2HC → HC2R → original)
//
// NOTE: Generated code references `crate::kernel::Float` and `crate::kernel::Complex<T>`.
// Since adding `oxifft` as a dev-dep here would create a circular dependency
// (oxifft → oxifft-codegen), we re-declare the minimal kernel stub below,
// mirroring the pattern in `notw_small_sizes.rs`.
//
// Clippy lints suppressed in this test file:
//   - cast_precision_loss: small FFT indices (n ≤ 8) cast to f64 are exact.
//   - approx_constant: generated code uses explicit f64 literals (trig values).
//   - suboptimal_flops: generated code uses naive add/mul structure.
//   - derive_partial_eq_without_eq: stub Complex<T> is PartialEq only.
//   - missing_const_for_fn: zero() cannot be const due to trait bound.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::approx_constant,
    clippy::suboptimal_flops,
    clippy::derive_partial_eq_without_eq,
    clippy::missing_const_for_fn
)]

use oxifft_codegen::gen_rdft_codelet;

// ============================================================================
// Minimal kernel stub — mirrors the types / impls the generated code needs.
// (Identical to the stub in notw_small_sizes.rs.)
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
// Generated codelets (proc-macro invocations)
// ============================================================================

gen_rdft_codelet!(size = 2, kind = R2hc);
gen_rdft_codelet!(size = 4, kind = R2hc);
gen_rdft_codelet!(size = 8, kind = R2hc);
gen_rdft_codelet!(size = 2, kind = Hc2r);
gen_rdft_codelet!(size = 4, kind = Hc2r);
gen_rdft_codelet!(size = 8, kind = Hc2r);

// ============================================================================
// Reference naive DFT (forward, real input → half-complex output, k=0..n/2)
// ============================================================================

fn naive_r2hc_f64(x: &[f64]) -> Vec<kernel::Complex<f64>> {
    let n = x.len();
    (0..=n / 2)
        .map(|k| {
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for (j, &xj) in x.iter().enumerate() {
                let angle = -2.0 * core::f64::consts::PI * (j * k) as f64 / n as f64;
                re += xj * angle.cos();
                im += xj * angle.sin();
            }
            kernel::Complex::new(re, im)
        })
        .collect()
}

fn naive_r2hc_f32(x: &[f32]) -> Vec<kernel::Complex<f32>> {
    let n = x.len();
    (0..=n / 2)
        .map(|k| {
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for (j, &xj) in x.iter().enumerate() {
                let angle = -2.0 * core::f64::consts::PI * (j * k) as f64 / n as f64;
                re += f64::from(xj) * angle.cos();
                im += f64::from(xj) * angle.sin();
            }
            kernel::Complex::new(re as f32, im as f32)
        })
        .collect()
}

// ============================================================================
// Small deterministic LCG (same as notw_small_sizes.rs)
// ============================================================================

fn lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let mantissa_bits = *state >> 11;
    let scale = 1.0_f64 / (1u64 << 52) as f64;
    mantissa_bits as f64 * scale - 1.0
}

fn make_real_f64(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n).map(|_| lcg_next(&mut s)).collect()
}

fn make_real_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| lcg_next(&mut s) as f32).collect()
}

// ============================================================================
// Tolerance checkers
// ============================================================================

fn check_r2hc_f64(
    x: &[f64],
    got: &[kernel::Complex<f64>],
    expected: &[kernel::Complex<f64>],
    tol: f64,
    label: &str,
) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: length mismatch (input: {x:?})"
    );
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let err_re = (g.re - e.re).abs();
        let err_im = (g.im - e.im).abs();
        assert!(
            err_re < tol && err_im < tol,
            "{label}[{i}]: got {g:?}, expected {e:?}, err=({err_re:.2e}, {err_im:.2e}) >= tol={tol:.2e}"
        );
    }
}

fn check_r2hc_f32(
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
            "{label}[{i}]: got {g:?}, expected {e:?}, err=({err_re:.2e}, {err_im:.2e}) >= tol={tol:.2e}"
        );
    }
}

// ============================================================================
// R2HC size-2 — f64
// ============================================================================

#[test]
fn r2hc_2_gen_f64_vs_naive() {
    let x = make_real_f64(2, 0xDEAD_BEEF_1001_u64);
    let expected = naive_r2hc_f64(&x);
    let mut got = vec![kernel::Complex::<f64>::zero(); 2];
    r2hc_2_gen(&x, &mut got);
    check_r2hc_f64(&x, &got, &expected, 1e-12, "r2hc_2_gen_f64");
}

#[test]
fn r2hc_2_gen_f64_dc_signal() {
    let x = [3.0_f64, 3.0];
    let mut got = vec![kernel::Complex::<f64>::zero(); 2];
    r2hc_2_gen(&x, &mut got);
    assert!((got[0].re - 6.0).abs() < 1e-14, "DC: {}", got[0].re);
    assert!((got[0].im).abs() < 1e-14, "DC.im: {}", got[0].im);
    assert!((got[1].re).abs() < 1e-14, "Nyq.re: {}", got[1].re);
    assert!((got[1].im).abs() < 1e-14, "Nyq.im: {}", got[1].im);
}

// ============================================================================
// R2HC size-2 — f32
// ============================================================================

#[test]
fn r2hc_2_gen_f32_vs_naive() {
    let x = make_real_f32(2, 0xCAFE_BABE_1001_u64);
    let expected = naive_r2hc_f32(&x);
    let mut got = vec![kernel::Complex::<f32>::zero(); 2];
    r2hc_2_gen(&x, &mut got);
    check_r2hc_f32(&got, &expected, 1e-5, "r2hc_2_gen_f32");
}

// ============================================================================
// R2HC size-4 — f64
// ============================================================================

#[test]
fn r2hc_4_gen_f64_vs_naive() {
    let x = make_real_f64(4, 0xDEAD_BEEF_1004_u64);
    let expected = naive_r2hc_f64(&x);
    let mut got = vec![kernel::Complex::<f64>::zero(); 3];
    r2hc_4_gen(&x, &mut got);
    check_r2hc_f64(&x, &got, &expected, 1e-12, "r2hc_4_gen_f64");
}

#[test]
fn r2hc_4_gen_f64_known_values() {
    // x = [1, 2, 3, 4] → Y[0]=10, Y[1]=(-2+2i), Y[2]=-2
    let x = [1.0_f64, 2.0, 3.0, 4.0];
    let mut got = vec![kernel::Complex::<f64>::zero(); 3];
    r2hc_4_gen(&x, &mut got);
    assert!((got[0].re - 10.0).abs() < 1e-12, "Y[0].re={}", got[0].re);
    assert!((got[0].im).abs() < 1e-14, "Y[0].im={}", got[0].im);
    assert!((got[1].re - (-2.0)).abs() < 1e-12, "Y[1].re={}", got[1].re);
    assert!((got[1].im - 2.0).abs() < 1e-12, "Y[1].im={}", got[1].im);
    assert!((got[2].re - (-2.0)).abs() < 1e-12, "Y[2].re={}", got[2].re);
    assert!((got[2].im).abs() < 1e-14, "Y[2].im={}", got[2].im);
}

// ============================================================================
// R2HC size-4 — f32
// ============================================================================

#[test]
fn r2hc_4_gen_f32_vs_naive() {
    let x = make_real_f32(4, 0xCAFE_BABE_1004_u64);
    let expected = naive_r2hc_f32(&x);
    let mut got = vec![kernel::Complex::<f32>::zero(); 3];
    r2hc_4_gen(&x, &mut got);
    check_r2hc_f32(&got, &expected, 1e-5, "r2hc_4_gen_f32");
}

// ============================================================================
// R2HC size-8 — f64
// ============================================================================

#[test]
fn r2hc_8_gen_f64_vs_naive() {
    let x = make_real_f64(8, 0xDEAD_BEEF_1008_u64);
    let expected = naive_r2hc_f64(&x);
    let mut got = vec![kernel::Complex::<f64>::zero(); 5];
    r2hc_8_gen(&x, &mut got);
    check_r2hc_f64(&x, &got, &expected, 1e-11, "r2hc_8_gen_f64");
}

#[test]
fn r2hc_8_gen_f64_dc_signal() {
    let x = [1.0_f64; 8];
    let mut got = vec![kernel::Complex::<f64>::zero(); 5];
    r2hc_8_gen(&x, &mut got);
    assert!((got[0].re - 8.0).abs() < 1e-12, "DC={}", got[0].re);
    for (k, bin) in got.iter().enumerate().skip(1) {
        assert!(bin.re.abs() < 1e-12, "Y[{k}].re={} expected ~0", bin.re);
        assert!(bin.im.abs() < 1e-12, "Y[{k}].im={} expected ~0", bin.im);
    }
}

// ============================================================================
// R2HC size-8 — f32
// ============================================================================

#[test]
fn r2hc_8_gen_f32_vs_naive() {
    let x = make_real_f32(8, 0xCAFE_BABE_1008_u64);
    let expected = naive_r2hc_f32(&x);
    let mut got = vec![kernel::Complex::<f32>::zero(); 5];
    r2hc_8_gen(&x, &mut got);
    check_r2hc_f32(&got, &expected, 1e-5, "r2hc_8_gen_f32");
}

// ============================================================================
// HC2R roundtrip tests (R2HC → HC2R recovers N× original signal)
// ============================================================================

fn check_hc2r_roundtrip_f64(n: usize, seed: u64, tol: f64, label: &str) {
    let original = make_real_f64(n, seed);
    let half = n / 2;
    let mut spectrum = vec![kernel::Complex::<f64>::zero(); half + 1];
    let mut recovered = vec![0.0_f64; n];

    match n {
        2 => {
            r2hc_2_gen(&original, &mut spectrum);
            hc2r_2_gen(&spectrum, &mut recovered);
        }
        4 => {
            r2hc_4_gen(&original, &mut spectrum);
            hc2r_4_gen(&spectrum, &mut recovered);
        }
        8 => {
            r2hc_8_gen(&original, &mut spectrum);
            hc2r_8_gen(&spectrum, &mut recovered);
        }
        _ => panic!("unsupported size {n}"),
    }

    // HC2R is unnormalized: recovered[j] should equal n * original[j]
    let n_f = n as f64;
    for (i, (r, o)) in recovered.iter().zip(original.iter()).enumerate() {
        let err = (r / n_f - o).abs();
        assert!(
            err < tol,
            "{label} idx={i}: recovered/n={}, original={o}, err={err:.2e}",
            r / n_f
        );
    }
}

fn check_hc2r_roundtrip_f32(n: usize, seed: u64, tol: f32, label: &str) {
    let original = make_real_f32(n, seed);
    let half = n / 2;
    let mut spectrum = vec![kernel::Complex::<f32>::zero(); half + 1];
    let mut recovered = vec![0.0_f32; n];

    match n {
        2 => {
            r2hc_2_gen(&original, &mut spectrum);
            hc2r_2_gen(&spectrum, &mut recovered);
        }
        4 => {
            r2hc_4_gen(&original, &mut spectrum);
            hc2r_4_gen(&spectrum, &mut recovered);
        }
        8 => {
            r2hc_8_gen(&original, &mut spectrum);
            hc2r_8_gen(&spectrum, &mut recovered);
        }
        _ => panic!("unsupported size {n}"),
    }

    let n_f = n as f32;
    for (i, (r, o)) in recovered.iter().zip(original.iter()).enumerate() {
        let err = (r / n_f - o).abs();
        assert!(
            err < tol,
            "{label} idx={i}: recovered/n={}, original={o}, err={err:.2e}",
            r / n_f
        );
    }
}

#[test]
fn hc2r_2_gen_roundtrip_f64() {
    check_hc2r_roundtrip_f64(2, 0xDEAD_BEEF_2002_u64, 1e-12, "hc2r_2_gen_f64");
}

#[test]
fn hc2r_2_gen_roundtrip_f32() {
    check_hc2r_roundtrip_f32(2, 0xCAFE_BABE_2002_u64, 1e-5, "hc2r_2_gen_f32");
}

#[test]
fn hc2r_4_gen_roundtrip_f64() {
    check_hc2r_roundtrip_f64(4, 0xDEAD_BEEF_2004_u64, 1e-12, "hc2r_4_gen_f64");
}

#[test]
fn hc2r_4_gen_roundtrip_f32() {
    check_hc2r_roundtrip_f32(4, 0xCAFE_BABE_2004_u64, 1e-5, "hc2r_4_gen_f32");
}

#[test]
fn hc2r_8_gen_roundtrip_f64() {
    check_hc2r_roundtrip_f64(8, 0xDEAD_BEEF_2008_u64, 1e-11, "hc2r_8_gen_f64");
}

#[test]
fn hc2r_8_gen_roundtrip_f32() {
    check_hc2r_roundtrip_f32(8, 0xCAFE_BABE_2008_u64, 1e-4, "hc2r_8_gen_f32");
}
