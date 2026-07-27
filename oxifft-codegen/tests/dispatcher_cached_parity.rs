// Numerical correctness + `no_std`-safety regression test for the *cached*
// (`AtomicU8`-backed) ISA dispatcher emitted by `gen_dispatcher_codelet!`.
//
// This closes a gap in the existing test suite: `simd_f32_parity.rs` and
// friends exercise `gen_simd_codelet!`'s *uncached* dispatcher, but nothing
// previously compiled `gen_dispatcher_codelet!` for real (only string-content
// checks existed in `oxifft-codegen-impl`'s unit tests). That mattered
// because the cached dispatcher's `detect_isa_{size}_{ty}` function used to
// emit an unconditional `is_x86_feature_detected!` (`x86_64`) and
// `std::arch::is_aarch64_feature_detected!` (`aarch64`) — both `std`-only —
// which does not compile under `#![no_std]`.
//
// `oxifft-codegen`'s own `std` Cargo feature (default-on) is forwarded into
// the `#[cfg(feature = "std")]` gate inside the generated dispatcher (see
// `oxifft-codegen/Cargo.toml`). Running this file both ways proves both
// branches actually compile *and* produce correct results:
//
//   cargo test -p oxifft-codegen --test dispatcher_cached_parity
//   cargo test -p oxifft-codegen --test dispatcher_cached_parity --no-default-features
//
// The `--no-default-features` run disables the `std` gate, so the generated
// code takes the `cfg!(target_feature = ...)` no_std-safe compile-time path
// instead of `is_x86_feature_detected!` — proving that path is not just
// syntactically present but numerically correct.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::approx_constant,
    clippy::assign_op_pattern,
    clippy::derive_partial_eq_without_eq,
    clippy::missing_const_for_fn,
    clippy::suboptimal_flops
)]

use oxifft_codegen::gen_simd_codelet;

// ============================================================================
// Minimal kernel stub (same shape as notw_small_sizes.rs / simd_f32_parity.rs)
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

use kernel::Complex;

// ============================================================================
// gen_simd_codelet! provides the arch-specific inner functions
// (codelet_simd_4_sse2_f32, _avx2_f64, _neon_f32, ...) that the cached
// dispatcher below delegates to.
// ============================================================================

gen_simd_codelet!(4);

// ============================================================================
// gen_dispatcher_codelet! — the code under test. Each invocation lives in its
// own submodule because the macro emits module-level `const` ISA-level
// declarations that would collide if multiple invocations shared a namespace
// (mirrors `oxifft::dft::codelets::generated_simd`'s own layout).
// ============================================================================

mod cached_4_f64 {
    use oxifft_codegen::gen_dispatcher_codelet;
    gen_dispatcher_codelet!(size = 4, ty = f64);
}
use cached_4_f64::codelet_simd_4_cached_f64;

mod cached_4_f32 {
    use oxifft_codegen::gen_dispatcher_codelet;
    gen_dispatcher_codelet!(size = 4, ty = f32);
}
use cached_4_f32::codelet_simd_4_cached_f32;

// ============================================================================
// Naive O(n^2) DFT reference.
// ============================================================================

fn dft_naive_f64(input: &[Complex<f64>], sign: i32) -> Vec<Complex<f64>> {
    let n = input.len();
    let n_f = n as f64;
    (0..n)
        .map(|k| {
            input
                .iter()
                .enumerate()
                .fold(Complex::new(0.0_f64, 0.0), |acc, (j, &x)| {
                    let angle =
                        f64::from(sign) * 2.0 * core::f64::consts::PI * (j * k) as f64 / n_f;
                    let (ws, wc) = angle.sin_cos();
                    acc + Complex::new(x.re * wc - x.im * ws, x.re * ws + x.im * wc)
                })
        })
        .collect()
}

fn dft_naive_f32(input: &[Complex<f32>], sign: i32) -> Vec<Complex<f32>> {
    let f64_input: Vec<Complex<f64>> = input
        .iter()
        .map(|c| Complex::new(f64::from(c.re), f64::from(c.im)))
        .collect();
    dft_naive_f64(&f64_input, sign)
        .into_iter()
        .map(|c| Complex::new(c.re as f32, c.im as f32))
        .collect()
}

// ============================================================================
// Tests — cached dispatcher vs. naive DFT, both precisions.
// ============================================================================

#[test]
fn cached_dispatcher_4_f64_forward_vs_naive() {
    let input = [
        Complex {
            re: 1.0f64,
            im: 0.0,
        },
        Complex {
            re: 0.0f64,
            im: 1.0,
        },
        Complex {
            re: -1.0f64,
            im: 0.0,
        },
        Complex {
            re: 0.0f64,
            im: -1.0,
        },
    ];
    let expected = dft_naive_f64(&input, -1);
    let mut data = input;
    codelet_simd_4_cached_f64(&mut data, -1);
    for (got, exp) in data.iter().zip(expected.iter()) {
        assert!(
            (got.re - exp.re).abs() < 1e-10,
            "re mismatch: {got:?} vs {exp:?}"
        );
        assert!(
            (got.im - exp.im).abs() < 1e-10,
            "im mismatch: {got:?} vs {exp:?}"
        );
    }
}

#[test]
fn cached_dispatcher_4_f64_roundtrip() {
    let original = [
        Complex {
            re: 1.0f64,
            im: 2.0,
        },
        Complex {
            re: 3.0f64,
            im: 4.0,
        },
        Complex {
            re: 5.0f64,
            im: 6.0,
        },
        Complex {
            re: 7.0f64,
            im: 8.0,
        },
    ];
    let mut data = original;
    codelet_simd_4_cached_f64(&mut data, -1);
    codelet_simd_4_cached_f64(&mut data, 1);
    let n = original.len() as f64;
    for (got, orig) in data.iter().zip(original.iter()) {
        assert!((got.re / n - orig.re).abs() < 1e-10);
        assert!((got.im / n - orig.im).abs() < 1e-10);
    }
}

/// Proves the cached dispatcher reaches the f32 SIMD/scalar path correctly
/// too — under `--no-default-features` this exercises the `cfg!(target_feature)`
/// `no_std` fallback branch on `x86_64` hosts (on `aarch64` the NEON arm is
/// unconditional regardless of the `std` feature).
#[test]
fn cached_dispatcher_4_f32_forward_vs_naive() {
    let input = [
        Complex {
            re: 1.0f32,
            im: 0.0,
        },
        Complex {
            re: 0.0f32,
            im: 1.0,
        },
        Complex {
            re: -1.0f32,
            im: 0.0,
        },
        Complex {
            re: 0.0f32,
            im: -1.0,
        },
    ];
    let expected = dft_naive_f32(&input, -1);
    let mut data = input;
    codelet_simd_4_cached_f32(&mut data, -1);
    for (got, exp) in data.iter().zip(expected.iter()) {
        assert!(
            (got.re - exp.re).abs() < 1e-5,
            "re mismatch: {got:?} vs {exp:?}"
        );
        assert!(
            (got.im - exp.im).abs() < 1e-5,
            "im mismatch: {got:?} vs {exp:?}"
        );
    }
}

#[test]
fn cached_dispatcher_4_f32_roundtrip() {
    let original = [
        Complex {
            re: 1.0f32,
            im: 2.0,
        },
        Complex {
            re: 3.0f32,
            im: 4.0,
        },
        Complex {
            re: 5.0f32,
            im: 6.0,
        },
        Complex {
            re: 7.0f32,
            im: 8.0,
        },
    ];
    let mut data = original;
    codelet_simd_4_cached_f32(&mut data, -1);
    codelet_simd_4_cached_f32(&mut data, 1);
    let n = original.len() as f32;
    for (got, orig) in data.iter().zip(original.iter()) {
        assert!((got.re / n - orig.re).abs() < 1e-5);
        assert!((got.im / n - orig.im).abs() < 1e-5);
    }
}

#[test]
fn cached_dispatcher_4_f64_deterministic() {
    let input = [
        Complex {
            re: 1.0f64,
            im: 2.0,
        },
        Complex {
            re: 3.0f64,
            im: 4.0,
        },
        Complex {
            re: 5.0f64,
            im: 6.0,
        },
        Complex {
            re: 7.0f64,
            im: 8.0,
        },
    ];
    let mut data_a = input;
    let mut data_b = input;
    codelet_simd_4_cached_f64(&mut data_a, -1);
    codelet_simd_4_cached_f64(&mut data_b, -1);
    for (a, b) in data_a.iter().zip(data_b.iter()) {
        assert!((a.re - b.re).abs() < 1e-15 && (a.im - b.im).abs() < 1e-15);
    }
}
