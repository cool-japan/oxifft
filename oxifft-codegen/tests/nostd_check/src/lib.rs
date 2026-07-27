//! Standalone `#![no_std]` compile check for `oxifft-codegen`'s emitted dispatchers.
//!
//! This crate is intentionally detached from the main `oxifft` Cargo
//! workspace (see its own `Cargo.toml`) so it can be built with a feature
//! configuration (`std` disabled, `avx512` enabled) that differs from the
//! rest of the repository. Its only job is to prove that the code emitted by
//! `gen_simd_codelet!` and `gen_dispatcher_codelet!` *type-checks* under
//! `#![no_std]` — i.e. that it contains no accidental `std`-only construct
//! (an unconditional `is_x86_feature_detected!`, a `std::`-qualified path,
//! `OnceLock`, etc.).
//!
//! `cargo check` (rather than `cargo build`/`cargo test`) is sufficient and
//! deliberately used: a *library* target never needs a `fn main`, panic
//! handler, or executable entry point, so there is no bare-metal-entry-point
//! plumbing to fight with — only real type/borrow/macro-expansion checking,
//! which is exactly what proves the generated code is `no_std`-safe.
//!
//! Run with (from the `oxifft-codegen/tests/nostd_check` directory, or via
//! `--manifest-path`):
//! ```text
//! cargo check --target x86_64-unknown-linux-gnu   # exercises the x86_64 arms
//! cargo check                                     # exercises the host arch's arms
//! ```
//!
//! Numerical correctness of the generated codelets is already covered
//! end-to-end (under both `std` and `no_std` feature configurations) by
//! `oxifft-codegen/tests/simd_f32_parity.rs` and
//! `oxifft-codegen/tests/dispatcher_cached_parity.rs`; the trigonometric
//! bodies below are therefore trivial stubs — no `libm`/`std` transcendental
//! functions are needed (or available) here, and none are exercised since
//! this crate is never executed, only type-checked.
#![no_std]
// Lints on the *emitted* codelet bodies (matching the `#![allow(...)]` list
// every other fixture in `oxifft-codegen/tests/` already carries for the
// same reason — these are stylistic properties of generated code, not of
// this file, and are out of scope for a `no_std` compile check).
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::approx_constant,
    clippy::assign_op_pattern,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::derive_partial_eq_without_eq
)]

/// Minimal `crate::kernel` contract (core-only). See
/// `oxifft_codegen_impl::kernel_contract` for the std-based reference
/// implementation used by the rest of the test suite (its `sin`/`cos`/`sqrt`
/// delegate to the real `f32`/`f64` methods, which require `std`).
pub mod kernel {
    use core::fmt;
    use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

    pub trait Float:
        Copy
        + Clone
        + Default
        + fmt::Debug
        + PartialOrd
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
        + Neg<Output = Self>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + 'static
    {
        const ZERO: Self;
        const ONE: Self;
        fn zero() -> Self;
        fn from_f64(v: f64) -> Self;
        fn from_usize(v: usize) -> Self;
        fn sin(self) -> Self;
        fn cos(self) -> Self;
        fn sin_cos(self) -> (Self, Self);
        fn sqrt(self) -> Self;
        fn abs(self) -> Self;
    }

    macro_rules! impl_float {
        ($t:ty) => {
            impl Float for $t {
                const ZERO: Self = 0.0;
                const ONE: Self = 1.0;
                fn zero() -> Self {
                    0.0
                }
                fn from_f64(v: f64) -> Self {
                    v as Self
                }
                fn from_usize(v: usize) -> Self {
                    v as Self
                }
                // Dummy bodies: this crate is only ever `cargo check`-ed, never
                // executed, and no `libm`/`std` transcendental functions are
                // available under `#![no_std]` without an extra dependency —
                // numeric correctness is proven elsewhere (see module docs).
                fn sin(self) -> Self {
                    self
                }
                fn cos(self) -> Self {
                    self
                }
                fn sin_cos(self) -> (Self, Self) {
                    (self, self)
                }
                fn sqrt(self) -> Self {
                    self
                }
                fn abs(self) -> Self {
                    if self < 0.0 {
                        -self
                    } else {
                        self
                    }
                }
            }
        };
    }
    impl_float!(f32);
    impl_float!(f64);

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

use oxifft_codegen::gen_simd_codelet;

// The uncached dispatchers — this is what previously failed to compile under
// `#![no_std]` on `x86_64` with an unconditional `is_x86_feature_detected!`.
gen_simd_codelet!(2);
gen_simd_codelet!(4);
gen_simd_codelet!(8);
gen_simd_codelet!(16);

// The cached (`AtomicU8`-backed) dispatchers — previously used an
// unconditional `is_x86_feature_detected!` (`x86_64`) and
// `std::arch::is_aarch64_feature_detected!` (`aarch64`).
mod cached_4_f64 {
    use oxifft_codegen::gen_dispatcher_codelet;
    gen_dispatcher_codelet!(size = 4, ty = f64);
}
mod cached_4_f32 {
    use oxifft_codegen::gen_dispatcher_codelet;
    gen_dispatcher_codelet!(size = 4, ty = f32);
}
mod cached_16_f32 {
    use oxifft_codegen::gen_dispatcher_codelet;
    gen_dispatcher_codelet!(size = 16, ty = f32);
}

// Note: `gen_any_codelet!`'s runtime-wrapper classes (smooth-7 `MixedRadix`,
// `RaderPrime`, `Bluestein`) additionally delegate to `::oxifft`'s
// `Plan::dft_1d`, which would be a circular dependency to compile here (this
// crate cannot depend on `oxifft`, which itself depends on `oxifft-codegen`).
// Their `no_std` safety (no `::std::vec::Vec`, resolved by type inference
// instead) is proven at the token-stream level by
// `oxifft_codegen_impl::gen_any::tests::runtime_wrapper_has_no_qualified_std_path`.
