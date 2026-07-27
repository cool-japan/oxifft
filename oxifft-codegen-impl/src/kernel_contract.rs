//! Reference `Float`/`Complex` contract that generated codelets target.
//!
//! Every codelet emitted by the `oxifft-codegen` proc-macros (and by the
//! programmatic emitters in this crate) refers to two types by the **relative**
//! path `crate::kernel::Float` and `crate::kernel::Complex<T>`.  That means the
//! crate that *invokes* a codelet macro must expose a module named `kernel` at
//! its crate root providing:
//!
//! - a `Float` trait implemented for the scalar element type (`f32` / `f64`), and
//! - a `Complex<T: Float>` type with `re` / `im` fields and the usual arithmetic.
//!
//! The production `oxifft` crate satisfies this contract with its own
//! `oxifft::kernel` module (which layers `num-traits` on top).  This module is a
//! **minimal, dependency-free reference implementation** of the same contract so
//! that:
//!
//! 1. Doc-tests and integration tests can compile real generated codelets
//!    without pulling in `oxifft` (which would be a circular dependency), and
//! 2. downstream users have an executable specification of exactly what a
//!    `kernel` module must provide.
//!
//! # Using it as `crate::kernel`
//!
//! Re-export it under the name `kernel` at your crate root and the generated
//! codelets resolve against it:
//!
//! ```
//! mod kernel {
//!     pub use oxifft_codegen_impl::kernel_contract::{Complex, Float};
//! }
//!
//! // A codelet macro expands to
//! //   pub fn codelet_notw_2<T: crate::kernel::Float>(x: &mut [crate::kernel::Complex<T>], sign: i32)
//! // which now resolves against the re-exported contract above.
//! # fn main() {}
//! ```

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::missing_const_for_fn,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::suboptimal_flops,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use
)]

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Scalar element contract required by generated codelets.
///
/// Implemented here for `f32` and `f64`.  The generated code only ever calls the
/// members declared on this trait, so any type satisfying it can host a codelet.
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
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;

    /// Additive identity as a function (mirrors `num_traits::Zero::zero`).
    fn zero() -> Self;
    /// Convert an `f64` literal to this precision.
    fn from_f64(v: f64) -> Self;
    /// Convert a `usize` index to this precision.
    fn from_usize(v: usize) -> Self;
    /// Sine.
    fn sin(self) -> Self;
    /// Cosine.
    fn cos(self) -> Self;
    /// Simultaneous sine and cosine.
    fn sin_cos(self) -> (Self, Self);
    /// Square root.
    fn sqrt(self) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
}

macro_rules! impl_float {
    ($t:ty) => {
        impl Float for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;

            #[inline]
            fn zero() -> Self {
                0.0
            }
            #[inline]
            fn from_f64(v: f64) -> Self {
                v as Self
            }
            #[inline]
            fn from_usize(v: usize) -> Self {
                v as Self
            }
            #[inline]
            fn sin(self) -> Self {
                <$t>::sin(self)
            }
            #[inline]
            fn cos(self) -> Self {
                <$t>::cos(self)
            }
            #[inline]
            fn sin_cos(self) -> (Self, Self) {
                <$t>::sin_cos(self)
            }
            #[inline]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            #[inline]
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
        }
    };
}

impl_float!(f32);
impl_float!(f64);

/// Minimal complex type mirroring the layout of the production `oxifft` kernel.
///
/// `#[repr(C)]` with `(re, im)` field order so that `&mut [Complex<T>]` and
/// `&mut [T]` (length `2 * n`) share a layout — exactly the reinterpretation the
/// SIMD codelets rely on.
#[derive(Copy, Clone, Default, PartialEq)]
#[repr(C)]
pub struct Complex<T: Float> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}

impl<T: Float> Complex<T> {
    /// Construct a complex value from real and imaginary parts.
    #[inline]
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }

    /// The complex zero, `0 + 0i`.
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
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}
impl<T: Float> AddAssign for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}
impl<T: Float> Sub for Complex<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}
impl<T: Float> SubAssign for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}
impl<T: Float> Mul for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}
impl<T: Float> MulAssign for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl<T: Float> Div for Complex<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let norm_sq = rhs.re * rhs.re + rhs.im * rhs.im;
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / norm_sq,
            (self.im * rhs.re - self.re * rhs.im) / norm_sq,
        )
    }
}
impl<T: Float> DivAssign for Complex<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
impl<T: Float> Neg for Complex<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}
impl<T: Float> Mul<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}
