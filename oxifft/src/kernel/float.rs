//! Generic floating-point trait for FFT operations.

use core::fmt::Debug;
use core::ops::{Add, Div, Mul, Neg, Sub};

use num_traits::{Float as NumFloat, FloatConst, NumAssign};

/// Trait for floating-point types supported by OxiFFT.
///
/// This trait combines numeric operations with trigonometric functions
/// needed for FFT computation.
pub trait Float:
    Copy
    + Clone
    + Default
    + Debug
    + Send
    + Sync
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + NumAssign
    + NumFloat
    + FloatConst
    + 'static
{
    /// The value 0.
    const ZERO: Self;
    /// The value 1.
    const ONE: Self;
    /// The value 2.
    const TWO: Self;
    /// The value π.
    const PI: Self;
    /// The value 2π.
    const TWO_PI: Self;

    /// Compute sine.
    fn sin(self) -> Self;

    /// Compute cosine.
    fn cos(self) -> Self;

    /// Compute sine and cosine together (more efficient).
    fn sin_cos(self) -> (Self, Self);

    /// Compute square root.
    fn sqrt(self) -> Self;

    /// Compute absolute value.
    fn abs(self) -> Self;

    /// Convert from usize.
    fn from_usize(n: usize) -> Self;

    /// Convert from isize.
    fn from_isize(n: isize) -> Self;

    /// Convert from f64.
    fn from_f64(n: f64) -> Self;
}

impl Float for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const PI: Self = core::f32::consts::PI;
    const TWO_PI: Self = core::f32::consts::TAU;

    #[inline]
    fn sin(self) -> Self {
        num_traits::Float::sin(self)
    }

    #[inline]
    fn cos(self) -> Self {
        num_traits::Float::cos(self)
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        num_traits::Float::sin_cos(self)
    }

    #[inline]
    fn sqrt(self) -> Self {
        num_traits::Float::sqrt(self)
    }

    #[inline]
    fn abs(self) -> Self {
        num_traits::Float::abs(self)
    }

    #[inline]
    fn from_usize(n: usize) -> Self {
        n as Self
    }

    #[inline]
    fn from_isize(n: isize) -> Self {
        n as Self
    }

    #[inline]
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

    #[inline]
    fn sin(self) -> Self {
        num_traits::Float::sin(self)
    }

    #[inline]
    fn cos(self) -> Self {
        num_traits::Float::cos(self)
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        num_traits::Float::sin_cos(self)
    }

    #[inline]
    fn sqrt(self) -> Self {
        num_traits::Float::sqrt(self)
    }

    #[inline]
    fn abs(self) -> Self {
        num_traits::Float::abs(self)
    }

    #[inline]
    fn from_usize(n: usize) -> Self {
        n as Self
    }

    #[inline]
    fn from_isize(n: isize) -> Self {
        n as Self
    }

    #[inline]
    fn from_f64(n: f64) -> Self {
        n
    }
}

#[cfg(feature = "f128-support")]
impl Float for super::f128_type::F128 {
    const ZERO: Self = super::f128_type::F128::ZERO;
    const ONE: Self = super::f128_type::F128::ONE;
    const TWO: Self = super::f128_type::F128::TWO;
    const PI: Self = super::f128_type::F128::PI;
    const TWO_PI: Self = super::f128_type::F128::TWO_PI;

    #[inline]
    fn sin(self) -> Self {
        super::f128_type::F128::sin(self)
    }

    #[inline]
    fn cos(self) -> Self {
        super::f128_type::F128::cos(self)
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        super::f128_type::F128::sin_cos(self)
    }

    #[inline]
    fn sqrt(self) -> Self {
        super::f128_type::F128::sqrt(self)
    }

    #[inline]
    fn abs(self) -> Self {
        super::f128_type::F128::abs(self)
    }

    #[inline]
    fn from_usize(n: usize) -> Self {
        super::f128_type::F128::from_usize(n)
    }

    #[inline]
    fn from_isize(n: isize) -> Self {
        super::f128_type::F128::from_isize(n)
    }

    #[inline]
    fn from_f64(n: f64) -> Self {
        super::f128_type::F128::from_f64(n)
    }
}
