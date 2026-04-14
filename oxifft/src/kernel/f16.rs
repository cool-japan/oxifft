//! Half-precision (f16) floating-point support.
//!
//! This module provides a pure Rust implementation of IEEE 754 half-precision
//! floating-point numbers for FFT operations.
//!
//! # Features
//!
//! - Pure Rust, no external dependencies
//! - Implements the `Float` trait for use with OxiFFT
//! - Efficient conversion to/from f32/f64
//!
//! # Use Cases
//!
//! Half-precision is useful for:
//! - Machine learning inference (transformers, audio models)
//! - Memory-constrained applications
//! - GPU processing (most GPUs have excellent f16 support)
//!
//! # Precision Notes
//!
//! f16 has limited precision:
//! - 1 sign bit, 5 exponent bits, 10 mantissa bits
//! - ~3.3 decimal digits of precision
//! - Range: ±65504, smallest normal: 6.1e-5
//!
//! For high-precision FFT, use f32 or f64 instead.

use core::cmp::Ordering;
use core::fmt::{Debug, Display};
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// Half-precision floating-point number (IEEE 754-2008 binary16).
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct F16(u16);

// Constants
impl F16 {
    /// Zero (0.0).
    pub const ZERO: Self = Self(0x0000);
    /// One (1.0).
    pub const ONE: Self = Self(0x3C00);
    /// Negative one (-1.0).
    pub const NEG_ONE: Self = Self(0xBC00);
    /// Two (2.0).
    pub const TWO: Self = Self(0x4000);
    /// Half (0.5).
    pub const HALF: Self = Self(0x3800);
    /// Pi (π ≈ 3.14159).
    pub const PI: Self = Self(0x4248); // 3.140625
    /// Two times pi (2π).
    pub const TWO_PI: Self = Self(0x4648); // 6.28125
    /// Pi divided by two (π/2).
    pub const FRAC_PI_2: Self = Self(0x3E48); // 1.5703125
    /// Pi divided by four (π/4).
    pub const FRAC_PI_4: Self = Self(0x3A48); // 0.78515625
    /// E (Euler's number).
    pub const E: Self = Self(0x4170); // 2.71875
    /// ln(2).
    pub const LN_2: Self = Self(0x398C); // 0.693359375
    /// 1/sqrt(2).
    pub const FRAC_1_SQRT_2: Self = Self(0x3B50); // 0.70703125
    /// Positive infinity.
    pub const INFINITY: Self = Self(0x7C00);
    /// Negative infinity.
    pub const NEG_INFINITY: Self = Self(0xFC00);
    /// Not a Number (NaN).
    pub const NAN: Self = Self(0x7E00);
    /// Machine epsilon (smallest x such that 1 + x ≠ 1).
    pub const EPSILON: Self = Self(0x1400); // 2^-10 ≈ 0.000977
    /// Minimum positive normal value.
    pub const MIN_POSITIVE: Self = Self(0x0400); // 2^-14 ≈ 6.1e-5
    /// Maximum finite value.
    pub const MAX: Self = Self(0x7BFF); // 65504
    /// Minimum finite value.
    pub const MIN: Self = Self(0xFBFF); // -65504

    /// Create F16 from raw bits.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Get the raw bits of this F16.
    #[inline]
    #[must_use]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Check if this value is NaN.
    #[inline]
    #[must_use]
    pub const fn is_nan(self) -> bool {
        (self.0 & 0x7FFF) > 0x7C00
    }

    /// Check if this value is infinite.
    #[inline]
    #[must_use]
    pub const fn is_infinite(self) -> bool {
        (self.0 & 0x7FFF) == 0x7C00
    }

    /// Check if this value is finite (not NaN or infinity).
    #[inline]
    #[must_use]
    pub const fn is_finite(self) -> bool {
        (self.0 & 0x7C00) != 0x7C00
    }

    /// Check if this value is zero.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0.trailing_zeros() >= 15
    }

    /// Check if this value is negative (including -0).
    #[inline]
    #[must_use]
    pub const fn is_sign_negative(self) -> bool {
        (self.0 & 0x8000) != 0
    }

    /// Check if this value is positive (including +0).
    #[inline]
    #[must_use]
    pub const fn is_sign_positive(self) -> bool {
        (self.0 & 0x8000) == 0
    }

    /// Check if this value is normal.
    #[inline]
    #[must_use]
    pub fn is_normal(self) -> bool {
        let exp = (self.0 >> 10) & 0x1F;
        exp != 0 && exp != 0x1F
    }

    /// Get the absolute value.
    #[inline]
    #[must_use]
    pub const fn abs(self) -> Self {
        Self(self.0 & 0x7FFF)
    }

    /// Negate this value.
    #[inline]
    #[must_use]
    pub const fn neg(self) -> Self {
        Self(self.0 ^ 0x8000)
    }

    /// Copy the sign from another value.
    #[inline]
    #[must_use]
    pub const fn copysign(self, sign: Self) -> Self {
        Self((self.0 & 0x7FFF) | (sign.0 & 0x8000))
    }

    /// Sign function.
    #[inline]
    #[must_use]
    pub fn signum(self) -> Self {
        if self.is_nan() {
            Self::NAN
        } else if self.is_zero() {
            Self::ZERO
        } else if self.is_sign_negative() {
            Self::NEG_ONE
        } else {
            Self::ONE
        }
    }

    /// Reciprocal (1/x).
    #[inline]
    #[must_use]
    pub fn recip(self) -> Self {
        Self::ONE / self
    }

    /// Classify the value.
    #[inline]
    #[must_use]
    pub fn classify(self) -> core::num::FpCategory {
        let exp = (self.0 >> 10) & 0x1F;
        let mantissa = self.0 & 0x03FF;

        if exp == 0x1F {
            if mantissa == 0 {
                core::num::FpCategory::Infinite
            } else {
                core::num::FpCategory::Nan
            }
        } else if exp == 0 {
            if mantissa == 0 {
                core::num::FpCategory::Zero
            } else {
                core::num::FpCategory::Subnormal
            }
        } else {
            core::num::FpCategory::Normal
        }
    }
}

// Conversion from f32
impl F16 {
    /// Convert from f32 to f16.
    #[inline]
    #[must_use]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exponent = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x007F_FFFF;

        // Handle special cases
        if exponent == 255 {
            // NaN or infinity
            if mantissa != 0 {
                return Self::NAN;
            }
            return Self((sign | 0x7C00) as u16);
        }

        if exponent == 0 {
            // Zero or denormal (becomes zero in f16)
            return Self(sign as u16);
        }

        // Bias conversion: f32 bias = 127, f16 bias = 15
        let new_exp = exponent - 127 + 15;

        if new_exp >= 31 {
            // Overflow to infinity
            return Self((sign | 0x7C00) as u16);
        }

        if new_exp <= 0 {
            // Underflow to zero or denormal
            if new_exp < -10 {
                return Self(sign as u16);
            }
            // Denormal
            let mant = (mantissa | 0x0080_0000) >> (14 - new_exp);
            return Self((sign | (mant >> 13)) as u16);
        }

        // Normal number
        let new_mant = mantissa >> 13;
        Self((sign | ((new_exp as u32) << 10) | new_mant) as u16)
    }

    /// Convert f16 to f32.
    #[inline]
    #[must_use]
    pub fn to_f32(self) -> f32 {
        let sign = u32::from(self.0 & 0x8000) << 16;
        let exponent = (self.0 >> 10) & 0x1F;
        let mantissa = u32::from(self.0 & 0x03FF);

        if exponent == 0 {
            if mantissa == 0 {
                // Zero
                return f32::from_bits(sign);
            }
            // Denormal - normalize it
            let mut e = 1i32;
            let mut m = mantissa;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e += 1;
            }
            // e is now the shift count needed to normalize
            // f32 exponent = 127 (bias) - 15 (half bias) + 1 - e
            let f32_exp = 127i32 - 15i32 + 1i32 - e;
            if f32_exp <= 0 {
                // Underflows to zero in f32
                return f32::from_bits(sign);
            }
            let new_exp = (f32_exp as u32) << 23;
            let new_mant = (m & 0x03FF) << 13;
            return f32::from_bits(sign | new_exp | new_mant);
        }

        if exponent == 31 {
            // Infinity or NaN
            let inf_or_nan = if mantissa == 0 {
                0x7F80_0000
            } else {
                0x7FC0_0000
            };
            return f32::from_bits(sign | inf_or_nan);
        }

        // Normal number
        // Use signed arithmetic to avoid underflow: exponent - 15 + 127 = exponent + 112
        let new_exp = (i32::from(exponent) - 15 + 127) as u32;
        let new_mant = mantissa << 13;
        f32::from_bits(sign | (new_exp << 23) | new_mant)
    }
}

// Conversion from f64
impl F16 {
    /// Convert from f64 to f16.
    #[inline]
    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        // Convert via f32 (simpler and sufficient for f16 precision)
        Self::from_f32(value as f32)
    }

    /// Convert f16 to f64.
    #[inline]
    #[must_use]
    pub fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }
}

// Mathematical functions
impl F16 {
    /// Square root.
    #[inline]
    pub fn sqrt(self) -> Self {
        Self::from_f32(self.to_f32().sqrt())
    }

    /// Sine.
    #[inline]
    pub fn sin(self) -> Self {
        Self::from_f32(self.to_f32().sin())
    }

    /// Cosine.
    #[inline]
    pub fn cos(self) -> Self {
        Self::from_f32(self.to_f32().cos())
    }

    /// Sine and cosine.
    #[inline]
    pub fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.to_f32().sin_cos();
        (Self::from_f32(s), Self::from_f32(c))
    }

    /// Tangent.
    #[inline]
    pub fn tan(self) -> Self {
        Self::from_f32(self.to_f32().tan())
    }

    /// Arc sine.
    #[inline]
    pub fn asin(self) -> Self {
        Self::from_f32(self.to_f32().asin())
    }

    /// Arc cosine.
    #[inline]
    pub fn acos(self) -> Self {
        Self::from_f32(self.to_f32().acos())
    }

    /// Arc tangent.
    #[inline]
    pub fn atan(self) -> Self {
        Self::from_f32(self.to_f32().atan())
    }

    /// Arc tangent of y/x.
    #[inline]
    pub fn atan2(self, other: Self) -> Self {
        Self::from_f32(self.to_f32().atan2(other.to_f32()))
    }

    /// Exponential function.
    #[inline]
    pub fn exp(self) -> Self {
        Self::from_f32(self.to_f32().exp())
    }

    /// exp(x) - 1.
    #[inline]
    pub fn exp_m1(self) -> Self {
        Self::from_f32(self.to_f32().exp_m1())
    }

    /// 2^x.
    #[inline]
    pub fn exp2(self) -> Self {
        Self::from_f32(self.to_f32().exp2())
    }

    /// Natural logarithm.
    #[inline]
    pub fn ln(self) -> Self {
        Self::from_f32(self.to_f32().ln())
    }

    /// ln(1 + x).
    #[inline]
    pub fn ln_1p(self) -> Self {
        Self::from_f32(self.to_f32().ln_1p())
    }

    /// Logarithm with arbitrary base.
    #[inline]
    pub fn log(self, base: Self) -> Self {
        Self::from_f32(self.to_f32().log(base.to_f32()))
    }

    /// Base-2 logarithm.
    #[inline]
    pub fn log2(self) -> Self {
        Self::from_f32(self.to_f32().log2())
    }

    /// Base-10 logarithm.
    #[inline]
    pub fn log10(self) -> Self {
        Self::from_f32(self.to_f32().log10())
    }

    /// Power function x^y.
    #[inline]
    pub fn powf(self, exp: Self) -> Self {
        Self::from_f32(self.to_f32().powf(exp.to_f32()))
    }

    /// Integer power function.
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        Self::from_f32(self.to_f32().powi(n))
    }

    /// Cube root.
    #[inline]
    pub fn cbrt(self) -> Self {
        Self::from_f32(self.to_f32().cbrt())
    }

    /// Hypotenuse.
    #[inline]
    pub fn hypot(self, other: Self) -> Self {
        Self::from_f32(self.to_f32().hypot(other.to_f32()))
    }

    /// Fused multiply-add.
    #[inline]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self::from_f32(self.to_f32().mul_add(a.to_f32(), b.to_f32()))
    }

    /// Floor function.
    #[inline]
    pub fn floor(self) -> Self {
        Self::from_f32(self.to_f32().floor())
    }

    /// Ceiling function.
    #[inline]
    pub fn ceil(self) -> Self {
        Self::from_f32(self.to_f32().ceil())
    }

    /// Round function.
    #[inline]
    pub fn round(self) -> Self {
        Self::from_f32(self.to_f32().round())
    }

    /// Truncate function.
    #[inline]
    pub fn trunc(self) -> Self {
        Self::from_f32(self.to_f32().trunc())
    }

    /// Fractional part.
    #[inline]
    pub fn fract(self) -> Self {
        Self::from_f32(self.to_f32().fract())
    }

    /// Minimum.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        Self::from_f32(self.to_f32().min(other.to_f32()))
    }

    /// Maximum.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        Self::from_f32(self.to_f32().max(other.to_f32()))
    }

    /// Absolute difference.
    #[inline]
    pub fn abs_sub(self, other: Self) -> Self {
        Self::abs(self - other)
    }

    /// Hyperbolic sine.
    #[inline]
    pub fn sinh(self) -> Self {
        Self::from_f32(self.to_f32().sinh())
    }

    /// Hyperbolic cosine.
    #[inline]
    pub fn cosh(self) -> Self {
        Self::from_f32(self.to_f32().cosh())
    }

    /// Hyperbolic tangent.
    #[inline]
    pub fn tanh(self) -> Self {
        Self::from_f32(self.to_f32().tanh())
    }

    /// Inverse hyperbolic sine.
    #[inline]
    pub fn asinh(self) -> Self {
        Self::from_f32(self.to_f32().asinh())
    }

    /// Inverse hyperbolic cosine.
    #[inline]
    pub fn acosh(self) -> Self {
        Self::from_f32(self.to_f32().acosh())
    }

    /// Inverse hyperbolic tangent.
    #[inline]
    pub fn atanh(self) -> Self {
        Self::from_f32(self.to_f32().atanh())
    }

    /// Convert to degrees.
    #[inline]
    pub fn to_degrees(self) -> Self {
        Self::from_f32(self.to_f32().to_degrees())
    }

    /// Convert to radians.
    #[inline]
    pub fn to_radians(self) -> Self {
        Self::from_f32(self.to_f32().to_radians())
    }
}

// Arithmetic operations
impl Add for F16 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl AddAssign for F16 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for F16 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl SubAssign for F16 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for F16 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl MulAssign for F16 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for F16 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl DivAssign for F16 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Rem for F16 {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() % rhs.to_f32())
    }
}

impl RemAssign for F16 {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl Neg for F16 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::neg(self)
    }
}

// Arithmetic with references (required for NumAssign)
impl Add<&F16> for F16 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &F16) -> Self {
        self + *rhs
    }
}

impl Sub<&F16> for F16 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &F16) -> Self {
        self - *rhs
    }
}

impl Mul<&F16> for F16 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &F16) -> Self {
        self * *rhs
    }
}

impl Div<&F16> for F16 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: &F16) -> Self {
        self / *rhs
    }
}

impl Rem<&F16> for F16 {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: &F16) -> Self {
        self % *rhs
    }
}

impl AddAssign<&F16> for F16 {
    #[inline]
    fn add_assign(&mut self, rhs: &F16) {
        *self = *self + *rhs;
    }
}

impl SubAssign<&F16> for F16 {
    #[inline]
    fn sub_assign(&mut self, rhs: &F16) {
        *self = *self - *rhs;
    }
}

impl MulAssign<&F16> for F16 {
    #[inline]
    fn mul_assign(&mut self, rhs: &F16) {
        *self = *self * *rhs;
    }
}

impl DivAssign<&F16> for F16 {
    #[inline]
    fn div_assign(&mut self, rhs: &F16) {
        *self = *self / *rhs;
    }
}

impl RemAssign<&F16> for F16 {
    #[inline]
    fn rem_assign(&mut self, rhs: &F16) {
        *self = *self % *rhs;
    }
}

// Comparison
impl PartialEq for F16 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Handle NaN
        if self.is_nan() || other.is_nan() {
            return false;
        }
        // Handle +0 == -0
        if self.is_zero() && other.is_zero() {
            return true;
        }
        self.0 == other.0
    }
}

impl PartialOrd for F16 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Handle NaN
        if self.is_nan() || other.is_nan() {
            return None;
        }
        // Compare as f32 for simplicity
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

// Debug and Display
impl Debug for F16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "F16({:?})", self.to_f32())
    }
}

impl Display for F16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

// From implementations
impl From<f32> for F16 {
    #[inline]
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl From<f64> for F16 {
    #[inline]
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

impl From<F16> for f32 {
    #[inline]
    fn from(value: F16) -> Self {
        value.to_f32()
    }
}

impl From<F16> for f64 {
    #[inline]
    fn from(value: F16) -> Self {
        value.to_f64()
    }
}

// num_traits implementations

impl num_traits::Zero for F16 {
    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    fn is_zero(&self) -> bool {
        Self::is_zero(*self)
    }
}

impl num_traits::One for F16 {
    #[inline]
    fn one() -> Self {
        Self::ONE
    }
}

impl num_traits::Num for F16 {
    type FromStrRadixErr = num_traits::ParseFloatError;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f32::from_str_radix(str, radix).map(Self::from_f32)
    }
}

impl num_traits::NumCast for F16 {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        n.to_f32().map(Self::from_f32)
    }
}

impl num_traits::ToPrimitive for F16 {
    fn to_i64(&self) -> Option<i64> {
        Some(Self::to_f32(*self) as i64)
    }

    fn to_u64(&self) -> Option<u64> {
        Some(Self::to_f32(*self) as u64)
    }

    fn to_f32(&self) -> Option<f32> {
        Some(Self::to_f32(*self))
    }

    fn to_f64(&self) -> Option<f64> {
        Some(Self::to_f64(*self))
    }
}

impl num_traits::Float for F16 {
    fn nan() -> Self {
        Self::NAN
    }
    fn infinity() -> Self {
        Self::INFINITY
    }
    fn neg_infinity() -> Self {
        Self::NEG_INFINITY
    }
    fn neg_zero() -> Self {
        Self(0x8000)
    }
    fn min_value() -> Self {
        Self::MIN
    }
    fn min_positive_value() -> Self {
        Self::MIN_POSITIVE
    }
    fn max_value() -> Self {
        Self::MAX
    }

    fn is_nan(self) -> bool {
        Self::is_nan(self)
    }
    fn is_infinite(self) -> bool {
        Self::is_infinite(self)
    }
    fn is_finite(self) -> bool {
        Self::is_finite(self)
    }
    fn is_normal(self) -> bool {
        Self::is_normal(self)
    }
    fn classify(self) -> core::num::FpCategory {
        Self::classify(self)
    }

    fn floor(self) -> Self {
        Self::floor(self)
    }
    fn ceil(self) -> Self {
        Self::ceil(self)
    }
    fn round(self) -> Self {
        Self::round(self)
    }
    fn trunc(self) -> Self {
        Self::trunc(self)
    }
    fn fract(self) -> Self {
        Self::fract(self)
    }
    fn abs(self) -> Self {
        Self::abs(self)
    }
    fn signum(self) -> Self {
        Self::signum(self)
    }
    fn is_sign_positive(self) -> bool {
        Self::is_sign_positive(self)
    }
    fn is_sign_negative(self) -> bool {
        Self::is_sign_negative(self)
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self::mul_add(self, a, b)
    }
    fn recip(self) -> Self {
        Self::recip(self)
    }
    fn powi(self, n: i32) -> Self {
        Self::powi(self, n)
    }
    fn powf(self, n: Self) -> Self {
        Self::powf(self, n)
    }
    fn sqrt(self) -> Self {
        Self::sqrt(self)
    }
    fn exp(self) -> Self {
        Self::exp(self)
    }
    fn exp2(self) -> Self {
        Self::exp2(self)
    }
    fn ln(self) -> Self {
        Self::ln(self)
    }
    fn log(self, base: Self) -> Self {
        Self::log(self, base)
    }
    fn log2(self) -> Self {
        Self::log2(self)
    }
    fn log10(self) -> Self {
        Self::log10(self)
    }
    fn max(self, other: Self) -> Self {
        Self::max(self, other)
    }
    fn min(self, other: Self) -> Self {
        Self::min(self, other)
    }
    fn abs_sub(self, other: Self) -> Self {
        Self::abs_sub(self, other)
    }
    fn cbrt(self) -> Self {
        Self::cbrt(self)
    }
    fn hypot(self, other: Self) -> Self {
        Self::hypot(self, other)
    }
    fn sin(self) -> Self {
        Self::sin(self)
    }
    fn cos(self) -> Self {
        Self::cos(self)
    }
    fn tan(self) -> Self {
        Self::tan(self)
    }
    fn asin(self) -> Self {
        Self::asin(self)
    }
    fn acos(self) -> Self {
        Self::acos(self)
    }
    fn atan(self) -> Self {
        Self::atan(self)
    }
    fn atan2(self, other: Self) -> Self {
        Self::atan2(self, other)
    }
    fn sin_cos(self) -> (Self, Self) {
        Self::sin_cos(self)
    }
    fn exp_m1(self) -> Self {
        Self::exp_m1(self)
    }
    fn ln_1p(self) -> Self {
        Self::ln_1p(self)
    }
    fn sinh(self) -> Self {
        Self::sinh(self)
    }
    fn cosh(self) -> Self {
        Self::cosh(self)
    }
    fn tanh(self) -> Self {
        Self::tanh(self)
    }
    fn asinh(self) -> Self {
        Self::asinh(self)
    }
    fn acosh(self) -> Self {
        Self::acosh(self)
    }
    fn atanh(self) -> Self {
        Self::atanh(self)
    }
    fn integer_decode(self) -> (u64, i16, i8) {
        Self::to_f32(self).integer_decode()
    }
    fn epsilon() -> Self {
        Self::EPSILON
    }
    fn to_degrees(self) -> Self {
        Self::to_degrees(self)
    }
    fn to_radians(self) -> Self {
        Self::to_radians(self)
    }
    fn copysign(self, sign: Self) -> Self {
        Self::copysign(self, sign)
    }
}

impl num_traits::FloatConst for F16 {
    fn E() -> Self {
        Self::E
    }
    fn FRAC_1_PI() -> Self {
        Self::ONE / Self::PI
    }
    fn FRAC_1_SQRT_2() -> Self {
        Self::FRAC_1_SQRT_2
    }
    fn FRAC_2_PI() -> Self {
        Self::TWO / Self::PI
    }
    fn FRAC_2_SQRT_PI() -> Self {
        Self::TWO / Self::sqrt(Self::PI)
    }
    fn FRAC_PI_2() -> Self {
        Self::FRAC_PI_2
    }
    fn FRAC_PI_3() -> Self {
        Self::PI / Self::from_f32(3.0)
    }
    fn FRAC_PI_4() -> Self {
        Self::FRAC_PI_4
    }
    fn FRAC_PI_6() -> Self {
        Self::PI / Self::from_f32(6.0)
    }
    fn FRAC_PI_8() -> Self {
        Self::PI / Self::from_f32(8.0)
    }
    fn LN_10() -> Self {
        Self::from_f32(core::f32::consts::LN_10)
    }
    fn LN_2() -> Self {
        Self::LN_2
    }
    fn LOG10_E() -> Self {
        Self::from_f32(core::f32::consts::LOG10_E)
    }
    fn LOG2_E() -> Self {
        Self::from_f32(core::f32::consts::LOG2_E)
    }
    fn PI() -> Self {
        Self::PI
    }
    fn SQRT_2() -> Self {
        Self::from_f32(core::f32::consts::SQRT_2)
    }
    fn TAU() -> Self {
        Self::TWO_PI
    }
}

// Float trait implementation (from our kernel)
impl super::Float for F16 {
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;
    const TWO: Self = Self::TWO;
    const PI: Self = Self::PI;
    const TWO_PI: Self = Self::TWO_PI;

    #[inline]
    fn sin(self) -> Self {
        Self::sin(self)
    }

    #[inline]
    fn cos(self) -> Self {
        Self::cos(self)
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        Self::sin_cos(self)
    }

    #[inline]
    fn sqrt(self) -> Self {
        Self::sqrt(self)
    }

    #[inline]
    fn abs(self) -> Self {
        Self::abs(self)
    }

    #[inline]
    fn from_usize(n: usize) -> Self {
        Self::from_f32(n as f32)
    }

    #[inline]
    fn from_isize(n: isize) -> Self {
        Self::from_f32(n as f32)
    }

    #[inline]
    fn from_f64(n: f64) -> Self {
        Self::from_f64(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_constants() {
        assert!(!F16::ZERO.is_nan());
        assert!(!F16::ONE.is_nan());
        assert!(F16::NAN.is_nan());
        assert!(F16::INFINITY.is_infinite());
        assert!(!F16::ONE.is_infinite());
    }

    #[test]
    fn test_f16_from_f32() {
        assert_eq!(F16::from_f32(0.0), F16::ZERO);
        assert_eq!(F16::from_f32(1.0), F16::ONE);
        assert_eq!(F16::from_f32(-1.0), F16::NEG_ONE);

        // Check approximate equality for non-exact values
        let pi = F16::from_f32(core::f32::consts::PI);
        assert!((pi.to_f32() - core::f32::consts::PI).abs() < 0.01);
    }

    #[test]
    fn test_f16_to_f32() {
        assert_eq!(F16::ZERO.to_f32(), 0.0);
        assert_eq!(F16::ONE.to_f32(), 1.0);
        assert_eq!(F16::NEG_ONE.to_f32(), -1.0);
    }

    #[test]
    fn test_f16_arithmetic() {
        let a = F16::from_f32(2.0);
        let b = F16::from_f32(3.0);

        assert!(((a + b).to_f32() - 5.0).abs() < 0.01);
        assert!(((a - b).to_f32() - (-1.0)).abs() < 0.01);
        assert!(((a * b).to_f32() - 6.0).abs() < 0.01);
        assert!(((b / a).to_f32() - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_f16_negation() {
        let a = F16::from_f32(2.5);
        assert!(((-a).to_f32() - (-2.5)).abs() < 0.01);
    }

    #[test]
    fn test_f16_comparison() {
        let a = F16::from_f32(1.0);
        let b = F16::from_f32(2.0);

        assert!(a < b);
        assert!(b > a);
        assert!(a == F16::ONE);
        assert!(F16::ZERO == F16::from_f32(-0.0)); // +0 == -0
    }

    #[test]
    fn test_f16_nan() {
        assert!(F16::NAN.is_nan());
        assert!(F16::NAN != F16::NAN); // NaN != NaN
        assert!(!(F16::NAN == F16::NAN));
    }

    #[test]
    fn test_f16_infinity() {
        assert!(F16::INFINITY.is_infinite());
        assert!(F16::NEG_INFINITY.is_infinite());
        assert!(!F16::INFINITY.is_nan());
        assert!(F16::INFINITY > F16::MAX);
    }

    #[test]
    fn test_f16_trig() {
        let x = F16::from_f32(0.0);
        assert!((x.sin().to_f32() - 0.0).abs() < 0.01);
        assert!((x.cos().to_f32() - 1.0).abs() < 0.01);

        let pi_2 = F16::from_f32(core::f32::consts::FRAC_PI_2);
        assert!((pi_2.sin().to_f32() - 1.0).abs() < 0.01);
        assert!(pi_2.cos().to_f32().abs() < 0.1);
    }

    #[test]
    fn test_f16_float_trait() {
        use super::super::Float;

        let a = F16::from_usize(10);
        assert!((a.to_f32() - 10.0).abs() < 0.1);

        let b = <F16 as Float>::from_f64(2.5);
        assert!((b.to_f32() - 2.5).abs() < 0.01);

        let c = F16::from_f32(4.0);
        assert!((<F16 as Float>::sqrt(c).to_f32() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_f16_abs() {
        let neg = F16::from_f32(-5.0);
        let pos = F16::from_f32(5.0);

        assert!((neg.abs().to_f32() - 5.0).abs() < 0.01);
        assert!((pos.abs().to_f32() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_f16_round_trip() {
        // Test various values round-trip through f32
        let values = [0.0_f32, 1.0, -1.0, 0.5, 100.0, 0.001, 65504.0];

        for &v in &values {
            let f16_val = F16::from_f32(v);
            let back = f16_val.to_f32();
            // Allow for f16 precision loss
            if v.abs() > 0.0 {
                assert!(
                    (back - v).abs() / v.abs() < 0.01,
                    "Round-trip failed for {v}"
                );
            } else {
                assert!(back == 0.0);
            }
        }
    }

    #[test]
    fn test_f16_num_traits() {
        use num_traits::{Float, One, Zero};

        assert!(F16::zero().is_zero());
        assert!(F16::one() == F16::ONE);
        assert!(F16::infinity().is_infinite());
        assert!(F16::nan().is_nan());
    }
}
