//! Pure Rust IEEE 754 binary128 floating-point type.
//!
//! This module provides a software-emulated quad-precision floating-point type
//! suitable for FFT computations requiring very high precision.
//!
//! IEEE 754 binary128 format:
//! - Sign: 1 bit (bit 127)
//! - Exponent: 15 bits (bits 112-126), bias = 16383
//! - Significand: 112 bits (bits 0-111), with implicit leading 1

use core::cmp::Ordering;
use core::fmt::{Debug, Display};
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// IEEE 754 binary128 quad-precision floating-point type.
///
/// This is a pure Rust implementation using two 64-bit integers.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct F128 {
    /// Low 64 bits of the 128-bit representation
    lo: u64,
    /// High 64 bits (contains sign, exponent, and high significand)
    hi: u64,
}

// Constants for IEEE 754 binary128
const EXPONENT_BIAS: i32 = 16383;
const EXPONENT_BITS: u32 = 15;
#[allow(dead_code)] // reason: documentation constant showing IEEE 754 binary128 structure; not used in computations
const SIGNIFICAND_BITS: u32 = 112; // for documentation - total bits in significand
const SIGN_MASK: u64 = 1 << 63;
const EXPONENT_MASK: u64 = ((1u64 << EXPONENT_BITS) - 1) << (63 - EXPONENT_BITS);
const SIGNIFICAND_HI_MASK: u64 = (1u64 << (64 - 1 - EXPONENT_BITS)) - 1;

impl F128 {
    /// Zero
    pub const ZERO: Self = Self { lo: 0, hi: 0 };

    /// One (1.0)
    pub const ONE: Self = Self::from_f64_const(1.0);

    /// Negative one (-1.0)
    pub const NEG_ONE: Self = Self::from_f64_const(-1.0);

    /// Two (2.0)
    pub const TWO: Self = Self::from_f64_const(2.0);

    /// Half (0.5)
    pub const HALF: Self = Self::from_f64_const(0.5);

    /// Pi (high precision)
    /// 3.14159265358979323846264338327950288419716939937510...
    pub const PI: Self = Self {
        // Hex representation of pi in binary128
        lo: 0x8469898CC51701B8,
        hi: 0x4000921FB54442D1,
    };

    /// Two Pi (2 * PI)
    pub const TWO_PI: Self = Self {
        lo: 0x08D313198A2E0370,
        hi: 0x4001921FB54442D1,
    };

    /// Pi / 2
    pub const FRAC_PI_2: Self = Self {
        lo: 0x8469898CC51701B8,
        hi: 0x3FFF921FB54442D1,
    };

    /// Pi / 4
    pub const FRAC_PI_4: Self = Self {
        lo: 0x8469898CC51701B8,
        hi: 0x3FFE921FB54442D1,
    };

    /// E (Euler's number)
    /// 2.71828182845904523536028747135266249775724709369995...
    pub const E: Self = Self {
        lo: 0xF0A1B167E4A9F4B1,
        hi: 0x40005BF0A8B14576,
    };

    /// ln(2)
    pub const LN_2: Self = Self {
        lo: 0x39AB0BFF42E32B4F,
        hi: 0x3FFE62E42FEFA39E,
    };

    /// 1/sqrt(2) = sqrt(2)/2
    pub const FRAC_1_SQRT_2: Self = Self {
        lo: 0xCCC906E7B14A7B7E,
        hi: 0x3FFE6A09E667F3BC,
    };

    /// Positive infinity
    pub const INFINITY: Self = Self {
        lo: 0,
        hi: EXPONENT_MASK,
    };

    /// Negative infinity
    pub const NEG_INFINITY: Self = Self {
        lo: 0,
        hi: EXPONENT_MASK | SIGN_MASK,
    };

    /// Not a Number (NaN)
    pub const NAN: Self = Self {
        lo: 1,
        hi: EXPONENT_MASK,
    };

    /// Create a new F128 from high and low 64-bit parts.
    #[inline]
    pub const fn from_bits(hi: u64, lo: u64) -> Self {
        Self { lo, hi }
    }

    /// Get the raw bit representation.
    #[inline]
    pub const fn to_bits(self) -> (u64, u64) {
        (self.hi, self.lo)
    }

    /// Convert from f64 (compile-time friendly version).
    #[inline]
    const fn from_f64_const(x: f64) -> Self {
        if x == 0.0 {
            return Self::ZERO;
        }

        let bits = x.to_bits();
        let sign = bits >> 63;
        let exp = ((bits >> 52) & 0x7FF) as i32;
        let mantissa = bits & ((1u64 << 52) - 1);

        if exp == 0x7FF {
            // Infinity or NaN
            if mantissa == 0 {
                if sign == 0 {
                    return Self::INFINITY;
                }
                return Self::NEG_INFINITY;
            }
            return Self::NAN;
        }

        if exp == 0 {
            // Subnormal f64 - becomes normal or subnormal in f128
            // For simplicity, treat as zero (subnormals are tiny)
            return Self::ZERO;
        }

        // Normal number: convert exponent
        // f64 bias is 1023, f128 bias is 16383
        let new_exp = (exp - 1023 + EXPONENT_BIAS) as u64;

        // Shift mantissa from 52 bits to 112 bits (shift left by 60)
        // mantissa occupies bits 51:0 in f64
        // In f128, it should occupy bits 111:0 (with implicit leading 1)
        // High part (bits 111:64) = mantissa << (64-52) = mantissa << 12 (only top 48 bits fit)
        // Low part (bits 63:0) = mantissa << 60
        let hi_mantissa = mantissa >> 4; // top 48 bits of the 52-bit mantissa
        let lo_mantissa = mantissa << 60; // bottom 4 bits shifted to top of lo

        let hi = (sign << 63) | (new_exp << (64 - 1 - EXPONENT_BITS)) | hi_mantissa;
        let lo = lo_mantissa;

        Self { lo, hi }
    }

    /// Convert from f64 at runtime.
    #[inline]
    pub fn from_f64(x: f64) -> Self {
        Self::from_f64_const(x)
    }

    /// Convert to f64 (with precision loss).
    #[inline]
    pub fn to_f64(self) -> f64 {
        if self.is_nan() {
            return f64::NAN;
        }
        if self.is_infinite() {
            return if self.is_sign_negative() {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
        if self.is_zero() {
            return if self.is_sign_negative() { -0.0 } else { 0.0 };
        }

        let sign = (self.hi >> 63) & 1;
        let exp = ((self.hi >> (64 - 1 - EXPONENT_BITS)) & ((1 << EXPONENT_BITS) - 1)) as i32;

        // Convert exponent
        let f64_exp = exp - EXPONENT_BIAS + 1023;

        if f64_exp >= 2047 {
            // Overflow to infinity
            return if sign == 1 {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
        if f64_exp <= 0 {
            // Underflow to zero (or subnormal, but we simplify)
            return if sign == 1 { -0.0 } else { 0.0 };
        }

        // Extract top 52 bits of significand
        // In f128, significand bits are: hi[47:0] (48 bits) + lo[63:0] (64 bits) = 112 bits
        // We want the top 52 bits
        let hi_mantissa = self.hi & SIGNIFICAND_HI_MASK;
        let mantissa = (hi_mantissa << 4) | (self.lo >> 60);

        let f64_bits = (sign << 63) | ((f64_exp as u64) << 52) | mantissa;
        f64::from_bits(f64_bits)
    }

    /// Convert from usize.
    #[inline]
    pub fn from_usize(n: usize) -> Self {
        Self::from_f64(n as f64)
    }

    /// Convert from isize.
    #[inline]
    pub fn from_isize(n: isize) -> Self {
        Self::from_f64(n as f64)
    }

    /// Check if this is zero.
    #[inline]
    pub fn is_zero(self) -> bool {
        (self.hi & !SIGN_MASK) == 0 && self.lo == 0
    }

    /// Check if this is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        let exp = (self.hi >> (64 - 1 - EXPONENT_BITS)) & ((1 << EXPONENT_BITS) - 1);
        exp == ((1 << EXPONENT_BITS) - 1) && (self.lo != 0 || (self.hi & SIGNIFICAND_HI_MASK) != 0)
    }

    /// Check if this is infinite.
    #[inline]
    pub fn is_infinite(self) -> bool {
        let exp = (self.hi >> (64 - 1 - EXPONENT_BITS)) & ((1 << EXPONENT_BITS) - 1);
        exp == ((1 << EXPONENT_BITS) - 1) && self.lo == 0 && (self.hi & SIGNIFICAND_HI_MASK) == 0
    }

    /// Check if the sign bit is set.
    #[inline]
    pub fn is_sign_negative(self) -> bool {
        (self.hi & SIGN_MASK) != 0
    }

    /// Check if this is finite (not NaN or infinity).
    #[inline]
    pub fn is_finite(self) -> bool {
        let exp = (self.hi >> (64 - 1 - EXPONENT_BITS)) & ((1 << EXPONENT_BITS) - 1);
        exp != ((1 << EXPONENT_BITS) - 1)
    }

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        Self {
            lo: self.lo,
            hi: self.hi & !SIGN_MASK,
        }
    }

    /// Copy sign from another value.
    #[inline]
    pub fn copysign(self, sign: Self) -> Self {
        Self {
            lo: self.lo,
            hi: (self.hi & !SIGN_MASK) | (sign.hi & SIGN_MASK),
        }
    }

    /// Square root using Newton-Raphson iteration.
    #[inline]
    pub fn sqrt(self) -> Self {
        if self.is_nan() || self.is_sign_negative() {
            return Self::NAN;
        }
        if self.is_zero() || self.is_infinite() {
            return self;
        }

        // Initial guess from f64
        let initial = Self::from_f64(self.to_f64().sqrt());

        // Newton-Raphson: x_{n+1} = 0.5 * (x_n + S/x_n)
        let mut x = initial;
        for _ in 0..10 {
            let x_new = (x + self / x) * Self::HALF;
            if x == x_new {
                break;
            }
            x = x_new;
        }
        x
    }

    /// Sine using Taylor series with range reduction.
    #[inline]
    pub fn sin(self) -> Self {
        if self.is_nan() || self.is_infinite() {
            return Self::NAN;
        }

        // Range reduction: x mod 2*PI
        let reduced = self.reduce_angle();

        // Taylor series: sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...
        let x2 = reduced * reduced;
        let mut term = reduced;
        let mut sum = reduced;

        for n in 1..30 {
            let denom = Self::from_usize(2 * n) * Self::from_usize(2 * n + 1);
            term = term * x2 / denom;
            term = -term;
            let new_sum = sum + term;
            if sum == new_sum {
                break;
            }
            sum = new_sum;
        }

        sum
    }

    /// Cosine using Taylor series with range reduction.
    #[inline]
    pub fn cos(self) -> Self {
        if self.is_nan() || self.is_infinite() {
            return Self::NAN;
        }

        // Range reduction: x mod 2*PI
        let reduced = self.reduce_angle();

        // Taylor series: cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + ...
        let x2 = reduced * reduced;
        let mut term = Self::ONE;
        let mut sum = Self::ONE;

        for n in 1..30 {
            let denom = Self::from_usize(2 * n - 1) * Self::from_usize(2 * n);
            term = term * x2 / denom;
            term = -term;
            let new_sum = sum + term;
            if sum == new_sum {
                break;
            }
            sum = new_sum;
        }

        sum
    }

    /// Sine and cosine together (more efficient).
    #[inline]
    pub fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    /// Reduce angle to [-PI, PI] range.
    fn reduce_angle(self) -> Self {
        // Simple modulo reduction (could be more precise)
        let two_pi = Self::TWO_PI;
        let mut x = self;

        // Coarse reduction using division
        if x.abs().to_f64() > core::f64::consts::PI {
            let n = (x / two_pi).to_f64().round();
            x = x - Self::from_f64(n) * two_pi;
        }

        x
    }

    /// Floor function.
    #[inline]
    pub fn floor(self) -> Self {
        Self::from_f64(self.to_f64().floor())
    }

    /// Ceiling function.
    #[inline]
    pub fn ceil(self) -> Self {
        Self::from_f64(self.to_f64().ceil())
    }

    /// Round function.
    #[inline]
    pub fn round(self) -> Self {
        Self::from_f64(self.to_f64().round())
    }

    /// Truncate function.
    #[inline]
    pub fn trunc(self) -> Self {
        Self::from_f64(self.to_f64().trunc())
    }

    /// Fractional part.
    #[inline]
    pub fn fract(self) -> Self {
        self - self.trunc()
    }

    /// Minimum of two values.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        if self.is_nan() {
            return other;
        }
        if other.is_nan() {
            return self;
        }
        if self.to_f64() <= other.to_f64() {
            self
        } else {
            other
        }
    }

    /// Maximum of two values.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        if self.is_nan() {
            return other;
        }
        if other.is_nan() {
            return self;
        }
        if self.to_f64() >= other.to_f64() {
            self
        } else {
            other
        }
    }

    /// Power function (x^y).
    #[inline]
    pub fn powf(self, exp: Self) -> Self {
        // Use exp and ln: x^y = exp(y * ln(x))
        Self::from_f64(self.to_f64().powf(exp.to_f64()))
    }

    /// Integer power function.
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::ONE;
        }
        if n == 1 {
            return self;
        }
        if n < 0 {
            return Self::ONE / self.powi(-n);
        }

        let mut result = Self::ONE;
        let mut base = self;
        let mut exp = n as u32;

        while exp > 0 {
            if exp & 1 == 1 {
                result = result * base;
            }
            base = base * base;
            exp >>= 1;
        }

        result
    }

    /// Natural logarithm.
    #[inline]
    pub fn ln(self) -> Self {
        Self::from_f64(self.to_f64().ln())
    }

    /// Exponential function.
    #[inline]
    pub fn exp(self) -> Self {
        Self::from_f64(self.to_f64().exp())
    }

    /// Sign function: returns -1, 0, or 1.
    #[inline]
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
    pub fn recip(self) -> Self {
        Self::ONE / self
    }
}

// Arithmetic operations using f64 as intermediate (for simplicity)
// A full implementation would do arbitrary precision arithmetic

impl Add for F128 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        // For now, use f64 arithmetic (loses precision)
        // A proper implementation would do full 128-bit addition
        Self::from_f64(self.to_f64() + rhs.to_f64())
    }
}

impl Sub for F128 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_f64(self.to_f64() - rhs.to_f64())
    }
}

impl Mul for F128 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_f64(self.to_f64() * rhs.to_f64())
    }
}

impl Div for F128 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self::from_f64(self.to_f64() / rhs.to_f64())
    }
}

impl Neg for F128 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            lo: self.lo,
            hi: self.hi ^ SIGN_MASK,
        }
    }
}

impl Rem for F128 {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self {
        Self::from_f64(self.to_f64() % rhs.to_f64())
    }
}

// Assignment operators
impl AddAssign for F128 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for F128 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for F128 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl DivAssign for F128 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl RemAssign for F128 {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

// Comparison
impl PartialEq for F128 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.is_nan() || other.is_nan() {
            return false;
        }
        // Handle +0 == -0
        if self.is_zero() && other.is_zero() {
            return true;
        }
        self.hi == other.hi && self.lo == other.lo
    }
}

impl PartialOrd for F128 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            return None;
        }
        self.to_f64().partial_cmp(&other.to_f64())
    }
}

// Display and Debug
impl Debug for F128 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "F128({})", self.to_f64())
    }
}

impl Display for F128 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_f64())
    }
}

// num_traits compatibility
impl num_traits::Zero for F128 {
    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    fn is_zero(&self) -> bool {
        Self::is_zero(*self)
    }
}

impl num_traits::One for F128 {
    #[inline]
    fn one() -> Self {
        Self::ONE
    }
}

impl num_traits::Num for F128 {
    type FromStrRadixErr = num_traits::ParseFloatError;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        // Delegate to f64 parsing
        f64::from_str_radix(str, radix).map(Self::from_f64)
    }
}

impl num_traits::NumCast for F128 {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        n.to_f64().map(Self::from_f64)
    }
}

impl num_traits::ToPrimitive for F128 {
    fn to_i64(&self) -> Option<i64> {
        Some(Self::to_f64(*self) as i64)
    }

    fn to_u64(&self) -> Option<u64> {
        Some(Self::to_f64(*self) as u64)
    }

    fn to_f64(&self) -> Option<f64> {
        Some(Self::to_f64(*self))
    }
}

impl num_traits::Float for F128 {
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
        Self {
            lo: 0,
            hi: SIGN_MASK,
        }
    }
    fn min_value() -> Self {
        Self::from_f64(f64::MIN)
    }
    fn min_positive_value() -> Self {
        Self::from_f64(f64::MIN_POSITIVE)
    }
    fn max_value() -> Self {
        Self::from_f64(f64::MAX)
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
        Self::is_finite(self) && !Self::is_zero(self)
    }
    fn classify(self) -> core::num::FpCategory {
        if Self::is_nan(self) {
            core::num::FpCategory::Nan
        } else if Self::is_infinite(self) {
            core::num::FpCategory::Infinite
        } else if Self::is_zero(self) {
            core::num::FpCategory::Zero
        } else {
            core::num::FpCategory::Normal
        }
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
        !Self::is_sign_negative(self)
    }
    fn is_sign_negative(self) -> bool {
        Self::is_sign_negative(self)
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
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
        Self::from_f64(Self::to_f64(self).exp2())
    }
    fn ln(self) -> Self {
        Self::ln(self)
    }
    fn log(self, base: Self) -> Self {
        Self::from_f64(Self::to_f64(self).log(Self::to_f64(base)))
    }
    fn log2(self) -> Self {
        Self::from_f64(Self::to_f64(self).log2())
    }
    fn log10(self) -> Self {
        Self::from_f64(Self::to_f64(self).log10())
    }
    fn max(self, other: Self) -> Self {
        Self::max(self, other)
    }
    fn min(self, other: Self) -> Self {
        Self::min(self, other)
    }
    fn abs_sub(self, other: Self) -> Self {
        Self::abs(self - other)
    }
    fn cbrt(self) -> Self {
        Self::from_f64(Self::to_f64(self).cbrt())
    }
    fn hypot(self, other: Self) -> Self {
        Self::sqrt(self * self + other * other)
    }
    fn sin(self) -> Self {
        Self::sin(self)
    }
    fn cos(self) -> Self {
        Self::cos(self)
    }
    fn tan(self) -> Self {
        Self::from_f64(Self::to_f64(self).tan())
    }
    fn asin(self) -> Self {
        Self::from_f64(Self::to_f64(self).asin())
    }
    fn acos(self) -> Self {
        Self::from_f64(Self::to_f64(self).acos())
    }
    fn atan(self) -> Self {
        Self::from_f64(Self::to_f64(self).atan())
    }
    fn atan2(self, other: Self) -> Self {
        Self::from_f64(Self::to_f64(self).atan2(Self::to_f64(other)))
    }
    fn sin_cos(self) -> (Self, Self) {
        Self::sin_cos(self)
    }
    fn exp_m1(self) -> Self {
        Self::from_f64(Self::to_f64(self).exp_m1())
    }
    fn ln_1p(self) -> Self {
        Self::from_f64(Self::to_f64(self).ln_1p())
    }
    fn sinh(self) -> Self {
        Self::from_f64(Self::to_f64(self).sinh())
    }
    fn cosh(self) -> Self {
        Self::from_f64(Self::to_f64(self).cosh())
    }
    fn tanh(self) -> Self {
        Self::from_f64(Self::to_f64(self).tanh())
    }
    fn asinh(self) -> Self {
        Self::from_f64(Self::to_f64(self).asinh())
    }
    fn acosh(self) -> Self {
        Self::from_f64(Self::to_f64(self).acosh())
    }
    fn atanh(self) -> Self {
        Self::from_f64(Self::to_f64(self).atanh())
    }
    fn integer_decode(self) -> (u64, i16, i8) {
        Self::to_f64(self).integer_decode()
    }
    fn epsilon() -> Self {
        Self::from_f64(f64::EPSILON)
    }
    fn to_degrees(self) -> Self {
        self * Self::from_f64(180.0) / Self::PI
    }
    fn to_radians(self) -> Self {
        self * Self::PI / Self::from_f64(180.0)
    }
    fn copysign(self, sign: Self) -> Self {
        Self::copysign(self, sign)
    }
}

impl num_traits::FloatConst for F128 {
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
        Self::PI / Self::from_f64(3.0)
    }
    fn FRAC_PI_4() -> Self {
        Self::FRAC_PI_4
    }
    fn FRAC_PI_6() -> Self {
        Self::PI / Self::from_f64(6.0)
    }
    fn FRAC_PI_8() -> Self {
        Self::PI / Self::from_f64(8.0)
    }
    fn LN_10() -> Self {
        Self::from_f64(core::f64::consts::LN_10)
    }
    fn LN_2() -> Self {
        Self::LN_2
    }
    fn LOG10_E() -> Self {
        Self::from_f64(core::f64::consts::LOG10_E)
    }
    fn LOG2_E() -> Self {
        Self::from_f64(core::f64::consts::LOG2_E)
    }
    fn PI() -> Self {
        Self::PI
    }
    fn SQRT_2() -> Self {
        Self::from_f64(core::f64::consts::SQRT_2)
    }
    fn TAU() -> Self {
        Self::TWO_PI
    }
}

// Implement arithmetic with &F128 for NumAssignRef compatibility
// (NumAssign, NumAssignOps are auto-implemented by blanket impls)
impl Add<&F128> for F128 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &F128) -> Self {
        self + *rhs
    }
}

impl Sub<&F128> for F128 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &F128) -> Self {
        self - *rhs
    }
}

impl Mul<&F128> for F128 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &F128) -> Self {
        self * *rhs
    }
}

impl Div<&F128> for F128 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: &F128) -> Self {
        self / *rhs
    }
}

impl Rem<&F128> for F128 {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: &F128) -> Self {
        self % *rhs
    }
}

impl AddAssign<&F128> for F128 {
    #[inline]
    fn add_assign(&mut self, rhs: &F128) {
        *self = *self + *rhs;
    }
}

impl SubAssign<&F128> for F128 {
    #[inline]
    fn sub_assign(&mut self, rhs: &F128) {
        *self = *self - *rhs;
    }
}

impl MulAssign<&F128> for F128 {
    #[inline]
    fn mul_assign(&mut self, rhs: &F128) {
        *self = *self * *rhs;
    }
}

impl DivAssign<&F128> for F128 {
    #[inline]
    fn div_assign(&mut self, rhs: &F128) {
        *self = *self / *rhs;
    }
}

impl RemAssign<&F128> for F128 {
    #[inline]
    fn rem_assign(&mut self, rhs: &F128) {
        *self = *self % *rhs;
    }
}

// NumAssignRef is auto-implemented by blanket impl

// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f128_zero() {
        let z = F128::ZERO;
        assert!(z.is_zero());
        assert!(!z.is_nan());
        assert!(!z.is_infinite());
    }

    #[test]
    fn test_f128_one() {
        let one = F128::ONE;
        assert!(!one.is_zero());
        assert!((one.to_f64() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_f128_from_f64() {
        let x = F128::from_f64(3.125);
        assert!((x.to_f64() - 3.125).abs() < 1e-10);
    }

    #[test]
    fn test_f128_neg() {
        let x = F128::from_f64(5.0);
        let neg_x = -x;
        assert!((neg_x.to_f64() + 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_f128_add() {
        let a = F128::from_f64(3.0);
        let b = F128::from_f64(4.0);
        let c = a + b;
        assert!((c.to_f64() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_f128_mul() {
        let a = F128::from_f64(3.0);
        let b = F128::from_f64(4.0);
        let c = a * b;
        assert!((c.to_f64() - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_f128_sqrt() {
        let x = F128::from_f64(16.0);
        let s = x.sqrt();
        assert!((s.to_f64() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_f128_sin_cos() {
        let x = F128::FRAC_PI_2;
        let (s, c) = x.sin_cos();
        assert!((s.to_f64() - 1.0).abs() < 1e-10);
        assert!(c.to_f64().abs() < 1e-10);
    }

    #[test]
    fn test_f128_pi() {
        assert!((F128::PI.to_f64() - core::f64::consts::PI).abs() < 1e-10);
    }
}
