//! Complex number type for FFT operations.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::Float;

/// A complex number with real and imaginary parts.
#[derive(Copy, Clone, Default, PartialEq)]
#[repr(C)]
pub struct Complex<T: Float> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}

impl<T: Float> Complex<T> {
    /// Create a new complex number.
    #[inline]
    #[must_use]
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }

    /// Create zero (0 + 0i).
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        Self::new(T::ZERO, T::ZERO)
    }

    /// Create one (1 + 0i).
    #[inline]
    #[must_use]
    pub fn one() -> Self {
        Self::new(T::ONE, T::ZERO)
    }

    /// Create the imaginary unit (0 + 1i).
    #[inline]
    #[must_use]
    pub fn i() -> Self {
        Self::new(T::ZERO, T::ONE)
    }

    /// Create from polar coordinates (r, θ).
    #[inline]
    #[must_use]
    pub fn from_polar(r: T, theta: T) -> Self {
        let (sin, cos) = Float::sin_cos(theta);
        Self::new(r * cos, r * sin)
    }

    /// Complex conjugate.
    #[inline]
    #[must_use]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// Squared magnitude |z|².
    #[inline]
    #[must_use]
    pub fn norm_sqr(self) -> T {
        self.re * self.re + self.im * self.im
    }

    /// Magnitude |z|.
    #[inline]
    #[must_use]
    pub fn norm(self) -> T {
        Float::sqrt(self.norm_sqr())
    }

    /// Phase angle (argument).
    #[inline]
    #[must_use]
    pub fn arg(self) -> T {
        num_traits::Float::atan2(self.im, self.re)
    }

    /// Multiplicative inverse: 1/z.
    #[inline]
    #[must_use]
    pub fn inv(self) -> Self {
        let norm_sq = self.norm_sqr();
        Self::new(self.re / norm_sq, -self.im / norm_sq)
    }

    /// Compute e^(i*theta) = cos(theta) + i*sin(theta).
    #[inline]
    #[must_use]
    pub fn cis(theta: T) -> Self {
        let (sin, cos) = Float::sin_cos(theta);
        Self::new(cos, sin)
    }

    /// Scale by a real number.
    #[inline]
    #[must_use]
    pub fn scale(self, s: T) -> Self {
        Self::new(self.re * s, self.im * s)
    }
}

impl<T: Float> fmt::Debug for Complex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} + {:?}i", self.re, self.im)
    }
}

impl<T: Float> fmt::Display for Complex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} + {:?}i", self.re, self.im)
    }
}

// Arithmetic operations

impl<T: Float> Add for Complex<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
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
    fn sub(self, rhs: Self) -> Self::Output {
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
    fn mul(self, rhs: Self) -> Self::Output {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
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
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inv()
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
    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.im)
    }
}

// Scalar operations

impl<T: Float> Mul<T> for Complex<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl<T: Float> Div<T> for Complex<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(1.0_f64, 2.0);
        let b = Complex::new(3.0, 4.0);

        let sum = a + b;
        assert!((sum.re - 4.0).abs() < 1e-10);
        assert!((sum.im - 6.0).abs() < 1e-10);

        let prod = a * b;
        // (1 + 2i)(3 + 4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
        assert!((prod.re - (-5.0)).abs() < 1e-10);
        assert!((prod.im - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_complex_conjugate() {
        let z = Complex::new(3.0_f64, 4.0);
        let conj = z.conj();
        assert!((conj.re - 3.0).abs() < 1e-10);
        assert!((conj.im - (-4.0)).abs() < 1e-10);
    }

    #[test]
    fn test_complex_norm() {
        let z = Complex::new(3.0_f64, 4.0);
        assert!((z.norm() - 5.0).abs() < 1e-10);
    }
}
