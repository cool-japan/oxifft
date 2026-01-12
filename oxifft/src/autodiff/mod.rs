//! Automatic differentiation for FFT operations.
//!
//! This module provides forward and backward mode automatic differentiation
//! through FFT operations, enabling gradient-based optimization in signal
//! processing and machine learning pipelines.
//!
//! # Key Insight
//!
//! The FFT is a linear operation, so its Jacobian is simply the DFT matrix.
//! For gradient computation:
//! - Forward FFT derivative: FFT of the input tangent
//! - Backward FFT gradient: IFFT of the output gradient (scaled)
//!
//! # Applications
//!
//! - Machine learning with spectral features
//! - Inverse problems in signal processing
//! - Optimization of filter designs
//! - Phase retrieval algorithms
//!
//! # Example
//!
//! ```ignore
//! use oxifft::autodiff::{DualComplex, fft_dual, grad_fft};
//!
//! // Forward mode: compute FFT and its directional derivative
//! let x = vec![DualComplex::new(1.0, 0.0, 1.0, 0.0); 8];
//! let (y, dy) = fft_dual(&x);
//!
//! // Backward mode: compute gradient of loss w.r.t. FFT input
//! let grad_output = vec![Complex::new(1.0, 0.0); 8];
//! let grad_input = grad_fft(&grad_output);
//! ```

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};

/// Dual number for forward-mode automatic differentiation.
///
/// Represents a value and its derivative: x + ε·dx where ε² = 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual<T: Float> {
    /// Primal value.
    pub value: T,
    /// Derivative (tangent).
    pub deriv: T,
}

impl<T: Float> Dual<T> {
    /// Create a new dual number.
    pub fn new(value: T, deriv: T) -> Self {
        Self { value, deriv }
    }

    /// Create a constant (derivative is zero).
    pub fn constant(value: T) -> Self {
        Self {
            value,
            deriv: T::ZERO,
        }
    }

    /// Create a variable (derivative is one).
    pub fn variable(value: T) -> Self {
        Self {
            value,
            deriv: T::ONE,
        }
    }
}

impl<T: Float> core::ops::Add for Dual<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.value + rhs.value, self.deriv + rhs.deriv)
    }
}

impl<T: Float> core::ops::Sub for Dual<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.value - rhs.value, self.deriv - rhs.deriv)
    }
}

impl<T: Float> core::ops::Mul for Dual<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        // (a + εb)(c + εd) = ac + ε(ad + bc)
        Self::new(
            self.value * rhs.value,
            self.value * rhs.deriv + self.deriv * rhs.value,
        )
    }
}

impl<T: Float> core::ops::Div for Dual<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        // (a + εb)/(c + εd) = a/c + ε(bc - ad)/c²
        let val = self.value / rhs.value;
        let deriv = (self.deriv * rhs.value - self.value * rhs.deriv) / (rhs.value * rhs.value);
        Self::new(val, deriv)
    }
}

/// Complex dual number for differentiating complex-valued functions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualComplex<T: Float> {
    /// Primal value (complex).
    pub value: Complex<T>,
    /// Derivative (complex tangent).
    pub deriv: Complex<T>,
}

impl<T: Float> DualComplex<T> {
    /// Create a new complex dual number.
    pub fn new(re: T, im: T, dre: T, dim: T) -> Self {
        Self {
            value: Complex::new(re, im),
            deriv: Complex::new(dre, dim),
        }
    }

    /// Create from complex values.
    pub fn from_complex(value: Complex<T>, deriv: Complex<T>) -> Self {
        Self { value, deriv }
    }

    /// Create a constant (derivative is zero).
    pub fn constant(value: Complex<T>) -> Self {
        Self {
            value,
            deriv: Complex::zero(),
        }
    }

    /// Create a variable (derivative equals identity direction).
    pub fn variable(value: Complex<T>) -> Self {
        Self {
            value,
            deriv: Complex::new(T::ONE, T::ZERO),
        }
    }

    /// Get the zero dual complex.
    pub fn zero() -> Self {
        Self {
            value: Complex::zero(),
            deriv: Complex::zero(),
        }
    }
}

impl<T: Float> core::ops::Add for DualComplex<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_complex(self.value + rhs.value, self.deriv + rhs.deriv)
    }
}

impl<T: Float> core::ops::Sub for DualComplex<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_complex(self.value - rhs.value, self.deriv - rhs.deriv)
    }
}

impl<T: Float> core::ops::Mul for DualComplex<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        // (a + εb)(c + εd) = ac + ε(ad + bc)
        Self::from_complex(
            self.value * rhs.value,
            self.value * rhs.deriv + self.deriv * rhs.value,
        )
    }
}

impl<T: Float> core::ops::Mul<Complex<T>> for DualComplex<T> {
    type Output = Self;
    fn mul(self, rhs: Complex<T>) -> Self {
        Self::from_complex(self.value * rhs, self.deriv * rhs)
    }
}

/// Differentiable FFT plan for automatic differentiation.
///
/// Wraps a standard FFT plan and provides differentiation capabilities.
pub struct DiffFftPlan<T: Float> {
    /// Forward FFT plan.
    fwd_plan: Plan<T>,
    /// Inverse FFT plan (for gradients).
    inv_plan: Plan<T>,
    /// Transform size.
    size: usize,
}

impl<T: Float> DiffFftPlan<T> {
    /// Create a new differentiable FFT plan.
    pub fn new(size: usize) -> Option<Self> {
        let fwd_plan = Plan::dft_1d(size, Direction::Forward, Flags::MEASURE)?;
        let inv_plan = Plan::dft_1d(size, Direction::Backward, Flags::MEASURE)?;

        Some(Self {
            fwd_plan,
            inv_plan,
            size,
        })
    }

    /// Execute forward FFT.
    pub fn forward(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        self.fwd_plan.execute(input, output);
    }

    /// Execute inverse FFT (normalized).
    pub fn inverse(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        self.inv_plan.execute(input, output);
        let scale = T::ONE / T::from_usize(self.size);
        for c in output.iter_mut() {
            *c = Complex::new(c.re * scale, c.im * scale);
        }
    }

    /// Compute forward FFT with its Jacobian-vector product (forward mode AD).
    ///
    /// Given input x and tangent dx, computes:
    /// - y = FFT(x)
    /// - dy = FFT(dx) (the directional derivative)
    pub fn forward_dual(&self, input: &[DualComplex<T>]) -> (Vec<Complex<T>>, Vec<Complex<T>>) {
        let n = input.len();

        // Extract values and tangents
        let values: Vec<Complex<T>> = input.iter().map(|d| d.value).collect();
        let tangents: Vec<Complex<T>> = input.iter().map(|d| d.deriv).collect();

        // FFT of values
        let mut y = vec![Complex::<T>::zero(); n];
        self.forward(&values, &mut y);

        // FFT of tangents (this is the directional derivative)
        let mut dy = vec![Complex::<T>::zero(); n];
        self.forward(&tangents, &mut dy);

        (y, dy)
    }

    /// Compute gradient of a scalar loss with respect to FFT input (backward mode AD).
    ///
    /// Given the gradient of loss w.r.t. FFT output (grad_output),
    /// computes the gradient w.r.t. FFT input.
    ///
    /// For FFT: y = F·x where F is the DFT matrix `F[k,n] = exp(-2πi·k·n/N)`
    /// The adjoint F^H satisfies: <v, F·x> = <F^H·v, x>
    /// F^H = conj(F^T) = IFFT_unnormalized = N · IFFT
    /// Gradient: ∂L/∂x = F^H · ∂L/∂y = conj(FFT(conj(∂L/∂y)))
    pub fn backward(&self, grad_output: &[Complex<T>]) -> Vec<Complex<T>> {
        let n = grad_output.len();
        let mut grad_input = vec![Complex::<T>::zero(); n];

        // F^H · v = conj(FFT(conj(v))) which equals N * IFFT(v)
        // Using IFFT directly is simpler
        self.inv_plan.execute(grad_output, &mut grad_input);

        // Our IFFT is unnormalized, so result = F^H · grad_output
        // No additional scaling needed for adjoint property
        grad_input
    }

    /// Compute gradient of a scalar loss with respect to IFFT input.
    ///
    /// For normalized IFFT: y = (1/N)·Fᴴ·x
    /// The adjoint of (1/N)·Fᴴ is (1/N)·F
    /// Gradient: ∂L/∂x = (1/N)·F · ∂L/∂y = (1/N)·FFT(∂L/∂y)
    pub fn backward_inverse(&self, grad_output: &[Complex<T>]) -> Vec<Complex<T>> {
        let n = grad_output.len();
        let mut grad_input = vec![Complex::<T>::zero(); n];

        self.forward(grad_output, &mut grad_input);

        // Scale by 1/N for the normalized IFFT's adjoint
        let scale = T::ONE / T::from_usize(n);
        for c in &mut grad_input {
            *c = Complex::new(c.re * scale, c.im * scale);
        }

        grad_input
    }

    /// Get the transform size.
    pub fn size(&self) -> usize {
        self.size
    }
}

// Convenience functions

/// Compute FFT with forward-mode automatic differentiation.
///
/// Returns (output, output_tangent) where output_tangent is the
/// directional derivative of FFT in the direction of input tangents.
pub fn fft_dual<T: Float>(input: &[DualComplex<T>]) -> Option<(Vec<Complex<T>>, Vec<Complex<T>>)> {
    let plan = DiffFftPlan::new(input.len())?;
    Some(plan.forward_dual(input))
}

/// Compute gradient of a scalar loss with respect to FFT input.
///
/// Given ∂L/∂(FFT(x)), computes ∂L/∂x.
pub fn grad_fft<T: Float>(grad_output: &[Complex<T>]) -> Option<Vec<Complex<T>>> {
    let plan = DiffFftPlan::new(grad_output.len())?;
    Some(plan.backward(grad_output))
}

/// Compute gradient of a scalar loss with respect to IFFT input.
///
/// Given ∂L/∂(IFFT(x)), computes ∂L/∂x.
pub fn grad_ifft<T: Float>(grad_output: &[Complex<T>]) -> Option<Vec<Complex<T>>> {
    let plan = DiffFftPlan::new(grad_output.len())?;
    Some(plan.backward_inverse(grad_output))
}

/// Vector-Jacobian product for FFT (used in reverse mode AD).
///
/// Computes vᵀ·J where J is the Jacobian of FFT and v is a vector.
pub fn vjp_fft<T: Float>(v: &[Complex<T>]) -> Option<Vec<Complex<T>>> {
    grad_fft(v)
}

/// Jacobian-vector product for FFT (used in forward mode AD).
///
/// Computes J·v where J is the Jacobian of FFT and v is a vector.
/// This is simply FFT(v) since FFT is linear.
pub fn jvp_fft<T: Float>(v: &[Complex<T>]) -> Option<Vec<Complex<T>>> {
    use crate::api::fft;
    Some(fft(v))
}

/// Compute the full Jacobian matrix of the FFT.
///
/// For an N-point FFT, returns an NxN complex matrix where
/// `J[k,n] = exp(-2πi·k·n/N) / √N` (normalized DFT matrix).
///
/// This is memory-intensive and should only be used for small N.
pub fn fft_jacobian<T: Float>(n: usize) -> Vec<Vec<Complex<T>>> {
    let two_pi = T::from_f64(2.0 * core::f64::consts::PI);
    let n_t = T::from_usize(n);

    (0..n)
        .map(|k| {
            (0..n)
                .map(|j| {
                    let angle = -two_pi * T::from_usize(k) * T::from_usize(j) / n_t;
                    Complex::new(Float::cos(angle), Float::sin(angle))
                })
                .collect()
        })
        .collect()
}

/// Differentiable real FFT functions.
pub mod real {
    use super::*;

    /// Compute gradient of a scalar loss with respect to real FFT input.
    ///
    /// Real FFT: R^N → C^(N/2+1)
    /// The gradient computation accounts for the conjugate symmetry.
    pub fn grad_rfft<T: Float>(grad_output: &[Complex<T>], n: usize) -> Option<Vec<T>> {
        let fft_plan = Plan::<T>::dft_1d(n, Direction::Backward, Flags::ESTIMATE)?;

        // Reconstruct full spectrum from half (conjugate symmetry)
        let mut full_grad = vec![Complex::<T>::zero(); n];
        for (i, &g) in grad_output.iter().enumerate() {
            full_grad[i] = g;
        }

        // Fill in conjugate symmetric part
        for i in 1..n / 2 {
            full_grad[n - i] = grad_output[i].conj();
        }

        // IFFT
        let mut result = vec![Complex::<T>::zero(); n];
        fft_plan.execute(&full_grad, &mut result);

        // Scale and extract real part
        let scale = T::ONE / T::from_usize(n);
        Some(result.iter().map(|c| c.re * scale).collect())
    }

    /// Compute gradient of a scalar loss with respect to inverse real FFT input.
    pub fn grad_irfft<T: Float>(grad_output: &[T], n_output: usize) -> Option<Vec<Complex<T>>> {
        let fft_plan = Plan::<T>::dft_1d(n_output, Direction::Forward, Flags::ESTIMATE)?;

        // Convert real gradient to complex
        let complex_grad: Vec<Complex<T>> = grad_output
            .iter()
            .map(|&r| Complex::new(r, T::ZERO))
            .collect();

        // FFT
        let mut result = vec![Complex::<T>::zero(); n_output];
        fft_plan.execute(&complex_grad, &mut result);

        // Take first N/2+1 elements (due to conjugate symmetry of real input)
        let n_freq = n_output / 2 + 1;
        let scale = T::ONE / T::from_usize(n_output);
        Some(
            result
                .into_iter()
                .take(n_freq)
                .map(|c| Complex::new(c.re * scale, c.im * scale))
                .collect(),
        )
    }
}

/// Differentiable 2D FFT functions.
pub mod fft2d {
    use super::*;

    /// Compute gradient of 2D FFT.
    ///
    /// The gradient of a 2D FFT is computed by applying 1D FFT gradients
    /// along each axis.
    pub fn grad_fft2d<T: Float>(
        grad_output: &[Complex<T>],
        rows: usize,
        cols: usize,
    ) -> Option<Vec<Complex<T>>> {
        if grad_output.len() != rows * cols {
            return None;
        }

        let row_plan = DiffFftPlan::new(cols)?;
        let col_plan = DiffFftPlan::new(rows)?;

        // Apply gradient along columns first
        let mut temp = vec![Complex::<T>::zero(); rows * cols];
        for c in 0..cols {
            let col: Vec<Complex<T>> = (0..rows).map(|r| grad_output[r * cols + c]).collect();
            let grad_col = col_plan.backward(&col);
            for (r, &g) in grad_col.iter().enumerate() {
                temp[r * cols + c] = g;
            }
        }

        // Apply gradient along rows
        let mut result = vec![Complex::<T>::zero(); rows * cols];
        for r in 0..rows {
            let row: Vec<Complex<T>> = (0..cols).map(|c| temp[r * cols + c]).collect();
            let grad_row = row_plan.backward(&row);
            for (c, &g) in grad_row.iter().enumerate() {
                result[r * cols + c] = g;
            }
        }

        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_dual_arithmetic() {
        let a = Dual::new(2.0, 1.0);
        let b = Dual::new(3.0, 0.0);

        let sum = a + b;
        assert!(approx_eq(sum.value, 5.0, 1e-10));
        assert!(approx_eq(sum.deriv, 1.0, 1e-10));

        let prod = a * b;
        assert!(approx_eq(prod.value, 6.0, 1e-10));
        assert!(approx_eq(prod.deriv, 3.0, 1e-10)); // d(2x*3)/dx = 3
    }

    #[test]
    fn test_dual_complex_arithmetic() {
        let a = DualComplex::new(1.0, 0.0, 1.0, 0.0);
        let b = DualComplex::new(0.0, 1.0, 0.0, 0.0);

        let sum = a + b;
        assert!(approx_eq(sum.value.re, 1.0, 1e-10));
        assert!(approx_eq(sum.value.im, 1.0, 1e-10));
        assert!(approx_eq(sum.deriv.re, 1.0, 1e-10));
        assert!(approx_eq(sum.deriv.im, 0.0, 1e-10));
    }

    #[test]
    fn test_fft_forward_mode() {
        // For a constant input, the derivative should be FFT of the tangent
        let n = 8;
        let input: Vec<DualComplex<f64>> = (0..n)
            .map(|k| DualComplex::new(1.0, 0.0, if k == 0 { 1.0 } else { 0.0 }, 0.0))
            .collect();

        let result = fft_dual(&input);
        assert!(result.is_some());

        let (y, dy) = result.expect("fft_dual failed");

        // FFT of constant 1 should have first element = N, rest = 0
        assert!(approx_eq(y[0].re, n as f64, 1e-10));
        for i in 1..n {
            assert!(approx_eq(y[i].re, 0.0, 1e-10));
            assert!(approx_eq(y[i].im, 0.0, 1e-10));
        }

        // FFT of delta at 0 should be constant 1
        for i in 0..n {
            assert!(approx_eq(dy[i].re, 1.0, 1e-10));
            assert!(approx_eq(dy[i].im, 0.0, 1e-10));
        }
    }

    #[test]
    fn test_fft_backward_mode() {
        // Test that backward is adjoint of forward
        let n = 8;

        let x: Vec<Complex<f64>> = (0..n)
            .map(|k| Complex::new((k as f64).cos(), (k as f64).sin()))
            .collect();

        let v: Vec<Complex<f64>> = (0..n)
            .map(|k| Complex::new(((k + 1) as f64).sin(), ((k + 1) as f64).cos()))
            .collect();

        let plan = DiffFftPlan::new(n).expect("Plan creation failed");

        // Compute y = FFT(x)
        let mut y = vec![Complex::<f64>::zero(); n];
        plan.forward(&x, &mut y);

        // Compute gradient: ∂L/∂x where L = <v, y>
        let grad_x = plan.backward(&v);

        // The adjoint property: <v, FFT(x)> = <FFT*(v), x>
        // where FFT* is the adjoint (conjugate transpose)
        // For verification: compute <grad_x, x> and <v, y>

        let inner_vy: Complex<f64> = v
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| a.conj() * b)
            .fold(Complex::zero(), |acc, x| acc + x);

        let inner_gx: Complex<f64> = grad_x
            .iter()
            .zip(x.iter())
            .map(|(&a, &b)| a.conj() * b)
            .fold(Complex::zero(), |acc, x| acc + x);

        // These should be equal (up to numerical precision)
        assert!(
            approx_eq(inner_vy.re, inner_gx.re, 1e-8),
            "Adjoint property failed: {} != {}",
            inner_vy.re,
            inner_gx.re
        );
    }

    #[test]
    fn test_fft_jacobian_small() {
        let n = 4;
        let jac = fft_jacobian::<f64>(n);

        // Jacobian should be NxN
        assert_eq!(jac.len(), n);
        for row in &jac {
            assert_eq!(row.len(), n);
        }

        // J[0,j] should all be 1 (DC component sums all inputs)
        for j in 0..n {
            assert!(approx_eq(jac[0][j].re, 1.0, 1e-10));
            assert!(approx_eq(jac[0][j].im, 0.0, 1e-10));
        }
    }

    #[test]
    fn test_vjp_jvp_consistency() {
        // For linear functions, VJP and JVP should satisfy:
        // <v, JVP(u)> = <VJP(v), u>
        let n = 8;

        let u: Vec<Complex<f64>> = (0..n)
            .map(|k| Complex::new(f64::from(k) * 0.1, 0.0))
            .collect();

        let v: Vec<Complex<f64>> = (0..n)
            .map(|k| Complex::new(0.0, f64::from(k) * 0.1))
            .collect();

        let jvp_u = jvp_fft(&u).expect("JVP failed");
        let vjp_v = vjp_fft(&v).expect("VJP failed");

        // <v, JVP(u)>
        let inner1: Complex<f64> = v
            .iter()
            .zip(jvp_u.iter())
            .map(|(&a, &b)| a.conj() * b)
            .fold(Complex::zero(), |acc, x| acc + x);

        // <VJP(v), u>
        let inner2: Complex<f64> = vjp_v
            .iter()
            .zip(u.iter())
            .map(|(&a, &b)| a.conj() * b)
            .fold(Complex::zero(), |acc, x| acc + x);

        assert!(
            approx_eq(inner1.re, inner2.re, 1e-8),
            "VJP/JVP consistency failed"
        );
    }
}
