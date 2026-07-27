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
//! # Gradient conventions
//!
//! All `grad_*` / `backward*` functions compute the reverse-mode adjoint
//! (vector-Jacobian product) of the corresponding transform, for a **real
//! scalar** loss `L`. Cotangents use the PyTorch/autograd layout: the cotangent
//! of a complex value `z = a + i·b` is `∂L/∂a + i·∂L/∂b`, and the cotangent of a
//! real value `r` is `∂L/∂r`.
//!
//! Transform conventions (matching the crate's public API):
//! - Forward DFT is unnormalised: `y[k] = Σ_n x[n]·exp(-2πi·kn/N)`; its adjoint
//!   is the unnormalised inverse DFT (`Direction::Backward`, no `1/N`).
//! - Inverse DFT is normalised: `y[n] = (1/N)·Σ_k x[k]·exp(+2πi·kn/N)`; its
//!   adjoint is `(1/N)·`forward DFT.
//! - Real transforms (`rfft`/`irfft`) additionally account for the discarded
//!   conjugate-symmetric half: see [`real::grad_rfft`] and [`real::grad_irfft`].
//!
//! Every public gradient function in this module is validated against central
//! finite differences (see the module tests).
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
//! ```
//! use oxifft::autodiff::{DualComplex, fft_dual, grad_fft};
//! use oxifft::Complex;
//!
//! // Forward mode: compute FFT and its directional derivative.
//! let x = vec![DualComplex::new(1.0, 0.0, 1.0, 0.0); 8];
//! let (y, dy) = fft_dual(&x).expect("size 8 is supported");
//! # let _ = (y, dy);
//!
//! // Backward mode: compute gradient of loss w.r.t. FFT input.
//! let grad_output = vec![Complex::new(1.0, 0.0); 8];
//! let grad_input = grad_fft(&grad_output).expect("size 8 is supported");
//! # let _ = grad_input;
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
    ///
    /// # Examples
    ///
    /// ```
    /// use oxifft::autodiff::DiffFftPlan;
    /// use oxifft::Complex;
    ///
    /// let plan = DiffFftPlan::<f64>::new(8).expect("plan creation failed");
    /// let input: Vec<Complex<f64>> = (0..8).map(|i| Complex::new(i as f64, 0.0)).collect();
    /// let mut output = vec![Complex::<f64>::zero(); 8];
    /// plan.forward(&input, &mut output);
    /// // DC bin = sum of 0+1+...+7 = 28
    /// assert!((output[0].re - 28.0_f64).abs() < 1e-9);
    /// ```
    pub fn new(size: usize) -> Option<Self> {
        // Default to `ESTIMATE`: `DiffFftPlan` is the API most likely to be
        // (re)constructed inside a training loop, once per step. `MEASURE`
        // planning costs ~100µs-13ms per construction with no runtime benefit
        // for the short-lived plans typical of autodiff use. Callers that keep
        // a plan alive across many transforms and want the tuned schedule can
        // opt in via [`DiffFftPlan::with_flags`].
        Self::with_flags(size, Flags::ESTIMATE)
    }

    /// Create a new differentiable FFT plan with explicit planner flags.
    ///
    /// Use this when the plan is long-lived and the one-time `MEASURE`/`PATIENT`
    /// planning cost is amortised over many transforms. For short-lived plans
    /// (the common autodiff case) prefer [`DiffFftPlan::new`], which uses
    /// [`Flags::ESTIMATE`].
    ///
    /// # Examples
    ///
    /// ```
    /// use oxifft::autodiff::DiffFftPlan;
    /// use oxifft::Flags;
    ///
    /// let plan = DiffFftPlan::<f64>::with_flags(8, Flags::ESTIMATE)
    ///     .expect("plan creation failed");
    /// assert_eq!(plan.size(), 8);
    /// ```
    pub fn with_flags(size: usize, flags: Flags) -> Option<Self> {
        let fwd_plan = Plan::dft_1d(size, Direction::Forward, flags)?;
        let inv_plan = Plan::dft_1d(size, Direction::Backward, flags)?;

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
///
/// # Conventions
///
/// These functions are the reverse-mode adjoints (vector-Jacobian products) of
/// the crate's [`rfft`](crate::rfft) / [`irfft`](crate::irfft) transforms, for a
/// **real scalar** loss `L`.
///
/// Cotangents use the PyTorch/autograd layout: the cotangent of a complex value
/// `z = a + i·b` is `∂L/∂a + i·∂L/∂b`, and the cotangent of a real value `r` is
/// `∂L/∂r`.
///
/// * `rfft`: `R^N → C^(N/2+1)`, `Y[k] = Σ_n x[n]·exp(-2πi·kn/N)` for `k = 0..=N/2`.
/// * `irfft`: `C^(N/2+1) → R^N`, the normalised (÷N) Hermitian-completion inverse.
///   Because the inverse assumes conjugate symmetry, the imaginary parts of the
///   DC bin (and the Nyquist bin when `N` is even) are ignored.
pub mod real {
    use super::*;

    /// Gradient of a real scalar loss with respect to the input of a real FFT.
    ///
    /// Given the cotangent `grad_output` of the `N/2+1` half-spectrum bins of
    /// `Y = rfft(x)` (layout: `∂L/∂Re + i·∂L/∂Im` per bin), returns the real
    /// cotangent `∂L/∂x` (length `n`).
    ///
    /// Each returned rfft bin is an **independent** output of the forward
    /// transform, so its contribution is counted exactly once — there is no
    /// conjugate-symmetric completion and no `1/N` factor here. Concretely,
    /// `x̄[m] = Re( Σ_{k=0}^{N/2} grad_output[k]·exp(+2πi·km/N) )`, which is the
    /// real part of the *unnormalised* inverse DFT of the zero-padded cotangent
    /// spectrum.
    ///
    /// Returns `None` if `grad_output.len() != n/2 + 1` or the plan cannot be
    /// built.
    pub fn grad_rfft<T: Float>(grad_output: &[Complex<T>], n: usize) -> Option<Vec<T>> {
        if grad_output.len() != n / 2 + 1 {
            return None;
        }
        let ifft_plan = Plan::<T>::dft_1d(n, Direction::Backward, Flags::ESTIMATE)?;

        // Place the cotangent bins into a full-length spectrum, leaving the
        // redundant (conjugate) half as zeros. The forward rfft returns only the
        // first N/2+1 bins, so their adjoint must NOT re-add a mirrored copy.
        let mut spectrum = vec![Complex::<T>::zero(); n];
        for (dst, &g) in spectrum.iter_mut().zip(grad_output.iter()) {
            *dst = g;
        }

        // Unnormalised inverse DFT (Direction::Backward, no 1/N scaling).
        let mut result = vec![Complex::<T>::zero(); n];
        ifft_plan.execute(&spectrum, &mut result);

        // x is real, so its cotangent is the real part.
        Some(result.iter().map(|c| c.re).collect())
    }

    /// Gradient of a real scalar loss with respect to the input of an inverse
    /// real FFT.
    ///
    /// Given the real cotangent `grad_output = ∂L/∂x` of the length-`n_output`
    /// output `x = irfft(X)`, returns the complex cotangent `∂L/∂X` of the
    /// `n_output/2 + 1` half-spectrum input bins (layout `∂L/∂Re + i·∂L/∂Im`).
    ///
    /// Because `irfft` conjugate-mirrors the interior bins (each interior input
    /// bin feeds both itself and its mirror), the adjoint of an interior bin
    /// carries a factor of two. The DC bin — and the Nyquist bin when
    /// `n_output` is even — are self-conjugate single real degrees of freedom:
    /// they carry no factor of two and their imaginary cotangent is exactly zero
    /// (the forward transform ignores it). Everything is scaled by `1/n_output`
    /// to match the normalised inverse.
    ///
    /// Returns `None` if `grad_output.len() != n_output` or the plan cannot be
    /// built.
    pub fn grad_irfft<T: Float>(grad_output: &[T], n_output: usize) -> Option<Vec<Complex<T>>> {
        if grad_output.len() != n_output {
            return None;
        }
        let fft_plan = Plan::<T>::dft_1d(n_output, Direction::Forward, Flags::ESTIMATE)?;

        // Unnormalised forward DFT of the real output-cotangent.
        let complex_grad: Vec<Complex<T>> = grad_output
            .iter()
            .map(|&r| Complex::new(r, T::ZERO))
            .collect();
        let mut spectrum = vec![Complex::<T>::zero(); n_output];
        fft_plan.execute(&complex_grad, &mut spectrum);

        let n_freq = n_output / 2 + 1;
        let scale = T::ONE / T::from_usize(n_output);
        let two = T::from_usize(2);

        let grad_input = spectrum
            .iter()
            .take(n_freq)
            .enumerate()
            .map(|(k, c)| {
                // Self-conjugate bins (DC, and Nyquist for even n_output) map to
                // a single real degree of freedom in the forward transform.
                if k == 0 || 2 * k == n_output {
                    Complex::new(c.re * scale, T::ZERO)
                } else {
                    // Interior bins are conjugate-mirror-doubled by irfft.
                    Complex::new(c.re * scale * two, c.im * scale * two)
                }
            })
            .collect();

        Some(grad_input)
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

    // ---------------------------------------------------------------------
    // Finite-difference validation of every public gradient function.
    //
    // Each transform is linear and every scalar loss below is linear in the
    // transform output, so the loss is exactly linear in the (real) input
    // components. A central finite difference of an exactly-linear function is
    // therefore analytically exact; only floating-point round-off separates it
    // from the analytic adjoint. This makes the checks a sharp regression gate:
    // the historic missing/extra factor-of-two bugs are 100% errors and cannot
    // hide under any reasonable tolerance.
    // ---------------------------------------------------------------------

    /// Small deterministic xorshift PRNG (keeps tests dependency-free).
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self {
                state: seed | 1, // avoid the all-zero fixed point
            }
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }

        /// Uniform sample in `[-1, 1)`.
        fn sample<T: Float>(&mut self) -> T {
            let u = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
            T::from_f64(u * 2.0 - 1.0)
        }
    }

    fn assert_close<T: Float>(got: T, want: T, tol: T, msg: &str) {
        let diff = Float::abs(got - want);
        let bound = tol * (T::ONE + Float::abs(want));
        assert!(
            diff <= bound,
            "{msg}: got {got:?} want {want:?} (diff {diff:?} > bound {bound:?})"
        );
    }

    // ---- Reference (naive O(n^2)) transforms, independent of the library ----

    fn ref_fft<T: Float>(x: &[Complex<T>]) -> Vec<Complex<T>> {
        let n = x.len();
        let n_t = T::from_usize(n);
        (0..n)
            .map(|k| {
                let mut acc = Complex::<T>::zero();
                for (m, &xm) in x.iter().enumerate() {
                    let ang = -T::TWO_PI * T::from_usize(k) * T::from_usize(m) / n_t;
                    let (s, c) = Float::sin_cos(ang);
                    acc = acc + xm * Complex::new(c, s);
                }
                acc
            })
            .collect()
    }

    fn ref_ifft<T: Float>(x: &[Complex<T>]) -> Vec<Complex<T>> {
        let n = x.len();
        let n_t = T::from_usize(n);
        let scale = T::ONE / n_t;
        (0..n)
            .map(|m| {
                let mut acc = Complex::<T>::zero();
                for (k, &xk) in x.iter().enumerate() {
                    let ang = T::TWO_PI * T::from_usize(k) * T::from_usize(m) / n_t;
                    let (s, c) = Float::sin_cos(ang);
                    acc = acc + xk * Complex::new(c, s);
                }
                Complex::new(acc.re * scale, acc.im * scale)
            })
            .collect()
    }

    fn ref_rfft<T: Float>(x: &[T]) -> Vec<Complex<T>> {
        let n = x.len();
        let n_t = T::from_usize(n);
        (0..=n / 2)
            .map(|k| {
                let mut acc = Complex::<T>::zero();
                for (m, &xm) in x.iter().enumerate() {
                    let ang = -T::TWO_PI * T::from_usize(k) * T::from_usize(m) / n_t;
                    let (s, c) = Float::sin_cos(ang);
                    acc = acc + Complex::new(xm * c, xm * s);
                }
                acc
            })
            .collect()
    }

    fn ref_irfft<T: Float>(x_half: &[Complex<T>], n: usize) -> Vec<T> {
        let n_freq = n / 2 + 1;
        let mut full = vec![Complex::<T>::zero(); n];
        for (dst, &src) in full.iter_mut().zip(x_half.iter()) {
            *dst = src;
        }
        // irfft treats the self-conjugate bins as purely real.
        full[0] = Complex::new(x_half[0].re, T::ZERO);
        if n.is_multiple_of(2) {
            full[n / 2] = Complex::new(x_half[n / 2].re, T::ZERO);
        }
        // Conjugate-mirror the interior bins.
        for k in 1..n_freq {
            if 2 * k != n {
                full[n - k] = full[k].conj();
            }
        }
        let n_t = T::from_usize(n);
        let scale = T::ONE / n_t;
        (0..n)
            .map(|m| {
                let mut acc = Complex::<T>::zero();
                for (k, &fk) in full.iter().enumerate() {
                    let ang = T::TWO_PI * T::from_usize(k) * T::from_usize(m) / n_t;
                    let (s, c) = Float::sin_cos(ang);
                    acc = acc + fk * Complex::new(c, s);
                }
                acc.re * scale
            })
            .collect()
    }

    fn ref_fft2d<T: Float>(x: &[Complex<T>], rows: usize, cols: usize) -> Vec<Complex<T>> {
        let rows_t = T::from_usize(rows);
        let cols_t = T::from_usize(cols);
        let mut out = vec![Complex::<T>::zero(); rows * cols];
        for p in 0..rows {
            for q in 0..cols {
                let mut acc = Complex::<T>::zero();
                for a in 0..rows {
                    for b in 0..cols {
                        let ang = -T::TWO_PI
                            * (T::from_usize(p) * T::from_usize(a) / rows_t
                                + T::from_usize(q) * T::from_usize(b) / cols_t);
                        let (s, c) = Float::sin_cos(ang);
                        acc = acc + x[a * cols + b] * Complex::new(c, s);
                    }
                }
                out[p * cols + q] = acc;
            }
        }
        out
    }

    // ---- Generic finite-difference drivers ----

    /// Validate a complex-input / complex-output VJP against finite differences.
    fn check_complex_vjp<T, F, G>(sizes: &[usize], tol: T, h: T, forward: F, grad: G)
    where
        T: Float,
        F: Fn(&[Complex<T>]) -> Vec<Complex<T>>,
        G: Fn(&[Complex<T>]) -> Vec<Complex<T>>,
    {
        for &n in sizes {
            let mut rng = Lcg::new(0x51ED_2701 ^ (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let x: Vec<Complex<T>> = (0..n)
                .map(|_| Complex::new(rng.sample::<T>(), rng.sample::<T>()))
                .collect();
            let m = forward(&x).len();
            let cr: Vec<T> = (0..m).map(|_| rng.sample::<T>()).collect();
            let ci: Vec<T> = (0..m).map(|_| rng.sample::<T>()).collect();
            let cot: Vec<Complex<T>> = (0..m).map(|k| Complex::new(cr[k], ci[k])).collect();

            let analytic = grad(&cot);
            assert_eq!(analytic.len(), n, "n={n}: gradient length mismatch");

            let loss = |xx: &[Complex<T>]| -> T {
                let yy = forward(xx);
                let mut s = T::ZERO;
                for (k, y) in yy.iter().enumerate() {
                    s = s + cr[k] * y.re + ci[k] * y.im;
                }
                s
            };
            let two_h = h + h;
            for idx in 0..n {
                let mut xp = x.clone();
                xp[idx].re = xp[idx].re + h;
                let mut xm = x.clone();
                xm[idx].re = xm[idx].re - h;
                let d_re = (loss(&xp) - loss(&xm)) / two_h;

                let mut xp = x.clone();
                xp[idx].im = xp[idx].im + h;
                let mut xm = x.clone();
                xm[idx].im = xm[idx].im - h;
                let d_im = (loss(&xp) - loss(&xm)) / two_h;

                assert_close(analytic[idx].re, d_re, tol, &format!("n={n} re[{idx}]"));
                assert_close(analytic[idx].im, d_im, tol, &format!("n={n} im[{idx}]"));
            }
        }
    }

    /// Validate `grad_rfft` (real input, complex half-spectrum output).
    fn check_rfft_vjp<T: Float>(sizes: &[usize], tol: T, h: T) {
        for &n in sizes {
            let mut rng = Lcg::new(0x2718_2818 ^ (n as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
            let x: Vec<T> = (0..n).map(|_| rng.sample::<T>()).collect();
            let m = n / 2 + 1;
            let cr: Vec<T> = (0..m).map(|_| rng.sample::<T>()).collect();
            let ci: Vec<T> = (0..m).map(|_| rng.sample::<T>()).collect();
            let cot: Vec<Complex<T>> = (0..m).map(|k| Complex::new(cr[k], ci[k])).collect();

            let analytic = real::grad_rfft(&cot, n).expect("grad_rfft");
            assert_eq!(analytic.len(), n, "n={n}: grad_rfft length");

            let loss = |xx: &[T]| -> T {
                let yy = ref_rfft(xx);
                let mut s = T::ZERO;
                for (k, y) in yy.iter().enumerate() {
                    s = s + cr[k] * y.re + ci[k] * y.im;
                }
                s
            };
            let two_h = h + h;
            for idx in 0..n {
                let mut xp = x.clone();
                xp[idx] = xp[idx] + h;
                let mut xm = x.clone();
                xm[idx] = xm[idx] - h;
                let slope = (loss(&xp) - loss(&xm)) / two_h;
                assert_close(analytic[idx], slope, tol, &format!("n={n} rfft[{idx}]"));
            }
        }
    }

    /// Validate `grad_irfft` (complex half-spectrum input, real output).
    fn check_irfft_vjp<T: Float>(sizes: &[usize], tol: T, h: T) {
        for &n in sizes {
            let mut rng = Lcg::new(0x1414_2135 ^ (n as u64).wrapping_mul(0xA24B_AED4_963E_E407));
            let n_freq = n / 2 + 1;
            let x: Vec<Complex<T>> = (0..n_freq)
                .map(|_| Complex::new(rng.sample::<T>(), rng.sample::<T>()))
                .collect();
            let w: Vec<T> = (0..n).map(|_| rng.sample::<T>()).collect();

            let analytic = real::grad_irfft(&w, n).expect("grad_irfft");
            assert_eq!(analytic.len(), n_freq, "n={n}: grad_irfft length");

            let loss = |xx: &[Complex<T>]| -> T {
                let yy = ref_irfft(xx, n);
                let mut s = T::ZERO;
                for (m, y) in yy.iter().enumerate() {
                    s = s + w[m] * *y;
                }
                s
            };
            let two_h = h + h;
            for k in 0..n_freq {
                let mut xp = x.clone();
                xp[k].re = xp[k].re + h;
                let mut xm = x.clone();
                xm[k].re = xm[k].re - h;
                let d_re = (loss(&xp) - loss(&xm)) / two_h;

                let mut xp = x.clone();
                xp[k].im = xp[k].im + h;
                let mut xm = x.clone();
                xm[k].im = xm[k].im - h;
                let d_im = (loss(&xp) - loss(&xm)) / two_h;

                assert_close(analytic[k].re, d_re, tol, &format!("n={n} irfft re[{k}]"));
                assert_close(analytic[k].im, d_im, tol, &format!("n={n} irfft im[{k}]"));
            }
        }
    }

    /// Validate forward-mode (JVP) against finite differences of the transform.
    fn check_forward_mode<T: Float>(sizes: &[usize], tol: T, h: T) {
        for &n in sizes {
            let mut rng = Lcg::new(0x6022_1407 ^ (n as u64).wrapping_mul(0xB492_B66F_BE98_F273));
            let x: Vec<Complex<T>> = (0..n)
                .map(|_| Complex::new(rng.sample::<T>(), rng.sample::<T>()))
                .collect();
            let t: Vec<Complex<T>> = (0..n)
                .map(|_| Complex::new(rng.sample::<T>(), rng.sample::<T>()))
                .collect();

            let dual: Vec<DualComplex<T>> = (0..n)
                .map(|i| DualComplex::from_complex(x[i], t[i]))
                .collect();
            let (y, dy) = fft_dual(&dual).expect("fft_dual");

            // Primal must match the reference forward transform.
            let y_ref = ref_fft(&x);
            for k in 0..n {
                assert_close(y[k].re, y_ref[k].re, tol, &format!("n={n} y.re[{k}]"));
                assert_close(y[k].im, y_ref[k].im, tol, &format!("n={n} y.im[{k}]"));
            }

            // Directional derivative via central difference along tangent t.
            let two_h = h + h;
            let xp: Vec<Complex<T>> = (0..n).map(|i| x[i] + t[i] * h).collect();
            let xm: Vec<Complex<T>> = (0..n).map(|i| x[i] - t[i] * h).collect();
            let yp = ref_fft(&xp);
            let ym = ref_fft(&xm);
            for k in 0..n {
                let d = (yp[k] - ym[k]) / two_h;
                assert_close(dy[k].re, d.re, tol, &format!("n={n} dy.re[{k}]"));
                assert_close(dy[k].im, d.im, tol, &format!("n={n} dy.im[{k}]"));
            }
        }
    }

    const FD_SIZES: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8, 12, 16];
    const FD_SIZES_2D: &[(usize, usize)] =
        &[(1, 1), (2, 2), (2, 3), (3, 2), (3, 3), (4, 4), (2, 5)];

    #[test]
    fn fd_grad_fft_f64() {
        check_complex_vjp::<f64, _, _>(FD_SIZES, 1e-6, 0.125, ref_fft, |v| {
            grad_fft(v).expect("grad_fft")
        });
    }

    #[test]
    fn fd_grad_ifft_f64() {
        check_complex_vjp::<f64, _, _>(FD_SIZES, 1e-6, 0.125, ref_ifft, |v| {
            grad_ifft(v).expect("grad_ifft")
        });
    }

    #[test]
    fn fd_grad_rfft_f64() {
        check_rfft_vjp::<f64>(FD_SIZES, 1e-6, 0.125);
    }

    #[test]
    fn fd_grad_irfft_f64() {
        check_irfft_vjp::<f64>(FD_SIZES, 1e-6, 0.125);
    }

    #[test]
    fn fd_forward_mode_f64() {
        check_forward_mode::<f64>(FD_SIZES, 1e-6, 0.125);
    }

    #[test]
    fn fd_grad_fft2d_f64() {
        for &(rows, cols) in FD_SIZES_2D {
            check_complex_vjp::<f64, _, _>(
                &[rows * cols],
                1e-6,
                0.125,
                move |x| ref_fft2d(x, rows, cols),
                move |v| fft2d::grad_fft2d(v, rows, cols).expect("grad_fft2d"),
            );
        }
    }

    #[test]
    fn fd_grad_fft_f32() {
        check_complex_vjp::<f32, _, _>(&[1, 2, 3, 4, 5, 8], 2e-2, 0.125, ref_fft, |v| {
            grad_fft(v).expect("grad_fft")
        });
    }

    #[test]
    fn fd_grad_rfft_f32() {
        check_rfft_vjp::<f32>(&[1, 2, 3, 4, 5, 8], 2e-2, 0.125);
    }

    #[test]
    fn fd_grad_irfft_f32() {
        check_irfft_vjp::<f32>(&[1, 2, 3, 4, 5, 8], 2e-2, 0.125);
    }

    #[test]
    fn fd_forward_mode_f32() {
        check_forward_mode::<f32>(&[1, 2, 3, 4, 5, 8], 2e-2, 0.125);
    }
}
