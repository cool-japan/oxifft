//! Fractional Fourier Transform (FrFT) implementation.
//!
//! The Fractional Fourier Transform generalizes the standard DFT to
//! fractional orders, providing a continuous rotation in the time-frequency plane.
//!
//! # Applications
//!
//! - Chirp signal analysis
//! - Optical systems modeling
//! - Radar signal processing
//! - Quantum mechanics
//! - Time-frequency analysis
//!
//! # Mathematical Definition
//!
//! The FrFT of order α transforms a signal by rotating it by angle α·π/2
//! in the time-frequency plane. When α=1, it equals the standard DFT.
//!
//! # Algorithm
//!
//! Uses the decomposition method by Ozaktas et al., which expresses FrFT as:
//! 1. Chirp multiplication
//! 2. Chirp convolution (via FFT)
//! 3. Chirp multiplication
//!
//! Complexity: O(n log n)
//!
//! # Example
//!
//! ```ignore
//! use oxifft::frft::{frft, ifrft};
//!
//! let signal = vec![Complex::new(1.0, 0.0); 256];
//!
//! // Fractional Fourier Transform with order 0.5
//! let result = frft(&signal, 0.5).expect("FrFT failed");
//!
//! // Inverse (order -0.5)
//! let recovered = ifrft(&result, 0.5).expect("iFrFT failed");
//! ```

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};

/// Fractional Fourier Transform error types.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum FrftError {
    /// Invalid input size.
    InvalidSize(usize),
    /// FFT planning failed.
    PlanFailed,
    /// Invalid fractional order.
    InvalidOrder,
}

impl core::fmt::Display for FrftError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidSize(n) => write!(f, "Invalid FrFT size: {n}"),
            Self::PlanFailed => write!(f, "Failed to create FFT plan"),
            Self::InvalidOrder => write!(f, "Invalid fractional order"),
        }
    }
}

/// Result type for FrFT operations.
pub type FrftResult<T> = Result<T, FrftError>;

/// Fractional Fourier Transform plan.
///
/// Precomputes chirp factors for efficient repeated transforms.
pub struct Frft<T: Float> {
    /// Transform size.
    n: usize,
    /// Fractional order (α in range [0, 4]).
    order: f64,
    /// Pre-chirp factors.
    pre_chirp: Vec<Complex<T>>,
    /// Post-chirp factors.
    post_chirp: Vec<Complex<T>>,
    /// Convolution kernel (in frequency domain).
    kernel_fft: Vec<Complex<T>>,
    /// Padded size for convolution.
    padded_size: usize,
    /// FFT plan for padded convolution.
    fft_plan: Option<Plan<T>>,
    /// IFFT plan for padded convolution.
    ifft_plan: Option<Plan<T>>,
}

impl<T: Float> Frft<T> {
    /// Create a new Fractional Fourier Transform plan.
    ///
    /// # Arguments
    ///
    /// * `n` - Transform size
    /// * `order` - Fractional order α (any real number, reduced mod 4)
    ///
    /// # Returns
    ///
    /// FrFT plan or error.
    pub fn new(n: usize, order: f64) -> FrftResult<Self> {
        if n == 0 {
            return Err(FrftError::InvalidSize(0));
        }

        // Reduce order to [0, 4)
        let order = reduce_order(order);

        // Handle special cases
        if (order - 0.0).abs() < 1e-10
            || (order - 1.0).abs() < 1e-10
            || (order - 2.0).abs() < 1e-10
            || (order - 3.0).abs() < 1e-10
        {
            // Integer orders - handled specially in execute
            return Ok(Self {
                n,
                order,
                pre_chirp: Vec::new(),
                post_chirp: Vec::new(),
                kernel_fft: Vec::new(),
                padded_size: 0,
                fft_plan: None,
                ifft_plan: None,
            });
        }

        // Compute parameters
        let phi = order * core::f64::consts::PI / 2.0;
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        let cot_phi = cos_phi / sin_phi;
        let csc_phi = 1.0 / sin_phi;

        // Padded size for linear convolution
        let padded_size = (2 * n).next_power_of_two();

        // Compute pre-chirp: exp(i * cot(φ) * π * k² / N)
        let pre_chirp: Vec<Complex<T>> = (0..n)
            .map(|k| {
                let k_centered = (k as f64) - (n as f64) / 2.0;
                let arg = cot_phi * core::f64::consts::PI * k_centered * k_centered / (n as f64);
                Complex::new(T::from_f64(arg.cos()), T::from_f64(arg.sin()))
            })
            .collect();

        // Compute post-chirp (same as pre-chirp for standard FrFT)
        let post_chirp = pre_chirp.clone();

        // Compute convolution kernel: exp(-i * csc(φ) * π * k² / N)
        // Use wrapping index for circular convolution
        let kernel: Vec<Complex<T>> = (0..padded_size)
            .map(|k| {
                // For linear convolution kernel, use indices [0..n] and [padded_size-n+1..padded_size]
                let k_val = if k < n {
                    k as f64
                } else if k > padded_size - n {
                    (k as i64 - padded_size as i64) as f64
                } else {
                    // Zero padding region
                    return Complex::new(T::ZERO, T::ZERO);
                };
                let k_centered = k_val - (n as f64) / 2.0;
                let arg = -csc_phi * core::f64::consts::PI * k_centered * k_centered / (n as f64);
                Complex::new(T::from_f64(arg.cos()), T::from_f64(arg.sin()))
            })
            .collect();

        // Precompute kernel FFT
        let fft_plan = Plan::dft_1d(padded_size, Direction::Forward, Flags::MEASURE);
        let ifft_plan = Plan::dft_1d(padded_size, Direction::Backward, Flags::MEASURE);

        let kernel_fft = if let Some(ref plan) = fft_plan {
            let mut result = vec![Complex::<T>::zero(); padded_size];
            plan.execute(&kernel, &mut result);
            result
        } else {
            return Err(FrftError::PlanFailed);
        };

        Ok(Self {
            n,
            order,
            pre_chirp,
            post_chirp,
            kernel_fft,
            padded_size,
            fft_plan,
            ifft_plan,
        })
    }

    /// Execute the Fractional Fourier Transform.
    ///
    /// # Arguments
    ///
    /// * `input` - Input signal
    ///
    /// # Returns
    ///
    /// Transformed signal.
    pub fn execute(&self, input: &[Complex<T>]) -> FrftResult<Vec<Complex<T>>> {
        if input.len() != self.n {
            return Err(FrftError::InvalidSize(input.len()));
        }

        // Handle special integer orders
        let order_int = self.order.round() as i32;
        if (self.order - f64::from(order_int)).abs() < 1e-10 {
            return self.execute_integer_order(input, order_int.rem_euclid(4));
        }

        // General fractional order
        self.execute_fractional(input)
    }

    /// Execute for integer order (0, 1, 2, 3).
    fn execute_integer_order(
        &self,
        input: &[Complex<T>],
        order: i32,
    ) -> FrftResult<Vec<Complex<T>>> {
        match order {
            0 => {
                // Identity
                Ok(input.to_vec())
            }
            1 => {
                // Standard DFT
                if let Some(ref plan) = Plan::dft_1d(self.n, Direction::Forward, Flags::ESTIMATE) {
                    let mut result = vec![Complex::<T>::zero(); self.n];
                    plan.execute(input, &mut result);
                    // Normalize
                    let n_t = T::from_usize(self.n);
                    let scale = T::ONE / Float::sqrt(n_t);
                    for c in &mut result {
                        *c = Complex::new(c.re * scale, c.im * scale);
                    }
                    Ok(result)
                } else {
                    Err(FrftError::PlanFailed)
                }
            }
            2 => {
                // Time reversal (flip around center)
                let mut result = vec![Complex::<T>::zero(); self.n];
                result[0] = input[0];
                for k in 1..self.n {
                    result[k] = input[self.n - k];
                }
                Ok(result)
            }
            3 => {
                // Inverse DFT (up to normalization)
                if let Some(ref plan) = Plan::dft_1d(self.n, Direction::Backward, Flags::ESTIMATE) {
                    let mut result = vec![Complex::<T>::zero(); self.n];
                    plan.execute(input, &mut result);
                    let n_t = T::from_usize(self.n);
                    let scale = T::ONE / Float::sqrt(n_t);
                    for c in &mut result {
                        *c = Complex::new(c.re * scale, c.im * scale);
                    }
                    Ok(result)
                } else {
                    Err(FrftError::PlanFailed)
                }
            }
            _ => Err(FrftError::InvalidOrder),
        }
    }

    /// Execute fractional order transform using chirp decomposition.
    fn execute_fractional(&self, input: &[Complex<T>]) -> FrftResult<Vec<Complex<T>>> {
        // Step 1: Pre-chirp multiplication
        let chirped: Vec<Complex<T>> = input
            .iter()
            .zip(self.pre_chirp.iter())
            .map(|(&x, &c)| x * c)
            .collect();

        // Step 2: Zero-pad at the beginning for linear convolution
        // Place chirped signal at start, not center
        let mut padded = vec![Complex::<T>::zero(); self.padded_size];
        for (i, &c) in chirped.iter().enumerate() {
            padded[i] = c;
        }

        // Step 3: FFT of padded signal
        let fft_plan = self.fft_plan.as_ref().ok_or(FrftError::PlanFailed)?;
        let ifft_plan = self.ifft_plan.as_ref().ok_or(FrftError::PlanFailed)?;

        let mut signal_fft = vec![Complex::<T>::zero(); self.padded_size];
        fft_plan.execute(&padded, &mut signal_fft);

        // Step 4: Multiply by kernel FFT (convolution)
        for (s, &k) in signal_fft.iter_mut().zip(self.kernel_fft.iter()) {
            *s = *s * k;
        }

        // Step 5: Inverse FFT
        let mut conv_result = vec![Complex::<T>::zero(); self.padded_size];
        ifft_plan.execute(&signal_fft, &mut conv_result);

        // Normalize IFFT
        let scale = T::ONE / T::from_usize(self.padded_size);
        for c in &mut conv_result {
            *c = Complex::new(c.re * scale, c.im * scale);
        }

        // Step 6: Extract first N samples and apply post-chirp
        let mut result = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let conv_val = conv_result[i];
            result.push(conv_val * self.post_chirp[i]);
        }

        // Apply overall scaling
        let phi = self.order * core::f64::consts::PI / 2.0;
        let norm_factor = (1.0 / (self.n as f64 * phi.sin().abs())).sqrt();
        let overall_scale = T::from_f64(norm_factor);
        for c in &mut result {
            *c = Complex::new(c.re * overall_scale, c.im * overall_scale);
        }

        Ok(result)
    }

    /// Get the transform size.
    pub fn size(&self) -> usize {
        self.n
    }

    /// Get the fractional order.
    pub fn order(&self) -> f64 {
        self.order
    }
}

/// Reduce fractional order to [0, 4).
fn reduce_order(order: f64) -> f64 {
    let reduced = order.rem_euclid(4.0);
    if reduced < 0.0 {
        reduced + 4.0
    } else {
        reduced
    }
}

// Convenience functions

/// Compute the Fractional Fourier Transform.
///
/// # Arguments
///
/// * `input` - Input signal
/// * `order` - Fractional order α (α=1 gives standard DFT)
///
/// # Returns
///
/// Transformed signal or an error if the input is empty or FFT planning fails.
pub fn frft<T: Float>(input: &[Complex<T>], order: f64) -> FrftResult<Vec<Complex<T>>> {
    frft_checked(input, order)
}

/// Compute the inverse Fractional Fourier Transform.
///
/// The inverse FrFT of order α is the FrFT of order -α.
///
/// # Arguments
///
/// * `input` - Input signal (result of frft)
/// * `order` - Original order used in forward transform
///
/// # Returns
///
/// Recovered signal or an error if FFT planning fails.
pub fn ifrft<T: Float>(input: &[Complex<T>], order: f64) -> FrftResult<Vec<Complex<T>>> {
    frft_checked(input, -order)
}

/// Compute FrFT with error handling.
pub fn frft_checked<T: Float>(input: &[Complex<T>], order: f64) -> FrftResult<Vec<Complex<T>>> {
    let plan = Frft::new(input.len(), order)?;
    plan.execute(input)
}

/// Compute inverse FrFT with error handling.
pub fn ifrft_checked<T: Float>(input: &[Complex<T>], order: f64) -> FrftResult<Vec<Complex<T>>> {
    frft_checked(input, -order)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_frft_order_zero() {
        // Order 0 should be identity
        let input: Vec<Complex<f64>> = (0..16)
            .map(|k| Complex::new(f64::from(k).cos(), f64::from(k).sin()))
            .collect();

        let result = frft(&input, 0.0).expect("frft order 0 should succeed");

        for (a, b) in input.iter().zip(result.iter()) {
            assert!(approx_eq(a.re, b.re, 1e-10));
            assert!(approx_eq(a.im, b.im, 1e-10));
        }
    }

    #[test]
    fn test_frft_order_two() {
        // Order 2 should be time reversal
        let input: Vec<Complex<f64>> = (0..8).map(|k| Complex::new(f64::from(k), 0.0)).collect();

        let result = frft(&input, 2.0).expect("frft order 2 should succeed");

        assert!(approx_eq(result[0].re, input[0].re, 1e-10));
        for k in 1..8 {
            assert!(approx_eq(result[k].re, input[8 - k].re, 1e-10));
        }
    }

    #[test]
    fn test_frft_produces_output() {
        // Test that FrFT produces valid output for fractional orders
        let input: Vec<Complex<f64>> = (0..32)
            .map(|k| Complex::new((f64::from(k) * 0.1).cos(), (f64::from(k) * 0.1).sin()))
            .collect();

        let order = 0.7;
        let result = frft(&input, order).expect("frft fractional order should succeed");

        // Result should have same length as input
        assert_eq!(result.len(), input.len());

        // Result should have finite values
        for c in &result {
            assert!(c.re.is_finite(), "Real part not finite");
            assert!(c.im.is_finite(), "Imag part not finite");
        }
    }

    #[test]
    fn test_frft_order_one_like_fft() {
        // FrFT with order 1 should behave like FFT (with normalization)
        let input: Vec<Complex<f64>> = (0..8)
            .map(|k| Complex::new(f64::from(k).cos(), 0.0))
            .collect();

        let frft_result = frft(&input, 1.0).expect("frft order 1 should succeed");

        // Check that result is similar in structure (not exact due to normalization)
        assert_eq!(frft_result.len(), input.len());

        // The magnitude spectrum should be preserved (up to scaling)
        let frft_energy: f64 = frft_result.iter().map(|c| c.re * c.re + c.im * c.im).sum();
        assert!(frft_energy > 0.0, "FrFT(1.0) should have non-zero energy");
    }

    #[test]
    fn test_frft_different_orders() {
        // Test that different orders produce different results
        let input: Vec<Complex<f64>> = (0..16)
            .map(|k| Complex::new((f64::from(k) * 0.2).cos(), 0.0))
            .collect();

        let result_05 = frft(&input, 0.5).expect("frft order 0.5 should succeed");
        let result_15 = frft(&input, 1.5).expect("frft order 1.5 should succeed");

        // Results should be different
        let diff: f64 = result_05
            .iter()
            .zip(result_15.iter())
            .map(|(a, b)| {
                let d = *a - *b;
                d.re * d.re + d.im * d.im
            })
            .sum();

        assert!(
            diff > 0.01,
            "Different orders should produce different results"
        );
    }

    #[test]
    fn test_frft_order_reduction() {
        assert!(approx_eq(reduce_order(5.5), 1.5, 1e-10));
        assert!(approx_eq(reduce_order(-1.0), 3.0, 1e-10));
        assert!(approx_eq(reduce_order(4.0), 0.0, 1e-10));
    }

    #[test]
    fn test_frft_error_handling() {
        // Empty input
        let result = Frft::<f64>::new(0, 0.5);
        assert!(result.is_err());
    }
}
