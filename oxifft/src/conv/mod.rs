//! FFT-based convolution and polynomial multiplication.
//!
//! This module provides efficient O(n log n) convolution using FFT,
//! which is fundamental for:
//! - Signal processing (filtering)
//! - Polynomial multiplication
//! - Cross-correlation
//! - Image processing
//!
//! # Types of Convolution
//!
//! - **Linear convolution**: Standard (a * b), output length = len(a) + len(b) - 1
//! - **Circular convolution**: Wraps around, output length = max(len(a), len(b))
//! - **Correlation**: Similar to convolution but without reversal
//!
//! # Algorithm
//!
//! Linear convolution via FFT:
//! 1. Zero-pad both signals to length n ≥ len(a) + len(b) - 1
//! 2. Compute FFT of both
//! 3. Multiply element-wise
//! 4. Compute inverse FFT
//!
//! Complexity: O(n log n) vs O(n²) for direct convolution.
//!
//! # Example
//!
//! ```ignore
//! use oxifft::conv::{convolve, correlate, polynomial_multiply};
//!
//! // Signal convolution
//! let signal = vec![1.0, 2.0, 3.0, 4.0];
//! let kernel = vec![0.5, 0.5];
//! let result = convolve(&signal, &kernel);
//!
//! // Polynomial multiplication: (1 + 2x) * (3 + 4x) = 3 + 10x + 8x²
//! let p1 = vec![1.0, 2.0];  // 1 + 2x
//! let p2 = vec![3.0, 4.0];  // 3 + 4x
//! let product = polynomial_multiply(&p1, &p2);  // [3, 10, 8]
//! ```

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};

/// Convolution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvMode {
    /// Full convolution, output length = len(a) + len(b) - 1.
    Full,
    /// Same size as larger input.
    Same,
    /// Only the parts where signals fully overlap.
    Valid,
}

/// Compute linear convolution of two real signals.
///
/// The convolution `(a * b)[n] = Σ_k a[k] * b[n-k]`
///
/// # Arguments
///
/// * `a` - First signal
/// * `b` - Second signal (kernel)
///
/// # Returns
///
/// Convolution result of length len(a) + len(b) - 1.
pub fn convolve<T: Float>(a: &[T], b: &[T]) -> Vec<T> {
    convolve_mode(a, b, ConvMode::Full)
}

/// Compute linear convolution with specified output mode.
///
/// # Arguments
///
/// * `a` - First signal
/// * `b` - Second signal (kernel)
/// * `mode` - Output mode (Full, Same, Valid)
///
/// # Returns
///
/// Convolution result.
pub fn convolve_mode<T: Float>(a: &[T], b: &[T], mode: ConvMode) -> Vec<T> {
    convolve_with_mode(a, b, mode)
}

/// Compute linear convolution with specified output mode.
pub fn convolve_with_mode<T: Float>(a: &[T], b: &[T], mode: ConvMode) -> Vec<T> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    // For very short signals, use direct convolution
    if a.len() < 32 && b.len() < 32 {
        return convolve_direct(a, b, mode);
    }

    // FFT-based convolution
    let full_len = a.len() + b.len() - 1;
    let fft_len = full_len.next_power_of_two();

    // Convert to complex and zero-pad
    let mut a_complex = vec![Complex::<T>::zero(); fft_len];
    let mut b_complex = vec![Complex::<T>::zero(); fft_len];

    for (i, &val) in a.iter().enumerate() {
        a_complex[i] = Complex::new(val, T::ZERO);
    }
    for (i, &val) in b.iter().enumerate() {
        b_complex[i] = Complex::new(val, T::ZERO);
    }

    // Forward FFT
    let fft_plan = Plan::dft_1d(fft_len, Direction::Forward, Flags::ESTIMATE);
    let ifft_plan = Plan::dft_1d(fft_len, Direction::Backward, Flags::ESTIMATE);

    if fft_plan.is_none() || ifft_plan.is_none() {
        return convolve_direct(a, b, mode);
    }

    let fft_plan = fft_plan.expect("FFT plan");
    let ifft_plan = ifft_plan.expect("IFFT plan");

    let mut a_fft = vec![Complex::<T>::zero(); fft_len];
    let mut b_fft = vec![Complex::<T>::zero(); fft_len];

    fft_plan.execute(&a_complex, &mut a_fft);
    fft_plan.execute(&b_complex, &mut b_fft);

    // Element-wise multiplication
    let mut product = vec![Complex::<T>::zero(); fft_len];
    for i in 0..fft_len {
        product[i] = a_fft[i] * b_fft[i];
    }

    // Inverse FFT
    let mut result_complex = vec![Complex::<T>::zero(); fft_len];
    ifft_plan.execute(&product, &mut result_complex);

    // Normalize and extract real parts
    let scale = T::ONE / T::from_usize(fft_len);
    let full_result: Vec<T> = result_complex
        .iter()
        .take(full_len)
        .map(|c| c.re * scale)
        .collect();

    // Apply output mode
    extract_mode(&full_result, a.len(), b.len(), mode)
}

/// Compute linear convolution of two complex signals.
pub fn convolve_complex<T: Float>(a: &[Complex<T>], b: &[Complex<T>]) -> Vec<Complex<T>> {
    convolve_complex_mode(a, b, ConvMode::Full)
}

/// Compute linear convolution of complex signals with specified mode.
pub fn convolve_complex_mode<T: Float>(
    a: &[Complex<T>],
    b: &[Complex<T>],
    mode: ConvMode,
) -> Vec<Complex<T>> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    let full_len = a.len() + b.len() - 1;
    let fft_len = full_len.next_power_of_two();

    // Zero-pad
    let mut a_padded = vec![Complex::<T>::zero(); fft_len];
    let mut b_padded = vec![Complex::<T>::zero(); fft_len];

    a_padded[..a.len()].copy_from_slice(a);
    b_padded[..b.len()].copy_from_slice(b);

    // Forward FFT
    let fft_plan = match Plan::dft_1d(fft_len, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return convolve_complex_direct(a, b, mode),
    };
    let ifft_plan = match Plan::dft_1d(fft_len, Direction::Backward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return convolve_complex_direct(a, b, mode),
    };

    let mut a_fft = vec![Complex::<T>::zero(); fft_len];
    let mut b_fft = vec![Complex::<T>::zero(); fft_len];

    fft_plan.execute(&a_padded, &mut a_fft);
    fft_plan.execute(&b_padded, &mut b_fft);

    // Element-wise multiplication
    for i in 0..fft_len {
        a_fft[i] = a_fft[i] * b_fft[i];
    }

    // Inverse FFT
    let mut result = vec![Complex::<T>::zero(); fft_len];
    ifft_plan.execute(&a_fft, &mut result);

    // Normalize
    let scale = T::ONE / T::from_usize(fft_len);
    for c in &mut result {
        *c = Complex::new(c.re * scale, c.im * scale);
    }

    // Apply output mode
    let full_result: Vec<Complex<T>> = result.into_iter().take(full_len).collect();
    extract_mode_complex(&full_result, a.len(), b.len(), mode)
}

/// Compute circular convolution (wraps around).
pub fn convolve_circular<T: Float>(a: &[T], b: &[T]) -> Vec<T> {
    let n = a.len().max(b.len());

    // Zero-pad to same length
    let mut a_padded = vec![T::ZERO; n];
    let mut b_padded = vec![T::ZERO; n];

    for (i, &val) in a.iter().enumerate() {
        a_padded[i] = val;
    }
    for (i, &val) in b.iter().enumerate() {
        b_padded[i] = val;
    }

    // Convert to complex
    let a_complex: Vec<Complex<T>> = a_padded.iter().map(|&x| Complex::new(x, T::ZERO)).collect();
    let b_complex: Vec<Complex<T>> = b_padded.iter().map(|&x| Complex::new(x, T::ZERO)).collect();

    // FFT-based circular convolution
    let fft_plan = match Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return convolve_circular_direct(&a_padded, &b_padded),
    };
    let ifft_plan = match Plan::dft_1d(n, Direction::Backward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return convolve_circular_direct(&a_padded, &b_padded),
    };

    let mut a_fft = vec![Complex::<T>::zero(); n];
    let mut b_fft = vec![Complex::<T>::zero(); n];

    fft_plan.execute(&a_complex, &mut a_fft);
    fft_plan.execute(&b_complex, &mut b_fft);

    // Element-wise multiplication
    for i in 0..n {
        a_fft[i] = a_fft[i] * b_fft[i];
    }

    // Inverse FFT
    let mut result = vec![Complex::<T>::zero(); n];
    ifft_plan.execute(&a_fft, &mut result);

    // Normalize and extract real parts
    let scale = T::ONE / T::from_usize(n);
    result.iter().map(|c| c.re * scale).collect()
}

/// Compute cross-correlation of two signals.
///
/// Correlation is similar to convolution but without reversing b:
/// `(a ⋆ b)[n] = Σ_k a[k] * conj(b[k-n])`
///
/// For real signals, this equals convolve(a, reverse(b)).
pub fn correlate<T: Float>(a: &[T], b: &[T]) -> Vec<T> {
    correlate_mode(a, b, ConvMode::Full)
}

/// Compute cross-correlation with specified mode.
pub fn correlate_mode<T: Float>(a: &[T], b: &[T], mode: ConvMode) -> Vec<T> {
    if b.is_empty() {
        return Vec::new();
    }

    // For real signals, correlation = convolution with reversed kernel
    let b_reversed: Vec<T> = b.iter().rev().copied().collect();
    convolve_with_mode(a, &b_reversed, mode)
}

/// Compute cross-correlation of complex signals.
pub fn correlate_complex<T: Float>(a: &[Complex<T>], b: &[Complex<T>]) -> Vec<Complex<T>> {
    correlate_complex_mode(a, b, ConvMode::Full)
}

/// Compute cross-correlation of complex signals with mode.
pub fn correlate_complex_mode<T: Float>(
    a: &[Complex<T>],
    b: &[Complex<T>],
    mode: ConvMode,
) -> Vec<Complex<T>> {
    if b.is_empty() {
        return Vec::new();
    }

    // For complex signals, correlation uses conjugate of reversed b
    let b_conj_rev: Vec<Complex<T>> = b.iter().rev().map(|c| c.conj()).collect();
    convolve_complex_mode(a, &b_conj_rev, mode)
}

/// Multiply two polynomials using FFT.
///
/// Given polynomials `p(x) = Σ a[i] * x^i` and `q(x) = Σ b[i] * x^i`,
/// computes their product r(x) = p(x) * q(x).
///
/// # Arguments
///
/// * `a` - Coefficients of first polynomial [a_0, a_1, ..., a_n]
/// * `b` - Coefficients of second polynomial [b_0, b_1, ..., b_m]
///
/// # Returns
///
/// Coefficients of product polynomial with length n + m + 1.
pub fn polynomial_multiply<T: Float>(a: &[T], b: &[T]) -> Vec<T> {
    convolve(a, b)
}

/// Multiply two polynomials with complex coefficients.
pub fn polynomial_multiply_complex<T: Float>(
    a: &[Complex<T>],
    b: &[Complex<T>],
) -> Vec<Complex<T>> {
    convolve_complex(a, b)
}

/// Compute polynomial power using repeated squaring.
///
/// Computes p(x)^n efficiently.
pub fn polynomial_power<T: Float>(p: &[T], n: u32) -> Vec<T> {
    if n == 0 {
        return vec![T::ONE];
    }
    if n == 1 {
        return p.to_vec();
    }
    if p.is_empty() {
        return Vec::new();
    }

    // Binary exponentiation
    let mut result = vec![T::ONE];
    let mut base = p.to_vec();
    let mut exp = n;

    while exp > 0 {
        if exp & 1 == 1 {
            result = polynomial_multiply(&result, &base);
        }
        base = polynomial_multiply(&base, &base);
        exp >>= 1;
    }

    result
}

// Direct implementations for small inputs

fn convolve_direct<T: Float>(a: &[T], b: &[T], mode: ConvMode) -> Vec<T> {
    let full_len = a.len() + b.len() - 1;
    let mut result = vec![T::ZERO; full_len];

    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] = result[i + j] + ai * bj;
        }
    }

    extract_mode(&result, a.len(), b.len(), mode)
}

fn convolve_complex_direct<T: Float>(
    a: &[Complex<T>],
    b: &[Complex<T>],
    mode: ConvMode,
) -> Vec<Complex<T>> {
    let full_len = a.len() + b.len() - 1;
    let mut result = vec![Complex::<T>::zero(); full_len];

    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] = result[i + j] + ai * bj;
        }
    }

    extract_mode_complex(&result, a.len(), b.len(), mode)
}

fn convolve_circular_direct<T: Float>(a: &[T], b: &[T]) -> Vec<T> {
    let n = a.len();
    let mut result = vec![T::ZERO; n];

    for (i, r) in result.iter_mut().enumerate() {
        for j in 0..n {
            let b_idx = (n + i - j) % n;
            *r = *r + a[j] * b[b_idx];
        }
    }

    result
}

fn extract_mode<T: Clone>(full: &[T], a_len: usize, b_len: usize, mode: ConvMode) -> Vec<T> {
    match mode {
        ConvMode::Full => full.to_vec(),
        ConvMode::Same => {
            let start = (b_len - 1) / 2;
            let len = a_len.max(b_len);
            full[start..start + len].to_vec()
        }
        ConvMode::Valid => {
            let len = a_len.max(b_len) - a_len.min(b_len) + 1;
            let start = a_len.min(b_len) - 1;
            full[start..start + len].to_vec()
        }
    }
}

fn extract_mode_complex<T: Float>(
    full: &[Complex<T>],
    a_len: usize,
    b_len: usize,
    mode: ConvMode,
) -> Vec<Complex<T>> {
    match mode {
        ConvMode::Full => full.to_vec(),
        ConvMode::Same => {
            let start = (b_len - 1) / 2;
            let len = a_len.max(b_len);
            full[start..start + len].to_vec()
        }
        ConvMode::Valid => {
            let len = a_len.max(b_len) - a_len.min(b_len) + 1;
            let start = a_len.min(b_len) - 1;
            full[start..start + len].to_vec()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_convolve_simple() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 1.0, 0.5];

        let result = convolve(&a, &b);

        // Expected: [0, 1, 2.5, 4, 1.5]
        assert_eq!(result.len(), 5);
        assert!(approx_eq(result[0], 0.0, 1e-10));
        assert!(approx_eq(result[1], 1.0, 1e-10));
        assert!(approx_eq(result[2], 2.5, 1e-10));
        assert!(approx_eq(result[3], 4.0, 1e-10));
        assert!(approx_eq(result[4], 1.5, 1e-10));
    }

    #[test]
    fn test_polynomial_multiply() {
        // (1 + 2x)(3 + 4x) = 3 + 10x + 8x²
        let p1 = vec![1.0, 2.0];
        let p2 = vec![3.0, 4.0];

        let result = polynomial_multiply(&p1, &p2);

        assert_eq!(result.len(), 3);
        assert!(approx_eq(result[0], 3.0, 1e-10));
        assert!(approx_eq(result[1], 10.0, 1e-10));
        assert!(approx_eq(result[2], 8.0, 1e-10));
    }

    #[test]
    fn test_polynomial_power() {
        // (1 + x)^2 = 1 + 2x + x²
        let p = vec![1.0, 1.0];
        let result = polynomial_power(&p, 2);

        assert_eq!(result.len(), 3);
        assert!(approx_eq(result[0], 1.0, 1e-10));
        assert!(approx_eq(result[1], 2.0, 1e-10));
        assert!(approx_eq(result[2], 1.0, 1e-10));
    }

    #[test]
    fn test_polynomial_power_cubic() {
        // (1 + x)^3 = 1 + 3x + 3x² + x³
        let p = vec![1.0, 1.0];
        let result = polynomial_power(&p, 3);

        assert_eq!(result.len(), 4);
        assert!(approx_eq(result[0], 1.0, 1e-10));
        assert!(approx_eq(result[1], 3.0, 1e-10));
        assert!(approx_eq(result[2], 3.0, 1e-10));
        assert!(approx_eq(result[3], 1.0, 1e-10));
    }

    #[test]
    fn test_correlate() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 1.0, 2.0];

        let corr = correlate(&a, &b);

        // Correlation = convolution with reversed b
        let b_rev = vec![2.0, 1.0, 0.0];
        let conv = convolve(&a, &b_rev);

        for (c, v) in corr.iter().zip(conv.iter()) {
            assert!(approx_eq(*c, *v, 1e-10));
        }
    }

    #[test]
    fn test_circular_convolution() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 0.0];

        // Convolving with [1, 0, 0, 0] should return the original
        let result = convolve_circular(&a, &b);

        for (r, &expected) in result.iter().zip(a.iter()) {
            assert!(approx_eq(*r, expected, 1e-10));
        }
    }

    #[test]
    fn test_convolve_empty() {
        let a: Vec<f64> = vec![];
        let b = vec![1.0, 2.0];

        let result = convolve(&a, &b);
        assert!(result.is_empty());
    }

    #[test]
    fn test_convolve_mode_same() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 1.0, 1.0];

        let result = convolve_with_mode(&a, &b, ConvMode::Same);

        // Same mode: output length = max(len(a), len(b)) = 5
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_convolve_mode_valid() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 1.0, 1.0];

        let result = convolve_with_mode(&a, &b, ConvMode::Valid);

        // Valid mode: output length = max - min + 1 = 5 - 3 + 1 = 3
        assert_eq!(result.len(), 3);
    }
}
