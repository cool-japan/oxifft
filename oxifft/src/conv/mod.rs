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
//! ```
//! use oxifft::conv::{convolve, polynomial_multiply};
//!
//! // Signal convolution
//! let signal = vec![1.0, 2.0, 3.0, 4.0];
//! let kernel = vec![0.5, 0.5];
//! let result = convolve(&signal, &kernel);
//! assert_eq!(result.len(), signal.len() + kernel.len() - 1);
//!
//! // Polynomial multiplication: (1 + 2x) * (3 + 4x) = 3 + 10x + 8x²
//! let p1 = vec![1.0, 2.0]; // 1 + 2x
//! let p2 = vec![3.0, 4.0]; // 3 + 4x
//! let product = polynomial_multiply(&p1, &p2); // ≈ [3, 10, 8]
//! assert_eq!(product.len(), 3);
//! ```

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};

/// Convolution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
    let Some(fft_plan) = Plan::dft_1d(fft_len, Direction::Forward, Flags::ESTIMATE) else {
        return convolve_direct(a, b, mode);
    };
    let Some(ifft_plan) = Plan::dft_1d(fft_len, Direction::Backward, Flags::ESTIMATE) else {
        return convolve_direct(a, b, mode);
    };

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

    // -------------------------------------------------------------------
    // FFT-based convolution path (N >= 32) direct tests.
    //
    // `convolve_with_mode` only takes the FFT branch when *both* inputs are
    // >= 32 elements (see the `a.len() < 32 && b.len() < 32` check above);
    // every test above this point uses 2-5 element inputs and therefore only
    // exercises `convolve_direct`.  These tests use inputs >= 32 elements
    // and independently-computed O(n^2) references (not calling the
    // library's own `convolve_direct`/`convolve_circular_direct` helpers) so
    // that the FFT-based code paths (lines computing `fft_len`,
    // `Plan::dft_1d`, and the pointwise-multiply + inverse FFT) are actually
    // exercised and checked against ground truth.
    // -------------------------------------------------------------------

    /// Independent O(n*m) reference for full linear convolution (f64).
    fn naive_convolve_full_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0_f64; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                result[i + j] += ai * bj;
            }
        }
        result
    }

    /// Independent O(n*m) reference for full linear convolution (f32).
    fn naive_convolve_full_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0_f32; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                result[i + j] += ai * bj;
            }
        }
        result
    }

    /// Independent O(n*m) reference for complex full linear convolution (f64).
    fn naive_convolve_full_complex_f64(
        a: &[Complex<f64>],
        b: &[Complex<f64>],
    ) -> Vec<Complex<f64>> {
        let mut result = vec![Complex::<f64>::zero(); a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                result[i + j] = result[i + j] + ai * bj;
            }
        }
        result
    }

    /// Independent O(n^2) reference for circular convolution:
    /// `c[i] = sum_j a[j] * b[(i-j) mod n]`, zero-padding the shorter input.
    fn naive_circular_convolve_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
        let n = a.len().max(b.len());
        let mut ap = vec![0.0_f64; n];
        ap[..a.len()].copy_from_slice(a);
        let mut bp = vec![0.0_f64; n];
        bp[..b.len()].copy_from_slice(b);

        let mut result = vec![0.0_f64; n];
        for i in 0..n {
            let mut sum = 0.0_f64;
            for j in 0..n {
                sum += ap[j] * bp[(i + n - j) % n];
            }
            result[i] = sum;
        }
        result
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    fn max_abs_diff_f32(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    fn max_abs_diff_complex(a: &[Complex<f64>], b: &[Complex<f64>]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).norm())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn test_convolve_fft_path_even_lengths_f64() {
        let a: Vec<f64> = (0..40).map(|i| (f64::from(i) * 0.37).sin()).collect();
        let b: Vec<f64> = (0..36).map(|i| (f64::from(i) * 0.61).cos()).collect();
        assert!(a.len() >= 32 && b.len() >= 32, "must exercise FFT path");
        assert!(a.len().is_multiple_of(2) && b.len().is_multiple_of(2));

        let result = convolve(&a, &b);
        let reference = naive_convolve_full_f64(&a, &b);
        assert_eq!(result.len(), reference.len());

        let err = max_abs_diff(&result, &reference);
        assert!(
            err < 1e-9,
            "even-length FFT convolution error {err:.2e} > 1e-9"
        );
    }

    #[test]
    fn test_convolve_fft_path_odd_lengths_f64() {
        let a: Vec<f64> = (0..41).map(|i| (f64::from(i) * 0.23).sin()).collect();
        let b: Vec<f64> = (0..35).map(|i| (f64::from(i) * 0.53).cos()).collect();
        assert!(a.len() >= 32 && b.len() >= 32, "must exercise FFT path");
        assert!(!a.len().is_multiple_of(2) && !b.len().is_multiple_of(2));

        let result = convolve(&a, &b);
        let reference = naive_convolve_full_f64(&a, &b);
        assert_eq!(result.len(), reference.len());

        let err = max_abs_diff(&result, &reference);
        assert!(
            err < 1e-9,
            "odd-length FFT convolution error {err:.2e} > 1e-9"
        );
    }

    #[test]
    fn test_convolve_fft_path_f32() {
        let a: Vec<f32> = (0..50).map(|i| (i as f32 * 0.17).sin()).collect();
        let b: Vec<f32> = (0..45).map(|i| (i as f32 * 0.29).cos()).collect();
        assert!(a.len() >= 32 && b.len() >= 32, "must exercise FFT path");

        let result = convolve(&a, &b);
        let reference = naive_convolve_full_f32(&a, &b);
        assert_eq!(result.len(), reference.len());

        let err = max_abs_diff_f32(&result, &reference);
        // f32 FFT convolution accumulates more rounding than f64; the
        // dynamic range here (~50-tap convolution of O(1) values) stays
        // well within a 1e-3 absolute budget.
        assert!(err < 1e-3, "f32 FFT convolution error {err:.2e} > 1e-3");
    }

    #[test]
    fn test_convolve_complex_fft_path_f64() {
        let a: Vec<Complex<f64>> = (0..48)
            .map(|i| Complex::new((f64::from(i) * 0.11).sin(), (f64::from(i) * 0.19).cos()))
            .collect();
        let b: Vec<Complex<f64>> = (0..37)
            .map(|i| Complex::new((f64::from(i) * 0.31).cos(), (f64::from(i) * 0.41).sin()))
            .collect();
        assert!(a.len() >= 32 && b.len() >= 32, "must exercise FFT path");

        let result = convolve_complex(&a, &b);
        let reference = naive_convolve_full_complex_f64(&a, &b);
        assert_eq!(result.len(), reference.len());

        let err = max_abs_diff_complex(&result, &reference);
        assert!(err < 1e-9, "complex FFT convolution error {err:.2e} > 1e-9");
    }

    #[test]
    fn test_convolve_circular_fft_path_f64() {
        // `convolve_circular` always uses the FFT path (no small-size direct
        // fallback other than on plan-construction failure), but existing
        // coverage only used a 4-element identity kernel; use a real N>=32
        // kernel here.
        let n = 64;
        let a: Vec<f64> = (0..n).map(|i| (f64::from(i) * 0.15).sin()).collect();
        let b: Vec<f64> = (0..n).map(|i| (f64::from(i) * 0.27).cos() * 0.5).collect();

        let result = convolve_circular(&a, &b);
        let reference = naive_circular_convolve_f64(&a, &b);
        assert_eq!(result.len(), reference.len());

        let err = max_abs_diff(&result, &reference);
        assert!(
            err < 1e-9,
            "circular FFT convolution error {err:.2e} > 1e-9"
        );
    }

    #[test]
    fn test_convolve_circular_fft_path_odd_length_f64() {
        let n = 33; // odd, still >= 32
        let a: Vec<f64> = (0..n).map(|i| (f64::from(i) * 0.19).sin()).collect();
        let b: Vec<f64> = (0..n).map(|i| (f64::from(i) * 0.33).cos()).collect();

        let result = convolve_circular(&a, &b);
        let reference = naive_circular_convolve_f64(&a, &b);
        assert_eq!(result.len(), reference.len());

        let err = max_abs_diff(&result, &reference);
        assert!(
            err < 1e-8,
            "odd-length circular FFT convolution error {err:.2e} > 1e-8"
        );
    }

    #[test]
    fn test_correlate_fft_path_f64() {
        let a: Vec<f64> = (0..45).map(|i| (f64::from(i) * 0.12).sin()).collect();
        let b: Vec<f64> = (0..38).map(|i| (f64::from(i) * 0.22).cos()).collect();
        assert!(a.len() >= 32 && b.len() >= 32, "must exercise FFT path");

        let result = correlate(&a, &b);

        // Independent reference: correlation = convolution of a with
        // reversed b (definition used throughout this module).
        let b_rev: Vec<f64> = b.iter().rev().copied().collect();
        let reference = naive_convolve_full_f64(&a, &b_rev);
        assert_eq!(result.len(), reference.len());

        let err = max_abs_diff(&result, &reference);
        assert!(err < 1e-9, "FFT correlation error {err:.2e} > 1e-9");
    }

    #[test]
    fn test_correlate_complex_fft_path_f64() {
        let a: Vec<Complex<f64>> = (0..40)
            .map(|i| Complex::new((f64::from(i) * 0.08).sin(), (f64::from(i) * 0.14).cos()))
            .collect();
        let b: Vec<Complex<f64>> = (0..33)
            .map(|i| Complex::new((f64::from(i) * 0.18).cos(), (f64::from(i) * 0.26).sin()))
            .collect();
        assert!(a.len() >= 32 && b.len() >= 32, "must exercise FFT path");

        let result = correlate_complex(&a, &b);

        // Independent reference: correlation = convolution of a with
        // conj(reversed(b)) (definition used throughout this module).
        let b_conj_rev: Vec<Complex<f64>> = b.iter().rev().map(|c| c.conj()).collect();
        let reference = naive_convolve_full_complex_f64(&a, &b_conj_rev);
        assert_eq!(result.len(), reference.len());

        let err = max_abs_diff_complex(&result, &reference);
        assert!(err < 1e-9, "complex FFT correlation error {err:.2e} > 1e-9");
    }

    #[test]
    fn test_convolve_mode_same_and_valid_fft_path_f64() {
        // Verify Same/Valid extraction is correct (not just length) once the
        // FFT path is actually taken.
        let a: Vec<f64> = (0..50).map(|i| (f64::from(i) * 0.09).sin()).collect();
        let b: Vec<f64> = (0..32).map(|i| (f64::from(i) * 0.21).cos()).collect();
        assert!(a.len() >= 32 && b.len() >= 32, "must exercise FFT path");

        let full = naive_convolve_full_f64(&a, &b);

        let same = convolve_with_mode(&a, &b, ConvMode::Same);
        let start_same = (b.len() - 1) / 2;
        let len_same = a.len().max(b.len());
        let expected_same = &full[start_same..start_same + len_same];
        assert_eq!(same.len(), expected_same.len());
        assert!(max_abs_diff(&same, expected_same) < 1e-9);

        let valid = convolve_with_mode(&a, &b, ConvMode::Valid);
        let start_valid = a.len().min(b.len()) - 1;
        let len_valid = a.len().max(b.len()) - a.len().min(b.len()) + 1;
        let expected_valid = &full[start_valid..start_valid + len_valid];
        assert_eq!(valid.len(), expected_valid.len());
        assert!(max_abs_diff(&valid, expected_valid) < 1e-9);
    }
}
