//! Half-Complex to Half-Complex solver.
//!
//! Performs operations on data in half-complex format, keeping it in that format.
//! This is useful for applying twiddle factors and other operations directly
//! on half-complex data without converting to/from complex format.

use crate::kernel::Float;
#[allow(unused_imports)]
use crate::prelude::*;

/// Half-Complex to Half-Complex solver.
///
/// Operates on data in FFTW half-complex format:
/// - hc\[0\] = Re(X\[0\]) (DC)
/// - hc\[1\], hc\[2\] = Re(X\[1\]), Im(X\[1\])
/// - hc\[3\], hc\[4\] = Re(X\[2\]), Im(X\[2\])
/// - ...
/// - For even N: hc\[N-1\] = Re(X\[N/2\]) (Nyquist)
pub struct Hc2hcSolver<T: Float> {
    /// Transform size
    n: usize,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> Default for Hc2hcSolver<T> {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<T: Float> Hc2hcSolver<T> {
    /// Create a new half-complex to half-complex solver.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            n,
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns the solver name identifier (`"rdft-hc2hc"`).
    #[must_use]
    pub fn name(&self) -> &'static str {
        "rdft-hc2hc"
    }

    /// Get the transform size.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Apply a scale factor to half-complex data.
    ///
    /// Scales all elements by the given factor.
    pub fn scale(&self, data: &mut [T], factor: T) {
        assert_eq!(data.len(), self.n, "Data must have length n");
        for x in data.iter_mut() {
            *x = *x * factor;
        }
    }

    /// Normalize half-complex data by 1/n.
    pub fn normalize(&self, data: &mut [T]) {
        let factor = T::ONE / T::from_usize(self.n);
        self.scale(data, factor);
    }

    /// Add two half-complex arrays element-wise.
    ///
    /// result\[i\] = a\[i\] + b\[i\]
    pub fn add(&self, a: &[T], b: &[T], result: &mut [T]) {
        assert_eq!(a.len(), self.n);
        assert_eq!(b.len(), self.n);
        assert_eq!(result.len(), self.n);

        for i in 0..self.n {
            result[i] = a[i] + b[i];
        }
    }

    /// Subtract two half-complex arrays element-wise.
    ///
    /// result\[i\] = a\[i\] - b\[i\]
    pub fn sub(&self, a: &[T], b: &[T], result: &mut [T]) {
        assert_eq!(a.len(), self.n);
        assert_eq!(b.len(), self.n);
        assert_eq!(result.len(), self.n);

        for i in 0..self.n {
            result[i] = a[i] - b[i];
        }
    }

    /// Multiply two half-complex arrays element-wise (complex multiplication).
    ///
    /// Treats the half-complex data as complex numbers and performs
    /// element-wise complex multiplication.
    pub fn mul(&self, a: &[T], b: &[T], result: &mut [T]) {
        assert_eq!(a.len(), self.n);
        assert_eq!(b.len(), self.n);
        assert_eq!(result.len(), self.n);

        if self.n == 0 {
            return;
        }

        // DC component (real * real)
        result[0] = a[0] * b[0];

        if self.n == 1 {
            return;
        }

        // Middle components (complex multiplication)
        let num_pairs = (self.n - 1) / 2;
        for k in 1..=num_pairs {
            let re_idx = 2 * k - 1;
            let im_idx = 2 * k;

            let a_re = a[re_idx];
            let a_im = a[im_idx];
            let b_re = b[re_idx];
            let b_im = b[im_idx];

            // (a_re + i*a_im) * (b_re + i*b_im)
            // = (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)
            result[re_idx] = a_re * b_re - a_im * b_im;
            result[im_idx] = a_re * b_im + a_im * b_re;
        }

        // Nyquist component for even N (real * real)
        if self.n.is_multiple_of(2) {
            result[self.n - 1] = a[self.n - 1] * b[self.n - 1];
        }
    }

    /// Compute the complex conjugate of half-complex data.
    ///
    /// Negates all imaginary parts.
    pub fn conj(&self, data: &[T], result: &mut [T]) {
        assert_eq!(data.len(), self.n);
        assert_eq!(result.len(), self.n);

        if self.n == 0 {
            return;
        }

        // DC (real)
        result[0] = data[0];

        if self.n == 1 {
            return;
        }

        // Middle components
        let num_pairs = (self.n - 1) / 2;
        for k in 1..=num_pairs {
            let re_idx = 2 * k - 1;
            let im_idx = 2 * k;
            result[re_idx] = data[re_idx];
            result[im_idx] = -data[im_idx]; // Negate imaginary part
        }

        // Nyquist (real)
        if self.n.is_multiple_of(2) {
            result[self.n - 1] = data[self.n - 1];
        }
    }

    /// Compute the squared magnitude of each complex component.
    ///
    /// For each frequency k: |X\[k\]|² = Re(X\[k\])² + Im(X\[k\])²
    /// Output is in the same half-complex layout but with Im parts set to 0.
    pub fn mag_squared(&self, data: &[T], result: &mut [T]) {
        assert_eq!(data.len(), self.n);
        assert_eq!(result.len(), self.n);

        if self.n == 0 {
            return;
        }

        // DC
        result[0] = data[0] * data[0];

        if self.n == 1 {
            return;
        }

        // Middle components
        let num_pairs = (self.n - 1) / 2;
        for k in 1..=num_pairs {
            let re_idx = 2 * k - 1;
            let im_idx = 2 * k;
            let mag_sq = data[re_idx] * data[re_idx] + data[im_idx] * data[im_idx];
            result[re_idx] = mag_sq;
            result[im_idx] = T::ZERO;
        }

        // Nyquist
        if self.n.is_multiple_of(2) {
            result[self.n - 1] = data[self.n - 1] * data[self.n - 1];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_hc2hc_scale() {
        let solver = Hc2hcSolver::<f64>::new(4);
        let mut data = [1.0, 2.0, 3.0, 4.0];

        solver.scale(&mut data, 2.0);

        assert!(approx_eq(data[0], 2.0, 1e-10));
        assert!(approx_eq(data[1], 4.0, 1e-10));
        assert!(approx_eq(data[2], 6.0, 1e-10));
        assert!(approx_eq(data[3], 8.0, 1e-10));
    }

    #[test]
    fn test_hc2hc_normalize() {
        let solver = Hc2hcSolver::<f64>::new(4);
        let mut data = [4.0, 8.0, 12.0, 16.0];

        solver.normalize(&mut data);

        assert!(approx_eq(data[0], 1.0, 1e-10));
        assert!(approx_eq(data[1], 2.0, 1e-10));
        assert!(approx_eq(data[2], 3.0, 1e-10));
        assert!(approx_eq(data[3], 4.0, 1e-10));
    }

    #[test]
    fn test_hc2hc_add() {
        let solver = Hc2hcSolver::<f64>::new(4);
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut result = [0.0; 4];

        solver.add(&a, &b, &mut result);

        assert!(approx_eq(result[0], 6.0, 1e-10));
        assert!(approx_eq(result[1], 8.0, 1e-10));
        assert!(approx_eq(result[2], 10.0, 1e-10));
        assert!(approx_eq(result[3], 12.0, 1e-10));
    }

    #[test]
    fn test_hc2hc_mul_size_4() {
        // n=4: [r0, r1, i1, r2]
        // a = [2, 1, 1, 3] represents DC=2, X[1]=1+i, X[2]=3
        // b = [1, 2, -1, 2] represents DC=1, X[1]=2-i, X[2]=2
        // Result: DC=2*1=2, X[1]=(1+i)*(2-i)=2-i+2i-i²=3+i, X[2]=3*2=6
        let solver = Hc2hcSolver::<f64>::new(4);
        let a = [2.0, 1.0, 1.0, 3.0];
        let b = [1.0, 2.0, -1.0, 2.0];
        let mut result = [0.0; 4];

        solver.mul(&a, &b, &mut result);

        assert!(approx_eq(result[0], 2.0, 1e-10)); // DC
        assert!(approx_eq(result[1], 3.0, 1e-10)); // Re(X[1])
        assert!(approx_eq(result[2], 1.0, 1e-10)); // Im(X[1])
        assert!(approx_eq(result[3], 6.0, 1e-10)); // X[2] (Nyquist)
    }

    #[test]
    fn test_hc2hc_conj() {
        let solver = Hc2hcSolver::<f64>::new(4);
        let data = [1.0, 2.0, 3.0, 4.0]; // DC=1, X[1]=2+3i, X[2]=4
        let mut result = [0.0; 4];

        solver.conj(&data, &mut result);

        assert!(approx_eq(result[0], 1.0, 1e-10)); // DC unchanged
        assert!(approx_eq(result[1], 2.0, 1e-10)); // Re unchanged
        assert!(approx_eq(result[2], -3.0, 1e-10)); // Im negated
        assert!(approx_eq(result[3], 4.0, 1e-10)); // Nyquist unchanged
    }

    #[test]
    fn test_hc2hc_mag_squared() {
        let solver = Hc2hcSolver::<f64>::new(4);
        // DC=2, X[1]=3+4i (|X[1]|²=25), X[2]=5
        let data = [2.0, 3.0, 4.0, 5.0];
        let mut result = [0.0; 4];

        solver.mag_squared(&data, &mut result);

        assert!(approx_eq(result[0], 4.0, 1e-10)); // |DC|² = 4
        assert!(approx_eq(result[1], 25.0, 1e-10)); // |X[1]|² = 9+16 = 25
        assert!(approx_eq(result[2], 0.0, 1e-10)); // Im = 0
        assert!(approx_eq(result[3], 25.0, 1e-10)); // |X[2]|² = 25
    }
}
