//! Half-Complex to Complex solver.
//!
//! Converts FFTW's half-complex format to standard complex format.
//!
//! Half-complex format stores the output of a real FFT in a compact real array:
//! - For N real inputs: [r0, r1, i1, r2, i2, ..., r(N/2)] for even N
//! - For N real inputs: [r0, r1, i1, r2, i2, ..., r(N/2), i(N/2)] for odd N
//!
//! This is converted to standard complex format: [c0, c1, c2, ..., c(N/2)]
//! where each ck = rk + i*ik (with i0 = 0 and iN/2 = 0 for even N).

use crate::kernel::{Complex, Float};
#[allow(unused_imports)]
// reason: prelude glob re-exports are selectively used per feature gate (std vs no_std)
use crate::prelude::*;

/// Half-Complex to Complex solver.
///
/// Converts FFTW half-complex format to standard interleaved complex format.
pub struct Hc2cSolver<T: Float> {
    /// Transform size (original real FFT size)
    n: usize,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> Default for Hc2cSolver<T> {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<T: Float> Hc2cSolver<T> {
    /// Create a new half-complex to complex solver.
    ///
    /// # Arguments
    /// * `n` - The original real FFT size (length of the half-complex array)
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            n,
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns the solver name identifier (`"rdft-hc2c"`).
    #[must_use]
    pub fn name(&self) -> &'static str {
        "rdft-hc2c"
    }

    /// Get the transform size.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the number of complex outputs.
    #[must_use]
    pub fn output_len(&self) -> usize {
        self.n / 2 + 1
    }

    /// Convert half-complex format to complex format.
    ///
    /// # Arguments
    /// * `halfcomplex` - Input in half-complex format (length n)
    /// * `complex` - Output in complex format (length n/2+1)
    ///
    /// # Half-Complex Format
    /// For size N:
    /// - hc\[0\] = Re(X\[0\]) (DC component, purely real)
    /// - hc\[1\] = Re(X\[1\])
    /// - hc\[2\] = Im(X\[1\])
    /// - hc\[3\] = Re(X\[2\])
    /// - hc\[4\] = Im(X\[2\])
    /// - ...
    /// - For even N: hc\[N-1\] = Re(X\[N/2\]) (Nyquist, purely real)
    /// - For odd N: hc\[N-2\] = Re(X\[N/2\]), hc\[N-1\] = Im(X\[N/2\])
    pub fn execute(&self, halfcomplex: &[T], complex: &mut [Complex<T>]) {
        assert_eq!(
            halfcomplex.len(),
            self.n,
            "Half-complex input must have length n"
        );
        assert_eq!(
            complex.len(),
            self.output_len(),
            "Complex output must have length n/2+1"
        );

        if self.n == 0 {
            return;
        }

        // DC component (index 0) is purely real
        complex[0] = Complex::new(halfcomplex[0], T::ZERO);

        if self.n == 1 {
            return;
        }

        // Middle components
        let num_pairs = (self.n - 1) / 2;
        for k in 1..=num_pairs {
            let re_idx = 2 * k - 1;
            let im_idx = 2 * k;
            complex[k] = Complex::new(halfcomplex[re_idx], halfcomplex[im_idx]);
        }

        // Nyquist component for even N (purely real)
        if self.n.is_multiple_of(2) {
            complex[self.n / 2] = Complex::new(halfcomplex[self.n - 1], T::ZERO);
        }
    }
}

/// Convert complex format to half-complex format.
///
/// This is the inverse operation of `Hc2cSolver::execute`.
pub struct C2hcSolver<T: Float> {
    /// Transform size (target half-complex array size)
    n: usize,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> Default for C2hcSolver<T> {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<T: Float> C2hcSolver<T> {
    /// Create a new complex to half-complex solver.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            n,
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns the solver name identifier (`"rdft-c2hc"`).
    #[must_use]
    pub fn name(&self) -> &'static str {
        "rdft-c2hc"
    }

    /// Get the transform size.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the number of complex inputs.
    #[must_use]
    pub fn input_len(&self) -> usize {
        self.n / 2 + 1
    }

    /// Convert complex format to half-complex format.
    ///
    /// # Arguments
    /// * `complex` - Input in complex format (length n/2+1)
    /// * `halfcomplex` - Output in half-complex format (length n)
    pub fn execute(&self, complex: &[Complex<T>], halfcomplex: &mut [T]) {
        assert_eq!(
            complex.len(),
            self.input_len(),
            "Complex input must have length n/2+1"
        );
        assert_eq!(
            halfcomplex.len(),
            self.n,
            "Half-complex output must have length n"
        );

        if self.n == 0 {
            return;
        }

        // DC component
        halfcomplex[0] = complex[0].re;

        if self.n == 1 {
            return;
        }

        // Middle components
        let num_pairs = (self.n - 1) / 2;
        for k in 1..=num_pairs {
            let re_idx = 2 * k - 1;
            let im_idx = 2 * k;
            halfcomplex[re_idx] = complex[k].re;
            halfcomplex[im_idx] = complex[k].im;
        }

        // Nyquist component for even N
        if self.n.is_multiple_of(2) {
            halfcomplex[self.n - 1] = complex[self.n / 2].re;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
        approx_eq(a.re, b.re, eps) && approx_eq(a.im, b.im, eps)
    }

    #[test]
    fn test_hc2c_size_4_even() {
        // For n=4: half-complex is [r0, r1, i1, r2]
        // complex output should be [c0, c1, c2] where c0.im=0, c2.im=0
        let solver = Hc2cSolver::<f64>::new(4);
        assert_eq!(solver.output_len(), 3);

        let hc = [1.0, 2.0, 3.0, 4.0]; // r0=1, r1=2, i1=3, r2=4
        let mut complex = vec![Complex::zero(); 3];

        solver.execute(&hc, &mut complex);

        assert!(complex_approx_eq(complex[0], Complex::new(1.0, 0.0), 1e-10));
        assert!(complex_approx_eq(complex[1], Complex::new(2.0, 3.0), 1e-10));
        assert!(complex_approx_eq(complex[2], Complex::new(4.0, 0.0), 1e-10));
    }

    #[test]
    fn test_hc2c_size_5_odd() {
        // For n=5: half-complex is [r0, r1, i1, r2, i2]
        // complex output should be [c0, c1, c2] where c0.im=0
        let solver = Hc2cSolver::<f64>::new(5);
        assert_eq!(solver.output_len(), 3);

        let hc = [1.0, 2.0, 3.0, 4.0, 5.0]; // r0=1, r1=2, i1=3, r2=4, i2=5
        let mut complex = vec![Complex::zero(); 3];

        solver.execute(&hc, &mut complex);

        assert!(complex_approx_eq(complex[0], Complex::new(1.0, 0.0), 1e-10));
        assert!(complex_approx_eq(complex[1], Complex::new(2.0, 3.0), 1e-10));
        assert!(complex_approx_eq(complex[2], Complex::new(4.0, 5.0), 1e-10));
    }

    #[test]
    fn test_hc2c_c2hc_roundtrip() {
        let n = 8;
        let hc2c = Hc2cSolver::<f64>::new(n);
        let c2hc = C2hcSolver::<f64>::new(n);

        let original: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let mut complex = vec![Complex::zero(); hc2c.output_len()];
        let mut recovered = vec![0.0_f64; n];

        hc2c.execute(&original, &mut complex);
        c2hc.execute(&complex, &mut recovered);

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_c2hc_size_4() {
        let solver = C2hcSolver::<f64>::new(4);
        assert_eq!(solver.input_len(), 3);

        let complex = [
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 3.0),
            Complex::new(4.0, 0.0),
        ];
        let mut hc = vec![0.0_f64; 4];

        solver.execute(&complex, &mut hc);

        assert!(approx_eq(hc[0], 1.0, 1e-10)); // r0
        assert!(approx_eq(hc[1], 2.0, 1e-10)); // r1
        assert!(approx_eq(hc[2], 3.0, 1e-10)); // i1
        assert!(approx_eq(hc[3], 4.0, 1e-10)); // r2
    }
}
