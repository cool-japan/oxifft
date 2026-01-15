//! Indirect solver for non-contiguous strides.
//!
//! The indirect solver handles FFTs on non-contiguous data by:
//! 1. Gathering scattered elements into a contiguous buffer
//! 2. Executing the FFT using a sub-solver on the contiguous buffer
//! 3. Scattering results back to non-contiguous output locations
//!
//! This is essential for multi-dimensional FFTs where we need to transform
//! along non-contiguous dimensions (e.g., columns in row-major storage).
//!
//! Unlike BufferedSolver which assumes uniform input/output strides, IndirectSolver
//! supports arbitrary stride patterns via index arrays or stride-based addressing.

use crate::dft::problem::Sign;
use crate::kernel::{Complex, Float};
use crate::prelude::*;

/// Strategy for accessing non-contiguous data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // IndexArray is planned for future use
pub enum IndirectStrategy {
    /// Use explicit stride for regular patterns.
    Stride,
    /// Use index array for irregular patterns (not yet implemented).
    IndexArray,
}

/// Indirect solver for non-contiguous memory layouts.
///
/// Wraps a contiguous FFT solver and handles gather/scatter for strided data.
/// This is used internally by multi-dimensional and batched transforms.
pub struct IndirectSolver<T: Float> {
    /// Transform size.
    n: usize,
    /// Input stride (elements between consecutive input values).
    input_stride: isize,
    /// Output stride (elements between consecutive output values).
    output_stride: isize,
    /// Access strategy.
    strategy: IndirectStrategy,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> Default for IndirectSolver<T> {
    fn default() -> Self {
        Self::new_contiguous(1)
    }
}

impl<T: Float> IndirectSolver<T> {
    /// Create a new indirect solver for contiguous data.
    #[must_use]
    pub fn new_contiguous(n: usize) -> Self {
        Self {
            n,
            input_stride: 1,
            output_stride: 1,
            strategy: IndirectStrategy::Stride,
            _marker: core::marker::PhantomData,
        }
    }

    /// Create a new indirect solver with uniform stride for both input and output.
    #[must_use]
    pub fn new_uniform(n: usize, stride: isize) -> Self {
        Self {
            n,
            input_stride: stride,
            output_stride: stride,
            strategy: IndirectStrategy::Stride,
            _marker: core::marker::PhantomData,
        }
    }

    /// Create a new indirect solver with different input and output strides.
    #[must_use]
    pub fn new(n: usize, input_stride: isize, output_stride: isize) -> Self {
        Self {
            n,
            input_stride,
            output_stride,
            strategy: IndirectStrategy::Stride,
            _marker: core::marker::PhantomData,
        }
    }

    /// Solver name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        "dft-indirect"
    }

    /// Get the transform size.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get input stride.
    #[must_use]
    pub fn input_stride(&self) -> isize {
        self.input_stride
    }

    /// Get output stride.
    #[must_use]
    pub fn output_stride(&self) -> isize {
        self.output_stride
    }

    /// Check if data is contiguous (no indirection needed).
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        self.input_stride == 1 && self.output_stride == 1
    }

    /// Check if input and output use the same stride.
    #[must_use]
    pub fn has_uniform_stride(&self) -> bool {
        self.input_stride == self.output_stride
    }

    /// Execute FFT with gather/scatter for non-contiguous data.
    ///
    /// The `fft_fn` closure receives contiguous input/output buffers and performs
    /// the actual FFT computation.
    ///
    /// # Parameters
    /// - `input`: Input data (accessed at strided positions from `input_base`)
    /// - `input_base`: Starting index in input array
    /// - `output`: Output data (written at strided positions from `output_base`)
    /// - `output_base`: Starting index in output array
    /// - `sign`: FFT direction (Forward or Backward)
    /// - `fft_fn`: Function that performs FFT on contiguous buffers
    pub fn execute<F>(
        &self,
        input: &[Complex<T>],
        input_base: usize,
        output: &mut [Complex<T>],
        output_base: usize,
        sign: Sign,
        fft_fn: F,
    ) where
        F: FnOnce(&[Complex<T>], &mut [Complex<T>], Sign),
    {
        if self.n == 0 {
            return;
        }

        // Fast path: contiguous data with no offset
        if self.is_contiguous() && input_base == 0 && output_base == 0 {
            fft_fn(&input[..self.n], &mut output[..self.n], sign);
            return;
        }

        // Gather input into contiguous buffer
        let mut in_buf = vec![Complex::zero(); self.n];
        self.gather(input, input_base, &mut in_buf);

        // Execute FFT on contiguous buffer
        let mut out_buf = vec![Complex::zero(); self.n];
        fft_fn(&in_buf, &mut out_buf, sign);

        // Scatter output to non-contiguous locations
        self.scatter(&out_buf, output, output_base);
    }

    /// Execute FFT in-place with gather/scatter for non-contiguous data.
    pub fn execute_inplace<F>(&self, data: &mut [Complex<T>], base: usize, sign: Sign, fft_fn: F)
    where
        F: FnOnce(&mut [Complex<T>], Sign),
    {
        if self.n == 0 {
            return;
        }

        // Fast path: contiguous data with no offset
        if self.is_contiguous() && base == 0 {
            fft_fn(&mut data[..self.n], sign);
            return;
        }

        // Gather into buffer
        let mut buf = vec![Complex::zero(); self.n];
        self.gather(data, base, &mut buf);

        // In-place FFT on buffer
        fft_fn(&mut buf, sign);

        // Scatter back (using output stride since it's in-place)
        self.scatter(&buf, data, base);
    }

    /// Simplified execute for cases where input_base and output_base are 0.
    pub fn execute_simple<F>(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        sign: Sign,
        fft_fn: F,
    ) where
        F: FnOnce(&[Complex<T>], &mut [Complex<T>], Sign),
    {
        self.execute(input, 0, output, 0, sign, fft_fn);
    }

    /// Gather elements from strided source into contiguous destination.
    fn gather(&self, src: &[Complex<T>], src_base: usize, dst: &mut [Complex<T>]) {
        match self.strategy {
            IndirectStrategy::Stride => {
                for i in 0..self.n {
                    let src_idx = (src_base as isize + i as isize * self.input_stride) as usize;
                    dst[i] = src[src_idx];
                }
            }
            IndirectStrategy::IndexArray => {
                // Future: use explicit index array for irregular patterns
                unimplemented!("Index array strategy not yet implemented")
            }
        }
    }

    /// Scatter elements from contiguous source to strided destination.
    fn scatter(&self, src: &[Complex<T>], dst: &mut [Complex<T>], dst_base: usize) {
        match self.strategy {
            IndirectStrategy::Stride => {
                for i in 0..self.n {
                    let dst_idx = (dst_base as isize + i as isize * self.output_stride) as usize;
                    dst[dst_idx] = src[i];
                }
            }
            IndirectStrategy::IndexArray => {
                unimplemented!("Index array strategy not yet implemented")
            }
        }
    }

    /// Execute using Cooley-Tukey for power-of-2 sizes.
    pub fn execute_ct(
        &self,
        input: &[Complex<T>],
        input_base: usize,
        output: &mut [Complex<T>],
        output_base: usize,
        sign: Sign,
    ) {
        use super::{CooleyTukeySolver, CtVariant};

        if !CooleyTukeySolver::<T>::applicable(self.n) {
            panic!("IndirectSolver::execute_ct requires power-of-2 size");
        }

        let solver = CooleyTukeySolver::new(CtVariant::Dit);
        self.execute(input, input_base, output, output_base, sign, |i, o, s| {
            solver.execute(i, o, s);
        });
    }

    /// Execute using Bluestein for arbitrary sizes.
    pub fn execute_bluestein(
        &self,
        input: &[Complex<T>],
        input_base: usize,
        output: &mut [Complex<T>],
        output_base: usize,
        sign: Sign,
    ) {
        use super::BluesteinSolver;

        let solver = BluesteinSolver::new(self.n);
        self.execute(input, input_base, output, output_base, sign, |i, o, s| {
            solver.execute(i, o, s);
        });
    }

    /// Execute using the best available solver for the size.
    pub fn execute_auto(
        &self,
        input: &[Complex<T>],
        input_base: usize,
        output: &mut [Complex<T>],
        output_base: usize,
        sign: Sign,
    ) {
        use super::{CooleyTukeySolver, GenericSolver, RaderSolver};
        use crate::kernel::is_prime;

        let n = self.n;

        // Choose appropriate solver
        if n <= 1 {
            // Handle trivial cases
            if n == 1 {
                let in_idx = input_base;
                let out_idx = output_base;
                output[out_idx] = input[in_idx];
            }
            return;
        }

        if CooleyTukeySolver::<T>::applicable(n) {
            self.execute_ct(input, input_base, output, output_base, sign);
        } else if is_prime(n) && n <= 1021 {
            if let Some(solver) = RaderSolver::new(n) {
                self.execute(input, input_base, output, output_base, sign, |i, o, s| {
                    solver.execute(i, o, s);
                });
            } else {
                // Fall back to Bluestein if Rader fails
                self.execute_bluestein(input, input_base, output, output_base, sign);
            }
        } else if is_smooth(n, 7) {
            let solver = GenericSolver::new(n);
            self.execute(input, input_base, output, output_base, sign, |i, o, s| {
                solver.execute(i, o, s);
            });
        } else {
            self.execute_bluestein(input, input_base, output, output_base, sign);
        }
    }
}

/// Check if n is B-smooth (all prime factors <= B).
fn is_smooth(n: usize, b: usize) -> bool {
    if n <= 1 {
        return true;
    }

    let mut m = n;
    for p in 2..=b {
        while m.is_multiple_of(p) {
            m /= p;
        }
        if m == 1 {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::super::{CooleyTukeySolver, CtVariant};
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
        approx_eq(a.re, b.re, eps) && approx_eq(a.im, b.im, eps)
    }

    #[test]
    fn test_indirect_contiguous() {
        let n = 8;
        let solver = IndirectSolver::<f64>::new_contiguous(n);
        assert!(solver.is_contiguous());

        let input: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut output = vec![Complex::zero(); n];

        let ct_solver = CooleyTukeySolver::new(CtVariant::Dit);
        solver.execute_simple(&input, &mut output, Sign::Forward, |i, o, s| {
            ct_solver.execute(i, o, s);
        });

        // DC should be sum: 0+1+2+3+4+5+6+7 = 28
        assert!(complex_approx_eq(output[0], Complex::new(28.0, 0.0), 1e-10));
    }

    #[test]
    fn test_indirect_uniform_stride() {
        // Test extracting every other element (stride 2)
        let n = 4;
        let solver = IndirectSolver::<f64>::new_uniform(n, 2);
        assert!(solver.has_uniform_stride());

        // Interleaved data: [0, X, 1, X, 2, X, 3, X]
        let input: Vec<Complex<f64>> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(99.0, 99.0),
            Complex::new(1.0, 0.0),
            Complex::new(99.0, 99.0),
            Complex::new(2.0, 0.0),
            Complex::new(99.0, 99.0),
            Complex::new(3.0, 0.0),
            Complex::new(99.0, 99.0),
        ];
        let mut output = vec![Complex::zero(); 8];

        solver.execute_ct(&input, 0, &mut output, 0, Sign::Forward);

        // DC at index 0 should be sum: 0+1+2+3 = 6
        assert!(complex_approx_eq(output[0], Complex::new(6.0, 0.0), 1e-10));
    }

    #[test]
    fn test_indirect_with_offset() {
        let n = 4;
        let solver = IndirectSolver::<f64>::new_contiguous(n);

        // Data with offset: [X, X, 1, 2, 3, 4]
        let input: Vec<Complex<f64>> = vec![
            Complex::new(99.0, 99.0),
            Complex::new(99.0, 99.0),
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut output = vec![Complex::zero(); 6];

        let ct_solver = CooleyTukeySolver::new(CtVariant::Dit);
        solver.execute(&input, 2, &mut output, 2, Sign::Forward, |i, o, s| {
            ct_solver.execute(i, o, s);
        });

        // DC at output[2] should be sum: 1+2+3+4 = 10
        assert!(complex_approx_eq(output[2], Complex::new(10.0, 0.0), 1e-10));
    }

    #[test]
    fn test_indirect_roundtrip() {
        let n = 8;
        let solver = IndirectSolver::<f64>::new_uniform(n, 2);

        // Create input at stride-2 positions
        let mut data: Vec<Complex<f64>> = vec![Complex::zero(); 16];
        for i in 0..n {
            data[i * 2] = Complex::new((i as f64).sin(), (i as f64).cos());
        }
        let original: Vec<Complex<f64>> = (0..n).map(|i| data[i * 2]).collect();

        // Forward transform
        let mut transformed = vec![Complex::zero(); 16];
        solver.execute_ct(&data, 0, &mut transformed, 0, Sign::Forward);

        // Inverse transform
        let mut recovered = vec![Complex::zero(); 16];
        solver.execute_ct(&transformed, 0, &mut recovered, 0, Sign::Backward);

        // Verify roundtrip (with normalization)
        let scale = n as f64;
        for i in 0..n {
            let idx = i * 2;
            let normalized = Complex::new(recovered[idx].re / scale, recovered[idx].im / scale);
            assert!(complex_approx_eq(original[i], normalized, 1e-10));
        }
    }

    #[test]
    fn test_indirect_inplace() {
        let n = 4;
        let solver = IndirectSolver::<f64>::new_uniform(n, 2);

        // Interleaved data: [0, X, 1, X, 2, X, 3, X]
        let mut data: Vec<Complex<f64>> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(99.0, 99.0),
            Complex::new(1.0, 0.0),
            Complex::new(99.0, 99.0),
            Complex::new(2.0, 0.0),
            Complex::new(99.0, 99.0),
            Complex::new(3.0, 0.0),
            Complex::new(99.0, 99.0),
        ];

        // Out-of-place result for comparison
        let input = data.clone();
        let mut expected = vec![Complex::zero(); 8];
        solver.execute_ct(&input, 0, &mut expected, 0, Sign::Forward);

        // In-place
        let ct_solver = CooleyTukeySolver::new(CtVariant::Dit);
        solver.execute_inplace(&mut data, 0, Sign::Forward, |d, s| {
            ct_solver.execute_inplace(d, s);
        });

        // Compare strided elements
        for i in 0..n {
            assert!(complex_approx_eq(data[i * 2], expected[i * 2], 1e-10));
        }
    }

    #[test]
    fn test_indirect_different_io_strides() {
        let n = 4;
        // Input at stride 2, output at stride 3
        let solver = IndirectSolver::<f64>::new(n, 2, 3);
        assert!(!solver.has_uniform_stride());

        // Input: [0, X, 1, X, 2, X, 3, X]
        let input: Vec<Complex<f64>> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(99.0, 99.0),
            Complex::new(1.0, 0.0),
            Complex::new(99.0, 99.0),
            Complex::new(2.0, 0.0),
            Complex::new(99.0, 99.0),
            Complex::new(3.0, 0.0),
            Complex::new(99.0, 99.0),
        ];
        let mut output = vec![Complex::zero(); 12];

        solver.execute_ct(&input, 0, &mut output, 0, Sign::Forward);

        // DC at output[0] should be sum: 0+1+2+3 = 6
        assert!(complex_approx_eq(output[0], Complex::new(6.0, 0.0), 1e-10));
        // Results are at stride-3 positions: 0, 3, 6, 9
    }

    #[test]
    fn test_indirect_execute_auto() {
        // Test with different size types
        for &n in &[4, 8, 16, 5, 7, 12, 15] {
            let solver = IndirectSolver::<f64>::new_contiguous(n);

            let input: Vec<Complex<f64>> = (0..n)
                .map(|i| Complex::new(i as f64, (i as f64) * 0.5))
                .collect();
            let mut output = vec![Complex::zero(); n];

            solver.execute_auto(&input, 0, &mut output, 0, Sign::Forward);

            // DC component should be sum of real parts
            let expected_dc: f64 = (0..n).map(|i| i as f64).sum();
            assert!(
                approx_eq(output[0].re, expected_dc, 1e-9),
                "Failed for n={}: expected {}, got {}",
                n,
                expected_dc,
                output[0].re
            );
        }
    }
}
