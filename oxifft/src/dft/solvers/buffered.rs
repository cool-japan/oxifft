//! Buffered solver for cache-optimized execution.
//!
//! The buffered solver improves cache locality by:
//! 1. Copying non-contiguous input to a contiguous buffer
//! 2. Executing the FFT on the buffer
//! 3. Copying the result back to the output
//!
//! This is particularly effective when:
//! - Input/output have non-unit strides (e.g., column access in row-major matrix)
//! - The FFT size fits in L1/L2 cache but would otherwise thrash due to stride

use crate::dft::problem::Sign;
use crate::kernel::{Complex, Float};
use crate::prelude::*;

/// Default buffer alignment in bytes (typically cache line size).
const BUFFER_ALIGN: usize = 64;

/// Buffered solver for improved cache locality.
///
/// Wraps another FFT solver and adds buffering for strided access patterns.
pub struct BufferedSolver<T: Float> {
    /// Transform size
    n: usize,
    /// Input stride
    input_stride: isize,
    /// Output stride
    output_stride: isize,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> Default for BufferedSolver<T> {
    fn default() -> Self {
        Self::new_contiguous(1)
    }
}

impl<T: Float> BufferedSolver<T> {
    /// Create a new buffered solver for contiguous data.
    #[must_use]
    pub fn new_contiguous(n: usize) -> Self {
        Self {
            n,
            input_stride: 1,
            output_stride: 1,
            _marker: core::marker::PhantomData,
        }
    }

    /// Create a new buffered solver with custom strides.
    #[must_use]
    pub fn new(n: usize, input_stride: isize, output_stride: isize) -> Self {
        Self {
            n,
            input_stride,
            output_stride,
            _marker: core::marker::PhantomData,
        }
    }

    /// Solver name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        "dft-buffered"
    }

    /// Get the transform size.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Check if buffering is needed (non-unit strides).
    #[must_use]
    pub fn needs_buffering(&self) -> bool {
        self.input_stride != 1 || self.output_stride != 1
    }

    /// Execute the FFT with buffering for non-contiguous access.
    ///
    /// Uses a contiguous buffer to improve cache locality when strides are non-unit.
    pub fn execute<F>(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign, fft_fn: F)
    where
        F: FnOnce(&[Complex<T>], &mut [Complex<T>], Sign),
    {
        if self.n == 0 {
            return;
        }

        // Fast path: contiguous data, no buffering needed
        if self.input_stride == 1 && self.output_stride == 1 {
            fft_fn(input, output, sign);
            return;
        }

        // Allocate aligned buffers for cache efficiency
        let mut in_buf = aligned_buffer(self.n);
        let mut out_buf = aligned_buffer(self.n);

        // Gather input with stride
        for i in 0..self.n {
            let idx = (i as isize * self.input_stride) as usize;
            in_buf[i] = input[idx];
        }

        // Execute FFT on contiguous buffer
        fft_fn(&in_buf, &mut out_buf, sign);

        // Scatter output with stride
        for i in 0..self.n {
            let idx = (i as isize * self.output_stride) as usize;
            output[idx] = out_buf[i];
        }
    }

    /// Execute FFT in-place with buffering.
    pub fn execute_inplace<F>(&self, data: &mut [Complex<T>], sign: Sign, fft_fn: F)
    where
        F: FnOnce(&mut [Complex<T>], Sign),
    {
        if self.n == 0 {
            return;
        }

        // Fast path: contiguous data
        if self.input_stride == 1 && self.output_stride == 1 {
            fft_fn(data, sign);
            return;
        }

        // Gather to buffer
        let mut buf = aligned_buffer(self.n);
        for i in 0..self.n {
            let idx = (i as isize * self.input_stride) as usize;
            buf[i] = data[idx];
        }

        // Execute in-place on buffer
        fft_fn(&mut buf, sign);

        // Scatter back
        for i in 0..self.n {
            let idx = (i as isize * self.output_stride) as usize;
            data[idx] = buf[i];
        }
    }

    /// Execute using Cooley-Tukey for power-of-2 sizes.
    pub fn execute_ct(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        use super::{CooleyTukeySolver, CtVariant};

        if !CooleyTukeySolver::<T>::applicable(self.n) {
            panic!("BufferedSolver::execute_ct requires power-of-2 size");
        }

        let solver = CooleyTukeySolver::new(CtVariant::Dit);
        self.execute(input, output, sign, |i, o, s| solver.execute(i, o, s));
    }

    /// Execute in-place using Cooley-Tukey.
    pub fn execute_ct_inplace(&self, data: &mut [Complex<T>], sign: Sign) {
        use super::{CooleyTukeySolver, CtVariant};

        if !CooleyTukeySolver::<T>::applicable(self.n) {
            panic!("BufferedSolver::execute_ct_inplace requires power-of-2 size");
        }

        let solver = CooleyTukeySolver::new(CtVariant::Dit);
        self.execute_inplace(data, sign, |d, s| solver.execute_inplace(d, s));
    }
}

/// Allocate a cache-aligned buffer.
fn aligned_buffer<T: Float>(n: usize) -> Vec<Complex<T>> {
    // For now, just use a regular Vec
    // A more sophisticated implementation would use aligned allocation
    let _ = BUFFER_ALIGN; // Suppress unused warning
    vec![Complex::zero(); n]
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
    fn test_buffered_contiguous() {
        let n = 8;
        let solver = BufferedSolver::<f64>::new_contiguous(n);
        assert!(!solver.needs_buffering());

        let input: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut output = vec![Complex::zero(); n];

        solver.execute_ct(&input, &mut output, Sign::Forward);

        // DC should be sum
        assert!(complex_approx_eq(output[0], Complex::new(28.0, 0.0), 1e-10));
    }

    #[test]
    fn test_buffered_strided() {
        let n = 4;
        // Data with stride 2: access elements at 0, 2, 4, 6
        let solver = BufferedSolver::<f64>::new(n, 2, 2);
        assert!(solver.needs_buffering());

        // Interleaved data: [0, _, 1, _, 2, _, 3, _]
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

        solver.execute_ct(&input, &mut output, Sign::Forward);

        // DC at index 0 should be sum: 0+1+2+3 = 6
        assert!(complex_approx_eq(output[0], Complex::new(6.0, 0.0), 1e-10));
        // Index 2 should have next frequency bin
        // (output is strided too)
    }

    #[test]
    fn test_buffered_roundtrip() {
        let n = 8;
        let solver = BufferedSolver::<f64>::new_contiguous(n);

        let original: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut transformed = vec![Complex::zero(); n];
        let mut recovered = vec![Complex::zero(); n];

        solver.execute_ct(&original, &mut transformed, Sign::Forward);
        solver.execute_ct(&transformed, &mut recovered, Sign::Backward);

        // Normalize and compare
        let scale = n as f64;
        for (a, b) in original.iter().zip(recovered.iter()) {
            let normalized = Complex::new(b.re / scale, b.im / scale);
            assert!(complex_approx_eq(*a, normalized, 1e-10));
        }
    }

    #[test]
    fn test_buffered_inplace() {
        let n = 8;
        let solver = BufferedSolver::<f64>::new_contiguous(n);

        let input: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(i as f64, 0.0)).collect();

        // Out-of-place
        let mut out_of_place = vec![Complex::zero(); n];
        solver.execute_ct(&input, &mut out_of_place, Sign::Forward);

        // In-place
        let mut in_place = input;
        solver.execute_ct_inplace(&mut in_place, Sign::Forward);

        for (a, b) in out_of_place.iter().zip(in_place.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_buffered_generic_callback() {
        let n = 8;
        let solver = BufferedSolver::<f64>::new(n, 2, 2);

        // Custom FFT using Cooley-Tukey
        let ct_solver = CooleyTukeySolver::<f64>::new(CtVariant::Dit);

        let input: Vec<Complex<f64>> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(7.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let mut output = vec![Complex::zero(); 16];

        solver.execute(&input, &mut output, Sign::Forward, |i, o, s| {
            ct_solver.execute(i, o, s);
        });

        // DC at strided position 0
        assert!(complex_approx_eq(output[0], Complex::new(28.0, 0.0), 1e-10));
    }
}
