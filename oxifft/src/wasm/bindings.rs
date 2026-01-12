//! wasm-bindgen bindings for JavaScript interop.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::api::{Direction, Flags, Plan};
use crate::kernel::Complex;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WebAssembly FFT wrapper for JavaScript.
///
/// Provides a plan-based interface for repeated FFT operations.
///
/// # Example (JavaScript)
///
/// ```javascript
/// import { WasmFft } from 'oxifft';
///
/// const fft = new WasmFft(256);
/// const real = new Float64Array([1, 2, 3, ...]);
/// const imag = new Float64Array([0, 0, 0, ...]);
/// const result = fft.forward(real, imag);
/// ```
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmFft {
    /// Size of the transform.
    size: usize,
    /// Forward FFT plan.
    forward_plan: Option<Plan<f64>>,
    /// Inverse FFT plan.
    inverse_plan: Option<Plan<f64>>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmFft {
    /// Create a new WASM FFT wrapper.
    ///
    /// # Arguments
    ///
    /// * `size` - Transform size
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(size: usize) -> Self {
        let forward_plan = Plan::dft_1d(size, Direction::Forward, Flags::ESTIMATE);
        let inverse_plan = Plan::dft_1d(size, Direction::Backward, Flags::ESTIMATE);

        Self {
            size,
            forward_plan,
            inverse_plan,
        }
    }

    /// Get the transform size.
    #[cfg_attr(feature = "wasm", wasm_bindgen(getter))]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Execute forward FFT.
    ///
    /// # Arguments
    ///
    /// * `real` - Real part of input (length must equal size)
    /// * `imag` - Imaginary part of input (length must equal size)
    ///
    /// # Returns
    ///
    /// Interleaved output: [re0, im0, re1, im1, ...]
    pub fn forward(&self, real: &[f64], imag: &[f64]) -> Vec<f64> {
        if real.len() != self.size || imag.len() != self.size {
            return Vec::new();
        }

        let plan = match &self.forward_plan {
            Some(p) => p,
            None => return Vec::new(),
        };

        // Create complex input
        let input: Vec<Complex<f64>> = real
            .iter()
            .zip(imag.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();

        let mut output = vec![Complex::<f64>::zero(); self.size];
        plan.execute(&input, &mut output);

        // Return interleaved result
        let mut result = Vec::with_capacity(self.size * 2);
        for c in output {
            result.push(c.re);
            result.push(c.im);
        }
        result
    }

    /// Execute inverse FFT.
    ///
    /// # Arguments
    ///
    /// * `real` - Real part of input (length must equal size)
    /// * `imag` - Imaginary part of input (length must equal size)
    ///
    /// # Returns
    ///
    /// Interleaved output: [re0, im0, re1, im1, ...]
    pub fn inverse(&self, real: &[f64], imag: &[f64]) -> Vec<f64> {
        if real.len() != self.size || imag.len() != self.size {
            return Vec::new();
        }

        let plan = match &self.inverse_plan {
            Some(p) => p,
            None => return Vec::new(),
        };

        // Create complex input
        let input: Vec<Complex<f64>> = real
            .iter()
            .zip(imag.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();

        let mut output = vec![Complex::<f64>::zero(); self.size];
        plan.execute(&input, &mut output);

        // Normalize by size for proper inverse
        let scale = 1.0 / (self.size as f64);
        let mut result = Vec::with_capacity(self.size * 2);
        for c in output {
            result.push(c.re * scale);
            result.push(c.im * scale);
        }
        result
    }

    /// Execute forward FFT with interleaved input.
    ///
    /// # Arguments
    ///
    /// * `interleaved` - Interleaved complex input [re0, im0, re1, im1, ...]
    ///
    /// # Returns
    ///
    /// Interleaved output.
    pub fn forward_interleaved(&self, interleaved: &[f64]) -> Vec<f64> {
        if interleaved.len() != self.size * 2 {
            return Vec::new();
        }

        let plan = match &self.forward_plan {
            Some(p) => p,
            None => return Vec::new(),
        };

        // Create complex input from interleaved data
        let input: Vec<Complex<f64>> = interleaved
            .chunks_exact(2)
            .map(|c| Complex::new(c[0], c[1]))
            .collect();

        let mut output = vec![Complex::<f64>::zero(); self.size];
        plan.execute(&input, &mut output);

        // Return interleaved result
        let mut result = Vec::with_capacity(self.size * 2);
        for c in output {
            result.push(c.re);
            result.push(c.im);
        }
        result
    }

    /// Execute inverse FFT with interleaved input.
    ///
    /// # Arguments
    ///
    /// * `interleaved` - Interleaved complex input [re0, im0, re1, im1, ...]
    ///
    /// # Returns
    ///
    /// Interleaved output (normalized).
    pub fn inverse_interleaved(&self, interleaved: &[f64]) -> Vec<f64> {
        if interleaved.len() != self.size * 2 {
            return Vec::new();
        }

        let plan = match &self.inverse_plan {
            Some(p) => p,
            None => return Vec::new(),
        };

        let input: Vec<Complex<f64>> = interleaved
            .chunks_exact(2)
            .map(|c| Complex::new(c[0], c[1]))
            .collect();

        let mut output = vec![Complex::<f64>::zero(); self.size];
        plan.execute(&input, &mut output);

        let scale = 1.0 / (self.size as f64);
        let mut result = Vec::with_capacity(self.size * 2);
        for c in output {
            result.push(c.re * scale);
            result.push(c.im * scale);
        }
        result
    }
}

/// One-shot forward FFT (f64).
///
/// # Arguments
///
/// * `real` - Real part of input
/// * `imag` - Imaginary part of input
///
/// # Returns
///
/// Interleaved output: [re0, im0, re1, im1, ...]
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn fft_f64(real: &[f64], imag: &[f64]) -> Vec<f64> {
    if real.len() != imag.len() || real.is_empty() {
        return Vec::new();
    }

    let n = real.len();
    let plan = match Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let input: Vec<Complex<f64>> = real
        .iter()
        .zip(imag.iter())
        .map(|(&r, &i)| Complex::new(r, i))
        .collect();

    let mut output = vec![Complex::<f64>::zero(); n];
    plan.execute(&input, &mut output);

    let mut result = Vec::with_capacity(n * 2);
    for c in output {
        result.push(c.re);
        result.push(c.im);
    }
    result
}

/// One-shot inverse FFT (f64).
///
/// # Arguments
///
/// * `real` - Real part of input
/// * `imag` - Imaginary part of input
///
/// # Returns
///
/// Interleaved output: [re0, im0, re1, im1, ...] (normalized)
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn ifft_f64(real: &[f64], imag: &[f64]) -> Vec<f64> {
    if real.len() != imag.len() || real.is_empty() {
        return Vec::new();
    }

    let n = real.len();
    let plan = match Plan::dft_1d(n, Direction::Backward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let input: Vec<Complex<f64>> = real
        .iter()
        .zip(imag.iter())
        .map(|(&r, &i)| Complex::new(r, i))
        .collect();

    let mut output = vec![Complex::<f64>::zero(); n];
    plan.execute(&input, &mut output);

    let scale = 1.0 / (n as f64);
    let mut result = Vec::with_capacity(n * 2);
    for c in output {
        result.push(c.re * scale);
        result.push(c.im * scale);
    }
    result
}

/// One-shot forward FFT (f32).
///
/// # Arguments
///
/// * `real` - Real part of input
/// * `imag` - Imaginary part of input
///
/// # Returns
///
/// Interleaved output: [re0, im0, re1, im1, ...]
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn fft_f32(real: &[f32], imag: &[f32]) -> Vec<f32> {
    if real.len() != imag.len() || real.is_empty() {
        return Vec::new();
    }

    let n = real.len();
    let plan = match Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let input: Vec<Complex<f32>> = real
        .iter()
        .zip(imag.iter())
        .map(|(&r, &i)| Complex::new(r, i))
        .collect();

    let mut output = vec![Complex::<f32>::zero(); n];
    plan.execute(&input, &mut output);

    let mut result = Vec::with_capacity(n * 2);
    for c in output {
        result.push(c.re);
        result.push(c.im);
    }
    result
}

/// One-shot inverse FFT (f32).
///
/// # Arguments
///
/// * `real` - Real part of input
/// * `imag` - Imaginary part of input
///
/// # Returns
///
/// Interleaved output: [re0, im0, re1, im1, ...] (normalized)
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn ifft_f32(real: &[f32], imag: &[f32]) -> Vec<f32> {
    if real.len() != imag.len() || real.is_empty() {
        return Vec::new();
    }

    let n = real.len();
    let plan = match Plan::dft_1d(n, Direction::Backward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let input: Vec<Complex<f32>> = real
        .iter()
        .zip(imag.iter())
        .map(|(&r, &i)| Complex::new(r, i))
        .collect();

    let mut output = vec![Complex::<f32>::zero(); n];
    plan.execute(&input, &mut output);

    let scale = 1.0 / (n as f32);
    let mut result = Vec::with_capacity(n * 2);
    for c in output {
        result.push(c.re * scale);
        result.push(c.im * scale);
    }
    result
}

/// Real-to-complex FFT (f64).
///
/// # Arguments
///
/// * `input` - Real input signal
///
/// # Returns
///
/// Interleaved complex output (hermitian symmetric, n/2+1 complex values)
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn rfft_f64(input: &[f64]) -> Vec<f64> {
    if input.is_empty() {
        return Vec::new();
    }

    let n = input.len();

    // Convert to complex
    let complex_input: Vec<Complex<f64>> = input.iter().map(|&r| Complex::new(r, 0.0)).collect();

    let plan = match Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut output = vec![Complex::<f64>::zero(); n];
    plan.execute(&complex_input, &mut output);

    // Return only first n/2+1 due to hermitian symmetry
    let out_len = n / 2 + 1;
    let mut result = Vec::with_capacity(out_len * 2);
    for c in output.iter().take(out_len) {
        result.push(c.re);
        result.push(c.im);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_fft_forward() {
        let fft = WasmFft::new(4);

        let real = [1.0, 1.0, 1.0, 1.0];
        let imag = [0.0, 0.0, 0.0, 0.0];

        let result = fft.forward(&real, &imag);

        // DC component should be 4
        assert!((result[0] - 4.0).abs() < 1e-10);
        assert!(result[1].abs() < 1e-10);

        // Other components should be 0
        assert!(result[2].abs() < 1e-10);
        assert!(result[3].abs() < 1e-10);
    }

    #[test]
    fn test_wasm_fft_inverse() {
        let fft = WasmFft::new(4);

        let real = [1.0, 2.0, 3.0, 4.0];
        let imag = [0.0, 0.0, 0.0, 0.0];

        // Forward then inverse should recover original
        let forward = fft.forward(&real, &imag);

        let fwd_real: Vec<f64> = forward.iter().step_by(2).copied().collect();
        let fwd_imag: Vec<f64> = forward.iter().skip(1).step_by(2).copied().collect();

        let inverse = fft.inverse(&fwd_real, &fwd_imag);

        for i in 0..4 {
            assert!((inverse[i * 2] - real[i]).abs() < 1e-10);
            assert!(inverse[i * 2 + 1].abs() < 1e-10);
        }
    }

    #[test]
    fn test_fft_f64_oneshot() {
        let real = [1.0, 0.0, 0.0, 0.0];
        let imag = [0.0, 0.0, 0.0, 0.0];

        let result = fft_f64(&real, &imag);

        // DFT of impulse is all ones
        for i in 0..4 {
            assert!((result[i * 2] - 1.0).abs() < 1e-10);
            assert!(result[i * 2 + 1].abs() < 1e-10);
        }
    }

    #[test]
    fn test_ifft_f64_oneshot() {
        let real = [4.0, 0.0, 0.0, 0.0];
        let imag = [0.0, 0.0, 0.0, 0.0];

        let result = ifft_f64(&real, &imag);

        // IDFT should recover uniform signal
        for i in 0..4 {
            assert!((result[i * 2] - 1.0).abs() < 1e-10);
            assert!(result[i * 2 + 1].abs() < 1e-10);
        }
    }

    #[test]
    fn test_fft_f32_oneshot() {
        let real = [1.0_f32, 1.0, 1.0, 1.0];
        let imag = [0.0_f32, 0.0, 0.0, 0.0];

        let result = fft_f32(&real, &imag);

        // DC should be 4
        assert!((result[0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_rfft_f64() {
        let input = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = rfft_f64(&input);

        // Impulse -> all ones (first n/2+1 = 5 complex values)
        assert_eq!(result.len(), 10); // 5 complex = 10 floats

        for chunk in result.chunks(2) {
            assert!((chunk[0] - 1.0).abs() < 1e-10);
            assert!(chunk[1].abs() < 1e-10);
        }
    }

    #[test]
    fn test_wasm_fft_interleaved() {
        let fft = WasmFft::new(4);

        // [1+0i, 1+0i, 1+0i, 1+0i]
        let interleaved = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        let result = fft.forward_interleaved(&interleaved);

        // DC component should be 4
        assert!((result[0] - 4.0).abs() < 1e-10);
    }
}
