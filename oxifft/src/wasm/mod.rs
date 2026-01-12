//! WebAssembly bindings for OxiFFT.
//!
//! This module provides JavaScript bindings for FFT operations in the browser
//! via WebAssembly.
//!
//! # Usage from JavaScript
//!
//! ```javascript
//! import init, { WasmFft, fft_f64, ifft_f64 } from 'oxifft';
//!
//! await init();
//!
//! // Create a plan for repeated use
//! const fft = new WasmFft(256);
//!
//! // Execute FFT
//! const real = new Float64Array([1, 2, 3, ...]);
//! const imag = new Float64Array([0, 0, 0, ...]);
//! const result = fft.forward(real, imag);
//!
//! // Result is interleaved [re0, im0, re1, im1, ...]
//! console.log(result);
//!
//! // One-shot API
//! const output = fft_f64(real, imag);
//! ```
//!
//! # Features
//!
//! - Plan-based API for repeated transforms
//! - One-shot API for single transforms
//! - TypedArray support for zero-copy data exchange
//! - WASM SIMD acceleration (when available)

mod bindings;
mod simd;

pub use bindings::*;
pub use simd::*;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
