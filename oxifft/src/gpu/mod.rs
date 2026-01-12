//! GPU-accelerated FFT support.
//!
//! This module provides GPU backends for high-performance FFT computation.
//!
//! # Supported Backends
//!
//! | Backend | Platform | Feature Flag | Library |
//! |---------|----------|--------------|---------|
//! | CUDA | NVIDIA GPUs | `cuda` | cuFFT |
//! | Metal | Apple GPUs | `metal` | Metal Performance Shaders |
//!
//! # Example
//!
//! ```ignore
//! use oxifft::gpu::{GpuFft, GpuBackend};
//! use oxifft::Complex;
//!
//! // Auto-detect best available backend
//! let gpu = GpuFft::new(1024, GpuBackend::Auto)?;
//!
//! let input: Vec<Complex<f32>> = vec![Complex::new(1.0, 0.0); 1024];
//! let output = gpu.forward(&input)?;
//! ```
//!
//! # Performance Notes
//!
//! GPU FFT is most beneficial for:
//! - Large transforms (N > 4096)
//! - Batched transforms
//! - Single-precision (f32) data
//!
//! For small transforms, CPU FFT is often faster due to GPU overhead.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

mod backend;
mod buffer;
mod error;
mod plan;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "metal")]
pub mod metal;

pub use backend::{GpuBackend, GpuCapabilities};
pub use buffer::GpuBuffer;
pub use error::{GpuError, GpuResult};
pub use plan::{GpuDirection, GpuFft, GpuPlan};

use crate::kernel::{Complex, Float};

/// Trait for GPU FFT implementations.
pub trait GpuFftEngine<T: Float>: Send + Sync {
    /// Execute forward FFT.
    fn forward(&self, input: &[Complex<T>], output: &mut [Complex<T>]) -> GpuResult<()>;

    /// Execute inverse FFT.
    fn inverse(&self, input: &[Complex<T>], output: &mut [Complex<T>]) -> GpuResult<()>;

    /// Execute forward FFT in-place.
    fn forward_inplace(&self, data: &mut [Complex<T>]) -> GpuResult<()>;

    /// Execute inverse FFT in-place.
    fn inverse_inplace(&self, data: &mut [Complex<T>]) -> GpuResult<()>;

    /// Get the transform size.
    fn size(&self) -> usize;

    /// Get the backend type.
    fn backend(&self) -> GpuBackend;

    /// Synchronize GPU operations.
    fn sync(&self) -> GpuResult<()>;
}

/// Trait for batched GPU FFT.
pub trait GpuBatchFft<T: Float>: GpuFftEngine<T> {
    /// Execute batched forward FFT.
    fn forward_batch(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        batch_size: usize,
    ) -> GpuResult<()>;

    /// Execute batched inverse FFT.
    fn inverse_batch(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        batch_size: usize,
    ) -> GpuResult<()>;
}

/// Check if any GPU backend is available.
#[must_use]
pub fn is_gpu_available() -> bool {
    #[cfg(feature = "cuda")]
    if cuda::is_available() {
        return true;
    }

    #[cfg(feature = "metal")]
    if metal::is_available() {
        return true;
    }

    false
}

/// Get the best available GPU backend.
#[must_use]
pub fn best_backend() -> Option<GpuBackend> {
    #[cfg(feature = "cuda")]
    if cuda::is_available() {
        return Some(GpuBackend::Cuda);
    }

    #[cfg(feature = "metal")]
    if metal::is_available() {
        return Some(GpuBackend::Metal);
    }

    None
}

/// Query GPU capabilities.
pub fn query_capabilities() -> GpuResult<GpuCapabilities> {
    #[cfg(feature = "cuda")]
    if cuda::is_available() {
        return cuda::query_capabilities();
    }

    #[cfg(feature = "metal")]
    if metal::is_available() {
        return metal::query_capabilities();
    }

    Err(GpuError::NoBackendAvailable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_gpu_available() {
        // This test just verifies the function doesn't panic
        let _ = is_gpu_available();
    }

    #[test]
    fn test_best_backend() {
        // This test just verifies the function doesn't panic
        let _ = best_backend();
    }
}
