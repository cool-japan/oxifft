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

pub mod batch;
pub mod pool;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "metal")]
pub mod metal;

pub use backend::{GpuBackend, GpuCapabilities};
pub use batch::GpuBatchFft;
pub use buffer::GpuBuffer;
pub use error::{GpuError, GpuResult};
pub use plan::{GpuDirection, GpuFft, GpuPlan};
pub use pool::{BufferKind, GpuBufferPool, PoolKey};

/// Return a reference to the process-global GPU buffer pool.
///
/// The pool is lazily initialised on first call with a default budget of 256 MiB.
/// See [`pool::global_pool`] for full documentation.
#[must_use]
pub fn global_gpu_pool() -> &'static GpuBufferPool {
    pool::global_pool()
}

/// Clear all cached buffers in the global GPU buffer pool.
///
/// All GPU allocations held by the pool are freed.  In-flight buffers (those
/// currently acquired but not yet released) are not affected.
///
/// This is primarily useful in tests to force reallocation.
pub fn clear_gpu_buffer_pool() {
    pool::global_pool().clear();
}

use crate::kernel::{Complex, Float};

/// Trait for GPU FFT implementations.
pub trait GpuFftEngine<T: Float>: Send + Sync {
    /// Execute forward FFT.
    ///
    /// # Errors
    ///
    /// Returns a `GpuError` if sizes are mismatched or the GPU operation fails.
    fn forward(&self, input: &[Complex<T>], output: &mut [Complex<T>]) -> GpuResult<()>;

    /// Execute inverse FFT.
    ///
    /// # Errors
    ///
    /// Returns a `GpuError` if sizes are mismatched or the GPU operation fails.
    fn inverse(&self, input: &[Complex<T>], output: &mut [Complex<T>]) -> GpuResult<()>;

    /// Execute forward FFT in-place.
    ///
    /// # Errors
    ///
    /// Returns a `GpuError` if sizes are mismatched or the GPU operation fails.
    fn forward_inplace(&self, data: &mut [Complex<T>]) -> GpuResult<()>;

    /// Execute inverse FFT in-place.
    ///
    /// # Errors
    ///
    /// Returns a `GpuError` if sizes are mismatched or the GPU operation fails.
    fn inverse_inplace(&self, data: &mut [Complex<T>]) -> GpuResult<()>;

    /// Get the transform size.
    fn size(&self) -> usize;

    /// Get the backend type.
    fn backend(&self) -> GpuBackend;

    /// Synchronize GPU operations.
    ///
    /// # Errors
    ///
    /// Returns a `GpuError` if GPU synchronisation fails.
    fn sync(&self) -> GpuResult<()>;
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
///
/// # Errors
///
/// Returns `GpuError::NoBackendAvailable` if no GPU backend (CUDA or Metal)
/// is present, or propagates backend-specific capability query errors.
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
