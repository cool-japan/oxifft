//! GPU error types.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::string::String;

use core::fmt;

/// GPU operation result type.
pub type GpuResult<T> = Result<T, GpuError>;

/// GPU error types.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum GpuError {
    /// No GPU backend is available.
    NoBackendAvailable,
    /// Failed to initialize GPU.
    InitializationFailed(String),
    /// Failed to allocate GPU memory.
    AllocationFailed(String),
    /// Failed to transfer data to/from GPU.
    TransferFailed(String),
    /// FFT execution failed.
    ExecutionFailed(String),
    /// Invalid transform size.
    InvalidSize(usize),
    /// Size mismatch between input and output.
    SizeMismatch { expected: usize, got: usize },
    /// Backend-specific error.
    BackendError(String),
    /// Feature not supported.
    Unsupported(String),
    /// Synchronization failed.
    SyncFailed(String),
    /// Metal device was lost or reset (device removed / GPU preemption).
    DeviceLost,
    /// GPU buffer allocation failed due to insufficient device memory.
    OutOfMemory {
        /// Number of bytes that were requested when the allocation failed.
        requested_bytes: usize,
    },
    /// MSL or CUDA kernel source failed to compile.
    ShaderCompileFailed(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoBackendAvailable => write!(f, "No GPU backend available"),
            Self::InitializationFailed(msg) => write!(f, "GPU initialization failed: {msg}"),
            Self::AllocationFailed(msg) => write!(f, "GPU memory allocation failed: {msg}"),
            Self::TransferFailed(msg) => write!(f, "GPU data transfer failed: {msg}"),
            Self::ExecutionFailed(msg) => write!(f, "GPU FFT execution failed: {msg}"),
            Self::InvalidSize(size) => write!(f, "Invalid FFT size: {size}"),
            Self::SizeMismatch { expected, got } => {
                write!(f, "Size mismatch: expected {expected}, got {got}")
            }
            Self::BackendError(msg) => write!(f, "GPU backend error: {msg}"),
            Self::Unsupported(msg) => write!(f, "Unsupported operation: {msg}"),
            Self::SyncFailed(msg) => write!(f, "GPU synchronization failed: {msg}"),
            Self::DeviceLost => write!(f, "GPU device lost or reset"),
            Self::OutOfMemory { requested_bytes } => {
                write!(f, "GPU out of memory: requested {requested_bytes} bytes")
            }
            Self::ShaderCompileFailed(msg) => write!(f, "Shader compilation failed: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for GpuError {}

// ---------------------------------------------------------------------------
// From conversions — Metal backend
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
impl From<oxicuda_metal::error::MetalError> for GpuError {
    fn from(e: oxicuda_metal::error::MetalError) -> Self {
        use oxicuda_metal::error::MetalError;
        match e {
            MetalError::NoDevice => Self::DeviceLost,
            MetalError::UnsupportedPlatform => Self::Unsupported("Metal requires macOS".into()),
            MetalError::OutOfMemory => Self::OutOfMemory { requested_bytes: 0 },
            MetalError::ShaderCompilation(msg) => Self::ShaderCompileFailed(msg),
            MetalError::PipelineCreation(msg) => Self::ShaderCompileFailed(msg),
            MetalError::NotInitialized => Self::InitializationFailed("not initialized".into()),
            MetalError::Unsupported(msg) => Self::Unsupported(msg),
            MetalError::InvalidArgument(msg) => Self::BackendError(msg),
            MetalError::CommandBufferError(msg) => Self::ExecutionFailed(msg),
        }
    }
}
