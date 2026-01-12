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
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for GpuError {}
