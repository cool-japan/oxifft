//! Error types for MPI operations.

use core::fmt;

/// Error type for MPI FFT operations.
#[derive(Debug, Clone)]
pub enum MpiError {
    /// Invalid dimension specification.
    InvalidDimension {
        /// The dimension index.
        dim: usize,
        /// The invalid size.
        size: usize,
        /// Description of the error.
        message: String,
    },
    /// Data size mismatch.
    SizeMismatch {
        /// Expected size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },
    /// Communication error.
    CommunicationError {
        /// Description of the error.
        message: String,
    },
    /// Not enough processes for the problem.
    InsufficientProcesses {
        /// Required number of processes.
        required: usize,
        /// Available number of processes.
        available: usize,
    },
    /// Internal FFT error.
    FftError {
        /// Description of the error.
        message: String,
    },
}

impl fmt::Display for MpiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension { dim, size, message } => {
                write!(f, "Invalid dimension {dim} (size {size}): {message}")
            }
            Self::SizeMismatch { expected, actual } => {
                write!(f, "Size mismatch: expected {expected}, got {actual}")
            }
            Self::CommunicationError { message } => {
                write!(f, "MPI communication error: {message}")
            }
            Self::InsufficientProcesses {
                required,
                available,
            } => {
                write!(
                    f,
                    "Insufficient processes: need {required}, have {available}"
                )
            }
            Self::FftError { message } => {
                write!(f, "FFT error: {message}")
            }
        }
    }
}

impl std::error::Error for MpiError {}
