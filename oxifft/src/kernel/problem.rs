//! Problem abstraction for FFT transforms.
//!
//! A problem represents an FFT operation to be planned and executed.

use core::fmt::Debug;
use core::hash::Hash;

/// Problem kind identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProblemKind {
    /// Complex DFT (Discrete Fourier Transform)
    Dft,
    /// Real DFT (Real-to-Complex or Complex-to-Real)
    Rdft,
    /// Real Even/Odd DFT (DCT/DST)
    Reodft,
}

/// Base trait for all FFT problems.
///
/// A problem encapsulates:
/// - The transform dimensions and strides
/// - Input/output buffer pointers
/// - Transform parameters (direction, kind)
pub trait Problem: Hash + Debug + Clone + Send + Sync {
    /// Get the problem kind identifier.
    fn kind(&self) -> ProblemKind;

    /// Zero the output (if applicable).
    fn zero(&self);

    /// Get the total number of elements.
    fn total_size(&self) -> usize;

    /// Check if this is an in-place transform.
    fn is_inplace(&self) -> bool;
}
