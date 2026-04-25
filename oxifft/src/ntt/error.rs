//! Error types for the NTT module.

use core::fmt;

/// Errors that can occur during NTT operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NttError {
    /// Transform size is not a power of two.
    NotPowerOfTwo(usize),
    /// The provided modulus is not a prime number.
    NotPrime(u64),
    /// Transform size exceeds what the modulus can support.
    ///
    /// For a prime `p = c · 2^k + 1`, the maximum NTT size is `2^k`.
    SizeTooLarge {
        /// Requested transform size.
        n: usize,
        /// Maximum supported size for this modulus.
        max: usize,
    },
    /// No primitive n-th root of unity exists for this modulus.
    NoRootOfUnity {
        /// Requested transform size.
        n: usize,
        /// The prime modulus.
        modulus: u64,
    },
}

impl fmt::Display for NttError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NttError::NotPowerOfTwo(n) => {
                write!(f, "NTT size {n} is not a power of two")
            }
            NttError::NotPrime(p) => {
                write!(f, "modulus {p} is not a prime number")
            }
            NttError::SizeTooLarge { n, max } => {
                write!(f, "NTT size {n} exceeds maximum {max} for this modulus")
            }
            NttError::NoRootOfUnity { n, modulus } => {
                write!(
                    f,
                    "no primitive {n}-th root of unity exists modulo {modulus}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for NttError {}
