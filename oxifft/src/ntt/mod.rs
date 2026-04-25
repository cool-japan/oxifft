//! Number Theoretic Transform (NTT) — exact integer FFT over prime fields.
//!
//! The NTT operates on integers modulo a prime `p`, providing exact arithmetic
//! (no floating-point rounding) for polynomial multiplication, cryptographic
//! applications, and large integer arithmetic.
//!
//! # Overview
//!
//! The NTT is the finite-field analogue of the discrete Fourier transform.
//! Where the DFT uses complex roots of unity (e^{2πi/n}), the NTT uses
//! primitive roots of unity in Z/pZ — integers `ω` such that `ω^n ≡ 1 (mod p)`
//! and no smaller power of `ω` equals 1.
//!
//! # NTT-Friendly Primes
//!
//! For an NTT of size `n = 2^k`, the prime `p` must satisfy `p ≡ 1 (mod n)`.
//! Primes of the form `p = c · 2^k + 1` are ideal because they support NTT
//! sizes up to `2^k`.
//!
//! This module provides three well-known NTT-friendly primes:
//! - [`NTT_PRIME_998244353`] = 119 · 2²³ + 1 (supports up to 2²³ = 8M elements)
//! - [`NTT_PRIME_MOD1`] = 7 · 2²⁶ + 1 (supports up to 2²⁶ = 64M elements)
//! - [`NTT_PRIME_MOD2`] = 5 · 2²⁵ + 1 (supports up to 2²⁵ = 32M elements)
//!
//! # Example
//!
//! ```
//! use oxifft::ntt::{ntt_convolve_default, NttPlan, NTT_PRIME_998244353};
//!
//! // Polynomial multiplication: (1 + 2x)(3 + 4x) = 3 + 10x + 8x²
//! let a = vec![1u64, 2];
//! let b = vec![3u64, 4];
//! let result = ntt_convolve_default(&a, &b).unwrap();
//! assert_eq!(result[0], 3);
//! assert_eq!(result[1], 10);
//! assert_eq!(result[2], 8);
//!
//! // Plan-based API for repeated transforms
//! let plan = NttPlan::new(4, NTT_PRIME_998244353).unwrap();
//! let mut data = vec![1u64, 2, 3, 4];
//! plan.forward(&mut data);
//! plan.inverse(&mut data);
//! assert_eq!(data, vec![1, 2, 3, 4]);
//! ```

mod arith;
mod error;
mod plan;
#[cfg(test)]
mod tests;

pub use arith::{is_prime, mod_inv, mod_mul, mod_pow, primitive_root};
pub use error::NttError;
pub use plan::NttPlan;

use crate::prelude::*;

// ============================================================================
// NTT-friendly primes: p = c * 2^k + 1
// ============================================================================

/// 998244353 = 119 · 2²³ + 1, primitive root = 3.
///
/// Supports NTT sizes up to 2²³ = 8,388,608 elements.
/// This is the most widely used NTT prime in competitive programming.
pub const NTT_PRIME_998244353: u64 = 998_244_353;

/// 469762049 = 7 · 2²⁶ + 1, primitive root = 3.
///
/// Supports NTT sizes up to 2²⁶ = 67,108,864 elements.
pub const NTT_PRIME_MOD1: u64 = 469_762_049;

/// 167772161 = 5 · 2²⁵ + 1, primitive root = 3.
///
/// Supports NTT sizes up to 2²⁵ = 33,554,432 elements.
pub const NTT_PRIME_MOD2: u64 = 167_772_161;

// ============================================================================
// Convenience functions
// ============================================================================

/// Perform a forward NTT in-place.
///
/// Creates a temporary [`NttPlan`] and applies the forward transform.
/// For repeated transforms of the same size, prefer creating a plan once.
///
/// # Errors
///
/// Returns [`NttError`] if the length is not a power of two, the modulus is
/// not prime, or the modulus does not support the requested size.
pub fn ntt(data: &mut [u64], modulus: u64) -> Result<(), NttError> {
    let plan = NttPlan::new(data.len(), modulus)?;
    plan.forward(data);
    Ok(())
}

/// Perform an inverse NTT in-place (includes 1/n scaling).
///
/// # Errors
///
/// Returns [`NttError`] if the length is not a power of two, the modulus is
/// not prime, or the modulus does not support the requested size.
pub fn intt(data: &mut [u64], modulus: u64) -> Result<(), NttError> {
    let plan = NttPlan::new(data.len(), modulus)?;
    plan.inverse(data);
    Ok(())
}

/// Exact polynomial multiplication via NTT convolution.
///
/// Computes the product of polynomials `a` and `b` over Z/pZ.
/// The result has length `len(a) + len(b) - 1` (trimmed of trailing zeros
/// from padding).
///
/// # Errors
///
/// Returns [`NttError`] if the required padded size exceeds the modulus capacity.
pub fn ntt_convolve(a: &[u64], b: &[u64], modulus: u64) -> Result<Vec<u64>, NttError> {
    if a.is_empty() || b.is_empty() {
        return Ok(Vec::new());
    }

    let result_len = a.len() + b.len() - 1;
    let n = result_len.next_power_of_two();

    let plan = NttPlan::new(n, modulus)?;

    // Pad inputs to length n
    let mut fa = vec![0u64; n];
    fa[..a.len()].copy_from_slice(a);

    let mut fb = vec![0u64; n];
    fb[..b.len()].copy_from_slice(b);

    // Forward NTT
    plan.forward(&mut fa);
    plan.forward(&mut fb);

    // Pointwise multiplication in frequency domain
    for i in 0..n {
        fa[i] = mod_mul(fa[i], fb[i], modulus);
    }

    // Inverse NTT
    plan.inverse(&mut fa);

    // Trim to actual result length
    fa.truncate(result_len);
    Ok(fa)
}

/// Exact polynomial multiplication using the default prime (998244353).
///
/// This is a convenience wrapper around [`ntt_convolve`] using
/// [`NTT_PRIME_998244353`]. Coefficients must be less than 998244353.
///
/// # Errors
///
/// Returns [`NttError`] if the required padded size exceeds 2²³.
pub fn ntt_convolve_default(a: &[u64], b: &[u64]) -> Result<Vec<u64>, NttError> {
    ntt_convolve(a, b, NTT_PRIME_998244353)
}
