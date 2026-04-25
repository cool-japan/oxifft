//! Precomputed lookup tables for Rader's algorithm.
//!
//! Rader's algorithm converts a prime-size DFT to a cyclic convolution,
//! requiring precomputed powers of the primitive root.

use super::{Complex, Float};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Precomputed tables for Rader's algorithm.
#[allow(dead_code)] // reason: Rader's algorithm tables; struct fields are accessed via pub fields in the Bluestein solver
pub struct RaderOmega<T: Float> {
    /// Prime size.
    pub p: usize,
    /// Primitive root.
    pub g: usize,
    /// Powers of g: g^0, g^1, ..., g^(p-2) mod p.
    pub g_powers: Vec<usize>,
    /// Inverse powers: g^(-0), g^(-1), ..., g^(-(p-2)) mod p.
    pub g_inv_powers: Vec<usize>,
    /// Omega values: e^(-2πi * g^k / p) for k = 0..p-1.
    pub omega: Vec<Complex<T>>,
}

#[allow(dead_code)] // reason: impl block for RaderOmega; methods used by Bluestein/Rader solvers but not in all build configurations
impl<T: Float> RaderOmega<T> {
    /// Compute Rader omega tables for prime p.
    ///
    /// Returns `None` if p is not prime.
    #[must_use]
    pub fn new(p: usize) -> Option<Self> {
        use super::primes::{is_prime, mod_inv, primitive_root};

        if !is_prime(p) || p < 2 {
            return None;
        }

        let g = primitive_root(p)?;
        let n = p - 1;

        // Compute g^k mod p for k = 0..n
        let mut g_powers = Vec::with_capacity(n);
        let mut val = 1;
        for _ in 0..n {
            g_powers.push(val);
            val = (val * g) % p;
        }

        // Compute g^(-k) mod p
        let g_inv = mod_inv(g, p)?;
        let mut g_inv_powers = Vec::with_capacity(n);
        val = 1;
        for _ in 0..n {
            g_inv_powers.push(val);
            val = (val * g_inv) % p;
        }

        // Compute omega values
        let mut omega = Vec::with_capacity(n);
        for &gk in &g_powers {
            let theta = -T::TWO_PI * T::from_usize(gk) / T::from_usize(p);
            omega.push(Complex::cis(theta));
        }

        Some(Self {
            p,
            g,
            g_powers,
            g_inv_powers,
            omega,
        })
    }

    /// Get the convolution size (p - 1).
    #[must_use]
    pub fn conv_size(&self) -> usize {
        self.p - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rader_omega() {
        let omega: Option<RaderOmega<f64>> = RaderOmega::new(7);
        assert!(omega.is_some());

        let omega = omega.unwrap();
        assert_eq!(omega.p, 7);
        assert_eq!(omega.conv_size(), 6);
        assert_eq!(omega.g_powers.len(), 6);
    }

    #[test]
    fn test_rader_omega_non_prime() {
        let omega: Option<RaderOmega<f64>> = RaderOmega::new(8);
        assert!(omega.is_none());
    }
}
