//! NTT plan: precomputed state for forward/inverse Number Theoretic Transforms.

use crate::prelude::*;

use super::arith::is_prime;
use super::arith::{mod_inv, mod_mul, mod_pow, primitive_root, two_adic_valuation};
use super::error::NttError;

/// Precomputed NTT plan for a given size and prime modulus.
///
/// The plan stores twiddle factors and auxiliary constants so that multiple
/// transforms of the same size can be executed without recomputation.
#[derive(Debug, Clone)]
pub struct NttPlan {
    /// Transform size (must be a power of 2).
    n: usize,
    /// Prime modulus.
    modulus: u64,
    /// Primitive n-th root of unity mod p.
    root: u64,
    /// Modular inverse of n mod p (for 1/n scaling in INTT).
    inv_n: u64,
    /// Precomputed twiddle factors for forward transform.
    twiddles: Vec<u64>,
    /// Precomputed twiddle factors for inverse transform.
    inv_twiddles: Vec<u64>,
}

impl NttPlan {
    /// Create a new NTT plan for transforms of size `n` modulo `modulus`.
    ///
    /// # Requirements
    ///
    /// - `n` must be a power of two (n = 1 is allowed).
    /// - `modulus` must be a prime number.
    /// - `modulus - 1` must be divisible by `n` (i.e., an n-th root of unity must exist).
    ///
    /// # Errors
    ///
    /// - [`NttError::NotPowerOfTwo`] if `n` is not a power of two.
    /// - [`NttError::NotPrime`] if `modulus` is not prime.
    /// - [`NttError::SizeTooLarge`] if `n` exceeds the modulus capacity.
    /// - [`NttError::NoRootOfUnity`] if no primitive root can be found.
    ///
    /// # Implementation note (u32 bit-reversal bound)
    ///
    /// The internal bit-reversal permutation casts indices to `u32`. This is safe
    /// because `n` is bounded by `1 << k` where `k` is the two-adic valuation of
    /// `modulus − 1`. For all practical NTT primes (e.g. 998244353, k = 23;
    /// 2013265921, k = 27), `n ≤ 2^27 ≪ 2^32`, so no truncation can occur.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxifft::{NttPlan, NTT_PRIME_998244353};
    ///
    /// let plan = NttPlan::new(8, NTT_PRIME_998244353).expect("NTT plan creation failed");
    /// let original = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
    /// let mut data = original.clone();
    ///
    /// plan.forward(&mut data);
    /// // Data is now NTT-transformed (values are mod p)
    /// assert_ne!(data, original);
    ///
    /// plan.inverse(&mut data);
    /// // After round-trip, data should match the original
    /// assert_eq!(data, original);
    /// ```
    pub fn new(n: usize, modulus: u64) -> Result<Self, NttError> {
        // n = 0 is not a power of two (0 & (0-1) wraps, but we catch it)
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(NttError::NotPowerOfTwo(n));
        }

        if modulus < 2 || !is_prime(modulus) {
            return Err(NttError::NotPrime(modulus));
        }

        // Check that the modulus supports this size.
        // For p = c * 2^k + 1, the max NTT size is 2^k.
        let p_minus_1 = modulus - 1;
        let k = two_adic_valuation(p_minus_1).unwrap_or(0);
        let max_n = 1usize << k;
        if n > max_n {
            return Err(NttError::SizeTooLarge { n, max: max_n });
        }

        // n = 1: trivial transform
        if n == 1 {
            return Ok(Self {
                n: 1,
                modulus,
                root: 1,
                inv_n: 1,
                twiddles: vec![1],
                inv_twiddles: vec![1],
            });
        }

        // Find primitive root of p, then derive the n-th root of unity.
        let g = primitive_root(modulus).ok_or(NttError::NoRootOfUnity { n, modulus })?;

        // g is a generator of the full group of order p-1.
        // The n-th root of unity is w = g^((p-1)/n).
        let root = mod_pow(g, p_minus_1 / n as u64, modulus);
        let inv_root = mod_inv(root, modulus).ok_or(NttError::NoRootOfUnity { n, modulus })?;
        let inv_n = mod_inv(n as u64, modulus).ok_or(NttError::NoRootOfUnity { n, modulus })?;

        // Precompute twiddle factors for all butterfly stages.
        // twiddles[i] = root^i mod p for i = 0..n/2
        let half = n / 2;
        let mut twiddles = Vec::with_capacity(half);
        let mut tw = 1u64;
        for _ in 0..half {
            twiddles.push(tw);
            tw = mod_mul(tw, root, modulus);
        }

        let mut inv_twiddles = Vec::with_capacity(half);
        let mut itw = 1u64;
        for _ in 0..half {
            inv_twiddles.push(itw);
            itw = mod_mul(itw, inv_root, modulus);
        }

        Ok(Self {
            n,
            modulus,
            root,
            inv_n,
            twiddles,
            inv_twiddles,
        })
    }

    /// Transform size.
    #[inline]
    pub fn size(&self) -> usize {
        self.n
    }

    /// Prime modulus.
    #[inline]
    pub fn modulus(&self) -> u64 {
        self.modulus
    }

    /// The primitive n-th root of unity.
    #[inline]
    pub fn root(&self) -> u64 {
        self.root
    }

    /// Forward NTT (in-place, decimation-in-time with bit-reversal).
    ///
    /// After this call, `data` contains the NTT of the input in standard order.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != self.n`.
    pub fn forward(&self, data: &mut [u64]) {
        assert_eq!(data.len(), self.n, "data length must equal plan size");
        if self.n <= 1 {
            return;
        }
        bit_reverse_permutation(data);
        self.butterfly_dit(data, &self.twiddles);
    }

    /// Inverse NTT (in-place, includes 1/n scaling).
    ///
    /// After this call, `data` contains the original sequence (if it was
    /// previously transformed with [`forward`](Self::forward)).
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != self.n`.
    pub fn inverse(&self, data: &mut [u64]) {
        assert_eq!(data.len(), self.n, "data length must equal plan size");
        if self.n <= 1 {
            return;
        }
        bit_reverse_permutation(data);
        self.butterfly_dit(data, &self.inv_twiddles);
        // Scale by 1/n
        let m = self.modulus;
        let inv_n = self.inv_n;
        for x in data.iter_mut() {
            *x = mod_mul(*x, inv_n, m);
        }
    }

    /// Forward NTT, out-of-place.
    ///
    /// Copies `input` into `output`, then performs the forward transform on `output`.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != self.n` or `output.len() != self.n`.
    pub fn forward_into(&self, input: &[u64], output: &mut [u64]) {
        assert_eq!(input.len(), self.n, "input length must equal plan size");
        assert_eq!(output.len(), self.n, "output length must equal plan size");
        output.copy_from_slice(input);
        self.forward(output);
    }

    /// Cooley-Tukey DIT butterfly with precomputed twiddle factors.
    ///
    /// `tw` contains the powers of the root of unity: tw[i] = ω^i for i = 0..n/2.
    fn butterfly_dit(&self, data: &mut [u64], tw: &[u64]) {
        let n = self.n;
        let m = self.modulus;
        let mut len = 2; // butterfly width, doubles each stage
        while len <= n {
            let half = len / 2;
            // Step through twiddle factors: for this stage, the stride is n/len
            let step = n / len;
            for start in (0..n).step_by(len) {
                for j in 0..half {
                    let w = tw[j * step];
                    let u = data[start + j];
                    let v = mod_mul(data[start + j + half], w, m);
                    data[start + j] = if u + v >= m { u + v - m } else { u + v };
                    data[start + j + half] = if u >= v { u - v } else { u + m - v };
                }
            }
            len <<= 1;
        }
    }
}

/// Bit-reversal permutation on a slice whose length is a power of two.
fn bit_reverse_permutation(data: &mut [u64]) {
    let n = data.len();
    if n <= 2 {
        if n == 2 {
            // No reversal needed for n=2 (bit reverse of 0 is 0, of 1 is 1)
        }
        return;
    }
    let log_n = n.trailing_zeros();
    for i in 0..n {
        // SAFETY: NttPlan::new validates that n ≤ 1 << k where k is the two-adic
        // valuation of modulus − 1. For all supported NTT primes (e.g. 998244353
        // where k = 23) the maximum n is 2^23 < 2^32, so i < n fits in u32.
        #[allow(clippy::cast_possible_truncation)]
        let i_u32 = i as u32;
        let j = reverse_bits(i_u32, log_n) as usize;
        if i < j {
            data.swap(i, j);
        }
    }
}

/// Reverse the lowest `bits` bits of `x`.
#[inline]
fn reverse_bits(mut x: u32, bits: u32) -> u32 {
    let mut result = 0u32;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_bits() {
        assert_eq!(reverse_bits(0b000, 3), 0b000);
        assert_eq!(reverse_bits(0b001, 3), 0b100);
        assert_eq!(reverse_bits(0b011, 3), 0b110);
        assert_eq!(reverse_bits(0b101, 3), 0b101);
    }

    #[test]
    fn test_bit_reverse_permutation() {
        // For n=8, the bit-reverse permutation is: 0,4,2,6,1,5,3,7
        let mut data = vec![0, 1, 2, 3, 4, 5, 6, 7];
        bit_reverse_permutation(&mut data);
        assert_eq!(data, vec![0, 4, 2, 6, 1, 5, 3, 7]);
    }

    #[test]
    fn test_plan_n1() {
        let plan = NttPlan::new(1, 998_244_353);
        assert!(plan.is_ok());
        let plan = plan.expect("n=1 plan");
        let mut data = vec![42u64];
        plan.forward(&mut data);
        assert_eq!(data, vec![42]);
        plan.inverse(&mut data);
        assert_eq!(data, vec![42]);
    }

    #[test]
    fn test_plan_errors() {
        // Not power of two
        assert!(matches!(
            NttPlan::new(3, 998_244_353),
            Err(NttError::NotPowerOfTwo(3))
        ));
        assert!(matches!(
            NttPlan::new(0, 998_244_353),
            Err(NttError::NotPowerOfTwo(0))
        ));
        // Not prime
        assert!(matches!(NttPlan::new(4, 15), Err(NttError::NotPrime(15))));
        // Size too large: 998244353 supports up to 2^23
        assert!(matches!(
            NttPlan::new(1 << 24, 998_244_353),
            Err(NttError::SizeTooLarge { .. })
        ));
    }
}
