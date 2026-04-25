//! Modular arithmetic primitives for NTT.
//!
//! All operations use `u128` intermediates to avoid overflow when working
//! with 64-bit moduli.

/// Modular multiplication: `(a * b) mod m`.
///
/// Uses 128-bit intermediate to avoid overflow.
#[inline]
pub fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Modular exponentiation: `base^exp mod m` via binary (square-and-multiply) method.
#[inline]
pub fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    let mut result: u64 = 1;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base, m);
        }
        exp >>= 1;
        base = mod_mul(base, base, m);
    }
    result
}

/// Modular inverse of `a` modulo `m` via the extended Euclidean algorithm.
///
/// Returns `None` if `gcd(a, m) != 1` (i.e., the inverse does not exist).
pub fn mod_inv(a: u64, m: u64) -> Option<u64> {
    if m == 0 {
        return None;
    }
    if m == 1 {
        return Some(0);
    }

    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);

    while r != 0 {
        let quotient = old_r / r;
        let temp_r = r;
        r = old_r - quotient * r;
        old_r = temp_r;

        let temp_s = s;
        s = old_s - quotient * s;
        old_s = temp_s;
    }

    // gcd != 1 means no inverse
    if old_r != 1 {
        return None;
    }

    // Normalize to positive
    let result = ((old_s % m as i128) + m as i128) % m as i128;
    Some(result as u64)
}

/// Deterministic Miller-Rabin primality test for all `u64` values.
///
/// Uses a set of witnesses that is sufficient to deterministically test
/// any 64-bit integer: {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}.
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    // Small prime check
    const SMALL_PRIMES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    for &p in &SMALL_PRIMES {
        if n == p {
            return true;
        }
        if n.is_multiple_of(p) {
            return false;
        }
    }

    // Write n - 1 = d * 2^r
    let mut d = n - 1;
    let mut r = 0u32;
    while d & 1 == 0 {
        d >>= 1;
        r += 1;
    }

    // Test with all witnesses — deterministic for all u64
    'witness: for &a in &SMALL_PRIMES {
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = mod_mul(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Find the smallest primitive root of a prime `p`.
///
/// A primitive root `g` generates the entire multiplicative group (Z/pZ)*,
/// meaning `g^k` for `k = 0, 1, ..., p-2` produces all non-zero residues.
///
/// Returns `None` if `p` is not a prime number (or if `p < 2`).
pub fn primitive_root(p: u64) -> Option<u64> {
    if p < 2 {
        return None;
    }
    if p == 2 {
        return Some(1);
    }
    if !is_prime(p) {
        return None;
    }

    // Factor p - 1
    let phi = p - 1;
    let factors = factorize(phi);

    // Try candidates starting from 2
    'candidate: for g in 2..p {
        for &f in &factors {
            if mod_pow(g, phi / f, p) == 1 {
                continue 'candidate;
            }
        }
        return Some(g);
    }

    None // Should not reach here for valid primes
}

/// Return the distinct prime factors of `n`.
fn factorize(mut n: u64) -> crate::prelude::Vec<u64> {
    let mut factors = crate::prelude::Vec::new();
    let mut d = 2u64;
    while d * d <= n {
        if n.is_multiple_of(d) {
            factors.push(d);
            while n.is_multiple_of(d) {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

/// Compute the largest power of 2 dividing `n`.
///
/// Returns the exponent `k` such that `2^k | n` but `2^(k+1)` does not.
/// Returns 0 if `n` is odd, and `None` if `n == 0`.
pub fn two_adic_valuation(n: u64) -> Option<u32> {
    if n == 0 {
        return None;
    }
    Some(n.trailing_zeros())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_mul() {
        assert_eq!(mod_mul(0, 0, 7), 0);
        assert_eq!(mod_mul(3, 4, 7), 5); // 12 mod 7 = 5
        assert_eq!(mod_mul(6, 6, 7), 1); // 36 mod 7 = 1
                                         // Large values that would overflow u64
        let big = (1u64 << 62) - 1;
        let m = 998_244_353;
        let result = mod_mul(big, big, m);
        assert!(result < m);
    }

    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 10, 1000), 24); // 1024 mod 1000
        assert_eq!(mod_pow(3, 0, 7), 1);
        assert_eq!(mod_pow(0, 0, 7), 1); // by convention
        assert_eq!(mod_pow(5, 3, 13), 8); // 125 mod 13 = 8
        assert_eq!(mod_pow(2, 23, 998_244_353), 8_388_608);
    }

    #[test]
    fn test_mod_pow_modulus_one() {
        assert_eq!(mod_pow(5, 3, 1), 0);
    }

    #[test]
    fn test_mod_inv() {
        // 3 * 5 = 15 ≡ 1 (mod 7)
        assert_eq!(mod_inv(3, 7), Some(5));
        // No inverse for 0
        assert_eq!(mod_inv(0, 7), None);
        // No inverse when gcd > 1
        assert_eq!(mod_inv(4, 8), None);
        // Self-inverse
        assert_eq!(mod_inv(1, 998_244_353), Some(1));
        // Verify round-trip
        let inv = mod_inv(42, 998_244_353);
        assert!(inv.is_some());
        assert_eq!(mod_mul(42, inv.expect("just checked"), 998_244_353), 1);
    }

    #[test]
    fn test_mod_inv_edge() {
        assert_eq!(mod_inv(1, 0), None);
        assert_eq!(mod_inv(5, 1), Some(0));
    }

    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(is_prime(998_244_353));
        assert!(is_prime(469_762_049));
        assert!(is_prime(167_772_161));
        assert!(!is_prime(998_244_354));
        // Large known prime
        assert!(is_prime(1_000_000_007));
    }

    #[test]
    fn test_primitive_root() {
        // All three NTT primes have primitive root 3
        assert_eq!(primitive_root(998_244_353), Some(3));
        assert_eq!(primitive_root(469_762_049), Some(3));
        assert_eq!(primitive_root(167_772_161), Some(3));
        // Small primes
        assert_eq!(primitive_root(2), Some(1));
        assert_eq!(primitive_root(7), Some(3));
        // Not prime
        assert_eq!(primitive_root(4), None);
        assert_eq!(primitive_root(0), None);
        assert_eq!(primitive_root(1), None);
    }

    #[test]
    fn test_two_adic_valuation() {
        assert_eq!(two_adic_valuation(0), None);
        assert_eq!(two_adic_valuation(1), Some(0));
        assert_eq!(two_adic_valuation(8), Some(3));
        // 998244353 - 1 = 998244352 = 119 * 2^23
        assert_eq!(two_adic_valuation(998_244_352), Some(23));
    }

    #[test]
    fn test_factorize() {
        assert_eq!(factorize(1), vec![]);
        assert_eq!(factorize(12), vec![2, 3]);
        assert_eq!(factorize(998_244_352), vec![2, 7, 17]);
    }
}
