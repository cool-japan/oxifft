//! Prime number utilities for FFT algorithms.
//!
//! Provides factorization, primality testing, and primitive root computation
//! needed by algorithms like Rader's.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Check if a number is prime.
#[must_use]
pub fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }

    let mut i = 5;
    while i * i <= n {
        if n.is_multiple_of(i) || n.is_multiple_of(i + 2) {
            return false;
        }
        i += 6;
    }
    true
}

/// Factor a number into primes.
///
/// Returns a vector of (prime, exponent) pairs.
#[must_use]
pub fn factor(mut n: usize) -> Vec<(usize, usize)> {
    let mut factors = Vec::new();

    // Factor out 2s
    if n.is_multiple_of(2) {
        let mut exp = 0;
        while n.is_multiple_of(2) {
            n /= 2;
            exp += 1;
        }
        factors.push((2, exp));
    }

    // Factor odd primes
    let mut p = 3;
    while p * p <= n {
        if n.is_multiple_of(p) {
            let mut exp = 0;
            while n.is_multiple_of(p) {
                n /= p;
                exp += 1;
            }
            factors.push((p, exp));
        }
        p += 2;
    }

    if n > 1 {
        factors.push((n, 1));
    }

    factors
}

/// Get all prime factors (with repetition).
#[allow(dead_code)] // reason: public utility for prime factorization; not used in all solver paths
#[must_use]
pub fn prime_factors(n: usize) -> Vec<usize> {
    let mut result = Vec::new();
    for (p, exp) in factor(n) {
        for _ in 0..exp {
            result.push(p);
        }
    }
    result
}

/// Compute the smallest primitive root modulo p (p must be prime).
///
/// Returns `None` if p is not prime or p < 2.
#[must_use]
pub fn primitive_root(p: usize) -> Option<usize> {
    if !is_prime(p) || p < 2 {
        return None;
    }
    if p == 2 {
        return Some(1);
    }

    let phi = p - 1;
    let factors = factor(phi);

    'outer: for g in 2..p {
        for (f, _) in &factors {
            if mod_pow(g, phi / f, p) == 1 {
                continue 'outer;
            }
        }
        return Some(g);
    }

    None
}

/// Modular exponentiation: base^exp mod modulus.
#[must_use]
pub fn mod_pow(mut base: usize, mut exp: usize, modulus: usize) -> usize {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        exp /= 2;
        base = (base * base) % modulus;
    }
    result
}

/// Modular inverse using extended Euclidean algorithm.
///
/// Returns `None` if the inverse doesn't exist.
#[allow(dead_code)] // reason: modular inverse utility for Rader's algorithm; not called in all solver configurations
#[must_use]
pub fn mod_inv(a: usize, m: usize) -> Option<usize> {
    let (g, x, _) = extended_gcd(a as isize, m as isize);
    if g != 1 {
        None
    } else {
        Some(((x % m as isize + m as isize) % m as isize) as usize)
    }
}

/// Extended Euclidean algorithm.
///
/// Returns (gcd, x, y) such that ax + by = gcd.
#[allow(dead_code)] // reason: helper for mod_inv; not called when modular inverse path is inactive
fn extended_gcd(a: isize, b: isize) -> (isize, isize, isize) {
    if a == 0 {
        (b, 0, 1)
    } else {
        let (g, x, y) = extended_gcd(b % a, a);
        (g, y - (b / a) * x, x)
    }
}

/// Find the next power of 2 >= n.
#[allow(dead_code)] // reason: power-of-two utility; not used in all solver paths
#[must_use]
pub fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        n.next_power_of_two()
    }
}

/// Check if n is a power of 2.
#[allow(dead_code)] // reason: power-of-two predicate; not used in all solver paths
#[must_use]
pub const fn is_power_of_two(n: usize) -> bool {
    n.is_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(is_prime(17));
        assert!(is_prime(2017));
        assert!(!is_prime(2016));
    }

    #[test]
    fn test_factor() {
        assert_eq!(factor(12), vec![(2, 2), (3, 1)]);
        assert_eq!(factor(256), vec![(2, 8)]);
        assert_eq!(factor(17), vec![(17, 1)]);
    }

    #[test]
    fn test_primitive_root() {
        assert_eq!(primitive_root(7), Some(3));
        assert_eq!(primitive_root(17), Some(3));
        assert_eq!(primitive_root(4), None); // not prime
    }

    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 10, 1000), 24);
        assert_eq!(mod_pow(3, 4, 17), 13);
    }
}
