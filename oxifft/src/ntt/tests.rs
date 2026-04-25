//! Integration tests for the NTT module.

use super::*;

/// Naive polynomial multiplication for reference (O(n²)).
fn naive_poly_mul(a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut result = vec![0u64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] = (result[i + j] + mod_mul(ai, bj, modulus)) % modulus;
        }
    }
    result
}

// =========================================================================
// Roundtrip tests
// =========================================================================

#[test]
fn test_roundtrip_small() {
    let plan = NttPlan::new(4, NTT_PRIME_998244353).expect("valid plan");
    let original = vec![1, 2, 3, 4];
    let mut data = original.clone();
    plan.forward(&mut data);
    // After forward, data should differ from original (unless trivial)
    assert_ne!(data, original);
    plan.inverse(&mut data);
    assert_eq!(data, original);
}

#[test]
fn test_roundtrip_n2() {
    let plan = NttPlan::new(2, NTT_PRIME_998244353).expect("valid plan");
    let original = vec![10, 20];
    let mut data = original.clone();
    plan.forward(&mut data);
    plan.inverse(&mut data);
    assert_eq!(data, original);
}

#[test]
fn test_roundtrip_n1() {
    let plan = NttPlan::new(1, NTT_PRIME_998244353).expect("valid plan");
    let mut data = vec![42];
    plan.forward(&mut data);
    plan.inverse(&mut data);
    assert_eq!(data, vec![42]);
}

#[test]
fn test_roundtrip_zeros() {
    let plan = NttPlan::new(8, NTT_PRIME_998244353).expect("valid plan");
    let mut data = vec![0u64; 8];
    plan.forward(&mut data);
    assert_eq!(data, vec![0u64; 8]);
    plan.inverse(&mut data);
    assert_eq!(data, vec![0u64; 8]);
}

#[test]
fn test_roundtrip_large() {
    let n = 1024;
    let plan = NttPlan::new(n, NTT_PRIME_998244353).expect("valid plan");
    let original: Vec<u64> = (0..n as u64).collect();
    let mut data = original.clone();
    plan.forward(&mut data);
    plan.inverse(&mut data);
    assert_eq!(data, original);
}

#[test]
fn test_roundtrip_all_primes() {
    let primes = [NTT_PRIME_998244353, NTT_PRIME_MOD1, NTT_PRIME_MOD2];
    for &p in &primes {
        let plan = NttPlan::new(8, p).expect("valid plan");
        let original = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut data = original.clone();
        plan.forward(&mut data);
        plan.inverse(&mut data);
        assert_eq!(data, original, "roundtrip failed for prime {p}");
    }
}

// =========================================================================
// Forward-into (out-of-place) test
// =========================================================================

#[test]
fn test_forward_into() {
    let plan = NttPlan::new(4, NTT_PRIME_998244353).expect("valid plan");
    let input = vec![1, 2, 3, 4];
    let mut output = vec![0u64; 4];
    plan.forward_into(&input, &mut output);

    // Compare with in-place result
    let mut inplace = input;
    plan.forward(&mut inplace);
    assert_eq!(output, inplace);
}

// =========================================================================
// Convolution tests
// =========================================================================

#[test]
fn test_convolve_simple() {
    // (1 + 2x)(3 + 4x) = 3 + 10x + 8x²
    let a = vec![1u64, 2];
    let b = vec![3u64, 4];
    let result = ntt_convolve_default(&a, &b).expect("convolve");
    assert_eq!(result, vec![3, 10, 8]);
}

#[test]
fn test_convolve_vs_naive() {
    let modulus = NTT_PRIME_998244353;
    let a: Vec<u64> = vec![5, 3, 7, 2, 1];
    let b: Vec<u64> = vec![4, 8, 6];
    let expected = naive_poly_mul(&a, &b, modulus);
    let result = ntt_convolve(&a, &b, modulus).expect("convolve");
    assert_eq!(result, expected);
}

#[test]
fn test_convolve_vs_naive_all_primes() {
    let primes = [NTT_PRIME_998244353, NTT_PRIME_MOD1, NTT_PRIME_MOD2];
    let a: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let b: Vec<u64> = vec![8, 7, 6, 5, 4, 3, 2, 1];
    for &p in &primes {
        let expected = naive_poly_mul(&a, &b, p);
        let result = ntt_convolve(&a, &b, p).expect("convolve");
        assert_eq!(result, expected, "convolution mismatch for prime {p}");
    }
}

#[test]
fn test_convolve_large_coefficients() {
    let modulus = NTT_PRIME_998244353;
    let a = vec![modulus - 1, modulus - 2]; // large coefficients
    let b = vec![modulus - 3, 1];
    let expected = naive_poly_mul(&a, &b, modulus);
    let result = ntt_convolve(&a, &b, modulus).expect("convolve");
    assert_eq!(result, expected);
}

#[test]
fn test_convolve_empty() {
    let result = ntt_convolve_default(&[], &[1, 2, 3]).expect("empty input");
    assert!(result.is_empty());
    let result = ntt_convolve_default(&[1, 2, 3], &[]).expect("empty input");
    assert!(result.is_empty());
}

#[test]
fn test_convolve_single_element() {
    // Multiply by scalar
    let result = ntt_convolve_default(&[5], &[3]).expect("single element");
    assert_eq!(result, vec![15]);
}

// =========================================================================
// One-shot ntt/intt tests
// =========================================================================

#[test]
fn test_oneshot_ntt_intt() {
    let mut data = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
    let original = data.clone();
    ntt(&mut data, NTT_PRIME_998244353).expect("ntt");
    intt(&mut data, NTT_PRIME_998244353).expect("intt");
    assert_eq!(data, original);
}

// =========================================================================
// Error case tests
// =========================================================================

#[test]
fn test_error_not_power_of_two() {
    let err = ntt(&mut [1, 2, 3], NTT_PRIME_998244353);
    assert!(matches!(err, Err(NttError::NotPowerOfTwo(3))));
}

#[test]
fn test_error_not_prime() {
    let err = NttPlan::new(4, 100);
    assert!(matches!(err, Err(NttError::NotPrime(100))));
}

#[test]
fn test_error_size_too_large() {
    // 998244353 supports up to 2^23
    let err = NttPlan::new(1 << 24, NTT_PRIME_998244353);
    assert!(matches!(
        err,
        Err(NttError::SizeTooLarge {
            n: 16777216,
            max: 8388608
        })
    ));
}

// =========================================================================
// Parseval-like identity: sum of squares in both domains
// =========================================================================

#[test]
fn test_linearity() {
    let modulus = NTT_PRIME_998244353;
    let plan = NttPlan::new(8, modulus).expect("plan");

    let mut a = vec![1, 2, 3, 4, 0, 0, 0, 0];
    let mut b = vec![5, 6, 7, 8, 0, 0, 0, 0];
    let mut sum: Vec<u64> = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x + y) % modulus)
        .collect();

    plan.forward(&mut a);
    plan.forward(&mut b);
    plan.forward(&mut sum);

    // NTT(a + b) should equal NTT(a) + NTT(b) (mod p)
    for i in 0..8 {
        let expected = (a[i] + b[i]) % modulus;
        assert_eq!(sum[i], expected, "linearity failed at index {i}");
    }
}

// =========================================================================
// Convolution theorem: pointwise product in NTT domain
// =========================================================================

#[test]
fn test_convolution_theorem() {
    let modulus = NTT_PRIME_998244353;
    let plan = NttPlan::new(8, modulus).expect("plan");

    let a_orig = vec![1u64, 2, 3, 0, 0, 0, 0, 0];
    let b_orig = vec![4u64, 5, 0, 0, 0, 0, 0, 0];

    let mut a_ntt = a_orig;
    let mut b_ntt = b_orig;
    plan.forward(&mut a_ntt);
    plan.forward(&mut b_ntt);

    // Pointwise multiply
    let mut prod: Vec<u64> = a_ntt
        .iter()
        .zip(b_ntt.iter())
        .map(|(&x, &y)| mod_mul(x, y, modulus))
        .collect();

    plan.inverse(&mut prod);

    // Compare with naive convolution
    let expected = naive_poly_mul(&[1, 2, 3], &[4, 5], modulus);
    for (i, &e) in expected.iter().enumerate() {
        assert_eq!(prod[i], e, "convolution theorem failed at index {i}");
    }
}
