//! Integration tests for `gen_any` — classify and generate for arbitrary FFT sizes.

use oxifft_codegen_impl::gen_any::{classify, generate, CodegenError, SizeClass};
use oxifft_codegen_impl::CodeletBuilder;

// ============================================================================
// classify — direct codelet sizes
// ============================================================================

#[test]
fn classify_notw_sizes() {
    for &n in &[2usize, 4, 8, 16, 32, 64] {
        assert!(
            matches!(classify(n).unwrap(), SizeClass::Notw(_)),
            "n={n} should be Notw"
        );
    }
}

#[test]
fn classify_odd_sizes() {
    for &n in &[3usize, 5, 7] {
        assert!(
            matches!(classify(n).unwrap(), SizeClass::Odd(_)),
            "n={n} should be Odd"
        );
    }
}

#[test]
fn classify_rader_hardcoded_sizes() {
    assert_eq!(classify(11).unwrap(), SizeClass::RaderHardcoded(11));
    assert_eq!(classify(13).unwrap(), SizeClass::RaderHardcoded(13));
}

// ============================================================================
// classify — mixed radix (smooth-7 composites)
// ============================================================================

#[test]
fn classify_mixed_radix_15() {
    match classify(15).unwrap() {
        SizeClass::MixedRadix(factors) => {
            assert!(factors.contains(&5), "15 = 5 × 3, must contain 5");
            assert!(factors.contains(&3), "15 = 5 × 3, must contain 3");
        }
        other => panic!("expected MixedRadix, got {other:?}"),
    }
}

#[test]
fn classify_mixed_radix_variety() {
    for &n in &[
        6usize, 10, 12, 14, 21, 24, 28, 30, 35, 40, 42, 48, 56, 60, 80, 84, 96, 112, 120,
    ] {
        assert!(
            matches!(classify(n).unwrap(), SizeClass::MixedRadix(_)),
            "n={n} expected MixedRadix"
        );
    }
}

// ============================================================================
// classify — Rader prime (runtime path)
// ============================================================================

#[test]
fn classify_rader_prime_runtime() {
    for &p in &[17usize, 19, 23, 29, 31, 37, 41, 43, 47, 97, 101, 1013, 1021] {
        assert!(
            matches!(classify(p).unwrap(), SizeClass::RaderPrime(_)),
            "n={p} should be RaderPrime (runtime)"
        );
    }
}

// ============================================================================
// classify — Bluestein (large primes and non-smooth composites)
// ============================================================================

#[test]
fn classify_bluestein_large_prime() {
    // 2003 is prime and > 1021
    assert_eq!(classify(2003).unwrap(), SizeClass::Bluestein(2003));
}

#[test]
fn classify_bluestein_non_smooth() {
    // 2006 = 2 × 17 × 59 — has prime factor 17 and 59 (>7)
    assert!(matches!(classify(2006).unwrap(), SizeClass::Bluestein(_)));
}

// ============================================================================
// classify — edge cases
// ============================================================================

#[test]
fn classify_zero_is_err() {
    assert_eq!(classify(0).unwrap_err(), CodegenError::InvalidSize(0));
}

#[test]
fn classify_one_is_notw() {
    // N=1 is the identity; classified as Notw(1) and emitted as a trivial codelet
    assert_eq!(classify(1).unwrap(), SizeClass::Notw(1));
}

// ============================================================================
// generate — smoke tests (non-empty TokenStream)
// ============================================================================

#[test]
fn generate_emits_tokens_for_direct_size() {
    let ts = generate(8).unwrap();
    assert!(!ts.to_string().is_empty());
}

#[test]
fn generate_emits_tokens_for_odd_size() {
    let ts = generate(7).unwrap();
    assert!(!ts.to_string().is_empty());
}

#[test]
fn generate_emits_tokens_for_rader_hardcoded() {
    let ts = generate(11).unwrap();
    assert!(!ts.to_string().is_empty());
    let ts13 = generate(13).unwrap();
    assert!(!ts13.to_string().is_empty());
}

#[test]
fn generate_emits_tokens_for_mixed_radix() {
    let ts = generate(15).unwrap();
    let s = ts.to_string();
    assert!(!s.is_empty());
}

#[test]
fn generate_emits_tokens_for_rader_prime_runtime() {
    let ts = generate(17).unwrap();
    assert!(!ts.to_string().is_empty());
}

#[test]
fn generate_emits_tokens_for_bluestein() {
    let ts = generate(2003).unwrap();
    let s = ts.to_string();
    assert!(!s.is_empty());
}

#[test]
fn generate_emits_tokens_for_identity() {
    let ts = generate(1).unwrap();
    assert!(!ts.to_string().is_empty());
}

#[test]
fn generate_zero_returns_err() {
    assert!(matches!(generate(0), Err(CodegenError::InvalidSize(0))));
}

// ============================================================================
// CodeletBuilder
// ============================================================================

#[test]
fn codelet_builder_zero_returns_err() {
    let result = CodeletBuilder::new(0).build();
    assert!(result.is_err());
}

#[test]
fn codelet_builder_direct_size() {
    let ts = CodeletBuilder::new(8).build().unwrap();
    assert!(!ts.to_string().is_empty());
}

#[test]
fn codelet_builder_mixed_radix() {
    let ts = CodeletBuilder::new(15).build().unwrap();
    assert!(!ts.to_string().is_empty());
}

#[test]
fn codelet_builder_bluestein() {
    let ts = CodeletBuilder::new(2003).build().unwrap();
    assert!(!ts.to_string().is_empty());
}

#[test]
fn codelet_builder_with_name() {
    // name() is reserved; verify it doesn't alter the result being non-empty
    let ts = CodeletBuilder::new(8).name("my_codelet").build().unwrap();
    assert!(!ts.to_string().is_empty());
}
