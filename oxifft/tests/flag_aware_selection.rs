//! Flag-aware algorithm selection and wisdom feedback.
//!
//! Covers the v0.4.0 planner work:
//! - MEASURE/PATIENT/EXHAUSTIVE produce correct results (never regress vs naive DFT).
//! - Rader is reachable from the ESTIMATE heuristic for prime sizes.
//! - Every algorithm reachable through measured/imported wisdom is correct
//!   (deterministic injection, bypassing timing-dependent selection).
//! - The build-time baseline binary format round-trips.

#![allow(clippy::cast_precision_loss)]
// reason: the naive DFT is a deliberately plain, readable reference; fused
// multiply-add would obscure the complex-multiply structure for no test benefit.
#![allow(clippy::suboptimal_flops)]

use oxifft::api::{store_wisdom, WisdomCache};
use oxifft::kernel::WisdomEntry;
use oxifft::{Complex, Direction, Flags, Plan};

/// Naive O(n²) forward DFT reference (sign = -1 exponent).
fn naive_dft(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let n = input.len();
    let mut out = vec![Complex::new(0.0, 0.0); n];
    for (k, slot) in out.iter_mut().enumerate() {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (j, x) in input.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
            let (c, s) = (angle.cos(), angle.sin());
            re += x.re * c - x.im * s;
            im += x.re * s + x.im * c;
        }
        *slot = Complex::new(re, im);
    }
    out
}

fn sample_input(n: usize) -> Vec<Complex<f64>> {
    (0..n)
        .map(|i| Complex::new((i as f64 * 0.3).sin(), (i as f64 * 0.17).cos()))
        .collect()
}

fn assert_matches_naive(n: usize, flags: Flags) {
    let input = sample_input(n);
    let plan =
        Plan::<f64>::dft_1d(n, Direction::Forward, flags).expect("plan construction must succeed");
    let mut out = vec![Complex::new(0.0, 0.0); n];
    plan.execute(&input, &mut out);
    let reference = naive_dft(&input);
    for (k, (got, want)) in out.iter().zip(reference.iter()).enumerate() {
        assert!(
            (got.re - want.re).abs() < 1e-6 && (got.im - want.im).abs() < 1e-6,
            "n={n} flags={flags:?} algo={} k={k}: got {got:?}, want ({}, {})",
            plan.algorithm_name(),
            want.re,
            want.im
        );
    }
}

/// Every planning mode must yield correct spectra for a broad size mix
/// (powers of two, primes routed to Rader, smooth composites, mixed-radix).
#[test]
fn all_flag_modes_match_naive_dft() {
    let sizes = [
        2usize, 3, 4, 5, 7, 8, 11, 12, 13, 15, 16, 17, 19, 31, 32, 41, 60, 64,
    ];
    for &n in &sizes {
        for flags in [Flags::ESTIMATE, Flags::MEASURE, Flags::PATIENT] {
            assert_matches_naive(n, flags);
        }
    }
    // EXHAUSTIVE is the priciest mode; run it on a representative subset.
    for &n in &[2usize, 5, 8, 12, 16, 17, 31, 64] {
        assert_matches_naive(n, Flags::EXHAUSTIVE);
    }
    // Exercise the Stockham candidate (proposed for n >= 256) under PATIENT.
    assert_matches_naive(256, Flags::PATIENT);
}

/// Rader must be reachable from the ESTIMATE heuristic for prime sizes
/// (the README's "Full Algorithm Support: Rader" claim). ESTIMATE ignores
/// runtime wisdom, so this is independent of any other test's measurements.
#[test]
fn rader_selected_for_primes_in_estimate() {
    for &p in &[101usize, 251, 521] {
        let plan = Plan::<f64>::dft_1d(p, Direction::Forward, Flags::ESTIMATE).expect("prime plan");
        assert_eq!(
            plan.algorithm_name(),
            "Rader",
            "prime {p} should be planned with Rader in ESTIMATE mode"
        );
        assert_matches_naive(p, Flags::ESTIMATE);
    }
}

/// Every algorithm reconstructable from a wisdom entry — including the ones the
/// heuristic never picks (cache-oblivious, split-radix, radix-4/8, DIF,
/// Stockham, Direct) — must produce correct output. This pins the correctness
/// of algorithms made reachable via the measured/wisdom path.
///
/// Uses sizes not shared with any other test so the injected `GLOBAL_WISDOM`
/// entries cannot race with concurrent tests in this binary.
#[test]
fn injected_wisdom_algorithms_are_correct() {
    // (size, wisdom solver name, expected display name)
    let cases: &[(usize, &str, &str)] = &[
        (1024, "cache-oblivious", "CacheOblivious"),
        (1024, "stockham", "Stockham"),
        (1024, "ct-dif", "CooleyTukey(Dif)"),
        (1024, "ct-radix4", "CooleyTukey(DitRadix4)"),
        (512, "ct-radix8", "CooleyTukey(DitRadix8)"),
        (1024, "ct-splitradix", "CooleyTukey(SplitRadix)"),
        (47, "direct", "Direct"),
        (47, "rader", "Rader"),
    ];

    for &(n, solver_name, expected_display) in cases {
        store_wisdom(WisdomEntry {
            problem_hash: n as u64,
            solver_name: solver_name.to_string(),
            cost: 1.0,
        });
        // MEASURE hits the just-stored wisdom entry and reconstructs the exact
        // algorithm without benchmarking.
        let plan = Plan::<f64>::dft_1d(n, Direction::Forward, Flags::MEASURE)
            .unwrap_or_else(|| panic!("plan for n={n} solver={solver_name}"));
        assert_eq!(
            plan.algorithm_name(),
            expected_display,
            "wisdom '{solver_name}' for n={n} must reconstruct to {expected_display}"
        );

        let input = sample_input(n);
        let mut out = vec![Complex::new(0.0, 0.0); n];
        plan.execute(&input, &mut out);
        let reference = naive_dft(&input);
        for (k, (got, want)) in out.iter().zip(reference.iter()).enumerate() {
            assert!(
                (got.re - want.re).abs() < 1e-5 && (got.im - want.im).abs() < 1e-5,
                "n={n} algo={solver_name} k={k}: got {got:?}, want ({}, {})",
                want.re,
                want.im
            );
        }
    }
}

/// The exact V1 binary layout `build.rs` emits under `OXIFFT_TUNE=1`
/// (magic | version=1 | count u16 | reserved u32 | 30-byte entries) must keep
/// decoding through the runtime baseline reader. Because a build script cannot
/// link the crate, `build.rs` hand-encodes this frozen legacy format; if the
/// runtime ever dropped V1 support, embedded build-time baselines would
/// silently stop loading — this test guards that contract.
#[test]
fn build_time_baseline_v1_bytes_are_accepted_by_runtime() {
    const MAGIC: &[u8; 8] = b"OXIWISDM";
    let sizes: Vec<u64> = (1..=16u32).map(|k| 1u64 << k).collect();
    let count = u16::try_from(sizes.len()).expect("fits u16");

    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&1u16.to_le_bytes()); // format version 1
    bytes.extend_from_slice(&count.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes()); // reserved
    for &size in &sizes {
        bytes.extend_from_slice(&size.to_le_bytes()); // size_key
        bytes.push(0u8); // algo_tag = CooleyTukey → "ct-dit"
        bytes.push(0u8); // factors_len
        bytes.extend_from_slice(&[0u8; 12]); // factors [u16; 6]
        bytes.extend_from_slice(&0u64.to_le_bytes()); // elapsed_ns
    }
    assert_eq!(bytes.len(), 16 + 16 * 30, "V1 baseline must be 496 bytes");

    let restored = WisdomCache::from_binary(&bytes)
        .expect("runtime must accept the V1 baseline build.rs emits");
    let entry = restored.lookup(1024).expect("1024 entry must decode");
    assert_eq!(
        entry.solver_name, "ct-dit",
        "power-of-two baseline entry must decode to ct-dit"
    );
}
