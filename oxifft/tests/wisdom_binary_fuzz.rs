//! Property-based ("fuzz-style") tests for the binary wisdom decoder.
//!
//! `oxifft/fuzz/fuzz_targets/wisdom_parse_binary.rs` exercises the same
//! surface (`WisdomCache::from_binary`/`import_binary`) under `cargo fuzz`,
//! which requires a nightly toolchain and `cargo-fuzz`. This file runs the
//! same corpus logic under plain `cargo test` on stable, so the property is
//! still checked in environments without the nightly fuzzing toolchain.
//!
//! The binary wisdom format is the one wisdom representation that is
//! currently consulted to reconstruct an executable algorithm at
//! plan-construction time (via the build-time baseline embedded through
//! `OXIFFT_TUNE=1`), so `from_binary`/`import_binary` must never panic on
//! arbitrary bytes — not on truncated data, not on a bad magic/version, and
//! not on a degenerate `MixedRadix` factor list that would otherwise divide
//! by zero or hit an `unreachable!()` when replayed.

use oxifft::api::WisdomCache;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// Arbitrary bytes must never panic `from_binary`, regardless of
    /// content: bad magic, bad version, truncated entries, or degenerate
    /// `MixedRadix` factor lists must all become an `Err`, never a panic.
    #[test]
    fn from_binary_never_panics(data in proptest::collection::vec(any::<u8>(), 0..256)) {
        let _ = WisdomCache::from_binary(&data);
    }

    /// Same property for the stats-reporting `import_binary` entry point.
    #[test]
    fn import_binary_never_panics(data in proptest::collection::vec(any::<u8>(), 0..256)) {
        let mut cache = WisdomCache::new();
        let _ = cache.import_binary(&data);
    }

    /// Whatever successfully decodes must re-encode/re-decode without
    /// panicking either (round-trip stability under fuzzing).
    #[test]
    fn decoded_cache_round_trips_without_panicking(data in proptest::collection::vec(any::<u8>(), 0..256)) {
        if let Ok(cache) = WisdomCache::from_binary(&data) {
            let reencoded = cache.to_binary();
            let _ = WisdomCache::from_binary(&reencoded);
            let _ = cache.to_binary_checked();
        }
    }

    /// A structurally valid header (correct magic/version) with arbitrary
    /// entry bytes must still never panic, even when the declared
    /// `entry_count` disagrees with the actual data length.
    #[test]
    fn from_binary_never_panics_with_valid_header(
        version in prop_oneof![Just(1u16), Just(2u16), any::<u16>()],
        entry_count_field in any::<u32>(),
        tail in proptest::collection::vec(any::<u8>(), 0..512),
    ) {
        let mut data = Vec::new();
        data.extend_from_slice(b"OXIWISDM");
        data.extend_from_slice(&version.to_le_bytes());
        // Bytes 10..16 differ in meaning between v1 (u16 count + u32
        // reserved) and v2 (u16 reserved + u32 count); either way, feeding
        // an arbitrary u32 here covers both interpretations.
        data.extend_from_slice(&0u16.to_le_bytes());
        data.extend_from_slice(&entry_count_field.to_le_bytes());
        data.extend_from_slice(&tail);

        let _ = WisdomCache::from_binary(&data);
    }
}

/// Regression test mirroring the exact crash class the mixed-radix fix
/// targets: a `MixedRadix`-tagged entry with a factor of zero must decode to
/// an `Ok` result with the entry skipped, never a panic (integer
/// divide-by-zero) and never a stored, replayable entry.
#[test]
fn from_binary_rejects_zero_factor_mixed_radix_without_panicking() {
    const ALGO_TAG_MIXED_RADIX: u8 = 5;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"OXIWISDM");
    bytes.extend_from_slice(&2u16.to_le_bytes()); // format_version = 2
    bytes.extend_from_slice(&0u16.to_le_bytes()); // reserved
    bytes.extend_from_slice(&1u32.to_le_bytes()); // entry_count = 1

    bytes.extend_from_slice(&8u64.to_le_bytes()); // size_key
    bytes.push(ALGO_TAG_MIXED_RADIX);
    bytes.push(1); // factors_len
    bytes.extend_from_slice(&0u16.to_le_bytes()); // factor = 0 (degenerate)
    bytes.extend_from_slice(&1000u64.to_le_bytes()); // elapsed_ns

    let cache =
        WisdomCache::from_binary(&bytes).expect("structurally valid data must not hard-error");
    assert!(
        cache.lookup(8).is_none(),
        "a degenerate zero-factor MixedRadix entry must never be stored"
    );
}
