//! Unit tests for `api::wisdom`.
//!
//! Split out of `wisdom.rs` (via `#[path = "wisdom_tests.rs"] mod tests;`,
//! the same convention already used by `dft/codelets/codegen_tests.rs`) to
//! keep the implementation file under the workspace's 2000-line limit. This
//! file is included as the body of `wisdom`'s private `tests` submodule, so
//! everything here has the same access to `wisdom`'s private items as if it
//! were still inlined via `use super::*;`.

use super::*;

// Tests that access the global wisdom cache must not run concurrently with
// each other — they share `GLOBAL_WISDOM` state.  This mutex is used as a
// cooperative semaphore so that only one such test holds the lock at a
// time.  The lock is intentionally acquired for the duration of the whole
// test body and released (via `_guard` drop) when the test returns.
static GLOBAL_WISDOM_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ── Basic cache operations ────────────────────────────────────────────────

#[test]
fn test_wisdom_cache_basic() {
    let mut cache = WisdomCache::new();
    assert!(cache.is_empty());

    let entry = WisdomEntry {
        problem_hash: 12345,
        solver_name: "ct-dit".to_string(),
        cost: 100.0,
    };
    cache.store(entry);

    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());

    let looked_up = cache.lookup(12345).expect("entry not found");
    assert_eq!(looked_up.solver_name, "ct-dit");
    assert!((looked_up.cost - 100.0).abs() < f64::EPSILON);
}

// ── Export / import round-trip (v1 format) ────────────────────────────────

#[test]
fn test_wisdom_export_import_v1() {
    let mut cache = WisdomCache::new();
    cache.store(WisdomEntry {
        problem_hash: 111,
        solver_name: "rader".to_string(),
        cost: 50.0,
    });
    cache.store(WisdomEntry {
        problem_hash: 222,
        solver_name: "bluestein".to_string(),
        cost: 75.0,
    });

    let exported = cache.export_string();
    assert!(exported.contains(WISDOM_MARKER));
    assert!(exported.contains("format_version"));
    assert!(exported.contains("111"));
    assert!(exported.contains("rader"));

    let mut cache2 = WisdomCache::new();
    let result = cache2.import_string(&exported).expect("import failed");
    assert_eq!(result.imported, 2);
    assert_eq!(result.skipped_invalid, 0);
    assert_eq!(result.format_version, WISDOM_FORMAT_VERSION);
    assert_eq!(cache2.len(), 2);

    let entry = cache2.lookup(111).expect("entry not found");
    assert_eq!(entry.solver_name, "rader");
}

// ── Legacy format (v0) accepted ───────────────────────────────────────────

#[test]
fn test_wisdom_legacy_format_accepted() {
    let legacy = "(oxifft-wisdom-1.0\n  (111 \"rader\" 50)\n  (222 \"bluestein\" 75)\n)";
    let mut cache = WisdomCache::new();
    let result = cache.import_string(legacy).expect("legacy import failed");
    assert_eq!(result.format_version, 0);
    assert_eq!(result.imported, 2);
    assert_eq!(cache.len(), 2);
}

// ── Incompatible (future) version rejected ────────────────────────────────

#[test]
fn test_wisdom_incompatible_version_rejected() {
    let future_version = WISDOM_FORMAT_VERSION + 1;
    let future_wisdom =
        format!("({WISDOM_MARKER}\n  (format_version {future_version})\n  (111 \"rader\" 50)\n)");
    let mut cache = WisdomCache::new();
    let err = cache
        .import_string(&future_wisdom)
        .expect_err("should have rejected future version");
    assert!(
        matches!(
            err,
            WisdomError::IncompatibleVersion {
                found,
                expected
            } if found == future_version && expected == WISDOM_FORMAT_VERSION
        ),
        "unexpected error: {err}"
    );
}

// ── Import validation: invalid entries are skipped ────────────────────────

#[test]
fn test_wisdom_import_skips_invalid_entries() {
    // Mix of valid entries and various forms of invalid data.
    // Entry with hash 0 — invalid.
    // Entry with empty solver — invalid.
    // Entry with NaN cost — invalid.
    // Entry with negative cost — invalid.
    // Entry (999, "ct-dit", 42.0) — valid.
    let wisdom_str = format!(
        "({WISDOM_MARKER}\n\
           (format_version {WISDOM_FORMAT_VERSION})\n\
           (0 \"ct-dit\" 1.0)\n\
           (333 \"\" 1.0)\n\
           (444 \"ct-dit\" NaN)\n\
           (555 \"ct-dit\" -1.0)\n\
           (999 \"ct-dit\" 42.0)\n\
         )"
    );

    let mut cache = WisdomCache::new();
    let result = cache
        .import_string(&wisdom_str)
        .expect("import should succeed");
    assert_eq!(
        result.imported, 1,
        "only the valid entry should be imported"
    );
    assert_eq!(
        result.skipped_invalid, 4,
        "four invalid entries should be skipped"
    );
    assert!(cache.lookup(999).is_some());
    assert!(cache.lookup(0).is_none());
    assert!(cache.lookup(333).is_none());
}

// ── Merge: new entries are added ──────────────────────────────────────────

#[test]
fn test_wisdom_merge_adds_new_entries() {
    let mut cache_a = WisdomCache::new();
    cache_a.store(WisdomEntry {
        problem_hash: 100,
        solver_name: "ct-dit".to_string(),
        cost: 10.0,
    });

    let mut cache_b = WisdomCache::new();
    cache_b.store(WisdomEntry {
        problem_hash: 200,
        solver_name: "bluestein".to_string(),
        cost: 20.0,
    });

    let b_str = cache_b.export_string();
    let merge = cache_a.merge_string(&b_str).expect("merge failed");

    assert_eq!(merge.added, 1);
    assert_eq!(merge.replaced, 0);
    assert_eq!(merge.kept_existing, 0);
    assert_eq!(merge.skipped_invalid, 0);
    assert_eq!(cache_a.len(), 2);
}

// ── Merge: lower cost wins ────────────────────────────────────────────────

#[test]
fn test_wisdom_merge_lower_cost_wins() {
    let mut cache_a = WisdomCache::new();
    cache_a.store(WisdomEntry {
        problem_hash: 100,
        solver_name: "ct-dit".to_string(),
        cost: 50.0,
    });

    // Incoming: same hash, lower cost.
    let incoming = format!(
        "({WISDOM_MARKER}\n\
           (format_version {WISDOM_FORMAT_VERSION})\n\
           (100 \"stockham\" 20.0)\n\
         )"
    );
    let merge = cache_a.merge_string(&incoming).expect("merge failed");

    assert_eq!(merge.replaced, 1);
    assert_eq!(merge.added, 0);
    assert_eq!(merge.kept_existing, 0);
    let entry = cache_a.lookup(100).expect("entry must still exist");
    assert_eq!(entry.solver_name, "stockham");
    assert!((entry.cost - 20.0).abs() < f64::EPSILON);
}

// ── Merge: higher cost in incoming — existing kept ────────────────────────

#[test]
fn test_wisdom_merge_keeps_existing_if_better() {
    let mut cache_a = WisdomCache::new();
    cache_a.store(WisdomEntry {
        problem_hash: 100,
        solver_name: "ct-dit".to_string(),
        cost: 10.0,
    });

    // Incoming: same hash, higher cost — should be ignored.
    let incoming = format!(
        "({WISDOM_MARKER}\n\
           (format_version {WISDOM_FORMAT_VERSION})\n\
           (100 \"rader\" 99.0)\n\
         )"
    );
    let merge = cache_a.merge_string(&incoming).expect("merge failed");

    assert_eq!(merge.kept_existing, 1);
    assert_eq!(merge.replaced, 0);
    let entry = cache_a.lookup(100).expect("entry must still exist");
    assert_eq!(entry.solver_name, "ct-dit"); // unchanged
}

// ── Merge rejects future format version ───────────────────────────────────

#[test]
fn test_wisdom_merge_rejects_future_version() {
    let future_version = WISDOM_FORMAT_VERSION + 5;
    let future_wisdom =
        format!("({WISDOM_MARKER}\n  (format_version {future_version})\n  (100 \"rader\" 1.0)\n)");
    let mut cache = WisdomCache::new();
    let err = cache
        .merge_string(&future_wisdom)
        .expect_err("should have rejected future version");
    assert!(matches!(
        err,
        WisdomError::IncompatibleVersion { found, .. } if found == future_version
    ));
}

// ── Global API ────────────────────────────────────────────────────────────

#[test]
fn test_wisdom_version_mismatch_unknown_header() {
    let mut cache = WisdomCache::new();
    let result = cache.import_string("(totally-unknown-header\n)");
    assert!(matches!(result, Err(WisdomError::ParseError(_))));
}

#[test]
fn test_global_wisdom_functions() {
    let _guard = GLOBAL_WISDOM_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // Clear any existing wisdom
    forget();
    assert_eq!(wisdom_count(), 0);

    // Store an entry
    store_wisdom(WisdomEntry {
        problem_hash: 999,
        solver_name: "generic".to_string(),
        cost: 200.0,
    });
    assert_eq!(wisdom_count(), 1);

    // Look it up
    let entry = lookup_wisdom(999).expect("entry not found");
    assert_eq!(entry.solver_name, "generic");

    // Export and reimport
    let exported = export_to_string();
    forget();
    assert_eq!(wisdom_count(), 0);

    let result = import_from_string(&exported).expect("import failed");
    assert_eq!(result.imported, 1);
    assert_eq!(wisdom_count(), 1);

    // Cleanup
    forget();
}

#[test]
fn test_global_merge_from_string() {
    let _guard = GLOBAL_WISDOM_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    forget();

    // Seed the global cache.
    store_wisdom(WisdomEntry {
        problem_hash: 1,
        solver_name: "ct-dit".to_string(),
        cost: 30.0,
    });

    // Merge with data carrying a better entry for hash 1 and a new entry
    // for hash 2.
    let incoming = format!(
        "({WISDOM_MARKER}\n\
           (format_version {WISDOM_FORMAT_VERSION})\n\
           (1 \"stockham\" 5.0)\n\
           (2 \"rader\" 10.0)\n\
         )"
    );
    let merge = merge_from_string(&incoming).expect("merge failed");
    assert_eq!(merge.replaced, 1);
    assert_eq!(merge.added, 1);
    assert_eq!(merge.kept_existing, 0);
    assert_eq!(wisdom_count(), 2);

    forget();
}

// ── File-backed API ───────────────────────────────────────────────────────

#[cfg(feature = "std")]
#[test]
// File I/O requires `-Zmiri-disable-isolation`; excluded from default MIRI runs.
// The underlying `export_to_file`/`import_from_file` logic is tested in native mode.
#[cfg_attr(miri, ignore)]
fn test_import_export_file_roundtrip() {
    let _guard = GLOBAL_WISDOM_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let dir = std::env::temp_dir();
    let path = dir.join("oxifft_wisdom_test_roundtrip.txt");

    forget();
    store_wisdom(WisdomEntry {
        problem_hash: 42,
        solver_name: "bluestein".to_string(),
        cost: 7.5,
    });

    export_to_file(&path).expect("export failed");
    forget();
    assert_eq!(wisdom_count(), 0);

    let result = import_from_file(&path).expect("import failed");
    assert_eq!(result.imported, 1);
    assert_eq!(wisdom_count(), 1);
    assert_eq!(
        lookup_wisdom(42).expect("entry missing").solver_name,
        "bluestein"
    );

    let _ = std::fs::remove_file(&path);
    forget();
}

#[cfg(feature = "std")]
#[test]
// File I/O requires `-Zmiri-disable-isolation`; excluded from default MIRI runs.
// The underlying `merge_from_file` logic is tested in native mode.
#[cfg_attr(miri, ignore)]
fn test_merge_from_file() {
    let _guard = GLOBAL_WISDOM_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let dir = std::env::temp_dir();
    let path = dir.join("oxifft_wisdom_test_merge.txt");

    forget();
    store_wisdom(WisdomEntry {
        problem_hash: 7,
        solver_name: "ct-dit".to_string(),
        cost: 100.0,
    });

    // Write a file with a better entry for hash 7.
    let content = format!(
        "({WISDOM_MARKER}\n\
           (format_version {WISDOM_FORMAT_VERSION})\n\
           (7 \"stockham\" 25.0)\n\
         )"
    );
    std::fs::write(&path, &content).expect("write failed");

    let merge = merge_from_file(&path).expect("merge failed");
    assert_eq!(merge.replaced, 1);
    assert_eq!(
        lookup_wisdom(7).expect("entry missing").solver_name,
        "stockham"
    );

    let _ = std::fs::remove_file(&path);
    forget();
}

// ── Poison recovery ───────────────────────────────────────────────────────

#[test]
fn test_global_wisdom_recovers_from_poisoning() {
    let _guard = GLOBAL_WISDOM_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    forget();

    // Poison GLOBAL_WISDOM's RwLock by panicking while a write guard
    // (obtained via the same private `with_wisdom_mut` path every public
    // mutating wisdom function uses) is held.
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_wisdom_mut(|_cache| {
            panic!("intentional panic to poison GLOBAL_WISDOM for testing recovery");
        });
    }));
    assert!(panicked.is_err(), "the closure should have panicked");

    // The lock is now poisoned. Earlier versions of this module treated
    // that as fatal (`.expect("Global wisdom lock poisoned")`), so every
    // subsequent call would panic too. The fix recovers the guard
    // instead, so ordinary wisdom operations must keep working.
    store_wisdom(WisdomEntry {
        problem_hash: 424_242,
        solver_name: "ct-dit".to_string(),
        cost: 1.0,
    });
    assert_eq!(
        wisdom_count(),
        1,
        "wisdom operations must survive lock poisoning"
    );
    let entry = lookup_wisdom(424_242).expect("entry not found after poison recovery");
    assert_eq!(entry.solver_name, "ct-dit");

    forget();
}

// ── Untrusted-input ceilings ──────────────────────────────────────────────

#[test]
fn test_import_string_rejects_oversized_input() {
    let huge = "x".repeat(MAX_WISDOM_TEXT_BYTES + 1);
    let mut cache = WisdomCache::new();
    let err = cache
        .import_string(&huge)
        .expect_err("oversized input must be rejected");
    assert!(matches!(err, WisdomError::TooLarge { kind: "bytes", .. }));
}

#[test]
fn test_merge_string_rejects_oversized_input() {
    let huge = "x".repeat(MAX_WISDOM_TEXT_BYTES + 1);
    let mut cache = WisdomCache::new();
    let err = cache
        .merge_string(&huge)
        .expect_err("oversized input must be rejected");
    assert!(matches!(err, WisdomError::TooLarge { kind: "bytes", .. }));
}

#[test]
fn test_import_string_rejects_too_many_entries() {
    // Build a small-in-bytes-but-many-entries wisdom string exceeding
    // MAX_WISDOM_TEXT_ENTRIES, using the shortest possible entry lines.
    let mut s = format!("({WISDOM_MARKER}\n  (format_version {WISDOM_FORMAT_VERSION})\n");
    for i in 1..=(MAX_WISDOM_TEXT_ENTRIES + 10) {
        s.push_str(&format!("  ({i} \"a\" 1)\n"));
    }
    s.push(')');

    let mut cache = WisdomCache::new();
    let err = cache
        .import_string(&s)
        .expect_err("too many entries must be rejected");
    assert!(matches!(
        err,
        WisdomError::TooLarge {
            kind: "entries",
            ..
        }
    ));
}

// ── import_system_wisdom error propagation ────────────────────────────────

#[cfg(feature = "std")]
#[test]
fn test_import_from_first_existing_propagates_real_error() {
    let dir = std::env::temp_dir();
    let corrupt_path = dir.join("oxifft_wisdom_test_corrupt_system_wisdom.txt");
    let missing_path = dir.join("oxifft_wisdom_test_definitely_missing_file_xyz.txt");

    // A candidate path that EXISTS but is not valid wisdom data at all.
    std::fs::write(&corrupt_path, b"not a wisdom file").expect("write failed");

    // Only the corrupt (existing) path is a candidate: the specific parse
    // error must be propagated, not masked as "not found".
    let err = import_from_first_existing(std::slice::from_ref(&corrupt_path))
        .expect_err("corrupt existing file must fail to import");
    assert!(
        matches!(err, WisdomError::ParseError(_)),
        "expected a ParseError from the corrupt file, got: {err}"
    );

    // A nonexistent-only candidate list still yields the generic NotFound.
    let err =
        import_from_first_existing(&[missing_path]).expect_err("no candidate existing must fail");
    assert!(matches!(err, WisdomError::IoError(_)));

    let _ = std::fs::remove_file(&corrupt_path);
}

#[cfg(feature = "std")]
#[test]
fn test_import_from_first_existing_incompatible_version_propagated() {
    let dir = std::env::temp_dir();
    let path = dir.join("oxifft_wisdom_test_future_version_system_wisdom.txt");

    let future_version = WISDOM_FORMAT_VERSION + 1;
    let content =
        format!("({WISDOM_MARKER}\n  (format_version {future_version})\n  (1 \"x\" 1.0)\n)");
    std::fs::write(&path, &content).expect("write failed");

    let err = import_from_first_existing(std::slice::from_ref(&path))
        .expect_err("future format version must fail to import");
    assert!(
        matches!(err, WisdomError::IncompatibleVersion { found, .. } if found == future_version),
        "expected IncompatibleVersion, got: {err}"
    );

    let _ = std::fs::remove_file(&path);
}

// ── Binary format ─────────────────────────────────────────────────────────

#[test]
fn test_binary_round_trip_seven_factor_mixed_radix() {
    // n = 2187 = 3^7 needs 7 mixed-radix stages — one more than the old
    // fixed `[u16; 6]` binary layout could hold. This must survive a
    // to_binary/from_binary round trip exactly under the v2 format.
    let mut cache = WisdomCache::new();
    cache.store(WisdomEntry {
        problem_hash: 2187,
        solver_name: "mixed-radix-3-3-3-3-3-3-3".to_string(),
        cost: 123.0,
    });

    let bytes = cache.to_binary();
    let restored = WisdomCache::from_binary(&bytes).expect("round-trip must succeed");
    let entry = restored
        .lookup(2187)
        .expect("entry for n=2187 must survive round-trip");
    assert_eq!(
        entry.solver_name, "mixed-radix-3-3-3-3-3-3-3",
        "all 7 factors must be preserved, not truncated to 6"
    );
}

#[test]
fn test_binary_round_trip_preserves_many_entries() {
    // Regression for the old `entry_count.min(u16::MAX)` truncation:
    // deliberately exceed u16::MAX (65_535) entries, which the old
    // format would have silently dropped down to. The new u32
    // entry-count field has no trouble representing this.
    let n: u64 = u64::from(u16::MAX) + 1000;
    let mut cache = WisdomCache::new();
    for i in 1..=n {
        cache.store(WisdomEntry {
            problem_hash: i,
            solver_name: "ct-dit".to_string(),
            cost: i as f64,
        });
    }
    let bytes = cache.to_binary();
    let restored = WisdomCache::from_binary(&bytes).expect("round-trip must succeed");
    assert_eq!(restored.entry_count(), n as usize);
}

#[test]
fn test_from_binary_rejects_bad_magic() {
    let mut bytes = WisdomCache::new().to_binary();
    bytes[0] = b'X';
    let err = WisdomCache::from_binary(&bytes).expect_err("bad magic must be rejected");
    assert!(matches!(err, WisdomError::ParseError(_)));
}

#[test]
fn test_from_binary_rejects_future_version() {
    let mut bytes = WisdomCache::new().to_binary();
    // Byte offset 8..10 is the format_version field.
    bytes[8] = 99;
    bytes[9] = 0;
    let err = WisdomCache::from_binary(&bytes).expect_err("future version must be rejected");
    assert!(matches!(
        err,
        WisdomError::IncompatibleVersion { found: 99, .. }
    ));
}

#[test]
fn test_from_binary_rejects_unrecognised_old_version() {
    let mut bytes = WisdomCache::new().to_binary();
    // Version 0 was never a valid binary format version (only the text
    // format has a legacy version 0).
    bytes[8] = 0;
    bytes[9] = 0;
    let err = WisdomCache::from_binary(&bytes).expect_err("version 0 must be rejected");
    assert!(matches!(err, WisdomError::ParseError(_)));
}

#[test]
fn test_from_binary_rejects_truncated_header() {
    let err = WisdomCache::from_binary(&[0u8; 10])
        .expect_err("data shorter than the 16-byte header must be rejected");
    assert!(matches!(err, WisdomError::ParseError(_)));
}

#[test]
fn test_from_binary_rejects_truncated_entries() {
    let mut cache = WisdomCache::new();
    cache.store(WisdomEntry {
        problem_hash: 16,
        solver_name: "ct-dit".to_string(),
        cost: 1.0,
    });
    let mut bytes = cache.to_binary();
    bytes.truncate(bytes.len() - 4); // chop off part of the last entry
    let err = WisdomCache::from_binary(&bytes).expect_err("truncated entry data must be rejected");
    assert!(matches!(err, WisdomError::ParseError(_)));
}

#[test]
fn test_from_binary_rejects_oversized_blob() {
    let huge = vec![0u8; MAX_WISDOM_BINARY_BYTES + 1];
    let err = WisdomCache::from_binary(&huge).expect_err("oversized blob must be rejected");
    assert!(matches!(err, WisdomError::TooLarge { kind: "bytes", .. }));
}

#[test]
fn test_from_binary_rejects_oversized_entry_count_header() {
    // A tiny blob whose header claims far more entries than
    // MAX_WISDOM_BINARY_ENTRIES must be rejected before any per-entry
    // decoding is attempted (no huge bounds-checked scan).
    let mut bytes = WisdomCache::new().to_binary();
    let bogus_count = (MAX_WISDOM_BINARY_ENTRIES as u32) + 1;
    bytes[12..16].copy_from_slice(&bogus_count.to_le_bytes());
    let err = WisdomCache::from_binary(&bytes)
        .expect_err("declared entry count over the ceiling must be rejected");
    assert!(matches!(
        err,
        WisdomError::TooLarge {
            kind: "entries",
            ..
        }
    ));
}

#[test]
fn test_from_binary_skips_degenerate_mixed_radix_factor() {
    // Hand-craft a v2 blob with one entry whose "mixed-radix" factor is
    // 0 — this is exactly the shape that used to reach
    // `algorithm_from_solver_name` and panic (integer divide-by-zero)
    // when replayed. It must now be silently skipped instead.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(BINARY_MAGIC);
    bytes.extend_from_slice(&2u16.to_le_bytes()); // format_version = 2
    bytes.extend_from_slice(&0u16.to_le_bytes()); // reserved
    bytes.extend_from_slice(&1u32.to_le_bytes()); // entry_count = 1

    bytes.extend_from_slice(&42u64.to_le_bytes()); // size_key
    bytes.push(ALGO_TAG_MIXED_RADIX);
    bytes.push(1); // factors_len
    bytes.extend_from_slice(&0u16.to_le_bytes()); // factor = 0 (degenerate)
    bytes.extend_from_slice(&1000u64.to_le_bytes()); // elapsed_ns

    let cache = WisdomCache::from_binary(&bytes)
        .expect("structurally well-formed data must decode without a hard error");
    assert!(
        cache.lookup(42).is_none(),
        "the degenerate mixed-radix entry must be skipped, not stored"
    );
}

#[test]
fn test_from_binary_skips_unsupported_radix_and_product_mismatch() {
    // Radix 6 is not in {2,3,4,5,7,8,16}; even though 6*7=42 matches the
    // size, the radix itself is invalid and must be rejected.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(BINARY_MAGIC);
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&42u64.to_le_bytes());
    bytes.push(ALGO_TAG_MIXED_RADIX);
    bytes.push(2);
    bytes.extend_from_slice(&6u16.to_le_bytes());
    bytes.extend_from_slice(&7u16.to_le_bytes());
    bytes.extend_from_slice(&1000u64.to_le_bytes());
    let cache = WisdomCache::from_binary(&bytes).expect("must not hard-error");
    assert!(
        cache.lookup(42).is_none(),
        "unsupported radix 6 must be rejected"
    );

    // Valid radices {2,7} but their product (14) does not match the
    // declared size (100) — must also be rejected.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(BINARY_MAGIC);
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&100u64.to_le_bytes());
    bytes.push(ALGO_TAG_MIXED_RADIX);
    bytes.push(2);
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&7u16.to_le_bytes());
    bytes.extend_from_slice(&1000u64.to_le_bytes());
    let cache = WisdomCache::from_binary(&bytes).expect("must not hard-error");
    assert!(
        cache.lookup(100).is_none(),
        "factor product (14) mismatching the declared size (100) must be rejected"
    );
}

#[test]
fn test_from_binary_skips_zero_hash_and_unknown_tag() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(BINARY_MAGIC);
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes()); // entry_count = 2

    // Entry 1: hash == 0 (invalid, matches text-format rule).
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.push(ALGO_TAG_COOLEY_TUKEY);
    bytes.push(0);
    bytes.extend_from_slice(&1u64.to_le_bytes());

    // Entry 2: unrecognised algo_tag (255 = Unknown).
    bytes.extend_from_slice(&7u64.to_le_bytes());
    bytes.push(ALGO_TAG_UNKNOWN);
    bytes.push(0);
    bytes.extend_from_slice(&1u64.to_le_bytes());

    let cache = WisdomCache::from_binary(&bytes).expect("must not hard-error");
    assert_eq!(cache.entry_count(), 0, "both entries must be skipped");
}

#[test]
fn test_import_binary_reports_skipped_invalid() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(BINARY_MAGIC);
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());

    // Entry 1: valid ct-dit entry.
    bytes.extend_from_slice(&16u64.to_le_bytes());
    bytes.push(ALGO_TAG_COOLEY_TUKEY);
    bytes.push(0);
    bytes.extend_from_slice(&500u64.to_le_bytes());

    // Entry 2: invalid (hash == 0).
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.push(ALGO_TAG_COOLEY_TUKEY);
    bytes.push(0);
    bytes.extend_from_slice(&500u64.to_le_bytes());

    let mut cache = WisdomCache::new();
    let result = cache.import_binary(&bytes).expect("import must succeed");
    assert_eq!(result.imported, 1);
    assert_eq!(result.skipped_invalid, 1);
    assert_eq!(result.format_version, 2);
    assert!(cache.lookup(16).is_some());
}

#[test]
fn test_from_binary_reads_legacy_v1_format() {
    // Hand-craft a version-1 blob (fixed [u16; 6] factors, u16 entry
    // count) to verify backward-compat reading of files written by
    // OxiFFT <= 0.3.x still works under the new decoder.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(BINARY_MAGIC);
    bytes.extend_from_slice(&1u16.to_le_bytes()); // format_version = 1
    bytes.extend_from_slice(&1u16.to_le_bytes()); // entry_count = 1
    bytes.extend_from_slice(&0u32.to_le_bytes()); // reserved

    bytes.extend_from_slice(&6u64.to_le_bytes()); // size_key = 6 = 2*3
    bytes.push(ALGO_TAG_MIXED_RADIX);
    bytes.push(2); // factors_len
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&3u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&12345u64.to_le_bytes()); // elapsed_ns

    let cache = WisdomCache::from_binary(&bytes).expect("legacy v1 blob must decode");
    let entry = cache.lookup(6).expect("entry must be present");
    assert_eq!(entry.solver_name, "mixed-radix-2-3");
    assert!((entry.cost - 12345.0).abs() < f64::EPSILON);
}

#[test]
fn test_to_binary_checked_rejects_too_many_factors() {
    // Craft a solver name with more dash-separated numeric factors than
    // MAX_BINARY_FACTORS can represent; `to_binary_checked` must return
    // an error rather than silently truncating (which is exactly the
    // v1 bug this module fixes, just pushed to a much larger bound).
    let factors: Vec<String> = (0..=MAX_BINARY_FACTORS).map(|_| "2".to_string()).collect();
    let mut cache = WisdomCache::new();
    cache.store(WisdomEntry {
        problem_hash: 1,
        solver_name: format!("mixed-radix-{}", factors.join("-")),
        cost: 1.0,
    });

    let err = cache
        .to_binary_checked()
        .expect_err("factor list beyond MAX_BINARY_FACTORS must be rejected");
    assert!(matches!(
        err,
        WisdomError::TooLarge {
            kind: "factors",
            ..
        }
    ));

    // The infallible convenience form must still never panic: it caps
    // instead of erroring.
    let bytes = cache.to_binary();
    assert!(!bytes.is_empty());
}

#[test]
fn test_is_valid_mixed_radix_factors() {
    assert!(is_valid_mixed_radix_factors(&[2, 3], 6));
    assert!(is_valid_mixed_radix_factors(&[3, 3, 3, 3, 3, 3, 3], 2187));
    assert!(
        !is_valid_mixed_radix_factors(&[], 6),
        "empty factors invalid"
    );
    assert!(
        !is_valid_mixed_radix_factors(&[0], 0),
        "zero factor invalid"
    );
    assert!(
        !is_valid_mixed_radix_factors(&[6], 6),
        "radix 6 is not supported"
    );
    assert!(
        !is_valid_mixed_radix_factors(&[2, 3], 7),
        "product must match size exactly"
    );
    // Must not panic even with values chosen to stress the overflow guard.
    assert!(!is_valid_mixed_radix_factors(&[16; 20], u64::MAX));
}

// ── Utility ───────────────────────────────────────────────────────────────

#[test]
fn test_wisdom_clear() {
    let mut cache = WisdomCache::new();
    cache.store(WisdomEntry {
        problem_hash: 1,
        solver_name: "test".to_string(),
        cost: 1.0,
    });
    assert!(!cache.is_empty());

    cache.clear();
    assert!(cache.is_empty());
}

#[cfg(feature = "std")]
#[test]
fn test_user_wisdom_path() {
    // Just verify it doesn't panic
    let _path = get_user_wisdom_path();
}
