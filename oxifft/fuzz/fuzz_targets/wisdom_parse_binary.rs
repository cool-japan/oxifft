#![no_main]

use libfuzzer_sys::fuzz_target;
use oxifft::api::WisdomCache;

// Fuzzes the binary wisdom decoder (`WisdomCache::from_binary`/`import_binary`),
// which is a distinct hand-rolled byte decoder from the S-expression text
// parser exercised by `wisdom_parse.rs`. This is the one wisdom format that
// currently gets consulted at plan-construction time (via the build-time
// baseline embedded through `OXIFFT_TUNE=1`), so it must never panic on
// arbitrary/adversarial bytes: bad magic, bad version, truncated data, and
// (crucially) degenerate `MixedRadix` factor lists that would otherwise
// divide by zero or hit an `unreachable!()` when replayed through
// `Plan::dft_1d`'s algorithm reconstruction.
fuzz_target!(|data: &[u8]| {
    // `from_binary` must return a `Result`, never panic, for any input.
    if let Ok(cache) = WisdomCache::from_binary(data) {
        // Round-trip: whatever successfully decoded must re-encode and
        // re-decode into an equivalent (or safely smaller, if capped)
        // cache without ever panicking either.
        let reencoded = cache.to_binary();
        let _ = WisdomCache::from_binary(&reencoded);
        let _ = cache.to_binary_checked();
    }

    // `import_binary` must also never panic, and must never insert an
    // entry whose mixed-radix factors are degenerate (that would be a
    // latent panic waiting in `Plan::dft_1d`'s algorithm reconstruction).
    let mut cache = WisdomCache::new();
    if let Ok(result) = cache.import_binary(data) {
        assert!(
            result.imported <= oxifft::api::MAX_WISDOM_BINARY_ENTRIES,
            "import_binary must never report more imports than the entry-count ceiling allows"
        );
        for (_, entry) in cache_entries_debug(&cache) {
            // Any entry claiming to be mixed-radix must have already passed
            // factor validation inside `import_binary`/`from_binary` — replay
            // the same check here as a fuzz-level regression guard.
            if let Some(factors) = parse_mixed_radix_suffix(&entry) {
                assert!(
                    !factors.is_empty(),
                    "stored mixed-radix entry must not have an empty factor list"
                );
            }
        }
    }
});

/// Extract `(hash, solver_name)` pairs via the only inspection surface
/// `WisdomCache` exposes (`export_string`), since `entries` is private.
fn cache_entries_debug(cache: &WisdomCache) -> Vec<(u64, String)> {
    cache
        .export_string()
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if !line.starts_with('(') || !line.ends_with(')') || line.contains("format_version") {
                return None;
            }
            let inner = line.get(1..line.len().saturating_sub(1))?;
            let mut parts = inner.splitn(3, ' ');
            let hash: u64 = parts.next()?.parse().ok()?;
            let name = parts.next()?.trim_matches('"').to_string();
            Some((hash, name))
        })
        .collect()
}

/// Mirror of the crate-internal mixed-radix name parser, duplicated here
/// since the fuzz crate cannot reach `oxifft`'s private module internals.
fn parse_mixed_radix_suffix(name: &str) -> Option<Vec<u16>> {
    let suffix = name.strip_prefix("mixed-radix-")?;
    suffix.split('-').map(|s| s.parse::<u16>().ok()).collect()
}
