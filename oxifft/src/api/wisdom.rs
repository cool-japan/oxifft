//! Wisdom management for plan caching.
//!
//! The wisdom system caches optimal plans for specific problem sizes,
//! avoiding the cost of re-measuring algorithms on subsequent runs.
//!
//! # Format Versioning
//!
//! Wisdom files carry a `format_version` field so that future OxiFFT versions
//! can detect incompatibility:
//!
//! - **Version 0 (legacy)**: header `(oxifft-wisdom-1.0 …)`, no `(format_version …)` line.
//! - **Version 1**: `(oxifft-wisdom` header with `(format_version 1)`. The `MixedRadix`
//!   solver was stored as the literal string `"MixedRadix"` (a stub; not reconstructible).
//! - **Version 2 (current, [`WISDOM_FORMAT_VERSION`])**: `MixedRadix` is encoded as
//!   `"mixed-radix-R1-R2-..."` (innermost-first factor sequence), so the solver can be
//!   fully reconstructed from the stored name.
//!
//! Importing wisdom whose `format_version` is *greater* than
//! [`WISDOM_FORMAT_VERSION`] returns
//! [`WisdomError::IncompatibleVersion`].  A lower or equal version is accepted
//! (with silent best-effort parsing).
//!
//! See `docs/wisdom_format.md` in the repository root for the full grammar,
//! validation rules, and the binary format (`to_binary`/`from_binary`) spec.
//!
//! # Merge Semantics
//!
//! [`merge_from_string`] (and its file counterpart) can combine wisdom gathered
//! on different machines or runs.  When the same problem hash appears in both
//! the current cache and the incoming data, the entry with the **lower cost**
//! (faster measured time) is kept.
//!
//! # Untrusted input
//!
//! [`WisdomCache::import_string`]/[`merge_string`](WisdomCache::merge_string) and
//! [`WisdomCache::from_binary`]/[`import_binary`](WisdomCache::import_binary) enforce
//! byte-size and entry-count ceilings ([`MAX_WISDOM_TEXT_BYTES`], [`MAX_WISDOM_TEXT_ENTRIES`],
//! [`MAX_WISDOM_BINARY_BYTES`], [`MAX_WISDOM_BINARY_ENTRIES`]) before parsing, returning
//! [`WisdomError::TooLarge`] rather than spending unbounded CPU/memory on hostile input.
//!
//! # Example (requires `std` feature)
//!
//! ```rust,no_run
//! # #[cfg(feature = "std")]
//! # {
//! use oxifft::api::{fft, import_from_file, export_to_file};
//! use std::path::Path;
//!
//! // Import wisdom from previous runs
//! let _ = import_from_file(Path::new("wisdom.txt"));
//!
//! // Run FFTs - they may use cached wisdom
//! // ...
//!
//! // Export accumulated wisdom for future runs
//! let _ = export_to_file(Path::new("wisdom.txt"));
//! # }
//! ```

use crate::kernel::{Planner, WisdomEntry};
use crate::prelude::*;

// ─── Constants ───────────────────────────────────────────────────────────────

/// Marker token that begins every wisdom string.
const WISDOM_MARKER: &str = "oxifft-wisdom";

/// Legacy header produced by `Planner::wisdom_export()` (format version 0).
const WISDOM_LEGACY_HEADER: &str = "oxifft-wisdom-1.0";

/// Current wisdom format version.
///
/// Increment this constant when the serialisation format changes in a
/// backwards-incompatible way.
///
/// Version history:
/// - 1: Initial format. MixedRadix solver stored as literal `"MixedRadix"` (stub, unimplemented).
/// - 2: MixedRadix encoded as `"mixed-radix-R1-R2-..."` (innermost-first factor sequence).
///   The planner now selects MixedRadix for smooth-7 composite sizes.
pub const WISDOM_FORMAT_VERSION: u32 = 2;

/// Maximum size, in bytes, of a **text**-format wisdom blob accepted by
/// [`WisdomCache::import_string`]/[`WisdomCache::merge_string`] (and the
/// corresponding free functions and file-backed wrappers).
///
/// Guards against unbounded memory/CPU consumption when parsing wisdom data
/// from an untrusted source (a file supplied by an untrusted user, or
/// fetched over a network). Input larger than this is rejected up front with
/// [`WisdomError::TooLarge`] instead of being parsed.
pub const MAX_WISDOM_TEXT_BYTES: usize = 64 * 1024 * 1024; // 64 MiB

/// Maximum number of entry lines accepted from a single text-format wisdom
/// import/merge operation.
///
/// Parsing aborts with [`WisdomError::TooLarge`] once this many entry lines
/// have been encountered, even if the input is smaller than
/// [`MAX_WISDOM_TEXT_BYTES`] (a small file can still contain an unreasonable
/// number of very short lines).
pub const MAX_WISDOM_TEXT_ENTRIES: usize = 1_000_000;

/// Maximum size, in bytes, of a **binary**-format wisdom blob accepted by
/// [`WisdomCache::from_binary`]/[`WisdomCache::import_binary`].
pub const MAX_WISDOM_BINARY_BYTES: usize = 64 * 1024 * 1024; // 64 MiB

/// Maximum number of entries a binary wisdom blob's header may declare.
///
/// Headers claiming more than this are rejected with
/// [`WisdomError::TooLarge`] before any per-entry decoding is attempted, so a
/// tiny hostile blob cannot claim billions of entries and force a large
/// bounds-checked scan.
pub const MAX_WISDOM_BINARY_ENTRIES: usize = 1_000_000;

// ─── Result types ────────────────────────────────────────────────────────────

/// Statistics returned from a wisdom import operation.
///
/// Returned by [`import_from_string`], [`import_from_file`], and the
/// corresponding [`WisdomCache::import_string`] method.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct WisdomImportResult {
    /// Number of entries that were successfully imported.
    pub imported: usize,
    /// Number of entries that were skipped due to invalid data.
    pub skipped_invalid: usize,
    /// Format version found in the wisdom data.
    pub format_version: u32,
}

/// Statistics returned from a wisdom merge operation.
///
/// Returned by [`merge_from_string`], [`merge_from_file`], and the
/// corresponding [`WisdomCache::merge_string`] method.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct WisdomMergeResult {
    /// Number of entries inserted because they were absent from the cache.
    pub added: usize,
    /// Number of entries from the incoming data that replaced existing ones
    /// because they had a lower cost.
    pub replaced: usize,
    /// Number of entries from the incoming data that were discarded because
    /// the existing cache entry already had a lower or equal cost.
    pub kept_existing: usize,
    /// Number of entries skipped because of invalid / corrupt data.
    pub skipped_invalid: usize,
    /// Format version found in the wisdom data.
    pub format_version: u32,
}

// ─── Cache ───────────────────────────────────────────────────────────────────

/// Global wisdom cache, shared across all planners.
static GLOBAL_WISDOM: RwLock<Option<WisdomCache>> = RwLock::new(None);

/// A cache of wisdom entries.
#[derive(Debug, Clone, Default)]
pub struct WisdomCache {
    /// Map from problem hash to wisdom entry.
    entries: HashMap<u64, WisdomEntry>,
}

impl WisdomCache {
    /// Create a new empty wisdom cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Get the number of entries in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a wisdom entry by problem hash.
    #[must_use]
    pub fn lookup(&self, hash: u64) -> Option<&WisdomEntry> {
        self.entries.get(&hash)
    }

    /// Store a wisdom entry.
    pub fn store(&mut self, entry: WisdomEntry) {
        self.entries.insert(entry.problem_hash, entry);
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Import wisdom from a planner.
    pub fn import_from_planner<T: crate::kernel::Float>(&mut self, planner: &Planner<T>) {
        let exported = planner.wisdom_export();
        let _ = self.import_string(&exported);
    }

    /// Export wisdom to a planner.
    pub fn export_to_planner<T: crate::kernel::Float>(&self, planner: &mut Planner<T>) {
        let exported = self.export_string();
        let _ = planner.wisdom_import(&exported);
    }

    /// Export wisdom to a string (version 1 format).
    ///
    /// The returned string begins with `(oxifft-wisdom`, followed by a
    /// `(format_version N)` line, then one `(hash "solver" cost)` line per
    /// entry, and ends with `)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxifft::api::WisdomCache;
    ///
    /// let cache = WisdomCache::new();
    /// let s = cache.export_string();
    /// assert!(s.contains("oxifft-wisdom"));
    /// assert!(s.contains("format_version"));
    /// ```
    #[must_use]
    pub fn export_string(&self) -> String {
        use core::fmt::Write;
        let mut result = format!("({WISDOM_MARKER}\n  (format_version {WISDOM_FORMAT_VERSION})\n");
        for entry in self.entries.values() {
            let _ = writeln!(
                result,
                "  ({} \"{}\" {})",
                entry.problem_hash, entry.solver_name, entry.cost
            );
        }
        result.push(')');
        result
    }

    /// Import wisdom from a string, with format version negotiation and
    /// per-entry validation.
    ///
    /// Entries that fail validation (zero hash, empty solver name, non-finite
    /// or negative cost) are silently skipped; the import continues with the
    /// remaining entries.
    ///
    /// # Errors
    ///
    /// - [`WisdomError::IncompatibleVersion`] when `format_version` in the
    ///   data is greater than [`WISDOM_FORMAT_VERSION`].
    /// - [`WisdomError::ParseError`] when the overall structure is
    ///   unrecognisable (not even a legacy wisdom header).
    pub fn import_string(&mut self, s: &str) -> Result<WisdomImportResult, WisdomError> {
        let s = s.trim();

        if s.len() > MAX_WISDOM_TEXT_BYTES {
            return Err(WisdomError::TooLarge {
                kind: "bytes",
                limit: MAX_WISDOM_TEXT_BYTES,
                actual: s.len(),
            });
        }

        // Detect format version from the header line.
        let format_version = detect_format_version(s)?;

        // Refuse future formats we don't know how to parse.
        if format_version > WISDOM_FORMAT_VERSION {
            return Err(WisdomError::IncompatibleVersion {
                found: format_version,
                expected: WISDOM_FORMAT_VERSION,
            });
        }

        let mut imported = 0usize;
        let mut skipped_invalid = 0usize;
        let mut entry_lines_seen = 0usize;

        for line in s.lines().skip(1) {
            let line = line.trim();
            if !is_entry_line(line) {
                continue;
            }

            entry_lines_seen += 1;
            if entry_lines_seen > MAX_WISDOM_TEXT_ENTRIES {
                return Err(WisdomError::TooLarge {
                    kind: "entries",
                    limit: MAX_WISDOM_TEXT_ENTRIES,
                    actual: entry_lines_seen,
                });
            }

            match parse_entry_line(line) {
                Some(entry) if is_valid_entry(&entry) => {
                    self.entries.insert(entry.problem_hash, entry);
                    imported += 1;
                }
                Some(_) => {
                    skipped_invalid += 1;
                }
                None => {
                    // Malformed line but not a fatal error — skip silently.
                    skipped_invalid += 1;
                }
            }
        }

        Ok(WisdomImportResult {
            imported,
            skipped_invalid,
            format_version,
        })
    }

    /// Merge incoming wisdom into this cache.
    ///
    /// For every entry in `s`:
    /// - If the hash is absent from the cache, insert it.
    /// - If the hash is present but the incoming entry has a **lower** cost
    ///   (better measured performance), replace the existing entry.
    /// - Otherwise keep the existing entry.
    ///
    /// Entries with invalid data are silently skipped.
    ///
    /// # Errors
    ///
    /// - [`WisdomError::IncompatibleVersion`] when `format_version` in the
    ///   data is greater than [`WISDOM_FORMAT_VERSION`].
    /// - [`WisdomError::ParseError`] when the overall structure is
    ///   unrecognisable.
    pub fn merge_string(&mut self, s: &str) -> Result<WisdomMergeResult, WisdomError> {
        let s = s.trim();

        if s.len() > MAX_WISDOM_TEXT_BYTES {
            return Err(WisdomError::TooLarge {
                kind: "bytes",
                limit: MAX_WISDOM_TEXT_BYTES,
                actual: s.len(),
            });
        }

        let format_version = detect_format_version(s)?;

        if format_version > WISDOM_FORMAT_VERSION {
            return Err(WisdomError::IncompatibleVersion {
                found: format_version,
                expected: WISDOM_FORMAT_VERSION,
            });
        }

        let mut added = 0usize;
        let mut replaced = 0usize;
        let mut kept_existing = 0usize;
        let mut skipped_invalid = 0usize;
        let mut entry_lines_seen = 0usize;

        for line in s.lines().skip(1) {
            let line = line.trim();
            if !is_entry_line(line) {
                continue;
            }

            entry_lines_seen += 1;
            if entry_lines_seen > MAX_WISDOM_TEXT_ENTRIES {
                return Err(WisdomError::TooLarge {
                    kind: "entries",
                    limit: MAX_WISDOM_TEXT_ENTRIES,
                    actual: entry_lines_seen,
                });
            }

            match parse_entry_line(line) {
                Some(entry) if is_valid_entry(&entry) => {
                    match self.entries.get(&entry.problem_hash) {
                        None => {
                            self.entries.insert(entry.problem_hash, entry);
                            added += 1;
                        }
                        Some(existing) if entry.cost < existing.cost => {
                            self.entries.insert(entry.problem_hash, entry);
                            replaced += 1;
                        }
                        Some(_) => {
                            kept_existing += 1;
                        }
                    }
                }
                _ => {
                    skipped_invalid += 1;
                }
            }
        }

        Ok(WisdomMergeResult {
            added,
            replaced,
            kept_existing,
            skipped_invalid,
            format_version,
        })
    }
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Detect the format version encoded in a wisdom string.
///
/// Accepts both the legacy `(oxifft-wisdom-1.0 …)` header (returns 0) and the
/// current `(oxifft-wisdom` + `(format_version N)` form.
///
/// Returns [`WisdomError::ParseError`] when no recognisable header is found.
fn detect_format_version(s: &str) -> Result<u32, WisdomError> {
    let first_line = s.lines().next().unwrap_or("").trim();

    // Legacy format: "(oxifft-wisdom-1.0"
    if first_line.starts_with(&format!("({WISDOM_LEGACY_HEADER}")) {
        return Ok(0);
    }

    // Current format: "(oxifft-wisdom" (without the "-1.0" suffix)
    if first_line.starts_with(&format!("({WISDOM_MARKER}")) {
        // Search for "(format_version N)" among the first few lines.
        for line in s.lines().skip(1).take(5) {
            let line = line.trim();
            if let Some(ver) = parse_format_version_line(line) {
                return Ok(ver);
            }
        }
        // Header recognised but no version line found — treat as version 1.
        return Ok(1);
    }

    Err(WisdomError::ParseError(
        "missing oxifft-wisdom header".to_string(),
    ))
}

/// Parse a `(format_version N)` line, returning `N` on success.
fn parse_format_version_line(line: &str) -> Option<u32> {
    let line = line.trim();
    if !line.starts_with("(format_version ") || !line.ends_with(')') {
        return None;
    }
    let inner = &line["(format_version ".len()..line.len() - 1];
    inner.trim().parse::<u32>().ok()
}

/// True if a wisdom line looks like a data entry (`(hash "solver" cost)`).
fn is_entry_line(line: &str) -> bool {
    line.starts_with('(')
        && line.ends_with(')')
        && !line.starts_with(&format!("({WISDOM_MARKER}"))
        && !line.starts_with(&format!("({WISDOM_LEGACY_HEADER}"))
        && !line.starts_with("(format_version ")
}

/// Attempt to parse `(hash "solver" cost)` into a [`WisdomEntry`].
fn parse_entry_line(line: &str) -> Option<WisdomEntry> {
    let inner = line.get(1..line.len().checked_sub(1)?)?;
    let parts: Vec<&str> = inner.split_whitespace().collect();
    if parts.len() < 3 {
        return None;
    }
    let hash = parts[0].parse::<u64>().ok()?;
    let solver_name = parts[1].trim_matches('"').to_string();
    let cost = parts[2].parse::<f64>().ok()?;
    Some(WisdomEntry {
        problem_hash: hash,
        solver_name,
        cost,
    })
}

/// Validate that a [`WisdomEntry`] contains sensible data.
///
/// An entry is considered invalid when:
/// - `problem_hash == 0`
/// - `solver_name` is empty
/// - `cost` is NaN, infinite, or negative
/// - `solver_name` uses the `"mixed-radix-R1-R2-..."` naming convention but the
///   factors are not all supported radices, or their product does not equal
///   `problem_hash` (see [`is_valid_mixed_radix_factors`])
fn is_valid_entry(entry: &WisdomEntry) -> bool {
    if entry.problem_hash == 0
        || entry.solver_name.is_empty()
        || !entry.cost.is_finite()
        || entry.cost < 0.0
    {
        return false;
    }
    match parse_mixed_radix_name(&entry.solver_name) {
        Some(factors) => is_valid_mixed_radix_factors(&factors, entry.problem_hash),
        None => true,
    }
}

/// Radices the mixed-radix DIT engine (`api::plan::types::execute_mixed_radix_inplace`)
/// supports. Duplicated here (rather than imported from `api::plan::types`) so that
/// wisdom validation has no dependency on plan-construction internals — this module
/// must be able to reject corrupt mixed-radix wisdom on its own.
const MIXED_RADIX_VALID_RADICES: [u16; 7] = [2, 3, 4, 5, 7, 8, 16];

/// Parse a `"mixed-radix-R1-R2-..."` solver name into its ordered factor list.
///
/// Returns `None` when `name` does not use the mixed-radix naming convention,
/// or when any dash-separated component fails to parse as a `u16` (an
/// all-or-nothing parse: a name with even one malformed component is treated
/// as not being a well-formed mixed-radix name at all, rather than producing
/// a partially-populated factor list).
fn parse_mixed_radix_name(name: &str) -> Option<Vec<u16>> {
    let suffix = name.strip_prefix("mixed-radix-")?;
    suffix.split('-').map(|s| s.parse::<u16>().ok()).collect()
}

/// Validate a decoded mixed-radix factor list against the wisdom entry's
/// declared problem size.
///
/// Returns `true` only when `factors` is non-empty, every factor is one of
/// the radices the mixed-radix DIT engine supports (`{2,3,4,5,7,8,16}`), and
/// — using saturating/checked arithmetic so this can never panic on
/// attacker-controlled input — their product equals `size` exactly.
///
/// This check exists because the only code path that ever reconstructs an
/// executable algorithm from a stored solver name
/// (`api::plan::types::algorithm_from_solver_name`) trusts the factor list
/// completely: a zero factor divides by zero, and an unsupported radix hits
/// an `unreachable!()`, inside the mixed-radix execution path. Wisdom data
/// that fails this check must never be allowed to reach that reconstruction,
/// so both `is_valid_entry` (text format) and the binary decoder reject it
/// here, before it is ever stored in a cache.
///
/// `pub(crate)` so `api::plan::types::algorithm_from_solver_name` can reuse
/// this exact check as defense-in-depth at the reconstruction site itself,
/// instead of duplicating (and risking divergence from) this validation.
// reason: clippy's suggestion to use plain `pub` is wrong here — `api::mod`
// does `pub use wisdom::*;`, so a genuinely `pub` item in this (privately
// declared) module WOULD leak into the crate's public API, whereas
// `pub(crate)` is capped at crate-visibility even through that glob
// re-export (verified: an external-crate-style caller gets E0603 "private
// function" against `pub(crate)`, but would succeed against plain `pub`).
#[allow(clippy::redundant_pub_crate)]
pub(crate) fn is_valid_mixed_radix_factors(factors: &[u16], size: u64) -> bool {
    if factors.is_empty() {
        return false;
    }
    let mut product: u128 = 1;
    for &f in factors {
        if !MIXED_RADIX_VALID_RADICES.contains(&f) {
            return false;
        }
        product = product.saturating_mul(u128::from(f));
        if product > u128::from(u64::MAX) {
            return false;
        }
    }
    product == u128::from(size)
}

// ─── Global cache initialisation ─────────────────────────────────────────────

/// Initialize global wisdom if not already initialized.
///
/// # Poisoning
///
/// A `std::sync::RwLock` is poisoned when a panic unwinds while a guard is
/// held. Since `GLOBAL_WISDOM` is shared process-wide, treating poisoning as
/// fatal (as earlier versions of this module did via `.expect(...)`) would
/// mean a single unrelated panic on any thread permanently breaks every
/// future wisdom operation for the lifetime of the process. The cached data
/// here is purely advisory (a cache of previously measured plan choices) and
/// safe to keep using even if it was concurrently modified during a panic,
/// so we recover the guard instead: `PoisonError::into_inner()` returns the
/// underlying data unchanged, and normal operation resumes immediately.
fn ensure_global_wisdom() {
    #[cfg(feature = "std")]
    {
        let needs_init = GLOBAL_WISDOM
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_none();
        if needs_init {
            let mut write_guard = GLOBAL_WISDOM
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            write_guard.get_or_insert_with(WisdomCache::new);
        }
    }
    #[cfg(not(feature = "std"))]
    {
        // `spin::RwLock` has no poisoning concept.
        let needs_init = GLOBAL_WISDOM.read().is_none();
        if needs_init {
            let mut write_guard = GLOBAL_WISDOM.write();
            write_guard.get_or_insert_with(WisdomCache::new);
        }
    }
}

/// Get access to the global wisdom cache for reading.
fn with_wisdom<F, R>(f: F) -> R
where
    F: FnOnce(&WisdomCache) -> R,
{
    ensure_global_wisdom();
    #[cfg(feature = "std")]
    {
        let guard = GLOBAL_WISDOM
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match guard.as_ref() {
            Some(cache) => f(cache),
            // Defense in depth: even after `ensure_global_wisdom()` recovers
            // from poisoning, a pathological interleaving could in theory
            // leave the slot empty. Fall back to a fresh, empty cache rather
            // than panicking — wisdom lookups must never become fatal.
            None => f(&WisdomCache::new()),
        }
    }
    #[cfg(not(feature = "std"))]
    {
        let guard = GLOBAL_WISDOM.read();
        match guard.as_ref() {
            Some(cache) => f(cache),
            None => f(&WisdomCache::new()),
        }
    }
}

/// Get access to the global wisdom cache for writing.
fn with_wisdom_mut<F, R>(f: F) -> R
where
    F: FnOnce(&mut WisdomCache) -> R,
{
    ensure_global_wisdom();
    #[cfg(feature = "std")]
    {
        let mut guard = GLOBAL_WISDOM
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        f(guard.get_or_insert_with(WisdomCache::new))
    }
    #[cfg(not(feature = "std"))]
    {
        let mut guard = GLOBAL_WISDOM.write();
        f(guard.get_or_insert_with(WisdomCache::new))
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Export current wisdom to a string.
///
/// Returns a string representation of all accumulated wisdom that can be
/// saved to a file or transmitted to another process.
///
/// # Example
///
/// ```rust
/// use oxifft::api::export_to_string;
///
/// let wisdom = export_to_string();
/// println!("{wisdom}");
/// ```
#[must_use]
pub fn export_to_string() -> String {
    with_wisdom(WisdomCache::export_string)
}

/// Import wisdom from a string.
///
/// Entries with invalid data are skipped silently; the returned
/// [`WisdomImportResult`] reports how many were imported and how many were
/// skipped.
///
/// # Arguments
/// * `s` - The wisdom string to import
///
/// # Errors
/// Returns an error if the wisdom string is structurally unrecognisable or its
/// `format_version` is newer than [`WISDOM_FORMAT_VERSION`].
///
/// # Example
///
/// ```rust
/// use oxifft::api::import_from_string;
///
/// let wisdom_str = "(oxifft-wisdom\n  (format_version 1)\n)";
/// let result = import_from_string(wisdom_str).unwrap();
/// assert_eq!(result.format_version, 1);
/// ```
pub fn import_from_string(s: &str) -> Result<WisdomImportResult, WisdomError> {
    with_wisdom_mut(|cache| cache.import_string(s))
}

/// Merge incoming wisdom into the global cache.
///
/// For each entry in the incoming wisdom string:
/// - If the problem hash is not yet in the global cache, it is inserted.
/// - If it is already present and the incoming cost is lower, the existing
///   entry is replaced.
/// - Otherwise the existing entry is kept.
///
/// # Errors
/// Returns an error if the wisdom string is structurally unrecognisable or its
/// `format_version` is newer than [`WISDOM_FORMAT_VERSION`].
///
/// # Example
///
/// ```rust
/// use oxifft::api::{export_to_string, merge_from_string, forget};
///
/// forget();
/// let wisdom = export_to_string();
/// let result = merge_from_string(&wisdom).unwrap();
/// assert_eq!(result.added, 0);
/// ```
pub fn merge_from_string(s: &str) -> Result<WisdomMergeResult, WisdomError> {
    with_wisdom_mut(|cache| cache.merge_string(s))
}

/// Export wisdom to a file.
///
/// # Arguments
/// * `path` - Path to the file to write wisdom to
///
/// # Errors
/// Returns an error if the file cannot be written.
///
/// # Example
///
/// ```rust,no_run
/// use oxifft::api::export_to_file;
/// use std::path::Path;
///
/// export_to_file(Path::new("wisdom.txt")).unwrap();
/// ```
#[cfg(feature = "std")]
pub fn export_to_file(path: &std::path::Path) -> std::io::Result<()> {
    let wisdom = export_to_string();
    std::fs::write(path, wisdom)
}

/// Check a file's size against [`MAX_WISDOM_TEXT_BYTES`] before reading it in
/// full, so a hostile multi-gigabyte "wisdom file" cannot force a large
/// allocation before any validation happens.
#[cfg(feature = "std")]
fn check_wisdom_file_size(path: &std::path::Path) -> Result<(), WisdomError> {
    let len = std::fs::metadata(path)?.len();
    if len > MAX_WISDOM_TEXT_BYTES as u64 {
        return Err(WisdomError::TooLarge {
            kind: "bytes",
            limit: MAX_WISDOM_TEXT_BYTES,
            actual: usize::try_from(len).unwrap_or(usize::MAX),
        });
    }
    Ok(())
}

/// Import wisdom from a file.
///
/// # Arguments
/// * `path` - Path to the file to read wisdom from
///
/// # Errors
/// Returns an error if the file cannot be read, exceeds
/// [`MAX_WISDOM_TEXT_BYTES`], is structurally unrecognisable, or its
/// `format_version` is newer than [`WISDOM_FORMAT_VERSION`].
///
/// # Example
///
/// ```rust,no_run
/// use oxifft::api::import_from_file;
/// use std::path::Path;
///
/// import_from_file(Path::new("wisdom.txt")).unwrap();
/// ```
#[cfg(feature = "std")]
pub fn import_from_file(path: &std::path::Path) -> Result<WisdomImportResult, WisdomError> {
    check_wisdom_file_size(path)?;
    let contents = std::fs::read_to_string(path)?;
    import_from_string(&contents)
}

/// Merge wisdom from a file into the global cache.
///
/// Combines the file's wisdom with the currently cached data, keeping the
/// lower-cost (better) entry for any hash that appears in both.
///
/// # Errors
/// Returns an error if the file cannot be read, exceeds
/// [`MAX_WISDOM_TEXT_BYTES`], is structurally unrecognisable, or its
/// `format_version` is newer than [`WISDOM_FORMAT_VERSION`].
///
/// # Example
///
/// ```rust,no_run
/// use oxifft::api::merge_from_file;
/// use std::path::Path;
///
/// merge_from_file(Path::new("extra_wisdom.txt")).unwrap();
/// ```
#[cfg(feature = "std")]
pub fn merge_from_file(path: &std::path::Path) -> Result<WisdomMergeResult, WisdomError> {
    check_wisdom_file_size(path)?;
    let contents = std::fs::read_to_string(path)?;
    merge_from_string(&contents)
}

/// Import wisdom from the first candidate in `paths` that exists, propagating
/// the real error from that candidate (rather than a generic "not found") if
/// it exists but fails to import.
///
/// Factored out of [`import_system_wisdom`] so the path-selection/error
/// propagation logic can be exercised directly in tests with temporary
/// files, independent of the OS-specific real system paths.
#[cfg(feature = "std")]
fn import_from_first_existing(
    paths: &[std::path::PathBuf],
) -> Result<WisdomImportResult, WisdomError> {
    let mut last_error: Option<WisdomError> = None;

    for path in paths {
        if !path.exists() {
            continue;
        }
        match import_from_file(path) {
            Ok(result) => return Ok(result),
            Err(e) => {
                // Remember the most specific error so a candidate that DOES
                // exist but is corrupt/incompatible is reported accurately
                // instead of being masked by a generic "not found" fallback.
                last_error = Some(e);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        WisdomError::IoError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No system wisdom found",
        ))
    }))
}

/// Import wisdom from the system default location.
///
/// Searches for wisdom files in standard locations:
/// - Linux: `~/.config/oxifft/wisdom` or `/etc/oxifft/wisdom`
/// - macOS: `~/Library/Application Support/oxifft/wisdom`
/// - Windows: `%APPDATA%\oxifft\wisdom`
///
/// If a candidate path exists but fails to import (corrupt data, an
/// incompatible `format_version`, an oversized file, ...), that specific
/// error is returned rather than a generic "not found" — only when *no*
/// candidate path exists at all does this return
/// `WisdomError::IoError(NotFound, ...)`.
///
/// # Errors
/// Returns an error if no system wisdom file exists, or if the first
/// existing candidate fails to load (see above).
#[cfg(feature = "std")]
pub fn import_system_wisdom() -> Result<WisdomImportResult, WisdomError> {
    import_from_first_existing(&get_system_wisdom_paths())
}

/// Get the default user wisdom file path.
///
/// Returns the path where user wisdom should be stored:
/// - Linux: `~/.config/oxifft/wisdom`
/// - macOS: `~/Library/Application Support/oxifft/wisdom`
/// - Windows: `%APPDATA%\oxifft\wisdom`
#[cfg(feature = "std")]
#[must_use]
pub fn get_user_wisdom_path() -> Option<std::path::PathBuf> {
    #[cfg(target_os = "linux")]
    {
        if let Some(config_dir) = std::env::var_os("XDG_CONFIG_HOME") {
            let mut path = std::path::PathBuf::from(config_dir);
            path.push("oxifft");
            path.push("wisdom");
            return Some(path);
        }
        if let Some(home) = std::env::var_os("HOME") {
            let mut path = std::path::PathBuf::from(home);
            path.push(".config");
            path.push("oxifft");
            path.push("wisdom");
            return Some(path);
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(home) = std::env::var_os("HOME") {
            let mut path = std::path::PathBuf::from(home);
            path.push("Library");
            path.push("Application Support");
            path.push("oxifft");
            path.push("wisdom");
            return Some(path);
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Some(appdata) = std::env::var_os("APPDATA") {
            let mut path = std::path::PathBuf::from(appdata);
            path.push("oxifft");
            path.push("wisdom");
            return Some(path);
        }
    }

    None
}

/// Get all system wisdom paths to search.
#[cfg(feature = "std")]
fn get_system_wisdom_paths() -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();

    // User wisdom path (highest priority)
    if let Some(user_path) = get_user_wisdom_path() {
        paths.push(user_path);
    }

    // System-wide wisdom paths
    #[cfg(target_os = "linux")]
    {
        paths.push(std::path::PathBuf::from("/etc/oxifft/wisdom"));
        paths.push(std::path::PathBuf::from("/usr/share/oxifft/wisdom"));
    }

    #[cfg(target_os = "macos")]
    {
        paths.push(std::path::PathBuf::from(
            "/Library/Application Support/oxifft/wisdom",
        ));
    }

    paths
}

/// Forget all accumulated wisdom.
///
/// Clears the global wisdom cache. Any subsequent planning will start fresh.
///
/// # Example
///
/// ```rust
/// use oxifft::api::forget;
///
/// forget();
/// ```
pub fn forget() {
    with_wisdom_mut(WisdomCache::clear);
}

/// Get the number of wisdom entries currently cached.
///
/// # Example
///
/// ```rust
/// use oxifft::api::wisdom_count;
///
/// let count = wisdom_count();
/// println!("Cached {} wisdom entries", count);
/// ```
#[must_use]
pub fn wisdom_count() -> usize {
    with_wisdom(WisdomCache::len)
}

/// Store a wisdom entry in the global cache.
///
/// This is typically called internally by the planner after measuring
/// algorithm performance.
pub fn store_wisdom(entry: WisdomEntry) {
    with_wisdom_mut(|cache| cache.store(entry));
}

/// Look up wisdom for a problem hash.
///
/// Returns the cached wisdom entry if available.
#[must_use]
pub fn lookup_wisdom(hash: u64) -> Option<WisdomEntry> {
    with_wisdom(|cache| cache.lookup(hash).cloned())
}

// ─── Error type ───────────────────────────────────────────────────────────────

/// Error type for wisdom operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum WisdomError {
    /// The wisdom string/file is malformed.
    ParseError(String),
    /// The wisdom data uses a `format_version` that is strictly newer than
    /// the version this build of OxiFFT understands.
    ///
    /// Upgrade OxiFFT to a version that supports format version `found`, or
    /// regenerate the wisdom file with an older OxiFFT build.
    IncompatibleVersion {
        /// The `format_version` found in the wisdom data.
        found: u32,
        /// The highest `format_version` this build can parse.
        expected: u32,
    },
    /// I/O error (only available with std feature).
    #[cfg(feature = "std")]
    IoError(std::io::Error),
    /// The input exceeded a configured safety ceiling before it was parsed.
    ///
    /// Returned by `import_string`/`merge_string`/`from_binary` (and their
    /// file-backed wrappers) when untrusted wisdom data is larger, or
    /// declares more entries, than this build is willing to process. See
    /// [`MAX_WISDOM_TEXT_BYTES`], [`MAX_WISDOM_TEXT_ENTRIES`],
    /// [`MAX_WISDOM_BINARY_BYTES`], and [`MAX_WISDOM_BINARY_ENTRIES`].
    TooLarge {
        /// What was too large: `"bytes"`, `"entries"`, or `"factors"`.
        kind: &'static str,
        /// The configured ceiling that was exceeded.
        limit: usize,
        /// The actual size/count encountered in the input.
        actual: usize,
    },
}

impl core::fmt::Display for WisdomError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "Wisdom parse error: {msg}"),
            Self::IncompatibleVersion { found, expected } => write!(
                f,
                "Wisdom format version {found} is not supported \
                 (this build understands up to version {expected})"
            ),
            #[cfg(feature = "std")]
            Self::IoError(e) => write!(f, "I/O error: {e}"),
            Self::TooLarge {
                kind,
                limit,
                actual,
            } => write!(
                f,
                "Wisdom input exceeds the maximum allowed {kind} ({actual} > {limit}); \
                 rejected before parsing to avoid unbounded resource use"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for WisdomError {}

#[cfg(feature = "std")]
impl From<std::io::Error> for WisdomError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

// ─── Binary wisdom format ─────────────────────────────────────────────────────
//
// See `docs/wisdom_format.md` for the full specification of both the v1
// (legacy) and v2 (current) binary layouts.

/// Magic bytes that identify an OxiFFT binary wisdom file.
const BINARY_MAGIC: &[u8; 8] = b"OXIWISDM";

/// Binary format version 1 (legacy — written by OxiFFT ≤ 0.3.x).
///
/// Fixed `[u16; 6]` mixed-radix factor slots and a `u16` entry count. Writing
/// this format silently truncated any mixed-radix solver name needing more
/// than 6 factors (e.g. `n = 2187 = 3^7`) and capped the entry count at
/// `u16::MAX`; both hazards are fixed in format 2. `from_binary` still reads
/// v1 blobs for backward compatibility with wisdom files generated by older
/// builds — only *writing* v1 was ever unsafe.
const BINARY_FORMAT_VERSION_V1: u16 = 1;

/// Binary format version 2 (current). Variable-length mixed-radix factor
/// lists (up to [`MAX_BINARY_FACTORS`] entries) and a `u32` entry count.
/// This is the format [`WisdomCache::to_binary`]/`to_binary_checked` write.
const BINARY_FORMAT_VERSION_V2: u16 = 2;

/// Binary format version this build writes.
const BINARY_FORMAT_VERSION: u16 = BINARY_FORMAT_VERSION_V2;

/// Fixed on-disk size, in bytes, of a version-1 binary entry.
/// Layout: u64 size_key + u8 algo_tag + u8 factors_len + [u16; 6] factors + u64 elapsed_ns
/// = 8 + 1 + 1 + 12 + 8 = 30 bytes
const BINARY_ENTRY_SIZE_V1: usize = 30;

/// Maximum number of mixed-radix factors representable in a single v2 entry.
///
/// `factors_len` is encoded as a `u8`. No real transform size needs anywhere
/// near this many stages — a size requiring 256 radix-2 stages alone would be
/// `2^256` elements, far beyond any machine's addressable memory — so this is
/// a defensive ceiling, not a practical limitation.
pub const MAX_BINARY_FACTORS: usize = u8::MAX as usize;

/// Algorithm tag discriminants for the binary format (stable across versions).
/// 0=CooleyTukey  1=SplitRadix  2=Stockham  3=Bluestein  4=Rader
/// 5=MixedRadix   6=Winograd    7=Direct    8=Generic    9=Composite
/// 10=Nop         255=Unknown
const ALGO_TAG_COOLEY_TUKEY: u8 = 0;
const ALGO_TAG_SPLIT_RADIX: u8 = 1;
const ALGO_TAG_STOCKHAM: u8 = 2;
const ALGO_TAG_BLUESTEIN: u8 = 3;
const ALGO_TAG_RADER: u8 = 4;
const ALGO_TAG_MIXED_RADIX: u8 = 5;
const ALGO_TAG_WINOGRAD: u8 = 6;
const ALGO_TAG_DIRECT: u8 = 7;
const ALGO_TAG_GENERIC: u8 = 8;
const ALGO_TAG_COMPOSITE: u8 = 9;
const ALGO_TAG_NOP: u8 = 10;
const ALGO_TAG_UNKNOWN: u8 = 255;

/// Derive an algorithm tag and (for `MixedRadix`) the ordered factor list
/// from a solver name string.
///
/// The factor list has no fixed bound here — [`WisdomCache::to_binary`] and
/// [`WisdomCache::to_binary_checked`] are responsible for applying
/// [`MAX_BINARY_FACTORS`] when encoding. A `"mixed-radix-..."` name with any
/// non-numeric component is treated as unrecognised (`ALGO_TAG_UNKNOWN`)
/// rather than producing a partially-populated factor list.
fn algo_tag_from_solver_name(name: &str) -> (u8, Vec<u16>) {
    if let Some(factors) = parse_mixed_radix_name(name) {
        return (ALGO_TAG_MIXED_RADIX, factors);
    }
    let tag = match name {
        "ct-dit" | "ct-dif" | "ct-radix4" | "ct-radix8" => ALGO_TAG_COOLEY_TUKEY,
        "ct-splitradix" => ALGO_TAG_SPLIT_RADIX,
        "stockham" => ALGO_TAG_STOCKHAM,
        "bluestein" => ALGO_TAG_BLUESTEIN,
        "rader" => ALGO_TAG_RADER,
        "winograd" | "winograd-pfa" => ALGO_TAG_WINOGRAD,
        "direct" => ALGO_TAG_DIRECT,
        "generic" | "cache-oblivious" => ALGO_TAG_GENERIC,
        "composite" => ALGO_TAG_COMPOSITE,
        "nop" => ALGO_TAG_NOP,
        // CooleyTukey variants that include parentheses in name
        n if n.starts_with("CooleyTukey") => ALGO_TAG_COOLEY_TUKEY,
        n if n.starts_with("Winograd") => ALGO_TAG_WINOGRAD,
        _ => ALGO_TAG_UNKNOWN,
    };
    (tag, Vec::new())
}

/// Recover a solver name from an algorithm tag and factor list.
fn solver_name_from_algo_tag(tag: u8, factors: &[u16]) -> String {
    match tag {
        ALGO_TAG_COOLEY_TUKEY => "ct-dit".to_string(),
        ALGO_TAG_SPLIT_RADIX => "ct-splitradix".to_string(),
        ALGO_TAG_STOCKHAM => "stockham".to_string(),
        ALGO_TAG_BLUESTEIN => "bluestein".to_string(),
        ALGO_TAG_RADER => "rader".to_string(),
        ALGO_TAG_MIXED_RADIX => {
            let parts: Vec<String> = factors.iter().map(|r| r.to_string()).collect();
            format!("mixed-radix-{}", parts.join("-"))
        }
        ALGO_TAG_WINOGRAD => "winograd".to_string(),
        ALGO_TAG_DIRECT => "direct".to_string(),
        ALGO_TAG_GENERIC => "generic".to_string(),
        ALGO_TAG_COMPOSITE => "composite".to_string(),
        ALGO_TAG_NOP => "nop".to_string(),
        _ => "unknown".to_string(),
    }
}

/// Semantic validity check shared by the v1 and v2 binary decoders — mirrors
/// [`is_valid_entry`] (the text-format check) so both wisdom formats reject
/// the same classes of corrupt/adversarial data:
/// - `size_key == 0` is invalid (matches the text format's `hash == 0` rule).
/// - An unrecognised algorithm tag carries no reconstructible information.
/// - A `MixedRadix` entry must have factors that are all supported radices
///   and multiply out exactly to `size_key` (see
///   [`is_valid_mixed_radix_factors`]) — this is the check that keeps a
///   corrupt/degenerate binary wisdom entry from ever reaching
///   `api::plan::types::algorithm_from_solver_name`, where replaying it
///   would panic.
fn is_valid_binary_entry(size_key: u64, algo_tag: u8, factors: &[u16]) -> bool {
    if size_key == 0 || algo_tag == ALGO_TAG_UNKNOWN {
        return false;
    }
    if algo_tag == ALGO_TAG_MIXED_RADIX {
        return is_valid_mixed_radix_factors(factors, size_key);
    }
    true
}

/// Overflow-handling policy for the binary encoder — see
/// [`WisdomCache::to_binary`] (caps) vs. [`WisdomCache::to_binary_checked`]
/// (rejects).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinaryOverflowPolicy {
    /// Cap the offending value and keep encoding.
    Cap,
    /// Return [`WisdomError::TooLarge`] instead of encoding.
    Reject,
}

impl WisdomCache {
    /// Shared encoder for [`to_binary`](Self::to_binary) and
    /// [`to_binary_checked`](Self::to_binary_checked); see those methods for
    /// the format description and the two overflow-handling behaviours.
    fn to_binary_with_policy(&self, policy: BinaryOverflowPolicy) -> Result<Vec<u8>, WisdomError> {
        let total_entries = self.entries.len();
        if total_entries > u32::MAX as usize && policy == BinaryOverflowPolicy::Reject {
            return Err(WisdomError::TooLarge {
                kind: "entries",
                limit: u32::MAX as usize,
                actual: total_entries,
            });
        }
        let entry_count = total_entries.min(u32::MAX as usize);

        let mut buf = Vec::with_capacity(16 + entry_count * 18);

        // Header (16 bytes): magic(8) + version:u16 + reserved:u16 + entry_count:u32
        buf.extend_from_slice(BINARY_MAGIC);
        buf.extend_from_slice(&BINARY_FORMAT_VERSION_V2.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes()); // reserved
        buf.extend_from_slice(&(entry_count as u32).to_le_bytes());

        // Entries (variable length): size_key:u64 + algo_tag:u8 + factors_len:u8
        //                             + factors:[u16; factors_len] + elapsed_ns:u64
        for entry in self.entries.values().take(entry_count) {
            let (tag, mut factors) = algo_tag_from_solver_name(&entry.solver_name);
            if factors.len() > MAX_BINARY_FACTORS {
                match policy {
                    BinaryOverflowPolicy::Reject => {
                        return Err(WisdomError::TooLarge {
                            kind: "factors",
                            limit: MAX_BINARY_FACTORS,
                            actual: factors.len(),
                        });
                    }
                    BinaryOverflowPolicy::Cap => factors.truncate(MAX_BINARY_FACTORS),
                }
            }
            let elapsed_ns = entry.cost as u64;

            buf.extend_from_slice(&entry.problem_hash.to_le_bytes());
            buf.push(tag);
            buf.push(factors.len() as u8);
            for f in &factors {
                buf.extend_from_slice(&f.to_le_bytes());
            }
            buf.extend_from_slice(&elapsed_ns.to_le_bytes());
        }

        Ok(buf)
    }

    /// Serialize the wisdom cache to a compact binary blob (format version 2).
    ///
    /// Binary header (16 bytes, all LE):
    /// - `OXIWISDM` magic (8 bytes)
    /// - format_version: u16 = 2
    /// - reserved: u16 = 0
    /// - entry_count: u32
    ///
    /// Each entry is variable-length (all LE):
    /// - size_key: u64 (= problem_hash, which encodes the transform size)
    /// - algo_tag: u8
    /// - factors_len: u8
    /// - factors: `[u16; factors_len]`
    /// - elapsed_ns: u64 (= cost cast to u64)
    ///
    /// This is the infallible convenience form: entries beyond `u32::MAX` or
    /// a mixed-radix solver name needing more than [`MAX_BINARY_FACTORS`]
    /// stages are capped rather than rejected. Neither condition is
    /// reachable through OxiFFT's own planner — both require a transform
    /// size no real machine could compute an FFT of. Use
    /// [`WisdomCache::to_binary_checked`] for a hard error instead of a
    /// best-effort cap (relevant mainly for hand-constructed caches built
    /// via the public [`WisdomCache::store`] API with synthetic data).
    #[must_use]
    pub fn to_binary(&self) -> Vec<u8> {
        self.to_binary_with_policy(BinaryOverflowPolicy::Cap)
            .unwrap_or_default()
    }

    /// Serialize the wisdom cache to a compact binary blob, like
    /// [`to_binary`](Self::to_binary), but return an error instead of
    /// silently capping entries/factors that do not fit the format.
    ///
    /// # Errors
    /// Returns [`WisdomError::TooLarge`] if the cache holds more than
    /// `u32::MAX` entries, or if any entry's mixed-radix solver name needs
    /// more than [`MAX_BINARY_FACTORS`] factors.
    pub fn to_binary_checked(&self) -> Result<Vec<u8>, WisdomError> {
        self.to_binary_with_policy(BinaryOverflowPolicy::Reject)
    }

    /// Decode a binary wisdom blob, returning the reconstructed cache, the
    /// number of entries skipped as semantically invalid, and the binary
    /// format version that was parsed.
    fn decode_binary(data: &[u8]) -> Result<(Self, usize, u16), WisdomError> {
        if data.len() > MAX_WISDOM_BINARY_BYTES {
            return Err(WisdomError::TooLarge {
                kind: "bytes",
                limit: MAX_WISDOM_BINARY_BYTES,
                actual: data.len(),
            });
        }
        if data.len() < 16 {
            return Err(WisdomError::ParseError(format!(
                "binary wisdom data truncated: header requires 16 bytes, got {}",
                data.len()
            )));
        }
        if &data[..8] != BINARY_MAGIC {
            return Err(WisdomError::ParseError(
                "binary wisdom data: bad magic bytes (not an OxiFFT wisdom blob)".to_string(),
            ));
        }

        let version = u16::from_le_bytes([data[8], data[9]]);
        match version {
            BINARY_FORMAT_VERSION_V1 => {
                Self::from_binary_v1(data).map(|(cache, skipped)| (cache, skipped, version))
            }
            BINARY_FORMAT_VERSION_V2 => {
                Self::from_binary_v2(data).map(|(cache, skipped)| (cache, skipped, version))
            }
            v if u32::from(v) > u32::from(BINARY_FORMAT_VERSION) => {
                Err(WisdomError::IncompatibleVersion {
                    found: u32::from(v),
                    expected: u32::from(BINARY_FORMAT_VERSION),
                })
            }
            v => Err(WisdomError::ParseError(format!(
                "binary wisdom data: unrecognised format version {v}"
            ))),
        }
    }

    /// Decode a version-1 (legacy, fixed `[u16; 6]` factors) binary blob.
    fn from_binary_v1(data: &[u8]) -> Result<(Self, usize), WisdomError> {
        let entry_count = u16::from_le_bytes([data[10], data[11]]) as usize;
        // bytes 12..16 are reserved.

        if entry_count > MAX_WISDOM_BINARY_ENTRIES {
            return Err(WisdomError::TooLarge {
                kind: "entries",
                limit: MAX_WISDOM_BINARY_ENTRIES,
                actual: entry_count,
            });
        }

        let expected_len = 16 + entry_count * BINARY_ENTRY_SIZE_V1;
        if data.len() < expected_len {
            return Err(WisdomError::ParseError(format!(
                "binary wisdom data truncated: header declares {entry_count} v1 entries \
                 ({expected_len} bytes total) but only {} bytes are present",
                data.len()
            )));
        }

        let mut cache = WisdomCache::new();
        let mut skipped_invalid = 0usize;
        for i in 0..entry_count {
            let offset = 16 + i * BINARY_ENTRY_SIZE_V1;
            let entry_bytes = &data[offset..offset + BINARY_ENTRY_SIZE_V1];

            let size_key = u64::from_le_bytes([
                entry_bytes[0],
                entry_bytes[1],
                entry_bytes[2],
                entry_bytes[3],
                entry_bytes[4],
                entry_bytes[5],
                entry_bytes[6],
                entry_bytes[7],
            ]);
            let algo_tag = entry_bytes[8];
            let factors_len = (entry_bytes[9] as usize).min(6);
            let factors: [u16; 6] = [
                u16::from_le_bytes([entry_bytes[10], entry_bytes[11]]),
                u16::from_le_bytes([entry_bytes[12], entry_bytes[13]]),
                u16::from_le_bytes([entry_bytes[14], entry_bytes[15]]),
                u16::from_le_bytes([entry_bytes[16], entry_bytes[17]]),
                u16::from_le_bytes([entry_bytes[18], entry_bytes[19]]),
                u16::from_le_bytes([entry_bytes[20], entry_bytes[21]]),
            ];
            let elapsed_ns = u64::from_le_bytes([
                entry_bytes[22],
                entry_bytes[23],
                entry_bytes[24],
                entry_bytes[25],
                entry_bytes[26],
                entry_bytes[27],
                entry_bytes[28],
                entry_bytes[29],
            ]);

            let factor_slice = &factors[..factors_len];
            if !is_valid_binary_entry(size_key, algo_tag, factor_slice) {
                skipped_invalid += 1;
                continue;
            }

            let solver_name = solver_name_from_algo_tag(algo_tag, factor_slice);
            cache.store(WisdomEntry {
                problem_hash: size_key,
                solver_name,
                cost: elapsed_ns as f64,
            });
        }

        Ok((cache, skipped_invalid))
    }

    /// Decode a version-2 (current, variable-length factors) binary blob.
    fn from_binary_v2(data: &[u8]) -> Result<(Self, usize), WisdomError> {
        let entry_count = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
        // bytes 10..12 are reserved.

        if entry_count > MAX_WISDOM_BINARY_ENTRIES {
            return Err(WisdomError::TooLarge {
                kind: "entries",
                limit: MAX_WISDOM_BINARY_ENTRIES,
                actual: entry_count,
            });
        }

        let mut cache = WisdomCache::new();
        let mut skipped_invalid = 0usize;
        let mut offset = 16usize;

        for _ in 0..entry_count {
            // Fixed-size entry prefix: size_key(8) + algo_tag(1) + factors_len(1).
            if data.len() < offset + 10 {
                return Err(WisdomError::ParseError(format!(
                    "binary wisdom data truncated: expected an entry header at byte {offset}, \
                     only {} bytes are present",
                    data.len()
                )));
            }
            let size_key = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            let algo_tag = data[offset + 8];
            let factors_len = data[offset + 9] as usize;
            offset += 10;

            let factors_bytes_len = factors_len * 2;
            if data.len() < offset + factors_bytes_len + 8 {
                return Err(WisdomError::ParseError(format!(
                    "binary wisdom data truncated: entry declares {factors_len} factors at byte \
                     {offset} but the blob ends before its data"
                )));
            }

            let mut factors: Vec<u16> = Vec::with_capacity(factors_len);
            for i in 0..factors_len {
                let fo = offset + i * 2;
                factors.push(u16::from_le_bytes([data[fo], data[fo + 1]]));
            }
            offset += factors_bytes_len;

            let elapsed_ns = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            offset += 8;

            if !is_valid_binary_entry(size_key, algo_tag, &factors) {
                skipped_invalid += 1;
                continue;
            }

            let solver_name = solver_name_from_algo_tag(algo_tag, &factors);
            cache.store(WisdomEntry {
                problem_hash: size_key,
                solver_name,
                cost: elapsed_ns as f64,
            });
        }

        Ok((cache, skipped_invalid))
    }

    /// Deserialize wisdom from a binary blob produced by
    /// [`to_binary`](Self::to_binary)/[`to_binary_checked`](Self::to_binary_checked).
    ///
    /// Reads both the current (v2, variable-length factors) and legacy (v1,
    /// fixed 6-factor) binary formats. Entries that fail the same semantic
    /// validation the text-format parser applies (zero hash, unrecognised
    /// algorithm tag, or — for `MixedRadix` — factors that are not all
    /// supported radices or whose product does not match the declared size)
    /// are silently skipped rather than stored; use
    /// [`import_binary`](Self::import_binary) if you need the skipped count.
    ///
    /// # Errors
    /// Returns an error if `data` exceeds [`MAX_WISDOM_BINARY_BYTES`] or
    /// declares more than [`MAX_WISDOM_BINARY_ENTRIES`], if the magic bytes
    /// are wrong, if the format version is unrecognised or newer than this
    /// build supports, or if the data is truncated relative to what the
    /// header declares.
    pub fn from_binary(data: &[u8]) -> Result<Self, WisdomError> {
        Self::decode_binary(data).map(|(cache, _skipped, _version)| cache)
    }

    /// Import wisdom from a binary blob into `self`, merging entries
    /// (existing entries for the same hash are overwritten — matching
    /// [`import_string`](Self::import_string)'s semantics, not
    /// [`merge_string`](Self::merge_string)'s lower-cost-wins semantics).
    ///
    /// Unlike [`from_binary`](Self::from_binary), this reports how many
    /// entries were imported vs. skipped as invalid, mirroring
    /// [`WisdomImportResult`] from the text-format import path. The
    /// `format_version` field carries the *binary* format version (1 or 2),
    /// not the S-expr [`WISDOM_FORMAT_VERSION`].
    ///
    /// # Errors
    /// See [`from_binary`](Self::from_binary).
    pub fn import_binary(&mut self, data: &[u8]) -> Result<WisdomImportResult, WisdomError> {
        let (decoded, skipped_invalid, version) = Self::decode_binary(data)?;
        let imported = decoded.entries.len();
        self.entries.extend(decoded.entries);
        Ok(WisdomImportResult {
            imported,
            skipped_invalid,
            format_version: u32::from(version),
        })
    }

    /// Return the number of entries (alias for `len` — used by CLI tooling).
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "wisdom_tests.rs"]
mod tests;
