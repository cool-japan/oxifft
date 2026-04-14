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
//! - **Version 0 (legacy)**: header `(oxifft-wisdom-1.0 …)`, no `(format_version …)` line
//! - **Version 1 (current)**: `(oxifft-wisdom` header with `(format_version 1)` on the
//!   second line
//!
//! Importing wisdom whose `format_version` is *greater* than
//! [`WISDOM_FORMAT_VERSION`] returns
//! [`WisdomError::IncompatibleVersion`].  A lower or equal version is accepted
//! (with silent best-effort parsing).
//!
//! # Merge Semantics
//!
//! [`merge_from_string`] (and its file counterpart) can combine wisdom gathered
//! on different machines or runs.  When the same problem hash appears in both
//! the current cache and the incoming data, the entry with the **lower cost**
//! (faster measured time) is kept.
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
pub const WISDOM_FORMAT_VERSION: u32 = 1;

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

        for line in s.lines().skip(1) {
            let line = line.trim();
            if !is_entry_line(line) {
                continue;
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

        for line in s.lines().skip(1) {
            let line = line.trim();
            if !is_entry_line(line) {
                continue;
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
fn is_valid_entry(entry: &WisdomEntry) -> bool {
    entry.problem_hash != 0
        && !entry.solver_name.is_empty()
        && entry.cost.is_finite()
        && entry.cost >= 0.0
}

// ─── Global cache initialisation ─────────────────────────────────────────────

/// Initialize global wisdom if not already initialized.
fn ensure_global_wisdom() {
    #[cfg(feature = "std")]
    {
        let read_guard = GLOBAL_WISDOM.read().expect("Global wisdom lock poisoned");
        if read_guard.is_none() {
            drop(read_guard);
            let mut write_guard = GLOBAL_WISDOM.write().expect("Global wisdom lock poisoned");
            if write_guard.is_none() {
                *write_guard = Some(WisdomCache::new());
            }
        }
    }
    #[cfg(not(feature = "std"))]
    {
        let read_guard = GLOBAL_WISDOM.read();
        if read_guard.is_none() {
            drop(read_guard);
            let mut write_guard = GLOBAL_WISDOM.write();
            if write_guard.is_none() {
                *write_guard = Some(WisdomCache::new());
            }
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
        let guard = GLOBAL_WISDOM.read().expect("Global wisdom lock poisoned");
        f(guard.as_ref().expect("Global wisdom not initialized"))
    }
    #[cfg(not(feature = "std"))]
    {
        let guard = GLOBAL_WISDOM.read();
        f(guard.as_ref().expect("Global wisdom not initialized"))
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
        let mut guard = GLOBAL_WISDOM.write().expect("Global wisdom lock poisoned");
        f(guard.as_mut().expect("Global wisdom not initialized"))
    }
    #[cfg(not(feature = "std"))]
    {
        let mut guard = GLOBAL_WISDOM.write();
        f(guard.as_mut().expect("Global wisdom not initialized"))
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

/// Import wisdom from a file.
///
/// # Arguments
/// * `path` - Path to the file to read wisdom from
///
/// # Errors
/// Returns an error if the file cannot be read, is structurally unrecognisable,
/// or its `format_version` is newer than [`WISDOM_FORMAT_VERSION`].
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
    let contents = std::fs::read_to_string(path)?;
    import_from_string(&contents)
}

/// Merge wisdom from a file into the global cache.
///
/// Combines the file's wisdom with the currently cached data, keeping the
/// lower-cost (better) entry for any hash that appears in both.
///
/// # Errors
/// Returns an error if the file cannot be read, is structurally unrecognisable,
/// or its `format_version` is newer than [`WISDOM_FORMAT_VERSION`].
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
    let contents = std::fs::read_to_string(path)?;
    merge_from_string(&contents)
}

/// Import wisdom from the system default location.
///
/// Searches for wisdom files in standard locations:
/// - Linux: `~/.config/oxifft/wisdom` or `/etc/oxifft/wisdom`
/// - macOS: `~/Library/Application Support/oxifft/wisdom`
/// - Windows: `%APPDATA%\oxifft\wisdom`
///
/// # Errors
/// Returns an error if no system wisdom is found or it cannot be loaded.
#[cfg(feature = "std")]
pub fn import_system_wisdom() -> Result<WisdomImportResult, WisdomError> {
    let paths = get_system_wisdom_paths();

    for path in paths {
        if path.exists() {
            if let Ok(result) = import_from_file(&path) {
                return Ok(result);
            }
        }
    }

    Err(WisdomError::IoError(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "No system wisdom found",
    )))
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

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
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
        let future_wisdom = format!(
            "({WISDOM_MARKER}\n  (format_version {future_version})\n  (111 \"rader\" 50)\n)"
        );
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
        let future_wisdom = format!(
            "({WISDOM_MARKER}\n  (format_version {future_version})\n  (100 \"rader\" 1.0)\n)"
        );
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
}
