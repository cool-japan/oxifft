//! Wisdom management for plan caching.
//!
//! The wisdom system caches optimal plans for specific problem sizes,
//! avoiding the cost of re-measuring algorithms on subsequent runs.
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

/// Version string for wisdom format.
const WISDOM_VERSION: &str = "oxifft-wisdom-1.0";

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

    /// Export wisdom to a string.
    #[must_use]
    pub fn export_string(&self) -> String {
        use core::fmt::Write;
        let mut result = format!("({WISDOM_VERSION}\n");
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

    /// Import wisdom from a string.
    ///
    /// # Errors
    /// Returns error if the wisdom format is invalid.
    pub fn import_string(&mut self, s: &str) -> Result<usize, WisdomError> {
        let s = s.trim();
        if !s.starts_with(&format!("({WISDOM_VERSION}")) {
            return Err(WisdomError::VersionMismatch);
        }

        let mut count = 0;
        for line in s.lines().skip(1) {
            let line = line.trim();
            if line.starts_with('(') && line.ends_with(')') && !line.starts_with("(oxifft") {
                // Parse: (hash "solver" cost)
                let inner = &line[1..line.len() - 1];
                let parts: Vec<&str> = inner.split_whitespace().collect();
                if parts.len() >= 3 {
                    let hash = parts[0]
                        .parse::<u64>()
                        .map_err(|_| WisdomError::ParseError(String::from("invalid hash")))?;
                    let solver_name = parts[1].trim_matches('"').to_string();
                    let cost = parts[2]
                        .parse::<f64>()
                        .map_err(|_| WisdomError::ParseError(String::from("invalid cost")))?;

                    self.entries.insert(
                        hash,
                        WisdomEntry {
                            problem_hash: hash,
                            solver_name,
                            cost,
                        },
                    );
                    count += 1;
                }
            }
        }

        Ok(count)
    }
}

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
    with_wisdom(|cache| cache.export_string())
}

/// Import wisdom from a string.
///
/// # Arguments
/// * `s` - The wisdom string to import
///
/// # Errors
/// Returns an error if the wisdom string is malformed or version mismatched.
///
/// # Example
///
/// ```rust
/// use oxifft::api::import_from_string;
///
/// let wisdom_str = "(oxifft-wisdom-1.0\n)";
/// import_from_string(wisdom_str).unwrap();
/// ```
pub fn import_from_string(s: &str) -> Result<usize, WisdomError> {
    with_wisdom_mut(|cache| cache.import_string(s))
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
/// Returns an error if the file cannot be read or is malformed.
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
pub fn import_from_file(path: &std::path::Path) -> Result<usize, WisdomError> {
    let contents = std::fs::read_to_string(path)?;
    import_from_string(&contents)
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
pub fn import_system_wisdom() -> Result<usize, WisdomError> {
    let paths = get_system_wisdom_paths();

    for path in paths {
        if path.exists() {
            if let Ok(count) = import_from_file(&path) {
                return Ok(count);
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
    with_wisdom_mut(|cache| cache.clear());
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
    with_wisdom(|cache| cache.len())
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

/// Error type for wisdom operations.
#[derive(Debug)]
pub enum WisdomError {
    /// The wisdom string/file is malformed
    ParseError(String),
    /// Version mismatch
    VersionMismatch,
    /// I/O error (only available with std feature)
    #[cfg(feature = "std")]
    IoError(std::io::Error),
}

impl core::fmt::Display for WisdomError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "Wisdom parse error: {msg}"),
            Self::VersionMismatch => write!(f, "Wisdom version mismatch"),
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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_wisdom_export_import() {
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
        assert!(exported.contains(WISDOM_VERSION));
        assert!(exported.contains("111"));
        assert!(exported.contains("rader"));

        let mut cache2 = WisdomCache::new();
        let count = cache2.import_string(&exported).expect("import failed");
        assert_eq!(count, 2);
        assert_eq!(cache2.len(), 2);

        let entry = cache2.lookup(111).expect("entry not found");
        assert_eq!(entry.solver_name, "rader");
    }

    #[test]
    fn test_wisdom_version_mismatch() {
        let mut cache = WisdomCache::new();
        let result = cache.import_string("(oxifft-wisdom-0.9\n)");
        assert!(matches!(result, Err(WisdomError::VersionMismatch)));
    }

    #[test]
    fn test_global_wisdom_functions() {
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

        let count = import_from_string(&exported).expect("import failed");
        assert_eq!(count, 1);
        assert_eq!(wisdom_count(), 1);

        // Cleanup
        forget();
    }

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
