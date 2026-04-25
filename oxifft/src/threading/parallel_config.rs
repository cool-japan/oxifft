//! Parallel execution configuration and task granularity tuning.
//!
//! Provides [`ParallelConfig`] for controlling when and how FFT operations
//! are parallelised, preventing excessive overhead from spawning tasks on
//! small problems.
//!
//! # Defaults
//!
//! The default thresholds are tuned for modern multi-core CPUs:
//!
//! | Parameter | Default | Rationale |
//! |-----------|---------|-----------|
//! | `min_fft_size` | 4096 | Below this, threading overhead > parallel gain |
//! | `min_batch_chunk` | 4 | Minimum transforms per thread in batch mode |
//! | `min_rows_per_thread` | 4 | Minimum rows/slices per thread in 2D/3D |
//! | `enabled` | `true` | Master switch for parallelism |
//!
//! # Example
//!
//! ```
//! use oxifft::threading::ParallelConfig;
//!
//! let config = ParallelConfig::new()
//!     .with_min_fft_size(8192)
//!     .with_min_batch_chunk(8)
//!     .with_min_rows_per_thread(8);
//!
//! assert!(config.should_parallelize_fft(16384));
//! assert!(!config.should_parallelize_fft(2048));
//! ```

/// Configuration for parallel FFT execution granularity.
///
/// Controls minimum problem sizes and chunk sizes to avoid excessive
/// parallelisation overhead on small workloads.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Minimum FFT size (number of complex elements) to parallelise a single
    /// 1D transform. Below this threshold, the transform runs sequentially.
    pub min_fft_size: usize,

    /// Minimum number of transforms per thread when running batch FFTs.
    /// The batch is split into chunks of at least this many transforms.
    pub min_batch_chunk: usize,

    /// Minimum number of rows (or slices in 3D) assigned to each thread when
    /// parallelising multi-dimensional FFTs.
    pub min_rows_per_thread: usize,

    /// Master switch. When `false`, all operations run single-threaded
    /// regardless of other settings.
    pub enabled: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            min_fft_size: 4096,
            min_batch_chunk: 4,
            min_rows_per_thread: 4,
            enabled: true,
        }
    }
}

impl ParallelConfig {
    /// Create a new configuration with default thresholds.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum 1D FFT size for parallelisation.
    #[must_use]
    pub fn with_min_fft_size(mut self, size: usize) -> Self {
        self.min_fft_size = size;
        self
    }

    /// Set the minimum batch chunk size (transforms per thread).
    #[must_use]
    pub fn with_min_batch_chunk(mut self, chunk: usize) -> Self {
        self.min_batch_chunk = chunk;
        self
    }

    /// Set the minimum rows/slices per thread for multi-dimensional FFTs.
    #[must_use]
    pub fn with_min_rows_per_thread(mut self, rows: usize) -> Self {
        self.min_rows_per_thread = rows;
        self
    }

    /// Enable or disable parallelism entirely.
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Create a configuration that disables all parallelism.
    #[must_use]
    pub fn serial() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    // ----- query methods -----

    /// Should a single 1D FFT of `n` elements be parallelised?
    #[inline]
    #[must_use]
    pub fn should_parallelize_fft(&self, n: usize) -> bool {
        self.enabled && n >= self.min_fft_size
    }

    /// Compute the chunk size for a batch of `batch_size` transforms on
    /// `num_threads` threads.
    ///
    /// Returns the number of transforms each thread should process. If the
    /// batch is too small to benefit from parallelism, returns `batch_size`
    /// (i.e. run everything on one thread).
    #[inline]
    #[must_use]
    pub fn batch_chunk_size(&self, batch_size: usize, num_threads: usize) -> usize {
        if !self.enabled || num_threads <= 1 {
            return batch_size;
        }
        let ideal = (batch_size + num_threads - 1) / num_threads;
        let chunk = ideal.max(self.min_batch_chunk);
        chunk.min(batch_size)
    }

    /// Compute the number of rows each thread should process for a
    /// multi-dimensional FFT with `total_rows` rows across `num_threads`
    /// threads.
    ///
    /// Returns `total_rows` when parallelism is not worthwhile.
    #[inline]
    #[must_use]
    pub fn rows_per_thread(&self, total_rows: usize, num_threads: usize) -> usize {
        if !self.enabled || num_threads <= 1 {
            return total_rows;
        }
        let ideal = (total_rows + num_threads - 1) / num_threads;
        let rows = ideal.max(self.min_rows_per_thread);
        rows.min(total_rows)
    }

    /// Should batch execution of `batch_size` transforms be parallelised on
    /// `num_threads` threads?
    #[inline]
    #[must_use]
    pub fn should_parallelize_batch(&self, batch_size: usize, num_threads: usize) -> bool {
        if !self.enabled || num_threads <= 1 {
            return false;
        }
        // Only parallelise if each thread gets at least min_batch_chunk work.
        batch_size >= num_threads * self.min_batch_chunk
    }

    /// Should row/column processing of a 2D/3D FFT be parallelised given
    /// `total_rows` and `num_threads`?
    #[inline]
    #[must_use]
    pub fn should_parallelize_rows(&self, total_rows: usize, num_threads: usize) -> bool {
        if !self.enabled || num_threads <= 1 {
            return false;
        }
        total_rows >= num_threads * self.min_rows_per_thread
    }
}

// ---------------------------------------------------------------------------
// Global default
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
use std::sync::OnceLock;

#[cfg(feature = "std")]
static GLOBAL_CONFIG: OnceLock<ParallelConfig> = OnceLock::new();

/// Get the global parallel configuration (read-only reference).
///
/// Returns the configuration set by [`set_global_parallel_config`], or the
/// default if none has been set.
#[cfg(feature = "std")]
#[must_use]
pub fn global_parallel_config() -> &'static ParallelConfig {
    GLOBAL_CONFIG.get_or_init(ParallelConfig::default)
}

/// Set the global parallel configuration.
///
/// Returns `Err(config)` if a configuration has already been set (the global
/// can only be initialised once per process).
///
/// # Errors
///
/// Returns `Err(config)` (wrapping the supplied configuration back to the
/// caller) if the global has already been initialised.  The global can only
/// be set once per process lifetime.
#[cfg(feature = "std")]
pub fn set_global_parallel_config(config: ParallelConfig) -> Result<(), ParallelConfig> {
    GLOBAL_CONFIG.set(config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults() {
        let cfg = ParallelConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.min_fft_size, 4096);
        assert_eq!(cfg.min_batch_chunk, 4);
        assert_eq!(cfg.min_rows_per_thread, 4);
    }

    #[test]
    fn test_should_parallelize_fft() {
        let cfg = ParallelConfig::new().with_min_fft_size(4096);
        assert!(!cfg.should_parallelize_fft(1024));
        assert!(!cfg.should_parallelize_fft(4095));
        assert!(cfg.should_parallelize_fft(4096));
        assert!(cfg.should_parallelize_fft(65536));
    }

    #[test]
    fn test_disabled_never_parallelizes() {
        let cfg = ParallelConfig::serial();
        assert!(!cfg.should_parallelize_fft(1_000_000));
        assert!(!cfg.should_parallelize_batch(1000, 8));
        assert!(!cfg.should_parallelize_rows(1000, 8));
        assert_eq!(cfg.batch_chunk_size(1000, 8), 1000);
        assert_eq!(cfg.rows_per_thread(1000, 8), 1000);
    }

    #[test]
    fn test_batch_chunk_size() {
        let cfg = ParallelConfig::new().with_min_batch_chunk(4);

        // 100 items across 8 threads => ideal 13, >= min_batch_chunk 4 => 13
        assert_eq!(cfg.batch_chunk_size(100, 8), 13);

        // 8 items across 8 threads => ideal 1, but min is 4 => 4
        assert_eq!(cfg.batch_chunk_size(8, 8), 4);

        // 2 items across 8 threads => chunk capped to batch_size
        assert_eq!(cfg.batch_chunk_size(2, 8), 2);

        // single thread => whole batch
        assert_eq!(cfg.batch_chunk_size(100, 1), 100);
    }

    #[test]
    fn test_rows_per_thread() {
        let cfg = ParallelConfig::new().with_min_rows_per_thread(4);

        // 64 rows across 8 threads => 8 rows each (>= 4)
        assert_eq!(cfg.rows_per_thread(64, 8), 8);

        // 8 rows across 8 threads => ideal 1, but min is 4 => 4
        assert_eq!(cfg.rows_per_thread(8, 8), 4);
    }

    #[test]
    fn test_should_parallelize_batch() {
        let cfg = ParallelConfig::new().with_min_batch_chunk(4);

        // 32 items, 8 threads: 32 >= 8*4=32 => true
        assert!(cfg.should_parallelize_batch(32, 8));

        // 31 items, 8 threads: 31 < 32 => false
        assert!(!cfg.should_parallelize_batch(31, 8));
    }

    #[test]
    fn test_should_parallelize_rows() {
        let cfg = ParallelConfig::new().with_min_rows_per_thread(4);

        // 32 rows, 8 threads: 32 >= 32 => true
        assert!(cfg.should_parallelize_rows(32, 8));

        // 31 rows, 8 threads: 31 < 32 => false
        assert!(!cfg.should_parallelize_rows(31, 8));
    }

    #[test]
    fn test_builder_chain() {
        let cfg = ParallelConfig::new()
            .with_min_fft_size(8192)
            .with_min_batch_chunk(8)
            .with_min_rows_per_thread(16)
            .with_enabled(true);

        assert_eq!(cfg.min_fft_size, 8192);
        assert_eq!(cfg.min_batch_chunk, 8);
        assert_eq!(cfg.min_rows_per_thread, 16);
        assert!(cfg.enabled);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_global_config() {
        // global_parallel_config should return a valid reference
        let cfg = global_parallel_config();
        assert!(cfg.enabled);
    }
}
