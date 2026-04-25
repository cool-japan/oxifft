//! Parallel execution support for multi-threaded FFT computation.
//!
//! Provides a threading abstraction layer with optional Rayon integration for
//! parallelizing FFT operations across multiple CPU cores.
//!
//! # Overview
//!
//! OxiFFT's threading system provides:
//! - **Pluggable thread pools** via the [`ThreadPool`] trait
//! - **Automatic parallelization** of batch and multi-dimensional FFTs
//! - **Zero-cost abstraction** when threading is disabled
//!
//! # Feature Flag
//!
//! Threading is controlled by the `threading` Cargo feature:
//!
//! ```toml
//! [dependencies]
//! oxifft = { version = "0.1", features = ["threading"] }
//! ```
//!
//! When disabled, all parallel operations fall back to single-threaded execution
//! with minimal overhead.
//!
//! # Thread Pool Implementations
//!
//! | Pool | Threads | Feature | Use Case |
//! |------|---------|---------|----------|
//! | [`SerialPool`] | 1 | Always | Debug, deterministic execution |
//! | [`RayonPool`] | N | `threading` | Production, maximum throughput |
//!
//! # Configuration
//!
//! ## Using Default Pool
//!
//! The simplest approach uses [`get_default_pool()`]:
//!
//! ```
//! use oxifft::threading::{ThreadPool, get_default_pool};
//!
//! let pool = get_default_pool();
//! println!("Using {} threads", pool.num_threads());
//! ```
//!
//! ## Explicit Thread Count
//!
//! Use [`pool_with_threads()`] or [`PoolConfig`] for explicit control:
//!
//! ```
//! use oxifft::threading::{PoolConfig, pool_with_threads};
//!
//! // Method 1: Direct function
//! let pool = pool_with_threads(4);
//!
//! // Method 2: Builder pattern
//! let pool = PoolConfig::new()
//!     .threads(8)
//!     .build();
//! ```
//!
//! ## Thread Count Guidelines
//!
//! | Scenario | Recommended Threads |
//! |----------|---------------------|
//! | Large 1D FFT (>1M points) | CPU cores |
//! | Batch FFT | CPU cores |
//! | 2D/3D FFT | CPU cores |
//! | Small FFT (<4K points) | 1 (overhead dominates) |
//! | Memory-bound workloads | CPU cores / 2 |
//!
//! A value of 0 in [`PoolConfig::threads()`] uses the system default (typically
//! the number of logical CPU cores).
//!
//! # Parallelization Strategies
//!
//! OxiFFT parallelizes FFTs using two strategies:
//!
//! ## 1. Batch Parallelism
//!
//! For batch transforms, each FFT in the batch can run independently:
//!
//! ```ignore
//! // 1000 independent 1024-point FFTs
//! pool.parallel_for(1000, |batch_idx| {
//!     compute_single_fft(batch_idx);
//! });
//! ```
//!
//! ## 2. Dimensional Parallelism
//!
//! For multi-dimensional FFTs, rows/columns can be processed in parallel:
//!
//! ```ignore
//! // 2D FFT: parallelize over rows
//! pool.parallel_for(height, |row| {
//!     fft_1d(&mut data[row * width..(row + 1) * width]);
//! });
//! // Then parallelize over columns
//! pool.parallel_for(width, |col| {
//!     fft_1d_strided(&mut data, col, height);
//! });
//! ```
//!
//! # Thread Pool Methods
//!
//! The [`ThreadPool`] trait provides several parallel primitives:
//!
//! | Method | Description | Use Case |
//! |--------|-------------|----------|
//! | [`parallel_for`](ThreadPool::parallel_for) | Execute over `0..count` | Batch processing |
//! | [`parallel_for_chunks`](ThreadPool::parallel_for_chunks) | Chunked iteration | Cache-friendly access |
//! | [`parallel_split`](ThreadPool::parallel_split) | Recursive divide-and-conquer | Tree algorithms |
//! | [`join`](ThreadPool::join) | Fork-join two tasks | Divide and conquer |
//!
//! # Performance Considerations
//!
//! ## Overhead
//!
//! Threading introduces overhead from:
//! - Thread synchronization barriers
//! - Work stealing (with Rayon)
//! - Cache coherency traffic
//!
//! For small FFTs (<4K points), this overhead can exceed the parallel speedup.
//! OxiFFT automatically falls back to serial execution for small transforms.
//!
//! ## Scaling
//!
//! Expected parallel efficiency:
//! - **Batch FFT**: Near-linear scaling up to 8-16 cores
//! - **Large 1D FFT**: 2-4x speedup with 8 cores (memory-bound)
//! - **2D/3D FFT**: 4-8x speedup with 8 cores
//!
//! ## Memory Bandwidth
//!
//! FFTs are often memory-bandwidth limited. On systems with limited memory
//! bandwidth per core, using fewer threads may actually improve performance.
//!
//! # Example: Parallel Batch FFT
//!
//! ```ignore
//! use oxifft::threading::{ThreadPool, get_default_pool};
//! use oxifft::{Complex, fft};
//!
//! let pool = get_default_pool();
//! let batch_size = 1000;
//! let fft_size = 1024;
//!
//! // Allocate batch data
//! let mut batches: Vec<Vec<Complex<f64>>> = (0..batch_size)
//!     .map(|_| vec![Complex::new(0.0, 0.0); fft_size])
//!     .collect();
//!
//! // Process in parallel
//! pool.parallel_for(batch_size, |i| {
//!     fft(&mut batches[i]);
//! });
//! ```
//!
//! # Thread Safety
//!
//! All OxiFFT types implement `Send` and `Sync` where appropriate:
//! - Plans can be shared across threads (read-only execution)
//! - Input/output buffers must have exclusive access per FFT
//! - The global wisdom cache uses interior locking

mod serial;
mod spawn;

mod parallel_config;

#[cfg(feature = "threading")]
mod rayon_impl;

#[cfg(feature = "threading")]
pub mod work_stealing;

#[cfg(feature = "threading")]
pub use work_stealing::WorkStealingContext;

pub use parallel_config::ParallelConfig;
#[cfg(feature = "std")]
pub use parallel_config::{global_parallel_config, set_global_parallel_config};
pub use serial::SerialPool;
pub use spawn::ThreadPool;

#[cfg(feature = "threading")]
pub use rayon_impl::RayonPool;

/// Get the default thread pool based on available features.
///
/// Returns `RayonPool` if the `threading` feature is enabled,
/// otherwise returns `SerialPool`.
#[cfg(feature = "threading")]
#[must_use]
pub fn get_default_pool() -> RayonPool {
    RayonPool::new()
}

/// Get the default thread pool based on available features.
///
/// Returns `SerialPool` when `threading` feature is not enabled.
#[cfg(not(feature = "threading"))]
#[must_use]
pub fn get_default_pool() -> SerialPool {
    SerialPool::new()
}

/// Get a thread pool with the specified number of threads.
///
/// When the `threading` feature is enabled, returns a `RayonPool` with
/// the specified thread count. Otherwise returns a `SerialPool`.
#[cfg(feature = "threading")]
#[must_use]
pub fn pool_with_threads(num_threads: usize) -> RayonPool {
    RayonPool::with_num_threads(num_threads)
}

/// Get a thread pool with the specified number of threads.
///
/// Without the `threading` feature, this returns a `SerialPool` regardless
/// of the requested thread count.
#[cfg(not(feature = "threading"))]
#[must_use]
pub fn pool_with_threads(_num_threads: usize) -> SerialPool {
    SerialPool::new()
}

/// Configuration for creating thread pools.
#[derive(Clone, Debug, Default)]
pub struct PoolConfig {
    /// Number of threads to use. 0 means use system default.
    pub num_threads: usize,
}

impl PoolConfig {
    /// Create a new pool configuration with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of threads.
    ///
    /// A value of 0 means use the system default (typically the number of CPU cores).
    #[must_use]
    pub fn threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Build a thread pool with this configuration.
    ///
    /// When the `threading` feature is enabled, returns a `RayonPool`.
    /// Otherwise returns a `SerialPool`.
    #[cfg(feature = "threading")]
    #[must_use]
    pub fn build(self) -> RayonPool {
        if self.num_threads == 0 {
            RayonPool::new()
        } else {
            RayonPool::with_num_threads(self.num_threads)
        }
    }

    /// Build a thread pool with this configuration.
    #[cfg(not(feature = "threading"))]
    #[must_use]
    pub fn build(self) -> SerialPool {
        SerialPool::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_default_pool() {
        let pool = get_default_pool();
        assert!(pool.num_threads() >= 1);
    }

    #[test]
    fn test_pool_config() {
        let config = PoolConfig::new().threads(2);
        let pool = config.build();
        // SerialPool always returns 1, RayonPool returns configured value
        assert!(pool.num_threads() >= 1);
    }

    #[test]
    fn test_serial_pool() {
        let pool = SerialPool::new();
        assert_eq!(pool.num_threads(), 1);

        pool.parallel_for(5, |i| {
            // This is intentionally simple since SerialPool is sequential
            let _ = i;
        });

        // Test join
        let (a, b) = pool.join(|| 1, || 2);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
    }

    #[test]
    fn test_parallel_for_chunks() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let pool = get_default_pool();
        let counter = AtomicUsize::new(0);
        let n = 100;

        // Callback receives (start, len) where len is chunk length
        pool.parallel_for_chunks(n, 10, |_start, len| {
            counter.fetch_add(len, Ordering::SeqCst);
        });

        assert_eq!(counter.load(Ordering::SeqCst), n);
    }

    #[test]
    fn test_parallel_split() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let pool = get_default_pool();
        let counter = AtomicUsize::new(0);

        // Split 100 items with min chunk size 10
        pool.parallel_split(0, 100, 10, &|start, count| {
            for i in start..(start + count) {
                counter.fetch_add(i, Ordering::SeqCst);
            }
        });

        // Sum of 0..100 = 100*99/2 = 4950
        assert_eq!(counter.load(Ordering::SeqCst), 4950);
    }

    #[test]
    fn test_join() {
        let pool = get_default_pool();

        // Test join returns results from both tasks
        let (a, b) = pool.join(|| 42, || 43);
        assert_eq!(a, 42);
        assert_eq!(b, 43);
    }

    #[test]
    fn test_parallel_for_data_integrity() {
        use std::sync::atomic::{AtomicU64, Ordering};

        let pool = get_default_pool();
        let sum = AtomicU64::new(0);
        let n = 1000;

        pool.parallel_for(n, |i| {
            sum.fetch_add(i as u64, Ordering::SeqCst);
        });

        // Sum of 0..n = n*(n-1)/2
        let expected = (n as u64 * (n as u64 - 1)) / 2;
        assert_eq!(sum.load(Ordering::SeqCst), expected);
    }

    #[cfg(feature = "threading")]
    #[test]
    fn test_rayon_pool_threads() {
        let pool = RayonPool::with_num_threads(4);
        assert_eq!(pool.num_threads(), 4);
    }

    #[cfg(feature = "threading")]
    #[test]
    fn test_parallel_correctness_with_mutex() {
        use std::sync::Mutex;

        let pool = get_default_pool();
        let results = Mutex::new(vec![0usize; 100]);

        pool.parallel_for(100, |i| {
            let mut r = results.lock().unwrap();
            r[i] = i * 2;
        });

        let r = results.lock().unwrap();
        for i in 0..100 {
            assert_eq!(r[i], i * 2, "Element {i} has wrong value");
        }
    }

    #[cfg(feature = "threading")]
    #[test]
    fn test_parallel_chunks_boundary() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Test edge case: n not evenly divisible by chunk_size
        let pool = get_default_pool();
        let counter = AtomicUsize::new(0);
        let n = 97; // Prime number

        // Callback receives (start, len)
        pool.parallel_for_chunks(n, 10, |start, len| {
            for i in start..(start + len) {
                counter.fetch_add(i, Ordering::SeqCst);
            }
        });

        let expected: usize = (0..n).sum();
        assert_eq!(counter.load(Ordering::SeqCst), expected);
    }

    #[cfg(feature = "threading")]
    #[test]
    fn test_nested_parallel() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let pool = get_default_pool();
        let counter = AtomicUsize::new(0);

        // Nested parallel operations
        pool.parallel_for(4, |_i| {
            // Inner loop (sequential within this thread)
            for j in 0..25 {
                counter.fetch_add(j, Ordering::SeqCst);
            }
        });

        // 4 outer iterations, each doing sum(0..25) = 25*24/2 = 300
        let expected = 4 * (25 * 24 / 2);
        assert_eq!(counter.load(Ordering::SeqCst), expected);
    }
}
