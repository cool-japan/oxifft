//! Work-stealing parallel execution context for multi-dimensional FFT transforms.
//!
//! Provides a thin wrapper around either the global rayon thread pool or a
//! user-supplied rayon `ThreadPool`, dispatching slice operations via
//! rayon's work-stealing scheduler for near-linear parallel scaling on
//! balanced data-parallel workloads (e.g., row-wise 2D FFT, plane-wise 3D FFT).
//!
//! # Design
//!
//! `WorkStealingContext` is designed to be embedded in plan types such as
//! `Plan2D` and `Plan3D`. When no custom pool is configured, it falls back
//! to the global rayon pool automatically.
//!
//! ```rust
//! use oxifft::threading::work_stealing::WorkStealingContext;
//!
//! let ctx = WorkStealingContext::new();
//! let mut data = vec![0u64; 64];
//! ctx.par_map_slices_mut(&mut data, 16, |slice| {
//!     for x in slice.iter_mut() {
//!         *x = x.wrapping_add(1);
//!     }
//! });
//! ```

use std::sync::Arc;

/// A parallel execution context backed by rayon's work-stealing scheduler.
///
/// By default it delegates to the global rayon thread pool. Optionally a
/// caller can supply a dedicated `rayon::ThreadPool` to isolate FFT work
/// from other rayon-based computation in the same process.
#[derive(Clone, Default)]
pub struct WorkStealingContext {
    /// Optional user-supplied rayon thread pool. `None` means "use global pool".
    pool: Option<Arc<rayon::ThreadPool>>,
}

impl WorkStealingContext {
    /// Create a context that uses the global rayon thread pool.
    #[must_use]
    pub fn new() -> Self {
        Self { pool: None }
    }

    /// Create a context that uses the provided rayon thread pool.
    ///
    /// The pool is shared via `Arc`, so cloning this context is cheap and
    /// all clones share the same pool.
    #[must_use]
    pub fn with_pool(pool: Arc<rayon::ThreadPool>) -> Self {
        Self { pool: Some(pool) }
    }

    /// Replace the pool in this context, returning a new context.
    ///
    /// Used by builder methods on plan types.
    #[must_use]
    pub fn with_rayon_pool(mut self, pool: Arc<rayon::ThreadPool>) -> Self {
        self.pool = Some(pool);
        self
    }

    /// Return the number of worker threads in the configured pool, or
    /// the global rayon pool thread count when no custom pool is set.
    #[must_use]
    pub fn num_threads(&self) -> usize {
        match &self.pool {
            Some(p) => p.current_num_threads(),
            None => rayon::current_num_threads(),
        }
    }

    /// Run `f` inside the configured thread pool.
    ///
    /// When a custom pool is set, `f` is `install`ed into that pool so all
    /// rayon work spawned by `f` runs on the pool's workers.  When no pool is
    /// configured, `f` is called directly — rayon work inside `f` uses the
    /// global pool as usual.
    pub fn install<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        match &self.pool {
            Some(pool) => pool.install(f),
            None => f(),
        }
    }

    /// Apply `f` to non-overlapping mutable slices of `data`, each of length
    /// `chunk_size` (the last slice may be shorter), in parallel using rayon.
    ///
    /// When a custom pool is configured, the work is `install`ed inside that
    /// pool; otherwise the global rayon pool is used directly.
    ///
    /// # Type Parameters
    ///
    /// * `T` — Element type; must be `Send` so slices can be sent across threads.
    /// * `F` — Closure applied to each slice; must be `Fn` (not `FnMut`) so it
    ///   can be called concurrently on multiple threads.  Must be `Send + Sync`.
    pub fn par_map_slices_mut<T, F>(&self, data: &mut [T], chunk_size: usize, f: F)
    where
        T: Send,
        F: Fn(&mut [T]) + Send + Sync,
    {
        use rayon::prelude::*;

        if chunk_size == 0 || data.is_empty() {
            return;
        }

        match &self.pool {
            Some(pool) => {
                pool.install(|| {
                    data.par_chunks_mut(chunk_size).for_each(&f);
                });
            }
            None => {
                data.par_chunks_mut(chunk_size).for_each(f);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_context_increments_all_chunks() {
        let ctx = WorkStealingContext::new();
        let mut data = vec![0u32; 100];
        ctx.par_map_slices_mut(&mut data, 10, |chunk| {
            for v in chunk.iter_mut() {
                *v = 1;
            }
        });
        assert!(data.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_with_custom_pool() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("failed to build pool");
        let ctx = WorkStealingContext::with_pool(Arc::new(pool));
        assert_eq!(ctx.num_threads(), 2);

        let mut data = vec![0i64; 64];
        ctx.par_map_slices_mut(&mut data, 16, |chunk| {
            for v in chunk.iter_mut() {
                *v += 7;
            }
        });
        assert!(data.iter().all(|&v| v == 7));
    }

    #[test]
    fn test_with_rayon_pool_builder() {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("pool build failed"),
        );
        let ctx = WorkStealingContext::new().with_rayon_pool(pool);
        let mut data = vec![1u8; 32];
        ctx.par_map_slices_mut(&mut data, 8, |chunk| {
            for v in chunk.iter_mut() {
                *v = v.wrapping_mul(2);
            }
        });
        assert!(data.iter().all(|&v| v == 2));
    }

    #[test]
    fn test_empty_data_is_noop() {
        let ctx = WorkStealingContext::new();
        let mut data: Vec<u32> = vec![];
        // Must not panic.
        ctx.par_map_slices_mut(&mut data, 8, |_chunk| panic!("should not be called"));
    }

    #[test]
    fn test_chunk_size_zero_is_noop() {
        let ctx = WorkStealingContext::new();
        let mut data = vec![42u32; 8];
        // chunk_size == 0: must not panic, data is unchanged.
        ctx.par_map_slices_mut(&mut data, 0, |_chunk| panic!("should not be called"));
        assert!(data.iter().all(|&v| v == 42));
    }
}
