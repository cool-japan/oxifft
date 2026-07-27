//! Rayon-based parallel execution.

use std::sync::Arc;

use rayon::prelude::*;

use super::spawn::ThreadPool;

/// Rayon-based thread pool.
///
/// By default (see [`RayonPool::new`]) this dispatches work onto rayon's
/// ambient/global thread pool. When constructed with an explicit thread
/// count via [`RayonPool::with_num_threads`], it instead builds a dedicated
/// `rayon::ThreadPool` sized to exactly that many worker threads and runs
/// all dispatched work through [`rayon::ThreadPool::install`], so the
/// requested limit is a real, enforced bound rather than a reporting-only
/// value.
#[derive(Clone)]
pub struct RayonPool {
    /// Dedicated rayon thread pool, or `None` to use rayon's ambient/global
    /// pool. When `Some`, all work is dispatched via `install` so it never
    /// spills onto the ambient pool's threads.
    pool: Option<Arc<rayon::ThreadPool>>,
    /// Thread count to report when `pool` is `None` (cached at construction
    /// time from rayon's ambient pool). Ignored when `pool` is `Some`, since
    /// `num_threads()` then queries the dedicated pool directly.
    num_threads: usize,
}

impl Default for RayonPool {
    fn default() -> Self {
        Self::new()
    }
}

impl RayonPool {
    /// Create a new Rayon pool that dispatches onto rayon's ambient/global
    /// thread pool.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pool: None,
            num_threads: rayon::current_num_threads(),
        }
    }

    /// Create a Rayon pool bounded to a specific thread count.
    ///
    /// Builds a dedicated `rayon::ThreadPool` with exactly `num_threads`
    /// worker threads. All work submitted through this pool's
    /// [`ThreadPool`] methods (`parallel_for`, `join`, etc.) is dispatched
    /// via [`rayon::ThreadPool::install`], so the requested thread count is
    /// an actual, enforced upper bound on parallelism rather than a purely
    /// informational value.
    ///
    /// A `num_threads` of `0` behaves like [`RayonPool::new`] and uses the
    /// ambient global pool. If the dedicated pool fails to build (e.g. the
    /// OS refuses to spawn worker threads), this falls back to the ambient
    /// global pool rather than panicking; callers can detect this by
    /// checking [`RayonPool::num_threads`] against the requested value.
    #[must_use]
    pub fn with_num_threads(num_threads: usize) -> Self {
        if num_threads == 0 {
            return Self::new();
        }
        match rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
        {
            Ok(pool) => Self {
                pool: Some(Arc::new(pool)),
                num_threads,
            },
            Err(_) => Self::new(),
        }
    }

    /// Returns `true` if this pool holds a dedicated (non-ambient) rayon
    /// thread pool, i.e. it was constructed via a successful
    /// [`RayonPool::with_num_threads`] call.
    #[must_use]
    pub fn has_dedicated_pool(&self) -> bool {
        self.pool.is_some()
    }
}

impl ThreadPool for RayonPool {
    fn parallel_for<F>(&self, count: usize, f: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        if !super::should_parallelize(count, self.num_threads()) {
            for i in 0..count {
                f(i);
            }
            return;
        }

        match &self.pool {
            Some(pool) => pool.install(|| (0..count).into_par_iter().for_each(f)),
            None => (0..count).into_par_iter().for_each(f),
        }
    }

    fn num_threads(&self) -> usize {
        match &self.pool {
            Some(pool) => pool.current_num_threads(),
            None => self.num_threads,
        }
    }

    fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        match &self.pool {
            Some(pool) => pool.install(|| rayon::join(a, b)),
            None => rayon::join(a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Mutex;
    use std::thread::ThreadId;

    #[test]
    fn test_with_num_threads_builds_dedicated_pool() {
        let pool = RayonPool::with_num_threads(3);
        assert!(pool.has_dedicated_pool());
        assert_eq!(pool.num_threads(), 3);
    }

    #[test]
    fn test_zero_threads_uses_ambient_pool() {
        let pool = RayonPool::with_num_threads(0);
        assert!(!pool.has_dedicated_pool());
        assert_eq!(pool.num_threads(), rayon::current_num_threads());
    }

    /// The configured thread count must be a *real* bound: code running
    /// inside `parallel_for` (dispatched via `install`) must observe rayon's
    /// ambient thread count as the dedicated pool's size, not the process's
    /// global pool size.
    #[test]
    fn test_parallel_for_installs_into_dedicated_pool() {
        let pool = RayonPool::with_num_threads(3);

        // Use a count well above the global ParallelConfig threshold
        // (num_threads * min_batch_chunk == 3*4 == 12 by default) so this
        // is guaranteed to dispatch via `install` rather than the small-
        // workload serial fallback.
        let observed: Mutex<HashSet<usize>> = Mutex::new(HashSet::new());
        pool.parallel_for(64, |_| {
            observed
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(rayon::current_num_threads());
        });

        // Every task, wherever it ran, must have observed rayon's ambient
        // thread count as the *dedicated* pool's size -- never the
        // process's global pool size.
        assert_eq!(
            *observed.lock().unwrap_or_else(|e| e.into_inner()),
            HashSet::from([3])
        );
    }

    /// Requesting exactly one thread must bound execution to a single OS
    /// thread: with only one worker in the dedicated pool, every task in
    /// `parallel_for` is structurally forced onto that same thread, so this
    /// assertion cannot be flaky.
    #[test]
    fn test_single_thread_pool_uses_one_os_thread() {
        let pool = RayonPool::with_num_threads(1);
        assert_eq!(pool.num_threads(), 1);

        let seen: Mutex<HashSet<ThreadId>> = Mutex::new(HashSet::new());
        // Use a large count and force actual parallel dispatch regardless
        // of ParallelConfig thresholds by exceeding min_batch_chunk * 1.
        pool.parallel_for(64, |_| {
            seen.lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(std::thread::current().id());
        });

        assert_eq!(seen.lock().unwrap_or_else(|e| e.into_inner()).len(), 1);
    }

    /// A dedicated pool with `n` worker threads can never involve more than
    /// `n` distinct OS threads while executing work through `install`, no
    /// matter how much work is submitted or how it interleaves. This is a
    /// structural guarantee, not a timing-dependent one.
    #[test]
    fn test_dedicated_pool_bounds_distinct_threads() {
        let pool = RayonPool::with_num_threads(2);

        let seen: Mutex<HashSet<ThreadId>> = Mutex::new(HashSet::new());
        pool.parallel_for(64, |i| {
            // A tiny amount of work so the scheduler has a chance to spread
            // iterations across both workers instead of draining serially
            // on one.
            std::thread::yield_now();
            let _ = i;
            seen.lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(std::thread::current().id());
        });

        let distinct = seen.lock().unwrap_or_else(|e| e.into_inner()).len();
        assert!(
            (1..=2).contains(&distinct),
            "observed {distinct} distinct threads, expected <= 2"
        );
    }

    /// `join` must also be scoped to the dedicated pool so fork-join code
    /// paths (used by `parallel_split`) respect the configured limit too.
    #[test]
    fn test_join_installs_into_dedicated_pool() {
        let pool = RayonPool::with_num_threads(2);
        let (a, b) = pool.join(rayon::current_num_threads, rayon::current_num_threads);
        assert_eq!(a, 2);
        assert_eq!(b, 2);
    }

    /// Nested-parallelism guard: dispatching a second, inner `parallel_for`
    /// from within a task already running under `install` on a dedicated
    /// pool must remain scoped to that same pool (no leakage back onto the
    /// ambient/global pool, and no additional threads spun up per nesting
    /// level).
    #[test]
    fn test_nested_parallel_for_stays_within_dedicated_pool() {
        let pool = RayonPool::with_num_threads(2);

        // Outer count well above the 2*4=8 default threshold so the outer
        // level is guaranteed to dispatch via `install` rather than the
        // small-workload serial fallback.
        let inner_observed: Mutex<HashSet<usize>> = Mutex::new(HashSet::new());
        pool.parallel_for(32, |_outer| {
            // Nested dispatch: plain rayon parallel iterator, not routed
            // through our `ThreadPool` abstraction, mirroring how Plan2D's
            // row/column passes nest parallel work inside `ws.install`.
            (0..4).into_par_iter().for_each(|_inner| {
                inner_observed
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(rayon::current_num_threads());
            });
        });

        // Every observation, at every nesting depth, must see the dedicated
        // pool's thread count -- never the ambient global pool's count.
        let observed = inner_observed.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(
            observed.len(),
            1,
            "nested work observed multiple thread counts: {observed:?}"
        );
        assert!(observed.contains(&2));
    }

    #[test]
    fn test_small_workload_falls_back_to_serial_bound_by_config() {
        // A count far below the default min_batch_chunk*num_threads
        // threshold should still visit every index exactly once, whether
        // or not it actually dispatches in parallel.
        let pool = RayonPool::with_num_threads(8);
        let seen: Mutex<Vec<usize>> = Mutex::new(Vec::new());
        pool.parallel_for(3, |i| {
            seen.lock().unwrap_or_else(|e| e.into_inner()).push(i);
        });
        let mut got = seen.lock().unwrap_or_else(|e| e.into_inner()).clone();
        got.sort_unstable();
        assert_eq!(got, vec![0, 1, 2]);
    }
}
