//! Rayon-based parallel execution.

use rayon::prelude::*;

use super::spawn::ThreadPool;

/// Rayon-based thread pool.
#[derive(Clone)]
pub struct RayonPool {
    num_threads: usize,
}

impl Default for RayonPool {
    fn default() -> Self {
        Self::new()
    }
}

impl RayonPool {
    /// Create a new Rayon pool with default thread count.
    #[must_use]
    pub fn new() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
        }
    }

    /// Create a Rayon pool with specific thread count.
    #[must_use]
    pub fn with_num_threads(num_threads: usize) -> Self {
        Self { num_threads }
    }
}

impl ThreadPool for RayonPool {
    fn parallel_for<F>(&self, count: usize, f: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        (0..count).into_par_iter().for_each(|i| f(i));
    }

    fn num_threads(&self) -> usize {
        self.num_threads
    }

    fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        rayon::join(a, b)
    }
}
