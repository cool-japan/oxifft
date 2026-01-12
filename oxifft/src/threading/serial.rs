//! Single-threaded execution.

use super::spawn::ThreadPool;

/// Single-threaded "pool" for serial execution.
#[derive(Clone, Copy)]
pub struct SerialPool;

impl Default for SerialPool {
    fn default() -> Self {
        Self::new()
    }
}

impl SerialPool {
    /// Create a new serial pool.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl ThreadPool for SerialPool {
    fn parallel_for<F>(&self, count: usize, f: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        for i in 0..count {
            f(i);
        }
    }

    fn num_threads(&self) -> usize {
        1
    }

    fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        (a(), b())
    }
}
