//! Thread spawning abstraction.

/// Thread pool trait for parallel execution.
///
/// This trait provides an abstraction over different threading backends.
/// Two implementations are provided:
/// - [`SerialPool`](super::SerialPool): Single-threaded execution (always available)
/// - [`RayonPool`](super::RayonPool): Rayon-based parallel execution (requires `threading` feature)
pub trait ThreadPool: Send + Sync {
    /// Execute function in parallel over range [0, count).
    ///
    /// The function `f` is called for each index in `0..count`.
    /// Implementations may execute these calls in parallel across multiple threads.
    fn parallel_for<F>(&self, count: usize, f: F)
    where
        F: Fn(usize) + Send + Sync;

    /// Number of threads available in this pool.
    fn num_threads(&self) -> usize;

    /// Execute two tasks in parallel and return both results.
    ///
    /// This is useful for fork-join parallelism patterns like divide-and-conquer.
    fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send;

    /// Execute function in parallel over range with chunking.
    ///
    /// Divides `count` iterations into chunks, each chunk processed by one thread.
    /// The function `f` receives the chunk start index and chunk size.
    fn parallel_for_chunks<F>(&self, count: usize, chunk_size: usize, f: F)
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        let num_chunks = (count + chunk_size - 1) / chunk_size;
        self.parallel_for(num_chunks, |chunk_idx| {
            let start = chunk_idx * chunk_size;
            let len = core::cmp::min(chunk_size, count - start);
            f(start, len);
        });
    }

    /// Recursively split work in parallel using join.
    ///
    /// Splits the range `[start, start + count)` recursively until chunks are
    /// smaller than `min_chunk_size`, then executes `f` on each chunk.
    fn parallel_split<F>(&self, start: usize, count: usize, min_chunk_size: usize, f: &F)
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        if count <= min_chunk_size || self.num_threads() <= 1 {
            f(start, count);
        } else {
            let mid = count / 2;
            self.join(
                || self.parallel_split(start, mid, min_chunk_size, f),
                || self.parallel_split(start + mid, count - mid, min_chunk_size, f),
            );
        }
    }
}
