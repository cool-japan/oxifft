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
    ///
    /// # Contract
    ///
    /// A correct implementation calls `f` **exactly once for every index in
    /// `0..count`**, and never with an index outside that range. Callers that
    /// partition a buffer by index (see [`crate::api::ParallelPlan2D`]) rely on
    /// this to obtain disjoint sub-buffers.
    ///
    /// This is a *safe* trait, so the crate cannot assume the contract holds:
    /// every in-crate raw-pointer partitioning site independently rejects
    /// out-of-range and repeated indices, so a misbehaving implementation
    /// produces incomplete results rather than undefined behaviour.
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
    ///
    /// `f` is never called when `count == 0` or `chunk_size == 0`; a zero chunk
    /// size describes no work rather than being an error (this mirrors
    /// [`WorkStealingContext::par_map_slices_mut`](super::WorkStealingContext::par_map_slices_mut)).
    /// All index arithmetic is saturating, so no argument combination can
    /// overflow, divide by zero, or underflow.
    fn parallel_for_chunks<F>(&self, count: usize, chunk_size: usize, f: F)
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        if count == 0 || chunk_size == 0 {
            return;
        }
        // `div_ceil` avoids the `count + chunk_size - 1` overflow/underflow of a
        // hand-rolled ceiling division.
        let num_chunks = count.div_ceil(chunk_size);
        self.parallel_for(num_chunks, |chunk_idx| {
            // `saturating_mul` + the `>= count` guard keep the reported window
            // inside `0..count` even if `parallel_for` hands back an
            // out-of-range index (the trait is safe, so that is possible).
            let start = chunk_idx.saturating_mul(chunk_size);
            if start >= count {
                return;
            }
            let len = core::cmp::min(chunk_size, count - start);
            f(start, len);
        });
    }

    /// Recursively split work in parallel using join.
    ///
    /// Splits the range `[start, start + count)` recursively until chunks are
    /// smaller than `min_chunk_size`, then executes `f` on each chunk.
    ///
    /// `min_chunk_size` is clamped to at least 1: with a literal `0` the
    /// recursion could never shrink a one-element range (`mid == 0`, so the
    /// right half stays at length 1 forever) and would overflow the stack.
    fn parallel_split<F>(&self, start: usize, count: usize, min_chunk_size: usize, f: &F)
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        let min_chunk_size = min_chunk_size.max(1);
        if count <= min_chunk_size || self.num_threads() <= 1 {
            f(start, count);
        } else {
            let mid = count / 2;
            self.join(
                || self.parallel_split(start, mid, min_chunk_size, f),
                || self.parallel_split(start.saturating_add(mid), count - mid, min_chunk_size, f),
            );
        }
    }
}
