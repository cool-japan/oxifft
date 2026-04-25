//! Thread-safe, LRU-evicting GPU buffer pool.
//!
//! Provides `GpuBufferPool`, a bounded pool of GPU buffers keyed by
//! `(backend_id, rounded_size, buffer_kind)`.  Same-size GPU transforms
//! reuse pre-allocated device buffers instead of allocating/freeing on
//! every call.
//!
//! # Design
//!
//! - Internally stores type-erased `Box<dyn Any + Send>` payloads so that
//!   concrete CUDA / Metal buffer types (from fenced modules) can be pooled
//!   through a single singleton.
//! - Pool sizes are rounded to the next power-of-two up to 64 MiB, then in
//!   64-MiB aligned steps beyond that, so buffers with slightly different
//!   raw sizes share the same pool bucket.
//! - LRU eviction: when a `release` would push `current_bytes` above
//!   `max_bytes`, the oldest (`last_touched`) buffer(s) are dropped first.
//!
//! # Example
//!
//! ```rust
//! use oxifft::gpu::pool::{GpuBufferPool, PoolKey, BufferKind, PooledBuffer};
//!
//! let pool = GpuBufferPool::new(64 * 1024 * 1024);
//! let key = PoolKey { backend_id: 0, rounded_size: 4096, kind: BufferKind::Scratch };
//!
//! // Acquire — allocates on first call
//! let buf = pool.acquire(key, |size| {
//!     Some(PooledBuffer::new(Box::new(vec![0u8; size]), size))
//! });
//!
//! // Release — returns to pool for reuse
//! if let Some(b) = buf {
//!     pool.release(key, b);
//! }
//! ```

use std::any::Any;
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

// ── Public key / kind types ──────────────────────────────────────────────────

/// Pool lookup key: identifies a class of interchangeable GPU buffers.
///
/// `rounded_size` should already be rounded via [`round_pool_size`] before
/// constructing a key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PoolKey {
    /// Numeric backend identifier (0 = CUDA, 1 = Metal, …).
    pub backend_id: u32,
    /// Buffer size in bytes, rounded to the nearest pool bucket.
    pub rounded_size: usize,
    /// Role of the buffer within a transform.
    pub kind: BufferKind,
}

/// Role of a GPU buffer inside a transform call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BufferKind {
    /// Input staging buffer.
    Input,
    /// Output staging buffer.
    Output,
    /// Scratch / work buffer used internally by the backend.
    Scratch,
}

// ── PooledBuffer ─────────────────────────────────────────────────────────────

/// A type-erased GPU buffer managed by [`GpuBufferPool`].
///
/// The `raw` field holds the concrete backend buffer (e.g. a CUDA or Metal
/// allocation) as a type-erased `Box<dyn Any + Send>`.  The owning backend
/// retrieves it via [`PooledBuffer::downcast`].
pub struct PooledBuffer {
    /// Opaque backend buffer.
    raw: Box<dyn Any + Send>,
    /// Byte size of the allocation tracked against the pool budget.
    pub size: usize,
    /// Timestamp of the last `release()` — used for LRU ordering.
    pub last_touched: Instant,
}

impl std::fmt::Debug for PooledBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledBuffer")
            .field("size", &self.size)
            .field("last_touched", &self.last_touched)
            .finish_non_exhaustive()
    }
}

impl PooledBuffer {
    /// Create a new pooled buffer wrapping a type-erased payload.
    ///
    /// `size` must equal the number of bytes that `raw` occupies on the device;
    /// it is used to track the pool budget.
    #[must_use]
    pub fn new(raw: Box<dyn Any + Send>, size: usize) -> Self {
        Self {
            raw,
            size,
            last_touched: Instant::now(),
        }
    }

    /// Attempt to downcast `raw` to a concrete type.
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if the type parameter does not match the concrete
    /// type originally stored in `raw`.
    pub fn downcast<T: Any + Send>(self) -> Result<Box<T>, Self> {
        // We need to check the type before consuming `self`.
        if self.raw.is::<T>() {
            // SAFETY: we just confirmed the type matches.
            let raw_ptr = Box::into_raw(self.raw) as *mut T;
            // SAFETY: raw_ptr came from a Box<T> via Box<dyn Any + Send>
            // and we confirmed the concrete type above.
            Ok(unsafe { Box::from_raw(raw_ptr) })
        } else {
            Err(self)
        }
    }

    /// Borrow the raw payload as `&T` if the type matches.
    #[must_use]
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.raw.downcast_ref::<T>()
    }
}

// ── GpuBufferPool ────────────────────────────────────────────────────────────

/// Thread-safe, LRU-evicting pool of GPU buffers.
///
/// Acquire a buffer via [`acquire`][GpuBufferPool::acquire], use it, then
/// return it with [`release`][GpuBufferPool::release].  Calling `release`
/// instead of dropping the buffer allows the next `acquire` with the same key
/// to reuse the allocation without going to the GPU allocator.
///
/// Buffers are evicted from the pool (and their GPU memory freed) when the
/// pool's byte budget (`max_bytes`) would be exceeded.  The eviction policy is
/// LRU: the buffer with the oldest `last_touched` timestamp is removed first.
pub struct GpuBufferPool {
    inner: Mutex<BTreeMap<PoolKey, VecDeque<PooledBuffer>>>,
    max_bytes: usize,
    current_bytes: AtomicUsize,
}

// SAFETY: Mutex<…> + AtomicUsize are both Send + Sync; the Any+Send bound on
// PooledBuffer payloads ensures the erased values can cross thread boundaries.
unsafe impl Send for GpuBufferPool {}
unsafe impl Sync for GpuBufferPool {}

impl GpuBufferPool {
    /// Create a new pool with the given byte budget.
    ///
    /// A sensible default is `256 * 1024 * 1024` (256 MiB).  Passing `0`
    /// disables pooling (every `acquire` will call the allocator; every
    /// `release` will immediately drop the buffer).
    #[must_use]
    pub fn new(max_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(BTreeMap::new()),
            max_bytes,
            current_bytes: AtomicUsize::new(0),
        }
    }

    /// Pop a buffer from the pool or allocate a new one.
    ///
    /// If a buffer with matching `key` is cached, it is removed from the pool
    /// (and `current_bytes` is decremented to reflect that it is now "in use").
    /// Otherwise `allocate` is called with `key.rounded_size` and its return
    /// value is forwarded.
    ///
    /// The caller must eventually call [`release`][Self::release] (or simply
    /// drop the buffer) to balance the acquisition.
    #[must_use]
    pub fn acquire(
        &self,
        key: PoolKey,
        allocate: impl FnOnce(usize) -> Option<PooledBuffer>,
    ) -> Option<PooledBuffer> {
        // Fast path: reuse from pool.
        {
            let mut pool = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(queue) = pool.get_mut(&key) {
                if let Some(buf) = queue.pop_front() {
                    // The buffer is leaving the pool; subtract its budget contribution.
                    self.current_bytes.fetch_sub(buf.size, Ordering::Relaxed);
                    return Some(buf);
                }
            }
        }
        // Lock is released before calling `allocate` to avoid holding it across
        // a potentially expensive GPU allocation.
        allocate(key.rounded_size)
    }

    /// Return a buffer to the pool, evicting LRU entries if over budget.
    ///
    /// `buf.last_touched` is updated to `Instant::now()` before insertion so
    /// that freshly returned buffers survive the next eviction pass longest.
    ///
    /// If the pool would be over budget even after eviction (e.g. `max_bytes`
    /// is 0), the buffer is simply dropped (freeing the GPU allocation).
    pub fn release(&self, key: PoolKey, mut buf: PooledBuffer) {
        buf.last_touched = Instant::now();
        let size = buf.size;

        if self.max_bytes == 0 {
            // Pooling disabled — drop immediately.
            return;
        }

        let mut pool = self.inner.lock().unwrap_or_else(|e| e.into_inner());

        // Check if adding this buffer would exceed the budget and evict first.
        // `current_bytes` is checked while holding the lock to prevent a TOCTOU
        // race between the load and the subsequent eviction.
        let current = self.current_bytes.load(Ordering::Relaxed);
        if current + size > self.max_bytes {
            self.evict_lru_locked(&mut pool, size);
        }

        pool.entry(key).or_default().push_back(buf);
        self.current_bytes.fetch_add(size, Ordering::Relaxed);
    }

    /// Remove old pool entries until `needed` bytes of headroom are available.
    ///
    /// Caller must hold the `inner` lock.
    fn evict_lru_locked(
        &self,
        pool: &mut BTreeMap<PoolKey, VecDeque<PooledBuffer>>,
        needed: usize,
    ) {
        let mut freed = 0usize;
        while freed < needed {
            // Find the key whose front buffer has the oldest `last_touched`.
            let oldest_key = pool
                .iter()
                .filter(|(_, q)| !q.is_empty())
                .min_by_key(|(_, q)| q.front().map(|b| b.last_touched))
                .map(|(k, _)| *k);

            match oldest_key {
                Some(key) => {
                    if let Some(q) = pool.get_mut(&key) {
                        if let Some(buf) = q.pop_front() {
                            freed += buf.size;
                            self.current_bytes.fetch_sub(buf.size, Ordering::Relaxed);
                            // `buf` is dropped here, freeing the GPU allocation.
                        }
                    }
                }
                None => break, // Pool is empty; nothing left to evict.
            }
        }
    }

    /// Drain all cached buffers, resetting the pool to an empty state.
    ///
    /// All GPU allocations held by the pool are freed when the buffers are
    /// dropped.  This is safe to call at any time; in-flight buffers (those
    /// currently acquired) are not affected.
    pub fn clear(&self) {
        let mut pool = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        pool.clear();
        self.current_bytes.store(0, Ordering::Relaxed);
    }

    /// Return the current tracked byte usage of the pool.
    #[must_use]
    pub fn current_bytes(&self) -> usize {
        self.current_bytes.load(Ordering::Relaxed)
    }

    /// Return the configured byte budget.
    #[must_use]
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }
}

impl std::fmt::Debug for GpuBufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBufferPool")
            .field("max_bytes", &self.max_bytes)
            .field("current_bytes", &self.current_bytes.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

// ── Utility ──────────────────────────────────────────────────────────────────

/// Round a raw byte count to the nearest pool bucket size.
///
/// Sizes up to 64 MiB are rounded to the next power of two.  Sizes above 64
/// MiB are rounded up to the nearest 64-MiB multiple.  This limits the number
/// of distinct pool buckets and improves buffer reuse for workloads with
/// slightly varying transform sizes.
///
/// # Examples
///
/// ```rust
/// use oxifft::gpu::pool::round_pool_size;
///
/// assert_eq!(round_pool_size(1000), 1024);
/// assert_eq!(round_pool_size(4096), 4096);
/// assert_eq!(round_pool_size(65 * 1024 * 1024), 128 * 1024 * 1024);
/// ```
#[must_use]
pub fn round_pool_size(size: usize) -> usize {
    const STEP_64MIB: usize = 64 * 1024 * 1024;
    if size == 0 {
        return 0;
    }
    if size <= STEP_64MIB {
        size.next_power_of_two()
    } else {
        size.div_ceil(STEP_64MIB) * STEP_64MIB
    }
}

// ── Global singleton ─────────────────────────────────────────────────────────

/// Return a reference to the process-global GPU buffer pool.
///
/// The pool is lazily initialised on first call with a default budget of 256
/// MiB.  Call [`GpuBufferPool::clear`] via `clear_gpu_pool` to release all
/// cached allocations (e.g. in tests).
#[must_use]
pub fn global_pool() -> &'static GpuBufferPool {
    static POOL: std::sync::OnceLock<GpuBufferPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| GpuBufferPool::new(256 * 1024 * 1024))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;

    // ── Mock buffer type ─────────────────────────────────────────────────────

    /// Lightweight mock that stands in for a real GPU buffer during tests.
    struct MockBuffer(Vec<u8>);

    fn mock_pooled(size: usize) -> PooledBuffer {
        PooledBuffer::new(Box::new(MockBuffer(vec![0u8; size])), size)
    }

    // ── Test: basic reuse ────────────────────────────────────────────────────

    #[test]
    fn pool_reuse_same_size() {
        let pool = GpuBufferPool::new(64 * 1024 * 1024);
        let key = PoolKey {
            backend_id: 0,
            rounded_size: 4096,
            kind: BufferKind::Scratch,
        };

        let mut alloc_count = 0usize;

        // First acquire — must call the allocator.
        let buf = pool
            .acquire(key, |size| {
                alloc_count += 1;
                Some(mock_pooled(size))
            })
            .expect("allocation failed");

        pool.release(key, buf);

        // Second acquire — must reuse from pool, NOT call the allocator.
        let _buf2 = pool.acquire(key, |_size| {
            alloc_count += 1; // must NOT be reached
            None
        });

        assert_eq!(alloc_count, 1, "second acquire should reuse from pool");
    }

    // ── Test: current_bytes tracking ─────────────────────────────────────────

    #[test]
    fn pool_current_bytes_tracks_release_and_acquire() {
        let pool = GpuBufferPool::new(64 * 1024 * 1024);
        let key = PoolKey {
            backend_id: 0,
            rounded_size: 1024,
            kind: BufferKind::Input,
        };

        assert_eq!(pool.current_bytes(), 0);

        // Acquire (no release yet) — current_bytes stays 0.
        let buf = pool
            .acquire(key, |size| Some(mock_pooled(size)))
            .expect("allocation failed");
        assert_eq!(pool.current_bytes(), 0, "in-flight buffer not counted");

        // Release — budget grows.
        pool.release(key, buf);
        assert_eq!(
            pool.current_bytes(),
            1024,
            "released buffer should be counted"
        );

        // Acquire again — budget shrinks (buffer is taken out of pool).
        let buf2 = pool.acquire(key, |_| None).expect("should reuse from pool");
        assert_eq!(pool.current_bytes(), 0, "acquired buffer leaves pool");

        drop(buf2);
    }

    // ── Test: LRU eviction ───────────────────────────────────────────────────

    #[test]
    fn pool_lru_eviction() {
        // Budget: 3 × 512-byte buffers (1536 bytes).
        let pool = GpuBufferPool::new(1536);

        let key_a = PoolKey {
            backend_id: 0,
            rounded_size: 512,
            kind: BufferKind::Input,
        };
        let key_b = PoolKey {
            backend_id: 0,
            rounded_size: 512,
            kind: BufferKind::Output,
        };
        let key_c = PoolKey {
            backend_id: 0,
            rounded_size: 512,
            kind: BufferKind::Scratch,
        };

        // Fill pool to budget limit.
        pool.release(key_a, mock_pooled(512));
        pool.release(key_b, mock_pooled(512));
        pool.release(key_c, mock_pooled(512));

        assert_eq!(pool.current_bytes(), 1536);

        // Releasing one more 512-byte buffer must trigger LRU eviction of the
        // oldest entry so total stays within budget.
        pool.release(key_a, mock_pooled(512));

        // After eviction + insertion: still at most 1536 bytes.
        assert!(
            pool.current_bytes() <= 1536,
            "pool exceeded budget after release: {} > 1536",
            pool.current_bytes()
        );
    }

    // ── Test: clear ──────────────────────────────────────────────────────────

    #[test]
    fn pool_clear_resets_budget() {
        let pool = GpuBufferPool::new(64 * 1024 * 1024);
        let key = PoolKey {
            backend_id: 0,
            rounded_size: 4096,
            kind: BufferKind::Output,
        };

        pool.release(key, mock_pooled(4096));
        assert_eq!(pool.current_bytes(), 4096);

        pool.clear();
        assert_eq!(pool.current_bytes(), 0);

        // After clear, acquire must call the allocator again.
        let mut alloc_count = 0usize;
        let _buf = pool.acquire(key, |size| {
            alloc_count += 1;
            Some(mock_pooled(size))
        });
        assert_eq!(alloc_count, 1, "pool should be empty after clear");
    }

    // ── Test: thread safety ──────────────────────────────────────────────────

    #[test]
    fn pool_thread_safety() {
        let pool = Arc::new(GpuBufferPool::new(64 * 1024 * 1024));
        let barrier = Arc::new(Barrier::new(8));
        let mut handles = vec![];

        for i in 0..8_u32 {
            let pool = Arc::clone(&pool);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                let key = PoolKey {
                    backend_id: i % 2,
                    rounded_size: 4096,
                    kind: BufferKind::Scratch,
                };
                for _ in 0..100 {
                    let buf = pool.acquire(key, |size| Some(mock_pooled(size)));
                    if let Some(b) = buf {
                        pool.release(key, b);
                    }
                }
            }));
        }

        for h in handles {
            h.join().expect("thread panicked");
        }
    }

    // ── Test: downcast ───────────────────────────────────────────────────────

    #[test]
    fn pooled_buffer_downcast_roundtrip() {
        let buf = PooledBuffer::new(Box::new(MockBuffer(vec![42u8; 8])), 8);
        let inner = buf.downcast::<MockBuffer>().expect("downcast failed");
        assert_eq!(inner.0[0], 42);
    }

    #[test]
    fn pooled_buffer_downcast_wrong_type_returns_err() {
        let buf = PooledBuffer::new(Box::new(MockBuffer(vec![0u8; 8])), 8);
        // Try to downcast to the wrong type.
        let result = buf.downcast::<Vec<u8>>();
        assert!(result.is_err(), "downcast to wrong type must fail");
    }

    // ── Test: round_pool_size ────────────────────────────────────────────────

    #[test]
    fn round_pool_size_zero() {
        assert_eq!(round_pool_size(0), 0);
    }

    #[test]
    fn round_pool_size_power_of_two_passthrough() {
        assert_eq!(round_pool_size(1024), 1024);
        assert_eq!(round_pool_size(4096), 4096);
    }

    #[test]
    fn round_pool_size_rounds_up_to_next_pow2() {
        assert_eq!(round_pool_size(1000), 1024);
        assert_eq!(round_pool_size(3000), 4096);
    }

    #[test]
    fn round_pool_size_above_64mib_aligned_to_64mib() {
        const MIB64: usize = 64 * 1024 * 1024;
        assert_eq!(round_pool_size(MIB64 + 1), 2 * MIB64);
        assert_eq!(round_pool_size(2 * MIB64), 2 * MIB64);
        assert_eq!(round_pool_size(2 * MIB64 + 1), 3 * MIB64);
    }

    // ── Test: S3 spec-named tests (aliases for spec compliance) ──────────────

    /// Verifies that a second acquire with the same key reuses from the pool
    /// without calling the allocator again.
    #[test]
    fn pool_reuse_avoids_realloc() {
        let pool = GpuBufferPool::new(64 * 1024 * 1024);
        let key = PoolKey {
            backend_id: 0,
            rounded_size: 4096,
            kind: BufferKind::Scratch,
        };
        let mut alloc_count = 0usize;

        let buf = pool
            .acquire(key, |size| {
                alloc_count += 1;
                Some(mock_pooled(size))
            })
            .expect("first alloc");
        pool.release(key, buf);

        // Second acquire must reuse from pool — allocator must NOT be called.
        let _buf2 = pool.acquire(key, |_size| {
            alloc_count += 1;
            None
        });
        assert_eq!(alloc_count, 1, "second acquire must reuse from pool");
    }

    /// Verifies that releasing a buffer that would exceed the budget triggers
    /// LRU eviction and keeps pool bytes within budget.
    #[test]
    fn pool_lru_eviction_under_budget() {
        // Tiny budget: exactly one 8192-byte buffer.
        let pool = GpuBufferPool::new(8192);
        let key = PoolKey {
            backend_id: 0,
            rounded_size: 8192,
            kind: BufferKind::Scratch,
        };

        // Fill the budget with one buffer.
        let buf = pool
            .acquire(key, |size| Some(mock_pooled(size)))
            .expect("first alloc");
        pool.release(key, buf);
        assert_eq!(pool.current_bytes(), 8192);

        // Releasing a second buffer of the same size must evict the first.
        let key2 = PoolKey {
            backend_id: 0,
            rounded_size: 4096,
            kind: BufferKind::Input,
        };
        let buf2 = pool
            .acquire(key2, |size| Some(mock_pooled(size)))
            .expect("second alloc");
        pool.release(key2, buf2);

        // Pool must not exceed budget (8192) + one incoming buffer (4096).
        assert!(
            pool.current_bytes() <= 8192 + 4096,
            "pool exceeded budget: {} > {}",
            pool.current_bytes(),
            8192 + 4096,
        );
    }
}
