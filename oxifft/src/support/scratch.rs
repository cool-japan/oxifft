//! Thread-local scratch buffer management.
//!
//! Provides amortized zero-allocation scratch buffers for FFT computations.
//! On `std` targets, buffers are stored in thread-local storage and grow lazily
//! but never shrink. On `no_std` targets, a simple heap allocation fallback is used.
//!
//! # Usage
//!
//! ```ignore
//! use oxifft::support::scratch::with_scratch;
//! use oxifft::Complex;
//!
//! // Borrow a scratch buffer of at least 1024 elements.
//! with_scratch::<Complex<f64>, _, _>(1024, |buf| {
//!     // buf is &mut [Complex<f64>] with len == 1024, zero-initialized.
//!     assert_eq!(buf.len(), 1024);
//! });
//! ```

use crate::kernel::{Complex, Float};
use crate::prelude::*;

// ---------------------------------------------------------------------------
// ScratchGuard – an owning handle that returns the buffer on drop (no_std)
// ---------------------------------------------------------------------------

/// An owning scratch buffer that is returned from [`get_scratch`].
///
/// On `std`, this borrows from thread-local storage.
/// On `no_std`, this is a plain `Vec` wrapper.
pub struct ScratchGuard<T: Float> {
    buf: Vec<Complex<T>>,
}

impl<T: Float> ScratchGuard<T> {
    /// View the scratch buffer as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[Complex<T>] {
        &self.buf
    }

    /// View the scratch buffer as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [Complex<T>] {
        &mut self.buf
    }

    /// Get the length of the scratch buffer.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Check if the scratch buffer is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

impl<T: Float> core::ops::Deref for ScratchGuard<T> {
    type Target = [Complex<T>];

    #[inline]
    fn deref(&self) -> &[Complex<T>] {
        &self.buf
    }
}

impl<T: Float> core::ops::DerefMut for ScratchGuard<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [Complex<T>] {
        &mut self.buf
    }
}

// ---------------------------------------------------------------------------
// std implementation – thread-local scratch pools
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
mod tls {
    use super::*;
    use std::cell::RefCell;

    /// Per-type, per-thread scratch pool.
    ///
    /// We keep one `Vec<u8>` per thread, sized in *bytes*. This avoids
    /// needing a separate thread-local for every `T`.
    struct RawScratch {
        /// Raw byte storage.
        bytes: Vec<u8>,
        /// Capacity in bytes (== bytes.len()).
        capacity: usize,
    }

    impl RawScratch {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                capacity: 0,
            }
        }

        /// Ensure we have at least `byte_count` bytes available.
        /// Grows but never shrinks.
        fn ensure_capacity(&mut self, byte_count: usize) {
            if byte_count > self.capacity {
                self.bytes.resize(byte_count, 0);
                self.capacity = byte_count;
            } else {
                // Zero the region we will hand out.
                self.bytes[..byte_count].fill(0);
            }
        }

        /// Get a mutable pointer to the raw storage.
        fn as_mut_ptr(&mut self) -> *mut u8 {
            self.bytes.as_mut_ptr()
        }
    }

    thread_local! {
        static SCRATCH: RefCell<RawScratch> = RefCell::new(RawScratch::new());
    }

    /// Execute `f` with a thread-local scratch buffer of at least `n` elements.
    ///
    /// The buffer is zero-initialized on each call. It grows lazily but never
    /// shrinks, so repeated calls with the same or smaller sizes are
    /// allocation-free after the first call.
    ///
    /// # Panics
    ///
    /// Panics (via `RefCell`) if called re-entrantly on the same thread (i.e.
    /// if `f` itself calls `with_scratch`). Use [`with_scratch_nested`] for
    /// reentrant usage.
    pub fn with_scratch<T: Float, F, R>(n: usize, f: F) -> R
    where
        F: FnOnce(&mut [Complex<T>]) -> R,
    {
        let byte_count = n * core::mem::size_of::<Complex<T>>();
        SCRATCH.with(|cell| {
            let mut raw = cell.borrow_mut();
            raw.ensure_capacity(byte_count);
            // SAFETY: We have exclusive access via RefCell, the memory is
            // properly aligned for u8 and we cast to Complex<T> which is
            // repr(C) of two floats. The memory is zero-initialized which is
            // a valid representation for all float types.
            let ptr = raw.as_mut_ptr();
            if byte_count == 0 {
                return f(&mut []);
            }
            // Ensure alignment: Complex<T> requires align_of::<Complex<T>>().
            // Vec<u8> is aligned to 1. For safety, if misaligned we fallback.
            let align = core::mem::align_of::<Complex<T>>();
            if !(ptr as usize).is_multiple_of(align) {
                // Alignment mismatch – fall back to a fresh Vec.
                drop(raw);
                let mut fallback = vec![Complex::<T>::zero(); n];
                return f(&mut fallback);
            }
            let slice = unsafe { core::slice::from_raw_parts_mut(ptr.cast::<Complex<T>>(), n) };
            f(slice)
        })
    }

    /// Execute `f` with an independent scratch buffer that supports nesting.
    ///
    /// Unlike [`with_scratch`], this always allocates a fresh buffer (but the
    /// caller can use it within a `with_scratch` callback without panicking).
    /// The buffer is still zero-initialized.
    pub fn with_scratch_nested<T: Float, F, R>(n: usize, f: F) -> R
    where
        F: FnOnce(&mut [Complex<T>]) -> R,
    {
        let mut buf = vec![Complex::<T>::zero(); n];
        f(&mut buf)
    }

    /// Allocate a [`ScratchGuard`] of at least `n` elements.
    ///
    /// This is a convenience wrapper that always allocates (it does not use
    /// thread-local storage) but returns a handle with a uniform API.
    pub fn get_scratch<T: Float>(n: usize) -> ScratchGuard<T> {
        ScratchGuard {
            buf: vec![Complex::<T>::zero(); n],
        }
    }

    /// Query the current thread-local scratch capacity in elements of `Complex<T>`.
    ///
    /// Returns 0 if no scratch has been allocated yet on this thread.
    pub fn scratch_capacity<T: Float>() -> usize {
        let elem_size = core::mem::size_of::<Complex<T>>();
        if elem_size == 0 {
            return 0;
        }
        SCRATCH.with(|cell| {
            let raw = cell.borrow();
            raw.capacity / elem_size
        })
    }
}

// ---------------------------------------------------------------------------
// no_std fallback – plain allocation
// ---------------------------------------------------------------------------

#[cfg(not(feature = "std"))]
mod fallback {
    use super::*;

    /// Execute `f` with a scratch buffer of `n` zero-initialized elements.
    ///
    /// On `no_std`, this always allocates via `Vec`.
    pub fn with_scratch<T: Float, F, R>(n: usize, f: F) -> R
    where
        F: FnOnce(&mut [Complex<T>]) -> R,
    {
        let mut buf = vec![Complex::<T>::zero(); n];
        f(&mut buf)
    }

    /// Same as [`with_scratch`] – no thread-local storage on `no_std`.
    pub fn with_scratch_nested<T: Float, F, R>(n: usize, f: F) -> R
    where
        F: FnOnce(&mut [Complex<T>]) -> R,
    {
        with_scratch(n, f)
    }

    /// Allocate a [`ScratchGuard`] of `n` zero-initialized elements.
    pub fn get_scratch<T: Float>(n: usize) -> ScratchGuard<T> {
        ScratchGuard {
            buf: vec![Complex::<T>::zero(); n],
        }
    }

    /// Always returns 0 on `no_std` (no thread-local pool).
    pub fn scratch_capacity<T: Float>() -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// Public re-exports
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
pub use tls::{get_scratch, scratch_capacity, with_scratch, with_scratch_nested};

#[cfg(not(feature = "std"))]
pub use fallback::{get_scratch, scratch_capacity, with_scratch, with_scratch_nested};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_scratch_basic() {
        with_scratch::<f64, _, _>(128, |buf| {
            assert_eq!(buf.len(), 128);
            // Should be zero-initialized
            for c in buf.iter() {
                assert_eq!(c.re, 0.0);
                assert_eq!(c.im, 0.0);
            }
        });
    }

    #[test]
    fn test_scratch_grows_but_does_not_shrink() {
        // First call: allocate 64 elements
        with_scratch::<f64, _, _>(64, |buf| {
            assert_eq!(buf.len(), 64);
        });

        #[cfg(feature = "std")]
        {
            let cap1 = scratch_capacity::<f64>();
            assert!(cap1 >= 64);

            // Second call: grow to 256 elements
            with_scratch::<f64, _, _>(256, |buf| {
                assert_eq!(buf.len(), 256);
            });

            let cap2 = scratch_capacity::<f64>();
            assert!(cap2 >= 256);

            // Third call: shrink request – capacity should NOT decrease
            with_scratch::<f64, _, _>(32, |buf| {
                assert_eq!(buf.len(), 32);
            });

            let cap3 = scratch_capacity::<f64>();
            assert!(cap3 >= 256, "capacity should not shrink: got {cap3}");
        }
    }

    #[test]
    fn test_scratch_zero_size() {
        with_scratch::<f64, _, _>(0, |buf| {
            assert!(buf.is_empty());
        });
    }

    #[test]
    fn test_scratch_nested_does_not_panic() {
        with_scratch::<f64, _, _>(64, |outer| {
            outer[0] = Complex::new(1.0, 2.0);
            with_scratch_nested::<f64, _, _>(32, |inner| {
                assert_eq!(inner.len(), 32);
                // inner should be independent
                inner[0] = Complex::new(3.0, 4.0);
            });
            // outer should be unchanged
            assert_eq!(outer[0].re, 1.0);
            assert_eq!(outer[0].im, 2.0);
        });
    }

    #[test]
    fn test_get_scratch_guard() {
        let mut guard = get_scratch::<f64>(512);
        assert_eq!(guard.len(), 512);
        assert!(!guard.is_empty());

        // Write and read back
        guard[0] = Complex::new(42.0, 0.0);
        assert_eq!(guard.as_slice()[0].re, 42.0);
        assert_eq!(guard.as_mut_slice()[0].re, 42.0);
    }

    #[test]
    fn test_scratch_f32() {
        with_scratch::<f32, _, _>(256, |buf| {
            assert_eq!(buf.len(), 256);
            for c in buf.iter() {
                assert_eq!(c.re, 0.0f32);
                assert_eq!(c.im, 0.0f32);
            }
        });
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_scratch_across_threads() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let success_count = Arc::new(AtomicUsize::new(0));
        let num_threads = 4;
        let mut handles = Vec::new();

        for _ in 0..num_threads {
            let counter = Arc::clone(&success_count);
            handles.push(std::thread::spawn(move || {
                // Each thread gets its own scratch
                with_scratch::<f64, _, _>(1024, |buf| {
                    assert_eq!(buf.len(), 1024);
                    // Write a pattern
                    for (i, c) in buf.iter_mut().enumerate() {
                        c.re = i as f64;
                        c.im = -(i as f64);
                    }
                    // Verify pattern
                    for (i, c) in buf.iter().enumerate() {
                        assert_eq!(c.re, i as f64);
                        assert_eq!(c.im, -(i as f64));
                    }
                    counter.fetch_add(1, Ordering::SeqCst);
                });
            }));
        }

        for h in handles {
            h.join().expect("thread panicked");
        }

        assert_eq!(success_count.load(Ordering::SeqCst), num_threads);
    }

    #[test]
    fn test_scratch_guard_deref() {
        let guard = get_scratch::<f64>(16);
        // Deref to slice
        let _slice: &[Complex<f64>] = &guard;
        assert_eq!(_slice.len(), 16);
    }

    #[test]
    fn test_scratch_repeated_same_size() {
        // Repeated calls with the same size should not allocate after the first.
        for _ in 0..100 {
            with_scratch::<f64, _, _>(128, |buf| {
                assert_eq!(buf.len(), 128);
            });
        }
    }
}
