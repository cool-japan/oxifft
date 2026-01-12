//! Memory allocation utilities for FFT buffers.
//!
//! Provides aligned memory allocation for optimal SIMD performance.

use crate::kernel::{Complex, Float};
use crate::prelude::*;
use core::alloc::Layout;

/// Default alignment for FFT buffers (64 bytes for AVX-512).
pub const DEFAULT_ALIGNMENT: usize = 64;

/// An aligned buffer that guarantees proper alignment for SIMD operations.
///
/// This is a wrapper around a raw allocation that ensures the data is aligned
/// to `DEFAULT_ALIGNMENT` bytes (64 bytes for AVX-512 compatibility).
pub struct AlignedBuffer<T> {
    ptr: *mut T,
    len: usize,
    capacity: usize,
}

impl<T: Clone + Default> AlignedBuffer<T> {
    /// Create a new aligned buffer with the given size, initialized to default values.
    ///
    /// # Panics
    /// Panics if memory allocation fails.
    #[must_use]
    pub fn new(size: usize) -> Self {
        if size == 0 {
            return Self {
                ptr: core::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                capacity: 0,
            };
        }

        let layout = Layout::from_size_align(
            size * core::mem::size_of::<T>(),
            DEFAULT_ALIGNMENT.max(core::mem::align_of::<T>()),
        )
        .expect("Invalid layout");

        // SAFETY: layout is non-zero size
        #[cfg(feature = "std")]
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut T };
        #[cfg(not(feature = "std"))]
        let ptr = unsafe { alloc::alloc::alloc_zeroed(layout) as *mut T };

        if ptr.is_null() {
            #[cfg(feature = "std")]
            std::alloc::handle_alloc_error(layout);
            #[cfg(not(feature = "std"))]
            alloc::alloc::handle_alloc_error(layout);
        }

        // Initialize with default values
        for i in 0..size {
            // SAFETY: ptr is valid for size elements, and we're within bounds
            unsafe {
                core::ptr::write(ptr.add(i), T::default());
            }
        }

        Self {
            ptr,
            len: size,
            capacity: size,
        }
    }

    /// Get the length of the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a raw pointer to the data.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Get a mutable raw pointer to the data.
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Get a slice view of the buffer.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: ptr is valid for len elements
            unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    /// Get a mutable slice view of the buffer.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: ptr is valid for len elements
            unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }
}

impl<T> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            let layout = Layout::from_size_align(
                self.capacity * core::mem::size_of::<T>(),
                DEFAULT_ALIGNMENT.max(core::mem::align_of::<T>()),
            )
            .expect("Invalid layout");

            // SAFETY: ptr was allocated with this layout
            unsafe {
                // Drop all elements first
                for i in 0..self.len {
                    core::ptr::drop_in_place(self.ptr.add(i));
                }
                #[cfg(feature = "std")]
                std::alloc::dealloc(self.ptr as *mut u8, layout);
                #[cfg(not(feature = "std"))]
                alloc::alloc::dealloc(self.ptr as *mut u8, layout);
            }
        }
    }
}

// SAFETY: AlignedBuffer is Send if T is Send
unsafe impl<T: Send> Send for AlignedBuffer<T> {}

// SAFETY: AlignedBuffer is Sync if T is Sync
unsafe impl<T: Sync> Sync for AlignedBuffer<T> {}

impl<T> core::ops::Deref for AlignedBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: ptr is valid for len elements
            unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

impl<T> core::ops::DerefMut for AlignedBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: ptr is valid for len elements
            unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }
}

/// Allocate an aligned buffer for complex values.
///
/// The returned vector is guaranteed to have its data pointer
/// aligned to `DEFAULT_ALIGNMENT` bytes (64 bytes for AVX-512).
///
/// Note: This returns a standard Vec which may not be aligned.
/// For guaranteed alignment, use `AlignedBuffer::new()`.
pub fn alloc_complex<T: Float>(size: usize) -> Vec<Complex<T>> {
    vec![Complex::zero(); size]
}

/// Allocate an aligned buffer for complex values with guaranteed alignment.
///
/// The returned buffer is guaranteed to have its data pointer
/// aligned to `DEFAULT_ALIGNMENT` bytes (64 bytes for AVX-512).
pub fn alloc_complex_aligned<T: Float>(size: usize) -> AlignedBuffer<Complex<T>> {
    AlignedBuffer::new(size)
}

/// Allocate an aligned buffer for real values.
///
/// Note: This returns a standard Vec which may not be aligned.
/// For guaranteed alignment, use `AlignedBuffer::new()`.
pub fn alloc_real<T: Float>(size: usize) -> Vec<T> {
    vec![T::ZERO; size]
}

/// Allocate an aligned buffer for real values with guaranteed alignment.
///
/// The returned buffer is guaranteed to have its data pointer
/// aligned to `DEFAULT_ALIGNMENT` bytes (64 bytes for AVX-512).
pub fn alloc_real_aligned<T: Float>(size: usize) -> AlignedBuffer<T> {
    AlignedBuffer::new(size)
}

/// Free an aligned buffer (for FFI compatibility).
///
/// # Safety
/// The pointer must have been allocated by `alloc_complex` or `alloc_real`.
pub unsafe fn free<T>(_ptr: *mut T) {
    // Standard Rust Vec handles deallocation automatically
    // This is mainly for FFI compatibility
}

/// Check if a pointer is properly aligned for SIMD operations.
pub fn is_aligned<T>(ptr: *const T) -> bool {
    (ptr as usize) % DEFAULT_ALIGNMENT == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_buffer_alignment() {
        let buf: AlignedBuffer<f64> = AlignedBuffer::new(64);
        assert!(is_aligned(buf.as_ptr()), "Buffer should be aligned");
        assert_eq!(buf.len(), 64);
    }

    #[test]
    fn test_aligned_buffer_complex() {
        let buf = alloc_complex_aligned::<f64>(32);
        assert!(is_aligned(buf.as_ptr()), "Complex buffer should be aligned");
        assert_eq!(buf.len(), 32);
    }

    #[test]
    fn test_aligned_buffer_real() {
        let buf = alloc_real_aligned::<f64>(32);
        assert!(is_aligned(buf.as_ptr()), "Real buffer should be aligned");
        assert_eq!(buf.len(), 32);
    }

    #[test]
    fn test_aligned_buffer_empty() {
        let buf: AlignedBuffer<f64> = AlignedBuffer::new(0);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_aligned_buffer_access() {
        let mut buf: AlignedBuffer<f64> = AlignedBuffer::new(4);
        buf[0] = 1.0;
        buf[1] = 2.0;
        buf[2] = 3.0;
        buf[3] = 4.0;

        assert_eq!(buf[0], 1.0);
        assert_eq!(buf[1], 2.0);
        assert_eq!(buf[2], 3.0);
        assert_eq!(buf[3], 4.0);
    }

    #[test]
    fn test_aligned_buffer_slice() {
        let buf: AlignedBuffer<f64> = AlignedBuffer::new(4);
        let slice = buf.as_slice();
        assert_eq!(slice.len(), 4);
    }
}
