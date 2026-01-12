//! Memory alignment utilities.

/// Default alignment for SIMD operations (64 bytes for AVX-512).
pub const SIMD_ALIGNMENT: usize = 64;

/// Check if a pointer is properly aligned.
#[inline]
#[must_use]
pub fn is_aligned<T>(ptr: *const T, alignment: usize) -> bool {
    (ptr as usize) % alignment == 0
}

/// Check if a pointer is SIMD-aligned.
#[inline]
#[must_use]
pub fn is_simd_aligned<T>(ptr: *const T) -> bool {
    is_aligned(ptr, SIMD_ALIGNMENT)
}

/// Round up to the next multiple of alignment.
#[inline]
#[must_use]
pub const fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

/// Round down to the previous multiple of alignment.
#[inline]
#[must_use]
pub const fn align_down(value: usize, alignment: usize) -> usize {
    value & !(alignment - 1)
}
