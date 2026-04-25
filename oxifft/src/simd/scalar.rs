//! Scalar fallback implementation.

use super::traits::SimdVector;
use crate::kernel::Float;

/// Scalar "SIMD" type (1-lane fallback).
#[derive(Copy, Clone, Debug)]
pub struct Scalar<T: Float>(pub T);

impl<T: Float> SimdVector for Scalar<T> {
    type Scalar = T;
    const LANES: usize = 1;

    #[inline]
    fn splat(value: T) -> Self {
        Self(value)
    }

    /// # Safety
    ///
    /// See [`SimdVector::load_aligned`] for the full safety contract.
    #[inline]
    unsafe fn load_aligned(ptr: *const T) -> Self {
        unsafe { Self(*ptr) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::load_unaligned`] for the full safety contract.
    #[inline]
    unsafe fn load_unaligned(ptr: *const T) -> Self {
        unsafe { Self(*ptr) }
    }

    /// # Safety
    ///
    /// See [`SimdVector::store_aligned`] for the full safety contract.
    #[inline]
    unsafe fn store_aligned(self, ptr: *mut T) {
        unsafe { *ptr = self.0 }
    }

    /// # Safety
    ///
    /// See [`SimdVector::store_unaligned`] for the full safety contract.
    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut T) {
        unsafe { *ptr = self.0 }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        Self(self.0 / other.0)
    }
}
