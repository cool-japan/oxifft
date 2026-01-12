//! SIMD operation traits.

use crate::kernel::Float;

/// Core SIMD vector trait.
pub trait SimdVector: Copy + Clone + Send + Sync + Sized {
    /// Scalar type of vector elements.
    type Scalar: Float;

    /// Number of lanes in the vector.
    const LANES: usize;

    /// Create vector with all lanes set to value.
    fn splat(value: Self::Scalar) -> Self;

    /// Load from aligned memory.
    ///
    /// # Safety
    /// Pointer must be aligned to vector size and valid for LANES elements.
    unsafe fn load_aligned(ptr: *const Self::Scalar) -> Self;

    /// Load from unaligned memory.
    ///
    /// # Safety
    /// Pointer must be valid for LANES elements.
    unsafe fn load_unaligned(ptr: *const Self::Scalar) -> Self;

    /// Store to aligned memory.
    ///
    /// # Safety
    /// Pointer must be aligned to vector size and valid for LANES elements.
    unsafe fn store_aligned(self, ptr: *mut Self::Scalar);

    /// Store to unaligned memory.
    ///
    /// # Safety
    /// Pointer must be valid for LANES elements.
    unsafe fn store_unaligned(self, ptr: *mut Self::Scalar);

    /// Vector addition.
    fn add(self, other: Self) -> Self;

    /// Vector subtraction.
    fn sub(self, other: Self) -> Self;

    /// Vector multiplication.
    fn mul(self, other: Self) -> Self;

    /// Vector division.
    fn div(self, other: Self) -> Self;

    /// Fused multiply-add: self * a + b
    fn fmadd(self, a: Self, b: Self) -> Self {
        self.mul(a).add(b)
    }

    /// Fused multiply-subtract: self * a - b
    fn fmsub(self, a: Self, b: Self) -> Self {
        self.mul(a).sub(b)
    }

    /// Negated fused multiply-add: -(self * a) + b
    fn fnmadd(self, a: Self, b: Self) -> Self {
        b.sub(self.mul(a))
    }
}

/// Complex SIMD operations.
pub trait SimdComplex: SimdVector {
    /// Complex multiply.
    fn cmul(self, other: Self) -> Self;

    /// Complex conjugate multiply.
    fn cmul_conj(self, other: Self) -> Self;

    /// Butterfly operation: (a+b, a-b)
    fn butterfly(a: Self, b: Self) -> (Self, Self) {
        (a.add(b), a.sub(b))
    }
}
