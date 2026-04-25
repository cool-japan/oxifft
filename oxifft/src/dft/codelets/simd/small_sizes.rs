//! SIMD dispatch functions for small DFT sizes (2, 4, 8, 16, 32).

use crate::kernel::Complex;
#[cfg(target_arch = "x86_64")]
use crate::simd::{detect_simd_level, SimdLevel};

/// Size-2 DFT with SIMD acceleration for f64.
#[inline]
pub fn notw_2_simd_f64(x: &mut [Complex<f64>]) {
    #[cfg(target_arch = "x86_64")]
    {
        let level = detect_simd_level();
        if matches!(
            level,
            SimdLevel::Sse2 | SimdLevel::Avx | SimdLevel::Avx2 | SimdLevel::Avx512
        ) {
            // Safety: We checked SSE2 is available
            unsafe {
                super::backends::sse2_f64::notw_2_sse2(x);
            }
            return;
        }
    }

    // NEON is always available on aarch64
    #[cfg(target_arch = "aarch64")]
    {
        // Safety: NEON is baseline on aarch64
        unsafe {
            super::backends::neon_f64::notw_2_neon(x);
        }
    }

    // Fallback to scalar
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    super::super::notw_2(x);
}

/// Size-4 DFT with SIMD acceleration for f64.
#[inline]
pub fn notw_4_simd_f64(x: &mut [Complex<f64>], sign: i32) {
    #[cfg(target_arch = "x86_64")]
    {
        let level = detect_simd_level();
        if matches!(level, SimdLevel::Avx2 | SimdLevel::Avx512) {
            // Safety: We checked AVX2 is available
            unsafe {
                super::backends::avx2_f64::notw_4_avx2(x, sign);
            }
            return;
        }
        if matches!(level, SimdLevel::Sse2 | SimdLevel::Avx) {
            // Safety: We checked SSE2 is available
            unsafe {
                super::backends::sse2_f64::notw_4_sse2(x, sign);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Safety: NEON is baseline on aarch64
        unsafe {
            super::backends::neon_f64::notw_4_neon(x, sign);
        }
    }

    // Fallback to scalar
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    super::super::notw_4(x, sign);
}

/// Size-8 DFT with SIMD acceleration for f64.
#[inline]
pub fn notw_8_simd_f64(x: &mut [Complex<f64>], sign: i32) {
    #[cfg(target_arch = "x86_64")]
    {
        let level = detect_simd_level();
        if matches!(
            level,
            SimdLevel::Sse2 | SimdLevel::Avx | SimdLevel::Avx2 | SimdLevel::Avx512
        ) {
            // Safety: We checked SSE2 is available
            unsafe {
                super::backends::sse2_f64::notw_8_sse2(x, sign);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Safety: NEON is baseline on aarch64
        unsafe {
            super::backends::neon_f64::notw_8_neon(x, sign);
        }
    }

    // Fallback to scalar
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    super::super::notw_8(x, sign);
}

/// Size-16 DFT with SIMD acceleration for f64.
///
/// Currently uses the scalar optimized codelet which has hardcoded twiddle factors.
/// Future optimization: implement full SIMD version.
#[inline]
pub fn notw_16_simd_f64(x: &mut [Complex<f64>], sign: i32) {
    // The scalar notw_16 is already highly optimized with hardcoded twiddles
    super::super::notw_16(x, sign);
}

/// Size-32 DFT with SIMD acceleration for f64.
///
/// Currently uses the scalar optimized codelet which has hardcoded twiddle factors.
/// Future optimization: implement full SIMD version.
#[inline]
pub fn notw_32_simd_f64(x: &mut [Complex<f64>], sign: i32) {
    // The scalar notw_32 is already highly optimized with hardcoded twiddles
    super::super::notw_32(x, sign);
}
