//! Runtime SIMD feature detection.

/// SIMD capability level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimdLevel {
    /// Scalar (no SIMD)
    Scalar,
    /// SSE2 (x86_64)
    Sse2,
    /// AVX (x86_64)
    Avx,
    /// AVX2 (x86_64)
    Avx2,
    /// AVX-512 (x86_64)
    Avx512,
    /// NEON (aarch64)
    Neon,
    /// SVE (aarch64, scalable vectors)
    Sve,
}

/// Detect the highest available SIMD level.
#[must_use]
pub fn detect_simd_level() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return SimdLevel::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdLevel::Avx2;
        }
        if is_x86_feature_detected!("avx") {
            return SimdLevel::Avx;
        }
        if is_x86_feature_detected!("sse2") {
            return SimdLevel::Sse2;
        }
        // No SIMD features detected on x86_64
        SimdLevel::Scalar
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Check for SVE first (if enabled)
        #[cfg(feature = "sve")]
        {
            if has_sve_runtime() {
                return SimdLevel::Sve;
            }
        }
        // NEON is mandatory on aarch64
        SimdLevel::Neon
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    SimdLevel::Scalar
}

/// Check if AVX is available.
#[allow(dead_code)]
#[must_use]
#[cfg(target_arch = "x86_64")]
pub fn has_avx() -> bool {
    is_x86_feature_detected!("avx")
}

/// Check if AVX2 is available.
#[allow(dead_code)]
#[must_use]
#[cfg(target_arch = "x86_64")]
pub fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

/// Check if AVX-512 is available.
#[allow(dead_code)]
#[must_use]
#[cfg(target_arch = "x86_64")]
pub fn has_avx512() -> bool {
    is_x86_feature_detected!("avx512f")
}

/// Check if SVE is available at runtime.
#[allow(dead_code)]
#[must_use]
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "sve"))]
pub fn has_sve_runtime() -> bool {
    // Check HWCAP for SVE support
    // HWCAP_SVE = 1 << 22
    #[allow(unsafe_code)]
    unsafe {
        const AT_HWCAP: libc::c_ulong = 16;
        const HWCAP_SVE: u64 = 1 << 22;
        let hwcap = libc::getauxval(AT_HWCAP);
        (hwcap & HWCAP_SVE) != 0
    }
}

/// SVE not available on non-Linux aarch64 or without feature flag.
#[allow(dead_code)]
#[must_use]
#[cfg(not(all(target_arch = "aarch64", target_os = "linux", feature = "sve")))]
pub fn has_sve_runtime() -> bool {
    false
}
