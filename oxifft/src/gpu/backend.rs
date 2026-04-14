//! GPU backend abstraction.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::string::String;

/// GPU backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GpuBackend {
    /// Automatically select best available backend.
    Auto,
    /// NVIDIA CUDA backend (uses cuFFT).
    Cuda,
    /// Apple Metal backend.
    Metal,
    /// OpenCL backend (future).
    #[allow(dead_code)]
    OpenCL,
    /// Vulkan compute backend (future).
    #[allow(dead_code)]
    Vulkan,
}

impl GpuBackend {
    /// Check if this backend is available on the current system.
    #[must_use]
    pub fn is_available(self) -> bool {
        match self {
            Self::Auto => {
                #[cfg(feature = "cuda")]
                if super::cuda::is_available() {
                    return true;
                }
                #[cfg(feature = "metal")]
                if super::metal::is_available() {
                    return true;
                }
                false
            }
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    super::cuda::is_available()
                }
                #[cfg(not(feature = "cuda"))]
                {
                    false
                }
            }
            Self::Metal => {
                #[cfg(feature = "metal")]
                {
                    super::metal::is_available()
                }
                #[cfg(not(feature = "metal"))]
                {
                    false
                }
            }
            Self::OpenCL | Self::Vulkan => false, // Not yet implemented
        }
    }

    /// Get the human-readable name of this backend.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Cuda => "CUDA",
            Self::Metal => "Metal",
            Self::OpenCL => "OpenCL",
            Self::Vulkan => "Vulkan",
        }
    }
}

/// GPU device capabilities.
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// Backend type.
    pub backend: GpuBackend,
    /// Device name.
    pub device_name: String,
    /// Total device memory in bytes.
    pub total_memory: u64,
    /// Available device memory in bytes.
    pub available_memory: u64,
    /// Maximum supported FFT size.
    pub max_fft_size: usize,
    /// Whether f64 (double precision) is supported.
    pub supports_f64: bool,
    /// Whether f16 (half precision) is supported.
    pub supports_f16: bool,
    /// Number of compute units.
    pub compute_units: u32,
    /// Maximum work group size.
    pub max_workgroup_size: u32,
}

impl Default for GpuCapabilities {
    fn default() -> Self {
        Self {
            backend: GpuBackend::Auto,
            device_name: String::new(),
            total_memory: 0,
            available_memory: 0,
            max_fft_size: 0,
            supports_f64: false,
            supports_f16: false,
            compute_units: 0,
            max_workgroup_size: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_name() {
        assert_eq!(GpuBackend::Cuda.name(), "CUDA");
        assert_eq!(GpuBackend::Metal.name(), "Metal");
        assert_eq!(GpuBackend::Auto.name(), "Auto");
    }

    #[test]
    fn test_backend_availability() {
        // These tests just verify the functions don't panic
        let _ = GpuBackend::Cuda.is_available();
        let _ = GpuBackend::Metal.is_available();
        let _ = GpuBackend::Auto.is_available();
    }
}
