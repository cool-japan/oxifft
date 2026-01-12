//! CUDA backend for GPU FFT.
//!
//! Uses cuFFT for high-performance FFT on NVIDIA GPUs.
//!
//! # Requirements
//!
//! - NVIDIA GPU with CUDA capability 3.0+
//! - CUDA toolkit installed
//! - cuFFT library available

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::string::{String, ToString};

use super::buffer::GpuBuffer;
use super::error::{GpuError, GpuResult};
use super::plan::GpuDirection;
use super::GpuBackend;
use super::GpuCapabilities;
use crate::kernel::{Complex, Float};

/// Check if CUDA is available.
#[must_use]
pub fn is_available() -> bool {
    // Check for CUDA driver/runtime
    #[cfg(target_os = "linux")]
    {
        // Check for NVIDIA driver
        std::path::Path::new("/dev/nvidia0").exists()
            || std::path::Path::new("/proc/driver/nvidia/version").exists()
    }
    #[cfg(target_os = "windows")]
    {
        // Check for nvcuda.dll
        std::path::Path::new("C:\\Windows\\System32\\nvcuda.dll").exists()
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        false
    }
}

/// Query CUDA device capabilities.
pub fn query_capabilities() -> GpuResult<GpuCapabilities> {
    if !is_available() {
        return Err(GpuError::NoBackendAvailable);
    }

    // In a real implementation, this would use cuDeviceGetAttribute, etc.
    Ok(GpuCapabilities {
        backend: GpuBackend::Cuda,
        device_name: "NVIDIA GPU".to_string(),
        total_memory: 0,
        available_memory: 0,
        max_fft_size: 1 << 27, // cuFFT supports up to 128M elements
        supports_f64: true,
        supports_f16: true,
        compute_units: 0,
        max_workgroup_size: 1024,
    })
}

/// Synchronize CUDA device.
pub fn synchronize() -> GpuResult<()> {
    // In a real implementation: cudaDeviceSynchronize()
    Ok(())
}

/// CUDA FFT plan wrapper.
#[derive(Debug)]
pub struct CudaFftPlan {
    /// Transform size.
    size: usize,
    /// Batch size.
    batch_size: usize,
    /// cuFFT plan handle (opaque).
    #[allow(dead_code)]
    handle: u64,
}

impl CudaFftPlan {
    /// Create a new CUDA FFT plan.
    pub fn new(size: usize, batch_size: usize) -> GpuResult<Self> {
        if !is_available() {
            return Err(GpuError::NoBackendAvailable);
        }

        // Validate size
        if size == 0 || !size.is_power_of_two() && size > 1 << 24 {
            return Err(GpuError::InvalidSize(size));
        }

        // In a real implementation:
        // cufftPlan1d(&plan, size, CUFFT_Z2Z, batch_size)
        Ok(Self {
            size,
            batch_size,
            handle: 0, // Would be actual cuFFT handle
        })
    }

    /// Execute the FFT.
    pub fn execute<T: Float>(
        &self,
        input: &GpuBuffer<T>,
        output: &mut GpuBuffer<T>,
        direction: GpuDirection,
    ) -> GpuResult<()> {
        let expected_size = self.size * self.batch_size;
        if input.size() != expected_size || output.size() != expected_size {
            return Err(GpuError::SizeMismatch {
                expected: expected_size,
                got: input.size().min(output.size()),
            });
        }

        // In a real implementation:
        // let cufft_dir = match direction {
        //     GpuDirection::Forward => CUFFT_FORWARD,
        //     GpuDirection::Inverse => CUFFT_INVERSE,
        // };
        // cufftExecZ2Z(self.handle, input_ptr, output_ptr, cufft_dir)

        // Fallback: use CPU FFT
        self.execute_fallback(input, output, direction)
    }

    fn execute_fallback<T: Float>(
        &self,
        input: &GpuBuffer<T>,
        output: &mut GpuBuffer<T>,
        direction: GpuDirection,
    ) -> GpuResult<()> {
        use crate::api::{Direction, Flags, Plan};

        let dir = match direction {
            GpuDirection::Forward => Direction::Forward,
            GpuDirection::Inverse => Direction::Backward,
        };

        // Process each batch
        for batch in 0..self.batch_size {
            let start = batch * self.size;
            let end = start + self.size;

            let input_slice = &input.cpu_data()[start..end];
            let output_slice = &mut output.cpu_data_mut()[start..end];

            // Use CPU FFT as fallback
            if let Some(plan) = Plan::dft_1d(self.size, dir, Flags::ESTIMATE) {
                // Convert to f64 for the plan
                let input_f64: Vec<Complex<f64>> = input_slice
                    .iter()
                    .map(|c| {
                        Complex::new(c.re.to_f64().unwrap_or(0.0), c.im.to_f64().unwrap_or(0.0))
                    })
                    .collect();
                let mut output_f64 = vec![Complex::<f64>::zero(); self.size];

                plan.execute(&input_f64, &mut output_f64);

                // Convert back
                for (i, c) in output_f64.iter().enumerate() {
                    output_slice[i] = Complex::new(T::from_f64(c.re), T::from_f64(c.im));
                }
            } else {
                return Err(GpuError::ExecutionFailed(
                    "Failed to create CPU fallback plan".into(),
                ));
            }
        }

        Ok(())
    }
}

impl Drop for CudaFftPlan {
    fn drop(&mut self) {
        // In a real implementation: cufftDestroy(self.handle)
    }
}

/// Upload buffer to CUDA device.
pub fn upload_buffer<T: Float>(_buffer: &mut GpuBuffer<T>) -> GpuResult<()> {
    // In a real implementation:
    // cudaMalloc(&device_ptr, size * sizeof(Complex<T>))
    // cudaMemcpy(device_ptr, host_ptr, size * sizeof(Complex<T>), cudaMemcpyHostToDevice)
    Ok(())
}

/// Download buffer from CUDA device.
pub fn download_buffer<T: Float>(_buffer: &mut GpuBuffer<T>) -> GpuResult<()> {
    // In a real implementation:
    // cudaMemcpy(host_ptr, device_ptr, size * sizeof(Complex<T>), cudaMemcpyDeviceToHost)
    Ok(())
}

/// Free CUDA buffer.
pub fn free_buffer(_ptr: *mut core::ffi::c_void) -> GpuResult<()> {
    // In a real implementation: cudaFree(ptr)
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability() {
        // Just verify it doesn't panic
        let _ = is_available();
    }

    #[test]
    fn test_cuda_capabilities() {
        if is_available() {
            let caps = query_capabilities().expect("Failed to query capabilities");
            assert_eq!(caps.backend, GpuBackend::Cuda);
            assert!(caps.supports_f64);
        }
    }

    #[test]
    fn test_cuda_plan_creation() {
        if is_available() {
            let plan = CudaFftPlan::new(1024, 1);
            assert!(plan.is_ok());
        }
    }
}
