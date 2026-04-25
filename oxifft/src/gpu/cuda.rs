//! CUDA backend for GPU FFT using real oxicuda types.
//!
//! Uses oxicuda-driver for device management and oxicuda-fft for plan
//! creation. Actual GPU kernel execution is deferred until oxicuda-launch
//! integration; CPU FFT is used as the computation engine in the meantime.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::string::{String, ToString};

use std::sync::Arc;

use oxicuda_driver::Context;
use oxicuda_fft::{FftHandle, FftPlan, FftType};

use super::buffer::GpuBuffer;
use super::error::{GpuError, GpuResult};
use super::plan::GpuDirection;
use super::GpuBackend;
use super::GpuCapabilities;
use crate::kernel::{Complex, Float};

/// Check if CUDA is available.
///
/// On macOS, NVIDIA dropped support so `init()` will always fail.
/// On Linux/Windows without an NVIDIA GPU, `Device::get(0)` will fail.
#[must_use]
pub fn is_available() -> bool {
    oxicuda_driver::init().is_ok() && oxicuda_driver::Device::get(0).is_ok()
}

/// Query CUDA device capabilities using the real driver API.
///
/// # Errors
///
/// Returns `GpuError::NoBackendAvailable` if no CUDA device is present, or
/// `GpuError::InitializationFailed` if the driver fails to return device info.
pub fn query_capabilities() -> GpuResult<GpuCapabilities> {
    if !is_available() {
        return Err(GpuError::NoBackendAvailable);
    }
    let device = oxicuda_driver::Device::get(0)
        .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
    let name = device
        .name()
        .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
    let total_memory = device
        .total_memory()
        .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
    Ok(GpuCapabilities {
        backend: GpuBackend::Cuda,
        device_name: name,
        total_memory: total_memory as u64,
        available_memory: 0,
        max_fft_size: 1 << 27,
        supports_f64: true,
        supports_f16: true,
        compute_units: 0,
        max_workgroup_size: 1024,
    })
}

/// Synchronize CUDA device (no-op until GPU stream sync is active).
///
/// # Errors
///
/// This function currently cannot return an error; the `Result` signature is
/// retained for API symmetry with backends that perform real GPU stream
/// synchronisation (once `oxicuda-launch` integration lands).
pub fn synchronize() -> GpuResult<()> {
    // GPU stream sync will be needed when the GPU execution path is active.
    Ok(())
}

/// CUDA FFT plan wrapper holding real oxicuda resources.
///
/// Note: `CudaFftPlan` is NOT `Clone` because `FftHandle` contains a
/// non-cloneable `Stream`.
pub struct CudaFftPlan {
    /// Transform size.
    size: usize,
    /// Batch size.
    batch_size: usize,
    /// CUDA context (held for RAII — keeps the Context alive for the FFT handle lifetime).
    _context: Arc<Context>,
    /// oxicuda-fft executor handle (owns a CUDA Stream).
    fft_handle: FftHandle,
    /// oxicuda-fft plan (size, type, batch).
    fft_plan: FftPlan,
}

impl std::fmt::Debug for CudaFftPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaFftPlan")
            .field("size", &self.size)
            .field("batch_size", &self.batch_size)
            .field("fft_handle", &self.fft_handle)
            .field("fft_plan", &self.fft_plan)
            .finish_non_exhaustive()
    }
}

impl CudaFftPlan {
    /// Create a new CUDA FFT plan, initialising real oxicuda resources.
    ///
    /// `FftPlan::new_1d` accepts any non-zero size (not only powers of two),
    /// so arbitrary factorisation is supported.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::NoBackendAvailable` if no CUDA device is present,
    /// `GpuError::InvalidSize` if `size` is zero, or
    /// `GpuError::InitializationFailed` if driver/context/plan allocation fails.
    pub fn new(size: usize, batch_size: usize) -> GpuResult<Self> {
        if !is_available() {
            return Err(GpuError::NoBackendAvailable);
        }
        if size == 0 {
            return Err(GpuError::InvalidSize(size));
        }

        oxicuda_driver::init().map_err(|e| GpuError::InitializationFailed(e.to_string()))?;

        let device = oxicuda_driver::Device::get(0)
            .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;

        let raw_ctx =
            Context::new(&device).map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
        let context = Arc::new(raw_ctx);

        let fft_handle =
            FftHandle::new(&context).map_err(|e| GpuError::InitializationFailed(e.to_string()))?;

        let fft_plan = FftPlan::new_1d(size, FftType::C2C, batch_size)
            .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;

        Ok(Self {
            size,
            batch_size,
            _context: context,
            fft_handle,
            fft_plan,
        })
    }

    /// Execute the FFT.
    ///
    /// GPU kernel execution is pending oxicuda-launch integration.
    /// CPU FFT is used as the computation engine until then.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::SizeMismatch` if buffer sizes do not match the plan,
    /// or `GpuError::ExecutionFailed` if the CPU fallback plan cannot be created.
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
        // GPU kernel execution is pending oxicuda-launch integration.
        // Use CPU FFT computation until GPU kernels are compiled.
        self.execute_cpu(input, output, direction)
    }

    fn execute_cpu<T: Float>(
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

    /// Execute a real-to-complex forward FFT.
    ///
    /// // KNOWN LIMITATION: CUDA R2C/C2R runs on CPU until oxicuda-launch kernel dispatch is integrated
    ///
    /// Wraps the existing CPU-fallback path.  An `eprintln!` warning is
    /// emitted to notify callers of the CPU path.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::SizeMismatch` if `input` or `output` lengths are
    /// inconsistent, or `GpuError::ExecutionFailed` if the CPU fallback plan
    /// cannot be created.
    pub fn forward_r2c(
        &self,
        input: &[f32],
        output: &mut [num_complex::Complex<f32>],
    ) -> GpuResult<()> {
        let n = self.size;
        let half = n / 2 + 1;

        if input.len() != n {
            return Err(GpuError::SizeMismatch {
                expected: n,
                got: input.len(),
            });
        }
        if output.len() != half {
            return Err(GpuError::SizeMismatch {
                expected: half,
                got: output.len(),
            });
        }

        eprintln!(
            "CUDA R2C/C2R: using CPU fallback (known limitation \
             — GPU path pending oxicuda-launch kernel dispatch integration)"
        );

        use crate::api::{Direction, Flags, Plan};

        // Zero-extend real input → complex.
        let input_f64: Vec<Complex<f64>> = input
            .iter()
            .map(|&x| Complex::new(x as f64, 0.0_f64))
            .collect();
        let mut output_f64 = vec![Complex::<f64>::zero(); n];

        let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
            GpuError::ExecutionFailed("Failed to create CPU fallback plan for R2C".into())
        })?;
        plan.execute(&input_f64, &mut output_f64);

        for (i, c) in output_f64[..half].iter().enumerate() {
            output[i] = num_complex::Complex::new(c.re as f32, c.im as f32);
        }
        Ok(())
    }

    /// Execute a complex-to-real inverse FFT.
    ///
    /// // KNOWN LIMITATION: CUDA R2C/C2R runs on CPU until oxicuda-launch kernel dispatch is integrated
    ///
    /// Wraps the existing CPU-fallback path.  An `eprintln!` warning is
    /// emitted to notify callers of the CPU path.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::SizeMismatch` if `input` or `output` lengths are
    /// inconsistent, or `GpuError::ExecutionFailed` if the CPU fallback plan
    /// cannot be created.
    pub fn inverse_c2r(
        &self,
        input: &[num_complex::Complex<f32>],
        output: &mut [f32],
    ) -> GpuResult<()> {
        let n = self.size;
        let half = n / 2 + 1;

        if input.len() != half {
            return Err(GpuError::SizeMismatch {
                expected: half,
                got: input.len(),
            });
        }
        if output.len() != n {
            return Err(GpuError::SizeMismatch {
                expected: n,
                got: output.len(),
            });
        }

        eprintln!(
            "CUDA R2C/C2R: using CPU fallback (known limitation \
             — GPU path pending oxicuda-launch kernel dispatch integration)"
        );

        use crate::api::{Direction, Flags, Plan};

        // Reconstruct conjugate-symmetric full spectrum.
        let mut full_f64 = vec![Complex::<f64>::zero(); n];
        for (k, c) in input.iter().enumerate() {
            full_f64[k] = Complex::new(c.re as f64, c.im as f64);
        }
        for k in 1..n / 2 {
            full_f64[n - k] = full_f64[k].conj();
        }

        let mut time_f64 = vec![Complex::<f64>::zero(); n];
        let plan = Plan::dft_1d(n, Direction::Backward, Flags::ESTIMATE).ok_or_else(|| {
            GpuError::ExecutionFailed("Failed to create CPU fallback plan for C2R".into())
        })?;
        plan.execute(&full_f64, &mut time_f64);

        // Normalise by 1/n (IFFT convention) and take real parts.
        let norm = 1.0_f64 / n as f64;
        for (i, c) in time_f64.iter().enumerate() {
            output[i] = (c.re * norm) as f32;
        }
        Ok(())
    }

    /// Return the transform size this plan was created for.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Return the batch size this plan was created for.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

impl Drop for CudaFftPlan {
    fn drop(&mut self) {
        // oxicuda types handle RAII automatically.
    }
}

/// Upload buffer to CUDA device (no-op; GPU memory managed in execute).
///
/// # Errors
///
/// This function currently cannot return an error; the `Result` signature is
/// retained for API symmetry with backends that perform real device transfers.
pub fn upload_buffer<T: Float>(_buffer: &mut GpuBuffer<T>) -> GpuResult<()> {
    Ok(())
}

/// Download buffer from CUDA device (no-op; GPU memory managed in execute).
///
/// # Errors
///
/// This function currently cannot return an error; the `Result` signature is
/// retained for API symmetry with backends that perform real device transfers.
pub fn download_buffer<T: Float>(_buffer: &mut GpuBuffer<T>) -> GpuResult<()> {
    Ok(())
}

/// Free CUDA buffer (no-op; GPU memory managed in execute).
///
/// # Errors
///
/// This function currently cannot return an error; the `Result` signature is
/// retained for API symmetry with backends that perform real GPU memory
/// deallocation.
pub fn free_buffer(_ptr: *mut core::ffi::c_void) -> GpuResult<()> {
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
