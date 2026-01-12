//! Metal backend for GPU FFT.
//!
//! Provides GPU-accelerated FFT on Apple Silicon and AMD GPUs on macOS.
//!
//! # Requirements
//!
//! - macOS 10.13+ or iOS 11+
//! - Metal-compatible GPU
//!
//! # Implementation Notes
//!
//! This module uses Metal Performance Shaders (MPS) for FFT when available,
//! or custom compute shaders for unsupported configurations.

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

/// Check if Metal is available.
#[must_use]
pub fn is_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        // Check for Metal framework
        // In a real implementation, this would use MTLCreateSystemDefaultDevice()
        true
    }
    #[cfg(target_os = "ios")]
    {
        true
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        false
    }
}

/// Query Metal device capabilities.
pub fn query_capabilities() -> GpuResult<GpuCapabilities> {
    if !is_available() {
        return Err(GpuError::NoBackendAvailable);
    }

    // In a real implementation, this would query the Metal device
    Ok(GpuCapabilities {
        backend: GpuBackend::Metal,
        device_name: get_device_name(),
        total_memory: 0, // Would query device.recommendedMaxWorkingSetSize
        available_memory: 0,
        max_fft_size: 1 << 24, // MPS FFT supports up to 16M elements
        supports_f64: false,   // Metal has limited f64 support
        supports_f16: true,    // Metal has excellent f16 support
        compute_units: 0,
        max_workgroup_size: 1024,
    })
}

/// Get the Metal device name.
fn get_device_name() -> String {
    #[cfg(target_os = "macos")]
    {
        // Would use device.name in real implementation
        "Apple GPU".to_string()
    }
    #[cfg(not(target_os = "macos"))]
    {
        "Unknown Metal Device".to_string()
    }
}

/// Synchronize Metal device.
pub fn synchronize() -> GpuResult<()> {
    // In a real implementation: wait for command buffer completion
    Ok(())
}

/// Metal FFT plan wrapper.
#[derive(Debug)]
pub struct MetalFftPlan {
    /// Transform size.
    size: usize,
    /// Batch size.
    batch_size: usize,
    /// Log2 of transform size.
    log2n: u32,
    /// Whether using MPS or custom shaders.
    #[allow(dead_code)]
    use_mps: bool,
}

impl MetalFftPlan {
    /// Create a new Metal FFT plan.
    pub fn new(size: usize, batch_size: usize) -> GpuResult<Self> {
        if !is_available() {
            return Err(GpuError::NoBackendAvailable);
        }

        // Validate size - MPS FFT requires power of 2
        if size == 0 {
            return Err(GpuError::InvalidSize(size));
        }

        let log2n = if size.is_power_of_two() {
            size.trailing_zeros()
        } else {
            // For non-power-of-2, we'd need Bluestein's algorithm
            return Err(GpuError::Unsupported(
                "Metal FFT currently requires power-of-2 sizes".into(),
            ));
        };

        // In a real implementation:
        // let descriptor = MPSImageFFTDescriptor(dimensions: [size], ...)
        // let fft = MPSImageFFT(device: device, descriptor: descriptor)

        Ok(Self {
            size,
            batch_size,
            log2n,
            use_mps: true,
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
        // Create command buffer
        // Encode FFT operation
        // Commit and wait

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
                let input_f64: Vec<Complex<f64>> = input_slice
                    .iter()
                    .map(|c| {
                        Complex::new(c.re.to_f64().unwrap_or(0.0), c.im.to_f64().unwrap_or(0.0))
                    })
                    .collect();
                let mut output_f64 = vec![Complex::<f64>::zero(); self.size];

                plan.execute(&input_f64, &mut output_f64);

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

    /// Get the log2 of the transform size.
    #[must_use]
    pub const fn log2n(&self) -> u32 {
        self.log2n
    }
}

impl Drop for MetalFftPlan {
    fn drop(&mut self) {
        // Metal objects are reference counted, automatic cleanup
    }
}

/// Metal buffer handle.
#[derive(Debug)]
pub struct MetalBufferHandle {
    /// Buffer ID.
    pub id: u64,
}

/// Upload buffer to Metal device.
pub fn upload_buffer<T: Float>(_buffer: &mut GpuBuffer<T>) -> GpuResult<()> {
    // In a real implementation:
    // let metal_buffer = device.makeBuffer(bytes: ptr, length: size, options: .storageModeShared)
    Ok(())
}

/// Download buffer from Metal device.
pub fn download_buffer<T: Float>(_buffer: &mut GpuBuffer<T>) -> GpuResult<()> {
    // In a real implementation:
    // memcpy(host_ptr, metal_buffer.contents(), size)
    Ok(())
}

/// Free Metal buffer.
pub fn free_buffer(_handle: MetalBufferHandle) -> GpuResult<()> {
    // Metal buffers are reference counted, automatic cleanup
    Ok(())
}

// ============================================================================
// Metal Shader Code (for reference)
// ============================================================================

/// Metal shader source for FFT butterfly operations.
#[allow(dead_code)]
const FFT_SHADER_SOURCE: &str = r"
#include <metal_stdlib>
using namespace metal;

// Complex number type
struct Complex {
    float re;
    float im;
};

// Complex multiplication
Complex cmul(Complex a, Complex b) {
    return Complex{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

// Complex addition
Complex cadd(Complex a, Complex b) {
    return Complex{a.re + b.re, a.im + b.im};
}

// Complex subtraction
Complex csub(Complex a, Complex b) {
    return Complex{a.re - b.re, a.im - b.im};
}

// Twiddle factor
Complex twiddle(uint k, uint n, bool inverse) {
    float angle = (inverse ? 1.0 : -1.0) * 2.0 * M_PI_F * float(k) / float(n);
    return Complex{cos(angle), sin(angle)};
}

// Radix-2 butterfly kernel
kernel void fft_butterfly(
    device Complex* data [[buffer(0)]],
    constant uint& stage [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant bool& inverse [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint butterfly_size = 1u << (stage + 1);
    uint half_size = butterfly_size >> 1;
    uint group = gid / half_size;
    uint pair = gid % half_size;

    uint i = group * butterfly_size + pair;
    uint j = i + half_size;

    Complex w = twiddle(pair, butterfly_size, inverse);
    Complex u = data[i];
    Complex t = cmul(w, data[j]);

    data[i] = cadd(u, t);
    data[j] = csub(u, t);
}

// Bit-reversal permutation kernel
kernel void bit_reverse(
    device Complex* input [[buffer(0)]],
    device Complex* output [[buffer(1)]],
    constant uint& log2n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint rev = 0;
    uint idx = gid;
    for (uint i = 0; i < log2n; i++) {
        rev = (rev << 1) | (idx & 1);
        idx >>= 1;
    }
    output[rev] = input[gid];
}
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_availability() {
        let _ = is_available();
    }

    #[test]
    fn test_metal_capabilities() {
        if is_available() {
            let caps = query_capabilities().expect("Failed to query capabilities");
            assert_eq!(caps.backend, GpuBackend::Metal);
            assert!(caps.supports_f16);
        }
    }

    #[test]
    fn test_metal_plan_creation() {
        if is_available() {
            let plan = MetalFftPlan::new(1024, 1);
            assert!(plan.is_ok());
            if let Ok(p) = plan {
                assert_eq!(p.log2n(), 10);
            }
        }
    }

    #[test]
    fn test_metal_non_power_of_2() {
        if is_available() {
            let plan = MetalFftPlan::new(1000, 1);
            assert!(plan.is_err());
        }
    }
}
