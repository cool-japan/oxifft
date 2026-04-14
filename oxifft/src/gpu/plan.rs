//! GPU FFT plan.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::backend::GpuBackend;
use super::buffer::GpuBuffer;
use super::error::{GpuError, GpuResult};
use super::GpuFftEngine;
use crate::kernel::{Complex, Float};

/// FFT direction for GPU transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GpuDirection {
    /// Forward FFT (time to frequency domain).
    Forward,
    /// Inverse FFT (frequency to time domain).
    Inverse,
}

/// GPU FFT plan configuration.
#[derive(Debug, Clone)]
pub struct GpuPlanConfig {
    /// Transform size.
    pub size: usize,
    /// Number of transforms in batch.
    pub batch_size: usize,
    /// Backend to use.
    pub backend: GpuBackend,
    /// Whether to normalize inverse transform.
    pub normalize_inverse: bool,
}

impl Default for GpuPlanConfig {
    fn default() -> Self {
        Self {
            size: 0,
            batch_size: 1,
            backend: GpuBackend::Auto,
            normalize_inverse: true,
        }
    }
}

/// High-level GPU FFT interface.
///
/// Provides a simple API for GPU-accelerated FFT with automatic
/// backend selection and memory management.
pub struct GpuFft<T: Float> {
    /// Transform size.
    size: usize,
    /// Batch size.
    batch_size: usize,
    /// Active backend.
    backend: GpuBackend,
    /// Whether to normalize inverse.
    normalize_inverse: bool,
    /// Input buffer.
    input_buffer: GpuBuffer<T>,
    /// Output buffer.
    output_buffer: GpuBuffer<T>,
    /// Backend-specific plan handle.
    #[cfg(feature = "cuda")]
    cuda_plan: Option<super::cuda::CudaFftPlan>,
    #[cfg(feature = "metal")]
    metal_plan: Option<super::metal::MetalFftPlan>,
}

impl<T: Float> GpuFft<T> {
    /// Create a new GPU FFT plan.
    ///
    /// # Arguments
    ///
    /// * `size` - Transform size
    /// * `backend` - GPU backend to use
    ///
    /// # Returns
    ///
    /// GPU FFT plan or error if GPU is not available.
    pub fn new(size: usize, backend: GpuBackend) -> GpuResult<Self> {
        Self::with_config(GpuPlanConfig {
            size,
            batch_size: 1,
            backend,
            normalize_inverse: true,
        })
    }

    /// Create a GPU FFT plan with custom configuration.
    pub fn with_config(config: GpuPlanConfig) -> GpuResult<Self> {
        if config.size == 0 {
            return Err(GpuError::InvalidSize(0));
        }

        let total_size = config.size * config.batch_size;

        // Determine actual backend to use
        let actual_backend = match config.backend {
            GpuBackend::Auto => super::best_backend().ok_or(GpuError::NoBackendAvailable)?,
            other => {
                if !other.is_available() {
                    return Err(GpuError::NoBackendAvailable);
                }
                other
            }
        };

        // Allocate buffers
        let input_buffer = GpuBuffer::new(total_size, actual_backend)?;
        let output_buffer = GpuBuffer::new(total_size, actual_backend)?;

        // Create backend-specific plan
        #[cfg(feature = "cuda")]
        let cuda_plan = if actual_backend == GpuBackend::Cuda {
            Some(super::cuda::CudaFftPlan::new(
                config.size,
                config.batch_size,
            )?)
        } else {
            None
        };

        #[cfg(feature = "metal")]
        let metal_plan = if actual_backend == GpuBackend::Metal {
            Some(super::metal::MetalFftPlan::new(
                config.size,
                config.batch_size,
            )?)
        } else {
            None
        };

        Ok(Self {
            size: config.size,
            batch_size: config.batch_size,
            backend: actual_backend,
            normalize_inverse: config.normalize_inverse,
            input_buffer,
            output_buffer,
            #[cfg(feature = "cuda")]
            cuda_plan,
            #[cfg(feature = "metal")]
            metal_plan,
        })
    }

    /// Create a batched GPU FFT plan.
    pub fn batched(size: usize, batch_size: usize, backend: GpuBackend) -> GpuResult<Self> {
        Self::with_config(GpuPlanConfig {
            size,
            batch_size,
            backend,
            normalize_inverse: true,
        })
    }

    /// Execute forward FFT.
    pub fn forward(&mut self, input: &[Complex<T>]) -> GpuResult<Vec<Complex<T>>> {
        let expected_size = self.size * self.batch_size;
        if input.len() != expected_size {
            return Err(GpuError::SizeMismatch {
                expected: expected_size,
                got: input.len(),
            });
        }

        // Upload input
        self.input_buffer.upload(input)?;

        // Execute FFT
        self.execute_internal(GpuDirection::Forward)?;

        // Download output
        let mut output = vec![Complex::<T>::zero(); expected_size];
        self.output_buffer.download(&mut output)?;

        Ok(output)
    }

    /// Execute inverse FFT.
    pub fn inverse(&mut self, input: &[Complex<T>]) -> GpuResult<Vec<Complex<T>>> {
        let expected_size = self.size * self.batch_size;
        if input.len() != expected_size {
            return Err(GpuError::SizeMismatch {
                expected: expected_size,
                got: input.len(),
            });
        }

        // Upload input
        self.input_buffer.upload(input)?;

        // Execute FFT
        self.execute_internal(GpuDirection::Inverse)?;

        // Download output
        let mut output = vec![Complex::<T>::zero(); expected_size];
        self.output_buffer.download(&mut output)?;

        // Normalize if requested
        if self.normalize_inverse {
            let scale = T::ONE / T::from_usize(self.size);
            for c in &mut output {
                *c = Complex::new(c.re * scale, c.im * scale);
            }
        }

        Ok(output)
    }

    /// Execute forward FFT with pre-allocated output.
    pub fn forward_into(
        &mut self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
    ) -> GpuResult<()> {
        let expected_size = self.size * self.batch_size;
        if input.len() != expected_size || output.len() != expected_size {
            return Err(GpuError::SizeMismatch {
                expected: expected_size,
                got: input.len().min(output.len()),
            });
        }

        self.input_buffer.upload(input)?;
        self.execute_internal(GpuDirection::Forward)?;
        self.output_buffer.download(output)?;

        Ok(())
    }

    /// Execute inverse FFT with pre-allocated output.
    pub fn inverse_into(
        &mut self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
    ) -> GpuResult<()> {
        let expected_size = self.size * self.batch_size;
        if input.len() != expected_size || output.len() != expected_size {
            return Err(GpuError::SizeMismatch {
                expected: expected_size,
                got: input.len().min(output.len()),
            });
        }

        self.input_buffer.upload(input)?;
        self.execute_internal(GpuDirection::Inverse)?;
        self.output_buffer.download(output)?;

        if self.normalize_inverse {
            let scale = T::ONE / T::from_usize(self.size);
            for c in output.iter_mut() {
                *c = Complex::new(c.re * scale, c.im * scale);
            }
        }

        Ok(())
    }

    fn execute_internal(&mut self, _direction: GpuDirection) -> GpuResult<()> {
        match self.backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    if let Some(ref plan) = self.cuda_plan {
                        return plan.execute(
                            &self.input_buffer,
                            &mut self.output_buffer,
                            _direction,
                        );
                    }
                }
                Err(GpuError::NoBackendAvailable)
            }
            GpuBackend::Metal => {
                #[cfg(feature = "metal")]
                {
                    if let Some(ref plan) = self.metal_plan {
                        return plan.execute(
                            &self.input_buffer,
                            &mut self.output_buffer,
                            _direction,
                        );
                    }
                }
                Err(GpuError::NoBackendAvailable)
            }
            _ => Err(GpuError::Unsupported("Backend not implemented".into())),
        }
    }
}

impl<T: Float> GpuFftEngine<T> for GpuFft<T> {
    fn forward(&self, _input: &[Complex<T>], _output: &mut [Complex<T>]) -> GpuResult<()> {
        // This requires a mutable self, so we can't implement the trait method directly
        // Users should use forward_into instead
        Err(GpuError::Unsupported(
            "Use forward_into for non-mutable access".into(),
        ))
    }

    fn inverse(&self, _input: &[Complex<T>], _output: &mut [Complex<T>]) -> GpuResult<()> {
        Err(GpuError::Unsupported(
            "Use inverse_into for non-mutable access".into(),
        ))
    }

    fn forward_inplace(&self, _data: &mut [Complex<T>]) -> GpuResult<()> {
        Err(GpuError::Unsupported(
            "In-place GPU FFT not implemented".into(),
        ))
    }

    fn inverse_inplace(&self, _data: &mut [Complex<T>]) -> GpuResult<()> {
        Err(GpuError::Unsupported(
            "In-place GPU FFT not implemented".into(),
        ))
    }

    fn size(&self) -> usize {
        self.size
    }

    fn backend(&self) -> GpuBackend {
        self.backend
    }

    fn sync(&self) -> GpuResult<()> {
        match self.backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                return super::cuda::synchronize();
                #[cfg(not(feature = "cuda"))]
                Err(GpuError::NoBackendAvailable)
            }
            GpuBackend::Metal => {
                #[cfg(feature = "metal")]
                return super::metal::synchronize();
                #[cfg(not(feature = "metal"))]
                Err(GpuError::NoBackendAvailable)
            }
            _ => Ok(()), // No-op for unsupported backends
        }
    }
}

/// Convenience type alias for GPU FFT plan.
pub type GpuPlan<T> = GpuFft<T>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_plan_config_default() {
        let config = GpuPlanConfig::default();
        assert_eq!(config.size, 0);
        assert_eq!(config.batch_size, 1);
        assert!(config.normalize_inverse);
    }

    #[test]
    fn test_gpu_fft_size_validation() {
        let result: GpuResult<GpuFft<f64>> = GpuFft::new(0, GpuBackend::Auto);
        assert!(result.is_err());
    }
}
