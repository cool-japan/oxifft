//! GPU memory buffer abstraction.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::error::{GpuError, GpuResult};
use super::GpuBackend;
use crate::kernel::{Complex, Float};

/// GPU memory buffer.
///
/// Manages memory allocation and data transfer between CPU and GPU.
#[derive(Debug)]
pub struct GpuBuffer<T: Float> {
    /// Size of the buffer in elements.
    size: usize,
    /// Backend type.
    backend: GpuBackend,
    /// CPU-side data (for fallback/staging).
    cpu_data: Vec<Complex<T>>,
    /// GPU device pointer (opaque, backend-specific).
    #[cfg(feature = "cuda")]
    cuda_ptr: Option<*mut core::ffi::c_void>,
    #[cfg(feature = "metal")]
    metal_buffer: Option<super::metal::MetalBufferHandle>,
}

// Safety: GPU pointers are managed by the backend and are thread-safe
unsafe impl<T: Float> Send for GpuBuffer<T> {}
unsafe impl<T: Float> Sync for GpuBuffer<T> {}

impl<T: Float> GpuBuffer<T> {
    /// Create a new GPU buffer with the specified size.
    pub fn new(size: usize, backend: GpuBackend) -> GpuResult<Self> {
        if size == 0 {
            return Err(GpuError::InvalidSize(size));
        }

        let cpu_data = vec![Complex::<T>::zero(); size];

        Ok(Self {
            size,
            backend,
            cpu_data,
            #[cfg(feature = "cuda")]
            cuda_ptr: None,
            #[cfg(feature = "metal")]
            metal_buffer: None,
        })
    }

    /// Create a GPU buffer from existing data.
    pub fn from_slice(data: &[Complex<T>], backend: GpuBackend) -> GpuResult<Self> {
        if data.is_empty() {
            return Err(GpuError::InvalidSize(0));
        }

        let mut buffer = Self::new(data.len(), backend)?;
        buffer.upload(data)?;
        Ok(buffer)
    }

    /// Get the size of the buffer in elements.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Get the backend type.
    #[must_use]
    pub const fn backend(&self) -> GpuBackend {
        self.backend
    }

    /// Upload data from CPU to GPU.
    pub fn upload(&mut self, data: &[Complex<T>]) -> GpuResult<()> {
        if data.len() != self.size {
            return Err(GpuError::SizeMismatch {
                expected: self.size,
                got: data.len(),
            });
        }

        // Copy to CPU staging buffer
        self.cpu_data.copy_from_slice(data);

        // Upload to GPU based on backend
        match self.backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    self.upload_cuda()?;
                    Ok(())
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(GpuError::NoBackendAvailable);
                }
            }
            GpuBackend::Metal => {
                #[cfg(feature = "metal")]
                {
                    self.upload_metal()?;
                    Ok(())
                }
                #[cfg(not(feature = "metal"))]
                {
                    return Err(GpuError::NoBackendAvailable);
                }
            }
            GpuBackend::Auto => {
                // Use first available backend
                #[cfg(feature = "cuda")]
                if super::cuda::is_available() {
                    self.upload_cuda()?;
                    return Ok(());
                }
                #[cfg(feature = "metal")]
                if super::metal::is_available() {
                    self.upload_metal()?;
                    return Ok(());
                }
                Err(GpuError::NoBackendAvailable)
            }
            _ => Err(GpuError::Unsupported("Backend not implemented".into())),
        }
    }

    /// Download data from GPU to CPU.
    pub fn download(&mut self, data: &mut [Complex<T>]) -> GpuResult<()> {
        if data.len() != self.size {
            return Err(GpuError::SizeMismatch {
                expected: self.size,
                got: data.len(),
            });
        }

        // Download from GPU based on backend
        match self.backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    self.download_cuda()?;
                    data.copy_from_slice(&self.cpu_data);
                    Ok(())
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(GpuError::NoBackendAvailable);
                }
            }
            GpuBackend::Metal => {
                #[cfg(feature = "metal")]
                {
                    self.download_metal()?;
                    data.copy_from_slice(&self.cpu_data);
                    Ok(())
                }
                #[cfg(not(feature = "metal"))]
                {
                    return Err(GpuError::NoBackendAvailable);
                }
            }
            GpuBackend::Auto => {
                #[cfg(feature = "cuda")]
                if super::cuda::is_available() {
                    self.download_cuda()?;
                    data.copy_from_slice(&self.cpu_data);
                    return Ok(());
                }
                #[cfg(feature = "metal")]
                if super::metal::is_available() {
                    self.download_metal()?;
                    data.copy_from_slice(&self.cpu_data);
                    return Ok(());
                }
                Err(GpuError::NoBackendAvailable)
            }
            _ => Err(GpuError::Unsupported("Backend not implemented".into())),
        }
    }

    /// Get a reference to the CPU staging data.
    #[must_use]
    pub fn cpu_data(&self) -> &[Complex<T>] {
        &self.cpu_data
    }

    /// Get a mutable reference to the CPU staging data.
    pub fn cpu_data_mut(&mut self) -> &mut [Complex<T>] {
        &mut self.cpu_data
    }

    #[cfg(feature = "cuda")]
    fn upload_cuda(&mut self) -> GpuResult<()> {
        // CUDA upload implementation
        super::cuda::upload_buffer(self)
    }

    #[cfg(feature = "cuda")]
    fn download_cuda(&mut self) -> GpuResult<()> {
        // CUDA download implementation
        super::cuda::download_buffer(self)
    }

    #[cfg(feature = "metal")]
    fn upload_metal(&mut self) -> GpuResult<()> {
        // Metal upload implementation
        super::metal::upload_buffer(self)
    }

    #[cfg(feature = "metal")]
    fn download_metal(&mut self) -> GpuResult<()> {
        // Metal download implementation
        super::metal::download_buffer(self)
    }
}

impl<T: Float> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        // Free GPU memory based on backend
        #[cfg(feature = "cuda")]
        if let Some(ptr) = self.cuda_ptr.take() {
            let _ = super::cuda::free_buffer(ptr);
        }

        #[cfg(feature = "metal")]
        if let Some(handle) = self.metal_buffer.take() {
            let _ = super::metal::free_buffer(handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_buffer_creation() {
        // This should work even without GPU
        let buffer: GpuBuffer<f64> =
            GpuBuffer::new(1024, GpuBackend::Auto).expect("Failed to create buffer");
        assert_eq!(buffer.size(), 1024);
    }

    #[test]
    fn test_gpu_buffer_cpu_data() {
        let mut buffer: GpuBuffer<f64> =
            GpuBuffer::new(8, GpuBackend::Auto).expect("Failed to create buffer");

        // Modify CPU data
        buffer.cpu_data_mut()[0] = Complex::new(1.0, 2.0);

        assert_eq!(buffer.cpu_data()[0], Complex::new(1.0, 2.0));
    }
}
