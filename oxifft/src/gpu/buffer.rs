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
/// All actual GPU I/O is handled inside `plan::execute()` via RAII types;
/// this struct serves only as a CPU staging buffer.
#[derive(Debug)]
pub struct GpuBuffer<T: Float> {
    /// Size of the buffer in elements.
    size: usize,
    /// Backend type.
    backend: GpuBackend,
    /// CPU-side staging data; populated/consumed by plan::execute().
    cpu_data: Vec<Complex<T>>,
}

// Safety: GpuBuffer only contains a Vec (which is Send+Sync) and plain data.
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
    ///
    /// Copies `data` into the CPU staging buffer.  The actual GPU transfer
    /// happens inside `plan::execute()`.
    pub fn upload(&mut self, data: &[Complex<T>]) -> GpuResult<()> {
        if data.len() != self.size {
            return Err(GpuError::SizeMismatch {
                expected: self.size,
                got: data.len(),
            });
        }
        // Copy to CPU staging buffer; GPU transfer happens inside plan::execute().
        self.cpu_data.copy_from_slice(data);
        Ok(())
    }

    /// Download data from GPU to CPU.
    ///
    /// Copies from the CPU staging buffer (populated by `plan::execute()`) into
    /// `data`.
    pub fn download(&mut self, data: &mut [Complex<T>]) -> GpuResult<()> {
        if data.len() != self.size {
            return Err(GpuError::SizeMismatch {
                expected: self.size,
                got: data.len(),
            });
        }
        // CPU staging buffer is populated by plan::execute(); copy out.
        data.copy_from_slice(&self.cpu_data);
        Ok(())
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
}

impl<T: Float> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        // GPU memory is managed inside plan::execute() via RAII types.
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
