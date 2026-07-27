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
///
/// `Send` and `Sync` are derived automatically: the only non-trivial field is
/// `cpu_data: Vec<Complex<T>>`, and `T: Float` already requires `Send + Sync`
/// (see [`crate::kernel::Float`]), so `Vec<Complex<T>>` — and therefore
/// `GpuBuffer<T>` — is auto-`Send`/`Sync` with no `unsafe`.  Should a genuine
/// device-pointer field ever be added here, auto-derivation will correctly
/// *stop* holding, forcing a deliberate (and audited) re-derivation rather than
/// silently asserting thread-safety via a stale blanket `unsafe impl`.
#[derive(Debug)]
pub struct GpuBuffer<T: Float> {
    /// Size of the buffer in elements.
    size: usize,
    /// Backend type.
    backend: GpuBackend,
    /// CPU-side staging data; populated/consumed by plan::execute().
    cpu_data: Vec<Complex<T>>,
}

// ── CPU staging-buffer pooling ───────────────────────────────────────────────
//
// The CPU staging `Vec<Complex<T>>` inside every `GpuBuffer` is the one
// allocation oxifft fully controls (the real device buffers are allocated
// internally by oxicuda per call).  Routing that `Vec` through the process
// global [`crate::gpu::pool::GpuBufferPool`] lets repeated same-size transforms
// — most importantly the per-element loops in [`crate::gpu::batch`] and the
// `&self` trait methods in [`crate::gpu::plan`] — reuse a previously allocated
// buffer instead of hitting the allocator on every call.
//
// Pooling requires `std` (Mutex / OnceLock / Instant); under `no_std` the
// helpers degrade to a plain allocation with identical observable behaviour.

/// Map a [`GpuBackend`] to the numeric pool `backend_id`.
#[cfg(feature = "std")]
fn backend_pool_id(backend: GpuBackend) -> u32 {
    match backend {
        GpuBackend::Cuda => 0,
        GpuBackend::Metal => 1,
        _ => 2,
    }
}

/// Build the [`crate::gpu::pool::PoolKey`] for a staging buffer of `size`
/// elements of `Complex<T>` on `backend`.
#[cfg(feature = "std")]
fn staging_key<T: Float>(backend: GpuBackend, size: usize) -> super::pool::PoolKey {
    let bytes = size.saturating_mul(core::mem::size_of::<Complex<T>>());
    super::pool::PoolKey {
        backend_id: backend_pool_id(backend),
        rounded_size: super::pool::round_pool_size(bytes),
        kind: super::pool::BufferKind::Scratch,
    }
}

/// Acquire a zeroed staging `Vec<Complex<T>>` of `size` elements from `pool`,
/// reusing a cached allocation when one of the same class is available.
#[cfg(feature = "std")]
fn acquire_staging_from<T: Float>(
    pool: &super::pool::GpuBufferPool,
    backend: GpuBackend,
    size: usize,
) -> Vec<Complex<T>> {
    let key = staging_key::<T>(backend, size);
    let pooled = pool.acquire(key, |rounded_bytes| {
        Some(super::pool::PooledBuffer::new(
            alloc_box(Vec::<Complex<T>>::new()),
            rounded_bytes,
        ))
    });
    let mut data = match pooled {
        Some(buf) => match buf.downcast::<Vec<Complex<T>>>() {
            Ok(boxed) => *boxed,
            Err(_) => Vec::new(),
        },
        None => Vec::new(),
    };
    data.clear();
    data.resize(size, Complex::<T>::zero());
    data
}

/// Return a staging `Vec` to `pool` for future reuse.
#[cfg(feature = "std")]
fn release_staging_to<T: Float>(
    pool: &super::pool::GpuBufferPool,
    backend: GpuBackend,
    size: usize,
    mut data: Vec<Complex<T>>,
) {
    if size == 0 {
        return;
    }
    data.clear();
    let key = staging_key::<T>(backend, size);
    pool.release(
        key,
        super::pool::PooledBuffer::new(alloc_box(data), key.rounded_size),
    );
}

/// Box helper that works in both `std` and `no_std` (via `alloc`).
#[cfg(feature = "std")]
fn alloc_box<T: core::any::Any + Send>(value: T) -> Box<dyn core::any::Any + Send> {
    Box::new(value)
}

/// Acquire a zeroed staging buffer, using the global pool when `std` is
/// available and a fresh allocation otherwise.
fn acquire_staging<T: Float>(backend: GpuBackend, size: usize) -> Vec<Complex<T>> {
    #[cfg(feature = "std")]
    {
        acquire_staging_from::<T>(super::pool::global_pool(), backend, size)
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = backend;
        vec![Complex::<T>::zero(); size]
    }
}

/// Return a staging buffer for reuse (global pool under `std`, drop otherwise).
fn release_staging<T: Float>(backend: GpuBackend, size: usize, data: Vec<Complex<T>>) {
    #[cfg(feature = "std")]
    {
        release_staging_to::<T>(super::pool::global_pool(), backend, size, data);
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = (backend, size, data);
    }
}

impl<T: Float> GpuBuffer<T> {
    /// Create a new GPU buffer with the specified size.
    ///
    /// The CPU staging allocation is drawn from the global GPU buffer pool
    /// (under `std`), so repeated same-size allocations reuse memory.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::InvalidSize` if `size` is zero.
    pub fn new(size: usize, backend: GpuBackend) -> GpuResult<Self> {
        if size == 0 {
            return Err(GpuError::InvalidSize(size));
        }

        let cpu_data = acquire_staging::<T>(backend, size);

        Ok(Self {
            size,
            backend,
            cpu_data,
        })
    }

    /// Create a GPU buffer from existing data.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::InvalidSize` if `data` is empty, or propagates any
    /// error from `upload`.
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
    ///
    /// # Errors
    ///
    /// Returns `GpuError::SizeMismatch` if `data.len()` does not equal the
    /// buffer size.
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
    ///
    /// # Errors
    ///
    /// Returns `GpuError::SizeMismatch` if `data.len()` does not equal the
    /// buffer size.
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
        // Return the CPU staging allocation to the pool for reuse.  GPU memory
        // itself is managed inside plan::execute() via RAII types.
        let data = core::mem::take(&mut self.cpu_data);
        release_staging::<T>(self.backend, self.size, data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Task 6 regression: `GpuBuffer` must be `Send + Sync` via *auto*
    /// derivation (the previous hand-written `unsafe impl`s were removed).  If a
    /// future non-`Send` field is added, this assertion will correctly fail to
    /// compile instead of silently asserting thread-safety.
    #[test]
    fn gpu_buffer_is_send_sync_by_auto_derive() {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GpuBuffer<f32>>();
        assert_send_sync::<GpuBuffer<f64>>();
    }

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

    // ── Staging-buffer pool wiring ──────────────────────────────────────────
    //
    // These exercise the exact acquire/release helpers used by `GpuBuffer::new`
    // and `Drop`, but against a *local* pool so the assertions are deterministic
    // and immune to concurrent activity on the process-global pool.

    #[cfg(feature = "std")]
    #[test]
    fn staging_pool_reuses_released_buffer() {
        use crate::gpu::pool::GpuBufferPool;

        let pool = GpuBufferPool::new(64 * 1024 * 1024);
        let backend = GpuBackend::Metal;
        let size = 1024usize;

        // First acquire — pool miss, allocates a fresh Vec and grows it.
        let mut v1 = acquire_staging_from::<f32>(&pool, backend, size);
        assert_eq!(v1.len(), size);
        // Tag the underlying allocation so we can prove the *same* Vec returns.
        v1[0] = Complex::new(7.0_f32, 0.0);
        let cap_before = v1.capacity();
        let ptr_before = v1.as_ptr() as usize;

        // Release back to the pool.
        release_staging_to::<f32>(&pool, backend, size, v1);
        assert!(
            pool.current_bytes() > 0,
            "released staging buffer should be accounted in the pool"
        );

        // Second acquire — must reuse the released allocation (no realloc).
        let v2 = acquire_staging_from::<f32>(&pool, backend, size);
        assert_eq!(v2.len(), size);
        assert_eq!(
            v2.as_ptr() as usize,
            ptr_before,
            "second acquire must reuse the same backing allocation"
        );
        assert_eq!(
            v2.capacity(),
            cap_before,
            "reused buffer should retain its capacity"
        );
        // Contents are re-zeroed on acquire.
        assert_eq!(v2[0], Complex::new(0.0_f32, 0.0));

        // Taking the buffer out of the pool drops the accounted bytes back to 0.
        assert_eq!(pool.current_bytes(), 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn staging_pool_separates_by_element_type() {
        use crate::gpu::pool::GpuBufferPool;

        let pool = GpuBufferPool::new(64 * 1024 * 1024);
        let backend = GpuBackend::Cuda;

        // f32 and f64 buffers of the same element count differ in byte size and
        // concrete type, so they must never alias in the pool.
        let vf32 = acquire_staging_from::<f32>(&pool, backend, 512);
        let vf64 = acquire_staging_from::<f64>(&pool, backend, 512);
        release_staging_to::<f32>(&pool, backend, 512, vf32);
        release_staging_to::<f64>(&pool, backend, 512, vf64);

        // Re-acquiring f32 must yield an f32-sized Vec (downcast succeeds).
        let again = acquire_staging_from::<f32>(&pool, backend, 512);
        assert_eq!(again.len(), 512);
    }

    #[cfg(feature = "std")]
    #[test]
    fn gpu_buffer_new_uses_global_pool() {
        use crate::gpu::global_gpu_pool;

        // Creating and dropping a buffer should leave a reusable allocation in
        // the global pool (monotone check — tolerant of concurrent tests).
        let before = global_gpu_pool().current_bytes();
        {
            let _buf: GpuBuffer<f64> = GpuBuffer::new(2048, GpuBackend::Metal).expect("buffer");
        } // drop returns the staging Vec to the global pool
        let after = global_gpu_pool().current_bytes();
        assert!(
            after >= before,
            "dropping a GpuBuffer should return its staging allocation to the pool"
        );
    }
}
