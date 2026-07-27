//! Metal backend for GPU FFT.
//!
//! Provides GPU-accelerated FFT on Apple Silicon and AMD GPUs on macOS
//! via the real `oxicuda_metal::fft::MetalFftPlan` implementation.
//!
//! # Requirements
//!
//! - macOS 10.13+ or iOS 11+
//! - Metal-compatible GPU
//!
//! # Implementation Notes
//!
//! On macOS with a Metal-capable GPU this module dispatches radix-2 DIT
//! FFTs to the GPU through oxicuda-metal.  On non-macOS targets
//! `is_available()` returns `false` and `MetalFftPlan::new()` returns
//! `Err(GpuError::NoBackendAvailable)`.

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

/// Check if Metal is available on this system.
///
/// Attempts to open the system-default Metal device; returns `true` when
/// that succeeds.  Always `false` on non-macOS targets.
#[must_use]
pub fn is_available() -> bool {
    oxicuda_metal::device::MetalDevice::new().is_ok()
}

/// Query Metal device capabilities.
///
/// # Errors
///
/// Returns `GpuError::InitializationFailed` if no Metal-capable device is
/// found (non-macOS or unsupported hardware).
pub fn query_capabilities() -> GpuResult<GpuCapabilities> {
    let device = oxicuda_metal::device::MetalDevice::new()
        .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;

    // `max_buffer_length()` is the largest single Metal buffer allocation the
    // device permits.  Derive the maximum FFT length from it (each element is a
    // `Complex<f32>` = 8 bytes) rather than hardcoding a constant.
    let max_buffer_bytes = device.max_buffer_length();
    let elem_size = core::mem::size_of::<num_complex::Complex<f32>>() as u64;
    let max_fft_size = if elem_size == 0 || max_buffer_bytes == 0 {
        1 << 24
    } else {
        // Largest power-of-two element count that fits in one buffer.
        let elems = (max_buffer_bytes / elem_size) as usize;
        elems
            .checked_next_power_of_two()
            .map_or(elems, |p| if p > elems { p >> 1 } else { p })
    };

    Ok(GpuCapabilities {
        backend: GpuBackend::Metal,
        device_name: device.name().to_string(),
        // Metal's public API exposes the maximum single-buffer allocation, not
        // total device VRAM.  Report it as the total-memory ceiling (best
        // available proxy) and leave free-memory unknown (0).
        total_memory: max_buffer_bytes,
        available_memory: 0, // Metal does not expose free VRAM directly.
        max_fft_size,
        supports_f64: false, // Metal has limited f64 support.
        supports_f16: true,
        compute_units: 0, // oxicuda-metal does not expose the GPU core count.
        max_workgroup_size: 1024,
        hardware_accelerated: true,
    })
}

/// Synchronize Metal device.
///
/// # Errors
///
/// This function currently cannot return an error; command buffers are
/// submitted synchronously by oxicuda-metal.  The `Result` signature is
/// retained for API symmetry with asynchronous backends.
pub fn synchronize() -> GpuResult<()> {
    // Metal command buffers are submitted synchronously in oxicuda-metal;
    // no additional synchronisation is required here.
    Ok(())
}

/// Metal FFT plan backed by the real `oxicuda_metal::fft::MetalFftPlan`.
pub struct MetalFftPlan {
    /// Transform size.
    size: usize,
    /// Batch size.
    batch_size: usize,
    /// The real oxicuda-metal plan that dispatches to the GPU.
    inner: oxicuda_metal::fft::MetalFftPlan,
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for MetalFftPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalFftPlan")
            .field("size", &self.size)
            .field("batch_size", &self.batch_size)
            .finish_non_exhaustive()
    }
}

impl MetalFftPlan {
    /// Create a new Metal FFT plan.
    ///
    /// # Errors
    ///
    /// - `GpuError::NoBackendAvailable` — no Metal device found (or non-macOS).
    /// - `GpuError::InvalidSize` — `size` is zero.
    /// - `GpuError::Unsupported` — `size` is not a power of two.
    /// - `GpuError::InitializationFailed` — oxicuda-metal plan creation error.
    pub fn new(size: usize, batch_size: usize) -> GpuResult<Self> {
        if !is_available() {
            return Err(GpuError::NoBackendAvailable);
        }

        if size == 0 {
            return Err(GpuError::InvalidSize(size));
        }

        if !size.is_power_of_two() {
            return Err(GpuError::Unsupported(
                "Metal FFT requires power-of-2 sizes".into(),
            ));
        }

        let inner =
            oxicuda_metal::fft::MetalFftPlan::new(size, batch_size).map_err(GpuError::from)?;

        Ok(Self {
            size,
            batch_size,
            inner,
        })
    }

    /// Execute the FFT on the Metal GPU.
    ///
    /// Input samples are converted from `Complex<T>` to `Complex<f32>` (Metal
    /// natively operates on f32), dispatched to the GPU, and the results are
    /// converted back to `Complex<T>`.
    ///
    /// # Normalisation convention
    ///
    /// This method returns the **unnormalised** transform in both directions
    /// (the FFTW convention), matching the CUDA backend's CPU path.  The
    /// underlying `oxicuda_metal` inverse transform applies an internal `1/N`
    /// scale, so this method multiplies the inverse result back by `N` to undo
    /// it.  The single, user-controllable `1/N` normalisation is applied once,
    /// at the high-level [`crate::gpu::GpuFft`] layer, governed by
    /// `GpuPlanConfig::normalize_inverse`.  This keeps both GPU backends
    /// numerically identical and prevents the historical double-normalisation
    /// bug where `GpuFft::inverse()` returned results scaled by `1/N²`.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::SizeMismatch` if buffer sizes do not match the plan,
    /// or propagates Metal execution errors from the oxicuda-metal backend.
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

        // Convert input Complex<T> → Complex<f32> (Metal operates on f32).
        let input_f32: Vec<num_complex::Complex<f32>> = input
            .cpu_data()
            .iter()
            .map(|c| {
                let re = num_traits::ToPrimitive::to_f64(&c.re)
                    .map(|v| v as f32)
                    .unwrap_or(0.0_f32);
                let im = num_traits::ToPrimitive::to_f64(&c.im)
                    .map(|v| v as f32)
                    .unwrap_or(0.0_f32);
                num_complex::Complex::new(re, im)
            })
            .collect();

        let mut output_f32 = vec![num_complex::Complex::<f32>::new(0.0, 0.0); expected_size];

        // Map direction to the oxicuda-metal enum.
        let metal_dir = match direction {
            GpuDirection::Forward => oxicuda_metal::fft::MetalFftDirection::Forward,
            GpuDirection::Inverse => oxicuda_metal::fft::MetalFftDirection::Inverse,
        };

        // Execute on the Metal GPU.
        self.inner
            .execute(&input_f32, &mut output_f32, metal_dir)
            .map_err(GpuError::from)?;

        // `oxicuda_metal` normalises the inverse transform by 1/N internally.
        // Undo it here so this method is unnormalised (FFTW convention),
        // matching the CUDA backend; the high-level GpuFft layer owns the
        // single, opt-out `1/N` normalisation.  N is the per-transform size.
        if direction == GpuDirection::Inverse {
            let scale = self.size as f32;
            for c in &mut output_f32 {
                c.re *= scale;
                c.im *= scale;
            }
        }

        // Convert output Complex<f32> → Complex<T>.
        let out_data = output.cpu_data_mut();
        for (i, c) in output_f32.iter().enumerate() {
            out_data[i] = Complex::new(T::from_f64(c.re as f64), T::from_f64(c.im as f64));
        }

        Ok(())
    }

    /// Execute a batch of independent size-`n` transforms, packing/unpacking
    /// `Complex<T>` ↔ `Complex<f32>` around native oxicuda-metal dispatches.
    ///
    /// Each element of `inputs`/`outputs` is one transform of length
    /// `self.size()`.  Transforms are processed in chunks of this plan's own
    /// `batch_size`, packed into one contiguous buffer per chunk and submitted
    /// through the **already-compiled** inner plan — so the cached MSL pipelines
    /// are reused with no shader recompilation on any call (the plan built by
    /// `GpuFft::batched(n, b, Metal)` batches `b` transforms per GPU submission).
    /// The final partial chunk is zero-padded up to `batch_size`, and the padded
    /// results are discarded.  This is the native-batch path used by
    /// [`crate::gpu::batch::GpuBatchFft`]; because it never routes a size-`n`
    /// buffer through the per-buffer `size * batch_size` check, it works for
    /// plans built with any `batch_size` (the old per-element loop errored out
    /// for `batch_size > 1`).
    ///
    /// Follows the same unnormalised convention as [`Self::execute`].
    ///
    /// # Errors
    ///
    /// Propagates Metal execution errors, or `GpuError::SizeMismatch` if any
    /// slice length differs from `self.size()`.
    pub(crate) fn execute_batch_native<T: Float>(
        &self,
        inputs: &[&[Complex<T>]],
        outputs: &mut [&mut [Complex<T>]],
        direction: GpuDirection,
    ) -> GpuResult<()> {
        let n = self.size;
        let chunk = self.batch_size.max(1);
        let count = inputs.len();
        debug_assert_eq!(count, outputs.len());
        if count == 0 {
            return Ok(());
        }

        let metal_dir = match direction {
            GpuDirection::Forward => oxicuda_metal::fft::MetalFftDirection::Forward,
            GpuDirection::Inverse => oxicuda_metal::fft::MetalFftDirection::Inverse,
        };
        let inv_scale = n as f32;

        // Reusable pack/unpack scratch sized for one full `chunk` (n*chunk).
        let mut packed_in = vec![num_complex::Complex::<f32>::new(0.0, 0.0); n * chunk];
        let mut packed_out = vec![num_complex::Complex::<f32>::new(0.0, 0.0); n * chunk];

        let mut start = 0;
        while start < count {
            let end = (start + chunk).min(count);

            // Pack this chunk's inputs; zero the padding tail (if the last chunk
            // is short) so the inner plan always sees exactly n*chunk elements.
            for c in &mut packed_in {
                *c = num_complex::Complex::new(0.0, 0.0);
            }
            for (ci, idx) in (start..end).enumerate() {
                let inp = inputs[idx];
                if inp.len() != n {
                    return Err(GpuError::SizeMismatch {
                        expected: n,
                        got: inp.len(),
                    });
                }
                let base = ci * n;
                for (k, c) in inp.iter().enumerate() {
                    let re = num_traits::ToPrimitive::to_f64(&c.re)
                        .map(|v| v as f32)
                        .unwrap_or(0.0_f32);
                    let im = num_traits::ToPrimitive::to_f64(&c.im)
                        .map(|v| v as f32)
                        .unwrap_or(0.0_f32);
                    packed_in[base + k] = num_complex::Complex::new(re, im);
                }
            }

            self.inner
                .execute(&packed_in, &mut packed_out, metal_dir)
                .map_err(GpuError::from)?;

            // Undo oxicuda-metal's internal 1/N on the inverse (see `execute`).
            if direction == GpuDirection::Inverse {
                for c in &mut packed_out {
                    c.re *= inv_scale;
                    c.im *= inv_scale;
                }
            }

            // Unpack only the real (non-padding) transforms.
            for (ci, idx) in (start..end).enumerate() {
                let out = &mut outputs[idx];
                if out.len() != n {
                    return Err(GpuError::SizeMismatch {
                        expected: n,
                        got: out.len(),
                    });
                }
                let base = ci * n;
                for (k, slot) in out.iter_mut().enumerate() {
                    let c = packed_out[base + k];
                    *slot = Complex::new(T::from_f64(c.re as f64), T::from_f64(c.im as f64));
                }
            }

            start = end;
        }

        Ok(())
    }

    /// Execute a real-to-complex forward FFT on the Metal GPU.
    ///
    /// # Strategy
    ///
    /// Real input is zero-extended to complex (imaginary parts = 0), then a
    /// full C2C forward FFT is run.  The first `n/2 + 1` frequency bins are
    /// extracted into `output` (the half-spectrum sufficient to describe a
    /// real signal by the Hermitian symmetry property).
    ///
    /// # Errors
    ///
    /// - `GpuError::SizeMismatch` — lengths are inconsistent.
    /// - Propagated Metal execution errors.
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

        // Zero-extend real input → full complex buffer.
        let complex_input: Vec<num_complex::Complex<f32>> = input
            .iter()
            .map(|&x| num_complex::Complex::new(x, 0.0_f32))
            .collect();
        let mut complex_output = vec![num_complex::Complex::<f32>::new(0.0, 0.0); n];

        self.inner
            .execute(
                &complex_input,
                &mut complex_output,
                oxicuda_metal::fft::MetalFftDirection::Forward,
            )
            .map_err(GpuError::from)?;

        // Extract the first n/2 + 1 bins (half-spectrum).
        output.copy_from_slice(&complex_output[..half]);
        Ok(())
    }

    /// Execute a complex-to-real inverse FFT on the Metal GPU.
    ///
    /// # Strategy
    ///
    /// Input is `n/2 + 1` complex bins (the positive-frequency half-spectrum).
    /// A full `n`-point conjugate-symmetric spectrum is reconstructed via
    /// `X[n-k] = conj(X[k])` for `k in 1..n/2`, then an inverse C2C FFT is
    /// run.  The real parts of the time-domain output are written to `output`;
    /// imaginary parts are discarded (they are numerically zero by construction).
    ///
    /// # Errors
    ///
    /// - `GpuError::SizeMismatch` — lengths are inconsistent.
    /// - Propagated Metal execution errors.
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

        // Reconstruct the full conjugate-symmetric spectrum.
        let mut full_spectrum = vec![num_complex::Complex::<f32>::new(0.0, 0.0); n];
        // Copy the positive-frequency half (indices 0..=n/2).
        full_spectrum[..half].copy_from_slice(input);
        // Mirror: X[n-k] = conj(X[k]) for k in 1..n/2.
        for k in 1..n / 2 {
            full_spectrum[n - k] = input[k].conj();
        }

        let mut time_domain = vec![num_complex::Complex::<f32>::new(0.0, 0.0); n];

        self.inner
            .execute(
                &full_spectrum,
                &mut time_domain,
                oxicuda_metal::fft::MetalFftDirection::Inverse,
            )
            .map_err(GpuError::from)?;

        // Extract real parts; imaginary parts are ~0 by Hermitian symmetry.
        for (i, c) in time_domain.iter().enumerate() {
            output[i] = c.re;
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

    /// Return log₂ of the transform size.
    #[must_use]
    pub fn log2n(&self) -> u32 {
        self.inner.log2n()
    }
}

impl Drop for MetalFftPlan {
    fn drop(&mut self) {
        // Metal objects are reference-counted — cleanup is automatic.
    }
}

/// Metal buffer handle.
///
/// Kept for backwards compatibility with the `Drop` implementation in
/// `buffer.rs`.  Metal buffer management is handled internally by
/// oxicuda-metal during `execute`.
#[derive(Debug)]
pub struct MetalBufferHandle {
    /// Buffer identifier.
    pub id: u64,
}

/// Upload a buffer to the Metal device.
///
/// Metal buffer staging is handled transparently inside `execute`; this
/// function is a no-op.
///
/// # Errors
///
/// This function currently cannot return an error; the `Result` signature is
/// retained for API symmetry with backends that perform explicit buffer uploads.
pub fn upload_buffer<T: Float>(_buffer: &mut GpuBuffer<T>) -> GpuResult<()> {
    Ok(())
}

/// Download a buffer from the Metal device.
///
/// Metal buffer readback is handled transparently inside `execute`; this
/// function is a no-op.
///
/// # Errors
///
/// This function currently cannot return an error; the `Result` signature is
/// retained for API symmetry with backends that perform explicit buffer readback.
pub fn download_buffer<T: Float>(_buffer: &mut GpuBuffer<T>) -> GpuResult<()> {
    Ok(())
}

/// Free a Metal buffer handle.
///
/// Metal buffers are reference-counted and freed automatically; this
/// function is a no-op.
///
/// # Errors
///
/// This function currently cannot return an error; the `Result` signature is
/// retained for API symmetry with backends that perform explicit GPU memory
/// deallocation.
pub fn free_buffer(_handle: MetalBufferHandle) -> GpuResult<()> {
    Ok(())
}

#[cfg(all(test, feature = "metal"))]
mod gpu_r2c_tests {
    use super::*;

    fn run_r2c_roundtrip(n: usize) {
        if !is_available() {
            return;
        }
        let plan = MetalFftPlan::new(n, 1).expect("MetalFftPlan::new");
        let half = n / 2 + 1;
        let tolerance = 1e-6_f32 * n as f32;

        let input: Vec<f32> = (0..n)
            .map(|k| {
                let t = k as f32 / n as f32;
                (2.0 * std::f32::consts::PI * t).sin()
                    + 0.5 * (6.0 * std::f32::consts::PI * t).cos()
            })
            .collect();

        let mut spectrum = vec![num_complex::Complex::<f32>::new(0.0, 0.0); half];
        plan.forward_r2c(&input, &mut spectrum)
            .expect("forward_r2c");

        let mut recovered = vec![0.0_f32; n];
        plan.inverse_c2r(&spectrum, &mut recovered)
            .expect("inverse_c2r");

        for (i, (&orig, &rec)) in input.iter().zip(recovered.iter()).enumerate() {
            let err = (orig - rec).abs();
            assert!(
                err <= tolerance,
                "n={n} sample {i}: expected {orig}, got {rec}, error {err} > {tolerance}"
            );
        }
    }

    #[test]
    fn metal_r2c_roundtrip_size64() {
        run_r2c_roundtrip(64);
    }

    #[test]
    fn metal_r2c_roundtrip_size256() {
        run_r2c_roundtrip(256);
    }

    #[test]
    fn metal_r2c_roundtrip_size1024() {
        run_r2c_roundtrip(1024);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_availability() {
        // Must not panic — just probe whether Metal is present.
        let _ = is_available();
    }

    #[test]
    fn test_metal_capabilities() {
        if is_available() {
            let caps = query_capabilities().expect("Failed to query capabilities");
            assert_eq!(caps.backend, GpuBackend::Metal);
            assert!(caps.supports_f16);
            assert!(!caps.supports_f64);
            assert!(caps.hardware_accelerated, "Metal runs on the GPU");
            assert!(caps.max_fft_size >= 1 << 20, "max_fft_size from device");
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

    #[test]
    fn test_metal_fft_correctness_impulse() {
        if !is_available() {
            return;
        }

        let n = 64usize;
        let plan = MetalFftPlan::new(n, 1).expect("plan creation");

        let mut input: GpuBuffer<f32> = GpuBuffer::new(n, GpuBackend::Metal).expect("buffer");
        let mut output: GpuBuffer<f32> = GpuBuffer::new(n, GpuBackend::Metal).expect("buffer");

        // Impulse at index 0
        let mut data = vec![Complex::<f32>::zero(); n];
        data[0] = Complex::new(1.0f32, 0.0f32);
        input.upload(&data).expect("upload");

        plan.execute(&input, &mut output, GpuDirection::Forward)
            .expect("FFT execute");

        let mut result = vec![Complex::<f32>::zero(); n];
        output.download(&mut result).expect("download");

        for (i, c) in result.iter().enumerate() {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            assert!(
                (mag - 1.0).abs() < 1e-4,
                "bin {i}: expected magnitude 1.0, got {mag}"
            );
        }
    }

    #[test]
    fn test_metal_fft_round_trip() {
        if !is_available() {
            return;
        }

        let n = 128usize;
        let plan = MetalFftPlan::new(n, 1).expect("plan");

        let original: Vec<Complex<f32>> = (0..n)
            .map(|k| {
                let t = k as f32 / n as f32;
                Complex::new(t.sin(), 0.0f32)
            })
            .collect();

        let mut buf_in: GpuBuffer<f32> = GpuBuffer::new(n, GpuBackend::Metal).expect("buf");
        let mut buf_mid: GpuBuffer<f32> = GpuBuffer::new(n, GpuBackend::Metal).expect("buf");
        let mut buf_out: GpuBuffer<f32> = GpuBuffer::new(n, GpuBackend::Metal).expect("buf");

        buf_in.upload(&original).expect("upload");

        plan.execute(&buf_in, &mut buf_mid, GpuDirection::Forward)
            .expect("forward");

        plan.execute(&buf_mid, &mut buf_out, GpuDirection::Inverse)
            .expect("inverse");

        let mut recovered = vec![Complex::<f32>::zero(); n];
        buf_out.download(&mut recovered).expect("download");

        // `MetalFftPlan::execute` is unnormalised (FFTW convention), so the
        // forward→inverse round trip returns N·x; normalise here for comparison.
        let inv_n = 1.0_f32 / n as f32;
        for c in &mut recovered {
            c.re *= inv_n;
            c.im *= inv_n;
        }

        for i in 0..n {
            let err = ((recovered[i].re - original[i].re).powi(2)
                + (recovered[i].im - original[i].im).powi(2))
            .sqrt();
            assert!(
                err < 1e-4,
                "sample {i}: expected ({}, {}), got ({}, {}), error={err}",
                original[i].re,
                original[i].im,
                recovered[i].re,
                recovered[i].im
            );
        }
    }

    #[test]
    fn metal_roundtrip_sizes_6_to_16() {
        if !is_available() {
            return;
        }
        for n_exp in 6usize..=16 {
            let n = 1usize << n_exp;
            let plan = MetalFftPlan::new(n, 1).expect("MetalFftPlan::new failed");

            let original: Vec<Complex<f32>> = (0..n)
                .map(|i| {
                    let t = i as f32 / n as f32;
                    Complex::new(t.sin(), t.cos())
                })
                .collect();

            let mut buf_in: GpuBuffer<f32> = GpuBuffer::new(n, GpuBackend::Metal).expect("buf_in");
            let mut buf_mid: GpuBuffer<f32> =
                GpuBuffer::new(n, GpuBackend::Metal).expect("buf_mid");
            let mut buf_out: GpuBuffer<f32> =
                GpuBuffer::new(n, GpuBackend::Metal).expect("buf_out");

            buf_in.upload(&original).expect("upload");

            plan.execute(&buf_in, &mut buf_mid, GpuDirection::Forward)
                .expect("forward");

            plan.execute(&buf_mid, &mut buf_out, GpuDirection::Inverse)
                .expect("inverse");

            let mut recovered = vec![Complex::<f32>::zero(); n];
            buf_out.download(&mut recovered).expect("download");

            // `execute` is unnormalised; normalise the round trip by 1/N.
            let inv_n = 1.0_f32 / n as f32;
            for c in &mut recovered {
                c.re *= inv_n;
                c.im *= inv_n;
            }

            for (i, (orig, rec)) in original.iter().zip(recovered.iter()).enumerate() {
                let err = ((rec.re - orig.re).powi(2) + (rec.im - orig.im).powi(2)).sqrt();
                assert!(
                    err < 1e-3,
                    "size={n} (n_exp={n_exp}) sample {i}: expected ({}, {}), got ({}, {}), error={err}",
                    orig.re,
                    orig.im,
                    rec.re,
                    rec.im
                );
            }
        }
    }
}
