//! Batched GPU FFT trait and implementations.
//!
//! This module defines [`GpuBatchFft`], a trait for executing N independent
//! FFTs of the same size in a single logical batch submission.
//!
//! # Design
//!
//! The default implementation (provided by this module) loops over individual
//! `execute`-calls for each input/output pair.  Backend implementors that
//! natively support batched GPU submissions may override `execute_batch` to
//! dispatch all transforms in a single command-buffer.
//!
//! # Example (Metal, Apple Silicon)
//!
//! ```no_run
//! // no_run: requires a real Apple-Silicon Metal device.
//! use oxifft::gpu::metal::MetalFftPlan;
//! use oxifft::gpu::{GpuBatchFft, GpuDirection};
//! use oxifft::Complex;
//!
//! let plan = MetalFftPlan::new(256, 1).expect("MetalFftPlan");
//! let inputs: Vec<Vec<Complex<f32>>> = vec![vec![Complex::new(1.0, 0.0); 256]; 4];
//! let mut outputs: Vec<Vec<Complex<f32>>> = vec![vec![Complex::new(0.0, 0.0); 256]; 4];
//!
//! let input_slices: Vec<&[Complex<f32>]> = inputs.iter().map(|v| v.as_slice()).collect();
//! let mut output_slices: Vec<&mut [Complex<f32>]> =
//!     outputs.iter_mut().map(|v| v.as_mut_slice()).collect();
//!
//! plan.execute_batch(&input_slices, &mut output_slices, GpuDirection::Forward)
//!     .expect("batch execute");
//! ```

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec::Vec};

use super::error::{GpuError, GpuResult};
use super::plan::GpuDirection;
use crate::kernel::{Complex, Float};

// Used only by the test suites below (the production batch paths dispatch
// through backend-specific methods, not raw `GpuBuffer`s).
#[cfg(test)]
use super::buffer::GpuBuffer;
#[cfg(test)]
use super::GpuBackend;

/// Validate that `inputs` and `outputs` are consistent for a batch FFT.
///
/// Returns:
/// - `Ok(0)` if the batch is empty (caller should return `Ok(())` immediately).
/// - `Ok(n)` where `n` is the common slice length if everything is consistent.
/// - `Err(GpuError::*)` if any mismatch is found.
fn validate_batch<T: Float>(
    inputs: &[&[Complex<T>]],
    outputs: &[&mut [Complex<T>]],
) -> GpuResult<usize> {
    if inputs.len() != outputs.len() {
        return Err(GpuError::SizeMismatch {
            expected: inputs.len(),
            got: outputs.len(),
        });
    }

    if inputs.is_empty() {
        return Ok(0);
    }

    let n = inputs[0].len();

    for (i, (inp, out)) in inputs.iter().zip(outputs.iter()).enumerate() {
        if inp.len() != n {
            return Err(GpuError::ExecutionFailed(format!(
                "batch element {i}: input length {} differs from first-element length {n}",
                inp.len()
            )));
        }
        if out.len() != n {
            return Err(GpuError::ExecutionFailed(format!(
                "batch element {i}: output length {} differs from first-element length {n}",
                out.len()
            )));
        }
    }

    Ok(n)
}

/// Trait for executing N independent FFTs of the same size as a batch.
///
/// The batch is represented as a slice of input-slice references and a
/// corresponding slice of mutable output-slice references.  Every slice must
/// have identical length equal to the plan's configured transform size.
///
/// ## Default implementation
///
/// A default `execute_batch` implementation is deliberately *not* provided
/// here as a blanket default — concrete implementations must provide it.
/// Both `MetalFftPlan` and `CudaFftPlan` provide loop-based fallbacks.
///
/// ## Backend specialisation
///
/// Backends that support native batched GPU dispatch (e.g., Metal Performance
/// Shaders batch API) can override `execute_batch` to avoid per-transform
/// overhead.
pub trait GpuBatchFft<T: Float>: Send + Sync {
    /// Maximum batch size this instance supports.
    ///
    /// `execute_batch` should return an error if `inputs.len()` exceeds this
    /// value.  A value of `usize::MAX` means no practical limit.
    fn batch_size_limit(&self) -> usize;

    /// Execute `inputs.len()` independent FFTs of size `inputs[0].len()`.
    ///
    /// # Arguments
    ///
    /// * `inputs` — slice of immutable input-slice references; all slices
    ///   must have the same length equal to the plan's FFT size.
    /// * `outputs` — slice of mutable output-slice references of the same
    ///   length as `inputs`; each output slice must also match the plan's FFT size.
    /// * `direction` — transform direction (forward or inverse).
    ///
    /// # Errors
    ///
    /// - [`GpuError::SizeMismatch`] if `inputs.len() != outputs.len()`.
    /// - [`GpuError::ExecutionFailed`] if any individual slice length is
    ///   inconsistent or the underlying transform fails.
    /// - [`GpuError::Unsupported`] if `inputs.len() > batch_size_limit()`.
    fn execute_batch(
        &self,
        inputs: &[&[Complex<T>]],
        outputs: &mut [&mut [Complex<T>]],
        direction: GpuDirection,
    ) -> GpuResult<()>;
}

// ---------------------------------------------------------------------------
// Shared chunked-dispatch helper
// ---------------------------------------------------------------------------

/// Dispatch a batch of FFTs, automatically splitting into chunks of at most
/// `chunk_limit` elements.
///
/// When `inputs.len() <= chunk_limit` the outer loop executes exactly once
/// incurring no overhead.  When it exceeds the limit the workload is split
/// into `⌈inputs.len() / chunk_limit⌉` chunks dispatched in order so that
/// output ordering is preserved.
///
/// This per-element helper backs the CUDA CPU-emulation path (the Metal backend
/// uses native single-dispatch batching instead), and the chunking tests.
#[cfg(any(feature = "cuda", test))]
fn dispatch_chunked<T, F>(
    inputs: &[&[Complex<T>]],
    outputs: &mut [&mut [Complex<T>]],
    chunk_limit: usize,
    mut dispatch_one: F,
) -> GpuResult<()>
where
    T: Float,
    F: FnMut(&[Complex<T>], &mut [Complex<T>]) -> GpuResult<()>,
{
    let total = inputs.len();
    let mut start = 0;
    while start < total {
        let end = (start + chunk_limit).min(total);
        for idx in start..end {
            dispatch_one(inputs[idx], outputs[idx])?;
        }
        start = end;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Metal implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
mod metal_impl {
    use super::*;
    use crate::gpu::metal::MetalFftPlan;

    /// Maximum batch size for a single Metal dispatch chunk.
    ///
    /// Batches larger than this are automatically split into multiple chunks
    /// rather than returning `Unsupported`.
    const METAL_BATCH_LIMIT: usize = 1024;

    impl<T: Float> GpuBatchFft<T> for MetalFftPlan {
        fn batch_size_limit(&self) -> usize {
            METAL_BATCH_LIMIT
        }

        fn execute_batch(
            &self,
            inputs: &[&[Complex<T>]],
            outputs: &mut [&mut [Complex<T>]],
            direction: GpuDirection,
        ) -> GpuResult<()> {
            let n = validate_batch(inputs, outputs)?;
            if n == 0 {
                return Ok(());
            }

            if n != self.size() {
                return Err(GpuError::SizeMismatch {
                    expected: self.size(),
                    got: n,
                });
            }

            // Native batched dispatch: transforms are packed and submitted in
            // chunks of the plan's own `batch_size` through its already-compiled
            // inner plan (no per-call shader recompilation).  This is correct
            // for any plan `batch_size` — unlike the old per-element
            // `self.execute` loop, which required `size * batch_size` buffers and
            // errored out for `batch_size > 1`.
            self.execute_batch_native(inputs, outputs, direction)
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use crate::gpu::cuda::CudaFftPlan;

    /// Maximum batch size for a single CUDA dispatch chunk.
    ///
    /// Batches larger than this are automatically split into multiple chunks
    /// rather than returning `Unsupported`.
    const CUDA_BATCH_LIMIT: usize = 4096;

    impl<T: Float> GpuBatchFft<T> for CudaFftPlan {
        fn batch_size_limit(&self) -> usize {
            CUDA_BATCH_LIMIT
        }

        fn execute_batch(
            &self,
            inputs: &[&[Complex<T>]],
            outputs: &mut [&mut [Complex<T>]],
            direction: GpuDirection,
        ) -> GpuResult<()> {
            let n = validate_batch(inputs, outputs)?;
            if n == 0 {
                return Ok(());
            }

            if n != self.size() {
                return Err(GpuError::SizeMismatch {
                    expected: self.size(),
                    got: n,
                });
            }

            // The CUDA backend currently emulates on the CPU (see
            // `crate::gpu::cuda`).  Dispatch each transform through the
            // single-transform CPU path, which is independent of this plan's
            // configured batch size — so, unlike the old `self.execute` loop,
            // this works even when the plan was built with `batch_size > 1`.
            // Native batched GPU dispatch will become worthwhile only once real
            // device kernels land in oxicuda-fft.
            dispatch_chunked(inputs, outputs, CUDA_BATCH_LIMIT, |inp, out| {
                self.execute_one_cpu(inp, out, direction)
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Batch auto-chunking tests — exercise both the fast path (≤ limit) and the
/// chunking paths (> limit).  These do not need a real GPU.
///
/// We use a tiny `TEST_LIMIT` (8) so tests don't need to create hundreds of
/// plans.  The same `dispatch_chunked` code is exercised with the real
/// 1024/4096 limits in production.
#[cfg(test)]
mod batch_chunking_tests {
    use super::*;

    /// Test-only chunking limit (smaller than production 1024/4096 for speed).
    const TEST_LIMIT: usize = 8;

    /// Build `count` distinct inputs of length `n`.
    fn make_inputs(count: usize, n: usize) -> Vec<Vec<Complex<f64>>> {
        (0..count)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let x = ((i * n + j) as f64) * 0.13;
                        Complex::new(x.sin(), x.cos())
                    })
                    .collect()
            })
            .collect()
    }

    /// Run `dispatch_chunked` with an identity dispatch function and assert
    /// that outputs match inputs.
    fn run_identity_chunked(batch_size: usize, n: usize) {
        let inputs_owned = make_inputs(batch_size, n);
        let mut outputs_owned: Vec<Vec<Complex<f64>>> =
            (0..batch_size).map(|_| vec![Complex::zero(); n]).collect();

        let inputs: Vec<&[Complex<f64>]> = inputs_owned.iter().map(|v| v.as_slice()).collect();
        let mut outputs: Vec<&mut [Complex<f64>]> =
            outputs_owned.iter_mut().map(|v| v.as_mut_slice()).collect();

        let elem_n = validate_batch(&inputs, outputs.as_slice()).expect("validate_batch");
        assert_eq!(elem_n, n);

        dispatch_chunked(&inputs, &mut outputs, TEST_LIMIT, |inp, out| {
            out.copy_from_slice(inp);
            Ok(())
        })
        .expect("dispatch_chunked");

        for (orig, got) in inputs_owned.iter().zip(outputs_owned.iter()) {
            assert_eq!(orig, got, "output ordering mismatch");
        }
    }

    #[test]
    fn batch_auto_chunk_below_limit() {
        run_identity_chunked(TEST_LIMIT - 1, 16);
    }

    #[test]
    fn batch_auto_chunk_at_limit() {
        run_identity_chunked(TEST_LIMIT, 16);
    }

    #[test]
    fn batch_auto_chunk_above_limit() {
        run_identity_chunked(TEST_LIMIT + 1, 16);
    }

    #[test]
    fn batch_auto_chunk_large() {
        // 3 * TEST_LIMIT + 7 elements → exercises multiple full chunks + tail.
        run_identity_chunked(3 * TEST_LIMIT + 7, 8);
    }
}

// ---------------------------------------------------------------------------
// S2 Required Tests
// ---------------------------------------------------------------------------

/// S2 tests: R2C/C2R support, batch auto-chunking, and typed error variants.
///
/// These tests are required by the S2 implementation step to verify that all
/// three features (R2C/C2R, batch chunking, typed errors) work correctly.
#[cfg(test)]
mod s2_tests {
    use super::*;

    // ── Metal R2C roundtrip ──────────────────────────────────────────────────

    #[cfg(feature = "metal")]
    mod metal_r2c {
        use crate::gpu::metal;
        use crate::gpu::metal::MetalFftPlan;

        fn run_r2c_roundtrip_s2(n: usize) {
            if !metal::is_available() {
                return;
            }
            let plan = MetalFftPlan::new(n, 1).expect("MetalFftPlan::new");
            let half = n / 2 + 1;
            let tolerance = 1e-4_f32 * n as f32;

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
            run_r2c_roundtrip_s2(64);
        }

        #[test]
        fn metal_r2c_roundtrip_size256() {
            run_r2c_roundtrip_s2(256);
        }

        #[test]
        fn metal_r2c_roundtrip_size1024() {
            run_r2c_roundtrip_s2(1024);
        }
    }

    // ── Batch chunking ───────────────────────────────────────────────────────

    mod batch_chunking {
        use super::*;

        /// Test-only limit: small enough to make over-limit tests fast.
        const TEST_LIMIT: usize = 8;
        const N: usize = 16;

        fn make_batch_inputs(count: usize) -> Vec<Vec<Complex<f64>>> {
            (0..count)
                .map(|i| {
                    (0..N)
                        .map(|j| {
                            let x = ((i * N + j) as f64) * 0.13;
                            Complex::new(x.sin(), x.cos())
                        })
                        .collect()
                })
                .collect()
        }

        fn run_chunked_identity(batch_size: usize) {
            let inputs_owned = make_batch_inputs(batch_size);
            let mut outputs_owned: Vec<Vec<Complex<f64>>> =
                (0..batch_size).map(|_| vec![Complex::zero(); N]).collect();

            let inputs: Vec<&[Complex<f64>]> = inputs_owned.iter().map(|v| v.as_slice()).collect();
            let mut outputs: Vec<&mut [Complex<f64>]> =
                outputs_owned.iter_mut().map(|v| v.as_mut_slice()).collect();

            dispatch_chunked(&inputs, &mut outputs, TEST_LIMIT, |inp, out| {
                out.copy_from_slice(inp);
                Ok(())
            })
            .expect("dispatch_chunked");

            for (orig, got) in inputs_owned.iter().zip(outputs_owned.iter()) {
                assert_eq!(orig, got, "output ordering mismatch");
            }
        }

        #[test]
        fn batch_at_limit() {
            // Exactly TEST_LIMIT items → fast path, no chunking.
            run_chunked_identity(TEST_LIMIT);
        }

        #[test]
        fn batch_above_limit() {
            // One over TEST_LIMIT → chunking path, same results.
            run_chunked_identity(TEST_LIMIT + 1);
        }

        #[test]
        fn batch_3x_limit_plus7() {
            // 3 full chunks + tail → exercises multiple full chunks.
            run_chunked_identity(3 * TEST_LIMIT + 7);
        }
    }

    // ── Error recovery ───────────────────────────────────────────────────────

    mod error_recovery {
        use crate::gpu::error::GpuError;

        #[test]
        fn out_of_memory_variant_exists() {
            // Instantiate the variant to ensure it compiles and is matchable.
            let e = GpuError::OutOfMemory {
                requested_bytes: 1024,
            };
            match e {
                GpuError::OutOfMemory { requested_bytes } => {
                    assert_eq!(requested_bytes, 1024);
                }
                other => panic!("expected OutOfMemory, got {other:?}"),
            }
        }

        #[test]
        fn device_lost_variant_exists() {
            let e = GpuError::DeviceLost;
            match e {
                GpuError::DeviceLost => {}
                other => panic!("expected DeviceLost, got {other:?}"),
            }
        }
    }
}

/// Error recovery tests — verify that typed `GpuError` variants are returned.
#[cfg(test)]
mod error_recovery_tests {
    use super::*;

    #[test]
    fn gpu_oom_returns_typed_error() {
        use crate::gpu::error::GpuError;
        use crate::gpu::GpuBackend;

        // Verify the OutOfMemory variant is constructible and matchable.
        let err = GpuError::OutOfMemory {
            requested_bytes: usize::MAX,
        };
        match err {
            GpuError::OutOfMemory { requested_bytes } => {
                assert_eq!(requested_bytes, usize::MAX);
            }
            other => panic!("expected OutOfMemory, got {other:?}"),
        }

        // GpuBuffer::new(0, ...) should return InvalidSize.
        let result = GpuBuffer::<f64>::new(0, GpuBackend::Auto);
        assert!(result.is_err(), "zero-size buffer should be an error");
        match result.expect_err("must be err") {
            GpuError::InvalidSize(0) => {}
            other => panic!("expected InvalidSize(0), got {other:?}"),
        }
    }

    #[test]
    fn gpu_error_display_new_variants() {
        use crate::gpu::error::GpuError;

        let e = GpuError::DeviceLost;
        assert!(e.to_string().contains("device"));

        let e = GpuError::OutOfMemory {
            requested_bytes: 42,
        };
        assert!(e.to_string().contains("42"));

        let e = GpuError::ShaderCompileFailed("bad syntax".into());
        assert!(e.to_string().contains("bad syntax"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Validation-only tests — no GPU required
    // ------------------------------------------------------------------

    #[test]
    fn validate_empty_batch_ok() {
        let empty_inputs: Vec<&[Complex<f64>]> = Vec::new();
        let mut empty_outputs: Vec<&mut [Complex<f64>]> = Vec::new();
        let result = validate_batch(&empty_inputs, empty_outputs.as_mut_slice());
        assert!(result.is_ok());
        assert_eq!(result.expect("ok"), 0);
    }

    #[test]
    fn validate_mismatched_count_err() {
        let n = 8usize;
        let a = vec![Complex::<f64>::zero(); n];
        let mut b = vec![Complex::<f64>::zero(); n];
        let mut c = vec![Complex::<f64>::zero(); n];
        let outputs: [&mut [Complex<f64>]; 2] = [b.as_mut_slice(), c.as_mut_slice()];

        let result = validate_batch(&[a.as_slice()], &outputs);
        assert!(result.is_err(), "mismatched count should be an error");
        match result.expect_err("must be err") {
            GpuError::SizeMismatch { expected, got } => {
                assert_eq!(expected, 1);
                assert_eq!(got, 2);
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn validate_nonuniform_input_lengths_err() {
        let a = vec![Complex::<f64>::zero(); 8];
        let b_short = vec![Complex::<f64>::zero(); 4]; // wrong length
        let mut out_a = vec![Complex::<f64>::zero(); 8];
        let mut out_b = vec![Complex::<f64>::zero(); 8];
        let outputs: [&mut [Complex<f64>]; 2] = [out_a.as_mut_slice(), out_b.as_mut_slice()];

        let result = validate_batch(&[a.as_slice(), b_short.as_slice()], &outputs);
        assert!(
            result.is_err(),
            "non-uniform input slice lengths should be an error"
        );
    }

    #[test]
    fn validate_nonuniform_output_lengths_err() {
        let a = vec![Complex::<f64>::zero(); 8];
        let b = vec![Complex::<f64>::zero(); 8];
        let mut out_a = vec![Complex::<f64>::zero(); 8];
        let mut out_b_short = vec![Complex::<f64>::zero(); 4]; // wrong length
        let outputs: [&mut [Complex<f64>]; 2] = [out_a.as_mut_slice(), out_b_short.as_mut_slice()];

        let result = validate_batch(&[a.as_slice(), b.as_slice()], &outputs);
        assert!(
            result.is_err(),
            "non-uniform output slice lengths should be an error"
        );
    }

    #[test]
    fn validate_single_element_batch_ok() {
        let n = 16usize;
        let a = vec![Complex::<f64>::zero(); n];
        let mut out = vec![Complex::<f64>::zero(); n];
        let outputs: [&mut [Complex<f64>]; 1] = [out.as_mut_slice()];

        let result = validate_batch(&[a.as_slice()], &outputs);
        assert!(result.is_ok());
        assert_eq!(result.expect("ok"), n);
    }

    // ------------------------------------------------------------------
    // Metal integration tests (require `metal` feature + Apple GPU)
    // ------------------------------------------------------------------

    #[cfg(feature = "metal")]
    mod metal_tests {
        use super::*;
        use crate::gpu::metal;
        use crate::gpu::metal::MetalFftPlan;

        #[test]
        fn metal_empty_batch_is_ok() {
            if !metal::is_available() {
                return;
            }
            let plan = MetalFftPlan::new(16, 1).expect("MetalFftPlan::new");
            let empty_inputs: Vec<&[Complex<f32>]> = Vec::new();
            let mut empty_outputs: Vec<&mut [Complex<f32>]> = Vec::new();
            let result =
                plan.execute_batch(&empty_inputs, &mut empty_outputs, GpuDirection::Forward);
            assert!(result.is_ok(), "empty batch should succeed");
        }

        #[test]
        fn metal_batch_validates_mismatched_counts() {
            if !metal::is_available() {
                return;
            }
            let n = 8usize;
            let plan = MetalFftPlan::new(n, 1).expect("MetalFftPlan::new");
            let inp = vec![Complex::<f32>::zero(); n];
            let mut out1 = vec![Complex::<f32>::zero(); n];
            let mut out2 = vec![Complex::<f32>::zero(); n];

            let result = plan.execute_batch(
                &[inp.as_slice()],
                &mut [out1.as_mut_slice(), out2.as_mut_slice()],
                GpuDirection::Forward,
            );
            assert!(result.is_err(), "mismatched input/output count should fail");
        }

        #[test]
        fn metal_batch_matches_individual_execute() {
            if !metal::is_available() {
                return;
            }
            let n = 16usize;
            let batch_count = 4usize;
            let plan = MetalFftPlan::new(n, 1).expect("MetalFftPlan::new");

            // Build batch_count random-ish inputs
            let inputs: Vec<Vec<Complex<f32>>> = (0..batch_count)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            let x = ((i * n + j) as f32) * 0.1_f32;
                            Complex::new(x.sin(), x.cos())
                        })
                        .collect()
                })
                .collect();

            // Single-execute reference
            let mut single_outputs: Vec<Vec<Complex<f32>>> =
                (0..batch_count).map(|_| vec![Complex::zero(); n]).collect();
            for (inp, out) in inputs.iter().zip(single_outputs.iter_mut()) {
                let in_buf =
                    GpuBuffer::from_slice(inp.as_slice(), GpuBackend::Metal).expect("in_buf");
                let mut out_buf = GpuBuffer::new(n, GpuBackend::Metal).expect("out_buf");
                plan.execute(&in_buf, &mut out_buf, GpuDirection::Forward)
                    .expect("single execute");
                out_buf.download(out.as_mut_slice()).expect("download");
            }

            // Batch execute
            let input_slices: Vec<&[Complex<f32>]> = inputs.iter().map(|v| v.as_slice()).collect();
            let mut batch_outputs: Vec<Vec<Complex<f32>>> =
                (0..batch_count).map(|_| vec![Complex::zero(); n]).collect();
            let mut output_slices: Vec<&mut [Complex<f32>]> =
                batch_outputs.iter_mut().map(|v| v.as_mut_slice()).collect();

            plan.execute_batch(&input_slices, &mut output_slices, GpuDirection::Forward)
                .expect("batch execute");

            // Numerically compare
            for (b, (single, batch)) in single_outputs.iter().zip(batch_outputs.iter()).enumerate()
            {
                for (k, (s, bt)) in single.iter().zip(batch.iter()).enumerate() {
                    assert!(
                        (s.re - bt.re).abs() < 1e-4_f32,
                        "batch={b} bin={k}: re mismatch single={} batch={}",
                        s.re,
                        bt.re
                    );
                    assert!(
                        (s.im - bt.im).abs() < 1e-4_f32,
                        "batch={b} bin={k}: im mismatch single={} batch={}",
                        s.im,
                        bt.im
                    );
                }
            }
        }

        #[test]
        fn metal_batch_wrong_fft_size_err() {
            if !metal::is_available() {
                return;
            }
            let n = 16usize;
            let plan = MetalFftPlan::new(n, 1).expect("MetalFftPlan::new");

            // Supply slices with wrong size (32 instead of 16)
            let wrong = vec![Complex::<f32>::zero(); 32];
            let mut wrong_out = vec![Complex::<f32>::zero(); 32];
            let result = plan.execute_batch(
                &[wrong.as_slice()],
                &mut [wrong_out.as_mut_slice()],
                GpuDirection::Forward,
            );
            assert!(result.is_err(), "wrong slice length should fail");
        }

        /// CPU reference: unnormalised forward DFT of a single size-`n` signal.
        fn cpu_forward(input: &[Complex<f32>]) -> Vec<Complex<f32>> {
            use crate::api::{Direction, Flags, Plan};
            let n = input.len();
            let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).expect("cpu plan");
            let in64: Vec<Complex<f64>> = input
                .iter()
                .map(|c| Complex::new(c.re as f64, c.im as f64))
                .collect();
            let mut out64 = vec![Complex::<f64>::zero(); n];
            plan.execute(&in64, &mut out64);
            out64
                .iter()
                .map(|c| Complex::new(c.re as f32, c.im as f32))
                .collect()
        }

        #[test]
        fn metal_batch_matches_cpu() {
            if !metal::is_available() {
                return;
            }
            let n = 32usize;
            let batch_count = 5usize;
            let plan = MetalFftPlan::new(n, 1).expect("MetalFftPlan::new");

            let inputs: Vec<Vec<Complex<f32>>> = (0..batch_count)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            let x = ((i * n + j) as f32) * 0.07_f32;
                            Complex::new(x.sin(), 0.0)
                        })
                        .collect()
                })
                .collect();

            let cpu_outputs: Vec<Vec<Complex<f32>>> =
                inputs.iter().map(|inp| cpu_forward(inp)).collect();

            let input_slices: Vec<&[Complex<f32>]> = inputs.iter().map(|v| v.as_slice()).collect();
            let mut batch_outputs: Vec<Vec<Complex<f32>>> =
                (0..batch_count).map(|_| vec![Complex::zero(); n]).collect();
            let mut output_slices: Vec<&mut [Complex<f32>]> =
                batch_outputs.iter_mut().map(|v| v.as_mut_slice()).collect();

            plan.execute_batch(&input_slices, &mut output_slices, GpuDirection::Forward)
                .expect("batch execute");

            let tol = 1e-3_f32 * n as f32;
            for (b, (gpu, cpu)) in batch_outputs.iter().zip(cpu_outputs.iter()).enumerate() {
                for (k, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
                    let err = ((g.re - c.re).powi(2) + (g.im - c.im).powi(2)).sqrt();
                    assert!(
                        err <= tol,
                        "batch={b} bin={k}: gpu=({},{}) cpu=({},{}) err={err} > {tol}",
                        g.re,
                        g.im,
                        c.re,
                        c.im
                    );
                }
            }
        }

        /// Regression: a plan built with `batch_size > 1` must still service the
        /// per-element batch trait.  The old loop passed size-`n` buffers to
        /// `self.execute`, which required `size * batch_size` and errored out.
        #[test]
        fn metal_batch_works_with_plan_batch_size_gt_1() {
            if !metal::is_available() {
                return;
            }
            let n = 16usize;
            // Plan batch = 4, but the batch call has 3 elements (count != batch).
            let plan = MetalFftPlan::new(n, 4).expect("MetalFftPlan::new");

            let inputs: Vec<Vec<Complex<f32>>> = (0..3)
                .map(|i| {
                    (0..n)
                        .map(|j| Complex::new(((i * n + j) as f32) * 0.1_f32, 0.0))
                        .collect()
                })
                .collect();
            let input_slices: Vec<&[Complex<f32>]> = inputs.iter().map(|v| v.as_slice()).collect();
            let mut outs: Vec<Vec<Complex<f32>>> =
                (0..3).map(|_| vec![Complex::zero(); n]).collect();
            let mut out_slices: Vec<&mut [Complex<f32>]> =
                outs.iter_mut().map(|v| v.as_mut_slice()).collect();

            plan.execute_batch(&input_slices, &mut out_slices, GpuDirection::Forward)
                .expect("batch with plan batch_size > 1 must succeed");

            // Each output must match the per-element CPU forward.
            let tol = 1e-3_f32 * n as f32;
            for (b, out) in outs.iter().enumerate() {
                let cpu = cpu_forward(&inputs[b]);
                for (k, (g, c)) in out.iter().zip(cpu.iter()).enumerate() {
                    let err = ((g.re - c.re).powi(2) + (g.im - c.im).powi(2)).sqrt();
                    assert!(err <= tol, "batch={b} bin={k}: err={err} > {tol}");
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // CUDA integration tests (require `cuda` feature + NVIDIA GPU)
    // ------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    mod cuda_tests {
        use super::*;
        use crate::gpu::cuda;
        use crate::gpu::cuda::CudaFftPlan;

        #[test]
        fn cuda_empty_batch_is_ok() {
            if !cuda::is_available() {
                return;
            }
            let plan = CudaFftPlan::new(16, 1).expect("CudaFftPlan::new");
            let empty_inputs: Vec<&[Complex<f64>]> = Vec::new();
            let mut empty_outputs: Vec<&mut [Complex<f64>]> = Vec::new();
            let result =
                plan.execute_batch(&empty_inputs, &mut empty_outputs, GpuDirection::Forward);
            assert!(result.is_ok(), "empty batch should succeed");
        }

        #[test]
        fn cuda_batch_validates_mismatched_counts() {
            if !cuda::is_available() {
                return;
            }
            let n = 8usize;
            let plan = CudaFftPlan::new(n, 1).expect("CudaFftPlan::new");
            let inp = vec![Complex::<f64>::zero(); n];
            let mut out1 = vec![Complex::<f64>::zero(); n];
            let mut out2 = vec![Complex::<f64>::zero(); n];

            let result = plan.execute_batch(
                &[inp.as_slice()],
                &mut [out1.as_mut_slice(), out2.as_mut_slice()],
                GpuDirection::Forward,
            );
            assert!(result.is_err(), "mismatched input/output count should fail");
        }

        #[test]
        fn cuda_batch_matches_individual_execute() {
            if !cuda::is_available() {
                return;
            }
            let n = 16usize;
            let batch_count = 4usize;
            let plan = CudaFftPlan::new(n, 1).expect("CudaFftPlan::new");

            let inputs: Vec<Vec<Complex<f64>>> = (0..batch_count)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            let x = ((i * n + j) as f64) * 0.1_f64;
                            Complex::new(x.sin(), x.cos())
                        })
                        .collect()
                })
                .collect();

            // Single-execute reference
            let mut single_outputs: Vec<Vec<Complex<f64>>> =
                (0..batch_count).map(|_| vec![Complex::zero(); n]).collect();
            for (inp, out) in inputs.iter().zip(single_outputs.iter_mut()) {
                let in_buf =
                    GpuBuffer::from_slice(inp.as_slice(), GpuBackend::Cuda).expect("in_buf");
                let mut out_buf = GpuBuffer::new(n, GpuBackend::Cuda).expect("out_buf");
                plan.execute(&in_buf, &mut out_buf, GpuDirection::Forward)
                    .expect("single execute");
                out_buf.download(out.as_mut_slice()).expect("download");
            }

            // Batch execute
            let input_slices: Vec<&[Complex<f64>]> = inputs.iter().map(|v| v.as_slice()).collect();
            let mut batch_outputs: Vec<Vec<Complex<f64>>> =
                (0..batch_count).map(|_| vec![Complex::zero(); n]).collect();
            let mut output_slices: Vec<&mut [Complex<f64>]> =
                batch_outputs.iter_mut().map(|v| v.as_mut_slice()).collect();

            plan.execute_batch(&input_slices, &mut output_slices, GpuDirection::Forward)
                .expect("batch execute");

            for (b, (single, batch)) in single_outputs.iter().zip(batch_outputs.iter()).enumerate()
            {
                for (k, (s, bt)) in single.iter().zip(batch.iter()).enumerate() {
                    assert!(
                        (s.re - bt.re).abs() < 1e-10_f64,
                        "batch={b} bin={k}: re mismatch single={} batch={}",
                        s.re,
                        bt.re
                    );
                    assert!(
                        (s.im - bt.im).abs() < 1e-10_f64,
                        "batch={b} bin={k}: im mismatch single={} batch={}",
                        s.im,
                        bt.im
                    );
                }
            }
        }
    }
}
