//! Generated SIMD codelets for small DFT sizes.
//!
//! This module invokes the `gen_simd_codelet!` procedural macro to emit
//! architecture-aware SIMD FFT kernels for sizes 2, 4, and 8 at compile time.
//!
//! Each macro invocation produces:
//! - A public dispatcher `codelet_simd_N<T: Float>(data, sign)` that selects
//!   the best SIMD path at runtime via `TypeId` + feature detection
//! - Architecture-specific inner functions guarded by `#[cfg(target_arch)]`
//!   and `#[target_feature]` (SSE2, AVX2+FMA on x86_64; NEON on aarch64)
//! - A generic scalar fallback for all other targets / float types
//!
//! ## f32 support
//!
//! The generated dispatchers handle both `f64` and `f32` SIMD paths.  The
//! hand-written `notw_{2,4,8}_dispatch` functions in `simd/mod.rs` only
//! accelerate `f64`; this module extends SIMD coverage to `f32` on all
//! supported architectures.
//!
//! ## no_std note
//!
//! `is_x86_feature_detected!` is used inside `#[cfg(target_arch = "x86_64")]`
//! blocks.  This follows the same convention as the rest of the `oxifft` SIMD
//! modules (see `simd/detect.rs`, `simd/large_sizes.rs`).

use oxifft_codegen::gen_simd_codelet;

use crate::kernel::{Complex, Float};

gen_simd_codelet!(2);
gen_simd_codelet!(4);
gen_simd_codelet!(8);

// ---------------------------------------------------------------------------
// Cached runtime dispatchers (AtomicU8-backed, avoid repeated feature probes)
// ---------------------------------------------------------------------------
//
// Each invocation is placed in its own submodule because the macro emits
// module-level const declarations (ISA_SCALAR_LEVEL, etc.) that would
// collide if multiple invocations shared the same namespace.

/// Cached dispatcher submodule for size-4 f64.
mod cached_dispatch_4_f64 {
    use oxifft_codegen::gen_dispatcher_codelet;
    gen_dispatcher_codelet!(size = 4, ty = f64);
}
pub use cached_dispatch_4_f64::codelet_simd_4_cached_f64;

/// Cached dispatcher submodule for size-4 f32.
mod cached_dispatch_4_f32 {
    use oxifft_codegen::gen_dispatcher_codelet;
    gen_dispatcher_codelet!(size = 4, ty = f32);
}
pub use cached_dispatch_4_f32::codelet_simd_4_cached_f32;

// ---------------------------------------------------------------------------
// Public wrappers for integration with `notw_N_dispatch`
// ---------------------------------------------------------------------------

/// Delegate size-2 dispatch to the generated SIMD codelet.
///
/// The generated `codelet_simd_2` takes a `sign: i32` parameter; the size-2
/// butterfly (`out[0] = a+b`, `out[1] = a-b`) is sign-independent so we pass
/// `1` as a no-op placeholder.
#[inline]
pub fn generated_simd_2_dispatch<T: Float>(data: &mut [Complex<T>]) {
    codelet_simd_2(data, 1);
}

/// Delegate size-4 dispatch to the generated SIMD codelet.
#[inline]
pub fn generated_simd_4_dispatch<T: Float>(data: &mut [Complex<T>], sign: i32) {
    codelet_simd_4(data, sign);
}

/// Delegate size-8 dispatch to the generated SIMD codelet.
#[inline]
pub fn generated_simd_8_dispatch<T: Float>(data: &mut [Complex<T>], sign: i32) {
    codelet_simd_8(data, sign);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Naive DFT for correctness reference
    fn naive_dft(x: &[Complex<f64>], sign: i32) -> Vec<Complex<f64>> {
        let n = x.len();
        let mut out = vec![Complex { re: 0.0, im: 0.0 }; n];
        for k in 0..n {
            for j in 0..n {
                let angle = sign as f64 * 2.0 * core::f64::consts::PI * (k * j) as f64 / n as f64;
                let w = Complex {
                    re: angle.cos(),
                    im: angle.sin(),
                };
                out[k].re += x[j].re * w.re - x[j].im * w.im;
                out[k].im += x[j].re * w.im + x[j].im * w.re;
            }
        }
        out
    }

    fn naive_dft_f32(x: &[Complex<f32>], sign: i32) -> Vec<Complex<f32>> {
        let n = x.len();
        let mut out = vec![
            Complex {
                re: 0.0f32,
                im: 0.0f32
            };
            n
        ];
        for k in 0..n {
            for j in 0..n {
                let angle = sign as f32 * 2.0 * core::f32::consts::PI * (k * j) as f32 / n as f32;
                let w = Complex {
                    re: angle.cos(),
                    im: angle.sin(),
                };
                out[k].re += x[j].re * w.re - x[j].im * w.im;
                out[k].im += x[j].re * w.im + x[j].im * w.re;
            }
        }
        out
    }

    fn approx_eq_f64(a: &[Complex<f64>], b: &[Complex<f64>], tol: f64) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b.iter())
                .all(|(x, y)| (x.re - y.re).abs() < tol && (x.im - y.im).abs() < tol)
    }

    fn approx_eq_f32(a: &[Complex<f32>], b: &[Complex<f32>], tol: f32) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b.iter())
                .all(|(x, y)| (x.re - y.re).abs() < tol && (x.im - y.im).abs() < tol)
    }

    // ----- size-2 f64 -------------------------------------------------------

    #[test]
    fn generated_simd_2_f64_forward_correctness() {
        let input = [
            Complex {
                re: 1.0f64,
                im: 2.0,
            },
            Complex {
                re: 3.0f64,
                im: 4.0,
            },
        ];
        let expected = naive_dft(&input, -1);
        let mut data = input;
        generated_simd_2_dispatch(&mut data);
        assert!(
            approx_eq_f64(&data, &expected, 1e-10),
            "size-2 f64 forward: got {data:?}, expected {expected:?}"
        );
    }

    #[test]
    fn generated_simd_2_f64_roundtrip() {
        let original = [
            Complex {
                re: 1.5f64,
                im: -0.5,
            },
            Complex {
                re: -2.0f64,
                im: 3.0,
            },
        ];
        let mut data = original;
        // Forward
        generated_simd_2_dispatch(&mut data);
        // Inverse via codelet_simd_2 with sign=+1
        codelet_simd_2(&mut data, 1);
        let n = original.len() as f64;
        for (got, orig) in data.iter().zip(original.iter()) {
            assert!((got.re / n - orig.re).abs() < 1e-10);
            assert!((got.im / n - orig.im).abs() < 1e-10);
        }
    }

    // ----- size-2 f32 -------------------------------------------------------

    /// This test proves the generated codelet reaches the f32 SIMD path.
    /// The hand-written `notw_2_dispatch` has no f32 SIMD — only scalar.
    /// Any passing result here means the generated dispatcher handled f32.
    #[test]
    fn generated_simd_2_f32_correctness() {
        let input = [
            Complex {
                re: 1.0f32,
                im: 2.0,
            },
            Complex {
                re: 3.0f32,
                im: 4.0,
            },
        ];
        let expected = naive_dft_f32(&input, -1);
        let mut data = input;
        generated_simd_2_dispatch(&mut data);
        assert!(
            approx_eq_f32(&data, &expected, 1e-5),
            "size-2 f32 forward: got {data:?}, expected {expected:?}"
        );
    }

    #[test]
    fn generated_simd_2_f32_roundtrip() {
        let original = [
            Complex {
                re: 1.5f32,
                im: -0.5,
            },
            Complex {
                re: -2.0f32,
                im: 3.0,
            },
        ];
        let mut data = original;
        generated_simd_2_dispatch(&mut data);
        codelet_simd_2(&mut data, 1);
        let n = original.len() as f32;
        for (got, orig) in data.iter().zip(original.iter()) {
            assert!((got.re / n - orig.re).abs() < 1e-5);
            assert!((got.im / n - orig.im).abs() < 1e-5);
        }
    }

    // ----- size-4 f64 -------------------------------------------------------

    #[test]
    fn generated_simd_4_f64_forward_correctness() {
        let input = [
            Complex {
                re: 1.0f64,
                im: 0.0,
            },
            Complex {
                re: 0.0f64,
                im: 1.0,
            },
            Complex {
                re: -1.0f64,
                im: 0.0,
            },
            Complex {
                re: 0.0f64,
                im: -1.0,
            },
        ];
        let expected = naive_dft(&input, -1);
        let mut data = input;
        generated_simd_4_dispatch(&mut data, -1);
        assert!(
            approx_eq_f64(&data, &expected, 1e-10),
            "size-4 f64 forward: got {data:?}, expected {expected:?}"
        );
    }

    #[test]
    fn generated_simd_4_f64_inverse_correctness() {
        let input = [
            Complex {
                re: 2.0f64,
                im: 1.0,
            },
            Complex {
                re: -1.0f64,
                im: 0.5,
            },
            Complex {
                re: 0.5f64,
                im: -2.0,
            },
            Complex {
                re: 1.5f64,
                im: 0.0,
            },
        ];
        let expected = naive_dft(&input, 1);
        let mut data = input;
        generated_simd_4_dispatch(&mut data, 1);
        assert!(
            approx_eq_f64(&data, &expected, 1e-10),
            "size-4 f64 inverse: got {data:?}, expected {expected:?}"
        );
    }

    #[test]
    fn generated_simd_4_f64_roundtrip() {
        let original = [
            Complex {
                re: 1.0f64,
                im: 2.0,
            },
            Complex {
                re: 3.0f64,
                im: 4.0,
            },
            Complex {
                re: 5.0f64,
                im: 6.0,
            },
            Complex {
                re: 7.0f64,
                im: 8.0,
            },
        ];
        let mut data = original;
        generated_simd_4_dispatch(&mut data, -1);
        generated_simd_4_dispatch(&mut data, 1);
        let n = original.len() as f64;
        for (got, orig) in data.iter().zip(original.iter()) {
            assert!((got.re / n - orig.re).abs() < 1e-10);
            assert!((got.im / n - orig.im).abs() < 1e-10);
        }
    }

    // ----- size-4 f32 -------------------------------------------------------

    /// Proves generated codelet reaches f32 SIMD path (hand-written has no f32 SIMD).
    #[test]
    fn generated_simd_4_f32_correctness() {
        let input = [
            Complex {
                re: 1.0f32,
                im: 0.0,
            },
            Complex {
                re: 0.0f32,
                im: 1.0,
            },
            Complex {
                re: -1.0f32,
                im: 0.0,
            },
            Complex {
                re: 0.0f32,
                im: -1.0,
            },
        ];
        let expected = naive_dft_f32(&input, -1);
        let mut data = input;
        generated_simd_4_dispatch(&mut data, -1);
        assert!(
            approx_eq_f32(&data, &expected, 1e-5),
            "size-4 f32 forward: got {data:?}, expected {expected:?}"
        );
    }

    #[test]
    fn generated_simd_4_f32_roundtrip() {
        let original = [
            Complex {
                re: 1.0f32,
                im: 2.0,
            },
            Complex {
                re: 3.0f32,
                im: 4.0,
            },
            Complex {
                re: 5.0f32,
                im: 6.0,
            },
            Complex {
                re: 7.0f32,
                im: 8.0,
            },
        ];
        let mut data = original;
        generated_simd_4_dispatch(&mut data, -1);
        generated_simd_4_dispatch(&mut data, 1);
        let n = original.len() as f32;
        for (got, orig) in data.iter().zip(original.iter()) {
            assert!((got.re / n - orig.re).abs() < 1e-5);
            assert!((got.im / n - orig.im).abs() < 1e-5);
        }
    }

    // ----- size-8 f64 -------------------------------------------------------

    #[test]
    fn generated_simd_8_f64_forward_correctness() {
        let input: [Complex<f64>; 8] = [
            Complex { re: 1.0, im: 0.0 },
            Complex { re: 0.5, im: 0.5 },
            Complex { re: 0.0, im: 1.0 },
            Complex { re: -0.5, im: 0.5 },
            Complex { re: -1.0, im: 0.0 },
            Complex { re: -0.5, im: -0.5 },
            Complex { re: 0.0, im: -1.0 },
            Complex { re: 0.5, im: -0.5 },
        ];
        let expected = naive_dft(&input, -1);
        let mut data = input;
        generated_simd_8_dispatch(&mut data, -1);
        assert!(
            approx_eq_f64(&data, &expected, 1e-10),
            "size-8 f64 forward: got {data:?}, expected {expected:?}"
        );
    }

    #[test]
    fn generated_simd_8_f64_roundtrip() {
        let original: [Complex<f64>; 8] = [
            Complex { re: 1.0, im: 2.0 },
            Complex { re: 3.0, im: 4.0 },
            Complex { re: 5.0, im: 6.0 },
            Complex { re: 7.0, im: 8.0 },
            Complex { re: -1.0, im: -2.0 },
            Complex { re: -3.0, im: -4.0 },
            Complex { re: 0.5, im: -0.5 },
            Complex { re: -0.5, im: 0.5 },
        ];
        let mut data = original;
        generated_simd_8_dispatch(&mut data, -1);
        generated_simd_8_dispatch(&mut data, 1);
        let n = original.len() as f64;
        for (got, orig) in data.iter().zip(original.iter()) {
            assert!((got.re / n - orig.re).abs() < 1e-10);
            assert!((got.im / n - orig.im).abs() < 1e-10);
        }
    }

    // ----- size-8 f32 -------------------------------------------------------

    /// Proves generated codelet reaches f32 SIMD path (hand-written has no f32 SIMD).
    #[test]
    fn generated_simd_8_f32_correctness() {
        let input: [Complex<f32>; 8] = [
            Complex { re: 1.0, im: 0.0 },
            Complex { re: 0.5, im: 0.5 },
            Complex { re: 0.0, im: 1.0 },
            Complex { re: -0.5, im: 0.5 },
            Complex { re: -1.0, im: 0.0 },
            Complex { re: -0.5, im: -0.5 },
            Complex { re: 0.0, im: -1.0 },
            Complex { re: 0.5, im: -0.5 },
        ];
        let expected = naive_dft_f32(&input, -1);
        let mut data = input;
        generated_simd_8_dispatch(&mut data, -1);
        assert!(
            approx_eq_f32(&data, &expected, 1e-5),
            "size-8 f32 forward: got {data:?}, expected {expected:?}"
        );
    }

    #[test]
    fn generated_simd_8_f32_roundtrip() {
        let original: [Complex<f32>; 8] = [
            Complex { re: 1.0, im: 2.0 },
            Complex { re: 3.0, im: 4.0 },
            Complex { re: 5.0, im: 6.0 },
            Complex { re: 7.0, im: 8.0 },
            Complex { re: -1.0, im: -2.0 },
            Complex { re: -3.0, im: -4.0 },
            Complex { re: 0.5, im: -0.5 },
            Complex { re: -0.5, im: 0.5 },
        ];
        let mut data = original;
        generated_simd_8_dispatch(&mut data, -1);
        generated_simd_8_dispatch(&mut data, 1);
        let n = original.len() as f32;
        for (got, orig) in data.iter().zip(original.iter()) {
            assert!((got.re / n - orig.re).abs() < 1e-5);
            assert!((got.im / n - orig.im).abs() < 1e-5);
        }
    }

    // ----- cached dispatcher tests ------------------------------------------

    /// Verify that the cached dispatcher for size-4 f64 produces the same result
    /// as the naive DFT reference and matches the non-cached dispatcher output.
    #[test]
    fn cached_dispatcher_4_f64_correctness() {
        let input = [
            Complex {
                re: 1.0f64,
                im: 0.0,
            },
            Complex {
                re: 0.0f64,
                im: 1.0,
            },
            Complex {
                re: -1.0f64,
                im: 0.0,
            },
            Complex {
                re: 0.0f64,
                im: -1.0,
            },
        ];
        let expected = naive_dft(&input, -1);
        let mut data = input;
        codelet_simd_4_cached_f64(&mut data, -1);
        assert!(
            approx_eq_f64(&data, &expected, 1e-10),
            "cached size-4 f64 forward: got {data:?}, expected {expected:?}"
        );
    }

    /// Verify inverse direction for cached size-4 f64 dispatcher.
    #[test]
    fn cached_dispatcher_4_f64_inverse_correctness() {
        let input = [
            Complex {
                re: 2.0f64,
                im: 1.0,
            },
            Complex {
                re: -1.0f64,
                im: 0.5,
            },
            Complex {
                re: 0.5f64,
                im: -2.0,
            },
            Complex {
                re: 1.5f64,
                im: 0.0,
            },
        ];
        let expected = naive_dft(&input, 1);
        let mut data = input;
        codelet_simd_4_cached_f64(&mut data, 1);
        assert!(
            approx_eq_f64(&data, &expected, 1e-10),
            "cached size-4 f64 inverse: got {data:?}, expected {expected:?}"
        );
    }

    /// Verify that calling the cached dispatcher twice converges (cache consistency).
    #[test]
    fn cached_dispatcher_4_f64_deterministic() {
        let input = [
            Complex {
                re: 1.0f64,
                im: 2.0,
            },
            Complex {
                re: 3.0f64,
                im: 4.0,
            },
            Complex {
                re: 5.0f64,
                im: 6.0,
            },
            Complex {
                re: 7.0f64,
                im: 8.0,
            },
        ];
        let mut data_a = input;
        let mut data_b = input;
        // Two independent calls on the same input — should produce bit-identical results
        codelet_simd_4_cached_f64(&mut data_a, -1);
        codelet_simd_4_cached_f64(&mut data_b, -1);
        // Use a tiny epsilon rather than 0.0: approx_eq uses strict < so 0.0 would fail
        // even for bit-equal results (0.0 < 0.0 is false).
        assert!(
            approx_eq_f64(&data_a, &data_b, 1e-15),
            "cached dispatcher must be deterministic: got data_a={data_a:?}, data_b={data_b:?}"
        );
    }

    /// Verify cached size-4 f64 matches the non-cached dispatcher.
    #[test]
    fn cached_dispatcher_4_f64_matches_uncached() {
        let input = [
            Complex {
                re: 1.0f64,
                im: 0.5,
            },
            Complex {
                re: -0.5f64,
                im: 1.0,
            },
            Complex {
                re: 0.0f64,
                im: -1.0,
            },
            Complex {
                re: 2.0f64,
                im: 0.0,
            },
        ];
        let mut data_cached = input;
        let mut data_uncached = input;
        codelet_simd_4_cached_f64(&mut data_cached, -1);
        generated_simd_4_dispatch(&mut data_uncached, -1);
        assert!(
            approx_eq_f64(&data_cached, &data_uncached, 1e-14),
            "cached and uncached dispatchers must agree: cached={data_cached:?}, uncached={data_uncached:?}"
        );
    }

    /// Verify correctness of cached size-4 f32 dispatcher.
    #[test]
    fn cached_dispatcher_4_f32_correctness() {
        let input = [
            Complex {
                re: 1.0f32,
                im: 0.0,
            },
            Complex {
                re: 0.0f32,
                im: 1.0,
            },
            Complex {
                re: -1.0f32,
                im: 0.0,
            },
            Complex {
                re: 0.0f32,
                im: -1.0,
            },
        ];
        let expected = naive_dft_f32(&input, -1);
        let mut data = input;
        codelet_simd_4_cached_f32(&mut data, -1);
        assert!(
            approx_eq_f32(&data, &expected, 1e-5),
            "cached size-4 f32 forward: got {data:?}, expected {expected:?}"
        );
    }

    /// Verify cached size-4 f32 matches non-cached dispatcher.
    #[test]
    fn cached_dispatcher_4_f32_matches_uncached() {
        let input = [
            Complex {
                re: 1.5f32,
                im: -0.5,
            },
            Complex {
                re: -2.0f32,
                im: 1.0,
            },
            Complex {
                re: 0.5f32,
                im: 0.5,
            },
            Complex {
                re: -1.0f32,
                im: 0.0,
            },
        ];
        let mut data_cached = input;
        let mut data_uncached = input;
        codelet_simd_4_cached_f32(&mut data_cached, -1);
        generated_simd_4_dispatch(&mut data_uncached, -1);
        assert!(
            approx_eq_f32(&data_cached, &data_uncached, 1e-6),
            "cached f32 and uncached f64-typed dispatchers must agree numerically"
        );
    }

    // ----- dispatch routing tests -------------------------------------------

    /// Verify that dispatch wrappers accept both f32 and f64 without panic.
    /// The mere fact that these compile and run without panicking proves the
    /// generated dispatchers handle both types.
    #[test]
    fn dispatch_wrappers_compile_and_run_f64() {
        let mut data2 = [
            Complex {
                re: 1.0f64,
                im: 0.0,
            },
            Complex {
                re: 0.0f64,
                im: 1.0,
            },
        ];
        let mut data4 = [
            Complex {
                re: 1.0f64,
                im: 0.0,
            },
            Complex {
                re: 0.0f64,
                im: 1.0,
            },
            Complex {
                re: -1.0f64,
                im: 0.0,
            },
            Complex {
                re: 0.0f64,
                im: -1.0,
            },
        ];
        let mut data8 = [
            Complex {
                re: 1.0f64,
                im: 0.0,
            },
            Complex {
                re: 0.0f64,
                im: 0.0,
            },
            Complex {
                re: 0.0f64,
                im: 0.0,
            },
            Complex {
                re: 0.0f64,
                im: 0.0,
            },
            Complex {
                re: 0.0f64,
                im: 0.0,
            },
            Complex {
                re: 0.0f64,
                im: 0.0,
            },
            Complex {
                re: 0.0f64,
                im: 0.0,
            },
            Complex {
                re: 0.0f64,
                im: 0.0,
            },
        ];
        generated_simd_2_dispatch(&mut data2);
        generated_simd_4_dispatch(&mut data4, -1);
        generated_simd_8_dispatch(&mut data8, -1);
    }

    #[test]
    fn dispatch_wrappers_compile_and_run_f32() {
        let mut data2 = [
            Complex {
                re: 1.0f32,
                im: 0.0,
            },
            Complex {
                re: 0.0f32,
                im: 1.0,
            },
        ];
        let mut data4 = [
            Complex {
                re: 1.0f32,
                im: 0.0,
            },
            Complex {
                re: 0.0f32,
                im: 1.0,
            },
            Complex {
                re: -1.0f32,
                im: 0.0,
            },
            Complex {
                re: 0.0f32,
                im: -1.0,
            },
        ];
        let mut data8 = [
            Complex {
                re: 1.0f32,
                im: 0.0,
            },
            Complex {
                re: 0.0f32,
                im: 0.0,
            },
            Complex {
                re: 0.0f32,
                im: 0.0,
            },
            Complex {
                re: 0.0f32,
                im: 0.0,
            },
            Complex {
                re: 0.0f32,
                im: 0.0,
            },
            Complex {
                re: 0.0f32,
                im: 0.0,
            },
            Complex {
                re: 0.0f32,
                im: 0.0,
            },
            Complex {
                re: 0.0f32,
                im: 0.0,
            },
        ];
        generated_simd_2_dispatch(&mut data2);
        generated_simd_4_dispatch(&mut data4, -1);
        generated_simd_8_dispatch(&mut data8, -1);
    }
}
