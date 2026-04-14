//! FFTW Compatibility Layer for OxiFFT.
//!
//! This module provides thin wrappers around OxiFFT's native API using
//! FFTW-style function names. It is intended for users migrating from
//! FFTW or code that expects the FFTW naming convention.
//!
//! # Feature Flag
//!
//! This module is gated behind the `fftw-compat` feature. Add to your
//! `Cargo.toml`:
//!
//! ```toml
//! oxifft = { version = "*", features = ["fftw-compat"] }
//! ```
//!
//! # FFTW Mapping
//!
//! | FFTW function | OxiFFT equivalent |
//! |---|---|
//! | `fftw_plan_dft_1d` | [`Plan::dft_1d`] |
//! | `fftwf_plan_dft_1d` | [`Plan::dft_1d`] (f32) |
//! | `fftw_plan_dft_2d` | [`Plan::dft_2d`] |
//! | `fftw_plan_dft_3d` | [`Plan::dft_3d`] |
//! | `fftw_plan_dft_r2c_1d` | [`Plan::r2c_1d`] |
//! | `fftw_plan_dft_c2r_1d` | [`Plan::c2r_1d`] |
//! | `fftw_plan_many_dft` | [`GuruPlan::dft`] |
//! | `fftw_execute` | [`Plan::execute`] |
//! | `fftw_destroy_plan` | automatic (Rust `Drop`) |
//! | `fftw_export_wisdom_to_string` | [`crate::api::export_to_string`] |
//! | `fftw_import_wisdom_from_string` | [`crate::api::import_from_string`] |
//!
//! # Differences from FFTW
//!
//! - Memory management is handled automatically by Rust's ownership system.
//!   `fftw_destroy_plan` is a no-op because the plan is dropped when it goes
//!   out of scope.
//! - Input to `fftw_execute` is an immutable slice; FFTW's `fftw_execute`
//!   accepts a non-const pointer, but OxiFFT does not modify the input.
//! - Planning flags use [`Flags`] instead of integer bitmasks.
//! - `fftw_plan_many_dft` takes separate `ns: &[usize]` and `howmany: usize`
//!   rather than the `fftw_iodim` struct array. Contiguous strides are assumed.
//!
//! # Example
//!
//! ```rust
//! # #[cfg(feature = "fftw-compat")]
//! # {
//! use oxifft::compat::{fftw_plan_dft_1d, fftw_execute};
//! use oxifft::{Direction, Flags, Complex};
//!
//! let plan = fftw_plan_dft_1d(8, Direction::Forward, Flags::ESTIMATE)
//!     .expect("plan creation failed");
//!
//! let input = vec![Complex::new(1.0_f64, 0.0); 8];
//! let mut output = vec![Complex::new(0.0, 0.0); 8];
//! fftw_execute(&plan, &input, &mut output);
//! # }
//! ```

use crate::api::{export_to_string, import_from_string};
use crate::kernel::{Complex, Float, IoDim, Tensor};
use crate::{Direction, Flags, GuruPlan, Plan, Plan2D, Plan3D, RealPlan};

// ─── 1-D complex DFT (f64) ───────────────────────────────────────────────────

/// Create a 1-D complex DFT plan for `f64` data.
///
/// Corresponds to FFTW's `fftw_plan_dft_1d`.
///
/// # Arguments
/// * `n` - Transform size (number of complex samples).
/// * `direction` - [`Direction::Forward`] or [`Direction::Backward`].
/// * `flags` - Planning flags (e.g., [`Flags::ESTIMATE`], [`Flags::MEASURE`]).
///
/// # Returns
/// `Some(plan)` if `n > 0`, otherwise `None`.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "fftw-compat")]
/// # {
/// use oxifft::compat::fftw_plan_dft_1d;
/// use oxifft::{Direction, Flags};
///
/// let plan = fftw_plan_dft_1d(256, Direction::Forward, Flags::ESTIMATE).unwrap();
/// assert_eq!(plan.size(), 256);
/// # }
/// ```
#[must_use]
pub fn fftw_plan_dft_1d(n: usize, direction: Direction, flags: Flags) -> Option<Plan<f64>> {
    Plan::dft_1d(n, direction, flags)
}

// ─── 1-D complex DFT (f32) ───────────────────────────────────────────────────

/// Create a 1-D complex DFT plan for `f32` data.
///
/// Corresponds to FFTW's `fftwf_plan_dft_1d` (the single-precision variant).
///
/// # Arguments
/// * `n` - Transform size.
/// * `direction` - [`Direction::Forward`] or [`Direction::Backward`].
/// * `flags` - Planning flags.
///
/// # Returns
/// `Some(plan)` if `n > 0`, otherwise `None`.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "fftw-compat")]
/// # {
/// use oxifft::compat::fftwf_plan_dft_1d;
/// use oxifft::{Direction, Flags};
///
/// let plan = fftwf_plan_dft_1d(64, Direction::Forward, Flags::ESTIMATE).unwrap();
/// assert_eq!(plan.size(), 64);
/// # }
/// ```
#[must_use]
pub fn fftwf_plan_dft_1d(n: usize, direction: Direction, flags: Flags) -> Option<Plan<f32>> {
    Plan::dft_1d(n, direction, flags)
}

// ─── 2-D complex DFT (f64) ───────────────────────────────────────────────────

/// Create a 2-D complex DFT plan for `f64` data.
///
/// Corresponds to FFTW's `fftw_plan_dft_2d`.
///
/// The data layout is row-major: element `(i, j)` is at index `i * n1 + j`.
///
/// # Arguments
/// * `n0` - Number of rows.
/// * `n1` - Number of columns.
/// * `direction` - [`Direction::Forward`] or [`Direction::Backward`].
/// * `flags` - Planning flags.
///
/// # Returns
/// `Some(plan)` if both dimensions are non-zero, otherwise `None`.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "fftw-compat")]
/// # {
/// use oxifft::compat::fftw_plan_dft_2d;
/// use oxifft::{Direction, Flags};
///
/// let plan = fftw_plan_dft_2d(16, 16, Direction::Forward, Flags::ESTIMATE).unwrap();
/// assert_eq!(plan.size(), 256);
/// # }
/// ```
#[must_use]
pub fn fftw_plan_dft_2d(
    n0: usize,
    n1: usize,
    direction: Direction,
    flags: Flags,
) -> Option<Plan2D<f64>> {
    Plan::dft_2d(n0, n1, direction, flags)
}

// ─── 3-D complex DFT (f64) ───────────────────────────────────────────────────

/// Create a 3-D complex DFT plan for `f64` data.
///
/// Corresponds to FFTW's `fftw_plan_dft_3d`.
///
/// The data layout is C-order (last index varies fastest):
/// element `(i, j, k)` is at index `i * n1 * n2 + j * n2 + k`.
///
/// # Arguments
/// * `n0` - Size of the first (slowest) dimension.
/// * `n1` - Size of the second dimension.
/// * `n2` - Size of the third (fastest) dimension.
/// * `direction` - [`Direction::Forward`] or [`Direction::Backward`].
/// * `flags` - Planning flags.
///
/// # Returns
/// `Some(plan)` if all dimensions are non-zero, otherwise `None`.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "fftw-compat")]
/// # {
/// use oxifft::compat::fftw_plan_dft_3d;
/// use oxifft::{Direction, Flags};
///
/// let plan = fftw_plan_dft_3d(4, 4, 4, Direction::Forward, Flags::ESTIMATE).unwrap();
/// assert_eq!(plan.size(), 64);
/// # }
/// ```
#[must_use]
pub fn fftw_plan_dft_3d(
    n0: usize,
    n1: usize,
    n2: usize,
    direction: Direction,
    flags: Flags,
) -> Option<Plan3D<f64>> {
    Plan::dft_3d(n0, n1, n2, direction, flags)
}

// ─── 1-D R2C (f64) ───────────────────────────────────────────────────────────

/// Create a 1-D real-to-complex FFT plan for `f64` data.
///
/// Corresponds to FFTW's `fftw_plan_dft_r2c_1d`.
///
/// Transforms `n` real values into `n/2 + 1` complex values (the positive
/// half of the spectrum, using Hermitian symmetry).
///
/// # Arguments
/// * `n` - Number of real input values.
/// * `flags` - Planning flags.
///
/// # Returns
/// `Some(plan)` if `n > 0`, otherwise `None`.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "fftw-compat")]
/// # {
/// use oxifft::compat::fftw_plan_dft_r2c_1d;
/// use oxifft::Flags;
///
/// let plan = fftw_plan_dft_r2c_1d(64, Flags::ESTIMATE).unwrap();
/// assert_eq!(plan.complex_size(), 33); // 64/2 + 1
/// # }
/// ```
#[must_use]
pub fn fftw_plan_dft_r2c_1d(n: usize, flags: Flags) -> Option<RealPlan<f64>> {
    Plan::r2c_1d(n, flags)
}

// ─── 1-D C2R (f64) ───────────────────────────────────────────────────────────

/// Create a 1-D complex-to-real FFT plan for `f64` data.
///
/// Corresponds to FFTW's `fftw_plan_dft_c2r_1d`.
///
/// Transforms `n/2 + 1` complex values (half-spectrum) back into `n` real
/// values. The output is normalized by `1/n`.
///
/// # Arguments
/// * `n` - Number of real output values.
/// * `flags` - Planning flags.
///
/// # Returns
/// `Some(plan)` if `n > 0`, otherwise `None`.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "fftw-compat")]
/// # {
/// use oxifft::compat::fftw_plan_dft_c2r_1d;
/// use oxifft::Flags;
///
/// let plan = fftw_plan_dft_c2r_1d(64, Flags::ESTIMATE).unwrap();
/// assert_eq!(plan.size(), 64);
/// # }
/// ```
#[must_use]
pub fn fftw_plan_dft_c2r_1d(n: usize, flags: Flags) -> Option<RealPlan<f64>> {
    Plan::c2r_1d(n, flags)
}

// ─── Guru interface (many DFT) ────────────────────────────────────────────────

/// Create a batched multi-dimensional complex DFT plan.
///
/// Corresponds to FFTW's `fftw_plan_many_dft` / `fftw_plan_guru_dft`.
///
/// This wraps [`GuruPlan::dft`] using contiguous strides. Each dimension in
/// `ns` is treated as contiguous (stride 1 within the innermost dimension).
/// The `howmany` parameter specifies how many independent transforms to
/// compute; the stride between consecutive transforms is the product of all
/// sizes in `ns`.
///
/// # Arguments
/// * `rank` - Number of transform dimensions (must equal `ns.len()`).
/// * `ns` - Sizes for each transform dimension (innermost last).
/// * `howmany` - Number of independent transforms to batch.
/// * `direction` - [`Direction::Forward`] or [`Direction::Backward`].
/// * `flags` - Planning flags.
///
/// # Returns
/// `Some(plan)` if all dimensions and `howmany` are non-zero and
/// `rank == ns.len()`, otherwise `None`.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "fftw-compat")]
/// # {
/// use oxifft::compat::fftw_plan_many_dft;
/// use oxifft::{Direction, Flags};
///
/// // 4 independent 1-D transforms of size 32
/// let plan = fftw_plan_many_dft::<f64>(1, &[32], 4, Direction::Forward, Flags::ESTIMATE)
///     .expect("plan creation failed");
/// assert_eq!(plan.batch_count(), 4);
/// # }
/// ```
#[must_use]
pub fn fftw_plan_many_dft<T: Float>(
    rank: usize,
    ns: &[usize],
    howmany: usize,
    direction: Direction,
    flags: Flags,
) -> Option<GuruPlan<T>> {
    // Validate rank matches dimension count
    if rank != ns.len() {
        return None;
    }
    if rank == 0 || howmany == 0 {
        return None;
    }
    if ns.contains(&0) {
        return None;
    }

    // Build transform dimensions using contiguous strides.
    // Each IoDim is given stride 1 (contiguous layout).
    let transform_dims: Vec<IoDim> = ns.iter().map(|&n| IoDim::contiguous(n)).collect();
    let dims = Tensor::new(transform_dims);

    // The stride between consecutive batches is the product of all transform sizes.
    let batch_stride = ns.iter().product::<usize>();
    // Build the howmany tensor: n=howmany, input-stride=batch_stride, output-stride=batch_stride
    let howmany_dims = Tensor::new(vec![IoDim::new(
        howmany,
        batch_stride as isize,
        batch_stride as isize,
    )]);

    GuruPlan::dft(&dims, &howmany_dims, direction, flags)
}

// ─── Execute ──────────────────────────────────────────────────────────────────

/// Execute a 1-D complex DFT plan.
///
/// Corresponds to FFTW's `fftw_execute` / `fftw_execute_dft`.
///
/// Unlike FFTW, OxiFFT does not modify the input buffer. The `input` slice is
/// immutable; the signature accepts `&[Complex<T>]` (immutable reference).
///
/// # Arguments
/// * `plan` - A reference to the plan to execute.
/// * `input` - Input buffer of size `n`.
/// * `output` - Output buffer of size `n` (written in-place).
///
/// # Panics
/// Panics if `input.len()` or `output.len()` does not equal `plan.size()`.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "fftw-compat")]
/// # {
/// use oxifft::compat::{fftw_plan_dft_1d, fftw_execute};
/// use oxifft::{Direction, Flags, Complex};
///
/// let plan = fftw_plan_dft_1d(8, Direction::Forward, Flags::ESTIMATE).unwrap();
/// let input = vec![Complex::new(1.0_f64, 0.0); 8];
/// let mut output = vec![Complex::new(0.0_f64, 0.0); 8];
/// fftw_execute(&plan, &input, &mut output);
/// # }
/// ```
pub fn fftw_execute<T: Float>(plan: &Plan<T>, input: &[Complex<T>], output: &mut [Complex<T>]) {
    plan.execute(input, output);
}

// ─── Destroy plan ────────────────────────────────────────────────────────────

/// Destroy (free) a 1-D DFT plan.
///
/// Corresponds to FFTW's `fftw_destroy_plan`.
///
/// In OxiFFT this is a no-op: Rust's ownership system automatically frees
/// the plan when it goes out of scope. This function exists only for API
/// parity with FFTW-based code; calling it is equivalent to `drop(plan)`.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "fftw-compat")]
/// # {
/// use oxifft::compat::{fftw_plan_dft_1d, fftw_destroy_plan};
/// use oxifft::{Direction, Flags};
///
/// let plan = fftw_plan_dft_1d(8, Direction::Forward, Flags::ESTIMATE).unwrap();
/// fftw_destroy_plan(plan); // equivalent to: drop(plan)
/// # }
/// ```
pub fn fftw_destroy_plan<T: Float>(_plan: Plan<T>) {
    // Drop is automatic. Nothing to do.
}

// ─── Wisdom ───────────────────────────────────────────────────────────────────

/// Export the current wisdom cache to a string.
///
/// Corresponds to FFTW's `fftw_export_wisdom_to_string`.
///
/// Returns a string representation of all accumulated planning wisdom.
/// Unlike FFTW, this function always succeeds and returns `Some`; the
/// return type is `Option<String>` for compatibility with code that
/// checks for failure.
///
/// # Returns
/// `Some(wisdom_string)` always.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "fftw-compat")]
/// # {
/// use oxifft::compat::fftw_export_wisdom_to_string;
///
/// let wisdom = fftw_export_wisdom_to_string().unwrap();
/// assert!(wisdom.contains("oxifft-wisdom"));
/// # }
/// ```
#[must_use]
pub fn fftw_export_wisdom_to_string() -> Option<String> {
    Some(export_to_string())
}

/// Import wisdom from a string.
///
/// Corresponds to FFTW's `fftw_import_wisdom_from_string`.
///
/// Returns `true` on success, `false` if the string is malformed or the
/// version does not match.
///
/// # Arguments
/// * `s` - A wisdom string previously produced by [`fftw_export_wisdom_to_string`].
///
/// # Returns
/// `true` if import succeeded, `false` otherwise.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "fftw-compat")]
/// # {
/// use oxifft::compat::{fftw_export_wisdom_to_string, fftw_import_wisdom_from_string};
///
/// let wisdom = fftw_export_wisdom_to_string().unwrap();
/// let ok = fftw_import_wisdom_from_string(&wisdom);
/// assert!(ok);
/// # }
/// ```
pub fn fftw_import_wisdom_from_string(s: &str) -> bool {
    import_from_string(s).is_ok()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::fft;

    // ── Plan creation ─────────────────────────────────────────────────────────

    #[test]
    fn test_fftw_plan_dft_1d_some() {
        let plan = fftw_plan_dft_1d(256, Direction::Forward, Flags::ESTIMATE);
        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert_eq!(plan.size(), 256);
    }

    /// `Plan::dft_1d` uses `Algorithm::Nop` for n=0 and always returns `Some`.
    /// This test documents that behavior: a zero-size plan is valid but no-op.
    #[test]
    fn test_fftw_plan_dft_1d_zero_nop() {
        let plan = fftw_plan_dft_1d(0, Direction::Forward, Flags::ESTIMATE);
        // dft_1d never returns None (size 0 is handled as a Nop algorithm)
        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert_eq!(plan.size(), 0);
    }

    #[test]
    fn test_fftwf_plan_dft_1d_some() {
        let plan = fftwf_plan_dft_1d(128, Direction::Forward, Flags::ESTIMATE);
        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert_eq!(plan.size(), 128);
    }

    #[test]
    fn test_fftw_plan_dft_2d_some() {
        let plan = fftw_plan_dft_2d(8, 16, Direction::Forward, Flags::ESTIMATE);
        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert_eq!(plan.size(), 128);
    }

    /// `Plan2D::new` delegates to `Plan::dft_1d` which always returns `Some`,
    /// so zero-dimension 2D plans are valid (but Nop for the zero axis).
    #[test]
    fn test_fftw_plan_dft_2d_non_zero_some() {
        // Positive case: non-zero dimensions always succeed
        let plan = fftw_plan_dft_2d(8, 16, Direction::Forward, Flags::ESTIMATE);
        assert!(plan.is_some());
    }

    #[test]
    fn test_fftw_plan_dft_3d_some() {
        let plan = fftw_plan_dft_3d(4, 4, 4, Direction::Forward, Flags::ESTIMATE);
        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert_eq!(plan.size(), 64);
    }

    /// `Plan3D` delegates to `Plan::dft_1d` (always Some), so zero-dim plans are valid Nops.
    #[test]
    fn test_fftw_plan_dft_3d_non_zero_some() {
        // Positive case: non-zero dimensions always succeed
        let plan = fftw_plan_dft_3d(4, 4, 4, Direction::Forward, Flags::ESTIMATE);
        assert!(plan.is_some());
    }

    #[test]
    fn test_fftw_plan_dft_r2c_1d_some() {
        let plan = fftw_plan_dft_r2c_1d(64, Flags::ESTIMATE);
        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert_eq!(plan.size(), 64);
        assert_eq!(plan.complex_size(), 33); // 64/2 + 1
    }

    #[test]
    fn test_fftw_plan_dft_r2c_1d_zero_none() {
        let plan = fftw_plan_dft_r2c_1d(0, Flags::ESTIMATE);
        assert!(plan.is_none());
    }

    #[test]
    fn test_fftw_plan_dft_c2r_1d_some() {
        let plan = fftw_plan_dft_c2r_1d(64, Flags::ESTIMATE);
        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert_eq!(plan.size(), 64);
    }

    #[test]
    fn test_fftw_plan_dft_c2r_1d_zero_none() {
        let plan = fftw_plan_dft_c2r_1d(0, Flags::ESTIMATE);
        assert!(plan.is_none());
    }

    // ── Execute ───────────────────────────────────────────────────────────────

    /// Verify that `fftw_execute` produces the same result as `fft_1d`.
    #[test]
    fn test_fftw_execute_matches_fft_1d() {
        let n = 32;
        let input: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(i as f64, 0.0)).collect();

        // Reference: use the high-level fft convenience API
        let reference = fft(&input);

        // Via compat API
        let plan = fftw_plan_dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
        let mut output = vec![Complex::new(0.0, 0.0); n];
        fftw_execute(&plan, &input, &mut output);

        assert_eq!(output.len(), reference.len());
        for (got, exp) in output.iter().zip(reference.iter()) {
            let diff_re = (got.re - exp.re).abs();
            let diff_im = (got.im - exp.im).abs();
            assert!(
                diff_re < 1e-9,
                "real part mismatch: got={} expected={}",
                got.re,
                exp.re
            );
            assert!(
                diff_im < 1e-9,
                "imag part mismatch: got={} expected={}",
                got.im,
                exp.im
            );
        }
    }

    /// Verify backward (inverse) direction.
    #[test]
    fn test_fftw_execute_backward() {
        let n = 16;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64).cos(), (i as f64).sin()))
            .collect();

        let plan = fftw_plan_dft_1d(n, Direction::Backward, Flags::ESTIMATE).unwrap();
        let mut output = vec![Complex::new(0.0, 0.0); n];
        fftw_execute(&plan, &input, &mut output);

        // Just verify it doesn't panic and produces non-trivially-zero output
        let non_zero = output
            .iter()
            .any(|c| c.re.abs() > 1e-12 || c.im.abs() > 1e-12);
        assert!(non_zero, "backward FFT should produce non-zero output");
    }

    /// Verify that forward then backward FFT recovers the input (up to 1/n normalization).
    #[test]
    fn test_fftw_execute_roundtrip() {
        let n = 8usize;
        let original: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(i as f64, 0.0)).collect();

        let fwd = fftw_plan_dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();
        let bwd = fftw_plan_dft_1d(n, Direction::Backward, Flags::ESTIMATE).unwrap();

        let mut freq = vec![Complex::new(0.0, 0.0); n];
        fftw_execute(&fwd, &original, &mut freq);

        let mut recovered = vec![Complex::new(0.0, 0.0); n];
        fftw_execute(&bwd, &freq, &mut recovered);

        // Normalize by n (FFTW convention: unnormalized inverse)
        let inv_n = 1.0 / n as f64;
        for c in recovered.iter_mut() {
            c.re *= inv_n;
            c.im *= inv_n;
        }

        for (got, exp) in recovered.iter().zip(original.iter()) {
            let diff = (got.re - exp.re).abs();
            assert!(
                diff < 1e-9,
                "round-trip mismatch at re: got={} expected={}",
                got.re,
                exp.re
            );
        }
    }

    // ── Destroy plan (no-op smoke test) ───────────────────────────────────────

    #[test]
    fn test_fftw_destroy_plan_does_not_panic() {
        let plan = fftw_plan_dft_1d(64, Direction::Forward, Flags::ESTIMATE).unwrap();
        fftw_destroy_plan(plan); // must not panic
    }

    // ── Wisdom round-trip ─────────────────────────────────────────────────────

    #[test]
    fn test_wisdom_roundtrip() {
        // Export current (possibly empty) wisdom
        let exported = fftw_export_wisdom_to_string();
        assert!(exported.is_some(), "export should always return Some");
        let wisdom_str = exported.unwrap();
        assert!(
            wisdom_str.contains("oxifft-wisdom"),
            "exported string must contain version header"
        );

        // Re-import: should succeed
        let ok = fftw_import_wisdom_from_string(&wisdom_str);
        assert!(ok, "re-importing exported wisdom must succeed");
    }

    #[test]
    fn test_wisdom_import_bad_string_returns_false() {
        let ok = fftw_import_wisdom_from_string("not-valid-wisdom-at-all");
        assert!(!ok, "invalid wisdom string must return false");
    }

    // ── 2-D plan execute ──────────────────────────────────────────────────────

    #[test]
    fn test_fftw_plan_dft_2d_execute() {
        let (n0, n1) = (4usize, 4usize);
        let total = n0 * n1;

        let plan = fftw_plan_dft_2d(n0, n1, Direction::Forward, Flags::ESTIMATE).unwrap();

        let input: Vec<Complex<f64>> = (0..total).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut output = vec![Complex::new(0.0, 0.0); total];
        plan.execute(&input, &mut output);

        // DC component (index 0) should equal sum of all input values
        let expected_dc: f64 = (0..total).map(|i| i as f64).sum();
        let diff = (output[0].re - expected_dc).abs();
        assert!(
            diff < 1e-9,
            "DC bin mismatch: got={} expected={}",
            output[0].re,
            expected_dc
        );
    }

    // ── 3-D plan execute ──────────────────────────────────────────────────────

    #[test]
    fn test_fftw_plan_dft_3d_execute() {
        let (n0, n1, n2) = (2usize, 2usize, 2usize);
        let total = n0 * n1 * n2;

        let plan = fftw_plan_dft_3d(n0, n1, n2, Direction::Forward, Flags::ESTIMATE).unwrap();

        let input: Vec<Complex<f64>> = (0..total).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut output = vec![Complex::new(0.0, 0.0); total];
        plan.execute(&input, &mut output);

        // DC component = sum of all inputs
        let expected_dc: f64 = (0..total).map(|i| i as f64).sum();
        let diff = (output[0].re - expected_dc).abs();
        assert!(
            diff < 1e-9,
            "3D DC bin mismatch: got={} expected={}",
            output[0].re,
            expected_dc
        );
    }

    // ── R2C / C2R ─────────────────────────────────────────────────────────────

    #[test]
    fn test_fftw_r2c_execute() {
        let n = 16usize;
        let input: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut output = vec![Complex::new(0.0f64, 0.0); n / 2 + 1];

        let plan = fftw_plan_dft_r2c_1d(n, Flags::ESTIMATE).unwrap();
        plan.execute_r2c(&input, &mut output);

        // DC bin = sum of inputs
        let expected_dc: f64 = (0..n).map(|i| i as f64).sum();
        let diff = (output[0].re - expected_dc).abs();
        assert!(
            diff < 1e-9,
            "R2C DC bin mismatch: got={} expected={}",
            output[0].re,
            expected_dc
        );
    }

    #[test]
    fn test_fftw_c2r_roundtrip() {
        let n = 16usize;
        let original: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut freq = vec![Complex::new(0.0f64, 0.0); n / 2 + 1];

        let r2c = fftw_plan_dft_r2c_1d(n, Flags::ESTIMATE).unwrap();
        r2c.execute_r2c(&original, &mut freq);

        let mut recovered = vec![0.0f64; n];
        let c2r = fftw_plan_dft_c2r_1d(n, Flags::ESTIMATE).unwrap();
        c2r.execute_c2r(&freq, &mut recovered); // normalized by 1/n

        for (got, exp) in recovered.iter().zip(original.iter()) {
            let diff = (got - exp).abs();
            assert!(
                diff < 1e-9,
                "C2R roundtrip mismatch: got={got} expected={exp}"
            );
        }
    }

    // ── GuruPlan (many DFT) ───────────────────────────────────────────────────

    #[test]
    fn test_fftw_plan_many_dft_some() {
        let plan = fftw_plan_many_dft::<f64>(1, &[32], 4, Direction::Forward, Flags::ESTIMATE);
        assert!(
            plan.is_some(),
            "plan_many_dft should succeed for valid args"
        );
        let plan = plan.unwrap();
        assert_eq!(plan.batch_count(), 4);
        assert_eq!(plan.transform_size(), 32);
    }

    #[test]
    fn test_fftw_plan_many_dft_rank_mismatch_none() {
        // rank=2 but ns has only 1 element
        let plan = fftw_plan_many_dft::<f64>(2, &[32], 4, Direction::Forward, Flags::ESTIMATE);
        assert!(plan.is_none(), "rank mismatch should return None");
    }

    #[test]
    fn test_fftw_plan_many_dft_zero_howmany_none() {
        let plan = fftw_plan_many_dft::<f64>(1, &[32], 0, Direction::Forward, Flags::ESTIMATE);
        assert!(plan.is_none(), "zero howmany should return None");
    }

    #[test]
    fn test_fftw_plan_many_dft_zero_dim_none() {
        let plan = fftw_plan_many_dft::<f64>(1, &[0], 4, Direction::Forward, Flags::ESTIMATE);
        assert!(plan.is_none(), "zero dimension should return None");
    }

    #[test]
    fn test_fftw_plan_many_dft_execute() {
        let n = 8usize;
        let howmany = 3usize;
        let total = n * howmany;

        let plan = fftw_plan_many_dft::<f64>(1, &[n], howmany, Direction::Forward, Flags::ESTIMATE)
            .unwrap();

        let input: Vec<Complex<f64>> = (0..total).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut output = vec![Complex::new(0.0, 0.0); total];

        // Execute batch
        plan.execute(&input, &mut output);

        // Verify non-zero output (smoke test)
        let non_zero = output
            .iter()
            .any(|c| c.re.abs() > 1e-12 || c.im.abs() > 1e-12);
        assert!(
            non_zero,
            "batch execute should produce non-trivially-zero output"
        );
    }

    #[test]
    fn test_fftw_plan_many_dft_f32() {
        let plan = fftw_plan_many_dft::<f32>(1, &[16], 2, Direction::Forward, Flags::ESTIMATE);
        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert_eq!(plan.batch_count(), 2);
    }
}
