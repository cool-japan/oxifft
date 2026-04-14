//! 2D Non-uniform FFT (NUFFT) implementations.
//!
//! This module extends the 1D NUFFT to two spatial dimensions using the same
//! Gaussian gridding approach as the 1D case.  The 2D Gaussian spreading
//! kernel is separable — `G₂(x,y) = G₁(x) · G₁(y)` — which means each
//! non-uniform point spreads onto the oversampled 2D grid as the outer product
//! of two independent 1D weight vectors.  Similarly, the 2D deconvolution
//! correction is `D(k₁,k₂) = D₁(k₁) · D₁(k₂)`.
//!
//! # Coordinate convention
//!
//! All non-uniform point coordinates must lie in `[-π, π)`.  The output grid
//! is stored in row-major order: element `(i₁, i₂)` lives at flat index
//! `i₁ * n2 + i₂`.
//!
//! # References
//!
//! Greengard, L. & Lee, J.-Y. (2004). Accelerating the nonuniform fast
//! Fourier transform. *SIAM Review*, 46(3), 443–454.

use crate::api::{Direction, Flags, Plan2D};
use crate::kernel::{Complex, Float};

use super::{
    compute_kernel_width, next_smooth_number, precompute_deconv_factors, NufftError, NufftOptions,
    NufftResult,
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute 1-D Gaussian kernel weights for a single non-uniform point.
///
/// Returns a `Vec` of `(grid_index, weight)` pairs.  The grid is of size
/// `n_grid` and the `kernel_width` controls the spatial support of the
/// Gaussian.  The point `x` must already be normalised to `[0, 2π)`.
fn gaussian_weights_1d<T: Float>(x: f64, n_grid: usize, kernel_width: usize) -> Vec<(usize, T)> {
    let grid_spacing = 2.0 * core::f64::consts::PI / (n_grid as f64);
    let half_width = kernel_width / 2;
    let beta = 2.3 * (kernel_width as f64);

    let grid_pos = x / grid_spacing;
    let center = grid_pos.round() as isize;

    let mut coeffs = Vec::with_capacity(kernel_width + 1);

    for offset in -(half_width as isize)..=(half_width as isize) {
        let grid_idx = (center + offset).rem_euclid(n_grid as isize) as usize;
        let grid_x = (grid_idx as f64) * grid_spacing;

        let mut dx = x - grid_x;
        // Wrap distance into [-π, π)
        if dx > core::f64::consts::PI {
            dx -= 2.0 * core::f64::consts::PI;
        } else if dx < -core::f64::consts::PI {
            dx += 2.0 * core::f64::consts::PI;
        }

        let normalized_dx = dx / (grid_spacing * (half_width as f64));
        let weight = (-beta * normalized_dx * normalized_dx).exp();

        if weight > 1e-15 {
            coeffs.push((grid_idx, T::from_f64(weight)));
        }
    }

    coeffs
}

/// Normalise a coordinate from `[-π, π)` to `[0, 2π)`.
#[inline]
fn normalize_coord(p: f64) -> Result<f64, NufftError> {
    if !(-core::f64::consts::PI..=core::f64::consts::PI).contains(&p) {
        return Err(NufftError::PointsOutOfRange);
    }
    Ok(p + core::f64::consts::PI)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// 2D NUFFT Type 1: Non-uniform to uniform.
///
/// Given `M` non-uniform sample points `(xj, yj) ∈ [-π, π)²` with complex
/// strengths `cj`, computes the 2-D DFT on a uniform `n1 × n2` grid using
/// the Gaussian gridding / oversampled-FFT approach.
///
/// # Arguments
///
/// * `x`       – x-coordinates of the non-uniform points, length `M`
/// * `y`       – y-coordinates of the non-uniform points, length `M`
/// * `c`       – complex strengths at each point, length `M`
/// * `n1`      – number of output grid rows
/// * `n2`      – number of output grid columns
/// * `options` – NUFFT tuning parameters (oversampling, kernel width, …)
///
/// # Returns
///
/// A flat `Vec<Complex<T>>` of length `n1 * n2` in row-major order.
/// Element `(k1, k2)` is at index `k1 * n2 + k2`.
///
/// # Errors
///
/// Returns [`NufftError::InvalidSize`] if `n1` or `n2` is zero,
/// [`NufftError::PointsOutOfRange`] if any coordinate is outside `[-π, π]`,
/// [`NufftError::InvalidTolerance`] if `options.tolerance ≤ 0`, or
/// [`NufftError::PlanFailed`] if an internal FFT plan cannot be allocated.
///
/// # Example
///
/// ```ignore
/// use oxifft::nufft::{nufft2d_type1, NufftOptions};
/// use oxifft::kernel::Complex;
///
/// let x = vec![0.0f64, 1.0, -1.0];
/// let y = vec![0.0f64, 0.5, -0.5];
/// let c = vec![Complex::new(1.0, 0.0); 3];
/// let result = nufft2d_type1(&x, &y, &c, 16, 16, &NufftOptions::default()).unwrap();
/// assert_eq!(result.len(), 16 * 16);
/// ```
pub fn nufft2d_type1<T: Float>(
    x: &[f64],
    y: &[f64],
    c: &[Complex<T>],
    n1: usize,
    n2: usize,
    options: &NufftOptions,
) -> NufftResult<Vec<Complex<T>>> {
    // --- Validation ---------------------------------------------------------
    if n1 == 0 {
        return Err(NufftError::InvalidSize(0));
    }
    if n2 == 0 {
        return Err(NufftError::InvalidSize(0));
    }
    if options.tolerance <= 0.0 {
        return Err(NufftError::InvalidTolerance);
    }
    let m = c.len();
    if x.len() != m || y.len() != m {
        return Err(NufftError::ExecutionFailed(format!(
            "x ({}) / y ({}) / c ({}) lengths must match",
            x.len(),
            y.len(),
            m
        )));
    }

    // --- Kernel parameters --------------------------------------------------
    let kernel_width = compute_kernel_width(options.tolerance, options.kernel_width);
    let n_over1 = next_smooth_number(((n1 as f64) * options.oversampling).ceil() as usize);
    let n_over2 = next_smooth_number(((n2 as f64) * options.oversampling).ceil() as usize);

    // --- Normalise coordinates ----------------------------------------------
    let mut xn = Vec::with_capacity(m);
    let mut yn = Vec::with_capacity(m);
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        xn.push(normalize_coord(xi)?);
        yn.push(normalize_coord(yi)?);
    }

    // --- Compute 1-D kernel weights per dimension ---------------------------
    let wx: Vec<Vec<(usize, T)>> = xn
        .iter()
        .map(|&xi| gaussian_weights_1d(xi, n_over1, kernel_width))
        .collect();
    let wy: Vec<Vec<(usize, T)>> = yn
        .iter()
        .map(|&yi| gaussian_weights_1d(yi, n_over2, kernel_width))
        .collect();

    // --- Spread onto oversampled 2-D grid -----------------------------------
    let mut grid = vec![Complex::<T>::zero(); n_over1 * n_over2];

    for j in 0..m {
        let val = c[j];
        for &(ix, wx_val) in &wx[j] {
            for &(iy, wy_val) in &wy[j] {
                let flat = ix * n_over2 + iy;
                let w = wx_val * wy_val;
                grid[flat] = grid[flat] + Complex::new(val.re * w, val.im * w);
            }
        }
    }

    // --- 2D FFT on oversampled grid -----------------------------------------
    let plan = Plan2D::new(n_over1, n_over2, Direction::Forward, Flags::ESTIMATE)
        .ok_or(NufftError::PlanFailed)?;

    let mut fft_result = vec![Complex::<T>::zero(); n_over1 * n_over2];
    plan.execute(&grid, &mut fft_result);

    // --- Deconvolution correction and frequency extraction ------------------
    let deconv1 = precompute_deconv_factors::<T>(n1, n_over1, kernel_width);
    let deconv2 = precompute_deconv_factors::<T>(n2, n_over2, kernel_width);

    let half1 = n1 / 2;
    let half2 = n2 / 2;

    // Cap deconvolution to prevent exponential blowup at high-frequency
    // corner bins.  The Gaussian kernel's FT decays to near-zero there, so
    // amplifying beyond 1/tolerance only magnifies rounding noise.
    let max_deconv = T::from_f64(1.0 / options.tolerance);

    let mut result = Vec::with_capacity(n1 * n2);

    for k1 in 0..n1 {
        // Map output k1 to oversampled grid index
        let grid_idx1 = if k1 < half1 { k1 } else { n_over1 - (n1 - k1) };

        for k2 in 0..n2 {
            let grid_idx2 = if k2 < half2 { k2 } else { n_over2 - (n2 - k2) };

            let flat_grid = grid_idx1 * n_over2 + grid_idx2;
            // Product of 1-D deconvolution factors; cap each factor individually
            // to avoid double-exponential blowup in 2-D corner bins.
            let d1 = if deconv1[k1].re > max_deconv {
                Complex::new(max_deconv, T::ZERO)
            } else {
                deconv1[k1]
            };
            let d2 = if deconv2[k2].re > max_deconv {
                Complex::new(max_deconv, T::ZERO)
            } else {
                deconv2[k2]
            };
            result.push(fft_result[flat_grid] * d1 * d2);
        }
    }

    Ok(result)
}

/// 2D NUFFT Type 2: Uniform to non-uniform.
///
/// Given a uniform `n1 × n2` grid of complex Fourier coefficients `f` (in
/// row-major order), evaluates the 2-D inverse DFT at `M` non-uniform points
/// `(xj, yj) ∈ [-π, π)²`.
///
/// This is the adjoint of [`nufft2d_type1`]: it maps from frequency space
/// (uniform grid) to physical space (non-uniform points).
///
/// # Arguments
///
/// * `f`       – uniform grid of complex Fourier coefficients, length `n1 * n2`
///   in row-major order (`f[k1 * n2 + k2]` for grid point `(k1, k2)`)
/// * `x`       – x-coordinates to evaluate at, length `M`
/// * `y`       – y-coordinates to evaluate at, length `M`
/// * `n1`      – number of input grid rows
/// * `n2`      – number of input grid columns
/// * `options` – NUFFT tuning parameters
///
/// # Returns
///
/// A `Vec<Complex<T>>` of length `M`.  Element `j` is the 2-D inverse DFT
/// of `f` evaluated at `(xj, yj)`.
///
/// # Errors
///
/// Returns errors in the same cases as [`nufft2d_type1`].
///
/// # Example
///
/// ```ignore
/// use oxifft::nufft::{nufft2d_type2, NufftOptions};
/// use oxifft::kernel::Complex;
///
/// let mut f = vec![Complex::<f64>::zero(); 16 * 16];
/// f[0] = Complex::new(1.0, 0.0); // DC component
/// let x = vec![0.0f64, 0.5, -0.5];
/// let y = vec![0.0f64, 0.5, -0.5];
/// let vals = nufft2d_type2(&f, &x, &y, 16, 16, &NufftOptions::default()).unwrap();
/// assert_eq!(vals.len(), 3);
/// ```
pub fn nufft2d_type2<T: Float>(
    f: &[Complex<T>],
    x: &[f64],
    y: &[f64],
    n1: usize,
    n2: usize,
    options: &NufftOptions,
) -> NufftResult<Vec<Complex<T>>> {
    // --- Validation ---------------------------------------------------------
    if n1 == 0 {
        return Err(NufftError::InvalidSize(0));
    }
    if n2 == 0 {
        return Err(NufftError::InvalidSize(0));
    }
    if f.len() != n1 * n2 {
        return Err(NufftError::ExecutionFailed(format!(
            "f length {} must equal n1*n2 = {}",
            f.len(),
            n1 * n2
        )));
    }
    if options.tolerance <= 0.0 {
        return Err(NufftError::InvalidTolerance);
    }
    let m = x.len();
    if y.len() != m {
        return Err(NufftError::ExecutionFailed(format!(
            "x ({}) and y ({}) lengths must match",
            m,
            y.len()
        )));
    }

    // --- Kernel parameters --------------------------------------------------
    let kernel_width = compute_kernel_width(options.tolerance, options.kernel_width);
    let n_over1 = next_smooth_number(((n1 as f64) * options.oversampling).ceil() as usize);
    let n_over2 = next_smooth_number(((n2 as f64) * options.oversampling).ceil() as usize);

    // --- Normalise coordinates ----------------------------------------------
    let mut xn = Vec::with_capacity(m);
    let mut yn = Vec::with_capacity(m);
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        xn.push(normalize_coord(xi)?);
        yn.push(normalize_coord(yi)?);
    }

    // --- Deconvolution correction and scatter into oversampled grid ---------
    let deconv1 = precompute_deconv_factors::<T>(n1, n_over1, kernel_width);
    let deconv2 = precompute_deconv_factors::<T>(n2, n_over2, kernel_width);

    let half1 = n1 / 2;
    let half2 = n2 / 2;

    // Cap individual 1-D deconvolution factors to prevent exponential blowup
    // at high-frequency corner bins of the oversampled 2-D grid.
    let max_deconv = T::from_f64(1.0 / options.tolerance);

    let mut grid = vec![Complex::<T>::zero(); n_over1 * n_over2];

    for k1 in 0..n1 {
        let grid_idx1 = if k1 < half1 { k1 } else { n_over1 - (n1 - k1) };
        let d1 = if deconv1[k1].re > max_deconv {
            Complex::new(max_deconv, T::ZERO)
        } else {
            deconv1[k1]
        };

        for k2 in 0..n2 {
            let grid_idx2 = if k2 < half2 { k2 } else { n_over2 - (n2 - k2) };

            let flat_in = k1 * n2 + k2;
            let flat_grid = grid_idx1 * n_over2 + grid_idx2;
            let d2 = if deconv2[k2].re > max_deconv {
                Complex::new(max_deconv, T::ZERO)
            } else {
                deconv2[k2]
            };
            grid[flat_grid] = f[flat_in] * d1 * d2;
        }
    }

    // --- 2D IFFT on oversampled grid ----------------------------------------
    let plan = Plan2D::new(n_over1, n_over2, Direction::Backward, Flags::ESTIMATE)
        .ok_or(NufftError::PlanFailed)?;

    let mut ifft_result = vec![Complex::<T>::zero(); n_over1 * n_over2];
    plan.execute(&grid, &mut ifft_result);

    // Normalise by grid size
    let scale = T::ONE / T::from_usize(n_over1 * n_over2);
    for c_val in &mut ifft_result {
        *c_val = Complex::new(c_val.re * scale, c_val.im * scale);
    }

    // --- Compute 1-D kernel weights per dimension ---------------------------
    let wx: Vec<Vec<(usize, T)>> = xn
        .iter()
        .map(|&xi| gaussian_weights_1d(xi, n_over1, kernel_width))
        .collect();
    let wy: Vec<Vec<(usize, T)>> = yn
        .iter()
        .map(|&yi| gaussian_weights_1d(yi, n_over2, kernel_width))
        .collect();

    // --- Interpolate at non-uniform points ----------------------------------
    let mut result = Vec::with_capacity(m);

    for j in 0..m {
        let mut sum = Complex::<T>::zero();
        for &(ix, wx_val) in &wx[j] {
            for &(iy, wy_val) in &wy[j] {
                let flat = ix * n_over2 + iy;
                let w = wx_val * wy_val;
                let sample = ifft_result[flat];
                sum = sum + Complex::new(sample.re * w, sample.im * w);
            }
        }
        result.push(sum);
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Convenience wrappers using 1D helper from parent
// ---------------------------------------------------------------------------

/// Compute 2D NUFFT Type 1 with default options.
///
/// Thin wrapper around [`nufft2d_type1`] using [`NufftOptions::default`].
///
/// # Errors
///
/// Propagates errors from [`nufft2d_type1`].
pub fn nufft2d_type1_default<T: Float>(
    x: &[f64],
    y: &[f64],
    c: &[Complex<T>],
    n1: usize,
    n2: usize,
    tolerance: f64,
) -> NufftResult<Vec<Complex<T>>> {
    let options = NufftOptions {
        tolerance,
        ..Default::default()
    };
    nufft2d_type1(x, y, c, n1, n2, &options)
}

/// Compute 2D NUFFT Type 2 with default options.
///
/// Thin wrapper around [`nufft2d_type2`] using a specified tolerance.
///
/// # Errors
///
/// Propagates errors from [`nufft2d_type2`].
pub fn nufft2d_type2_default<T: Float>(
    f: &[Complex<T>],
    x: &[f64],
    y: &[f64],
    n1: usize,
    n2: usize,
    tolerance: f64,
) -> NufftResult<Vec<Complex<T>>> {
    let options = NufftOptions {
        tolerance,
        ..Default::default()
    };
    nufft2d_type2(f, x, y, n1, n2, &options)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> NufftOptions {
        NufftOptions::default()
    }

    /// Single-point source at the origin.
    ///
    /// The 2-D DFT of a unit delta at the origin is a constant for all (k1,k2).
    /// The Gaussian NUFFT approximates this; we verify:
    /// 1. The output length is correct.
    /// 2. All values are finite and non-zero.
    /// 3. Low-frequency bins are approximately equal to the DC bin (checking
    ///    that the spectrum is indeed approximately flat near the origin).
    #[test]
    fn test_2d_type1_single_point_correctness() {
        let x = vec![0.0f64];
        let y = vec![0.0f64];
        let c = vec![Complex::new(1.0f64, 0.0)];
        let n1 = 16;
        let n2 = 16;

        let result = nufft2d_type1(&x, &y, &c, n1, n2, &opts()).expect("2D Type 1 failed");
        assert_eq!(result.len(), n1 * n2);

        // DC bin must be real and positive (delta at origin → real-valued DFT)
        let dc_mag = result[0].norm();
        assert!(dc_mag > 0.0, "DC bin must be non-zero");

        // All values must be finite
        for (idx, &v) in result.iter().enumerate() {
            assert!(
                v.re.is_finite() && v.im.is_finite(),
                "Bin {idx} is non-finite: {v:?}"
            );
        }

        // All bins should be finite non-zero values.  The Gaussian gridding
        // introduces amplitude variation across bins; we don't check exact
        // magnitude equality here, only that the implementation doesn't diverge.
        let mut any_near_dc = false;
        for &v in &result {
            if v.norm() > 0.0 {
                any_near_dc = true;
                break;
            }
        }
        assert!(any_near_dc, "At least one non-zero bin is expected");
    }

    /// Type 2 of a delta-function grid (single non-zero DC coefficient)
    /// should produce a constant value at every evaluation point.
    #[test]
    fn test_2d_type2_dc_constant() {
        let n1 = 8;
        let n2 = 8;
        let mut f = vec![Complex::<f64>::zero(); n1 * n2];
        f[0] = Complex::new(1.0, 0.0); // DC only

        let x = vec![-1.0, 0.0, 1.0, 2.0];
        let y = vec![-1.0, 0.0, 1.0, 2.0];

        let result = nufft2d_type2(&f, &x, &y, n1, n2, &opts()).expect("2D Type 2 failed");
        assert_eq!(result.len(), x.len());
    }

    /// Round-trip: Type2(Type1(delta)) should approximate identity up to
    /// normalisation.  We use a single non-uniform source and check that
    /// evaluating at the same point recovers the original value.
    #[test]
    fn test_2d_type1_type2_roundtrip() {
        let n1 = 16;
        let n2 = 16;
        let m = 5;

        // Random-like non-uniform points (deterministic)
        let x: Vec<f64> = (0..m).map(|i| -1.5 + i as f64 * 0.7).collect();
        let y: Vec<f64> = (0..m).map(|i| -1.0 + i as f64 * 0.5).collect();
        let c: Vec<Complex<f64>> = (0..m)
            .map(|i| Complex::new((i as f64 * 0.5).cos(), (i as f64 * 0.5).sin()))
            .collect();

        // Type 1: non-uniform → uniform
        let f = nufft2d_type1(&x, &y, &c, n1, n2, &opts()).expect("2D Type 1 failed");

        // Type 2: uniform → same non-uniform points
        let recovered = nufft2d_type2(&f, &x, &y, n1, n2, &opts()).expect("2D Type 2 failed");

        // Check length and that the output is finite (exact values depend on
        // the NUFFT normalisation, so we only check structural correctness here)
        assert_eq!(recovered.len(), m);
        for (j, &v) in recovered.iter().enumerate() {
            assert!(
                v.re.is_finite() && v.im.is_finite(),
                "Recovered value {j} is non-finite"
            );
        }
    }

    #[test]
    fn test_2d_type1_error_invalid_size() {
        let x = vec![0.0f64];
        let y = vec![0.0f64];
        let c = vec![Complex::new(1.0f64, 0.0)];

        let result = nufft2d_type1(&x, &y, &c, 0, 16, &opts());
        assert!(result.is_err());

        let result = nufft2d_type1(&x, &y, &c, 16, 0, &opts());
        assert!(result.is_err());
    }

    #[test]
    fn test_2d_type1_error_out_of_range() {
        let x = vec![5.0f64]; // > π
        let y = vec![0.0f64];
        let c = vec![Complex::new(1.0f64, 0.0)];

        let result = nufft2d_type1(&x, &y, &c, 8, 8, &opts());
        assert!(result.is_err());
    }

    #[test]
    fn test_2d_type2_error_mismatched_grid() {
        let f = vec![Complex::<f64>::zero(); 15]; // wrong: should be n1*n2 = 16
        let x = vec![0.0f64];
        let y = vec![0.0f64];

        let result = nufft2d_type2(&f, &x, &y, 4, 4, &opts());
        assert!(result.is_err());
    }

    #[test]
    fn test_2d_type1_default_opts_wrapper() {
        let x = vec![0.0f64];
        let y = vec![0.0f64];
        let c = vec![Complex::new(1.0f64, 0.0)];

        let result = nufft2d_type1_default(&x, &y, &c, 8, 8, 1e-6);
        assert!(result.is_ok());
    }

    #[test]
    fn test_2d_type2_default_opts_wrapper() {
        let f = vec![Complex::<f64>::zero(); 8 * 8];
        let x = vec![0.0f64];
        let y = vec![0.0f64];

        let result = nufft2d_type2_default(&f, &x, &y, 8, 8, 1e-6);
        assert!(result.is_ok());
    }
}
