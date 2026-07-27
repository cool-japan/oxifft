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
//! # Frequency-index convention
//!
//! Output index `k` along each dimension corresponds to the **centered**
//! frequency `freq = k - n/2` — the same convention documented and used by
//! the 1D [`crate::nufft::Nufft::type1`] / [`crate::nufft::Nufft::type2`]
//! API (and the FINUFFT Type 1/2 convention):
//!
//! ```text
//! k=0      -> freq = -n/2   (most negative)
//! k=n/2    -> freq = 0      (DC)
//! k=n-1    -> freq = n/2-1  (most positive)
//! ```
//!
//! This applies independently to each axis, so `result[k1 * n2 + k2]`
//! corresponds to the 2-D frequency `(k1 - n1/2, k2 - n2/2)`.
//!
//! # References
//!
//! Greengard, L. & Lee, J.-Y. (2004). Accelerating the nonuniform fast
//! Fourier transform. *SIAM Review*, 46(3), 443–454.

use crate::api::{Direction, Flags, Plan2D};
use crate::kernel::{Complex, Float};

use super::{
    centered_freq_indices, compute_kernel_width, next_smooth_number, precompute_deconv_factors,
    NufftError, NufftOptions, NufftResult,
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
    // β scales with W = half_width, not kernel_width, so that exp(-β) ≈ 0.10
    // at the kernel edges (j = ±W), giving adequate taper with low sub-grid error.
    let beta = 2.3 * (half_width as f64);

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
/// ```
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
    let kernel_width = compute_kernel_width(
        options.tolerance,
        options.oversampling,
        options.kernel_width,
    );
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

    // Cap deconvolution to prevent exponential blowup at high-frequency
    // corner bins.  The Gaussian kernel's FT decays to near-zero there, so
    // amplifying beyond 1/tolerance only magnifies rounding noise.
    let max_deconv = T::from_f64(1.0 / options.tolerance);

    let mut result = Vec::with_capacity(n1 * n2);

    for k1 in 0..n1 {
        // Map output index k1 to the oversampled grid index and FFT-order
        // deconvolution index using the centered frequency convention
        // (freq = k1 - n1/2), shared with the 1D and 3D NUFFT APIs.
        let (grid_idx1, deconv_idx1) = centered_freq_indices(k1, n1, n_over1);
        let d1 = if deconv1[deconv_idx1].re > max_deconv {
            Complex::new(max_deconv, T::ZERO)
        } else {
            deconv1[deconv_idx1]
        };

        for k2 in 0..n2 {
            let (grid_idx2, deconv_idx2) = centered_freq_indices(k2, n2, n_over2);

            let flat_grid = grid_idx1 * n_over2 + grid_idx2;
            // Product of 1-D deconvolution factors; cap each factor individually
            // to avoid double-exponential blowup in 2-D corner bins.
            let d2 = if deconv2[deconv_idx2].re > max_deconv {
                Complex::new(max_deconv, T::ZERO)
            } else {
                deconv2[deconv_idx2]
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
/// ```
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
    let kernel_width = compute_kernel_width(
        options.tolerance,
        options.oversampling,
        options.kernel_width,
    );
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

    // Cap individual 1-D deconvolution factors to prevent exponential blowup
    // at high-frequency corner bins of the oversampled 2-D grid.
    let max_deconv = T::from_f64(1.0 / options.tolerance);

    // Type 2 deconvolution differs from Type 1 by a factor equal to the total
    // oversampled grid size (same derivation as the 1D `type2` in
    // `crate::nufft::Nufft`, extended to 2D): after the properly-normalised
    // IFFT (1/(n_over1*n_over2)) and kernel interpolation, the output would
    // otherwise be too small by exactly that factor.  Multiplying the
    // deconvolution product by `n_over1*n_over2` here cancels it.
    let n_os_scale = T::from_usize(n_over1 * n_over2);

    let mut grid = vec![Complex::<T>::zero(); n_over1 * n_over2];

    for k1 in 0..n1 {
        // Centered frequency convention (freq = k1 - n1/2), matching
        // nufft2d_type1 and the 1D NUFFT API.
        let (grid_idx1, deconv_idx1) = centered_freq_indices(k1, n1, n_over1);
        let d1 = if deconv1[deconv_idx1].re > max_deconv {
            Complex::new(max_deconv, T::ZERO)
        } else {
            deconv1[deconv_idx1]
        };

        for k2 in 0..n2 {
            let (grid_idx2, deconv_idx2) = centered_freq_indices(k2, n2, n_over2);

            let flat_in = k1 * n2 + k2;
            let flat_grid = grid_idx1 * n_over2 + grid_idx2;
            let d2 = if deconv2[deconv_idx2].re > max_deconv {
                Complex::new(max_deconv, T::ZERO)
            } else {
                deconv2[deconv_idx2]
            };
            grid[flat_grid] = f[flat_in] * d1 * d2 * n_os_scale;
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

    // -----------------------------------------------------------------------
    // Dense NDFT reference (O(n1*n2*m)), used to numerically validate the
    // Gaussian-gridding NUFFT against ground truth rather than only checking
    // shape/finiteness.  Uses the same centered frequency convention
    // (freq = k - n/2) documented on the module and implemented via
    // `centered_freq_indices`.
    // -----------------------------------------------------------------------

    /// Dense 2-D NDFT Type 1 (non-uniform -> uniform).
    ///
    /// `f_hat[k1,k2] = sum_j c[j] * exp(-i*(freq1*x[j] + freq2*y[j]))`
    fn dense_ndft2d_type1(
        x: &[f64],
        y: &[f64],
        c: &[Complex<f64>],
        n1: usize,
        n2: usize,
    ) -> Vec<Complex<f64>> {
        let half1 = (n1 / 2) as isize;
        let half2 = (n2 / 2) as isize;
        let mut out = Vec::with_capacity(n1 * n2);
        for k1 in 0..n1 {
            let freq1 = (k1 as isize - half1) as f64;
            for k2 in 0..n2 {
                let freq2 = (k2 as isize - half2) as f64;
                let mut acc = Complex::new(0.0_f64, 0.0);
                for (j, &cj) in c.iter().enumerate() {
                    let angle = -(freq1 * x[j] + freq2 * y[j]);
                    acc = acc + cj * Complex::new(angle.cos(), angle.sin());
                }
                out.push(acc);
            }
        }
        out
    }

    /// Dense 2-D NDFT Type 2 (uniform -> non-uniform).
    ///
    /// `f[j] = sum_{k1,k2} f_hat[k1,k2] * exp(+i*(freq1*x[j] + freq2*y[j]))`
    fn dense_ndft2d_type2(
        f: &[Complex<f64>],
        x: &[f64],
        y: &[f64],
        n1: usize,
        n2: usize,
    ) -> Vec<Complex<f64>> {
        let half1 = (n1 / 2) as isize;
        let half2 = (n2 / 2) as isize;
        x.iter()
            .zip(y.iter())
            .map(|(&xj, &yj)| {
                let mut acc = Complex::new(0.0_f64, 0.0);
                for k1 in 0..n1 {
                    let freq1 = (k1 as isize - half1) as f64;
                    for k2 in 0..n2 {
                        let freq2 = (k2 as isize - half2) as f64;
                        let angle = freq1 * xj + freq2 * yj;
                        acc = acc + f[k1 * n2 + k2] * Complex::new(angle.cos(), angle.sin());
                    }
                }
                acc
            })
            .collect()
    }

    /// Maximum `|nufft[i] - ref[i]| / max(|ref[i]|)` over all bins.
    fn max_relative_error(nufft_out: &[Complex<f64>], reference: &[Complex<f64>]) -> f64 {
        let ref_max = reference.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
        if ref_max < 1e-30 {
            return 0.0;
        }
        nufft_out
            .iter()
            .zip(reference.iter())
            .map(|(n, r)| (*n - *r).norm() / ref_max)
            .fold(0.0_f64, f64::max)
    }

    /// Relative-error headroom for the default `opts()` (tol=1e-6, os=2.0).
    ///
    /// Empirically the 2-D Gaussian-gridding error is of the same order as
    /// the 1-D case (see `oxifft/tests/nufft_tolerance_sweep.rs`) since the
    /// kernel is separable per axis; `tol * 10` gives comfortable headroom
    /// without masking real regressions.
    fn headroom() -> f64 {
        opts().tolerance * 10.0
    }

    // -----------------------------------------------------------------------
    // Type 1 numeric correctness
    // -----------------------------------------------------------------------

    /// Single-point source at the origin: the dense 2-D NDFT of a unit delta
    /// at the origin is exactly `1.0` for every (k1,k2) bin (all phases are
    /// zero).  This directly validates the frequency-index convention: a bug
    /// in the k-to-frequency mapping would leave the *set* of output values
    /// unchanged (still all equal) but this test additionally cross-checks
    /// magnitude against the dense reference at every bin.
    #[test]
    fn test_2d_type1_single_point_matches_dense_ndft() {
        let x = vec![0.0f64];
        let y = vec![0.0f64];
        let c = vec![Complex::new(1.0f64, 0.0)];
        let n1 = 16;
        let n2 = 16;

        let result = nufft2d_type1(&x, &y, &c, n1, n2, &opts()).expect("2D Type 1 failed");
        let reference = dense_ndft2d_type1(&x, &y, &c, n1, n2);
        assert_eq!(result.len(), reference.len());

        let rel_err = max_relative_error(&result, &reference);
        assert!(
            rel_err <= headroom(),
            "single-point-at-origin rel_err {rel_err:.2e} exceeds headroom {:.2e}",
            headroom()
        );
    }

    /// Random (deterministic) non-uniform points, Type 1 vs. dense NDFT.
    #[test]
    fn test_2d_type1_random_points_matches_dense_ndft() {
        let n1 = 16;
        let n2 = 16;
        let m = 12;

        let x: Vec<f64> = (0..m).map(|i| -2.9 + (i as f64) * 0.5).collect();
        let y: Vec<f64> = (0..m).map(|i| -2.5 + (i as f64) * 0.43).collect();
        let c: Vec<Complex<f64>> = (0..m)
            .map(|i| Complex::new(((i as f64) * 0.37).cos(), ((i as f64) * 0.61).sin()))
            .collect();

        let result = nufft2d_type1(&x, &y, &c, n1, n2, &opts()).expect("2D Type 1 failed");
        let reference = dense_ndft2d_type1(&x, &y, &c, n1, n2);

        let rel_err = max_relative_error(&result, &reference);
        assert!(
            rel_err <= headroom(),
            "random-points rel_err {rel_err:.2e} exceeds headroom {:.2e}",
            headroom()
        );
    }

    /// Points exactly at the domain edges (`-π`, `π`) must not panic and
    /// must still match the dense reference.
    #[test]
    fn test_2d_type1_edge_points_matches_dense_ndft() {
        let pi = core::f64::consts::PI;
        let x = vec![-pi, pi, 0.0];
        let y = vec![pi, -pi, 0.0];
        let c = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.5, -0.5),
            Complex::new(-0.3, 0.2),
        ];
        let n1 = 16;
        let n2 = 16;

        let result =
            nufft2d_type1(&x, &y, &c, n1, n2, &opts()).expect("edge-point 2D Type 1 failed");
        let reference = dense_ndft2d_type1(&x, &y, &c, n1, n2);

        let rel_err = max_relative_error(&result, &reference);
        assert!(
            rel_err <= headroom(),
            "edge-points rel_err {rel_err:.2e} exceeds headroom {:.2e}",
            headroom()
        );
    }

    /// Two coincident non-uniform points with values `c1`,`c2` must produce
    /// exactly the same result (up to floating rounding) as a single point
    /// at the same location with value `c1+c2` — the spreading kernel is
    /// linear in the input strengths and depends only on point location.
    #[test]
    fn test_2d_type1_coincident_points_linearity() {
        let n1 = 16;
        let n2 = 16;
        let x_single = vec![0.4f64];
        let y_single = vec![-0.3f64];
        let c_combined = vec![Complex::new(1.7, -0.5)];

        let x_dup = vec![0.4f64, 0.4f64];
        let y_dup = vec![-0.3f64, -0.3f64];
        let c_dup = vec![Complex::new(1.0, -0.2), Complex::new(0.7, -0.3)];

        let result_single = nufft2d_type1(&x_single, &y_single, &c_combined, n1, n2, &opts())
            .expect("single failed");
        let result_dup =
            nufft2d_type1(&x_dup, &y_dup, &c_dup, n1, n2, &opts()).expect("dup failed");

        for (a, b) in result_single.iter().zip(result_dup.iter()) {
            assert!(
                (*a - *b).norm() < 1e-9,
                "coincident-point linearity violated: {a:?} vs {b:?}"
            );
        }
    }

    /// Empty non-uniform input must yield an all-zero uniform grid (not a
    /// panic or garbage output).
    #[test]
    fn test_2d_type1_empty_input_returns_zero_grid() {
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];
        let c: Vec<Complex<f64>> = vec![];
        let n1 = 8;
        let n2 = 8;

        let result = nufft2d_type1(&x, &y, &c, n1, n2, &opts()).expect("empty 2D Type 1 failed");
        assert_eq!(result.len(), n1 * n2);
        for v in &result {
            assert_eq!(v.re, 0.0);
            assert_eq!(v.im, 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // Type 2 numeric correctness
    // -----------------------------------------------------------------------

    /// Type 2 of a delta-function grid (single non-zero DC coefficient, at
    /// the centered-convention DC index `(n1/2, n2/2)`) must produce exactly
    /// the coefficient value at every evaluation point (DC has zero phase
    /// everywhere).
    #[test]
    fn test_2d_type2_dc_constant_matches_dense_ndft() {
        let n1 = 8;
        let n2 = 8;
        let (half1, half2) = (n1 / 2, n2 / 2);
        let mut f = vec![Complex::<f64>::zero(); n1 * n2];
        f[half1 * n2 + half2] = Complex::new(1.0, 0.0); // DC (centered convention)

        let x = vec![-1.0, 0.0, 1.0, 2.0];
        let y = vec![-1.0, 0.0, 1.0, 2.0];

        let result = nufft2d_type2(&f, &x, &y, n1, n2, &opts()).expect("2D Type 2 failed");
        assert_eq!(result.len(), x.len());

        for (j, v) in result.iter().enumerate() {
            assert!(
                (v.re - 1.0).abs() <= headroom(),
                "point {j}: expected re~1.0, got {v:?}"
            );
            assert!(
                v.im.abs() <= headroom(),
                "point {j}: expected im~0, got {v:?}"
            );
        }
    }

    /// Random (deterministic) uniform coefficients, Type 2 vs. dense NDFT.
    #[test]
    fn test_2d_type2_random_points_matches_dense_ndft() {
        let n1 = 16;
        let n2 = 16;
        let f: Vec<Complex<f64>> = (0..n1 * n2)
            .map(|i| {
                Complex::new(
                    ((i as f64) * 0.017).sin() * 0.1,
                    ((i as f64) * 0.013).cos() * 0.1,
                )
            })
            .collect();

        let m = 10;
        let x: Vec<f64> = (0..m).map(|i| -2.7 + (i as f64) * 0.55).collect();
        let y: Vec<f64> = (0..m).map(|i| -2.2 + (i as f64) * 0.41).collect();

        let result = nufft2d_type2(&f, &x, &y, n1, n2, &opts()).expect("2D Type 2 failed");
        let reference = dense_ndft2d_type2(&f, &x, &y, n1, n2);

        let rel_err = max_relative_error(&result, &reference);
        assert!(
            rel_err <= headroom(),
            "type2 random-points rel_err {rel_err:.2e} exceeds headroom {:.2e}",
            headroom()
        );
    }

    /// Empty evaluation-point input must yield an empty result (not a panic).
    #[test]
    fn test_2d_type2_empty_points_returns_empty() {
        let n1 = 8;
        let n2 = 8;
        let f = vec![Complex::<f64>::zero(); n1 * n2];
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];

        let result = nufft2d_type2(&f, &x, &y, n1, n2, &opts()).expect("empty 2D Type 2 failed");
        assert!(result.is_empty());
    }

    /// Type2(Type1(x)) is not an exact identity in general (Type 1 is the
    /// adjoint, not the inverse, of Type 2), but for a well-resolved signal
    /// (few, well-separated points on a moderately oversampled grid) it
    /// should recover values that are finite and of comparable magnitude to
    /// the input — used here as a numerical-stability smoke test in addition
    /// to the independent dense-NDFT comparisons above.
    #[test]
    fn test_2d_type1_type2_roundtrip_bounded() {
        let n1 = 16;
        let n2 = 16;
        let m = 5;

        let x: Vec<f64> = (0..m).map(|i| -1.5 + (i as f64) * 0.7).collect();
        let y: Vec<f64> = (0..m).map(|i| -1.0 + (i as f64) * 0.5).collect();
        let c: Vec<Complex<f64>> = (0..m)
            .map(|i| Complex::new(((i as f64) * 0.5).cos(), ((i as f64) * 0.5).sin()))
            .collect();
        let input_sum: f64 = c.iter().map(|c| c.norm()).sum();

        let f = nufft2d_type1(&x, &y, &c, n1, n2, &opts()).expect("2D Type 1 failed");
        let recovered = nufft2d_type2(&f, &x, &y, n1, n2, &opts()).expect("2D Type 2 failed");

        assert_eq!(recovered.len(), m);
        // Very loose upper bound (order of magnitude, not tight accuracy):
        // this is a numerical-stability smoke test guarding against gross
        // scaling errors (e.g. an accidental extra factor of the oversampled
        // grid size or of `max_deconv`), not a precision check — the
        // dense-NDFT comparisons above cover precision.
        let loose_bound = (input_sum * (n1 * n2) as f64).max(1.0) * 10.0;
        for (j, &v) in recovered.iter().enumerate() {
            assert!(
                v.re.is_finite() && v.im.is_finite(),
                "Recovered value {j} is non-finite"
            );
            assert!(
                v.norm() <= loose_bound,
                "Recovered value {j} magnitude {} unexpectedly large (bound {loose_bound})",
                v.norm()
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
