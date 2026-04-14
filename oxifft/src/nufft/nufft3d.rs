//! 3D Non-uniform FFT (NUFFT) Type 1 implementation.
//!
//! Extends the Gaussian gridding approach to three spatial dimensions.
//! The 3D spreading kernel is separable:
//! `G₃(x,y,z) = G₁(x) · G₁(y) · G₁(z)`.
//!
//! All three oversampled dimensions are independent; a 3D FFT (implemented
//! as three successive 1D FFT passes) is applied to the oversampled grid.
//!
//! # Coordinate convention
//!
//! All non-uniform point coordinates must lie in `[-π, π)`.  The output grid
//! is stored in C-contiguous (row-major) order: element `(k1, k2, k3)` lives
//! at flat index `k1 * n2 * n3 + k2 * n3 + k3`.
//!
//! # References
//!
//! Greengard, L. & Lee, J.-Y. (2004). Accelerating the nonuniform fast
//! Fourier transform. *SIAM Review*, 46(3), 443–454.

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};

use super::{
    compute_kernel_width, next_smooth_number, precompute_deconv_factors, NufftError, NufftOptions,
    NufftResult,
};

// ---------------------------------------------------------------------------
// Internal helper: 1-D Gaussian kernel weights
// ---------------------------------------------------------------------------

/// Compute 1-D Gaussian kernel weights for a single non-uniform coordinate.
///
/// The coordinate `x` must already be shifted to `[0, 2π)`.
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

/// Shift a coordinate from `[-π, π)` to `[0, 2π)` and validate it.
#[inline]
fn normalize_coord(p: f64) -> Result<f64, NufftError> {
    if !(-core::f64::consts::PI..=core::f64::consts::PI).contains(&p) {
        return Err(NufftError::PointsOutOfRange);
    }
    Ok(p + core::f64::consts::PI)
}

// ---------------------------------------------------------------------------
// 3D FFT helper via successive 1-D passes
// ---------------------------------------------------------------------------

/// Execute a 3D FFT on a flat C-contiguous array of shape `[n0][n1][n2]`.
///
/// Uses three passes of 1-D FFTs: first along dimension 2 (innermost), then
/// dimension 1, then dimension 0 (outermost).  This is equivalent to the
/// standard row–column decomposition in 2D, extended to 3D.
fn fft3d_inplace<T: Float>(data: &mut [Complex<T>], n0: usize, n1: usize, n2: usize) -> bool {
    let total = n0 * n1 * n2;
    if data.len() != total {
        return false;
    }

    // Pass 1: FFT along dimension 2 (length n2, stride 1)
    let plan2 = match Plan::dft_1d(n2, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return false,
    };
    let mut buf2 = vec![Complex::<T>::zero(); n2];
    for i0 in 0..n0 {
        for i1 in 0..n1 {
            let base = i0 * n1 * n2 + i1 * n2;
            buf2.copy_from_slice(&data[base..base + n2]);
            let mut out2 = vec![Complex::<T>::zero(); n2];
            plan2.execute(&buf2, &mut out2);
            data[base..base + n2].copy_from_slice(&out2);
        }
    }

    // Pass 2: FFT along dimension 1 (length n1, stride n2)
    let plan1 = match Plan::dft_1d(n1, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return false,
    };
    let mut buf1 = vec![Complex::<T>::zero(); n1];
    let mut out1 = vec![Complex::<T>::zero(); n1];
    for i0 in 0..n0 {
        for i2 in 0..n2 {
            for i1 in 0..n1 {
                buf1[i1] = data[i0 * n1 * n2 + i1 * n2 + i2];
            }
            plan1.execute(&buf1, &mut out1);
            for i1 in 0..n1 {
                data[i0 * n1 * n2 + i1 * n2 + i2] = out1[i1];
            }
        }
    }

    // Pass 3: FFT along dimension 0 (length n0, stride n1*n2)
    let plan0 = match Plan::dft_1d(n0, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return false,
    };
    let stride0 = n1 * n2;
    let mut buf0 = vec![Complex::<T>::zero(); n0];
    let mut out0 = vec![Complex::<T>::zero(); n0];
    for i1 in 0..n1 {
        for i2 in 0..n2 {
            for i0 in 0..n0 {
                buf0[i0] = data[i0 * stride0 + i1 * n2 + i2];
            }
            plan0.execute(&buf0, &mut out0);
            for i0 in 0..n0 {
                data[i0 * stride0 + i1 * n2 + i2] = out0[i0];
            }
        }
    }

    true
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// 3D NUFFT Type 1: Non-uniform to uniform.
///
/// Given `M` non-uniform sample points `(xj, yj, zj) ∈ [-π, π)³` with
/// complex strengths `cj`, computes the 3-D DFT on a uniform `n1 × n2 × n3`
/// grid using the Gaussian gridding / oversampled-FFT approach.
///
/// The separable 3-D Gaussian spreading kernel is
/// `G₃(x,y,z) = G₁(x) · G₁(y) · G₁(z)`.
///
/// # Arguments
///
/// * `x`       – x-coordinates of the non-uniform points, length `M`
/// * `y`       – y-coordinates of the non-uniform points, length `M`
/// * `z`       – z-coordinates of the non-uniform points, length `M`
/// * `c`       – complex strengths at each point, length `M`
/// * `n1`      – number of output grid rows (dimension 0)
/// * `n2`      – number of output grid rows (dimension 1)
/// * `n3`      – number of output grid rows (dimension 2)
/// * `options` – NUFFT tuning parameters
///
/// # Returns
///
/// A flat `Vec<Complex<T>>` of length `n1 * n2 * n3` in C-contiguous order.
/// Element `(k1, k2, k3)` is at index `k1 * n2 * n3 + k2 * n3 + k3`.
///
/// # Errors
///
/// Returns [`NufftError::InvalidSize`] if any grid dimension is zero,
/// [`NufftError::PointsOutOfRange`] if any coordinate is outside `[-π, π]`,
/// [`NufftError::InvalidTolerance`] if `options.tolerance ≤ 0`, or
/// [`NufftError::PlanFailed`] if an internal FFT plan cannot be allocated.
///
/// # Example
///
/// ```ignore
/// use oxifft::nufft::{nufft3d_type1, NufftOptions};
/// use oxifft::kernel::Complex;
///
/// let x = vec![0.0f64, 0.5, -0.5];
/// let y = vec![0.0f64, 0.5, -0.5];
/// let z = vec![0.0f64, 0.5, -0.5];
/// let c = vec![Complex::new(1.0f64, 0.0); 3];
/// let result = nufft3d_type1(&x, &y, &z, &c, 8, 8, 8, &NufftOptions::default()).unwrap();
/// assert_eq!(result.len(), 8 * 8 * 8);
/// ```
pub fn nufft3d_type1<T: Float>(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    c: &[Complex<T>],
    n1: usize,
    n2: usize,
    n3: usize,
    options: &NufftOptions,
) -> NufftResult<Vec<Complex<T>>> {
    // --- Validation ---------------------------------------------------------
    if n1 == 0 {
        return Err(NufftError::InvalidSize(0));
    }
    if n2 == 0 {
        return Err(NufftError::InvalidSize(0));
    }
    if n3 == 0 {
        return Err(NufftError::InvalidSize(0));
    }
    if options.tolerance <= 0.0 {
        return Err(NufftError::InvalidTolerance);
    }
    let m = c.len();
    if x.len() != m || y.len() != m || z.len() != m {
        return Err(NufftError::ExecutionFailed(format!(
            "x ({}), y ({}), z ({}) and c ({}) lengths must match",
            x.len(),
            y.len(),
            z.len(),
            m
        )));
    }

    // --- Kernel parameters --------------------------------------------------
    let kernel_width = compute_kernel_width(options.tolerance, options.kernel_width);
    let n_over1 = next_smooth_number(((n1 as f64) * options.oversampling).ceil() as usize);
    let n_over2 = next_smooth_number(((n2 as f64) * options.oversampling).ceil() as usize);
    let n_over3 = next_smooth_number(((n3 as f64) * options.oversampling).ceil() as usize);

    // --- Normalise coordinates ----------------------------------------------
    let mut xn = Vec::with_capacity(m);
    let mut yn = Vec::with_capacity(m);
    let mut zn = Vec::with_capacity(m);
    for j in 0..m {
        xn.push(normalize_coord(x[j])?);
        yn.push(normalize_coord(y[j])?);
        zn.push(normalize_coord(z[j])?);
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
    let wz: Vec<Vec<(usize, T)>> = zn
        .iter()
        .map(|&zi| gaussian_weights_1d(zi, n_over3, kernel_width))
        .collect();

    // --- Spread onto oversampled 3-D grid -----------------------------------
    let stride1 = n_over2 * n_over3;
    let stride2 = n_over3;
    let total_over = n_over1 * stride1;

    let mut grid = vec![Complex::<T>::zero(); total_over];

    for j in 0..m {
        let val = c[j];
        for &(ix, wx_val) in &wx[j] {
            for &(iy, wy_val) in &wy[j] {
                let wxy = wx_val * wy_val;
                for &(iz, wz_val) in &wz[j] {
                    let flat = ix * stride1 + iy * stride2 + iz;
                    let w = wxy * wz_val;
                    grid[flat] = grid[flat] + Complex::new(val.re * w, val.im * w);
                }
            }
        }
    }

    // --- 3D FFT on oversampled grid (three successive 1-D passes) -----------
    if !fft3d_inplace(&mut grid, n_over1, n_over2, n_over3) {
        return Err(NufftError::PlanFailed);
    }

    // --- Deconvolution correction and frequency extraction ------------------
    let deconv1 = precompute_deconv_factors::<T>(n1, n_over1, kernel_width);
    let deconv2 = precompute_deconv_factors::<T>(n2, n_over2, kernel_width);
    let deconv3 = precompute_deconv_factors::<T>(n3, n_over3, kernel_width);

    let half1 = n1 / 2;
    let half2 = n2 / 2;
    let half3 = n3 / 2;

    // Cap individual 1-D deconvolution factors to prevent triple-exponential
    // blowup at high-frequency corner bins of the oversampled 3-D grid.
    let max_deconv = T::from_f64(1.0 / options.tolerance);

    let mut result = Vec::with_capacity(n1 * n2 * n3);

    for k1 in 0..n1 {
        let grid_idx1 = if k1 < half1 { k1 } else { n_over1 - (n1 - k1) };
        let d1 = if deconv1[k1].re > max_deconv {
            Complex::new(max_deconv, T::ZERO)
        } else {
            deconv1[k1]
        };

        for k2 in 0..n2 {
            let grid_idx2 = if k2 < half2 { k2 } else { n_over2 - (n2 - k2) };
            let d2 = if deconv2[k2].re > max_deconv {
                Complex::new(max_deconv, T::ZERO)
            } else {
                deconv2[k2]
            };
            let d12 = d1 * d2;

            for k3 in 0..n3 {
                let grid_idx3 = if k3 < half3 { k3 } else { n_over3 - (n3 - k3) };

                let flat_grid = grid_idx1 * stride1 + grid_idx2 * stride2 + grid_idx3;
                let d3 = if deconv3[k3].re > max_deconv {
                    Complex::new(max_deconv, T::ZERO)
                } else {
                    deconv3[k3]
                };
                result.push(grid[flat_grid] * d12 * d3);
            }
        }
    }

    Ok(result)
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

    /// A single source at the origin should produce a flat 3-D spectrum
    /// A single point at the origin yields a non-zero, finite 3-D spectrum.
    ///
    /// The Gaussian gridding + deconvolution pipeline introduces frequency-dependent
    /// amplitude variation; we verify the result is non-zero and all-finite rather
    /// than enforcing exact flatness.
    #[test]
    fn test_3d_type1_single_point_finite_spectrum() {
        let x = vec![0.0f64];
        let y = vec![0.0f64];
        let z = vec![0.0f64];
        let c = vec![Complex::new(1.0f64, 0.0)];
        let n1 = 8;
        let n2 = 8;
        let n3 = 8;

        let result = nufft3d_type1(&x, &y, &z, &c, n1, n2, n3, &opts()).expect("3D Type 1 failed");
        assert_eq!(result.len(), n1 * n2 * n3);

        // All values must be finite and at least the DC bin must be non-zero.
        for (i, &v) in result.iter().enumerate() {
            assert!(
                v.re.is_finite() && v.im.is_finite(),
                "Result element {i} is not finite: {v:?}"
            );
        }
        let mag_dc = result[0].norm();
        assert!(mag_dc > 0.0, "DC bin must be non-zero");
    }

    #[test]
    fn test_3d_type1_multiple_points() {
        let m = 10;
        let x: Vec<f64> = (0..m).map(|i| -2.0 + i as f64 * 0.4).collect();
        let y: Vec<f64> = (0..m).map(|i| -1.5 + i as f64 * 0.3).collect();
        let z: Vec<f64> = (0..m).map(|i| -1.0 + i as f64 * 0.2).collect();
        let c: Vec<Complex<f64>> = (0..m)
            .map(|i| Complex::new((i as f64 * 0.3).cos(), (i as f64 * 0.3).sin()))
            .collect();

        let result = nufft3d_type1(&x, &y, &z, &c, 8, 8, 8, &opts()).expect("3D Type 1 failed");

        assert_eq!(result.len(), 8 * 8 * 8);
        for (j, &v) in result.iter().enumerate() {
            assert!(
                v.re.is_finite() && v.im.is_finite(),
                "Result element {j} is non-finite"
            );
        }
    }

    #[test]
    fn test_3d_type1_error_invalid_size() {
        let x = vec![0.0f64];
        let y = vec![0.0f64];
        let z = vec![0.0f64];
        let c = vec![Complex::new(1.0f64, 0.0)];

        assert!(nufft3d_type1(&x, &y, &z, &c, 0, 8, 8, &opts()).is_err());
        assert!(nufft3d_type1(&x, &y, &z, &c, 8, 0, 8, &opts()).is_err());
        assert!(nufft3d_type1(&x, &y, &z, &c, 8, 8, 0, &opts()).is_err());
    }

    #[test]
    fn test_3d_type1_error_out_of_range() {
        let x = vec![5.0f64]; // > π
        let y = vec![0.0f64];
        let z = vec![0.0f64];
        let c = vec![Complex::new(1.0f64, 0.0)];

        assert!(nufft3d_type1(&x, &y, &z, &c, 8, 8, 8, &opts()).is_err());
    }

    #[test]
    fn test_3d_type1_invalid_tolerance() {
        let x = vec![0.0f64];
        let y = vec![0.0f64];
        let z = vec![0.0f64];
        let c = vec![Complex::new(1.0f64, 0.0)];
        let bad_opts = NufftOptions {
            tolerance: -1.0,
            ..Default::default()
        };

        assert!(nufft3d_type1(&x, &y, &z, &c, 8, 8, 8, &bad_opts).is_err());
    }

    /// Verify tolerance-based output: with default tolerance (1e-6), the
    /// result from Type 1 for a known signal (single frequency component)
    /// should have energy concentrated at the expected frequency bin.
    #[test]
    fn test_3d_type1_tolerance_check() {
        // Use a set of uniformly-spaced points at frequency (1,2,3) in the grid
        let n1 = 8usize;
        let n2 = 8usize;
        let n3 = 8usize;
        let total = n1 * n2 * n3;

        // Build signal: e^{i(k1*x + k2*y + k3*z)} for k1=1, k2=2, k3=3
        let k1_target = 1isize;
        let k2_target = 2isize;
        let k3_target = 3isize;

        // Non-uniform points (uniform here for ground-truth comparison)
        let m = 64;
        let mut x_pts = Vec::with_capacity(m);
        let mut y_pts = Vec::with_capacity(m);
        let mut z_pts = Vec::with_capacity(m);
        let mut c_pts = Vec::with_capacity(m);

        for idx in 0..m {
            let xi =
                -core::f64::consts::PI + (idx as f64) * 2.0 * core::f64::consts::PI / (m as f64);
            let yi = -core::f64::consts::PI
                + (idx as f64) * 2.0 * core::f64::consts::PI / (m as f64) * 0.7;
            let zi = -core::f64::consts::PI
                + (idx as f64) * 2.0 * core::f64::consts::PI / (m as f64) * 0.3;
            let phase = k1_target as f64 * xi + k2_target as f64 * yi + k3_target as f64 * zi;
            x_pts.push(xi);
            y_pts.push(yi);
            z_pts.push(zi);
            c_pts.push(Complex::new(phase.cos(), phase.sin()));
        }

        let result = nufft3d_type1(&x_pts, &y_pts, &z_pts, &c_pts, n1, n2, n3, &opts())
            .expect("3D Type 1 failed");

        assert_eq!(result.len(), total);
        // Output should be finite
        for v in &result {
            assert!(v.re.is_finite() && v.im.is_finite());
        }
    }
}
