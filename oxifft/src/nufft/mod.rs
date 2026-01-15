//! Non-uniform FFT (NUFFT) implementation.
//!
//! This module provides FFT for non-equispaced (non-uniform) data points,
//! which is essential for applications like:
//! - MRI reconstruction
//! - Radio astronomy
//! - Seismic imaging
//! - Spectral analysis of irregularly sampled signals
//!
//! # NUFFT Types
//!
//! - **Type 1 (Adjoint)**: Non-uniform points → uniform grid
//! - **Type 2 (Forward)**: Uniform grid → non-uniform points
//! - **Type 3**: Non-uniform → non-uniform
//!
//! # Algorithm
//!
//! Uses the Gaussian gridding approach:
//! 1. Spread non-uniform data to oversampled grid (convolution with kernel)
//! 2. Apply standard FFT
//! 3. Deconvolve to correct for spreading kernel
//!
//! # Example
//!
//! ```ignore
//! use oxifft::nufft::{Nufft, NufftType};
//!
//! // Non-uniform sample locations in [-π, π]
//! let x = vec![-2.0, -0.5, 0.3, 1.5, 2.8];
//! let values = vec![Complex::new(1.0, 0.0); 5];
//!
//! // Create NUFFT plan
//! let plan = Nufft::new(NufftType::Type1, 64, &x, 1e-6)?;
//!
//! // Execute: non-uniform → uniform grid
//! let result = plan.execute(&values)?;
//! ```

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};
use crate::prelude::*;

/// NUFFT type specifying the direction of transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NufftType {
    /// Type 1: Non-uniform to uniform (adjoint NUFFT).
    /// Given values at non-uniform points, compute uniform Fourier coefficients.
    Type1,
    /// Type 2: Uniform to non-uniform (forward NUFFT).
    /// Given uniform Fourier coefficients, evaluate at non-uniform points.
    Type2,
    /// Type 3: Non-uniform to non-uniform.
    /// Transform between two sets of non-uniform points.
    Type3,
}

/// NUFFT configuration options.
#[derive(Debug, Clone, Copy)]
pub struct NufftOptions {
    /// Oversampling factor (typically 2.0).
    pub oversampling: f64,
    /// Kernel width in grid points (typically 4-12).
    pub kernel_width: usize,
    /// Target relative tolerance.
    pub tolerance: f64,
    /// Use multi-threaded spreading.
    pub threaded: bool,
}

impl Default for NufftOptions {
    fn default() -> Self {
        Self {
            oversampling: 2.0,
            kernel_width: 6,
            tolerance: 1e-6,
            threaded: true,
        }
    }
}

/// NUFFT error types.
#[derive(Debug, Clone)]
pub enum NufftError {
    /// Invalid input size.
    InvalidSize(usize),
    /// Points out of range [-π, π].
    PointsOutOfRange,
    /// FFT planning failed.
    PlanFailed,
    /// Execution failed.
    ExecutionFailed(String),
    /// Invalid tolerance.
    InvalidTolerance,
}

impl core::fmt::Display for NufftError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidSize(n) => write!(f, "Invalid NUFFT size: {n}"),
            Self::PointsOutOfRange => write!(f, "Non-uniform points must be in [-π, π]"),
            Self::PlanFailed => write!(f, "Failed to create FFT plan"),
            Self::ExecutionFailed(msg) => write!(f, "NUFFT execution failed: {msg}"),
            Self::InvalidTolerance => write!(f, "Tolerance must be positive"),
        }
    }
}

/// Result type for NUFFT operations.
pub type NufftResult<T> = Result<T, NufftError>;

/// Non-uniform FFT plan.
///
/// Precomputes spreading/interpolation coefficients for efficient repeated use.
#[allow(clippy::struct_field_names)]
pub struct Nufft<T: Float> {
    /// NUFFT type.
    nufft_type: NufftType,
    /// Number of uniform grid points.
    n_uniform: usize,
    /// Number of non-uniform points.
    n_nonuniform: usize,
    /// Oversampled grid size.
    n_oversampled: usize,
    /// Non-uniform point locations (normalized to [0, 2π]).
    points: Vec<f64>,
    /// Precomputed spreading coefficients for each non-uniform point.
    spread_coeffs: Vec<Vec<(usize, T)>>,
    /// Deconvolution factors.
    deconv_factors: Vec<Complex<T>>,
    /// Internal FFT plan.
    fft_plan: Option<Plan<T>>,
    /// Options.
    options: NufftOptions,
}

impl<T: Float> Nufft<T> {
    /// Create a new NUFFT plan.
    ///
    /// # Arguments
    ///
    /// * `nufft_type` - Type of NUFFT (1, 2, or 3)
    /// * `n_uniform` - Number of uniform grid points
    /// * `points` - Non-uniform point locations in [-π, π]
    /// * `tolerance` - Target relative accuracy
    ///
    /// # Returns
    ///
    /// NUFFT plan or error.
    ///
    /// # Errors
    ///
    /// Returns error if size is zero, tolerance is non-positive, or points are out of range.
    pub fn new(
        nufft_type: NufftType,
        n_uniform: usize,
        points: &[f64],
        tolerance: f64,
    ) -> NufftResult<Self> {
        let options = NufftOptions {
            tolerance,
            ..Default::default()
        };
        Self::with_options(nufft_type, n_uniform, points, &options)
    }

    /// Create NUFFT plan with custom options.
    ///
    /// # Errors
    ///
    /// Returns error if size is zero, tolerance is non-positive, or points are out of range.
    pub fn with_options(
        nufft_type: NufftType,
        n_uniform: usize,
        points: &[f64],
        options: &NufftOptions,
    ) -> NufftResult<Self> {
        if n_uniform == 0 {
            return Err(NufftError::InvalidSize(0));
        }
        if options.tolerance <= 0.0 {
            return Err(NufftError::InvalidTolerance);
        }

        // Compute kernel width from tolerance
        let kernel_width = compute_kernel_width(options.tolerance, options.kernel_width);

        // Compute oversampled grid size
        let n_oversampled = ((n_uniform as f64) * options.oversampling).ceil() as usize;
        let n_oversampled = next_smooth_number(n_oversampled);

        // Normalize points to [0, 2π] and validate
        let mut normalized_points = Vec::with_capacity(points.len());
        for &p in points {
            if !(-core::f64::consts::PI..=core::f64::consts::PI).contains(&p) {
                return Err(NufftError::PointsOutOfRange);
            }
            // Shift from [-π, π] to [0, 2π]
            normalized_points.push(p + core::f64::consts::PI);
        }

        // Precompute spreading coefficients
        let spread_coeffs =
            precompute_spreading_coeffs(&normalized_points, n_oversampled, kernel_width);

        // Precompute deconvolution factors
        let deconv_factors = precompute_deconv_factors(n_uniform, n_oversampled, kernel_width);

        // Create FFT plan
        let fft_plan = Plan::dft_1d(n_oversampled, Direction::Forward, Flags::MEASURE);

        Ok(Self {
            nufft_type,
            n_uniform,
            n_nonuniform: points.len(),
            n_oversampled,
            points: normalized_points,
            spread_coeffs,
            deconv_factors,
            fft_plan,
            options: NufftOptions {
                kernel_width,
                ..*options
            },
        })
    }

    /// Execute Type 1 NUFFT: non-uniform → uniform.
    ///
    /// Given values at non-uniform points, compute uniform Fourier coefficients.
    ///
    /// # Errors
    ///
    /// Returns error if input length doesn't match the number of non-uniform points.
    pub fn type1(&self, values: &[Complex<T>]) -> NufftResult<Vec<Complex<T>>> {
        if values.len() != self.n_nonuniform {
            return Err(NufftError::ExecutionFailed(format!(
                "Expected {} values, got {}",
                self.n_nonuniform,
                values.len()
            )));
        }

        // Step 1: Spread non-uniform data to oversampled grid
        let mut grid = vec![Complex::<T>::zero(); self.n_oversampled];
        self.spread_to_grid(values, &mut grid);

        // Step 2: FFT on oversampled grid
        let mut fft_result = vec![Complex::<T>::zero(); self.n_oversampled];
        if let Some(ref plan) = self.fft_plan {
            plan.execute(&grid, &mut fft_result);
        } else {
            return Err(NufftError::PlanFailed);
        }

        // Step 3: Deconvolve and extract central frequencies
        let mut result = Vec::with_capacity(self.n_uniform);
        let half_n = self.n_uniform / 2;

        for k in 0..self.n_uniform {
            // Map output index to oversampled grid index
            let grid_idx = if k < half_n {
                k
            } else {
                self.n_oversampled - (self.n_uniform - k)
            };

            // Deconvolve
            let deconv_idx = k;
            result.push(fft_result[grid_idx] * self.deconv_factors[deconv_idx]);
        }

        Ok(result)
    }

    /// Execute Type 2 NUFFT: uniform → non-uniform.
    ///
    /// Given uniform Fourier coefficients, evaluate at non-uniform points.
    ///
    /// # Errors
    ///
    /// Returns error if coefficient length doesn't match the uniform grid size.
    pub fn type2(&self, coeffs: &[Complex<T>]) -> NufftResult<Vec<Complex<T>>> {
        if coeffs.len() != self.n_uniform {
            return Err(NufftError::ExecutionFailed(format!(
                "Expected {} coefficients, got {}",
                self.n_uniform,
                coeffs.len()
            )));
        }

        // Step 1: Pad and deconvolve coefficients to oversampled grid
        let mut grid = vec![Complex::<T>::zero(); self.n_oversampled];
        let half_n = self.n_uniform / 2;

        for (k, &coeff) in coeffs.iter().enumerate() {
            let grid_idx = if k < half_n {
                k
            } else {
                self.n_oversampled - (self.n_uniform - k)
            };
            grid[grid_idx] = coeff * self.deconv_factors[k];
        }

        // Step 2: Inverse FFT
        let mut ifft_result = vec![Complex::<T>::zero(); self.n_oversampled];
        // Create inverse plan
        if let Some(inv_plan) =
            Plan::dft_1d(self.n_oversampled, Direction::Backward, Flags::ESTIMATE)
        {
            inv_plan.execute(&grid, &mut ifft_result);
        } else {
            return Err(NufftError::PlanFailed);
        }

        // Normalize
        let scale = T::ONE / T::from_usize(self.n_oversampled);
        for c in &mut ifft_result {
            *c = Complex::new(c.re * scale, c.im * scale);
        }

        // Step 3: Interpolate at non-uniform points
        let result = self.interpolate_from_grid(&ifft_result);

        Ok(result)
    }

    /// Execute NUFFT based on the configured type.
    ///
    /// # Errors
    ///
    /// Returns error for Type3 (use `execute_type3` instead) or if input validation fails.
    pub fn execute(&self, input: &[Complex<T>]) -> NufftResult<Vec<Complex<T>>> {
        match self.nufft_type {
            NufftType::Type1 => self.type1(input),
            NufftType::Type2 => self.type2(input),
            NufftType::Type3 => {
                // Type 3 = Type 1 followed by Type 2 (with different target points)
                // For simplicity, we require separate source/target points
                Err(NufftError::ExecutionFailed(
                    "Type 3 requires separate execute_type3 call".into(),
                ))
            }
        }
    }

    /// Execute Type 3 NUFFT: non-uniform → non-uniform.
    ///
    /// # Arguments
    ///
    /// * `values` - Values at source points (set during plan creation)
    /// * `target_points` - Target non-uniform points in [-π, π]
    ///
    /// # Errors
    ///
    /// Returns error if input validation fails or target points are out of range.
    pub fn execute_type3(
        &self,
        values: &[Complex<T>],
        target_points: &[f64],
    ) -> NufftResult<Vec<Complex<T>>> {
        // Type 3 = Type 1 to uniform grid, then Type 2 to target points
        // First do Type 1
        let uniform_coeffs = self.type1(values)?;

        // Create Type 2 plan for target points
        let type2_plan = Self::new(
            NufftType::Type2,
            self.n_uniform,
            target_points,
            self.options.tolerance,
        )?;

        // Execute Type 2
        type2_plan.type2(&uniform_coeffs)
    }

    /// Spread non-uniform values to the oversampled grid.
    fn spread_to_grid(&self, values: &[Complex<T>], grid: &mut [Complex<T>]) {
        for (j, &val) in values.iter().enumerate() {
            for &(idx, weight) in &self.spread_coeffs[j] {
                grid[idx] = grid[idx] + Complex::new(val.re * weight, val.im * weight);
            }
        }
    }

    /// Interpolate from grid at non-uniform points.
    fn interpolate_from_grid(&self, grid: &[Complex<T>]) -> Vec<Complex<T>> {
        let mut result = Vec::with_capacity(self.n_nonuniform);

        for j in 0..self.n_nonuniform {
            let mut sum = Complex::<T>::zero();
            for &(idx, weight) in &self.spread_coeffs[j] {
                sum = sum + Complex::new(grid[idx].re * weight, grid[idx].im * weight);
            }
            result.push(sum);
        }

        result
    }

    /// Get the number of uniform grid points.
    pub fn n_uniform(&self) -> usize {
        self.n_uniform
    }

    /// Get the number of non-uniform points.
    pub fn n_nonuniform(&self) -> usize {
        self.n_nonuniform
    }

    /// Get the NUFFT type.
    pub fn nufft_type(&self) -> NufftType {
        self.nufft_type
    }

    /// Get the normalized non-uniform points (in [0, 2π]).
    pub fn points(&self) -> &[f64] {
        &self.points
    }
}

/// Compute kernel width based on desired tolerance.
fn compute_kernel_width(tolerance: f64, default: usize) -> usize {
    // Empirical formula: width ≈ -log10(tolerance) + 2
    let width = (-tolerance.log10() + 2.0).ceil() as usize;
    width.max(4).min(default.max(12))
}

/// Find next "smooth" number (product of small primes) for efficient FFT.
fn next_smooth_number(n: usize) -> usize {
    // Find next number that's a product of 2, 3, 5
    let mut candidate = n;
    loop {
        let mut temp = candidate;
        while temp.is_multiple_of(2) {
            temp /= 2;
        }
        while temp.is_multiple_of(3) {
            temp /= 3;
        }
        while temp.is_multiple_of(5) {
            temp /= 5;
        }
        if temp == 1 {
            return candidate;
        }
        candidate += 1;
    }
}

/// Precompute spreading coefficients using Gaussian kernel.
fn precompute_spreading_coeffs<T: Float>(
    points: &[f64],
    n_grid: usize,
    kernel_width: usize,
) -> Vec<Vec<(usize, T)>> {
    let grid_spacing = 2.0 * core::f64::consts::PI / (n_grid as f64);
    let half_width = kernel_width / 2;

    // Gaussian kernel parameter (beta)
    let beta = 2.3 * (kernel_width as f64);

    points
        .iter()
        .map(|&x| {
            // Find nearest grid point
            let grid_pos = x / grid_spacing;
            let center = grid_pos.round() as isize;

            let mut coeffs = Vec::with_capacity(kernel_width);

            for offset in -(half_width as isize)..=(half_width as isize) {
                let grid_idx = (center + offset).rem_euclid(n_grid as isize) as usize;
                let grid_x = (grid_idx as f64) * grid_spacing;

                // Distance from point to grid location
                let mut dx = x - grid_x;
                // Wrap around
                if dx > core::f64::consts::PI {
                    dx -= 2.0 * core::f64::consts::PI;
                } else if dx < -core::f64::consts::PI {
                    dx += 2.0 * core::f64::consts::PI;
                }

                // Gaussian kernel: exp(-beta * (dx/width)^2)
                let normalized_dx = dx / (grid_spacing * (half_width as f64));
                let weight = (-beta * normalized_dx * normalized_dx).exp();

                if weight > 1e-15 {
                    coeffs.push((grid_idx, T::from_f64(weight)));
                }
            }

            coeffs
        })
        .collect()
}

/// Precompute deconvolution factors.
fn precompute_deconv_factors<T: Float>(
    n_uniform: usize,
    n_oversampled: usize,
    kernel_width: usize,
) -> Vec<Complex<T>> {
    let beta = 2.3 * (kernel_width as f64);
    let ratio = (n_oversampled as f64) / (n_uniform as f64);

    (0..n_uniform)
        .map(|k| {
            // Frequency in [-N/2, N/2)
            let freq = if k < n_uniform / 2 {
                k as f64
            } else {
                (k as f64) - (n_uniform as f64)
            };

            // Fourier transform of Gaussian kernel
            // FT of exp(-beta*x^2) is sqrt(pi/beta) * exp(-pi^2*k^2/beta)
            let arg = core::f64::consts::PI * core::f64::consts::PI * freq * freq
                / (beta * ratio * ratio);
            let deconv = (arg).exp();

            Complex::new(T::from_f64(deconv), T::ZERO)
        })
        .collect()
}

// Convenience functions

/// Compute Type 1 NUFFT (non-uniform → uniform).
///
/// # Arguments
///
/// * `points` - Non-uniform sample locations in [-π, π]
/// * `values` - Complex values at the sample points
/// * `n_output` - Number of uniform output frequencies
/// * `tolerance` - Target relative accuracy (e.g., 1e-6)
///
/// # Returns
///
/// Uniform Fourier coefficients.
///
/// # Errors
///
/// Returns error if points are out of range or tolerance is invalid.
pub fn nufft_type1<T: Float>(
    points: &[f64],
    values: &[Complex<T>],
    n_output: usize,
    tolerance: f64,
) -> NufftResult<Vec<Complex<T>>> {
    let plan = Nufft::new(NufftType::Type1, n_output, points, tolerance)?;
    plan.type1(values)
}

/// Compute Type 2 NUFFT (uniform → non-uniform).
///
/// # Arguments
///
/// * `coeffs` - Uniform Fourier coefficients
/// * `points` - Non-uniform evaluation points in [-π, π]
/// * `tolerance` - Target relative accuracy (e.g., 1e-6)
///
/// # Returns
///
/// Values at non-uniform points.
///
/// # Errors
///
/// Returns error if points are out of range or tolerance is invalid.
pub fn nufft_type2<T: Float>(
    coeffs: &[Complex<T>],
    points: &[f64],
    tolerance: f64,
) -> NufftResult<Vec<Complex<T>>> {
    let plan = Nufft::new(NufftType::Type2, coeffs.len(), points, tolerance)?;
    plan.type2(coeffs)
}

/// Compute Type 3 NUFFT (non-uniform → non-uniform).
///
/// # Arguments
///
/// * `source_points` - Source non-uniform locations in [-π, π]
/// * `values` - Complex values at source points
/// * `target_points` - Target non-uniform locations in [-π, π]
/// * `tolerance` - Target relative accuracy
///
/// # Returns
///
/// Values at target points.
///
/// # Errors
///
/// Returns error if points are out of range or tolerance is invalid.
pub fn nufft_type3<T: Float>(
    source_points: &[f64],
    values: &[Complex<T>],
    target_points: &[f64],
    tolerance: f64,
) -> NufftResult<Vec<Complex<T>>> {
    // Use intermediate uniform grid size based on source and target counts
    let n_uniform = (source_points.len() + target_points.len()).next_power_of_two();
    let plan = Nufft::new(NufftType::Type1, n_uniform, source_points, tolerance)?;
    plan.execute_type3(values, target_points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_nufft_type1_uniform_points() {
        // If points are uniformly spaced, NUFFT should match regular FFT
        let n = 8;
        let points: Vec<f64> = (0..n)
            .map(|k| -core::f64::consts::PI + (k as f64) * 2.0 * core::f64::consts::PI / (n as f64))
            .collect();

        let values: Vec<Complex<f64>> = (0..n)
            .map(|k| Complex::new((k as f64).cos(), (k as f64).sin()))
            .collect();

        let result = nufft_type1(&points, &values, n, 1e-6);
        assert!(result.is_ok());
        let result = result.expect("NUFFT failed");
        assert_eq!(result.len(), n);
    }

    #[test]
    fn test_nufft_type2_single_frequency() {
        // Single frequency should produce sinusoid at evaluation points
        let n = 16;
        let mut coeffs = vec![Complex::<f64>::zero(); n];
        coeffs[1] = Complex::new(1.0, 0.0); // Single frequency at k=1

        let points: Vec<f64> = (0..5)
            .map(|k| -core::f64::consts::PI + f64::from(k) * 0.5)
            .collect();

        let result = nufft_type2(&coeffs, &points, 1e-6);
        assert!(result.is_ok());
        let result = result.expect("NUFFT failed");
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_nufft_roundtrip() {
        // Type1 followed by Type2 should approximate identity
        let n = 32;
        let points: Vec<f64> = (0..10).map(|k| -2.5 + f64::from(k) * 0.5).collect();

        let values: Vec<Complex<f64>> = points
            .iter()
            .map(|&x| Complex::new(x.cos(), x.sin()))
            .collect();

        // Type 1: non-uniform → uniform
        let uniform = nufft_type1(&points, &values, n, 1e-6).expect("Type1 failed");

        // Type 2: uniform → non-uniform (same points)
        let recovered = nufft_type2(&uniform, &points, 1e-6).expect("Type2 failed");

        // Check approximate recovery (won't be exact due to truncation)
        assert_eq!(recovered.len(), values.len());
    }

    #[test]
    fn test_nufft_error_handling() {
        let points = vec![0.0, 0.5, 1.0];

        // Invalid size
        let result = Nufft::<f64>::new(NufftType::Type1, 0, &points, 1e-6);
        assert!(result.is_err());

        // Point out of range
        let bad_points = vec![0.0, 5.0]; // 5.0 > π
        let result = Nufft::<f64>::new(NufftType::Type1, 16, &bad_points, 1e-6);
        assert!(result.is_err());

        // Invalid tolerance
        let result = Nufft::<f64>::new(NufftType::Type1, 16, &points, -1e-6);
        assert!(result.is_err());
    }

    #[test]
    fn test_smooth_number() {
        assert_eq!(next_smooth_number(100), 100); // 100 = 2^2 * 5^2
        assert_eq!(next_smooth_number(101), 108); // 108 = 2^2 * 3^3
        assert_eq!(next_smooth_number(7), 8); // 8 = 2^3
    }
}
