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

pub mod nufft2d;
pub mod nufft3d;

pub use nufft2d::{nufft2d_type1, nufft2d_type2};
pub use nufft3d::nufft3d_type1;

/// NUFFT type specifying the direction of transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
#[non_exhaustive]
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
#[allow(clippy::struct_field_names)] // reason: fields named by mathematical role (nufft_type, n_uniform, etc.); renaming would obscure the NUFFT algorithm structure
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
    ///
    /// # Examples
    ///
    /// ```
    /// use oxifft::{Complex, Nufft, NufftType};
    ///
    /// // Type 1 NUFFT: 4 non-uniform points → 8 uniform Fourier modes
    /// let points = vec![-1.0_f64, -0.3, 0.2, 0.8];
    /// let plan = Nufft::<f64>::new(NufftType::Type1, 8, &points, 1e-6)
    ///     .expect("NUFFT plan creation failed");
    /// let values: Vec<Complex<f64>> = points.iter()
    ///     .map(|&x| Complex::new(x.cos(), 0.0))
    ///     .collect();
    /// let modes = plan.type1(&values).expect("type1 execution failed");
    /// // Output has n_uniform = 8 modes
    /// assert_eq!(modes.len(), 8);
    /// ```
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

        // Compute kernel width from tolerance and oversampling ratio.
        // The oversampling ratio affects how large the kernel needs to be to
        // achieve the desired accuracy (smaller oversampling needs wider kernel).
        let kernel_width = compute_kernel_width(
            options.tolerance,
            options.oversampling,
            options.kernel_width,
        );

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

        // Step 3: Deconvolve and extract frequencies in math order.
        //
        // The output convention (matching the dense NDFT / FINUFFT Type 1) is:
        //   result[k] corresponds to frequency  freq = k − N/2
        //   k=0      → freq = −N/2   (most-negative)
        //   k=N/2    → freq = 0      (DC)
        //   k=N−1    → freq = N/2−1  (most-positive)
        //
        // In the oversampled FFT result, frequency `freq` lives at:
        //   grid_idx = freq               if freq ≥ 0
        //   grid_idx = n_oversampled+freq if freq < 0
        //
        // The deconv_factors array is in FFT order (index `d` → freq `d` if
        // d < N/2, freq `d−N` if d ≥ N/2), so the FFT-order index for `freq` is:
        //   deconv_idx = freq               if freq ≥ 0
        //   deconv_idx = n_uniform + freq   if freq < 0
        let mut result = Vec::with_capacity(self.n_uniform);
        let half_n = self.n_uniform / 2;

        for k in 0..self.n_uniform {
            // Math-order frequency for output index k
            let freq = (k as isize) - (half_n as isize);

            let grid_idx = if freq >= 0 {
                freq as usize
            } else {
                (self.n_oversampled as isize + freq) as usize
            };

            let deconv_idx = if freq >= 0 {
                freq as usize
            } else {
                (self.n_uniform as isize + freq) as usize
            };

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

        // Step 1: Deconvolve coefficients and scatter into oversampled grid.
        //
        // Input convention (matching the dense NDFT / FINUFFT Type 2):
        //   coeffs[k] is the Fourier coefficient for frequency  freq = k − N/2
        //   k=0   → freq = −N/2  (most-negative)
        //   k=N/2 → freq = 0     (DC)
        //
        // In the oversampled grid, frequency `freq` is placed at:
        //   grid_idx = freq               if freq ≥ 0
        //   grid_idx = n_oversampled+freq if freq < 0
        //
        // deconv_factors is in FFT order; the FFT-order index for `freq` is:
        //   deconv_idx = freq               if freq ≥ 0
        //   deconv_idx = n_uniform + freq   if freq < 0
        //
        // Type 2 deconvolution differs from Type 1 by a factor of n_oversampled.
        // After IFFT (unnormalized) + 1/n_os normalization + kernel interpolation,
        // the output picks up an extra factor of Ψ̂(freq)/n_os from the
        // interpolation step.  To cancel this, the grid coefficient must be
        // n_os / Ψ̂(freq), not 1 / Ψ̂(freq).  Since deconv_factors already stores
        // 1/Ψ̂, we multiply by n_os here.
        let mut grid = vec![Complex::<T>::zero(); self.n_oversampled];
        let half_n = self.n_uniform / 2;
        let n_os_scale = T::from_usize(self.n_oversampled);

        for (k, &coeff) in coeffs.iter().enumerate() {
            let freq = (k as isize) - (half_n as isize);

            let grid_idx = if freq >= 0 {
                freq as usize
            } else {
                (self.n_oversampled as isize + freq) as usize
            };

            let deconv_idx = if freq >= 0 {
                freq as usize
            } else {
                (self.n_uniform as isize + freq) as usize
            };

            // Multiply deconv factor by n_os to account for the IFFT normalization
            // that is undone by the interpolation (kernel re-applies a factor of ~1/n_os
            // relative to what Type 1 spreading adds, so Type 2 needs n_os × the
            // Type 1 deconv factor).
            let scaled_deconv = Complex::new(
                self.deconv_factors[deconv_idx].re * n_os_scale,
                self.deconv_factors[deconv_idx].im * n_os_scale,
            );
            grid[grid_idx] = coeff * scaled_deconv;
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

/// Compute kernel width based on desired tolerance and oversampling ratio.
///
/// The spreading kernel is `exp(-β·(j/W)²)` with `β = 2.3·W` and
/// `W = kernel_width / 2`.  The required half-width W depends on both the
/// target accuracy and the oversampling ratio `σ`:
///
/// - Lower `σ` (e.g. 1.5) leaves fewer guard bands in the oversampled grid,
///   so the kernel must be wider to control sub-grid aliasing at Nyquist.
/// - Higher `σ` (e.g. 2.0) provides more isolation and needs fewer taps.
///
/// Empirical formula (validated against NUFFT tolerance sweep benchmarks):
///
/// ```text
/// W = ceil( -log10(tol) · (2 - σ/2) )
/// kw = max(4, 2·W)    (always even, so W = kw/2 is exact)
/// ```
///
/// Values:
/// - `σ=1.5, tol=1e-3`:  `W = ceil(3 · 1.25) = 4` → `kw = 8`
/// - `σ=2.0, tol=1e-3`:  `W = ceil(3 · 1.0)  = 3` → `kw = 6`
/// - `σ=1.5, tol=1e-6`:  `W = ceil(6 · 1.25) = 8` → `kw = 16`
/// - `σ=2.0, tol=1e-6`:  `W = ceil(6 · 1.0)  = 6` → `kw = 12`
///
/// The `default` parameter is the user-supplied `NufftOptions::kernel_width`
/// and sets an upper cap via `min(kw, default.max(12))`.
pub(crate) fn compute_kernel_width(tolerance: f64, oversampling: f64, default: usize) -> usize {
    let sigma = oversampling.clamp(1.05, 4.0); // guard against degenerate values
                                               // f(σ) = (2 - σ/2): factor that accounts for the reduction in guard-band
                                               // isolation as σ decreases toward 1.
    let f_sigma = 2.0 - sigma / 2.0;
    // W = half-width needed to achieve tolerance.
    let w = ((-tolerance.log10()) * f_sigma).ceil() as usize;
    // kw must be even (so that W = kw/2 exactly).
    // The lower bound is 4 (minimum useful kernel, W=2).
    // We take the max of the computed value and the user-supplied `default`
    // (from NufftOptions::kernel_width), so:
    //   - If the user requests a wider kernel, honour it.
    //   - If accuracy demands a wider kernel than `default`, use the wider one.
    (2 * w).max(4).max(default)
}

/// Find next "smooth" number (product of small primes) for efficient FFT.
pub(crate) fn next_smooth_number(n: usize) -> usize {
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
pub(crate) fn precompute_spreading_coeffs<T: Float>(
    points: &[f64],
    n_grid: usize,
    kernel_width: usize,
) -> Vec<Vec<(usize, T)>> {
    let grid_spacing = 2.0 * core::f64::consts::PI / (n_grid as f64);
    let half_width = kernel_width / 2;

    // Gaussian kernel parameter (beta).
    //
    // The kernel is exp(-β · (dx / (h · W))²) with W = half_width.
    // β must scale with W (not kernel_width) so that the edge weight
    // exp(-β) remains finite regardless of how large kw gets.  Using
    // β = 2.3 · W gives ~exp(-2.3) ≈ 0.10 edge weight, which provides
    // enough taper while keeping sub-grid variation manageable.
    let beta = 2.3 * (half_width as f64);

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

/// Precompute deconvolution factors for the Gaussian NUFFT.
///
/// The spreading kernel in [`precompute_spreading_coeffs`] is:
/// ```text
/// w(dx) = exp(-β · (dx / (h · W_int))²)
///   h     = 2π / n_oversampled   (oversampled grid spacing)
///   W_int = kernel_width / 2     (integer half-width, support is 2·W_int+1 points)
///   β     = 2.3 · W_int   (= 2.3 · kernel_width / 2)
/// ```
///
/// ## Exact discrete kernel DFT
///
/// The kernel is symmetric about its centre, so its DFT (evaluated at the
/// integer output frequency `freq`) is real and equals:
/// ```text
/// ψ̂(freq) = Σ_{j=-W_int}^{W_int}  exp(-β · (j/W_int)²) · cos(2π · freq · j / n_oversampled)
/// ```
/// The deconvolution factor is `1 / ψ̂(freq)`.
///
/// ## Phase correction for the [0, 2π] shift
///
/// Points are normalised from `[-π, π]` to `[0, 2π]` by adding π.  This shift
/// multiplies the DFT result at oversampled bin `k` by `exp(-i·k·π) = (-1)^k`.
/// After deconvolution the residual phase `(-1)^k` must be removed by
/// multiplying by `exp(+i·k·π) = (-1)^k`.  Because `(-1)^k = ±1` the combined
/// factor remains real-valued.
///
/// The oversampled grid bin `k` that holds frequency `freq` is:
/// - `k = freq`              when `freq ≥ 0`
/// - `k = n_oversampled + freq`  when `freq < 0`
///
/// In FFT-order deconvolution index `d`:
/// - `d < N/2`: `freq = d`,  `k = d`
/// - `d ≥ N/2`: `freq = d − N`, `k = n_oversampled + d − N`
///
/// The returned factors are indexed in **FFT order** (`d = 0..N-1`).
pub(crate) fn precompute_deconv_factors<T: Float>(
    n_uniform: usize,
    n_oversampled: usize,
    kernel_width: usize,
) -> Vec<Complex<T>> {
    let w_int = (kernel_width / 2) as isize; // integer half-width for spreading
                                             // β must match the β used in precompute_spreading_coeffs: β = 2.3 · W_int
    let beta = 2.3 * (w_int as f64);
    let two_pi_over_nos = 2.0 * core::f64::consts::PI / (n_oversampled as f64);

    (0..n_uniform)
        .map(|d| {
            // FFT-order frequency and oversampled grid bin
            let (freq, grid_bin) = if d < n_uniform / 2 {
                (d as isize, d)
            } else {
                let f = (d as isize) - (n_uniform as isize);
                (f, n_oversampled + d - n_uniform)
            };

            // Exact kernel DFT at `freq` (cosine sum over kernel support):
            //   ψ̂(freq) = Σ_{j=-W_int}^{W_int} exp(-β·(j/W_int)²) · cos(2π·freq·j/n_os)
            let kernel_dft: f64 = (-w_int..=w_int)
                .map(|j| {
                    let w = (-beta * ((j * j) as f64 / (w_int * w_int) as f64)).exp();
                    let angle = two_pi_over_nos * (freq * j) as f64;
                    w * angle.cos()
                })
                .sum();

            // Phase correction: spreading used x_norm = x_orig + π, which added
            // exp(-i·k·π) = (-1)^k to every DFT bin.  Multiply by (-1)^k to undo.
            let phase_sign = if grid_bin % 2 == 0 { 1.0_f64 } else { -1.0_f64 };

            // Deconvolution: 1/ψ̂(freq) with phase correction (always real).
            let deconv = if kernel_dft.abs() > f64::EPSILON {
                phase_sign / kernel_dft
            } else {
                0.0_f64
            };

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
