//! Aliasing filters for sparse FFT.
//!
//! Implements filters used in the subsampling stage of FFAST algorithm.
//! These filters help isolate frequency components during bucketization.

// Allow dead code - infrastructure for future algorithm enhancements
#![allow(dead_code)]

use crate::kernel::{Complex, Float};

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Filter type for sparse FFT subsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FilterType {
    /// Flat (box) filter - simple but has spectral leakage.
    Flat,
    /// Dirichlet kernel filter - better frequency isolation.
    Dirichlet,
    /// Gaussian filter - smooth roll-off.
    Gaussian,
    /// Blackman-Harris filter - very low sidelobes.
    BlackmanHarris,
}

/// Aliasing filter for sparse FFT.
#[derive(Debug, Clone)]
pub struct AliasingFilter<T: Float> {
    /// Filter coefficients in frequency domain.
    pub coeffs: Vec<Complex<T>>,
    /// Filter type.
    pub filter_type: FilterType,
    /// Filter width parameter.
    pub width: usize,
    /// Signal length.
    pub n: usize,
}

impl<T: Float> AliasingFilter<T> {
    /// Create a flat (box) filter.
    ///
    /// # Arguments
    ///
    /// * `n` - Signal length
    /// * `width` - Filter width (number of frequency bins to pass)
    pub fn flat(n: usize, width: usize) -> Self {
        let mut coeffs = vec![Complex::<T>::zero(); n];
        let half_width = width / 2;

        // Pass frequencies in [-width/2, width/2]
        for i in 0..half_width {
            coeffs[i] = Complex::new(T::ONE, T::ZERO);
        }
        for i in (n - half_width)..n {
            coeffs[i] = Complex::new(T::ONE, T::ZERO);
        }

        Self {
            coeffs,
            filter_type: FilterType::Flat,
            width,
            n,
        }
    }

    /// Create a Dirichlet kernel filter.
    ///
    /// The Dirichlet kernel provides better frequency isolation than flat filter.
    ///
    /// # Arguments
    ///
    /// * `n` - Signal length
    /// * `width` - Filter width parameter
    pub fn dirichlet(n: usize, width: usize) -> Self {
        let mut coeffs = vec![Complex::<T>::zero(); n];

        // Dirichlet kernel: D_M(x) = sin(Mx/2) / sin(x/2)
        let m = width as f64;
        let n_f = n as f64;

        for i in 0..n {
            let x = 2.0 * core::f64::consts::PI * (i as f64) / n_f;

            let val = if x.abs() < 1e-10 {
                m // Limit as x -> 0
            } else {
                (m * x / 2.0).sin() / (x / 2.0).sin()
            };

            // Normalize and apply window
            let normalized = val / m;
            coeffs[i] = Complex::new(T::from_f64(normalized.abs()), T::ZERO);
        }

        Self {
            coeffs,
            filter_type: FilterType::Dirichlet,
            width,
            n,
        }
    }

    /// Create a Gaussian filter.
    ///
    /// # Arguments
    ///
    /// * `n` - Signal length
    /// * `sigma` - Standard deviation parameter
    pub fn gaussian(n: usize, sigma: f64) -> Self {
        let mut coeffs = vec![Complex::<T>::zero(); n];
        let n_f = n as f64;

        for i in 0..n {
            // Map index to [-n/2, n/2]
            let freq = if i <= n / 2 { i as f64 } else { i as f64 - n_f };

            // Gaussian: exp(-freq^2 / (2*sigma^2))
            let val = (-freq * freq / (2.0 * sigma * sigma)).exp();
            coeffs[i] = Complex::new(T::from_f64(val), T::ZERO);
        }

        Self {
            coeffs,
            filter_type: FilterType::Gaussian,
            width: (4.0 * sigma).ceil() as usize,
            n,
        }
    }

    /// Create a Blackman-Harris filter for very low sidelobes.
    ///
    /// # Arguments
    ///
    /// * `n` - Signal length
    /// * `width` - Filter width
    pub fn blackman_harris(n: usize, width: usize) -> Self {
        let mut coeffs = vec![Complex::<T>::zero(); n];
        let width_f = width as f64;

        // Blackman-Harris window coefficients
        let a0 = 0.355_768;
        let a1 = 0.487_396;
        let a2 = 0.144_232;
        let a3 = 0.012_604;

        let half_width = width / 2;

        for i in 0..width {
            let x = (i as f64) / width_f;
            let two_pi = 2.0 * core::f64::consts::PI;

            let val = a0 - a1 * (two_pi * x).cos() + a2 * (2.0 * two_pi * x).cos()
                - a3 * (3.0 * two_pi * x).cos();

            // Place in frequency domain centered at 0
            let idx = if i < half_width { i } else { n - width + i };
            if idx < n {
                coeffs[idx] = Complex::new(T::from_f64(val), T::ZERO);
            }
        }

        Self {
            coeffs,
            filter_type: FilterType::BlackmanHarris,
            width,
            n,
        }
    }

    /// Apply filter to signal in frequency domain.
    ///
    /// # Arguments
    ///
    /// * `signal_fft` - Signal in frequency domain
    ///
    /// # Returns
    ///
    /// Filtered signal in frequency domain.
    pub fn apply(&self, signal_fft: &[Complex<T>]) -> Vec<Complex<T>> {
        debug_assert_eq!(signal_fft.len(), self.n);

        signal_fft
            .iter()
            .zip(self.coeffs.iter())
            .map(|(&s, &f)| s * f)
            .collect()
    }

    /// Apply filter to signal in time domain (convolution).
    ///
    /// This is less efficient than frequency domain multiplication.
    /// Use `apply()` on FFT of signal for better performance.
    pub fn apply_time_domain(&self, signal: &[Complex<T>]) -> Vec<Complex<T>> {
        // Convert filter to time domain via IFFT
        // For now, use simple multiplication approach
        // In production, use FFT-based convolution
        self.apply(signal)
    }

    /// Get filter response at a specific frequency.
    pub fn response_at(&self, freq_idx: usize) -> Complex<T> {
        if freq_idx < self.n {
            self.coeffs[freq_idx]
        } else {
            Complex::<T>::zero()
        }
    }

    /// Get the -3dB bandwidth of the filter.
    pub fn bandwidth_3db(&self) -> usize {
        let half_power = T::from_f64(0.5); // -3dB = half power
        let mut count = 0;

        for coeff in &self.coeffs {
            if coeff.norm_sqr() >= half_power {
                count += 1;
            }
        }

        count
    }
}

/// Create optimal filter for sparse FFT based on sparsity.
///
/// # Arguments
///
/// * `n` - Signal length
/// * `k` - Expected sparsity
/// * `num_buckets` - Number of buckets
pub fn create_optimal_filter<T: Float>(
    n: usize,
    k: usize,
    num_buckets: usize,
) -> AliasingFilter<T> {
    // For sparse FFT, we want a filter that:
    // 1. Has main lobe width approximately n/num_buckets
    // 2. Has low sidelobes to minimize aliasing errors

    let width = n / num_buckets;

    // Use Dirichlet for moderate sparsity, Gaussian for very sparse
    if k < n / 100 {
        // Very sparse: use wider Gaussian for better isolation
        AliasingFilter::gaussian(n, (width as f64) / 2.0)
    } else if k < n / 20 {
        // Moderately sparse: use Dirichlet
        AliasingFilter::dirichlet(n, width)
    } else {
        // Less sparse: use flat filter for simplicity
        AliasingFilter::flat(n, width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_filter() {
        let filter: AliasingFilter<f64> = AliasingFilter::flat(64, 16);
        assert_eq!(filter.coeffs.len(), 64);
        assert_eq!(filter.filter_type, FilterType::Flat);

        // Check that passband has unit response
        assert_eq!(filter.response_at(0).re, 1.0);
    }

    #[test]
    fn test_gaussian_filter() {
        let filter: AliasingFilter<f64> = AliasingFilter::gaussian(64, 8.0);
        assert_eq!(filter.coeffs.len(), 64);
        assert_eq!(filter.filter_type, FilterType::Gaussian);

        // Peak at DC
        assert!(filter.response_at(0).re > 0.9);
    }

    #[test]
    fn test_dirichlet_filter() {
        let filter: AliasingFilter<f64> = AliasingFilter::dirichlet(64, 8);
        assert_eq!(filter.coeffs.len(), 64);
        assert_eq!(filter.filter_type, FilterType::Dirichlet);
    }

    #[test]
    fn test_filter_apply() {
        let filter: AliasingFilter<f64> = AliasingFilter::flat(8, 4);
        let signal = vec![Complex::new(1.0_f64, 0.0); 8];

        let filtered = filter.apply(&signal);
        assert_eq!(filtered.len(), 8);
    }

    #[test]
    fn test_optimal_filter() {
        // Very sparse
        let filter1: AliasingFilter<f64> = create_optimal_filter(1024, 5, 32);
        assert_eq!(filter1.filter_type, FilterType::Gaussian);

        // Moderately sparse
        let filter2: AliasingFilter<f64> = create_optimal_filter(1024, 30, 32);
        assert_eq!(filter2.filter_type, FilterType::Dirichlet);

        // Less sparse
        let filter3: AliasingFilter<f64> = create_optimal_filter(1024, 100, 32);
        assert_eq!(filter3.filter_type, FilterType::Flat);
    }
}
