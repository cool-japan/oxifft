//! Chirp-Z Transform (CZT) for arbitrary evaluation grids.
//!
//! The CZT computes:
//!
//! ```text
//! X[k] = Σ_{n=0}^{N-1}  x[n] · A^{-n} · W^{nk},   k = 0 .. M-1
//! ```
//!
//! where `A` and `W` are arbitrary complex numbers that define the evaluation
//! contour on the z-plane.  When `A = 1` and `W = e^{-2πi/N}`, the CZT
//! reduces to the standard forward DFT.
//!
//! # Algorithm
//!
//! Implements the Bluestein / Rabiner-Schafer-Rader (1969) chirp decomposition
//! that rewrites the CZT as a length-`L = next_pow2(N+M-1)` convolution,
//! computable in `O(L log L)` time:
//!
//! 1. **Pre-modulate**: `y[n] = x[n] · chirp_y[n]` where `chirp_y[n] = A^{-n}·W^{n²/2}`
//! 2. **Zero-pad** `y` to length `L`
//! 3. **FFT**: `Y = FFT_L(y)`
//! 4. **Pointwise multiply**: `R[k] = Y[k] · H_fft[k]`  (H_fft precomputed)
//! 5. **IFFT**: `r = IFFT_L(R)` (with manual 1/L normalisation)
//! 6. **Post-modulate**: `X[k] = r[k] · chirp_post[k]` for `k = 0..M-1`
//!
//! # Examples
//!
//! ```
//! use oxifft::{Complex, chirp_z::CztPlan};
//!
//! // Standard DFT via CZT (N=M=16, A=1, W=e^{-2πi/16})
//! let n = 16_usize;
//! let two_pi_over_n = -2.0 * std::f64::consts::PI / n as f64;
//! let a = Complex::one();
//! let w = Complex::from_polar(1.0_f64, two_pi_over_n);
//! let plan = CztPlan::<f64>::new(n, n, a, w).expect("plan failed");
//!
//! let input: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(i as f64, 0.0)).collect();
//! let mut output = vec![Complex::zero(); n];
//! plan.execute(&input, &mut output).expect("execute failed");
//! // DC bin = sum(0..15) = 120
//! assert!((output[0].re - 120.0).abs() < 1e-8);
//! ```

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};

pub mod chirp_tables;
#[cfg(test)]
mod tests;

use chirp_tables::{build_chirp_filter, build_chirp_post, build_chirp_y};

/// Errors that can occur during CZT plan creation or execution.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum CztError {
    /// Input or output length is zero.
    InvalidSize(usize),
    /// A parameter (`A` or `W`) would cause numerical overflow or underflow.
    InvalidParameter,
    /// Internal FFT plan could not be constructed (should not normally occur for
    /// power-of-two sizes).
    PlanFailed,
    /// Input or output slice length does not match the plan dimensions.
    MismatchedLength {
        /// Expected length.
        expected: usize,
        /// Actual length provided.
        actual: usize,
    },
}

impl core::fmt::Display for CztError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidSize(n) => write!(f, "invalid CZT size: {n}"),
            Self::InvalidParameter => write!(f, "CZT parameter A or W causes overflow"),
            Self::PlanFailed => write!(f, "internal FFT plan creation failed"),
            Self::MismatchedLength { expected, actual } => {
                write!(f, "length mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

/// Result type for CZT operations.
pub type CztResult<T> = Result<T, CztError>;

/// Chirp-Z Transform plan.
///
/// Precomputes all chirp sequences and the FFT of the convolution kernel so
/// that repeated calls to [`execute`](Self::execute) run in `O(L log L)` time.
pub struct CztPlan<T: Float> {
    /// Input length N.
    n: usize,
    /// Output length M.
    m: usize,
    /// Convolution length (next power of two >= N+M-1).
    l: usize,
    /// Pre-modulation sequence: `A^{-n}·W^{n²/2}` for `n = 0..N`.
    chirp_y: Vec<Complex<T>>,
    /// Precomputed `FFT_L` of the circular convolution kernel.
    h_fft: Vec<Complex<T>>,
    /// Post-modulation sequence: `W^{k²/2}` for `k = 0..M`.
    chirp_post: Vec<Complex<T>>,
    /// Forward FFT plan for length L.
    fft_plan: Plan<T>,
    /// Inverse FFT plan for length L (caller divides by L after use).
    ifft_plan: Plan<T>,
}

impl<T: Float> CztPlan<T> {
    /// Create a new CZT plan.
    ///
    /// # Arguments
    ///
    /// * `n` – number of input samples
    /// * `m` – number of output frequency bins
    /// * `a` – starting point on the z-plane (`A` in the CZT formula)
    /// * `w` – step ratio between successive evaluation points (`W`)
    ///
    /// # Errors
    ///
    /// Returns [`CztError::InvalidSize`] when `n` or `m` is zero.
    /// Returns [`CztError::InvalidParameter`] when `|W|^{N²/2}` would overflow
    /// (i.e. `N²/2 · ln|W| > 700`).
    /// Returns [`CztError::PlanFailed`] when the internal FFT plan cannot be
    /// constructed (extremely unlikely for power-of-two sizes).
    pub fn new(n: usize, m: usize, a: Complex<T>, w: Complex<T>) -> CztResult<Self> {
        if n == 0 {
            return Err(CztError::InvalidSize(n));
        }
        if m == 0 {
            return Err(CztError::InvalidSize(m));
        }

        // Overflow guard: |W|^{N²/2} must not blow up.
        let w_abs = w.norm();
        if w_abs > T::ONE {
            let n_f = T::from_usize(n);
            let max_exp = n_f * n_f / T::TWO;
            let log_factor = max_exp * num_traits::Float::ln(w_abs);
            if log_factor > T::from_f64(700.0) {
                return Err(CztError::InvalidParameter);
            }
        }

        // Convolution length: smallest power-of-two >= N+M-1.
        let l = (n + m - 1).next_power_of_two();

        // Build pre-modulation table.
        let chirp_y = build_chirp_y(n, a, w);

        // Build filter in circular-convolution order and FFT it.
        let h_circ = build_chirp_filter(n, m, l, w);

        let fft_plan =
            Plan::<T>::dft_1d(l, Direction::Forward, Flags::MEASURE).ok_or(CztError::PlanFailed)?;
        let ifft_plan = Plan::<T>::dft_1d(l, Direction::Backward, Flags::MEASURE)
            .ok_or(CztError::PlanFailed)?;

        let mut h_fft = vec![Complex::<T>::zero(); l];
        fft_plan.execute(&h_circ, &mut h_fft);

        // Build post-modulation table.
        let chirp_post = build_chirp_post(m, w);

        Ok(Self {
            n,
            m,
            l,
            chirp_y,
            h_fft,
            chirp_post,
            fft_plan,
            ifft_plan,
        })
    }

    /// Execute the Chirp-Z Transform.
    ///
    /// Writes `M` output samples into `output`.  The method allocates two
    /// temporary length-`L` buffers internally (one allocation per call).
    ///
    /// # Arguments
    ///
    /// * `input`  – input slice of length `N`
    /// * `output` – output slice of length `M` (overwritten on success)
    ///
    /// # Errors
    ///
    /// Returns [`CztError::MismatchedLength`] when the slice lengths do not
    /// match the plan dimensions.
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>]) -> CztResult<()> {
        if input.len() != self.n {
            return Err(CztError::MismatchedLength {
                expected: self.n,
                actual: input.len(),
            });
        }
        if output.len() != self.m {
            return Err(CztError::MismatchedLength {
                expected: self.m,
                actual: output.len(),
            });
        }

        let l = self.l;

        // Step 1: y[n] = x[n] · chirp_y[n], zero-padded to length L.
        let mut y_padded = vec![Complex::<T>::zero(); l];
        for i in 0..self.n {
            y_padded[i] = input[i] * self.chirp_y[i];
        }

        // Step 2: Y = FFT_L(y_padded).
        let mut y_fft = vec![Complex::<T>::zero(); l];
        self.fft_plan.execute(&y_padded, &mut y_fft);

        // Step 3: R = Y * H_fft (pointwise multiply).
        let mut r_fft = vec![Complex::<T>::zero(); l];
        for i in 0..l {
            r_fft[i] = y_fft[i] * self.h_fft[i];
        }

        // Step 4: r = IFFT_L(R).
        // Plan follows FFTW convention: no automatic 1/L normalisation.
        let mut r = vec![Complex::<T>::zero(); l];
        self.ifft_plan.execute(&r_fft, &mut r);

        // Step 5: normalise 1/L and apply post-modulation.
        let inv_l = T::ONE / T::from_usize(l);
        for k in 0..self.m {
            let r_k = Complex::<T>::new(r[k].re * inv_l, r[k].im * inv_l);
            output[k] = r_k * self.chirp_post[k];
        }

        Ok(())
    }

    /// Convenience constructor: compute a zoom-FFT on the frequency range
    /// `[f_start, f_stop)` with `M` equally-spaced bins and sample rate `fs`.
    ///
    /// Internally sets:
    /// - `A = exp(+2πi · f_start / fs)`  (first evaluation point)
    /// - `W = exp(−2πi · (f_stop − f_start) / (M · fs))` (forward-DFT convention)
    ///
    /// # Errors
    ///
    /// Returns [`CztError::InvalidParameter`] when `f_start >= f_stop`.
    /// Propagates any error from [`new`](Self::new).
    pub fn zoom_fft(n: usize, m: usize, f_start: T, f_stop: T, fs: T) -> CztResult<Self> {
        if f_start >= f_stop {
            return Err(CztError::InvalidParameter);
        }

        let two_pi = T::TWO_PI;
        let m_f = T::from_usize(m);

        // First evaluation point on the unit circle.
        let a = Complex::cis(two_pi * f_start / fs);

        // Step ratio (forward-DFT convention: negative exponent).
        let bw = f_stop - f_start;
        let w = Complex::cis(-two_pi * bw / (m_f * fs));

        Self::new(n, m, a, w)
    }

    /// Return the input length `N`.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Return the output length `M`.
    #[must_use]
    pub fn m(&self) -> usize {
        self.m
    }

    /// Return the internal convolution length `L`.
    #[must_use]
    pub fn l(&self) -> usize {
        self.l
    }
}
