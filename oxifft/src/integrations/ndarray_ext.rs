//! ndarray integration for OxiFFT.
//!
//! Provides extension traits [`FftExt`] and [`RealFftExt`] that add FFT methods
//! directly to ndarray arrays. Gated by the `ndarray` cargo feature.
//!
//! Arrays must use [`crate::Complex<T>`] as the element type for complex
//! transforms, or `T: Float` for the real-input `RealFftExt` trait.
//!
//! # Examples
//!
//! ```
//! # #[cfg(feature = "ndarray")] {
//! use ndarray::Array1;
//! use oxifft::integrations::ndarray_ext::{FftExt, NdarrayFftError};
//! use oxifft::Complex;
//!
//! let data: Vec<Complex<f64>> = (0..8)
//!     .map(|i| Complex::new(i as f64, 0.0))
//!     .collect();
//! let arr = Array1::from_vec(data);
//! let spectrum = arr.fft().expect("fft failed");
//! // DC bin = 0+1+...+7 = 28
//! assert!((spectrum[ndarray::IxDyn(&[0])].re - 28.0_f64).abs() < 1e-9);
//! # }
//! ```

use ndarray::{Array, ArrayBase, Data, DataMut, Ix1, Ix2, IxDyn};

use crate::kernel::{Complex, Float};
use crate::{Direction, Flags, Plan, RealPlan};

// ─── Error type ────────────────────────────────────────────────────────────────

/// Errors that can occur when performing ndarray FFT operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NdarrayFftError {
    /// Plan construction failed (e.g., zero-length array).
    PlanCreationFailed,
    /// The array has zero elements — no transform is possible.
    EmptyArray,
    /// Shape conversion failed (internal error).
    ShapeError,
}

impl core::fmt::Display for NdarrayFftError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::PlanCreationFailed => write!(f, "FFT plan creation failed"),
            Self::EmptyArray => write!(f, "Cannot FFT an empty array"),
            Self::ShapeError => write!(f, "Failed to construct ndarray with given shape"),
        }
    }
}

// ─── Helper: build a Plan or return an error ───────────────────────────────────

fn make_plan<T: Float>(n: usize, dir: Direction) -> Result<Plan<T>, NdarrayFftError> {
    if n == 0 {
        return Err(NdarrayFftError::EmptyArray);
    }
    Plan::<T>::dft_1d(n, dir, Flags::ESTIMATE).ok_or(NdarrayFftError::PlanCreationFailed)
}

fn make_real_plan<T: Float>(n: usize) -> Result<RealPlan<T>, NdarrayFftError> {
    if n == 0 {
        return Err(NdarrayFftError::EmptyArray);
    }
    RealPlan::<T>::r2c_1d(n, Flags::ESTIMATE).ok_or(NdarrayFftError::PlanCreationFailed)
}

// ─── Complex FFT extension trait ───────────────────────────────────────────────

/// Extension trait adding FFT methods to ndarray arrays of [`Complex<T>`].
///
/// Implemented for:
/// - `ArrayBase<S, Ix1>` — 1D arrays (N-point DFT)
/// - `ArrayBase<S, Ix2>` — 2D arrays (row-column 2D DFT)
///
/// All transforms follow FFTW convention: **unnormalized forward, unnormalized
/// inverse**. Divide by N (or N×M for 2D) after `ifft` to recover the original.
///
/// Requires the `ndarray` cargo feature.
pub trait FftExt<T: Float> {
    /// Forward FFT, returning a new dynamically-dimensioned array.
    ///
    /// For 1D: N-point forward DFT (unnormalized).
    /// For 2D: row-column 2D forward DFT (unnormalized).
    ///
    /// # Errors
    /// Returns [`NdarrayFftError`] if the array is empty or plan creation fails.
    fn fft(&self) -> Result<Array<Complex<T>, IxDyn>, NdarrayFftError>;

    /// Inverse FFT, returning a new dynamically-dimensioned array.
    ///
    /// Result is **unnormalized** (FFTW convention). Divide by N (or N×M for 2D)
    /// after calling to recover the original signal.
    ///
    /// # Errors
    /// Returns [`NdarrayFftError`] if the array is empty or plan creation fails.
    fn ifft(&self) -> Result<Array<Complex<T>, IxDyn>, NdarrayFftError>;

    /// In-place forward FFT, modifying `self`.
    ///
    /// # Errors
    /// Returns [`NdarrayFftError`] if the array is empty or plan creation fails.
    fn fft_inplace(&mut self) -> Result<(), NdarrayFftError>;

    /// In-place inverse FFT, modifying `self`.
    ///
    /// Result is **unnormalized**.
    ///
    /// # Errors
    /// Returns [`NdarrayFftError`] if the array is empty or plan creation fails.
    fn ifft_inplace(&mut self) -> Result<(), NdarrayFftError>;
}

// ─── 1D implementation ─────────────────────────────────────────────────────────

impl<T: Float, S: DataMut<Elem = Complex<T>>> FftExt<T> for ArrayBase<S, Ix1> {
    fn fft(&self) -> Result<Array<Complex<T>, IxDyn>, NdarrayFftError> {
        let n = self.len();
        let plan = make_plan::<T>(n, Direction::Forward)?;
        let input: Vec<Complex<T>> = self.iter().copied().collect();
        let mut output = vec![Complex::<T>::zero(); n];
        plan.execute(&input, &mut output);
        Array::from_shape_vec(IxDyn(&[n]), output).map_err(|_| NdarrayFftError::ShapeError)
    }

    fn ifft(&self) -> Result<Array<Complex<T>, IxDyn>, NdarrayFftError> {
        let n = self.len();
        let plan = make_plan::<T>(n, Direction::Backward)?;
        let input: Vec<Complex<T>> = self.iter().copied().collect();
        let mut output = vec![Complex::<T>::zero(); n];
        plan.execute(&input, &mut output);
        Array::from_shape_vec(IxDyn(&[n]), output).map_err(|_| NdarrayFftError::ShapeError)
    }

    fn fft_inplace(&mut self) -> Result<(), NdarrayFftError> {
        let n = self.len();
        let plan = make_plan::<T>(n, Direction::Forward)?;
        let input: Vec<Complex<T>> = self.iter().copied().collect();
        let mut output = vec![Complex::<T>::zero(); n];
        plan.execute(&input, &mut output);
        for (dst, src) in self.iter_mut().zip(output.iter()) {
            *dst = *src;
        }
        Ok(())
    }

    fn ifft_inplace(&mut self) -> Result<(), NdarrayFftError> {
        let n = self.len();
        let plan = make_plan::<T>(n, Direction::Backward)?;
        let input: Vec<Complex<T>> = self.iter().copied().collect();
        let mut output = vec![Complex::<T>::zero(); n];
        plan.execute(&input, &mut output);
        for (dst, src) in self.iter_mut().zip(output.iter()) {
            *dst = *src;
        }
        Ok(())
    }
}

// ─── 2D implementation ─────────────────────────────────────────────────────────

impl<T: Float, S: DataMut<Elem = Complex<T>>> FftExt<T> for ArrayBase<S, Ix2> {
    fn fft(&self) -> Result<Array<Complex<T>, IxDyn>, NdarrayFftError> {
        let (rows, cols) = self.dim();
        let row_plan = make_plan::<T>(cols, Direction::Forward)?;
        let col_plan = make_plan::<T>(rows, Direction::Forward)?;

        // Flat row-major scratch buffer
        let mut buf: Vec<Complex<T>> = self.iter().copied().collect();

        // Row FFTs
        let mut row_out = vec![Complex::<T>::zero(); cols];
        for row_idx in 0..rows {
            let start = row_idx * cols;
            row_plan.execute(&buf[start..start + cols], &mut row_out);
            buf[start..start + cols].copy_from_slice(&row_out);
        }

        // Column FFTs
        let mut col_in = vec![Complex::<T>::zero(); rows];
        let mut col_out = vec![Complex::<T>::zero(); rows];
        for col_idx in 0..cols {
            for (r, val) in col_in.iter_mut().enumerate() {
                *val = buf[r * cols + col_idx];
            }
            col_plan.execute(&col_in, &mut col_out);
            for (r, val) in col_out.iter().enumerate() {
                buf[r * cols + col_idx] = *val;
            }
        }

        Array::from_shape_vec(IxDyn(&[rows, cols]), buf).map_err(|_| NdarrayFftError::ShapeError)
    }

    fn ifft(&self) -> Result<Array<Complex<T>, IxDyn>, NdarrayFftError> {
        let (rows, cols) = self.dim();
        let row_plan = make_plan::<T>(cols, Direction::Backward)?;
        let col_plan = make_plan::<T>(rows, Direction::Backward)?;

        let mut buf: Vec<Complex<T>> = self.iter().copied().collect();

        // Inverse row FFTs
        let mut row_out = vec![Complex::<T>::zero(); cols];
        for row_idx in 0..rows {
            let start = row_idx * cols;
            row_plan.execute(&buf[start..start + cols], &mut row_out);
            buf[start..start + cols].copy_from_slice(&row_out);
        }

        // Inverse column FFTs
        let mut col_in = vec![Complex::<T>::zero(); rows];
        let mut col_out = vec![Complex::<T>::zero(); rows];
        for col_idx in 0..cols {
            for (r, val) in col_in.iter_mut().enumerate() {
                *val = buf[r * cols + col_idx];
            }
            col_plan.execute(&col_in, &mut col_out);
            for (r, val) in col_out.iter().enumerate() {
                buf[r * cols + col_idx] = *val;
            }
        }

        Array::from_shape_vec(IxDyn(&[rows, cols]), buf).map_err(|_| NdarrayFftError::ShapeError)
    }

    fn fft_inplace(&mut self) -> Result<(), NdarrayFftError> {
        let (rows, cols) = self.dim();
        let row_plan = make_plan::<T>(cols, Direction::Forward)?;
        let col_plan = make_plan::<T>(rows, Direction::Forward)?;

        // Row pass: read from self, write back after FFT
        let mut row_in = vec![Complex::<T>::zero(); cols];
        let mut row_out = vec![Complex::<T>::zero(); cols];
        for row_idx in 0..rows {
            for (c, val) in row_in.iter_mut().enumerate() {
                *val = self[[row_idx, c]];
            }
            row_plan.execute(&row_in, &mut row_out);
            for (c, val) in row_out.iter().enumerate() {
                self[[row_idx, c]] = *val;
            }
        }

        // Column pass
        let mut col_in = vec![Complex::<T>::zero(); rows];
        let mut col_out = vec![Complex::<T>::zero(); rows];
        for col_idx in 0..cols {
            for (r, val) in col_in.iter_mut().enumerate() {
                *val = self[[r, col_idx]];
            }
            col_plan.execute(&col_in, &mut col_out);
            for (r, val) in col_out.iter().enumerate() {
                self[[r, col_idx]] = *val;
            }
        }
        Ok(())
    }

    fn ifft_inplace(&mut self) -> Result<(), NdarrayFftError> {
        let (rows, cols) = self.dim();
        let row_plan = make_plan::<T>(cols, Direction::Backward)?;
        let col_plan = make_plan::<T>(rows, Direction::Backward)?;

        // Row pass
        let mut row_in = vec![Complex::<T>::zero(); cols];
        let mut row_out = vec![Complex::<T>::zero(); cols];
        for row_idx in 0..rows {
            for (c, val) in row_in.iter_mut().enumerate() {
                *val = self[[row_idx, c]];
            }
            row_plan.execute(&row_in, &mut row_out);
            for (c, val) in row_out.iter().enumerate() {
                self[[row_idx, c]] = *val;
            }
        }

        // Column pass
        let mut col_in = vec![Complex::<T>::zero(); rows];
        let mut col_out = vec![Complex::<T>::zero(); rows];
        for col_idx in 0..cols {
            for (r, val) in col_in.iter_mut().enumerate() {
                *val = self[[r, col_idx]];
            }
            col_plan.execute(&col_in, &mut col_out);
            for (r, val) in col_out.iter().enumerate() {
                self[[r, col_idx]] = *val;
            }
        }
        Ok(())
    }
}

// ─── Real-input FFT extension trait ────────────────────────────────────────────

/// Extension trait adding real-to-complex FFT to 1D ndarray arrays of real `T`.
///
/// Requires the `ndarray` cargo feature.
pub trait RealFftExt<T: Float> {
    /// Compute the R2C FFT.
    ///
    /// Returns an array of [`Complex<T>`] with `N/2 + 1` frequency bins
    /// (the non-redundant half of the spectrum, exploiting Hermitian symmetry).
    ///
    /// The output is **unnormalized**.
    ///
    /// # Errors
    /// Returns [`NdarrayFftError`] if the array is empty or plan creation fails.
    fn fft_real(&self) -> Result<Array<Complex<T>, IxDyn>, NdarrayFftError>;
}

impl<T: Float, S: Data<Elem = T>> RealFftExt<T> for ArrayBase<S, Ix1> {
    fn fft_real(&self) -> Result<Array<Complex<T>, IxDyn>, NdarrayFftError> {
        let n = self.len();
        let plan = make_real_plan::<T>(n)?;
        let input: Vec<T> = self.iter().copied().collect();
        let out_len = plan.complex_size();
        let mut output = vec![Complex::<T>::zero(); out_len];
        plan.execute_r2c(&input, &mut output);
        Array::from_shape_vec(IxDyn(&[out_len]), output).map_err(|_| NdarrayFftError::ShapeError)
    }
}
