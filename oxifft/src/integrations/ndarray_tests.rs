//! Tests for the ndarray FFT extension traits.
//!
//! Gated by `#[cfg(all(test, feature = "ndarray"))]` in integrations/mod.rs.

use ndarray::{Array1, Array2};

use crate::integrations::ndarray_ext::{FftExt, NdarrayFftError, RealFftExt};
use crate::kernel::Complex;
use crate::{Direction, Flags, Plan, RealPlan};

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn max_abs_diff(a: &[Complex<f64>], b: &[Complex<f64>]) -> f64 {
    a.iter().zip(b.iter()).fold(0.0_f64, |acc, (x, y)| {
        let dr = (x.re - y.re).abs();
        let di = (x.im - y.im).abs();
        let d = if dr > di { dr } else { di };
        if d > acc {
            d
        } else {
            acc
        }
    })
}

// ─── 1D: fft roundtrip ────────────────────────────────────────────────────────

/// Forward FFT followed by inverse FFT (scaled by 1/N) must recover the original.
#[test]
fn test_1d_fft_roundtrip() {
    let n = 64_usize;
    let original: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new(i as f64, (n - i) as f64))
        .collect();

    let arr = Array1::from_vec(original.clone());

    let spectrum = arr.fft().expect("fft failed");
    assert_eq!(spectrum.len(), n);

    // Collect the spectrum as a 1D dynamic array and IFFT it
    let spec_1d: Array1<Complex<f64>> = spectrum
        .into_dimensionality::<ndarray::Ix1>()
        .expect("reshape to Ix1");

    let recovered = spec_1d.ifft().expect("ifft failed");
    assert_eq!(recovered.len(), n);

    // Scale by 1/N
    let scale = 1.0_f64 / n as f64;
    let recovered_scaled: Vec<Complex<f64>> = recovered
        .iter()
        .map(|c| Complex::new(c.re * scale, c.im * scale))
        .collect();

    let err = max_abs_diff(&original, &recovered_scaled);
    assert!(err < 1e-12, "Roundtrip error {err:.2e} exceeds 1e-12");
}

// ─── 1D: parity vs Plan::dft_1d ──────────────────────────────────────────────

/// FftExt::fft must produce the same result as Plan::dft_1d.execute().
#[test]
fn test_1d_parity_vs_plan() {
    let n = 32_usize;
    let data: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
        .collect();

    // Reference: direct Plan
    let plan =
        Plan::<f64>::dft_1d(n, Direction::Forward, Flags::ESTIMATE).expect("plan creation failed");
    let mut expected = vec![Complex::<f64>::zero(); n];
    plan.execute(&data, &mut expected);

    // Extension trait
    let arr = Array1::from_vec(data);
    let result = arr.fft().expect("fft ext failed");
    let result_vec: Vec<Complex<f64>> = result.iter().copied().collect();

    let err = max_abs_diff(&result_vec, &expected);
    assert!(err < 1e-12, "Parity error {err:.2e} exceeds 1e-12");
}

// ─── 1D: fft_inplace ─────────────────────────────────────────────────────────

/// In-place forward FFT must match the out-of-place version.
#[test]
fn test_1d_fft_inplace_matches_fft() {
    let n = 16_usize;
    let data: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(i as f64, 0.0)).collect();

    let arr_out = Array1::from_vec(data.clone());
    let out_of_place = arr_out.fft().expect("fft failed");
    let out_vec: Vec<Complex<f64>> = out_of_place.iter().copied().collect();

    let mut arr_inplace = Array1::from_vec(data);
    arr_inplace.fft_inplace().expect("fft_inplace failed");
    let inplace_vec: Vec<Complex<f64>> = arr_inplace.iter().copied().collect();

    let err = max_abs_diff(&inplace_vec, &out_vec);
    assert!(err < 1e-12, "In-place/out-of-place mismatch {err:.2e}");
}

// ─── 2D: fft roundtrip ────────────────────────────────────────────────────────

/// 2D fft followed by ifft (scaled by 1/(rows×cols)) must recover the original.
#[test]
fn test_2d_fft_roundtrip() {
    let rows = 8_usize;
    let cols = 8_usize;
    let total = rows * cols;

    let flat: Vec<Complex<f64>> = (0..total)
        .map(|i| Complex::new(i as f64, -(i as f64)))
        .collect();

    let arr = Array2::from_shape_vec((rows, cols), flat.clone()).expect("shape error");

    let spectrum = arr.fft().expect("2D fft failed");
    assert_eq!(spectrum.len(), total);

    // Reshape back to 2D
    let spec_2d: Array2<Complex<f64>> = spectrum
        .into_dimensionality::<ndarray::Ix2>()
        .expect("reshape to Ix2");

    let recovered_dyn = spec_2d.ifft().expect("2D ifft failed");
    let scale = 1.0_f64 / total as f64;
    let recovered: Vec<Complex<f64>> = recovered_dyn
        .iter()
        .map(|c| Complex::new(c.re * scale, c.im * scale))
        .collect();

    let err = max_abs_diff(&flat, &recovered);
    assert!(err < 1e-10, "2D roundtrip error {err:.2e} exceeds 1e-10");
}

// ─── 2D: fft parity vs row-then-column Plan ───────────────────────────────────

/// 2D FftExt::fft must equal manually applying 1D FFT row-wise then column-wise.
#[test]
fn test_2d_parity_vs_row_col_plans() {
    let rows = 4_usize;
    let cols = 8_usize;

    let flat: Vec<Complex<f64>> = (0..rows * cols)
        .map(|i| Complex::new((i as f64).sin(), (i as f64 * 0.5).cos()))
        .collect();

    // Manual row-column approach runs on an independent copy of flat
    let arr = Array2::from_shape_vec((rows, cols), flat.clone()).expect("shape error");

    // Extension trait result
    let ext_result = arr.fft().expect("2D fft ext failed");
    let ext_vec: Vec<Complex<f64>> = ext_result.iter().copied().collect();

    let row_plan =
        Plan::<f64>::dft_1d(cols, Direction::Forward, Flags::ESTIMATE).expect("row plan");
    let col_plan =
        Plan::<f64>::dft_1d(rows, Direction::Forward, Flags::ESTIMATE).expect("col plan");

    // Reuse flat by moving into buf (no second clone needed)
    let mut buf = flat;
    let mut row_out = vec![Complex::<f64>::zero(); cols];
    for r in 0..rows {
        let start = r * cols;
        row_plan.execute(&buf[start..start + cols], &mut row_out);
        buf[start..start + cols].copy_from_slice(&row_out);
    }
    let mut col_in = vec![Complex::<f64>::zero(); rows];
    let mut col_out = vec![Complex::<f64>::zero(); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_in[r] = buf[r * cols + c];
        }
        col_plan.execute(&col_in, &mut col_out);
        for r in 0..rows {
            buf[r * cols + c] = col_out[r];
        }
    }

    let err = max_abs_diff(&ext_vec, &buf);
    assert!(err < 1e-12, "2D parity error {err:.2e} exceeds 1e-12");
}

// ─── 2D: fft_inplace ─────────────────────────────────────────────────────────

/// In-place 2D FFT must match the out-of-place version.
#[test]
fn test_2d_fft_inplace_matches_fft() {
    let rows = 8_usize;
    let cols = 4_usize;
    let total = rows * cols;

    let flat: Vec<Complex<f64>> = (0..total).map(|i| Complex::new(i as f64, 0.0)).collect();

    let arr_oop = Array2::from_shape_vec((rows, cols), flat.clone()).expect("shape error");
    let out_of_place = arr_oop.fft().expect("2D fft failed");
    let oop_vec: Vec<Complex<f64>> = out_of_place.iter().copied().collect();

    let mut arr_ip = Array2::from_shape_vec((rows, cols), flat).expect("shape error");
    arr_ip.fft_inplace().expect("2D fft_inplace failed");
    let ip_vec: Vec<Complex<f64>> = arr_ip.iter().copied().collect();

    let err = max_abs_diff(&ip_vec, &oop_vec);
    assert!(err < 1e-12, "2D in-place/out-of-place mismatch {err:.2e}");
}

// ─── Non-contiguous mutable view ─────────────────────────────────────────────

/// A mutable sub-array view (non-contiguous in memory) transformed with
/// `fft_inplace` must produce the same result as converting it to an owned
/// array and calling `fft`.
///
/// `arr.slice_mut(s![1.., ..])` returns an `ArrayViewMut2` whose rows are not
/// adjacent in memory, exercising the scratch-buffer gather/scatter paths
/// inside `fft_inplace`.
#[test]
fn test_noncontiguous_view_fft_inplace() {
    let rows = 8_usize;
    let cols = 16_usize;
    let sub_rows = rows - 1; // rows 1..8

    let flat: Vec<Complex<f64>> = (0..rows * cols)
        .map(|i| Complex::new(i as f64, 0.0))
        .collect();

    let mut arr_inplace = Array2::from_shape_vec((rows, cols), flat.clone()).expect("shape error");
    let arr_ref = Array2::from_shape_vec((rows, cols), flat).expect("shape error");

    // Reference: extract sub-array as owned, run fft
    let sub_owned: Array2<Complex<f64>> = arr_ref.slice(ndarray::s![1.., ..]).to_owned();
    let reference = sub_owned.fft().expect("reference fft failed");
    let ref_vec: Vec<Complex<f64>> = reference.iter().copied().collect();

    // Actual: run fft_inplace on the mutable non-contiguous view directly
    arr_inplace
        .slice_mut(ndarray::s![1.., ..])
        .fft_inplace()
        .expect("fft_inplace on mutable view failed");

    let result_vec: Vec<Complex<f64>> = arr_inplace
        .slice(ndarray::s![1.., ..])
        .iter()
        .copied()
        .collect();

    assert_eq!(result_vec.len(), sub_rows * cols);

    let err = max_abs_diff(&result_vec, &ref_vec);
    assert!(
        err < 1e-12,
        "Non-contiguous view FFT mismatch: err = {err:.2e}"
    );

    // Rows 0..1 must be untouched
    let row0_unchanged: bool = arr_inplace
        .row(0)
        .iter()
        .enumerate()
        .all(|(c, &v)| v.re == c as f64 && v.im == 0.0);
    assert!(row0_unchanged, "Row 0 was mutated but should be untouched");
}

// ─── Real-input fft_real ──────────────────────────────────────────────────────

/// fft_real must produce the same spectrum as RealPlan::execute_r2c.
#[test]
fn test_real_fft_parity_vs_real_plan() {
    let n = 64_usize;
    let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

    // Reference
    let plan = RealPlan::<f64>::r2c_1d(n, Flags::ESTIMATE).expect("real plan failed");
    let expected_len = plan.complex_size();
    let mut expected = vec![Complex::<f64>::zero(); expected_len];
    plan.execute_r2c(&data, &mut expected);

    // Extension trait
    let arr = Array1::from_vec(data);
    let result = arr.fft_real().expect("fft_real failed");
    assert_eq!(result.len(), expected_len);

    let result_vec: Vec<Complex<f64>> = result.iter().copied().collect();
    let err = max_abs_diff(&result_vec, &expected);
    assert!(err < 1e-12, "Real FFT parity error {err:.2e} exceeds 1e-12");
}

// ─── DC bin correctness ────────────────────────────────────────────────────────

/// For an all-ones 1D array, the DC bin must equal N (sum of all elements).
#[test]
fn test_1d_dc_bin() {
    let n = 32_usize;
    let arr = Array1::from_vec(vec![Complex::<f64>::new(1.0, 0.0); n]);
    let result = arr.fft().expect("fft failed");
    let dc = result[ndarray::IxDyn(&[0])];
    let dc_err = (dc.re - n as f64).abs();
    assert!(
        dc_err < 1e-10,
        "DC bin {:.4} != {n} (err = {dc_err:.2e})",
        dc.re
    );
    // All other bins must be zero for constant input
    for k in 1..n {
        let bin = result[ndarray::IxDyn(&[k])];
        assert!(
            bin.re.abs() < 1e-10 && bin.im.abs() < 1e-10,
            "Bin {k} should be zero for constant input"
        );
    }
}

// ─── Error: empty array ────────────────────────────────────────────────────────

#[test]
fn test_empty_1d_returns_error() {
    let arr: Array1<Complex<f64>> = Array1::from_vec(vec![]);
    assert_eq!(arr.fft(), Err(NdarrayFftError::EmptyArray));
    assert_eq!(arr.ifft(), Err(NdarrayFftError::EmptyArray));
}

#[test]
fn test_empty_real_returns_error() {
    let arr: Array1<f64> = Array1::from_vec(vec![]);
    assert_eq!(arr.fft_real(), Err(NdarrayFftError::EmptyArray));
}
