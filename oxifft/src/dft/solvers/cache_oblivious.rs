//! Cache-oblivious FFT solver.
//!
//! Implements the four-step FFT algorithm with recursive decomposition,
//! based on the Frigo/Johnson approach. The key insight is that by decomposing
//! N = N1 × N2 and performing column FFTs, twiddle multiplication, and row FFTs,
//! the algorithm naturally adapts to any cache hierarchy without needing to know
//! cache sizes.
//!
//! For large transforms (≥ threshold), this solver:
//! 1. Views N-point data as an N1 × N2 matrix (row-major)
//! 2. Computes N2 column-FFTs of size N1
//! 3. Multiplies by twiddle factors
//! 4. Computes N1 row-FFTs of size N2
//! 5. Transposes the result
//!
//! The recursive decomposition continues until sub-problems are small enough
//! to be handled by a base solver (Cooley-Tukey or codelet).

use crate::dft::problem::Sign;
use crate::kernel::{Complex, Float};
use crate::prelude::*;

use super::ct::{CooleyTukeySolver, CtVariant};

/// Minimum size for cache-oblivious decomposition.
/// Below this threshold, we fall back to Cooley-Tukey.
const CACHE_OBLIVIOUS_THRESHOLD: usize = 1024;

/// Block size for cache-friendly matrix transpose.
const TRANSPOSE_BLOCK_SIZE: usize = 64;

/// Cache-oblivious FFT solver.
///
/// Uses the four-step FFT algorithm with recursive decomposition.
/// Decomposes N = N1 × N2 and performs:
/// 1. N2 column-FFTs of size N1
/// 2. Twiddle factor multiplication
/// 3. N1 row-FFTs of size N2
/// 4. Matrix transpose
///
/// This approach naturally fits in cache at every level of the memory hierarchy
/// without needing to know cache sizes explicitly.
///
/// Time complexity: O(n log n)
/// Space complexity: O(n) for scratch buffer
pub struct CacheObliviousSolver<T: Float> {
    /// Base solver for sub-problems that fit in cache.
    base_solver: CooleyTukeySolver<T>,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> Default for CacheObliviousSolver<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> CacheObliviousSolver<T> {
    /// Create a new cache-oblivious solver.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base_solver: CooleyTukeySolver::new(CtVariant::Dit),
            _marker: core::marker::PhantomData,
        }
    }

    /// Solver name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        "dft-cache-oblivious"
    }

    /// Check if size is applicable (must be power of 2, ≥ threshold).
    #[must_use]
    pub fn applicable(n: usize) -> bool {
        n.is_power_of_two() && n >= CACHE_OBLIVIOUS_THRESHOLD
    }

    /// Execute the cache-oblivious FFT.
    ///
    /// # Arguments
    /// * `input` - Input array of complex values
    /// * `output` - Output array (must be same length as input)
    /// * `sign` - Transform direction (Forward or Backward)
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        let n = input.len();
        debug_assert_eq!(n, output.len());
        debug_assert!(n.is_power_of_two(), "Size must be power of 2");

        if n < CACHE_OBLIVIOUS_THRESHOLD {
            // Fall back to base solver for small sizes
            self.base_solver.execute(input, output, sign);
            return;
        }

        // Decompose N = N1 × N2 with N1 ≈ √N
        let (n1, n2) = balanced_factorization(n);

        // Working buffer: we need n elements of scratch space
        let mut scratch = vec![Complex::<T>::zero(); n];

        // Step 1: Copy input into scratch, viewed as N1×N2 row-major matrix
        scratch.copy_from_slice(input);

        // Step 2: Compute N2 column-FFTs of size N1
        // Each column j has elements at indices j, j+N2, j+2*N2, ..., j+(N1-1)*N2
        self.execute_column_ffts(&mut scratch, n1, n2, sign);

        // Step 3: Multiply by twiddle factors W_N^(j*k) for row j, column k
        apply_twiddle_factors(&mut scratch, n1, n2, n, sign);

        // Step 4: Compute N1 row-FFTs of size N2
        // Each row i has elements at indices i*N2, i*N2+1, ..., i*N2+(N2-1)
        self.execute_row_ffts(&scratch, output, n1, n2, sign);

        // Step 5: Transpose output from row-major N1×N2 to column-major (N2×N1)
        // The result needs to be read in column-major order
        transpose_output(output, n1, n2);
    }

    /// Execute in-place cache-oblivious FFT.
    pub fn execute_inplace(&self, data: &mut [Complex<T>], sign: Sign) {
        let n = data.len();
        if n < CACHE_OBLIVIOUS_THRESHOLD {
            self.base_solver.execute_inplace(data, sign);
            return;
        }

        // Need temporary buffer for out-of-place execution
        let input: Vec<Complex<T>> = data.to_vec();
        self.execute(&input, data, sign);
    }

    /// Execute column FFTs on an N1×N2 matrix stored in row-major order.
    ///
    /// For each column j (0..N2), gather elements at stride N2,
    /// perform FFT of size N1, and scatter back.
    fn execute_column_ffts(&self, matrix: &mut [Complex<T>], n1: usize, n2: usize, sign: Sign) {
        let mut col_buf = vec![Complex::<T>::zero(); n1];
        let mut col_out = vec![Complex::<T>::zero(); n1];

        for j in 0..n2 {
            // Gather column j: elements at indices j, j+N2, j+2*N2, ...
            for i in 0..n1 {
                col_buf[i] = matrix[i * n2 + j];
            }

            // Execute FFT of size N1 on this column
            if n1 >= CACHE_OBLIVIOUS_THRESHOLD {
                // Recurse for large sub-problems
                self.execute(&col_buf, &mut col_out, sign);
            } else {
                self.base_solver.execute(&col_buf, &mut col_out, sign);
            }

            // Scatter back into column j
            for i in 0..n1 {
                matrix[i * n2 + j] = col_out[i];
            }
        }
    }

    /// Execute row FFTs on an N1×N2 matrix stored in row-major order.
    ///
    /// For each row i (0..N1), the elements are contiguous: matrix[i*N2 .. (i+1)*N2].
    fn execute_row_ffts(
        &self,
        matrix: &[Complex<T>],
        output: &mut [Complex<T>],
        n1: usize,
        n2: usize,
        sign: Sign,
    ) {
        let mut row_out = vec![Complex::<T>::zero(); n2];

        for i in 0..n1 {
            let row_start = i * n2;
            let row_end = row_start + n2;
            let row_in = &matrix[row_start..row_end];

            // Execute FFT of size N2 on this row
            if n2 >= CACHE_OBLIVIOUS_THRESHOLD {
                // Recurse for large sub-problems
                self.execute(row_in, &mut row_out, sign);
            } else {
                self.base_solver.execute(row_in, &mut row_out, sign);
            }

            // Store result
            output[row_start..row_end].copy_from_slice(&row_out);
        }
    }
}

/// Choose a balanced factorization N = N1 × N2 where N1 ≈ √N.
///
/// For power-of-2 N, we pick N1 = 2^(log2(N)/2) so both factors are
/// as close to √N as possible. This ensures balanced recursive decomposition.
fn balanced_factorization(n: usize) -> (usize, usize) {
    debug_assert!(n.is_power_of_two());
    let log_n = n.trailing_zeros();
    // Split the bits roughly in half
    let log_n1 = log_n / 2;
    let n1 = 1usize << log_n1;
    let n2 = n / n1;
    (n1, n2)
}

/// Apply twiddle factors to the matrix after column FFTs.
///
/// For element (i, j) in the N1×N2 matrix, multiply by W_N^(i*j)
/// where W_N = e^(sign * 2πi / N).
fn apply_twiddle_factors<T: Float>(
    matrix: &mut [Complex<T>],
    n1: usize,
    n2: usize,
    n: usize,
    sign: Sign,
) {
    let sign_val = T::from_isize(sign.value() as isize);
    let two_pi_over_n = T::TWO_PI / T::from_usize(n);

    for i in 0..n1 {
        // Skip row 0 (twiddle factor is always 1)
        if i == 0 {
            continue;
        }
        let row_start = i * n2;
        for j in 1..n2 {
            // W_N^(i*j) = e^(sign * 2πi * i * j / N)
            let angle = sign_val * two_pi_over_n * T::from_usize(i) * T::from_usize(j);
            let twiddle = Complex::cis(angle);
            matrix[row_start + j] = matrix[row_start + j] * twiddle;
        }
    }
}

/// Cache-friendly blocked matrix transpose.
///
/// Transposes an N1×N2 row-major matrix in-place using blocking
/// to improve cache utilization. The result is the transposed matrix
/// stored in row-major order (i.e., N2×N1).
fn transpose_output<T: Float>(data: &mut [Complex<T>], n1: usize, n2: usize) {
    if n1 == n2 {
        // Square matrix: can transpose in-place directly
        transpose_square_blocked(data, n1);
    } else {
        // Non-square: need temporary buffer
        transpose_rectangular(data, n1, n2);
    }
}

/// In-place blocked transpose for square N×N matrix stored in row-major order.
fn transpose_square_blocked<T: Float>(data: &mut [Complex<T>], n: usize) {
    let block = TRANSPOSE_BLOCK_SIZE.min(n);

    // Process blocks
    let mut bi = 0;
    while bi < n {
        let bi_end = (bi + block).min(n);
        let mut bj = bi;
        while bj < n {
            let bj_end = (bj + block).min(n);

            if bi == bj {
                // Diagonal block: swap upper-triangle with lower-triangle
                for i in bi..bi_end {
                    for j in (i + 1)..bj_end {
                        let idx_ij = i * n + j;
                        let idx_ji = j * n + i;
                        data.swap(idx_ij, idx_ji);
                    }
                }
            } else {
                // Off-diagonal block pair: swap block (bi,bj) with block (bj,bi)
                for i in bi..bi_end {
                    for j in bj..bj_end {
                        let idx_ij = i * n + j;
                        let idx_ji = j * n + i;
                        data.swap(idx_ij, idx_ji);
                    }
                }
            }

            bj += block;
        }
        bi += block;
    }
}

/// Out-of-place transpose for rectangular N1×N2 matrix.
///
/// Transposes N1×N2 row-major into N2×N1 row-major using a temporary buffer.
fn transpose_rectangular<T: Float>(data: &mut [Complex<T>], n1: usize, n2: usize) {
    let total = n1 * n2;
    let mut temp = vec![Complex::<T>::zero(); total];

    // Blocked copy for cache efficiency
    let block = TRANSPOSE_BLOCK_SIZE.min(n1.min(n2));

    let mut bi = 0;
    while bi < n1 {
        let bi_end = (bi + block).min(n1);
        let mut bj = 0;
        while bj < n2 {
            let bj_end = (bj + block).min(n2);

            for i in bi..bi_end {
                for j in bj..bj_end {
                    // Source: row i, col j in N1×N2
                    // Dest: row j, col i in N2×N1
                    temp[j * n1 + i] = data[i * n2 + j];
                }
            }

            bj += block;
        }
        bi += block;
    }

    data[..total].copy_from_slice(&temp);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dft::solvers::direct::DirectSolver;

    /// Helper: check if two complex numbers are approximately equal.
    fn complex_approx_eq<T: Float>(a: Complex<T>, b: Complex<T>, tol: f64) -> bool {
        let dr = num_traits::Float::abs(a.re - b.re);
        let di = num_traits::Float::abs(a.im - b.im);
        dr < T::from_f64(tol) && di < T::from_f64(tol)
    }

    /// Helper: max absolute error between two complex arrays.
    fn max_abs_error<T: Float>(a: &[Complex<T>], b: &[Complex<T>]) -> f64 {
        let mut max_err = 0.0_f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let dr = num_traits::Float::abs((*x - *y).re);
            let di = num_traits::Float::abs((*x - *y).im);
            // Convert to f64 for comparison: use the fact that T: NumFloat
            let err_r = num_traits::ToPrimitive::to_f64(&dr).unwrap_or(f64::MAX);
            let err_i = num_traits::ToPrimitive::to_f64(&di).unwrap_or(f64::MAX);
            if err_r > max_err {
                max_err = err_r;
            }
            if err_i > max_err {
                max_err = err_i;
            }
        }
        max_err
    }

    #[test]
    fn test_balanced_factorization() {
        assert_eq!(balanced_factorization(1024), (32, 32));
        assert_eq!(balanced_factorization(2048), (32, 64));
        assert_eq!(balanced_factorization(4096), (64, 64));
        assert_eq!(balanced_factorization(8192), (64, 128));
        assert_eq!(balanced_factorization(16384), (128, 128));
    }

    #[test]
    fn test_transpose_square() {
        // 4x4 matrix
        let mut data: Vec<Complex<f64>> =
            (0..16).map(|i| Complex::new(f64::from(i), 0.0)).collect();
        let original = data.clone();
        transpose_square_blocked(&mut data, 4);

        // Check element (i,j) in original is at (j,i) in result
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    complex_approx_eq(data[j * 4 + i], original[i * 4 + j], 1e-15),
                    "Transpose failed at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn test_transpose_rectangular() {
        // 2x8 matrix -> 8x2 matrix
        let mut data: Vec<Complex<f64>> = (0..16)
            .map(|i| Complex::new(f64::from(i), f64::from(i) * 0.5))
            .collect();
        let original = data.clone();
        transpose_rectangular(&mut data, 2, 8);

        for i in 0..2 {
            for j in 0..8 {
                assert!(
                    complex_approx_eq(data[j * 2 + i], original[i * 8 + j], 1e-15),
                    "Transpose failed at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn test_cache_oblivious_vs_direct_1024() {
        let n = 1024;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let t = core::f64::consts::TAU * (i as f64) / (n as f64);
                Complex::new(t.sin(), t.cos())
            })
            .collect();

        let solver = CacheObliviousSolver::<f64>::new();
        let direct = DirectSolver::<f64>::new();

        let mut output_co = vec![Complex::zero(); n];
        let mut output_direct = vec![Complex::zero(); n];

        solver.execute(&input, &mut output_co, Sign::Forward);
        direct.execute(&input, &mut output_direct, Sign::Forward);

        let err = max_abs_error(&output_co, &output_direct);
        assert!(
            err < 1e-6,
            "Cache-oblivious vs direct error too large: {err} for N={n}"
        );
    }

    #[test]
    fn test_cache_oblivious_vs_direct_2048() {
        let n = 2048;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let t = core::f64::consts::TAU * (i as f64) / (n as f64);
                Complex::new(t.sin(), (2.0 * t).cos())
            })
            .collect();

        let solver = CacheObliviousSolver::<f64>::new();
        let direct = DirectSolver::<f64>::new();

        let mut output_co = vec![Complex::zero(); n];
        let mut output_direct = vec![Complex::zero(); n];

        solver.execute(&input, &mut output_co, Sign::Forward);
        direct.execute(&input, &mut output_direct, Sign::Forward);

        let err = max_abs_error(&output_co, &output_direct);
        assert!(
            err < 1e-6,
            "Cache-oblivious vs direct error too large: {err} for N={n}"
        );
    }

    #[test]
    fn test_cache_oblivious_vs_direct_4096() {
        let n = 4096;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let t = core::f64::consts::TAU * (i as f64) / (n as f64);
                Complex::new(t.sin(), (3.0 * t).cos())
            })
            .collect();

        let solver = CacheObliviousSolver::<f64>::new();
        let direct = DirectSolver::<f64>::new();

        let mut output_co = vec![Complex::zero(); n];
        let mut output_direct = vec![Complex::zero(); n];

        solver.execute(&input, &mut output_co, Sign::Forward);
        direct.execute(&input, &mut output_direct, Sign::Forward);

        let err = max_abs_error(&output_co, &output_direct);
        assert!(
            err < 1e-5,
            "Cache-oblivious vs direct error too large: {err} for N={n}"
        );
    }

    #[test]
    fn test_cache_oblivious_vs_ct_8192() {
        let n = 8192;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let t = core::f64::consts::TAU * (i as f64) / (n as f64);
                Complex::new(t.sin(), (5.0 * t).cos())
            })
            .collect();

        let solver = CacheObliviousSolver::<f64>::new();
        let ct_solver = CooleyTukeySolver::<f64>::new(CtVariant::Dit);

        let mut output_co = vec![Complex::zero(); n];
        let mut output_ct = vec![Complex::zero(); n];

        solver.execute(&input, &mut output_co, Sign::Forward);
        ct_solver.execute(&input, &mut output_ct, Sign::Forward);

        let err = max_abs_error(&output_co, &output_ct);
        assert!(
            err < 1e-6,
            "Cache-oblivious vs CT error too large: {err} for N={n}"
        );
    }

    #[test]
    fn test_cache_oblivious_round_trip_1024() {
        let n = 1024;
        let original: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(f64::from(i as u32), f64::from(i as u32) * 0.5))
            .collect();

        let solver = CacheObliviousSolver::<f64>::new();
        let mut transformed = vec![Complex::zero(); n];
        let mut recovered = vec![Complex::zero(); n];

        solver.execute(&original, &mut transformed, Sign::Forward);
        solver.execute(&transformed, &mut recovered, Sign::Backward);

        // Normalize
        let scale = 1.0 / n as f64;
        for x in &mut recovered {
            *x = Complex::new(x.re * scale, x.im * scale);
        }

        let err = max_abs_error(&original, &recovered);
        assert!(err < 1e-9, "Round-trip error too large: {err} for N={n}");
    }

    #[test]
    fn test_cache_oblivious_round_trip_4096() {
        let n = 4096;
        let original: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let t = core::f64::consts::TAU * (i as f64) / (n as f64);
                Complex::new(t.sin(), t.cos())
            })
            .collect();

        let solver = CacheObliviousSolver::<f64>::new();
        let mut transformed = vec![Complex::zero(); n];
        let mut recovered = vec![Complex::zero(); n];

        solver.execute(&original, &mut transformed, Sign::Forward);
        solver.execute(&transformed, &mut recovered, Sign::Backward);

        // Normalize
        let scale = 1.0 / n as f64;
        for x in &mut recovered {
            *x = Complex::new(x.re * scale, x.im * scale);
        }

        let err = max_abs_error(&original, &recovered);
        assert!(err < 1e-8, "Round-trip error too large: {err} for N={n}");
    }

    #[test]
    fn test_cache_oblivious_round_trip_8192() {
        let n = 8192;
        let original: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let t = core::f64::consts::TAU * (i as f64) / (n as f64);
                Complex::new(t.sin(), (7.0 * t).cos())
            })
            .collect();

        let solver = CacheObliviousSolver::<f64>::new();
        let mut transformed = vec![Complex::zero(); n];
        let mut recovered = vec![Complex::zero(); n];

        solver.execute(&original, &mut transformed, Sign::Forward);
        solver.execute(&transformed, &mut recovered, Sign::Backward);

        // Normalize
        let scale = 1.0 / n as f64;
        for x in &mut recovered {
            *x = Complex::new(x.re * scale, x.im * scale);
        }

        let err = max_abs_error(&original, &recovered);
        assert!(err < 1e-7, "Round-trip error too large: {err} for N={n}");
    }

    #[test]
    fn test_cache_oblivious_inplace_matches_outofplace() {
        let n = 1024;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let t = core::f64::consts::TAU * (i as f64) / (n as f64);
                Complex::new(t.sin(), t.cos())
            })
            .collect();

        let solver = CacheObliviousSolver::<f64>::new();

        let mut out_of_place = vec![Complex::zero(); n];
        solver.execute(&input, &mut out_of_place, Sign::Forward);

        let mut in_place = input;
        solver.execute_inplace(&mut in_place, Sign::Forward);

        let err = max_abs_error(&out_of_place, &in_place);
        assert!(err < 1e-15, "In-place vs out-of-place error: {err}");
    }

    #[test]
    fn test_cache_oblivious_f32() {
        let n = 1024;
        let input: Vec<Complex<f32>> = (0..n)
            .map(|i| {
                let t = core::f32::consts::TAU * (i as f32) / (n as f32);
                Complex::new(t.sin(), t.cos())
            })
            .collect();

        let solver = CacheObliviousSolver::<f32>::new();
        let ct_solver = CooleyTukeySolver::<f32>::new(CtVariant::Dit);

        let mut output_co = vec![Complex::zero(); n];
        let mut output_ct = vec![Complex::zero(); n];

        solver.execute(&input, &mut output_co, Sign::Forward);
        ct_solver.execute(&input, &mut output_ct, Sign::Forward);

        let err = max_abs_error(&output_co, &output_ct);
        assert!(
            err < 1e-2,
            "f32 cache-oblivious vs CT error too large: {err}"
        );
    }

    #[test]
    fn test_below_threshold_falls_back() {
        // Size 512 < threshold, should use base solver and still be correct
        let n = 512;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(f64::from(i as u32), 0.0))
            .collect();

        let solver = CacheObliviousSolver::<f64>::new();
        let ct_solver = CooleyTukeySolver::<f64>::new(CtVariant::Dit);

        let mut output_co = vec![Complex::zero(); n];
        let mut output_ct = vec![Complex::zero(); n];

        solver.execute(&input, &mut output_co, Sign::Forward);
        ct_solver.execute(&input, &mut output_ct, Sign::Forward);

        let err = max_abs_error(&output_co, &output_ct);
        assert!(err < 1e-10, "Below-threshold fallback error: {err}");
    }

    #[test]
    fn test_twiddle_factors_identity() {
        // For N=4 (N1=2, N2=2), twiddle factor at (0,j) and (i,0) should be 1
        let n = 4;
        let n1 = 2;
        let n2 = 2;
        let mut matrix: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); n];
        apply_twiddle_factors(&mut matrix, n1, n2, n, Sign::Forward);

        // (0,0), (0,1), (1,0) should be unchanged (twiddle = 1)
        assert!(complex_approx_eq(matrix[0], Complex::new(1.0, 0.0), 1e-15));
        assert!(complex_approx_eq(matrix[1], Complex::new(1.0, 0.0), 1e-15));
        assert!(complex_approx_eq(matrix[2], Complex::new(1.0, 0.0), 1e-15));
        // (1,1) should be W_4^1 = e^(-2πi/4) = -i (for forward)
        let expected = Complex::cis(-core::f64::consts::TAU / 4.0);
        assert!(
            complex_approx_eq(matrix[3], expected, 1e-14),
            "Twiddle at (1,1): got {:?}, expected {:?}",
            matrix[3],
            expected
        );
    }
}
