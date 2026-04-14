//! Generic mixed-radix DFT solver.
//!
//! This solver implements the Cooley-Tukey algorithm for composite sizes.
//! For N = N1 × N2, the DFT can be computed as:
//! 1. N1 DFTs of size N2
//! 2. Multiply by twiddle factors
//! 3. N2 DFTs of size N1
//!
//! Time complexity: O(N log N) for highly composite N
//! Space complexity: O(N)

use crate::dft::codelets::{notw_3, notw_5, notw_7};
use crate::dft::problem::Sign;
use crate::kernel::{factor, Complex, Float};
use crate::prelude::*;

use super::bluestein::BluesteinSolver;
use super::ct::CooleyTukeySolver;
use super::direct::DirectSolver;

/// Cached sub-solver for a specific size.
///
/// Avoids repeated planning overhead for recursive DFT calls.
enum SubSolver<T: Float> {
    /// Size 1 - no transform needed
    Identity,
    /// Small primes with dedicated codelets (3, 5, 7)
    Codelet(usize),
    /// Power-of-2 sizes - use Cooley-Tukey
    CooleyTukey,
    /// Composite sizes - use cached GenericSolver
    Generic(Box<GenericSolver<T>>),
    /// Large primes - use cached BluesteinSolver
    Bluestein(Box<BluesteinSolver<T>>),
    /// Small primes - use DirectSolver
    Direct,
}

impl<T: Float> SubSolver<T> {
    /// Create a sub-solver for the given size.
    fn new(n: usize) -> Self {
        match n {
            0 | 1 => Self::Identity,
            3 | 5 | 7 => Self::Codelet(n),
            n if n.is_power_of_two() => Self::CooleyTukey,
            n if GenericSolver::<T>::applicable(n) => {
                Self::Generic(Box::new(GenericSolver::new(n)))
            }
            n if n <= 13 => Self::Direct,
            n => Self::Bluestein(Box::new(BluesteinSolver::new(n))),
        }
    }

    /// Execute DFT using the cached solver.
    fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        match self {
            Self::Identity => {
                if !input.is_empty() {
                    output[0] = input[0];
                }
            }
            Self::Codelet(3) => {
                output[..3].copy_from_slice(&input[..3]);
                let sign_int = if sign == Sign::Forward { -1 } else { 1 };
                notw_3(output, sign_int);
            }
            Self::Codelet(5) => {
                output[..5].copy_from_slice(&input[..5]);
                let sign_int = if sign == Sign::Forward { -1 } else { 1 };
                notw_5(output, sign_int);
            }
            Self::Codelet(7) => {
                output[..7].copy_from_slice(&input[..7]);
                let sign_int = if sign == Sign::Forward { -1 } else { 1 };
                notw_7(output, sign_int);
            }
            Self::Codelet(_) => unreachable!(),
            Self::CooleyTukey => {
                CooleyTukeySolver::default().execute(input, output, sign);
            }
            Self::Generic(solver) => {
                solver.execute(input, output, sign);
            }
            Self::Bluestein(solver) => {
                solver.execute(input, output, sign);
            }
            Self::Direct => {
                DirectSolver::new().execute(input, output, sign);
            }
        }
    }
}

/// Generic mixed-radix recursive solver.
///
/// This solver factorizes N and applies the Cooley-Tukey decomposition
/// recursively. For prime factors, it falls back to Bluestein or direct.
///
/// Pre-allocates work buffers to avoid per-execution allocations.
/// Uses `Mutex` for thread-safe interior mutability with `try_lock()` fallback.
pub struct GenericSolver<T: Float> {
    /// Transform size
    n: usize,
    /// Factorization: N = n1 × n2 where n1 is the smallest prime factor
    n1: usize,
    n2: usize,
    /// Precomputed twiddle factors for forward transform
    twiddles_fwd: Vec<Complex<T>>,
    /// Precomputed twiddle factors for backward transform
    twiddles_bwd: Vec<Complex<T>>,
    /// Cached sub-solver for size n1
    solver_n1: SubSolver<T>,
    /// Cached sub-solver for size n2
    solver_n2: SubSolver<T>,
    /// Pre-allocated temp buffer (size n)
    #[cfg(feature = "std")]
    work_temp: Mutex<Vec<Complex<T>>>,
    /// Pre-allocated row input buffer for size n1
    #[cfg(feature = "std")]
    work_row_in_1: Mutex<Vec<Complex<T>>>,
    /// Pre-allocated row output buffer for size n1
    #[cfg(feature = "std")]
    work_row_out_1: Mutex<Vec<Complex<T>>>,
    /// Pre-allocated row input buffer for size n2
    #[cfg(feature = "std")]
    work_row_in_2: Mutex<Vec<Complex<T>>>,
    /// Pre-allocated row output buffer for size n2
    #[cfg(feature = "std")]
    work_row_out_2: Mutex<Vec<Complex<T>>>,
}

impl<T: Float> GenericSolver<T> {
    /// Create a new mixed-radix solver for the given size.
    #[must_use]
    pub fn new(n: usize) -> Self {
        if n <= 1 {
            return Self {
                n,
                n1: 1,
                n2: n,
                twiddles_fwd: Vec::new(),
                twiddles_bwd: Vec::new(),
                solver_n1: SubSolver::Identity,
                solver_n2: SubSolver::Identity,
                #[cfg(feature = "std")]
                work_temp: Mutex::new(Vec::new()),
                #[cfg(feature = "std")]
                work_row_in_1: Mutex::new(Vec::new()),
                #[cfg(feature = "std")]
                work_row_out_1: Mutex::new(Vec::new()),
                #[cfg(feature = "std")]
                work_row_in_2: Mutex::new(Vec::new()),
                #[cfg(feature = "std")]
                work_row_out_2: Mutex::new(Vec::new()),
            };
        }

        // Get optimal factorization (prefers larger radices we have kernels for)
        let (n1, n2) = Self::optimal_factorization(n);

        // Precompute twiddle factors: W_N^(k1 * k2) for k1 = 0..n1-1, k2 = 0..n2-1
        // Store in row-major order (same as temp buffer) for SIMD-friendly access:
        // twiddles[k1 * n2 + k2] = W_N^(k1 * k2)
        let mut twiddles_fwd = vec![Complex::zero(); n];
        let mut twiddles_bwd = vec![Complex::zero(); n];

        // Precompute row twiddle steps: W_N^k2 for k2 = 0..n2-1
        let angle_step_fwd = -<T as Float>::TWO_PI / T::from_usize(n);
        let angle_step_bwd = <T as Float>::TWO_PI / T::from_usize(n);

        for k2 in 0..n2 {
            // For each column k2, compute twiddles using recurrence
            // W_N^(k1 * k2) = (W_N^k2)^k1
            let w_step_fwd = Complex::cis(angle_step_fwd * T::from_usize(k2));
            let w_step_bwd = Complex::cis(angle_step_bwd * T::from_usize(k2));

            let mut w_fwd = Complex::new(T::ONE, T::ZERO);
            let mut w_bwd = Complex::new(T::ONE, T::ZERO);

            for k1 in 0..n1 {
                // Store in row-major order: index = k1 * n2 + k2
                let idx = k1 * n2 + k2;
                twiddles_fwd[idx] = w_fwd;
                twiddles_bwd[idx] = w_bwd;
                w_fwd = w_fwd * w_step_fwd;
                w_bwd = w_bwd * w_step_bwd;
            }
        }

        // Create cached sub-solvers for n1 and n2
        let solver_n1 = SubSolver::new(n1);
        let solver_n2 = SubSolver::new(n2);

        Self {
            n,
            n1,
            n2,
            twiddles_fwd,
            twiddles_bwd,
            solver_n1,
            solver_n2,
            #[cfg(feature = "std")]
            work_temp: Mutex::new(vec![Complex::zero(); n]),
            #[cfg(feature = "std")]
            work_row_in_1: Mutex::new(vec![Complex::zero(); n1]),
            #[cfg(feature = "std")]
            work_row_out_1: Mutex::new(vec![Complex::zero(); n1]),
            #[cfg(feature = "std")]
            work_row_in_2: Mutex::new(vec![Complex::zero(); n2]),
            #[cfg(feature = "std")]
            work_row_out_2: Mutex::new(vec![Complex::zero(); n2]),
        }
    }

    /// Solver name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        "dft-generic"
    }

    /// Get the transform size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.n
    }

    /// Check if this solver is applicable.
    ///
    /// Generic solver can handle any composite size > 1.
    #[must_use]
    pub fn applicable(n: usize) -> bool {
        if n <= 1 {
            return false;
        }
        // Check if composite (has more than one prime factor or prime power > 1)
        let factors = factor(n);
        factors.len() > 1 || factors[0].1 > 1
    }

    /// Apply twiddle factors element-wise to the temp buffer.
    ///
    /// Uses SIMD for f64 when available, scalar fallback otherwise.
    #[inline]
    fn apply_twiddles(temp: &mut [Complex<T>], twiddles: &[Complex<T>]) {
        // Try SIMD path for f64
        if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f64>() {
            // Safety: We've verified T is f64
            let temp_f64: &mut [Complex<f64>] =
                unsafe { &mut *(std::ptr::from_mut::<[Complex<T>]>(temp) as *mut [Complex<f64>]) };
            let twiddles_f64: &[Complex<f64>] = unsafe {
                &*(std::ptr::from_ref::<[Complex<T>]>(twiddles) as *const [Complex<f64>])
            };

            Self::apply_twiddles_f64(temp_f64, twiddles_f64);
            return;
        }

        // Scalar fallback
        for (t, tw) in temp.iter_mut().zip(twiddles.iter()) {
            *t = *t * *tw;
        }
    }

    /// Apply twiddle factors for f64 with SIMD acceleration.
    #[cfg(target_arch = "x86_64")]
    fn apply_twiddles_f64(temp: &mut [Complex<f64>], twiddles: &[Complex<f64>]) {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // Safety: We've verified AVX2+FMA are available
            unsafe { Self::apply_twiddles_avx2(temp, twiddles) }
        } else {
            Self::apply_twiddles_scalar_f64(temp, twiddles);
        }
    }

    /// Apply twiddle factors for f64 (non-x86).
    #[cfg(not(target_arch = "x86_64"))]
    fn apply_twiddles_f64(temp: &mut [Complex<f64>], twiddles: &[Complex<f64>]) {
        Self::apply_twiddles_scalar_f64(temp, twiddles);
    }

    /// Scalar implementation for f64.
    #[inline]
    fn apply_twiddles_scalar_f64(temp: &mut [Complex<f64>], twiddles: &[Complex<f64>]) {
        for (t, tw) in temp.iter_mut().zip(twiddles.iter()) {
            *t = *t * *tw;
        }
    }

    /// AVX2+FMA implementation for twiddle multiplication.
    ///
    /// Processes 2 complex values (4 f64s) per iteration.
    ///
    /// # Safety
    /// Caller must ensure AVX2 and FMA are available.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn apply_twiddles_avx2(temp: &mut [Complex<f64>], twiddles: &[Complex<f64>]) {
        use core::arch::x86_64::*;

        let n = temp.len();
        let ptr = temp.as_mut_ptr() as *mut f64;
        let tw_ptr = twiddles.as_ptr() as *const f64;

        let mut i = 0;

        // Process 2 complex numbers at a time (4 f64s)
        while i + 2 <= n {
            // Safety: AVX2 is guaranteed available via target_feature
            unsafe {
                // Load temp[i], temp[i+1] as [t0_re, t0_im, t1_re, t1_im]
                let t = _mm256_loadu_pd(ptr.add(i * 2));
                // Load twiddles[i], twiddles[i+1]
                let tw = _mm256_loadu_pd(tw_ptr.add(i * 2));

                // Complex multiply: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
                // t = [t0_re, t0_im, t1_re, t1_im]
                // tw = [tw0_re, tw0_im, tw1_re, tw1_im]

                // Get real and imaginary parts
                let t_re = _mm256_permute_pd(t, 0b0000); // [t0_re, t0_re, t1_re, t1_re]
                let t_im = _mm256_permute_pd(t, 0b1111); // [t0_im, t0_im, t1_im, t1_im]

                // Prepare twiddle for FMA
                // For result_re: need tw_re at even positions
                // For result_im: need tw_im at even positions
                let tw_ri = tw; // [tw0_re, tw0_im, tw1_re, tw1_im]
                let tw_ir = _mm256_permute_pd(tw, 0b0101); // [tw0_im, tw0_re, tw1_im, tw1_re]

                // result_re = t_re * tw_re - t_im * tw_im
                // result_im = t_re * tw_im + t_im * tw_re
                let prod1 = _mm256_mul_pd(t_re, tw_ri); // [t_re*tw_re, t_re*tw_im, ...]
                let prod2 = _mm256_mul_pd(t_im, tw_ir); // [t_im*tw_im, t_im*tw_re, ...]

                // addsub: result = [prod1[0]-prod2[0], prod1[1]+prod2[1], ...]
                // = [t_re*tw_re - t_im*tw_im, t_re*tw_im + t_im*tw_re, ...]
                // = [result_re, result_im, ...]
                let result = _mm256_addsub_pd(prod1, prod2);

                _mm256_storeu_pd(ptr.add(i * 2), result);
            }
            i += 2;
        }

        // Handle remaining element
        while i < n {
            let t = temp[i];
            let tw = twiddles[i];
            temp[i] = Complex::new(t.re * tw.re - t.im * tw.im, t.re * tw.im + t.im * tw.re);
            i += 1;
        }
    }

    /// Find optimal factorization N = n1 × n2 for mixed-radix FFT.
    ///
    /// Strategy (in priority order):
    /// 1. Extract largest power-of-2 factor (up to 16) - we have efficient kernels
    /// 2. Extract small prime factors (3, 5, 7) - we'll add kernels for these
    /// 3. Balance factors close to sqrt(N) - minimizes recursion depth
    /// 4. Fallback to smallest prime factor
    ///
    /// This significantly reduces recursion depth compared to always picking
    /// the smallest prime factor. For example:
    /// - 360 with smallest prime: 2→2→2→3→3→5 (6 levels)
    /// - 360 with optimal: 8×45 or 5×72 (2 levels)
    fn optimal_factorization(n: usize) -> (usize, usize) {
        debug_assert!(n > 1);

        // Priority 1: Try to extract powers of 2 (largest first)
        // These map directly to efficient Cooley-Tukey radix-2 kernels
        for radix in [16, 8, 4, 2] {
            if n.is_multiple_of(radix) {
                let n2 = n / radix;
                if n2 > 0 {
                    return (radix, n2);
                }
            }
        }

        // Priority 2: Try small odd primes we'll have kernels for
        for radix in [3, 5, 7] {
            if n.is_multiple_of(radix) {
                let n2 = n / radix;
                if n2 > 0 {
                    return (radix, n2);
                }
            }
        }

        // Priority 3: For remaining composites, find balanced factorization
        // Try to split close to sqrt(n) to minimize recursion depth
        let sqrt_n = (n as f64).sqrt() as usize;

        // Search from sqrt(n) downward for a divisor
        for d in (2..=sqrt_n).rev() {
            if n.is_multiple_of(d) {
                return (d, n / d);
            }
        }

        // Priority 4: If nothing found (shouldn't happen for composites),
        // fall back to smallest prime factor
        let factors = factor(n);
        let (p, _) = factors[0];
        (p, n / p)
    }

    /// Execute the mixed-radix FFT with provided work buffers.
    #[allow(clippy::too_many_arguments)]
    fn execute_with_buffers(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        sign: Sign,
        temp: &mut [Complex<T>],
        row_in_1: &mut [Complex<T>],
        row_out_1: &mut [Complex<T>],
        row_in_2: &mut [Complex<T>],
        row_out_2: &mut [Complex<T>],
    ) {
        let n1 = self.n1;
        let n2 = self.n2;

        // Step 1: Perform n2 DFTs of size n1
        // Input is viewed as n2 rows of n1 elements (row-major)
        // We compute DFT along each row
        for j in 0..n2 {
            // Extract row j into pre-allocated buffer
            for i in 0..n1 {
                row_in_1[i] = input[i * n2 + j];
            }

            // DFT of size n1 using cached sub-solver
            self.solver_n1.execute(row_in_1, row_out_1, sign);

            // Store result
            for i in 0..n1 {
                temp[i * n2 + j] = row_out_1[i];
            }
        }

        // Step 2: Multiply by twiddle factors
        // Twiddles are now stored in row-major order matching temp layout
        let twiddles = match sign {
            Sign::Forward => &self.twiddles_fwd,
            Sign::Backward => &self.twiddles_bwd,
        };

        // Contiguous SIMD-friendly multiplication since both temp and twiddles
        // are in row-major order: temp[k1*n2+k2] *= twiddles[k1*n2+k2]
        Self::apply_twiddles(temp, twiddles);

        // Step 3: Perform n1 DFTs of size n2
        // Now viewed as n1 rows of n2 elements
        for i in 0..n1 {
            // Extract row i into pre-allocated buffer
            for j in 0..n2 {
                row_in_2[j] = temp[i * n2 + j];
            }

            // DFT of size n2 using cached sub-solver
            self.solver_n2.execute(row_in_2, row_out_2, sign);

            // Store result (transposed output)
            for j in 0..n2 {
                output[j * n1 + i] = row_out_2[j];
            }
        }
    }

    /// Execute the mixed-radix FFT.
    ///
    /// Uses pre-allocated work buffers when available (single-threaded case).
    /// Falls back to fresh allocation when buffers are locked (parallel execution).
    #[cfg(feature = "std")]
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        let n = self.n;

        debug_assert_eq!(input.len(), n);
        debug_assert_eq!(output.len(), n);

        if n == 0 {
            return;
        }

        if n == 1 {
            output[0] = input[0];
            return;
        }

        let n1 = self.n1;
        let n2 = self.n2;

        // Try to acquire all locks. If any fails, allocate fresh buffers.
        let temp_guard = self.work_temp.try_lock();
        let row_in_1_guard = self.work_row_in_1.try_lock();
        let row_out_1_guard = self.work_row_out_1.try_lock();
        let row_in_2_guard = self.work_row_in_2.try_lock();
        let row_out_2_guard = self.work_row_out_2.try_lock();

        if let (
            Ok(mut temp),
            Ok(mut row_in_1),
            Ok(mut row_out_1),
            Ok(mut row_in_2),
            Ok(mut row_out_2),
        ) = (
            temp_guard,
            row_in_1_guard,
            row_out_1_guard,
            row_in_2_guard,
            row_out_2_guard,
        ) {
            // Use pre-allocated buffers
            self.execute_with_buffers(
                input,
                output,
                sign,
                &mut temp,
                &mut row_in_1,
                &mut row_out_1,
                &mut row_in_2,
                &mut row_out_2,
            );
        } else {
            // Fallback: allocate fresh buffers (parallel execution case)
            let mut temp = vec![Complex::zero(); n];
            let mut row_in_1 = vec![Complex::zero(); n1];
            let mut row_out_1 = vec![Complex::zero(); n1];
            let mut row_in_2 = vec![Complex::zero(); n2];
            let mut row_out_2 = vec![Complex::zero(); n2];
            self.execute_with_buffers(
                input,
                output,
                sign,
                &mut temp,
                &mut row_in_1,
                &mut row_out_1,
                &mut row_in_2,
                &mut row_out_2,
            );
        }
    }

    /// Execute the mixed-radix FFT (no_std version - always allocates).
    #[cfg(not(feature = "std"))]
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        let n = self.n;

        debug_assert_eq!(input.len(), n);
        debug_assert_eq!(output.len(), n);

        if n == 0 {
            return;
        }

        if n == 1 {
            output[0] = input[0];
            return;
        }

        let n1 = self.n1;
        let n2 = self.n2;

        // no_std: always allocate fresh buffers
        let mut temp = vec![Complex::zero(); n];
        let mut row_in_1 = vec![Complex::zero(); n1];
        let mut row_out_1 = vec![Complex::zero(); n1];
        let mut row_in_2 = vec![Complex::zero(); n2];
        let mut row_out_2 = vec![Complex::zero(); n2];
        self.execute_with_buffers(
            input,
            output,
            sign,
            &mut temp,
            &mut row_in_1,
            &mut row_out_1,
            &mut row_in_2,
            &mut row_out_2,
        );
    }

    /// Execute in-place mixed-radix FFT.
    pub fn execute_inplace(&self, data: &mut [Complex<T>], sign: Sign) {
        let n = self.n;
        debug_assert_eq!(data.len(), n);

        if n <= 1 {
            return;
        }

        // Need temporary storage
        let input: Vec<Complex<T>> = data.to_vec();
        self.execute(&input, data, sign);
    }
}

impl<T: Float> Default for GenericSolver<T> {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
        approx_eq(a.re, b.re, eps) && approx_eq(a.im, b.im, eps)
    }

    #[test]
    fn test_generic_applicable() {
        // Not applicable for primes
        assert!(!GenericSolver::<f64>::applicable(1));
        assert!(!GenericSolver::<f64>::applicable(2));
        assert!(!GenericSolver::<f64>::applicable(3));
        assert!(!GenericSolver::<f64>::applicable(5));
        assert!(!GenericSolver::<f64>::applicable(7));

        // Applicable for composites
        assert!(GenericSolver::<f64>::applicable(4)); // 2^2
        assert!(GenericSolver::<f64>::applicable(6)); // 2×3
        assert!(GenericSolver::<f64>::applicable(8)); // 2^3
        assert!(GenericSolver::<f64>::applicable(9)); // 3^2
        assert!(GenericSolver::<f64>::applicable(10)); // 2×5
        assert!(GenericSolver::<f64>::applicable(12)); // 2^2×3
        assert!(GenericSolver::<f64>::applicable(15)); // 3×5
    }

    #[test]
    fn test_generic_size_6() {
        // 6 = 2 × 3
        let input: Vec<Complex<f64>> = (0..6).map(|i| Complex::new(f64::from(i), 0.0)).collect();
        let mut output_generic = vec![Complex::zero(); 6];
        let mut output_direct = vec![Complex::zero(); 6];

        GenericSolver::new(6).execute(&input, &mut output_generic, Sign::Forward);
        DirectSolver::new().execute(&input, &mut output_direct, Sign::Forward);

        for (a, b) in output_generic.iter().zip(output_direct.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_generic_size_9() {
        // 9 = 3^2
        let input: Vec<Complex<f64>> = (0..9)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut output_generic = vec![Complex::zero(); 9];
        let mut output_direct = vec![Complex::zero(); 9];

        GenericSolver::new(9).execute(&input, &mut output_generic, Sign::Forward);
        DirectSolver::new().execute(&input, &mut output_direct, Sign::Forward);

        for (a, b) in output_generic.iter().zip(output_direct.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_generic_size_12() {
        // 12 = 2^2 × 3
        let input: Vec<Complex<f64>> = (0..12)
            .map(|i| Complex::new(f64::from(i), f64::from(i) * 0.5))
            .collect();
        let mut output_generic = vec![Complex::zero(); 12];
        let mut output_direct = vec![Complex::zero(); 12];

        GenericSolver::new(12).execute(&input, &mut output_generic, Sign::Forward);
        DirectSolver::new().execute(&input, &mut output_direct, Sign::Forward);

        for (a, b) in output_generic.iter().zip(output_direct.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_generic_size_15() {
        // 15 = 3 × 5
        let input: Vec<Complex<f64>> = (0..15)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut output_generic = vec![Complex::zero(); 15];
        let mut output_direct = vec![Complex::zero(); 15];

        GenericSolver::new(15).execute(&input, &mut output_generic, Sign::Forward);
        DirectSolver::new().execute(&input, &mut output_direct, Sign::Forward);

        for (a, b) in output_generic.iter().zip(output_direct.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_generic_size_18() {
        // 18 = 2 × 3^2
        let input: Vec<Complex<f64>> = (0..18).map(|i| Complex::new(f64::from(i), 0.0)).collect();
        let mut output_generic = vec![Complex::zero(); 18];
        let mut output_direct = vec![Complex::zero(); 18];

        GenericSolver::new(18).execute(&input, &mut output_generic, Sign::Forward);
        DirectSolver::new().execute(&input, &mut output_direct, Sign::Forward);

        for (a, b) in output_generic.iter().zip(output_direct.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-8));
        }
    }

    #[test]
    fn test_generic_inverse_recovers_input() {
        let original: Vec<Complex<f64>> = (0..12)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut transformed = vec![Complex::zero(); 12];
        let mut recovered = vec![Complex::zero(); 12];

        let solver = GenericSolver::new(12);
        solver.execute(&original, &mut transformed, Sign::Forward);
        solver.execute(&transformed, &mut recovered, Sign::Backward);

        // Normalize
        let n = original.len() as f64;
        for x in &mut recovered {
            *x = *x / n;
        }

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_generic_inplace() {
        let original: Vec<Complex<f64>> =
            (0..10).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        // Out-of-place reference
        let mut out_of_place = vec![Complex::zero(); 10];
        let solver = GenericSolver::new(10);
        solver.execute(&original, &mut out_of_place, Sign::Forward);

        // In-place
        let mut in_place = original;
        solver.execute_inplace(&mut in_place, Sign::Forward);

        for (a, b) in out_of_place.iter().zip(in_place.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }
}
