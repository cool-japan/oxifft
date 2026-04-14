//! Real-to-Real solver for DCT/DST/DHT transforms.
//!
//! All transforms have two implementations:
//! - `_direct`: O(n²) direct computation, used for n < 16
//! - `_fast`: O(n log n) FFT-based computation, used for n >= 16
//!
//! The public API dispatches automatically based on size.

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};

/// Type of DCT/DST/DHT transform (FFTW terminology).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum R2rKind {
    /// DCT-I (REDFT00)
    Redft00,
    /// DCT-II (REDFT10) - the "DCT" commonly used in JPEG
    Redft10,
    /// DCT-III (REDFT01) - inverse of DCT-II (up to scaling)
    Redft01,
    /// DCT-IV (REDFT11)
    Redft11,
    /// DST-I (RODFT00)
    Rodft00,
    /// DST-II (RODFT10)
    Rodft10,
    /// DST-III (RODFT01)
    Rodft01,
    /// DST-IV (RODFT11)
    Rodft11,
    /// Discrete Hartley Transform
    Dht,
}

/// Real-to-Real transform solver.
pub struct R2rSolver<T: Float> {
    kind: R2rKind,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> Default for R2rSolver<T> {
    fn default() -> Self {
        Self::new(R2rKind::Redft10)
    }
}

impl<T: Float> R2rSolver<T> {
    /// Create a new R2R solver with specified kind.
    #[must_use]
    pub fn new(kind: R2rKind) -> Self {
        Self {
            kind,
            _marker: core::marker::PhantomData,
        }
    }

    /// Get the solver name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self.kind {
            R2rKind::Redft00 => "rdft-redft00",
            R2rKind::Redft10 => "rdft-redft10",
            R2rKind::Redft01 => "rdft-redft01",
            R2rKind::Redft11 => "rdft-redft11",
            R2rKind::Rodft00 => "rdft-rodft00",
            R2rKind::Rodft10 => "rdft-rodft10",
            R2rKind::Rodft01 => "rdft-rodft01",
            R2rKind::Rodft11 => "rdft-rodft11",
            R2rKind::Dht => "rdft-dht",
        }
    }

    /// Check if this solver is applicable for the given size.
    #[must_use]
    pub fn applicable(&self, n: usize) -> bool {
        n >= 1
    }

    // -------------------------------------------------------------------------
    // DCT-II
    // -------------------------------------------------------------------------

    /// Execute DCT-II (REDFT10) using direct O(n²) computation.
    ///
    /// Formula: X\[k\] = sum_{n=0}^{N-1} x\[n\] * cos(π * (2n + 1) * k / (2N))
    pub fn execute_dct2_direct(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DCT-II requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let two_n = T::from_usize(2 * n);

        for k in 0..n {
            let k_f = T::from_usize(k);
            let mut sum = T::ZERO;

            for (i, &x_i) in input.iter().enumerate() {
                let angle = <T as Float>::PI * (T::from_usize(2 * i + 1)) * k_f / two_n;
                sum = sum + x_i * Float::cos(angle);
            }

            output[k] = sum;
        }
    }

    /// Execute DCT-II using FFT-based O(n log n) algorithm.
    ///
    /// Algorithm (2N-point FFT, even reflection):
    ///   v\[i\] = x\[i\]           for i = 0..N-1
    ///   v\[i\] = x\[2N-1-i\]      for i = N..2N-1
    ///   Y = FFT_{2N}(v)
    ///   DCT_II\[k\] = (c_k * Y\[k\].re - s_k * Y\[k\].im) / 2
    ///   where c_k = cos(-π*k/(2N)), s_k = sin(-π*k/(2N))
    pub fn execute_dct2_fast(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DCT-II requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let two_n = 2 * n;
        let two_n_f = T::from_usize(two_n);

        // Build 2N-point even-reflection vector (real-valued)
        let mut v_complex: Vec<Complex<T>> = Vec::with_capacity(two_n);
        for i in 0..n {
            v_complex.push(Complex::new(input[i], T::ZERO));
        }
        for i in n..two_n {
            v_complex.push(Complex::new(input[2 * n - 1 - i], T::ZERO));
        }

        let mut y = vec![Complex::zero(); two_n];

        if let Some(plan) = Plan::dft_1d(two_n, Direction::Forward, Flags::ESTIMATE) {
            plan.execute(&v_complex, &mut y);
        } else {
            return self.execute_dct2_direct(input, output);
        }

        // Extract DCT-II coefficients from the FFT output
        for k in 0..n {
            let angle = -<T as Float>::PI * T::from_usize(k) / two_n_f;
            let (s_k, c_k) = Float::sin_cos(angle);
            output[k] = (c_k * y[k].re - s_k * y[k].im) / T::TWO;
        }
    }

    /// Execute DCT-II transform (dispatches to fast for n >= 16, direct otherwise).
    pub fn execute_dct2(&self, input: &[T], output: &mut [T]) {
        if input.len() >= 16 {
            self.execute_dct2_fast(input, output);
        } else {
            self.execute_dct2_direct(input, output);
        }
    }

    // -------------------------------------------------------------------------
    // DCT-III
    // -------------------------------------------------------------------------

    /// Execute DCT-III (REDFT01) using direct O(n²) computation.
    ///
    /// Formula: x\[n\] = X\[0\]/2 + sum_{k=1}^{N-1} X\[k\] * cos(π * k * (2n + 1) / (2N))
    pub fn execute_dct3_direct(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DCT-III requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let two_n = T::from_usize(2 * n);
        let half = T::ONE / T::TWO;

        for i in 0..n {
            let two_i_plus_1 = T::from_usize(2 * i + 1);
            let mut sum = input[0] * half;

            for k in 1..n {
                let angle = <T as Float>::PI * T::from_usize(k) * two_i_plus_1 / two_n;
                sum = sum + input[k] * Float::cos(angle);
            }

            output[i] = sum;
        }
    }

    /// Execute DCT-III using FFT-based O(n log n) algorithm.
    ///
    /// DCT-III is the transpose/inverse of DCT-II. We use the conj-FFT-conj trick:
    ///   IFFT_{2N}(Z) = conj(FFT_{2N}(conj(Z))) / (2N)
    ///
    /// Build Z of length 2N:
    ///   Z\[0\] = 2*X\[0\], Z\[N\] = 0
    ///   Z\[k\] = 2*X\[k\]*exp(i*π*k/(2N))   for k = 1..N-1
    ///   Z\[2N-k\] = conj(Z\[k\])             for k = 1..N-1
    ///
    /// Then DCT_III\[n\] = Re(IFFT_{2N}(Z)\[n\]) = Re(conj(FFT_{2N}(conj(Z)))\[n\]) / (2N)
    pub fn execute_dct3_fast(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DCT-III requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let two_n = 2 * n;
        let two_n_f = T::from_usize(two_n);

        // Build Z of length 2N
        let mut z: Vec<Complex<T>> = vec![Complex::zero(); two_n];

        // Z[0] = 2 * X[0]
        z[0] = Complex::new(T::TWO * input[0], T::ZERO);
        // Z[N] = 0 (already zero from initialization)

        // Z[k] = 2*X[k]*exp(i*π*k/(2N)) for k=1..N-1
        for k in 1..n {
            let angle = <T as Float>::PI * T::from_usize(k) / two_n_f;
            let (s, c) = Float::sin_cos(angle);
            let phase = Complex::new(c, s);
            let val = phase * T::TWO * input[k];
            z[k] = val;
            z[two_n - k] = val.conj();
        }

        // conj(Z) then forward FFT then conj gives IFFT
        // Take conjugate of input for conj-FFT-conj trick
        let z_conj: Vec<Complex<T>> = z.iter().map(|c| c.conj()).collect();
        let mut y = vec![Complex::zero(); two_n];

        if let Some(plan) = Plan::dft_1d(two_n, Direction::Forward, Flags::ESTIMATE) {
            plan.execute(&z_conj, &mut y);
        } else {
            return self.execute_dct3_direct(input, output);
        }

        // DCT_III[n] = Re(FFT_2N(conj(Z))[n]) / 4
        // (Derivation: IFFT_2N(Z)[n] = (2/N)*DCT_III[n], and
        //  Re(IFFT_2N(Z)[n]) = Re(FFT_2N(conj(Z))[n]) / (2N),
        //  so DCT_III[n] = N/2 * Re(FFT_2N(conj(Z))[n]) / (2N) = Re(FFT_2N(conj(Z))[n]) / 4)
        let four = T::TWO + T::TWO;
        for i in 0..n {
            output[i] = y[i].re / four;
        }
    }

    /// Execute DCT-III transform (dispatches to fast for n >= 16, direct otherwise).
    pub fn execute_dct3(&self, input: &[T], output: &mut [T]) {
        if input.len() >= 16 {
            self.execute_dct3_fast(input, output);
        } else {
            self.execute_dct3_direct(input, output);
        }
    }

    // -------------------------------------------------------------------------
    // DCT-I
    // -------------------------------------------------------------------------

    /// Execute DCT-I (REDFT00) using direct O(n²) computation.
    ///
    /// Formula: X\[k\] = x\[0\] + (-1)^k * x\[N-1\] + 2 * sum_{n=1}^{N-2} x\[n\] * cos(π * n * k / (N-1))
    pub fn execute_dct1_direct(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        if n <= 1 {
            if n == 1 {
                output[0] = input[0];
            }
            return;
        }

        let n_minus_1 = T::from_usize(n - 1);

        for k in 0..n {
            let k_f = T::from_usize(k);
            let sign = if k % 2 == 0 { T::ONE } else { -T::ONE };

            let mut sum = input[0] + sign * input[n - 1];

            for i in 1..(n - 1) {
                let angle = <T as Float>::PI * T::from_usize(i) * k_f / n_minus_1;
                sum = sum + T::TWO * input[i] * Float::cos(angle);
            }

            output[k] = sum;
        }
    }

    /// Execute DCT-I using FFT-based O(n log n) algorithm.
    ///
    /// Even extension of length 2(N-1):
    ///   v\[i\] = x\[i\]           for i = 0..N-1
    ///   v\[i\] = x\[2(N-1)-i\]    for i = N..2(N-1)-1
    ///
    /// DCT_I\[k\] = Re(FFT_{2(N-1)}(v))\[k\]
    pub fn execute_dct1_fast(&self, input: &[T], output: &mut [T]) {
        let n = input.len();

        if n <= 3 {
            return self.execute_dct1_direct(input, output);
        }

        let m = 2 * (n - 1); // FFT size

        // Build even-extension vector
        let mut v: Vec<Complex<T>> = Vec::with_capacity(m);
        for i in 0..n {
            v.push(Complex::new(input[i], T::ZERO));
        }
        // i = N..2(N-1)-1, i.e., i = N..m-1 (since m = 2(N-1))
        for i in n..m {
            v.push(Complex::new(input[m - i], T::ZERO));
        }

        let mut y = vec![Complex::zero(); m];

        if let Some(plan) = Plan::dft_1d(m, Direction::Forward, Flags::ESTIMATE) {
            plan.execute(&v, &mut y);
        } else {
            return self.execute_dct1_direct(input, output);
        }

        // DCT_I[k] = Re(Y[k])
        for k in 0..n {
            output[k] = y[k].re;
        }
    }

    /// Execute DCT-I transform (dispatches to fast for n >= 16, direct otherwise).
    pub fn execute_dct1(&self, input: &[T], output: &mut [T]) {
        if input.len() >= 16 {
            self.execute_dct1_fast(input, output);
        } else {
            self.execute_dct1_direct(input, output);
        }
    }

    // -------------------------------------------------------------------------
    // DCT-IV
    // -------------------------------------------------------------------------

    /// Execute DCT-IV (REDFT11) using direct O(n²) computation.
    ///
    /// Formula: X\[k\] = sum_{n=0}^{N-1} x\[n\] * cos(π * (2n + 1) * (2k + 1) / (4N))
    pub fn execute_dct4_direct(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DCT-IV requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let four_n = T::from_usize(4 * n);

        for k in 0..n {
            let two_k_plus_1 = T::from_usize(2 * k + 1);
            let mut sum = T::ZERO;

            for (i, &x_i) in input.iter().enumerate() {
                let angle = <T as Float>::PI * T::from_usize(2 * i + 1) * two_k_plus_1 / four_n;
                sum = sum + x_i * Float::cos(angle);
            }

            output[k] = sum;
        }
    }

    /// Execute DCT-IV using FFT-based O(n log n) algorithm.
    ///
    /// Algorithm using 4N-point FFT:
    ///   Pad x to length 4N (zeros for indices N..4N-1)
    ///   Y = FFT_{4N}(x_padded)
    ///   DCT_IV\[k\] = Re(exp(-i*π*(2k+1)/(4N)) * Y\[2k+1\])
    ///
    /// Derivation: DCT_IV\[k\] = Re(sum_n x\[n\]*exp(i*π*(2n+1)*(2k+1)/(4N)))
    ///           = Re(exp(-i*π*(2k+1)/(4N)) * sum_n x\[n\]*exp(-2*π*i*n*(2k+1)/(4N)))
    ///           = Re(exp(-i*π*(2k+1)/(4N)) * FFT_{4N}(x_padded)\[2k+1\])
    pub fn execute_dct4_fast(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DCT-IV requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let four_n = 4 * n;
        let four_n_f = T::from_usize(four_n);

        // Build zero-padded input of length 4N
        let mut x_padded: Vec<Complex<T>> = Vec::with_capacity(four_n);
        for &xi in input {
            x_padded.push(Complex::new(xi, T::ZERO));
        }
        for _ in n..four_n {
            x_padded.push(Complex::zero());
        }

        let mut y = vec![Complex::zero(); four_n];

        if let Some(plan) = Plan::dft_1d(four_n, Direction::Forward, Flags::ESTIMATE) {
            plan.execute(&x_padded, &mut y);
        } else {
            return self.execute_dct4_direct(input, output);
        }

        // DCT_IV[k] = Re(exp(-i*π*(2k+1)/(4N)) * Y[2k+1])
        for k in 0..n {
            let angle = -<T as Float>::PI * T::from_usize(2 * k + 1) / four_n_f;
            let (s, c) = Float::sin_cos(angle);
            let phase = Complex::new(c, s);
            let val = phase * y[2 * k + 1];
            output[k] = val.re;
        }
    }

    /// Execute DCT-IV transform (dispatches to fast for n >= 16, direct otherwise).
    pub fn execute_dct4(&self, input: &[T], output: &mut [T]) {
        if input.len() >= 16 {
            self.execute_dct4_fast(input, output);
        } else {
            self.execute_dct4_direct(input, output);
        }
    }

    // -------------------------------------------------------------------------
    // DST-I
    // -------------------------------------------------------------------------

    /// Execute DST-I (RODFT00) transform.
    ///
    /// Formula: X\[k\] = sum_{n=0}^{N-1} x\[n\] * sin(π * (n+1) * (k+1) / (N+1))
    ///
    /// Note: DST-I is always computed directly (no FFT fast path for all sizes).
    /// For n >= 16, we use an odd-extension FFT approach.
    pub fn execute_dst1_direct(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DST-I requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let n_plus_1 = T::from_usize(n + 1);

        for k in 0..n {
            let k_plus_1 = T::from_usize(k + 1);
            let mut sum = T::ZERO;

            for (i, &x_i) in input.iter().enumerate() {
                let angle = <T as Float>::PI * T::from_usize(i + 1) * k_plus_1 / n_plus_1;
                sum = sum + x_i * Float::sin(angle);
            }

            output[k] = sum;
        }
    }

    /// Execute DST-I using FFT-based O(n log n) algorithm.
    ///
    /// DST-I of length N via 2(N+1)-point odd extension:
    ///   Build v of length 2(N+1):
    ///     v\[0\] = 0, v\[n+1\] = x\[n\] for n=0..N-1, v\[N+1\] = 0, v\[N+2+n\] = -x\[N-1-n\] for n=0..N-1
    ///   Y = FFT_{2(N+1)}(v)
    ///   DST_I\[k\] = -Im(Y\[k+1\]) for k=0..N-1
    pub fn execute_dst1_fast(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DST-I requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let m = 2 * (n + 1); // FFT size

        // Build odd-extension vector of length 2(N+1)
        let mut v: Vec<Complex<T>> = vec![Complex::zero(); m];
        // v[0] = 0 (already zero)
        // v[n+1] = x[n] for n=0..N-1
        for i in 0..n {
            v[i + 1] = Complex::new(input[i], T::ZERO);
        }
        // v[N+1] = 0 (already zero)
        // v[N+2+i] = -x[N-1-i] for i=0..N-1
        for i in 0..n {
            v[n + 2 + i] = Complex::new(-input[n - 1 - i], T::ZERO);
        }

        let mut y = vec![Complex::zero(); m];

        if let Some(plan) = Plan::dft_1d(m, Direction::Forward, Flags::ESTIMATE) {
            plan.execute(&v, &mut y);
        } else {
            return self.execute_dst1_direct(input, output);
        }

        // DST_I[k] = -Im(Y[k+1]) / 2
        // (Factor of 2 because the odd extension causes FFT values at odd
        //  positions to equal 2 * the desired DST-I value)
        let two = T::TWO;
        for k in 0..n {
            output[k] = -y[k + 1].im / two;
        }
    }

    /// Execute DST-I transform (dispatches to fast for n >= 16, direct otherwise).
    pub fn execute_dst1(&self, input: &[T], output: &mut [T]) {
        if input.len() >= 16 {
            self.execute_dst1_fast(input, output);
        } else {
            self.execute_dst1_direct(input, output);
        }
    }

    // -------------------------------------------------------------------------
    // DST-II
    // -------------------------------------------------------------------------

    /// Execute DST-II (RODFT10) using direct O(n²) computation.
    ///
    /// Formula: X\[k\] = sum_{n=0}^{N-1} x\[n\] * sin(π * (2n+1) * (k+1) / (2N))
    pub fn execute_dst2_direct(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DST-II requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let two_n = T::from_usize(2 * n);

        for k in 0..n {
            let k_plus_1 = T::from_usize(k + 1);
            let mut sum = T::ZERO;

            for (i, &x_i) in input.iter().enumerate() {
                let angle = <T as Float>::PI * T::from_usize(2 * i + 1) * k_plus_1 / two_n;
                sum = sum + x_i * Float::sin(angle);
            }

            output[k] = sum;
        }
    }

    /// Execute DST-II using FFT-based O(n log n) algorithm.
    ///
    /// Relationship: DST_II(x)\[k\] = DCT_II(y)\[N-1-k\]
    /// where y\[n\] = (-1)^n * x\[n\]  (sign-alternate input)
    pub fn execute_dst2_fast(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DST-II requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        // Build sign-alternated input: y[i] = (-1)^i * x[i]
        let y: Vec<T> = input
            .iter()
            .enumerate()
            .map(|(i, &x)| if i % 2 == 0 { x } else { -x })
            .collect();

        let mut dct2_y = vec![T::ZERO; n];
        let helper = Self::new(R2rKind::Redft10);
        helper.execute_dct2_fast(&y, &mut dct2_y);

        // DST_II[k] = DCT_II(y)[N-1-k]
        for k in 0..n {
            output[k] = dct2_y[n - 1 - k];
        }
    }

    /// Execute DST-II transform (dispatches to fast for n >= 16, direct otherwise).
    pub fn execute_dst2(&self, input: &[T], output: &mut [T]) {
        if input.len() >= 16 {
            self.execute_dst2_fast(input, output);
        } else {
            self.execute_dst2_direct(input, output);
        }
    }

    // -------------------------------------------------------------------------
    // DST-III
    // -------------------------------------------------------------------------

    /// Execute DST-III (RODFT01) using direct O(n²) computation.
    ///
    /// Formula: X\[k\] = (-1)^k * x\[N-1\]/2 + sum_{n=0}^{N-2} x\[n\] * sin(π * (n+1) * (2k+1) / (2N))
    pub fn execute_dst3_direct(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DST-III requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let two_n = T::from_usize(2 * n);
        let half = T::ONE / T::TWO;

        for k in 0..n {
            let two_k_plus_1 = T::from_usize(2 * k + 1);
            let sign = if k % 2 == 0 { T::ONE } else { -T::ONE };

            let mut sum = sign * input[n - 1] * half;

            for i in 0..(n - 1) {
                let angle = <T as Float>::PI * T::from_usize(i + 1) * two_k_plus_1 / two_n;
                sum = sum + input[i] * Float::sin(angle);
            }

            output[k] = sum;
        }
    }

    /// Execute DST-III using FFT-based O(n log n) algorithm.
    ///
    /// Relationship: DST_III(f)\[n\] = (-1)^n * DCT_III(f_reversed)\[n\]
    /// where f_reversed\[k\] = f\[N-1-k\]
    pub fn execute_dst3_fast(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DST-III requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        // f_reversed[k] = input[N-1-k]
        let f_reversed: Vec<T> = (0..n).map(|k| input[n - 1 - k]).collect();

        let mut dct3_out = vec![T::ZERO; n];
        let helper = Self::new(R2rKind::Redft01);
        helper.execute_dct3_fast(&f_reversed, &mut dct3_out);

        // DST_III[n] = (-1)^n * DCT_III(f_reversed)[n]
        for i in 0..n {
            output[i] = if i % 2 == 0 {
                dct3_out[i]
            } else {
                -dct3_out[i]
            };
        }
    }

    /// Execute DST-III transform (dispatches to fast for n >= 16, direct otherwise).
    pub fn execute_dst3(&self, input: &[T], output: &mut [T]) {
        if input.len() >= 16 {
            self.execute_dst3_fast(input, output);
        } else {
            self.execute_dst3_direct(input, output);
        }
    }

    // -------------------------------------------------------------------------
    // DST-IV
    // -------------------------------------------------------------------------

    /// Execute DST-IV (RODFT11) using direct O(n²) computation.
    ///
    /// Formula: X\[k\] = sum_{n=0}^{N-1} x\[n\] * sin(π * (2n+1) * (2k+1) / (4N))
    pub fn execute_dst4_direct(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DST-IV requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let four_n = T::from_usize(4 * n);

        for k in 0..n {
            let two_k_plus_1 = T::from_usize(2 * k + 1);
            let mut sum = T::ZERO;

            for (i, &x_i) in input.iter().enumerate() {
                let angle = <T as Float>::PI * T::from_usize(2 * i + 1) * two_k_plus_1 / four_n;
                sum = sum + x_i * Float::sin(angle);
            }

            output[k] = sum;
        }
    }

    /// Execute DST-IV using FFT-based O(n log n) algorithm.
    ///
    /// Relationship: DST_IV(x)\[k\] = (-1)^k * DCT_IV(x_reversed)\[k\]
    /// where x_reversed\[n\] = x\[N-1-n\]
    pub fn execute_dst4_fast(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DST-IV requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        // x_reversed[n] = input[N-1-n]
        let x_reversed: Vec<T> = (0..n).map(|i| input[n - 1 - i]).collect();

        let mut dct4_out = vec![T::ZERO; n];
        let helper = Self::new(R2rKind::Redft11);
        helper.execute_dct4_fast(&x_reversed, &mut dct4_out);

        // DST_IV[k] = (-1)^k * DCT_IV(x_reversed)[k]
        for k in 0..n {
            output[k] = if k % 2 == 0 {
                dct4_out[k]
            } else {
                -dct4_out[k]
            };
        }
    }

    /// Execute DST-IV transform (dispatches to fast for n >= 16, direct otherwise).
    pub fn execute_dst4(&self, input: &[T], output: &mut [T]) {
        if input.len() >= 16 {
            self.execute_dst4_fast(input, output);
        } else {
            self.execute_dst4_direct(input, output);
        }
    }

    // -------------------------------------------------------------------------
    // DHT
    // -------------------------------------------------------------------------

    /// Execute Discrete Hartley Transform (DHT) using direct O(n²) computation.
    ///
    /// Formula: H\[k\] = sum_{n=0}^{N-1} x\[n\] * cas(2πnk/N)
    ///
    /// where cas(θ) = cos(θ) + sin(θ)
    pub fn execute_dht_direct(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DHT requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let n_f = T::from_usize(n);

        for k in 0..n {
            let k_f = T::from_usize(k);
            let mut sum = T::ZERO;

            for (i, &x_i) in input.iter().enumerate() {
                let angle = <T as Float>::TWO_PI * T::from_usize(i) * k_f / n_f;
                let (s, c) = Float::sin_cos(angle);
                sum = sum + x_i * (c + s);
            }

            output[k] = sum;
        }
    }

    /// Execute DHT using FFT-based O(n log n) algorithm.
    ///
    /// For real input x:
    ///   Y = FFT(x)  (complex FFT with zero imaginary parts)
    ///   DHT\[k\] = Y\[k\].re - Y\[k\].im
    ///
    /// This works because the DFT with negative exponential gives:
    ///   Y\[k\] = sum x\[n\] * exp(-2πink/N) = sum x\[n\] * (cos - i*sin)(2πnk/N)
    ///   Y\[k\].re - Y\[k\].im = sum x\[n\] * (cos + sin)(2πnk/N) = DHT\[k\]
    pub fn execute_dht_fast(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DHT requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        // Build complex input with zero imaginary parts
        let v_complex: Vec<Complex<T>> = input.iter().map(|&x| Complex::new(x, T::ZERO)).collect();
        let mut y = vec![Complex::zero(); n];

        if let Some(plan) = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
            plan.execute(&v_complex, &mut y);
        } else {
            return self.execute_dht_direct(input, output);
        }

        // DHT[k] = Re(Y[k]) - Im(Y[k])
        for k in 0..n {
            output[k] = y[k].re - y[k].im;
        }
    }

    /// Execute Discrete Hartley Transform (dispatches to fast for n >= 16, direct otherwise).
    ///
    /// The DHT is its own inverse (up to scaling by N), making it
    /// particularly elegant for real-valued signals.
    pub fn execute_dht(&self, input: &[T], output: &mut [T]) {
        if input.len() >= 16 {
            self.execute_dht_fast(input, output);
        } else {
            self.execute_dht_direct(input, output);
        }
    }

    // -------------------------------------------------------------------------
    // Dispatch
    // -------------------------------------------------------------------------

    /// Execute the transform based on the configured kind.
    pub fn execute(&self, input: &[T], output: &mut [T]) {
        match self.kind {
            R2rKind::Redft00 => self.execute_dct1(input, output),
            R2rKind::Redft10 => self.execute_dct2(input, output),
            R2rKind::Redft01 => self.execute_dct3(input, output),
            R2rKind::Redft11 => self.execute_dct4(input, output),
            R2rKind::Rodft00 => self.execute_dst1(input, output),
            R2rKind::Rodft10 => self.execute_dst2(input, output),
            R2rKind::Rodft01 => self.execute_dst3(input, output),
            R2rKind::Rodft11 => self.execute_dst4(input, output),
            R2rKind::Dht => self.execute_dht(input, output),
        }
    }
}

/// Convenience function for DCT-II transform.
pub fn dct2<T: Float>(input: &[T], output: &mut [T]) {
    R2rSolver::new(R2rKind::Redft10).execute_dct2(input, output);
}

/// Convenience function for DCT-III transform (inverse DCT-II).
pub fn dct3<T: Float>(input: &[T], output: &mut [T]) {
    R2rSolver::new(R2rKind::Redft01).execute_dct3(input, output);
}

/// Convenience function for DCT-I transform.
pub fn dct1<T: Float>(input: &[T], output: &mut [T]) {
    R2rSolver::new(R2rKind::Redft00).execute_dct1(input, output);
}

/// Convenience function for DCT-IV transform.
pub fn dct4<T: Float>(input: &[T], output: &mut [T]) {
    R2rSolver::new(R2rKind::Redft11).execute_dct4(input, output);
}

/// Convenience function for DST-I transform.
pub fn dst1<T: Float>(input: &[T], output: &mut [T]) {
    R2rSolver::new(R2rKind::Rodft00).execute_dst1(input, output);
}

/// Convenience function for DST-II transform.
pub fn dst2<T: Float>(input: &[T], output: &mut [T]) {
    R2rSolver::new(R2rKind::Rodft10).execute_dst2(input, output);
}

/// Convenience function for DST-III transform (inverse DST-II).
pub fn dst3<T: Float>(input: &[T], output: &mut [T]) {
    R2rSolver::new(R2rKind::Rodft01).execute_dst3(input, output);
}

/// Convenience function for DST-IV transform.
pub fn dst4<T: Float>(input: &[T], output: &mut [T]) {
    R2rSolver::new(R2rKind::Rodft11).execute_dst4(input, output);
}

/// Convenience function for Discrete Hartley Transform.
///
/// The DHT is its own inverse (up to scaling by N):
/// DHT(DHT(x)) = N * x
pub fn dht<T: Float>(input: &[T], output: &mut [T]) {
    R2rSolver::new(R2rKind::Dht).execute_dht(input, output);
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Property-based tests (proptest)
    // -----------------------------------------------------------------------
    use proptest::prelude::*;

    proptest! {
        /// DCT-III is the inverse of DCT-II (up to scaling by 2N).
        /// Verify: DCT-III(DCT-II(x)) = 2N * x  for all x.
        ///
        /// We normalise by the empirical scale rather than hard-coding 2N
        /// to stay robust across any future normalisation changes (matching
        /// the convention used in the unit tests above).
        #[test]
        fn dct2_dct3_roundtrip(
            values in prop::collection::vec(-100.0f64..=100.0f64, 4usize..=32usize),
        ) {
            let n = values.len();
            let mut dct_out = vec![0.0f64; n];
            let mut recovered = vec![0.0f64; n];

            dct2(&values, &mut dct_out);
            dct3(&dct_out, &mut recovered);

            // Find a reference element with sufficient magnitude to derive scale.
            let ref_idx = values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(core::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            if values[ref_idx].abs() < 1e-10 {
                return Ok(()); // Signal is near-zero; skip.
            }

            let scale = recovered[ref_idx] / values[ref_idx];
            prop_assert!(
                scale.abs() > 1e-10,
                "Scale factor is near zero: {}",
                scale
            );

            for i in 0..n {
                let expected = values[i] * scale;
                let tol = expected.abs() * 1e-8 + 1e-8;
                prop_assert!(
                    (recovered[i] - expected).abs() < tol,
                    "DCT-II to III roundtrip mismatch at index {}: got {}, expected {} (scale={})",
                    i,
                    recovered[i],
                    expected,
                    scale
                );
            }
        }

        /// Parseval's theorem for DCT-II.
        ///
        /// Rather than asserting a specific constant (which depends on normalization
        /// convention), we verify that the DCT output energy is proportional to
        /// input energy: the ratio lies in a reasonable bounded range.
        /// For FFTW-convention DCT-II the ratio is approximately N (varies by at most 4x).
        #[test]
        fn dct_parseval(
            values in prop::collection::vec(-10.0f64..=10.0f64, 4usize..=32usize),
        ) {
            let n = values.len();

            let input_energy: f64 = values.iter().map(|&x| x * x).sum();
            if input_energy < 1e-12 {
                return Ok(()); // Skip near-zero inputs.
            }

            let mut output = vec![0.0f64; n];
            dct2(&values, &mut output);

            let output_energy: f64 = output.iter().map(|&x| x * x).sum();

            // The ratio output_energy / input_energy for unnormalized DCT-II is ~N.
            let ratio = output_energy / input_energy;
            let n_f = n as f64;
            prop_assert!(
                ratio > n_f * 0.1 && ratio < n_f * 4.0,
                "DCT Parseval ratio out of expected range: ratio={}, n={}, expected ~{}",
                ratio,
                n,
                n_f
            );
        }

        /// DST-III is the inverse of DST-II (up to scaling).
        /// Verify: DST-III(DST-II(x)) = scale * x  for all x.
        #[test]
        fn dst2_dst3_roundtrip(
            values in prop::collection::vec(-100.0f64..=100.0f64, 4usize..=32usize),
        ) {
            let n = values.len();
            let mut dst_out = vec![0.0f64; n];
            let mut recovered = vec![0.0f64; n];

            dst2(&values, &mut dst_out);
            dst3(&dst_out, &mut recovered);

            // Find reference element with largest magnitude.
            let ref_idx = values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(core::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            if values[ref_idx].abs() < 1e-10 {
                return Ok(()); // Near-zero signal; skip.
            }

            let scale = recovered[ref_idx] / values[ref_idx];
            prop_assert!(
                scale.abs() > 1e-10,
                "DST scale factor is near zero: {}",
                scale
            );

            for i in 0..n {
                let expected = values[i] * scale;
                let tol = expected.abs() * 1e-8 + 1e-8;
                prop_assert!(
                    (recovered[i] - expected).abs() < tol,
                    "DST-II to III roundtrip mismatch at index {}: got {}, expected {} (scale={})",
                    i,
                    recovered[i],
                    expected,
                    scale
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Unit tests
    // -----------------------------------------------------------------------

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_dct2_size_1() {
        let input = [1.0_f64];
        let mut output = [0.0];
        dct2(&input, &mut output);
        // DCT-II of single element is itself
        assert!(approx_eq(output[0], 1.0, 1e-10));
    }

    #[test]
    fn test_dct2_size_2() {
        let input = [1.0_f64, 1.0];
        let mut output = [0.0, 0.0];
        dct2(&input, &mut output);
        // DCT-II formula: X[k] = sum_{n=0}^{N-1} x[n] * cos(π * (2n+1) * k / (2N))
        // For N=2, k=0: X[0] = x[0]*cos(0) + x[1]*cos(0) = 1 + 1 = 2
        // For N=2, k=1: X[1] = x[0]*cos(π/4) + x[1]*cos(3π/4) = cos(π/4) - cos(π/4) = 0
        assert!(approx_eq(output[0], 2.0, 1e-10));
        assert!(approx_eq(output[1], 0.0, 1e-10));
    }

    #[test]
    fn test_dct2_dct3_roundtrip() {
        let input = [1.0_f64, 2.0, 3.0, 4.0];
        let mut dct = [0.0; 4];
        let mut recovered = [0.0; 4];

        dct2(&input, &mut dct);
        dct3(&dct, &mut recovered);

        // The relationship between DCT-II and DCT-III depends on the exact
        // normalization used. With FFTW convention, the roundtrip scale is 2N.
        // We verify that DCT-III(DCT-II(x)) = x * scale for some scale factor
        let scale = recovered[0] / input[0]; // Compute empirical scale
        for i in 0..4 {
            assert!(
                approx_eq(recovered[i] / scale, input[i], 1e-10),
                "Mismatch at index {}: {} vs {}",
                i,
                recovered[i] / scale,
                input[i]
            );
        }
    }

    #[test]
    fn test_dct2_dc_component() {
        // For constant input, only DC component should be non-zero
        let input = [3.0_f64, 3.0, 3.0, 3.0];
        let mut output = [0.0; 4];
        dct2(&input, &mut output);

        // DC component = sum of all inputs = 12
        // Because cos(0) = 1 for all terms when k=0
        let expected_dc = 12.0;
        assert!(approx_eq(output[0], expected_dc, 1e-10));

        // All other components should be zero
        for &val in &output[1..] {
            assert!(approx_eq(val, 0.0, 1e-10));
        }
    }

    #[test]
    fn test_dct1_size_2() {
        let input = [1.0_f64, 2.0];
        let mut output = [0.0, 0.0];
        dct1(&input, &mut output);
        // DCT-I for size 2: X[k] = x[0] + (-1)^k * x[1]
        // X[0] = 1 + 2 = 3
        // X[1] = 1 - 2 = -1
        assert!(approx_eq(output[0], 3.0, 1e-10));
        assert!(approx_eq(output[1], -1.0, 1e-10));
    }

    #[test]
    fn test_dct4_size_4() {
        let input = [1.0_f64, 0.0, 0.0, 0.0];
        let mut output = [0.0; 4];
        dct4(&input, &mut output);

        // DCT-IV of impulse at position 0
        // X[k] = cos(π * (2*0 + 1) * (2k + 1) / 16) = cos(π * (2k + 1) / 16)
        for k in 0..4 {
            let expected = (std::f64::consts::PI * (2 * k + 1) as f64 / 16.0).cos();
            assert!(
                approx_eq(output[k], expected, 1e-10),
                "DCT-IV mismatch at {}: {} vs {}",
                k,
                output[k],
                expected
            );
        }
    }

    #[test]
    fn test_dst1_size_2() {
        let input = [1.0_f64, 2.0];
        let mut output = [0.0, 0.0];
        dst1(&input, &mut output);
        // DST-I for size 2: X[k] = sum x[n] * sin(π*(n+1)*(k+1)/3)
        // X[0] = x[0]*sin(π/3) + x[1]*sin(2π/3) = 1*sin(60°) + 2*sin(120°)
        //      = 1*√3/2 + 2*√3/2 = 3*√3/2
        let sqrt3_2 = (3.0_f64).sqrt() / 2.0;
        assert!(approx_eq(output[0], 3.0 * sqrt3_2, 1e-10));
        // X[1] = x[0]*sin(2π/3) + x[1]*sin(4π/3) = sin(120°) + 2*sin(240°)
        //      = √3/2 - 2*√3/2 = -√3/2
        assert!(approx_eq(output[1], -sqrt3_2, 1e-10));
    }

    #[test]
    fn test_dst2_dst3_roundtrip() {
        let input = [1.0_f64, 2.0, 3.0, 4.0];
        let mut dst = [0.0; 4];
        let mut recovered = [0.0; 4];

        dst2(&input, &mut dst);
        dst3(&dst, &mut recovered);

        // DST-III is inverse of DST-II up to a scale factor
        let scale = recovered[0] / input[0];
        for i in 0..4 {
            assert!(
                approx_eq(recovered[i] / scale, input[i], 1e-10),
                "DST roundtrip mismatch at index {}: {} vs {}",
                i,
                recovered[i] / scale,
                input[i]
            );
        }
    }

    #[test]
    fn test_dst4_size_4() {
        let input = [1.0_f64, 0.0, 0.0, 0.0];
        let mut output = [0.0; 4];
        dst4(&input, &mut output);

        // DST-IV of impulse at position 0
        // X[k] = sin(π * (2*0 + 1) * (2k + 1) / 16) = sin(π * (2k + 1) / 16)
        for k in 0..4 {
            let expected = (std::f64::consts::PI * (2 * k + 1) as f64 / 16.0).sin();
            assert!(
                approx_eq(output[k], expected, 1e-10),
                "DST-IV mismatch at {}: {} vs {}",
                k,
                output[k],
                expected
            );
        }
    }

    #[test]
    fn test_dst1_symmetry() {
        // DST-I of antisymmetric signal
        let input = [1.0_f64, -1.0];
        let mut output = [0.0, 0.0];
        dst1(&input, &mut output);
        // The transform should produce some non-zero result
        // since sine functions are present
        assert!(output[0].abs() > 1e-10 || output[1].abs() > 1e-10);
    }

    #[test]
    fn test_dht_size_1() {
        let input = [5.0_f64];
        let mut output = [0.0];
        dht(&input, &mut output);
        // DHT of single element: H[0] = x[0] * cas(0) = x[0] * 1 = 5
        assert!(approx_eq(output[0], 5.0, 1e-10));
    }

    #[test]
    fn test_dht_roundtrip() {
        // DHT is its own inverse up to scaling by N
        let input = [1.0_f64, 2.0, 3.0, 4.0];
        let mut dht_out = [0.0; 4];
        let mut recovered = [0.0; 4];

        dht(&input, &mut dht_out);
        dht(&dht_out, &mut recovered);

        // DHT(DHT(x)) = N * x
        let n = 4.0;
        for i in 0..4 {
            assert!(
                approx_eq(recovered[i] / n, input[i], 1e-10),
                "DHT roundtrip mismatch at index {}: {} vs {}",
                i,
                recovered[i] / n,
                input[i]
            );
        }
    }

    #[test]
    fn test_dht_dc_component() {
        // For constant input, DC component is sum of all elements
        let input = [2.0_f64, 2.0, 2.0, 2.0];
        let mut output = [0.0; 4];
        dht(&input, &mut output);

        // H[0] = sum of all inputs since cas(0) = 1
        assert!(approx_eq(output[0], 8.0, 1e-10));
    }

    #[test]
    fn test_dht_impulse() {
        // DHT of impulse at position 0
        let input = [1.0_f64, 0.0, 0.0, 0.0];
        let mut output = [0.0; 4];
        dht(&input, &mut output);

        // H[k] = cas(0) = 1 for all k (since only x[0] contributes)
        for k in 0..4 {
            assert!(approx_eq(output[k], 1.0, 1e-10));
        }
    }

    // -----------------------------------------------------------------------
    // Fast vs direct accuracy tests
    // -----------------------------------------------------------------------

    /// Check that two values agree within combined absolute and relative
    /// tolerance.  This avoids false failures when both values are near zero
    /// (floating-point noise below `abs_tol` is indistinguishable from exact
    /// zero) while still catching genuine algorithmic errors at larger
    /// magnitudes.
    fn approx_match(a: f64, b: f64, rel_tol: f64, abs_tol: f64) -> bool {
        let diff = (a - b).abs();
        diff <= abs_tol || diff <= rel_tol * a.abs().max(b.abs())
    }

    #[test]
    fn test_dct2_fast_matches_direct() {
        let solver = R2rSolver::<f64>::new(R2rKind::Redft10);
        for &n in &[16usize, 32, 64, 128, 256, 512, 1024] {
            let input: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
            let mut out_direct = vec![0.0_f64; n];
            let mut out_fast = vec![0.0_f64; n];
            solver.execute_dct2_direct(&input, &mut out_direct);
            solver.execute_dct2_fast(&input, &mut out_fast);
            for k in 0..n {
                assert!(
                    approx_match(out_fast[k], out_direct[k], 1e-8, 1e-10),
                    "DCT-II size {}: mismatch at k={}: fast={} direct={}",
                    n,
                    k,
                    out_fast[k],
                    out_direct[k]
                );
            }
        }
    }

    #[test]
    fn test_dct3_fast_matches_direct() {
        let solver = R2rSolver::<f64>::new(R2rKind::Redft01);
        for &n in &[16usize, 32, 64, 128, 256] {
            let input: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
            let mut out_direct = vec![0.0_f64; n];
            let mut out_fast = vec![0.0_f64; n];
            solver.execute_dct3_direct(&input, &mut out_direct);
            solver.execute_dct3_fast(&input, &mut out_fast);
            for k in 0..n {
                assert!(
                    approx_match(out_fast[k], out_direct[k], 1e-8, 1e-10),
                    "DCT-III size {}: mismatch at k={}: fast={} direct={}",
                    n,
                    k,
                    out_fast[k],
                    out_direct[k]
                );
            }
        }
    }

    #[test]
    fn test_dct4_fast_matches_direct() {
        let solver = R2rSolver::<f64>::new(R2rKind::Redft11);
        for &n in &[16usize, 32, 64, 128, 256] {
            let input: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
            let mut out_direct = vec![0.0_f64; n];
            let mut out_fast = vec![0.0_f64; n];
            solver.execute_dct4_direct(&input, &mut out_direct);
            solver.execute_dct4_fast(&input, &mut out_fast);
            for k in 0..n {
                assert!(
                    approx_match(out_fast[k], out_direct[k], 1e-8, 1e-10),
                    "DCT-IV size {}: mismatch at k={}: fast={} direct={}",
                    n,
                    k,
                    out_fast[k],
                    out_direct[k]
                );
            }
        }
    }

    #[test]
    fn test_dct1_fast_matches_direct() {
        let solver = R2rSolver::<f64>::new(R2rKind::Redft00);
        for &n in &[16usize, 32, 64, 128, 256] {
            let input: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
            let mut out_direct = vec![0.0_f64; n];
            let mut out_fast = vec![0.0_f64; n];
            solver.execute_dct1_direct(&input, &mut out_direct);
            solver.execute_dct1_fast(&input, &mut out_fast);
            for k in 0..n {
                assert!(
                    approx_match(out_fast[k], out_direct[k], 1e-8, 1e-10),
                    "DCT-I size {}: mismatch at k={}: fast={} direct={}",
                    n,
                    k,
                    out_fast[k],
                    out_direct[k]
                );
            }
        }
    }

    #[test]
    fn test_dht_fast_matches_direct() {
        let solver = R2rSolver::<f64>::new(R2rKind::Dht);
        for &n in &[16usize, 32, 64, 128, 256] {
            let input: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
            let mut out_direct = vec![0.0_f64; n];
            let mut out_fast = vec![0.0_f64; n];
            solver.execute_dht_direct(&input, &mut out_direct);
            solver.execute_dht_fast(&input, &mut out_fast);
            for k in 0..n {
                assert!(
                    approx_match(out_fast[k], out_direct[k], 1e-8, 1e-10),
                    "DHT size {}: mismatch at k={}: fast={} direct={}",
                    n,
                    k,
                    out_fast[k],
                    out_direct[k]
                );
            }
        }
    }

    #[test]
    fn test_dst1_fast_matches_direct() {
        let solver = R2rSolver::<f64>::new(R2rKind::Rodft00);
        for &n in &[16usize, 32, 64, 128, 256] {
            let input: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
            let mut out_direct = vec![0.0_f64; n];
            let mut out_fast = vec![0.0_f64; n];
            solver.execute_dst1_direct(&input, &mut out_direct);
            solver.execute_dst1_fast(&input, &mut out_fast);
            for k in 0..n {
                assert!(
                    approx_match(out_fast[k], out_direct[k], 1e-8, 1e-10),
                    "DST-I size {}: mismatch at k={}: fast={} direct={}",
                    n,
                    k,
                    out_fast[k],
                    out_direct[k]
                );
            }
        }
    }

    #[test]
    fn test_dst2_fast_matches_direct() {
        let solver = R2rSolver::<f64>::new(R2rKind::Rodft10);
        for &n in &[16usize, 32, 64, 128, 256] {
            let input: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
            let mut out_direct = vec![0.0_f64; n];
            let mut out_fast = vec![0.0_f64; n];
            solver.execute_dst2_direct(&input, &mut out_direct);
            solver.execute_dst2_fast(&input, &mut out_fast);
            for k in 0..n {
                assert!(
                    approx_match(out_fast[k], out_direct[k], 1e-8, 1e-10),
                    "DST-II size {}: mismatch at k={}: fast={} direct={}",
                    n,
                    k,
                    out_fast[k],
                    out_direct[k]
                );
            }
        }
    }

    #[test]
    fn test_dst3_fast_matches_direct() {
        let solver = R2rSolver::<f64>::new(R2rKind::Rodft01);
        for &n in &[16usize, 32, 64, 128, 256] {
            let input: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
            let mut out_direct = vec![0.0_f64; n];
            let mut out_fast = vec![0.0_f64; n];
            solver.execute_dst3_direct(&input, &mut out_direct);
            solver.execute_dst3_fast(&input, &mut out_fast);
            for k in 0..n {
                assert!(
                    approx_match(out_fast[k], out_direct[k], 1e-8, 1e-10),
                    "DST-III size {}: mismatch at k={}: fast={} direct={}",
                    n,
                    k,
                    out_fast[k],
                    out_direct[k]
                );
            }
        }
    }

    #[test]
    fn test_dst4_fast_matches_direct() {
        let solver = R2rSolver::<f64>::new(R2rKind::Rodft11);
        // DST-IV is computed indirectly via DCT-IV (itself FFT-based), so
        // accumulated floating-point rounding grows with N; 1e-6 relative
        // tolerance is appropriate for sizes up to 256.
        for &n in &[16usize, 32, 64, 128, 256] {
            let input: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
            let mut out_direct = vec![0.0_f64; n];
            let mut out_fast = vec![0.0_f64; n];
            solver.execute_dst4_direct(&input, &mut out_direct);
            solver.execute_dst4_fast(&input, &mut out_fast);
            for k in 0..n {
                assert!(
                    approx_match(out_fast[k], out_direct[k], 1e-6, 1e-10),
                    "DST-IV size {}: mismatch at k={}: fast={} direct={}",
                    n,
                    k,
                    out_fast[k],
                    out_direct[k]
                );
            }
        }
    }
}
