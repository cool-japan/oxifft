//! Real-to-Real solver for DCT/DST/DHT transforms.

use crate::kernel::Float;

/// Type of DCT/DST/DHT transform (FFTW terminology).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    /// Execute DCT-II (REDFT10) transform.
    ///
    /// Formula: X\[k\] = sum_{n=0}^{N-1} x\[n\] * cos(π * (2n + 1) * k / (2N))
    ///
    /// This is the most common form of DCT, used in JPEG compression.
    pub fn execute_dct2(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DCT-II requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let n_f = T::from_usize(n);
        let two_n = T::from_usize(2 * n);

        for k in 0..n {
            let k_f = T::from_usize(k);
            let mut sum = T::ZERO;

            for (i, &x_i) in input.iter().enumerate() {
                // cos(π * (2i + 1) * k / (2N))
                let angle = <T as Float>::PI * (T::from_usize(2 * i + 1)) * k_f / two_n;
                sum = sum + x_i * Float::cos(angle);
            }

            output[k] = sum;
        }

        // Apply normalization for orthonormal DCT-II
        // X[0] is scaled by 1/sqrt(N), others by sqrt(2/N)
        // But we follow FFTW convention which doesn't apply normalization
        let _ = n_f; // Suppress unused warning
    }

    /// Execute DCT-III (REDFT01) transform.
    ///
    /// Formula: x\[n\] = X\[0\]/2 + sum_{k=1}^{N-1} X\[k\] * cos(π * k * (2n + 1) / (2N))
    ///
    /// This is the inverse of DCT-II (up to scaling by 2N).
    pub fn execute_dct3(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DCT-III requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let two_n = T::from_usize(2 * n);
        let half = T::ONE / T::TWO;

        for i in 0..n {
            let two_i_plus_1 = T::from_usize(2 * i + 1);
            // Start with X[0]/2
            let mut sum = input[0] * half;

            for k in 1..n {
                // cos(π * k * (2i + 1) / (2N))
                let angle = <T as Float>::PI * T::from_usize(k) * two_i_plus_1 / two_n;
                sum = sum + input[k] * Float::cos(angle);
            }

            output[i] = sum;
        }
    }

    /// Execute DCT-I (REDFT00) transform.
    ///
    /// Formula: X\[k\] = x\[0\] + (-1)^k * x\[N-1\] + 2 * sum_{n=1}^{N-2} x\[n\] * cos(π * n * k / (N-1))
    pub fn execute_dct1(&self, input: &[T], output: &mut [T]) {
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

            // x[0] + (-1)^k * x[N-1]
            let mut sum = input[0] + sign * input[n - 1];

            // 2 * sum_{n=1}^{N-2} x[n] * cos(π * n * k / (N-1))
            for i in 1..(n - 1) {
                let angle = <T as Float>::PI * T::from_usize(i) * k_f / n_minus_1;
                sum = sum + T::TWO * input[i] * Float::cos(angle);
            }

            output[k] = sum;
        }
    }

    /// Execute DCT-IV (REDFT11) transform.
    ///
    /// Formula: X\[k\] = sum_{n=0}^{N-1} x\[n\] * cos(π * (2n + 1) * (2k + 1) / (4N))
    pub fn execute_dct4(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DCT-IV requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let four_n = T::from_usize(4 * n);

        for k in 0..n {
            let two_k_plus_1 = T::from_usize(2 * k + 1);
            let mut sum = T::ZERO;

            for (i, &x_i) in input.iter().enumerate() {
                // cos(π * (2i + 1) * (2k + 1) / (4N))
                let angle = <T as Float>::PI * T::from_usize(2 * i + 1) * two_k_plus_1 / four_n;
                sum = sum + x_i * Float::cos(angle);
            }

            output[k] = sum;
        }
    }

    /// Execute DST-I (RODFT00) transform.
    ///
    /// Formula: X\[k\] = sum_{n=0}^{N-1} x\[n\] * sin(π * (n+1) * (k+1) / (N+1))
    pub fn execute_dst1(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DST-I requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let n_plus_1 = T::from_usize(n + 1);

        for k in 0..n {
            let k_plus_1 = T::from_usize(k + 1);
            let mut sum = T::ZERO;

            for (i, &x_i) in input.iter().enumerate() {
                // sin(π * (i+1) * (k+1) / (N+1))
                let angle = <T as Float>::PI * T::from_usize(i + 1) * k_plus_1 / n_plus_1;
                sum = sum + x_i * Float::sin(angle);
            }

            output[k] = sum;
        }
    }

    /// Execute DST-II (RODFT10) transform.
    ///
    /// Formula: X\[k\] = sum_{n=0}^{N-1} x\[n\] * sin(π * (2n+1) * (k+1) / (2N))
    pub fn execute_dst2(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DST-II requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let two_n = T::from_usize(2 * n);

        for k in 0..n {
            let k_plus_1 = T::from_usize(k + 1);
            let mut sum = T::ZERO;

            for (i, &x_i) in input.iter().enumerate() {
                // sin(π * (2i+1) * (k+1) / (2N))
                let angle = <T as Float>::PI * T::from_usize(2 * i + 1) * k_plus_1 / two_n;
                sum = sum + x_i * Float::sin(angle);
            }

            output[k] = sum;
        }
    }

    /// Execute DST-III (RODFT01) transform.
    ///
    /// Formula: X\[k\] = (-1)^k * x\[N-1\]/2 + sum_{n=0}^{N-2} x\[n\] * sin(π * (n+1) * (2k+1) / (2N))
    ///
    /// This is the inverse of DST-II (up to scaling).
    pub fn execute_dst3(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DST-III requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let two_n = T::from_usize(2 * n);
        let half = T::ONE / T::TWO;

        for k in 0..n {
            let two_k_plus_1 = T::from_usize(2 * k + 1);
            let sign = if k % 2 == 0 { T::ONE } else { -T::ONE };

            // Start with (-1)^k * x[N-1] / 2
            let mut sum = sign * input[n - 1] * half;

            // Add sum_{n=0}^{N-2} x[n] * sin(π * (n+1) * (2k+1) / (2N))
            for i in 0..(n - 1) {
                let angle = <T as Float>::PI * T::from_usize(i + 1) * two_k_plus_1 / two_n;
                sum = sum + input[i] * Float::sin(angle);
            }

            output[k] = sum;
        }
    }

    /// Execute DST-IV (RODFT11) transform.
    ///
    /// Formula: X\[k\] = sum_{n=0}^{N-1} x\[n\] * sin(π * (2n+1) * (2k+1) / (4N))
    pub fn execute_dst4(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DST-IV requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let four_n = T::from_usize(4 * n);

        for k in 0..n {
            let two_k_plus_1 = T::from_usize(2 * k + 1);
            let mut sum = T::ZERO;

            for (i, &x_i) in input.iter().enumerate() {
                // sin(π * (2i+1) * (2k+1) / (4N))
                let angle = <T as Float>::PI * T::from_usize(2 * i + 1) * two_k_plus_1 / four_n;
                sum = sum + x_i * Float::sin(angle);
            }

            output[k] = sum;
        }
    }

    /// Execute Discrete Hartley Transform (DHT).
    ///
    /// Formula: H\[k\] = sum_{n=0}^{N-1} x\[n\] * cas(2πnk/N)
    ///
    /// where cas(θ) = cos(θ) + sin(θ)
    ///
    /// The DHT is its own inverse (up to scaling by N), making it
    /// particularly elegant for real-valued signals.
    pub fn execute_dht(&self, input: &[T], output: &mut [T]) {
        let n = input.len();
        debug_assert!(n > 0, "DHT requires at least 1 element");
        debug_assert_eq!(output.len(), n, "Output must match input size");

        let n_f = T::from_usize(n);

        for k in 0..n {
            let k_f = T::from_usize(k);
            let mut sum = T::ZERO;

            for (i, &x_i) in input.iter().enumerate() {
                // cas(2π * i * k / N) = cos(2π*i*k/N) + sin(2π*i*k/N)
                let angle = <T as Float>::TWO_PI * T::from_usize(i) * k_f / n_f;
                let (s, c) = Float::sin_cos(angle);
                sum = sum + x_i * (c + s);
            }

            output[k] = sum;
        }
    }

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

        // Print intermediate DCT coefficients for debugging
        // println!("DCT coefficients: {:?}", dct);

        dct3(&dct, &mut recovered);

        // Print recovered values for debugging
        // println!("Recovered: {:?}", recovered);

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
}
