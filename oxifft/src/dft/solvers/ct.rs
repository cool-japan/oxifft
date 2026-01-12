//! Cooley-Tukey FFT solver.
//!
//! Implements the classic Cooley-Tukey algorithm with DIT/DIF variants.
//! This is the workhorse algorithm for power-of-2 and composite sizes.

// Static lookup table after statements is intentional for readability
#![allow(clippy::items_after_statements)]

use crate::dft::codelets::{
    notw_1024_dispatch, notw_128_dispatch, notw_16_dispatch, notw_256_dispatch, notw_2_dispatch,
    notw_32_dispatch, notw_4096_dispatch, notw_4_dispatch, notw_512_dispatch, notw_64_dispatch,
    notw_8_dispatch,
};
use crate::kernel::{Complex, Float};
use crate::prelude::*;

use super::super::problem::Sign;
use super::simd_butterfly::dit_butterflies_f64;

/// Cooley-Tukey variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CtVariant {
    /// Decimation-in-Time: input is bit-reversed, output is in order
    Dit,
    /// Decimation-in-Frequency: input is in order, output is bit-reversed
    Dif,
    /// Radix-4 DIT: More efficient for sizes divisible by 4
    DitRadix4,
    /// Radix-8 DIT: More efficient for sizes divisible by 8
    DitRadix8,
    /// Split-radix DIT: Combines radix-2 and radix-4 for near-optimal operation count
    SplitRadix,
}

/// Cooley-Tukey FFT solver.
///
/// Implements the Cooley-Tukey algorithm with radix-2, radix-4, radix-8, and split-radix variants:
/// - DIT (Decimation-in-Time): Processes input in bit-reversed order
/// - DIF (Decimation-in-Frequency): Produces output in bit-reversed order
/// - DitRadix4: More efficient for power-of-4 sizes (75% of radix-2 multiplies)
/// - DitRadix8: More efficient for power-of-8 sizes (further reduced multiplies)
/// - SplitRadix: Combines radix-2 and radix-4 for near-optimal operation count
///
/// Time complexity: O(n log n)
/// Space complexity: O(1) for in-place, O(n) for out-of-place
pub struct CooleyTukeySolver<T: Float> {
    /// DIT or DIF variant.
    pub variant: CtVariant,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> Default for CooleyTukeySolver<T> {
    fn default() -> Self {
        Self::new(CtVariant::Dit)
    }
}

impl<T: Float> CooleyTukeySolver<T> {
    /// Create a new Cooley-Tukey solver.
    #[must_use]
    pub fn new(variant: CtVariant) -> Self {
        Self {
            variant,
            _marker: core::marker::PhantomData,
        }
    }

    /// Solver name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self.variant {
            CtVariant::Dit => "dft-ct-dit",
            CtVariant::Dif => "dft-ct-dif",
            CtVariant::DitRadix4 => "dft-ct-dit-radix4",
            CtVariant::DitRadix8 => "dft-ct-dit-radix8",
            CtVariant::SplitRadix => "dft-ct-split-radix",
        }
    }

    /// Check if size is a power of 8 (suitable for radix-8).
    #[must_use]
    pub fn is_power_of_8(n: usize) -> bool {
        n > 0 && (n & (n - 1)) == 0 && (n.trailing_zeros() % 3 == 0)
    }

    /// Check if size is a power of 4 (suitable for radix-4).
    #[must_use]
    pub fn is_power_of_4(n: usize) -> bool {
        n > 0 && (n & (n - 1)) == 0 && (n.trailing_zeros() % 2 == 0)
    }

    /// Check if size is a power of 2.
    #[must_use]
    pub fn applicable(n: usize) -> bool {
        n > 0 && (n & (n - 1)) == 0
    }

    /// Execute the Cooley-Tukey FFT.
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        let n = input.len();
        debug_assert_eq!(n, output.len());
        debug_assert!(Self::applicable(n), "Size must be power of 2");

        if n <= 1 {
            if n == 1 {
                output[0] = input[0];
            }
            return;
        }

        // Use optimized codelets for small sizes (with SIMD dispatch for f64)
        let sign_int = sign.value();
        match n {
            2 => {
                output.copy_from_slice(input);
                notw_2_dispatch(output);
                return;
            }
            4 => {
                output.copy_from_slice(input);
                notw_4_dispatch(output, sign_int);
                return;
            }
            8 => {
                output.copy_from_slice(input);
                notw_8_dispatch(output, sign_int);
                return;
            }
            16 => {
                output.copy_from_slice(input);
                notw_16_dispatch(output, sign_int);
                return;
            }
            32 => {
                output.copy_from_slice(input);
                notw_32_dispatch(output, sign_int);
                return;
            }
            64 => {
                output.copy_from_slice(input);
                notw_64_dispatch(output, sign_int);
                return;
            }
            128 => {
                output.copy_from_slice(input);
                notw_128_dispatch(output, sign_int);
                return;
            }
            256 => {
                output.copy_from_slice(input);
                notw_256_dispatch(output, sign_int);
                return;
            }
            512 => {
                output.copy_from_slice(input);
                notw_512_dispatch(output, sign_int);
                return;
            }
            1024 => {
                output.copy_from_slice(input);
                notw_1024_dispatch(output, sign_int);
                return;
            }
            4096 => {
                output.copy_from_slice(input);
                notw_4096_dispatch(output, sign_int);
                return;
            }
            _ => {}
        }

        match self.variant {
            CtVariant::Dit => self.execute_dit(input, output, sign),
            CtVariant::Dif => self.execute_dif(input, output, sign),
            CtVariant::DitRadix4 => self.execute_dit_radix4(input, output, sign),
            CtVariant::DitRadix8 => self.execute_dit_radix8(input, output, sign),
            CtVariant::SplitRadix => self.execute_split_radix(input, output, sign),
        }
    }

    /// Execute in-place Cooley-Tukey FFT.
    pub fn execute_inplace(&self, data: &mut [Complex<T>], sign: Sign) {
        let n = data.len();
        debug_assert!(Self::applicable(n), "Size must be power of 2");

        if n <= 1 {
            return;
        }

        // Use optimized codelets for small sizes (with SIMD dispatch for f64)
        let sign_int = sign.value();
        match n {
            2 => {
                notw_2_dispatch(data);
                return;
            }
            4 => {
                notw_4_dispatch(data, sign_int);
                return;
            }
            8 => {
                notw_8_dispatch(data, sign_int);
                return;
            }
            16 => {
                notw_16_dispatch(data, sign_int);
                return;
            }
            32 => {
                notw_32_dispatch(data, sign_int);
                return;
            }
            64 => {
                notw_64_dispatch(data, sign_int);
                return;
            }
            128 => {
                notw_128_dispatch(data, sign_int);
                return;
            }
            256 => {
                notw_256_dispatch(data, sign_int);
                return;
            }
            512 => {
                notw_512_dispatch(data, sign_int);
                return;
            }
            1024 => {
                notw_1024_dispatch(data, sign_int);
                return;
            }
            4096 => {
                notw_4096_dispatch(data, sign_int);
                return;
            }
            _ => {}
        }

        match self.variant {
            CtVariant::Dit => self.execute_dit_inplace(data, sign),
            CtVariant::Dif => self.execute_dif_inplace(data, sign),
            CtVariant::DitRadix4 => self.execute_dit_radix4_inplace(data, sign),
            CtVariant::DitRadix8 => self.execute_dit_radix8_inplace(data, sign),
            CtVariant::SplitRadix => self.execute_split_radix_inplace(data, sign),
        }
    }

    /// Decimation-in-Time out-of-place.
    fn execute_dit(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        let n = input.len();

        // Copy input with bit-reversal permutation
        for i in 0..n {
            output[bit_reverse(i, n)] = input[i];
        }

        // Iterative butterfly stages
        self.dit_butterflies(output, sign);
    }

    /// Decimation-in-Time in-place.
    fn execute_dit_inplace(&self, data: &mut [Complex<T>], sign: Sign) {
        // Bit-reversal permutation
        bit_reverse_permute(data);

        // Iterative butterfly stages
        self.dit_butterflies(data, sign);
    }

    /// DIT butterfly stages.
    fn dit_butterflies(&self, data: &mut [Complex<T>], sign: Sign) {
        // Use SIMD-accelerated version for f64
        if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f64>() {
            // Safety: We've verified T is f64, so this pointer cast is safe
            let data_f64: &mut [Complex<f64>] =
                unsafe { &mut *(data as *mut [Complex<T>] as *mut [Complex<f64>]) };
            dit_butterflies_f64(data_f64, sign);
            return;
        }

        // Scalar fallback for other types
        self.dit_butterflies_scalar(data, sign);
    }

    /// Scalar DIT butterfly stages (for non-f64 types).
    /// Uses twiddle recurrence to avoid repeated sin/cos calls.
    fn dit_butterflies_scalar(&self, data: &mut [Complex<T>], sign: Sign) {
        let n = data.len();
        let log_n = n.trailing_zeros() as usize;
        let sign_val = T::from_isize(sign.value() as isize);

        // Process each stage
        let mut m = 2; // Current butterfly size
        for _ in 0..log_n {
            let half_m = m / 2;
            let angle_step = sign_val * T::TWO_PI / T::from_usize(m);
            let w_step = Complex::cis(angle_step);

            // Process each group of butterflies
            for k in (0..n).step_by(m) {
                let mut w = Complex::new(T::ONE, T::ZERO);

                for j in 0..half_m {
                    let u = data[k + j];
                    let t = data[k + j + half_m] * w;

                    data[k + j] = u + t;
                    data[k + j + half_m] = u - t;

                    w = w * w_step;
                }
            }

            m *= 2;
        }
    }

    /// Decimation-in-Frequency out-of-place.
    fn execute_dif(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        // Copy input to output
        output.copy_from_slice(input);

        // DIF butterflies
        self.dif_butterflies(output, sign);

        // Bit-reversal permutation on output
        bit_reverse_permute(output);
    }

    /// Decimation-in-Frequency in-place.
    fn execute_dif_inplace(&self, data: &mut [Complex<T>], sign: Sign) {
        // DIF butterflies first
        self.dif_butterflies(data, sign);

        // Then bit-reversal permutation
        bit_reverse_permute(data);
    }

    /// DIF butterfly stages.
    /// Uses twiddle recurrence to avoid repeated sin/cos calls.
    fn dif_butterflies(&self, data: &mut [Complex<T>], sign: Sign) {
        let n = data.len();
        let log_n = n.trailing_zeros() as usize;
        let sign_val = T::from_isize(sign.value() as isize);

        // Process each stage (from large to small)
        let mut m = n;
        for _ in 0..log_n {
            let half_m = m / 2;
            let angle_step = sign_val * T::TWO_PI / T::from_usize(m);
            let w_step = Complex::cis(angle_step);

            // Process each group of butterflies
            for k in (0..n).step_by(m) {
                let mut w = Complex::new(T::ONE, T::ZERO);

                for j in 0..half_m {
                    let u = data[k + j];
                    let v = data[k + j + half_m];

                    data[k + j] = u + v;
                    data[k + j + half_m] = (u - v) * w;

                    w = w * w_step;
                }
            }

            m /= 2;
        }
    }

    /// Radix-4 DIT out-of-place.
    ///
    /// Implements a true radix-4 FFT that reduces multiplications by ~25% compared to radix-2.
    /// Uses standard bit-reversal followed by combined radix-4 butterfly stages.
    fn execute_dit_radix4(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        let n = input.len();

        // Copy input with bit-reversal permutation (same as radix-2)
        for i in 0..n {
            output[bit_reverse(i, n)] = input[i];
        }

        // Radix-4 butterflies
        self.dit_radix4_butterflies(output, sign);
    }

    /// Radix-4 DIT in-place.
    fn execute_dit_radix4_inplace(&self, data: &mut [Complex<T>], sign: Sign) {
        // Bit-reversal permutation (same as radix-2)
        bit_reverse_permute(data);

        // Radix-4 butterflies
        self.dit_radix4_butterflies(data, sign);
    }

    /// Radix-4 DIT butterfly stages.
    ///
    /// After standard bit-reversal, we can perform radix-4 butterflies that
    /// combine two radix-2 stages into one, reducing the number of complex
    /// multiplications by ~25%.
    ///
    /// For sizes that aren't powers of 4, we do one radix-2 stage first.
    ///
    /// The key insight is that a radix-4 butterfly combines two radix-2 stages:
    /// - First radix-2 stage: pairs at distance 1 (within sub-blocks of 2)
    /// - Second radix-2 stage: pairs at distance 2 (within sub-blocks of 4)
    fn dit_radix4_butterflies(&self, data: &mut [Complex<T>], sign: Sign) {
        let n = data.len();
        let log_n = n.trailing_zeros() as usize;
        let sign_val = T::from_isize(sign.value() as isize);

        let mut s = 0; // Current stage (in terms of radix-2 stages)

        // If log_n is odd, do one radix-2 stage first
        if log_n % 2 == 1 {
            let m = 2;
            // Radix-2 butterflies for size 2
            for k in (0..n).step_by(m) {
                let u = data[k];
                let v = data[k + 1];
                data[k] = u + v;
                data[k + 1] = u - v;
            }
            s = 1;
        }

        // Radix-4 stages: each combines stages s, s+1 (which would be m=2^s and m=2^(s+1))
        // After the combined stage, s += 2
        while s + 1 < log_n {
            // This radix-4 stage combines radix-2 stages for m1=2^(s+1) and m2=2^(s+2)
            let m1 = 1 << (s + 1); // Distance for first radix-2 (half of first stage butterfly size)
            let m2 = 1 << (s + 2); // Full butterfly size for this radix-4 stage

            let half_m1 = m1 / 2; // Distance between elements in first radix-2: 2^s
            let half_m2 = m2 / 2; // Distance for second radix-2 within m2 group: 2^(s+1)

            // Twiddles for the equivalent radix-2 stages
            let angle_step_1 = sign_val * T::TWO_PI / T::from_usize(m1);
            let angle_step_2 = sign_val * T::TWO_PI / T::from_usize(m2);

            // Process groups of size m2
            for k in (0..n).step_by(m2) {
                // Within each group, we have m2/4 = half_m1 butterflies
                for j in 0..half_m1 {
                    // Twiddles: W_{m1}^j for first stage, W_{m2}^j and W_{m2}^(j+half_m2/2) for second
                    let angle1 = angle_step_1 * T::from_usize(j);
                    let w1 = Complex::cis(angle1);

                    let angle2_a = angle_step_2 * T::from_usize(j);
                    let angle2_b = angle_step_2 * T::from_usize(j + half_m1);
                    let w2_a = Complex::cis(angle2_a);
                    let w2_b = Complex::cis(angle2_b);

                    // Four elements for this butterfly
                    // In the first radix-2 stage, we'd pair (k+j, k+j+half_m1) and (k+j+half_m2, k+j+half_m2+half_m1)
                    // In the second radix-2 stage, we'd then pair results at distance half_m2
                    let i0 = k + j;
                    let i1 = k + j + half_m1;
                    let i2 = k + j + half_m2;
                    let i3 = k + j + half_m2 + half_m1;

                    let x0 = data[i0];
                    let x1 = data[i1];
                    let x2 = data[i2];
                    let x3 = data[i3];

                    // First radix-2 stage (distance half_m1), with twiddle w1
                    let a0 = x0 + x1 * w1;
                    let a1 = x0 - x1 * w1;
                    let a2 = x2 + x3 * w1;
                    let a3 = x2 - x3 * w1;

                    // Second radix-2 stage (distance half_m2), with twiddles w2_a, w2_b
                    data[i0] = a0 + a2 * w2_a;
                    data[i2] = a0 - a2 * w2_a;
                    data[i1] = a1 + a3 * w2_b;
                    data[i3] = a1 - a3 * w2_b;
                }
            }

            s += 2;
        }

        // If there's one more radix-2 stage left
        if s < log_n {
            let m = 1 << (s + 1);
            let half_m = m / 2;
            let angle_step = sign_val * T::TWO_PI / T::from_usize(m);
            let w_step = Complex::cis(angle_step);

            for k in (0..n).step_by(m) {
                let mut w = Complex::new(T::ONE, T::ZERO);
                for j in 0..half_m {
                    let u = data[k + j];
                    let t = data[k + j + half_m] * w;
                    data[k + j] = u + t;
                    data[k + j + half_m] = u - t;
                    w = w * w_step;
                }
            }
        }
    }

    /// Radix-8 DIT out-of-place.
    ///
    /// Implements a radix-8 FFT that combines three radix-2 stages into one,
    /// providing further optimization for sizes divisible by 8.
    fn execute_dit_radix8(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        let n = input.len();

        // Copy input with bit-reversal permutation
        for i in 0..n {
            output[bit_reverse(i, n)] = input[i];
        }

        // Radix-8 butterflies
        self.dit_radix8_butterflies(output, sign);
    }

    /// Radix-8 DIT in-place.
    fn execute_dit_radix8_inplace(&self, data: &mut [Complex<T>], sign: Sign) {
        // Bit-reversal permutation
        bit_reverse_permute(data);

        // Radix-8 butterflies
        self.dit_radix8_butterflies(data, sign);
    }

    /// Radix-8 DIT butterfly stages.
    ///
    /// Combines three radix-2 stages into one radix-8 stage.
    /// For log_n not divisible by 3, uses radix-2 and/or radix-4 for remaining stages.
    fn dit_radix8_butterflies(&self, data: &mut [Complex<T>], sign: Sign) {
        let n = data.len();
        let log_n = n.trailing_zeros() as usize;
        let sign_val = T::from_isize(sign.value() as isize);

        let mut s = 0; // Current stage (in radix-2 stages)

        // Handle initial stages if log_n % 3 != 0
        let remainder = log_n % 3;
        if remainder == 1 {
            // Do one radix-2 stage first
            for k in (0..n).step_by(2) {
                let u = data[k];
                let v = data[k + 1];
                data[k] = u + v;
                data[k + 1] = u - v;
            }
            s = 1;
        } else if remainder == 2 {
            // Do one radix-4 stage first (combines 2 radix-2 stages)
            self.radix4_stage(data, 0, sign_val);
            s = 2;
        }

        // Radix-8 stages: each combines 3 radix-2 stages
        while s + 2 < log_n {
            self.radix8_stage(data, s, sign_val);
            s += 3;
        }

        // Handle remaining stages
        if s + 1 < log_n {
            // Two stages left - use radix-4
            self.radix4_stage(data, s, sign_val);
            s += 2;
        }

        if s < log_n {
            // One stage left - use radix-2 with twiddle recurrence
            let m = 1 << (s + 1);
            let half_m = m / 2;
            let angle_step = sign_val * T::TWO_PI / T::from_usize(m);
            let w_step = Complex::cis(angle_step);

            for k in (0..n).step_by(m) {
                let mut w = Complex::new(T::ONE, T::ZERO);
                for j in 0..half_m {
                    let u = data[k + j];
                    let t = data[k + j + half_m] * w;
                    data[k + j] = u + t;
                    data[k + j + half_m] = u - t;
                    w = w * w_step;
                }
            }
        }
    }

    /// Perform a radix-4 stage starting at radix-2 stage `s`.
    #[inline]
    fn radix4_stage(&self, data: &mut [Complex<T>], s: usize, sign_val: T) {
        let n = data.len();
        let m1 = 1 << (s + 1);
        let m2 = 1 << (s + 2);
        let half_m1 = m1 / 2;
        let half_m2 = m2 / 2;

        let angle_step_1 = sign_val * T::TWO_PI / T::from_usize(m1);
        let angle_step_2 = sign_val * T::TWO_PI / T::from_usize(m2);

        for k in (0..n).step_by(m2) {
            for j in 0..half_m1 {
                let angle1 = angle_step_1 * T::from_usize(j);
                let w1 = Complex::cis(angle1);

                let angle2_a = angle_step_2 * T::from_usize(j);
                let angle2_b = angle_step_2 * T::from_usize(j + half_m1);
                let w2_a = Complex::cis(angle2_a);
                let w2_b = Complex::cis(angle2_b);

                let i0 = k + j;
                let i1 = k + j + half_m1;
                let i2 = k + j + half_m2;
                let i3 = k + j + half_m2 + half_m1;

                let x0 = data[i0];
                let x1 = data[i1];
                let x2 = data[i2];
                let x3 = data[i3];

                let a0 = x0 + x1 * w1;
                let a1 = x0 - x1 * w1;
                let a2 = x2 + x3 * w1;
                let a3 = x2 - x3 * w1;

                data[i0] = a0 + a2 * w2_a;
                data[i2] = a0 - a2 * w2_a;
                data[i1] = a1 + a3 * w2_b;
                data[i3] = a1 - a3 * w2_b;
            }
        }
    }

    /// Perform a radix-8 stage starting at radix-2 stage `s`.
    ///
    /// Combines three radix-2 stages into one radix-8 butterfly.
    /// Processes 8 elements at a time: indices at distances 2^s, 2^(s+1), 2^(s+2).
    #[inline]
    fn radix8_stage(&self, data: &mut [Complex<T>], s: usize, sign_val: T) {
        let n = data.len();

        // Block sizes for the three combined radix-2 stages
        let m1 = 1 << (s + 1); // First radix-2 block size
        let m2 = 1 << (s + 2); // Second radix-2 block size
        let m3 = 1 << (s + 3); // Third radix-2 block size (total radix-8 block)

        let d0 = 1 << s; // Distance for first stage: 2^s
        let d1 = 1 << (s + 1); // Distance for second stage: 2^(s+1)
        let d2 = 1 << (s + 2); // Distance for third stage: 2^(s+2)

        // Twiddle angle steps for each of the three stages
        let angle_step_1 = sign_val * T::TWO_PI / T::from_usize(m1);
        let angle_step_2 = sign_val * T::TWO_PI / T::from_usize(m2);
        let angle_step_3 = sign_val * T::TWO_PI / T::from_usize(m3);

        // Pre-compute all twiddles using recurrence (much faster than sin/cos per element)
        let w1_step = Complex::cis(angle_step_1);
        let w2_step = Complex::cis(angle_step_2);
        let w3_step = Complex::cis(angle_step_3);

        // Pre-compute offset factors (computed only once per stage)
        let w2_offset_d0 = Complex::cis(angle_step_2 * T::from_usize(d0));
        let w3_offset_d0 = Complex::cis(angle_step_3 * T::from_usize(d0));
        let w3_offset_d1 = Complex::cis(angle_step_3 * T::from_usize(d1));
        let w3_offset_d0_d1 = Complex::cis(angle_step_3 * T::from_usize(d0 + d1));

        // Pre-allocate twiddle arrays
        let mut tw1: Vec<Complex<T>> = Vec::with_capacity(d0);
        let mut tw2_0: Vec<Complex<T>> = Vec::with_capacity(d0);
        let mut tw2_1: Vec<Complex<T>> = Vec::with_capacity(d0);
        let mut tw3_0: Vec<Complex<T>> = Vec::with_capacity(d0);
        let mut tw3_1: Vec<Complex<T>> = Vec::with_capacity(d0);
        let mut tw3_2: Vec<Complex<T>> = Vec::with_capacity(d0);
        let mut tw3_3: Vec<Complex<T>> = Vec::with_capacity(d0);

        // Generate twiddles using recurrence (only 3 complex multiplies per iteration)
        let mut w1 = Complex::new(T::ONE, T::ZERO);
        let mut w2 = Complex::new(T::ONE, T::ZERO);
        let mut w3 = Complex::new(T::ONE, T::ZERO);

        for _ in 0..d0 {
            tw1.push(w1);
            tw2_0.push(w2);
            tw2_1.push(w2 * w2_offset_d0);
            tw3_0.push(w3);
            tw3_1.push(w3 * w3_offset_d0);
            tw3_2.push(w3 * w3_offset_d1);
            tw3_3.push(w3 * w3_offset_d0_d1);

            // Advance using recurrence
            w1 = w1 * w1_step;
            w2 = w2 * w2_step;
            w3 = w3 * w3_step;
        }

        // Process groups of size m3 (8 * 2^s elements)
        for k in (0..n).step_by(m3) {
            // Within each group, process d0 butterflies
            for j in 0..d0 {
                // 8 input indices
                let i0 = k + j;
                let i1 = k + j + d0;
                let i2 = k + j + d1;
                let i3 = k + j + d0 + d1;
                let i4 = k + j + d2;
                let i5 = k + j + d0 + d2;
                let i6 = k + j + d1 + d2;
                let i7 = k + j + d0 + d1 + d2;

                let x0 = data[i0];
                let x1 = data[i1];
                let x2 = data[i2];
                let x3 = data[i3];
                let x4 = data[i4];
                let x5 = data[i5];
                let x6 = data[i6];
                let x7 = data[i7];

                // First radix-2 stage: combine pairs at distance d0
                let a0 = x0 + x1 * tw1[j];
                let a1 = x0 - x1 * tw1[j];
                let a2 = x2 + x3 * tw1[j];
                let a3 = x2 - x3 * tw1[j];
                let a4 = x4 + x5 * tw1[j];
                let a5 = x4 - x5 * tw1[j];
                let a6 = x6 + x7 * tw1[j];
                let a7 = x6 - x7 * tw1[j];

                // Second radix-2 stage: combine pairs at distance d1
                let b0 = a0 + a2 * tw2_0[j];
                let b2 = a0 - a2 * tw2_0[j];
                let b1 = a1 + a3 * tw2_1[j];
                let b3 = a1 - a3 * tw2_1[j];
                let b4 = a4 + a6 * tw2_0[j];
                let b6 = a4 - a6 * tw2_0[j];
                let b5 = a5 + a7 * tw2_1[j];
                let b7 = a5 - a7 * tw2_1[j];

                // Third radix-2 stage: combine pairs at distance d2
                data[i0] = b0 + b4 * tw3_0[j];
                data[i4] = b0 - b4 * tw3_0[j];
                data[i1] = b1 + b5 * tw3_1[j];
                data[i5] = b1 - b5 * tw3_1[j];
                data[i2] = b2 + b6 * tw3_2[j];
                data[i6] = b2 - b6 * tw3_2[j];
                data[i3] = b3 + b7 * tw3_3[j];
                data[i7] = b3 - b7 * tw3_3[j];
            }
        }
    }

    /// Split-radix FFT out-of-place.
    ///
    /// Implements the split-radix FFT algorithm which combines radix-2 and radix-4
    /// decompositions to achieve near-optimal operation count.
    ///
    /// The key idea: decompose DFT(n) as:
    /// - DFT(n/2) on even indices (radix-2)
    /// - Two DFT(n/4) on odd indices (radix-4 style)
    ///
    /// The split-radix algorithm achieves close to the theoretical minimum
    /// number of multiplications for power-of-2 sizes.
    fn execute_split_radix(&self, input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
        // Copy input to output
        output.copy_from_slice(input);

        // Recursive split-radix DIF
        let sign_val = T::from_isize(sign.value() as isize);
        split_radix_dif_recursive(output, sign_val);

        // Bit-reversal permutation
        bit_reverse_permute(output);
    }

    /// Split-radix FFT in-place.
    fn execute_split_radix_inplace(&self, data: &mut [Complex<T>], sign: Sign) {
        let sign_val = T::from_isize(sign.value() as isize);
        split_radix_dif_recursive(data, sign_val);

        // Bit-reversal permutation
        bit_reverse_permute(data);
    }
}

/// Recursive split-radix DIF implementation.
///
/// The split-radix algorithm uses a recursive structure:
/// - X[k] for k even uses radix-2 decomposition
/// - X[k] for k odd (specifically k = 4m+1 and k = 4m+3) uses radix-4 decomposition
fn split_radix_dif_recursive<T: Float>(data: &mut [Complex<T>], sign_val: T) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    if n == 2 {
        let t0 = data[0];
        let t1 = data[1];
        data[0] = t0 + t1;
        data[1] = t0 - t1;
        return;
    }

    // For n >= 4, apply split-radix butterflies
    let half_n = n / 2;
    let quarter_n = n / 4;

    // Compute twiddle factors
    let angle_step = sign_val * T::TWO_PI / T::from_usize(n);

    // Split-radix L-shaped butterfly
    for k in 0..quarter_n {
        let angle1 = angle_step * T::from_usize(k);
        let angle3 = angle_step * T::from_usize(3 * k);

        let w1 = Complex::cis(angle1);
        let w3 = Complex::cis(angle3);

        // Indices
        let i0 = k;
        let i1 = k + quarter_n;
        let i2 = k + half_n;
        let i3 = k + half_n + quarter_n;

        let x0 = data[i0];
        let x1 = data[i1];
        let x2 = data[i2];
        let x3 = data[i3];

        // Radix-2 part: even outputs
        let t0 = x0 + x2;
        let t1 = x1 + x3;

        // Radix-4 part: odd outputs with j*sign rotation
        let t2 = x0 - x2;
        let t3 = x1 - x3;

        // Apply +/- j rotation for split-radix
        // For forward: W_4^1 = -j, W_4^3 = j
        // u2 = (t2 - j*t3) * W_n^k
        // u3 = (t2 + j*t3) * W_n^{3k}
        let j_t3 = Complex::new(-sign_val * t3.im, sign_val * t3.re);
        let u2 = (t2 + j_t3) * w1;
        let u3 = (t2 - j_t3) * w3;

        // Store in DIF order
        data[i0] = t0;
        data[i1] = t1;
        data[i2] = u2;
        data[i3] = u3;
    }

    // Recursively process the two halves and two quarters
    // First half: n/2 elements at positions 0..n/2
    split_radix_dif_recursive(&mut data[0..half_n], sign_val);

    // The odd parts are now at positions n/2..n
    // They need to be processed as two n/4 DFTs
    split_radix_dif_recursive(&mut data[half_n..half_n + quarter_n], sign_val);
    split_radix_dif_recursive(&mut data[half_n + quarter_n..n], sign_val);
}

/// Compute bit-reversed index.
#[inline]
fn bit_reverse(mut x: usize, n: usize) -> usize {
    let log_n = n.trailing_zeros() as usize;
    let mut result = 0;

    for _ in 0..log_n {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }

    result
}

/// In-place bit-reversal permutation.
///
/// Uses byte-level lookup table for fast bit-reversal of indices.
fn bit_reverse_permute<T: Float>(data: &mut [Complex<T>]) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    let log_n = n.trailing_zeros() as usize;

    // Byte-reverse lookup table (256 entries)
    static BIT_REV_TABLE: [u8; 256] = {
        let mut table = [0u8; 256];
        let mut i = 0;
        while i < 256 {
            let mut x = i as u8;
            let mut rev = 0u8;
            let mut j = 0;
            while j < 8 {
                rev = (rev << 1) | (x & 1);
                x >>= 1;
                j += 1;
            }
            table[i] = rev;
            i += 1;
        }
        table
    };

    // Fast bit-reverse using lookup table
    let bit_reverse_fast = |mut x: usize, bits: usize| -> usize {
        let mut result: usize = 0;
        let mut remaining_bits = bits;

        while remaining_bits >= 8 {
            result = (result << 8) | (BIT_REV_TABLE[x & 0xFF] as usize);
            x >>= 8;
            remaining_bits -= 8;
        }

        if remaining_bits > 0 {
            result = (result << remaining_bits)
                | ((BIT_REV_TABLE[x & 0xFF] as usize) >> (8 - remaining_bits));
        }

        result
    };

    for i in 0..n {
        let j = bit_reverse_fast(i, log_n);
        if i < j {
            data.swap(i, j);
        }
    }
}

/// Convenience function for forward FFT using DIT.
pub fn fft_radix2<T: Float>(input: &[Complex<T>], output: &mut [Complex<T>]) {
    CooleyTukeySolver::new(CtVariant::Dit).execute(input, output, Sign::Forward);
}

/// Convenience function for inverse FFT using DIT (without normalization).
pub fn ifft_radix2<T: Float>(input: &[Complex<T>], output: &mut [Complex<T>]) {
    CooleyTukeySolver::new(CtVariant::Dit).execute(input, output, Sign::Backward);
}

/// Convenience function for inverse FFT with normalization.
pub fn ifft_radix2_normalized<T: Float>(input: &[Complex<T>], output: &mut [Complex<T>]) {
    CooleyTukeySolver::new(CtVariant::Dit).execute(input, output, Sign::Backward);

    let n = T::from_usize(output.len());
    for x in output.iter_mut() {
        *x = *x / n;
    }
}

/// Convenience function for in-place forward FFT.
pub fn fft_radix2_inplace<T: Float>(data: &mut [Complex<T>]) {
    CooleyTukeySolver::new(CtVariant::Dit).execute_inplace(data, Sign::Forward);
}

/// Convenience function for forward FFT using radix-4.
pub fn fft_radix4<T: Float>(input: &[Complex<T>], output: &mut [Complex<T>]) {
    CooleyTukeySolver::new(CtVariant::DitRadix4).execute(input, output, Sign::Forward);
}

/// Convenience function for in-place forward FFT using radix-4.
pub fn fft_radix4_inplace<T: Float>(data: &mut [Complex<T>]) {
    CooleyTukeySolver::new(CtVariant::DitRadix4).execute_inplace(data, Sign::Forward);
}

/// Convenience function for in-place inverse FFT (without normalization).
pub fn ifft_radix2_inplace<T: Float>(data: &mut [Complex<T>]) {
    CooleyTukeySolver::new(CtVariant::Dit).execute_inplace(data, Sign::Backward);
}

/// Convenience function for forward FFT using radix-8.
pub fn fft_radix8<T: Float>(input: &[Complex<T>], output: &mut [Complex<T>]) {
    CooleyTukeySolver::new(CtVariant::DitRadix8).execute(input, output, Sign::Forward);
}

/// Convenience function for in-place forward FFT using radix-8.
pub fn fft_radix8_inplace<T: Float>(data: &mut [Complex<T>]) {
    CooleyTukeySolver::new(CtVariant::DitRadix8).execute_inplace(data, Sign::Forward);
}

/// Convenience function for forward FFT using split-radix.
pub fn fft_split_radix<T: Float>(input: &[Complex<T>], output: &mut [Complex<T>]) {
    CooleyTukeySolver::new(CtVariant::SplitRadix).execute(input, output, Sign::Forward);
}

/// Convenience function for in-place forward FFT using split-radix.
pub fn fft_split_radix_inplace<T: Float>(data: &mut [Complex<T>]) {
    CooleyTukeySolver::new(CtVariant::SplitRadix).execute_inplace(data, Sign::Forward);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dft::solvers::direct::dft_direct;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
        approx_eq(a.re, b.re, eps) && approx_eq(a.im, b.im, eps)
    }

    #[test]
    fn test_bit_reverse() {
        // For n=8 (log_n=3):
        // 0 (000) -> 0 (000)
        // 1 (001) -> 4 (100)
        // 2 (010) -> 2 (010)
        // 3 (011) -> 6 (110)
        // 4 (100) -> 1 (001)
        // 5 (101) -> 5 (101)
        // 6 (110) -> 3 (011)
        // 7 (111) -> 7 (111)
        assert_eq!(bit_reverse(0, 8), 0);
        assert_eq!(bit_reverse(1, 8), 4);
        assert_eq!(bit_reverse(2, 8), 2);
        assert_eq!(bit_reverse(3, 8), 6);
        assert_eq!(bit_reverse(4, 8), 1);
        assert_eq!(bit_reverse(5, 8), 5);
        assert_eq!(bit_reverse(6, 8), 3);
        assert_eq!(bit_reverse(7, 8), 7);
    }

    #[test]
    fn test_fft_size_2() {
        let input = [Complex::new(1.0_f64, 0.0), Complex::new(2.0, 0.0)];
        let mut output_fft = [Complex::zero(); 2];
        let mut output_direct = [Complex::zero(); 2];

        fft_radix2(&input, &mut output_fft);
        dft_direct(&input, &mut output_direct);

        for (a, b) in output_fft.iter().zip(output_direct.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_fft_size_4() {
        let input = [
            Complex::new(1.0_f64, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut output_fft = [Complex::zero(); 4];
        let mut output_direct = [Complex::zero(); 4];

        fft_radix2(&input, &mut output_fft);
        dft_direct(&input, &mut output_direct);

        for (a, b) in output_fft.iter().zip(output_direct.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_fft_size_8() {
        let input: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new(f64::from(i), f64::from(i) * 0.5))
            .collect();
        let mut output_fft = vec![Complex::zero(); 8];
        let mut output_direct = vec![Complex::zero(); 8];

        fft_radix2(&input, &mut output_fft);
        dft_direct(&input, &mut output_direct);

        for (a, b) in output_fft.iter().zip(output_direct.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_fft_size_16() {
        let input: Vec<Complex<f64>> = (0..16)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut output_fft = vec![Complex::zero(); 16];
        let mut output_direct = vec![Complex::zero(); 16];

        fft_radix2(&input, &mut output_fft);
        dft_direct(&input, &mut output_direct);

        for (a, b) in output_fft.iter().zip(output_direct.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_fft_inverse_recovers_input() {
        let original: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new(f64::from(i), f64::from(i) * 0.5))
            .collect();
        let mut transformed = vec![Complex::zero(); 8];
        let mut recovered = vec![Complex::zero(); 8];

        fft_radix2(&original, &mut transformed);
        ifft_radix2_normalized(&transformed, &mut recovered);

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_fft_inplace_matches_outofplace() {
        let input: Vec<Complex<f64>> = (0..8).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        let mut out_of_place = vec![Complex::zero(); 8];
        fft_radix2(&input, &mut out_of_place);

        let mut in_place = input;
        fft_radix2_inplace(&mut in_place);

        for (a, b) in out_of_place.iter().zip(in_place.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_dif_matches_dit() {
        let input: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new(f64::from(i), f64::from(i) * 0.5))
            .collect();

        let mut output_dit = vec![Complex::zero(); 8];
        let mut output_dif = vec![Complex::zero(); 8];

        CooleyTukeySolver::new(CtVariant::Dit).execute(&input, &mut output_dit, Sign::Forward);
        CooleyTukeySolver::new(CtVariant::Dif).execute(&input, &mut output_dif, Sign::Forward);

        for (a, b) in output_dit.iter().zip(output_dif.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_applicable() {
        assert!(CooleyTukeySolver::<f64>::applicable(1));
        assert!(CooleyTukeySolver::<f64>::applicable(2));
        assert!(CooleyTukeySolver::<f64>::applicable(4));
        assert!(CooleyTukeySolver::<f64>::applicable(8));
        assert!(CooleyTukeySolver::<f64>::applicable(1024));

        assert!(!CooleyTukeySolver::<f64>::applicable(0));
        assert!(!CooleyTukeySolver::<f64>::applicable(3));
        assert!(!CooleyTukeySolver::<f64>::applicable(5));
        assert!(!CooleyTukeySolver::<f64>::applicable(6));
        assert!(!CooleyTukeySolver::<f64>::applicable(7));
    }

    #[test]
    fn test_is_power_of_4() {
        assert!(CooleyTukeySolver::<f64>::is_power_of_4(1));
        assert!(!CooleyTukeySolver::<f64>::is_power_of_4(2));
        assert!(CooleyTukeySolver::<f64>::is_power_of_4(4));
        assert!(!CooleyTukeySolver::<f64>::is_power_of_4(8));
        assert!(CooleyTukeySolver::<f64>::is_power_of_4(16));
        assert!(!CooleyTukeySolver::<f64>::is_power_of_4(32));
        assert!(CooleyTukeySolver::<f64>::is_power_of_4(64));
        assert!(CooleyTukeySolver::<f64>::is_power_of_4(256));
        assert!(CooleyTukeySolver::<f64>::is_power_of_4(1024));

        assert!(!CooleyTukeySolver::<f64>::is_power_of_4(0));
        assert!(!CooleyTukeySolver::<f64>::is_power_of_4(3));
        assert!(!CooleyTukeySolver::<f64>::is_power_of_4(5));
    }

    #[test]
    fn test_radix4_matches_radix2_size_4() {
        let input = [
            Complex::new(1.0_f64, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut output_radix2 = [Complex::zero(); 4];
        let mut output_radix4 = [Complex::zero(); 4];

        fft_radix2(&input, &mut output_radix2);
        fft_radix4(&input, &mut output_radix4);

        for (a, b) in output_radix2.iter().zip(output_radix4.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_radix4_matches_radix2_size_16() {
        let input: Vec<Complex<f64>> = (0..16)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut output_radix2 = vec![Complex::zero(); 16];
        let mut output_radix4 = vec![Complex::zero(); 16];

        fft_radix2(&input, &mut output_radix2);
        fft_radix4(&input, &mut output_radix4);

        for (a, b) in output_radix2.iter().zip(output_radix4.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_radix4_matches_radix2_size_64() {
        let input: Vec<Complex<f64>> = (0..64)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut output_radix2 = vec![Complex::zero(); 64];
        let mut output_radix4 = vec![Complex::zero(); 64];

        fft_radix2(&input, &mut output_radix2);
        fft_radix4(&input, &mut output_radix4);

        for (a, b) in output_radix2.iter().zip(output_radix4.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_radix4_matches_radix2_size_8() {
        // Size 8 = power of 2 but not power of 4 - tests mixed radix-4/radix-2
        let input: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new(f64::from(i), f64::from(i) * 0.5))
            .collect();
        let mut output_radix2 = vec![Complex::zero(); 8];
        let mut output_radix4 = vec![Complex::zero(); 8];

        fft_radix2(&input, &mut output_radix2);
        fft_radix4(&input, &mut output_radix4);

        for (a, b) in output_radix2.iter().zip(output_radix4.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_radix4_inverse_recovers_input() {
        let original: Vec<Complex<f64>> = (0..16)
            .map(|i| Complex::new(f64::from(i), f64::from(i) * 0.5))
            .collect();
        let mut transformed = vec![Complex::zero(); 16];
        let mut recovered = vec![Complex::zero(); 16];

        CooleyTukeySolver::new(CtVariant::DitRadix4).execute(
            &original,
            &mut transformed,
            Sign::Forward,
        );
        CooleyTukeySolver::new(CtVariant::DitRadix4).execute(
            &transformed,
            &mut recovered,
            Sign::Backward,
        );

        // Normalize
        let n = 16.0_f64;
        for x in &mut recovered {
            *x = *x / n;
        }

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_radix4_inplace() {
        let input: Vec<Complex<f64>> = (0..16).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        let mut out_of_place = vec![Complex::zero(); 16];
        fft_radix4(&input, &mut out_of_place);

        let mut in_place = input;
        fft_radix4_inplace(&mut in_place);

        for (a, b) in out_of_place.iter().zip(in_place.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_is_power_of_8() {
        assert!(CooleyTukeySolver::<f64>::is_power_of_8(1));
        assert!(!CooleyTukeySolver::<f64>::is_power_of_8(2));
        assert!(!CooleyTukeySolver::<f64>::is_power_of_8(4));
        assert!(CooleyTukeySolver::<f64>::is_power_of_8(8));
        assert!(!CooleyTukeySolver::<f64>::is_power_of_8(16));
        assert!(!CooleyTukeySolver::<f64>::is_power_of_8(32));
        assert!(CooleyTukeySolver::<f64>::is_power_of_8(64));
        assert!(!CooleyTukeySolver::<f64>::is_power_of_8(128));
        assert!(!CooleyTukeySolver::<f64>::is_power_of_8(256));
        assert!(CooleyTukeySolver::<f64>::is_power_of_8(512));
        assert!(!CooleyTukeySolver::<f64>::is_power_of_8(0));
        assert!(!CooleyTukeySolver::<f64>::is_power_of_8(3));
        assert!(!CooleyTukeySolver::<f64>::is_power_of_8(5));
    }

    #[test]
    fn test_radix8_matches_radix2_size_8() {
        let input: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new(f64::from(i), f64::from(i) * 0.5))
            .collect();
        let mut output_radix2 = vec![Complex::zero(); 8];
        let mut output_radix8 = vec![Complex::zero(); 8];

        fft_radix2(&input, &mut output_radix2);
        fft_radix8(&input, &mut output_radix8);

        for (a, b) in output_radix2.iter().zip(output_radix8.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_radix8_matches_radix2_size_16() {
        // Size 16: log_n = 4, remainder = 1, so one radix-2 + one radix-8 stage
        let input: Vec<Complex<f64>> = (0..16)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut output_radix2 = vec![Complex::zero(); 16];
        let mut output_radix8 = vec![Complex::zero(); 16];

        fft_radix2(&input, &mut output_radix2);
        fft_radix8(&input, &mut output_radix8);

        for (a, b) in output_radix2.iter().zip(output_radix8.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_radix8_matches_radix2_size_32() {
        // Size 32: log_n = 5, remainder = 2, so one radix-4 + one radix-8 stage
        let input: Vec<Complex<f64>> = (0..32)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut output_radix2 = vec![Complex::zero(); 32];
        let mut output_radix8 = vec![Complex::zero(); 32];

        fft_radix2(&input, &mut output_radix2);
        fft_radix8(&input, &mut output_radix8);

        for (a, b) in output_radix2.iter().zip(output_radix8.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_radix8_matches_radix2_size_64() {
        // Size 64: log_n = 6, remainder = 0, so two pure radix-8 stages
        let input: Vec<Complex<f64>> = (0..64)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut output_radix2 = vec![Complex::zero(); 64];
        let mut output_radix8 = vec![Complex::zero(); 64];

        fft_radix2(&input, &mut output_radix2);
        fft_radix8(&input, &mut output_radix8);

        for (a, b) in output_radix2.iter().zip(output_radix8.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_radix8_matches_radix2_size_128() {
        // Size 128: log_n = 7, remainder = 1
        let input: Vec<Complex<f64>> = (0..128)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut output_radix2 = vec![Complex::zero(); 128];
        let mut output_radix8 = vec![Complex::zero(); 128];

        fft_radix2(&input, &mut output_radix2);
        fft_radix8(&input, &mut output_radix8);

        for (a, b) in output_radix2.iter().zip(output_radix8.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-8));
        }
    }

    #[test]
    fn test_radix8_matches_radix2_size_512() {
        // Size 512: log_n = 9, remainder = 0, pure radix-8 (3 stages)
        let input: Vec<Complex<f64>> = (0..512)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut output_radix2 = vec![Complex::zero(); 512];
        let mut output_radix8 = vec![Complex::zero(); 512];

        fft_radix2(&input, &mut output_radix2);
        fft_radix8(&input, &mut output_radix8);

        for (a, b) in output_radix2.iter().zip(output_radix8.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-8));
        }
    }

    #[test]
    fn test_radix8_inverse_recovers_input() {
        let original: Vec<Complex<f64>> = (0..64)
            .map(|i| Complex::new(f64::from(i), f64::from(i) * 0.5))
            .collect();
        let mut transformed = vec![Complex::zero(); 64];
        let mut recovered = vec![Complex::zero(); 64];

        CooleyTukeySolver::new(CtVariant::DitRadix8).execute(
            &original,
            &mut transformed,
            Sign::Forward,
        );
        CooleyTukeySolver::new(CtVariant::DitRadix8).execute(
            &transformed,
            &mut recovered,
            Sign::Backward,
        );

        // Normalize
        let n = 64.0_f64;
        for x in &mut recovered {
            *x = *x / n;
        }

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_radix8_inplace() {
        let input: Vec<Complex<f64>> = (0..64).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        let mut out_of_place = vec![Complex::zero(); 64];
        fft_radix8(&input, &mut out_of_place);

        let mut in_place = input;
        fft_radix8_inplace(&mut in_place);

        for (a, b) in out_of_place.iter().zip(in_place.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_split_radix_matches_radix2_size_4() {
        let input = [
            Complex::new(1.0_f64, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut output_radix2 = [Complex::zero(); 4];
        let mut output_split = [Complex::zero(); 4];

        fft_radix2(&input, &mut output_radix2);
        fft_split_radix(&input, &mut output_split);

        for (a, b) in output_radix2.iter().zip(output_split.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_split_radix_matches_radix2_size_8() {
        let input: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new(f64::from(i), f64::from(i) * 0.5))
            .collect();
        let mut output_radix2 = vec![Complex::zero(); 8];
        let mut output_split = vec![Complex::zero(); 8];

        fft_radix2(&input, &mut output_radix2);
        fft_split_radix(&input, &mut output_split);

        for (a, b) in output_radix2.iter().zip(output_split.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_split_radix_matches_radix2_size_16() {
        let input: Vec<Complex<f64>> = (0..16)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut output_radix2 = vec![Complex::zero(); 16];
        let mut output_split = vec![Complex::zero(); 16];

        fft_radix2(&input, &mut output_radix2);
        fft_split_radix(&input, &mut output_split);

        for (a, b) in output_radix2.iter().zip(output_split.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_split_radix_matches_radix2_size_64() {
        let input: Vec<Complex<f64>> = (0..64)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut output_radix2 = vec![Complex::zero(); 64];
        let mut output_split = vec![Complex::zero(); 64];

        fft_radix2(&input, &mut output_radix2);
        fft_split_radix(&input, &mut output_split);

        for (a, b) in output_radix2.iter().zip(output_split.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_split_radix_matches_radix2_size_256() {
        let input: Vec<Complex<f64>> = (0..256)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();
        let mut output_radix2 = vec![Complex::zero(); 256];
        let mut output_split = vec![Complex::zero(); 256];

        fft_radix2(&input, &mut output_radix2);
        fft_split_radix(&input, &mut output_split);

        for (a, b) in output_radix2.iter().zip(output_split.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-8));
        }
    }

    #[test]
    fn test_split_radix_inverse_recovers_input() {
        let original: Vec<Complex<f64>> = (0..64)
            .map(|i| Complex::new(f64::from(i), f64::from(i) * 0.5))
            .collect();
        let mut transformed = vec![Complex::zero(); 64];
        let mut recovered = vec![Complex::zero(); 64];

        CooleyTukeySolver::new(CtVariant::SplitRadix).execute(
            &original,
            &mut transformed,
            Sign::Forward,
        );
        CooleyTukeySolver::new(CtVariant::SplitRadix).execute(
            &transformed,
            &mut recovered,
            Sign::Backward,
        );

        // Normalize
        let n = 64.0_f64;
        for x in &mut recovered {
            *x = *x / n;
        }

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_split_radix_inplace() {
        let input: Vec<Complex<f64>> = (0..64).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        let mut out_of_place = vec![Complex::zero(); 64];
        fft_split_radix(&input, &mut out_of_place);

        let mut in_place = input;
        fft_split_radix_inplace(&mut in_place);

        for (a, b) in out_of_place.iter().zip(in_place.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }
}
