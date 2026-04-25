//! Compile-time radix-2 Cooley-Tukey FFT.
//!
//! Provides fixed-size FFT implementations for power-of-2 sizes.

use crate::kernel::Complex;
use core::f64::consts::PI;

/// Perform forward FFT on a fixed-size array.
///
/// # Arguments
///
/// * `input` - Input array of complex numbers
///
/// # Returns
///
/// Output array of complex numbers (frequency domain).
#[inline]
pub fn fft_fixed<const N: usize>(input: &[Complex<f64>; N]) -> [Complex<f64>; N] {
    let mut output = *input;
    fft_fixed_inplace(&mut output);
    output
}

/// Perform inverse FFT on a fixed-size array.
///
/// # Arguments
///
/// * `input` - Input array of complex numbers (frequency domain)
///
/// # Returns
///
/// Output array of complex numbers (time domain), normalized.
#[inline]
pub fn ifft_fixed<const N: usize>(input: &[Complex<f64>; N]) -> [Complex<f64>; N] {
    let mut output = *input;
    ifft_fixed_inplace(&mut output);
    output
}

/// Perform in-place forward FFT.
#[inline]
pub fn fft_fixed_inplace<const N: usize>(data: &mut [Complex<f64>; N]) {
    // Handle small sizes specially
    match N {
        0 | 1 => return,
        2 => {
            fft2_inplace(data);
            return;
        }
        4 => {
            fft4_inplace(data);
            return;
        }
        _ => {}
    }

    // Bit-reversal permutation
    bit_reverse_permutation(data);

    // Iterative Cooley-Tukey
    cooley_tukey_iterative(data, false);
}

/// Perform in-place inverse FFT with normalization.
#[inline]
pub fn ifft_fixed_inplace<const N: usize>(data: &mut [Complex<f64>; N]) {
    // Handle small sizes specially
    match N {
        0 | 1 => return,
        2 => {
            ifft2_inplace(data);
            return;
        }
        4 => {
            ifft4_inplace(data);
            return;
        }
        _ => {}
    }

    // Bit-reversal permutation
    bit_reverse_permutation(data);

    // Iterative Cooley-Tukey (inverse)
    cooley_tukey_iterative(data, true);

    // Normalize
    let scale = 1.0 / N as f64;
    for c in data.iter_mut() {
        *c = Complex::new(c.re * scale, c.im * scale);
    }
}

/// Size-2 FFT in-place.
#[inline]
fn fft2_inplace<const N: usize>(data: &mut [Complex<f64>; N]) {
    if N < 2 {
        return;
    }
    let a = data[0];
    let b = data[1];
    data[0] = Complex::new(a.re + b.re, a.im + b.im);
    data[1] = Complex::new(a.re - b.re, a.im - b.im);
}

/// Size-2 IFFT in-place (with normalization).
#[inline]
fn ifft2_inplace<const N: usize>(data: &mut [Complex<f64>; N]) {
    if N < 2 {
        return;
    }
    let a = data[0];
    let b = data[1];
    data[0] = Complex::new((a.re + b.re) * 0.5, (a.im + b.im) * 0.5);
    data[1] = Complex::new((a.re - b.re) * 0.5, (a.im - b.im) * 0.5);
}

/// Size-4 FFT in-place.
#[inline]
fn fft4_inplace<const N: usize>(data: &mut [Complex<f64>; N]) {
    if N < 4 {
        return;
    }

    // Load
    let x0 = data[0];
    let x1 = data[1];
    let x2 = data[2];
    let x3 = data[3];

    // Stage 1: butterflies
    let t0 = Complex::new(x0.re + x2.re, x0.im + x2.im);
    let t1 = Complex::new(x0.re - x2.re, x0.im - x2.im);
    let t2 = Complex::new(x1.re + x3.re, x1.im + x3.im);
    let t3 = Complex::new(x1.re - x3.re, x1.im - x3.im);

    // W_4^1 * t3 = -i * t3 = (t3.im, -t3.re)
    let w_t3 = Complex::new(t3.im, -t3.re);

    // Stage 2: output
    data[0] = Complex::new(t0.re + t2.re, t0.im + t2.im);
    data[1] = Complex::new(t1.re + w_t3.re, t1.im + w_t3.im);
    data[2] = Complex::new(t0.re - t2.re, t0.im - t2.im);
    data[3] = Complex::new(t1.re - w_t3.re, t1.im - w_t3.im);
}

/// Size-4 IFFT in-place (with normalization).
#[inline]
fn ifft4_inplace<const N: usize>(data: &mut [Complex<f64>; N]) {
    if N < 4 {
        return;
    }

    // Load
    let x0 = data[0];
    let x1 = data[1];
    let x2 = data[2];
    let x3 = data[3];

    // Stage 1: butterflies
    let t0 = Complex::new(x0.re + x2.re, x0.im + x2.im);
    let t1 = Complex::new(x0.re - x2.re, x0.im - x2.im);
    let t2 = Complex::new(x1.re + x3.re, x1.im + x3.im);
    let t3 = Complex::new(x1.re - x3.re, x1.im - x3.im);

    // W_4^{-1} * t3 = i * t3 = (-t3.im, t3.re)
    let w_t3 = Complex::new(-t3.im, t3.re);

    // Stage 2: output with normalization
    let scale = 0.25;
    data[0] = Complex::new((t0.re + t2.re) * scale, (t0.im + t2.im) * scale);
    data[1] = Complex::new((t1.re + w_t3.re) * scale, (t1.im + w_t3.im) * scale);
    data[2] = Complex::new((t0.re - t2.re) * scale, (t0.im - t2.im) * scale);
    data[3] = Complex::new((t1.re - w_t3.re) * scale, (t1.im - w_t3.im) * scale);
}

/// Bit-reversal permutation for array of size N.
#[inline]
fn bit_reverse_permutation<const N: usize>(data: &mut [Complex<f64>; N]) {
    let log_n = log2_usize(N);
    if log_n == 0 {
        return;
    }

    for i in 0..N {
        let j = bit_reverse(i, log_n);
        if i < j {
            data.swap(i, j);
        }
    }
}

/// Compute bit-reversal of index.
#[inline]
const fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    let mut i = 0;
    while i < bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
        i += 1;
    }
    result
}

/// Compute log2 of usize at compile time.
#[inline]
const fn log2_usize(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let mut log = 0;
    let mut val = n;
    while val > 1 {
        val >>= 1;
        log += 1;
    }
    log
}

/// Iterative Cooley-Tukey FFT algorithm.
///
/// Assumes data is already bit-reversed.
#[inline]
fn cooley_tukey_iterative<const N: usize>(data: &mut [Complex<f64>; N], inverse: bool) {
    let sign = if inverse { 1.0 } else { -1.0 };

    // Process each stage
    let mut len = 2;
    while len <= N {
        let half_len = len / 2;
        let angle_step = sign * 2.0 * PI / (len as f64);

        // Process each group in this stage
        let mut group_start = 0;
        while group_start < N {
            // Compute twiddle factors for this group
            let mut angle: f64 = 0.0;

            for k in 0..half_len {
                let c = libm::cos(angle);
                let s = libm::sin(angle);

                let i = group_start + k;
                let j = i + half_len;

                // Butterfly
                let u = data[i];
                let t_re = c * data[j].re - s * data[j].im;
                let t_im = c * data[j].im + s * data[j].re;

                data[i] = Complex::new(u.re + t_re, u.im + t_im);
                data[j] = Complex::new(u.re - t_re, u.im - t_im);

                angle += angle_step;
            }

            group_start += len;
        }

        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_reverse() {
        // For N=8 (3 bits):
        // 0 (000) -> 0 (000)
        // 1 (001) -> 4 (100)
        // 2 (010) -> 2 (010)
        // 3 (011) -> 6 (110)
        // 4 (100) -> 1 (001)
        // 5 (101) -> 5 (101)
        // 6 (110) -> 3 (011)
        // 7 (111) -> 7 (111)
        assert_eq!(bit_reverse(0, 3), 0);
        assert_eq!(bit_reverse(1, 3), 4);
        assert_eq!(bit_reverse(2, 3), 2);
        assert_eq!(bit_reverse(3, 3), 6);
        assert_eq!(bit_reverse(4, 3), 1);
        assert_eq!(bit_reverse(5, 3), 5);
        assert_eq!(bit_reverse(6, 3), 3);
        assert_eq!(bit_reverse(7, 3), 7);
    }

    #[test]
    fn test_log2() {
        assert_eq!(log2_usize(1), 0);
        assert_eq!(log2_usize(2), 1);
        assert_eq!(log2_usize(4), 2);
        assert_eq!(log2_usize(8), 3);
        assert_eq!(log2_usize(16), 4);
        assert_eq!(log2_usize(1024), 10);
    }

    #[test]
    fn test_fft2() {
        let input = [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        let output = fft_fixed(&input);

        // DFT of [1, 1]:
        // X[0] = 1 + 1 = 2
        // X[1] = 1 - 1 = 0
        assert!((output[0].re - 2.0).abs() < 1e-10);
        assert!(output[0].im.abs() < 1e-10);
        assert!(output[1].re.abs() < 1e-10);
        assert!(output[1].im.abs() < 1e-10);
    }

    #[test]
    fn test_fft4() {
        let input = [
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let output = fft_fixed(&input);

        // Impulse at 0 -> flat spectrum (all 1s)
        for c in &output {
            assert!((c.re - 1.0).abs() < 1e-10);
            assert!(c.im.abs() < 1e-10);
        }
    }

    #[test]
    fn test_fft8_parseval() {
        // Parseval's theorem: sum|x[n]|² = (1/N) * sum|X[k]|²
        let input: [Complex<f64>; 8] = [
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(7.0, 0.0),
            Complex::new(8.0, 0.0),
        ];

        let output = fft_fixed(&input);

        let time_energy: f64 = input.iter().map(|c| c.re * c.re + c.im * c.im).sum();
        let freq_energy: f64 = output.iter().map(|c| c.re * c.re + c.im * c.im).sum();

        assert!(
            (time_energy - freq_energy / 8.0).abs() < 1e-10,
            "Parseval: {} vs {}",
            time_energy,
            freq_energy / 8.0
        );
    }

    #[test]
    fn test_roundtrip_256() {
        let input: [Complex<f64>; 256] =
            core::array::from_fn(|i| Complex::new((i as f64 * 0.1).sin(), 0.0));

        let spectrum = fft_fixed(&input);
        let recovered = ifft_fixed(&spectrum);

        for i in 0..256 {
            assert!(
                (recovered[i].re - input[i].re).abs() < 1e-10,
                "Real mismatch at {}: {} vs {}",
                i,
                recovered[i].re,
                input[i].re
            );
            assert!(
                recovered[i].im.abs() < 1e-10,
                "Imag should be 0 at {}: {}",
                i,
                recovered[i].im
            );
        }
    }

    #[test]
    fn test_roundtrip_1024() {
        let input: [Complex<f64>; 1024] = core::array::from_fn(|i| {
            Complex::new((i as f64 * 0.01).cos(), (i as f64 * 0.02).sin())
        });

        let spectrum = fft_fixed(&input);
        let recovered = ifft_fixed(&spectrum);

        for i in 0..1024 {
            assert!(
                (recovered[i].re - input[i].re).abs() < 1e-9,
                "Real mismatch at {}: {} vs {}",
                i,
                recovered[i].re,
                input[i].re
            );
            assert!(
                (recovered[i].im - input[i].im).abs() < 1e-9,
                "Imag mismatch at {}: {} vs {}",
                i,
                recovered[i].im,
                input[i].im
            );
        }
    }
}
