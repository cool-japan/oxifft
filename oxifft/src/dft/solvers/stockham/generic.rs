//! Generic and scalar Stockham FFT implementations.
//!
//! Contains:
//! - Generic implementation for any Float type
//! - Scalar f64 implementations (radix-2, stage-fused radix-4)
//! - Specialized small-size kernels (size 2, 4, 8, 16)

use crate::dft::problem::Sign;
use crate::kernel::{Complex, Float};
use crate::prelude::*;

/// Generic Stockham implementation for any Float type.
pub fn stockham_generic<T: Float>(input: &[Complex<T>], output: &mut [Complex<T>], sign: Sign) {
    let n = input.len();
    let log_n = n.trailing_zeros() as usize;
    let sign_val = T::from_isize(sign.value() as isize);
    let half_n = n / 2;

    // Allocate scratch buffer
    let mut scratch: Vec<Complex<T>> = vec![Complex::zero(); n];

    // Copy input to appropriate buffer based on log_n parity
    // After log_n stages, we want result in output buffer
    let (mut src, mut dst) = if log_n.is_multiple_of(2) {
        output.copy_from_slice(input);
        (
            std::ptr::from_mut::<[Complex<T>]>(output),
            std::ptr::from_mut::<[Complex<T>]>(scratch.as_mut_slice()),
        )
    } else {
        scratch.copy_from_slice(input);
        (
            std::ptr::from_mut::<[Complex<T>]>(scratch.as_mut_slice()),
            std::ptr::from_mut::<[Complex<T>]>(output),
        )
    };

    // Process stages using correct Stockham formulation
    let mut m = 1; // Half-butterfly span
    for _ in 0..log_n {
        let m2 = m * 2;
        let angle_step = sign_val * T::TWO_PI / T::from_usize(m2);

        // Safety: src and dst are valid slices
        let src_slice = unsafe { &*src };
        let dst_slice = unsafe { &mut *dst };

        // Precompute twiddles for this stage
        let mut twiddles: Vec<Complex<T>> = Vec::with_capacity(m);
        let mut w = Complex::new(T::ONE, T::ZERO);
        let w_step = Complex::cis(angle_step);
        for _ in 0..m {
            twiddles.push(w);
            w = w * w_step;
        }

        // Process butterflies
        // Key: read at fixed stride n/2, write with pattern based on stage
        for k in 0..half_n {
            let j = k % m; // Position within butterfly pair
            let g = k / m; // Which group

            // Read indices: fixed stride n/2
            let src_u = k;
            let src_v = k + half_n;

            // Write indices: group-based pattern
            let dst_u = g * m2 + j;
            let dst_v = dst_u + m;

            // Butterfly with twiddle
            let u = src_slice[src_u];
            let v = src_slice[src_v] * twiddles[j];

            dst_slice[dst_u] = u + v;
            dst_slice[dst_v] = u - v;
        }

        // Swap buffers
        core::mem::swap(&mut src, &mut dst);
        m *= 2;
    }
}

/// Stage-fused Stockham implementation for f64 (scalar).
///
/// Fuses pairs of radix-2 stages to halve memory passes, achieving radix-4 equivalent performance.
#[allow(dead_code)]
pub fn stockham_radix4_scalar(input: &[Complex<f64>], output: &mut [Complex<f64>], sign: Sign) {
    // Use stage fusion for correctness and performance
    stockham_radix4_scalar_wip(input, output, sign);
}

/// Stockham FFT with stage fusion (scalar) - WORK IN PROGRESS.
///
/// Fuses pairs of radix-2 stages together to reduce memory passes by half.
/// This is NOT true radix-4, but combines two consecutive radix-2 stages.
#[allow(dead_code)]
fn stockham_radix4_scalar_wip(input: &[Complex<f64>], output: &mut [Complex<f64>], sign: Sign) {
    let n = input.len();
    let log_n = n.trailing_zeros() as usize;
    let sign_val = f64::from(sign.value());

    // Specialized small-size kernels (fully unrolled, no loops)
    match n {
        1 => {
            output[0] = input[0];
            return;
        }
        2 => {
            stockham_size2(input, output);
            return;
        }
        4 => {
            stockham_size4(input, output, sign);
            return;
        }
        8 => {
            stockham_size8(input, output, sign);
            return;
        }
        16 => {
            stockham_size16(input, output, sign);
            return;
        }
        _ => {}
    }

    let half_n = n / 2;

    // Allocate scratch buffer
    let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); n];

    // Calculate total number of "writes to dst" operations
    // Each fused pair counts as 1 write, final radix-2 stage counts as 1 write
    let num_fused = log_n / 2;
    let has_final = usize::from(log_n % 2 == 1);
    let total_writes = num_fused + has_final;

    // Ping-pong buffer logic:
    // - If total_writes is odd: final result in dst → start with dst=output
    // - If total_writes is even: final result in src after last swap → start with src=output
    let (mut src_ptr, mut dst_ptr): (*mut Complex<f64>, *mut Complex<f64>) =
        if total_writes.is_multiple_of(2) {
            output.copy_from_slice(input);
            (output.as_mut_ptr(), scratch.as_mut_ptr())
        } else {
            scratch.copy_from_slice(input);
            (scratch.as_mut_ptr(), output.as_mut_ptr())
        };

    // Process pairs of stages using stage fusion
    let mut stage = 0;
    let mut m = 1; // Half-butterfly span for current stage

    while stage + 1 < log_n {
        // Fuse stages `stage` and `stage+1`
        let m1 = m; // m for stage s
        let m2 = m * 2; // m for stage s+1
        let m4 = m * 4; // output span after both stages

        // Twiddle angles for both stages
        let angle_step1 = sign_val * core::f64::consts::TAU / (m2 as f64);
        let angle_step2 = sign_val * core::f64::consts::TAU / (m4 as f64);

        let src = src_ptr;
        let dst = dst_ptr;

        let num_groups = half_n / m2;

        let quarter_n = n / 4;

        for g in 0..num_groups {
            // For each position j in [0, m1), we process 4 elements
            for j in 0..m1 {
                // Source indices (stride n/4 for radix-4 equivalent)
                let k = g * m1 + j;
                let s0 = k;
                let s1 = k + quarter_n;
                let s2 = k + half_n;
                let s3 = k + half_n + quarter_n;

                // Destination indices after fused stages
                let dst_base = g * m4 + j;
                let d0 = dst_base;
                let d1 = dst_base + m1;
                let d2 = dst_base + m2;
                let d3 = dst_base + m2 + m1;

                // Load 4 inputs
                let x0 = unsafe { *src.add(s0) };
                let x1 = unsafe { *src.add(s1) };
                let x2 = unsafe { *src.add(s2) };
                let x3 = unsafe { *src.add(s3) };

                // Twiddles for stage s (applied to x1, x3)
                let angle1 = angle_step1 * (j as f64);
                let w1 = Complex::cis(angle1);

                // Apply stage s: pairs (x0,x2) and (x1,x3)
                let t2 = x2 * w1;
                let t3 = x3 * w1;
                let a0 = x0 + t2;
                let a1 = x0 - t2;
                let a2 = x1 + t3;
                let a3 = x1 - t3;

                // Twiddles for stage s+1
                // Pair (a0, a2) at position j uses W^j
                // Pair (a1, a3) at position j+m1 uses W^(j+m1)
                let angle2_a = angle_step2 * (j as f64);
                let angle2_b = angle_step2 * ((j + m1) as f64);
                let w2_a = Complex::cis(angle2_a);
                let w2_b = Complex::cis(angle2_b);

                // Apply stage s+1
                let b2 = a2 * w2_a;
                let b3 = a3 * w2_b;

                unsafe {
                    *dst.add(d0) = a0 + b2;
                    *dst.add(d2) = a0 - b2;
                    *dst.add(d1) = a1 + b3;
                    *dst.add(d3) = a1 - b3;
                }
            }
        }

        core::mem::swap(&mut src_ptr, &mut dst_ptr);
        stage += 2;
        m *= 4;
    }

    // Handle remaining single stage if log_n is odd
    if stage < log_n {
        let m2 = m * 2;
        let angle_step = sign_val * core::f64::consts::TAU / (m2 as f64);

        let src = src_ptr;
        let dst = dst_ptr;
        let num_groups = half_n / m;

        for g in 0..num_groups {
            let src_base = g * m;
            let dst_base = g * m2;

            for j in 0..m {
                let src_u = src_base + j;
                let src_v = src_u + half_n;
                let dst_u = dst_base + j;
                let dst_v = dst_u + m;

                let u = unsafe { *src.add(src_u) };
                let v = unsafe { *src.add(src_v) };

                let angle = angle_step * (j as f64);
                let w = Complex::cis(angle);
                let t = v * w;

                unsafe {
                    *dst.add(dst_u) = u + t;
                    *dst.add(dst_v) = u - t;
                }
            }
        }
    }
}

/// Scalar Stockham implementation for f64 (radix-2, for reference/testing).
///
/// Uses pre-computed twiddle factors per stage for efficiency.
#[allow(dead_code)]
pub fn stockham_scalar(input: &[Complex<f64>], output: &mut [Complex<f64>], sign: Sign) {
    let n = input.len();
    let log_n = n.trailing_zeros() as usize;
    let sign_val = f64::from(sign.value());
    let half_n = n / 2;

    // Allocate scratch buffer
    let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); n];

    // Pre-allocate twiddle buffer (reused across stages)
    let mut twiddles: Vec<Complex<f64>> = Vec::with_capacity(half_n);

    // Copy input to appropriate buffer
    let (mut src_ptr, mut dst_ptr): (*mut Complex<f64>, *mut Complex<f64>) =
        if log_n.is_multiple_of(2) {
            output.copy_from_slice(input);
            (output.as_mut_ptr(), scratch.as_mut_ptr())
        } else {
            scratch.copy_from_slice(input);
            (scratch.as_mut_ptr(), output.as_mut_ptr())
        };

    // Process stages
    let mut m = 1;
    for _ in 0..log_n {
        let m2 = m * 2;
        let angle_step = sign_val * core::f64::consts::TAU / (m2 as f64);
        let w_step = Complex::cis(angle_step);

        // Pre-compute twiddles for this stage using recurrence
        twiddles.clear();
        let mut w = Complex::new(1.0, 0.0);
        for _ in 0..m {
            twiddles.push(w);
            w = w * w_step;
        }

        let src = src_ptr;
        let dst = dst_ptr;
        let num_groups = half_n / m;

        // Process each group
        for g in 0..num_groups {
            let src_base = g * m;
            let dst_base = g * m2;

            for j in 0..m {
                let src_u = src_base + j;
                let src_v = src_u + half_n;
                let dst_u = dst_base + j;
                let dst_v = dst_u + m;

                // Butterfly with pre-computed twiddle
                let u = unsafe { *src.add(src_u) };
                let v = unsafe { *src.add(src_v) } * twiddles[j];

                unsafe {
                    *dst.add(dst_u) = u + v;
                    *dst.add(dst_v) = u - v;
                }
            }
        }

        core::mem::swap(&mut src_ptr, &mut dst_ptr);
        m *= 2;
    }
}

// ============================================================================
// Specialized small-size kernels (fully unrolled, no loops)
// ============================================================================

/// Size-2 FFT: single butterfly, no twiddle factors needed.
#[allow(clippy::inline_always, dead_code)]
#[inline(always)]
pub fn stockham_size2(input: &[Complex<f64>], output: &mut [Complex<f64>]) {
    let x0 = input[0];
    let x1 = input[1];
    output[0] = x0 + x1;
    output[1] = x0 - x1;
}

/// Size-4 FFT: radix-4 butterfly with ±i rotation.
#[allow(clippy::inline_always, dead_code)]
#[inline(always)]
pub fn stockham_size4(input: &[Complex<f64>], output: &mut [Complex<f64>], sign: Sign) {
    let x0 = input[0];
    let x1 = input[1];
    let x2 = input[2];
    let x3 = input[3];

    // First stage butterflies
    let a = x0 + x2;
    let b = x0 - x2;
    let c = x1 + x3;
    let diff = x1 - x3;

    // Rotate diff by ±90°
    let d = if sign.value() < 0 {
        // * (-i): [re, im] -> [im, -re]
        Complex::new(diff.im, -diff.re)
    } else {
        // * (+i): [re, im] -> [-im, re]
        Complex::new(-diff.im, diff.re)
    };

    output[0] = a + c;
    output[1] = b + d;
    output[2] = a - c;
    output[3] = b - d;
}

/// Size-8 FFT: Use 2 size-4 FFTs with twiddles (Cooley-Tukey decimation-in-time).
#[allow(clippy::inline_always, dead_code)]
#[inline(always)]
pub fn stockham_size8(input: &[Complex<f64>], output: &mut [Complex<f64>], sign: Sign) {
    let sign_val = f64::from(sign.value());

    // Split into even and odd indices
    let even = [input[0], input[2], input[4], input[6]];
    let odd = [input[1], input[3], input[5], input[7]];

    // Compute size-4 FFTs
    let mut even_out = [Complex::zero(); 4];
    let mut odd_out = [Complex::zero(); 4];
    stockham_size4(&even, &mut even_out, sign);
    stockham_size4(&odd, &mut odd_out, sign);

    // Combine with twiddle factors W_8^k
    for k in 0..4 {
        let angle = sign_val * core::f64::consts::TAU * (k as f64) / 8.0;
        let w = Complex::cis(angle);
        let t = odd_out[k] * w;

        output[k] = even_out[k] + t;
        output[k + 4] = even_out[k] - t;
    }
}

/// Size-16 FFT: Use 2 size-8 FFTs with twiddles.
#[allow(clippy::inline_always, dead_code)]
#[inline(always)]
pub fn stockham_size16(input: &[Complex<f64>], output: &mut [Complex<f64>], sign: Sign) {
    let sign_val = f64::from(sign.value());

    // Compute two size-8 sub-FFTs
    let mut even = [Complex::zero(); 8];
    let mut odd = [Complex::zero(); 8];

    // Split into even and odd indices
    for i in 0..8 {
        even[i] = input[2 * i];
        odd[i] = input[2 * i + 1];
    }

    // Transform even and odd parts using size-8 kernel
    let mut even_out = [Complex::zero(); 8];
    let mut odd_out = [Complex::zero(); 8];
    stockham_size8(&even, &mut even_out, sign);
    stockham_size8(&odd, &mut odd_out, sign);

    // Combine with twiddle factors W_16^k
    for k in 0..8 {
        let angle = sign_val * core::f64::consts::TAU * (k as f64) / 16.0;
        let w = Complex::cis(angle);
        let t = odd_out[k] * w;

        output[k] = even_out[k] + t;
        output[k + 8] = even_out[k] - t;
    }
}
