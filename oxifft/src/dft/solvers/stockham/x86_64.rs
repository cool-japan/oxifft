//! x86_64-specific Stockham FFT implementations.
//!
//! Contains AVX2 and AVX-512 optimized implementations.

use crate::dft::problem::Sign;
use crate::dft::solvers::simd_butterfly::get_twiddles;
use crate::kernel::Complex;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// AVX2-optimized radix-4 Stockham FFT for f64.
///
/// Uses stage fusion to halve memory passes, achieving radix-4 equivalent performance.
/// Uses precomputed twiddle tables for cache efficiency.
///
/// # Safety
///
/// Caller must ensure the target CPU supports the `avx2` and `fma` features.
/// Calling this function on a CPU that lacks these features causes undefined
/// behavior (illegal instruction trap at runtime).
/// Both `input` and `output` must have the same length, which must be a
/// power of two.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn stockham_radix4_avx2(
    input: &[Complex<f64>],
    output: &mut [Complex<f64>],
    sign: Sign,
) {
    unsafe {
        use core::arch::x86_64::*;

        let n = input.len();
        let log_n = n.trailing_zeros() as usize;

        // For very small sizes, use radix-2
        if n <= 4 {
            stockham_avx2(input, output, sign);
            return;
        }

        let half_n = n / 2;
        let quarter_n = n / 4;

        // Allocate scratch buffer
        let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); n];

        // Calculate total number of "writes to dst" operations
        let num_fused = log_n / 2;
        let has_final = usize::from(log_n % 2 == 1);
        let total_writes = num_fused + has_final;

        // Ping-pong buffer logic
        let (mut src_ptr, mut dst_ptr): (*mut f64, *mut f64) = if total_writes % 2 == 0 {
            output.copy_from_slice(input);
            (
                output.as_mut_ptr() as *mut f64,
                scratch.as_mut_ptr() as *mut f64,
            )
        } else {
            scratch.copy_from_slice(input);
            (
                scratch.as_mut_ptr() as *mut f64,
                output.as_mut_ptr() as *mut f64,
            )
        };

        // Process pairs of stages using stage fusion
        let mut stage = 0;
        let mut m = 1usize; // Half-butterfly span for current stage

        // Get precomputed twiddle table (computed once at program startup)
        let twiddles = get_twiddles();
        let forward = sign == Sign::Forward;

        while stage + 1 < log_n {
            // Fuse stages `stage` and `stage+1`
            let m1 = m;
            let m2 = m * 2;
            let m4 = m * 4;

            // Get base pointers to precomputed twiddle tables
            // Stage s: twiddles for W^k_{2^{s+1}}, k = 0..2^s-1
            // Stage s+1: twiddles for W^k_{2^{s+2}}, k = 0..2^{s+1}-1
            let tw1_base = if forward {
                twiddles.forward[twiddles.offsets[stage]..].as_ptr() as *const f64
            } else {
                twiddles.inverse[twiddles.offsets[stage]..].as_ptr() as *const f64
            };
            let tw2_base = if forward {
                twiddles.forward[twiddles.offsets[stage + 1]..].as_ptr() as *const f64
            } else {
                twiddles.inverse[twiddles.offsets[stage + 1]..].as_ptr() as *const f64
            };

            let src = src_ptr;
            let dst = dst_ptr;
            let num_groups = half_n / m2;

            for g in 0..num_groups {
                let mut j = 0;

                // 4x unrolled loop for better instruction-level parallelism
                while j + 4 <= m1 {
                    // Prefetch next iteration's data
                    if j + 8 <= m1 {
                        let k_next = g * m1 + j + 4;
                        _mm_prefetch(src.add(k_next * 2) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(src.add((k_next + quarter_n) * 2) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(src.add((k_next + half_n) * 2) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(
                            src.add((k_next + half_n + quarter_n) * 2) as *const i8,
                            _MM_HINT_T0,
                        );
                    }

                    // Process elements j, j+1, j+2, j+3 (2 AVX iterations)
                    for iter in 0..2 {
                        let jj = j + iter * 2;
                        let k0 = g * m1 + jj;

                        // Source indices
                        let s0_0 = k0;
                        let s1_0 = k0 + quarter_n;
                        let s2_0 = k0 + half_n;
                        let s3_0 = k0 + half_n + quarter_n;

                        // Destination indices
                        let dst_base0 = g * m4 + jj;

                        // Load pairs of complex values
                        let x0 = _mm256_loadu_pd(src.add(s0_0 * 2));
                        let x1 = _mm256_loadu_pd(src.add(s1_0 * 2));
                        let x2 = _mm256_loadu_pd(src.add(s2_0 * 2));
                        let x3 = _mm256_loadu_pd(src.add(s3_0 * 2));

                        // Load twiddles directly from precomputed table
                        // Each entry is [cos, sin] = 2 f64s, so jj*2 to load 2 consecutive entries
                        let tw1 = _mm256_loadu_pd(tw1_base.add(jj * 2));
                        let tw2_a_vec = _mm256_loadu_pd(tw2_base.add(jj * 2));
                        let tw2_b_vec = _mm256_loadu_pd(tw2_base.add((jj + m1) * 2));

                        // Expand twiddles
                        let tw1_re = _mm256_permute_pd(tw1, 0b0000);
                        let tw1_im = _mm256_permute_pd(tw1, 0b1111);
                        let tw2a_re = _mm256_permute_pd(tw2_a_vec, 0b0000);
                        let tw2a_im = _mm256_permute_pd(tw2_a_vec, 0b1111);
                        let tw2b_re = _mm256_permute_pd(tw2_b_vec, 0b0000);
                        let tw2b_im = _mm256_permute_pd(tw2_b_vec, 0b1111);

                        // Stage s: t2 = x2 * w1, t3 = x3 * w1
                        let x2_re = _mm256_permute_pd(x2, 0b0000);
                        let x2_im = _mm256_permute_pd(x2, 0b1111);
                        let t2_re = _mm256_fnmadd_pd(x2_im, tw1_im, _mm256_mul_pd(x2_re, tw1_re));
                        let t2_im = _mm256_fmadd_pd(x2_im, tw1_re, _mm256_mul_pd(x2_re, tw1_im));
                        let t2 = _mm256_blend_pd(t2_re, t2_im, 0b1010);

                        let x3_re = _mm256_permute_pd(x3, 0b0000);
                        let x3_im = _mm256_permute_pd(x3, 0b1111);
                        let t3_re = _mm256_fnmadd_pd(x3_im, tw1_im, _mm256_mul_pd(x3_re, tw1_re));
                        let t3_im = _mm256_fmadd_pd(x3_im, tw1_re, _mm256_mul_pd(x3_re, tw1_im));
                        let t3 = _mm256_blend_pd(t3_re, t3_im, 0b1010);

                        // a0 = x0 + t2, a1 = x0 - t2, a2 = x1 + t3, a3 = x1 - t3
                        let a0 = _mm256_add_pd(x0, t2);
                        let a1 = _mm256_sub_pd(x0, t2);
                        let a2 = _mm256_add_pd(x1, t3);
                        let a3 = _mm256_sub_pd(x1, t3);

                        // Stage s+1: b2 = a2 * w2_a, b3 = a3 * w2_b
                        let a2_re = _mm256_permute_pd(a2, 0b0000);
                        let a2_im = _mm256_permute_pd(a2, 0b1111);
                        let b2_re = _mm256_fnmadd_pd(a2_im, tw2a_im, _mm256_mul_pd(a2_re, tw2a_re));
                        let b2_im = _mm256_fmadd_pd(a2_im, tw2a_re, _mm256_mul_pd(a2_re, tw2a_im));
                        let b2 = _mm256_blend_pd(b2_re, b2_im, 0b1010);

                        let a3_re = _mm256_permute_pd(a3, 0b0000);
                        let a3_im = _mm256_permute_pd(a3, 0b1111);
                        let b3_re = _mm256_fnmadd_pd(a3_im, tw2b_im, _mm256_mul_pd(a3_re, tw2b_re));
                        let b3_im = _mm256_fmadd_pd(a3_im, tw2b_re, _mm256_mul_pd(a3_re, tw2b_im));
                        let b3 = _mm256_blend_pd(b3_re, b3_im, 0b1010);

                        // Compute outputs
                        let y0 = _mm256_add_pd(a0, b2);
                        let y2 = _mm256_sub_pd(a0, b2);
                        let y1 = _mm256_add_pd(a1, b3);
                        let y3 = _mm256_sub_pd(a1, b3);

                        // Store
                        _mm256_storeu_pd(dst.add(dst_base0 * 2), y0);
                        _mm256_storeu_pd(dst.add((dst_base0 + m1) * 2), y1);
                        _mm256_storeu_pd(dst.add((dst_base0 + m2) * 2), y2);
                        _mm256_storeu_pd(dst.add((dst_base0 + m2 + m1) * 2), y3);
                    }

                    j += 4;
                }

                // Handle remaining elements (0-3)
                while j + 2 <= m1 {
                    let k0 = g * m1 + j;

                    // Source indices
                    let s0_0 = k0;
                    let s1_0 = k0 + quarter_n;
                    let s2_0 = k0 + half_n;
                    let s3_0 = k0 + half_n + quarter_n;

                    // Destination indices
                    let dst_base0 = g * m4 + j;

                    // Load pairs of complex values
                    let x0 = _mm256_loadu_pd(src.add(s0_0 * 2));
                    let x1 = _mm256_loadu_pd(src.add(s1_0 * 2));
                    let x2 = _mm256_loadu_pd(src.add(s2_0 * 2));
                    let x3 = _mm256_loadu_pd(src.add(s3_0 * 2));

                    // Load twiddles directly from precomputed table
                    let tw1 = _mm256_loadu_pd(tw1_base.add(j * 2));
                    let tw2_a_vec = _mm256_loadu_pd(tw2_base.add(j * 2));
                    let tw2_b_vec = _mm256_loadu_pd(tw2_base.add((j + m1) * 2));

                    // Expand twiddles
                    let tw1_re = _mm256_permute_pd(tw1, 0b0000);
                    let tw1_im = _mm256_permute_pd(tw1, 0b1111);
                    let tw2a_re = _mm256_permute_pd(tw2_a_vec, 0b0000);
                    let tw2a_im = _mm256_permute_pd(tw2_a_vec, 0b1111);
                    let tw2b_re = _mm256_permute_pd(tw2_b_vec, 0b0000);
                    let tw2b_im = _mm256_permute_pd(tw2_b_vec, 0b1111);

                    // Stage s: t2 = x2 * w1, t3 = x3 * w1
                    let x2_re = _mm256_permute_pd(x2, 0b0000);
                    let x2_im = _mm256_permute_pd(x2, 0b1111);
                    let t2_re = _mm256_fnmadd_pd(x2_im, tw1_im, _mm256_mul_pd(x2_re, tw1_re));
                    let t2_im = _mm256_fmadd_pd(x2_im, tw1_re, _mm256_mul_pd(x2_re, tw1_im));
                    let t2 = _mm256_blend_pd(t2_re, t2_im, 0b1010);

                    let x3_re = _mm256_permute_pd(x3, 0b0000);
                    let x3_im = _mm256_permute_pd(x3, 0b1111);
                    let t3_re = _mm256_fnmadd_pd(x3_im, tw1_im, _mm256_mul_pd(x3_re, tw1_re));
                    let t3_im = _mm256_fmadd_pd(x3_im, tw1_re, _mm256_mul_pd(x3_re, tw1_im));
                    let t3 = _mm256_blend_pd(t3_re, t3_im, 0b1010);

                    // a0 = x0 + t2, a1 = x0 - t2, a2 = x1 + t3, a3 = x1 - t3
                    let a0 = _mm256_add_pd(x0, t2);
                    let a1 = _mm256_sub_pd(x0, t2);
                    let a2 = _mm256_add_pd(x1, t3);
                    let a3 = _mm256_sub_pd(x1, t3);

                    // Stage s+1: b2 = a2 * w2_a, b3 = a3 * w2_b
                    let a2_re = _mm256_permute_pd(a2, 0b0000);
                    let a2_im = _mm256_permute_pd(a2, 0b1111);
                    let b2_re = _mm256_fnmadd_pd(a2_im, tw2a_im, _mm256_mul_pd(a2_re, tw2a_re));
                    let b2_im = _mm256_fmadd_pd(a2_im, tw2a_re, _mm256_mul_pd(a2_re, tw2a_im));
                    let b2 = _mm256_blend_pd(b2_re, b2_im, 0b1010);

                    let a3_re = _mm256_permute_pd(a3, 0b0000);
                    let a3_im = _mm256_permute_pd(a3, 0b1111);
                    let b3_re = _mm256_fnmadd_pd(a3_im, tw2b_im, _mm256_mul_pd(a3_re, tw2b_re));
                    let b3_im = _mm256_fmadd_pd(a3_im, tw2b_re, _mm256_mul_pd(a3_re, tw2b_im));
                    let b3 = _mm256_blend_pd(b3_re, b3_im, 0b1010);

                    // Compute outputs
                    let y0 = _mm256_add_pd(a0, b2);
                    let y2 = _mm256_sub_pd(a0, b2);
                    let y1 = _mm256_add_pd(a1, b3);
                    let y3 = _mm256_sub_pd(a1, b3);

                    // Store
                    _mm256_storeu_pd(dst.add(dst_base0 * 2), y0);
                    _mm256_storeu_pd(dst.add((dst_base0 + m1) * 2), y1);
                    _mm256_storeu_pd(dst.add((dst_base0 + m2) * 2), y2);
                    _mm256_storeu_pd(dst.add((dst_base0 + m2 + m1) * 2), y3);

                    j += 2;
                }

                // Handle remaining odd element
                while j < m1 {
                    let k = g * m1 + j;
                    let s0 = k;
                    let s1 = k + quarter_n;
                    let s2 = k + half_n;
                    let s3 = k + half_n + quarter_n;

                    let dst_base = g * m4 + j;
                    let d0 = dst_base;
                    let d1 = dst_base + m1;
                    let d2 = dst_base + m2;
                    let d3 = dst_base + m2 + m1;

                    // Load twiddles from precomputed table (stored as [cos, sin] = [re, im])
                    let tw1_entry = *tw1_base.add(j * 2);
                    let tw1_entry_im = *tw1_base.add(j * 2 + 1);
                    let w1 = Complex::new(tw1_entry, tw1_entry_im);

                    let tw2a_entry = *tw2_base.add(j * 2);
                    let tw2a_entry_im = *tw2_base.add(j * 2 + 1);
                    let w2_a = Complex::new(tw2a_entry, tw2a_entry_im);

                    let tw2b_entry = *tw2_base.add((j + m1) * 2);
                    let tw2b_entry_im = *tw2_base.add((j + m1) * 2 + 1);
                    let w2_b = Complex::new(tw2b_entry, tw2b_entry_im);

                    // Load
                    let x0 = Complex::new(*src.add(s0 * 2), *src.add(s0 * 2 + 1));
                    let x1 = Complex::new(*src.add(s1 * 2), *src.add(s1 * 2 + 1));
                    let x2 = Complex::new(*src.add(s2 * 2), *src.add(s2 * 2 + 1));
                    let x3 = Complex::new(*src.add(s3 * 2), *src.add(s3 * 2 + 1));

                    // Stage s
                    let t2 = x2 * w1;
                    let t3 = x3 * w1;
                    let a0 = x0 + t2;
                    let a1 = x0 - t2;
                    let a2 = x1 + t3;
                    let a3 = x1 - t3;

                    // Stage s+1
                    let b2 = a2 * w2_a;
                    let b3 = a3 * w2_b;

                    // Store
                    *dst.add(d0 * 2) = (a0 + b2).re;
                    *dst.add(d0 * 2 + 1) = (a0 + b2).im;
                    *dst.add(d2 * 2) = (a0 - b2).re;
                    *dst.add(d2 * 2 + 1) = (a0 - b2).im;
                    *dst.add(d1 * 2) = (a1 + b3).re;
                    *dst.add(d1 * 2 + 1) = (a1 + b3).im;
                    *dst.add(d3 * 2) = (a1 - b3).re;
                    *dst.add(d3 * 2 + 1) = (a1 - b3).im;

                    j += 1;
                }
            }

            core::mem::swap(&mut src_ptr, &mut dst_ptr);
            stage += 2;
            m *= 4;
        }

        // Handle remaining single stage if log_n is odd
        if stage < log_n {
            let m2 = m * 2;

            // Get base pointer to precomputed twiddles for this stage
            let tw_final_base = if forward {
                twiddles.forward[twiddles.offsets[stage]..].as_ptr() as *const f64
            } else {
                twiddles.inverse[twiddles.offsets[stage]..].as_ptr() as *const f64
            };

            let src = src_ptr;
            let dst = dst_ptr;
            let num_groups = half_n / m;

            for g in 0..num_groups {
                let src_base = g * m;
                let dst_base = g * m2;

                let mut j = 0;
                while j + 2 <= m {
                    // Load twiddles directly from precomputed table
                    let tw = _mm256_loadu_pd(tw_final_base.add(j * 2));
                    let tw_re = _mm256_permute_pd(tw, 0b0000);
                    let tw_im = _mm256_permute_pd(tw, 0b1111);

                    // Load data
                    let u = _mm256_loadu_pd(src.add((src_base + j) * 2));
                    let v = _mm256_loadu_pd(src.add((src_base + j + half_n) * 2));

                    // Complex multiply: t = v * tw
                    let v_re = _mm256_permute_pd(v, 0b0000);
                    let v_im = _mm256_permute_pd(v, 0b1111);
                    let t_re = _mm256_fnmadd_pd(v_im, tw_im, _mm256_mul_pd(v_re, tw_re));
                    let t_im = _mm256_fmadd_pd(v_im, tw_re, _mm256_mul_pd(v_re, tw_im));
                    let t = _mm256_blend_pd(t_re, t_im, 0b1010);

                    // Butterfly
                    _mm256_storeu_pd(dst.add((dst_base + j) * 2), _mm256_add_pd(u, t));
                    _mm256_storeu_pd(dst.add((dst_base + j + m) * 2), _mm256_sub_pd(u, t));

                    j += 2;
                }

                // Handle remaining
                while j < m {
                    let src_u = src_base + j;
                    let src_v = src_u + half_n;
                    let dst_u = dst_base + j;
                    let dst_v = dst_u + m;

                    // Load twiddle from precomputed table
                    let tw_re = *tw_final_base.add(j * 2);
                    let tw_im = *tw_final_base.add(j * 2 + 1);
                    let w = Complex::new(tw_re, tw_im);

                    let u = Complex::new(*src.add(src_u * 2), *src.add(src_u * 2 + 1));
                    let v = Complex::new(*src.add(src_v * 2), *src.add(src_v * 2 + 1));
                    let t = v * w;

                    *dst.add(dst_u * 2) = (u + t).re;
                    *dst.add(dst_u * 2 + 1) = (u + t).im;
                    *dst.add(dst_v * 2) = (u - t).re;
                    *dst.add(dst_v * 2 + 1) = (u - t).im;

                    j += 1;
                }
            }
        }
    }
}

/// AVX-512 optimized radix-4 Stockham FFT for f64.
///
/// Uses 512-bit registers to process 4 complex f64 values at once.
/// Stage fusion halves memory passes for maximum throughput.
///
/// # Safety
///
/// Caller must ensure the target CPU supports the `avx512f` and `avx512dq`
/// features. Calling this function on a CPU that lacks these features causes
/// undefined behavior (illegal instruction trap at runtime).
/// Both `input` and `output` must have the same length, which must be a
/// power of two.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f", enable = "avx512dq")]
pub unsafe fn stockham_radix4_avx512(
    input: &[Complex<f64>],
    output: &mut [Complex<f64>],
    sign: Sign,
) {
    unsafe {
        use core::arch::x86_64::*;

        let n = input.len();
        let log_n = n.trailing_zeros() as usize;

        // For very small sizes, use scalar
        if n <= 4 {
            stockham_small_avx512(input, output, sign);
            return;
        }

        let half_n = n / 2;
        let quarter_n = n / 4;

        // Allocate scratch buffer
        let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); n];

        // Calculate total writes for ping-pong logic
        let num_fused = log_n / 2;
        let has_final = usize::from(log_n % 2 == 1);
        let total_writes = num_fused + has_final;

        let (mut src_ptr, mut dst_ptr): (*mut f64, *mut f64) = if total_writes % 2 == 0 {
            output.copy_from_slice(input);
            (
                output.as_mut_ptr() as *mut f64,
                scratch.as_mut_ptr() as *mut f64,
            )
        } else {
            scratch.copy_from_slice(input);
            (
                scratch.as_mut_ptr() as *mut f64,
                output.as_mut_ptr() as *mut f64,
            )
        };

        // Get precomputed twiddle table (computed once at program startup)
        let twiddles = get_twiddles();
        let forward = sign == Sign::Forward;

        // Fused stage processing (2 stages at a time)
        let mut stage = 0;
        let mut m = 1usize;

        while stage + 1 < log_n {
            let m1 = m;
            let m2 = m * 2;
            let m4 = m * 4;

            // Get base pointers to precomputed twiddle tables
            let tw1_base = if forward {
                twiddles.forward[twiddles.offsets[stage]..].as_ptr() as *const f64
            } else {
                twiddles.inverse[twiddles.offsets[stage]..].as_ptr() as *const f64
            };
            let tw2_base = if forward {
                twiddles.forward[twiddles.offsets[stage + 1]..].as_ptr() as *const f64
            } else {
                twiddles.inverse[twiddles.offsets[stage + 1]..].as_ptr() as *const f64
            };

            let src = src_ptr;
            let dst = dst_ptr;
            let num_groups = half_n / m2;

            for g in 0..num_groups {
                let mut j = 0;

                // Process 4 elements at a time
                while j + 4 <= m1 {
                    for delta in 0..4 {
                        let jj = j + delta;
                        let k = g * m1 + jj;

                        // Load twiddles from precomputed table
                        let tw1_re = *tw1_base.add(jj * 2);
                        let tw1_im = *tw1_base.add(jj * 2 + 1);
                        let tw1 = Complex::new(tw1_re, tw1_im);

                        let tw2a_re = *tw2_base.add(jj * 2);
                        let tw2a_im = *tw2_base.add(jj * 2 + 1);
                        let tw2a = Complex::new(tw2a_re, tw2a_im);

                        let tw2b_re = *tw2_base.add((jj + m1) * 2);
                        let tw2b_im = *tw2_base.add((jj + m1) * 2 + 1);
                        let tw2b = Complex::new(tw2b_re, tw2b_im);

                        let s0 = k;
                        let s1 = k + quarter_n;
                        let s2 = k + half_n;
                        let s3 = k + half_n + quarter_n;

                        let dst_base = g * m4 + jj;

                        // Load
                        let x0 = _mm_loadu_pd(src.add(s0 * 2));
                        let x1 = _mm_loadu_pd(src.add(s1 * 2));
                        let x2 = _mm_loadu_pd(src.add(s2 * 2));
                        let x3 = _mm_loadu_pd(src.add(s3 * 2));

                        // Stage s: multiply x2, x3 by tw1
                        let t2 = avx512_cmul_128(x2, tw1);
                        let t3 = avx512_cmul_128(x3, tw1);

                        let a0 = _mm_add_pd(x0, t2);
                        let a1 = _mm_sub_pd(x0, t2);
                        let a2 = _mm_add_pd(x1, t3);
                        let a3 = _mm_sub_pd(x1, t3);

                        // Stage s+1: multiply a2 by tw2a, a3 by tw2b
                        let b2 = avx512_cmul_128(a2, tw2a);
                        let b3 = avx512_cmul_128(a3, tw2b);

                        // Output
                        _mm_storeu_pd(dst.add(dst_base * 2), _mm_add_pd(a0, b2));
                        _mm_storeu_pd(dst.add((dst_base + m1) * 2), _mm_add_pd(a1, b3));
                        _mm_storeu_pd(dst.add((dst_base + m2) * 2), _mm_sub_pd(a0, b2));
                        _mm_storeu_pd(dst.add((dst_base + m2 + m1) * 2), _mm_sub_pd(a1, b3));
                    }

                    j += 4;
                }

                // Handle remaining elements
                while j < m1 {
                    let k = g * m1 + j;
                    let s0 = k;
                    let s1 = k + quarter_n;
                    let s2 = k + half_n;
                    let s3 = k + half_n + quarter_n;

                    let dst_base = g * m4 + j;

                    // Load twiddles from precomputed table
                    let tw1_re = *tw1_base.add(j * 2);
                    let tw1_im = *tw1_base.add(j * 2 + 1);
                    let w1 = Complex::new(tw1_re, tw1_im);

                    let tw2a_re = *tw2_base.add(j * 2);
                    let tw2a_im = *tw2_base.add(j * 2 + 1);
                    let w2_a = Complex::new(tw2a_re, tw2a_im);

                    let tw2b_re = *tw2_base.add((j + m1) * 2);
                    let tw2b_im = *tw2_base.add((j + m1) * 2 + 1);
                    let w2_b = Complex::new(tw2b_re, tw2b_im);

                    let x0 = Complex::new(*src.add(s0 * 2), *src.add(s0 * 2 + 1));
                    let x1 = Complex::new(*src.add(s1 * 2), *src.add(s1 * 2 + 1));
                    let x2 = Complex::new(*src.add(s2 * 2), *src.add(s2 * 2 + 1));
                    let x3 = Complex::new(*src.add(s3 * 2), *src.add(s3 * 2 + 1));

                    let t2 = x2 * w1;
                    let t3 = x3 * w1;
                    let a0 = x0 + t2;
                    let a1 = x0 - t2;
                    let a2 = x1 + t3;
                    let a3 = x1 - t3;

                    let b2 = a2 * w2_a;
                    let b3 = a3 * w2_b;

                    *dst.add(dst_base * 2) = (a0 + b2).re;
                    *dst.add(dst_base * 2 + 1) = (a0 + b2).im;
                    *dst.add((dst_base + m1) * 2) = (a1 + b3).re;
                    *dst.add((dst_base + m1) * 2 + 1) = (a1 + b3).im;
                    *dst.add((dst_base + m2) * 2) = (a0 - b2).re;
                    *dst.add((dst_base + m2) * 2 + 1) = (a0 - b2).im;
                    *dst.add((dst_base + m2 + m1) * 2) = (a1 - b3).re;
                    *dst.add((dst_base + m2 + m1) * 2 + 1) = (a1 - b3).im;

                    j += 1;
                }
            }

            core::mem::swap(&mut src_ptr, &mut dst_ptr);
            stage += 2;
            m *= 4;
        }

        // Handle remaining single stage if log_n is odd
        if stage < log_n {
            let m2 = m * 2;

            // Get base pointer to precomputed twiddles for this stage
            let tw_final_base = if forward {
                twiddles.forward[twiddles.offsets[stage]..].as_ptr() as *const f64
            } else {
                twiddles.inverse[twiddles.offsets[stage]..].as_ptr() as *const f64
            };

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

                    // Load twiddle from precomputed table
                    let tw_re = *tw_final_base.add(j * 2);
                    let tw_im = *tw_final_base.add(j * 2 + 1);
                    let w = Complex::new(tw_re, tw_im);

                    let u = _mm_loadu_pd(src.add(src_u * 2));
                    let v = _mm_loadu_pd(src.add(src_v * 2));

                    let t = avx512_cmul_128(v, w);

                    _mm_storeu_pd(dst.add(dst_u * 2), _mm_add_pd(u, t));
                    _mm_storeu_pd(dst.add(dst_v * 2), _mm_sub_pd(u, t));
                }
            }
        }
    }
}

/// AVX-512 complex multiply for single complex value (128-bit).
///
/// # Safety
///
/// Caller must ensure the target CPU supports the `avx512f` feature
/// (the `_mm_permute_pd` and `_mm_addsub_pd` intrinsics used internally
/// require at least SSE3/AVX, which is implied by `avx512f`).
#[cfg(feature = "avx512")]
#[inline(always)]
unsafe fn avx512_cmul_128(
    v: core::arch::x86_64::__m128d,
    w: Complex<f64>,
) -> core::arch::x86_64::__m128d {
    unsafe {
        use core::arch::x86_64::*;
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        let tw = _mm_set_pd(w.im, w.re);
        let v_re = _mm_permute_pd(v, 0b00); // [re, re]
        let v_im = _mm_permute_pd(v, 0b11); // [im, im]
        let tw_flip = _mm_permute_pd(tw, 0b01); // [im, re]

        let prod1 = _mm_mul_pd(v_re, tw); // [re*re, re*im]
        let prod2 = _mm_mul_pd(v_im, tw_flip); // [im*im, im*re]

        // Result: [re*re - im*im, re*im + im*re]
        _mm_addsub_pd(prod1, prod2)
    }
}

/// Specialized small-size Stockham for n <= 4 using AVX-512.
///
/// # Safety
///
/// Caller must ensure the target CPU supports the `avx512f` and `avx512dq` features.
/// Calling this function on a CPU that lacks these features causes undefined
/// behavior (illegal instruction trap at runtime).
/// `input` and `output` must have the same length, which must be 1, 2, or 4.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f", enable = "avx512dq")]
unsafe fn stockham_small_avx512(input: &[Complex<f64>], output: &mut [Complex<f64>], sign: Sign) {
    unsafe {
        use core::arch::x86_64::*;
        let n = input.len();
        let sign_val = f64::from(sign.value());

        match n {
            1 => {
                output[0] = input[0];
            }
            2 => {
                let x0 = _mm_loadu_pd(input.as_ptr() as *const f64);
                let x1 = _mm_loadu_pd((input.as_ptr() as *const f64).add(2));
                _mm_storeu_pd(output.as_mut_ptr() as *mut f64, _mm_add_pd(x0, x1));
                _mm_storeu_pd((output.as_mut_ptr() as *mut f64).add(2), _mm_sub_pd(x0, x1));
            }
            4 => {
                // Load all 4 complex values
                let data = _mm512_loadu_pd(input.as_ptr() as *const f64);

                // x0 = [0,1], x1 = [2,3], x2 = [4,5], x3 = [6,7] (indices into f64 array)
                let x0 = _mm512_extractf64x2_pd(data, 0);
                let x1 = _mm512_extractf64x2_pd(data, 1);
                let x2 = _mm512_extractf64x2_pd(data, 2);
                let x3 = _mm512_extractf64x2_pd(data, 3);

                // First stage butterflies
                let a = _mm_add_pd(x0, x2);
                let b = _mm_sub_pd(x0, x2);
                let c = _mm_add_pd(x1, x3);
                let diff = _mm_sub_pd(x1, x3);

                // Rotate diff by ±90°: swap and negate appropriately
                let swapped = _mm_permute_pd(diff, 0b01);
                let d = if sign_val < 0.0 {
                    // * (-i): [re, im] -> [im, -re]
                    _mm_mul_pd(swapped, _mm_set_pd(-1.0, 1.0))
                } else {
                    // * (+i): [re, im] -> [-im, re]
                    _mm_mul_pd(swapped, _mm_set_pd(1.0, -1.0))
                };

                // Output
                _mm_storeu_pd(output.as_mut_ptr() as *mut f64, _mm_add_pd(a, c));
                _mm_storeu_pd((output.as_mut_ptr() as *mut f64).add(2), _mm_add_pd(b, d));
                _mm_storeu_pd((output.as_mut_ptr() as *mut f64).add(4), _mm_sub_pd(a, c));
                _mm_storeu_pd((output.as_mut_ptr() as *mut f64).add(6), _mm_sub_pd(b, d));
            }
            _ => {
                output.copy_from_slice(input);
            }
        }
    }
}

/// AVX2-optimized Stockham FFT for f64 (radix-2, for reference).
///
/// # Safety
///
/// Caller must ensure the target CPU supports the `avx2` and `fma` features.
/// Calling this function on a CPU that lacks these features causes undefined
/// behavior (illegal instruction trap at runtime).
/// Both `input` and `output` must have the same length, which must be a
/// power of two.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn stockham_avx2(input: &[Complex<f64>], output: &mut [Complex<f64>], sign: Sign) {
    unsafe {
        use core::arch::x86_64::*;

        let n = input.len();
        let log_n = n.trailing_zeros() as usize;
        let sign_val = f64::from(sign.value());
        let half_n = n / 2;

        // Allocate scratch buffer
        let mut scratch: Vec<Complex<f64>> = vec![Complex::zero(); n];

        // Copy input to appropriate buffer
        let (mut src_ptr, mut dst_ptr): (*mut Complex<f64>, *mut Complex<f64>) = if log_n % 2 == 0 {
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
            let angle_base = sign_val * core::f64::consts::TAU / (m2 as f64);

            let src = src_ptr as *const f64;
            let dst = dst_ptr as *mut f64;

            // Precompute twiddles
            let mut twiddles: Vec<Complex<f64>> = Vec::with_capacity(m);
            for j in 0..m {
                let angle = angle_base * (j as f64);
                twiddles.push(Complex::cis(angle));
            }

            // Process 2 butterflies at once when possible
            let mut k = 0;
            while k + 1 < half_n {
                let j0 = k % m;
                let g0 = k / m;
                let j1 = (k + 1) % m;
                let g1 = (k + 1) / m;

                let src_u0 = k;
                let src_v0 = k + half_n;
                let src_u1 = k + 1;
                let src_v1 = k + 1 + half_n;

                let dst_u0 = g0 * m2 + j0;
                let dst_v0 = dst_u0 + m;
                let dst_u1 = g1 * m2 + j1;
                let dst_v1 = dst_u1 + m;

                let w0 = twiddles[j0];
                let w1 = twiddles[j1];

                // Load 2 u and 2 v values
                let u0 = _mm_loadu_pd(src.add(src_u0 * 2));
                let u1 = _mm_loadu_pd(src.add(src_u1 * 2));
                let v0 = _mm_loadu_pd(src.add(src_v0 * 2));
                let v1 = _mm_loadu_pd(src.add(src_v1 * 2));

                // Complex multiply v0 * w0
                let v0_re = _mm_shuffle_pd(v0, v0, 0b00);
                let v0_im = _mm_shuffle_pd(v0, v0, 0b11);
                let tw0 = _mm_set_pd(w0.im, w0.re);
                let t0_re = _mm_fnmadd_pd(v0_im, _mm_set_pd(w0.re, w0.im), _mm_mul_pd(v0_re, tw0));
                let t0_im = _mm_fmadd_pd(
                    v0_im,
                    _mm_set_pd(w0.im, w0.re),
                    _mm_mul_pd(v0_re, _mm_set_pd(w0.re, w0.im)),
                );
                let t0 = _mm_blend_pd(t0_re, _mm_shuffle_pd(t0_im, t0_im, 0b01), 0b10);

                // Complex multiply v1 * w1
                let v1_re = _mm_shuffle_pd(v1, v1, 0b00);
                let v1_im = _mm_shuffle_pd(v1, v1, 0b11);
                let t1_re_part = _mm_mul_pd(v1_re, _mm_set_pd(w1.im, w1.re));
                let t1_im_part = _mm_mul_pd(v1_re, _mm_set_pd(w1.re, w1.im));
                let t1_re = _mm_fnmadd_pd(v1_im, _mm_set_pd(w1.re, w1.im), t1_re_part);
                let t1_im = _mm_fmadd_pd(v1_im, _mm_set_pd(w1.im, w1.re), t1_im_part);
                let t1 = _mm_blend_pd(t1_re, _mm_shuffle_pd(t1_im, t1_im, 0b01), 0b10);

                // Butterflies
                let out_u0 = _mm_add_pd(u0, t0);
                let out_v0 = _mm_sub_pd(u0, t0);
                let out_u1 = _mm_add_pd(u1, t1);
                let out_v1 = _mm_sub_pd(u1, t1);

                // Store
                _mm_storeu_pd(dst.add(dst_u0 * 2), out_u0);
                _mm_storeu_pd(dst.add(dst_v0 * 2), out_v0);
                _mm_storeu_pd(dst.add(dst_u1 * 2), out_u1);
                _mm_storeu_pd(dst.add(dst_v1 * 2), out_v1);

                k += 2;
            }

            // Handle remaining odd butterfly
            if k < half_n {
                let j = k % m;
                let g = k / m;

                let src_u = k;
                let src_v = k + half_n;
                let dst_u = g * m2 + j;
                let dst_v = dst_u + m;

                let w = twiddles[j];

                let u = Complex::new(*src.add(src_u * 2), *src.add(src_u * 2 + 1));
                let v = Complex::new(*src.add(src_v * 2), *src.add(src_v * 2 + 1));
                let t = v * w;

                let out_u = u + t;
                let out_v = u - t;

                *dst.add(dst_u * 2) = out_u.re;
                *dst.add(dst_u * 2 + 1) = out_u.im;
                *dst.add(dst_v * 2) = out_v.re;
                *dst.add(dst_v * 2 + 1) = out_v.im;
            }

            core::mem::swap(&mut src_ptr, &mut dst_ptr);
            m *= 2;
        }
    }
}
