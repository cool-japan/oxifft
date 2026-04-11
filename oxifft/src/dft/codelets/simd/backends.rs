//! Architecture-specific SIMD backend codelets.
//!
//! Each submodule contains low-level SIMD implementations gated by target architecture.

// ============================================================================
// SSE2 f64 SIMD codelets
// ============================================================================

#[cfg(target_arch = "x86_64")]
pub mod sse2_f64 {
    use crate::kernel::Complex;
    use crate::simd::{SimdComplex, SimdVector, Sse2F64};

    /// SSE2 Size-2 butterfly for f64.
    ///
    /// Processes one pair of complex numbers using SSE2.
    /// Input: x = [x0, x1] where each is Complex<f64>
    /// Output: [x0+x1, x0-x1]
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn notw_2_sse2(x: &mut [Complex<f64>]) {
        unsafe {
            debug_assert!(x.len() >= 2);

            let ptr = x.as_mut_ptr() as *mut f64;

            // Load x0 = [re0, im0] and x1 = [re1, im1]
            let v0 = Sse2F64::load_unaligned(ptr);
            let v1 = Sse2F64::load_unaligned(ptr.add(2));

            // Butterfly: (v0+v1, v0-v1)
            let (sum, diff) = Sse2F64::butterfly(v0, v1);

            // Store results
            sum.store_unaligned(ptr);
            diff.store_unaligned(ptr.add(2));
        }
    }

    /// SSE2 Size-4 DFT for f64.
    ///
    /// Uses SSE2 intrinsics to compute 4-point DFT.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn notw_4_sse2(x: &mut [Complex<f64>], sign: i32) {
        unsafe {
            debug_assert!(x.len() >= 4);

            let ptr = x.as_mut_ptr() as *mut f64;

            // Load all 4 complex numbers
            let x0 = Sse2F64::load_unaligned(ptr); // [re0, im0]
            let x1 = Sse2F64::load_unaligned(ptr.add(2)); // [re1, im1]
            let x2 = Sse2F64::load_unaligned(ptr.add(4)); // [re2, im2]
            let x3 = Sse2F64::load_unaligned(ptr.add(6)); // [re3, im3]

            // Stage 1: 2 butterflies
            let (t0, t1) = Sse2F64::butterfly(x0, x2); // t0 = x0+x2, t1 = x0-x2
            let (t2, t3) = Sse2F64::butterfly(x1, x3); // t2 = x1+x3, t3 = x1-x3

            // Apply -i or +i rotation to t3
            // For forward (sign < 0): multiply by -i means [re,im] -> [im, -re]
            // For inverse (sign >= 0): multiply by +i means [re,im] -> [-im, re]
            let t3_rot = if sign < 0 {
                // Multiply by -i: [re, im] -> [im, -re]
                t3.swap().negate_high()
            } else {
                // Multiply by +i: [re, im] -> [-im, re]
                t3.swap().negate_low()
            };

            // Stage 2: Final butterflies
            let (y0, y2) = Sse2F64::butterfly(t0, t2); // y0 = t0+t2, y2 = t0-t2
            let (y1, y3) = Sse2F64::butterfly(t1, t3_rot); // y1 = t1+t3_rot, y3 = t1-t3_rot

            // Store results in bit-reversed order
            y0.store_unaligned(ptr); // X[0]
            y1.store_unaligned(ptr.add(2)); // X[1]
            y2.store_unaligned(ptr.add(4)); // X[2]
            y3.store_unaligned(ptr.add(6)); // X[3]
        }
    }

    /// SSE2 Size-8 DFT for f64.
    ///
    /// This matches the scalar notw_8 algorithm exactly:
    /// - Stage 1: 4 radix-2 butterflies
    /// - Stage 2: Apply W8 twiddles to t3, t5, t7, then 4 more butterflies
    /// - Stage 3: Apply W4 twiddles to u3, u7, then final butterflies
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn notw_8_sse2(x: &mut [Complex<f64>], sign: i32) {
        unsafe {
            debug_assert!(x.len() >= 8);

            let ptr = x.as_mut_ptr() as *mut f64;
            let sqrt2_2 = core::f64::consts::FRAC_1_SQRT_2;

            // Load all 8 complex numbers
            let x0 = Sse2F64::load_unaligned(ptr);
            let x1 = Sse2F64::load_unaligned(ptr.add(2));
            let x2 = Sse2F64::load_unaligned(ptr.add(4));
            let x3 = Sse2F64::load_unaligned(ptr.add(6));
            let x4 = Sse2F64::load_unaligned(ptr.add(8));
            let x5 = Sse2F64::load_unaligned(ptr.add(10));
            let x6 = Sse2F64::load_unaligned(ptr.add(12));
            let x7 = Sse2F64::load_unaligned(ptr.add(14));

            // Stage 1: 4 radix-2 butterflies (no twiddles)
            let (t0, t1) = Sse2F64::butterfly(x0, x4);
            let (t2, t3) = Sse2F64::butterfly(x2, x6);
            let (t4, t5) = Sse2F64::butterfly(x1, x5);
            let (t6, t7) = Sse2F64::butterfly(x3, x7);

            // Stage 2: Apply W8 twiddles
            // t3 *= W8^2 = -i (forward) or +i (inverse)
            let t3_rot = if sign < 0 {
                t3.swap().negate_high() // [im, -re] = multiply by -i
            } else {
                t3.swap().negate_low() // [-im, re] = multiply by +i
            };

            // t5 *= W8^1 = (1-i)/sqrt(2) (forward) or (1+i)/sqrt(2) (inverse)
            let t5_rot = {
                let re = t5.low();
                let im = t5.high();
                if sign < 0 {
                    // Complex multiply: (re + i*im) * (1 - i)/sqrt(2)
                    // = (re + im)/sqrt(2) + i*(im - re)/sqrt(2)
                    Sse2F64::new((re + im) * sqrt2_2, (-re + im) * sqrt2_2)
                } else {
                    // Complex multiply: (re + i*im) * (1 + i)/sqrt(2)
                    // = (re - im)/sqrt(2) + i*(re + im)/sqrt(2)
                    Sse2F64::new((re - im) * sqrt2_2, (re + im) * sqrt2_2)
                }
            };

            // t7 *= W8^3 = (-1-i)/sqrt(2) (forward) or (-1+i)/sqrt(2) (inverse)
            let t7_rot = {
                let re = t7.low();
                let im = t7.high();
                if sign < 0 {
                    // Complex multiply: (re + i*im) * (-1 - i)/sqrt(2)
                    // = (-re + im)/sqrt(2) + i*(-re - im)/sqrt(2)
                    Sse2F64::new((-re + im) * sqrt2_2, (-re - im) * sqrt2_2)
                } else {
                    // Complex multiply: (re + i*im) * (-1 + i)/sqrt(2)
                    // = (-re - im)/sqrt(2) + i*(re - im)/sqrt(2)
                    Sse2F64::new((-re - im) * sqrt2_2, (re - im) * sqrt2_2)
                }
            };

            // 4 more butterflies
            let (u0, u1) = Sse2F64::butterfly(t0, t2);
            let (u2, u3) = Sse2F64::butterfly(t4, t6);
            let (u4, u5) = Sse2F64::butterfly(t1, t3_rot);
            let (u6, u7) = Sse2F64::butterfly(t5_rot, t7_rot);

            // Stage 3: Apply W4 twiddles and final butterflies
            // u3 *= W4^1 = -i (forward) or +i (inverse)
            let u3_rot = if sign < 0 {
                u3.swap().negate_high() // multiply by -i
            } else {
                u3.swap().negate_low() // multiply by +i
            };

            // u7 *= W4^1 = -i (forward) or +i (inverse)
            let u7_rot = if sign < 0 {
                u7.swap().negate_high() // multiply by -i
            } else {
                u7.swap().negate_low() // multiply by +i
            };

            // Final outputs
            let (y0, y4) = Sse2F64::butterfly(u0, u2);
            let (y2, y6) = Sse2F64::butterfly(u1, u3_rot);
            let (y1, y5) = Sse2F64::butterfly(u4, u6);
            let (y3, y7) = Sse2F64::butterfly(u5, u7_rot);

            // Store results
            y0.store_unaligned(ptr);
            y1.store_unaligned(ptr.add(2));
            y2.store_unaligned(ptr.add(4));
            y3.store_unaligned(ptr.add(6));
            y4.store_unaligned(ptr.add(8));
            y5.store_unaligned(ptr.add(10));
            y6.store_unaligned(ptr.add(12));
            y7.store_unaligned(ptr.add(14));
        }
    }
}

// ============================================================================
// AVX2 f64 SIMD codelets (processes 2 complex numbers per register)
// ============================================================================

#[cfg(target_arch = "x86_64")]
pub mod avx2_f64 {
    use crate::kernel::Complex;
    use crate::simd::{Avx2F64, SimdVector};

    /// AVX2 Size-4 DFT for f64.
    ///
    /// Uses AVX2 with 256-bit registers to process pairs of complex numbers.
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn notw_4_avx2(x: &mut [Complex<f64>], sign: i32) {
        unsafe {
            debug_assert!(x.len() >= 4);

            let ptr = x.as_mut_ptr() as *mut f64;

            // Load 4 complex numbers (2 per AVX register)
            let x01 = Avx2F64::load_unaligned(ptr); // [re0, im0, re1, im1]
            let x23 = Avx2F64::load_unaligned(ptr.add(4)); // [re2, im2, re3, im3]

            // Extract individual complex numbers using permutation
            // For proper size-4 DFT we need access to x0, x1, x2, x3
            let re0 = x01.extract(0);
            let im0 = x01.extract(1);
            let re1 = x01.extract(2);
            let im1 = x01.extract(3);
            let re2 = x23.extract(0);
            let im2 = x23.extract(1);
            let re3 = x23.extract(2);
            let im3 = x23.extract(3);

            // Stage 1 butterflies
            let t0_re = re0 + re2;
            let t0_im = im0 + im2;
            let t1_re = re0 - re2;
            let t1_im = im0 - im2;
            let t2_re = re1 + re3;
            let t2_im = im1 + im3;
            let t3_re = re1 - re3;
            let t3_im = im1 - im3;

            // Apply -i or +i rotation to t3
            let (t3_rot_re, t3_rot_im) = if sign < 0 {
                (t3_im, -t3_re) // multiply by -i
            } else {
                (-t3_im, t3_re) // multiply by +i
            };

            // Stage 2 butterflies
            let y0_re = t0_re + t2_re;
            let y0_im = t0_im + t2_im;
            let y2_re = t0_re - t2_re;
            let y2_im = t0_im - t2_im;
            let y1_re = t1_re + t3_rot_re;
            let y1_im = t1_im + t3_rot_im;
            let y3_re = t1_re - t3_rot_re;
            let y3_im = t1_im - t3_rot_im;

            // Pack and store results
            let y01 = Avx2F64::new(y0_re, y0_im, y1_re, y1_im);
            let y23 = Avx2F64::new(y2_re, y2_im, y3_re, y3_im);

            y01.store_unaligned(ptr);
            y23.store_unaligned(ptr.add(4));
        }
    }
}

// ============================================================================
// NEON f64 SIMD codelets for aarch64 (Apple Silicon, etc.)
// ============================================================================
// Note: These NEON codelets were benchmarked and found to be slower than scalar
// due to memory copying overhead in even/odd deinterleaving. Kept for potential
// future optimization (e.g., in-place or streaming approaches).

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
pub mod neon_f64 {
    use crate::kernel::Complex;
    use core::arch::aarch64::*;

    /// NEON Size-64 DFT for f64.
    ///
    /// Uses recursive decomposition with SIMD twiddle combination.
    /// Twiddle recurrence eliminates sin/cos calls from the inner loop.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn notw_64_neon(x: &mut [Complex<f64>], sign: i32) {
        debug_assert!(x.len() >= 64);

        // Deinterleave into even and odd arrays
        let mut even = [Complex::<f64>::zero(); 32];
        let mut odd = [Complex::<f64>::zero(); 32];

        for i in 0..32 {
            even[i] = x[2 * i];
            odd[i] = x[2 * i + 1];
        }

        // Apply size-32 DFT to each half (uses SIMD dispatch)
        crate::dft::codelets::simd::notw_32_simd_f64(&mut even, sign);
        crate::dft::codelets::simd::notw_32_simd_f64(&mut odd, sign);

        // Combine with NEON SIMD using twiddle recurrence
        let sign_val = f64::from(sign);
        let angle_step = sign_val * core::f64::consts::PI / 32.0;
        let w_step = Complex::cis(angle_step);

        // Sign pattern for complex multiply: [-1, 1]
        let sign_arr = [-1.0_f64, 1.0];
        let sign_pattern = unsafe { vld1q_f64(sign_arr.as_ptr()) };

        // Start with w = 1 + 0i
        let mut w = Complex::new(1.0, 0.0);

        for k in 0..32 {
            // Load even[k] and odd[k]
            let e_arr = [even[k].re, even[k].im];
            let o_arr = [odd[k].re, odd[k].im];
            let e = unsafe { vld1q_f64(e_arr.as_ptr()) };
            let o = unsafe { vld1q_f64(o_arr.as_ptr()) };

            // Create twiddle vector: [w.re, w.im]
            let tw_arr = [w.re, w.im];
            let tw = unsafe { vld1q_f64(tw_arr.as_ptr()) };
            // Flipped twiddle: [w.im, w.re]
            let tw_flip = vextq_f64(tw, tw, 1);

            // Extract and broadcast o_re and o_im
            let o_re_scalar = vgetq_lane_f64(o, 0);
            let o_im_scalar = vgetq_lane_f64(o, 1);
            let o_re = vdupq_n_f64(o_re_scalar);
            let o_im = vdupq_n_f64(o_im_scalar);

            // Complex multiply: t = odd[k] * w
            let prod1 = vmulq_f64(o_re, tw);
            let prod2 = vmulq_f64(o_im, tw_flip);
            let t = vfmaq_f64(prod1, prod2, sign_pattern);

            // Butterfly: x[k] = e + t, x[k+32] = e - t
            let out_lo = vaddq_f64(e, t);
            let out_hi = vsubq_f64(e, t);

            // Store results
            let mut lo_arr = [0.0_f64; 2];
            let mut hi_arr = [0.0_f64; 2];
            unsafe {
                vst1q_f64(lo_arr.as_mut_ptr(), out_lo);
                vst1q_f64(hi_arr.as_mut_ptr(), out_hi);
            }
            x[k] = Complex::new(lo_arr[0], lo_arr[1]);
            x[k + 32] = Complex::new(hi_arr[0], hi_arr[1]);

            // Advance twiddle using recurrence
            w = w * w_step;
        }
    }

    /// NEON Size-128 DFT for f64.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn notw_128_neon(x: &mut [Complex<f64>], sign: i32) {
        debug_assert!(x.len() >= 128);

        // Deinterleave into even and odd arrays
        let mut even = [Complex::<f64>::zero(); 64];
        let mut odd = [Complex::<f64>::zero(); 64];

        for i in 0..64 {
            even[i] = x[2 * i];
            odd[i] = x[2 * i + 1];
        }

        // Apply size-64 DFT to each half
        unsafe {
            notw_64_neon(&mut even, sign);
            notw_64_neon(&mut odd, sign);
        }

        // Combine with NEON SIMD using twiddle recurrence
        let sign_val = f64::from(sign);
        let angle_step = sign_val * core::f64::consts::PI / 64.0;
        let w_step = Complex::cis(angle_step);

        let sign_arr = [-1.0_f64, 1.0];
        let sign_pattern = unsafe { vld1q_f64(sign_arr.as_ptr()) };

        let mut w = Complex::new(1.0, 0.0);

        for k in 0..64 {
            let e_arr = [even[k].re, even[k].im];
            let o_arr = [odd[k].re, odd[k].im];
            let e = unsafe { vld1q_f64(e_arr.as_ptr()) };
            let o = unsafe { vld1q_f64(o_arr.as_ptr()) };

            let tw_arr = [w.re, w.im];
            let tw = unsafe { vld1q_f64(tw_arr.as_ptr()) };
            let tw_flip = vextq_f64(tw, tw, 1);

            let o_re_scalar = vgetq_lane_f64(o, 0);
            let o_im_scalar = vgetq_lane_f64(o, 1);
            let o_re = vdupq_n_f64(o_re_scalar);
            let o_im = vdupq_n_f64(o_im_scalar);

            let prod1 = vmulq_f64(o_re, tw);
            let prod2 = vmulq_f64(o_im, tw_flip);
            let t = vfmaq_f64(prod1, prod2, sign_pattern);

            let out_lo = vaddq_f64(e, t);
            let out_hi = vsubq_f64(e, t);

            let mut lo_arr = [0.0_f64; 2];
            let mut hi_arr = [0.0_f64; 2];
            unsafe {
                vst1q_f64(lo_arr.as_mut_ptr(), out_lo);
                vst1q_f64(hi_arr.as_mut_ptr(), out_hi);
            }
            x[k] = Complex::new(lo_arr[0], lo_arr[1]);
            x[k + 64] = Complex::new(hi_arr[0], hi_arr[1]);

            w = w * w_step;
        }
    }

    /// NEON Size-256 DFT for f64.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn notw_256_neon(x: &mut [Complex<f64>], sign: i32) {
        debug_assert!(x.len() >= 256);

        // Deinterleave into even and odd arrays
        let mut even = [Complex::<f64>::zero(); 128];
        let mut odd = [Complex::<f64>::zero(); 128];

        for i in 0..128 {
            even[i] = x[2 * i];
            odd[i] = x[2 * i + 1];
        }

        // Apply size-128 DFT to each half
        unsafe {
            notw_128_neon(&mut even, sign);
            notw_128_neon(&mut odd, sign);
        }

        // Combine with NEON SIMD using twiddle recurrence
        let sign_val = f64::from(sign);
        let angle_step = sign_val * core::f64::consts::PI / 128.0;
        let w_step = Complex::cis(angle_step);

        let sign_arr = [-1.0_f64, 1.0];
        let sign_pattern = unsafe { vld1q_f64(sign_arr.as_ptr()) };

        let mut w = Complex::new(1.0, 0.0);

        for k in 0..128 {
            let e_arr = [even[k].re, even[k].im];
            let o_arr = [odd[k].re, odd[k].im];
            let e = unsafe { vld1q_f64(e_arr.as_ptr()) };
            let o = unsafe { vld1q_f64(o_arr.as_ptr()) };

            let tw_arr = [w.re, w.im];
            let tw = unsafe { vld1q_f64(tw_arr.as_ptr()) };
            let tw_flip = vextq_f64(tw, tw, 1);

            let o_re_scalar = vgetq_lane_f64(o, 0);
            let o_im_scalar = vgetq_lane_f64(o, 1);
            let o_re = vdupq_n_f64(o_re_scalar);
            let o_im = vdupq_n_f64(o_im_scalar);

            let prod1 = vmulq_f64(o_re, tw);
            let prod2 = vmulq_f64(o_im, tw_flip);
            let t = vfmaq_f64(prod1, prod2, sign_pattern);

            let out_lo = vaddq_f64(e, t);
            let out_hi = vsubq_f64(e, t);

            let mut lo_arr = [0.0_f64; 2];
            let mut hi_arr = [0.0_f64; 2];
            unsafe {
                vst1q_f64(lo_arr.as_mut_ptr(), out_lo);
                vst1q_f64(hi_arr.as_mut_ptr(), out_hi);
            }
            x[k] = Complex::new(lo_arr[0], lo_arr[1]);
            x[k + 128] = Complex::new(hi_arr[0], hi_arr[1]);

            w = w * w_step;
        }
    }
}

// ============================================================================
// x86_64 f64 SIMD codelets for larger sizes (AVX2)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub mod x86_64_f64 {
    use crate::kernel::Complex;

    /// AVX2 Size-64 DFT for f64.
    ///
    /// Uses recursive decomposition with SIMD twiddle combination.
    #[inline]
    pub fn notw_64_avx2(x: &mut [Complex<f64>], sign: i32) {
        // For now, fall back to scalar - AVX2 implementation can be added later
        crate::dft::codelets::notw_64(x, sign);
    }

    /// AVX2 Size-128 DFT for f64.
    #[inline]
    pub fn notw_128_avx2(x: &mut [Complex<f64>], sign: i32) {
        crate::dft::codelets::notw_128(x, sign);
    }

    /// AVX2 Size-256 DFT for f64.
    #[inline]
    pub fn notw_256_avx2(x: &mut [Complex<f64>], sign: i32) {
        crate::dft::codelets::notw_256(x, sign);
    }
}
