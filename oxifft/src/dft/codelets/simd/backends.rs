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
