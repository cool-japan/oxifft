//! Pure-AVX (non-AVX2, non-FMA) codelet emitters (`x86_64`, 256-bit f64).
//!
//! These emitters use ONLY AVX instructions available without AVX2 or FMA:
//! - `_mm256_add_pd`, `_mm256_sub_pd`, `_mm256_mul_pd`
//! - `_mm256_loadu_pd`, `_mm256_storeu_pd`
//! - `_mm256_permute_pd`, `_mm256_permute2f128_pd`
//! - `_mm256_unpacklo_pd`, `_mm256_unpackhi_pd`
//! - `_mm256_blend_pd`, `_mm256_blendv_pd`
//!
//! Complex multiply `(a+bi)(c+di)`:
//! - real part: `sub(mul(a_re, b_re), mul(a_im, b_im))`
//! - imag part: `add(mul(a_re, b_im), mul(a_im, b_re))`
//!
//! No FMA, no AVX2 integer intrinsics. All functions are gated with
//! `#[target_feature(enable = "avx")]`.

use proc_macro2::TokenStream;
use quote::quote;

/// Pure-AVX size-2 butterfly on f64 data.
///
/// Loads both complexes as a single 256-bit vector, splits into 128-bit halves,
/// butterflies, and repacks via `_mm256_permute2f128_pd` (no AVX2 needed).
///
/// Note: `_mm256_permute2f128_pd(a, b, 0x20)` = [a.lo, b.lo],
/// semantically equivalent to `_mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 1)`
/// but uses only the pure-AVX lane-permute instruction (3c latency vs 3c, same cost).
pub(super) fn gen_avx_size_2_f64() -> TokenStream {
    quote! {
        /// Size-2 butterfly using pure AVX intrinsics for f64 data.
        ///
        /// Uses only AVX instructions (no AVX2, no FMA). Each YMM register holds
        /// two complex f64 values as `[re0, im0, re1, im1]`.
        ///
        /// Butterfly: out[0] = a + b, out[1] = a - b.
        ///
        /// # Safety
        /// - Caller must verify AVX (but not necessarily AVX2/FMA) is available.
        /// - `data` must contain at least 4 f64 elements (2 complex numbers).
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx")]
        unsafe fn codelet_simd_2_avx_f64(data: &mut [f64], _sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();
            // Load [re0, im0, re1, im1] into one YMM register
            let v = _mm256_loadu_pd(ptr);

            // Extract 128-bit halves using pure-AVX instructions
            let a = _mm256_castpd256_pd128(v);        // [re0, im0]
            let b = _mm256_extractf128_pd(v, 1);      // [re1, im1]

            // Radix-2 butterfly: out0 = a+b, out1 = a-b
            let sum  = _mm_add_pd(a, b);
            let diff = _mm_sub_pd(a, b);

            // Repack using _mm256_permute2f128_pd: imm=0x20 → [src1.lo, src2.lo]
            // Equivalent to insertf128_pd but uses only pure-AVX permute.
            // _mm256_permute2f128_pd(A, B, 0x20) = [A.lo128, B.lo128]
            // latency: permute2f128_pd = 3c (same as insertf128_pd on Haswell)
            let sum_wide  = _mm256_castpd128_pd256(sum);   // sum in YMM lo
            let diff_wide = _mm256_castpd128_pd256(diff);  // diff in YMM lo
            let result = _mm256_permute2f128_pd(sum_wide, diff_wide, 0x20); // [sum.lo, diff.lo]
            _mm256_storeu_pd(ptr, result);
        }
    }
}

/// Pure-AVX size-4 radix-4 butterfly on f64 data.
///
/// Uses 256-bit loads and pure-AVX permutation to perform the two butterfly stages.
/// The ±i rotation on t3 is done via scalar lane-level shuffle (`_mm_shuffle_pd`)
/// and sign-flip (`_mm_xor_pd`) — these are SSE2 instructions available on any AVX host.
pub(super) fn gen_avx_size_4_f64() -> TokenStream {
    quote! {
        /// Size-4 radix-4 FFT using pure AVX intrinsics for f64 data.
        ///
        /// Uses only AVX instructions (no AVX2, no FMA). Two YMM registers hold
        /// the four complex f64 inputs: v_01=[re0,im0,re1,im1], v_23=[re2,im2,re3,im3].
        ///
        /// Stage 1: pairwise butterflies (x0±x2, x1±x3).
        /// Stage 2: final butterflies with ±i rotation on t3.
        ///
        /// # Safety
        /// - Caller must verify AVX is available.
        /// - `data` must contain at least 8 f64 elements.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx")]
        unsafe fn codelet_simd_4_avx_f64(data: &mut [f64], sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();

            // Load two pairs of complexes as 256-bit vectors
            let v_01 = _mm256_loadu_pd(ptr);          // [re0, im0, re1, im1]
            let v_23 = _mm256_loadu_pd(ptr.add(4));   // [re2, im2, re3, im3]

            // Stage 1: pairwise butterflies (x0±x2 and x1±x3 simultaneously)
            let sum  = _mm256_add_pd(v_01, v_23); // [t0_re, t0_im, t2_re, t2_im]
            let diff = _mm256_sub_pd(v_01, v_23); // [t1_re, t1_im, t3_re, t3_im]

            // Extract 128-bit complexes for stage 2
            let t0 = _mm256_castpd256_pd128(sum);       // [t0_re, t0_im]
            let t2 = _mm256_extractf128_pd(sum, 1);     // [t2_re, t2_im]
            let t1 = _mm256_castpd256_pd128(diff);      // [t1_re, t1_im]
            let t3 = _mm256_extractf128_pd(diff, 1);    // [t3_re, t3_im]

            // Rotate t3 by ±i: swap re↔im, then negate one lane
            // _mm_shuffle_pd(a,a,0b01): [lo,hi] → [hi,lo] (single SSE2 instruction, 1c)
            let t3_swapped = _mm_shuffle_pd(t3, t3, 0b01); // [t3_im, t3_re]
            let t3_rot = if sign < 0 {
                // Forward DFT: multiply by -i → [t3_im, -t3_re]
                _mm_xor_pd(t3_swapped, _mm_set_pd(-0.0, 0.0))
            } else {
                // Inverse DFT: multiply by +i → [-t3_im, t3_re]
                _mm_xor_pd(t3_swapped, _mm_set_pd(0.0, -0.0))
            };

            // Stage 2: final butterflies
            let out0 = _mm_add_pd(t0, t2);       // t0 + t2
            let out1 = _mm_add_pd(t1, t3_rot);   // t1 + t3_rot
            let out2 = _mm_sub_pd(t0, t2);       // t0 - t2
            let out3 = _mm_sub_pd(t1, t3_rot);   // t1 - t3_rot

            // Repack pairs into YMM using pure-AVX permute2f128 (avoids AVX2 insertf128 chaining)
            // _mm256_permute2f128_pd(A, B, 0x20) = [A.lo128, B.lo128] — 3c latency (Haswell)
            let v_out_01 = _mm256_permute2f128_pd(
                _mm256_castpd128_pd256(out0),
                _mm256_castpd128_pd256(out1),
                0x20, // [out0, out1]
            );
            let v_out_23 = _mm256_permute2f128_pd(
                _mm256_castpd128_pd256(out2),
                _mm256_castpd128_pd256(out3),
                0x20, // [out2, out3]
            );
            _mm256_storeu_pd(ptr,        v_out_01);
            _mm256_storeu_pd(ptr.add(4), v_out_23);
        }
    }
}

/// Pure-AVX size-8 radix-2 DIT butterfly on f64 data.
///
/// Mirrors the AVX2 structure but uses only pure-AVX intrinsics.
/// Complex twiddle multiply at stage 3 is computed without FMA:
/// - real part: `sub(mul(v_re, c), mul(v_im, d))`
/// - imag part: `add(mul(v_re, d), mul(v_im, c))`
///
/// The bit-reversal load order and 3-stage DIT structure are identical to
/// the AVX2 emitter. Store uses `_mm256_permute2f128_pd` instead of
/// `_mm256_insertf128_pd`.
#[allow(clippy::too_many_lines)]
pub(super) fn gen_avx_size_8_f64() -> TokenStream {
    quote! {
        /// Size-8 FFT using pure AVX intrinsics for f64 data (radix-2 DIT, 3 stages).
        ///
        /// Uses only AVX instructions (no AVX2, no FMA). Complex twiddle multiply
        /// for W8 twiddles uses explicit sub(mul,mul) + add(mul,mul) without FMA.
        ///
        /// # Safety
        /// - Caller must verify AVX is available.
        /// - `data` must contain at least 16 f64 elements.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx")]
        #[allow(clippy::too_many_lines)]
        unsafe fn codelet_simd_8_avx_f64(data: &mut [f64], sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();
            let inv_sqrt2 = _mm_set1_pd(0.707_106_781_186_547_6_f64);

            // Helper: rotate complex by ±i in SSE2 register [re, im]
            let rotate_pm_i = |v: __m128d, fwd: bool| -> __m128d {
                let swapped = _mm_shuffle_pd(v, v, 0b01);
                if fwd {
                    _mm_xor_pd(swapped, _mm_set_pd(-0.0, 0.0))
                } else {
                    _mm_xor_pd(swapped, _mm_set_pd(0.0, -0.0))
                }
            };

            // Helper: complex multiply v * (c + id) without FMA
            // real = v_re*c - v_im*d, imag = v_re*d + v_im*c
            let cmul_no_fma = |v: __m128d, twd: __m128d| -> __m128d {
                let c = _mm_shuffle_pd(twd, twd, 0b00); // broadcast re of twd
                let d = _mm_shuffle_pd(twd, twd, 0b11); // broadcast im of twd
                let v_re = _mm_shuffle_pd(v, v, 0b00);  // broadcast re of v
                let v_im = _mm_shuffle_pd(v, v, 0b11);  // broadcast im of v
                // real = v_re*c - v_im*d  (two muls + one sub, no FMA)
                let real = _mm_sub_pd(_mm_mul_pd(v_re, c), _mm_mul_pd(v_im, d));
                // imag = v_re*d + v_im*c  (two muls + one add, no FMA)
                let imag = _mm_add_pd(_mm_mul_pd(v_re, d), _mm_mul_pd(v_im, c));
                _mm_shuffle_pd(real, imag, 0b00) // [real.lo, imag.lo]
            };

            let fwd = sign < 0;

            // Bit-reversal load for N=8: indices [0,4,2,6,1,5,3,7]
            let mut a = [_mm_setzero_pd(); 8];
            a[0] = _mm_loadu_pd(ptr);           // x[0]
            a[1] = _mm_loadu_pd(ptr.add(8));    // x[4]
            a[2] = _mm_loadu_pd(ptr.add(4));    // x[2]
            a[3] = _mm_loadu_pd(ptr.add(12));   // x[6]
            a[4] = _mm_loadu_pd(ptr.add(2));    // x[1]
            a[5] = _mm_loadu_pd(ptr.add(10));   // x[5]
            a[6] = _mm_loadu_pd(ptr.add(6));    // x[3]
            a[7] = _mm_loadu_pd(ptr.add(14));   // x[7]

            // Stage 1: 4 span-1 butterflies (trivial twiddle W_2^0 = 1)
            for i in (0..8usize).step_by(2) {
                let t = a[i + 1];
                a[i + 1] = _mm_sub_pd(a[i], t);
                a[i]     = _mm_add_pd(a[i], t);
            }

            // Stage 2: 2 groups, span-2, W4 twiddles (W4^0=1, W4^1=∓i)
            for group in (0..8usize).step_by(4) {
                // k=0: twiddle = 1
                let t = a[group + 2];
                a[group + 2] = _mm_sub_pd(a[group], t);
                a[group]     = _mm_add_pd(a[group], t);

                // k=1: twiddle = W4^1 = ∓i
                let t = a[group + 3];
                let t_tw = rotate_pm_i(t, fwd);
                a[group + 3] = _mm_sub_pd(a[group + 1], t_tw);
                a[group + 1] = _mm_add_pd(a[group + 1], t_tw);
            }

            // Stage 3: 1 group, span-4, W8 twiddles (k=0..3)
            // k=0: W8^0 = 1 (trivial)
            {
                let t = a[4];
                a[4] = _mm_sub_pd(a[0], t);
                a[0] = _mm_add_pd(a[0], t);
            }

            // k=1: W8^1 = (1/√2, ∓1/√2) — pure-mul twiddle, no FMA
            // Forward: W8^1 = (cos(-π/4), sin(-π/4)) = (1/√2, -1/√2)
            // Inverse: W8^1 = (cos(+π/4), sin(+π/4)) = (1/√2, +1/√2)
            {
                let v = a[5];
                let swapped = _mm_shuffle_pd(v, v, 0b01);
                let t_tw = if fwd {
                    // Forward: [(re+im)/√2, (im-re)/√2]
                    let sum  = _mm_add_pd(v, swapped); // [re+im, im+re]
                    let diff = _mm_sub_pd(swapped, v); // [im-re, re-im]
                    let combined = _mm_shuffle_pd(sum, diff, 0b00);
                    _mm_mul_pd(combined, inv_sqrt2)
                } else {
                    // Inverse: [(re-im)/√2, (im+re)/√2]
                    let diff = _mm_sub_pd(v, swapped); // [re-im, im-re]
                    let sum  = _mm_add_pd(v, swapped); // [re+im, im+re]
                    let combined = _mm_shuffle_pd(diff, sum, 0b10);
                    _mm_mul_pd(combined, inv_sqrt2)
                };
                a[5] = _mm_sub_pd(a[1], t_tw);
                a[1] = _mm_add_pd(a[1], t_tw);
            }

            // k=2: W8^2 = ∓i
            {
                let t = a[6];
                let t_tw = rotate_pm_i(t, fwd);
                a[6] = _mm_sub_pd(a[2], t_tw);
                a[2] = _mm_add_pd(a[2], t_tw);
            }

            // k=3: W8^3 — complex multiply without FMA
            // Forward: W8^3 = (-1/√2, -1/√2)
            // Inverse: W8^3 = (-1/√2, +1/√2)
            {
                let v = a[7];
                let swapped = _mm_shuffle_pd(v, v, 0b01);
                let t_tw = if fwd {
                    // Forward: [(-re+im)/√2, (-im-re)/√2]
                    let t = _mm_sub_pd(swapped, v); // [im-re, re-im]
                    let neg_sum = _mm_xor_pd(
                        _mm_add_pd(v, swapped),
                        _mm_set1_pd(-0.0),
                    ); // [-(re+im), -(im+re)]
                    let combined = _mm_shuffle_pd(t, neg_sum, 0b00);
                    _mm_mul_pd(combined, inv_sqrt2)
                } else {
                    // Inverse: [(-re-im)/√2, (-im+re)/√2]
                    let neg_sum = _mm_xor_pd(
                        _mm_add_pd(v, swapped),
                        _mm_set1_pd(-0.0),
                    ); // [-(re+im), -(im+re)]
                    let diff = _mm_sub_pd(swapped, v); // [im-re, re-im]
                    let combined = _mm_shuffle_pd(neg_sum, diff, 0b10);
                    _mm_mul_pd(combined, inv_sqrt2)
                };
                a[7] = _mm_sub_pd(a[3], t_tw);
                a[3] = _mm_add_pd(a[3], t_tw);
            }

            // Store via 256-bit writes using _mm256_permute2f128_pd
            // (pure-AVX lane-permute, avoids AVX2 insert128 dependency)
            // _mm256_permute2f128_pd(A, B, 0x20) = [A.lo128, B.lo128] — 3c vs 1c+1c insert
            let _ = cmul_no_fma; // silence unused warning (only used above for W8^3)
            for i in (0..8usize).step_by(2) {
                let packed = _mm256_permute2f128_pd(
                    _mm256_castpd128_pd256(a[i]),
                    _mm256_castpd128_pd256(a[i + 1]),
                    0x20, // [a[i].lo, a[i+1].lo]
                );
                _mm256_storeu_pd(ptr.add(i * 2), packed);
            }
        }
    }
}
