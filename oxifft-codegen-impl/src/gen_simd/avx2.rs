//! AVX2+FMA codelet emitters (`x86_64`, 256-bit).
//!
//! f64 variant: 4×f64 = 2 complexes per `__m256d` register.
//! f32 variant: 8×f32 = 4 complexes per `__m256` register; uses `_ps` intrinsics.

use proc_macro2::TokenStream;
use quote::quote;

/// AVX2+FMA size-2 butterfly on f64 data.
///
/// Loads both complexes as a single 256-bit vector, splits into 128-bit halves,
/// butterflies, and stores as a single 256-bit vector.
pub(super) fn gen_avx2_size_2() -> TokenStream {
    quote! {
        /// Size-2 butterfly using AVX2 intrinsics for f64 data.
        ///
        /// # Safety
        /// - Caller must verify AVX2+FMA are available.
        /// - `data` must contain at least 4 f64 elements.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2", enable = "fma")]
        unsafe fn codelet_simd_2_avx2_f64(data: &mut [f64], _sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();
            // Load all 4 f64s as one YMM register: [re0, im0, re1, im1]
            let v = _mm256_loadu_pd(ptr);

            // Extract 128-bit halves
            let a = _mm256_castpd256_pd128(v);       // [re0, im0]
            let b = _mm256_extractf128_pd(v, 1);     // [re1, im1]

            // Butterfly
            let sum  = _mm_add_pd(a, b);  // a + b
            let diff = _mm_sub_pd(a, b);  // a - b

            // Pack back into 256-bit and store.
            // _mm256_permute2f128_pd(A, B, 0x20) = [A.lo128, B.lo128]
            // Same semantics as insertf128_pd(..., 1) but using the lane-permute
            // instruction directly: permute2f128_pd = 3c, insertf128_pd = 3c on Haswell.
            let result = _mm256_permute2f128_pd(
                _mm256_castpd128_pd256(sum),
                _mm256_castpd128_pd256(diff),
                0x20, // [sum.lo, diff.lo]
            );
            _mm256_storeu_pd(ptr, result);
        }
    }
}

/// AVX2+FMA size-4 radix-4 butterfly on f64 data.
///
/// Uses 256-bit operations for the pair-wise stages:
/// - `v_01` = [x0, x1], `v_23` = [x2, x3] (each 256 bits = 2 complexes)
/// - Vector add/sub for stage 1
/// - Lane-level rotation for the ±i term
pub(super) fn gen_avx2_size_4() -> TokenStream {
    quote! {
        /// Size-4 radix-4 FFT using AVX2+FMA intrinsics for f64 data.
        ///
        /// # Safety
        /// - Caller must verify AVX2+FMA are available.
        /// - `data` must contain at least 8 f64 elements.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2", enable = "fma")]
        unsafe fn codelet_simd_4_avx2_f64(data: &mut [f64], sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();

            // Load two pairs of complexes as 256-bit vectors
            let v_01 = _mm256_loadu_pd(ptr);         // [re0, im0, re1, im1]
            let v_23 = _mm256_loadu_pd(ptr.add(4));   // [re2, im2, re3, im3]

            // Stage 1: pair-wise butterflies (x0±x2, x1±x3)
            let sum  = _mm256_add_pd(v_01, v_23);
            // sum = [t0_re, t0_im, t2_re, t2_im]
            let diff = _mm256_sub_pd(v_01, v_23);
            // diff = [t1_re, t1_im, t3_re, t3_im]

            // Extract 128-bit halves of sum
            let t0 = _mm256_castpd256_pd128(sum);        // [t0_re, t0_im]
            let t2 = _mm256_extractf128_pd(sum, 1);      // [t2_re, t2_im]

            // Extract 128-bit halves of diff
            let t1 = _mm256_castpd256_pd128(diff);       // [t1_re, t1_im]
            let t3 = _mm256_extractf128_pd(diff, 1);     // [t3_re, t3_im]

            // Rotate t3 by ±i:
            //   Swap re <-> im, then negate one lane
            let t3_swapped = _mm_shuffle_pd(t3, t3, 0b01); // [t3_im, t3_re]
            let t3_rot = if sign < 0 {
                // Forward: -i rotation -> [im, -re]
                _mm_xor_pd(t3_swapped, _mm_set_pd(-0.0, 0.0))
            } else {
                // Inverse: +i rotation -> [-im, re]
                _mm_xor_pd(t3_swapped, _mm_set_pd(0.0, -0.0))
            };

            // Stage 2: final butterflies
            let out0 = _mm_add_pd(t0, t2);       // t0 + t2
            let out1 = _mm_add_pd(t1, t3_rot);   // t1 + t3_rot
            let out2 = _mm_sub_pd(t0, t2);       // t0 - t2
            let out3 = _mm_sub_pd(t1, t3_rot);   // t1 - t3_rot

            // Pack and store.
            // _mm256_permute2f128_pd(A, B, 0x20) = [A.lo128, B.lo128]
            // Replaces _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 1)
            // with a single lane-permute that takes both halves directly:
            // permute2f128_pd = 3c vs castpd128_pd256(1c) + insertf128_pd(3c) = 4c total.
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

/// AVX2+FMA size-8 radix-2 DIT butterfly on f64 data.
///
/// Uses 256-bit loads/stores and FMA for twiddle application.
pub(super) fn gen_avx2_size_8() -> TokenStream {
    quote! {
        /// Size-8 FFT using AVX2+FMA intrinsics for f64 data (radix-2 DIT).
        ///
        /// # Safety
        /// - Caller must verify AVX2+FMA are available.
        /// - `data` must contain at least 16 f64 elements.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2", enable = "fma")]
        #[allow(clippy::too_many_lines)]
        unsafe fn codelet_simd_8_avx2_f64(data: &mut [f64], sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();
            let inv_sqrt2 = _mm_set1_pd(0.707_106_781_186_547_6_f64);

            // Helper: rotate complex ±i in SSE2 register
            let rotate_pm_i_sse = |v: __m128d, fwd: bool| -> __m128d {
                let swapped = _mm_shuffle_pd(v, v, 0b01);
                if fwd {
                    _mm_xor_pd(swapped, _mm_set_pd(-0.0, 0.0))
                } else {
                    _mm_xor_pd(swapped, _mm_set_pd(0.0, -0.0))
                }
            };
            let fwd = sign < 0;

            // Bit-reversal load into SSE2 registers (1 complex each)
            let mut a = [_mm_setzero_pd(); 8];
            a[0] = _mm_loadu_pd(ptr);            // x[0]
            a[1] = _mm_loadu_pd(ptr.add(8));     // x[4]
            a[2] = _mm_loadu_pd(ptr.add(4));     // x[2]
            a[3] = _mm_loadu_pd(ptr.add(12));    // x[6]
            a[4] = _mm_loadu_pd(ptr.add(2));     // x[1]
            a[5] = _mm_loadu_pd(ptr.add(10));    // x[5]
            a[6] = _mm_loadu_pd(ptr.add(6));     // x[3]
            a[7] = _mm_loadu_pd(ptr.add(14));    // x[7]

            // Stage 1: 4 span-1 butterflies (trivial twiddle)
            for i in (0..8usize).step_by(2) {
                let t = a[i + 1];
                a[i + 1] = _mm_sub_pd(a[i], t);
                a[i]     = _mm_add_pd(a[i], t);
            }

            // Stage 2: 2 groups, span 2, W4 twiddles
            for group in (0..8usize).step_by(4) {
                let t = a[group + 2];
                a[group + 2] = _mm_sub_pd(a[group], t);
                a[group]     = _mm_add_pd(a[group], t);

                let t = a[group + 3];
                let t_tw = rotate_pm_i_sse(t, fwd);
                a[group + 3] = _mm_sub_pd(a[group + 1], t_tw);
                a[group + 1] = _mm_add_pd(a[group + 1], t_tw);
            }

            // Stage 3: 1 group, span 4, W8 twiddles (using FMA for k=1,3)
            // k=0: W8^0 = 1
            let t = a[4];
            a[4] = _mm_sub_pd(a[0], t);
            a[0] = _mm_add_pd(a[0], t);

            // k=1: W8^1  — use FMA for the twiddle application
            {
                let v = a[5];
                let swapped = _mm_shuffle_pd(v, v, 0b01);
                let t_tw = if fwd {
                    // [(re+im)/√2, (im-re)/√2]
                    let sum = _mm_add_pd(v, swapped);
                    let diff = _mm_sub_pd(swapped, v);
                    let combined = _mm_shuffle_pd(sum, diff, 0b00);
                    _mm_mul_pd(combined, inv_sqrt2)
                } else {
                    // [(re-im)/√2, (im+re)/√2]
                    let diff = _mm_sub_pd(v, swapped);
                    let sum = _mm_add_pd(v, swapped);
                    let combined = _mm_shuffle_pd(diff, sum, 0b10);
                    _mm_mul_pd(combined, inv_sqrt2)
                };
                a[5] = _mm_sub_pd(a[1], t_tw);
                a[1] = _mm_add_pd(a[1], t_tw);
            }

            // k=2: W8^2 = ∓i
            {
                let t = a[6];
                let t_tw = rotate_pm_i_sse(t, fwd);
                a[6] = _mm_sub_pd(a[2], t_tw);
                a[2] = _mm_add_pd(a[2], t_tw);
            }

            // k=3: W8^3
            {
                let v = a[7];
                let swapped = _mm_shuffle_pd(v, v, 0b01);
                let t_tw = if fwd {
                    // [(-re+im)/√2, (-im-re)/√2]
                    let t = _mm_sub_pd(swapped, v);
                    let neg_sum = _mm_xor_pd(
                        _mm_add_pd(v, swapped),
                        _mm_set1_pd(-0.0),
                    );
                    let combined = _mm_shuffle_pd(t, neg_sum, 0b00);
                    _mm_mul_pd(combined, inv_sqrt2)
                } else {
                    // [(-re-im)/√2, (-im+re)/√2]
                    let neg_sum = _mm_xor_pd(
                        _mm_add_pd(v, swapped),
                        _mm_set1_pd(-0.0),
                    );
                    let diff = _mm_sub_pd(swapped, v);
                    let combined = _mm_shuffle_pd(neg_sum, diff, 0b10);
                    _mm_mul_pd(combined, inv_sqrt2)
                };
                a[7] = _mm_sub_pd(a[3], t_tw);
                a[3] = _mm_add_pd(a[3], t_tw);
            }

            // Store via 256-bit writes (pairs of complex).
            // _mm256_permute2f128_pd(A, B, 0x20) = [A.lo128, B.lo128]
            // Single-instruction replacement for insertf128_pd(..., 1):
            // permute2f128_pd = 3c vs castpd128_pd256(1c) + insertf128_pd(3c) = 4c total.
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

// ---------------------------------------------------------------------------
// AVX2 f32 emitters
// ---------------------------------------------------------------------------

/// AVX2+FMA size-2 butterfly on f32 data.
///
/// Size-2 f32 data is only 4×f32 = 128 bits — cannot fill a 256-bit YMM.
/// We use 128-bit SSE `_ps` intrinsics under the AVX2 feature umbrella.
pub(super) fn gen_avx2_size_2_f32() -> TokenStream {
    quote! {
        /// Size-2 butterfly using AVX2-capable SSE intrinsics for f32 data.
        ///
        /// # Safety
        /// - Caller must verify AVX2+FMA are available.
        /// - `data` must contain at least 4 f32 elements.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2", enable = "fma")]
        unsafe fn codelet_simd_2_avx2_f32(data: &mut [f32], _sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();
            // Load [re0, im0, re1, im1] into one XMM
            let v = _mm_loadu_ps(ptr);
            // Extract a=[re0,im0,re0,im0], b=[re1,im1,re1,im1]
            let a = _mm_shuffle_ps(v, v, 0b01_00_01_00);
            let b = _mm_shuffle_ps(v, v, 0b11_10_11_10);
            let sum  = _mm_add_ps(a, b);
            let diff = _mm_sub_ps(a, b);
            // Merge low halves: [sum.lo, diff.lo]
            let out = _mm_shuffle_ps(sum, diff, 0b01_00_01_00);
            _mm_storeu_ps(ptr, out);
        }
    }
}

/// AVX2+FMA size-4 radix-4 butterfly on f32 data.
///
/// A single `__m256` holds 4 complexes as 8 f32 lanes: [re0,im0,re1,im1,re2,im2,re3,im3].
pub(super) fn gen_avx2_size_4_f32() -> TokenStream {
    quote! {
        /// Size-4 radix-4 FFT using AVX2+FMA intrinsics for f32 data.
        ///
        /// # Safety
        /// - Caller must verify AVX2+FMA are available.
        /// - `data` must contain at least 8 f32 elements.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2", enable = "fma")]
        unsafe fn codelet_simd_4_avx2_f32(data: &mut [f32], sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();

            // Load all 8 f32 as one YMM: [re0,im0,re1,im1,re2,im2,re3,im3]
            let all = _mm256_loadu_ps(ptr);

            // Split into 128-bit halves: v_01 = [x0,x1], v_23 = [x2,x3]
            let v_01 = _mm256_castps256_ps128(all);
            let v_23 = _mm256_extractf128_ps(all, 1);

            // Stage 1: pair-wise butterflies
            let sum  = _mm_add_ps(v_01, v_23); // [t0_re, t0_im, t2_re, t2_im]
            let diff = _mm_sub_ps(v_01, v_23); // [t1_re, t1_im, t3_re, t3_im]

            // Extract individual complexes
            let t0 = _mm_shuffle_ps(sum,  sum,  0b01_00_01_00);
            let t2 = _mm_shuffle_ps(sum,  sum,  0b11_10_11_10);
            let t1 = _mm_shuffle_ps(diff, diff, 0b01_00_01_00);
            let t3 = _mm_shuffle_ps(diff, diff, 0b11_10_11_10);

            // Rotate t3 by ±i
            let t3_swapped = _mm_shuffle_ps(t3, t3, 0b00_01_00_01);
            let t3_rot = if sign < 0 {
                // Forward: [im, -re, im, -re]
                let mask = _mm_set_ps(-0.0, 0.0, -0.0, 0.0);
                _mm_xor_ps(t3_swapped, mask)
            } else {
                // Inverse: [-im, re, -im, re]
                let mask = _mm_set_ps(0.0, -0.0, 0.0, -0.0);
                _mm_xor_ps(t3_swapped, mask)
            };

            // Stage 2: final butterflies
            let out0 = _mm_add_ps(t0, t2);
            let out1 = _mm_add_ps(t1, t3_rot);
            let out2 = _mm_sub_ps(t0, t2);
            let out3 = _mm_sub_ps(t1, t3_rot);

            // Pack and store via 256-bit store
            let packed_01 = _mm_shuffle_ps(out0, out1, 0b01_00_01_00);
            let packed_23 = _mm_shuffle_ps(out2, out3, 0b01_00_01_00);
            let result = _mm256_insertf128_ps(
                _mm256_castps128_ps256(packed_01),
                packed_23,
                1,
            );
            _mm256_storeu_ps(ptr, result);
        }
    }
}

/// AVX2+FMA size-8 radix-2 DIT butterfly on f32 data.
///
/// Uses `__m128` (2 complexes per register = 4 f32 lanes) with `_ps` intrinsics
/// and the same bit-reversal / 3-stage DIT structure as the f64 version.
#[allow(clippy::too_many_lines)]
pub(super) fn gen_avx2_size_8_f32() -> TokenStream {
    quote! {
        /// Size-8 FFT using AVX2+FMA intrinsics for f32 data (radix-2 DIT).
        ///
        /// # Safety
        /// - Caller must verify AVX2+FMA are available.
        /// - `data` must contain at least 16 f32 elements.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2", enable = "fma")]
        #[allow(clippy::too_many_lines)]
        unsafe fn codelet_simd_8_avx2_f32(data: &mut [f32], sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();
            let inv_sqrt2 = _mm_set1_ps(0.707_106_8_f32);

            // Helper: load 1 complex [re,im] into broadcast [re,im,re,im]
            // Uses _mm_loadl_epi64 (SSE2) to avoid the MMX __m64 type.
            let load_cx = |base: *const f32| -> __m128 {
                let v = _mm_castsi128_ps(_mm_loadl_epi64(base.cast::<__m128i>()));
                _mm_shuffle_ps(v, v, 0b01_00_01_00)
            };

            // Helper: store low 2 lanes of XMM
            // Uses _mm_storel_epi64 (SSE2) to avoid the MMX __m64 type.
            let store_cx = |base: *mut f32, v: __m128| {
                _mm_storel_epi64(base.cast::<__m128i>(), _mm_castps_si128(v));
            };

            // Helper: rotate complex by ±i (broadcast [re,im,re,im] layout)
            let rotate_pm_i_ps = |v: __m128, fwd: bool| -> __m128 {
                let sw = _mm_shuffle_ps(v, v, 0b00_01_00_01);
                if fwd {
                    let mask = _mm_set_ps(-0.0, 0.0, -0.0, 0.0);
                    _mm_xor_ps(sw, mask)
                } else {
                    let mask = _mm_set_ps(0.0, -0.0, 0.0, -0.0);
                    _mm_xor_ps(sw, mask)
                }
            };

            let fwd = sign < 0;

            // Bit-reversal load
            let mut a = [_mm_setzero_ps(); 8];
            a[0] = load_cx(ptr);             // x[0]
            a[1] = load_cx(ptr.add(8));      // x[4]
            a[2] = load_cx(ptr.add(4));      // x[2]
            a[3] = load_cx(ptr.add(12));     // x[6]
            a[4] = load_cx(ptr.add(2));      // x[1]
            a[5] = load_cx(ptr.add(10));     // x[5]
            a[6] = load_cx(ptr.add(6));      // x[3]
            a[7] = load_cx(ptr.add(14));     // x[7]

            // Stage 1: span-1 butterflies
            for i in (0..8usize).step_by(2) {
                let t = a[i + 1];
                a[i + 1] = _mm_sub_ps(a[i], t);
                a[i]     = _mm_add_ps(a[i], t);
            }

            // Stage 2: span-2 with W4 twiddles
            for group in (0..8usize).step_by(4) {
                let t = a[group + 2];
                a[group + 2] = _mm_sub_ps(a[group], t);
                a[group]     = _mm_add_ps(a[group], t);

                let t = a[group + 3];
                let t_tw = rotate_pm_i_ps(t, fwd);
                a[group + 3] = _mm_sub_ps(a[group + 1], t_tw);
                a[group + 1] = _mm_add_ps(a[group + 1], t_tw);
            }

            // Stage 3: span-4 with W8 twiddles
            // k=0: trivial
            let t = a[4];
            a[4] = _mm_sub_ps(a[0], t);
            a[0] = _mm_add_ps(a[0], t);

            // k=1: W8^1
            {
                let v = a[5];
                let sw = _mm_shuffle_ps(v, v, 0b00_01_00_01);
                let t_tw = if fwd {
                    // [(re+im)/√2, (im-re)/√2] repeated
                    let sum  = _mm_add_ps(v, sw);  // [re+im, im+re, re+im, im+re]
                    let diff = _mm_sub_ps(sw, v);  // [im-re, re-im, im-re, re-im]
                    // interleave: [sum[0], diff[0], sum[1], diff[1]] = [re+im, im-re, im+re, re-im]
                    let combined = _mm_unpacklo_ps(sum, diff);
                    // broadcast low pair: [re+im, im-re, re+im, im-re]
                    _mm_mul_ps(_mm_shuffle_ps(combined, combined, 0b01_00_01_00), inv_sqrt2)
                } else {
                    // [(re-im)/√2, (im+re)/√2] repeated
                    let diff = _mm_sub_ps(v, sw);  // [re-im, im-re, re-im, im-re]
                    let sum  = _mm_add_ps(v, sw);  // [re+im, im+re, re+im, im+re]
                    // swap sum to get [im+re, re+im, ...] at positions 0,1
                    let sum_sw = _mm_shuffle_ps(sum, sum, 0b00_01_00_01);
                    // interleave: [diff[0], sum_sw[0], diff[1], sum_sw[1]] = [re-im, im+re, im-re, re+im]
                    let combined = _mm_unpacklo_ps(diff, sum_sw);
                    // broadcast low pair: [re-im, im+re, re-im, im+re]
                    _mm_mul_ps(_mm_shuffle_ps(combined, combined, 0b01_00_01_00), inv_sqrt2)
                };
                a[5] = _mm_sub_ps(a[1], t_tw);
                a[1] = _mm_add_ps(a[1], t_tw);
            }

            // k=2: ∓i
            {
                let t = a[6];
                let t_tw = rotate_pm_i_ps(t, fwd);
                a[6] = _mm_sub_ps(a[2], t_tw);
                a[2] = _mm_add_ps(a[2], t_tw);
            }

            // k=3: W8^3
            {
                let v = a[7];
                let sw = _mm_shuffle_ps(v, v, 0b00_01_00_01);
                let t_tw = if fwd {
                    // [(im-re)/√2, -(re+im)/√2] repeated
                    let diff = _mm_sub_ps(sw, v);  // [im-re, re-im, im-re, re-im]
                    let neg_sum = _mm_xor_ps(_mm_add_ps(v, sw), _mm_set1_ps(-0.0));
                    // [-(re+im), -(im+re), -(re+im), -(im+re)]
                    // interleave: [diff[0], neg_sum[0], diff[1], neg_sum[1]]
                    //           = [im-re, -(re+im), re-im, -(im+re)]
                    let combined = _mm_unpacklo_ps(diff, neg_sum);
                    // broadcast low pair: [im-re, -(re+im), im-re, -(re+im)]
                    _mm_mul_ps(_mm_shuffle_ps(combined, combined, 0b01_00_01_00), inv_sqrt2)
                } else {
                    // [-(re+im)/√2, (re-im)/√2] repeated
                    let neg_sum = _mm_xor_ps(_mm_add_ps(v, sw), _mm_set1_ps(-0.0));
                    // [-(re+im), -(im+re), -(re+im), -(im+re)]
                    let diff = _mm_sub_ps(sw, v); // [im-re, re-im, im-re, re-im]
                    // swap diff to get [re-im, im-re, ...] at positions 0,1
                    let diff_sw = _mm_shuffle_ps(diff, diff, 0b00_01_00_01);
                    // interleave: [neg_sum[0], diff_sw[0], neg_sum[1], diff_sw[1]]
                    //           = [-(re+im), re-im, -(im+re), im-re]
                    let combined = _mm_unpacklo_ps(neg_sum, diff_sw);
                    // broadcast low pair: [-(re+im), re-im, -(re+im), re-im]
                    _mm_mul_ps(_mm_shuffle_ps(combined, combined, 0b01_00_01_00), inv_sqrt2)
                };
                a[7] = _mm_sub_ps(a[3], t_tw);
                a[3] = _mm_add_ps(a[3], t_tw);
            }

            // Store pairs via 256-bit writes
            for i in (0..8usize).step_by(2) {
                store_cx(ptr.add(i * 2),       a[i]);
                store_cx(ptr.add(i * 2 + 2),   a[i + 1]);
            }
        }
    }
}
