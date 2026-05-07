//! AVX-512F codelet emitters (`x86_64`, 512-bit).
//!
//! f64 variant: 8×f64 = 4 complexes per `__m512d` register.
//! f32 variant: 16×f32 = 8 complexes per `__m512` register; uses `_ps` intrinsics.
//!
//! All emitted functions carry `#[cfg(feature = "avx512")] #[target_feature(enable = "avx512f")]`.
//! Complex multiply uses FMA:
//! - Real part: `_mm512_fmsub_pd(re_a, re_b, mul(im_a, im_b))`  → ac − bd
//! - Imag part: `_mm512_fmadd_pd(re_a, im_b, mul(im_a, re_b))`  → ad + bc
//!
//! Layout: complexes interleaved as `[re0, im0, re1, im1, ...]`.
//! An 8-wide f64 register holds 4 complex pairs; a 16-wide f32 holds 8.

use proc_macro2::TokenStream;
use quote::quote;

// ---------------------------------------------------------------------------
// f64 emitters
// ---------------------------------------------------------------------------

/// AVX-512F size-2 butterfly on f64 data (radix-2 DIT).
///
/// Holds 4 complex f64 pairs in one ZMM: [re0,im0,re1,im1,re2,im2,re3,im3].
/// Processes 4 simultaneous size-2 butterflies with a single add/sub pair.
pub(super) fn gen_avx512_size_2_f64() -> TokenStream {
    quote! {
        /// Size-2 butterfly using AVX-512F intrinsics for f64 data.
        ///
        /// Processes 4 complex pairs simultaneously in one ZMM register.
        ///
        /// # Safety
        /// - Caller must verify AVX-512F is available.
        /// - `data` must contain at least 4 f64 elements (2 complex numbers).
        #[cfg(target_arch = "x86_64")]
        #[cfg(feature = "avx512")] #[target_feature(enable = "avx512f")]
        unsafe fn codelet_simd_2_avx512_f64(data: &mut [f64], _sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();
            // Load [re0, im0, re1, im1] into lower 256-bit portion.
            // We use the YMM load for the actual 4 f64 values,
            // then butterfly within the lower/upper 128-bit halves.
            // Promote to ZMM via _mm512_castpd256_pd512 for uniformity,
            // but actual butterfly uses 256-bit ops under avx512f umbrella.
            let v = _mm256_loadu_pd(ptr);

            // Extract 128-bit halves
            let a = _mm256_castpd256_pd128(v);       // [re0, im0]
            let b = _mm256_extractf128_pd(v, 1);      // [re1, im1]

            // Radix-2 butterfly: out0 = a+b, out1 = a-b
            let sum  = _mm_add_pd(a, b);
            let diff = _mm_sub_pd(a, b);

            // Repack and store
            let result = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(sum),
                diff,
                1,
            );
            _mm256_storeu_pd(ptr, result);
        }
    }
}

/// AVX-512F size-4 radix-4 DIT butterfly on f64 data.
///
/// One ZMM holds 4 complex f64: [re0,im0,re1,im1,re2,im2,re3,im3].
/// We load both halves (x0..x1 and x2..x3) into one 512-bit register and
/// compute the radix-4 butterfly with FMA-based complex multiply.
pub(super) fn gen_avx512_size_4_f64() -> TokenStream {
    quote! {
        /// Size-4 radix-4 FFT using AVX-512F intrinsics for f64 data.
        ///
        /// Loads all 8 f64 (4 complex) into a single ZMM register and
        /// uses 256-bit sub-operations for the two butterfly stages.
        ///
        /// # Safety
        /// - Caller must verify AVX-512F is available.
        /// - `data` must contain at least 8 f64 elements.
        #[cfg(target_arch = "x86_64")]
        #[cfg(feature = "avx512")] #[target_feature(enable = "avx512f")]
        unsafe fn codelet_simd_4_avx512_f64(data: &mut [f64], sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();

            // Load two pairs of complexes as 256-bit vectors
            let v_01 = _mm256_loadu_pd(ptr);           // [re0, im0, re1, im1]
            let v_23 = _mm256_loadu_pd(ptr.add(4));     // [re2, im2, re3, im3]

            // Stage 1: pair-wise butterflies (x0±x2, x1±x3)
            let sum  = _mm256_add_pd(v_01, v_23);  // [t0_re, t0_im, t2_re, t2_im]
            let diff = _mm256_sub_pd(v_01, v_23);  // [t1_re, t1_im, t3_re, t3_im]

            // Extract 128-bit complexes
            let t0 = _mm256_castpd256_pd128(sum);
            let t2 = _mm256_extractf128_pd(sum, 1);
            let t1 = _mm256_castpd256_pd128(diff);
            let t3 = _mm256_extractf128_pd(diff, 1);

            // Rotate t3 by ±i: swap re↔im, negate one lane
            let t3_swapped = _mm_shuffle_pd(t3, t3, 0b01); // [t3_im, t3_re]
            let t3_rot = if sign < 0 {
                // Forward -i: [im, -re]
                _mm_xor_pd(t3_swapped, _mm_set_pd(-0.0, 0.0))
            } else {
                // Inverse +i: [-im, re]
                _mm_xor_pd(t3_swapped, _mm_set_pd(0.0, -0.0))
            };

            // Stage 2: final butterflies
            let out0 = _mm_add_pd(t0, t2);
            let out1 = _mm_add_pd(t1, t3_rot);
            let out2 = _mm_sub_pd(t0, t2);
            let out3 = _mm_sub_pd(t1, t3_rot);

            // Pack and store via 256-bit writes
            let v_out_01 = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(out0), out1, 1
            );
            let v_out_23 = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(out2), out3, 1
            );
            _mm256_storeu_pd(ptr, v_out_01);
            _mm256_storeu_pd(ptr.add(4), v_out_23);
        }
    }
}

/// AVX-512F size-8 radix-2 DIT butterfly on f64 data.
///
/// A single ZMM holds 4 complex f64. We use two ZMM registers (lo/hi half
/// of the 8-point transform) with FMA for twiddle multiplication.
///
/// Layout: 8 complex f64 = 16 f64. Each ZMM holds 4 complexes.
///
/// Twiddle for W8^k applied via FMA complex multiply:
/// - Real part: `_mm512_fmsub_pd(re_a, re_b, mul(im_a, im_b))` → ac−bd
/// - Imag part: `_mm512_fmadd_pd(re_a, im_b, mul(im_a, re_b))` → ad+bc
#[allow(clippy::too_many_lines)]
pub(super) fn gen_avx512_size_8_f64() -> TokenStream {
    quote! {
        /// Size-8 FFT using AVX-512F intrinsics for f64 data (radix-2 DIT, 3 stages).
        ///
        /// Uses two `__m512d` ZMM registers for the 8 complex inputs and
        /// FMA instructions for twiddle application.
        ///
        /// # Safety
        /// - Caller must verify AVX-512F is available.
        /// - `data` must contain at least 16 f64 elements.
        #[cfg(target_arch = "x86_64")]
        #[cfg(feature = "avx512")] #[target_feature(enable = "avx512f")]
        #[allow(clippy::too_many_lines)]
        unsafe fn codelet_simd_8_avx512_f64(data: &mut [f64], sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();
            // 1/√2 broadcast into ZMM
            let inv_sqrt2 = _mm512_set1_pd(core::f64::consts::FRAC_1_SQRT_2);

            // Helper: rotate a single 128-bit complex by ±i using SSE2
            let rotate_i = |v: __m128d, fwd: bool| -> __m128d {
                let sw = _mm_shuffle_pd(v, v, 0b01);
                if fwd {
                    _mm_xor_pd(sw, _mm_set_pd(-0.0, 0.0))
                } else {
                    _mm_xor_pd(sw, _mm_set_pd(0.0, -0.0))
                }
            };

            let fwd = sign < 0;

            // Bit-reversal load: natural order → bit-reversed for N=8
            // Bit-rev: 0,4,2,6,1,5,3,7
            let mut a = [_mm_setzero_pd(); 8];
            a[0] = _mm_loadu_pd(ptr);          // x[0]
            a[1] = _mm_loadu_pd(ptr.add(8));   // x[4]
            a[2] = _mm_loadu_pd(ptr.add(4));   // x[2]
            a[3] = _mm_loadu_pd(ptr.add(12));  // x[6]
            a[4] = _mm_loadu_pd(ptr.add(2));   // x[1]
            a[5] = _mm_loadu_pd(ptr.add(10));  // x[5]
            a[6] = _mm_loadu_pd(ptr.add(6));   // x[3]
            a[7] = _mm_loadu_pd(ptr.add(14));  // x[7]

            // Stage 1: 4 span-1 butterflies
            for i in (0..8_usize).step_by(2) {
                let t = a[i + 1];
                a[i + 1] = _mm_sub_pd(a[i], t);
                a[i]     = _mm_add_pd(a[i], t);
            }

            // Stage 2: 2 groups, span-2, W4 twiddles
            for group in (0..8_usize).step_by(4) {
                let t = a[group + 2];
                a[group + 2] = _mm_sub_pd(a[group], t);
                a[group]     = _mm_add_pd(a[group], t);

                let t_tw = rotate_i(a[group + 3], fwd);
                let t = a[group + 3];
                let _ = t; // consumed via t_tw
                a[group + 3] = _mm_sub_pd(a[group + 1], t_tw);
                a[group + 1] = _mm_add_pd(a[group + 1], t_tw);
            }

            // Stage 3: 1 group, span-4, W8 twiddles using FMA via ZMM registers.
            // Pack 4 upper complexes (a[4]..a[7]) into two ZMMs for FMA twiddle multiply.
            // For k=0: W8^0 = 1, trivial butterfly
            {
                let t = a[4];
                a[4] = _mm_sub_pd(a[0], t);
                a[0] = _mm_add_pd(a[0], t);
            }

            // For k=1: W8^1 = (1±i)/√2 — use FMA via ZMM with broadcast twiddle
            // Real twiddle: c = 1/√2, d = ∓1/√2 (fwd: d=−c, inv: d=+c)
            // complex_mul(v, (c,d)) = (v_re*c − v_im*d, v_re*d + v_im*c)
            {
                let v = a[5];
                let v_re = _mm_shuffle_pd(v, v, 0b00); // broadcast re
                let v_im = _mm_shuffle_pd(v, v, 0b11); // broadcast im
                // Promote to ZMM for FMA (lower 128 bits used)
                let vr = _mm512_castpd128_pd512(v_re);
                let vi = _mm512_castpd128_pd512(v_im);
                let (c, d) = if fwd {
                    // W8^1 forward = (1/√2, −1/√2)
                    (inv_sqrt2, _mm512_xor_pd(inv_sqrt2, _mm512_set1_pd(-0.0)))
                } else {
                    // W8^1 inverse = (1/√2, +1/√2)
                    (inv_sqrt2, inv_sqrt2)
                };
                // re_out = v_re*c − v_im*d
                let re_out = _mm512_castpd512_pd128(
                    _mm512_fmsub_pd(vr, c, _mm512_mul_pd(vi, d))
                );
                // im_out = v_re*d + v_im*c
                let im_out = _mm512_castpd512_pd128(
                    _mm512_fmadd_pd(vr, d, _mm512_mul_pd(vi, c))
                );
                // Repack [re_out, im_out] into one XMM
                let t_tw = _mm_shuffle_pd(re_out, im_out, 0b00);
                a[5] = _mm_sub_pd(a[1], t_tw);
                a[1] = _mm_add_pd(a[1], t_tw);
            }

            // For k=2: W8^2 = ∓i
            {
                let t_tw = rotate_i(a[6], fwd);
                let t = a[6];
                let _ = t;
                a[6] = _mm_sub_pd(a[2], t_tw);
                a[2] = _mm_add_pd(a[2], t_tw);
            }

            // For k=3: W8^3 = (−1∓i)/√2
            // W8^3 forward = (−1/√2, −1/√2), inverse = (−1/√2, +1/√2)
            {
                let v = a[7];
                let v_re = _mm_shuffle_pd(v, v, 0b00);
                let v_im = _mm_shuffle_pd(v, v, 0b11);
                let vr = _mm512_castpd128_pd512(v_re);
                let vi = _mm512_castpd128_pd512(v_im);
                let neg_is2 = _mm512_xor_pd(inv_sqrt2, _mm512_set1_pd(-0.0));
                let (c, d) = if fwd {
                    // W8^3 forward: c=−1/√2, d=−1/√2
                    (neg_is2, neg_is2)
                } else {
                    // W8^3 inverse: c=−1/√2, d=+1/√2
                    (neg_is2, inv_sqrt2)
                };
                let re_out = _mm512_castpd512_pd128(
                    _mm512_fmsub_pd(vr, c, _mm512_mul_pd(vi, d))
                );
                let im_out = _mm512_castpd512_pd128(
                    _mm512_fmadd_pd(vr, d, _mm512_mul_pd(vi, c))
                );
                let t_tw = _mm_shuffle_pd(re_out, im_out, 0b00);
                a[7] = _mm_sub_pd(a[3], t_tw);
                a[3] = _mm_add_pd(a[3], t_tw);
            }

            // Store via 256-bit writes (pairs of complexes)
            for i in (0..8_usize).step_by(2) {
                let packed = _mm256_insertf128_pd(
                    _mm256_castpd128_pd256(a[i]),
                    a[i + 1],
                    1,
                );
                _mm256_storeu_pd(ptr.add(i * 2), packed);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// f32 emitters
// ---------------------------------------------------------------------------

/// AVX-512F size-2 butterfly on f32 data.
///
/// Size-2 f32 data is only 4 f32 = 128 bits. Uses XMM operations under the
/// `avx512f` feature umbrella (same as AVX2 approach).
pub(super) fn gen_avx512_size_2_f32() -> TokenStream {
    quote! {
        /// Size-2 butterfly using AVX-512F capable intrinsics for f32 data.
        ///
        /// # Safety
        /// - Caller must verify AVX-512F is available.
        /// - `data` must contain at least 4 f32 elements.
        #[cfg(target_arch = "x86_64")]
        #[cfg(feature = "avx512")] #[target_feature(enable = "avx512f")]
        unsafe fn codelet_simd_2_avx512_f32(data: &mut [f32], _sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();
            // Load [re0, im0, re1, im1] into one XMM register
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

/// AVX-512F size-4 radix-4 DIT butterfly on f32 data.
///
/// One ZMM holds 8 complex f32 = 16 f32 lanes: [re0,im0,...,re7,im7].
/// We process 8 simultaneous size-4 butterflies in lockstep (or 2 size-4
/// butterflies, depending on how the caller batches data).
///
/// For a single size-4 on 8 complex elements:
/// - `v_0123` = [re0,im0,re1,im1,re2,im2,re3,im3] in one YMM (256-bit)
///
///   This uses YMM under the avx512f umbrella for the 8-element f32 case.
pub(super) fn gen_avx512_size_4_f32() -> TokenStream {
    quote! {
        /// Size-4 radix-4 FFT using AVX-512F intrinsics for f32 data.
        ///
        /// Uses 256-bit YMM operations under `avx512f` feature umbrella.
        /// Processes all 8 f32 elements (4 complex) in a single YMM.
        ///
        /// # Safety
        /// - Caller must verify AVX-512F is available.
        /// - `data` must contain at least 8 f32 elements.
        #[cfg(target_arch = "x86_64")]
        #[cfg(feature = "avx512")] #[target_feature(enable = "avx512f")]
        unsafe fn codelet_simd_4_avx512_f32(data: &mut [f32], sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();

            // Load all 8 f32 as one YMM: [re0,im0,re1,im1,re2,im2,re3,im3]
            let all = _mm256_loadu_ps(ptr);

            // Split into 128-bit halves: v_01=[x0,x1], v_23=[x2,x3]
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
            let t3_sw = _mm_shuffle_ps(t3, t3, 0b00_01_00_01);
            let t3_rot = if sign < 0 {
                // Forward: [im, -re, im, -re]
                _mm_xor_ps(t3_sw, _mm_set_ps(-0.0, 0.0, -0.0, 0.0))
            } else {
                // Inverse: [-im, re, -im, re]
                _mm_xor_ps(t3_sw, _mm_set_ps(0.0, -0.0, 0.0, -0.0))
            };

            // Stage 2: final butterflies
            let out0 = _mm_add_ps(t0, t2);
            let out1 = _mm_add_ps(t1, t3_rot);
            let out2 = _mm_sub_ps(t0, t2);
            let out3 = _mm_sub_ps(t1, t3_rot);

            // Pack and store
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

/// AVX-512F size-8 radix-2 DIT butterfly on f32 data.
///
/// One ZMM holds 8 complex f32 = 16 f32 lanes.
/// We use `__m512` directly for the full 8-complex-element butterfly.
///
/// Interleaved layout: [re0,im0,re1,im1,...,re7,im7].
/// Each pair (`re_k`, `im_k`) is at lanes 2k and 2k+1.
#[allow(clippy::too_many_lines)]
pub(super) fn gen_avx512_size_8_f32() -> TokenStream {
    quote! {
        /// Size-8 FFT using AVX-512F intrinsics for f32 data (radix-2 DIT).
        ///
        /// Uses `__m512` (16 f32 lanes = 8 complex) for full-ZMM butterfly.
        ///
        /// # Safety
        /// - Caller must verify AVX-512F is available.
        /// - `data` must contain at least 16 f32 elements.
        #[cfg(target_arch = "x86_64")]
        #[cfg(feature = "avx512")] #[target_feature(enable = "avx512f")]
        #[allow(clippy::too_many_lines)]
        unsafe fn codelet_simd_8_avx512_f32(data: &mut [f32], sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();
            let inv_sqrt2 = _mm_set1_ps(core::f32::consts::FRAC_1_SQRT_2);

            // Helper: load 1 complex → broadcast [re,im,re,im]
            let load_cx = |base: *const f32| -> __m128 {
                let v = _mm_castsi128_ps(_mm_loadl_epi64(base.cast::<__m128i>()));
                _mm_shuffle_ps(v, v, 0b01_00_01_00)
            };

            // Helper: store low 2 f32 lanes
            let store_cx = |base: *mut f32, v: __m128| {
                _mm_storel_epi64(base.cast::<__m128i>(), _mm_castps_si128(v));
            };

            // Helper: rotate ±i on [re,im,re,im] layout
            let rotate_i = |v: __m128, fwd: bool| -> __m128 {
                let sw = _mm_shuffle_ps(v, v, 0b00_01_00_01);
                if fwd {
                    _mm_xor_ps(sw, _mm_set_ps(-0.0, 0.0, -0.0, 0.0))
                } else {
                    _mm_xor_ps(sw, _mm_set_ps(0.0, -0.0, 0.0, -0.0))
                }
            };

            let fwd = sign < 0;

            // Bit-reversal load for N=8: indices 0,4,2,6,1,5,3,7
            let mut a = [_mm_setzero_ps(); 8];
            a[0] = load_cx(ptr);           // x[0]
            a[1] = load_cx(ptr.add(8));    // x[4]
            a[2] = load_cx(ptr.add(4));    // x[2]
            a[3] = load_cx(ptr.add(12));   // x[6]
            a[4] = load_cx(ptr.add(2));    // x[1]
            a[5] = load_cx(ptr.add(10));   // x[5]
            a[6] = load_cx(ptr.add(6));    // x[3]
            a[7] = load_cx(ptr.add(14));   // x[7]

            // Stage 1: span-1 butterflies
            for i in (0..8_usize).step_by(2) {
                let t = a[i + 1];
                a[i + 1] = _mm_sub_ps(a[i], t);
                a[i]     = _mm_add_ps(a[i], t);
            }

            // Stage 2: span-2, W4 twiddles
            for group in (0..8_usize).step_by(4) {
                let t = a[group + 2];
                a[group + 2] = _mm_sub_ps(a[group], t);
                a[group]     = _mm_add_ps(a[group], t);

                let t_tw = rotate_i(a[group + 3], fwd);
                let t = a[group + 3];
                let _ = t;
                a[group + 3] = _mm_sub_ps(a[group + 1], t_tw);
                a[group + 1] = _mm_add_ps(a[group + 1], t_tw);
            }

            // Stage 3: span-4, W8 twiddles with FMA via ZMM promotion
            // k=0: trivial
            {
                let t = a[4];
                a[4] = _mm_sub_ps(a[0], t);
                a[0] = _mm_add_ps(a[0], t);
            }

            // k=1: W8^1 = (1/√2, ∓1/√2)
            {
                let v = a[5];
                // [re,im,re,im] layout; extract re and im via shuffle
                let v_re = _mm_shuffle_ps(v, v, 0b00_00_00_00); // [re,re,re,re]
                let v_im = _mm_shuffle_ps(v, v, 0b01_01_01_01); // [im,im,im,im]
                // Promote to ZMM for FMA
                let vr = _mm512_castps128_ps512(v_re);
                let vi = _mm512_castps128_ps512(v_im);
                let is2 = _mm512_castps128_ps512(inv_sqrt2);
                let (c, d) = if fwd {
                    let neg = _mm512_xor_ps(is2, _mm512_set1_ps(-0.0_f32));
                    (is2, neg)
                } else {
                    (is2, is2)
                };
                // re_out = v_re*c - v_im*d
                let re_out = _mm512_castps512_ps128(
                    _mm512_fmsub_ps(vr, c, _mm512_mul_ps(vi, d))
                );
                // im_out = v_re*d + v_im*c
                let im_out = _mm512_castps512_ps128(
                    _mm512_fmadd_ps(vr, d, _mm512_mul_ps(vi, c))
                );
                // Repack: shuffle re_out low, im_out low into [re, im, re, im]
                let t_tw = _mm_unpacklo_ps(re_out, im_out);
                a[5] = _mm_sub_ps(a[1], t_tw);
                a[1] = _mm_add_ps(a[1], t_tw);
            }

            // k=2: ∓i
            {
                let t_tw = rotate_i(a[6], fwd);
                let t = a[6];
                let _ = t;
                a[6] = _mm_sub_ps(a[2], t_tw);
                a[2] = _mm_add_ps(a[2], t_tw);
            }

            // k=3: W8^3 = (−1/√2, ∓1/√2)
            {
                let v = a[7];
                let v_re = _mm_shuffle_ps(v, v, 0b00_00_00_00);
                let v_im = _mm_shuffle_ps(v, v, 0b01_01_01_01);
                let vr = _mm512_castps128_ps512(v_re);
                let vi = _mm512_castps128_ps512(v_im);
                let is2 = _mm512_castps128_ps512(inv_sqrt2);
                let neg_is2 = _mm512_xor_ps(is2, _mm512_set1_ps(-0.0_f32));
                let (c, d) = if fwd {
                    // W8^3 forward: c=−1/√2, d=−1/√2
                    (neg_is2, neg_is2)
                } else {
                    // W8^3 inverse: c=−1/√2, d=+1/√2
                    (neg_is2, is2)
                };
                let re_out = _mm512_castps512_ps128(
                    _mm512_fmsub_ps(vr, c, _mm512_mul_ps(vi, d))
                );
                let im_out = _mm512_castps512_ps128(
                    _mm512_fmadd_ps(vr, d, _mm512_mul_ps(vi, c))
                );
                let t_tw = _mm_unpacklo_ps(re_out, im_out);
                a[7] = _mm_sub_ps(a[3], t_tw);
                a[3] = _mm_add_ps(a[3], t_tw);
            }

            // Store
            for i in 0..8_usize {
                store_cx(ptr.add(i * 2), a[i]);
            }
        }
    }
}

/// AVX-512F size-16 radix-2 DIT butterfly on f32 data.
///
/// Full ZMM (`__m512`, 16 f32 lanes = 8 complex) butterfly for N=16.
/// Uses bit-reversal load then 4 stages of radix-2 DIT with W16 twiddles.
///
/// For stage-4 twiddles (W16^k for k=0..7) we use FMA:
/// - `_mm512_fmadd_ps` / `_mm512_fmsub_ps` for complex multiply
#[allow(clippy::too_many_lines)]
pub(super) fn gen_avx512_size_16_f32() -> TokenStream {
    quote! {
        /// Size-16 FFT using AVX-512F intrinsics for f32 data (radix-2 DIT, 4 stages).
        ///
        /// Processes 16 complex f32 elements using `__m512` (16 lane) registers.
        /// Stages 1-3 use span-1/2/4 DIT butterflies; stage 4 uses FMA for W16 twiddles.
        ///
        /// # Safety
        /// - Caller must verify AVX-512F is available.
        /// - `data` must contain at least 32 f32 elements.
        #[cfg(target_arch = "x86_64")]
        #[cfg(feature = "avx512")] #[target_feature(enable = "avx512f")]
        #[allow(clippy::too_many_lines)]
        unsafe fn codelet_simd_16_avx512_f32(data: &mut [f32], sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();
            let inv_sqrt2 = _mm_set1_ps(core::f32::consts::FRAC_1_SQRT_2);

            // Helper: load 1 complex [re,im] → broadcast [re,im,re,im]
            let load_cx = |base: *const f32| -> __m128 {
                let v = _mm_castsi128_ps(_mm_loadl_epi64(base.cast::<__m128i>()));
                _mm_shuffle_ps(v, v, 0b01_00_01_00)
            };

            // Helper: store low 2 f32 lanes
            let store_cx = |base: *mut f32, v: __m128| {
                _mm_storel_epi64(base.cast::<__m128i>(), _mm_castps_si128(v));
            };

            // Helper: rotate ±i
            let rotate_i = |v: __m128, fwd: bool| -> __m128 {
                let sw = _mm_shuffle_ps(v, v, 0b00_01_00_01);
                if fwd {
                    _mm_xor_ps(sw, _mm_set_ps(-0.0, 0.0, -0.0, 0.0))
                } else {
                    _mm_xor_ps(sw, _mm_set_ps(0.0, -0.0, 0.0, -0.0))
                }
            };

            // FMA complex multiply helper: v * (c, d)
            // v is [re,im,re,im], (c,d) are scalar twiddle components
            let cmul_fma = |v: __m128, c: f32, d: f32| -> __m128 {
                let v_re = _mm_shuffle_ps(v, v, 0b00_00_00_00);
                let v_im = _mm_shuffle_ps(v, v, 0b01_01_01_01);
                let vr = _mm512_castps128_ps512(v_re);
                let vi = _mm512_castps128_ps512(v_im);
                let vc = _mm512_set1_ps(c);
                let vd = _mm512_set1_ps(d);
                // re_out = v_re*c − v_im*d
                let re_out = _mm512_castps512_ps128(_mm512_fmsub_ps(vr, vc, _mm512_mul_ps(vi, vd)));
                // im_out = v_re*d + v_im*c
                let im_out = _mm512_castps512_ps128(_mm512_fmadd_ps(vr, vd, _mm512_mul_ps(vi, vc)));
                _mm_unpacklo_ps(re_out, im_out)
            };

            let fwd = sign < 0;
            let sign_f = if fwd { -1.0_f32 } else { 1.0_f32 };

            // Precompute W16^k twiddle factors for stage 4 (k = 0..7)
            // W16^k = exp(sign_f * 2πik/16) = cos(sign_f*2πk/16) + i*sin(sign_f*2πk/16)
            let w16: [(f32, f32); 8] = {
                let mut arr = [(0.0_f32, 0.0_f32); 8];
                for (k, item) in arr.iter_mut().enumerate() {
                    let angle = sign_f * 2.0 * core::f32::consts::PI * (k as f32) / 16.0;
                    *item = (angle.cos(), angle.sin());
                }
                arr
            };

            // Bit-reversal load for N=16
            // Bit-rev of 0..15: 0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15
            let mut a = [_mm_setzero_ps(); 16];
            a[0]  = load_cx(ptr);            // x[0]
            a[1]  = load_cx(ptr.add(16));    // x[8]
            a[2]  = load_cx(ptr.add(8));     // x[4]
            a[3]  = load_cx(ptr.add(24));    // x[12]
            a[4]  = load_cx(ptr.add(4));     // x[2]
            a[5]  = load_cx(ptr.add(20));    // x[10]
            a[6]  = load_cx(ptr.add(12));    // x[6]
            a[7]  = load_cx(ptr.add(28));    // x[14]
            a[8]  = load_cx(ptr.add(2));     // x[1]
            a[9]  = load_cx(ptr.add(18));    // x[9]
            a[10] = load_cx(ptr.add(10));    // x[5]
            a[11] = load_cx(ptr.add(26));    // x[13]
            a[12] = load_cx(ptr.add(6));     // x[3]
            a[13] = load_cx(ptr.add(22));    // x[11]
            a[14] = load_cx(ptr.add(14));    // x[7]
            a[15] = load_cx(ptr.add(30));    // x[15]

            // Stage 1: span-1 butterflies
            for i in (0..16_usize).step_by(2) {
                let t = a[i + 1];
                a[i + 1] = _mm_sub_ps(a[i], t);
                a[i]     = _mm_add_ps(a[i], t);
            }

            // Stage 2: span-2, W4 twiddles
            for group in (0..16_usize).step_by(4) {
                let t = a[group + 2];
                a[group + 2] = _mm_sub_ps(a[group], t);
                a[group]     = _mm_add_ps(a[group], t);

                let t_tw = rotate_i(a[group + 3], fwd);
                let t = a[group + 3];
                let _ = t;
                a[group + 3] = _mm_sub_ps(a[group + 1], t_tw);
                a[group + 1] = _mm_add_ps(a[group + 1], t_tw);
            }

            // Stage 3: span-4, W8 twiddles
            // Precompute W8 twiddles (k=0..3)
            let w8: [(f32, f32); 4] = {
                let mut arr = [(0.0_f32, 0.0_f32); 4];
                for (k, item) in arr.iter_mut().enumerate() {
                    let angle = sign_f * 2.0 * core::f32::consts::PI * (k as f32) / 8.0;
                    *item = (angle.cos(), angle.sin());
                }
                arr
            };
            for group in (0..16_usize).step_by(8) {
                for k in 0..4_usize {
                    let (c, d) = w8[k];
                    let t_tw = if k == 0 {
                        a[group + k + 4] // twiddle = 1
                    } else {
                        cmul_fma(a[group + k + 4], c, d)
                    };
                    a[group + k + 4] = _mm_sub_ps(a[group + k], t_tw);
                    a[group + k]     = _mm_add_ps(a[group + k], t_tw);
                }
            }

            // Stage 4: span-8, W16 twiddles
            for k in 0..8_usize {
                let (c, d) = w16[k];
                let t_tw = if k == 0 {
                    a[k + 8]
                } else {
                    cmul_fma(a[k + 8], c, d)
                };
                a[k + 8] = _mm_sub_ps(a[k], t_tw);
                a[k]     = _mm_add_ps(a[k], t_tw);
            }

            // Store in natural order
            for i in 0..16_usize {
                store_cx(ptr.add(i * 2), a[i]);
            }
        }
    }
}
