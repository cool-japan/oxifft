//! SSE2 codelet emitters (`x86_64`, 128-bit).
//!
//! f64 variant: 2×f64 = 1 complex per `__m128d` register.
//! f32 variant: 4×f32 = 2 complexes per `__m128` register; uses `_ps` intrinsics.

use proc_macro2::TokenStream;
use quote::quote;

/// SSE2 size-2 butterfly on f64 data.
///
/// Layout: `[re0, im0, re1, im1]` — each complex is one XMM register.
pub(super) fn gen_sse2_size_2() -> TokenStream {
    quote! {
        /// Size-2 butterfly using SSE2 intrinsics for f64 data.
        ///
        /// # Safety
        /// - Caller must verify SSE2 is available (guaranteed on x86_64).
        /// - `data` must contain at least 4 f64 elements.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse2")]
        unsafe fn codelet_simd_2_sse2_f64(data: &mut [f64], _sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();
            // a = [re0, im0], b = [re1, im1]
            let a = _mm_loadu_pd(ptr);
            let b = _mm_loadu_pd(ptr.add(2));
            // Butterfly: out0 = a + b, out1 = a - b
            let sum = _mm_add_pd(a, b);
            let diff = _mm_sub_pd(a, b);
            _mm_storeu_pd(ptr, sum);
            _mm_storeu_pd(ptr.add(2), diff);
        }
    }
}

/// SSE2 size-4 radix-4 butterfly on f64 data.
///
/// Uses shuffle-based ±i rotation for the t3 term.
pub(super) fn gen_sse2_size_4() -> TokenStream {
    quote! {
        /// Size-4 radix-4 FFT using SSE2 intrinsics for f64 data.
        ///
        /// # Safety
        /// - Caller must verify SSE2 is available.
        /// - `data` must contain at least 8 f64 elements.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse2")]
        unsafe fn codelet_simd_4_sse2_f64(data: &mut [f64], sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();

            // Load 4 complex numbers: x0, x1, x2, x3
            let x0 = _mm_loadu_pd(ptr);           // [re0, im0]
            let x1 = _mm_loadu_pd(ptr.add(2));     // [re1, im1]
            let x2 = _mm_loadu_pd(ptr.add(4));     // [re2, im2]
            let x3 = _mm_loadu_pd(ptr.add(6));     // [re3, im3]

            // Stage 1: pair-wise butterflies
            let t0 = _mm_add_pd(x0, x2); // x0 + x2
            let t1 = _mm_sub_pd(x0, x2); // x0 - x2
            let t2 = _mm_add_pd(x1, x3); // x1 + x3
            let t3 = _mm_sub_pd(x1, x3); // x1 - x3

            // Rotate t3 by ±i:
            //   Forward (sign<0): (re,im) -> (im, -re)  [multiply by -i]
            //   Inverse (sign>0): (re,im) -> (-im, re)  [multiply by +i]
            // Step 1: swap re <-> im
            let t3_swapped = _mm_shuffle_pd(t3, t3, 0b01); // [im3, re3]
            // Step 2: negate the appropriate lane
            let t3_rot = if sign < 0 {
                // Need [im, -re]: negate high (the re, now in position 1)
                let mask = _mm_set_pd(-0.0, 0.0);
                _mm_xor_pd(t3_swapped, mask)
            } else {
                // Need [-im, re]: negate low (the im, now in position 0)
                let mask = _mm_set_pd(0.0, -0.0);
                _mm_xor_pd(t3_swapped, mask)
            };

            // Stage 2: final butterflies
            let out0 = _mm_add_pd(t0, t2);       // t0 + t2
            let out1 = _mm_add_pd(t1, t3_rot);   // t1 + t3_rot
            let out2 = _mm_sub_pd(t0, t2);       // t0 - t2
            let out3 = _mm_sub_pd(t1, t3_rot);   // t1 - t3_rot

            // Store results
            _mm_storeu_pd(ptr,       out0);
            _mm_storeu_pd(ptr.add(2), out1);
            _mm_storeu_pd(ptr.add(4), out2);
            _mm_storeu_pd(ptr.add(6), out3);
        }
    }
}

// ---------------------------------------------------------------------------
// SSE2 f32 emitters
// ---------------------------------------------------------------------------

/// SSE2 size-2 butterfly on f32 data.
///
/// `__m128` holds all 4 f32 lanes: [re0, im0, re1, im1].
/// Use 64-bit halves via shuffles to avoid crossing complex-number boundaries.
pub(super) fn gen_sse2_size_2_f32() -> TokenStream {
    quote! {
        /// Size-2 butterfly using SSE2 intrinsics for f32 data.
        ///
        /// # Safety
        /// - Caller must verify SSE2 is available (guaranteed on x86_64).
        /// - `data` must contain at least 4 f32 elements.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse2")]
        unsafe fn codelet_simd_2_sse2_f32(data: &mut [f32], _sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();
            // Load [re0, im0, re1, im1] as one XMM register
            let v = _mm_loadu_ps(ptr);

            // Extract lower/upper halves as f32x2 using shuffle:
            // a = [re0, im0, re0, im0] (movlhps pattern), b = [re1, im1, re1, im1]
            // For butterfly: out_lo = v_lo + v_hi, out_hi = v_lo - v_hi
            // We need to pick lo=elements 0,1 and hi=elements 2,3.
            // _mm_shuffle_ps(v, v, 0b01_00_01_00) = [e0,e1,e0,e1]
            let a = _mm_shuffle_ps(v, v, 0b01_00_01_00); // [re0, im0, re0, im0]
            let b = _mm_shuffle_ps(v, v, 0b11_10_11_10); // [re1, im1, re1, im1]
            let sum  = _mm_add_ps(a, b);
            let diff = _mm_sub_ps(a, b);
            // out = [sum[0], sum[1], diff[0], diff[1]] = [re0+re1, im0+im1, re0-re1, im0-im1]
            let out = _mm_shuffle_ps(sum, diff, 0b01_00_01_00); // [sum.lo, diff.lo]
            _mm_storeu_ps(ptr, out);
        }
    }
}

/// SSE2 size-4 radix-4 butterfly on f32 data.
///
/// Each complex is 2 f32 lanes; a single `__m128` holds 2 complexes.
/// We load data in two XMM registers: v01=[x0,x1], v23=[x2,x3].
pub(super) fn gen_sse2_size_4_f32() -> TokenStream {
    quote! {
        /// Size-4 radix-4 FFT using SSE2 intrinsics for f32 data.
        ///
        /// # Safety
        /// - Caller must verify SSE2 is available.
        /// - `data` must contain at least 8 f32 elements.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse2")]
        unsafe fn codelet_simd_4_sse2_f32(data: &mut [f32], sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();

            // Load two pairs of complexes: [re0,im0,re1,im1] and [re2,im2,re3,im3]
            let v01 = _mm_loadu_ps(ptr);
            let v23 = _mm_loadu_ps(ptr.add(4));

            // Stage 1: pair-wise butterflies (x0±x2, x1±x3) — vector ops
            let sum  = _mm_add_ps(v01, v23); // [t0_re, t0_im, t2_re, t2_im]
            let diff = _mm_sub_ps(v01, v23); // [t1_re, t1_im, t3_re, t3_im]

            // Extract individual complexes using shuffle
            // t0 = [t0_re, t0_im, *,*], t2 = [t2_re, t2_im, *,*]
            let t0 = _mm_shuffle_ps(sum,  sum,  0b01_00_01_00); // [t0_re, t0_im, t0_re, t0_im]
            let t2 = _mm_shuffle_ps(sum,  sum,  0b11_10_11_10); // [t2_re, t2_im, t2_re, t2_im]
            let t1 = _mm_shuffle_ps(diff, diff, 0b01_00_01_00); // [t1_re, t1_im, ...]
            let t3 = _mm_shuffle_ps(diff, diff, 0b11_10_11_10); // [t3_re, t3_im, ...]

            // Rotate t3 by ±i: swap re <-> im, then negate one
            // SSE2 doesn't have float32 shuffle for single pair, use _mm_shuffle_ps
            // t3_swapped = [t3_im, t3_re, t3_im, t3_re]
            let t3_swapped = _mm_shuffle_ps(t3, t3, 0b00_01_00_01);
            let t3_rot = if sign < 0 {
                // Forward: [im, -re, im, -re] — negate the re-lanes (indices 1,3)
                let mask = _mm_set_ps(-0.0, 0.0, -0.0, 0.0);
                _mm_xor_ps(t3_swapped, mask)
            } else {
                // Inverse: [-im, re, -im, re] — negate the im-lanes (indices 0,2)
                let mask = _mm_set_ps(0.0, -0.0, 0.0, -0.0);
                _mm_xor_ps(t3_swapped, mask)
            };

            // Stage 2: final butterflies
            let out0 = _mm_add_ps(t0, t2);
            let out1 = _mm_add_ps(t1, t3_rot);
            let out2 = _mm_sub_ps(t0, t2);
            let out3 = _mm_sub_ps(t1, t3_rot);

            // Pack back: [out0.lo, out1.lo] and [out2.lo, out3.lo]
            let packed_01 = _mm_shuffle_ps(out0, out1, 0b01_00_01_00);
            let packed_23 = _mm_shuffle_ps(out2, out3, 0b01_00_01_00);
            _mm_storeu_ps(ptr,       packed_01);
            _mm_storeu_ps(ptr.add(4), packed_23);
        }
    }
}

/// SSE2 size-8 radix-2 DIT butterfly on f64 data.
pub(super) fn gen_sse2_size_8() -> TokenStream {
    quote! {
        /// Size-8 FFT using SSE2 intrinsics for f64 data (radix-2 DIT).
        ///
        /// # Safety
        /// - Caller must verify SSE2 is available.
        /// - `data` must contain at least 16 f64 elements.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse2")]
        #[allow(clippy::too_many_lines)]
        unsafe fn codelet_simd_8_sse2_f64(data: &mut [f64], sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();

            // 1/sqrt(2)
            let inv_sqrt2 = _mm_set1_pd(0.707_106_781_186_547_6_f64);

            // Bit-reversal load: natural order -> bit-reversed order
            // Bit-rev mapping for N=8: 0,4,2,6,1,5,3,7
            let mut a = [_mm_setzero_pd(); 8];
            a[0] = _mm_loadu_pd(ptr);            // x[0]
            a[1] = _mm_loadu_pd(ptr.add(8));     // x[4]
            a[2] = _mm_loadu_pd(ptr.add(4));     // x[2]
            a[3] = _mm_loadu_pd(ptr.add(12));    // x[6]
            a[4] = _mm_loadu_pd(ptr.add(2));     // x[1]
            a[5] = _mm_loadu_pd(ptr.add(10));    // x[5]
            a[6] = _mm_loadu_pd(ptr.add(6));     // x[3]
            a[7] = _mm_loadu_pd(ptr.add(14));    // x[7]

            // Stage 1: 4 butterflies, span 1, trivial twiddle
            for i in (0..8usize).step_by(2) {
                let t = a[i + 1];
                a[i + 1] = _mm_sub_pd(a[i], t);
                a[i]     = _mm_add_pd(a[i], t);
            }

            // Stage 2: 2 groups of 2 butterflies, span 2, W4 twiddles
            // Helper: rotate complex by ±i using SSE2 shuffle
            let rotate_pm_i = |v: __m128d, fwd: bool| -> __m128d {
                let swapped = _mm_shuffle_pd(v, v, 0b01);
                if fwd {
                    // -i rotation: [im, -re]
                    _mm_xor_pd(swapped, _mm_set_pd(-0.0, 0.0))
                } else {
                    // +i rotation: [-im, re]
                    _mm_xor_pd(swapped, _mm_set_pd(0.0, -0.0))
                }
            };
            let fwd = sign < 0;

            for group in (0..8usize).step_by(4) {
                // k=0: twiddle = 1
                let t = a[group + 2];
                a[group + 2] = _mm_sub_pd(a[group], t);
                a[group]     = _mm_add_pd(a[group], t);

                // k=1: twiddle = ∓i
                let t = a[group + 3];
                let t_tw = rotate_pm_i(t, fwd);
                a[group + 3] = _mm_sub_pd(a[group + 1], t_tw);
                a[group + 1] = _mm_add_pd(a[group + 1], t_tw);
            }

            // Stage 3: 1 group of 4 butterflies, span 4, W8 twiddles
            // Helper: apply W8^k twiddle factor to a complex SSE2 vector
            let apply_w8_twiddle = |v: __m128d, k: usize, is_fwd: bool| -> __m128d {
                match k {
                    0 => v, // W8^0 = 1
                    1 => {
                        // W8^1: fwd = (1-i)/√2, inv = (1+i)/√2
                        // (re+im)/√2 or (re-im)/√2 for real part
                        // (im-re)/√2 or (im+re)/√2 for imag part
                        let re = v;
                        let swapped = _mm_shuffle_pd(v, v, 0b01); // [im, re]
                        if is_fwd {
                            // [(re+im)/√2, (im-re)/√2]
                            let sum = _mm_add_pd(re, swapped); // [re+im, im+re]
                            let diff = _mm_sub_pd(swapped, re); // [im-re, re-im]
                            let combined = _mm_shuffle_pd(sum, diff, 0b00); // [re+im, im-re]
                            _mm_mul_pd(combined, inv_sqrt2)
                        } else {
                            // [(re-im)/√2, (im+re)/√2]
                            let diff = _mm_sub_pd(re, swapped); // [re-im, im-re]
                            let sum = _mm_add_pd(re, swapped); // [re+im, im+re]
                            let combined = _mm_shuffle_pd(diff, sum, 0b10); // [re-im, im+re]
                            _mm_mul_pd(combined, inv_sqrt2)
                        }
                    }
                    2 => rotate_pm_i(v, is_fwd), // W8^2 = ∓i
                    3 => {
                        // W8^3: fwd = (-1-i)/√2, inv = (-1+i)/√2
                        let re = v;
                        let swapped = _mm_shuffle_pd(v, v, 0b01); // [im, re]
                        if is_fwd {
                            // [(-re+im)/√2, (-im-re)/√2]
                            let t = _mm_sub_pd(swapped, re); // [im-re, re-im]
                            let neg_sum = _mm_add_pd(re, swapped); // [re+im, im+re]
                            let neg_sum = _mm_xor_pd(neg_sum, _mm_set1_pd(-0.0));
                            let combined = _mm_shuffle_pd(t, neg_sum, 0b00);
                            _mm_mul_pd(combined, inv_sqrt2)
                        } else {
                            // [(-re-im)/√2, (-im+re)/√2]
                            let neg_sum = _mm_add_pd(re, swapped);
                            let neg_sum = _mm_xor_pd(neg_sum, _mm_set1_pd(-0.0));
                            let diff = _mm_sub_pd(swapped, re); // [im-re, re-im]
                            // want [-re-im, -im+re] = [-(re+im), re-im]
                            let combined = _mm_shuffle_pd(neg_sum, diff, 0b10);
                            _mm_mul_pd(combined, inv_sqrt2)
                        }
                    }
                    _ => v,
                }
            };

            for k in 0..4usize {
                let t = a[k + 4];
                let t_tw = apply_w8_twiddle(t, k, fwd);
                a[k + 4] = _mm_sub_pd(a[k], t_tw);
                a[k]     = _mm_add_pd(a[k], t_tw);
            }

            // Store in natural order
            for i in 0..8usize {
                _mm_storeu_pd(ptr.add(i * 2), a[i]);
            }
        }
    }
}

/// SSE2 size-8 radix-2 DIT butterfly on f32 data.
///
/// Each `__m128` holds 2 complexes. We keep 1 complex per "slot" using
/// `_mm_shuffle_ps` to load/isolate individual complexes for correctness.
#[allow(clippy::too_many_lines)]
pub(super) fn gen_sse2_size_8_f32() -> TokenStream {
    quote! {
        /// Size-8 FFT using SSE2 intrinsics for f32 data (radix-2 DIT).
        ///
        /// # Safety
        /// - Caller must verify SSE2 is available.
        /// - `data` must contain at least 16 f32 elements.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse2")]
        #[allow(clippy::too_many_lines)]
        unsafe fn codelet_simd_8_sse2_f32(data: &mut [f32], sign: i32) {
            use core::arch::x86_64::*;

            let ptr = data.as_mut_ptr();
            // 1/√2 broadcast
            let inv_sqrt2 = _mm_set1_ps(0.707_106_8_f32);

            // Helper: load one complex (2 f32 lanes) into the low half of XMM,
            // broadcast to both halves: [re, im, re, im]
            // Uses _mm_loadl_epi64 (SSE2) to avoid the MMX __m64 type.
            let load_cx = |base: *const f32| -> __m128 {
                let v = _mm_castsi128_ps(_mm_loadl_epi64(base.cast::<__m128i>()));
                _mm_shuffle_ps(v, v, 0b01_00_01_00)
            };

            // Helper: store the low 2 lanes of an XMM to 2 f32 positions
            // Uses _mm_storel_epi64 (SSE2) to avoid the MMX __m64 type.
            let store_cx = |base: *mut f32, v: __m128| {
                _mm_storel_epi64(base.cast::<__m128i>(), _mm_castps_si128(v));
            };

            // Helper: rotate complex by ±i — operate on [re,im,re,im] layout
            let rotate_pm_i = |v: __m128, fwd: bool| -> __m128 {
                // swap pairs: [im,re,im,re]
                let sw = _mm_shuffle_ps(v, v, 0b00_01_00_01);
                if fwd {
                    // [im, -re, im, -re]
                    let mask = _mm_set_ps(-0.0, 0.0, -0.0, 0.0);
                    _mm_xor_ps(sw, mask)
                } else {
                    // [-im, re, -im, re]
                    let mask = _mm_set_ps(0.0, -0.0, 0.0, -0.0);
                    _mm_xor_ps(sw, mask)
                }
            };

            let fwd = sign < 0;

            // Bit-reversal load (1 complex per XMM, broadcast to [re,im,re,im])
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

            // Stage 2: span-2 butterflies with W4 twiddles
            for group in (0..8usize).step_by(4) {
                let t = a[group + 2];
                a[group + 2] = _mm_sub_ps(a[group], t);
                a[group]     = _mm_add_ps(a[group], t);

                let t = a[group + 3];
                let t_tw = rotate_pm_i(t, fwd);
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
                // swap [re,im] -> [im,re] (within each pair)
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
                    // swap sum to get [im+re, re+im, im+re, re+im] at positions 0,1
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
                let t_tw = rotate_pm_i(t, fwd);
                a[6] = _mm_sub_ps(a[2], t_tw);
                a[2] = _mm_add_ps(a[2], t_tw);
            }

            // k=3: W8^3
            {
                let v = a[7];
                let sw = _mm_shuffle_ps(v, v, 0b00_01_00_01); // [im,re,im,re]
                let t_tw = if fwd {
                    // [(im-re)/√2, -(re+im)/√2] repeated
                    let diff = _mm_sub_ps(sw, v);  // [im-re, re-im, im-re, re-im]
                    let neg_sum = _mm_xor_ps(
                        _mm_add_ps(v, sw),
                        _mm_set1_ps(-0.0),
                    ); // [-(re+im), -(im+re), -(re+im), -(im+re)]
                    // interleave: [diff[0], neg_sum[0], diff[1], neg_sum[1]]
                    //           = [im-re, -(re+im), re-im, -(im+re)]
                    let combined = _mm_unpacklo_ps(diff, neg_sum);
                    // broadcast low pair: [im-re, -(re+im), im-re, -(re+im)]
                    _mm_mul_ps(_mm_shuffle_ps(combined, combined, 0b01_00_01_00), inv_sqrt2)
                } else {
                    // [-(re+im)/√2, (re-im)/√2] repeated
                    let neg_sum = _mm_xor_ps(
                        _mm_add_ps(v, sw),
                        _mm_set1_ps(-0.0),
                    ); // [-(re+im), -(im+re), -(re+im), -(im+re)]
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

            // Store in natural order (low 2 lanes per XMM)
            for i in 0..8usize {
                store_cx(ptr.add(i * 2), a[i]);
            }
        }
    }
}
