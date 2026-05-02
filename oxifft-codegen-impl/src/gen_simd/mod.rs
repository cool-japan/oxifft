//! SIMD codelet generation.
//!
//! Generates architecture-aware SIMD FFT codelets with multi-architecture dispatch.
//! At compile time, generates:
//! - AVX-512F variant (512-bit, `8×f64` / `16×f32`) for `x86_64`
//! - AVX2+FMA variant (256-bit, `4×f64`) for `x86_64`
//! - Pure-AVX variant (256-bit, `4×f64`, no FMA, no AVX2) for `x86_64`
//! - SSE2 variant (128-bit, `2×f64`) for `x86_64`
//! - NEON variant (128-bit, `2×f64`) for `aarch64`
//! - Scalar fallback for all architectures
//!
//! The dispatcher function selects the best SIMD path at runtime using
//! `is_x86_feature_detected!` (`x86_64`) or compile-time cfg (`aarch64`).
//!
//! Probe order for `x86_64`: AVX-512F > AVX2+FMA > AVX > SSE2 > scalar.
//! AVX-512F is probed first (when the host supports it) to enable
//! `_mm512_fmadd_pd`/`_mm512_fmsub_pd` based butterfly arithmetic.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::LitInt;

mod avx;
mod avx2;
mod avx512;
mod neon;
mod scalar;
mod sse2;

pub mod multi_transform;
pub mod runtime_dispatch;

/// Generate a SIMD-optimized codelet for the given FFT size.
///
/// Supports sizes 2, 4, 8 for all ISAs and size 16 (f32 only, AVX-512F or scalar).
/// The macro generates:
/// - A public dispatcher that picks the best SIMD path at runtime
/// - Architecture-specific inner functions with `#[target_feature]`
/// - A generic scalar fallback
///
/// AVX-512F is probed before AVX2+FMA for sizes that have AVX-512 emitters.
///
/// # Errors
/// Returns a `syn::Error` when the input does not parse as a valid size literal,
/// or when the size is not in the supported set {2, 4, 8, 16}.
pub fn generate(input: TokenStream) -> Result<TokenStream, syn::Error> {
    let size: LitInt = syn::parse2(input)?;
    let n: usize = size.base10_parse().map_err(|_| {
        syn::Error::new(
            size.span(),
            "gen_simd_codelet: expected an integer size literal",
        )
    })?;

    match n {
        2 => Ok(gen_simd_size_2()),
        4 => Ok(gen_simd_size_4()),
        8 => Ok(gen_simd_size_8()),
        16 => Ok(gen_simd_size_16()),
        _ => Err(syn::Error::new(
            size.span(),
            format!("gen_simd_codelet: unsupported size {n} (expected one of 2, 4, 8, 16)"),
        )),
    }
}

/// Generate size-2 SIMD butterfly codelet.
///
/// Size-2 butterfly: out[0] = a + b, out[1] = a - b
///
/// SIMD strategy:
/// - AVX-512F (`x86_64`): 256-bit YMM under avx512f feature umbrella, f64 only
/// - SSE2/AVX2 (`x86_64`): `__m128d` / `__m128` for f64/f32, vector add/sub
/// - NEON (`aarch64`): `float64x2_t` / `float32x2_t` for f64/f32, vector add/sub
fn gen_simd_size_2() -> TokenStream {
    let dispatcher = gen_dispatcher(2);
    let scalar = scalar::gen_scalar_size_2();
    let sse2_f64 = sse2::gen_sse2_size_2();
    let sse2_f32 = sse2::gen_sse2_size_2_f32();
    let plain_avx_f64 = avx::gen_avx_size_2_f64();
    let avx2_f64 = avx2::gen_avx2_size_2();
    let avx2_f32 = avx2::gen_avx2_size_2_f32();
    let avx512_f64 = avx512::gen_avx512_size_2_f64();
    let avx512_f32 = avx512::gen_avx512_size_2_f32();
    let neon_f64 = neon::gen_neon_size_2();
    let neon_f32 = neon::gen_neon_size_2_f32();

    quote! {
        #dispatcher
        #scalar
        #sse2_f64
        #sse2_f32
        #plain_avx_f64
        #avx2_f64
        #avx2_f32
        #avx512_f64
        #avx512_f32
        #neon_f64
        #neon_f32
    }
}

/// Generate size-4 SIMD radix-4 codelet.
///
/// Size-4 FFT: radix-4 butterfly with sign-dependent ±i rotation.
///
/// SIMD strategy:
/// - AVX-512F (`x86_64`): 256-bit f64 + f32 butterfly under avx512f feature
/// - SSE2/AVX2 (`x86_64`): `__m128d` / `__m128` for f64/f32, shuffle-based rotation
/// - NEON (`aarch64`): `float64x2_t` / `float32x2_t` for f64/f32, ext-based rotation
fn gen_simd_size_4() -> TokenStream {
    let dispatcher = gen_dispatcher(4);
    let scalar = scalar::gen_scalar_size_4();
    let sse2_f64 = sse2::gen_sse2_size_4();
    let sse2_f32 = sse2::gen_sse2_size_4_f32();
    let plain_avx_f64 = avx::gen_avx_size_4_f64();
    let avx2_f64 = avx2::gen_avx2_size_4();
    let avx2_f32 = avx2::gen_avx2_size_4_f32();
    let avx512_f64 = avx512::gen_avx512_size_4_f64();
    let avx512_f32 = avx512::gen_avx512_size_4_f32();
    let neon_f64 = neon::gen_neon_size_4();
    let neon_f32 = neon::gen_neon_size_4_f32();

    quote! {
        #dispatcher
        #scalar
        #sse2_f64
        #sse2_f32
        #plain_avx_f64
        #avx2_f64
        #avx2_f32
        #avx512_f64
        #avx512_f32
        #neon_f64
        #neon_f32
    }
}

/// Generate size-8 SIMD radix-8 codelet.
///
/// Size-8 FFT: radix-2 DIT with 3 butterfly stages.
///
/// SIMD strategy:
/// - AVX-512F (`x86_64`): 256-bit + FMA via ZMM promotion for twiddles
/// - SSE2/AVX2 (`x86_64`): `__m128d` / `__m128` for f64/f32, FMA twiddles
/// - NEON (`aarch64`): `float64x2_t` / `float32x2_t` for f64/f32, FMA twiddles
fn gen_simd_size_8() -> TokenStream {
    let dispatcher = gen_dispatcher(8);
    let scalar = scalar::gen_scalar_size_8();
    let sse2_f64 = sse2::gen_sse2_size_8();
    let sse2_f32 = sse2::gen_sse2_size_8_f32();
    let plain_avx_f64 = avx::gen_avx_size_8_f64();
    let avx2_f64 = avx2::gen_avx2_size_8();
    let avx2_f32 = avx2::gen_avx2_size_8_f32();
    let avx512_f64 = avx512::gen_avx512_size_8_f64();
    let avx512_f32 = avx512::gen_avx512_size_8_f32();
    let neon_f64 = neon::gen_neon_size_8();
    let neon_f32 = neon::gen_neon_size_8_f32();

    quote! {
        #dispatcher
        #scalar
        #sse2_f64
        #sse2_f32
        #plain_avx_f64
        #avx2_f64
        #avx2_f32
        #avx512_f64
        #avx512_f32
        #neon_f64
        #neon_f32
    }
}

/// Generate size-16 SIMD radix-2 DIT codelet (f32 only via AVX-512F).
///
/// Size-16 FFT: radix-2 DIT with 4 butterfly stages.
///
/// SIMD strategy:
/// - AVX-512F (`x86_64`): full `__m512` 16-lane f32 butterfly with FMA W16 twiddles
/// - Scalar fallback for all other architectures (AVX2/SSE2/NEON lack this size)
fn gen_simd_size_16() -> TokenStream {
    let dispatcher = gen_dispatcher_16();
    let scalar = scalar::gen_scalar_size_16();
    let avx512_f32 = avx512::gen_avx512_size_16_f32();

    quote! {
        #dispatcher
        #scalar
        #avx512_f32
    }
}

// ---------------------------------------------------------------------------
// Dispatcher generation
// ---------------------------------------------------------------------------

/// Generate the public dispatcher function for sizes 2, 4, 8 (all ISAs).
///
/// Priority on `x86_64` (f64): AVX-512F > AVX2+FMA > AVX > SSE2 > scalar.
/// Priority on `x86_64` (f32): AVX-512F > AVX2+FMA > SSE2 > scalar (no pure-AVX f32 path).
/// Priority on `aarch64`: NEON > scalar.
///
/// The dispatcher:
/// 1. Checks `T` via `core::any::TypeId` (since `Float: 'static`):
///    - `f64`: uses f64 SIMD path
///    - `f32`: uses f32 SIMD path
///    - other: scalar fallback
/// 2. On `x86_64`: probes AVX-512F first, then AVX2+FMA, then SSE2
/// 3. On `aarch64`: uses NEON unconditionally
/// 4. Falls back to the generic scalar implementation
fn gen_dispatcher(n: usize) -> proc_macro2::TokenStream {
    let fn_name = format_ident!("codelet_simd_{}", n);
    let scalar_name = format_ident!("codelet_simd_{}_scalar", n);
    let avx512_f64_name = format_ident!("codelet_simd_{}_avx512_f64", n);
    let avx2_f64_name = format_ident!("codelet_simd_{}_avx2_f64", n);
    let plain_avx_fn_name = format_ident!("codelet_simd_{}_avx_f64", n);
    let sse2_f64_name = format_ident!("codelet_simd_{}_sse2_f64", n);
    let neon_f64_name = format_ident!("codelet_simd_{}_neon_f64", n);
    let avx512_f32_name = format_ident!("codelet_simd_{}_avx512_f32", n);
    let avx2_f32_name = format_ident!("codelet_simd_{}_avx2_f32", n);
    let sse2_f32_name = format_ident!("codelet_simd_{}_sse2_f32", n);
    let neon_f32_name = format_ident!("codelet_simd_{}_neon_f32", n);
    let n_lit = n;
    let doc = format!(
        "Size-{n} SIMD-optimized FFT codelet with architecture dispatch.\n\n\
         Automatically selects the best SIMD path at runtime:\n\
         - x86_64: AVX-512F > AVX2+FMA > AVX > SSE2 > scalar  (f64; f32 uses AVX2/SSE2/512)\n\
         - aarch64: NEON > scalar                               (both f64 and f32)\n\
         - other: scalar fallback"
    );

    quote! {
        #[doc = #doc]
        #[inline]
        pub fn #fn_name<T: crate::kernel::Float>(
            data: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(
                data.len() >= #n_lit,
                "codelet_simd_{}: need >= {} elements, got {}",
                #n_lit,
                #n_lit,
                data.len(),
            );

            // Fast path: f64 SIMD
            if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f64>() {
                // Safety: Complex<T> is #[repr(C)] with (re, im) fields.
                // When T == f64, &mut [Complex<f64>] has the same layout as
                // &mut [f64] with twice the length: [re0, im0, re1, im1, ...].
                let len = data.len() * 2;
                let ptr = data.as_mut_ptr().cast::<f64>();
                let f64_data = unsafe { core::slice::from_raw_parts_mut(ptr, len) };

                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx512f") {
                        // Safety: AVX-512F detected, pointer valid for len f64s
                        unsafe { #avx512_f64_name(f64_data, sign); }
                        return;
                    }
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        // Safety: AVX2+FMA detected, pointer valid for len f64s
                        unsafe { #avx2_f64_name(f64_data, sign); }
                        return;
                    }
                    // Pure AVX (no FMA, no AVX2) — probe after AVX2+FMA, before SSE2
                    if is_x86_feature_detected!("avx") {
                        // Safety: AVX detected (superset of SSE2), pointer valid
                        unsafe { #plain_avx_fn_name(f64_data, sign); }
                        return;
                    }
                    if is_x86_feature_detected!("sse2") {
                        // Safety: SSE2 detected (guaranteed on x86_64), pointer valid
                        unsafe { #sse2_f64_name(f64_data, sign); }
                        return;
                    }
                }

                #[cfg(target_arch = "aarch64")]
                {
                    // NEON is mandatory on aarch64
                    unsafe { #neon_f64_name(f64_data, sign); }
                    return;
                }
            }

            // Fast path: f32 SIMD
            if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
                // Safety: Complex<T> is #[repr(C)] with (re, im) fields.
                // When T == f32, &mut [Complex<f32>] has the same layout as
                // &mut [f32] with twice the length: [re0, im0, re1, im1, ...].
                let len = data.len() * 2;
                let ptr = data.as_mut_ptr().cast::<f32>();
                let f32_data = unsafe { core::slice::from_raw_parts_mut(ptr, len) };

                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx512f") {
                        // Safety: AVX-512F detected
                        unsafe { #avx512_f32_name(f32_data, sign); }
                        return;
                    }
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        unsafe { #avx2_f32_name(f32_data, sign); }
                        return;
                    }
                    // f32 has no dedicated AVX (non-AVX2) path; fall through to SSE2
                    if is_x86_feature_detected!("sse2") {
                        unsafe { #sse2_f32_name(f32_data, sign); }
                        return;
                    }
                }

                #[cfg(target_arch = "aarch64")]
                {
                    // NEON is mandatory on aarch64
                    unsafe { #neon_f32_name(f32_data, sign); }
                    return;
                }
            }

            // Scalar fallback for other float types or unsupported architectures
            #scalar_name(data, sign);
        }
    }
}

/// Generate the dispatcher for size-16 (f32 only via AVX-512F; scalar fallback).
///
/// Size-16 is only available as f32 via AVX-512F. All other paths fall through
/// to the scalar implementation, including f64 (no size-16 f64 SIMD emitter).
fn gen_dispatcher_16() -> proc_macro2::TokenStream {
    let avx512_f32_name = format_ident!("codelet_simd_16_avx512_f32");
    let scalar_name = format_ident!("codelet_simd_16_scalar");

    quote! {
        /// Size-16 SIMD-optimized FFT codelet.
        ///
        /// Selects AVX-512F f32 path when available; otherwise falls back to scalar.
        /// No f64 SIMD path at size 16 (scalar is used instead).
        ///
        /// - x86_64 + avx512f: `__m512` 16-lane f32 butterfly with FMA twiddles
        /// - all other: scalar fallback
        #[inline]
        pub fn codelet_simd_16<T: crate::kernel::Float>(
            data: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(
                data.len() >= 16_usize,
                "codelet_simd_16: need >= 16 elements, got {}",
                data.len(),
            );

            // AVX-512F f32 path only
            if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
                let len = data.len() * 2;
                let ptr = data.as_mut_ptr().cast::<f32>();
                let f32_data = unsafe { core::slice::from_raw_parts_mut(ptr, len) };

                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx512f") {
                        // Safety: AVX-512F detected, pointer valid for len f32s
                        unsafe { #avx512_f32_name(f32_data, sign); }
                        return;
                    }
                }
            }

            // Scalar fallback for f64, other types, or no AVX-512F
            #scalar_name(data, sign);
        }
    }
}
