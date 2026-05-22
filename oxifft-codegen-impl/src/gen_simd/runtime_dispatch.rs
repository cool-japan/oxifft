//! Centralized ISA runtime dispatch codegen for `OxiFFT` SIMD codelets.
//!
//! This module generates **cached** runtime ISA dispatchers that extend the
//! inline dispatchers in [`super`] with an `AtomicU8`-based ISA level cache.
//!
//! # Motivation
//!
//! The basic dispatchers emitted by `super::gen_dispatcher` perform
//! `is_x86_feature_detected!` / `is_aarch64_feature_detected!` on every call.
//! While each call is cheap (typically one CPUID cache read), a hot codelet
//! invoked millions of times per second may benefit from the cached path, which
//! replaces repeated feature probes with a single `AtomicU8` load.
//!
//! # Priority order (high → low)
//!
//! ```text
//! x86_64: AVX-512F > AVX2+FMA > AVX > SSE2 > scalar
//! aarch64: NEON > scalar
//! other: scalar
//! ```
//!
//! # Generated code shape
//!
//! For each `(size, precision)` pair, the proc-macro emits:
//! - ISA level constants (`ISA_SCALAR`, `ISA_SSE2`, … `ISA_UNDETECTED`)
//! - A `static DETECTED_ISA_{size}_{TY}: AtomicU8` initialized to `ISA_UNDETECTED`
//! - A private `detect_isa_{size}_{ty}() -> u8` function that probes the CPU once
//! - A public `{fn_name}_cached(data, sign)` dispatcher that reads the cache first
//!
//! # Proc-macro entry
//!
//! ```ignore
//! // Generates a cached dispatcher for size-4 f32.
//! gen_dispatcher_codelet!(size = 4, ty = f32);
//! ```

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    LitInt, Token,
};

pub use super::multi_transform::Precision;

// ============================================================================
// Public types
// ============================================================================

/// Configuration for a cached runtime ISA dispatcher codelet.
#[derive(Debug, Clone, Copy)]
pub struct DispatcherConfig {
    /// DFT size — must be one of 2, 4, 8, or 16.
    pub size: usize,
    /// Floating-point precision.
    pub precision: Precision,
}

// ============================================================================
// ISA level constants (used in generated code and in host-detection helper)
// ============================================================================

/// ISA level for scalar fallback.
pub const ISA_SCALAR: u8 = 0;
/// ISA level for SSE2.
pub const ISA_SSE2: u8 = 1;
/// ISA level for pure AVX (no FMA, no AVX2).
pub const ISA_AVX: u8 = 2;
/// ISA level for AVX2 + FMA.
pub const ISA_AVX2_FMA: u8 = 3;
/// ISA level for AVX-512F.
pub const ISA_AVX512: u8 = 4;
/// ISA level for NEON (aarch64).
pub const ISA_NEON: u8 = 5;
/// Sentinel: ISA not yet detected (stored in the `AtomicU8` before first call).
pub const ISA_UNDETECTED: u8 = 255;

// ============================================================================
// Host-detection helper (used by tests and by the generated detection code)
// ============================================================================

/// Detect the best ISA available on the current host at runtime.
///
/// Returns one of the `ISA_*` constants.  Never returns `ISA_UNDETECTED`.
///
/// This function is also used in the unit tests to validate that we always
/// detect a valid ISA on the host machine.
#[must_use]
pub fn detect_host_isa() -> u8 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return ISA_AVX512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return ISA_AVX2_FMA;
        }
        if is_x86_feature_detected!("avx") {
            return ISA_AVX;
        }
        if is_x86_feature_detected!("sse2") {
            return ISA_SSE2;
        }
        return ISA_SCALAR;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return ISA_NEON;
        }
        return ISA_SCALAR;
    }

    // All other architectures (wasm32, riscv, etc.)
    #[allow(unreachable_code)]
    ISA_SCALAR
}

// ============================================================================
// Code generation helpers
// ============================================================================

/// Build the `x86_64` ISA detection body emitted inside the detect function.
fn build_detect_x86_body() -> TokenStream {
    quote! {
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(feature = "avx512")]
            if is_x86_feature_detected!("avx512f") {
                return ISA_AVX512_LEVEL;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return ISA_AVX2_FMA_LEVEL;
            }
            if is_x86_feature_detected!("avx") {
                return ISA_AVX_LEVEL;
            }
            if is_x86_feature_detected!("sse2") {
                return ISA_SSE2_LEVEL;
            }
            return ISA_SCALAR_LEVEL;
        }
    }
}

/// Build the aarch64 ISA detection body emitted inside the detect function.
fn build_detect_aarch64_body() -> TokenStream {
    quote! {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return ISA_NEON_LEVEL;
            }
            return ISA_SCALAR_LEVEL;
        }
    }
}

/// Build the `x86_64` dispatch branches for the cached dispatcher body.
///
/// For size-16 f32 only AVX-512 is available; for size-16 f64 no x86 SIMD
/// path exists.  For all other sizes (2, 4, 8), all ISA levels are probed.
///
/// Each branch creates its own local `data_inner` reinterpretation so that
/// the raw-pointer slice never aliases the original `data` borrow.
fn build_x86_64_branches(config: DispatcherConfig) -> TokenStream {
    let size = config.size;
    let ty_str = config.precision.type_str();
    let ty_tokens: TokenStream = ty_str
        .parse()
        .unwrap_or_else(|_| unreachable!("ty_str is always f32 or f64"));
    let avx512_fn = format_ident!("codelet_simd_{}_avx512_{}", size, ty_str);
    let avx2_fn = format_ident!("codelet_simd_{}_avx2_{}", size, ty_str);
    let sse2_fn = format_ident!("codelet_simd_{}_sse2_{}", size, ty_str);

    if size == 16 {
        if config.precision == Precision::F32 {
            return quote! {
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                {
                    if cached_level == ISA_AVX512_LEVEL {
                        // Safety: avx512f detected at runtime.
                        // Layout: Complex<f32> is #[repr(C)] (re, im) — same as [f32; 2*N].
                        let data_len = data.len() * 2;
                        let data_ptr = data.as_mut_ptr().cast::<#ty_tokens>();
                        let data_inner = unsafe { core::slice::from_raw_parts_mut(data_ptr, data_len) };
                        unsafe { super::#avx512_fn(data_inner, sign); }
                        return;
                    }
                }
            };
        }
        // size-16 f64: no dedicated SIMD on x86_64
        return quote! {};
    }

    // Pure-AVX path only exists for f64 (no pure-AVX f32 emitter)
    let avx_branch = if config.precision == Precision::F64 {
        let avx_f64_fn = format_ident!("codelet_simd_{}_avx_f64", size);
        quote! {
            if cached_level == ISA_AVX_LEVEL {
                // Safety: avx detected at runtime; function has #[target_feature(enable = "avx")].
                let data_len = data.len() * 2;
                let data_ptr = data.as_mut_ptr().cast::<#ty_tokens>();
                let data_inner = unsafe { core::slice::from_raw_parts_mut(data_ptr, data_len) };
                unsafe { super::#avx_f64_fn(data_inner, sign); }
                return;
            }
        }
    } else {
        quote! {}
    };

    quote! {
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(feature = "avx512")]
            if cached_level == ISA_AVX512_LEVEL {
                // Safety: avx512f detected at runtime.
                let data_len = data.len() * 2;
                let data_ptr = data.as_mut_ptr().cast::<#ty_tokens>();
                let data_inner = unsafe { core::slice::from_raw_parts_mut(data_ptr, data_len) };
                unsafe { super::#avx512_fn(data_inner, sign); }
                return;
            }
            if cached_level == ISA_AVX2_FMA_LEVEL {
                // Safety: avx2+fma detected at runtime.
                let data_len = data.len() * 2;
                let data_ptr = data.as_mut_ptr().cast::<#ty_tokens>();
                let data_inner = unsafe { core::slice::from_raw_parts_mut(data_ptr, data_len) };
                unsafe { super::#avx2_fn(data_inner, sign); }
                return;
            }
            #avx_branch
            if cached_level == ISA_SSE2_LEVEL {
                // Safety: sse2 detected at runtime.
                let data_len = data.len() * 2;
                let data_ptr = data.as_mut_ptr().cast::<#ty_tokens>();
                let data_inner = unsafe { core::slice::from_raw_parts_mut(data_ptr, data_len) };
                unsafe { super::#sse2_fn(data_inner, sign); }
                return;
            }
        }
    }
}

/// Build the aarch64 dispatch branch for the cached dispatcher body.
///
/// Size-16 has no NEON path.
fn build_aarch64_branch(config: DispatcherConfig) -> TokenStream {
    if config.size == 16 {
        return quote! {};
    }
    let ty_str = config.precision.type_str();
    let ty_tokens: TokenStream = ty_str
        .parse()
        .unwrap_or_else(|_| unreachable!("ty_str is always f32 or f64"));
    let neon_fn = format_ident!("codelet_simd_{}_neon_{}", config.size, ty_str);
    quote! {
        #[cfg(target_arch = "aarch64")]
        {
            if cached_level == ISA_NEON_LEVEL {
                // Safety: NEON detected at runtime; mandatory on aarch64.
                let data_len = data.len() * 2;
                let data_ptr = data.as_mut_ptr().cast::<#ty_tokens>();
                let data_inner = unsafe { core::slice::from_raw_parts_mut(data_ptr, data_len) };
                unsafe { super::#neon_fn(data_inner, sign); }
                return;
            }
        }
    }
}

// ============================================================================
// Code generation
// ============================================================================

/// Generate a cached runtime ISA dispatcher `TokenStream`.
///
/// The emitted code:
/// 1. Declares ISA constants (only once per invocation; the caller is
///    responsible for deduplication if multiple sizes share a module).
/// 2. Declares a `static DETECTED_ISA_{size}_{ty}: AtomicU8`.
/// 3. Emits a private `detect_isa_{size}_{ty}() -> u8` probe function.
/// 4. Emits a public `codelet_simd_{size}_cached_{ty}(data, sign)` dispatcher.
///
/// The dispatcher delegates to the same arch-specific inner functions that the
/// basic (uncached) dispatcher in [`super`] uses, following the exact same
/// naming convention: `codelet_simd_{size}_{isa}_{ty}`.
///
/// # Errors
///
/// Returns `syn::Error` when `config.size` is not one of 2, 4, 8, or 16.
#[allow(clippy::too_many_lines)] // reason: token-stream assembly requires many local variables
pub fn generate_dispatcher(config: DispatcherConfig) -> Result<TokenStream, syn::Error> {
    let size = config.size;
    if !matches!(size, 2 | 4 | 8 | 16) {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "gen_dispatcher_codelet: unsupported size {size} (expected one of 2, 4, 8, 16)"
            ),
        ));
    }

    let ty_str = config.precision.type_str();
    let ty_upper = ty_str.to_uppercase();
    let size_str = size.to_string();

    // AtomicU8 static name: DETECTED_ISA_4_F32
    let static_name = format_ident!("DETECTED_ISA_{}_{}", size_str, ty_upper);
    // Detect function name: detect_isa_4_f32
    let detect_fn = format_ident!("detect_isa_{}_{}", size_str, ty_str);
    // Cached dispatcher name: codelet_simd_4_cached_f32
    let cached_fn = format_ident!("codelet_simd_{}_cached_{}", size_str, ty_str);
    // Scalar fallback name: codelet_simd_4_scalar
    let scalar_fn = format_ident!("codelet_simd_{}_scalar", size);

    let detect_x86_body = build_detect_x86_body();
    let detect_aarch64_body = build_detect_aarch64_body();
    let x86_64_branches = build_x86_64_branches(config);
    let aarch64_branch = build_aarch64_branch(config);

    let ty_tokens: TokenStream = ty_str
        .parse()
        .unwrap_or_else(|_| unreachable!("ty_str is always f32 or f64"));

    let fn_doc = format!(
        "Cached runtime ISA dispatcher for size-{size} DFT ({ty_str}).\n\n\
         On first call, probes CPU features and stores the ISA level in a\n\
         thread-safe `AtomicU8` static.  Subsequent calls read the cache with\n\
         `Relaxed` ordering (benign-racy: all threads converge on the same answer).\n\n\
         Dispatch priority on `x86_64`: AVX-512F > AVX2+FMA > AVX > SSE2 > scalar.\n\
         Dispatch priority on `aarch64`: NEON > scalar.\n\
         Other architectures fall through to the scalar codelet."
    );

    let size_lit = size;

    Ok(quote! {
        // ISA level constants (private to the generated scope)
        const ISA_SCALAR_LEVEL:     u8 = 0;
        const ISA_SSE2_LEVEL:       u8 = 1;
        const ISA_AVX_LEVEL:        u8 = 2;
        const ISA_AVX2_FMA_LEVEL:   u8 = 3;
        const ISA_AVX512_LEVEL:     u8 = 4;
        const ISA_NEON_LEVEL:       u8 = 5;
        const ISA_UNDETECTED_LEVEL: u8 = 255;

        /// Cached ISA level for this (size, precision) pair.
        ///
        /// Initialized to `ISA_UNDETECTED_LEVEL`.  Written once on first dispatch call.
        static #static_name: core::sync::atomic::AtomicU8 =
            core::sync::atomic::AtomicU8::new(ISA_UNDETECTED_LEVEL);

        /// Probe the CPU once and return the best ISA level for this target.
        fn #detect_fn() -> u8 {
            #detect_x86_body
            #detect_aarch64_body
            #[allow(unreachable_code)]
            ISA_SCALAR_LEVEL
        }

        #[doc = #fn_doc]
        #[inline]
        pub fn #cached_fn(
            data: &mut [crate::kernel::Complex<#ty_tokens>],
            sign: i32,
        ) {
            debug_assert!(
                data.len() >= #size_lit,
                "codelet_simd_{}_cached_{}: need >= {} elements, got {}",
                #size_lit,
                stringify!(#ty_tokens),
                #size_lit,
                data.len(),
            );

            // Load cached ISA; detect on first call.
            let cached_level = {
                let level = #static_name.load(core::sync::atomic::Ordering::Relaxed);
                if level == ISA_UNDETECTED_LEVEL {
                    let detected = #detect_fn();
                    // Relaxed store: benign-racy — all threads converge on the same value.
                    #static_name.store(detected, core::sync::atomic::Ordering::Relaxed);
                    detected
                } else {
                    level
                }
            };

            // Architecture-specific SIMD paths.
            //
            // data_inner is created inside each cfg block so that the raw-pointer
            // reinterpretation and the original `data` borrow never overlap.
            // The scalar fallback uses `data` directly — no aliasing.
            #x86_64_branches
            #aarch64_branch

            // Scalar fallback: use the original Complex slice directly.
            // No reinterpretation needed — the scalar codelet accepts Complex<T>.
            super::#scalar_fn(data, sign);
        }
    })
}

// ============================================================================
// Proc-macro parse input
// ============================================================================

/// Parsed arguments from `gen_dispatcher_codelet!(size = 4, ty = f32)`.
struct MacroArgs {
    size: usize,
    precision: Precision,
}

impl Parse for MacroArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut size: Option<usize> = None;
        let mut precision: Option<Precision> = None;

        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            let _eq: Token![=] = input.parse()?;
            match key.to_string().as_str() {
                "size" => {
                    let lit: LitInt = input.parse()?;
                    size = Some(lit.base10_parse::<usize>().map_err(|_| {
                        syn::Error::new(lit.span(), "expected an integer literal for `size`")
                    })?);
                }
                "ty" => {
                    let ident: syn::Ident = input.parse()?;
                    precision = Some(match ident.to_string().as_str() {
                        "f32" => Precision::F32,
                        "f64" => Precision::F64,
                        other => {
                            return Err(syn::Error::new(
                                ident.span(),
                                format!("unknown ty `{other}`, expected f32 or f64"),
                            ));
                        }
                    });
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown key `{other}`, expected one of: size, ty"),
                    ));
                }
            }
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        let size = size.ok_or_else(|| {
            syn::Error::new(proc_macro2::Span::call_site(), "missing `size` argument")
        })?;
        let precision = precision.ok_or_else(|| {
            syn::Error::new(proc_macro2::Span::call_site(), "missing `ty` argument")
        })?;

        Ok(Self { size, precision })
    }
}

/// Entry point for the `gen_dispatcher_codelet!` proc-macro.
///
/// Parses `size = N, ty = TY` and calls [`generate_dispatcher`].
///
/// # Errors
///
/// Returns a `syn::Error` when the input does not parse as valid key-value
/// pairs, a required key is missing, or `size` / `ty` have unsupported values.
pub fn generate_from_macro(input: TokenStream) -> Result<TokenStream, syn::Error> {
    let args: MacroArgs = syn::parse2(input)?;
    generate_dispatcher(DispatcherConfig {
        size: args.size,
        precision: args.precision,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── DispatcherConfig construction ─────────────────────────────────────

    #[test]
    fn test_dispatcher_config_valid_f32() {
        let config = DispatcherConfig {
            size: 4,
            precision: Precision::F32,
        };
        assert_eq!(config.size, 4);
        assert_eq!(config.precision, Precision::F32);
    }

    #[test]
    fn test_dispatcher_config_valid_f64() {
        let config = DispatcherConfig {
            size: 8,
            precision: Precision::F64,
        };
        assert_eq!(config.size, 8);
        assert_eq!(config.precision, Precision::F64);
    }

    // ── ISA constants ─────────────────────────────────────────────────────

    #[test]
    fn test_isa_constants_are_ordered() {
        // Validate ordering as compile-time assertions embedded in a constant.
        const _: () = {
            assert!(ISA_SCALAR < ISA_SSE2);
            assert!(ISA_SSE2 < ISA_AVX);
            assert!(ISA_AVX < ISA_AVX2_FMA);
            assert!(ISA_AVX2_FMA < ISA_AVX512);
            assert!(ISA_NEON != ISA_SCALAR);
            assert!(ISA_UNDETECTED == 255);
        };
    }

    // ── generate_dispatcher: TokenStream checks ───────────────────────────

    #[test]
    fn test_generate_dispatcher_nonempty() {
        let ts = generate_dispatcher(DispatcherConfig {
            size: 4,
            precision: Precision::F32,
        })
        .expect("should generate for size 4 f32");
        assert!(!ts.is_empty(), "TokenStream must not be empty");
    }

    #[test]
    fn test_generate_dispatcher_nonempty_f64() {
        let ts = generate_dispatcher(DispatcherConfig {
            size: 8,
            precision: Precision::F64,
        })
        .expect("should generate for size 8 f64");
        assert!(!ts.is_empty(), "TokenStream must not be empty");
    }

    #[test]
    fn test_generate_dispatcher_contains_is_x86_feature_detected() {
        let ts = generate_dispatcher(DispatcherConfig {
            size: 4,
            precision: Precision::F32,
        })
        .expect("should generate");
        let s = ts.to_string();
        assert!(
            s.contains("is_x86_feature_detected"),
            "generated code must contain is_x86_feature_detected! macro; got snippet: {}",
            &s[..s.len().min(500)]
        );
    }

    #[test]
    fn test_generate_dispatcher_contains_atomic_u8() {
        let ts = generate_dispatcher(DispatcherConfig {
            size: 4,
            precision: Precision::F32,
        })
        .expect("should generate");
        let s = ts.to_string();
        assert!(
            s.contains("AtomicU8"),
            "generated code must contain AtomicU8 static; got snippet: {}",
            &s[..s.len().min(500)]
        );
    }

    #[test]
    fn test_generate_dispatcher_contains_isa_undetected() {
        let ts = generate_dispatcher(DispatcherConfig {
            size: 4,
            precision: Precision::F32,
        })
        .expect("should generate");
        let s = ts.to_string();
        assert!(
            s.contains("ISA_UNDETECTED_LEVEL") || s.contains("255"),
            "generated code must reference ISA_UNDETECTED_LEVEL sentinel"
        );
    }

    #[test]
    fn test_generate_dispatcher_function_name_size4_f32() {
        let ts = generate_dispatcher(DispatcherConfig {
            size: 4,
            precision: Precision::F32,
        })
        .expect("should generate");
        let s = ts.to_string();
        assert!(
            s.contains("codelet_simd_4_cached_f32"),
            "expected cached dispatcher name in output; snippet: {}",
            &s[..s.len().min(400)]
        );
    }

    #[test]
    fn test_generate_dispatcher_function_name_size8_f64() {
        let ts = generate_dispatcher(DispatcherConfig {
            size: 8,
            precision: Precision::F64,
        })
        .expect("should generate");
        let s = ts.to_string();
        assert!(
            s.contains("codelet_simd_8_cached_f64"),
            "expected cached dispatcher name in output"
        );
    }

    #[test]
    fn test_generate_dispatcher_all_valid_sizes() {
        for &size in &[2_usize, 4, 8, 16] {
            for &prec in &[Precision::F32, Precision::F64] {
                let result = generate_dispatcher(DispatcherConfig {
                    size,
                    precision: prec,
                });
                assert!(
                    result.is_ok(),
                    "size={size} prec={prec:?} should succeed, got: {:?}",
                    result.err()
                );
            }
        }
    }

    #[test]
    fn test_generate_dispatcher_unsupported_size_returns_error() {
        let result = generate_dispatcher(DispatcherConfig {
            size: 3,
            precision: Precision::F32,
        });
        assert!(result.is_err(), "size 3 must return Err");
    }

    #[test]
    fn test_generate_dispatcher_unsupported_size_6_returns_error() {
        let result = generate_dispatcher(DispatcherConfig {
            size: 6,
            precision: Precision::F64,
        });
        assert!(result.is_err(), "size 6 must return Err");
    }

    // ── detect_host_isa ───────────────────────────────────────────────────

    #[test]
    fn test_dispatcher_isa_detection() {
        // On the host machine, detect_host_isa() must always return a valid ISA level.
        // On aarch64 macOS (Apple Silicon) this should be ISA_NEON.
        // On x86_64 this should be ISA_SSE2 or higher.
        let isa = detect_host_isa();
        assert_ne!(
            isa, ISA_UNDETECTED,
            "detect_host_isa must never return ISA_UNDETECTED (255)"
        );
        // Must be one of the known constants
        assert!(
            matches!(
                isa,
                ISA_SCALAR | ISA_SSE2 | ISA_AVX | ISA_AVX2_FMA | ISA_AVX512 | ISA_NEON
            ),
            "detect_host_isa returned unknown level {isa}"
        );
    }

    #[test]
    fn test_detect_host_isa_is_deterministic() {
        let first = detect_host_isa();
        let second = detect_host_isa();
        assert_eq!(first, second, "detect_host_isa must be deterministic");
    }

    // ── generate_from_macro ───────────────────────────────────────────────

    #[test]
    fn test_generate_from_macro_size4_f32() {
        let input: TokenStream = "size = 4, ty = f32".parse().expect("valid token stream");
        let result = generate_from_macro(input);
        assert!(
            result.is_ok(),
            "size=4 ty=f32 must succeed: {:?}",
            result.err()
        );
        let s = result.expect("TokenStream").to_string();
        assert!(
            s.contains("codelet_simd_4_cached_f32"),
            "must contain cached dispatcher name"
        );
    }

    #[test]
    fn test_generate_from_macro_size8_f64() {
        let input: TokenStream = "size = 8, ty = f64".parse().expect("valid token stream");
        let result = generate_from_macro(input);
        assert!(
            result.is_ok(),
            "size=8 ty=f64 must succeed: {:?}",
            result.err()
        );
        let s = result.expect("TokenStream").to_string();
        assert!(
            s.contains("codelet_simd_8_cached_f64"),
            "must contain cached dispatcher name"
        );
    }

    #[test]
    fn test_generate_from_macro_size2_f64() {
        let input: TokenStream = "size = 2, ty = f64".parse().expect("valid token stream");
        let result = generate_from_macro(input);
        assert!(result.is_ok(), "size=2 ty=f64 must succeed");
    }

    #[test]
    fn test_generate_from_macro_size16_f32() {
        let input: TokenStream = "size = 16, ty = f32".parse().expect("valid token stream");
        let result = generate_from_macro(input);
        assert!(result.is_ok(), "size=16 ty=f32 must succeed");
    }

    #[test]
    fn test_generate_from_macro_missing_size_returns_error() {
        let input: TokenStream = "ty = f32".parse().expect("valid token stream");
        let result = generate_from_macro(input);
        assert!(result.is_err(), "missing size must return error");
    }

    #[test]
    fn test_generate_from_macro_missing_ty_returns_error() {
        let input: TokenStream = "size = 4".parse().expect("valid token stream");
        let result = generate_from_macro(input);
        assert!(result.is_err(), "missing ty must return error");
    }

    #[test]
    fn test_generate_from_macro_unknown_ty_returns_error() {
        let input: TokenStream = "size = 4, ty = f16".parse().expect("valid token stream");
        let result = generate_from_macro(input);
        assert!(result.is_err(), "unknown ty must return error");
    }

    #[test]
    fn test_generate_from_macro_unknown_key_returns_error() {
        let input: TokenStream = "size = 4, ty = f32, isa = avx2"
            .parse()
            .expect("valid token stream");
        let result = generate_from_macro(input);
        assert!(result.is_err(), "unknown key must return error");
    }

    #[test]
    fn test_generate_from_macro_unsupported_size_returns_error() {
        let input: TokenStream = "size = 5, ty = f32".parse().expect("valid token stream");
        let result = generate_from_macro(input);
        assert!(result.is_err(), "size=5 must return error");
    }
}
