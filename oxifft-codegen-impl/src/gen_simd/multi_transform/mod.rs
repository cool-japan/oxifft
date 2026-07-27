//! Build-time codegen for SIMD vrank multi-transform codelets.
//!
//! A multi-transform codelet processes `V` DFTs of size `N` simultaneously.
//!
//! # Implementations
//!
//! - **SSE2 f32 (V=4)**: true SIMD for sizes 2 and 4 via `notw_{size}_v4_sse2_f32_soa`.
//! - **AVX2 f32 (V=8)**: true SIMD for sizes 2, 4, and 8 via `notw_{size}_v8_avx2_f32_soa`.
//! - **All other combos**: sequential scalar fallback over `AoS` layout.
//!
//! # Data layouts
//!
//! ## `AoS` (Array-of-Structs) — outer function signature
//!
//! Transforms are grouped into batches of `V`; each batch is a contiguous block
//! of `N * V * 2` floats. For global transform `g` (batch `b = g / V`, lane
//! `t = g % V`), element `k`:
//! ```text
//! data[b * (N*V*2) + t * 2 + k * (V*2) + 0]  = re of x[k] for transform g
//! data[b * (N*V*2) + t * 2 + k * (V*2) + 1]  = im of x[k] for transform g
//! ```
//! (A single batch, `g < V`, reduces to `data[k * V*2 + g * 2 + c]`.)
//!
//! ## `SoA` (Struct-of-Arrays) — inner SIMD function signature
//!
//! For `V` transforms of size `N` (only used internally by SIMD paths):
//! ```text
//! re_in[element_idx * v + transform_idx] = real  part of x[element_idx] for transform transform_idx
//! im_in[element_idx * v + transform_idx] = imag  part of x[element_idx] for transform transform_idx
//! ```
//!
//! The SIMD functions operate natively in `SoA`. The outer `AoS` function is
//! **always** the sequential scalar loop — it never calls the inner `SoA`
//! function. When a SIMD path exists the `_soa` companion is emitted *alongside*
//! the outer function for callers that want true SIMD throughput; they must call
//! the `_soa` function directly (see the `has_simd_impl` dispatch predicate).
//!
//! The outer `AoS` function only supports the canonical layout where
//! `istride == ostride == 2 * V`; the transform-packing offset is hard-coded to
//! that stride. Passing any other stride is a contract violation, guarded by a
//! `debug_assert_eq!` in the generated body.
//!
//! # Generated function signatures
//!
//! Outer (`AoS`, called by users):
//! ```text
//! pub unsafe fn notw_4_v8_avx2_f32(
//!     input: *const f32, output: *mut f32,
//!     istride: usize, ostride: usize, count: usize,
//! )
//! ```
//!
//! Inner `SoA` SIMD helpers (emitted alongside, for direct use or testing):
//! ```text
//! pub unsafe fn notw_4_v8_avx2_f32_soa(
//!     re_in: *const f32, im_in: *const f32,
//!     re_out: *mut f32, im_out: *mut f32,
//! )
//! ```

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    LitInt, Token,
};

mod scalar;
mod simd_avx2_f32;
mod simd_sse2_f32;

#[cfg(test)]
mod tests;

// ============================================================================
// Public types
// ============================================================================

/// Target ISA for a multi-transform codelet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdIsa {
    /// SSE2 (128-bit, 4 f32 or 2 f64 lanes).
    Sse2,
    /// AVX2+FMA (256-bit, 8 f32 or 4 f64 lanes).
    Avx2,
    /// Scalar fallback (no SIMD).
    Scalar,
}

impl SimdIsa {
    /// Number of scalar lanes for `f32`.
    #[must_use]
    pub const fn lanes_f32(self) -> usize {
        match self {
            Self::Sse2 => 4,
            Self::Avx2 => 8,
            Self::Scalar => 1,
        }
    }

    /// Number of scalar lanes for `f64`.
    #[must_use]
    pub const fn lanes_f64(self) -> usize {
        match self {
            Self::Sse2 => 2,
            Self::Avx2 => 4,
            Self::Scalar => 1,
        }
    }

    /// Lowercase name used in generated identifiers.
    #[must_use]
    pub const fn ident_str(self) -> &'static str {
        match self {
            Self::Sse2 => "sse2",
            Self::Avx2 => "avx2",
            Self::Scalar => "scalar",
        }
    }
}

/// Floating-point precision for a multi-transform codelet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// 32-bit single precision.
    F32,
    /// 64-bit double precision.
    F64,
}

impl Precision {
    /// Lowercase type name used in generated identifiers and code.
    #[must_use]
    pub const fn type_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

/// Configuration for a vectorized multi-transform codelet.
///
/// Describes a (DFT size, `ISA`, V, precision) tuple used to emit a
/// batch-of-V-transforms function at build time.
#[derive(Debug, Clone)]
pub struct MultiTransformConfig {
    /// DFT size — must be 2, 4, or 8.
    pub size: usize,
    /// Number of simultaneous transforms (lane count: 4 for SSE2 f32, 8 for AVX2 f32, etc.).
    pub v: usize,
    /// Target ISA.
    pub isa: SimdIsa,
    /// `f32` or `f64`.
    pub precision: Precision,
}

// ============================================================================
// SIMD dispatch logic
// ============================================================================

/// Returns `true` when the (`ISA`, precision, size) combination has a true SIMD
/// multi-transform implementation (`SoA` inner function).
///
/// - SSE2 f32: sizes 2 and 4
/// - AVX2 f32: sizes 2, 4, and 8
/// - All f64 combos: scalar fallback only
const fn has_simd_impl(isa: SimdIsa, precision: Precision, size: usize) -> bool {
    matches!(
        (isa, precision, size),
        (SimdIsa::Sse2, Precision::F32, 2 | 4) | (SimdIsa::Avx2, Precision::F32, 2 | 4 | 8)
    )
}

/// Emit the inner `SoA` SIMD function `TokenStream` for the given config.
///
/// Returns `None` if the config has no SIMD implementation.
fn gen_simd_inner(config: &MultiTransformConfig) -> Option<TokenStream> {
    match (config.isa, config.precision, config.size) {
        (SimdIsa::Sse2, Precision::F32, 2) => Some(simd_sse2_f32::gen_sse2_f32_v4_size2_soa()),
        (SimdIsa::Sse2, Precision::F32, 4) => Some(simd_sse2_f32::gen_sse2_f32_v4_size4_soa()),
        (SimdIsa::Avx2, Precision::F32, 2) => Some(simd_avx2_f32::gen_avx2_f32_v8_size2_soa()),
        (SimdIsa::Avx2, Precision::F32, 4) => Some(simd_avx2_f32::gen_avx2_f32_v8_size4_soa()),
        (SimdIsa::Avx2, Precision::F32, 8) => Some(simd_avx2_f32::gen_avx2_f32_v8_size8_soa()),
        _ => None,
    }
}

// ============================================================================
// Code generation
// ============================================================================

/// Build the outer `AoS` function body for any config (scalar loop over all transforms).
///
/// The outer function always processes transforms sequentially (scalar `AoS` loop),
/// regardless of whether a companion `SoA` SIMD function is also emitted.
/// Callers that want true SIMD throughput should use the `_soa` companion directly.
///
/// The transform-packing offset is hard-coded to a stride of `2 * V`, so the
/// generated body opens with `debug_assert_eq!(istride, 2 * V)` /
/// `debug_assert_eq!(ostride, 2 * V)`: any other stride would silently corrupt
/// the interleaved output, so it is rejected as a contract violation in debug
/// builds.
///
/// # Panics
///
/// Panics only if internal constant string literals fail to parse — impossible
/// in practice.
fn gen_outer_body(config: &MultiTransformConfig, size: usize, v: usize) -> TokenStream {
    let butterfly_body = scalar::gen_scalar_butterfly(size, config.precision);
    let v_lit = v;
    let stride_lit = v * 2;
    // Floats per full V-transform batch block: `size` complex elements per
    // transform, `v` interleaved transforms, 2 floats per complex.
    let block_lit = size * v * 2;
    quote! {
        // The AoS transform packing hard-codes a stride of 2*V; only the
        // canonical layout is supported. Guard the contract in debug builds.
        debug_assert_eq!(
            istride, #stride_lit,
            "multi-transform outer AoS: istride must equal 2*V ({})", #stride_lit
        );
        debug_assert_eq!(
            ostride, #stride_lit,
            "multi-transform outer AoS: ostride must equal 2*V ({})", #stride_lit
        );

        let batches = count / #v_lit;
        let remainder = count % #v_lit;

        // Each batch of V transforms occupies its own contiguous block of
        // `size * V * 2` floats. Within a block, transform lane `t` starts at
        // `t * 2` and its element `e` is at `t*2 + e*istride` (istride == 2*V).
        for b in 0..batches {
            let block_base = b * #block_lit;
            for t in 0..#v_lit {
                let base_in  = block_base + t * 2;
                let base_out = block_base + t * 2;
                #butterfly_body
            }
        }
        // Trailing partial batch (fewer than V transforms) lives in the next block.
        let rem_base = batches * #block_lit;
        for t in 0..remainder {
            let base_in  = rem_base + t * 2;
            let base_out = rem_base + t * 2;
            #butterfly_body
        }
    }
}

/// Generate a multi-transform codelet `TokenStream`.
///
/// # Output
///
/// Always emits a public outer function `notw_{size}_v{v}_{isa}_{ty}` with
/// `AoS` signature `(input, output, istride, ostride, count)`.
///
/// For supported (`ISA`, precision, size) combinations (SSE2 f32 sizes 2/4,
/// AVX2 f32 sizes 2/4/8), also emits a companion inner function
/// `notw_{size}_v{v}_{isa}_{ty}_soa` with `SoA` signature
/// `(re_in, im_in, re_out, im_out)` that is the **true SIMD implementation**.
///
/// # Errors
///
/// Returns a [`syn::Error`] when:
/// - `config.size` is not one of 2, 4, or 8.
/// - `config.v` is 0.
///
/// # Panics
///
/// Panics only if internal constant string literals that are guaranteed to be
/// valid fail to parse as token streams — this cannot occur in practice.
pub fn generate_multi_transform(config: &MultiTransformConfig) -> Result<TokenStream, syn::Error> {
    if !matches!(config.size, 2 | 4 | 8) {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "multi_transform: unsupported size {} (expected 2, 4, or 8)",
                config.size
            ),
        ));
    }
    if config.v == 0 {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "multi_transform: v must be >= 1",
        ));
    }

    let fn_name = format_ident!(
        "notw_{}_v{}_{}_{}",
        config.size,
        config.v,
        config.isa.ident_str(),
        config.precision.type_str()
    );
    let size = config.size;
    let v = config.v;
    let ty_str = config.precision.type_str();
    let ty_tokens: TokenStream = ty_str.parse().expect("valid type token");

    let use_simd = has_simd_impl(config.isa, config.precision, size);
    let simd_inner = gen_simd_inner(config);
    let outer_body = gen_outer_body(config, size, v);

    let stride = v * 2;
    let simd_note = if use_simd {
        format!(
            "True SIMD available via `notw_{size}_v{v}_{isa}_{ty}_soa` (`SoA` layout).",
            isa = config.isa.ident_str(),
            ty = ty_str,
        )
    } else {
        "Sequential scalar fallback (no SIMD for this `ISA`+precision+size combination).".into()
    };

    let block = size * v * 2;
    let fn_doc = format!(
        "Process `count` transforms of size {size} in batches of {v} (v={v}) using {isa} ISA.\n\n\
         # Data layout (`AoS`)\n\
         Transforms are grouped into batches of {v}; each batch is a contiguous block\n\
         of `{block}` floats (`{size} * {v} * 2`). For global transform `g` the batch is\n\
         `b = g / {v}` and the lane is `t = g % {v}`; element `k` of that transform lives at\n\
         `data[b * {block} + t * 2 + k * {stride} + c]` where `c` is 0 for real, 1 for imag.\n\
         A trailing partial batch (`count % {v}` transforms) occupies the next block.\n\n\
         # SIMD acceleration\n\
         {simd_note}\n\n\
         # Safety\n\
         - `input` must be valid for `((count + {v} - 1) / {v}) * {block}` reads of `{ty_str}`.\n\
         - `output` must be valid for the same number of `{ty_str}` writes.\n\
         - `istride` / `ostride` MUST be `2 * {v}` = {stride} (the only supported layout;\n\
           debug-asserted in the body).\n\
         - No alignment requirement; uses unaligned loads.",
        size = size,
        v = v,
        isa = config.isa.ident_str(),
        stride = stride,
        block = block,
        ty_str = ty_str,
        simd_note = simd_note,
    );

    let outer_fn = quote! {
        #[doc = #fn_doc]
        pub unsafe fn #fn_name(
            input:   *const #ty_tokens,
            output:  *mut   #ty_tokens,
            istride: usize,
            ostride: usize,
            count:   usize,
        ) {
            #outer_body
        }
    };

    Ok(if let Some(inner) = simd_inner {
        quote! {
            #inner
            #outer_fn
        }
    } else {
        outer_fn
    })
}

// ============================================================================
// Proc-macro entry point
// ============================================================================

/// Parsed arguments from `gen_multi_transform_codelet!(size=4, v=8, isa=avx2, ty=f32)`.
struct MacroArgs {
    size: usize,
    v: usize,
    isa: SimdIsa,
    precision: Precision,
}

impl Parse for MacroArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut size: Option<usize> = None;
        let mut v: Option<usize> = None;
        let mut isa: Option<SimdIsa> = None;
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
                "v" => {
                    let lit: LitInt = input.parse()?;
                    v = Some(lit.base10_parse::<usize>().map_err(|_| {
                        syn::Error::new(lit.span(), "expected an integer literal for `v`")
                    })?);
                }
                "isa" => {
                    let ident: syn::Ident = input.parse()?;
                    isa = Some(match ident.to_string().as_str() {
                        "sse2" => SimdIsa::Sse2,
                        "avx2" => SimdIsa::Avx2,
                        "scalar" => SimdIsa::Scalar,
                        other => {
                            return Err(syn::Error::new(
                                ident.span(),
                                format!(
                                    "unknown isa `{other}`, expected one of: sse2, avx2, scalar"
                                ),
                            ));
                        }
                    });
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
                        format!("unknown key `{other}`, expected one of: size, v, isa, ty"),
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
        let v = v.ok_or_else(|| {
            syn::Error::new(proc_macro2::Span::call_site(), "missing `v` argument")
        })?;
        let isa = isa.ok_or_else(|| {
            syn::Error::new(proc_macro2::Span::call_site(), "missing `isa` argument")
        })?;
        let precision = precision.ok_or_else(|| {
            syn::Error::new(proc_macro2::Span::call_site(), "missing `ty` argument")
        })?;

        Ok(Self {
            size,
            v,
            isa,
            precision,
        })
    }
}

/// Entry point for the `gen_multi_transform_codelet!` proc-macro.
///
/// Parses `size=N, v=V, isa=ISA, ty=TY` from the token stream and calls
/// [`generate_multi_transform`].
///
/// # Example
/// ```text
/// gen_multi_transform_codelet!(size = 4, v = 8, isa = avx2, ty = f32);
/// ```
///
/// # Errors
///
/// Returns a [`syn::Error`] when the input does not parse as valid key-value
/// pairs, a required key is missing, or `size` / `isa` / `ty` have unsupported
/// values.
pub fn generate_from_macro(input: TokenStream) -> Result<TokenStream, syn::Error> {
    let args: MacroArgs = syn::parse2(input)?;
    let config = MultiTransformConfig {
        size: args.size,
        v: args.v,
        isa: args.isa,
        precision: args.precision,
    };
    generate_multi_transform(&config)
}
