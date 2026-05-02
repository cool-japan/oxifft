//! `OxiFFT` Codelet Generator
//!
//! This proc-macro crate generates optimized FFT codelets at compile time.
//! It replaces FFTW's OCaml-based genfft with Rust procedural macros.
//!
//! # Overview
//!
//! Codelets are highly optimized kernels for small FFT sizes (2-64).
//! They are generated at compile time with:
//! - Common subexpression elimination
//! - Strength reduction
//! - Optimal instruction ordering
//! - SIMD-aware code patterns
//!
//! # Usage
//!
//! ```ignore
//! use oxifft_codegen::gen_dft_codelet;
//!
//! // Generate size-8 DFT codelet
//! gen_dft_codelet!(8);
//! ```

extern crate proc_macro;

use proc_macro::TokenStream;

/// Generate a non-twiddle (base case) DFT codelet.
///
/// # Arguments
/// * `size` - The FFT size (must be 2, 4, 8, 16, 32, or 64)
///
/// # Example
/// ```ignore
/// gen_notw_codelet!(8);
/// ```
#[proc_macro]
pub fn gen_notw_codelet(input: TokenStream) -> TokenStream {
    let input2: proc_macro2::TokenStream = input.into();
    oxifft_codegen_impl::gen_notw::generate(input2)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Generate a twiddle-factor DFT codelet.
#[proc_macro]
pub fn gen_twiddle_codelet(input: TokenStream) -> TokenStream {
    let input2: proc_macro2::TokenStream = input.into();
    oxifft_codegen_impl::gen_twiddle::generate(input2)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Generate a split-radix twiddle codelet.
///
/// The split-radix FFT decomposes an N-point DFT into one N/2-point DFT
/// (even-indexed elements) and two N/4-point DFTs (odd-indexed elements)
/// with twiddle factors `W_N^k` and `W_N^{3k`}, reducing the total multiply count.
///
/// # Usage
/// ```ignore
/// // Generate generic runtime-parameterized split-radix twiddle codelet
/// gen_split_radix_twiddle_codelet!();
///
/// // Generate specialized unrolled version for N=8
/// gen_split_radix_twiddle_codelet!(8);
///
/// // Generate specialized unrolled version for N=16
/// gen_split_radix_twiddle_codelet!(16);
/// ```
#[proc_macro]
pub fn gen_split_radix_twiddle_codelet(input: TokenStream) -> TokenStream {
    let input2: proc_macro2::TokenStream = input.into();
    oxifft_codegen_impl::gen_twiddle::generate_split_radix(input2)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Generate a SIMD-optimized codelet.
#[proc_macro]
pub fn gen_simd_codelet(input: TokenStream) -> TokenStream {
    let input2: proc_macro2::TokenStream = input.into();
    oxifft_codegen_impl::gen_simd::generate(input2)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Convenience macro to generate all codelets for a size.
#[proc_macro]
pub fn gen_dft_codelet(input: TokenStream) -> TokenStream {
    let input2: proc_macro2::TokenStream = input.into();
    oxifft_codegen_impl::gen_notw::generate(input2)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Generate an odd-size (3, 5, 7) DFT codelet using Winograd minimum-multiply factorization.
///
/// The generated function is an in-place `&mut [Complex<T>]` codelet with `sign: i32`
/// for runtime forward/inverse dispatch (matching `gen_notw_codelet!` conventions).
///
/// # Arguments
/// * The size literal — must be 3, 5, or 7.
///
/// # Example
/// ```ignore
/// gen_odd_codelet!(3);  // emits `codelet_notw_3`
/// gen_odd_codelet!(5);  // emits `codelet_notw_5`
/// gen_odd_codelet!(7);  // emits `codelet_notw_7`
/// ```
#[proc_macro]
pub fn gen_odd_codelet(input: TokenStream) -> TokenStream {
    let input2: proc_macro2::TokenStream = input.into();
    oxifft_codegen_impl::gen_odd::generate_from_macro(input2)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Generate a Rader prime DFT codelet for primes 11 and 13.
///
/// The generated function uses the Rader algorithm to reduce the prime-size DFT
/// to a cyclic convolution, computed as straight-line code with hardcoded twiddle
/// factors.  Generator g = 2 for both supported primes.
///
/// # Arguments
/// * The prime literal — must be 11 or 13.
///
/// # Example
/// ```ignore
/// gen_rader_codelet!(11);  // emits `codelet_notw_11`
/// gen_rader_codelet!(13);  // emits `codelet_notw_13`
/// ```
#[proc_macro]
pub fn gen_rader_codelet(input: TokenStream) -> TokenStream {
    let input2: proc_macro2::TokenStream = input.into();
    oxifft_codegen_impl::gen_rader::generate_from_macro(input2)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Generate a vectorized multi-transform codelet.
///
/// Emits a function that processes V DFT transforms of size N simultaneously,
/// where V is the SIMD lane count for the chosen ISA and precision.
///
/// The generated function name follows `notw_{size}_v{v}_{isa}_{ty}`.
///
/// # Arguments
/// * `size` — DFT size: 2, 4, or 8
/// * `v`    — number of simultaneous transforms (lane count)
/// * `isa`  — target ISA: `sse2`, `avx2`, or `scalar`
/// * `ty`   — float type: `f32` or `f64`
///
/// # Example
/// ```ignore
/// gen_multi_transform_codelet!(size = 4, v = 8, isa = avx2, ty = f32);
/// // emits: pub unsafe fn notw_4_v8_avx2_f32(...)
/// ```
#[proc_macro]
pub fn gen_multi_transform_codelet(input: TokenStream) -> TokenStream {
    let input2: proc_macro2::TokenStream = input.into();
    oxifft_codegen_impl::gen_simd::multi_transform::generate_from_macro(input2)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Generate a cached ISA runtime dispatcher for a SIMD codelet.
///
/// Emits a function `codelet_simd_{size}_cached_{ty}(data, sign)` that caches
/// the best ISA level in an `AtomicU8` static, avoiding repeated
/// `is_x86_feature_detected!` / `is_aarch64_feature_detected!` calls on hot
/// paths.  The cached dispatcher delegates to the same arch-specific inner
/// functions as the uncached `codelet_simd_{size}<T>` dispatcher.
///
/// # Arguments
/// * `size` — DFT size: 2, 4, 8, or 16
/// * `ty`   — float type: `f32` or `f64`
///
/// # Priority order (high → low)
/// - `x86_64`: AVX-512F > AVX2+FMA > AVX > SSE2 > scalar
/// - `aarch64`: NEON > scalar
/// - other: scalar
///
/// # Example
/// ```ignore
/// gen_dispatcher_codelet!(size = 4, ty = f32);
/// // emits: pub fn codelet_simd_4_cached_f32(data: &mut [Complex<f32>], sign: i32)
/// ```
#[proc_macro]
pub fn gen_dispatcher_codelet(input: TokenStream) -> TokenStream {
    let input2: proc_macro2::TokenStream = input.into();
    oxifft_codegen_impl::gen_simd::runtime_dispatch::generate_from_macro(input2)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Generate a complete FFT codelet for any user-specified size N.
///
/// Routes to the most appropriate emitter based on the size:
/// - **Direct set** {2, 4, 8, 16, 32, 64}: optimised non-twiddle codelet.
/// - **Winograd odd** {3, 5, 7}: Winograd minimum-multiply codelet.
/// - **Rader hardcoded** {11, 13}: straight-line Rader cyclic-convolution codelet.
/// - **Smooth-7 composites** (all prime factors in {2, 3, 5, 7}):
///   runtime-delegating wrapper using `Plan::dft_1d` (mixed-radix path).
/// - **Primes p ≤ 1021**: runtime-delegating wrapper (runtime Rader/Generic).
/// - **All other sizes**: runtime-delegating Bluestein wrapper via `Plan::dft_1d`.
///
/// # Syntax
/// ```ignore
/// gen_any_codelet!(8);     // emits codelet_any_8  (direct notw codelet)
/// gen_any_codelet!(15);    // emits codelet_any_15 (runtime mixed-radix wrapper)
/// gen_any_codelet!(2003);  // emits codelet_any_2003 (Bluestein wrapper)
/// ```
///
/// The emitted function signature is:
/// ```ignore
/// pub fn codelet_any_{N}<T: crate::kernel::Float>(
///     x: &mut [crate::kernel::Complex<T>],
///     sign: i32,
/// )
/// ```
#[proc_macro]
pub fn gen_any_codelet(input: TokenStream) -> TokenStream {
    let input2: proc_macro2::TokenStream = input.into();
    oxifft_codegen_impl::gen_any::generate_from_macro(input2).into()
}

/// Generate a real-to-half-complex (R2HC) or half-complex-to-real (HC2R) codelet.
///
/// The generated function has the same signature and produces numerically equivalent
/// results to the hand-written codelets in `oxifft/src/rdft/codelets/mod.rs`.
///
/// # Usage
/// ```ignore
/// use oxifft_codegen::gen_rdft_codelet;
///
/// // Generates `pub fn r2hc_4_gen<T: crate::kernel::Float>(x: &[T], y: &mut [Complex<T>])`
/// gen_rdft_codelet!(size = 4, kind = R2hc);
///
/// // Generates `pub fn hc2r_4_gen<T: crate::kernel::Float>(y: &[Complex<T>], x: &mut [T])`
/// gen_rdft_codelet!(size = 4, kind = Hc2r);
/// ```
///
/// Supported sizes: 2, 4, 8.
#[proc_macro]
pub fn gen_rdft_codelet(input: TokenStream) -> TokenStream {
    let input2: proc_macro2::TokenStream = input.into();
    oxifft_codegen_impl::gen_rdft::generate(input2)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}
