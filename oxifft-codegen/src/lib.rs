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
