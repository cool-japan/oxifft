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

mod gen_notw;
mod gen_simd;
mod gen_twiddle;
mod symbolic;

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
    gen_notw::generate(input)
}

/// Generate a twiddle-factor DFT codelet.
#[proc_macro]
pub fn gen_twiddle_codelet(input: TokenStream) -> TokenStream {
    gen_twiddle::generate(input)
}

/// Generate a SIMD-optimized codelet.
#[proc_macro]
pub fn gen_simd_codelet(input: TokenStream) -> TokenStream {
    gen_simd::generate(input)
}

/// Convenience macro to generate all codelets for a size.
#[proc_macro]
pub fn gen_dft_codelet(input: TokenStream) -> TokenStream {
    gen_notw::generate(input)
}
