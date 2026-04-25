//! Code-size analysis for OxiFFT-generated codelets.
//!
//! Run with:
//!   `cargo run -p oxifft-codegen --example code_size_report`
//!
//! Prints a tab-separated table of token count, approximate line count, and
//! (where available) symbolic op count for every generated codelet.

use oxifft_codegen_impl::{gen_notw, gen_rdft, gen_simd, gen_twiddle, symbolic};
use proc_macro2::TokenStream;
use quote::quote;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Count the number of tokens in a `TokenStream`.
fn count_tokens(ts: &TokenStream) -> usize {
    ts.clone().into_iter().count()
}

/// Approximate line count by counting lines in the string representation.
///
/// This is a proxy — `prettyplease` is not in the workspace deps.
fn count_lines(ts: &TokenStream) -> usize {
    ts.to_string().lines().count()
}

/// Op count via `SymbolicFFT` (power-of-two sizes only).
///
/// Returns `None` when symbolic computation is not applicable to the given
/// codelet kind/size combination (twiddle, RDFT, etc.).
fn symbolic_op_count(n: usize) -> Option<usize> {
    if n.is_power_of_two() && (2..=64).contains(&n) {
        let sym = symbolic::SymbolicFFT::radix2_dit(n, true);
        Some(sym.op_count())
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Printer
// ---------------------------------------------------------------------------

fn print_row(kind: &str, label: &str, ts: &TokenStream, op_count: Option<usize>) {
    let ops = op_count.map_or_else(|| "-".to_string(), |n| n.to_string());
    println!(
        "{kind}\t{label}\t{}\t{}\t{ops}",
        count_tokens(ts),
        count_lines(ts),
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("kind\tlabel\ttokens\tlines\top_count");

    // -------------------------------------------------------------------------
    // gen_notw — non-twiddle base-case codelets
    // -------------------------------------------------------------------------
    for &n in &[2_usize, 4, 8, 16, 32, 64] {
        let ts = gen_notw::generate(quote! { #n }).expect("gen_notw::generate failed");
        print_row("notw", &n.to_string(), &ts, symbolic_op_count(n));
    }

    // -------------------------------------------------------------------------
    // gen_twiddle — twiddle codelets (radix)
    // -------------------------------------------------------------------------
    for &r in &[2_usize, 4, 8, 16] {
        let ts = gen_twiddle::generate(quote! { #r }).expect("gen_twiddle::generate failed");
        // Twiddle codelets apply twiddle factors — not comparable to SymbolicFFT
        print_row("twiddle", &r.to_string(), &ts, None);
    }

    // -------------------------------------------------------------------------
    // gen_twiddle split-radix — generic + specialised sizes
    // -------------------------------------------------------------------------
    {
        let ts = gen_twiddle::generate_split_radix(TokenStream::new())
            .expect("gen_twiddle::generate_split_radix (generic) failed");
        print_row("split_radix_twiddle", "generic", &ts, None);
    }
    for &n in &[8_usize, 16] {
        let ts = gen_twiddle::generate_split_radix(quote! { #n })
            .expect("gen_twiddle::generate_split_radix failed");
        print_row("split_radix_twiddle", &n.to_string(), &ts, None);
    }

    // -------------------------------------------------------------------------
    // gen_rdft — R2HC and HC2R codelets
    // -------------------------------------------------------------------------
    for &sz in &[2_usize, 4, 8] {
        for (kind_str, kind_tokens) in &[
            ("R2hc", quote! { size = #sz, kind = R2hc }),
            ("Hc2r", quote! { size = #sz, kind = Hc2r }),
        ] {
            let ts = gen_rdft::generate(kind_tokens.clone()).expect("gen_rdft::generate failed");
            let label = format!("{sz}_{kind_str}");
            print_row("rdft", &label, &ts, None);
        }
    }

    // -------------------------------------------------------------------------
    // gen_simd — SIMD-dispatched codelets
    // -------------------------------------------------------------------------
    for &n in &[2_usize, 4, 8] {
        let ts = gen_simd::generate(quote! { #n }).expect("gen_simd::generate failed");
        print_row("simd", &n.to_string(), &ts, None);
    }
}
