//! Wall-clock benchmarks for codelet generation time.
//!
//! This bench measures how long it takes to call each `generate()` function
//! in `oxifft-codegen-impl` — i.e., the cost of generating the proc-macro
//! output `TokenStream` for each codelet type and size.
//!
//! This complements `codelet_perf.rs` (which measures *running* the generated
//! code) by showing the *generation* overhead.
//!
//! # Running
//!
//! ```text
//! cargo bench -p oxifft-codegen --bench codegen_time
//! ```
//!
//! Or compile-only (CI gate):
//! ```text
//! cargo bench -p oxifft-codegen --bench codegen_time --no-run
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use proc_macro2::TokenStream;
use quote::quote;
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a `TokenStream` containing a single integer literal `n`.
///
/// Used as input to generators that accept `size = N` or just `N`.
fn make_size_input(n: usize) -> TokenStream {
    quote! { #n }
}

/// Build a `TokenStream` for `gen_rdft_codelet!(size = N, kind = R2hc)`.
fn make_rdft_r2hc_input(n: usize) -> TokenStream {
    quote! { size = #n, kind = R2hc }
}

/// Build a `TokenStream` for `gen_rdft_codelet!(size = N, kind = Hc2r)`.
fn make_rdft_hc2r_input(n: usize) -> TokenStream {
    quote! { size = #n, kind = Hc2r }
}

/// Build an empty `TokenStream`.
///
/// Used for `gen_split_radix_twiddle_codelet!()` (no-arg generic variant).
fn make_empty_input() -> TokenStream {
    TokenStream::new()
}

// ---------------------------------------------------------------------------
// gen_notw benchmarks — sizes 2, 4, 8, 16, 32, 64
// ---------------------------------------------------------------------------

fn bench_gen_notw(c: &mut Criterion) {
    let mut group = c.benchmark_group("gen_notw");

    for &size in &[2_usize, 4, 8, 16, 32, 64] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &s| {
            b.iter(|| {
                let input = make_size_input(s);
                black_box(
                    oxifft_codegen_impl::gen_notw::generate(input).expect("gen_notw::generate"),
                )
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// gen_twiddle benchmarks — radixes 2, 4, 8, 16
// ---------------------------------------------------------------------------

fn bench_gen_twiddle(c: &mut Criterion) {
    let mut group = c.benchmark_group("gen_twiddle");

    for &radix in &[2_usize, 4, 8, 16] {
        group.bench_with_input(BenchmarkId::from_parameter(radix), &radix, |b, &r| {
            b.iter(|| {
                let input = make_size_input(r);
                black_box(
                    oxifft_codegen_impl::gen_twiddle::generate(input)
                        .expect("gen_twiddle::generate"),
                )
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// gen_split_radix_twiddle benchmarks — generic, size 8, size 16
// ---------------------------------------------------------------------------

fn bench_gen_split_radix_twiddle(c: &mut Criterion) {
    let mut group = c.benchmark_group("gen_split_radix_twiddle");

    // Generic (empty input)
    group.bench_function("generic", |b| {
        b.iter(|| {
            let input = make_empty_input();
            black_box(
                oxifft_codegen_impl::gen_twiddle::generate_split_radix(input)
                    .expect("gen_twiddle::generate_split_radix generic"),
            )
        });
    });

    // Specialized sizes
    for &size in &[8_usize, 16] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &s| {
            b.iter(|| {
                let input = make_size_input(s);
                black_box(
                    oxifft_codegen_impl::gen_twiddle::generate_split_radix(input)
                        .expect("gen_twiddle::generate_split_radix"),
                )
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// gen_rdft_r2hc benchmarks — sizes 2, 4, 8
// ---------------------------------------------------------------------------

fn bench_gen_rdft_r2hc(c: &mut Criterion) {
    let mut group = c.benchmark_group("gen_rdft_r2hc");

    for &size in &[2_usize, 4, 8] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &s| {
            b.iter(|| {
                let input = make_rdft_r2hc_input(s);
                black_box(
                    oxifft_codegen_impl::gen_rdft::generate(input)
                        .expect("gen_rdft::generate R2hc"),
                )
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// gen_rdft_hc2r benchmarks — sizes 2, 4, 8
// ---------------------------------------------------------------------------

fn bench_gen_rdft_hc2r(c: &mut Criterion) {
    let mut group = c.benchmark_group("gen_rdft_hc2r");

    for &size in &[2_usize, 4, 8] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &s| {
            b.iter(|| {
                let input = make_rdft_hc2r_input(s);
                black_box(
                    oxifft_codegen_impl::gen_rdft::generate(input)
                        .expect("gen_rdft::generate Hc2r"),
                )
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// gen_simd benchmarks — sizes 2, 4, 8
//
// gen_simd::generate emits the full dispatcher + all ISA variants in one
// call, so we benchmark per-size rather than per-ISA.
// ---------------------------------------------------------------------------

fn bench_gen_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("gen_simd");

    for &size in &[2_usize, 4, 8] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &s| {
            b.iter(|| {
                let input = make_size_input(s);
                black_box(
                    oxifft_codegen_impl::gen_simd::generate(input).expect("gen_simd::generate"),
                )
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion configuration
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_gen_notw,
    bench_gen_twiddle,
    bench_gen_split_radix_twiddle,
    bench_gen_rdft_r2hc,
    bench_gen_rdft_hc2r,
    bench_gen_simd,
);
criterion_main!(benches);
