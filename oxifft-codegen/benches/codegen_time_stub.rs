use criterion::{criterion_group, criterion_main, Criterion};
use proc_macro2::TokenStream;
use quote::quote;

fn bench_test(c: &mut Criterion) {
    c.bench_function("notw_8", |b| {
        b.iter(|| {
            let input: TokenStream = quote! { 8 };
            oxifft_codegen_impl::gen_notw::generate(input).expect("generate ok");
        });
    });
}

criterion_group!(benches, bench_test);
criterion_main!(benches);
