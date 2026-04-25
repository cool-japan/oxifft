# oxifft-codegen-impl

Internal codelet generation logic for [OxiFFT](https://github.com/cool-japan/oxifft).

This crate provides the core symbolic computation and codelet-generation engine
used by `oxifft-codegen` (the procedural macro crate) and by `oxifft-bench`
(for benchmarking codelet generation throughput). It is **not a proc-macro**
crate — it is a regular library that `oxifft-codegen` depends on to work around
the proc-macro crate limitation that prevents exposing non-macro public APIs.

## Overview

- **Symbolic DFT engine** — represents FFT butterfly operations symbolically as
  expression trees, enabling algebraic simplification and strength reduction
  before code generation.
- **Codelet generators** — emit Rust token streams for radix-2/4/8 and
  mixed-radix NOTW (no-twiddle) kernels, RDFT codelets, and SIMD-specialized
  variants for x86-64 (AVX2/SSE2) and AArch64 (NEON/SVE).
- **Optimization passes** — dead-code elimination, constant folding, common
  sub-expression elimination, and strength reduction (mul-by-1 → identity,
  mul-by-0 → zero).

## Usage

This crate is an implementation detail of `oxifft-codegen`. Users of OxiFFT
should depend on `oxifft` directly; `oxifft-codegen-impl` is re-exported
through the proc-macro interface automatically.

```toml
# In your Cargo.toml — depend on oxifft, not on this crate directly
[dependencies]
oxifft = "0.3"
```

If you are building tooling that needs access to the codelet generation
internals (e.g. a custom benchmark harness or a code-generation analysis tool),
you may depend on this crate directly:

```toml
[dev-dependencies]
oxifft-codegen-impl = "0.3"
```

## License

Apache-2.0. See [LICENSE](../LICENSE).
