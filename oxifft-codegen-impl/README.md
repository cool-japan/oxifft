# oxifft-codegen-impl

**Version:** 0.3.1

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
- **Universal dispatcher** — `gen_any` module classifies any FFT size and
  routes it to the optimal codelet path (hardcoded, MixedRadix, Rader, or
  Bluestein).

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

## Modules

| Module | Description |
|--------|-------------|
| `emit` | Core token-stream emitters for NOTW, twiddle, RDFT, and SIMD codelets |
| `gen_any` | Universal dispatcher — classifies `N` and selects the optimal path |
| `gen_mixed_radix` | MixedRadix runtime-wrapper generator for smooth-7 composite sizes |
| `optimize` | DCE, constant folding, CSE, and strength-reduction passes |
| `symbolic` | Expression DAG for symbolic FFT butterfly operations |

## Public API

### `classify(n: usize) -> SizeClass`

Classifies an FFT size into the appropriate generation strategy:

```rust
use oxifft_codegen_impl::classify;

match classify(13) {
    SizeClass::Hardcoded  => { /* direct codelet */ }
    SizeClass::MixedRadix => { /* smooth-7 composite */ }
    SizeClass::Rader      => { /* prime ≤ 1021 */ }
    SizeClass::Bluestein  => { /* general fallback */ }
}
```

### `SizeClass` enum

```
SizeClass::Hardcoded   — N ∈ {2,3,4,5,7,8,11,13,16,32,64}
SizeClass::MixedRadix  — N smooth-7 composite not in the hardcoded set
SizeClass::Rader       — N prime, 1021 ≥ N > 64 and not hardcoded
SizeClass::Bluestein   — all other N
```

### `CodeletBuilder`

Programmatic API for constructing a single codelet:

```rust
use oxifft_codegen_impl::{CodeletBuilder, Precision, Direction};

let tokens = CodeletBuilder::new(16)
    .precision(Precision::F64)
    .direction(Direction::Forward)
    .build();
```

## Test Count

213 tests in `oxifft-codegen-impl` + 56 tests in `oxifft-codegen` = **269 total**.

## License

Apache-2.0. See [LICENSE](../LICENSE).
