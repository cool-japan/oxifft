# oxifft-codegen-impl

**Version:** 0.4.3

Internal codelet generation logic for [OxiFFT](https://github.com/cool-japan/oxifft).

This crate provides the core symbolic computation and codelet-generation engine
used by `oxifft-codegen` (the procedural macro crate) and by `oxifft-bench`
(for benchmarking codelet generation throughput). It is **not a proc-macro**
crate — it is a regular library that `oxifft-codegen` depends on to work around
the proc-macro crate limitation that prevents exposing non-macro public APIs.

All public items are **semver-unstable**: they may change at any time. External
code should prefer the proc-macro interface exposed by `oxifft-codegen`.

## Overview

- **Symbolic DFT engine** (`symbolic`) — represents FFT butterfly operations
  symbolically as expression trees, enabling algebraic simplification, constant
  folding, and strength reduction before code generation.
- **Codelet generators** — emit Rust token streams for non-twiddle
  (`gen_notw`), twiddle / split-radix (`gen_twiddle`), Winograd odd
  (`gen_odd`), hardcoded Rader (`gen_rader`), RDFT (`gen_rdft`), and
  SIMD-specialized (`gen_simd`) codelets.
- **Universal dispatcher** (`gen_any`) — classifies any FFT size and routes it
  to the optimal codelet path (hardcoded, MixedRadix, Rader, or Bluestein).
- **Kernel contract** (`kernel_contract`) — a dependency-free reference
  `Float` / `Complex<T>` implementation that generated codelets can target (see
  below).

## The `crate::kernel` contract

Every generated codelet refers to `crate::kernel::Float` and
`crate::kernel::Complex<T>`, so the crate that *invokes* a codelet must provide a
`kernel` module at its crate root satisfying that contract. `kernel_contract`
is an executable specification of exactly what is required and can be
re-exported as `kernel` in tests, doc-tests, and downstream experiments:

```rust
mod kernel {
    pub use oxifft_codegen_impl::kernel_contract::{Complex, Float};
}
```

## Modules

| Module | Description |
|--------|-------------|
| `gen_any` | Universal dispatcher — classifies `N` and selects the optimal path |
| `gen_mixed_radix` | MixedRadix runtime-wrapper generator for smooth-7 composites |
| `gen_notw` | Non-twiddle base-case codelets {2, 4, 8, 16, 32, 64} |
| `gen_odd` | Winograd odd codelets {3, 5, 7} |
| `gen_rader` | Hardcoded Rader codelets {11, 13} |
| `gen_rdft` | Real-to-halfcomplex / halfcomplex-to-real codelets {2, 4, 8} |
| `gen_simd` | SIMD codelets + runtime dispatch + multi-transform |
| `gen_twiddle` | Twiddle-factor and split-radix twiddle codelets |
| `kernel_contract` | Reference `Float` / `Complex<T>` contract for generated code |
| `symbolic` | Expression DAG + optimization passes (DCE, CSE, folding, strength reduction) |
| `winograd_constants` | Exact cos/sin constants for the Winograd odd codelets |

## Public API

### `classify`

```rust
pub fn classify(n: usize) -> Result<SizeClass, CodegenError>
```

Classifies an FFT size into a generation strategy:

```rust
use oxifft_codegen_impl::{classify, SizeClass};

assert!(matches!(classify(8),    Ok(SizeClass::Notw(8))));
assert!(matches!(classify(3),    Ok(SizeClass::Odd(3))));
assert!(matches!(classify(13),   Ok(SizeClass::RaderHardcoded(13))));
assert!(matches!(classify(15),   Ok(SizeClass::MixedRadix(_))));   // 15 = 5 × 3
assert!(matches!(classify(17),   Ok(SizeClass::RaderPrime(17))));
assert!(matches!(classify(2003), Ok(SizeClass::Bluestein(2003)))); // large prime
```

### `SizeClass`

```rust
pub enum SizeClass {
    Notw(usize),          // {1, 2, 4, 8, 16, 32, 64} — direct non-twiddle codelet
    Odd(usize),           // {3, 5, 7}                — Winograd minimum-multiply
    RaderHardcoded(usize),// {11, 13}                 — straight-line Rader
    MixedRadix(Vec<u16>), // smooth-7 composite       — runtime Plan::dft_1d wrapper
    RaderPrime(usize),    // prime ≤ 1021             — runtime Plan::dft_1d wrapper
    Bluestein(usize),     // everything else          — runtime Plan::dft_1d wrapper
}
```

### `CodegenError`

```rust
pub enum CodegenError {
    InvalidSize(usize),     // n == 0
    UnsupportedSize(usize), // n > MAX_ANY_SIZE (= 1 << 24)
    EmitError(String),      // downstream emission failure
}
```

### `CodeletBuilder`

Programmatic API for building a single codelet `TokenStream` without proc-macros
(the same output `gen_any_codelet!` would emit):

```rust
use oxifft_codegen_impl::CodeletBuilder;

let tokens = CodeletBuilder::new(15).build().expect("size 15 is supported");
assert!(!tokens.to_string().is_empty());
```

`CodeletBuilder` exposes `new(n)`, `name(..)` (reserved), and `build()`. There is
no precision/direction selector: generated codelets are generic over the `Float`
trait and take a runtime `sign` for forward/inverse.

## Testing

Run the crate's own suite with `cargo test -p oxifft-codegen-impl`. Numerical
parity of the generated codelets against a naive DFT is exercised from
`oxifft-codegen`'s integration tests (which supply the `kernel` contract).

## License

Apache-2.0. See [LICENSE](../LICENSE).
