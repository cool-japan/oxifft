# oxifft-codegen

**Version:** 0.4.0
**Status:** Stable — codelet generation implemented for all supported sizes (scalar + SIMD).

Procedural macro crate for OxiFFT codelet generation.

## Overview

This crate replaces FFTW's OCaml-based `genfft` code generator with Rust
procedural macros. It generates highly optimized FFT kernels (codelets) at
compile time: non-twiddle base cases, twiddle / split-radix kernels, Winograd
odd and Rader prime kernels, real-FFT (RDFT) kernels, and runtime-dispatched
SIMD codelets for x86-64 (AVX-512F / AVX2+FMA / AVX / SSE2) and AArch64 (NEON).

## The `crate::kernel` contract

Every generated codelet refers to `crate::kernel::Float` and
`crate::kernel::Complex<T>`. The crate that **invokes** a codelet macro must
therefore expose a `kernel` module at its crate root providing a `Float` trait
(implemented for the scalar element type) and a `Complex<T: Float>` type.

The production `oxifft` crate supplies its own `oxifft::kernel`. For tests,
doc-tests, and downstream experimentation, a dependency-free reference contract
is available in the companion crate and can simply be re-exported as `kernel`:

```rust
mod kernel {
    pub use oxifft_codegen_impl::kernel_contract::{Complex, Float};
}

use oxifft_codegen::gen_notw_codelet;

// Expands to `pub fn codelet_notw_8<T: crate::kernel::Float>(..)`.
gen_notw_codelet!(8);
```

The examples below assume such a `kernel` module (or the host crate's own) is in
scope.

## Procedural macros

| Macro | Emits | Sizes |
|-------|-------|-------|
| `gen_notw_codelet!(N)` | `codelet_notw_N` | 2, 4, 8, 16, 32, 64 |
| `gen_dft_codelet!(N)` | alias of `gen_notw_codelet!` | 2, 4, 8, 16, 32, 64 |
| `gen_twiddle_codelet!(R)` | `codelet_twiddle_R` | radix 2, 4, 8, 16 |
| `gen_split_radix_twiddle_codelet!([N])` | split-radix twiddle codelet | generic, 8, 16 |
| `gen_odd_codelet!(N)` | `codelet_notw_N` (Winograd) | 3, 5, 7 |
| `gen_rader_codelet!(P)` | `codelet_notw_P` (Rader) | 11, 13 |
| `gen_rdft_codelet!(size = N, kind = R2hc\|Hc2r)` | `r2hc_N_gen` / `hc2r_N_gen` | 2, 4, 8 |
| `gen_simd_codelet!(N)` | `codelet_simd_N` + arch inner fns | 2, 4, 8, 16 |
| `gen_dispatcher_codelet!(size = N, ty = f32\|f64)` | `codelet_simd_N_cached_ty` | 2, 4, 8, 16 |
| `gen_multi_transform_codelet!(size, v, isa, ty)` | `notw_S_vV_ISA_TY` (batch of V) | 2, 4, 8 |
| `gen_any_codelet!(N)` | best emitter for any `N` | any (see below) |

### Non-twiddle codelets

Base-case FFT kernels used at the leaves of the FFT recursion.

```rust
use oxifft_codegen::gen_notw_codelet;
gen_notw_codelet!(8); // -> codelet_notw_8
```

### Twiddle & split-radix codelets

```rust
use oxifft_codegen::{gen_twiddle_codelet, gen_split_radix_twiddle_codelet};
gen_twiddle_codelet!(4);              // -> codelet_twiddle_4
gen_split_radix_twiddle_codelet!(16); // specialized N=16 split-radix twiddle
```

### Winograd-odd & Rader codelets

```rust
use oxifft_codegen::{gen_odd_codelet, gen_rader_codelet};
gen_odd_codelet!(5);    // -> codelet_notw_5 (Winograd minimum-multiply)
gen_rader_codelet!(11); // -> codelet_notw_11 (straight-line Rader)
```

### SIMD codelets & cached dispatcher

`gen_simd_codelet!` emits a runtime-dispatched `codelet_simd_N<T>` plus
architecture-specific inner functions and a portable scalar fallback.
`gen_dispatcher_codelet!` adds an `AtomicU8`-cached dispatcher that delegates to
those same inner functions (so it must sit in a child module of the
`gen_simd_codelet!` expansion):

```rust
mod simd {
    use oxifft_codegen::gen_simd_codelet;
    gen_simd_codelet!(4); // provides codelet_simd_4_scalar / _sse2_f32 / ...

    pub mod cached {
        use oxifft_codegen::gen_dispatcher_codelet;
        gen_dispatcher_codelet!(size = 4, ty = f32); // -> codelet_simd_4_cached_f32
    }
}
```

Dispatch priority (high → low): `x86_64` AVX-512F > AVX2+FMA > AVX > SSE2 >
scalar; `aarch64` NEON > scalar. Each ISA falls back to the next-lower available
SIMD tier — e.g. an AVX-only host running f32 (which has no pure-AVX codelet)
correctly uses SSE2 rather than scalar. AVX-512 codelets are gated behind the
`avx512` feature (default-off).

### Real-FFT (RDFT) codelets

```rust
use oxifft_codegen::gen_rdft_codelet;
gen_rdft_codelet!(size = 4, kind = R2hc); // -> r2hc_4_gen
gen_rdft_codelet!(size = 4, kind = Hc2r); // -> hc2r_4_gen
```

### Universal dispatcher

`gen_any_codelet!(N)` classifies `N` and routes to the best emitter:

- `N ∈ {1,2,3,4,5,7,8,11,13,16,32,64}` — direct NOTW / Winograd-odd / Rader
  codelet (self-contained; only needs `crate::kernel`).
- smooth-7 composite — `MixedRadix` runtime wrapper.
- prime `≤ 1021` — runtime `RaderPrime` wrapper.
- otherwise — `Bluestein` runtime wrapper.
- `N > 2^24` — rejected at compile time (`compile_error!`).

```rust
use oxifft_codegen::gen_any_codelet;
gen_any_codelet!(8);  // direct -> codelet_notw_8
gen_any_codelet!(13); // Rader  -> codelet_notw_13
```

The runtime-wrapper classes additionally delegate to `::oxifft`'s
`Plan::dft_1d`, so they only compile inside a crate that depends on `oxifft`.
Those wrappers emit `codelet_any_N` returning `Result<(), &'static str>` (planner
failure is propagated, never panicked).

## Supported sizes

| Size | Non-twiddle | Twiddle | SIMD (`codelet_simd_N`) |
|------|-------------|---------|-------------------------|
| 2    | ✓           | ✓       | ✓ (all ISAs)            |
| 4    | ✓           | ✓       | ✓ (all ISAs)            |
| 8    | ✓           | ✓       | ✓ (all ISAs)            |
| 16   | ✓           | ✓       | ✓ (f32: AVX-512F; else scalar) |
| 32   | ✓           | —       | —                       |
| 64   | ✓           | —       | —                       |

Odd sizes 3/5/7 (Winograd) and primes 11/13 (Rader) are generated by the
dedicated macros; every other size is reachable through `gen_any_codelet!`.

## Code generation strategy

Following FFTW's approach:

1. **Symbolic representation** — build a DAG of FFT butterfly operations.
2. **Optimization passes** — common-subexpression elimination, constant
   folding, strength reduction, and dead-code elimination (`gen_simd`'s scalar
   bodies and the size 16/32/64 codelets are emitted from this pipeline).
3. **Code emission** — generate Rust with the `Float` trait and FFTW-style
   codelet naming.

## License

Apache-2.0 — Copyright (c) 2026 COOLJAPAN OU (Team Kitasan)
