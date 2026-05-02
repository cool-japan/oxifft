# oxifft-codegen

**Version:** 0.3.1  
**Status:** ✅ Stable — codelet generation complete for all supported sizes

Procedural macro crate for OxiFFT codelet generation.

## Overview

This crate replaces FFTW's OCaml-based `genfft` code generator with Rust procedural macros. It generates highly optimized FFT kernels (codelets) at compile time.

## Features

- **Compile-time code generation**: Zero runtime overhead (radix 2, 4, 8, 16)
- **SIMD-aware**: Can generate SIMD-specific code patterns (infrastructure in place)
- **Symbolic optimization**: Built-in symbolic expression DAG for optimization passes

## Procedural Macros

- `gen_notw_codelet!(size)` — Non-twiddle base case codelets
- `gen_twiddle_codelet!(radix)` — Twiddle-factor codelets for multi-radix FFT
- `gen_simd_codelet!(size)` — SIMD-optimized codelets (infrastructure, generation pending)
- `gen_dft_codelet!(size)` — Convenience wrapper (aliases `gen_notw_codelet!`)
- `gen_any_codelet!(N)` — Universal dispatcher: routes any size N to the best emitter

## Codelet Types

### Non-Twiddle Codelets (notw)

Base case FFT kernels that don't require twiddle factors. Used at the leaves of the FFT recursion.

```rust
use oxifft_codegen::gen_notw_codelet;

// Generates codelet_notw_8 function
gen_notw_codelet!(8);
```

### Twiddle Codelets

Codelets that apply twiddle factors as part of the Cooley-Tukey recursion.

```rust
use oxifft_codegen::gen_twiddle_codelet;

// Generates codelet_twiddle_4
gen_twiddle_codelet!(4);
```

### SIMD Codelets

SIMD-optimized codelets (infrastructure present, full generation pending).

```rust
use oxifft_codegen::gen_simd_codelet;

// Generates SIMD-optimized size-8 codelet
gen_simd_codelet!(8);
```

### Convenience Macro

```rust
use oxifft_codegen::gen_dft_codelet;

// Currently aliases gen_notw_codelet!
gen_dft_codelet!(8);
```

### Universal Dispatcher

`gen_any_codelet!(N)` selects the optimal implementation path at compile time:

- **N ∈ {2,3,4,5,7,8,11,13,16,32,64}**: direct hardcoded NOTW/Rader codelets
- **N smooth-7 composite**: MixedRadix runtime wrapper
- **N prime ≤ 1021**: Rader runtime path
- **Otherwise**: Bluestein runtime wrapper

```rust
use oxifft_codegen::gen_any_codelet;

// Routes size 13 to the hardcoded Rader codelet
gen_any_codelet!(13);

// Routes size 100 through the MixedRadix path (4×5×5, smooth-7 composite)
gen_any_codelet!(100);

// Routes a large prime through the Bluestein path
gen_any_codelet!(997);
```

## Supported Sizes

| Size | Non-Twiddle | Twiddle | SIMD |
|------|-------------|---------|------|
| 2    | ✓           | ✓       | Planned |
| 4    | ✓           | ✓       | Planned |
| 8    | ✓           | ✓       | Planned |
| 16   | ✓           | ✓       | Planned |
| 32   | ✓           | Planned | Planned |
| 64   | ✓           | Planned | Planned |

## Code Generation Strategy

The codelet generator follows FFTW's approach:

1. **Symbolic representation**: Build a DAG of FFT operations
2. **Optimization passes**:
   - Common subexpression elimination (CSE)
   - Strength reduction (replace multiplications with additions where possible)
   - Dead code elimination
3. **Code emission**: Generate Rust code with optimal instruction ordering

## Implementation Notes

- Uses `proc-macro2`, `quote`, and `syn` for macro implementation
- Generated code is generic over the `Float` trait (f32/f64)
- Follows FFTW's codelet naming conventions for compatibility
- All sizes 2–64 have generation infrastructure, with 2–16 fully implemented
- 56 tests passing

## License

Apache-2.0 — Copyright (c) 2026 COOLJAPAN OU (Team Kitasan)
