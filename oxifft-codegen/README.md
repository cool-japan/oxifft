# oxifft-codegen

Procedural macro crate for OxiFFT codelet generation.

## Overview

This crate replaces FFTW's OCaml-based `genfft` code generator with Rust procedural macros. It generates highly optimized FFT kernels (codelets) at compile time.

## Features

- **Compile-time code generation**: Zero runtime overhead
- **Optimized kernels**: Common subexpression elimination, strength reduction
- **Multiple sizes**: Support for sizes 2, 4, 8, 16, 32, 64
- **Twiddle variants**: Both non-twiddle (base case) and twiddle codelets
- **SIMD-aware**: Can generate SIMD-specific code patterns

## Codelet Types

### Non-Twiddle Codelets (notw)

Base case FFT kernels that don't require twiddle factors. Used at the leaves of the FFT recursion.

```rust
use oxifft_codegen::gen_notw_codelet;

// Generates codelet_notw_8 function
gen_notw_codelet!(8);
```

### Twiddle Codelets

FFT kernels that apply twiddle factors as part of the Cooley-Tukey recursion.

```rust
use oxifft_codegen::gen_twiddle_codelet;

// Generates codelet_twiddle_4 function
gen_twiddle_codelet!(4);
```

### SIMD Codelets

Architecture-specific SIMD-optimized kernels.

```rust
use oxifft_codegen::gen_simd_codelet;

// Generates SIMD-optimized size-8 codelet
gen_simd_codelet!(8);
```

## Code Generation Strategy

The codelet generator follows FFTW's approach:

1. **Symbolic representation**: Build a DAG of FFT operations
2. **Optimization passes**:
   - Common subexpression elimination (CSE)
   - Strength reduction (replace multiplications with additions where possible)
   - Dead code elimination
3. **Code emission**: Generate Rust code with optimal instruction ordering

## Supported Sizes

| Size | Non-Twiddle | Twiddle | SIMD |
|------|-------------|---------|------|
| 2    | ✓           | ✓       | Planned |
| 4    | ✓           | ✓       | Planned |
| 8    | ✓           | ✓       | Planned |
| 16   | ✓           | Planned | Planned |
| 32   | ✓           | Planned | Planned |
| 64   | ✓           | Planned | Planned |

## Implementation Notes

- Uses `proc-macro2`, `quote`, and `syn` for macro implementation
- Generated code is generic over the `Float` trait (f32/f64)
- Follows FFTW's codelet naming conventions for compatibility

## License

Same as the parent OxiFFT project.
