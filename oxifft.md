# OxiFFT Architecture Blueprint

## Overview

OxiFFT is a pure-Rust implementation of FFTW3 following its Problem-Plan-Solver design.

## Module Structure

```
oxifft/src/
├── api/            Public user-facing API (Plan, Plan2D, Plan3D, RealPlan, GuruPlan)
├── kernel/         Core planner, data structures, F16/F128 types, twiddle factors
├── dft/            Complex DFT implementations (codelets, solvers, composite)
├── rdft/           Real DFT (R2C, C2R, R2R)
├── reodft/         DCT/DST transforms (Types I-IV, DHT)
├── simd/           SIMD abstraction layer (SSE2, AVX, AVX2, AVX-512, NEON, SVE)
├── threading/      Parallel execution via Rayon
├── support/        Utilities (alignment, transpose, copy operations)
├── sparse/         Sparse FFT (FFAST algorithm, O(k log n))
├── pruned/         Pruned FFT (input/output pruning, Goertzel)
├── streaming/      STFT, window functions, mel filterbank, MFCC
├── signal/         Hilbert transform, Welch PSD, cepstrum, resampling
├── const_fft/      Compile-time FFT with const generics (sizes 2–1024)
├── nufft/          Non-uniform FFT (Type 1/2/3, Gaussian gridding)
├── frft/           Fractional Fourier Transform
├── conv/           FFT-based convolution and correlation
├── autodiff/       Automatic differentiation (forward/backward mode)
├── gpu/            GPU acceleration (CUDA, Metal)
├── mpi/            MPI distributed computing
└── wasm/           WebAssembly bindings
```

## Core Traits

```rust
pub trait Problem: Hash + Debug + Clone + Send + Sync { ... }
pub trait Plan: Send + Sync { ... }
pub trait Solver: Send + Sync { ... }
```

## Planning System

The planner selects the optimal solver for each problem via the Wisdom cache:

1. Hash the problem (size, direction, flags)
2. Check wisdom cache for a previously measured plan
3. If not cached, run timing trials (MEASURE/PATIENT/EXHAUSTIVE modes)
4. Store the winner in the wisdom cache for future reuse

## Architecture Diagrams

### Overall Module Organization

```mermaid
graph TD
    A["<b>OxiFFT</b><br/>lib.rs"] --> B["<b>api/</b><br/>User-facing API<br/>Plan, Plan2D, Plan3D<br/>RealPlan, GuruPlan"]
    A --> C["<b>kernel/</b><br/>Core types and planner<br/>Complex, Float, Planner<br/>Twiddle factors, Problem"]
    A --> D["<b>dft/</b><br/>Complex DFT<br/>Solvers and Codelets"]
    A --> E["<b>rdft/</b><br/>Real FFT<br/>R2C, C2R, R2R"]
    A --> F["<b>reodft/</b><br/>DCT/DST/DHT"]
    A --> G["<b>simd/</b><br/>SIMD abstraction<br/>CPU detect and backends"]
    A --> H["<b>threading/</b><br/>Parallel execution<br/>Rayon work-stealing"]
    D --> D1["<b>solvers/</b><br/>Cooley-Tukey, Stockham<br/>Rader, Bluestein, Generic"]
    D --> D2["<b>codelets/</b><br/>Optimized kernels<br/>SIMD dispatch"]
    G --> G1["<b>sse2.rs</b><br/>128-bit x86_64"]
    G --> G2["<b>avx.rs</b><br/>256-bit x86_64"]
    G --> G3["<b>avx2.rs</b><br/>256-bit+FMA"]
    G --> G4["<b>avx512.rs</b><br/>512-bit x86_64"]
    G --> G5["<b>neon.rs</b><br/>128-bit aarch64"]
    G --> G6["<b>sve.rs</b><br/>Scalable aarch64"]
    B -.->|creates| D
    B -.->|creates| E
    D2 -.->|dispatches to| G
```

### Planning and Solver Dispatch

```mermaid
flowchart LR
    A["User Code<br/>Plan::dft_1d(n, dir, flags)"] --> B["Plan::select_algorithm(n)"]
    B --> C{Size Check}
    C -->|n le 1| D["Algorithm::Nop<br/>NopSolver"]
    C -->|is_power_of_2| E["Algorithm::CooleyTukey<br/>CooleyTukeySolver<br/>O(n log n)"]
    C -->|has_composite_codelet| F["Algorithm::Composite<br/>sizes 12-4096<br/>execute_composite_codelet"]
    C -->|n le 16| G["Algorithm::Direct<br/>O(n^2) naive DFT"]
    C -->|composite_factorizable| H["Algorithm::Generic<br/>Mixed-radix N=N1*N2<br/>GenericSolver"]
    C -->|prime or arbitrary| I["Algorithm::Bluestein<br/>Chirp-Z transform<br/>O(n log n) fallback"]
    D --> J["plan.execute()"]
    E --> J
    F --> J
    G --> J
    H --> J
    I --> J
    J --> K["Output: transformed data"]
```

### Solver Hierarchy

```mermaid
graph TD
    CT["CooleyTukeySolver<br/>isPowerOf2(n)"]
    CT --> DIT["CtVariant::Dit<br/>Decimation-in-Time"]
    CT --> DIF["CtVariant::Dif<br/>Decimation-in-Frequency"]
    CT --> R4["CtVariant::DitRadix4<br/>Radix-4 (25% fewer muls)"]
    CT --> R8["CtVariant::DitRadix8"]
    CT --> SR["CtVariant::SplitRadix<br/>Optimal operation count"]
    GEN["GenericSolver<br/>Composite n = N1 x N2"] --> CT
    GEN --> DIR["DirectSolver n le 16"]
    GEN --> BLU
    BLU["BluesteinSolver<br/>Chirp-Z<br/>Arbitrary n"]
    RAD["RaderSolver<br/>Prime n"]
    SK["StockhamSolver<br/>Cache-oblivious<br/>Large power-of-2"]
    CT --> COD["Codelets<br/>Hand-optimized SIMD<br/>n in 2..4096"]
    GEN --> COD
```

### SIMD Runtime Dispatch

```mermaid
graph TD
    A["Codelet needs SIMD<br/>simd::detect_simd_level()"] --> B["x86_64"]
    A --> C["aarch64"]
    A --> D["Other"]
    B --> B1{"AVX-512F?"}
    B1 -->|Yes| B1a["Avx512F64 / Avx512F32<br/>8xf64 or 16xf32<br/>hand_avx512.rs"]
    B1 -->|No| B2{"AVX2+FMA?"}
    B2 -->|Yes| B2a["Avx2F64 / Avx2F32<br/>4xf64 or 8xf32 with FMA"]
    B2 -->|No| B3{"AVX?"}
    B3 -->|Yes| B3a["AvxF64 / AvxF32<br/>4xf64 or 8xf32"]
    B3 -->|No| B4["Sse2F64 / Sse2F32<br/>2xf64 or 4xf32"]
    C --> C1{"SVE?"}
    C1 -->|Yes| C1a["SveF64 / SveF32<br/>Scalable width"]
    C1 -->|No| C2["NeonF64 / NeonF32<br/>2xf64 or 4xf32"]
    D --> SC["Scalar<br/>fallback"]
```

## Algorithm Selection

| Transform Size | Algorithm |
|----------------|-----------|
| Power of 2 | Cooley-Tukey DIT (radix-2/4/8, split-radix) |
| Power of 2 (large) | Stockham (cache-friendly, auto-sort) |
| Prime | Rader's algorithm or Bluestein |
| Composite | Mixed-radix or Cooley-Tukey generic |
| Small (≤8) | Direct O(n²) codelet |
| Any size | Bluestein (Chirp-Z, fallback) |

## SIMD Architecture

Runtime CPU feature detection selects the best backend:

- **x86_64**: AVX-512 > AVX2 > AVX > SSE2 > scalar
- **aarch64**: SVE > NEON > scalar
- **wasm32**: simd128 > scalar
- **other**: scalar fallback

For detailed algorithm citations and implementation notes, see the inline rustdoc comments
in each module.
