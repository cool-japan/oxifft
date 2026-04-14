# oxifft

[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](https://crates.io/crates/oxifft)
[![Tests](https://img.shields.io/badge/tests-858%20passing-brightgreen.svg)](#)
[![API Items](https://img.shields.io/badge/API%20items-688-blue.svg)](#)
[![Features](https://img.shields.io/badge/features-18-orange.svg)](#features)

Pure Rust implementation of FFTW - the Fastest Fourier Transform in the West.

## Overview

OxiFFT v0.2.0 is a production-ready FFT library with **858 passing tests**, **688 public API items**, and **zero unimplemented features**. It provides:

- **Complex DFT** (Discrete Fourier Transform) - forward & inverse
- **Real FFT** (R2C/C2R) - optimized real-valued transforms
- **DCT/DST** (Discrete Cosine/Sine Transforms) - all 8 variants
- **Multi-dimensional transforms** - 1D, 2D, 3D, and N-dimensional
- **Batch processing** - vectorized multi-transform execution
- **SIMD optimization** - SSE2, AVX, AVX-512, NEON, SVE
- **Threading support** - Rayon-based parallelism
- **Wisdom system** - plan caching and serialization
- **GPU acceleration** - CUDA and Metal backends
- **Sparse FFT** - O(k log n) for k-sparse signals
- **Streaming FFT** - STFT, spectrograms, MFCC
- **Signal processing** - Welch's method, Hilbert transform, cepstral analysis
- **Advanced transforms** - NUFFT, FrFT, convolution, autodiff

## Module Structure

```
src/
├── lib.rs              # Public API exports (688 items)
├── prelude.rs          # Internal prelude for no_std
│
├── api/                # User-facing API
│   ├── plan/           # Plan creation (Plan, RealPlan, GuruPlan, etc.)
│   ├── execute.rs      # Execution functions
│   ├── wisdom.rs       # Wisdom import/export
│   ├── memory.rs       # Aligned allocation (AlignedBuffer)
│   ├── types.rs        # Direction, Flags, R2rKind
│   └── parallel.rs     # Parallel convenience functions
│
├── kernel/             # Core infrastructure
│   ├── float.rs        # Float trait (f32/f64/f16/f128)
│   ├── complex.rs      # Complex<T> type
│   ├── tensor.rs       # IoDim, Tensor
│   ├── problem.rs      # Problem trait
│   ├── plan.rs         # Plan trait
│   ├── solver.rs       # Solver trait
│   ├── planner.rs      # Main planner + registry
│   ├── twiddle.rs      # Twiddle factor cache
│   ├── primes.rs       # Prime factorization
│   ├── f16.rs          # 16-bit float support
│   ├── f128_type.rs    # 128-bit float support
│   └── ...
│
├── dft/                # Complex DFT
│   ├── problem.rs      # DftProblem
│   ├── plan.rs         # DftPlan
│   ├── solvers/        # Algorithm implementations
│   │   ├── ct.rs       # Cooley-Tukey (radix-2/3/4/5)
│   │   ├── rader.rs    # Rader's algorithm (prime sizes)
│   │   ├── bluestein.rs # Bluestein/Chirp-Z
│   │   ├── direct.rs   # Direct DFT (small sizes)
│   │   └── nop.rs      # No-op (size 1)
│   └── codelets/       # Hand-optimized kernels
│
├── rdft/               # Real DFT
│   ├── problem.rs      # RdftProblem
│   ├── plan.rs         # RdftPlan
│   └── solvers/        # R2C, C2R, DHC, HCR
│
├── reodft/             # DCT/DST/DHT
│   ├── problem.rs      # ReodftProblem
│   ├── plan.rs         # ReodftPlan
│   └── solvers/        # DCT/DST variants, DHT
│
├── simd/               # SIMD abstraction
│   ├── traits.rs       # SimdVector, SimdReal, SimdComplex
│   ├── detect.rs       # Runtime feature detection
│   ├── scalar.rs       # Fallback (no SIMD)
│   ├── sse2.rs         # x86_64 SSE2
│   ├── avx.rs          # x86_64 AVX/AVX2
│   ├── avx512.rs       # x86_64 AVX-512
│   ├── neon.rs         # ARM NEON
│   └── sve.rs          # ARM SVE (feature-gated)
│
├── threading/          # Parallel execution
│   ├── spawn.rs        # ThreadPool trait
│   ├── serial.rs       # Single-threaded fallback
│   └── rayon_impl.rs   # Rayon-based parallelism
│
├── support/            # Utilities
│   ├── align.rs        # SIMD-aligned allocation
│   ├── copy.rs         # Optimized memcpy
│   └── transpose.rs    # Matrix transpose
│
├── sparse/             # Sparse FFT (feature-gated)
├── pruned/             # Pruned FFT (feature-gated)
├── streaming/          # STFT, spectrograms (feature-gated)
├── const_fft/          # Compile-time FFT (feature-gated)
├── gpu/                # CUDA/Metal backends (feature-gated)
├── mpi/                # MPI distributed (feature-gated)
├── wasm/               # WebAssembly (feature-gated)
├── signal/             # Signal processing (feature-gated)
├── nufft/              # Non-uniform FFT
├── frft/               # Fractional FFT
├── conv/               # Convolution
├── autodiff/           # Automatic differentiation
└── compat/             # FFTW-compatible API (feature-gated)
```

## FFTW API Compatibility

This crate aims for high API compatibility with FFTW. Key mappings:

| FFTW Function | OxiFFT Equivalent |
|---------------|-------------------|
| `fftw_plan_dft_1d` | `Plan::dft_1d()` |
| `fftw_plan_dft_2d` | `Plan2D::new()` |
| `fftw_plan_dft_3d` | `Plan3D::new()` |
| `fftw_plan_dft_r2c` | `RealPlan::r2c_1d()` |
| `fftw_plan_dft_c2r` | `RealPlan::c2r_1d()` |
| `fftw_plan_guru_dft` | `GuruPlan::dft()` |
| `fftw_execute` | `plan.execute()` |
| `fftw_export_wisdom_to_string` | `wisdom::export_to_string()` |
| `FFTW_MEASURE` | `Flags::MEASURE` |
| `FFTW_FORWARD` | `Direction::Forward` |

## Quick Start

```rust
use oxifft::{Complex, Direction, Flags, Plan, Plan2D, RealPlan};

// 1D Complex FFT (256 points)
let plan = Plan::dft_1d(256, Direction::Forward, Flags::MEASURE)?;
let input = vec![Complex::new(1.0, 0.0); 256];
let mut output = vec![Complex::zero(); 256];
plan.execute(&input, &mut output);

// 2D FFT (64x64)
let plan_2d = Plan2D::new(64, 64, Direction::Forward, Flags::ESTIMATE)?;
let input_2d = vec![Complex::zero(); 64 * 64];
let mut output_2d = vec![Complex::zero(); 64 * 64];
plan_2d.execute(&input_2d, &mut output_2d);

// Real-to-Complex FFT (optimized for real input)
let plan_r2c = RealPlan::r2c_1d(256, Flags::MEASURE)?;
let real_input = vec![0.0f64; 256];
let mut complex_output = vec![Complex::zero(); 129]; // n/2 + 1
plan_r2c.execute_r2c(&real_input, &mut complex_output);

// Convenience functions (no plan needed)
use oxifft::{fft, ifft, rfft, irfft};
let data = vec![Complex::new(1.0, 0.0); 256];
let spectrum = fft(&data)?;
let recovered = ifft(&spectrum)?;
```

## Features

OxiFFT provides 18 configurable feature flags:

### Core Features (enabled by default)
- **`std`** - Standard library support (file I/O, timing, allocations)
- **`threading`** - Rayon-based parallelism (requires `std`)

### SIMD & Hardware Acceleration
- **`simd`** - Explicit SIMD optimizations (SSE2, AVX, NEON)
- **`portable_simd`** - Nightly portable SIMD API (requires nightly)
- **`sve`** - ARM SVE (Scalable Vector Extension) support

### Precision Options
- **`f16-support`** - Half-precision (16-bit) floats
- **`f128-support`** - Quad-precision (128-bit) floats

### Advanced Algorithms
- **`sparse`** - Sparse FFT (O(k log n) for k-sparse signals)
- **`pruned`** - Pruned FFT (partial input/output computation)
- **`streaming`** - STFT, spectrograms, window functions, MFCC
- **`const-fft`** - Compile-time FFT with const generics

### GPU & Distributed Computing
- **`gpu`** - GPU acceleration meta-feature (enables cuda + metal)
- **`cuda`** - CUDA backend for NVIDIA GPUs
- **`metal`** - Metal backend for Apple GPUs
- **`mpi`** - MPI distributed computing (requires MPI library)

### Additional Features
- **`signal`** - Signal processing (Welch, Hilbert, cepstral analysis)
- **`wasm`** - WebAssembly support
- **`fftw-compat`** - FFTW-compatible API surface

### Usage

```toml
[dependencies]
oxifft = { version = "0.2.0", features = ["sparse", "streaming", "signal"] }
```

## API Overview

OxiFFT exports 688 public API items organized into these categories:

### High-Level Plans
- `Plan`, `Plan2D`, `Plan3D`, `PlanND` - Complex FFT plans
- `RealPlan`, `RealPlan2D`, `RealPlan3D`, `RealPlanND` - Real FFT plans
- `SplitPlan`, `SplitPlan2D`, `SplitPlan3D`, `SplitPlanND` - Split-complex plans
- `GuruPlan` - Advanced multi-dimensional transforms
- `R2rPlan` - DCT/DST/DHT plans

### Convenience Functions
- `fft()`, `ifft()` - 1D complex transforms
- `fft2d()`, `ifft2d()` - 2D transforms
- `fft_nd()`, `ifft_nd()` - N-dimensional transforms
- `rfft()`, `irfft()` - Real FFT convenience functions
- `fft_batch()`, `ifft_batch()` - Batch processing
- `dct1()` through `dct4()`, `dst1()` through `dst4()` - DCT/DST
- `dht()` - Discrete Hartley Transform

### Advanced Transforms (feature-gated)
- `SparsePlan`, `sparse_fft()`, `sparse_ifft()` - Sparse FFT
- `PrunedPlan`, `fft_pruned_input()`, `fft_pruned_output()` - Pruned FFT
- `StreamingFft`, `stft()`, `istft()` - Short-time FFT
- `Nufft` - Non-uniform FFT (1D/2D/3D, all 3 types)
- `Frft` - Fractional Fourier Transform
- `ConstFft` - Compile-time FFT
- `GpuFft`, `GpuPlan` - GPU-accelerated FFT

### Signal Processing (with `signal` feature)
- `welch()` - Welch's power spectral density
- `hilbert()` - Hilbert transform (analytic signal)
- `envelope()`, `instantaneous_phase()`, `instantaneous_frequency()`
- `real_cepstrum()`, `complex_cepstrum()`, `minimum_phase()`
- `periodogram()`, `coherence()`, `cross_spectral_density()`
- `resample()`, `resample_to()` - Sample rate conversion
- `mel_spectrogram()`, `mfcc()` - Audio feature extraction

### Convolution
- `convolve()`, `correlate()` - Real/complex convolution
- `convolve_mode()` - 'full', 'same', 'valid' modes
- `polynomial_multiply()`, `polynomial_power()` - Polynomial operations

### Automatic Differentiation
- `fft_dual()`, `grad_fft()`, `grad_ifft()` - Forward/reverse mode AD
- `jvp_fft()`, `vjp_fft()` - Jacobian-vector products
- `DiffFftPlan` - Differentiable FFT plans

### Memory Management
- `alloc_complex()`, `alloc_real()` - Aligned allocation
- `AlignedBuffer<T>` - RAII aligned buffers
- `is_aligned()` - Alignment checking

### Core Types
- `Complex<T>` - Complex number type
- `Float` trait - Generic over `f32`/`f64`/`f16`/`f128`
- `Direction` - Forward/Backward
- `Flags` - MEASURE, ESTIMATE, EXHAUSTIVE, etc.
- `IoDim`, `Tensor` - Multi-dimensional array descriptors

## Examples

See the `examples/` directory for complete examples:
- `simple_fft.rs` - Basic 1D FFT usage
- `real_fft.rs` - Real-to-complex FFT
- `multidimensional.rs` - 2D/3D transforms
- `batch_fft.rs` - Batch processing
- `convolution.rs` - Convolution via FFT
- `autodiff_fft.rs` - Automatic differentiation
- `sparse_fft.rs` - Sparse FFT (requires `sparse` feature)
- `streaming_fft.rs` - STFT and spectrograms (requires `streaming` feature)
- `signal_processing.rs` - Welch, Hilbert, etc. (requires `signal` feature)
- `wisdom_usage.rs` - Plan caching
- `nufft_example.rs` - Non-uniform FFT

## Testing

OxiFFT has comprehensive test coverage:

- **858 tests passing** across unit, integration, and property-based tests
- **0 unimplemented items** - all features are fully implemented
- Fuzz testing with `proptest` and `cargo-fuzz`
- Comparison tests against `rustfft` and FFTW
- CI validation on multiple platforms (Linux, macOS, Windows)

## Performance

OxiFFT achieves competitive performance with FFTW:
- SIMD-optimized kernels for x86-64 (SSE2/AVX/AVX-512) and ARM (NEON/SVE)
- Parallel execution via Rayon for large transforms
- GPU acceleration via CUDA and Metal
- Wisdom system for amortizing planning costs

See `BENCHMARKING.md` and `PERFORMANCE_ANALYSIS.md` for detailed benchmarks.

## License

Same as the parent OxiFFT project.
