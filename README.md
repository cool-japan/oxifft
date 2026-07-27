# OxiFFT

[![Crates.io](https://img.shields.io/crates/v/oxifft.svg)](https://crates.io/crates/oxifft)
[![Documentation](https://docs.rs/oxifft/badge.svg)](https://docs.rs/oxifft)
[![License](https://img.shields.io/crates/l/oxifft.svg)](https://github.com/cool-japan/oxifft#license)
[![Downloads](https://img.shields.io/crates/d/oxifft.svg)](https://crates.io/crates/oxifft)

**Pure Rust implementation of FFTW (Fastest Fourier Transform in the West)**

OxiFFT is a 99% Rust port of FFTW3, the world's most respected FFT library. It brings FFTW's sophisticated algorithms, planning system, and performance optimizations to the Rust ecosystem while leveraging Rust's safety guarantees and modern language features.

## Features

### Core FFT Functionality
- **Pure Rust by default**: No C dependencies, no FFI, no bindgen — every feature of the `oxifft` crate is pure Rust (Pure Rust Policy compliant), including `sve` (ARM SVE detection uses `std::arch`, not `libc`). Distributed MPI support lives in the **separate** `oxifft-adapter-mpi` crate, which links a system MPI library — see [MPI notes](#mpi).
- **Full Algorithm Support**: Cooley-Tukey (radix-2/4/8, split-radix, mixed-radix for smooth-7 composites), Stockham auto-sort, cache-oblivious recursion, Rader (primes), Bluestein, Direct O(n²); `Algorithm::MixedRadix` handles sizes such as 6, 10, 12, 14, 24, 28, 40, 56, 80, 96, 112, 240, …
- **Transform Types**: Complex DFT, Real FFT (R2C/C2R), DCT/DST variants, DHT
- **Multi-Dimensional**: 1D, 2D, 3D, and N-D transforms
- **Batch Processing**: Efficient vector-rank handling for multiple transforms
- **SIMD Optimization**: SSE2, AVX, AVX2, AVX-512, ARM NEON, ARM SVE, WebAssembly SIMD (runtime-detected; currently applied to small fixed-size codelets and Rader/Bluestein pointwise multiplies — the general large-N Stockham butterfly path is still scalar)
- **Threading**: Rayon integration for parallel execution
- **Wisdom System**: Plan caching and persistence; `Flags::MEASURE`/`PATIENT`/`EXHAUSTIVE` consult the runtime wisdom cache before benchmarking, and `Flags::WISDOM_ONLY` plans exclusively from stored wisdom (FFTW semantics)
- **Auto-tuning**: `Flags::MEASURE`/`PATIENT`/`EXHAUSTIVE` genuinely compare multiple candidate algorithms at plan time and cache the fastest via the wisdom system; the `oxifft_tune` binary (built from `src/bin/oxifft_tune.rs`) benchmarks candidate plans and writes wisdom, and a build-time baseline can be embedded with `OXIFFT_TUNE=1`
- **Precision Support**: f16, f32, f64, and f128 floating-point types (`Plan<T>` is generic over the internal `Float` trait)

### Advanced Features (Beyond FFTW)
- **Sparse FFT**: O(k log n) complexity for k-sparse signals using FFAST algorithm
- **Pruned FFT**: Input/output pruning for partial computation, Goertzel algorithm
- **Streaming FFT**: STFT with window functions, mel-frequency filterbank, MFCC (`streaming` feature)
- **Compile-time FFT**: Const FFT for fixed-size arrays (sizes 2-1024)
- **Non-uniform FFT (NUFFT)**: Type 1/2/3 transforms with Gaussian gridding
- **Fractional Fourier Transform (FrFT)**: Chirp decomposition for fractional orders
- **Convolution**: FFT-based linear/circular convolution and correlation
- **Automatic Differentiation**: Forward and backward mode gradients for FFT operations
- **Signal Processing**: Hilbert transform, analytic signal, Welch's PSD, cross-spectral density, coherence, cepstral analysis, FFT-based resampling (`signal` feature)
- **GPU Acceleration**: Metal (Apple) runs transforms on the GPU today; the CUDA (NVIDIA) backend opens a real context/stream/plan but currently evaluates on the CPU (device kernel dispatch is pending in `oxicuda-fft`). Query the status at runtime via `GpuFft::execution_target()` / `GpuCapabilities::hardware_accelerated`
- **MPI Distributed Computing**: 2D/3D/N-D distributed FFTs with slab decomposition — provided by the separate `oxifft-adapter-mpi` crate (links a system OpenMPI/MPICH library; see [MPI notes](#mpi))
- **WebAssembly**: Browser-compatible FFT with WASM SIMD support

## Project Status

✅ **Core FFT functionality is COMPLETE**
✅ **1,730 tests passing** across the workspace (all features; 7 `#[ignore]`d long-running/environment tests), plus **107 doctests, 1 ignored**
✅ **Zero clippy warnings** (all features)
⚠️ **Performance vs. RustFFT/FFTW is mixed** — OxiFFT wins some sizes and loses others (notably large power-of-2, prime, and some composite sizes). The committed FFTW baseline (`benches/baselines/v0.3.0/`: geomean ratio 1.50×, 4/7 v1.0 gates passing) is the source of truth; see [BENCHMARKING.md](./BENCHMARKING.md) and [Known Performance Gaps](#known-performance-gaps-vs-fftw). General parity is **not** yet claimed.
✅ **Extensively documented public API** — every rustdoc example compiles and runs under `cargo test --doc`
✅ **78K+ lines of code** across 5 crates (78,090 SLoC)

See [PROJECT_STATUS.md](./PROJECT_STATUS.md) for comprehensive status, [oxifft.md](./oxifft.md) for architecture blueprint, and [TODO.md](./TODO.md) for detailed roadmap.

## Usage

```rust
use oxifft::{Complex, Direction, Flags, Plan, Plan2D, RealPlan};

// Simple 1D Complex FFT
let input: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); 256];
let mut output: Vec<Complex<f64>> = vec![Complex::zero(); 256];

let plan = Plan::dft_1d(256, Direction::Forward, Flags::MEASURE).unwrap();
plan.execute(&input, &mut output);

// 2D Complex FFT
let plan_2d = Plan2D::new(64, 64, Direction::Forward, Flags::ESTIMATE).unwrap();
let input_2d: Vec<Complex<f64>> = vec![Complex::zero(); 64 * 64];
let mut output_2d: Vec<Complex<f64>> = vec![Complex::zero(); 64 * 64];
plan_2d.execute(&input_2d, &mut output_2d);

// Real-to-Complex FFT
let real_input: Vec<f64> = vec![0.0; 256];
let mut complex_output: Vec<Complex<f64>> = vec![Complex::zero(); 129]; // n/2 + 1

let plan_r2c = RealPlan::r2c_1d(256, Flags::MEASURE).unwrap();
plan_r2c.execute_r2c(&real_input, &mut complex_output);
```

### Guru Interface (Maximum Flexibility)

```rust
use oxifft::{Complex, IoDim, Tensor, GuruPlan, Direction, Flags};

// Batch of 100 transforms, each 512-point
let dims = Tensor::new(vec![IoDim::contiguous(512)]);
let howmany = Tensor::new(vec![IoDim::new(100, 512, 512)]);

let plan = GuruPlan::dft(&dims, &howmany, Direction::Forward, Flags::MEASURE).unwrap();

let input: Vec<Complex<f64>> = vec![Complex::zero(); 512 * 100];
let mut output: Vec<Complex<f64>> = vec![Complex::zero(); 512 * 100];
plan.execute(&input, &mut output);
```

### Wisdom Management

```rust
use oxifft::api::{export_to_file, import_from_file, forget};
use std::path::Path;

// Export/import wisdom (both take &Path, not &str)
export_to_file(Path::new("my_wisdom.txt"))?;
import_from_file(Path::new("my_wisdom.txt"))?;

// Clear the in-process wisdom cache
forget();
```

### Advanced Features Examples

#### Sparse FFT (for k-sparse signals)
```rust
use oxifft::{sparse_fft, SparsePlan};

// Signal with only 10 non-zero frequency components
let signal = vec![Complex::new(1.0, 0.0); 1024];
let k = 10; // Expected sparsity

// One-shot API: O(k log n) instead of O(n log n)
let result = sparse_fft(&signal, k);
for (freq_idx, value) in result.indices.iter().zip(result.values.iter()) {
    println!("Frequency {}: {:?}", freq_idx, value);
}

// Plan-based API for repeated use
let plan = SparsePlan::new(1024, k, Flags::ESTIMATE).unwrap();
let result = plan.execute(&signal);
```

#### Streaming FFT (STFT for real-time processing)
```rust
use oxifft::{stft, istft, StreamingFft, WindowFunction};

// Perform Short-Time Fourier Transform
let window_size = 512;
let hop_size = 256;
let spectrogram = stft(&audio_signal, window_size, hop_size, WindowFunction::Hann);

// Reconstruct signal from STFT (istft takes hop_size + window; the frame size
// is recovered from the spectrogram itself)
let reconstructed = istft(&spectrogram, hop_size, WindowFunction::Hann);

// Real-time streaming: feed samples, then drain completed frames
let mut streaming_fft = StreamingFft::new(window_size, hop_size, WindowFunction::Hamming);
streaming_fft.feed(&audio_signal);
while let Some(spectrum) = streaming_fft.pop_frame() {
    // Process spectrum in real-time
    let _ = spectrum;
}
```

#### Compile-time FFT (zero runtime overhead)
```rust
use oxifft::{fft_fixed, ifft_fixed};

// Fixed-size FFT computed at compile time
let input: [Complex<f64>; 8] = [Complex::new(1.0, 0.0); 8];
let output = fft_fixed(&input);
let reconstructed = ifft_fixed(&output);
```

#### Non-uniform FFT (NUFFT for irregularly spaced data)
```rust
use oxifft::{nufft_type1, nufft_type2, Nufft, NufftType};

// Type 1: Non-uniform to uniform (analysis)
let non_uniform_points = vec![0.1, 0.3, 0.7, 0.9]; // Irregular sampling
let values = vec![Complex::new(1.0, 0.0); 4];
let spectrum = nufft_type1(&non_uniform_points, &values, 16, 1e-6)?;

// Type 2: Uniform to non-uniform (synthesis)
// Signature is nufft_type2(coeffs, points, tolerance) — spectrum first.
let uniform_spectrum = vec![Complex::new(1.0, 0.0); 16];
let interpolated = nufft_type2(&uniform_spectrum, &non_uniform_points, 1e-6)?;
```

#### Automatic Differentiation for FFT
```rust
use oxifft::{grad_fft, vjp_fft, fft_jacobian};

// Compute gradient of loss w.r.t. FFT input (returns Option)
let grad_output = vec![Complex::new(0.1, 0.0); 256]; // Gradient from loss
let grad_input = grad_fft(&grad_output);

// Vector-Jacobian product for backpropagation (takes only the cotangent)
let vjp = vjp_fft(&grad_output);

// Full Jacobian matrix (for analysis) — a plain Vec, no `?`
let jacobian: Vec<Vec<Complex<f64>>> = fft_jacobian(256);
```

#### WebAssembly (Browser)
```bash
# Build for web
wasm-pack build oxifft --target web --features wasm
```

```javascript
import init, { WasmFft, fft_f64, ifft_f64 } from './oxifft';

await init();

// Plan-based API (efficient for repeated use)
const fft = new WasmFft(256);
const real = new Float64Array([1, 2, 3, ...]);
const imag = new Float64Array([0, 0, 0, ...]);
const result = fft.forward(real, imag);  // [re0, im0, re1, im1, ...]

// One-shot API
const output = fft_f64(real, imag);
```

#### GPU Acceleration
```rust
use oxifft::gpu::{GpuFft, GpuBackend};

// Auto-detect the best available backend. Metal runs on the GPU; the CUDA
// backend currently evaluates on the CPU (see GPU note above). forward/inverse
// take `&mut self`, so the plan must be `mut`.
let mut gpu_fft = GpuFft::new(4096, GpuBackend::Auto)?;

let input = vec![Complex::new(1.0, 0.0); 4096];
let output = gpu_fft.forward(&input)?;
let reconstructed = gpu_fft.inverse(&output)?;
```

## Workspace Structure

```
oxifft/
├── src/                    # Main library source
│   ├── api/               # Public user-facing API
│   ├── kernel/            # Core planner & data structures (F16, F128 types)
│   ├── dft/               # Complex DFT implementations
│   ├── rdft/              # Real DFT implementations
│   ├── reodft/            # DCT/DST (Real Even/Odd DFT)
│   ├── simd/              # SIMD abstraction (SSE2, AVX, AVX2, AVX-512, NEON, SVE)
│   ├── threading/         # Parallel execution (Rayon integration)
│   ├── support/           # Utilities (alignment, transpose, copy)
│   ├── sparse/            # Sparse FFT (FFAST algorithm)
│   ├── pruned/            # Pruned FFT (input/output pruning, Goertzel)
│   ├── streaming/         # STFT and window functions
│   ├── signal/            # Hilbert transform, PSD, cepstrum, resampling
│   ├── const_fft/         # Compile-time FFT with const generics
│   ├── nufft/             # Non-uniform FFT (Type 1/2/3)
│   ├── frft/              # Fractional Fourier Transform
│   ├── conv/              # FFT-based convolution and correlation
│   ├── autodiff/          # Automatic differentiation for FFT
│   ├── conv/ ntt/ frft/   # Convolution, number-theoretic transform, fractional FFT
│   ├── chirp_z/           # Chirp-Z transform
│   ├── compat/            # FFTW-compatible (fftw_*) API surface
│   ├── gpu/               # GPU acceleration (Metal GPU; CUDA CPU-fallback)
│   └── wasm/              # WebAssembly bindings and WASM SIMD
├── oxifft-codegen/        # Proc-macro façade crate (re-exports the codelet macros)
├── oxifft-codegen-impl/   # Codelet-generation implementation crate
├── oxifft-adapter-mpi/    # MPI distributed FFT (separate crate; system MPI FFI, quarantined)
├── oxifft-bench/          # RustFFT/FFTW comparison benchmarks + fftw_ratio_report
├── benches/               # Additional benchmarks + committed baselines (baselines/)
├── examples/              # Usage examples
└── tests/                 # Integration tests (size coverage, FFTW comparison)
```

> Note: there is no `oxifft/src/mpi/` module — MPI lives entirely in the
> separate `oxifft-adapter-mpi` crate so that `oxifft` stays pure Rust under
> `--all-features`.

## Architecture

OxiFFT follows FFTW's proven design patterns:

- **Problem-Plan-Solver Hierarchy**: Trait-based abstractions for maximum flexibility
- **Wisdom System**: Cache optimal plans for repeated problem sizes
- **Modular Solvers**: Easy to add new algorithms without breaking existing code
- **Codelet Generation**: Proc-macros generate optimized kernels at compile-time

### Core Types

The public surface is a family of concrete, `Send + Sync` plan structs
(`Plan`, `Plan2D`, `Plan3D`, `PlanND`, `RealPlan*`, `R2rPlan*`, `SplitPlan*`,
`GuruPlan`). Internally, `kernel::Planner` enumerates candidate solvers for a
size and `wisdom::WisdomEntry` records the winning choice so it can be cached
and replayed:

```rust
use oxifft::{Direction, Flags, Plan};

// Construction selects (and, under MEASURE/PATIENT, benchmarks + caches) a solver.
let plan = Plan::<f64>::dft_1d(1024, Direction::Forward, Flags::MEASURE);
```

## Comparison with RustFFT

OxiFFT provides many features beyond RustFFT:

| Feature | OxiFFT | RustFFT |
|---------|--------|---------|
| **Basic FFT** | ✅ | ✅ |
| **Mixed-radix (smooth-7 composites)** | ✅ `Algorithm::MixedRadix` | partial |
| **Real FFT (R2C/C2R)** | ✅ | ✅ |
| **DCT/DST (8 types)** | ✅ | ❌ |
| **2D/3D/N-D FFT** | ✅ | ❌ (manual) |
| **Batch FFT** | ✅ | ❌ (loop) |
| **Wisdom System** | ✅ | ❌ |
| **WASM Support** | ✅ | ❌ |
| **Sparse FFT** | ✅ O(k log n) | ❌ |
| **Pruned FFT** | ✅ | ❌ |
| **STFT/Streaming** | ✅ | ❌ |
| **NUFFT** | ✅ | ❌ |
| **Fractional FFT** | ✅ | ❌ |
| **Convolution** | ✅ | ❌ |
| **Auto-Differentiation** | ✅ | ❌ |
| **GPU (Metal)** | ✅ GPU | ❌ |
| **GPU (CUDA)** | ⚠️ CPU-fallback | ❌ |
| **MPI Distributed** | ✅ (separate `oxifft-adapter-mpi` crate) | ❌ |
| **f16/f128 Support** | ✅ | ❌ |
| **Const-FFT** | ✅ | ❌ |
| **Signal Processing (Hilbert/PSD)** | ✅ | ❌ |
| **Mel-Frequency / MFCC** | ✅ | ❌ |
| **Split-Complex** | ✅ | ❌ |
| **Guru Interface (c2c DFT)** | ✅ | ❌ |

### When to Use OxiFFT

- **SAR/Radar Processing**: 2D FFT, batch processing, real FFT, convolution
- **Audio Processing**: STFT, mel-frequency filterbank, MFCC, Hilbert envelope
- **Scientific Computing**: NUFFT for irregular sampling, MPI for HPC
- **Machine Learning**: Auto-differentiation, GPU acceleration
- **Embedded/Web**: WASM support, const-FFT for fixed sizes
- **Signal Analysis**: Sparse FFT for compressed sensing, pruned FFT for specific frequencies

## Performance Targets

| Transform Type | Size | Target |
|----------------|------|--------|
| 1D Complex DFT | 2^10 | Within 2x of FFTW |
| 1D Complex DFT | 2^20 | Within 2x of FFTW |
| 1D Real FFT | 2^10 | Within 2x of FFTW |
| 2D Complex DFT | 1024x1024 | Within 2x of FFTW |
| Batch 1D DFT | 1000x256 | Within 2x of FFTW |
| Prime size DFT | 2017 | Within 3x of FFTW |

**Stretch goal**: Match or exceed FFTW performance for common sizes.

### Known Performance Gaps vs FFTW

<a name="known-performance-gaps-vs-fftw"></a>

These are **targets, not current results.** The committed FFTW baseline
(`benches/baselines/v0.3.0/fftw_ratios_2026-04-20.json`, measured mid-development
on Apple Silicon) records geomean ratio **1.50×** with **4 of 7** v1.0 gates
passing. The three failing gates (ratio = OxiFFT / FFTW, lower is better):

| Gate | Ratio | Target | Status |
|------|-------|--------|--------|
| `1d_cplx_2e20` (2^20 complex DFT) | 3.67× | < 2.0 | ❌ FAIL |
| `1d_real_2e10` (2^10 real FFT) | 3.95× | < 2.0 | ❌ FAIL |
| `dct2_1024` (DCT-II, 1024) | 3.90× | < 3.0 | ❌ FAIL |

Against RustFFT, OxiFFT is competitive at some small/medium sizes but currently
slower on large power-of-2 (up to ~3.4× at 65536), prime (up to ~7.8× at n=17),
and several composite sizes. The root cause is that SIMD acceleration is applied
only to small fixed-size codelets and pointwise multiplies today — the general
large-N Stockham butterfly path is still scalar. Re-run and refresh the baseline
with `fftw_ratio_report` (see [BENCHMARKING.md](./BENCHMARKING.md)) before any
v1.0 performance announcement.

## Dependencies

Add OxiFFT to your `Cargo.toml`:

```toml
[dependencies]
oxifft = "0.4"
```

The `oxifft` crate pulls in only pure-Rust dependencies: `num-complex`,
`num-traits`, `serde`/`serde_json`, `hashbrown`, `spin`, `libm`, and — behind
optional features — `rayon`, `wasm-bindgen`/`js-sys`, `ndarray`, and the
`oxicuda-*` GPU driver crates. There is **no** `mpi`, `libc`, `seahash`, or
`simd` dependency/feature: SIMD is always-on via runtime detection, and MPI is
a separate crate.

### Cargo features (matches `oxifft/Cargo.toml`)

```toml
[features]
default       = ["std", "threading"]
std           = [...]            # File I/O, timing, std collections
threading     = ["std", ...]     # Rayon-based parallel execution
avx512        = []               # AVX-512 SIMD tier (x86_64)
portable_simd = []               # Experimental core::simd backend (NIGHTLY only)
f16-support   = ["std"]          # Half-precision (f16)
f128-support  = ["std"]          # Quad-precision (f128)
sparse        = ["std"]          # Sparse FFT (FFAST)
pruned        = ["std"]          # Pruned / partial FFT
sve           = ["std"]          # ARM SVE (pure Rust; std::arch detection)
wasm          = ["std", ...]     # WebAssembly bindings
streaming     = ["std"]          # STFT / windows / mel / MFCC
const-fft     = []               # Compile-time fixed-size FFT
signal        = ["std"]          # Hilbert, Welch PSD, cepstrum, resampling
fftw-compat   = []               # FFTW-named (fftw_*) API surface
ndarray       = ["std", ...]     # ndarray integration
cuda          = ["std", ...]     # CUDA backend (currently CPU-fallback)
metal         = ["std", ...]     # Metal backend (real GPU on Apple Silicon)
gpu           = ["std", "cuda", "metal"]
```

## Documentation

### Project Overview

- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - 📊 Current project status, metrics, and priorities
- **[README.md](README.md)** - This file - project overview and quick start

### User Guides

- **[BENCHMARKING.md](BENCHMARKING.md)** - Comprehensive guide to running performance benchmarks
- **[PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md)** - Performance analysis and optimization guide
- **[TESTING.md](TESTING.md)** - Testing methodology and validation procedures
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines and project policies

### Architecture & Planning

- **[oxifft.md](oxifft.md)** - Architecture and implementation blueprint
- **[TODO.md](TODO.md)** - Detailed implementation status and roadmap
- **[CHANGELOG.md](CHANGELOG.md)** - Project history and release notes

### Benchmark Reports

- **[BENCHMARK_RESULTS_TEMPLATE.md](BENCHMARK_RESULTS_TEMPLATE.md)** - Template for documenting benchmark results

## References

- FFTW Paper: "The Design and Implementation of FFTW3" (Frigo & Johnson, 2005)
- Cooley-Tukey: "An Algorithm for the Machine Calculation of Complex Fourier Series" (1965)
- Rader's Algorithm: "Discrete Fourier transforms when the number of data samples is prime" (1968)
- Bluestein: "A linear filtering approach to the computation of discrete Fourier transform" (1970)

## MPI

<a name="mpi"></a>

Distributed FFT across multiple compute nodes lives in the **separate**
`oxifft-adapter-mpi` crate — it is **not** a feature of `oxifft`. Keeping MPI in
its own crate is deliberate: it quarantines the MPI FFI so that `oxifft` itself
stays 100% pure Rust even under `--all-features`.

**⚠ C system dependency:** `oxifft-adapter-mpi` links a system MPI library
(OpenMPI, MPICH, or equivalent) and needs `libclang` at build time (for
`bindgen`). This is intentionally not pure Rust. No pure-Rust MPI implementation
covering the required API surface exists as of 2026; if one appears, OxiFFT will
adopt it.

To use MPI:
```toml
[dependencies]
oxifft-adapter-mpi = "0.4"
```

Ensure a system MPI library is installed (`brew install openmpi` on macOS,
`apt-get install libopenmpi-dev libclang-dev` on Debian/Ubuntu), then launch
with `mpirun`/`mpiexec` as usual. See `oxifft-adapter-mpi/README.md` for the
full quickstart.

## Sponsorship

OxiFFT is developed and maintained by **COOLJAPAN OU (Team Kitasan)**.

If you find OxiFFT useful, please consider sponsoring the project to support continued development of the Pure Rust ecosystem.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps us:
- Maintain and improve the COOLJAPAN ecosystem
- Keep the entire ecosystem (OxiBLAS, OxiFFT, SciRS2, etc.) 100% Pure Rust
- Provide long-term support and security updates

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be licensed under Apache-2.0, without any additional terms or conditions.

Copyright (c) 2026 COOLJAPAN OU (Team KitaSan)
