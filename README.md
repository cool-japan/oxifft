# OxiFFT

[![Crates.io](https://img.shields.io/crates/v/oxifft.svg)](https://crates.io/crates/oxifft)
[![Documentation](https://docs.rs/oxifft/badge.svg)](https://docs.rs/oxifft)
[![License](https://img.shields.io/crates/l/oxifft.svg)](https://github.com/cool-japan/oxifft#license)
[![Downloads](https://img.shields.io/crates/d/oxifft.svg)](https://crates.io/crates/oxifft)

**Pure Rust implementation of FFTW (Fastest Fourier Transform in the West)**

OxiFFT is a 99% Rust port of FFTW3, the world's most respected FFT library. It brings FFTW's sophisticated algorithms, planning system, and performance optimizations to the Rust ecosystem while leveraging Rust's safety guarantees and modern language features.

## Features

### Core FFT Functionality
- **Pure Rust**: No C dependencies, no FFI, no bindgen (Pure Rust Policy compliant)
- **Full Algorithm Support**: Cooley-Tukey, Rader, Bluestein, Direct O(n²)
- **Transform Types**: Complex DFT, Real FFT (R2C/C2R), DCT/DST variants, DHT
- **Multi-Dimensional**: 1D, 2D, 3D, and N-D transforms
- **Batch Processing**: Efficient vector-rank handling for multiple transforms
- **SIMD Optimization**: SSE2, AVX, AVX2, AVX-512, ARM NEON, ARM SVE, WebAssembly SIMD
- **Threading**: Rayon integration for parallel execution
- **Wisdom System**: Plan caching and persistence for repeated transforms
- **Precision Support**: f16, f32, f64, and f128 floating-point types

### Advanced Features (Beyond FFTW)
- **Sparse FFT**: O(k log n) complexity for k-sparse signals using FFAST algorithm
- **Pruned FFT**: Input/output pruning for partial computation, Goertzel algorithm
- **Streaming FFT**: Short-Time Fourier Transform (STFT) with window functions
- **Compile-time FFT**: Const FFT for fixed-size arrays (sizes 2-1024)
- **Non-uniform FFT (NUFFT)**: Type 1/2/3 transforms with Gaussian gridding
- **Fractional Fourier Transform (FrFT)**: Chirp decomposition for fractional orders
- **Convolution**: FFT-based linear/circular convolution and correlation
- **Automatic Differentiation**: Forward and backward mode gradients for FFT operations
- **GPU Acceleration**: CUDA (NVIDIA) and Metal (Apple) backends
- **MPI Distributed Computing**: 2D/3D/N-D distributed FFTs with slab decomposition
- **WebAssembly**: Browser-compatible FFT with WASM SIMD support

## Project Status

✅ **Core FFT functionality is COMPLETE**
✅ **357 unit tests + doc tests passing** (100% pass rate)
✅ **Zero clippy warnings** (all features)
✅ **Performance optimized** (9/15 composite sizes faster than RustFFT)

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
let input_2d = vec![Complex::zero(); 64 * 64];
let mut output_2d = vec![Complex::zero(); 64 * 64];
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

let input = vec![Complex::zero(); 512 * 100];
let mut output = vec![Complex::zero(); 512 * 100];
plan.execute(&input, &mut output);
```

### Wisdom Management

```rust
use oxifft::wisdom;

// Export/import wisdom
wisdom::export_to_file("my_wisdom.txt")?;
wisdom::import_from_file("my_wisdom.txt")?;
wisdom::forget();
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

// Reconstruct signal from STFT
let reconstructed = istft(&spectrogram, window_size, hop_size, WindowFunction::Hann);

// Real-time streaming
let mut streaming_fft = StreamingFft::new(window_size, hop_size, WindowFunction::Hamming);
for frame in audio_chunks {
    let spectrum = streaming_fft.process_frame(&frame);
    // Process spectrum in real-time
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
let uniform_spectrum = vec![Complex::new(1.0, 0.0); 16];
let interpolated = nufft_type2(&non_uniform_points, &uniform_spectrum, 1e-6)?;
```

#### Automatic Differentiation for FFT
```rust
use oxifft::{grad_fft, vjp_fft, fft_jacobian};

// Compute gradient of loss w.r.t. FFT input
let input = vec![Complex::new(1.0, 0.0); 256];
let grad_output = vec![Complex::new(0.1, 0.0); 256]; // Gradient from loss
let grad_input = grad_fft(&grad_output, 256)?;

// Vector-Jacobian product for backpropagation
let vjp = vjp_fft(&input, &grad_output)?;

// Full Jacobian matrix (for analysis)
let jacobian = fft_jacobian(256)?;
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

// Auto-detect best available GPU backend (CUDA or Metal)
let gpu_fft = GpuFft::new(4096, GpuBackend::Auto)?;

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
│   ├── const_fft/         # Compile-time FFT with const generics
│   ├── nufft/             # Non-uniform FFT (Type 1/2/3)
│   ├── frft/              # Fractional Fourier Transform
│   ├── conv/              # FFT-based convolution and correlation
│   ├── autodiff/          # Automatic differentiation for FFT
│   ├── gpu/               # GPU acceleration (CUDA, Metal backends)
│   ├── mpi/               # MPI distributed computing
│   └── wasm/              # WebAssembly bindings and WASM SIMD
├── oxifft-codegen/        # Proc-macro crate for codelet generation
├── oxifft-bench/          # Benchmarks (including FFTW comparison)
├── benches/               # Additional benchmarks (beyond_fftw.rs)
├── examples/              # Usage examples
└── tests/                 # Integration tests (629 tests passing)
```

## Architecture

OxiFFT follows FFTW's proven design patterns:

- **Problem-Plan-Solver Hierarchy**: Trait-based abstractions for maximum flexibility
- **Wisdom System**: Cache optimal plans for repeated problem sizes
- **Modular Solvers**: Easy to add new algorithms without breaking existing code
- **Codelet Generation**: Proc-macros generate optimized kernels at compile-time

### Core Traits

```rust
pub trait Problem: Hash + Debug + Clone + Send + Sync { ... }
pub trait Plan: Send + Sync { ... }
pub trait Solver: Send + Sync { ... }
```

## Comparison with RustFFT

OxiFFT provides many features beyond RustFFT:

| Feature | OxiFFT | RustFFT |
|---------|--------|---------|
| **Basic FFT** | ✅ | ✅ |
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
| **GPU (CUDA/Metal)** | ✅ | ❌ |
| **MPI Distributed** | ✅ | ❌ |
| **f16/f128 Support** | ✅ | ❌ |
| **Const-FFT** | ✅ | ❌ |
| **Split-Complex** | ✅ | ❌ |
| **Guru Interface** | ✅ | ❌ |

### When to Use OxiFFT

- **SAR/Radar Processing**: 2D FFT, batch processing, real FFT, convolution
- **Audio Processing**: STFT, window functions, streaming FFT
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

## Dependencies

```toml
[dependencies]
num-complex = "0.4"
num-traits = "0.2"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
seahash = "4.1"
rayon = { version = "1.11", optional = true }
mpi = { version = "0.8", optional = true }
libc = { version = "0.2", optional = true }

[features]
default = ["std", "threading"]
std = []
threading = ["dep:rayon"]
simd = []
portable_simd = []
f128-support = []
f16-support = []
mpi = ["dep:mpi"]
sparse = []
pruned = []
sve = ["dep:libc"]
wasm = ["dep:wasm-bindgen", "dep:js-sys"]
streaming = []
const-fft = []
cuda = []
metal = []
gpu = []
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

- **[oxifft.md](oxifft.md)** - Complete architecture and implementation blueprint (32 KB)
- **[TODO.md](TODO.md)** - Detailed implementation status and roadmap (18 KB)
- **[CHANGELOG.md](CHANGELOG.md)** - Project history and release notes

### Benchmark Reports

- **[BENCHMARK_INVESTIGATION_SUMMARY.md](BENCHMARK_INVESTIGATION_SUMMARY.md)** - Investigation of benchmark methodology bugs (2026-01-07)
- **[benchmark_results_2026-01-07_CORRECTED.md](benchmark_results_2026-01-07_CORRECTED.md)** - Corrected benchmark results vs RustFFT
- **[BENCHMARK_RESULTS_TEMPLATE.md](BENCHMARK_RESULTS_TEMPLATE.md)** - Template for documenting benchmark results

## References

- FFTW Paper: "The Design and Implementation of FFTW3" (Frigo & Johnson, 2005)
- Cooley-Tukey: "An Algorithm for the Machine Calculation of Complex Fourier Series" (1965)
- Rader's Algorithm: "Discrete Fourier transforms when the number of data samples is prime" (1968)
- Bluestein: "A linear filtering approach to the computation of discrete Fourier transform" (1970)

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

Copyright (c) 2026 COOLJAPAN OU (Team KitaSan)
