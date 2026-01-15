# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_No unreleased changes._

## [0.1.1] - 2026-01-15

### Changed

- **Dependency updates**:
  - `hashbrown`: 0.15.5 → 0.16.1
  - `spin`: 0.9.8 → 0.10.0
- Removed `rust-version` field (MSRV) to allow using latest Rust features

### Fixed

- **48 clippy warnings eliminated**:
  - `manual_is_multiple_of`: Replaced `n % x == 0` with `n.is_multiple_of(x)`
  - `ref_as_ptr`: Replaced `x as *const _` with `std::ptr::from_ref(x)`
- All tests passing (652 tests)
- Zero clippy warnings with `-D warnings`

## [0.1.0] - 2026-01-12

### Highlights

- **Pure Rust FFT library** - No C/Fortran dependencies for default features
- **20+ features beyond RustFFT** - Sparse FFT, STFT, NUFFT, Auto-diff, GPU, MPI, WASM
- **FFTW-compatible API** - Easy migration from FFTW3
- **SAR/Radar optimized** - Benchmarks and optimizations for signal processing workloads

### Added

#### Core FFT Functionality (Phases 1-7)
- Complete implementation of complex DFT with multiple algorithms:
  - Cooley-Tukey FFT (DIT/DIF, radix-2/4/8, split-radix)
  - Rader's algorithm for prime-size transforms
  - Bluestein's Chirp-Z algorithm for arbitrary sizes
  - Direct O(n²) solver for small sizes
  - Generic mixed-radix solver for composite sizes
- Real FFT support:
  - R2C (Real-to-Complex) transforms
  - C2R (Complex-to-Real) transforms
  - R2R (Real-to-Real) transforms
- DCT/DST transforms:
  - All 8 DCT/DST variants (Types I-IV for both)
  - Discrete Hartley Transform (DHT)
- Multi-dimensional transforms:
  - 1D, 2D, 3D, and N-dimensional DFTs
  - Optimized row-column decomposition
  - Efficient transpose operations
- Batch processing:
  - Vector-rank handling for multiple simultaneous transforms
  - Efficient stride management
  - Cache-optimized buffered execution
- SIMD optimization:
  - SSE2, AVX, AVX2, AVX-512 (x86_64)
  - ARM NEON (aarch64)
  - ARM SVE (Scalable Vector Extension)
  - WebAssembly SIMD (simd128)
  - Runtime CPU feature detection
  - Portable SIMD fallback
- Threading support:
  - Rayon integration for parallel execution
  - Parallel dimension splitting
  - Parallel batch processing
  - Configurable thread pool
- Wisdom system:
  - Plan caching for optimal performance
  - Serialization and deserialization
  - File import/export
  - System wisdom location discovery
- Planning modes:
  - ESTIMATE (heuristic-based)
  - MEASURE (benchmark-based)
  - PATIENT (thorough search)
  - EXHAUSTIVE (comprehensive)
  - Time-limited planning
- Memory management:
  - Aligned memory allocation
  - Optimized copy operations
  - Matrix transpose utilities
- API completeness:
  - Simple convenience functions (fft, ifft, rfft, irfft)
  - 2D/3D convenience functions
  - Guru interface for maximum flexibility
  - Split-complex support (separate real/imaginary arrays)
  - In-place transform support

#### Advanced Features - Beyond FFTW (Phases 8-9)

- **Sparse FFT** (`sparse` feature):
  - FFAST (Fast Fourier Aliasing-based Sparse Transform) algorithm
  - O(k log n) complexity for k-sparse signals
  - Frequency bucketization and peeling decoder
  - One-shot API (`sparse_fft`, `sparse_ifft`)
  - Plan-based API (`SparsePlan`) for repeated use

- **Pruned FFT** (`pruned` feature):
  - Input-pruned FFT (sparse input, full output)
  - Output-pruned FFT (full input, sparse output)
  - Both-pruned FFT (sparse input and output)
  - Goertzel algorithm for single-frequency computation
  - `PrunedPlan` with configurable pruning modes

- **Streaming FFT** (`streaming` feature):
  - Short-Time Fourier Transform (STFT)
  - Inverse STFT with overlap-add reconstruction
  - Window functions: Hann, Hamming, Blackman, Kaiser, Rectangular
  - Real-time streaming processor (`StreamingFft`)
  - Ring buffer for efficient frame management
  - Magnitude, power, and phase spectrograms

- **Compile-time FFT** (`const-fft` feature):
  - Const generics for fixed-size arrays
  - Zero runtime overhead for known sizes
  - Taylor series sin/cos for twiddle factors
  - Implementations for sizes 2-1024
  - In-place and out-of-place variants
  - `ConstFft` trait for type-safe compile-time transforms

- **Non-Uniform FFT** (`nufft`):
  - Type 1: Non-uniform time → Uniform frequency
  - Type 2: Uniform frequency → Non-uniform time
  - Type 3: Non-uniform → Non-uniform
  - Gaussian gridding with spreading coefficients
  - Deconvolution for kernel correction
  - Configurable tolerance and oversampling
  - Plan-based API for repeated transforms

- **Fractional Fourier Transform** (`frft`):
  - Chirp decomposition for fractional orders
  - Integer order optimization (0, 1, 2, 3)
  - One-shot API (`frft`, `ifrft`)
  - Checked variants with error handling
  - Plan-based API (`Frft`) for efficiency

- **FFT-based Convolution** (`conv`):
  - Linear convolution (O(n log n) vs O(n²))
  - Circular convolution for periodic signals
  - Cross-correlation for pattern matching
  - Polynomial multiplication and power
  - Convolution modes: Full, Same, Valid
  - Complex signal support

- **Automatic Differentiation** (`autodiff`):
  - Forward-mode AD with dual numbers
  - Backward-mode AD for gradient computation
  - Vector-Jacobian product (VJP) for backpropagation
  - Jacobian-vector product (JVP) for forward sensitivity
  - Full Jacobian matrix computation
  - Real FFT gradients
  - 2D FFT gradients
  - `DiffFftPlan` for repeated differentiation

- **GPU Acceleration** (`gpu`, `cuda`, `metal` features):
  - CUDA backend for NVIDIA GPUs (via cuFFT)
  - Metal backend for Apple GPUs (via MPS)
  - Auto-detection of best available backend
  - GPU buffer management
  - Device capability querying
  - Forward and inverse transforms
  - Batch processing support

- **MPI Distributed Computing** (`mpi` feature):
  - 2D, 3D, and N-D distributed FFTs
  - Slab decomposition (row-major distribution)
  - Efficient all-to-all transpose operations
  - Compatible with FFTW-MPI data layouts
  - Transposed input/output modes
  - Local size computation utilities

- **WebAssembly Support** (`wasm` feature):
  - Browser-compatible FFT
  - JavaScript interop (`WasmFft` wrapper)
  - One-shot functions (fft_f64, ifft_f64, fft_f32, ifft_f32)
  - Real-to-complex transforms (rfft_f64)
  - WASM SIMD backend (simd128)
  - Portable fallback for non-SIMD environments

- **Extended Precision** (`f16-support`, `f128-support` features):
  - F16 (half-precision, 16-bit) floating-point
  - F128 (quad-precision, 128-bit) floating-point
  - IEEE 754 binary16/binary128 conversion
  - Full `num_traits` trait implementations
  - All FFT operations support both precisions

#### Testing and Validation

- Comprehensive test suite:
  - 629 unit tests passing
  - 3 integration tests (skipped for optional features)
  - Correctness validation against Direct O(n²) solver
  - Cross-validation with rustfft
  - FFTW comparison tests (28 tests, feature-gated)
  - Property-based tests (Parseval, linearity, inverse)
  - Size coverage tests (powers of 2, primes, composites, edge cases)
  - Multi-dimensional roundtrip tests
  - Batch transform correctness tests
  - Threading correctness tests
  - Wisdom persistence tests
  - Planning mode tests
  - SIMD backend tests

- Benchmarking suite:
  - Criterion-based benchmarks
  - 1D complex DFT (power-of-2, prime, composite sizes)
  - 1D real FFT
  - 2D complex DFT
  - Batch transforms
  - Comparison with rustfft
  - Optional FFTW comparison (feature-gated)
  - Beyond-FFTW features benchmark (sparse, pruned, streaming, const-fft)
  - **SAR Processing Benchmarks**: range compression, azimuth batch, 2D image formation, chirp convolution, roundtrip, real FFT

#### Documentation and Examples

- Complete API documentation with rustdoc
- 10 comprehensive examples:
  - `simple_fft.rs` - Basic 1D FFT usage
  - `real_fft.rs` - Real-to-complex transforms
  - `batch_fft.rs` - Batch processing
  - `multidimensional.rs` - 2D/3D/N-D transforms
  - `wisdom_usage.rs` - Wisdom system usage
  - `sparse_fft.rs` - Sparse FFT for k-sparse signals
  - `streaming_fft.rs` - STFT for real-time processing
  - `nufft_example.rs` - Non-uniform FFT for irregular sampling
  - `autodiff_fft.rs` - Automatic differentiation
  - `convolution.rs` - FFT-based convolution and correlation
- Architecture documentation (`oxifft.md`)
- Comprehensive README with feature overview
- Implementation TODO tracking (`TODO.md`)

#### Project Infrastructure

- Workspace structure with 3 crates:
  - `oxifft` - Main library
  - `oxifft-codegen` - Proc-macro crate for codelet generation
  - `oxifft-bench` - Benchmarking suite with FFTW comparison
- GitHub Actions CI/CD:
  - Multi-platform testing (Linux, macOS, Windows)
  - Clippy and rustfmt checks
  - Documentation build verification
  - Benchmark workflow
- Dual licensing (MIT OR Apache-2.0)
- Pure Rust implementation (100% Rust for default features)
- `no_std` support (with `std` feature flag)

### Changed

- **Architecture Refactoring** for 2000-line policy compliance:
  - Split `stockham.rs` (2406→4 modules) into `stockham/` directory
  - Split `composite.rs` (2531→5 modules) into `composite/` directory
  - Modular SIMD implementations by architecture (x86_64, aarch64)

### Performance

- **Composite FFT optimization**: 8×12 factorization for notw_96
- Many composite sizes now faster than RustFFT: 20, 30, 36, 45, 48, 50, 60, 80, 100
- Precomputed twiddle tables for Stockham radix-4 algorithm

### Fixed

- **186 clippy warnings eliminated** across codebase
- Example compilation errors with proper `required-features`
- Benchmark methodology bugs in FFTW comparison

### Documentation

- `BENCHMARKING.md` - Comprehensive benchmarking guide with SAR examples
- `PERFORMANCE_ANALYSIS.md` - Performance analysis methodology
- `TESTING.md` - Testing strategy and validation procedures
- RustFFT comparison table in README

### Security

- N/A (initial release)

## Project Statistics

- **Total Lines of Code**: 37,594 (Rust code only)
- **Rust Files**: 168
- **Test Coverage**: 357 unit tests + doc tests passing
- **Zero Warnings**: Clippy + rustdoc clean (all features)
- **Documentation**: 2,853 comment lines + 5,552 doc comment lines

## Supported Platforms

- **x86_64**: Linux, macOS, Windows (SSE2, AVX, AVX2, AVX-512)
- **aarch64**: Linux, macOS (NEON, SVE with feature flag)
- **wasm32**: Browser and Node.js (WASM SIMD)

## Dependencies

### Required
- num-complex 0.4
- num-traits 0.2
- serde 1.0
- serde_json 1.0
- seahash 4.1

### Optional
- rayon 1.11 (threading)
- mpi 0.8 (MPI distributed computing)
- libc 0.2 (SVE detection)
- wasm-bindgen 0.2 (WebAssembly bindings)
- js-sys 0.3 (JavaScript interop)

[Unreleased]: https://github.com/cool-japan/oxifft/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cool-japan/oxifft/releases/tag/v0.1.0
