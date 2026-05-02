# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_No unreleased changes._

## [0.3.1] - 2026-05-01

### Changed

- Version maintenance bump; no functional changes from 0.3.0.
- Applied cargo fmt formatting fixes to `build.rs` and `src/bin/oxifft_tune.rs` (pre-existing format drift).

## [0.3.0] - 2026-04-25

### Performance

- **Performance:** `R2rPlan` now caches `R2rSolver` at construction; solver twiddle tables and FFT plans are built once and reused on every `execute()` call, eliminating 2 Plan constructions + 2561 sin_cos calls per dct2_1024 invocation (v1.0 parity gate: dct2_1024 < 3.0×)

### Added

- DCT-II/III/IV FFT-based implementations via Makhoul reduction (N-point R2C + O(N)
  post-twiddle), replacing 2N/4N complex DFT approach; ~4× flop reduction vs v0.2.0.
  (`oxifft/src/rdft/solvers/r2r.rs`)
- R2r/R2c solver plan caching (`Option<Plan<T>>` + pre-computed twiddle tables)
  eliminating per-call `Plan::dft_1d` construction.
- Bluestein + Rader AoS SIMD pointwise-multiply helpers (`kernel/complex_mul.rs`:
  `complex_mul_aos_f64`, `complex_mul_aos_f32`) with AVX2+FMA, NEON, SSE2, scalar dispatch.
- Thread-local scratch for Bluestein/Rader keyed by solver ID, removing mutex contention.
- FFTW parity gate benchmark harness: 7 gates (1024 complex, 2^20 complex, 1024 real,
  1024×1024 2D, 1000×256 batch, 2017 prime, 1024 DCT-II) at
  `oxifft-bench/benches/fftw_parity_gates.rs`.
- FFTW parity ratio baseline JSON committed at `benches/baselines/v0.3.0/`.
- GPU batch FFT with automatic chunking (`gpu/batch.rs`,
  `METAL_BATCH_LIMIT=1024`, `CUDA_BATCH_LIMIT=4096`).
- Pencil decomposition for 3D MPI FFT (`mpi/plans/plan_3d_pencil.rs`).
- Real WASM SIMD v128 intrinsics via `core::arch::wasm32` with module-split fallback for
  non-simd128 targets (`wasm/simd.rs`).
- Work-stealing `WorkStealingContext` for Plan2D/Plan3D with user-pool override
  (`threading/work_stealing.rs`).
- `Send + Sync` compile-time assertions on all public plan types (`assertions.rs`).
- Hand-optimized AVX-512 codelets for sizes 16/32/64
  (`dft/codelets/hand_avx512.rs`, `dft/codelets/hand_avx512_twiddles.rs`).
- Cache-oblivious Frigo-Johnson 4-step FFT (`dft/solvers/cache_oblivious.rs`).
- Criterion DCT/DST benchmark group (`oxifft/benches/dct_benchmarks.rs`,
  `oxifft-bench/benches/dct_dst.rs`).
- Criterion R2C/C2R regression tracker (`oxifft-bench/benches/r2c_c2r.rs`).
- GPU vs CPU benchmark at 4096/16384/65536/262144 (`oxifft/benches/gpu_vs_cpu.rs`).
- Multi-dimensional NUFFT (2D/3D) (`nufft/nufft2d.rs`, `nufft3d.rs`).
- SoA twiddle layout for CT sizes ≥ 4096, reducing SIMD shuffle count
  (`kernel/twiddle.rs`).
- `GpuBatchFft<T>` trait for N independent same-size FFTs in a single GPU submission.
- Overlap-save STFT method as alternative to overlap-add (`streaming/stft.rs`).

### Changed

- DCT-II default path now FFT-based for n ≥ 16 (O(n log n)); O(n²) retained as reference
  fallback for n < 16.
- Metal backend uses real device probe via `oxicuda_metal::device::MetalDevice::new()`
  (was hardcoded placeholder).
- CUDA backend uses real driver probe via `oxicuda_driver::init()` (was filesystem check).
  GPU kernel dispatch uses CPU fallback pending `oxicuda-launch` integration.
- NEON dispatch wired into small-size path (sizes 2/4/8); no more scalar fallback on
  aarch64.
- Production `.unwrap()` removed from `rader_omega.rs`, `spectral.rs`, `threading/mod.rs`
  (test-only sites retained).
- SVE detection now uses `std::arch::is_aarch64_feature_detected!("sve")` instead of
  `libc::getauxval`; `libc` dependency removed.
- `#![warn(clippy::missing_safety_doc)]` and `#![warn(clippy::missing_errors_doc)]` added
  to `lib.rs` as compile-enforced invariants.

### Removed

- `GpuBackend::OpenCL` and `GpuBackend::Vulkan` placeholder variants (never had backing
  code; downstream match exhaustiveness breakage is acceptable pre-1.0).

### Fixed

- Bluestein `execute_inplace` no longer allocates via `to_vec()` — uses dedicated
  thread-local scratch.
- Rader `execute_inplace` mirrored fix.
- NEON `dit_64`/`dit_128`/`dit_512` eliminate stack round-trip in butterfly loops.

### Performance

- DCT-II @ 1024: ~4× faster vs v0.2.0 (O(n log n) vs O(n²)).
  FFTW ratio: v0.2.0 baseline 7.39× → see `benches/baselines/v0.3.0/` for post-Makhoul
  measurement.
- Power-of-2 1D complex FFT: see `benches/baselines/v0.3.0/` for FFTW ratio snapshots.

### Documentation

- `# Safety` rustdoc added to all 84+ unsafe functions (enforced via
  `#![warn(clippy::missing_safety_doc)]`).
- `# Errors` rustdoc added to 84+ fallible public functions (enforced via
  `#![warn(clippy::missing_errors_doc)]`).
- 1360 tests passing (up from 858 in v0.2.0).

## [0.2.0] - 2026-04-14

### Breaking Changes

- `Plan::dft_2d()` now returns `Option<Plan2D<T>>` instead of `Option<Plan<T>>`.
  Previously panicked at runtime; this is a compile-time breaking change that prevents a runtime crash.
- `Plan::dft_3d()` now returns `Option<Plan3D<T>>` instead of `Option<Plan<T>>`.
  Previously panicked at runtime; this is a compile-time breaking change that prevents a runtime crash.
- `Plan::r2c_1d()` now returns `Option<RealPlan<T>>` instead of `Option<Plan<T>>`.
  Previously panicked at runtime; this is a compile-time breaking change that prevents a runtime crash.
- `Plan::c2r_1d()` now returns `Option<RealPlan<T>>` instead of `Option<Plan<T>>`.
  Previously panicked at runtime; this is a compile-time breaking change that prevents a runtime crash.
- `IndirectStrategy` enum and its `IndexArray` variant removed (was dead code, never constructed).
- All public enums are now `#[non_exhaustive]`. Downstream `match` expressions on public enums
  need a wildcard `_ => ...` arm.

### New Features

- **FFTW Compatibility API** (`fftw-compat` feature): `oxifft::compat` module with FFTW-style
  function names (`fftw_plan_dft_1d`, `fftw_plan_dft_2d`, `fftw_execute`, etc.).
- `Debug` impl on all public plan types.
- `#[must_use]` on all plan creation methods returning `Option<Plan...>`.

### Improvements

- Reduced crate-level `#[allow(clippy::...)]` from 60 to under 30 by fixing underlying lint sites.
- Hardened FFAST sparse FFT peeling decoder for edge cases (k=0, k=n, pure noise).
- Added property-based tests for sparse FFT and all DCT/DST variants.

### Fixes

- Eliminated 6 runtime panics reachable from public API (4 `todo!()` + 2 `unimplemented!()`).
- Removed dead `#[allow(dead_code)]` attributes in production code.

## [0.1.4] - 2026-04-11

### Added

- **Signal processing module** (`signal` feature, requires `std`):
  - Hilbert transform (`hilbert()`) for computing the analytic signal via FFT
  - Envelope detection (`envelope()`) via analytic signal magnitude
  - Instantaneous phase (`instantaneous_phase()`) and frequency (`instantaneous_frequency()`) extraction
  - Power spectral density via Welch's method (`welch()`, `periodogram()`)
  - Cross-spectral density (`cross_spectral_density()`) for two-signal analysis
  - Magnitude-squared coherence (`coherence()`)
  - Real cepstrum (`real_cepstrum()`) — `IFFT(log(|FFT(x)|))`
  - Complex cepstrum (`complex_cepstrum()`) with phase unwrapping
  - Minimum-phase reconstruction (`minimum_phase()`)
  - `SpectralWindow` enum (Rectangular, Hann, Hamming, Blackman) and `WelchConfig` struct
  - FFT-based signal resampling (`resample()`, `resample_to()`) via spectral zero-padding/truncation
- **Mel-frequency analysis** (`streaming` feature):
  - `MelConfig` struct for mel filterbank configuration (sample rate, FFT size, hop size, n_mels, f_min, f_max)
  - `build_mel_filterbank()` — builds triangular mel filterbank matrix
  - `mel_spectrogram()` — log-mel spectrogram from a signal
  - `mfcc()` — Mel-Frequency Cepstral Coefficients via DCT of log-mel spectrogram
- **Example**: `signal_processing.rs` demonstrating all signal module functions

### Changed

- **SIMD codelet refactor**: Split `dft/codelets/simd.rs` (2813 lines, exceeding the 2000-line policy) into a directory module `dft/codelets/simd/` with 5 focused files:
  - `mod.rs` (261 lines): dispatch functions and re-exports
  - `backends.rs` (517 lines): SSE2, AVX2, NEON, and x86_64 SIMD backend implementations
  - `small_sizes.rs` (95 lines): f64-specific SIMD dispatch for sizes 2–32
  - `large_sizes.rs` (1600 lines): f64-specific SIMD dispatch for sizes 64–4096 with precomputed twiddles
  - `tests.rs` (360 lines): correctness and roundtrip tests for SIMD codelets

- **Version bump**: 0.1.3 → 0.1.4

## [0.1.3] - 2026-02-12

### Fixed

- **CUDA SIMD fallback infinite recursion**: Fixed infinite recursion bug in `notw_512_dispatch`, `notw_1024_dispatch`, and `notw_4096_dispatch` SIMD fallback paths
  - The fallback for non-f32/f64 types previously called `CooleyTukeySolver::execute`, which dispatched back to the same codelet, causing infinite recursion
  - Now calls `CooleyTukeySolver::execute_dit_inplace` directly to perform iterative DIT without re-entering the codelet dispatch
  - Made `execute_dit_inplace` public on `CooleyTukeySolver` to support this fix
  - Removed unnecessary `output` buffer allocation in fallback paths

### Changed

- **License consolidation**: Consolidated dual license files (`LICENSE-APACHE` + `LICENSE-MIT`) into a single `LICENSE` file (Apache-2.0)

## [0.1.2] - 2026-01-26

### Fixed

- **Windows compatibility**: Removed `examples/**/CLAUDE.md` directory which caused package unpacking errors on Windows
  - Windows does not allow `**` as directory or filename
  - Error: "The filename, directory name, or volume label syntax is incorrect. (os error 123)"
  - This fix enables cross-platform PyPI publishing for dependent crates (e.g., scirs2-python)

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
- Apache-2.0 licensing
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

[Unreleased]: https://github.com/cool-japan/oxifft/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/cool-japan/oxifft/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/cool-japan/oxifft/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/cool-japan/oxifft/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/cool-japan/oxifft/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/cool-japan/oxifft/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/cool-japan/oxifft/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/cool-japan/oxifft/releases/tag/v0.1.0
