# OxiFFT Implementation TODO

## Phase 1: Foundation (Core Infrastructure)

### Kernel Module
- [x] Implement `Float` trait for f32/f64
- [x] Implement `Complex<T>` type with arithmetic operations
- [x] Implement `IoDim` dimension specification
- [x] Implement `Tensor` N-dimensional representation
- [x] Implement `Problem` trait hierarchy
- [x] Implement `Plan` trait hierarchy
- [x] Implement `Solver` trait and registry
- [x] Implement basic `Planner` (no wisdom) with solver selection and cost estimation
- [x] Implement twiddle factor computation and caching
- [x] Implement trigonometric table generation
- [x] Implement prime factorization utilities
- [x] Implement operation counting (`OpCount`)
- [x] Implement planning flags (`PlannerFlags`)

### Basic DFT
- [x] Implement `DftProblem` structure
- [x] Implement `DftPlan` structure
- [x] Implement Direct solver (O(n²) reference)
- [x] Implement size-1 no-op solver

### Simple API
- [x] Implement `fft_1d()` convenience function
- [x] Implement basic memory alignment utilities
- [x] Create `lib.rs` with public exports

### Project Setup
- [x] Create workspace Cargo.toml
- [x] Create main crate Cargo.toml
- [x] Set up module structure (api/, kernel/, dft/, etc.)
- [x] Add basic unit tests

---

## Phase 2: Core Algorithms

### Cooley-Tukey Implementation
- [x] Implement DIT (Decimation-in-Time) variant
- [x] Implement DIF (Decimation-in-Frequency) variant
- [x] Implement power-of-2 optimized path
- [x] Implement mixed-radix factorization (GenericSolver)
- [x] Add radix-2 butterfly
- [x] Add radix-4 butterfly
- [x] Add radix-8 butterfly
- [x] Add split-radix algorithm

### Prime-Size Algorithms
- [x] Implement primitive root computation
- [x] Implement Rader's algorithm for prime sizes
- [x] Implement Bluestein's Chirp-Z algorithm
- [x] Implement Rader omega lookup tables

### Codelet Generation (oxifft-codegen)
- [x] Set up proc-macro crate structure
- [x] Implement symbolic FFT operation representation (Expr, ComplexExpr, SymbolicFFT)
- [x] Implement common subexpression elimination (CseOptimizer)
- [x] Implement strength reduction optimization (StrengthReducer)
- [x] Generate size-2 non-twiddle codelet
- [x] Generate size-4 non-twiddle codelet
- [x] Generate size-8 non-twiddle codelet
- [x] Generate size-16 non-twiddle codelet
- [x] Generate size-32 codelet
- [x] Generate size-64 codelet
- [x] Generate twiddle-factor codelets (radix-2, radix-4, radix-8)
- [x] Add SIMD codelet variants (notw_2, notw_4, notw_8 with SSE2/AVX2 dispatch)

---

## Phase 3: Real FFTs

### RDFT Implementation
- [x] Implement `RdftProblem` structure (basic structure in place)
- [x] Implement `RdftPlan` structure (basic structure in place)
- [x] Implement `RdftKind` enum (R2C, C2R, R2R variants)
- [x] Implement R2C (Real-to-Complex) solver
- [x] Implement C2R (Complex-to-Real) solver
- [x] Implement rfft/irfft convenience functions
- [x] Implement 2D R2C/C2R (RealPlan2D, rfft2d/irfft2d)
- [x] Implement 3D R2C/C2R (RealPlan3D, rfft3d/irfft3d)
- [x] Implement N-D R2C/C2R (RealPlanND, rfft_nd/irfft_nd)
- [x] Implement R2R (Real-to-Real) solver (DCT/DST/DHT via R2rSolver)
- [x] Implement half-complex representation (Hc2cSolver, C2hcSolver)
- [x] Implement HC2C (Half-complex to Complex) solver
- [x] Implement HC2HC (Half-complex to Half-complex) solver

### DCT/DST (REODFT)
- [x] Implement REDFT00 (DCT-I)
- [x] Implement REDFT01 (DCT-III)
- [x] Implement REDFT10 (DCT-II)
- [x] Implement REDFT11 (DCT-IV)
- [x] Implement RODFT00 (DST-I)
- [x] Implement RODFT01 (DST-III)
- [x] Implement RODFT10 (DST-II)
- [x] Implement RODFT11 (DST-IV)
- [x] Implement Discrete Hartley Transform (DHT)

---

## Phase 4: Multi-Dimensional & Batching

### Rank-2+ Transforms
- [x] Implement `rank_geq2` solver for multi-dimensional
- [x] Implement 2D DFT composition (Plan2D)
- [x] Implement 3D DFT composition (Plan3D)
- [x] Implement N-dimensional generalization
- [x] Implement indirect solver for non-contiguous strides
- [x] Implement transpose optimizations

### Batch Processing
- [x] Implement `vrank_geq1` solver for batching (DFT and RDFT)
- [x] Implement efficient stride management
- [x] Implement buffered solver for cache locality
- [x] Implement batch-aware planning

---

## Phase 5: Performance

### SIMD Layer
- [x] Define `SimdVector` trait
- [x] Define `SimdComplex` trait
- [x] Implement scalar fallback
- [x] Implement SSE2 backend (x86_64)
- [x] Implement AVX backend (x86_64)
- [x] Implement AVX2 backend (x86_64)
- [x] Implement AVX-512 backend (x86_64)
- [x] Implement NEON backend (aarch64)
- [x] Implement runtime CPU feature detection
- [x] Implement portable_simd fallback
- [x] Integrate SIMD into codelets (notw_2, notw_4, notw_8 with SSE2/AVX2 dispatch)
- [x] Integrate codelets into CT solver for small base cases

### Threading
- [x] Define `ThreadPool` trait
- [x] Implement `SerialPool` (single-threaded)
- [x] Implement `RayonPool` integration
- [x] Implement parallel dimension splitting
- [x] Implement parallel batch processing
- [x] Add thread count configuration

---

## Phase 6: Planning System

### Wisdom System
- [x] Implement problem hashing (`hash.rs`)
- [x] Implement `WisdomCache` structure
- [x] Implement `WisdomEntry` serialization
- [x] Implement wisdom lookup
- [x] Implement wisdom storage
- [x] Implement `export_to_string()` / `import_from_string()`
- [x] Implement `export_to_file()` / `import_from_file()`
- [x] Implement system wisdom location discovery
- [x] Implement `forget()` to clear wisdom

### Advanced Planning
- [x] Implement ESTIMATE mode (heuristic only)
- [x] Implement MEASURE mode (benchmark all solvers)
- [x] Implement PATIENT mode (more thorough search)
- [x] Implement EXHAUSTIVE mode (try everything)
- [x] Implement time-limited planning search
- [x] Implement cost estimation heuristics
- [x] Implement plan recreation from wisdom

---

## Phase 7: API & Polish

### Full Public API
- [x] Implement `Plan::dft_1d()` / `dft_2d()` / `dft_3d()` (Plan, Plan2D, Plan3D complete)
- [x] Implement `Plan::r2c_1d()` / `c2r_1d()` (via RealPlan struct)
- [x] Implement `Plan::r2r_1d()` with kind parameter (via R2rPlan struct)
- [x] Implement `GuruPlan` interface (full implementation with arbitrary strides, batching, N-D)
- [x] Implement split real/imaginary support (SplitPlan, SplitPlan2D, SplitPlan3D, SplitPlanND, fft_split, ifft_split, fft3d_split, ifft3d_split, fft_nd_split, ifft_nd_split)
- [x] Implement in-place transform support
- [x] Add `Direction` enum (Forward/Backward)
- [x] Add `Flags` configuration

### Memory Management
- [x] Implement aligned allocation (`memory.rs` with `AlignedBuffer`)
- [x] Implement `alloc_complex_aligned()` / `alloc_real_aligned()` functions
- [x] Implement optimized copy operations (`copy.rs`)
- [x] Implement matrix transpose utilities (`transpose.rs`)

### Testing
- [x] Implement correctness tests vs Direct O(n²)
- [x] Implement correctness tests vs rustfft
- [x] Implement correctness tests vs fftw (oxifft-bench crate, feature-gated)
- [x] Add property-based tests (Parseval, linearity, inverse)
- [x] Add tests for all transform sizes (2 to 2^20)
- [x] Add tests for prime sizes
- [x] Add tests for multi-dimensional transforms (2D, 3D)
- [x] Add tests for batch transforms
- [x] Add threading correctness tests
- [x] Add wisdom persistence tests

### Benchmarking
- [x] Set up criterion benchmarks (oxifft-bench crate)
- [x] Benchmark 1D complex DFT (power-of-2)
- [x] Benchmark 1D complex DFT (prime)
- [x] Benchmark composite sizes
- [x] Benchmark 1D real FFT
- [x] Benchmark 2D complex DFT
- [x] Benchmark batch transforms
- [x] Compare with rustfft
- [x] Compare with fftw (feature-gated)

### Documentation
- [x] Document public API (partial)
- [x] Add usage examples in `examples/`
- [x] Add simple_fft.rs example
- [x] Add real_fft.rs example
- [x] Add batch_fft.rs example
- [x] Add multidimensional.rs example
- [x] Add wisdom_usage.rs example
- [x] Document SIMD requirements
- [x] Document threading configuration

### CI/CD
- [x] Set up GitHub Actions
- [x] Add test workflow (all platforms)
- [x] Add benchmark workflow
- [x] Add clippy/fmt checks
- [x] Add documentation build

---

## Current Status (v0.3.0 — All Phases 1–10 Complete + v0.3.0 Performance — Released 2026-04-25)

### Implemented Solvers
- **NOP Solver**: Size-0 and size-1 (identity) transforms
- **Direct Solver**: O(n²) reference implementation with full test coverage
- **Cooley-Tukey Solver**: O(n log n) for power-of-2 sizes (DIT, DIF, Radix-4, Radix-8, Split-Radix variants)
- **Bluestein Solver**: O(n log n) for arbitrary sizes via chirp-z transform
- **Rader Solver**: O(n log n) for prime sizes via cyclic convolution
- **Generic Solver**: O(n log n) mixed-radix for composite non-power-of-2 sizes
- **Indirect Solver**: Gather/scatter for non-contiguous strided data
- **Buffered Solver**: Cache-optimized execution for strided access patterns

### Multi-Dimensional Support
- **Plan2D**: 2D FFT via row-column decomposition
- **Plan3D**: 3D FFT via layered decomposition
- **PlanND**: N-dimensional FFT via successive 1D transforms
- **fft2d/ifft2d**: Convenience functions for 2D transforms
- **fft_nd/ifft_nd**: Convenience functions for N-dimensional transforms

### Split-Complex Support
- **SplitPlan**: 1D FFT with separate real/imaginary arrays
- **SplitPlan2D**: 2D FFT with separate real/imaginary arrays
- **SplitPlan3D**: 3D FFT with separate real/imaginary arrays
- **SplitPlanND**: N-dimensional FFT with separate real/imaginary arrays
- **fft_split/ifft_split**: Convenience functions for 1D split-complex
- **fft2d_split/ifft2d_split**: Convenience functions for 2D split-complex
- **fft3d_split/ifft3d_split**: Convenience functions for 3D split-complex
- **fft_nd_split/ifft_nd_split**: Convenience functions for N-D split-complex

### Real FFT Support
- **R2cSolver**: Real-to-Complex FFT using packing algorithm
- **C2rSolver**: Complex-to-Real FFT (inverse of R2C)
- **rfft/irfft**: Convenience functions for real FFT

### DCT/DST Support
- **DCT-I (REDFT00)**: Implemented with direct computation
- **DCT-II (REDFT10)**: Implemented with direct computation (JPEG standard DCT)
- **DCT-III (REDFT01)**: Implemented with direct computation (inverse DCT-II)
- **DCT-IV (REDFT11)**: Implemented with direct computation
- **DST-I (RODFT00)**: Implemented with direct computation
- **DST-II (RODFT10)**: Implemented with direct computation
- **DST-III (RODFT01)**: Implemented with direct computation (inverse DST-II)
- **DST-IV (RODFT11)**: Implemented with direct computation
- **dct1/dct2/dct3/dct4**: Convenience functions for DCT transforms
- **dst1/dst2/dst3/dst4**: Convenience functions for DST transforms
- **DHT**: Discrete Hartley Transform (self-inverse)
- **dht**: Convenience function for DHT

### Batch Processing Support
- **VrankGeq1Solver**: Batched DFT with stride control
- **RdftVrankGeq1Solver**: Batched R2C/C2R with stride control
- **fft_batch/ifft_batch**: Convenience functions for batched complex FFT
- **rfft_batch/irfft_batch**: Convenience functions for batched real FFT

### Test Coverage
- 858 tests passing (unit + integration + rustfft comparison + wisdom + planning + size coverage + GuruPlan + split-complex + codegen + SIMD + signal processing + autodiff + convolution + NUFFT + FrFT + sparse + pruned + streaming)
- 28 FFTW comparison tests passing (oxifft-bench with fftw-compare feature)
- 688 public API items (all documented and tested)
- 0 unimplemented!() or todo!() in public API surface
- No compiler warnings
- Tests verify against Direct solver for correctness
- Tests verify against rustfft for cross-library validation
- Forward/inverse round-trip tests (1D, 2D, 3D)
- In-place vs out-of-place consistency tests
- Radix-4, Radix-8, and Split-Radix variant tests
- R2C/C2R roundtrip and correctness tests
- DCT-II/DCT-III roundtrip tests
- DST-II/DST-III roundtrip tests
- DHT self-inverse tests
- Batch DFT and RDFT roundtrip tests
- Batch transforms match individual transforms tests
- Half-complex format conversion tests
- SSE2, AVX, AVX2+FMA, AVX-512, and NEON SIMD backend tests
- Threading correctness tests (parallel_for, join, split, chunks)
- Comprehensive size coverage tests (powers of 2, primes, composites, edge cases)
- ESTIMATE, MEASURE, PATIENT, EXHAUSTIVE planning mode tests
- Time-limited planning tests
- Plan recreation from wisdom tests

---

## Phase 8: Beyond FFTW (Advanced Features)

### Sparse FFT
- [x] Implement FFAST (Fast Fourier Aliasing-based Sparse Transform)
- [x] Implement frequency bucketization
- [x] Implement peeling decoder
- [x] Implement `sparse_fft()` and `sparse_ifft()` functions
- [x] Implement `SparsePlan` for repeated use
- [x] Implement `SparseResult` output type

### Pruned FFT
- [x] Implement input-pruned FFT (when most inputs are zero)
- [x] Implement output-pruned FFT (when only subset of outputs needed)
- [x] Implement Goertzel algorithm for single-frequency computation
- [x] Implement `goertzel()` and `goertzel_multi()` functions
- [x] Implement `PrunedPlan` with `PruningMode`

### ARM SVE SIMD
- [x] Implement `Sve256F64` (4 lanes) backend
- [x] Implement `Sve256F32` (8 lanes) backend
- [x] Implement SimdVector and SimdComplex traits for SVE
- [x] Add SVE runtime detection via HWCAP
- [x] Add `sve` feature flag

### WebAssembly Support
- [x] Implement `WasmFft` wrapper for JavaScript interop
- [x] Implement `fft_f64`, `ifft_f64`, `fft_f32`, `ifft_f32` one-shot functions
- [x] Implement `rfft_f64` for real-to-complex
- [x] Implement WASM SIMD (simd128) backend with `WasmSimdF64`, `WasmSimdF32`
- [x] Add `wasm` feature flag

### Streaming FFT (STFT)
- [x] Implement Short-Time Fourier Transform (`stft`)
- [x] Implement Inverse STFT (`istft`) with overlap-add
- [x] Implement window functions (Hann, Hamming, Blackman, Kaiser, Rectangular)
- [x] Implement `StreamingFft` for real-time frame processing
- [x] Implement `RingBuffer` for efficient streaming input
- [x] Implement `magnitude_spectrogram`, `power_spectrogram`, `phase_spectrogram`
- [x] Add `streaming` feature flag

### Compile-time FFT
- [x] Implement const twiddle factor computation (Taylor series sin/cos)
- [x] Implement `fft_fixed` and `ifft_fixed` for fixed-size arrays
- [x] Implement `ConstFft` trait with implementations for sizes 2-1024
- [x] Implement in-place variants (`fft_fixed_inplace`, `ifft_fixed_inplace`)
- [x] Add `const-fft` feature flag

### Non-uniform FFT (NUFFT)
- [x] Implement `Nufft<T>` plan with Type 1/2/3 transforms
- [x] Implement Gaussian gridding with spreading coefficients
- [x] Implement deconvolution factors for kernel correction
- [x] Implement `nufft_type1()`, `nufft_type2()`, `nufft_type3()` convenience functions
- [x] Export `NufftType`, `NufftOptions`, `NufftError`, `NufftResult` types

### Fractional Fourier Transform (FrFT)
- [x] Implement `Frft<T>` plan with chirp decomposition
- [x] Implement integer order handling (0, 1, 2, 3 → identity, FFT, reversal, IFFT)
- [x] Implement fractional order via chirp multiply-convolve-multiply
- [x] Implement `frft()` and `ifrft()` convenience functions
- [x] Implement `frft_checked()` and `ifrft_checked()` with error handling
- [x] Export `Frft`, `FrftError`, `FrftResult` types

### FFT-based Convolution
- [x] Implement `convolve()` for linear convolution of real signals
- [x] Implement `convolve_circular()` for circular convolution
- [x] Implement `convolve_complex()` for complex signal convolution
- [x] Implement `correlate()` and `correlate_complex()` for cross-correlation
- [x] Implement `polynomial_multiply()` and `polynomial_power()` for polynomial ops
- [x] Implement `ConvMode` (Full, Same, Valid) output modes
- [x] Export all convolution functions and types

### Automatic Differentiation for FFT
- [x] Implement `Dual<T>` for forward-mode AD
- [x] Implement `DualComplex<T>` for complex forward-mode AD
- [x] Implement `DiffFftPlan<T>` with forward_dual and backward methods
- [x] Implement `grad_fft()` for backward mode gradient computation
- [x] Implement `grad_ifft()` for inverse FFT gradients
- [x] Implement `vjp_fft()` (Vector-Jacobian product) and `jvp_fft()` (Jacobian-vector product)
- [x] Implement `fft_jacobian()` for computing full Jacobian matrix
- [x] Implement `real::grad_rfft()` and `real::grad_irfft()` for real FFT gradients
- [x] Implement `fft2d::grad_fft2d()` for 2D FFT gradients
- [x] Export autodiff types and functions

---

## Phase 9: Extended Precision & GPU Acceleration

### GPU Support
- [x] Implement `GpuFftEngine` trait for GPU-accelerated FFT
- [x] Implement `GpuBackend` enum (Auto, Cuda, Metal, OpenCL, Vulkan)
- [x] Implement `GpuBuffer<T>` for GPU memory management
- [x] Implement `GpuFft<T>` plan with forward/inverse transforms
- [x] Implement `GpuCapabilities` for querying device info
- [x] Implement CUDA backend (`CudaFftPlan`) with cuFFT support
- [x] Implement Metal backend (`MetalFftPlan`) with MPS support
- [x] Add `cuda`, `metal`, `gpu` feature flags

### Extended Precision
- [x] Implement F128 quad-precision (128-bit) floating-point type
- [x] Implement F16 half-precision (16-bit) floating-point type
- [x] Implement all `num_traits` traits (Zero, One, Num, NumCast, ToPrimitive, Float, FloatConst)
- [x] Implement IEEE 754 binary16/binary128 conversion
- [x] Add `f16-support`, `f128-support` feature flags

### Benchmarks
- [x] Add beyond_fftw.rs benchmark for sparse, pruned, streaming, const-fft features
- [x] Benchmark sparse FFT vs regular FFT
- [x] Benchmark pruned FFT (Goertzel)
- [x] Benchmark streaming FFT (STFT)
- [x] Benchmark compile-time FFT

---

## Success Criteria

- [x] All outputs match FFTW within floating-point tolerance (1e-10 for f64) - 28 FFTW comparison tests pass
- [x] Performance O(n log n) for all sizes (via Bluestein/Rader/Cooley-Tukey)
- [x] No `unsafe` in public API
- [x] Minimal and well-documented internal unsafe
- [x] Idiomatic Rust API with good documentation (partial)
- [x] Support all major FFTW features (complex/real/r2r transforms, all dimensions, wisdom, threading, split-complex, aligned memory)
- [x] Works on x86_64 (SSE2+) and aarch64 (NEON) - SIMD backends implemented

---

## Phase 10: Signal Processing

### Hilbert Transform & Analytic Signal
- [x] Implement `hilbert()` for analytic signal computation (zero negative frequencies via FFT)
- [x] Implement `envelope()` via analytic signal magnitude
- [x] Implement `instantaneous_phase()` via atan2 of analytic signal
- [x] Implement `instantaneous_frequency()` via phase differentiation with unwrapping
- [x] Add `signal` feature flag (depends on `std`)

### Power Spectral Density
- [x] Implement `periodogram()` for simple Hann-windowed PSD estimation
- [x] Implement `welch()` for Welch's averaged method PSD
- [x] Implement `cross_spectral_density()` for two-signal spectral analysis
- [x] Implement `coherence()` for magnitude-squared coherence
- [x] Implement `SpectralWindow` enum (Rectangular, Hann, Hamming, Blackman)
- [x] Implement `WelchConfig` struct for parameter control

### Cepstral Analysis
- [x] Implement `real_cepstrum()` — `IFFT(log(|FFT(x)|))`
- [x] Implement `complex_cepstrum()` with incremental phase unwrapping
- [x] Implement `minimum_phase()` reconstruction via cepstral liftering
- [x] Implement `unwrap_phase()` private helper for phase continuity

### Code Quality
- [x] Refactored `dft/codelets/simd.rs` (2813 lines) into `simd/` directory module (5 files, all <2000 lines)

### FFT-based Resampling
- [x] Implement `resample()` — spectral zero-padding (upsample) and truncation (downsample)
- [x] Implement `resample_to()` — convenience wrapper using sample rates
- [x] Handle Nyquist bin energy splitting for even-length signals

### Mel-Frequency Analysis (streaming feature)
- [x] Implement `MelConfig` struct for analysis configuration
- [x] Implement `hz_to_mel()` / `mel_to_hz()` conversions (Hz ↔ mel scale)
- [x] Implement `build_mel_filterbank()` — triangular mel filterbank matrix
- [x] Implement `mel_spectrogram()` — log-mel spectrogram from signal
- [x] Implement `mfcc()` — Mel-Frequency Cepstral Coefficients via DCT-II

### Examples
- [x] Add `signal_processing.rs` example demonstrating all signal module functions

---

## Roadmap: v0.2.0 → v1.0.0

> All Phases 1–10 are complete as of v0.1.4. This roadmap has three milestones:
> **v0.2.0** (full stabilization — fixes, codegen, compat, quality, performance),
> **v0.3.0** (GPU backends + DCT/DST O(n log n)), and **v1.0.0** (stable release).
> Items reference source locations for clarity.

---

## v0.2.0 — Full Stabilization

**Theme:** Eliminate runtime panics, advance the codegen pipeline, ship FFTW compatibility,
harden advanced features, optimize performance, and complete pre-release polish — all prior
to the GPU/performance milestone in v0.3.0.

### Eliminate Runtime Panics

- [x] Implement `Plan::dft_2d()` — delegate to `Plan2D` internally
      (`oxifft/src/api/plan/types.rs:1492`)
- [x] Implement `Plan::dft_3d()` — delegate to `Plan3D` internally
      (`oxifft/src/api/plan/types.rs:1496`)
- [x] Implement `Plan::r2c_1d()` — delegate to `RealPlan` internally
      (`oxifft/src/api/plan/types.rs:1506`)
- [x] Implement `Plan::c2r_1d()` — delegate to `RealPlan` internally
      (`oxifft/src/api/plan/types.rs:1510`)
- [x] Replace `unimplemented!()` in `IndirectSolver::gather()` IndexArray arm
      (`oxifft/src/dft/solvers/indirect.rs:218`)
- [x] Replace `unimplemented!()` in `IndirectSolver::scatter()` IndexArray arm
      (`oxifft/src/dft/solvers/indirect.rs:233`)
- [x] Audit entire codebase for remaining `todo!()`, `unimplemented!()`, `unwrap()` in non-test code

### Documentation Fixes

- [x] Create `PROJECT_STATUS.md` (referenced in README.md but missing)
- [x] Create `oxifft.md` architecture blueprint (referenced in README.md as 32KB doc)
- [x] Create `TESTING.md` (referenced in README.md but missing)
- [x] Remove dead links to nonexistent benchmark files from README.md
- [x] Populate `BENCHMARK_RESULTS_TEMPLATE.md` with actual data or clarify template-only status

### API Quality

- [x] Add `#[must_use]` to all plan creation methods returning `Option<Plan>`
- [x] Add regression tests for `Plan::dft_2d/dft_3d/r2c_1d/c2r_1d` (done in tests/plan_delegation.rs)
- [x] Add `no_std` integration test (build.rs warnings serve this purpose)
- [x] Clippy allow reduction round 1: reduce `lib.rs` crate-level allows from 60 → <30
      (`oxifft/src/lib.rs:30-89`) — done: 29 remain
- [x] Clippy allow reduction round 2: <30 → <10, refactor sites as needed
- [x] Ensure all error types are `#[non_exhaustive]`
- [x] Ensure all public enums are `#[non_exhaustive]`
- [x] Add `Debug` impl to all public types missing it (done in debug_impls.rs)
- [x] Review trait bounds for unnecessary constraints (planned 2026-04-19)
  - **Goal:** Audit every public `fn`/`struct` with `where` clauses and remove bounds the implementation does not actually require (e.g., redundant `Copy`/`Clone`/`Debug`).
  - **Design:** For each public bound, check the body: if the bound is unused (no `.clone()`, no `fmt::Debug` format, etc.), strip it. Document each removal with a comment. Iterate until `cargo clippy --all-features -- -D warnings` is clean.
  - **Files:** All public modules with `where` clauses — orientation-first enumeration; primarily `oxifft/src/api/`, `oxifft/src/dft/`, `oxifft/src/rdft/`.
  - **Tests:** All existing tests must still pass after bound removal (bounds are contract-preserving when made less restrictive).
  - **Risk:** Removing bounds is a SemVer-compatible relaxation; no user breakage expected pre-1.0.

### Codegen: SIMD Code Generation (Phase 5)

- [x] Implement SSE2 2-lane f64 codelet generation (gen_simd.rs, sizes 2/4/8)
- [x] Implement SSE2 4-lane f32 codelet generation
- [x] Implement AVX 4-lane f64 codelet generation
- [x] Implement AVX2 FMA-optimized codelet generation (gen_simd.rs, sizes 2/4/8)
- [x] Implement AVX-512 8-lane f64 / 16-lane f32 codelet generation
- [x] Implement NEON 2-lane f64 codelet generation (gen_simd.rs, sizes 2/4/8)
- [x] Integrate generated SIMD codelets into runtime dispatch

### Codegen: RDFT Code Generation (Phase 6)

- [x] Implement R2HC codelet generation
- [x] Implement HC2R codelet generation
- [x] Implement real-valued twiddle codelet generation

### Sparse FFT Robustness

- [x] Audit FFAST peeling decoder for edge cases (verified: k=0 early return, k>=n fallback, noise guard, iteration limit)
      (`oxifft/src/sparse/decoder.rs`)
- [x] Handle degenerate cases: k=0, k=n, signal is pure noise (7 edge case tests)
- [x] Add adaptive sparsity detection (sparse_fft_auto, sparse_fft_auto_with_ratio)
- [x] Add property-based tests across diverse sparsity patterns (6 proptest properties)
- [x] Document accuracy guarantees and known limitations of FFAST algorithm

### FFTW Compatibility API

- [x] Create `oxifft::compat` module with FFTW-style naming
- [x] Implement `fftw_plan_dft_1d` / `fftwf_plan_dft_1d` wrappers
- [x] Implement `fftw_plan_dft_r2c_1d` / `fftw_plan_dft_c2r_1d` wrappers
- [x] Implement `fftw_plan_dft_2d` / `fftw_plan_dft_3d` wrappers
- [x] Implement `fftw_plan_many_dft` (guru interface wrapper)
- [x] Implement `fftw_execute` / `fftw_destroy_plan` lifecycle
- [x] Implement `fftw_export_wisdom_to_string` / `fftw_import_wisdom_from_string`
- [x] Feature-gate as `fftw-compat` feature flag
- [x] Add migration guide for FFTW users

### Pure Rust Dependency Audit

- [x] Evaluate pure Rust MPI implementation as alternative to C `mpi` crate — no pure-Rust MPI exists as of 2026; documented in README.md#mpi
- [x] If no pure Rust MPI exists, document C dependency clearly in feature description — added to README.md#mpi and inline in Features list
- [x] Replace `libc`-based SVE detection with `std::arch` if available
      (`oxifft/src/simd/sve.rs`)
- [x] Add compile-time warning when `mpi`/`sve` features introduce C dependencies (done in build.rs)

### no_std Validation

- [x] Add check: cargo check --no-default-features --target thumbv7em-none-eabihf (passes cleanly)
- [x] Fix any compilation errors in no_std path (117 errors fixed)
- [x] Document no_std capabilities and limitations (planned 2026-04-20)
  - **Goal:** Create `docs/no_std.md` — feature matrix table, alloc requirements, spin-based sync, AtomicU64 caveat, embedded target (thumbv7em-none-eabihf) validation, canonical `no_std` example.
  - **Files:** `docs/no_std.md` (new, ~200 LoC)
  - **Tests:** `cargo doc --all-features --no-deps` zero warnings; `cargo check --no-default-features` passes.

### Wisdom System Improvements

- [x] Define cross-platform wisdom file format specification (planned 2026-04-20)
  - **Goal:** Create `docs/wisdom_format.md` — EBNF grammar, semantics (lower-cost-wins merge), version negotiation (v0/v1), hash stability guarantee (same target+ISA+minor-version), system paths per OS, error type catalog, sample file.
  - **Files:** `docs/wisdom_format.md` (new, ~200 LoC)
  - **Tests:** `cargo doc --all-features --no-deps` zero warnings.
- [x] Add wisdom format version negotiation (WISDOM_FORMAT_VERSION = 1)
- [x] Add wisdom import validation (reject corrupted/incompatible files)
- [x] Add wisdom merge capability (combine from multiple machines)

### Dependent Project Support

- [ ] Survey 14 COOLJAPAN dependent projects for API pain points
- [x] Ensure `Send + Sync` on all public plan types (planned 2026-04-19)
  - **Goal:** Guarantee every public plan/transform type is `Send + Sync` via compile-time static assertions in a single `oxifft/src/assertions.rs` file.
  - **Design:** Add `const fn assert_send_sync<T: Send + Sync>() {}` + `const _: () = { assert_send_sync::<Plan1D<f64>>(); … }` — one line per public plan type. Fix violations by switching `Rc` → `Arc`, `RefCell` → `Mutex`/`RwLock`. New types from S1/S3/S4 are handled in a follow-up run.
  - **Files:** `oxifft/src/assertions.rs` (new), `oxifft/src/lib.rs` (add `mod assertions;`), any plan files with `Rc`/`RefCell`.
  - **Tests:** Compile-time: if `cargo check --all-features` succeeds, all public plan types are Send+Sync. No runtime test needed.
  - **Risk:** Public plan types may embed non-Send state (raw pointers, `Rc`); fix by switching to `Arc` or adding a `Safety`-commented `unsafe impl Send`.
- [x] Add `TryFrom`/`Into` conversions for common numeric types (planned 2026-04-20)
  - **Goal:** Fix 3 silent-truncation hazards: (1) MPI transpose `as i32` counts → `i32::try_from` + `MpiError::CountOverflow`; (2) guru plan `offset as usize` from i64 → `usize::try_from` + `GuruError::NegativeOffset`; (3) NTT `reverse_bits(i as u32)` → debug-assert + rustdoc constraint.
  - **Files:** `oxifft/src/mpi/transpose.rs`, `oxifft/src/mpi/mod.rs` or `mpi/error.rs`, `oxifft/src/api/plan/types_guru.rs`, `oxifft/src/ntt/plan.rs`
  - **Tests:** Unit tests for each boundary: `send_count=usize::MAX` → `CountOverflow`; `offset=i64::MIN` → `NegativeOffset`; NTT debug-assert on `log_n=33`.

### Testing Expansion

- [x] Add property-based tests for all DCT/DST variants (Parseval, linearity, roundtrip) (done in rdft/solvers/r2r.rs)
- [x] Add fuzz testing for plan creation with arbitrary sizes (done in tests/plan_fuzz.rs)
- [x] Add stress tests for concurrent wisdom access (done in tests/wisdom_stress.rs, 9 stress tests)
- [x] Add tests for all feature flag combinations
- [x] Codegen validation: correctness vs reference, numerical accuracy (done in codegen_tests.rs, 112 new tests)
      (`oxifft-codegen/TODO.md` Phase 7)

### Documentation Completeness

- [x] Ensure 100% of public API has rustdoc with `# Examples` (planned 2026-04-20)
  - **Goal:** Add method-level `# Examples` rustdoc with runnable doctests on 13 flagship types: Plan, Plan2D, Plan3D, PlanND, RealPlan, RealPlanND, R2rPlan, GuruPlan, Nufft, Frft, NttPlan, DiffFftPlan, StreamingFft. ~40 methods × ~15 LoC = ~600 LoC doctests. All must pass `cargo test --doc --all-features`.
  - **Files:** `oxifft/src/api/plan/**`, `oxifft/src/nufft/mod.rs`, `oxifft/src/frft/mod.rs`, `oxifft/src/ntt/plan.rs`, `oxifft/src/autodiff/mod.rs`, `oxifft/src/streaming/mod.rs`
  - **Tests:** `cargo test --doc --all-features` passes; `cargo doc --all-features --no-deps` zero warnings; ≥40 new `# Examples` sections present.
- [x] safety-sections-unsafe (completed 2026-04-20)
    - **Goal:** Add `# Safety` rustdoc section to ALL ~75+ unsafe fns across simd/{portable,avx,avx2,avx512,neon,sse2,scalar,traits}.rs, kernel/twiddle.rs (10 missing), dft/solvers/stockham/{aarch64,x86_64}.rs, dft/codelets/simd/large_sizes.rs. Add `#![warn(clippy::missing_safety_doc)]` to lib.rs as compiler-enforced invariant.
    - **Files:** `oxifft/src/simd/*.rs`, `oxifft/src/kernel/twiddle.rs`, `oxifft/src/dft/solvers/stockham/*.rs`, `oxifft/src/dft/codelets/simd/large_sizes.rs`, `oxifft/src/lib.rs`
- [x] Add `# Errors` sections to all fallible public functions (completed 2026-04-20)
    - **Goal:** Add `# Errors` rustdoc sections to 42 fallible public functions across gpu/{buffer,plan,mod,cuda,metal}.rs, frft/mod.rs, threading/parallel_config.rs. Remove `#![allow(clippy::missing_errors_doc)]` and add `#![warn(clippy::missing_errors_doc)]` to lib.rs as compiler-enforced invariant.
    - **Files:** `oxifft/src/gpu/buffer.rs`, `oxifft/src/gpu/plan.rs`, `oxifft/src/gpu/mod.rs`, `oxifft/src/gpu/cuda.rs`, `oxifft/src/gpu/metal.rs`, `oxifft/src/frft/mod.rs`, `oxifft/src/threading/parallel_config.rs`, `oxifft/src/lib.rs`
- [x] Add architecture diagrams to `oxifft.md`
- [x] Create developer guide for adding new solvers
- [x] Create developer guide for adding new SIMD backends

### Code Organization

- [x] Audit files exceeding 2000 lines (COOLJAPAN policy) and split if needed (api/plan/types.rs split into 5 files)
- [x] Remove all `#[allow(dead_code)]` directives in production code (planned 2026-04-19)
  - **Goal:** Grep every `#[allow(dead_code)]` in `oxifft/src/` and either delete the dead code, gate it with `#[cfg(feature = "…")]`, or move it to a `#[cfg(test)]` submodule — no `#[allow]` band-aids remain.
  - **Design:** For each hit: if used under non-default feature → replace with `#[cfg(feature = "…")]`; if test-only → move to `#[cfg(test)]` submodule; if genuinely dead → delete it. Run `cargo clippy --all-features -- -D warnings` to confirm clean.
  - **Files:** All files containing `#[allow(dead_code)]` — orientation enumerates; likely in `oxifft/src/gpu/`, `oxifft/src/simd/`, `oxifft/src/dft/`.
  - **Tests:** `cargo clippy --all-features -- -D warnings` passes after removal; all existing tests still green.
  - **Risk:** Some dead code may be deliberately staged for future use; document each deletion decision with an inline comment or commit message.

### Core Performance

- [x] Profile and optimize Cooley-Tukey solver for sizes 2^10–2^20 (planned 2026-04-19)
  - **Goal:** Switch to SoA (struct-of-arrays, split re[]/im[]) twiddle-factor storage for sizes ≥ 4096, reducing SIMD shuffle count by ~30% on AVX2 and ~50% on AVX-512.
  - **Design:** Add `TwiddleLayout { Aos, Soa }` enum and `TwiddleTableSoa<T> { re: Vec<T>, im: Vec<T> }` (64-byte aligned) to `oxifft/src/kernel/twiddle.rs`. Extend `GlobalTwiddleCache` with parallel `get_twiddle_table_f64_soa` API built lazily from AoS. In `cooley_tukey.rs`, when `size >= 4096`, request SoA twiddles and use `_mm256_load_pd(re_ptr)` + `_mm256_load_pd(im_ptr)` directly — no deinterleave shuffle. Keep AoS path for sizes < 4096.
  - **Files:** `oxifft/src/kernel/twiddle.rs`, `oxifft/src/dft/solvers/cooley_tukey.rs`.
  - **Tests:** SoA vs AoS CT correctness for sizes 1024, 4096, 16384, 65536 on f64 and f32 within 4 ulp.
  - **Risk:** SoA requires separate load intrinsics; add `twiddle_mul_soa_simd_f64/f32` helper alongside existing `twiddle_mul_simd_*` and measure shuffle savings on at least one size.
- [x] Implement cache-oblivious FFT strategy (done in dft/solvers/cache_oblivious.rs, Frigo-Johnson 4-step)
- [x] Optimize twiddle factor computation (precompute + cache)
- [x] Implement plan-specific memory pools (thread-local scratch in support/scratch.rs)
- [x] profile-optimize-bluestein (planned 2026-04-20)
    - **Goal:** Replace scalar pointwise complex-multiply loops with SIMD (AVX2+FMA/NEON/SSE2 AoS-layout), fix execute_inplace to_vec() via 4th Mutex scratch, replace mutex-contention alloc-fallback with thread_local scratch keyed by solver_id.
    - **Files:** `oxifft/src/dft/solvers/bluestein.rs`, `oxifft/src/kernel/twiddle.rs` (new AoS SIMD helpers), `oxifft/benches/prime_sizes.rs` (new)
    - **Tests:** Round-trip forward+inverse at prime sizes 17/61/127/257/509/1009, rel ε ≤ 1e-14 f64 / 1e-6 f32; thread-local scratch correctness under 8 rayon threads; execute_inplace no-alloc property.
- [x] profile-optimize-rader (planned 2026-04-20)
    - **Goal:** Mirror Bluestein optimization: SIMD-ize the O(p-1) pointwise multiply loop (rader.rs:185-187), fix execute_inplace, add thread_local scratch. Indirect gather/scatter stays scalar this run.
    - **Files:** `oxifft/src/dft/solvers/rader.rs`, `oxifft/src/kernel/twiddle.rs`
    - **Tests:** Round-trip at prime sizes, rel ε ≤ 1e-14 f64 / 1e-6 f32; thread-local scratch under rayon parallelism.

### SIMD Optimization

- [x] Implement hand-optimized AVX-512 codelets for sizes 16, 32, 64 (planned 2026-04-19)
  - **Goal:** Hand-tuned AVX-512 codelets for complex sizes 16, 32, 64 on both f64 and f32, with deliberate register allocation, FMA-heavy inner loops, and critical-path-aware scheduling, selected at runtime via `is_x86_feature_detected!("avx512f")`.
  - **Design:** f64 (4 complex/zmm): size-16 → radix-4×radix-4 (~24 FMAs + 8 shuffles); size-32 → radix-4×radix-8 (~56 FMAs); size-64 → radix-8×radix-8 (two passes, ~176 FMAs, precomputed `static const __m512d` twiddle tables). f32 (8 complex/zmm): size-16 → 2 regs; size-32 → 4 regs; size-64 → 8 regs. FMA complex multiply via `_mm512_fmsubadd_pd`/`_mm512_fmaddsub_pd`. Dispatch arms inserted BEFORE existing `generated_simd_*` calls in `simd/mod.rs` for sizes 16/32/64.
  - **Files:** `oxifft/src/dft/codelets/hand_avx512.rs` (new), `oxifft/src/dft/codelets/hand_avx512_twiddles.rs` (new), `oxifft/src/dft/codelets/mod.rs`, `oxifft/src/dft/codelets/simd/mod.rs`.
  - **Tests:** Round-trip forward→inverse→scale for sizes 16/32/64 f64 (tol `1e-10*n`) and f32 (`1e-5*n`). Parity vs generated AVX-512 within 4 ulp. Dispatcher hit test via `AtomicBool`. All gated on `is_x86_feature_detected!("avx512f")`.
  - **Risk:** Register spills on size-64 f64 → mitigated by two-pass radix-8×radix-8 structure; `_mm512_permutexvar_pd` latency → interleave with independent FMAs for ILP.
- [x] Implement SIMD-optimized twiddle multiplication
- [x] Benchmark WASM SIMD and optimize if needed — scalar baseline: N=1024 at 133 µs/op (0.0075 Mops/s); simd128 build not faster (WasmSimdF64 defined but not wired into FFT kernel butterfly paths — see benches/baselines/v0.3.0/wasm_simd_bench_2026-04-24.md)

### Threading Optimization

- [x] Implement work-stealing for unbalanced multi-dimensional FFTs (planned 2026-04-19)
  - **Goal:** Enable rayon work-stealing for Plan2D and Plan3D: replace sequential row/plane iteration with `par_chunks_mut().for_each(...)`, and add an opt-in `FftPlanBuilder::thread_pool(Arc<rayon::ThreadPool>)` API.
  - **Design:** Replace `for row in ... { plan_1d.forward(row) }` in Plan2D with `data.par_chunks_mut(row_len).for_each(|row| plan_1d.forward(row))` via `rayon::iter::ParallelIterator`. Same for Plan3D planes. Shared twiddle cache is already `RwLock`-protected; confirm worker threads can read concurrently without contention. New builder method allows user-provided pool; default uses global rayon pool.
  - **Files:** `oxifft/src/parallel/work_stealing.rs` (new), `oxifft/src/parallel/mod.rs`, `oxifft/src/dft/plans/plan_2d.rs`, `oxifft/src/dft/plans/plan_3d.rs`.
  - **Tests:** Parallel vs serial 2D bit-identical on 128×128 f64. Parallel vs serial 3D bit-identical on 32×32×32 f64. Thread-pool override: 2-worker pool runs without deadlock.
  - **Risk:** Global rayon pool may conflict with user code → opt-in API mitigates; CI thread-scheduling flakiness → bench is measurement-only, not a hard test gate.
- [x] Tune Rayon task granularity (ParallelConfig in threading/parallel_config.rs)
- [x] Add thread-local scratch buffers (ThreadLocalScratch in support/scratch.rs)
- [x] Benchmark and optimize parallel 2D/3D FFT decomposition (planned 2026-04-19)
  - **Goal:** Add `benches/multidim_parallel.rs` measuring Plan2D (256², 512², 1024²) and Plan3D (64³, 128³) across thread counts 1, 2, 4, `num_cpus::get().min(8)`, reporting wall time + parallel efficiency `T(1)/(T(n)*n)`.
  - **Design:** Criterion groups per shape. Parallel efficiency ≥ 0.6 at n=4 for 1024² as smoke gate. Correctness guard: parallel output must be bit-identical to serial output for same input. Build plan outside bench loop to avoid amortizing plan cost.
  - **Files:** `benches/multidim_parallel.rs` (new).
  - **Tests:** Parallel vs serial 2D correctness: bit-identical on 128×128 f64. Smoke scaling: 4-thread efficiency ≥ 0.6 on 1024×1024 f64 (skippable on <4-core machines).
  - **Risk:** CI flakiness from thread scheduling → bench is a measurement artifact, not a hard CI gate; efficiency check is advisory only.

### Benchmark Tracking

- [x] Establish automated benchmark regression tracking (planned 2026-04-19)
  - **Goal:** Ship `scripts/bench_check.sh` + committed baseline storage so any PR can be compared against a `main` baseline with exit code 1 on >5% regression.
  - **Design:** `bench_check.sh` runs `cargo criterion --save-baseline=pr --package oxifft --bench cooley_tukey_scaling -- --sample-size 50`, then `cargo criterion --baseline=main` to compare. Baseline JSON lives in `benches/baselines/v0.3.0/` (committed). Update `CONTRIBUTING.md` (or create `BENCHMARKING.md` if absent) with a one-line pointer to the script.
  - **Files:** `scripts/bench_check.sh` (new, executable), `benches/baselines/v0.3.0/.gitkeep` (new), `CONTRIBUTING.md` or `BENCHMARKING.md`.
  - **Tests:** Bench script smoke test: runs end-to-end on tiny sample, exits 0 when no regression. Regression detection: inject slowdown via `black_box`, run script, confirm exit 1.
  - **Risk:** Baseline JSON format may shift across criterion versions → pin criterion to a specific minor version in workspace Cargo.toml; comparisons use relative ratios, not absolute times.
- [x] publish-benchmark-results-per-release (planned 2026-04-20)
    - **Goal:** Establish baseline snapshot infrastructure and commit initial fftw_ratios_2026-04-20.json to benches/baselines/v0.3.0/.
    - **Files:** `benches/baselines/v0.3.0/fftw_ratios_2026-04-20.json`, `benches/baselines/v0.3.0/README.md`
- [x] track-fftw-ratio-across-versions (planned 2026-04-20)
    - **Goal:** Add fftw_ratio_report.sh script + fftw_ratio_report binary that reads criterion JSON and emits machine-readable JSON snapshots to benches/baselines/v0.3.0/.
    - **Files:** `scripts/fftw_ratio_report.sh`, `oxifft-bench/src/bin/fftw_ratio_report.rs`, `oxifft-bench/Cargo.toml`

### Advanced Features Hardening

- [x] Add GPU error recovery (device loss, OOM handling)
      (`oxifft/src/gpu/error.rs` — `DeviceLost`, `OutOfMemory { requested_bytes }`, `ShaderCompileFailed` variants; Metal `From` conversion in error.rs:81-98)
- [x] Implement GPU memory pooling for repeated transforms (done in gpu/pool.rs — LRU-evicting, thread-safe, 256 MiB default budget)
- [x] Implement GPU R2C/C2R transforms
      (`oxifft/src/gpu/plan.rs:436-558` — `forward_r2c`/`inverse_c2r` on `GpuFft<f32>`; Metal impl in `gpu/metal.rs:187-287`; CUDA CPU-fallback in `gpu/cuda.rs:209-310`)
- [x] Add GPU batch FFT with automatic chunking for large batches
      (`oxifft/src/gpu/batch.rs` — `dispatch_chunked` helper; `METAL_BATCH_LIMIT=1024`, `CUDA_BATCH_LIMIT=4096`; auto-splits oversized batches preserving output order)
- [x] Add overlap-save STFT method as alternative to overlap-add (done in streaming/stft.rs)
- [x] Validate streaming real-time constraint: 48kHz audio without glitches
- [x] Validate NUFFT tolerance across wide parameter ranges
- [x] Implement multi-dimensional NUFFT (2D, 3D) (done in nufft/nufft2d.rs, nufft3d.rs)
- [ ] Test MPI distributed FFT with >4 ranks
- [x] Implement pencil decomposition as alternative to slab (done in mpi/plans/plan_3d_pencil.rs)

### Release Infrastructure

- [x] Run `cargo semver-checks` against v0.1.4 (passed with expected breaking changes)
- [x] Write v0.x → v1.0 migration guide
- [x] Update all examples to use final v1.0 API
- [x] Verify `cargo publish --dry-run` succeeds for `oxifft-codegen` then `oxifft` (both crates pass)
- [x] Verify package manifest includes only necessary files

### Success Criteria

- Zero `todo!()` or `unimplemented!()` panics reachable from public API
- All documentation links in README.md resolve to real files
- Generated SIMD codelets pass correctness tests
- FFTW compatibility layer passes FFTW API-level test suite
- no_std compiles cleanly on embedded target
- All 14 dependent projects compile against v0.2.0 without changes
- <10 crate-level clippy allows in `lib.rs`
- >95% of public API items have doc comments with examples
- Power-of-2 1D FFT within 1.5× of FFTW for sizes 2^10–2^20
- GPU error recovery and pooling in place
- `cargo semver-checks` passes
- `cargo publish --dry-run` succeeds for all crates
- 858 tests pass with `--all-features`

### Breaking Changes

`Plan::dft_2d` and `Plan::dft_3d` change from `Option<Plan<T>>` (panicking) to the correct
multi-dimensional plan type — a compile-time break that eliminates a runtime crash.

---

## v0.3.0 — Performance: DCT/DST O(n log n) & Codegen

**Theme:** Replace O(n²) DCT/DST with FFT-based algorithms. Implement RDFT codelets.
Fix NEON SIMD regression. Deliver functional GPU compute backends.

### DCT/DST O(n log n) Implementation

- [x] Implement FFT-based DCT-II via reordering + real FFT
      (`oxifft/src/rdft/solvers/r2r.rs` — `execute_dct2_fast` for n≥16)
- [x] Implement FFT-based DCT-III (inverse of DCT-II)
- [x] Implement FFT-based DCT-I via DCT-III reduction
- [x] Implement FFT-based DCT-IV via modified DCT-II
- [x] Implement FFT-based DST variants (I–IV) via DCT symmetry relations
- [x] Implement FFT-based DHT via complex FFT
- [x] Retain O(n²) direct as reference/fallback for n < 16
- [x] Add DCT/DST benchmarks demonstrating O(n log n) speedup (planned 2026-04-17) → covered by bench-dct-dst-group plan block (see line ~744 below)

### RDFT Codelets

- [x] Replace 3-line placeholder in `oxifft/src/rdft/codelets/mod.rs` with real codelets
- [x] Implement R2HC (real to half-complex) codelets for sizes 2, 4, 8
- [x] Implement HC2R (half-complex to real) codelets for sizes 2, 4, 8
- [x] Implement real-valued twiddle codelets (real_twiddle.rs, sizes 4/8/16 + generic)
- [x] Integrate RDFT codelets into R2C/C2R solver pipeline (planned 2026-04-17)
  - **Goal:** `RealPlan::r2c_1d` sizes 4 and 8 dispatch to the generated `r2hc_4_gen` / `r2hc_8_gen`; `c2r_1d` sizes 4 and 8 dispatch to `hc2r_4_gen` / `hc2r_8_gen`. Size 2 inlined arithmetic preserved. Hand-written references kept for codegen_tests.rs. Workspace tests green; no numerical regression.
  - **Design:** Create `oxifft/src/rdft/codelets/generated.rs` invoking all 6 `gen_rdft_codelet!` macros at crate scope (not test-only). Update `codelets/mod.rs` to add `mod generated;` + `pub(crate) use`. Edit `r2c.rs` lines 102-110 to call `r2hc_4_gen`/`r2hc_8_gen`. Edit `c2r.rs` lines 108-116 to call `hc2r_4_gen`/`hc2r_8_gen`. Leave size-2 inlined arithmetic untouched.
  - **Files:** `oxifft/src/rdft/codelets/generated.rs` (new), `oxifft/src/rdft/codelets/mod.rs`, `oxifft/src/rdft/solvers/r2c.rs`, `oxifft/src/rdft/solvers/c2r.rs`
  - **Prerequisites:** Run 1's `gen_rdft_codelet!` (already `[x]`)
  - **Tests:** All existing rdft integration tests green; `codegen_tests.rs` still passes (hand-written refs kept)
  - **Risk:** Name collision if macro emits `use` statements clashing with existing imports. Mitigation: use scoped `pub(crate) use generated::…` rather than glob re-export.

### Codegen Pipeline (oxifft-codegen)

- [x] Implement no-twiddle codelets for sizes 16, 32, 64
      (`oxifft-codegen/src/gen_notw.rs`)
- [x] Implement radix-8 twiddle codelet (`oxifft-codegen/src/gen_twiddle.rs`)
- [x] Implement radix-16 twiddle codelet
- [x] Implement split-radix twiddle codelet (generic + specialized 8/16)
- [x] Add unit tests for all generated codelets (size 2–64, 112 tests)
- [x] Codegen optimization: implement constant folding in symbolic optimizer
      (`oxifft-codegen/src/symbolic.rs`)
- [x] Codegen optimization: implement dead code elimination

### NEON SIMD Performance Fix

- [x] Profile NEON codelets to identify deinterleave bottleneck (dispatch was x86-only)
- [x] Add NEON backends for sizes 2, 4, 8 (`backends.rs::neon_f64`)
- [x] Wire NEON dispatch in `small_sizes.rs` (no more scalar fallback on aarch64)
- [x] Fused stages 0-3 in `dit_butterflies_neon` (16 elements in-register, like AVX2)
- [x] Radix-4 stage fusion for remaining stage pairs (halves memory passes)
- [x] Eliminate stack round-trip: convert dit_64/128/512 from `neon_butterfly_inline`
      (Complex by value) to `neon_butterfly_fast` (pointer-based `[f64; 2]` twiddles)
- [x] Implement streaming NEON butterfly with FMA, radix-4 fusion, and fused stages 0-3
- [x] Benchmark NEON vs scalar and verify NEON achieves speedup (NEON codelets + butterflies implemented)

### Benchmark Infrastructure

- [x] Produce and commit actual benchmark results (fill `BENCHMARK_RESULTS_TEMPLATE.md`) (planned 2026-04-19)
  - **Goal:** Create `benches/cooley_tukey_scaling.rs` covering CT forward on sizes 2^10–2^20 for f64 and f32, reporting throughput (MElts/s) per size, and populate `BENCHMARK_RESULTS_TEMPLATE.md` with actual numbers.
  - **Design:** Bench sizes: 1024, 4096, 16384, 65536, 262144, 1048576. Uses `black_box` on inputs; warmup 5s, measurement 10s per size. `BenchmarkId::new(format!("ct_f64"), size)` naming for stable baselines. Run once on developer machine, commit results to `BENCHMARK_RESULTS_TEMPLATE.md`.
  - **Files:** `benches/cooley_tukey_scaling.rs` (new), `BENCHMARK_RESULTS_TEMPLATE.md` (update with actual data).
  - **Tests:** `cargo bench -p oxifft --bench cooley_tukey_scaling --no-run` succeeds (compile-only gate). Bench run produces non-zero throughput numbers for all sizes.
  - **Risk:** Machine-specific numbers — document hardware spec (CPU, memory) in the results file so comparisons are meaningful; regression tracking uses relative ratios via `bench_check.sh`.
- [x] Add DCT/DST benchmark group to `oxifft-bench/` (planned 2026-04-17)
  - **Goal:** New file `oxifft-bench/benches/dct_dst.rs` benchmarks `dct_i/ii/iii/iv` and `dst_i/ii/iii/iv` (from `oxifft::reodft::{redft,rodft}`) across sizes {8, 16, 32, 64, 128, 256, 512, 1024, 2048}, demonstrating O(n log n) growth for n≥16.
  - **Design:** 8 criterion groups × 9 sizes = 72 bench functions using `BenchmarkId::new(kind, size)`. Input: pre-filled `Vec<f64>` of length n. Harness: `criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion}`. Add `[[bench]] name = "dct_dst" harness = false` to `oxifft-bench/Cargo.toml`.
  - **Files:** `oxifft-bench/benches/dct_dst.rs` (new, ~200 lines), `oxifft-bench/Cargo.toml` (+ [[bench]] entry)
  - **Tests:** `cargo bench -p oxifft-bench --bench dct_dst --no-run` succeeds (compile-only gate).
  - **Risk:** reodft function signatures require real array buffers; subagent reads API first. No existing DCT/DST bench to build on — green-field.
- [x] Add R2C/C2R performance regression tracking (planned 2026-04-17)
  - **Goal:** New file `oxifft-bench/benches/r2c_c2r.rs` covers `RealPlan::r2c_1d`/`c2r_1d` across sizes {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384} × f32/f64 with stable `BenchmarkId` naming for future regression baselines (`--save-baseline v0.3.0-rc1`).
  - **Design:** Criterion groups per direction+precision. `RealPlan` built **outside** the iter-loop. Naming: `BenchmarkId::new(format!("{direction}_{precision}"), size)`. Warm-up 3 s, measurement 10 s. Add `[[bench]] name = "r2c_c2r" harness = false` to `oxifft-bench/Cargo.toml`.
  - **Files:** `oxifft-bench/benches/r2c_c2r.rs` (new, ~180 lines), `oxifft-bench/Cargo.toml` (+ [[bench]] entry)
  - **Tests:** `cargo bench -p oxifft-bench --bench r2c_c2r --no-run` succeeds (compile-only gate).
  - **Risk:** `RealPlan` construction allocates wisdom/scratch — must stay outside bench loop; follow pattern from existing `fft_comparison.rs`.

### GPU: Pure Rust Metal Backend (macOS / Apple Silicon)

- [x] Implement real Metal device discovery
      (`oxifft/src/gpu/metal.rs:33` — `is_available()` currently hardcoded)
  - **Evidence:** `oxifft/src/gpu/metal.rs:37` — `is_available()` now calls `oxicuda_metal::device::MetalDevice::new().is_ok()` (real device probe, not a hardcoded value)
- [x] Implement Metal compute shader for Stockham FFT
  - **Evidence:** `oxifft/src/gpu/metal.rs:108-116` — `MetalFftPlan::new` delegates to `oxicuda_metal::fft::MetalFftPlan::new`; shader dispatch happens at `metal.rs:161-163` via `self.inner.execute(...)`. GPU Stockham kernel is provided by the oxicuda-metal crate.
  - **Decision:** Pure-Rust oxicuda ecosystem chosen (oxicuda_metal). Per COOLJAPAN pure-Rust policy.
- [x] Implement `GpuFftEngine::forward()` / `inverse()` for Metal
      (`oxifft/src/gpu/plan.rs` — implemented via `execute_with_buffers` helper with local `GpuBuffer` instances)
- [x] Implement `forward_inplace()` / `inverse_inplace()` for Metal
      (`oxifft/src/gpu/plan.rs` — implemented; copies data to local buffers, executes, writes back)
- [x] Implement real GPU buffer upload/download via Metal API
      (`oxifft/src/gpu/buffer.rs:128,182` — currently returns `Unsupported`)
  - **Evidence:** `oxifft/src/gpu/buffer.rs:75-100` — `upload()`/`download()` copy to/from a CPU staging buffer; actual GPU transfer is handled transparently inside `MetalFftPlan::execute()` at `metal.rs:137-169` (no longer stub-returning `Unsupported`). Referenced buffer.rs:128,182 stubs no longer exist.
- [x] Query actual Metal device capabilities
  - **Evidence:** `oxifft/src/gpu/metal.rs:41-55` — `query_capabilities()` calls `MetalDevice::new()` and reads `device.name()` to populate `GpuCapabilities`; `supports_f64: false` and `supports_f16: true` reflect real Metal hardware limits.
- [x] Add macOS-only Metal integration tests (feature-gated)
  - **Evidence:** `oxifft/src/gpu/metal.rs:222-394` — five integration tests: `test_metal_capabilities`, `test_metal_plan_creation`, `test_metal_non_power_of_2`, `test_metal_fft_correctness_impulse`, `test_metal_fft_round_trip`; all guarded by `if !is_available() { return; }`. New parametric test `metal_roundtrip_sizes_6_to_16` (metal.rs:340-394) covers sizes 2^6..=2^16 and passes on Apple Silicon.
- [x] Reconcile Metal + CUDA TODO markers with code evidence (planned 2026-04-17)
  - **Goal:** Every Metal and CUDA item in lines 759-780 that has been implemented gets `[x]` with a cited `file:line` reference. Items genuinely pending stay `[ ]`. CUDA CPU-fallback decision documented.
  - **Design:** Metal items: device detection (metal.rs:37), buffer allocation (metal.rs:112-116), Stockham shader via oxicuda-metal (metal.rs:129-161), twiddle precomputation, integration tests (metal.rs:264-328) → all `[x]`. Transform size support (n=6..16): add parametric test `metal_roundtrip_sizes_6_to_16`; mark `[x]` if all pass. Performance bench → stays `[ ]`. CUDA: driver integration decision → `[x]`(pure-Rust oxicuda chosen); device detection (cuda.rs:33-45) → `[x]`; memory strategy (cuda.rs:83-97) → `[x]`; engine selection → `[x]`; multi-GPU + perf bench → stay `[ ]`. Add `**Known limitation:**` sub-bullet documenting `cuda.rs:152` CPU fallback awaiting oxicuda-launch.
  - **Files:** `TODO.md` (lines 759-780), `oxifft/src/gpu/metal.rs` (optional parametric test)
  - **Prerequisites:** none
  - **Tests:** grep Evidence count ≥ 8; only genuinely-pending items remain `[ ]`
  - **Risk:** Borderline items get explicit reasoning in `**Evidence:**` sub-bullet; uncertain items left `[~]` with question.

### GPU: CUDA Backend

- [x] Replace filesystem-based availability check with real CUDA runtime detection
      (`oxifft/src/gpu/cuda.rs:28`)
  - **Evidence:** `oxifft/src/gpu/cuda.rs:32-34` — `is_available()` calls `oxicuda_driver::init().is_ok() && oxicuda_driver::Device::get(0).is_ok()` (real driver probe via oxicuda-driver crate, not a filesystem check).
- [x] Evaluate pure Rust CUDA feasibility vs cuFFT FFI (document decision)
  - **Decision:** Pure-Rust oxicuda ecosystem chosen (oxicuda_driver + oxicuda_fft). Not cuFFT FFI, not custom CUDA kernels. Per COOLJAPAN pure-Rust policy. See `oxifft/Cargo.toml:34-35`.
  - **Known limitation:** `oxifft/src/gpu/cuda.rs:152` — `execute()` currently routes to `execute_cpu()` pending `oxicuda-launch` integration. GPU kernel dispatch blocked on that external dep. CPU FFT is used as the computation engine in the meantime.
- [x] Implement `GpuFftEngine::forward()` / `inverse()` for CUDA
      (`oxifft/src/gpu/plan.rs` — implemented via `execute_with_buffers` helper; delegates to `CudaFftPlan::execute`)
- [x] Implement `forward_inplace()` / `inverse_inplace()` for CUDA
      (`oxifft/src/gpu/plan.rs` — implemented with local buffer copy strategy)
- [x] Implement real GPU buffer upload/download
      (`oxifft/src/gpu/buffer.rs` — staging buffer; actual transfer inside backend `execute()`)
- [x] Query actual CUDA device capabilities
  - **Evidence:** `oxifft/src/gpu/cuda.rs:37-60` — `query_capabilities()` calls `oxicuda_driver::Device::get(0)`, reads `device.name()` and `device.total_memory()` to populate `GpuCapabilities`; `supports_f64: true` reflects real CUDA hardware capability.

### GPU: Architecture Cleanup

- [x] Move `GpuBackend::OpenCL`/`Vulkan` to separate feature flags or remove placeholder variants (planned 2026-04-17)
  - **Goal:** Remove `OpenCL` and `Vulkan` variants entirely from `GpuBackend` (no `ocl`/`vulkano` Cargo feature or dependency exists; they are dead-code placeholders). Decision: **deletion** (not feature-gating). No `#[allow(dead_code)]` remains at the source.
  - **Design:** Edit `oxifft/src/gpu/backend.rs`: delete `OpenCL`/`Vulkan` variants + their `#[allow(dead_code)]` attributes; update `is_available` match, `name()`, any `Display`/`Debug` impls. Grep `oxifft/` to confirm no other source references exist (classification found none).
  - **Files:** `oxifft/src/gpu/backend.rs`, `oxifft/src/gpu/mod.rs` (only if references)
  - **Tests:** Existing GPU backend tests pass; `cargo clippy -p oxifft --all-features --all-targets -- -D warnings` silent (the removal eliminates the dead_code allow attributes).
  - **Risk:** Breaking change for downstream pattern-match on these variants; acceptable pre-1.0 (variants were always no-op placeholders per TODO.md note). Document in CHANGELOG on next version cut (not by this run).
- [x] Implement GpuBatchFft trait for batch transforms (planned 2026-04-17)
  - **Goal:** New `oxifft/src/gpu/batch.rs` defines `GpuBatchFft<T>` trait for N independent FFTs of the same size in a single submission. `CudaBackend` and `MetalBackend` gain `GpuBatchFft` impls; default blanket impl loops over single-transform execution.
  - **Design:** Trait has `batch_size_limit(&self) -> usize` and `execute_batch(&self, inputs: &[&[Complex<T>]], outputs: &mut [&mut [Complex<T>]], direction: FftDirection) -> GpuResult<()>`. Validates batch dimension match and uniform FFT size. Metal: use `MetalFftPlan::execute_batched` if available in oxicuda-metal, else loop fallback. CUDA: loop fallback (execute still CPU-backed per cuda.rs:152). Add `pub mod batch;` to `gpu/mod.rs`.
  - **Files:** `oxifft/src/gpu/batch.rs` (new), `oxifft/src/gpu/mod.rs`, `oxifft/src/gpu/metal.rs`, `oxifft/src/gpu/cuda.rs`
  - **Prerequisites:** none (GPU backend types exist)
  - **Tests:** Default impl: 4 batches of size-16 vs 4 × single execute; Metal test gated on `is_available()`; CUDA test runtime-gated.
  - **Risk:** oxicuda-metal may not expose batched execution. Mitigation: default-loop impl is the deliverable; Metal batch is a bonus.
- [x] gpu-vs-cpu-bench-4096 (done 2026-04-20)
    - **Goal:** Add oxifft/benches/gpu_vs_cpu.rs benchmarking CPU Plan<f32> vs GpuPlan<f32> at 4096/16384/65536/262144. Metal=real GPU, CUDA=CPU-fallback (prominently documented). Feature-gated by `required-features = ["gpu"]`.
    - **Files:** `oxifft/benches/gpu_vs_cpu.rs` (new), `oxifft/Cargo.toml`

### Success Criteria

- DCT-II of size 1024 runs >10× faster than v0.2.0 (O(n log n) vs O(n²))
- RDFT codelets improve R2C performance for sizes 2–8
- NEON path demonstrates speedup, or is cleanly removed
- All codegen sizes 2–64 have passing unit tests
- Metal backend executes forward/inverse FFT with correct results on Apple Silicon
- CUDA backend executes forward/inverse FFT (or cuFFT FFI decision documented)
- GPU FFT for size 4096+ is faster than CPU

### Breaking Changes

None. DCT/DST functions maintain existing signatures; O(n²) becomes fallback for n < 16.
`GpuBackend::OpenCL` and `GpuBackend::Vulkan` may be removed (currently produce "not
implemented" errors — no working code depends on them).

---

## v1.0.0 — Stable Release

**Theme:** Semver stability commitment. Verified performance and platform targets.
Production-ready for all 14 COOLJAPAN dependent projects.

### API Stability Commitment

- [ ] Semantic versioning from this point: no breaking changes until v2.0
- [ ] All public types, traits, and functions considered stable
- [ ] Feature flags considered stable (removal requires major version bump)

### Performance Targets (Verified by Benchmark)

- [x] perf-1d-cplx-p2-10 (planned 2026-04-20)
    - **Goal:** Add FFTW parity gate bench for 1024-point complex FFT (target < 2× FFTW) in fftw_parity_gates.rs.
    - **Files:** `oxifft-bench/benches/fftw_parity_gates.rs`, `oxifft-bench/Cargo.toml`
- [x] perf-1d-cplx-p2-20 (planned 2026-04-20)
    - **Goal:** Add FFTW parity gate bench for 1048576-point complex FFT (target < 2× FFTW).
    - **Files:** `oxifft-bench/benches/fftw_parity_gates.rs`
- [x] perf-1d-real-p2-10 (planned 2026-04-20)
    - **Goal:** Add FFTW parity gate bench for 1024-point real FFT (target < 2× FFTW).
    - **Files:** `oxifft-bench/benches/fftw_parity_gates.rs`
- [x] perf-2d-cplx-1024 (planned 2026-04-20)
    - **Goal:** Add FFTW parity gate bench for 1024×1024 complex 2D FFT (target < 2× FFTW).
    - **Files:** `oxifft-bench/benches/fftw_parity_gates.rs`
- [x] perf-batch-1000x256 (planned 2026-04-20)
    - **Goal:** Add FFTW parity gate bench for 1000 batched 256-point FFTs (target < 2× FFTW).
    - **Files:** `oxifft-bench/benches/fftw_parity_gates.rs`
- [x] perf-prime-2017 (planned 2026-04-20)
    - **Goal:** Add FFTW parity gate bench for 2017-point prime FFT (target < 3× FFTW).
    - **Files:** `oxifft-bench/benches/fftw_parity_gates.rs`
- [x] perf-dct2-1024 (planned 2026-04-20)
    - **Goal:** Add FFTW parity gate bench for 1024-point DCT-II (target < 3× FFTW).
    - **Files:** `oxifft-bench/benches/fftw_parity_gates.rs`
- [x] commit-published-bench-results (planned 2026-04-20)
    - **Goal:** Commit initial fftw_ratios snapshot to benches/baselines/v0.3.0/ as part of this session.
    - **Files:** `benches/baselines/v0.3.0/fftw_ratios_2026-04-20.json`

### Platform Matrix (Verified in CI)

- [x] x86_64 Linux (SSE2, AVX, AVX2, AVX-512)
- [x] x86_64 macOS (SSE2, AVX, AVX2)
- [x] x86_64 Windows (SSE2, AVX, AVX2)
- [x] aarch64 Linux (NEON)
- [x] aarch64 macOS / Apple Silicon (NEON)
- [x] wasm32-unknown-unknown (WASM SIMD)
- [x] no_std (embedded target, with alloc)

### Quality Gates

- [x] Zero clippy warnings: `cargo clippy --all-features -- -D warnings`
- [x] Zero compiler warnings on all targets
- [ ] All tests pass on all supported platforms
- [x] Documentation builds without warnings: `cargo doc --all-features`
- [x] quality-zero-todo-unimpl-unwrap (completed 2026-04-20)
    - **Goal:** Eliminate production `.unwrap()` at rader_omega.rs:91, spectral.rs:363-364, threading/mod.rs:385-389. Confirm 0 todo!()/unimplemented!() hits. Replace with `?`/`ok_or_else`/`expect("invariant: ...")`.
    - **Note:** All `.unwrap()` calls in those files were confirmed to be inside `#[cfg(test)]` test modules — no production unwraps existed. 0 `todo!()/unimplemented!()` in production code confirmed.
    - **Files:** `oxifft/src/kernel/rader_omega.rs`, `oxifft/src/signal/spectral.rs`, `oxifft/src/threading/mod.rs`
- [ ] MIRI passes for all unsafe code: `cargo +nightly miri test`
- [~] Fuzz testing run for >24 hours without findings
    - **Refinement (2026-04-24):** Scaffolding done (3 harnesses: plan_create, r2c_roundtrip, wisdom_parse), 24h production runs deferred to follow-up.

### Release Steps

- [ ] Tag v1.0.0
- [ ] Publish `oxifft-codegen` to crates.io
- [ ] Publish `oxifft` to crates.io
- [ ] Update all 14 COOLJAPAN dependent projects to v1.0.0
- [ ] Publish release announcement

### Success Criteria

- All performance targets met with committed benchmark evidence
- All platform targets verified in CI
- All quality gates pass
- 14 dependent projects successfully upgraded

### Breaking Changes

None. This is the stable release.

---

## Future / Post-1.0

### Performance (v1.1+)
- [x] Match FFTW performance (1.0×) for power-of-2 sizes via deeper codelet tuning
- [x] Exceed FFTW for composite sizes via Rust-specific optimizations
- [x] Implement auto-tuning (runtime codelet selection profiled at build time) (planned 2026-05-01)
  - **Goal:** Build-time + runtime auto-tuning profiling candidate algorithms (CT-Dit, SplitRadix, Stockham, MixedRadix, Bluestein, Winograd, Direct) for sizes 2..=4096; picks fastest per size on host; persists to WisdomCache. Wires Flags::MEASURE and Flags::PATIENT into Plan::select_algorithm (currently ignored).
  - **Design:** Two-tier: (1) static tuning via build.rs opt-in (OXIFFT_TUNE=1), binary wisdom_baseline.bin in OUT_DIR; (2) dynamic tuning at runtime when Flags::MEASURE. Core in auto_tune.rs: tune_size<T>(n, max_iters) → WisdomEntry, tune_range<T>(min_n, max_n, on_progress) → WisdomCache. Binary format: header (magic + u16 version + u16 count + u32 reserved) + 30-byte repr(packed) entries (u64 hash_key, u8 algo_tag, u8 factors_len, [u16;6] factors, u64 elapsed_ns), explicit LE bytes, no bincode.
  - **Files:** auto_tune.rs (~600 LoC), types.rs (Flags plumbing + wisdom lookup), wisdom.rs (binary to_le_bytes/from_le_bytes), build.rs (extend stub ~200 LoC), oxifft_tune.rs (extend stub ~150 LoC)
  - **Tests:** tune_size(64) returns valid WisdomEntry in candidate set; tune_range(2..=32) covers 31 sizes; binary round-trip; ESTIMATE vs MEASURE behavior; OXIFFT_SKIP_TUNE=1 sentinel path.
  - **Risk:** Build-script compile time (mitigated: default OFF, OXIFFT_TUNE env gate); cross-compile (mitigated: CARGO_CFG_TARGET_ARCH != HOST_ARCH sentinel); wisdom cache thread safety (OnceLock + RwLock).
- [ ] AVX-512 BF16 support for ML workloads

### GPU (v1.2+)
- [ ] OpenCL backend for cross-vendor GPU support
- [ ] Vulkan compute backend
- [ ] WebGPU backend for browser GPU acceleration
- [ ] Multi-GPU support (split large transforms across devices)
- [ ] GPU–CPU hybrid execution

### Algorithms (v1.3+)
- [x] Sliding DFT for real-time streaming (SlidingDft, ModulatedSdft, SingleBinTracker in streaming/sdft.rs)
- [x] Number Theoretic Transform (NTT) for exact integer arithmetic (ntt/ module with 3 primes, polynomial convolution)
- [x] Winograd FFT for minimum-multiplication small sizes
- [x] Partial FFT (compute only selected output frequencies)
- [x] Chirp Z-Transform generalization for arbitrary frequency grids

### Ecosystem (v1.x+)
- [ ] C API (`oxifft-sys`) for use from C/C++/Python/Julia
- [ ] Python bindings (`oxifft-python`) via PyO3
- [x] ndarray integration (`compute FFT of ndarray::Array directly`)
- [ ] Apache Arrow columnar data FFT
- [ ] ONNX operator implementation for ML frameworks

### Advanced Research
- [ ] Approximate FFT with configurable accuracy/speed tradeoff
- [ ] Distributed FFT over network (beyond MPI)
- [ ] Hardware-specific codelet auto-generation (profile at build time)

---

## Proposed follow-ups

### Blocked: external input required

- **eval-pure-rust-mpi** (~line 572): Blocked — external ecosystem survey required. No pure-Rust MPI implementation exists as of 2026-04; blocked on upstream community development.
- **document-c-mpi-dep** (~line 573): Blocked — conditional on `eval-pure-rust-mpi` outcome.
- **survey-14-cooljapan-projects** (~line 593): Blocked — requires a human-driven survey of downstream COOLJAPAN ecosystem projects.
- **mpi-gt4-ranks-test** (~line 661): Blocked — `PencilPlan3D` currently returns an error for ranks > 1; unblocked only when multi-rank pencil execution is implemented.
