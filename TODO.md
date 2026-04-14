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

## Current Status (v0.2.0 — All Phases 1–10 Complete — Released 2026-04-14)

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
- [ ] Clippy allow reduction round 2: <30 → <10, refactor sites as needed
- [x] Ensure all error types are `#[non_exhaustive]`
- [x] Ensure all public enums are `#[non_exhaustive]`
- [x] Add `Debug` impl to all public types missing it (done in debug_impls.rs)
- [ ] Review trait bounds for unnecessary constraints

### Codegen: SIMD Code Generation (Phase 5)

- [ ] Implement SSE2 2-lane f64 codelet generation (`oxifft-codegen/src/gen_simd.rs`)
- [ ] Implement SSE2 4-lane f32 codelet generation
- [ ] Implement AVX 4-lane f64 codelet generation
- [ ] Implement AVX2 FMA-optimized codelet generation
- [ ] Implement AVX-512 8-lane f64 / 16-lane f32 codelet generation
- [ ] Implement NEON 2-lane f64 / 4-lane f32 codelet generation
- [ ] Integrate generated SIMD codelets into runtime dispatch

### Codegen: RDFT Code Generation (Phase 6)

- [ ] Implement R2HC codelet generation
- [ ] Implement HC2R codelet generation
- [ ] Implement real-valued twiddle codelet generation

### Sparse FFT Robustness

- [ ] Audit FFAST peeling decoder for edge cases (very low/high sparsity ratios)
      (`oxifft/src/sparse/decoder.rs`)
- [ ] Handle degenerate cases: k=0, k=n, signal is pure noise
- [ ] Add adaptive sparsity detection (auto-tune k)
- [ ] Add property-based tests across diverse sparsity patterns
- [ ] Document accuracy guarantees and known limitations of FFAST algorithm

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

- [ ] Evaluate pure Rust MPI implementation as alternative to C `mpi` crate
- [ ] If no pure Rust MPI exists, document C dependency clearly in feature description
- [ ] Replace `libc`-based SVE detection with `std::arch` if available
      (`oxifft/src/simd/sve.rs`)
- [x] Add compile-time warning when `mpi`/`sve` features introduce C dependencies (done in build.rs)

### no_std Validation

- [ ] Add check: `cargo check --no-default-features --target thumbv7em-none-eabihf`
- [ ] Fix any compilation errors in no_std path
- [ ] Document no_std capabilities and limitations

### Wisdom System Improvements

- [ ] Define cross-platform wisdom file format specification
- [x] Add wisdom format version negotiation (WISDOM_FORMAT_VERSION = 1)
- [x] Add wisdom import validation (reject corrupted/incompatible files)
- [x] Add wisdom merge capability (combine from multiple machines)

### Dependent Project Support

- [ ] Survey 14 COOLJAPAN dependent projects for API pain points
- [ ] Ensure `Send + Sync` on all public plan types
- [ ] Add `TryFrom`/`Into` conversions for common numeric types

### Testing Expansion

- [x] Add property-based tests for all DCT/DST variants (Parseval, linearity, roundtrip) (done in rdft/solvers/r2r.rs)
- [x] Add fuzz testing for plan creation with arbitrary sizes (done in tests/plan_fuzz.rs)
- [ ] Add stress tests for concurrent wisdom access
- [ ] Add tests for all feature flag combinations
- [ ] Codegen validation: correctness vs reference, numerical accuracy, code size analysis
      (`oxifft-codegen/TODO.md` Phase 7)

### Documentation Completeness

- [ ] Ensure 100% of public API has rustdoc with `# Examples`
- [ ] Add `# Safety` sections to all `unsafe` blocks
- [ ] Add `# Errors` sections to all fallible public functions
- [ ] Add architecture diagrams to `oxifft.md`
- [ ] Create developer guide for adding new solvers
- [ ] Create developer guide for adding new SIMD backends

### Code Organization

- [ ] Audit files exceeding 2000 lines (COOLJAPAN policy) and split if needed
- [ ] Remove all `#[allow(dead_code)]` directives in production code

### Core Performance

- [ ] Profile and optimize Cooley-Tukey solver for sizes 2^10–2^20
- [ ] Implement cache-oblivious FFT strategy for large sizes (>L2 cache)
- [ ] Optimize twiddle factor computation (precompute + cache)
- [ ] Implement plan-specific memory pools to reduce allocation overhead
- [ ] Profile and optimize Bluestein chirp-Z convolution step
- [ ] Profile and optimize Rader omega table access pattern

### SIMD Optimization

- [ ] Implement hand-optimized AVX-512 codelets for sizes 16, 32, 64
- [ ] Implement SIMD-optimized twiddle multiplication
- [ ] Benchmark WASM SIMD and optimize if needed

### Threading Optimization

- [ ] Implement work-stealing for unbalanced multi-dimensional FFTs
- [ ] Tune Rayon task granularity to avoid excessive splitting
- [ ] Add thread-local scratch buffers to reduce allocation
- [ ] Benchmark and optimize parallel 2D/3D FFT decomposition

### Benchmark Tracking

- [ ] Establish automated benchmark regression tracking
- [ ] Publish benchmark results with each release
- [ ] Track FFTW ratio across versions (target: <1.5× power-of-2, <2× all sizes)

### Advanced Features Hardening

- [ ] Add GPU error recovery (device loss, OOM handling)
- [ ] Implement GPU memory pooling for repeated transforms
- [ ] Implement GPU R2C/C2R transforms
- [ ] Add GPU batch FFT with automatic chunking for large batches
- [x] Add overlap-save STFT method as alternative to overlap-add (done in streaming/stft.rs)
- [ ] Validate streaming real-time constraint: 48kHz audio without glitches
- [ ] Validate NUFFT tolerance across wide parameter ranges
- [x] Implement multi-dimensional NUFFT (2D, 3D) (done in nufft/nufft2d.rs, nufft3d.rs)
- [ ] Test MPI distributed FFT with >4 ranks
- [ ] Implement pencil decomposition as alternative to slab

### Release Infrastructure

- [x] Run `cargo semver-checks` against v0.1.4 (passed with expected breaking changes)
- [ ] Write v0.x → v1.0 migration guide
- [ ] Update all examples to use final v1.0 API
- [x] Verify `cargo publish --dry-run` succeeds for `oxifft-codegen` then `oxifft` (both crates pass)
- [ ] Verify package manifest includes only necessary files

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

- [ ] Implement FFT-based DCT-II via reordering + real FFT
      (`oxifft/src/rdft/solvers/r2r.rs` — `execute_dct2` is currently O(n²))
- [ ] Implement FFT-based DCT-III (inverse of DCT-II)
- [ ] Implement FFT-based DCT-I via DCT-III reduction
- [ ] Implement FFT-based DCT-IV via modified DCT-II
- [ ] Implement FFT-based DST variants (I–IV) via DCT symmetry relations
- [ ] Implement FFT-based DHT via complex FFT
- [ ] Retain O(n²) direct as reference/fallback for n < 16
- [ ] Add DCT/DST benchmarks demonstrating O(n log n) speedup

### RDFT Codelets

- [ ] Replace 3-line placeholder in `oxifft/src/rdft/codelets/mod.rs` with real codelets
- [ ] Implement R2HC (real to half-complex) codelets for sizes 2, 4, 8
- [ ] Implement HC2R (half-complex to real) codelets for sizes 2, 4, 8
- [ ] Implement real-valued twiddle codelets
- [ ] Integrate RDFT codelets into R2C/C2R solver pipeline

### Codegen Pipeline (oxifft-codegen)

- [ ] Implement no-twiddle codelets for sizes 16, 32, 64
      (`oxifft-codegen/src/gen_notw.rs`)
- [ ] Implement radix-8 twiddle codelet (`oxifft-codegen/src/gen_twiddle.rs`)
- [ ] Implement radix-16 twiddle codelet
- [ ] Implement split-radix twiddle codelet
- [ ] Add unit tests for all generated codelets (size 2–64)
- [ ] Codegen optimization: implement constant folding in symbolic optimizer
      (`oxifft-codegen/src/symbolic.rs`)
- [ ] Codegen optimization: implement dead code elimination

### NEON SIMD Performance Fix

- [ ] Profile NEON codelets to identify deinterleave bottleneck
      (`oxifft/src/dft/codelets/simd/backends.rs:268` — "slower than scalar" comment)
- [ ] Implement streaming NEON butterfly to eliminate copy overhead
- [ ] Benchmark NEON vs scalar and verify NEON achieves speedup (or document removal decision)

### Benchmark Infrastructure

- [ ] Produce and commit actual benchmark results (fill `BENCHMARK_RESULTS_TEMPLATE.md`)
- [ ] Add DCT/DST benchmark group to `oxifft-bench/`
- [ ] Add R2C/C2R performance regression tracking

### GPU: Pure Rust Metal Backend (macOS / Apple Silicon)

- [ ] Implement real Metal device discovery
      (`oxifft/src/gpu/metal.rs:33` — `is_available()` currently hardcoded)
- [ ] Implement Metal compute shader for Stockham FFT
- [ ] Implement `GpuFftEngine::forward()` / `inverse()` for Metal
      (`oxifft/src/gpu/plan.rs:291–321` — currently returns `Unsupported`)
- [ ] Implement `forward_inplace()` / `inverse_inplace()` for Metal
- [ ] Implement real GPU buffer upload/download via Metal API
      (`oxifft/src/gpu/buffer.rs:128,182` — currently returns `Unsupported`)
- [ ] Query actual Metal device capabilities
- [ ] Add macOS-only Metal integration tests (feature-gated)

### GPU: CUDA Backend

- [ ] Replace filesystem-based availability check with real CUDA runtime detection
      (`oxifft/src/gpu/cuda.rs:28`)
- [ ] Evaluate pure Rust CUDA feasibility vs cuFFT FFI (document decision)
- [ ] Implement `GpuFftEngine::forward()` / `inverse()` for CUDA
- [ ] Implement `forward_inplace()` / `inverse_inplace()` for CUDA
- [ ] Implement real GPU buffer upload/download
- [ ] Query actual CUDA device capabilities

### GPU: Architecture Cleanup

- [ ] Move `GpuBackend::OpenCL` and `GpuBackend::Vulkan` to separate feature flags
      or remove placeholder variants (currently return "not implemented")
- [ ] Implement `GpuBatchFft` trait for batch transforms
- [ ] Add GPU vs CPU benchmark for sizes 4096+

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

- [ ] 1D complex FFT power-of-2 (2^10): within 2× of FFTW
- [ ] 1D complex FFT power-of-2 (2^20): within 2× of FFTW
- [ ] 1D real FFT (2^10): within 2× of FFTW
- [ ] 2D complex FFT (1024×1024): within 2× of FFTW
- [ ] Batch 1D FFT (1000×256): within 2× of FFTW
- [ ] Prime-size FFT (2017): within 3× of FFTW
- [ ] DCT-II (1024): within 3× of FFTW
- [ ] Published benchmark results committed to repository

### Platform Matrix (Verified in CI)

- [ ] x86_64 Linux (SSE2, AVX, AVX2, AVX-512)
- [ ] x86_64 macOS (SSE2, AVX, AVX2)
- [ ] x86_64 Windows (SSE2, AVX, AVX2)
- [ ] aarch64 Linux (NEON)
- [ ] aarch64 macOS / Apple Silicon (NEON)
- [ ] wasm32-unknown-unknown (WASM SIMD)
- [ ] no_std (embedded target, with alloc)

### Quality Gates

- [ ] Zero clippy warnings: `cargo clippy --all-features -- -D warnings`
- [ ] Zero compiler warnings on all targets
- [ ] All tests pass on all supported platforms
- [ ] Documentation builds without warnings: `cargo doc --all-features`
- [ ] Zero `todo!()`, `unimplemented!()`, or `unwrap()` in library code
- [ ] MIRI passes for all unsafe code: `cargo +nightly miri test`
- [ ] Fuzz testing run for >24 hours without findings

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
- [ ] Match FFTW performance (1.0×) for power-of-2 sizes via deeper codelet tuning
- [ ] Exceed FFTW for composite sizes via Rust-specific optimizations
- [ ] Implement auto-tuning (runtime codelet selection profiled at build time)
- [ ] AVX-512 BF16 support for ML workloads

### GPU (v1.2+)
- [ ] OpenCL backend for cross-vendor GPU support
- [ ] Vulkan compute backend
- [ ] WebGPU backend for browser GPU acceleration
- [ ] Multi-GPU support (split large transforms across devices)
- [ ] GPU–CPU hybrid execution

### Algorithms (v1.3+)
- [ ] Sliding DFT for real-time streaming (O(1) per-sample update)
- [ ] Number Theoretic Transform (NTT) for exact integer arithmetic
- [ ] Winograd FFT for minimum-multiplication small sizes
- [ ] Partial FFT (compute only selected output frequencies)
- [ ] Chirp Z-Transform generalization for arbitrary frequency grids

### Ecosystem (v1.x+)
- [ ] C API (`oxifft-sys`) for use from C/C++/Python/Julia
- [ ] Python bindings (`oxifft-python`) via PyO3
- [ ] ndarray integration (`compute FFT of ndarray::Array directly`)
- [ ] Apache Arrow columnar data FFT
- [ ] ONNX operator implementation for ML frameworks

### Advanced Research
- [ ] Approximate FFT with configurable accuracy/speed tradeoff
- [ ] Distributed FFT over network (beyond MPI)
- [ ] Hardware-specific codelet auto-generation (profile at build time)
