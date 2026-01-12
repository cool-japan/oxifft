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

## Current Status (Phase 1-4 Partial)

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
- 398 unit tests passing (main crate + rustfft comparison + wisdom tests + planning mode tests + size coverage tests + GuruPlan tests + split-complex tests + codegen tests + aligned memory tests)
- 28 FFTW comparison tests passing (oxifft-bench with fftw-compare feature)
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
