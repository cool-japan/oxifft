# OxiFFT Project Status

## Current Status (v0.3.0)

| Metric | Value |
|--------|-------|
| Version | 0.3.0 |
| Tests Passing | 1116 |
| Clippy Warnings | 0 |
| Lines of Code | ~63,000 (Rust) |

## Feature Completion

- [x] Core FFT (Cooley-Tukey, Rader, Bluestein, Direct)
- [x] Real FFT (R2C / C2R)
- [x] DCT/DST (8 types)
- [x] Multi-dimensional (1D/2D/3D/ND)
- [x] Batch processing
- [x] SIMD (SSE2, AVX, AVX2, AVX-512, NEON, SVE, WASM)
- [x] Threading (Rayon)
- [x] Wisdom system
- [x] Sparse FFT (FFAST, O(k log n))
- [x] Pruned FFT (Goertzel)
- [x] Streaming FFT (STFT, mel, MFCC)
- [x] Compile-time FFT (const generics)
- [x] NUFFT (Type 1/2/3)
- [x] Fractional FFT
- [x] FFT-based convolution
- [x] Auto-differentiation
- [x] GPU acceleration (CUDA, Metal)
- [x] MPI distributed computing
- [x] WebAssembly
- [x] f16/f128 precision
- [x] Signal processing (Hilbert, Welch PSD, cepstrum, resampling)
- [x] Sliding DFT (SlidingDft, ModulatedSdft, SingleBinTracker)
- [x] Number Theoretic Transform (NTT)
- [x] Thread-local scratch buffers
- [x] SIMD codelet generation (SSE2/AVX2/NEON)
- [x] Real-valued twiddle codelets
- [x] Parallel configuration (ParallelConfig)
- [x] Cache-oblivious FFT strategy

## Upcoming (v0.3.0)

### Completed This Session
- Cache-oblivious FFT strategy (Frigo-Johnson 4-step in dft/solvers/cache_oblivious.rs)
- Thread-local scratch buffers (ThreadLocalScratch in support/scratch.rs)
- Parallel configuration (ParallelConfig in threading/parallel_config.rs)
- SIMD codelet generation: SSE2/AVX2/NEON f64 sizes 2/4/8 (gen_simd.rs)
- Real-valued twiddle codelets (real_twiddle.rs, sizes 4/8/16 + generic)
- Sliding DFT (SlidingDft, ModulatedSdft, SingleBinTracker in streaming/sdft.rs)
- Number Theoretic Transform (NTT module with 3 primes, polynomial convolution)
- Tests: 1003 → 1116 (+113 tests)

### Remaining
- DCT/DST benchmarks demonstrating O(n log n) speedup
- GPU: Pure Rust Metal and CUDA backends
- RDFT codelets integration into R2C/C2R pipeline
- Benchmark infrastructure and regression tracking

See [TODO.md](TODO.md) for the full roadmap.
