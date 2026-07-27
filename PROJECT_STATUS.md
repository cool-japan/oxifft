# OxiFFT Project Status

## Current Status (v0.4.0)

| Metric | Value |
|--------|-------|
| Version | 0.4.0 |
| Tests passing | 1,692 (workspace, all features) + 107 doctests (0 ignored) |
| `#[ignore]`d tests | 6 (slow stress / timing-sensitive; run with `--ignored`) |
| Clippy warnings | 0 (all features) |
| Lines of code | 76,457 Rust (5 crates) |
| MSRV | 1.87 (empirically verified) |
| Performance vs FFTW | geomean 1.50×, 4/7 v1.0 gates passing (see below) |

Verified locally (no CI is configured; repository policy allows only publish
workflows). See [TESTING.md](TESTING.md) for the local verification gates.

## Workspace crates (5)

| Crate | Purpose | Published |
|-------|---------|-----------|
| `oxifft` | Core pure-Rust FFT library | yes |
| `oxifft-codegen` | Proc-macro façade (re-exports codelet macros) | yes |
| `oxifft-codegen-impl` | Codelet-generation implementation | yes |
| `oxifft-adapter-mpi` | Distributed MPI FFT (system MPI FFI, quarantined) | yes |
| `oxifft-bench` | RustFFT/FFTW comparison benches + `fftw_ratio_report` | no (`publish = false`) |

## Feature Completion

- [x] Core FFT (Cooley-Tukey radix-2/4/8, split-radix, mixed-radix, Stockham, cache-oblivious, Rader, Bluestein, Direct)
- [x] Real FFT (R2C / C2R) — 1D/2D/3D/ND, normalization unified across ranks
- [x] DCT/DST (8 types) + DHT; `R2rPlan`, `R2rPlan2D`, `R2rPlan3D`
- [x] Multi-dimensional (1D/2D/3D/ND), including multi-rank 3D pencil FFT
- [x] Batch processing (Guru interface, c2c DFT)
- [x] SIMD (SSE2, AVX, AVX2, AVX-512, NEON, SVE, WASM) — currently small codelets + pointwise multiplies (see performance note)
- [x] Threading (Rayon), `ParallelConfig` size-aware thresholds
- [x] Wisdom system — runtime cache consulted by `MEASURE`/`PATIENT`/`EXHAUSTIVE`/`WISDOM_ONLY`; binary format v2
- [x] Auto-tuning — genuine multi-candidate comparison at plan time, winner cached
- [x] Sparse FFT (FFAST, O(k log n))
- [x] Pruned FFT (Goertzel, input/output pruning)
- [x] Streaming FFT (STFT, mel, MFCC, sliding DFT)
- [x] Compile-time FFT (const generics)
- [x] NUFFT (Type 1/2/3)
- [x] Fractional FFT, Chirp-Z transform
- [x] FFT-based convolution, Number Theoretic Transform (NTT)
- [x] Auto-differentiation (forward + reverse mode)
- [~] GPU acceleration — **Metal runs on the GPU**; **CUDA currently evaluates on the CPU** (real kernel dispatch pending in `oxicuda-fft`). Status is queryable via `GpuFft::execution_target()` / `GpuCapabilities::hardware_accelerated`
- [x] MPI distributed computing — in the separate `oxifft-adapter-mpi` crate
- [x] WebAssembly (WASM SIMD)
- [x] f16/f128 precision (`Plan<T>` generic over the `Float` trait)
- [x] Signal processing (Hilbert, Welch PSD, cepstrum, resampling)
- [x] FFTW-compatible surface (`compat`: `fftw_*` names incl. `fftw_plan_r2r_1d/2d/3d`)

## v0.4.0 Sprint Outcomes (Production Readiness)

**Planner / wisdom**
- `Plan::dft_1d` now honors `ESTIMATE` (heuristic) vs `MEASURE`/`PATIENT`/`EXHAUSTIVE` (consult runtime wisdom via `lookup_wisdom`, else benchmark real candidates and cache the winner). Repeated `MEASURE` constructions hit the cache instead of re-benchmarking.
- `auto_tune::tune_size` gained a `Flags` parameter and genuinely compares candidate algorithms (Rader, split-radix, radix-4/8, Stockham, cache-oblivious, generic, mixed-radix).
- Added `Flags::WISDOM_ONLY` (FFTW semantics: fails when no wisdom exists).
- Real transforms (`RealPlan*`, `R2rPlan*`) and multi-dimensional column FFTs now thread real `flags` into their internal sub-plans, so `MEASURE`/`PATIENT`/wisdom influence them too.
- Removed dead `kernel::{Solver, ProblemHash, hash_problem}` (**breaking**).

**Wisdom format**
- Binary format **v2**: variable-length mixed-radix factor lists (fixes >6-factor truncation, e.g. n=2187=3^7) and a `u32` entry count (fixes >65535-entry drop). v1 still read for backward compatibility.
- `from_binary` now returns `Result` with distinct error causes; poisoned `RwLock` recovers instead of panicking; import size limits enforced.

**Correctness fixes (with regression tests)**
- Metal high-level inverse FFT was double-normalized (1/N twice) — fixed.
- Odd-N real FFT round-trip fixed.
- `RealPlan` c2r normalization reconciled across 1D/2D/3D/ND.
- `ifft2d_split` returned `N*x` — fixed; all `SplitPlan*::execute` are now unnormalized with normalization in the `ifft*_split` wrappers (**breaking**).
- `oxifft::R2rKind` is now the FFTW-named `REDFT`/`RODFT`/`DHT` enum (**breaking**).

**Packaging / docs**
- Version bump to 0.4.0; workspace-inherited internal deps; `deny.toml` (`cargo deny check bans` clean); `docs.rs` metadata; MSRV 1.87 documented.
- Every previously `ignore`-fenced rustdoc example (32) now compiles/runs under `cargo test --doc`; README, PROJECT_STATUS, TESTING, TODO, BENCHMARKING and `docs/wisdom_format.md` brought in line with the shipping code.

## Known Performance Gaps vs FFTW

The committed baseline (`benches/baselines/v0.3.0/fftw_ratios_2026-04-20.json`,
Apple Silicon, mid-development) records geomean **1.50×** with **4/7** v1.0
gates passing. Failing gates: `1d_cplx_2e20` (3.67×), `1d_real_2e10` (3.95×),
`dct2_1024` (3.90×). OxiFFT also currently trails RustFFT on large power-of-2,
prime, and several composite sizes — root cause: SIMD covers only small codelets
and pointwise multiplies, not the general large-N Stockham path. Refresh the
baseline with `fftw_ratio_report` before any v1.0 performance announcement.

## Remaining Toward v1.0

- Real GPU kernel dispatch for CUDA (`oxicuda-fft` integration); wire the GPU buffer pool into the execution path.
- Extend SIMD into the general Stockham butterfly path (closes the large-N gaps above).
- `no_std` (`--no-default-features`) core build (codegen `gen_simd` still emits std-only paths).
- Guru r2c/c2r/split/r2r constructors; broaden `fftw_execute_*` coverage.
- `syn` 3.0 migration for the codegen crates.

See [TODO.md](TODO.md) for the full roadmap and [CHANGELOG.md](CHANGELOG.md) for release notes.
