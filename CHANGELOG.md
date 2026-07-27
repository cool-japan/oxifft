# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_No unreleased changes._

## [0.4.1] - 2026-07-27

### Changed

- `oxicuda-driver` / `oxicuda-fft` / `oxicuda-metal` (optional GPU backends)
  updated from 0.5.2 to 0.5.3.

## [0.4.0] - 2026-07-27

### Breaking Changes

- **`oxifft::kernel::{Solver, ProblemHash, hash_problem}` removed.** `Solver`
  had zero implementors and `ProblemHash`/`hash_problem` were dead,
  misleadingly-commented code; `kernel::planner::Planner` /
  `kernel::wisdom::WisdomEntry` are the real, wired-in candidate model.
- **`auto_tune::tune_size` gained a `flags: Flags` parameter.** Callers must
  now pass planning flags; `MEASURE`/`PATIENT`/`EXHAUSTIVE` widen the
  candidate set that gets profiled (radix-4/8, split-radix, Stockham,
  cache-oblivious, generic, Rader, mixed-radix, …) instead of only timing the
  single heuristic-chosen algorithm.
- **`Flags::WISDOM_ONLY` added**, matching FFTW's `FFTW_WISDOM_ONLY`: plan
  construction now fails (`None`) when no matching wisdom exists instead of
  silently falling back to a fresh heuristic/measured plan.
- **Wisdom binary format v2** (`BINARY_FORMAT_VERSION` bumped to 2): the v1
  encoding used a fixed 30-byte entry layout and a `u16`-bounded factor count,
  both of which could silently truncate `MixedRadix` solver names with many
  factors. v2 uses variable-length factor lists with an explicit `u32` entry
  count. `from_binary` still reads v1 files; `to_binary`/`to_binary_checked`
  now always write v2.
- **Workspace `rust-version` corrected from `1.75` to `1.87`.** `1.75` was
  aspirational, not actual: `hashbrown` 0.17 (a default-feature, non-optional
  dependency) requires edition2024/rust-version 1.85 at the dependency-
  resolution level alone, and `oxifft`'s own `ntt`/`nufft` modules use
  `u32::is_multiple_of`, stabilized in 1.87. Verified empirically with
  `cargo +<toolchain> check -p oxifft --locked` across rustup toolchains
  1.75.0, 1.77.1, 1.82.0, 1.85.0, 1.86.0 (all fail) and 1.87.0, 1.88.0 (both
  succeed).

### Added

- **`oxifft-adapter-mpi`: distributed real transforms (`MpiRealPlan2D`,
  `MpiRealPlan3D`).** Slab-decomposed r2c/c2r in the FFTW half-complex layout
  (a real last dimension of size `n` is stored as `n/2 + 1` complex
  coefficients; odd last dims supported). c2r is normalized by
  `1 / product(dims)`, consistent with the core `RealPlan*::execute_c2r`
  (deliberately unlike FFTW, which leaves c2r unnormalized), so an r2c -> c2r
  round trip is the identity. r2c honours `MpiFlags::transposed_out`. Added
  `local_size_2d_r2c` / `local_size_3d_r2c` allocation-size helpers.

### Changed

- **`oxifft::Plan::dft_1d` is now flag-aware and wisdom-cached.** `ESTIMATE`
  keeps the pure heuristic; `MEASURE`/`PATIENT`/`EXHAUSTIVE` consult the
  runtime wisdom cache (`lookup_wisdom`) first, and on a miss, benchmark real
  candidate algorithms and cache the winner (`store_wisdom`) so repeated
  `MEASURE` planning of the same size no longer re-benchmarks on every call.
- Added `Algorithm::Rader` and `Algorithm::CacheOblivious`;
  `algorithm_from_solver_name` now reconstructs all stateful solvers so
  imported/measured wisdom actually feeds back into plan selection. Rader is
  now selected by the `ESTIMATE` heuristic for primes in `17..=1021`.
- `oxifft-bench`'s `rustfft` dependency is now optional and feature-gated
  (`rustfft-compare`, on by default — mirrors the existing, already-optional
  `fftw-compare`/`fftw` pairing) instead of an unconditional dependency. The
  default `cargo build`/`cargo bench -p oxifft-bench` experience is
  unchanged. `oxifft-bench` remains `publish = false`.
- `oxifft-codegen-impl`'s crates.io category corrected from the copy-pasted
  `development-tools::procedural-macro-helpers` (it is not a proc-macro
  crate) to `["development-tools", "algorithms"]`.
- **`oxifft-adapter-mpi`: `MpiPlanND::distributed_fft_dim0` now uses a single
  `alltoallv`-based transpose** (was one blocking `all_gather_v` per fiber):
  two collectives total regardless of fiber count, each rank holds only its
  `1/P` share of the transposed data instead of gathering full fibers.
  Validated under `mpirun -n 1/2/3/4`.

### Dependencies

- `oxicuda-driver` / `oxicuda-fft` / `oxicuda-metal` (optional GPU backends)
  bumped from 0.1.8 → 0.5.1 across five incremental updates. CUDA execution
  remains an honest CPU-fallback (`GpuFft::execution_target` returns
  `ExecutionTarget::Cpu` for the CUDA backend; Metal returns `Gpu`) pending
  device-side FFT support landing in `oxicuda-fft` itself — the bump tracks
  upstream API changes only, no behavior change on the `oxifft` side.

### Fixed / Dependency & Packaging Hygiene

- **Added `deny.toml`** at the workspace root enforcing the COOLJAPAN
  banned-crate policy (openblas, bincode, rustfft, Z3, rusqlite, zip, flate2,
  zstd, bzip2, lz4, tar, snap, brotli, miniz_oxide) via `cargo deny check
  bans`, with narrow, documented `wrappers` exceptions scoped only to
  `oxifft-bench` (`publish = false`) and its comparison-benchmark
  dependencies (`rustfft`, and the `fftw` → `fftw-sys` → `fftw-src` → `zip` →
  {`flate2` → `miniz_oxide`, `zstd`, `bzip2`} chain behind the non-default
  `fftw-compare` feature). `cargo deny check` (bans + licenses + advisories +
  sources) passes clean on this workspace as of this release.
- Fixed workspace dependency-inheritance violations: `oxifft/Cargo.toml`'s
  stale `oxifft-codegen = "0.3.1"` local pin, `oxifft-adapter-mpi/Cargo.toml`'s
  local pin on `oxifft`, and `oxifft-codegen/Cargo.toml`'s local `trybuild`
  dev-dependency pin now all route through `[workspace.dependencies]` /
  `.workspace = true`, matching the existing `oxifft-codegen-impl` pattern.
- Removed the dead `seahash` dependency (its only consumer, `kernel/hash.rs`,
  was deleted as part of the planner/wisdom rework above) and the dead
  workspace `libc` dependency (declared "for SVE detection" but never
  referenced anywhere; SVE detection uses
  `std::arch::is_aarch64_feature_detected!` instead).
- Added `[package.metadata.docs.rs]` to all four publishable crates
  (`oxifft`, `oxifft-codegen`, `oxifft-codegen-impl`, `oxifft-adapter-mpi`).
  `oxifft`'s docs.rs build now uses a curated feature list (`std`,
  `threading`, `avx512`, `f128-support`, `f16-support`, `sparse`, `pruned`,
  `sve`, `wasm`, `streaming`, `const-fft`, `signal`, `fftw-compat`, `ndarray`)
  rather than `all-features`, since `cuda`/`metal`/`gpu` pull in
  `oxicuda-metal`'s `metal` crate dependency, which does not build on
  docs.rs's Linux container.
- Added a real `README.md` for `oxifft-adapter-mpi` (previously shipped with
  no readme at all) documenting what the crate is, its MPI/libclang build
  prerequisites, a quickstart, and the `mpirun` integration-check workflow;
  added `readme = "README.md"` to its `Cargo.toml`.
- Every publishable crate (`oxifft`, `oxifft-codegen`, `oxifft-codegen-impl`,
  `oxifft-adapter-mpi`) now ships its own copy of `LICENSE` in its package
  tarball (previously only the workspace-root `LICENSE` existed, which is not
  auto-included in any individual crate's `cargo package` output).
- Excluded the prebuilt `pkg_simd/` wasm-pack artifacts (`.wasm`/`.js`/
  `.d.ts`, still tracked in git from before a `.gitignore` rule was added for
  them) from the `oxifft` crate's crates.io tarball via `Cargo.toml`'s
  `exclude` list — `.gitignore` cannot retroactively untrack already-committed
  files, so this is the mechanism that actually keeps them out of `cargo
  package`.
- Added `default-members = ["oxifft", "oxifft-codegen", "oxifft-codegen-impl",
  "oxifft-bench"]` to the workspace so a plain `cargo build`/`cargo test` at
  the repository root no longer requires a system MPI implementation or
  `libclang`; `oxifft-adapter-mpi` remains fully buildable via
  `cargo build --workspace` or `cargo build -p oxifft-adapter-mpi`.
- **`no_std` SIMD compilation fixed across both the generated and hand-written
  AVX-512 paths.** The codegen-emitted dispatchers (`gen_simd_codelet!`,
  `gen_dispatcher_codelet!`) are now fully `no_std`-safe: x86 feature detection
  goes through a self-contained, block-local `macro_rules!` that expands to
  `is_x86_feature_detected!` under `std` and to `cfg!(target_feature = ...)`
  under `no_std` (matched as `:tt`, not `:literal`, so `std_detect` sees the
  raw feature literal — a `:literal` fragment tripped a spurious "unknown x86
  target feature" error under `--features avx512,std`); the AVX-512 f32 size-16
  twiddles are precomputed into literal constants at proc-macro expansion time
  rather than emitting runtime `.sin()`/`.cos()` calls; and no `::std::` paths
  are emitted (aarch64 NEON is assumed, matching the uncached dispatcher). With
  the emitted code self-contained, the previously `std`-gated cached size-4
  dispatchers in `generated_simd.rs` are now available under `no_std` too. The
  hand-written AVX-512 codelets were fixed the same way: the dispatch wrappers
  route `avx512f` detection through the crate's `detect_x86_feature!` macro
  instead of the `std`-only `is_x86_feature_detected!`, and the precomputed
  twiddle-table builders resolve `sin_cos` via `num_traits::Float` under
  `no_std`. `cargo check -p oxifft --no-default-features --features avx512
  --target x86_64-unknown-linux-gnu` and the `--features avx512,std` variant
  both build clean.
- Fixed two remaining lint warnings found in final full-workspace
  verification: a `clippy::manual_midpoint` hit in
  `const_fft::radix2::ifft2_inplace` (now uses `f64::midpoint`, stabilized in
  the crate's MSRV 1.87) and a `rustdoc::private_intra_doc_links` warning in
  `oxifft-codegen-impl/src/gen_simd/mod.rs` (doc comment referenced the
  private `gen_x86_detect_macro` helper via `[...]` link syntax, which cannot
  resolve to a private item). `cargo clippy --workspace --all-targets
  --all-features` and `cargo doc --workspace --all-features --no-deps` are
  both zero-warning as of this release. Full gate re-verification: 1730
  tests passing workspace-wide, doctests passing, `cargo deny check` clean,
  `mpirun -n 2` / `-n 4` MPI integration tests passing.
- **Fixed release-build-only test flakiness in `kernel::twiddle`'s SoA-vs-AoS
  parity tests** (`soa_vs_aos_correctness_f64_{1024,4096,16384,65536}`):
  these passed under `cargo nextest run` (debug) but failed under `--release`,
  up to 2048 ULP apart at specific (size, index) pairs. Root cause: the AoS
  and SoA twiddle-table builders (`compute_twiddle_table_f64` vs
  `compute_twiddle_table_soa_f64`) use the identical scalar formula but as
  structurally different loops (iterator `.map().collect()` vs manual `for`
  + `.push()`), so under `-O` LLVM's auto-vectorizer is free to choose
  different, individually-valid instruction sequences for the division
  feeding `cos`/`sin` in each — a sub-ULP input perturbation that can
  occasionally amplify to a few thousand ULP in the trig output for specific
  inputs, without either path being wrong. A raw ULP bound is the wrong tool
  for comparing two independently-vectorized-and-compiled code paths for
  exactly this reason; the tests (f64 and f32 variants) now use an
  absolute/relative-error gate (`1e-11`/`1e-9` for f64, `1e-6`/`1e-5` for
  f32) that comfortably clears the observed noise (worst case ~4.5e-13
  relative) while staying far tighter than any real algorithmic bug would
  produce. No production code changed — this was a test-tolerance defect
  only, caught by adding a `cargo nextest run --release` pass (not
  previously part of this project's verification gates) to `/final-call`.

### Known Issues (tracked, not fixed this release)

- `syn` stays pinned to the 2.0 major workspace-wide (crates.io's current
  major is 3.0) because migrating `oxifft-codegen`/`oxifft-codegen-impl`'s
  proc-macro internals to syn 3.0 is a real source migration, not a drop-in
  bump. `serde_derive` already pulls in syn 3.0 transitively, so both majors
  compile in every build regardless; `cargo deny`'s `multiple-versions`
  duplicate check is informed of this via a documented `skip` entry.
- `oxifft-bench --no-default-features` does not build:
  `utils::rustfft_forward`/`utils::rustfft_inverse` and the `fft_comparison`
  bench still reference `rustfft` unconditionally in source. The
  `rustfft-compare` feature (on by default) makes the dependency itself
  optional at the `Cargo.toml` level, but the call sites in
  `oxifft-bench/src/utils.rs` need `#[cfg(feature = "rustfft-compare")]`
  guards to fully honor `--no-default-features`; default-feature builds are
  unaffected.
- `oxifft-adapter-mpi`'s docs.rs build is expected to fail regardless of its
  `[package.metadata.docs.rs]` stanza: its single, non-optional `mpi`
  dependency requires a real MPI implementation and `libclang` at build time,
  neither available on the docs.rs container, and there is no Cargo-level way
  to make a non-optional dependency skip that requirement.

## [0.3.2] - 2026-05-22

### Added

- **Multi-rank 3D pencil FFT execution**: `plan_3d_pencil` now supports multi-rank MPI
  execution with full forward/inverse pencil decomposition. Expanded `plan_nd` error
  handling for ND FFT plans.

### Changed

- **AVX-512 codelets and dispatchers gated behind a new default-off `avx512` feature**:
  `rustc` 1.95 stable treats `#[target_feature(enable = "avx512*")]` as unstable
  (rust-lang/rust#44839). Without `--features avx512`, builds fall through to the
  existing AVX-2 / SSE / scalar dispatch paths. No API or ABI change when the feature
  is enabled. Both `oxifft` and `oxifft-codegen-impl` are updated in concert.
- **Refactored error handling for row and column pools** in multi-rank 3D pencil FFT
  execution (`plan_3d_pencil`, `plan_nd`).

### Dependencies

- `oxicuda` (optional GPU backend) bumped from 0.1.4 → 0.1.8 across four incremental
  updates (0.1.5, 0.1.6, 0.1.7, 0.1.8).

## [0.3.1] - 2026-05-02

### Added
- **Mixed-radix Cooley-Tukey FFT** for smooth-7 sizes factoring into {2, 3, 4, 5, 7, 8, 16}
  (6, 10, 12, 14, 24, 28, 40, 56, 80, 96, 112, 240, …). Uses Winograd minimum-multiply
  radix-3/5/7 DIT butterflies; replaces Bluestein for these sizes with a proper cost model.
- **Auto-tuning** (`Flags::MEASURE` / `Flags::PATIENT`): runtime profiling of candidate
  algorithms via `auto_tune::tune_size<T>` / `tune_range<T>`. Binary wisdom format
  (30-byte packed LE entries). Build-time profiling opt-in via `OXIFFT_TUNE=1`.
  New `oxifft_tune` CLI binary for offline profiling.
- **`gen_any_codelet!` proc-macro** and `CodeletBuilder` API in `oxifft-codegen`:
  dispatches to direct codelets / Rader / MixedRadix / Bluestein for any user-specified N.
- **Wisdom format v2**: S-expression format with `(mixed-radix-R1-R2-...)` encoding;
  backward-compatible with v1 files.

### Changed
- 325 new tests; total workspace: **1 554 passing** (up from 1 229 in v0.3.0).
- `oxifft-codegen` now has 11 proc-macros (added `gen_any_codelet!`).

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

- **Total Lines of Code**: 78,065 (Rust code only, `tokei`)
- **Rust Files**: 300
- **Test Coverage**: 1,730 tests + doc tests passing (0 failed, 7 skipped)
- **Zero Warnings**: Clippy + rustdoc clean (all features, workspace-wide)
- **Documentation**: 7,737 comment lines (incl. doc comments)

## Supported Platforms

- **x86_64**: Linux, macOS, Windows (SSE2, AVX, AVX2, AVX-512 behind the
  default-off `avx512` feature — see [0.3.2](#032---2026-05-22))
- **aarch64**: Linux, macOS (NEON, SVE with feature flag)
- **wasm32**: Browser and Node.js (WASM SIMD)

## Dependencies

### Required
- num-complex 0.4
- num-traits 0.2
- serde 1.0
- serde_json 1.0

### Optional
- rayon 1.12 (threading)
- mpi 0.8 (MPI distributed computing, `oxifft-adapter-mpi`)
- oxicuda-driver / oxicuda-fft / oxicuda-metal 0.5 (GPU backends — Metal is
  real GPU execution, CUDA is an honest CPU-fallback pending upstream
  device-side FFT support)
- wasm-bindgen 0.2 (WebAssembly bindings)
- js-sys 0.3 (JavaScript interop)

[Unreleased]: https://github.com/cool-japan/oxifft/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/cool-japan/oxifft/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/cool-japan/oxifft/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/cool-japan/oxifft/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/cool-japan/oxifft/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/cool-japan/oxifft/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/cool-japan/oxifft/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/cool-japan/oxifft/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/cool-japan/oxifft/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/cool-japan/oxifft/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/cool-japan/oxifft/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/cool-japan/oxifft/releases/tag/v0.1.0
