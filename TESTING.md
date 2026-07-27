# OxiFFT Testing Guide

## Running Tests

```bash
# Run all tests with all features (workspace default members)
cargo nextest run --all-features

# Run only the core library tests
cargo nextest run -p oxifft --all-features

# Run a specific test
cargo nextest run --all-features -- test_name

# Run the documentation examples (every rustdoc example compiles + runs)
cargo test --doc --all-features
```

As of v0.4.0 this yields **1,692 tests passing** (all features), **6 `#[ignore]`d**
tests (see below), and **107 doctests, 0 ignored**.

## Test Categories

### Unit Tests

Each module contains inline unit tests (`#[cfg(test)]` blocks):

- **Correctness**: Compare against direct O(n²) solver for small sizes
- **Roundtrip**: `IFFT(FFT(x)) ≈ x` within floating-point tolerance
- **Parseval**: Energy conservation across transforms
- **Linearity**: `FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)`

### Integration Tests

Located in `oxifft/tests/`:

- `size_coverage.rs` — correctness for powers of 2, primes, composites, edge cases
- `fftw_comparison.rs` — compare against FFTW3 (requires the `fftw-compat`/`fftw` build)
- plus the v0.4.0 regression suites (odd-N real FFT, cross-rank c2r normalization,
  split-plan normalization, flag threading, etc.)

### Feature-Gated Tests

Some tests are skipped unless the relevant feature is enabled:

| Feature | Tests |
|---------|-------|
| `sparse` | Sparse FFT correctness |
| `streaming` | STFT roundtrip, mel filterbank |
| `pruned` | Goertzel, input/output pruning |
| `const-fft` | Compile-time FFT correctness |
| `signal` | Hilbert transform, Welch PSD |
| `metal` / `cuda` | GPU backend round-trips (Metal requires Apple GPU hardware) |

## `#[ignore]`d Tests

Six tests are marked `#[ignore]` because they are slow stress tests or
timing-sensitive smoke tests that are flaky on shared/low-core machines. Run
them explicitly:

```bash
# Run everything, including ignored tests
cargo nextest run --all-features --run-ignored all

# Or with the built-in test harness (shows --nocapture timing output)
cargo test --all-features -- --ignored --nocapture
```

| Test | Location | Why ignored |
|------|----------|-------------|
| large power-of-2 stress (2048–16384) | `oxifft/tests/size_coverage.rs` | slow |
| additional slow stress sweeps (×3) | `oxifft/tests/size_coverage.rs` | slow |
| pruned-vs-full timing benchmark | `oxifft/src/pruned/output_pruned.rs` | timing-sensitive |
| planner scaling smoke test | `oxifft/src/api/plan/types.rs` | timing-sensitive |

## MPI Tests (mpirun-based)

The distributed FFT code lives in the separate `oxifft-adapter-mpi` crate. Its
correctness checks are an **example binary**, not `#[test]` functions, on
purpose: the Rust test harness runs test bodies on spawned threads, which
deadlocks MPI collective calls. Build the example and launch it under `mpirun`:

```bash
# Requires a system MPI library + libclang (bindgen). Off the default member set.
cargo build -p oxifft-adapter-mpi --example mpi_integration

mpirun -n 1 target/debug/examples/mpi_integration
mpirun -n 2 target/debug/examples/mpi_integration
mpirun -n 4 target/debug/examples/mpi_integration
```

Plain `cargo build`/`cargo test` at the workspace root excludes
`oxifft-adapter-mpi` (it is not in `default-members`); use `-p
oxifft-adapter-mpi` or `--workspace` to include it. See
`oxifft-adapter-mpi/README.md` for the full prerequisites and workflow.

## Validation Strategy

1. **Direct solver validation**: For sizes ≤ 64, compare against O(n²) DFT
2. **Cross-validation**: Compare multiple algorithms for the same size
3. **Property tests**: Parseval, linearity, inverse, symmetry (via `proptest`)
4. **SIMD validation**: Each SIMD backend is validated against the scalar
   reference on the host running the tests (runtime feature detection selects
   the widest available backend). There is no CI host — validation is whatever
   your local machine supports; run on representative hardware before a release.

## Tolerance

Floating-point comparisons use these tolerances:

| Precision | Tolerance |
|-----------|-----------|
| f32 | 1e-5 |
| f64 | 1e-10 |
| f128 | 1e-15 |

## Local Verification Gates (no CI)

This repository has **no CI workflows** — policy permits only publish workflows
(`pypi-publish.yml` / `npm-publish.yml`), and none run tests. Platform/SIMD
coverage is therefore **manual and local**, not automated. Before a release, run
the following locally on each platform you can access:

```bash
cargo fmt --all -- --check
cargo clippy --all-features --workspace -- -D warnings
cargo nextest run --all-features
cargo test --doc --all-features
cargo deny check bans          # banned-crate policy
```

Platforms exercised manually when hardware is available: x86_64 Linux (AVX2),
aarch64 macOS (NEON), x86_64 Windows (SSE2), wasm32, and `no_std` builds. None
of these are gated by CI today; treat the matrix as "verified where run", not
"verified in CI".
