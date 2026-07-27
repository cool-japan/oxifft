# oxifft-bench

Benchmarks and FFTW comparison tests for [OxiFFT](https://github.com/cool-japan/oxifft).

This crate is internal to the OxiFFT project and is not published to crates.io.

## Usage

```bash
# Run benchmarks
cargo bench -p oxifft-bench

# Run FFTW comparison tests (FFTW built from bundled source — reproducible)
cargo test -p oxifft-bench --features fftw-compare
```

## FFTW comparison: bundled vs system

FFTW support is feature-gated and off by default (Pure-Rust default builds).
Two mutually complementary ways to provide FFTW are available:

| Feature        | FFTW provenance                                             | Use when                                              |
|----------------|------------------------------------------------------------|-------------------------------------------------------|
| `fftw-compare` | Built from the `fftw` crate's bundled `fftw-src` source     | Default. Reproducible regardless of the host.          |
| `fftw-system`  | Links the pre-installed system `libfftw3` (via pkg-config)  | Attributable numbers against a known FFTW build.       |

```bash
# Reproducible bundled build (default)
cargo bench -p oxifft-bench --features fftw-compare --bench fftw_parity_gates

# Link the system libfftw3 (e.g. a Homebrew `brew install fftw` build)
cargo bench -p oxifft-bench --features fftw-system  --bench fftw_parity_gates
```

`fftw-system` implies `fftw-compare` and additionally activates the `fftw`
crate's `system` backend. For `system` to fully replace the bundled `source`
build, the workspace-level `fftw` dependency should be declared with
`default-features = false` (see the note in `oxifft-bench/Cargo.toml`).

The `fftw_ratio_report` tool records which FFTW was used (source vs system) and
its version in every snapshot, so comparison numbers are attributable.

## FFTW ratio report

`fftw_ratio_report` post-processes the criterion output of the parity benches
into a dated ratio snapshot. It honours `CARGO_TARGET_DIR`, exits non-zero when
any gate's criterion estimates are missing, and writes snapshots to an
**untracked** directory by default:

```bash
# 1. Run the parity benches so criterion writes estimates
cargo bench -p oxifft-bench --features fftw-compare --bench fftw_parity_gates -- --save-baseline current

# 2. Summarise — snapshot lands in <target>/fftw-ratio-reports/ (untracked)
cargo run -p oxifft-bench --features fftw-compare --bin fftw_ratio_report

# Promote a completed run into the tracked benches/baselines/<version>/ history
cargo run -p oxifft-bench --features fftw-compare --bin fftw_ratio_report -- --commit-baseline
```

Options: `--criterion-dir <DIR>`, `--out <DIR>`, `--commit-baseline`, `--help`.
Committed baselines under `benches/baselines/` only change via
`--commit-baseline`, and that refuses to promote a run with any missing gate.

## License

Apache-2.0 — Copyright COOLJAPAN OU (Team Kitasan)
