//! `OxiFFT` build script.
//!
//! This build script emits compile-time warnings when features that introduce
//! C or Fortran dependencies are enabled, so downstream users are aware that
//! those features break the "Pure Rust" guarantee.
//!
//! It also writes `$OUT_DIR/wisdom_baseline.bin` so that the `include_bytes!`
//! macro in `types.rs` always succeeds, even when auto-tuning is disabled.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    // Detect features that pull in C/Fortran dependencies and warn the user.
    let mpi_enabled = env::var("CARGO_FEATURE_MPI").is_ok();
    let sve_enabled = env::var("CARGO_FEATURE_SVE").is_ok();

    if mpi_enabled {
        println!(
            "cargo:warning=\
oxifft: the `mpi` feature links against the system MPI library (C/Fortran), \
which violates the Pure Rust policy for default builds. \
This feature is provided for distributed computing and is explicitly \
feature-gated. No pure-Rust MPI implementation currently exists. \
See https://github.com/cool-japan/oxifft/blob/master/README.md#mpi for details."
        );
    }

    // sve feature now uses std::arch::is_aarch64_feature_detected! — no C dep.
    let _ = sve_enabled;

    // ── Auto-tuning environment variables ──────────────────────────────────────

    // Rerun the build script when these env vars change.
    println!("cargo:rerun-if-env-changed=OXIFFT_TUNE");
    println!("cargo:rerun-if-env-changed=OXIFFT_SKIP_TUNE");

    // ── Wisdom baseline file ───────────────────────────────────────────────────
    //
    // `include_bytes!(concat!(env!("OUT_DIR"), "/wisdom_baseline.bin"))` in
    // `types.rs` requires the file to exist at compile time.  We always write
    // it here — empty when tuning is disabled, which causes the runtime to fall
    // back to heuristics.
    //
    // When OXIFFT_TUNE=1 and OXIFFT_SKIP_TUNE≠1 and we are not cross-compiling,
    // a future implementation could run a sub-process tuner and write real data.
    // For now we unconditionally write an empty sentinel in all paths.

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set by Cargo"));
    let baseline_path = out_dir.join("wisdom_baseline.bin");

    // Always write an empty baseline — the runtime falls back to heuristics
    // when the baseline is empty.  The `oxifft_tune` binary can produce a
    // populated baseline file for embedding in a future build-integration step.
    fs::write(&baseline_path, []).expect("failed to write wisdom_baseline.bin in OUT_DIR");
}
