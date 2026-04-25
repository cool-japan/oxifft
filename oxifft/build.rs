//! `OxiFFT` build script.
//!
//! This build script emits compile-time warnings when features that introduce
//! C or Fortran dependencies are enabled, so downstream users are aware that
//! those features break the "Pure Rust" guarantee.

fn main() {
    // Detect features that pull in C/Fortran dependencies and warn the user.
    let mpi_enabled = std::env::var("CARGO_FEATURE_MPI").is_ok();
    let sve_enabled = std::env::var("CARGO_FEATURE_SVE").is_ok();

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
}
