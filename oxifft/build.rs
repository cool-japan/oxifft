//! `OxiFFT` build script.
//!
//! This build script emits compile-time warnings when features that introduce
//! C or Fortran dependencies are enabled, so downstream users are aware that
//! those features break the "Pure Rust" guarantee.
//!
//! It also writes `$OUT_DIR/wisdom_baseline.bin`, which the `include_bytes!`
//! macro in `types.rs` embeds and the runtime consults before heuristic
//! planning. By default this file is an empty sentinel (the runtime then falls
//! back to heuristics). When `OXIFFT_TUNE=1` is set for a native (non
//! cross-compiled) build, a real, non-empty baseline is embedded instead.

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

// ── Binary wisdom format (frozen V1; `api::wisdom::from_binary` still reads it)
//
// A build script cannot link the crate under construction, so it hand-encodes
// the wisdom blob. We deliberately target the stable *version 1* layout, which
// `WisdomCache::from_binary` supports permanently for backward compatibility,
// rather than a newer format that may still evolve.
//
// Header (16 bytes): magic (8) | format_version u16 LE = 1 | entry_count u16 LE |
//                    reserved u32 LE.
// Each entry (30 bytes): size_key u64 LE | algo_tag u8 | factors_len u8 |
//                        factors [u16; 6] LE | elapsed_ns u64 LE.
//
// The runtime `WisdomCache::from_binary` rejects any file whose magic or
// version does not match, so a format mismatch degrades gracefully to the
// heuristic path (never to wrong results).
const BINARY_MAGIC: &[u8; 8] = b"OXIWISDM";
const BINARY_FORMAT_VERSION: u16 = 1;
/// `CooleyTukey` algorithm tag (see `api::wisdom::ALGO_TAG_COOLEY_TUKEY`).
const ALGO_TAG_COOLEY_TUKEY: u8 = 0;

fn main() {
    // ── Portable SIMD gating ────────────────────────────────────────────────────
    //
    // The `portable_simd` Cargo feature routes f64 codelets through the unstable
    // `core::simd` API, which requires the `#![feature(portable_simd)]` crate
    // attribute. That attribute only compiles on a nightly toolchain (or when
    // `RUSTC_BOOTSTRAP` explicitly opts in), so gating the attribute directly on
    // the Cargo feature would break `--all-features` builds on stable — exactly
    // the configuration CI exercises.
    //
    // Instead we expose an internal `oxifft_portable_simd` cfg that is set only
    // when BOTH the Cargo feature is enabled AND the toolchain accepts
    // `#![feature]`. On stable the feature therefore degrades to a graceful
    // no-op (the crate builds, the portable tier is simply inactive); on nightly
    // it activates. All `core::simd` code — including the crate attribute — is
    // gated on this cfg rather than on the Cargo feature directly.
    //
    // The matching `rustc-check-cfg` declaration keeps the `unexpected_cfgs`
    // lint silent (zero-warning policy) on toolchains that support check-cfg.
    println!("cargo:rustc-check-cfg=cfg(oxifft_portable_simd)");
    println!("cargo:rerun-if-env-changed=RUSTC_BOOTSTRAP");
    let portable_simd_requested = env::var_os("CARGO_FEATURE_PORTABLE_SIMD").is_some();
    if portable_simd_requested && feature_attrs_allowed() {
        println!("cargo:rustc-cfg=oxifft_portable_simd");
    }

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

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set by Cargo"));
    let baseline_path = out_dir.join("wisdom_baseline.bin");

    // ── Decide whether to embed a real baseline ────────────────────────────────
    let tune_requested = env::var("OXIFFT_TUNE").is_ok_and(|v| v == "1");
    let tune_skipped = env::var("OXIFFT_SKIP_TUNE").is_ok_and(|v| v == "1");

    // Cross-compilation: a build-time baseline profiled on the host is not
    // representative of (and its solver set may not even be valid for) the
    // target, so we never embed one when host != target.
    let host = env::var("HOST").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_default();
    let cross_compiling = !host.is_empty() && !target.is_empty() && host != target;

    let baseline = if tune_requested && !tune_skipped && !cross_compiling {
        // Embed a real, non-empty baseline. A build script cannot benchmark the
        // not-yet-compiled crate, so we bake the deterministic optimal choice
        // for the sizes where it is unambiguous (every power of two resolves to
        // Cooley-Tukey DIT). This gives the runtime a valid, non-empty wisdom
        // blob to consult. For per-machine *measured* wisdom, run the bundled
        // `oxifft_tune` binary and load its output via `import_from_file`.
        println!(
            "cargo:warning=\
oxifft: OXIFFT_TUNE=1 — embedding a build-time wisdom baseline. For per-machine \
measured tuning, run `oxifft_tune` and import the result at runtime."
        );
        build_baseline_blob()
    } else {
        if tune_requested && cross_compiling {
            println!(
                "cargo:warning=\
oxifft: OXIFFT_TUNE=1 ignored for cross-compilation (host {host} != target {target}); \
writing empty wisdom baseline."
            );
        }
        // Empty sentinel — the runtime falls back to heuristics.
        Vec::new()
    };

    fs::write(&baseline_path, &baseline).expect("failed to write wisdom_baseline.bin in OUT_DIR");
}

/// Return `true` when the active toolchain permits the `#![feature(...)]` crate
/// attribute, i.e. a nightly/dev channel or an explicit `RUSTC_BOOTSTRAP` opt-in.
///
/// Beta and stable reject `#![feature]`, so this returns `false` for them and the
/// `portable_simd` tier stays inactive. Any probe failure is treated as "not
/// allowed" so builds fail closed onto the stable-safe scalar/intrinsic path.
fn feature_attrs_allowed() -> bool {
    // `RUSTC_BOOTSTRAP` lets `#![feature]` compile on any channel; honor it so an
    // opt-in bootstrap build still gets the portable tier.
    if env::var_os("RUSTC_BOOTSTRAP").is_some() {
        return true;
    }

    // Cargo sets `RUSTC` to the compiler it is driving; fall back to `rustc`.
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc"));
    let Ok(output) = Command::new(rustc).arg("-vV").output() else {
        return false;
    };
    if !output.status.success() {
        return false;
    }

    // Parse the `release:` line (e.g. `release: 1.95.0-nightly`). Only the
    // nightly and dev channels carry an unstable suffix that permits `#![feature]`.
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines()
        .find_map(|line| line.strip_prefix("release:"))
        .map(str::trim)
        .is_some_and(|release| release.contains("nightly") || release.contains("dev"))
}

/// Encode a non-empty wisdom baseline blob in the binary format understood by
/// `api::wisdom::WisdomCache::from_binary`.
///
/// Contains one Cooley-Tukey DIT entry per power-of-two size in `2..=65536`.
fn build_baseline_blob() -> Vec<u8> {
    // Power-of-two sizes: 2, 4, 8, …, 65536.
    let sizes: Vec<u64> = (1..=16).map(|k| 1u64 << k).collect();
    let entry_count_u16 = u16::try_from(sizes.len()).unwrap_or(u16::MAX);
    let entry_count = entry_count_u16 as usize;

    let mut buf = Vec::with_capacity(16 + entry_count * 30);

    // Header.
    buf.extend_from_slice(BINARY_MAGIC);
    buf.extend_from_slice(&BINARY_FORMAT_VERSION.to_le_bytes());
    buf.extend_from_slice(&entry_count_u16.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // reserved

    // Entries.
    for &size in sizes.iter().take(entry_count) {
        buf.extend_from_slice(&size.to_le_bytes()); // size_key
        buf.push(ALGO_TAG_COOLEY_TUKEY); // algo_tag → "ct-dit"
        buf.push(0u8); // factors_len
        buf.extend_from_slice(&[0u8; 12]); // factors [u16; 6]
        buf.extend_from_slice(&0u64.to_le_bytes()); // elapsed_ns (unknown)
    }

    buf
}
