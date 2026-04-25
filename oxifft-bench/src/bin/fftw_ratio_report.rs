//! FFTW Ratio Report — Post-processor for FFTW parity gate benchmarks.
//!
//! Reads criterion JSON from `target/criterion/*/current/estimates.json`
//! (written by `cargo bench -- --save-baseline current`), pairs `OxiFFT` and
//! FFTW entries by gate name, computes the timing ratio, and emits:
//!
//! 1. A JSON snapshot to `benches/baselines/v0.3.0/fftw_ratios_YYYY-MM-DD.json`
//! 2. A Markdown ratio table to stdout
//!
//! # Usage
//!
//! ```bash
//! cargo run --features fftw-compare -p oxifft-bench --bin fftw_ratio_report
//! ```
//!
//! Assumes the bench has been run with `--save-baseline current` so that
//! `target/criterion/<group>/<bench>/current/estimates.json` exists.
//! Falls back to `new/estimates.json` if `current/` is absent.

#![cfg(feature = "fftw-compare")]
#![allow(clippy::cast_precision_loss)] // FFT size math; values are always << 2^53

use serde::Serialize;
use serde_json::Value;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

/// A simple boxed error type alias for this binary.
type BoxError = Box<dyn std::error::Error + Send + Sync>;

// =============================================================================
// Constants
// =============================================================================

/// The 7 gate ids in canonical order.
const GATE_IDS: &[&str] = &[
    "1d_cplx_2e10",
    "1d_cplx_2e20",
    "1d_real_2e10",
    "2d_cplx_1024",
    "batch_1000x256",
    "prime_2017",
    "dct2_1024",
];

/// Target ratios (`oxifft / fftw`) per gate — the v1.0 performance goals.
const TARGET_RATIOS: &[f64] = &[2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0];

/// ISO date string for this baseline snapshot.
const BASELINE_DATE: &str = "2026-04-20";

// =============================================================================
// JSON output schema
// =============================================================================

#[derive(Debug, Serialize)]
struct GateResult {
    id: String,
    oxifft_ns: f64,
    fftw_ns: f64,
    ratio: f64,
    target_ratio: f64,
    pass: bool,
}

#[derive(Debug, Serialize)]
struct Summary {
    gates_passing: usize,
    gates_failing: usize,
    geomean_ratio: Option<f64>,
}

#[derive(Debug, Serialize)]
struct RatioReport {
    date: String,
    git_sha: String,
    cpu: String,
    rustc: String,
    gates: Vec<GateResult>,
    summary: Summary,
}

// =============================================================================
// System info helpers
// =============================================================================

/// Get the short git SHA of HEAD.
fn git_sha() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| o.status.success().then_some(o.stdout))
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .map_or_else(|| "unknown".to_string(), |s| s.trim().to_string())
}

/// Get the CPU brand string.
fn cpu_brand() -> String {
    #[cfg(target_os = "macos")]
    {
        Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|o| o.status.success().then_some(o.stdout))
            .and_then(|bytes| String::from_utf8(bytes).ok())
            .map_or_else(|| "unknown-macos".to_string(), |s| s.trim().to_string())
    }

    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|contents| {
                contents
                    .lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|s| s.trim().to_string())
            })
            .unwrap_or_else(|| "unknown-linux".to_string())
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        "unknown-platform".to_string()
    }
}

/// Get rustc version string.
fn rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| o.status.success().then_some(o.stdout))
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .map_or_else(|| "unknown".to_string(), |s| s.trim().to_string())
}

// =============================================================================
// Criterion output parsing
// =============================================================================

/// Locate the criterion `estimates.json` for a given group and bench name.
///
/// Tries `current/` first (from `--save-baseline current`), then `new/`.
fn locate_estimates_json(criterion_dir: &Path, group: &str, bench: &str) -> Option<PathBuf> {
    let base = criterion_dir.join(group).join(bench);

    let current_path = base.join("current").join("estimates.json");
    if current_path.exists() {
        return Some(current_path);
    }

    let new_path = base.join("new").join("estimates.json");
    if new_path.exists() {
        return Some(new_path);
    }

    None
}

/// Parse the `mean.point_estimate` (nanoseconds) from `estimates.json`.
fn parse_mean_ns(path: &Path) -> Option<f64> {
    let content = std::fs::read_to_string(path).ok()?;
    let json: Value = serde_json::from_str(&content).ok()?;

    // Try mean.point_estimate first (primary)
    let mean_ns = json
        .get("mean")
        .and_then(|m| m.get("point_estimate"))
        .and_then(Value::as_f64);

    if let Some(ns) = mean_ns {
        return Some(ns);
    }

    // Fallback: slope.point_estimate (criterion linear regression result)
    json.get("slope")
        .and_then(|s| s.get("point_estimate"))
        .and_then(Value::as_f64)
}

// =============================================================================
// Geometric mean helper
// =============================================================================

/// Compute the geometric mean of a non-empty slice of positive values.
fn geomean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let n = values.len() as f64;
    let log_sum: f64 = values.iter().map(|v| v.ln()).sum();
    Some((log_sum / n).exp())
}

// =============================================================================
// Gate collection
// =============================================================================

/// Collect per-gate results by reading criterion output.
fn collect_gates(criterion_dir: &Path) -> Vec<GateResult> {
    let mut gates: Vec<GateResult> = Vec::with_capacity(GATE_IDS.len());

    for (&gate_id, &target_ratio) in GATE_IDS.iter().zip(TARGET_RATIOS.iter()) {
        let oxi_path = locate_estimates_json(criterion_dir, gate_id, "oxifft");
        let fftw_path = locate_estimates_json(criterion_dir, gate_id, "fftw");

        match (oxi_path, fftw_path) {
            (Some(oxi_p), Some(fftw_p)) => {
                let oxifft_ns = parse_mean_ns(&oxi_p).unwrap_or(f64::NAN);
                let fftw_ns = parse_mean_ns(&fftw_p).unwrap_or(f64::NAN);

                let ratio = if fftw_ns.is_finite() && fftw_ns > 0.0 {
                    oxifft_ns / fftw_ns
                } else {
                    f64::NAN
                };

                let pass = ratio.is_finite() && ratio < target_ratio;

                eprintln!(
                    "  {gate_id}: oxifft={oxifft_ns:.1} ns  fftw={fftw_ns:.1} ns  \
                     ratio={ratio:.3}  target<{target_ratio:.1}  {}",
                    if pass { "PASS" } else { "FAIL" }
                );

                gates.push(GateResult {
                    id: gate_id.to_string(),
                    oxifft_ns,
                    fftw_ns,
                    ratio,
                    target_ratio,
                    pass,
                });
            }
            (oxi_p, fftw_p) => {
                eprintln!(
                    "  {gate_id}: MISSING estimates — oxifft={}, fftw={}",
                    if oxi_p.is_some() { "found" } else { "MISSING" },
                    if fftw_p.is_some() { "found" } else { "MISSING" }
                );
                gates.push(GateResult {
                    id: gate_id.to_string(),
                    oxifft_ns: f64::NAN,
                    fftw_ns: f64::NAN,
                    ratio: f64::NAN,
                    target_ratio,
                    pass: false,
                });
            }
        }
    }

    gates
}

// =============================================================================
// Report emission
// =============================================================================

/// Write the JSON report and emit the Markdown table.
fn write_report(report: &RatioReport, baseline_dir: &Path) -> Result<(), BoxError> {
    // Emit the JSON file
    std::fs::create_dir_all(baseline_dir)?;
    let json_path = baseline_dir.join(format!("fftw_ratios_{BASELINE_DATE}.json"));
    let json_content = serde_json::to_string_pretty(report)?;
    std::fs::write(&json_path, &json_content)?;
    eprintln!("\nRatios written to: {}", json_path.display());

    // Emit the Markdown table to stdout
    let stdout = io::stdout();
    let mut out = stdout.lock();

    writeln!(out, "# FFTW Parity Ratio Report — {}", report.date)?;
    writeln!(out)?;
    writeln!(
        out,
        "| Gate | OxiFFT (ns) | FFTW (ns) | Ratio | Target | Status |"
    )?;
    writeln!(
        out,
        "|------|-------------|-----------|-------|--------|--------|"
    )?;

    for gate in &report.gates {
        let status = if gate.pass { "PASS" } else { "FAIL" };
        let ratio_str = if gate.ratio.is_finite() {
            format!("{:.3}", gate.ratio)
        } else {
            "N/A".to_string()
        };
        let oxi_str = if gate.oxifft_ns.is_finite() {
            format!("{:.1}", gate.oxifft_ns)
        } else {
            "N/A".to_string()
        };
        let fftw_str = if gate.fftw_ns.is_finite() {
            format!("{:.1}", gate.fftw_ns)
        } else {
            "N/A".to_string()
        };
        writeln!(
            out,
            "| {} | {} | {} | {} | <{:.1} | {} |",
            gate.id, oxi_str, fftw_str, ratio_str, gate.target_ratio, status
        )?;
    }
    writeln!(out)?;

    if let Some(gm) = report.summary.geomean_ratio {
        writeln!(out, "Geometric mean ratio: {gm:.3}")?;
    }
    writeln!(
        out,
        "Gates passing: {}/{} | Git: {} | CPU: {}",
        report.summary.gates_passing,
        GATE_IDS.len(),
        report.git_sha,
        report.cpu
    )?;

    Ok(())
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<(), BoxError> {
    // Locate the workspace root (the directory containing `target/`)
    let bench_manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string()));
    // `oxifft-bench` is one level below workspace root
    let workspace_root = bench_manifest_dir
        .parent()
        .map_or_else(|| bench_manifest_dir.clone(), Path::to_path_buf);

    let criterion_dir = workspace_root.join("target").join("criterion");
    let baseline_dir = workspace_root
        .join("benches")
        .join("baselines")
        .join("v0.3.0");

    eprintln!("Criterion output dir: {}", criterion_dir.display());
    eprintln!("Baseline output dir:  {}", baseline_dir.display());

    let gates = collect_gates(&criterion_dir);

    let all_ratios: Vec<f64> = gates
        .iter()
        .filter_map(|g| g.ratio.is_finite().then_some(g.ratio))
        .collect();

    let gates_passing = gates.iter().filter(|g| g.pass).count();
    let gates_failing = gates.len() - gates_passing;
    let geomean_ratio = geomean(&all_ratios);

    let report = RatioReport {
        date: BASELINE_DATE.to_string(),
        git_sha: git_sha(),
        cpu: cpu_brand(),
        rustc: rustc_version(),
        gates,
        summary: Summary {
            gates_passing,
            gates_failing,
            geomean_ratio,
        },
    };

    write_report(&report, &baseline_dir)?;

    Ok(())
}

// =============================================================================
// Unit tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_computation_is_correct() {
        // Given oxifft_ns=200.0, fftw_ns=100.0 → ratio=2.0
        let oxifft_ns = 200.0_f64;
        let fftw_ns = 100.0_f64;
        let ratio = oxifft_ns / fftw_ns;
        assert!(
            (ratio - 2.0).abs() < 1e-12,
            "ratio should be 2.0, got {ratio}"
        );

        // target 2.0: ratio=2.0 is NOT strictly less than 2.0 → fail (2.0 >= 2.0)
        assert!(ratio >= 2.0, "ratio=2.0 should fail target<2.0");

        // target 2.0: ratio=1.99 passes
        let ratio_pass = 199.0 / 100.0;
        assert!(ratio_pass < 2.0, "ratio=1.99 should pass target<2.0");

        // target 3.0: ratio=2.0 passes
        assert!(ratio < 3.0, "ratio=2.0 should pass target<3.0");
    }

    #[test]
    fn geomean_of_equal_values() {
        let values = vec![2.0_f64, 2.0, 2.0];
        let gm = geomean(&values).expect("geomean of non-empty slice");
        assert!((gm - 2.0).abs() < 1e-12, "geomean of [2,2,2] = 2, got {gm}");
    }

    #[test]
    fn geomean_empty_returns_none() {
        assert!(geomean(&[]).is_none());
    }

    #[test]
    fn geomean_mixed_values() {
        // geomean(1, 4) = sqrt(1*4) = 2
        let values = vec![1.0_f64, 4.0];
        let gm = geomean(&values).expect("geomean of 2 values");
        assert!((gm - 2.0).abs() < 1e-12, "geomean(1,4)=2, got {gm}");
    }
}
