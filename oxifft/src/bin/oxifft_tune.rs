//! Runtime auto-tuning CLI for `OxiFFT`.
//!
//! Profiles FFT algorithms across a range of transform sizes and writes a
//! binary wisdom file that can be loaded by the library at startup.
//!
//! # Usage
//!
//! ```text
//! oxifft_tune [--min-size N] [--max-size N] [--reps N] [--output PATH]
//! ```
//!
//! Defaults:
//! - `--min-size 2`
//! - `--max-size 4096`
//! - `--reps 32`
//! - `--output wisdom_baseline.bin`

use oxifft::api::tune_range;
use oxifft::api::WisdomCache;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let min_size = parse_arg(&args, "--min-size").unwrap_or(2);
    let max_size = parse_arg(&args, "--max-size").unwrap_or(4096);
    let reps = parse_arg(&args, "--reps").unwrap_or(32);
    let output = args
        .iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "wisdom_baseline.bin".to_string());

    if max_size < min_size {
        eprintln!("error: --max-size ({max_size}) must be >= --min-size ({min_size})");
        std::process::exit(1);
    }

    eprintln!("OxiFFT tuner: profiling sizes {min_size}..={max_size} with {reps} reps each");
    eprintln!("Output: {output}");

    let total = max_size.saturating_sub(min_size) + 1;
    let mut progress_count = 0usize;

    let cache: WisdomCache = tune_range::<f64>(min_size, max_size, reps, |n| {
        progress_count += 1;
        // Print progress every 10% of the range.
        let pct_step = (total / 10).max(1);
        if progress_count % pct_step == 0 {
            // Use u64 arithmetic to avoid usize→f64 precision lint.
            let pct = progress_count as u64 * 100 / total as u64;
            eprintln!("  [{pct:3}%] tuned n={n}");
        }
    });

    let entry_count = cache.entry_count();
    let binary = cache.to_binary();

    match std::fs::write(&output, &binary) {
        Ok(()) => {
            eprintln!(
                "Wrote {entry_count} wisdom entries ({} bytes) to {output}",
                binary.len()
            );
        }
        Err(e) => {
            eprintln!("error: failed to write {output}: {e}");
            std::process::exit(1);
        }
    }
}

/// Parse a `--key value` pair from the argument list, returning the value
/// parsed as `usize`.  Returns `None` if the key is absent or unparseable.
fn parse_arg(args: &[String], key: &str) -> Option<usize> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}
