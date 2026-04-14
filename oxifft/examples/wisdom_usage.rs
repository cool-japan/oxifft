//! Wisdom usage example demonstrating plan caching.
//!
//! The wisdom system caches optimal plans to avoid re-measuring
//! algorithm performance on subsequent runs.

#![allow(clippy::cast_precision_loss)] // FFT size computations use float for math
#![allow(clippy::unreadable_literal)] // hash constants are more readable as-is

use oxifft::api::{
    export_to_file, export_to_string, fft, forget, import_from_file, import_from_string,
    merge_from_string, store_wisdom, wisdom_count, WISDOM_FORMAT_VERSION,
};
use oxifft::kernel::WisdomEntry;
use std::path::Path;

fn main() {
    println!("=== Wisdom System Example ===\n");

    // Start fresh
    forget();
    println!("Initial wisdom count: {}", wisdom_count());

    // ========================
    // Manual wisdom storage
    // ========================
    println!("\n--- Manual Wisdom Storage ---\n");

    // Store wisdom entries for specific problem sizes
    store_wisdom(WisdomEntry {
        problem_hash: 0xcbf29ce484222326, // Hash for size 2
        solver_name: "ct-dit".to_string(),
        cost: 10.0,
    });

    store_wisdom(WisdomEntry {
        problem_hash: 0xcbf29ce484222325, // Hash for size 1
        solver_name: "nop".to_string(),
        cost: 0.0,
    });

    println!("After storing 2 entries: {} entries", wisdom_count());

    // ========================
    // Export/Import as String
    // ========================
    println!("\n--- Export/Import as String ---\n");

    let wisdom_str = export_to_string();
    println!("Exported wisdom:\n{wisdom_str}");

    // Clear and reimport
    forget();
    println!("\nAfter forget: {} entries", wisdom_count());

    let result = import_from_string(&wisdom_str).expect("Failed to import wisdom");
    println!(
        "Imported {} entries from string (format v{})",
        result.imported, result.format_version
    );
    println!("Current wisdom count: {}", wisdom_count());

    // ========================
    // Export/Import to File
    // ========================
    println!("\n--- Export/Import to File ---\n");

    // Create a temporary file path
    let wisdom_path = Path::new("/tmp/oxifft_wisdom_example.txt");

    // Export to file
    export_to_file(wisdom_path).expect("Failed to export wisdom to file");
    println!("Exported wisdom to: {}", wisdom_path.display());

    // Clear and reimport from file
    forget();
    println!("After forget: {} entries", wisdom_count());

    match import_from_file(wisdom_path) {
        Ok(result) => {
            println!("Imported {} entries from file", result.imported);
            println!("Current wisdom count: {}", wisdom_count());
        }
        Err(e) => {
            println!("Error importing from file: {e}");
        }
    }

    // Clean up the temporary file
    let _ = std::fs::remove_file(wisdom_path);

    // ========================
    // Practical Usage Pattern
    // ========================
    println!("\n--- Practical Usage Pattern ---\n");

    // A typical application would:
    // 1. Load wisdom at startup
    // 2. Run FFTs (which may accumulate wisdom)
    // 3. Save wisdom at shutdown

    forget(); // Start fresh for this demo

    println!("Step 1: Attempt to load wisdom from file (may not exist)");
    if import_from_file(wisdom_path).is_ok() {
        println!("  Loaded existing wisdom");
    } else {
        println!("  No existing wisdom file found - starting fresh");
    }

    println!("\nStep 2: Run some FFTs");
    let input: Vec<oxifft::Complex<f64>> = (0..16)
        .map(|k| {
            let t = 2.0 * std::f64::consts::PI * f64::from(k) / 16.0;
            oxifft::Complex::new((2.0 * t).sin(), 0.0)
        })
        .collect();

    let _ = fft(&input);
    println!("  Computed FFT of size 16");

    // In a real application, the planner would store wisdom
    // when measuring algorithm performance (non-ESTIMATE modes)

    println!("\nStep 3: Save wisdom for future runs");
    if export_to_file(wisdom_path).is_ok() {
        println!("  Saved wisdom to: {}", wisdom_path.display());
    }

    // Clean up
    let _ = std::fs::remove_file(wisdom_path);

    // ========================
    // Version Handling
    // ========================
    println!("\n--- Version Handling ---\n");

    forget();

    // Legacy format (oxifft-wisdom-1.0 header) is accepted as format version 0.
    let legacy_wisdom = "(oxifft-wisdom-1.0\n  (12345 \"ct-dit\" 100.0)\n)";
    match import_from_string(legacy_wisdom) {
        Ok(r) => println!(
            "Imported {} entries from legacy format (format v{})",
            r.imported, r.format_version
        ),
        Err(e) => println!("Error importing legacy wisdom: {e}"),
    }

    // Future format (format_version higher than current) is rejected.
    let future_version = WISDOM_FORMAT_VERSION + 1;
    let future_wisdom = format!(
        "(oxifft-wisdom\n  (format_version {future_version})\n  (12345 \"ct-dit\" 100.0)\n)"
    );
    match import_from_string(&future_wisdom) {
        Ok(_) => println!("Accepted future wisdom (unexpected)"),
        Err(e) => println!("Correctly rejected future format: {e}"),
    }

    // ========================
    // Merge demonstration
    // ========================
    println!("\n--- Merge Demonstration ---\n");

    forget();
    store_wisdom(oxifft::kernel::WisdomEntry {
        problem_hash: 12345,
        solver_name: "ct-dit".to_string(),
        cost: 200.0,
    });

    // Merge with data that contains a better entry for hash 12345.
    let better_wisdom = format!(
        "(oxifft-wisdom\n  (format_version {WISDOM_FORMAT_VERSION})\n  (12345 \"stockham\" 50.0)\n  (99999 \"rader\" 30.0)\n)"
    );
    match merge_from_string(&better_wisdom) {
        Ok(m) => println!(
            "Merge result: added={}, replaced={}, kept_existing={}",
            m.added, m.replaced, m.kept_existing
        ),
        Err(e) => println!("Merge error: {e}"),
    }

    println!("\nFinal wisdom count: {}", wisdom_count());
}
