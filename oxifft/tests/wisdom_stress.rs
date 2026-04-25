//! Stress tests for concurrent wisdom access.
//!
//! These tests verify that the global wisdom cache handles concurrent
//! reads, writes, imports, exports, and forgets without deadlocks or
//! data corruption.

#![cfg(feature = "std")]
// Test-specific: index/hash→f64 casts are safe for small iteration counts,
// format_push_string is clear in test helpers, and drop order doesn't matter in tests.
#![allow(
    clippy::cast_precision_loss,
    clippy::format_push_string,
    clippy::significant_drop_tightening
)]

use std::sync::{Arc, Barrier};
use std::thread;

use std::sync::Mutex;

use oxifft::api::{
    export_to_file, export_to_string, forget, import_from_file, import_from_string,
    merge_from_string, store_wisdom, wisdom_count, WisdomCache, WISDOM_FORMAT_VERSION,
};
use oxifft::kernel::WisdomEntry;

/// Number of threads to use in stress tests.
const NUM_THREADS: usize = 8;

/// Number of iterations per thread.
const ITERATIONS: usize = 200;

/// Serialise tests that touch the global wisdom singleton so they don't
/// interfere with each other when `cargo test` runs them in parallel
/// within the same process.
static GLOBAL_LOCK: Mutex<()> = Mutex::new(());

/// Build a minimal valid wisdom string with `n` entries starting at `base_hash`.
fn make_wisdom_string(base_hash: u64, n: usize) -> String {
    let mut s = format!("(oxifft-wisdom\n  (format_version {WISDOM_FORMAT_VERSION})\n");
    for i in 0..n {
        let hash = base_hash + i as u64;
        let cost = (hash as f64).mul_add(0.1, 1.0);
        s.push_str(&format!("  ({hash} \"solver_{hash}\" {cost})\n"));
    }
    s.push(')');
    s
}

// ─── (a) Many readers, one writer ─────────────────────────────────────────────

/// Multiple threads reading wisdom while one thread creates entries.
///
/// Verifies no deadlock or panic occurs when concurrent reads overlap
/// with writes to the global wisdom cache.
#[test]
fn stress_many_readers_one_writer() {
    let _guard = GLOBAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    forget();
    let barrier = Arc::new(Barrier::new(NUM_THREADS + 1));

    // Writer thread: stores entries into global wisdom
    let writer_barrier = Arc::clone(&barrier);
    let writer = thread::spawn(move || {
        writer_barrier.wait();
        for i in 0..ITERATIONS {
            store_wisdom(WisdomEntry {
                problem_hash: 1000 + i as u64,
                solver_name: format!("writer_{i}"),
                cost: i as f64 + 1.0,
            });
        }
    });

    // Reader threads: repeatedly export and count wisdom
    let readers: Vec<_> = (0..NUM_THREADS)
        .map(|_| {
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                for _ in 0..ITERATIONS {
                    let _ = export_to_string();
                    let _ = wisdom_count();
                }
            })
        })
        .collect();

    writer.join().expect("writer thread should not panic");
    for r in readers {
        r.join().expect("reader thread should not panic");
    }

    // The writer stored ITERATIONS entries; count must be at least that
    // (could be more from previous tests in the same process).
    assert!(
        wisdom_count() >= ITERATIONS,
        "expected at least {ITERATIONS} entries, got {}",
        wisdom_count()
    );

    forget();
}

// ─── (b) Many writers ─────────────────────────────────────────────────────────

/// Multiple threads creating plans of different sizes simultaneously.
///
/// Each thread stores entries with a unique hash range so we can verify
/// all entries landed correctly.
#[test]
fn stress_many_writers() {
    let _guard = GLOBAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    forget();
    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|tid| {
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                let base = (tid as u64) * 10_000;
                for i in 0..ITERATIONS {
                    store_wisdom(WisdomEntry {
                        problem_hash: base + i as u64 + 1,
                        solver_name: format!("t{tid}_s{i}"),
                        cost: (i + 1) as f64,
                    });
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("writer thread should not panic");
    }

    // Each thread wrote ITERATIONS entries with unique hashes.
    let expected_total = NUM_THREADS * ITERATIONS;
    let count = wisdom_count();
    assert!(
        count >= expected_total,
        "expected at least {expected_total} entries, got {count}"
    );

    forget();
}

// ─── (c) Read-write-forget cycle ──────────────────────────────────────────────

/// Threads racing to add and remove wisdom entries.
///
/// Exercises the forget → store → count cycle under contention, checking
/// that the system remains consistent (no panics, no deadlocks).
#[test]
fn stress_read_write_forget_cycle() {
    let _guard = GLOBAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    forget();
    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|tid| {
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                for i in 0..ITERATIONS {
                    // Store
                    store_wisdom(WisdomEntry {
                        problem_hash: (tid as u64) * 100_000 + i as u64 + 1,
                        solver_name: format!("cycle_{tid}_{i}"),
                        cost: 1.0,
                    });

                    // Read
                    let _ = wisdom_count();
                    let _ = export_to_string();

                    // Occasionally forget everything (every ~20 iterations)
                    if i % 20 == 0 {
                        forget();
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("cycle thread should not panic");
    }

    // No specific count assertion — forgets are racing with stores.
    // The point is no panics or deadlocks occurred.
    forget();
}

// ─── (d) Export during mutation ───────────────────────────────────────────────

/// Export wisdom to string while other threads are modifying the cache.
///
/// Verifies that the exported string is always well-formed (parseable).
#[test]
fn stress_export_during_mutation() {
    let _guard = GLOBAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    forget();
    let barrier = Arc::new(Barrier::new(NUM_THREADS + 1));

    // Mutator thread: continuously stores entries
    let mut_barrier = Arc::clone(&barrier);
    let mutator = thread::spawn(move || {
        mut_barrier.wait();
        for i in 0..ITERATIONS * 2 {
            store_wisdom(WisdomEntry {
                problem_hash: 5000 + i as u64,
                solver_name: format!("mutator_{i}"),
                cost: i as f64 + 0.5,
            });
        }
    });

    // Exporter threads: repeatedly export and verify the result parses
    let exporters: Vec<_> = (0..NUM_THREADS)
        .map(|_| {
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                let mut parse_failures = 0u64;
                for _ in 0..ITERATIONS {
                    let s = export_to_string();
                    // Verify the exported string is re-importable
                    let mut cache = WisdomCache::new();
                    if cache.import_string(&s).is_err() {
                        parse_failures += 1;
                    }
                }
                parse_failures
            })
        })
        .collect();

    mutator.join().expect("mutator thread should not panic");

    let mut total_failures = 0u64;
    for e in exporters {
        total_failures += e.join().expect("exporter thread should not panic");
    }

    assert_eq!(
        total_failures, 0,
        "exported wisdom should always be parseable; got {total_failures} parse failures"
    );

    forget();
}

// ─── (e) Import collision ─────────────────────────────────────────────────────

/// Multiple threads importing different wisdom strings simultaneously.
///
/// Each thread imports entries from a unique hash range. After all
/// threads complete, every entry from every thread should be present.
#[test]
fn stress_import_collision() {
    let _guard = GLOBAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    forget();
    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|tid| {
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                for batch in 0..ITERATIONS {
                    let base = (tid as u64) * 1_000_000 + (batch as u64) * 100;
                    let wisdom_str = make_wisdom_string(base + 1, 5);
                    let result = import_from_string(&wisdom_str)
                        .expect("import should succeed for valid wisdom");
                    assert_eq!(
                        result.imported, 5,
                        "each import batch should import 5 entries"
                    );
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("import thread should not panic");
    }

    // Each thread did ITERATIONS batches × 5 entries per batch.
    // All hashes are unique across threads, so all must be present.
    let expected = NUM_THREADS * ITERATIONS * 5;
    let count = wisdom_count();
    assert!(
        count >= expected,
        "expected at least {expected} entries, got {count}"
    );

    forget();
}

// ─── (f) File I/O stress ──────────────────────────────────────────────────────

/// Multiple threads writing wisdom to temp files concurrently.
///
/// Each thread uses its own temp file (to avoid OS-level file-lock issues
/// that aren't wisdom-related), but they all race to export/import from
/// the shared global cache.
#[test]
fn stress_file_io() {
    let _guard = GLOBAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    forget();

    // Seed the cache with some entries
    for i in 0..50 {
        store_wisdom(WisdomEntry {
            problem_hash: 9000 + i,
            solver_name: format!("file_seed_{i}"),
            cost: i as f64 + 1.0,
        });
    }

    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|tid| {
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                let dir = std::env::temp_dir();
                let path = dir.join(format!("oxifft_stress_wisdom_{tid}.txt"));

                b.wait();
                for i in 0..ITERATIONS {
                    // Export to file
                    export_to_file(&path).expect("export_to_file should succeed");

                    // Add more entries to the global cache
                    store_wisdom(WisdomEntry {
                        problem_hash: (tid as u64 + 1) * 200_000 + i as u64,
                        solver_name: format!("fio_{tid}_{i}"),
                        cost: 42.0,
                    });

                    // Import from file (this re-loads what we just wrote)
                    let _ = import_from_file(&path);
                }

                // Cleanup
                let _ = std::fs::remove_file(&path);
            })
        })
        .collect();

    for h in handles {
        h.join().expect("file I/O thread should not panic");
    }

    // Verify cache is still usable
    let count = wisdom_count();
    assert!(count > 0, "cache should have entries after file I/O stress");
    let _ = export_to_string();

    forget();
}

// ─── (g) Merge contention ─────────────────────────────────────────────────────

/// Multiple threads merging different wisdom strings into the global cache.
///
/// Merge is more complex than import because it compares costs. This test
/// ensures the merge path is race-free.
#[test]
fn stress_merge_contention() {
    let _guard = GLOBAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    forget();
    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|tid| {
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                for batch in 0..ITERATIONS {
                    // Create wisdom with overlapping hashes but different costs.
                    // Threads 0..N all write to hashes (batch*10 + 1)..(batch*10 + 6)
                    // with cost = tid+1, so the thread with tid=0 should win for
                    // cost (it has the lowest).
                    let base = (batch as u64) * 10 + 1;
                    let cost = (tid as f64) + 1.0;
                    let mut s =
                        format!("(oxifft-wisdom\n  (format_version {WISDOM_FORMAT_VERSION})\n");
                    for j in 0..5u64 {
                        let hash = base + j;
                        s.push_str(&format!("  ({hash} \"merge_t{tid}_b{batch}\" {cost})\n"));
                    }
                    s.push(')');

                    let _ = merge_from_string(&s);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("merge thread should not panic");
    }

    // After merge contention, the cache should contain entries.
    // Each batch produces 5 unique hashes, the lowest cost should have won.
    let count = wisdom_count();
    assert!(count > 0, "cache must have entries after merge stress");

    forget();
}

// ─── (h) Interleaved operations ───────────────────────────────────────────────

/// All wisdom operations interleaved across threads.
///
/// Each thread performs a mix of store, export, import, merge, count, and
/// forget — the "everything at once" scenario.
#[test]
fn stress_interleaved_operations() {
    let _guard = GLOBAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    forget();
    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|tid| {
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                for i in 0..ITERATIONS {
                    let phase = i % 6;
                    match phase {
                        0 => {
                            // Store
                            store_wisdom(WisdomEntry {
                                problem_hash: (tid as u64 + 1) * 500_000 + i as u64,
                                solver_name: format!("interleaved_{tid}"),
                                cost: 10.0,
                            });
                        }
                        1 => {
                            // Export
                            let _ = export_to_string();
                        }
                        2 => {
                            // Import
                            let s =
                                make_wisdom_string((tid as u64 + 1) * 600_000 + (i as u64) * 10, 3);
                            let _ = import_from_string(&s);
                        }
                        3 => {
                            // Merge
                            let s =
                                make_wisdom_string((tid as u64 + 1) * 700_000 + (i as u64) * 10, 3);
                            let _ = merge_from_string(&s);
                        }
                        4 => {
                            // Count
                            let _ = wisdom_count();
                        }
                        5 => {
                            // Forget
                            forget();
                        }
                        _ => unreachable!(),
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("interleaved thread should not panic");
    }

    // No specific assertion — success means no deadlocks, panics, or corruption.
    forget();
}

// ─── (i) WisdomCache local stress (non-global) ───────────────────────────────

/// Stress-test a shared `WisdomCache` behind `Arc<RwLock<>>` without the
/// global singleton, to isolate cache logic from the global accessor.
#[test]
fn stress_local_wisdom_cache() {
    use std::sync::RwLock;

    let cache = Arc::new(RwLock::new(WisdomCache::new()));
    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|tid| {
            let c = Arc::clone(&cache);
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                for i in 0..ITERATIONS {
                    // Write
                    {
                        let mut w = c.write().expect("cache write lock should not be poisoned");
                        w.store(WisdomEntry {
                            problem_hash: (tid as u64 + 1) * 300_000 + i as u64,
                            solver_name: format!("local_{tid}_{i}"),
                            cost: i as f64 + 0.5,
                        });
                    }

                    // Read
                    {
                        let r = c.read().expect("cache read lock should not be poisoned");
                        let _ = r.len();
                        let _ = r.export_string();
                    }

                    // Import round-trip
                    {
                        let exported = {
                            let r = c.read().expect("cache read lock should not be poisoned");
                            r.export_string()
                        };
                        let mut w = c.write().expect("cache write lock should not be poisoned");
                        let _ = w.import_string(&exported);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("local cache thread should not panic");
    }

    let guard = cache
        .read()
        .expect("final read lock should not be poisoned");
    assert!(
        guard.len() >= NUM_THREADS * ITERATIONS,
        "expected at least {} entries, got {}",
        NUM_THREADS * ITERATIONS,
        guard.len()
    );
}
