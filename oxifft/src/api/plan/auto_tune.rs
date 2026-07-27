//! Auto-tuning support for OxiFFT.
//!
//! This module profiles candidate algorithms for specific transform sizes and
//! returns the fastest, allowing the planner to make evidence-based algorithm
//! selections rather than relying purely on heuristics.
//!
//! All tuning functions require the `std` feature because they use
//! [`std::time::Instant`] for wall-clock measurement.

#[cfg(feature = "std")]
use std::time::Instant;

#[cfg(feature = "std")]
use crate::api::wisdom::WisdomCache;
use crate::api::Direction;
#[cfg(feature = "std")]
use crate::api::Flags;
#[cfg(feature = "std")]
use crate::kernel::WisdomEntry;
#[cfg(feature = "std")]
use crate::kernel::{Complex, Float};
#[cfg(feature = "std")]
use crate::kernel::{Planner, PlannerFlags};
use crate::prelude::*;

#[cfg(feature = "std")]
use super::types::Plan;

// ─── Public types ─────────────────────────────────────────────────────────────

/// Result of tuning a single transform size.
#[derive(Debug, Clone)]
pub struct TuneResult {
    /// Transform size that was profiled.
    pub n: usize,
    /// Transform direction that was profiled.
    pub direction: Direction,
    /// Human-readable name of the winning algorithm.
    pub algorithm_name: String,
    /// Median elapsed time in nanoseconds across all repetitions.
    pub elapsed_ns: u64,
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Simple LCG-based deterministic input generator (no `rand` dependency).
///
/// Uses Knuth's multiplicative LCG: state′ = state × A + C (mod 2⁶⁴).
#[cfg(feature = "std")]
fn make_test_input<T: Float>(n: usize) -> Vec<Complex<T>> {
    let mut state: u64 = 0xdead_beef_cafe_1234;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let re = T::from_f64((state >> 32) as f64 / u32::MAX as f64 * 2.0 - 1.0);
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let im = T::from_f64((state >> 32) as f64 / u32::MAX as f64 * 2.0 - 1.0);
            Complex::new(re, im)
        })
        .collect()
}

/// Compute the median of a sorted slice of nanosecond timings.
///
/// Requires `timings` to be sorted in ascending order.
#[cfg(feature = "std")]
fn median_ns(timings: &[u64]) -> u64 {
    let len = timings.len();
    if len == 0 {
        return 0;
    }
    if len.is_multiple_of(2) {
        (timings[len / 2 - 1] / 2) + (timings[len / 2] / 2)
    } else {
        timings[len / 2]
    }
}

/// Time a single plan for `max_iters` repetitions and return the median
/// elapsed nanoseconds.
#[cfg(feature = "std")]
fn time_plan<T: Float>(plan: &Plan<T>, input: &[Complex<T>], max_iters: usize) -> u64 {
    let n = plan.size();
    let mut output = vec![Complex::<T>::new(T::ZERO, T::ZERO); n];

    // Warm-up: a few executions to fill caches and amortise first-call overhead.
    const WARMUP: usize = 4;
    for _ in 0..WARMUP {
        plan.execute(input, &mut output);
    }

    // Timed repetitions.
    let mut timings: Vec<u64> = Vec::with_capacity(max_iters);
    for _ in 0..max_iters {
        let t0 = Instant::now();
        plan.execute(input, &mut output);
        timings.push(t0.elapsed().as_nanos() as u64);
    }

    timings.sort_unstable();
    median_ns(&timings)
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Map public planning [`Flags`] onto the kernel planner's flags, which control
/// how wide a candidate set the planner proposes.
#[cfg(feature = "std")]
fn planner_flags_from(flags: Flags) -> PlannerFlags {
    if flags.is_exhaustive() {
        PlannerFlags::EXHAUSTIVE
    } else if flags.is_patient() {
        PlannerFlags::PATIENT
    } else {
        // MEASURE breadth (the planner's default).
        PlannerFlags::default()
    }
}

/// Profile every candidate algorithm valid for `n` and return the fastest.
///
/// `flags` controls how wide the candidate set is: MEASURE compares the core
/// alternatives, while PATIENT / EXHAUSTIVE add specialised variants (radix-4/8,
/// split-radix, Stockham, cache-oblivious, generic, …). The heuristic pick is
/// always included, so the winner is never slower than the ESTIMATE choice.
///
/// `max_iters` controls the number of timing repetitions per candidate; values
/// in 16–1000 are sensible. Smaller values are faster but noisier.
///
/// Each candidate is constructed with a fixed algorithm (never re-entering the
/// tuner) so there is no recursion. Returns `None` if no plan can be built.
#[cfg(feature = "std")]
pub fn tune_size<T: Float>(
    n: usize,
    direction: Direction,
    flags: Flags,
    max_iters: usize,
) -> Option<TuneResult> {
    // Clamp repetition count to a sane range.
    let iters = max_iters.clamp(4, 10_000);
    let input = make_test_input::<T>(n);

    // Assemble the candidate list: the heuristic pick first (guaranteeing the
    // tuner never regresses below ESTIMATE), then the planner's flag-aware
    // candidate set. Duplicates are dropped.
    let mut names: Vec<String> = Vec::new();
    if let Some(heuristic) = Plan::<T>::dft_1d(n, direction, Flags::ESTIMATE) {
        names.push(heuristic.wisdom_solver_name());
    }
    let planner = Planner::<T>::with_flags(planner_flags_from(flags));
    for name in planner.candidate_solver_names(n) {
        if !names.contains(&name) {
            names.push(name);
        }
    }

    // Time each candidate we can actually reconstruct and keep the fastest.
    let mut best: Option<(String, u64)> = None;
    for name in names {
        let Some(plan) = Plan::<T>::from_solver_name(n, direction, &name) else {
            continue;
        };
        let elapsed_ns = time_plan(&plan, &input, iters);
        match &best {
            Some((_, best_ns)) if *best_ns <= elapsed_ns => {}
            _ => best = Some((name, elapsed_ns)),
        }
    }

    let (algorithm_name, elapsed_ns) = best?;
    Some(TuneResult {
        n,
        direction,
        algorithm_name,
        elapsed_ns,
    })
}

/// Profile all sizes in `min_n..=max_n` (forward direction) and return a
/// [`WisdomCache`] populated with the timing winners.
///
/// `reps_per_size` is the number of timing repetitions per size (16–1000).
///
/// `on_progress` is called after each size is profiled with the size value,
/// so callers can print progress without coupling the tuner to any I/O.
#[cfg(feature = "std")]
pub fn tune_range<T: Float>(
    min_n: usize,
    max_n: usize,
    reps_per_size: usize,
    mut on_progress: impl FnMut(usize),
) -> WisdomCache {
    let mut cache = WisdomCache::new();

    for n in min_n..=max_n {
        if let Some(result) = tune_size::<T>(n, Direction::Forward, Flags::MEASURE, reps_per_size) {
            // Use `n` directly as the problem hash (matches the keying strategy
            // used by `from_baseline_wisdom` in types.rs).
            cache.store(WisdomEntry {
                problem_hash: n as u64,
                solver_name: result.algorithm_name.clone(),
                cost: result.elapsed_ns as f64,
            });
        }
        on_progress(n);
    }

    cache
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn tune_size_returns_result() {
        let result = tune_size::<f64>(64, Direction::Forward, Flags::MEASURE, 16);
        assert!(result.is_some(), "tune_size should succeed for n=64");
        let r = result.expect("already checked is_some");
        assert_eq!(r.n, 64);
        assert!(r.elapsed_ns > 0, "elapsed_ns must be non-zero");
        assert!(
            !r.algorithm_name.is_empty(),
            "algorithm_name must be non-empty"
        );
    }

    #[test]
    fn tune_size_various_sizes() {
        // Power of 2
        let r = tune_size::<f64>(8, Direction::Forward, Flags::MEASURE, 8);
        assert!(r.is_some());
        // Prime (Rader vs Bluestein vs Direct compared)
        let r = tune_size::<f64>(17, Direction::Forward, Flags::MEASURE, 8);
        assert!(r.is_some());
        // Smooth-7 composite
        let r = tune_size::<f64>(12, Direction::Forward, Flags::MEASURE, 8);
        assert!(r.is_some());
    }

    #[test]
    fn tune_size_compares_multiple_candidates() {
        // A large power-of-two exposes several genuinely different candidates
        // (ct-dit, ct-dif, stockham, cache-oblivious). PATIENT widens the set
        // further (split-radix, radix-4/8). The winner must be a real,
        // reconstructable algorithm, proving multi-candidate comparison runs.
        let planner = Planner::<f64>::with_flags(planner_flags_from(Flags::PATIENT));
        let candidates = planner.candidate_solver_names(1024);
        assert!(
            candidates.len() >= 3,
            "expected several candidates for n=1024 under PATIENT, got {candidates:?}"
        );

        let r = tune_size::<f64>(1024, Direction::Forward, Flags::PATIENT, 8)
            .expect("tune_size for n=1024 must succeed");
        assert!(
            Plan::<f64>::from_solver_name(1024, Direction::Forward, &r.algorithm_name).is_some(),
            "winning algorithm '{}' must be reconstructable",
            r.algorithm_name
        );
    }

    #[test]
    fn tune_range_covers_all_sizes() {
        let cache = tune_range::<f64>(2, 32, 8, |_| {});
        // Every size from 2 to 32 should have been profiled.
        assert!(
            cache.entry_count() >= 1,
            "at least one entry expected, got 0"
        );
        // We expect exactly 31 entries (one per size).
        assert_eq!(
            cache.entry_count(),
            31,
            "expected 31 entries for range 2..=32, got {}",
            cache.entry_count()
        );
    }

    #[test]
    fn binary_wisdom_round_trip() {
        let cache = tune_range::<f64>(2, 8, 4, |_| {});
        let bytes = cache.to_binary();
        assert!(!bytes.is_empty(), "binary output must not be empty");
        let restored =
            WisdomCache::from_binary(&bytes).expect("from_binary should succeed on valid data");
        assert_eq!(
            restored.entry_count(),
            cache.entry_count(),
            "entry count must be preserved in round-trip"
        );
    }

    #[test]
    fn binary_wisdom_round_trip_solver_name_content() {
        // n=6 = 2×3 uses MixedRadix in the heuristic; its wisdom name must
        // survive a binary round-trip. The innermost-first factor sequence for 6
        // is [2,3] (greedy-largest-first peels 3 then 2, reversed → [2,3]), so
        // the wisdom name = "mixed-radix-2-3". We take the name directly from the
        // heuristic plan (deterministic) rather than the tuner's fastest pick,
        // which is timing-dependent.
        let plan = Plan::<f64>::dft_1d(6, Direction::Forward, Flags::ESTIMATE)
            .expect("plan for n=6 must build");
        let solver_name = plan.wisdom_solver_name();
        assert_eq!(
            solver_name, "mixed-radix-2-3",
            "n=6 heuristic must map to mixed-radix-2-3, got: {solver_name}"
        );

        // Build a tiny cache with just n=6 and round-trip through binary.
        let mut cache = WisdomCache::new();
        cache.store(crate::kernel::WisdomEntry {
            problem_hash: 6,
            solver_name: solver_name.clone(),
            cost: 1.0,
        });
        let bytes = cache.to_binary();
        let restored = WisdomCache::from_binary(&bytes)
            .expect("from_binary must succeed for mixed-radix entry");

        // Verify that the restored entry carries the correct solver name.
        let entry = restored
            .lookup(6)
            .expect("entry for n=6 must survive round-trip");
        let restored_name = entry.solver_name.clone();
        assert!(
            restored_name.starts_with("mixed-radix-"),
            "round-tripped solver name must still be mixed-radix, got: {restored_name}"
        );
        assert_eq!(
            restored_name, solver_name,
            "solver name must be identical after binary round-trip"
        );
    }

    #[test]
    fn estimate_does_not_tune() {
        // ESTIMATE should return almost instantly — not invoke the tuner.
        let start = Instant::now();
        let _plan = Plan::<f64>::dft_1d(64, Direction::Forward, Flags::ESTIMATE);
        let elapsed = start.elapsed();
        // Should complete in << 100 ms (not 16+ repetitions of FFT timing).
        assert!(
            elapsed.as_millis() < 100,
            "ESTIMATE should be fast, took {elapsed:?}"
        );
    }

    #[test]
    fn make_test_input_length() {
        let v = make_test_input::<f64>(64);
        assert_eq!(v.len(), 64);
    }

    #[test]
    fn median_ns_basic() {
        let mut v = vec![3u64, 1, 4, 1, 5];
        v.sort_unstable();
        assert_eq!(median_ns(&v), 3);

        let mut even = vec![2u64, 4];
        even.sort_unstable();
        assert_eq!(median_ns(&even), 3);

        assert_eq!(median_ns(&[]), 0);
    }
}
