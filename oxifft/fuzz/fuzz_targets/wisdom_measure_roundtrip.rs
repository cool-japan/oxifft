#![no_main]

use libfuzzer_sys::fuzz_target;
use oxifft::{Direction, Flags, Plan};

// Drives the path the `algorithm_from_solver_name` applicability-gating fix
// (see CHANGELOG.md, Unreleased/Security) has to defend in production: import
// an attacker-controlled wisdom string, then plan and execute with a
// wisdom-consulting flag. `MEASURE` looks wisdom up as a fast path before
// falling back to benchmarking/heuristic selection on a miss; `WISDOM_ONLY`
// refuses to plan at all without a match (FFTW semantics). Either path can
// reconstruct an `Algorithm` straight from a fuzzer-chosen `(size,
// solver_name)` pairing, unlike the four names pinned by the regression
// tests in `oxifft/src/api/plan/types.rs` (`solver_name_gate_tests`) and
// `oxifft/src/api/wisdom_tests.rs`.
//
// The invariant under fuzzing is simply: no panic, and no non-finite output —
// a wisdom entry is never trusted enough to run an algorithm inapplicable to
// `n` (which could read/write out of bounds in a solver that assumes its own
// applicability, or otherwise produce garbage).
fuzz_target!(|data: &[u8]| {
    if data.len() < 5 {
        return;
    }

    let n = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    // Cap so a MEASURE miss (which benchmarks real candidates) stays fast.
    let n = n % 4097;
    let mode_byte = data[4];
    let wisdom_text = String::from_utf8_lossy(&data[5..]);

    // Must never panic on adversarial text (already fuzzed in isolation by
    // `wisdom_parse`); the result is deliberately ignored — planning below
    // must be safe whether the import succeeded, partially succeeded, or
    // failed outright.
    let _ = oxifft::api::import_from_string(&wisdom_text);

    let direction = if mode_byte & 1 == 0 {
        Direction::Forward
    } else {
        Direction::Backward
    };
    // Alternate between the two wisdom-consulting flags so both code paths in
    // `Plan::dft_1d` run against the freshly (possibly hostile) imported
    // wisdom.
    let flags = if mode_byte & 2 == 0 {
        Flags::MEASURE
    } else {
        Flags::WISDOM_ONLY
    };

    let Some(plan) = Plan::<f64>::dft_1d(n, direction, flags) else {
        return;
    };

    let input = vec![oxifft::Complex::<f64>::new(0.0, 0.0); n];
    let mut output = vec![oxifft::Complex::<f64>::new(0.0, 0.0); n];
    plan.execute(&input, &mut output);
    for c in &output {
        assert!(
            c.re.is_finite() && c.im.is_finite(),
            "execute produced a non-finite output for n={n}, flags={flags:?}"
        );
    }

    let mut inplace = vec![oxifft::Complex::<f64>::new(0.0, 0.0); n];
    plan.execute_inplace(&mut inplace);
    for c in &inplace {
        assert!(
            c.re.is_finite() && c.im.is_finite(),
            "execute_inplace produced a non-finite output for n={n}, flags={flags:?}"
        );
    }
});
