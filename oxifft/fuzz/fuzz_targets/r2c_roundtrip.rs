#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    // Parse n from first 2 bytes; cap at 1024 for speed
    let raw_n = u16::from_le_bytes([data[0], data[1]]) as usize;
    let n = (raw_n % 1025).max(2); // 2..=1024
                                   // Make n even (many r2c implementations optimise for even sizes; odd is also valid)
    let n = if n % 2 == 1 { n + 1 } else { n };

    let payload = &data[2..];
    if payload.len() < n * 4 {
        return;
    }

    let mut input = Vec::with_capacity(n);
    for i in 0..n {
        let bytes = [
            payload[4 * i],
            payload[4 * i + 1],
            payload[4 * i + 2],
            payload[4 * i + 3],
        ];
        let v = f32::from_le_bytes(bytes);
        if !v.is_finite() {
            return; // skip NaN / Inf
        }
        // Bound the amplitude, mirroring the proptest suite (plan_fuzz.rs).
        // A round trip intrinsically forms intermediate sums up to ~N * |x|
        // (e.g. the DC bin); for f32 those overflow to +/-inf once |x| nears
        // f32::MAX / N. That is a fundamental floating-point range limit, not a
        // library bug, so we exclude such gratuitously huge magnitudes here.
        // (With N <= 1024, 1e18 keeps N * |x| far below f32::MAX ~= 3.4e38.)
        if v.abs() > 1.0e18 {
            return;
        }
        input.push(v);
    }

    use oxifft::{Flags, RealPlan};

    let r2c = match RealPlan::<f32>::r2c_1d(n, Flags::ESTIMATE) {
        Some(p) => p,
        None => return,
    };
    let c2r = match RealPlan::<f32>::c2r_1d(n, Flags::ESTIMATE) {
        Some(p) => p,
        None => return,
    };

    let spectrum_len = n / 2 + 1;
    let mut spectrum = vec![oxifft::Complex::<f32>::new(0.0, 0.0); spectrum_len];
    let mut reconstructed = vec![0.0f32; n];

    r2c.execute_r2c(&input, &mut spectrum);
    // execute_c2r normalizes by 1/n automatically, so round-trip recovers input
    c2r.execute_c2r(&spectrum, &mut reconstructed);

    // Per-element tolerance combines a tight relative bound (catches real
    // correctness bugs at "normal" magnitudes) with an absolute floor tied to
    // the *whole input's* dynamic range. The floor is necessary because
    // roundoff at any one output element is bounded by error accumulated
    // across the whole O(n) transform, which scales with the *largest*
    // magnitude anywhere in the input — not with that element's own, possibly
    // near-zero, value. Without it, a fuzzer-found input mixing a huge value
    // with a near-zero one at another index (e.g. `[3.85e-34, 4.09e-34,
    // 4.09e-34, -139296.125, 9.47e-41, 9.95e-44]` at n=6, minimized from
    // `crash-a004db1abd95825e388d6a1007c08722a1b1f330`) makes a legitimate
    // ~0.0103 roundoff error at a near-zero index look like a bug, when it is
    // well within the f32 precision floor implied by the -139296.125 elsewhere
    // in the same input (16 * 6 * f32::EPSILON * 139296.125 =~ 1.6, i.e. this
    // specific case has ~150x headroom).
    let max_abs = input.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
    let atol = 16.0 * (n as f32) * f32::EPSILON * max_abs.max(1.0);
    let rtol = 5e-3_f32;
    for i in 0..n {
        let expected = input[i];
        let err = (reconstructed[i] - expected).abs();
        assert!(
            err <= atol + rtol * expected.abs(),
            "r2c/c2r round-trip error at index {}: got {} expected {} (n={}, atol={}, max_abs={})",
            i,
            reconstructed[i],
            expected,
            n,
            atol,
            max_abs,
        );
    }
});
