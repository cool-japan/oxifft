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

    for i in 0..n {
        let expected = input[i];
        let err = (reconstructed[i] - expected).abs();
        let scale = expected.abs().max(1.0);
        assert!(
            err / scale < 5e-3,
            "r2c/c2r round-trip error at index {}: got {} expected {} (n={})",
            i,
            reconstructed[i],
            expected,
            n,
        );
    }
});
