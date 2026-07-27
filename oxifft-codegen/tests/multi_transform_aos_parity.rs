//! Numerical parity tests for the **outer `AoS`** multi-transform function.
//!
//! `gen_multi_transform_codelet!` emits a public outer function
//! `notw_{size}_v{v}_{isa}_{ty}(input, output, istride, ostride, count)` that runs
//! `count` forward DFTs of size `N` over the canonical batch-blocked Array-of-Structs
//! layout. The companion `_soa` parity test covers only the inner SIMD function; this
//! file exercises the outer function itself (which the SoA test explicitly skips).
//!
//! We use `isa = scalar` so only the outer (portable) function is emitted, letting
//! these tests run on every architecture. Coverage:
//!   - single full batch (`count == v`),
//!   - a trailing partial batch (`count < v`),
//!   - multiple batches plus a remainder (`count == 2*v + 1`),
//! each compared against a naive O(N^2) forward DFT for both `f32` and `f64`.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::approx_constant,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::doc_lazy_continuation
)]

use oxifft_codegen::gen_multi_transform_codelet;

const V: usize = 4;

// Emit outer-only (scalar) multi-transform functions for f64 and f32.
gen_multi_transform_codelet!(size = 2, v = 4, isa = scalar, ty = f64);
gen_multi_transform_codelet!(size = 4, v = 4, isa = scalar, ty = f64);
gen_multi_transform_codelet!(size = 8, v = 4, isa = scalar, ty = f64);
gen_multi_transform_codelet!(size = 2, v = 4, isa = scalar, ty = f32);
gen_multi_transform_codelet!(size = 4, v = 4, isa = scalar, ty = f32);
gen_multi_transform_codelet!(size = 8, v = 4, isa = scalar, ty = f32);

// ── Naive forward DFT reference (computed in f64) ────────────────────────────

fn dft_naive(re: &[f64], im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = re.len();
    let mut out_re = vec![0.0; n];
    let mut out_im = vec![0.0; n];
    for k in 0..n {
        for j in 0..n {
            let angle = -2.0 * core::f64::consts::PI * (k * j) as f64 / n as f64;
            let (s, c) = angle.sin_cos();
            out_re[k] += re[j] * c - im[j] * s;
            out_im[k] += re[j] * s + im[j] * c;
        }
    }
    (out_re, out_im)
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// Global transform `g`, element `k`, real/imag selector `c`: batch-blocked index.
fn aos_index(size: usize, g: usize, k: usize, c: usize) -> usize {
    let b = g / V;
    let t = g % V;
    b * (size * V * 2) + t * 2 + k * (V * 2) + c
}

/// Run one (size, count) case for f64 and compare each transform to the naive DFT.
fn run_f64(size: usize, count: usize, outer: unsafe fn(*const f64, *mut f64, usize, usize, usize)) {
    let blocks = count.div_ceil(V);
    let buf_len = blocks * size * V * 2;
    let mut input = vec![0.0_f64; buf_len];

    // Fill each transform with deterministic pseudo-random data.
    let mut seed = 0x1234_5678_9abc_def0 ^ ((size as u64) << 8) ^ (count as u64);
    let mut refs = Vec::with_capacity(count);
    for g in 0..count {
        let mut re = vec![0.0; size];
        let mut im = vec![0.0; size];
        for k in 0..size {
            re[k] = lcg(&mut seed);
            im[k] = lcg(&mut seed);
            input[aos_index(size, g, k, 0)] = re[k];
            input[aos_index(size, g, k, 1)] = im[k];
        }
        refs.push(dft_naive(&re, &im));
    }

    let mut output = vec![0.0_f64; buf_len];
    // Safety: buffers are sized for `blocks` full batch blocks; istride/ostride = 2*V.
    unsafe { outer(input.as_ptr(), output.as_mut_ptr(), 2 * V, 2 * V, count) };

    for g in 0..count {
        let (exp_re, exp_im) = &refs[g];
        for k in 0..size {
            let got_re = output[aos_index(size, g, k, 0)];
            let got_im = output[aos_index(size, g, k, 1)];
            let dr = (got_re - exp_re[k]).abs();
            let di = (got_im - exp_im[k]).abs();
            assert!(
                dr < 1e-9 && di < 1e-9,
                "f64 size {size} count {count} transform {g} bin {k}: \
                 got ({got_re},{got_im}), expected ({},{}), err=({dr},{di})",
                exp_re[k],
                exp_im[k]
            );
        }
    }
}

/// Run one (size, count) case for f32 and compare each transform to the naive DFT.
fn run_f32(size: usize, count: usize, outer: unsafe fn(*const f32, *mut f32, usize, usize, usize)) {
    let blocks = count.div_ceil(V);
    let buf_len = blocks * size * V * 2;
    let mut input = vec![0.0_f32; buf_len];

    let mut seed = 0x0fed_cba9_8765_4321 ^ ((size as u64) << 8) ^ (count as u64);
    let mut refs = Vec::with_capacity(count);
    for g in 0..count {
        let mut re = vec![0.0; size];
        let mut im = vec![0.0; size];
        for k in 0..size {
            re[k] = lcg(&mut seed);
            im[k] = lcg(&mut seed);
            input[aos_index(size, g, k, 0)] = re[k] as f32;
            input[aos_index(size, g, k, 1)] = im[k] as f32;
        }
        refs.push(dft_naive(&re, &im));
    }

    let mut output = vec![0.0_f32; buf_len];
    // Safety: buffers are sized for `blocks` full batch blocks; istride/ostride = 2*V.
    unsafe { outer(input.as_ptr(), output.as_mut_ptr(), 2 * V, 2 * V, count) };

    for g in 0..count {
        let (exp_re, exp_im) = &refs[g];
        for k in 0..size {
            let got_re = f64::from(output[aos_index(size, g, k, 0)]);
            let got_im = f64::from(output[aos_index(size, g, k, 1)]);
            let dr = (got_re - exp_re[k]).abs();
            let di = (got_im - exp_im[k]).abs();
            assert!(
                dr < 1e-3 && di < 1e-3,
                "f32 size {size} count {count} transform {g} bin {k}: \
                 got ({got_re},{got_im}), expected ({},{}), err=({dr},{di})",
                exp_re[k],
                exp_im[k]
            );
        }
    }
}

// Counts: single full batch (V), partial batch (V-1), multi-batch + remainder (2V+1).
const COUNTS: [usize; 3] = [V, V - 1, 2 * V + 1];

#[test]
fn outer_aos_size2_f64() {
    for &c in &COUNTS {
        run_f64(2, c, notw_2_v4_scalar_f64);
    }
}

#[test]
fn outer_aos_size4_f64() {
    for &c in &COUNTS {
        run_f64(4, c, notw_4_v4_scalar_f64);
    }
}

#[test]
fn outer_aos_size8_f64() {
    for &c in &COUNTS {
        run_f64(8, c, notw_8_v4_scalar_f64);
    }
}

#[test]
fn outer_aos_size2_f32() {
    for &c in &COUNTS {
        run_f32(2, c, notw_2_v4_scalar_f32);
    }
}

#[test]
fn outer_aos_size4_f32() {
    for &c in &COUNTS {
        run_f32(4, c, notw_4_v4_scalar_f32);
    }
}

#[test]
fn outer_aos_size8_f32() {
    for &c in &COUNTS {
        run_f32(8, c, notw_8_v4_scalar_f32);
    }
}
