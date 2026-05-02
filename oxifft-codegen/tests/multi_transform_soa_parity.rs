// Correctness (parity) tests for the `_soa` SIMD multi-transform codelets.
//
// `gen_multi_transform_codelet!` emits two functions per invocation:
//   1. An outer `AoS` function `notw_{size}_v{v}_{isa}_{ty}` (scalar loop, not tested here).
//   2. An inner `SoA` SIMD function `notw_{size}_v{v}_{isa}_{ty}_soa` (tested here).
//
// The `SoA` layout for V transforms of size N:
//   `re_in [k * V + t]` = real part  of element k in transform t  (k in 0..N, t in 0..V)
//   `im_in [k * V + t]` = imag part  of element k in transform t
//   `re_out[k * V + t]` = real part  of X[k]       for transform t
//   `im_out[k * V + t]` = imag part  of X[k]       for transform t
//
// Test strategy:
//   - Generate V independent random complex vectors of size N.
//   - Pack them into SoA buffers.
//   - Call the `_soa` SIMD function.
//   - For each transform t, unpack its outputs and compare against the naïve O(N²) DFT.
//   - Tolerance: 1e-5 (f32 single precision).
//
// All tests are cfg-gated to x86/x86_64 because the `_soa` functions only exist
// on those targets (they use SSE2 / AVX2 intrinsics).

#![cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::approx_constant,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::assign_op_pattern
)]

use oxifft_codegen::gen_multi_transform_codelet;

// ── Emit the codelets under test ──────────────────────────────────────────────
//
// Each expansion emits (among other things):
//   pub unsafe fn notw_2_v4_sse2_f32_soa(re_in, im_in, re_out, im_out)
//   pub unsafe fn notw_4_v4_sse2_f32_soa(re_in, im_in, re_out, im_out)
//   pub unsafe fn notw_2_v8_avx2_f32_soa(re_in, im_in, re_out, im_out)
//   pub unsafe fn notw_4_v8_avx2_f32_soa(re_in, im_in, re_out, im_out)
//   pub unsafe fn notw_8_v8_avx2_f32_soa(re_in, im_in, re_out, im_out)

gen_multi_transform_codelet!(size = 2, v = 4, isa = sse2, ty = f32);
gen_multi_transform_codelet!(size = 4, v = 4, isa = sse2, ty = f32);
gen_multi_transform_codelet!(size = 2, v = 8, isa = avx2, ty = f32);
gen_multi_transform_codelet!(size = 4, v = 8, isa = avx2, ty = f32);
gen_multi_transform_codelet!(size = 8, v = 8, isa = avx2, ty = f32);

// ── Naïve DFT reference ───────────────────────────────────────────────────────

/// Naïve O(N²) forward DFT (sign = -1) of `input`, result into `output`.
fn dft_naive_f32(input_re: &[f32], input_im: &[f32], output_re: &mut [f32], output_im: &mut [f32]) {
    let n = input_re.len();
    debug_assert_eq!(input_im.len(), n);
    debug_assert_eq!(output_re.len(), n);
    debug_assert_eq!(output_im.len(), n);
    for k in 0..n {
        let mut re_acc = 0.0_f64;
        let mut im_acc = 0.0_f64;
        for j in 0..n {
            // forward DFT: angle = -2π k j / N
            let angle = -2.0 * core::f64::consts::PI * (k * j) as f64 / n as f64;
            let (ws, wc) = angle.sin_cos();
            let x_re = input_re[j] as f64;
            let x_im = input_im[j] as f64;
            re_acc += x_re * wc - x_im * ws;
            im_acc += x_re * ws + x_im * wc;
        }
        output_re[k] = re_acc as f32;
        output_im[k] = im_acc as f32;
    }
}

// ── LCG RNG ───────────────────────────────────────────────────────────────────

fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let mantissa_bits = *state >> 11;
    let scale = 1.0_f64 / (1u64 << 52) as f64;
    (mantissa_bits as f64 * scale * 2.0 - 1.0) as f32
}

/// Generate `count` random f32 values.
fn rand_f32(count: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..count).map(|_| lcg_next(&mut s)).collect()
}

// ── Tolerance check ───────────────────────────────────────────────────────────

fn check_close(got_re: f32, got_im: f32, exp_re: f32, exp_im: f32, tol: f32, label: &str) {
    let err_re = (got_re - exp_re).abs();
    let err_im = (got_im - exp_im).abs();
    assert!(
        err_re < tol && err_im < tol,
        "{label}: got ({got_re}, {got_im}i), expected ({exp_re}, {exp_im}i), \
         err=({err_re}, {err_im}) >= tol={tol}"
    );
}

// ── Core harness ─────────────────────────────────────────────────────────────

/// Pack V independent transforms (each of size N) into SoA buffers.
///
/// `inputs_re[t][k]` / `inputs_im[t][k]` → `re_soa[k*V+t]` / `im_soa[k*V+t]`.
fn pack_soa(
    inputs_re: &[Vec<f32>],
    inputs_im: &[Vec<f32>],
    n: usize,
    v: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut re_soa = vec![0.0_f32; n * v];
    let mut im_soa = vec![0.0_f32; n * v];
    for t in 0..v {
        for k in 0..n {
            re_soa[k * v + t] = inputs_re[t][k];
            im_soa[k * v + t] = inputs_im[t][k];
        }
    }
    (re_soa, im_soa)
}

/// Unpack SoA output for transform `t` into `(re_out[0..n], im_out[0..n])`.
fn unpack_soa(
    re_soa: &[f32],
    im_soa: &[f32],
    n: usize,
    v: usize,
    t: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut re = Vec::with_capacity(n);
    let mut im = Vec::with_capacity(n);
    for k in 0..n {
        re.push(re_soa[k * v + t]);
        im.push(im_soa[k * v + t]);
    }
    (re, im)
}

/// Run the full parity check for a single (size, V, ISA) combo.
///
/// `call_soa`: closure that calls the generated `_soa` function.
/// Returns `Ok(())` if all V transforms match the naïve DFT within `tol`.
fn run_parity<F>(size: usize, v: usize, seed: u64, tol: f32, label: &str, call_soa: F)
where
    F: Fn(&[f32], &[f32], &mut [f32], &mut [f32]),
{
    // Generate V independent random transforms
    let mut inputs_re: Vec<Vec<f32>> = Vec::with_capacity(v);
    let mut inputs_im: Vec<Vec<f32>> = Vec::with_capacity(v);
    for t in 0..v {
        let base_seed = seed.wrapping_add(t as u64 * 997);
        inputs_re.push(rand_f32(size, base_seed));
        inputs_im.push(rand_f32(size, base_seed ^ 0xDEAD_BEEF));
    }

    // Pack into SoA
    let (re_in, im_in) = pack_soa(&inputs_re, &inputs_im, size, v);
    let mut re_out = vec![0.0_f32; size * v];
    let mut im_out = vec![0.0_f32; size * v];

    // Call the SIMD SoA kernel
    call_soa(&re_in, &im_in, &mut re_out, &mut im_out);

    // For each transform, unpack and compare against naïve DFT
    for t in 0..v {
        let (got_re, got_im) = unpack_soa(&re_out, &im_out, size, v, t);
        let mut exp_re = vec![0.0_f32; size];
        let mut exp_im = vec![0.0_f32; size];
        dft_naive_f32(&inputs_re[t], &inputs_im[t], &mut exp_re, &mut exp_im);
        for k in 0..size {
            check_close(
                got_re[k],
                got_im[k],
                exp_re[k],
                exp_im[k],
                tol,
                &format!("{label}[t={t}][k={k}]"),
            );
        }
    }
}

// ── SSE2 f32 tests ───────────────────────────────────────────────────────────

/// SSE2 f32, V=4, size-2: 4 DFT-2 transforms processed simultaneously.
#[test]
fn sse2_f32_size2_soa_parity() {
    // SSE2 is baseline for x86_64; no runtime guard needed.
    run_parity(
        2,
        4,
        0xABCD_1234_0001,
        1e-5,
        "sse2_f32_size2",
        |ri, ii, ro, io| {
            unsafe {
                notw_2_v4_sse2_f32_soa(ri.as_ptr(), ii.as_ptr(), ro.as_mut_ptr(), io.as_mut_ptr())
            };
        },
    );
}

/// SSE2 f32, V=4, size-4: 4 DFT-4 transforms processed simultaneously.
#[test]
fn sse2_f32_size4_soa_parity() {
    run_parity(
        4,
        4,
        0xABCD_1234_0002,
        1e-5,
        "sse2_f32_size4",
        |ri, ii, ro, io| {
            unsafe {
                notw_4_v4_sse2_f32_soa(ri.as_ptr(), ii.as_ptr(), ro.as_mut_ptr(), io.as_mut_ptr())
            };
        },
    );
}

/// SSE2 f32, V=4, size-2: multiple independent seeds.
#[test]
fn sse2_f32_size2_soa_multi_seed() {
    for seed_idx in 0..5_u64 {
        run_parity(
            2,
            4,
            0xF00D_CAFE_0010u64.wrapping_add(seed_idx * 1_234_567),
            1e-5,
            &format!("sse2_f32_size2_seed{seed_idx}"),
            |ri, ii, ro, io| {
                unsafe {
                    notw_2_v4_sse2_f32_soa(
                        ri.as_ptr(),
                        ii.as_ptr(),
                        ro.as_mut_ptr(),
                        io.as_mut_ptr(),
                    )
                };
            },
        );
    }
}

/// SSE2 f32, V=4, size-4: multiple independent seeds.
#[test]
fn sse2_f32_size4_soa_multi_seed() {
    for seed_idx in 0..5_u64 {
        run_parity(
            4,
            4,
            0xF00D_CAFE_0020u64.wrapping_add(seed_idx * 1_234_567),
            1e-5,
            &format!("sse2_f32_size4_seed{seed_idx}"),
            |ri, ii, ro, io| {
                unsafe {
                    notw_4_v4_sse2_f32_soa(
                        ri.as_ptr(),
                        ii.as_ptr(),
                        ro.as_mut_ptr(),
                        io.as_mut_ptr(),
                    )
                };
            },
        );
    }
}

// ── AVX2 f32 tests ───────────────────────────────────────────────────────────

/// AVX2 f32, V=8, size-2: 8 DFT-2 transforms processed simultaneously.
#[test]
fn avx2_f32_size2_soa_parity() {
    if !std::is_x86_feature_detected!("avx2") {
        // Skip silently on older CPUs without AVX2.
        return;
    }
    run_parity(
        2,
        8,
        0xABCD_1234_0011,
        1e-5,
        "avx2_f32_size2",
        |ri, ii, ro, io| {
            unsafe {
                notw_2_v8_avx2_f32_soa(ri.as_ptr(), ii.as_ptr(), ro.as_mut_ptr(), io.as_mut_ptr())
            };
        },
    );
}

/// AVX2 f32, V=8, size-4: 8 DFT-4 transforms processed simultaneously.
#[test]
fn avx2_f32_size4_soa_parity() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    run_parity(
        4,
        8,
        0xABCD_1234_0012,
        1e-5,
        "avx2_f32_size4",
        |ri, ii, ro, io| {
            unsafe {
                notw_4_v8_avx2_f32_soa(ri.as_ptr(), ii.as_ptr(), ro.as_mut_ptr(), io.as_mut_ptr())
            };
        },
    );
}

/// AVX2 f32, V=8, size-8: 8 DFT-8 transforms processed simultaneously.
#[test]
fn avx2_f32_size8_soa_parity() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    run_parity(
        8,
        8,
        0xABCD_1234_0013,
        1e-5,
        "avx2_f32_size8",
        |ri, ii, ro, io| {
            unsafe {
                notw_8_v8_avx2_f32_soa(ri.as_ptr(), ii.as_ptr(), ro.as_mut_ptr(), io.as_mut_ptr())
            };
        },
    );
}

/// AVX2 f32, V=8, size-2: multiple seeds.
#[test]
fn avx2_f32_size2_soa_multi_seed() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    for seed_idx in 0..5_u64 {
        run_parity(
            2,
            8,
            0xBEEF_F00D_0030u64.wrapping_add(seed_idx * 7_654_321),
            1e-5,
            &format!("avx2_f32_size2_seed{seed_idx}"),
            |ri, ii, ro, io| {
                unsafe {
                    notw_2_v8_avx2_f32_soa(
                        ri.as_ptr(),
                        ii.as_ptr(),
                        ro.as_mut_ptr(),
                        io.as_mut_ptr(),
                    )
                };
            },
        );
    }
}

/// AVX2 f32, V=8, size-4: multiple seeds.
#[test]
fn avx2_f32_size4_soa_multi_seed() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    for seed_idx in 0..5_u64 {
        run_parity(
            4,
            8,
            0xBEEF_F00D_0040u64.wrapping_add(seed_idx * 7_654_321),
            1e-5,
            &format!("avx2_f32_size4_seed{seed_idx}"),
            |ri, ii, ro, io| {
                unsafe {
                    notw_4_v8_avx2_f32_soa(
                        ri.as_ptr(),
                        ii.as_ptr(),
                        ro.as_mut_ptr(),
                        io.as_mut_ptr(),
                    )
                };
            },
        );
    }
}

/// AVX2 f32, V=8, size-8: multiple seeds.
#[test]
fn avx2_f32_size8_soa_multi_seed() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    for seed_idx in 0..5_u64 {
        run_parity(
            8,
            8,
            0xBEEF_F00D_0050u64.wrapping_add(seed_idx * 7_654_321),
            1e-5,
            &format!("avx2_f32_size8_seed{seed_idx}"),
            |ri, ii, ro, io| {
                unsafe {
                    notw_8_v8_avx2_f32_soa(
                        ri.as_ptr(),
                        ii.as_ptr(),
                        ro.as_mut_ptr(),
                        io.as_mut_ptr(),
                    )
                };
            },
        );
    }
}
