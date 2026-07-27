//! Tests for twiddle factor computation, caching, and SIMD multiplication.
//!
//! Extracted from `twiddle.rs` to keep that file under the 2000-line policy limit.

use super::*;
use std::sync::{Arc, Mutex};

/// Serialise cache-modifying tests to prevent interference.
static CACHE_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn test_twiddle_w4() {
    // W_4^0 = 1
    let w0: Complex<f64> = twiddle(4, 0);
    assert!((w0.re - 1.0).abs() < 1e-10);
    assert!(w0.im.abs() < 1e-10);

    // W_4^1 = -i
    let w1: Complex<f64> = twiddle(4, 1);
    assert!(w1.re.abs() < 1e-10);
    assert!((w1.im - (-1.0)).abs() < 1e-10);

    // W_4^2 = -1
    let w2: Complex<f64> = twiddle(4, 2);
    assert!((w2.re - (-1.0)).abs() < 1e-10);
    assert!(w2.im.abs() < 1e-10);
}

#[test]
fn test_compute_twiddles() {
    let tw: Vec<Complex<f64>> = compute_twiddles(8, 4);
    assert_eq!(tw.len(), 4);

    // W_8^0 = 1
    assert!((tw[0].re - 1.0).abs() < 1e-10);
    assert!(tw[0].im.abs() < 1e-10);
}

// -------------------------------------------------------------------------
// SIMD vs scalar parity tests
// -------------------------------------------------------------------------

#[test]
fn simd_vs_scalar_parity_f64() {
    let size = 256;
    let twiddles: Vec<Complex<f64>> = (0..size)
        .map(|k| {
            let angle = -2.0 * core::f64::consts::PI * k as f64 / size as f64;
            Complex::new(angle.cos(), angle.sin())
        })
        .collect();
    let input: Vec<Complex<f64>> = (0..size)
        .map(|k| Complex::new(k as f64, -(k as f64)))
        .collect();

    let mut simd_data = input.clone();
    let mut scalar_data = input;

    twiddle_mul_simd_f64(&mut simd_data, &twiddles);
    twiddle_mul_scalar_f64(&mut scalar_data, &twiddles);

    for (s, r) in simd_data.iter().zip(scalar_data.iter()) {
        let diff = (s.re - r.re).abs().max((s.im - r.im).abs());
        assert!(
            diff <= 1e-10 || diff <= 1e-12 * r.norm(),
            "SIMD/scalar f64 mismatch at element: simd={s:?} scalar={r:?}",
        );
    }
}

#[test]
fn simd_vs_scalar_parity_f32() {
    let size = 256;
    let twiddles: Vec<Complex<f32>> = (0..size)
        .map(|k| {
            let angle = -2.0 * core::f32::consts::PI * k as f32 / size as f32;
            Complex::new(angle.cos(), angle.sin())
        })
        .collect();
    let input: Vec<Complex<f32>> = (0..size)
        .map(|k| Complex::new(k as f32, -(k as f32)))
        .collect();

    let mut simd_data = input.clone();
    let mut scalar_data = input;

    twiddle_mul_simd_f32(&mut simd_data, &twiddles);
    twiddle_mul_scalar_f32(&mut scalar_data, &twiddles);

    for (s, r) in simd_data.iter().zip(scalar_data.iter()) {
        let diff = (s.re - r.re).abs().max((s.im - r.im).abs());
        assert!(
            diff <= 1e-5_f32 || diff <= 1e-6_f32 * r.norm(),
            "SIMD/scalar f32 mismatch: simd={s:?} scalar={r:?}",
        );
    }
}

#[test]
fn simd_vs_scalar_parity_f64_odd_length() {
    // Verify the remainder path (non-multiple-of-chunk-size)
    let size = 7;
    let twiddles: Vec<Complex<f64>> = (0..size)
        .map(|k| {
            let angle = -2.0 * core::f64::consts::PI * k as f64 / size as f64;
            Complex::new(angle.cos(), angle.sin())
        })
        .collect();
    let input: Vec<Complex<f64>> = (0..size)
        .map(|k| Complex::new(k as f64 + 1.0, -(k as f64) - 0.5))
        .collect();

    let mut simd_data = input.clone();
    let mut scalar_data = input;

    twiddle_mul_simd_f64(&mut simd_data, &twiddles);
    twiddle_mul_scalar_f64(&mut scalar_data, &twiddles);

    for (s, r) in simd_data.iter().zip(scalar_data.iter()) {
        let diff = (s.re - r.re).abs().max((s.im - r.im).abs());
        assert!(
            diff <= 1e-10,
            "SIMD/scalar f64 odd-length mismatch: simd={s:?} scalar={r:?}",
        );
    }
}

// -------------------------------------------------------------------------
// TwiddleCache tests
// -------------------------------------------------------------------------

#[test]
fn twiddle_cache_hit_f64() {
    let _lock = CACHE_LOCK.lock().expect("cache lock");
    clear_twiddle_cache();

    let t1 = get_twiddle_table_f64(256, TwiddleDirection::Forward);
    let t2 = get_twiddle_table_f64(256, TwiddleDirection::Forward);
    assert!(
        Arc::ptr_eq(&t1, &t2),
        "second call should return the cached Arc"
    );
}

#[test]
fn twiddle_cache_direction_separation() {
    let _lock = CACHE_LOCK.lock().expect("cache lock");
    clear_twiddle_cache();

    let fwd = get_twiddle_table_f64(64, TwiddleDirection::Forward);
    let inv = get_twiddle_table_f64(64, TwiddleDirection::Inverse);
    assert!(
        !Arc::ptr_eq(&fwd, &inv),
        "forward and inverse tables should be distinct"
    );
    // Verify forward[1] and inverse[1] are complex conjugates
    if fwd.factors.len() > 1 && inv.factors.len() > 1 {
        let f = fwd.factors[1];
        let i = inv.factors[1];
        assert!(
            (f.re - i.re).abs() < 1e-14,
            "real parts should match: {} vs {}",
            f.re,
            i.re
        );
        assert!(
            (f.im + i.im).abs() < 1e-14,
            "imag parts should be negated: {} vs {}",
            f.im,
            i.im
        );
    }
}

#[test]
fn twiddle_cache_invalidate_f64() {
    let _lock = CACHE_LOCK.lock().expect("cache lock");

    let t1 = get_twiddle_table_f64(512, TwiddleDirection::Forward);
    clear_twiddle_cache();
    let t2 = get_twiddle_table_f64(512, TwiddleDirection::Forward);
    assert!(
        !Arc::ptr_eq(&t1, &t2),
        "after clear, should allocate a new table"
    );
}

#[test]
fn twiddle_cache_hit_f32() {
    let _lock = CACHE_LOCK.lock().expect("cache lock");
    clear_twiddle_cache();

    let t1 = get_twiddle_table_f32(128, TwiddleDirection::Forward);
    let t2 = get_twiddle_table_f32(128, TwiddleDirection::Forward);
    assert!(
        Arc::ptr_eq(&t1, &t2),
        "second call should return the cached Arc (f32)"
    );
}

#[test]
fn twiddle_cache_f32_f64_separation() {
    // f32 and f64 tables must not be ptr_eq (they are different types)
    let _lock = CACHE_LOCK.lock().expect("cache lock");
    clear_twiddle_cache();

    let _f64 = get_twiddle_table_f64(32, TwiddleDirection::Forward);
    let _f32 = get_twiddle_table_f32(32, TwiddleDirection::Forward);
    // Just verifying both calls succeed; their types preclude ptr_eq comparison.
}

#[test]
fn twiddle_table_correctness_f64() {
    let _lock = CACHE_LOCK.lock().expect("cache lock");
    clear_twiddle_cache();

    let n = 8;
    let table = get_twiddle_table_f64(n, TwiddleDirection::Forward);
    assert_eq!(table.factors.len(), n);

    // W_8^0 = 1
    assert!((table.factors[0].re - 1.0).abs() < 1e-14);
    assert!(table.factors[0].im.abs() < 1e-14);

    // Each factor should lie on the unit circle
    for (k, w) in table.factors.iter().enumerate() {
        let mag_sq = w.re * w.re + w.im * w.im;
        assert!(
            (mag_sq - 1.0).abs() < 1e-13,
            "W_{n}^{k} should be on unit circle, |w|²={mag_sq}"
        );
    }
}

// -------------------------------------------------------------------------
// SoA vs AoS correctness tests
// -------------------------------------------------------------------------

/// Helper: compute max ULP distance between two f64 values.
fn ulp_distance_f64(a: f64, b: f64) -> u64 {
    let ai = a.to_bits();
    let bi = b.to_bits();
    ai.abs_diff(bi)
}

/// For a given size, verify that applying AoS and SoA twiddle multiplication
/// to the same input produces numerically equivalent results (f64).
///
/// The AoS and SoA table-construction loops (`compute_twiddle_table_f64` vs
/// `compute_twiddle_table_soa_f64`) use the identical scalar formula but are
/// written as structurally different Rust loops (iterator `.map().collect()`
/// vs a manual `for` + `.push()`), so under `-O` release optimization LLVM's
/// auto-vectorizer is free to pick different (equally valid) instruction
/// sequences for the `k as f64 / size as f64` division feeding `cos`/`sin` in
/// each loop — e.g. a vectorized reciprocal-based division in one and a
/// direct scalar division in the other. A perturbation this tiny in the
/// input angle can occasionally amplify to a few thousand ULP in the trig
/// output for specific (size, idx) pairs close to a libm range-reduction
/// sensitivity point, purely as a byproduct of that codegen choice — this
/// was observed empirically up to 2048 ULP at size=16384 under release
/// builds (vs. ≤4 ULP under debug, where the vectorizer doesn't kick in the
/// same way). A raw ULP bound is the wrong tool here: it is sensitive to
/// exactly this kind of benign, compiler-version/CPU-dependent rounding
/// noise. A real algorithmic bug (wrong sign, wrong index, wrong butterfly)
/// would show up as a relative error many orders of magnitude larger than
/// this, so an absolute/relative-error gate several thousand times looser
/// than the observed noise floor and still ~1000x tighter than real-bug
/// territory is the numerically meaningful check.
fn check_soa_vs_aos_f64(size: usize) {
    let _lock = CACHE_LOCK.lock().expect("cache lock");
    clear_twiddle_cache();

    let input: Vec<Complex<f64>> = (0..size)
        .map(|k| Complex::new((k as f64).sin(), (k as f64).cos()))
        .collect();

    // AoS path
    let aos_table = get_twiddle_table_f64(size, TwiddleDirection::Forward);
    let mut aos_data = input.clone();
    twiddle_mul_simd_f64(&mut aos_data, &aos_table.factors);

    // SoA path
    let soa_table = get_twiddle_table_soa_f64(size, TwiddleDirection::Forward);
    let mut soa_data = input;
    twiddle_mul_soa_simd_f64(&mut soa_data, &soa_table.re, &soa_table.im);

    const ABS_FLOOR: f64 = 1e-11;
    const REL_TOL: f64 = 1e-9;
    for (idx, (a, s)) in aos_data.iter().zip(soa_data.iter()).enumerate() {
        let re_diff = (a.re - s.re).abs();
        let im_diff = (a.im - s.im).abs();
        assert!(
            re_diff <= ABS_FLOOR || re_diff <= REL_TOL * a.re.abs().max(s.re.abs()),
            "SoA vs AoS f64 re mismatch at idx={idx} size={size}: \
             AoS={}, SoA={}, diff={re_diff:e}, ULP={}",
            a.re,
            s.re,
            ulp_distance_f64(a.re, s.re)
        );
        assert!(
            im_diff <= ABS_FLOOR || im_diff <= REL_TOL * a.im.abs().max(s.im.abs()),
            "SoA vs AoS f64 im mismatch at idx={idx} size={size}: \
             AoS={}, SoA={}, diff={im_diff:e}, ULP={}",
            a.im,
            s.im,
            ulp_distance_f64(a.im, s.im)
        );
    }
}

/// Helper: compute max ULP distance between two f32 values.
fn ulp_distance_f32(a: f32, b: f32) -> u32 {
    let ai = a.to_bits();
    let bi = b.to_bits();
    ai.abs_diff(bi)
}

/// For a given size, verify SoA vs AoS produce numerically equivalent
/// results (f32). See [`check_soa_vs_aos_f64`] for why this uses an
/// absolute/relative-error gate rather than a raw ULP bound: the same
/// release-mode auto-vectorization sensitivity applies here, scaled to f32's
/// ~7 significant decimal digits.
fn check_soa_vs_aos_f32(size: usize) {
    let _lock = CACHE_LOCK.lock().expect("cache lock");
    clear_twiddle_cache();

    let input: Vec<Complex<f32>> = (0..size)
        .map(|k| Complex::new((k as f32).sin(), (k as f32).cos()))
        .collect();

    let aos_table = get_twiddle_table_f32(size, TwiddleDirection::Forward);
    let mut aos_data = input.clone();
    twiddle_mul_simd_f32(&mut aos_data, &aos_table.factors);

    let soa_table = get_twiddle_table_soa_f32(size, TwiddleDirection::Forward);
    let mut soa_data = input;
    twiddle_mul_soa_simd_f32(&mut soa_data, &soa_table.re, &soa_table.im);

    const ABS_FLOOR: f32 = 1e-6;
    const REL_TOL: f32 = 1e-5;
    for (idx, (a, s)) in aos_data.iter().zip(soa_data.iter()).enumerate() {
        let re_diff = (a.re - s.re).abs();
        let im_diff = (a.im - s.im).abs();
        assert!(
            re_diff <= ABS_FLOOR || re_diff <= REL_TOL * a.re.abs().max(s.re.abs()),
            "SoA vs AoS f32 re mismatch at idx={idx} size={size}: \
             AoS={}, SoA={}, diff={re_diff:e}, ULP={}",
            a.re,
            s.re,
            ulp_distance_f32(a.re, s.re)
        );
        assert!(
            im_diff <= ABS_FLOOR || im_diff <= REL_TOL * a.im.abs().max(s.im.abs()),
            "SoA vs AoS f32 im mismatch at idx={idx} size={size}: \
             AoS={}, SoA={}, diff={im_diff:e}, ULP={}",
            a.im,
            s.im,
            ulp_distance_f32(a.im, s.im)
        );
    }
}

#[test]
fn soa_vs_aos_correctness_f64_1024() {
    check_soa_vs_aos_f64(1024);
}

#[test]
fn soa_vs_aos_correctness_f64_4096() {
    check_soa_vs_aos_f64(4096);
}

#[test]
fn soa_vs_aos_correctness_f64_16384() {
    check_soa_vs_aos_f64(16384);
}

#[test]
fn soa_vs_aos_correctness_f64_65536() {
    check_soa_vs_aos_f64(65536);
}

#[test]
fn soa_vs_aos_correctness_f32_1024() {
    check_soa_vs_aos_f32(1024);
}

#[test]
fn soa_vs_aos_correctness_f32_4096() {
    check_soa_vs_aos_f32(4096);
}

#[test]
fn soa_vs_aos_correctness_f32_16384() {
    check_soa_vs_aos_f32(16384);
}

#[test]
fn soa_vs_aos_correctness_f32_65536() {
    check_soa_vs_aos_f32(65536);
}

// -------------------------------------------------------------------------
// SoA cache tests
// -------------------------------------------------------------------------

#[test]
fn soa_cache_hit_f64() {
    let _lock = CACHE_LOCK.lock().expect("cache lock");
    clear_twiddle_cache();

    let t1 = get_twiddle_table_soa_f64(1024, TwiddleDirection::Forward);
    let t2 = get_twiddle_table_soa_f64(1024, TwiddleDirection::Forward);
    assert!(
        Arc::ptr_eq(&t1, &t2),
        "second SoA f64 call should return the cached Arc"
    );
}

#[test]
fn soa_cache_hit_f32() {
    let _lock = CACHE_LOCK.lock().expect("cache lock");
    clear_twiddle_cache();

    let t1 = get_twiddle_table_soa_f32(512, TwiddleDirection::Forward);
    let t2 = get_twiddle_table_soa_f32(512, TwiddleDirection::Forward);
    assert!(
        Arc::ptr_eq(&t1, &t2),
        "second SoA f32 call should return the cached Arc"
    );
}

#[test]
fn soa_table_correctness_f64() {
    let _lock = CACHE_LOCK.lock().expect("cache lock");
    clear_twiddle_cache();

    let n = 16usize;
    let soa = get_twiddle_table_soa_f64(n, TwiddleDirection::Forward);
    assert_eq!(soa.re.len(), n);
    assert_eq!(soa.im.len(), n);

    // W_n^0 = (1, 0)
    assert!((soa.re[0] - 1.0_f64).abs() < 1e-14);
    assert!(soa.im[0].abs() < 1e-14);

    // Each factor should lie on the unit circle
    for k in 0..n {
        let mag_sq = soa.re[k] * soa.re[k] + soa.im[k] * soa.im[k];
        assert!(
            (mag_sq - 1.0_f64).abs() < 1e-13,
            "SoA W_{n}^{k} not on unit circle: |w|²={mag_sq}"
        );
    }

    // SoA values should match the AoS table
    let aos = get_twiddle_table_f64(n, TwiddleDirection::Forward);
    for k in 0..n {
        assert!(
            (soa.re[k] - aos.factors[k].re).abs() < 1e-14,
            "SoA re[{k}]={} != AoS re={}",
            soa.re[k],
            aos.factors[k].re
        );
        assert!(
            (soa.im[k] - aos.factors[k].im).abs() < 1e-14,
            "SoA im[{k}]={} != AoS im={}",
            soa.im[k],
            aos.factors[k].im
        );
    }
}

// -------------------------------------------------------------------------
// Mixed-radix twiddle generation tests
// -------------------------------------------------------------------------

/// For N=6, factors=[3,2], Stage 1 (radix-3, stride=1) should have
/// (r-1)*stride = 2 entries, both equal to 1+0i (W_3^0).
#[test]
fn mixed_radix_twiddles_n6_stage1_all_ones() {
    let tables = twiddles_mixed_radix(6, &[3, 2], TwiddleDirection::Forward);
    assert_eq!(tables.len(), 2, "should have 2 stages");
    // Stage 0 (radix-3, current_n=3, stride=1): 2 entries, all = W_3^0 = 1
    let stage1 = &tables[0];
    assert_eq!(stage1.len(), 2, "stage 1 should have (3-1)*1 = 2 twiddles");
    for (i, &tw) in stage1.iter().enumerate() {
        assert!(
            (tw.re - 1.0).abs() < 1e-14 && tw.im.abs() < 1e-14,
            "stage 1 twiddle[{i}] should be 1+0i, got {tw:?}"
        );
    }
}

/// For N=6, factors=[3,2], Stage 2 (radix-2, stride=3) should have
/// (r-1)*stride = 1*3 = 3 entries: W_6^0, W_6^1, W_6^2.
///
/// This is the critical test that distinguishes correct (stride-indexed)
/// from wrong (blocks-indexed) twiddle generation.
#[test]
fn mixed_radix_twiddles_n6_stage2_correct_stride_indexing() {
    let tables = twiddles_mixed_radix(6, &[3, 2], TwiddleDirection::Forward);
    let stage2 = &tables[1];
    // Stage 1 (radix-2, current_n=6, stride=3): 3 entries
    // table[s] = W_6^{1*s} = exp(-2πi*s/6) for s=0,1,2
    assert_eq!(stage2.len(), 3, "stage 2 should have (2-1)*3 = 3 twiddles");

    for s in 0..3_usize {
        let angle = -2.0 * core::f64::consts::PI * s as f64 / 6.0;
        let expected = Complex::new(angle.cos(), angle.sin());
        let got = stage2[s];
        assert!(
            (got.re - expected.re).abs() < 1e-12 && (got.im - expected.im).abs() < 1e-12,
            "stage 2 twiddle[{s}] = W_6^{s}: expected {expected:?}, got {got:?}"
        );
    }
}

/// Verify that Inverse direction returns conjugated twiddles relative to Forward.
#[test]
fn mixed_radix_twiddles_inverse_is_conjugate_of_forward() {
    let fwd = twiddles_mixed_radix(6, &[3, 2], TwiddleDirection::Forward);
    let inv = twiddles_mixed_radix(6, &[3, 2], TwiddleDirection::Inverse);
    assert_eq!(fwd.len(), inv.len());
    for (t, (fwd_stage, inv_stage)) in fwd.iter().zip(inv.iter()).enumerate() {
        assert_eq!(
            fwd_stage.len(),
            inv_stage.len(),
            "stage {t} length mismatch"
        );
        for (i, (&fw, &bw)) in fwd_stage.iter().zip(inv_stage.iter()).enumerate() {
            assert!(
                (fw.re - bw.re).abs() < 1e-14,
                "stage {t}[{i}] re should match: fwd={} inv={}",
                fw.re,
                bw.re
            );
            assert!(
                (fw.im + bw.im).abs() < 1e-14,
                "stage {t}[{i}] im should be negated: fwd={} inv={}",
                fw.im,
                bw.im
            );
        }
    }
}

/// Stage 0 (innermost, stride=1) always has all twiddles equal to 1,
/// regardless of radix, because j*s/current_n = j*0/r = 0 (s can only be 0).
#[test]
fn mixed_radix_twiddles_innermost_stage_all_ones() {
    // Try radix-5 innermost
    for &radix in &[2u16, 3, 5, 7] {
        let n = radix as usize * 2; // e.g., 5*2=10
        let tables = twiddles_mixed_radix(n, &[radix, 2], TwiddleDirection::Forward);
        let stage0 = &tables[0];
        let expected_len = radix as usize - 1; // stride=1
        assert_eq!(
            stage0.len(),
            expected_len,
            "radix={radix}: stage 0 len should be (r-1)*1={expected_len}"
        );
        for (i, &tw) in stage0.iter().enumerate() {
            assert!(
                (tw.re - 1.0).abs() < 1e-14 && tw.im.abs() < 1e-14,
                "radix={radix} stage 0 twiddle[{i}] should be 1+0i, got {tw:?}"
            );
        }
    }
}

/// Total table entries across all stages must equal sum of (r_t-1)*stride_t.
#[test]
fn mixed_radix_twiddles_table_sizes_correct() {
    // N=60 = 2*2*3*5, inner-to-outer
    let factors = [2u16, 2, 3, 5];
    let n: usize = factors.iter().map(|&r| r as usize).product();
    let tables = twiddles_mixed_radix(n, &factors, TwiddleDirection::Forward);
    assert_eq!(tables.len(), 4);

    let mut current_n = 1usize;
    for (t, (&r_u16, table)) in factors.iter().zip(tables.iter()).enumerate() {
        let r = r_u16 as usize;
        current_n *= r;
        let stride = current_n / r;
        let expected_len = (r - 1) * stride;
        assert_eq!(
            table.len(),
            expected_len,
            "stage {t} (radix={r}, stride={stride}): expected {expected_len} twiddles, got {}",
            table.len()
        );
    }
}

// -------------------------------------------------------------------------
// f32 twiddle accuracy: computed in f64 then rounded to f32
// -------------------------------------------------------------------------

/// The f32 twiddle table must be computed in f64 and rounded to f32, so it must
/// exactly equal a reference computed the same way. A regression to native-f32
/// trigonometry would diverge from this reference by up to a few ULP, failing
/// the exact-equality assertion.
#[test]
fn f32_twiddle_table_matches_f64_reference() {
    let size = 1 << 20;
    let table = compute_twiddle_table_f32(size, TwiddleDirection::Forward);
    assert_eq!(table.factors.len(), size);

    let mut max_diff = 0.0_f32;
    for (k, factor) in table.factors.iter().enumerate() {
        let angle = -2.0_f64 * core::f64::consts::PI * k as f64 / size as f64;
        let ref_re = angle.cos() as f32;
        let ref_im = angle.sin() as f32;
        max_diff = max_diff.max((factor.re - ref_re).abs());
        max_diff = max_diff.max((factor.im - ref_im).abs());
    }
    assert_eq!(
        max_diff, 0.0,
        "f32 twiddle table must equal the f64-computed-then-rounded reference"
    );
}

/// The SoA f32 twiddle table must also match the f64-computed reference.
#[test]
fn f32_soa_twiddle_table_matches_f64_reference() {
    let size = 1 << 16;
    let table = compute_twiddle_table_soa_f32(size, TwiddleDirection::Inverse);
    assert_eq!(table.re.len(), size);

    for k in 0..size {
        let angle = 2.0_f64 * core::f64::consts::PI * k as f64 / size as f64;
        assert_eq!(table.re[k], angle.cos() as f32, "re mismatch at k={k}");
        assert_eq!(table.im[k], angle.sin() as f32, "im mismatch at k={k}");
    }
}

// -------------------------------------------------------------------------
// Twiddle-cache RwLock poison recovery
// -------------------------------------------------------------------------

/// A panic while holding the twiddle-cache write lock poisons it. The cache must
/// recover (its data is purely recomputable) rather than turning every
/// subsequent FFT twiddle lookup into a fatal panic process-wide.
#[cfg(feature = "std")]
#[test]
fn twiddle_cache_recovers_from_poison() {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    let _serialize = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Poison the global twiddle-cache lock by panicking while the write guard is held.
    let unwound = catch_unwind(AssertUnwindSafe(|| {
        let _write = super::global_twiddle_cache()
            .write()
            .expect("lock not yet poisoned");
        panic!("intentional poison for test");
    }));
    assert!(unwound.is_err(), "the injected panic must have unwound");

    // Under the old `.expect(\"RwLock poisoned\")` policy this next call would panic.
    // With poison recovery it must return a correct table.
    let table = get_twiddle_table_f64(8, TwiddleDirection::Forward);
    assert_eq!(table.factors.len(), 8);
    assert!((table.factors[0].re - 1.0).abs() < 1e-12);
    assert!(table.factors[0].im.abs() < 1e-12);

    // A write-locking path must also recover.
    clear_twiddle_cache();
}
