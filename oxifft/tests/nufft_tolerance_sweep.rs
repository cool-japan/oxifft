//! NUFFT Type 1 / Type 2 tolerance × oversampling × size sweep.
//!
//! Sweeps a small parameter grid and compares NUFFT output against a dense
//! NDFT reference.  Each cell asserts that the maximum relative error is at
//! most `tolerance × 10` (10× headroom over the user-specified tolerance).
//!
//! The grid is deliberately small (`N ∈ {64,256}`, `M ∈ {100,1000}`,
//! `tol ∈ {1e-3,1e-6}`, `oversampling ∈ {1.5,2.0}`) so the test finishes in
//! well under 60 s with `--release`.
//!
//! # Algorithm
//!
//! The Gaussian NUFFT uses a spreading kernel `exp(-β·(j/W)²)` with `β = 2.3·W`
//! (where `W = kernel_width/2` is the integer half-width).  Deconvolution is
//! performed using the exact discrete kernel DFT (cosine sum over the support),
//! with a phase correction of `(-1)^k` to undo the coordinate shift from
//! `[-π,π]` to `[0,2π]`.  The kernel width is chosen via:
//!
//! ```text
//! W = ceil( -log10(tol) · (2 - σ/2) )
//! kw = 2·W
//! ```
//!
//! where σ is the oversampling ratio.  This accounts for the reduced guard-band
//! isolation at lower oversampling ratios.

#![cfg(feature = "std")]

use oxifft::nufft::{Nufft, NufftError, NufftOptions, NufftType};
use oxifft::Complex;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Splitmix64 PRNG (no external dependency)
// ---------------------------------------------------------------------------

const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Map a `u64` PRNG output to `f64` in `[-1.0, 1.0]` without precision-loss casts.
///
/// Constructs an IEEE 754 f64 in `[1.0, 2.0)` from the top 52 mantissa bits,
/// then maps to `[-1.0, 1.0]`.  No integer-to-float precision is lost.
fn u64_to_f64_bipolar(bits: u64) -> f64 {
    // Extract the top 52 bits (f64 mantissa width).
    let mantissa = bits >> 12;
    // Build exponent-1023 = 0 (i.e. value in [1.0, 2.0)).
    let ieee_bits = 0x3ff0_0000_0000_0000_u64 | mantissa;
    // Map [1.0, 2.0) → [-1.0, 1.0).
    f64::from_bits(ieee_bits) * 2.0 - 3.0
}

/// Generate `m` non-uniform points in `(-π, π)`, clamped away from the boundary
/// to avoid fp-rounding ejection by the range guard in `Nufft::new`.
fn gen_nonuniform_points(m: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..m)
        .map(|_| {
            let bits = splitmix64(&mut state);
            // Map to (-1, 1) then scale to (-π, π).
            let x = u64_to_f64_bipolar(bits) * PI;
            // Clamp defensively to stay strictly within [-π, π].
            x.clamp(-PI + 1e-12, PI - 1e-12)
        })
        .collect()
}

/// Generate `n` complex values with components in `[-1, 1]`.
fn gen_random_complex(n: usize, seed: u64) -> Vec<Complex<f64>> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            let re = u64_to_f64_bipolar(splitmix64(&mut state));
            let im = u64_to_f64_bipolar(splitmix64(&mut state));
            Complex::new(re, im)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Dense NDFT reference implementations
// ---------------------------------------------------------------------------

/// Dense NDFT Type 1 (non-uniform → uniform, O(N*M)).
///
/// `f_hat[k] = sum_{j=0}^{M-1}  c[j] * exp(-i * freq * x[j])`
///
/// where `freq ∈ [-N/2, N/2)` and `k` goes from `0` to `N-1`.
///
/// Grid sizes ≤ 256, so all casts through `i32::try_from` succeed.
fn dense_ndft_type1(points: &[f64], values: &[Complex<f64>], n: usize) -> Vec<Complex<f64>> {
    let n_i32 = i32::try_from(n).expect("n fits i32; grid max is 256");
    let half_i32 = n_i32 / 2;
    (0..n)
        .map(|k| {
            let freq_i32 = i32::try_from(k).expect("k fits i32") - half_i32;
            let freq = f64::from(freq_i32);
            values
                .iter()
                .zip(points.iter())
                .fold(Complex::new(0.0, 0.0), |acc, (&val, &xj)| {
                    let angle = -freq * xj;
                    acc + val * Complex::new(angle.cos(), angle.sin())
                })
        })
        .collect()
}

/// Dense NDFT Type 2 (uniform → non-uniform, O(N*M)).
///
/// `f[j] = sum_{k=-N/2}^{N/2-1}  f_hat[k] * exp(i * freq * x[j])`
///
/// Grid sizes ≤ 256, so casts through `i32::try_from` succeed.
fn dense_ndft_type2(fhat: &[Complex<f64>], points: &[f64]) -> Vec<Complex<f64>> {
    let n = fhat.len();
    let n_i32 = i32::try_from(n).expect("n fits i32; grid max is 256");
    let half_i32 = n_i32 / 2;
    points
        .iter()
        .map(|&xj| {
            (0..n).fold(Complex::new(0.0, 0.0), |acc, k| {
                let freq_i32 = i32::try_from(k).expect("k fits i32") - half_i32;
                let freq = f64::from(freq_i32);
                let angle = freq * xj;
                acc + fhat[k] * Complex::new(angle.cos(), angle.sin())
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Max relative error helper
// ---------------------------------------------------------------------------

/// Compute `max |nufft[i] - ref[i]| / max(|ref[i]|)` over all `i`.
/// Returns `0.0` if all reference values are near-zero.
fn max_relative_error(nufft_out: &[Complex<f64>], reference: &[Complex<f64>]) -> f64 {
    let ref_max = reference.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);

    if ref_max < 1e-30 {
        return 0.0;
    }

    nufft_out
        .iter()
        .zip(reference.iter())
        .map(|(n, r)| (*n - *r).norm() / ref_max)
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// Sweep parameter grid
// ---------------------------------------------------------------------------

/// Attempt to build a Nufft plan with given oversampling, returning `None` if
/// the combination is unsupported.
fn try_build_plan(
    nufft_type: NufftType,
    n: usize,
    points: &[f64],
    tol: f64,
    oversampling: f64,
) -> Option<Nufft<f64>> {
    let options = NufftOptions {
        oversampling,
        tolerance: tol,
        ..Default::default()
    };
    match Nufft::with_options(nufft_type, n, points, &options) {
        Ok(plan) => Some(plan),
        Err(NufftError::InvalidSize(_) | NufftError::PlanFailed) => None,
        Err(e) => {
            // Unexpected error — propagate as a test failure.
            panic!(
                "Unexpected NufftError building plan (n={n}, tol={tol}, os={oversampling}): {e}"
            );
        }
    }
}

/// NUFFT Type 1 tolerance sweep.
///
/// To run manually:
/// `cargo test --release -p oxifft --features std nufft_tolerance_sweep_type1`
#[test]
fn nufft_tolerance_sweep_type1() {
    // Grid parameters — kept small so the dense NDFT finishes quickly.
    let n_sizes: &[usize] = &[64, 256];
    let m_sizes: &[usize] = &[100, 1000];
    let tolerances: &[f64] = &[1e-3, 1e-6];
    let oversamplings: &[f64] = &[1.5, 2.0];

    let total_cells = n_sizes.len() * m_sizes.len() * tolerances.len() * oversamplings.len();
    let mut tested = 0usize;
    let mut passed = 0usize;
    let mut skipped = 0usize;

    for &n in n_sizes {
        for &m in m_sizes {
            // Seeds derived from (n,m) for independent inputs per cell.
            let base_pts: u64 = 0x0000_abcd_ef01_2345;
            let base_vals: u64 = 0x0000_fedc_ba54_3210;
            let seed_pts = base_pts
                .wrapping_add((n as u64) << 16)
                .wrapping_add(m as u64);
            let seed_vals = base_vals
                .wrapping_add((n as u64) << 16)
                .wrapping_add(m as u64);

            let points = gen_nonuniform_points(m, seed_pts);
            let values = gen_random_complex(m, seed_vals);

            // Dense reference (computed once per (n,m)).
            let reference = dense_ndft_type1(&points, &values, n);

            for &tol in tolerances {
                for &os in oversamplings {
                    tested += 1;

                    let Some(plan) = try_build_plan(NufftType::Type1, n, &points, tol, os) else {
                        skipped += 1;
                        eprintln!("[type1 SKIP] n={n} m={m} tol={tol:.0e} os={os}");
                        continue;
                    };

                    let nufft_out = match plan.type1(&values) {
                        Ok(out) => out,
                        Err(e) => {
                            panic!(
                                "[type1 FAIL] n={n} m={m} tol={tol:.0e} os={os}: \
                                 execution error: {e}"
                            );
                        }
                    };

                    assert_eq!(
                        nufft_out.len(),
                        n,
                        "Type1 output length mismatch: expected {n}, got {}",
                        nufft_out.len()
                    );

                    let rel_err = max_relative_error(&nufft_out, &reference);
                    let headroom = tol * 10.0;

                    println!(
                        "[type1] n={n} m={m} tol={tol:.0e} os={os} \
                         rel_err={rel_err:.2e} limit={headroom:.2e} {}",
                        if rel_err <= headroom { "PASS" } else { "FAIL" }
                    );

                    assert!(
                        rel_err <= headroom,
                        "Type1 NUFFT relative error {rel_err:.2e} exceeds \
                         tolerance×10 = {headroom:.2e} \
                         (n={n}, m={m}, tol={tol:.0e}, os={os})"
                    );

                    passed += 1;
                }
            }
        }
    }

    println!(
        "[nufft_tolerance_sweep_type1] cells={total_cells} \
         tested={tested} passed={passed} skipped={skipped}"
    );

    assert!(
        passed > 0,
        "No Type1 cells were successfully tested — check NUFFT feature gate"
    );
}

/// NUFFT Type 2 tolerance sweep.
///
/// To run manually:
/// `cargo test --release -p oxifft --features std nufft_tolerance_sweep_type2`
#[test]
fn nufft_tolerance_sweep_type2() {
    let n_sizes: &[usize] = &[64, 256];
    let m_sizes: &[usize] = &[100, 1000];
    let tolerances: &[f64] = &[1e-3, 1e-6];
    let oversamplings: &[f64] = &[1.5, 2.0];

    let total_cells = n_sizes.len() * m_sizes.len() * tolerances.len() * oversamplings.len();
    let mut tested = 0usize;
    let mut passed = 0usize;
    let mut skipped = 0usize;

    for &n in n_sizes {
        for &m in m_sizes {
            let base_pts: u64 = 0x0001_1111_aaaa_aaaa;
            let base_fhat: u64 = 0x0002_2222_bbbb_bbbb;
            let seed_pts = base_pts
                .wrapping_add((n as u64) << 16)
                .wrapping_add(m as u64);
            let seed_fhat = base_fhat
                .wrapping_add((n as u64) << 16)
                .wrapping_add(m as u64);

            let points = gen_nonuniform_points(m, seed_pts);
            let fhat = gen_random_complex(n, seed_fhat);

            // Dense reference.
            let reference = dense_ndft_type2(&fhat, &points);

            for &tol in tolerances {
                for &os in oversamplings {
                    tested += 1;

                    let Some(plan) = try_build_plan(NufftType::Type2, n, &points, tol, os) else {
                        skipped += 1;
                        eprintln!("[type2 SKIP] n={n} m={m} tol={tol:.0e} os={os}");
                        continue;
                    };

                    let nufft_out = match plan.type2(&fhat) {
                        Ok(out) => out,
                        Err(e) => {
                            panic!(
                                "[type2 FAIL] n={n} m={m} tol={tol:.0e} os={os}: \
                                 execution error: {e}"
                            );
                        }
                    };

                    assert_eq!(
                        nufft_out.len(),
                        m,
                        "Type2 output length mismatch: expected {m}, got {}",
                        nufft_out.len()
                    );

                    let rel_err = max_relative_error(&nufft_out, &reference);
                    let headroom = tol * 10.0;

                    println!(
                        "[type2] n={n} m={m} tol={tol:.0e} os={os} \
                         rel_err={rel_err:.2e} limit={headroom:.2e} {}",
                        if rel_err <= headroom { "PASS" } else { "FAIL" }
                    );

                    assert!(
                        rel_err <= headroom,
                        "Type2 NUFFT relative error {rel_err:.2e} exceeds \
                         tolerance×10 = {headroom:.2e} \
                         (n={n}, m={m}, tol={tol:.0e}, os={os})"
                    );

                    passed += 1;
                }
            }
        }
    }

    println!(
        "[nufft_tolerance_sweep_type2] cells={total_cells} \
         tested={tested} passed={passed} skipped={skipped}"
    );

    assert!(
        passed > 0,
        "No Type2 cells were successfully tested — check NUFFT feature gate"
    );
}
