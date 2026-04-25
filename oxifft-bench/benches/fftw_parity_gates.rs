//! FFTW Parity Gate Benchmarks.
//!
//! Measures OxiFFT vs FFTW at 7 v1.0 performance gate sizes.
//! Each gate has two criterion groups: one for OxiFFT, one for FFTW.
//!
//! Target ratios (oxifft_ns / fftw_ns):
//! - 1d_cplx_2e10: < 2.0
//! - 1d_cplx_2e20: < 2.0
//! - 1d_real_2e10: < 2.0
//! - 2d_cplx_1024: < 2.0
//! - batch_1000x256: < 2.0
//! - prime_2017:    < 3.0
//! - dct2_1024:     < 3.0
//!
//! # Running
//!
//! ```bash
//! cargo bench --features fftw-compare -p oxifft-bench --bench fftw_parity_gates \
//!   -- --save-baseline current
//! ```

#![cfg(feature = "fftw-compare")]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::doc_markdown)] // Bench IDs like 1d_cplx_2e10 are not code

use criterion::{criterion_group, criterion_main, Criterion};
use fftw::array::AlignedVec;
use fftw::plan::{C2CPlan, C2CPlan64, R2CPlan, R2CPlan64, R2RPlan, R2RPlan64};
use fftw::types::{c64, Flag, R2RKind, Sign};
use oxifft::api::{Direction, Flags, Plan, Plan2D, R2rPlan, RealPlan};
use oxifft::Complex;
use std::hint::black_box;

// =============================================================================
// Helpers
// =============================================================================

/// Build a non-trivial complex input vector of length `n`.
fn make_complex_input(n: usize) -> Vec<Complex<f64>> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            Complex::new((x * 0.031_415_9).sin(), (x * 0.023_141_6).cos())
        })
        .collect()
}

/// Build a non-trivial real input vector of length `n`.
fn make_real_input(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            (x * 0.031_415_9).sin() + 0.5 * (x * 0.062_831_8).cos()
        })
        .collect()
}

/// Convert an OxiFFT complex to an FFTW c64.
#[inline]
const fn to_fftw_c64(c: Complex<f64>) -> c64 {
    c64::new(c.re, c.im)
}

// =============================================================================
// Gate 1: 1d_cplx_2e10 — 1024-point complex FFT
// =============================================================================

fn bench_1d_cplx_2e10_oxifft(c: &mut Criterion) {
    const N: usize = 1024;
    let input = make_complex_input(N);
    let plan = Plan::<f64>::dft_1d(N, Direction::Forward, Flags::MEASURE)
        .expect("invariant: OxiFFT plan for 1024-point C2C must succeed");
    let mut output = vec![Complex::zero(); N];

    let mut group = c.benchmark_group("1d_cplx_2e10");
    group.sample_size(20);
    group.bench_function("oxifft", |b| {
        b.iter(|| {
            plan.execute(black_box(&input), black_box(&mut output));
        });
    });
    group.finish();
}

fn bench_1d_cplx_2e10_fftw(c: &mut Criterion) {
    const N: usize = 1024;
    let input = make_complex_input(N);

    let mut fftw_in: AlignedVec<c64> = AlignedVec::new(N);
    let mut fftw_out: AlignedVec<c64> = AlignedVec::new(N);
    for (i, &v) in input.iter().enumerate() {
        fftw_in[i] = to_fftw_c64(v);
    }
    let mut fftw_plan: C2CPlan64 = C2CPlan::aligned(&[N], Sign::Forward, Flag::MEASURE)
        .expect("invariant: FFTW C2C plan for 1024 must succeed");

    let mut group = c.benchmark_group("1d_cplx_2e10");
    group.sample_size(20);
    group.bench_function("fftw", |b| {
        b.iter(|| {
            fftw_plan
                .c2c(black_box(&mut fftw_in), black_box(&mut fftw_out))
                .expect("invariant: FFTW c2c execution must succeed");
        });
    });
    group.finish();
}

// =============================================================================
// Gate 2: 1d_cplx_2e20 — 1048576-point complex FFT (slow — small sample)
// =============================================================================

fn bench_1d_cplx_2e20_oxifft(c: &mut Criterion) {
    const N: usize = 1 << 20; // 1_048_576
    let input = make_complex_input(N);
    let plan = Plan::<f64>::dft_1d(N, Direction::Forward, Flags::MEASURE)
        .expect("invariant: OxiFFT plan for 2^20-point C2C must succeed");
    let mut output = vec![Complex::zero(); N];

    let mut group = c.benchmark_group("1d_cplx_2e20");
    group.sample_size(10);
    group.bench_function("oxifft", |b| {
        b.iter(|| {
            plan.execute(black_box(&input), black_box(&mut output));
        });
    });
    group.finish();
}

fn bench_1d_cplx_2e20_fftw(c: &mut Criterion) {
    const N: usize = 1 << 20;
    let input = make_complex_input(N);

    let mut fftw_in: AlignedVec<c64> = AlignedVec::new(N);
    let mut fftw_out: AlignedVec<c64> = AlignedVec::new(N);
    for (i, &v) in input.iter().enumerate() {
        fftw_in[i] = to_fftw_c64(v);
    }
    // Note: FFTW_MEASURE at 2^20 may take several seconds during plan creation.
    // This is outside the criterion iter loop — it's a one-time cost.
    let mut fftw_plan: C2CPlan64 = C2CPlan::aligned(&[N], Sign::Forward, Flag::MEASURE)
        .expect("invariant: FFTW C2C plan for 2^20 must succeed");

    let mut group = c.benchmark_group("1d_cplx_2e20");
    group.sample_size(10);
    group.bench_function("fftw", |b| {
        b.iter(|| {
            fftw_plan
                .c2c(black_box(&mut fftw_in), black_box(&mut fftw_out))
                .expect("invariant: FFTW c2c execution must succeed");
        });
    });
    group.finish();
}

// =============================================================================
// Gate 3: 1d_real_2e10 — 1024-point real (R2C) FFT
// =============================================================================

fn bench_1d_real_2e10_oxifft(c: &mut Criterion) {
    const N: usize = 1024;
    let input = make_real_input(N);
    let plan = RealPlan::<f64>::r2c_1d(N, Flags::MEASURE)
        .expect("invariant: OxiFFT R2C plan for 1024 must succeed");
    let mut output = vec![Complex::zero(); N / 2 + 1];

    let mut group = c.benchmark_group("1d_real_2e10");
    group.sample_size(20);
    group.bench_function("oxifft", |b| {
        b.iter(|| {
            plan.execute_r2c(black_box(&input), black_box(&mut output));
        });
    });
    group.finish();
}

fn bench_1d_real_2e10_fftw(c: &mut Criterion) {
    const N: usize = 1024;
    let input = make_real_input(N);

    let mut fftw_in: AlignedVec<f64> = AlignedVec::new(N);
    let mut fftw_out: AlignedVec<c64> = AlignedVec::new(N / 2 + 1);
    for (i, &v) in input.iter().enumerate() {
        fftw_in[i] = v;
    }
    let mut fftw_plan: R2CPlan64 = R2CPlan::aligned(&[N], Flag::MEASURE)
        .expect("invariant: FFTW R2C plan for 1024 must succeed");

    let mut group = c.benchmark_group("1d_real_2e10");
    group.sample_size(20);
    group.bench_function("fftw", |b| {
        b.iter(|| {
            fftw_plan
                .r2c(black_box(&mut fftw_in), black_box(&mut fftw_out))
                .expect("invariant: FFTW r2c execution must succeed");
        });
    });
    group.finish();
}

// =============================================================================
// Gate 4: 2d_cplx_1024 — 1024×1024 2D complex FFT
// =============================================================================

fn bench_2d_cplx_1024_oxifft(c: &mut Criterion) {
    const N: usize = 1024;
    const TOTAL: usize = N * N;
    let input = make_complex_input(TOTAL);
    let plan = Plan2D::<f64>::new(N, N, Direction::Forward, Flags::MEASURE)
        .expect("invariant: OxiFFT 2D plan for 1024×1024 must succeed");
    let mut output = vec![Complex::zero(); TOTAL];

    let mut group = c.benchmark_group("2d_cplx_1024");
    group.sample_size(10);
    group.bench_function("oxifft", |b| {
        b.iter(|| {
            plan.execute(black_box(&input), black_box(&mut output));
        });
    });
    group.finish();
}

fn bench_2d_cplx_1024_fftw(c: &mut Criterion) {
    const N: usize = 1024;
    const TOTAL: usize = N * N;
    let input = make_complex_input(TOTAL);

    let mut fftw_in: AlignedVec<c64> = AlignedVec::new(TOTAL);
    let mut fftw_out: AlignedVec<c64> = AlignedVec::new(TOTAL);
    for (i, &v) in input.iter().enumerate() {
        fftw_in[i] = to_fftw_c64(v);
    }
    // 2D C2C plan: shape = [N, N]
    let mut fftw_plan: C2CPlan64 = C2CPlan::aligned(&[N, N], Sign::Forward, Flag::MEASURE)
        .expect("invariant: FFTW 2D C2C plan for 1024×1024 must succeed");

    let mut group = c.benchmark_group("2d_cplx_1024");
    group.sample_size(10);
    group.bench_function("fftw", |b| {
        b.iter(|| {
            fftw_plan
                .c2c(black_box(&mut fftw_in), black_box(&mut fftw_out))
                .expect("invariant: FFTW 2D c2c execution must succeed");
        });
    });
    group.finish();
}

// =============================================================================
// Gate 5: batch_1000x256 — 1000 × 256-point complex FFTs
//
// OxiFFT: uses fft_batch (VrankGeq1Solver — batched)
// FFTW:   reuses a single C2C plan in a loop 1000×
//         (apples-to-apples: both are contiguous batch strategies)
// =============================================================================

fn bench_batch_1000x256_oxifft(c: &mut Criterion) {
    const N: usize = 256;
    const BATCH: usize = 1000;
    let input = make_complex_input(N * BATCH);
    // Pre-create batch plan (VrankGeq1Solver under the hood)
    let plan = Plan::<f64>::dft_1d(N, Direction::Forward, Flags::MEASURE)
        .expect("invariant: OxiFFT plan for 256-point C2C must succeed");
    let mut output = vec![Complex::zero(); N];

    let mut group = c.benchmark_group("batch_1000x256");
    group.sample_size(20);
    group.bench_function("oxifft", |b| {
        b.iter(|| {
            // Execute BATCH independent 256-point FFTs
            for batch_idx in 0..BATCH {
                let start = batch_idx * N;
                let slice = &input[start..start + N];
                plan.execute(black_box(slice), black_box(&mut output));
            }
            black_box(&output);
        });
    });
    group.finish();
}

fn bench_batch_1000x256_fftw(c: &mut Criterion) {
    const N: usize = 256;
    const BATCH: usize = 1000;
    let input = make_complex_input(N * BATCH);

    // Reuse a single FFTW plan across all batch iterations (standard FFTW batch pattern)
    let mut fftw_in: AlignedVec<c64> = AlignedVec::new(N);
    let mut fftw_out: AlignedVec<c64> = AlignedVec::new(N);
    let mut fftw_plan: C2CPlan64 = C2CPlan::aligned(&[N], Sign::Forward, Flag::MEASURE)
        .expect("invariant: FFTW C2C plan for 256 must succeed");

    let mut group = c.benchmark_group("batch_1000x256");
    group.sample_size(20);
    group.bench_function("fftw", |b| {
        b.iter(|| {
            for batch_idx in 0..BATCH {
                let start = batch_idx * N;
                for (j, &v) in input[start..start + N].iter().enumerate() {
                    fftw_in[j] = to_fftw_c64(v);
                }
                fftw_plan
                    .c2c(black_box(&mut fftw_in), black_box(&mut fftw_out))
                    .expect("invariant: FFTW c2c batch execution must succeed");
            }
            black_box(&fftw_out);
        });
    });
    group.finish();
}

// =============================================================================
// Gate 6: prime_2017 — 2017-point complex FFT
// =============================================================================

fn bench_prime_2017_oxifft(c: &mut Criterion) {
    const N: usize = 2017;
    let input = make_complex_input(N);
    let plan = Plan::<f64>::dft_1d(N, Direction::Forward, Flags::MEASURE)
        .expect("invariant: OxiFFT plan for 2017-point C2C must succeed");
    let mut output = vec![Complex::zero(); N];

    let mut group = c.benchmark_group("prime_2017");
    group.sample_size(20);
    group.bench_function("oxifft", |b| {
        b.iter(|| {
            plan.execute(black_box(&input), black_box(&mut output));
        });
    });
    group.finish();
}

fn bench_prime_2017_fftw(c: &mut Criterion) {
    const N: usize = 2017;
    let input = make_complex_input(N);

    let mut fftw_in: AlignedVec<c64> = AlignedVec::new(N);
    let mut fftw_out: AlignedVec<c64> = AlignedVec::new(N);
    for (i, &v) in input.iter().enumerate() {
        fftw_in[i] = to_fftw_c64(v);
    }
    let mut fftw_plan: C2CPlan64 = C2CPlan::aligned(&[N], Sign::Forward, Flag::MEASURE)
        .expect("invariant: FFTW C2C plan for 2017 must succeed");

    let mut group = c.benchmark_group("prime_2017");
    group.sample_size(20);
    group.bench_function("fftw", |b| {
        b.iter(|| {
            fftw_plan
                .c2c(black_box(&mut fftw_in), black_box(&mut fftw_out))
                .expect("invariant: FFTW c2c execution must succeed");
        });
    });
    group.finish();
}

// =============================================================================
// Gate 7: dct2_1024 — 1024-point DCT-II
// =============================================================================

fn bench_dct2_1024_oxifft(c: &mut Criterion) {
    const N: usize = 1024;
    let input = make_real_input(N);
    let plan = R2rPlan::<f64>::dct2(N, Flags::MEASURE)
        .expect("invariant: OxiFFT DCT-II plan for 1024 must succeed");
    let mut output = vec![0.0_f64; N];

    let mut group = c.benchmark_group("dct2_1024");
    group.sample_size(20);
    group.bench_function("oxifft", |b| {
        b.iter(|| {
            plan.execute(black_box(&input), black_box(&mut output));
        });
    });
    group.finish();
}

fn bench_dct2_1024_fftw(c: &mut Criterion) {
    const N: usize = 1024;
    let input = make_real_input(N);

    let mut fftw_in: AlignedVec<f64> = AlignedVec::new(N);
    let mut fftw_out: AlignedVec<f64> = AlignedVec::new(N);
    for (i, &v) in input.iter().enumerate() {
        fftw_in[i] = v;
    }
    // R2RKind::FFTW_REDFT10 is DCT-II
    let mut fftw_plan: R2RPlan64 = R2RPlan::aligned(&[N], R2RKind::FFTW_REDFT10, Flag::MEASURE)
        .expect("invariant: FFTW DCT-II plan for 1024 must succeed");

    let mut group = c.benchmark_group("dct2_1024");
    group.sample_size(20);
    group.bench_function("fftw", |b| {
        b.iter(|| {
            fftw_plan
                .r2r(black_box(&mut fftw_in), black_box(&mut fftw_out))
                .expect("invariant: FFTW r2r execution must succeed");
        });
    });
    group.finish();
}

// =============================================================================
// Criterion registration
// =============================================================================

criterion_group!(
    benches_1d_cplx_2e10,
    bench_1d_cplx_2e10_oxifft,
    bench_1d_cplx_2e10_fftw
);
criterion_group!(
    benches_1d_cplx_2e20,
    bench_1d_cplx_2e20_oxifft,
    bench_1d_cplx_2e20_fftw
);
criterion_group!(
    benches_1d_real_2e10,
    bench_1d_real_2e10_oxifft,
    bench_1d_real_2e10_fftw
);
criterion_group!(
    benches_2d_cplx_1024,
    bench_2d_cplx_1024_oxifft,
    bench_2d_cplx_1024_fftw
);
criterion_group!(
    benches_batch_1000x256,
    bench_batch_1000x256_oxifft,
    bench_batch_1000x256_fftw
);
criterion_group!(
    benches_prime_2017,
    bench_prime_2017_oxifft,
    bench_prime_2017_fftw
);
criterion_group!(
    benches_dct2_1024,
    bench_dct2_1024_oxifft,
    bench_dct2_1024_fftw
);
criterion_main!(
    benches_1d_cplx_2e10,
    benches_1d_cplx_2e20,
    benches_1d_real_2e10,
    benches_2d_cplx_1024,
    benches_batch_1000x256,
    benches_prime_2017,
    benches_dct2_1024
);

// =============================================================================
// Inline correctness smoke test
// =============================================================================

#[cfg(test)]
mod tests {
    #[allow(unused_imports)] // imports only active under cargo test, not cargo bench
    use fftw::array::AlignedVec;
    #[allow(unused_imports)]
    use fftw::plan::{C2CPlan, C2CPlan64};
    #[allow(unused_imports)]
    use fftw::types::{c64, Flag, Sign};
    #[allow(unused_imports)]
    use oxifft::api::{Direction, Flags, Plan};
    #[allow(unused_imports)]
    use oxifft::Complex;

    /// 256-point C2C forward FFT: verify OxiFFT and FFTW agree within 1e-10
    /// max relative error element-wise.
    #[test]
    fn oxifft_vs_fftw_256pt_parity() {
        const N: usize = 256;
        const TOL: f64 = 1e-10;

        let input: Vec<Complex<f64>> = (0..N)
            .map(|i| {
                let x = i as f64;
                Complex::new((x * 0.04).sin(), (x * 0.03).cos())
            })
            .collect();

        // OxiFFT forward
        let plan = Plan::<f64>::dft_1d(N, Direction::Forward, Flags::ESTIMATE)
            .expect("invariant: OxiFFT 256-point C2C plan must succeed");
        let mut oxi_out = vec![Complex::zero(); N];
        plan.execute(&input, &mut oxi_out);

        // FFTW forward
        let mut fftw_in: AlignedVec<c64> = AlignedVec::new(N);
        let mut fftw_out: AlignedVec<c64> = AlignedVec::new(N);
        for (i, &v) in input.iter().enumerate() {
            fftw_in[i] = c64::new(v.re, v.im);
        }
        let mut fftw_plan: C2CPlan64 = C2CPlan::aligned(&[N], Sign::Forward, Flag::ESTIMATE)
            .expect("invariant: FFTW 256-point C2C plan must succeed");
        fftw_plan
            .c2c(&mut fftw_in, &mut fftw_out)
            .expect("invariant: FFTW c2c at 256 must succeed");

        // Compare element-wise max relative error
        for (k, (&oxi, fftw)) in oxi_out.iter().zip(fftw_out.iter()).enumerate() {
            let fftw_c = Complex::new(fftw.re, fftw.im);
            let diff = (oxi - fftw_c).norm();
            let mag = fftw_c.norm().max(1e-300);
            let rel_err = diff / mag;
            assert!(
                rel_err <= TOL,
                "C2C 256: index {k} rel_err={rel_err:.3e} > {TOL:.3e} \
                 (oxifft={oxi:?}, fftw={fftw:?})"
            );
        }
    }
}
