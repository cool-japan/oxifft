//! Correctness smoke test: `OxiFFT` vs FFTW on a 256-point C2C forward FFT.
//!
//! Verifies that `OxiFFT` and FFTW agree within a max relative error of 1e-10
//! element-wise on a non-trivial complex input.
//!
//! # Running
//!
//! ```bash
//! cargo test --features fftw-compare -p oxifft-bench --test fftw_parity_smoke
//! ```

#![cfg(feature = "fftw-compare")]
#![allow(clippy::cast_precision_loss)] // Index arithmetic; values always << 2^53

use fftw::array::AlignedVec;
use fftw::plan::{C2CPlan, C2CPlan64};
use fftw::types::{c64, Flag, Sign};
use oxifft::api::{Direction, Flags, Plan};
use oxifft::Complex;

/// Verify that `OxiFFT` and FFTW agree on a 256-point C2C forward FFT
/// within a max relative error of 1e-10 element-wise.
#[test]
fn smoke_c2c_256_matches_fftw() {
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
        .expect("invariant: OxiFFT plan must succeed at 256");
    let mut oxi_out = vec![Complex::zero(); N];
    plan.execute(&input, &mut oxi_out);

    // FFTW forward
    let mut fftw_in: AlignedVec<c64> = AlignedVec::new(N);
    let mut fftw_out: AlignedVec<c64> = AlignedVec::new(N);
    for (i, &v) in input.iter().enumerate() {
        fftw_in[i] = c64::new(v.re, v.im);
    }
    let mut fftw_plan: C2CPlan64 = C2CPlan::aligned(&[N], Sign::Forward, Flag::ESTIMATE)
        .expect("invariant: FFTW C2C plan at 256 must succeed");
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
