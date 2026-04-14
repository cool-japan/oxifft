//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#![allow(clippy::approx_constant)] // Precomputed twiddle factors for FFT
#![allow(clippy::unreadable_literal)] // Precomputed FFT twiddle factors

use crate::dft::codelets::simd;
use crate::kernel::{Complex, Float};

use super::functions::{dft25, dft5, notw_12, notw_24, notw_36, notw_48};
use super::functions_2::{notw_60, notw_72, notw_96};
use super::functions_3::{notw_100, notw_15, notw_18, notw_20, notw_30, notw_45};

/// Optimized DFT of size 50.
///
/// Uses 2×25 mixed-radix decomposition.
/// sign: -1 for forward, +1 for inverse
#[inline]
pub fn notw_50<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 50);
    let sign_t = if sign < 0 { -T::ONE } else { T::ONE };
    let mut t: [Complex<T>; 50] = [Complex::zero(); 50];
    for j in 0..25 {
        let a0 = x[j];
        let a1 = x[j + 25];
        t[j] = a0 + a1;
        t[j + 25] = a0 - a1;
    }
    let cos_50: [T; 25] = [
        T::ONE,
        T::from_f64(0.9921147013144779),
        T::from_f64(0.9685831611286311),
        T::from_f64(0.9297764858882513),
        T::from_f64(0.8763066800438637),
        T::from_f64(0.8090169943749474),
        T::from_f64(0.7289686274214116),
        T::from_f64(0.6374239897486897),
        T::from_f64(0.5358267949789965),
        T::from_f64(0.4257792915650727),
        T::from_f64(0.30901699437494745),
        T::from_f64(0.18738131458572463),
        T::from_f64(0.06279051952931337),
        T::from_f64(-0.06279051952931337),
        T::from_f64(-0.18738131458572463),
        T::from_f64(-0.30901699437494745),
        T::from_f64(-0.4257792915650727),
        T::from_f64(-0.5358267949789965),
        T::from_f64(-0.6374239897486897),
        T::from_f64(-0.7289686274214116),
        T::from_f64(-0.8090169943749474),
        T::from_f64(-0.8763066800438637),
        T::from_f64(-0.9297764858882513),
        T::from_f64(-0.9685831611286311),
        T::from_f64(-0.9921147013144779),
    ];
    let sin_50: [T; 25] = [
        T::ZERO,
        T::from_f64(0.12533323356430426),
        T::from_f64(0.2486898871648548),
        T::from_f64(0.3681245526846779),
        T::from_f64(0.4817536741017153),
        T::from_f64(0.5877852522924731),
        T::from_f64(0.6845471059286887),
        T::from_f64(0.7705132427757893),
        T::from_f64(0.8443279255020151),
        T::from_f64(0.9048270524660195),
        T::from_f64(0.9510565162951535),
        T::from_f64(0.9822872507286887),
        T::from_f64(0.998026728428272),
        T::from_f64(0.998026728428272),
        T::from_f64(0.9822872507286887),
        T::from_f64(0.9510565162951535),
        T::from_f64(0.9048270524660195),
        T::from_f64(0.8443279255020151),
        T::from_f64(0.7705132427757893),
        T::from_f64(0.6845471059286887),
        T::from_f64(0.5877852522924731),
        T::from_f64(0.4817536741017153),
        T::from_f64(0.3681245526846779),
        T::from_f64(0.2486898871648548),
        T::from_f64(0.12533323356430426),
    ];
    for k2 in 1..25 {
        let c = cos_50[k2];
        let s = sin_50[k2];
        let idx = 25 + k2;
        let tw = t[idx];
        t[idx] = Complex::new(
            tw.re * c - sign_t * tw.im * s,
            sign_t * tw.re * s + tw.im * c,
        );
    }
    for k1 in 0..2 {
        let base = k1 * 25;
        let a: [Complex<T>; 25] = [
            t[base],
            t[base + 1],
            t[base + 2],
            t[base + 3],
            t[base + 4],
            t[base + 5],
            t[base + 6],
            t[base + 7],
            t[base + 8],
            t[base + 9],
            t[base + 10],
            t[base + 11],
            t[base + 12],
            t[base + 13],
            t[base + 14],
            t[base + 15],
            t[base + 16],
            t[base + 17],
            t[base + 18],
            t[base + 19],
            t[base + 20],
            t[base + 21],
            t[base + 22],
            t[base + 23],
            t[base + 24],
        ];
        let y = dft25(&a, sign_t);
        for k2 in 0..25 {
            x[k2 * 2 + k1] = y[k2];
        }
    }
}
/// Optimized DFT of size 80.
///
/// Uses 16×5 mixed-radix decomposition.
/// sign: -1 for forward, +1 for inverse
#[inline]
pub fn notw_80<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 80);
    let sign_t = if sign < 0 { -T::ONE } else { T::ONE };
    let mut t: [Complex<T>; 80] = [Complex::zero(); 80];
    for j in 0..5 {
        let mut row: [Complex<T>; 16] = [
            x[j],
            x[j + 5],
            x[j + 10],
            x[j + 15],
            x[j + 20],
            x[j + 25],
            x[j + 30],
            x[j + 35],
            x[j + 40],
            x[j + 45],
            x[j + 50],
            x[j + 55],
            x[j + 60],
            x[j + 65],
            x[j + 70],
            x[j + 75],
        ];
        simd::notw_16_dispatch(&mut row, sign);
        for k1 in 0..16 {
            t[k1 * 5 + j] = row[k1];
        }
    }
    let cos_80: [T; 61] = [
        T::ONE,
        T::from_f64(0.9969173337331281),
        T::from_f64(0.9876883405951378),
        T::from_f64(0.9723699203976766),
        T::from_f64(0.9510565162951535),
        T::from_f64(0.9238795325112867),
        T::from_f64(0.8910065241883679),
        T::from_f64(0.8526401643540922),
        T::from_f64(0.8090169943749474),
        T::from_f64(0.7604059656000309),
        T::from_f64(0.7071067811865476),
        T::from_f64(0.6494480483301837),
        T::from_f64(0.5877852522924731),
        T::from_f64(0.5224985647159488),
        T::from_f64(0.4539904997395468),
        T::from_f64(0.38268343236508984),
        T::from_f64(0.30901699437494745),
        T::from_f64(0.23344536385590525),
        T::from_f64(0.15643446504023092),
        T::from_f64(0.07845909572784494),
        T::ZERO,
        T::from_f64(-0.07845909572784494),
        T::from_f64(-0.15643446504023092),
        T::from_f64(-0.23344536385590525),
        T::from_f64(-0.30901699437494745),
        T::from_f64(-0.38268343236508984),
        T::from_f64(-0.4539904997395468),
        T::from_f64(-0.5224985647159488),
        T::from_f64(-0.5877852522924731),
        T::from_f64(-0.6494480483301837),
        T::from_f64(-0.7071067811865476),
        T::from_f64(-0.7604059656000309),
        T::from_f64(-0.8090169943749474),
        T::from_f64(-0.8526401643540922),
        T::from_f64(-0.8910065241883679),
        T::from_f64(-0.9238795325112867),
        T::from_f64(-0.9510565162951535),
        T::from_f64(-0.9723699203976766),
        T::from_f64(-0.9876883405951378),
        T::from_f64(-0.9969173337331281),
        -T::ONE,
        T::from_f64(-0.9969173337331281),
        T::from_f64(-0.9876883405951378),
        T::from_f64(-0.9723699203976766),
        T::from_f64(-0.9510565162951535),
        T::from_f64(-0.9238795325112867),
        T::from_f64(-0.8910065241883679),
        T::from_f64(-0.8526401643540922),
        T::from_f64(-0.8090169943749474),
        T::from_f64(-0.7604059656000309),
        T::from_f64(-0.7071067811865476),
        T::from_f64(-0.6494480483301837),
        T::from_f64(-0.5877852522924731),
        T::from_f64(-0.5224985647159488),
        T::from_f64(-0.4539904997395468),
        T::from_f64(-0.38268343236508984),
        T::from_f64(-0.30901699437494745),
        T::from_f64(-0.23344536385590525),
        T::from_f64(-0.15643446504023092),
        T::from_f64(-0.07845909572784494),
        T::ZERO,
    ];
    let sin_80: [T; 61] = [
        T::ZERO,
        T::from_f64(0.07845909572784494),
        T::from_f64(0.15643446504023087),
        T::from_f64(0.23344536385590525),
        T::from_f64(0.30901699437494745),
        T::from_f64(0.38268343236508984),
        T::from_f64(0.4539904997395468),
        T::from_f64(0.5224985647159488),
        T::from_f64(0.5877852522924731),
        T::from_f64(0.6494480483301837),
        T::from_f64(0.7071067811865476),
        T::from_f64(0.7604059656000309),
        T::from_f64(0.8090169943749474),
        T::from_f64(0.8526401643540922),
        T::from_f64(0.8910065241883679),
        T::from_f64(0.9238795325112867),
        T::from_f64(0.9510565162951535),
        T::from_f64(0.9723699203976766),
        T::from_f64(0.9876883405951378),
        T::from_f64(0.9969173337331281),
        T::ONE,
        T::from_f64(0.9969173337331281),
        T::from_f64(0.9876883405951378),
        T::from_f64(0.9723699203976766),
        T::from_f64(0.9510565162951535),
        T::from_f64(0.9238795325112867),
        T::from_f64(0.8910065241883679),
        T::from_f64(0.8526401643540922),
        T::from_f64(0.8090169943749474),
        T::from_f64(0.7604059656000309),
        T::from_f64(0.7071067811865476),
        T::from_f64(0.6494480483301837),
        T::from_f64(0.5877852522924731),
        T::from_f64(0.5224985647159488),
        T::from_f64(0.4539904997395468),
        T::from_f64(0.38268343236508984),
        T::from_f64(0.30901699437494745),
        T::from_f64(0.23344536385590525),
        T::from_f64(0.15643446504023087),
        T::from_f64(0.07845909572784494),
        T::ZERO,
        T::from_f64(-0.07845909572784494),
        T::from_f64(-0.15643446504023087),
        T::from_f64(-0.23344536385590525),
        T::from_f64(-0.30901699437494745),
        T::from_f64(-0.38268343236508984),
        T::from_f64(-0.4539904997395468),
        T::from_f64(-0.5224985647159488),
        T::from_f64(-0.5877852522924731),
        T::from_f64(-0.6494480483301837),
        T::from_f64(-0.7071067811865476),
        T::from_f64(-0.7604059656000309),
        T::from_f64(-0.8090169943749474),
        T::from_f64(-0.8526401643540922),
        T::from_f64(-0.8910065241883679),
        T::from_f64(-0.9238795325112867),
        T::from_f64(-0.9510565162951535),
        T::from_f64(-0.9723699203976766),
        T::from_f64(-0.9876883405951378),
        T::from_f64(-0.9969173337331281),
        -T::ONE,
    ];
    for k1 in 1..16 {
        for k2 in 1..5 {
            let k = k1 * k2;
            let c = cos_80[k];
            let s = sin_80[k];
            let idx = k1 * 5 + k2;
            let tw = t[idx];
            t[idx] = Complex::new(
                tw.re * c - sign_t * tw.im * s,
                sign_t * tw.re * s + tw.im * c,
            );
        }
    }
    for k1 in 0..16 {
        let base = k1 * 5;
        let a: [Complex<T>; 5] = [t[base], t[base + 1], t[base + 2], t[base + 3], t[base + 4]];
        let y = dft5(&a, sign_t);
        for k2 in 0..5 {
            x[k2 * 16 + k1] = y[k2];
        }
    }
}
/// Check if a composite codelet is available for size n.
#[inline]
#[must_use]
pub fn has_composite_codelet(n: usize) -> bool {
    matches!(
        n,
        12 | 15 | 18 | 20 | 24 | 30 | 36 | 45 | 48 | 50 | 60 | 72 | 80 | 96 | 100
    )
}
/// Execute composite codelet for size n.
/// Returns true if executed, false if no codelet available.
#[inline]
pub fn execute_composite_codelet<T: Float>(x: &mut [Complex<T>], n: usize, sign: i32) -> bool {
    match n {
        12 => {
            notw_12(x, sign);
            true
        }
        15 => {
            notw_15(x, sign);
            true
        }
        18 => {
            notw_18(x, sign);
            true
        }
        20 => {
            notw_20(x, sign);
            true
        }
        24 => {
            notw_24(x, sign);
            true
        }
        30 => {
            notw_30(x, sign);
            true
        }
        36 => {
            notw_36(x, sign);
            true
        }
        45 => {
            notw_45(x, sign);
            true
        }
        48 => {
            notw_48(x, sign);
            true
        }
        50 => {
            notw_50(x, sign);
            true
        }
        60 => {
            notw_60(x, sign);
            true
        }
        72 => {
            notw_72(x, sign);
            true
        }
        80 => {
            notw_80(x, sign);
            true
        }
        96 => {
            notw_96(x, sign);
            true
        }
        100 => {
            notw_100(x, sign);
            true
        }
        _ => false,
    }
}
#[cfg(test)]
#[allow(
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::redundant_clone
)]
mod tests {
    use super::*;
    use crate::dft::problem::Sign;
    use crate::dft::solvers::DirectSolver;
    #[allow(dead_code)]
    fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
        let err = ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)).sqrt();
        err < eps
    }
    fn test_composite_codelet(n: usize, codelet_fn: fn(&mut [Complex<f64>], i32)) {
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64).sin(), (i as f64 * 0.1).cos()))
            .collect();
        let mut ref_output = vec![Complex::zero(); n];
        DirectSolver::new().execute(&input, &mut ref_output, Sign::Forward);
        let mut comp_output = input.clone();
        codelet_fn(&mut comp_output, -1);
        let mut max_err = 0.0f64;
        let mut max_idx = 0;
        for (i, (r, c)) in ref_output.iter().zip(comp_output.iter()).enumerate() {
            let err = ((r.re - c.re).powi(2) + (r.im - c.im).powi(2)).sqrt();
            if err > max_err {
                max_err = err;
                max_idx = i;
            }
        }
        if max_err > 1e-9 {
            eprintln!("Size {} comparison (first 12 elements):", n);
            for i in 0..n.min(12) {
                eprintln!(
                    "  [{}] ref=({:.6}, {:.6}) comp=({:.6}, {:.6})",
                    i, ref_output[i].re, ref_output[i].im, comp_output[i].re, comp_output[i].im
                );
            }
            eprintln!("Max error at index {}: {}", max_idx, max_err);
        }
        assert!(
            max_err < 1e-9,
            "Size {}: max error {} exceeds threshold",
            n,
            max_err
        );
    }
    #[test]
    fn test_notw_12_correctness() {
        test_composite_codelet(12, notw_12);
    }
    #[test]
    fn test_notw_24_correctness() {
        test_composite_codelet(24, notw_24);
    }
    #[test]
    fn test_notw_36_correctness() {
        test_composite_codelet(36, notw_36);
    }
    #[test]
    fn test_notw_48_correctness() {
        test_composite_codelet(48, notw_48);
    }
    #[test]
    fn test_notw_60_correctness() {
        test_composite_codelet(60, notw_60);
    }
    #[test]
    fn test_notw_72_correctness() {
        test_composite_codelet(72, notw_72);
    }
    #[test]
    fn test_notw_96_correctness() {
        test_composite_codelet(96, notw_96);
    }
    #[test]
    fn test_notw_100_correctness() {
        test_composite_codelet(100, notw_100);
    }
    #[test]
    fn test_notw_15_correctness() {
        test_composite_codelet(15, notw_15);
    }
    #[test]
    fn test_notw_18_correctness() {
        test_composite_codelet(18, notw_18);
    }
    #[test]
    fn test_notw_20_correctness() {
        test_composite_codelet(20, notw_20);
    }
    #[test]
    fn test_notw_30_correctness() {
        test_composite_codelet(30, notw_30);
    }
    #[test]
    fn test_notw_45_correctness() {
        test_composite_codelet(45, notw_45);
    }
    #[test]
    fn test_notw_50_correctness() {
        test_composite_codelet(50, notw_50);
    }
    #[test]
    fn test_notw_80_correctness() {
        test_composite_codelet(80, notw_80);
    }
}
