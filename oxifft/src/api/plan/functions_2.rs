//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[allow(unused_imports)]
// reason: prelude glob re-exports are selectively used per feature gate (std vs no_std)
use crate::prelude::*;

#[cfg(test)]
#[allow(clippy::cast_lossless)] // reason: usize/u32 as f64 cast in test helpers is intentional and correct for test data
#[allow(clippy::redundant_clone)] // reason: clones in test data setup are explicit for readability
#[allow(clippy::uninlined_format_args)] // reason: generated test code uses explicit format args for clarity
mod tests {
    use super::super::functions::*;
    use super::super::types::{Plan, Plan2D, Plan3D, RealPlanKind};
    use super::super::types_nd::PlanND;
    use super::super::types_r2r::R2rPlan;
    use super::super::types_real::RealPlan;
    use crate::api::{Direction, Flags};
    use crate::kernel::Complex;
    use crate::rdft::solvers::R2rKind;
    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }
    fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
        approx_eq(a.re, b.re, eps) && approx_eq(a.im, b.im, eps)
    }
    #[test]
    fn test_plan_dft_1d_power_of_2() {
        let input: Vec<Complex<f64>> = (0..8).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut output = vec![Complex::zero(); 8];
        let plan = Plan::dft_1d(8, Direction::Forward, Flags::ESTIMATE).unwrap();
        plan.execute(&input, &mut output);
        assert!(complex_approx_eq(output[0], Complex::new(28.0, 0.0), 1e-10));
    }
    #[test]
    fn test_plan_dft_1d_non_power_of_2() {
        let input: Vec<Complex<f64>> = (0..5).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut output = vec![Complex::zero(); 5];
        let plan = Plan::dft_1d(5, Direction::Forward, Flags::ESTIMATE).unwrap();
        plan.execute(&input, &mut output);
        assert!(complex_approx_eq(output[0], Complex::new(10.0, 0.0), 1e-10));
    }
    #[test]
    fn test_plan_dft_1d_large_non_power_of_2() {
        let input: Vec<Complex<f64>> = (0..100)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut output = vec![Complex::zero(); 100];
        let plan = Plan::dft_1d(100, Direction::Forward, Flags::ESTIMATE).unwrap();
        plan.execute(&input, &mut output);
        let mut recovered = vec![Complex::zero(); 100];
        let plan_inv = Plan::dft_1d(100, Direction::Backward, Flags::ESTIMATE).unwrap();
        plan_inv.execute(&output, &mut recovered);
        for x in &mut recovered {
            *x = *x / 100.0;
        }
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }
    #[test]
    fn test_fft_ifft_roundtrip() {
        let original: Vec<Complex<f64>> = (0..16)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let transformed = fft(&original);
        let recovered = ifft(&transformed);
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }
    #[test]
    fn test_fft_ifft_roundtrip_non_power_of_2() {
        let original: Vec<Complex<f64>> = (0..37)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let transformed = fft(&original);
        let recovered = ifft(&transformed);
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }
    #[test]
    fn test_inplace_matches_out_of_place() {
        let input: Vec<Complex<f64>> = (0..8).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut out_of_place = vec![Complex::zero(); 8];
        let plan = Plan::dft_1d(8, Direction::Forward, Flags::ESTIMATE).unwrap();
        plan.execute(&input, &mut out_of_place);
        let mut in_place = input.clone();
        plan.execute_inplace(&mut in_place);
        for (a, b) in out_of_place.iter().zip(in_place.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }
    #[test]
    fn test_plan2d_power_of_2() {
        let n0 = 4;
        let n1 = 8;
        let input: Vec<Complex<f64>> = (0..(n0 * n1))
            .map(|i| Complex::new(i as f64, 0.0))
            .collect();
        let mut output = vec![Complex::zero(); n0 * n1];
        let plan = Plan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE).unwrap();
        plan.execute(&input, &mut output);
        let expected_sum: f64 = (0..(n0 * n1)).map(|i| i as f64).sum();
        assert!(complex_approx_eq(
            output[0],
            Complex::new(expected_sum, 0.0),
            1e-9
        ));
    }
    #[test]
    fn test_plan2d_roundtrip() {
        let n0 = 4;
        let n1 = 4;
        let original: Vec<Complex<f64>> = (0..(n0 * n1))
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut transformed = vec![Complex::zero(); n0 * n1];
        let mut recovered = vec![Complex::zero(); n0 * n1];
        let plan_fwd = Plan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE).unwrap();
        let plan_bwd = Plan2D::new(n0, n1, Direction::Backward, Flags::ESTIMATE).unwrap();
        plan_fwd.execute(&original, &mut transformed);
        plan_bwd.execute(&transformed, &mut recovered);
        let scale = (n0 * n1) as f64;
        for x in &mut recovered {
            *x = *x / scale;
        }
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }
    #[test]
    fn test_plan2d_non_power_of_2() {
        let n0 = 3;
        let n1 = 5;
        let original: Vec<Complex<f64>> = (0..(n0 * n1))
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let transformed = fft2d(&original, n0, n1);
        let recovered = ifft2d(&transformed, n0, n1);
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }
    #[test]
    fn test_plan2d_inplace() {
        let n0 = 4;
        let n1 = 4;
        let original: Vec<Complex<f64>> = (0..(n0 * n1))
            .map(|i| Complex::new(i as f64, 0.0))
            .collect();
        let mut out_of_place = vec![Complex::zero(); n0 * n1];
        let plan = Plan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE).unwrap();
        plan.execute(&original, &mut out_of_place);
        let mut in_place = original;
        plan.execute_inplace(&mut in_place);
        for (a, b) in out_of_place.iter().zip(in_place.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }
    #[test]
    fn test_plan3d_power_of_2() {
        let n0 = 2;
        let n1 = 4;
        let n2 = 4;
        let total = n0 * n1 * n2;
        let input: Vec<Complex<f64>> = (0..total).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut output = vec![Complex::zero(); total];
        let plan = Plan3D::new(n0, n1, n2, Direction::Forward, Flags::ESTIMATE).unwrap();
        plan.execute(&input, &mut output);
        let expected_sum: f64 = (0..total).map(|i| i as f64).sum();
        assert!(complex_approx_eq(
            output[0],
            Complex::new(expected_sum, 0.0),
            1e-9
        ));
    }
    #[test]
    fn test_plan3d_roundtrip() {
        let n0 = 2;
        let n1 = 3;
        let n2 = 4;
        let total = n0 * n1 * n2;
        let original: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut transformed = vec![Complex::zero(); total];
        let mut recovered = vec![Complex::zero(); total];
        let plan_fwd = Plan3D::new(n0, n1, n2, Direction::Forward, Flags::ESTIMATE).unwrap();
        let plan_bwd = Plan3D::new(n0, n1, n2, Direction::Backward, Flags::ESTIMATE).unwrap();
        plan_fwd.execute(&original, &mut transformed);
        plan_bwd.execute(&transformed, &mut recovered);
        let scale = total as f64;
        for x in &mut recovered {
            *x = *x / scale;
        }
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }
    #[test]
    fn test_plan3d_inplace() {
        let n0 = 2;
        let n1 = 2;
        let n2 = 4;
        let total = n0 * n1 * n2;
        let original: Vec<Complex<f64>> = (0..total).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut out_of_place = vec![Complex::zero(); total];
        let plan = Plan3D::new(n0, n1, n2, Direction::Forward, Flags::ESTIMATE).unwrap();
        plan.execute(&original, &mut out_of_place);
        let mut in_place = original;
        plan.execute_inplace(&mut in_place);
        for (a, b) in out_of_place.iter().zip(in_place.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }
    #[test]
    fn test_plannd_2d_matches_plan2d() {
        let n0 = 4;
        let n1 = 8;
        let total = n0 * n1;
        let input: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut output_2d = vec![Complex::zero(); total];
        let plan2d = Plan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE).unwrap();
        plan2d.execute(&input, &mut output_2d);
        let mut output_nd = vec![Complex::zero(); total];
        let plannd = PlanND::new(&[n0, n1], Direction::Forward, Flags::ESTIMATE).unwrap();
        plannd.execute(&input, &mut output_nd);
        for (a, b) in output_2d.iter().zip(output_nd.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }
    #[test]
    fn test_plannd_3d_matches_plan3d() {
        let n0 = 2;
        let n1 = 3;
        let n2 = 4;
        let total = n0 * n1 * n2;
        let input: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut output_3d = vec![Complex::zero(); total];
        let plan3d = Plan3D::new(n0, n1, n2, Direction::Forward, Flags::ESTIMATE).unwrap();
        plan3d.execute(&input, &mut output_3d);
        let mut output_nd = vec![Complex::zero(); total];
        let plannd = PlanND::new(&[n0, n1, n2], Direction::Forward, Flags::ESTIMATE).unwrap();
        plannd.execute(&input, &mut output_nd);
        for (a, b) in output_3d.iter().zip(output_nd.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }
    #[test]
    fn test_plannd_4d_roundtrip() {
        let dims = [2, 3, 4, 2];
        let total: usize = dims.iter().product();
        let original: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut transformed = vec![Complex::zero(); total];
        let mut recovered = vec![Complex::zero(); total];
        let plan_fwd = PlanND::new(&dims, Direction::Forward, Flags::ESTIMATE).unwrap();
        let plan_bwd = PlanND::new(&dims, Direction::Backward, Flags::ESTIMATE).unwrap();
        plan_fwd.execute(&original, &mut transformed);
        plan_bwd.execute(&transformed, &mut recovered);
        let scale = total as f64;
        for x in &mut recovered {
            *x = *x / scale;
        }
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }
    #[test]
    fn test_plannd_inplace() {
        let dims = [2, 4, 4];
        let total: usize = dims.iter().product();
        let original: Vec<Complex<f64>> = (0..total).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut out_of_place = vec![Complex::zero(); total];
        let plan = PlanND::new(&dims, Direction::Forward, Flags::ESTIMATE).unwrap();
        plan.execute(&original, &mut out_of_place);
        let mut in_place = original;
        plan.execute_inplace(&mut in_place);
        for (a, b) in out_of_place.iter().zip(in_place.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10));
        }
    }
    #[test]
    fn test_plannd_dc_component() {
        let dims = [2, 3, 4];
        let total: usize = dims.iter().product();
        let input: Vec<Complex<f64>> = (0..total).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut output = vec![Complex::zero(); total];
        let plan = PlanND::new(&dims, Direction::Forward, Flags::ESTIMATE).unwrap();
        plan.execute(&input, &mut output);
        let expected_sum: f64 = (0..total).map(|i| i as f64).sum();
        assert!(complex_approx_eq(
            output[0],
            Complex::new(expected_sum, 0.0),
            1e-9
        ));
    }
    #[test]
    fn test_fft_nd_ifft_nd_convenience() {
        let dims = [4, 4, 4];
        let total: usize = dims.iter().product();
        let original: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let transformed = fft_nd(&original, &dims);
        let recovered = ifft_nd(&transformed, &dims);
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-9));
        }
    }
    #[test]
    fn test_rfft_irfft_roundtrip() {
        let original: Vec<f64> = (0..16).map(|i| (i as f64).sin()).collect();
        let freq = rfft(&original);
        let recovered = irfft(&freq, original.len());
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(approx_eq(*a, *b, 1e-10), "got {}, expected {}", b, a);
        }
    }
    #[test]
    fn test_rfft_irfft_roundtrip_size_8() {
        let original: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let freq = rfft(&original);
        assert_eq!(freq.len(), 5, "R2C output should have N/2+1 elements");
        let recovered = irfft(&freq, original.len());
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(approx_eq(*a, *b, 1e-10), "got {}, expected {}", b, a);
        }
    }
    #[test]
    fn test_rfft_dc_component() {
        let input: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let freq = rfft(&input);
        let expected_dc: f64 = input.iter().sum();
        assert!(approx_eq(freq[0].re, expected_dc, 1e-10));
        assert!(approx_eq(freq[0].im, 0.0, 1e-10));
    }
    #[test]
    fn test_rfft_matches_complex_fft() {
        let input: Vec<f64> = (0..8).map(|i| (i as f64).sin()).collect();
        let rfft_output = rfft(&input);
        let complex_input: Vec<Complex<f64>> =
            input.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let fft_output = fft(&complex_input);
        for k in 0..5 {
            assert!(
                complex_approx_eq(rfft_output[k], fft_output[k], 1e-9),
                "Mismatch at k={}: rfft={:?}, fft={:?}",
                k,
                rfft_output[k],
                fft_output[k]
            );
        }
    }
    #[test]
    fn test_realplan_r2c() {
        let plan = RealPlan::<f64>::r2c_1d(8, Flags::ESTIMATE).unwrap();
        assert_eq!(plan.size(), 8);
        assert_eq!(plan.complex_size(), 5);
        assert_eq!(plan.kind(), RealPlanKind::R2C);
        let input: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let mut output = vec![Complex::zero(); 5];
        plan.execute_r2c(&input, &mut output);
        assert!(complex_approx_eq(output[0], Complex::new(28.0, 0.0), 1e-10));
    }
    #[test]
    fn test_realplan_c2r() {
        let plan = RealPlan::<f64>::c2r_1d(8, Flags::ESTIMATE).unwrap();
        assert_eq!(plan.size(), 8);
        assert_eq!(plan.complex_size(), 5);
        assert_eq!(plan.kind(), RealPlanKind::C2R);
    }
    #[test]
    fn test_realplan_roundtrip() {
        let original: Vec<f64> = (0..16).map(|i| (i as f64).sin()).collect();
        let r2c_plan = RealPlan::<f64>::r2c_1d(16, Flags::ESTIMATE).unwrap();
        let c2r_plan = RealPlan::<f64>::c2r_1d(16, Flags::ESTIMATE).unwrap();
        let mut freq = vec![Complex::zero(); r2c_plan.complex_size()];
        let mut recovered = vec![0.0_f64; 16];
        r2c_plan.execute_r2c(&original, &mut freq);
        c2r_plan.execute_c2r(&freq, &mut recovered);
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(approx_eq(*a, *b, 1e-10), "got {}, expected {}", b, a);
        }
    }
    #[test]
    fn test_r2rplan_dct2() {
        let plan = R2rPlan::<f64>::dct2(8, Flags::ESTIMATE).unwrap();
        assert_eq!(plan.size(), 8);
        assert_eq!(plan.kind(), R2rKind::Redft10);
        let input: Vec<f64> = (0..8).map(|i| (i as f64).sin()).collect();
        let mut output = vec![0.0_f64; 8];
        plan.execute(&input, &mut output);
        let mut different = false;
        for (a, b) in input.iter().zip(output.iter()) {
            if (a - b).abs() > 1e-10 {
                different = true;
                break;
            }
        }
        assert!(different, "DCT should produce different output");
    }
    #[test]
    fn test_r2rplan_dht_self_inverse() {
        let n = 8;
        let plan = R2rPlan::<f64>::dht(n, Flags::ESTIMATE).unwrap();
        let original: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let mut transformed = vec![0.0_f64; n];
        let mut recovered = vec![0.0_f64; n];
        plan.execute(&original, &mut transformed);
        plan.execute(&transformed, &mut recovered);
        let scale = n as f64;
        for (a, b) in original.iter().zip(recovered.iter()) {
            let normalized = b / scale;
            assert!(
                approx_eq(*a, normalized, 1e-10),
                "got {}, expected {}",
                normalized,
                a
            );
        }
    }
    #[test]
    fn test_r2rplan_inplace() {
        let plan = R2rPlan::<f64>::dct2(8, Flags::ESTIMATE).unwrap();
        let input: Vec<f64> = (0..8).map(|i| (i as f64).sin()).collect();
        let mut out_of_place = vec![0.0_f64; 8];
        plan.execute(&input, &mut out_of_place);
        let mut in_place = input;
        plan.execute_inplace(&mut in_place);
        for (a, b) in out_of_place.iter().zip(in_place.iter()) {
            assert!(approx_eq(*a, *b, 1e-10));
        }
    }
    #[test]
    fn test_fft_batch_roundtrip() {
        let n = 16;
        let howmany = 4;
        let total = n * howmany;
        let original: Vec<Complex<f64>> = (0..total)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let transformed = fft_batch(&original, n, howmany);
        let recovered = ifft_batch(&transformed, n, howmany);
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(
                complex_approx_eq(*a, *b, 1e-10),
                "got {:?}, expected {:?}",
                b,
                a
            );
        }
    }
    #[test]
    fn test_fft_batch_matches_individual() {
        let n = 16;
        let howmany = 3;
        let input: Vec<Complex<f64>> = (0..(n * howmany))
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let batch_result = fft_batch(&input, n, howmany);
        for batch_idx in 0..howmany {
            let start = batch_idx * n;
            let single_input = &input[start..start + n];
            let single_result = fft(single_input);
            for i in 0..n {
                assert!(
                    complex_approx_eq(batch_result[start + i], single_result[i], 1e-10),
                    "Batch {} index {}: got {:?}, expected {:?}",
                    batch_idx,
                    i,
                    batch_result[start + i],
                    single_result[i]
                );
            }
        }
    }
    #[test]
    fn test_fft_batch_linearity() {
        let n = 16;
        let howmany = 2;
        let a: Vec<Complex<f64>> = (0..(n * howmany))
            .map(|i| Complex::new((i as f64).sin(), 0.0))
            .collect();
        let b: Vec<Complex<f64>> = (0..(n * howmany))
            .map(|i| Complex::new((i as f64).cos(), 0.0))
            .collect();
        let scale = Complex::new(2.5, 0.0);
        let sum: Vec<Complex<f64>> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| *x + scale * *y)
            .collect();
        let fft_sum = fft_batch(&sum, n, howmany);
        let fft_a = fft_batch(&a, n, howmany);
        let fft_b = fft_batch(&b, n, howmany);
        let expected: Vec<Complex<f64>> = fft_a
            .iter()
            .zip(fft_b.iter())
            .map(|(x, y)| *x + scale * *y)
            .collect();
        for (got, exp) in fft_sum.iter().zip(expected.iter()) {
            assert!(
                complex_approx_eq(*got, *exp, 1e-9),
                "Linearity failed: got {:?}, expected {:?}",
                got,
                exp
            );
        }
    }
    #[test]
    fn test_fft_batch_power_of_2_sizes() {
        for n in [2, 4, 8, 16, 32, 64] {
            let howmany = 3;
            let original: Vec<Complex<f64>> = (0..(n * howmany))
                .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
                .collect();
            let transformed = fft_batch(&original, n, howmany);
            let recovered = ifft_batch(&transformed, n, howmany);
            for (a, b) in original.iter().zip(recovered.iter()) {
                assert!(
                    complex_approx_eq(*a, *b, 1e-9),
                    "Size {}: got {:?}, expected {:?}",
                    n,
                    b,
                    a
                );
            }
        }
    }
    #[test]
    fn test_fft_batch_non_power_of_2_sizes() {
        for n in [5, 7, 11, 13, 17, 19, 23] {
            let howmany = 2;
            let original: Vec<Complex<f64>> = (0..(n * howmany))
                .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
                .collect();
            let transformed = fft_batch(&original, n, howmany);
            let recovered = ifft_batch(&transformed, n, howmany);
            for (a, b) in original.iter().zip(recovered.iter()) {
                assert!(
                    complex_approx_eq(*a, *b, 1e-8),
                    "Size {}: got {:?}, expected {:?}",
                    n,
                    b,
                    a
                );
            }
        }
    }
    #[test]
    fn test_rfft_batch_roundtrip() {
        let n = 16;
        let howmany = 4;
        let original: Vec<f64> = (0..(n * howmany)).map(|i| (i as f64).sin()).collect();
        let freq = rfft_batch(&original, n, howmany);
        let recovered = irfft_batch(&freq, n, howmany);
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(approx_eq(*a, *b, 1e-10), "got {}, expected {}", b, a);
        }
    }
    #[test]
    fn test_rfft_batch_matches_individual() {
        let n = 16;
        let howmany = 3;
        let out_len = n / 2 + 1;
        let input: Vec<f64> = (0..(n * howmany)).map(|i| (i as f64).sin()).collect();
        let batch_result = rfft_batch(&input, n, howmany);
        for batch_idx in 0..howmany {
            let in_start = batch_idx * n;
            let out_start = batch_idx * out_len;
            let single_input = &input[in_start..in_start + n];
            let single_result = rfft(single_input);
            for i in 0..out_len {
                assert!(
                    complex_approx_eq(batch_result[out_start + i], single_result[i], 1e-10),
                    "Batch {} index {}: got {:?}, expected {:?}",
                    batch_idx,
                    i,
                    batch_result[out_start + i],
                    single_result[i]
                );
            }
        }
    }
    #[test]
    fn test_rfft_batch_power_of_2_sizes() {
        for n in [4, 8, 16, 32, 64] {
            let howmany = 3;
            let original: Vec<f64> = (0..(n * howmany)).map(|i| (i as f64).sin()).collect();
            let freq = rfft_batch(&original, n, howmany);
            let recovered = irfft_batch(&freq, n, howmany);
            for (a, b) in original.iter().zip(recovered.iter()) {
                assert!(
                    approx_eq(*a, *b, 1e-9),
                    "Size {}: got {}, expected {}",
                    n,
                    b,
                    a
                );
            }
        }
    }
    #[test]
    fn test_rfft_batch_non_power_of_2_sizes() {
        for n in [6, 10, 12, 14, 18, 20] {
            let howmany = 2;
            let original: Vec<f64> = (0..(n * howmany)).map(|i| (i as f64).sin()).collect();
            let freq = rfft_batch(&original, n, howmany);
            let recovered = irfft_batch(&freq, n, howmany);
            for (a, b) in original.iter().zip(recovered.iter()) {
                assert!(
                    approx_eq(*a, *b, 1e-8),
                    "Size {}: got {}, expected {}",
                    n,
                    b,
                    a
                );
            }
        }
    }
    #[test]
    fn test_rfft_batch_single_batch() {
        let n = 16;
        let howmany = 1;
        let original: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let freq = rfft_batch(&original, n, howmany);
        let freq_single = rfft(&original);
        for (a, b) in freq.iter().zip(freq_single.iter()) {
            assert!(
                complex_approx_eq(*a, *b, 1e-10),
                "got {:?}, expected {:?}",
                a,
                b
            );
        }
    }
    #[test]
    fn test_fft_batch_single_batch() {
        let n = 16;
        let howmany = 1;
        let original: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let result = fft_batch(&original, n, howmany);
        let result_single = fft(&original);
        for (a, b) in result.iter().zip(result_single.iter()) {
            assert!(
                complex_approx_eq(*a, *b, 1e-10),
                "got {:?}, expected {:?}",
                a,
                b
            );
        }
    }
}
