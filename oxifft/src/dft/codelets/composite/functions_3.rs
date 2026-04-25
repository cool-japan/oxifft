//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#![allow(clippy::unreadable_literal)] // reason: machine-generated FFT twiddle factor literals

use crate::kernel::{Complex, Float};

use super::functions::{dft25, dft5, dft9};

/// Optimized DFT of size 100.
///
/// Uses 4×25 mixed-radix decomposition.
/// sign: -1 for forward, +1 for inverse
#[inline]
pub fn notw_100<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 100);
    let sign_t = if sign < 0 { -T::ONE } else { T::ONE };
    let mut t: [Complex<T>; 100] = [Complex::zero(); 100];
    for j in 0..25 {
        let a0 = x[j];
        let a1 = x[j + 25];
        let a2 = x[j + 50];
        let a3 = x[j + 75];
        let s02 = a0 + a2;
        let d02 = a0 - a2;
        let s13 = a1 + a3;
        let d13 = a1 - a3;
        let rot_d13 = Complex::new(sign_t * d13.im, -sign_t * d13.re);
        t[j] = s02 + s13;
        t[j + 25] = d02 - rot_d13;
        t[j + 50] = s02 - s13;
        t[j + 75] = d02 + rot_d13;
    }
    #[rustfmt::skip]
    let cos_100: [T; 73] = [
        T::ONE,
        T::from_f64(0.998_026_728_428_271_6),
        T::from_f64(0.992_114_701_314_477_9),
        T::from_f64(0.982_287_250_728_688_7),
        T::from_f64(0.968_583_161_128_631_1),
        T::from_f64(0.951_056_516_295_153_5),
        T::from_f64(0.929_776_485_888_251_5),
        T::from_f64(0.904_827_052_466_019_6),
        T::from_f64(0.876_306_680_043_863_6),
        T::from_f64(0.844_327_925_502_015_1),
        T::from_f64(0.809_016_994_374_947_5),
        T::from_f64(0.770_513_242_775_789_3),
        T::from_f64(0.728_968_627_421_411_6),
        T::from_f64(0.684_547_105_928_688_6),
        T::from_f64(0.637_423_989_748_689_7),
        T::from_f64(0.587_785_252_292_473_2),
        T::from_f64(0.535_826_794_978_996_5),
        T::from_f64(0.48175367410171516),
        T::from_f64(0.42577929156507266),
        T::from_f64(0.368_124_552_684_678_1),
        T::from_f64(0.30901699437494745),
        T::from_f64(0.24868988716485474),
        T::from_f64(0.18738131458572474),
        T::from_f64(0.12533323356430448),
        T::from_f64(0.062_790_519_529_313_53),
        T::ZERO,
        T::from_f64(-0.062_790_519_529_313_4),
        T::from_f64(-0.12533323356430415),
        T::from_f64(-0.1873813145857246),
        T::from_f64(-0.24868988716485485),
        T::from_f64(-0.309_016_994_374_947_1),
        T::from_f64(-0.36812455268467775),
        T::from_f64(-0.425_779_291_565_072_7),
        T::from_f64(-0.48175367410171543),
        T::from_f64(-0.535_826_794_978_996_9),
        T::from_f64(-0.587_785_252_292_473),
        T::from_f64(-0.637_423_989_748_689_7),
        T::from_f64(-0.684_547_105_928_688_7),
        T::from_f64(-0.728_968_627_421_411_3),
        T::from_f64(-0.770_513_242_775_789_1),
        T::from_f64(-0.809_016_994_374_947_3),
        T::from_f64(-0.844_327_925_502_014_9),
        T::from_f64(-0.876_306_680_043_863_6),
        T::from_f64(-0.904_827_052_466_019_4),
        T::from_f64(-0.929_776_485_888_251_3),
        T::from_f64(-0.951_056_516_295_153_5),
        T::from_f64(-0.968_583_161_128_631),
        T::from_f64(-0.982_287_250_728_688_7),
        T::from_f64(-0.992_114_701_314_477_8),
        T::from_f64(-0.998_026_728_428_271_6),
        T::from_f64(-1.0),
        T::from_f64(-0.998_026_728_428_271_6),
        T::from_f64(-0.992_114_701_314_477_9),
        T::from_f64(-0.982_287_250_728_688_6),
        T::from_f64(-0.968_583_161_128_631_2),
        T::from_f64(-0.951_056_516_295_153_6),
        T::from_f64(-0.929_776_485_888_251_5),
        T::from_f64(-0.904_827_052_466_019_7),
        T::from_f64(-0.876_306_680_043_863_5),
        T::from_f64(-0.844_327_925_502_015_2),
        T::from_f64(-0.809_016_994_374_947_8),
        T::from_f64(-0.770_513_242_775_789_3),
        T::from_f64(-0.728_968_627_421_411_9),
        T::from_f64(-0.684_547_105_928_688_6),
        T::from_f64(-0.637_423_989_748_689_5),
        T::from_f64(-0.587_785_252_292_473_2),
        T::from_f64(-0.535_826_794_978_996_3),
        T::from_f64(-0.48175367410171527),
        T::from_f64(-0.42577929156507216),
        T::from_f64(-0.368_124_552_684_677_8),
        T::from_f64(-0.30901699437494756),
        T::from_f64(-0.24868988716485443),
        T::from_f64(-0.18738131458572463),
    ];
    #[rustfmt::skip]
    let sin_100: [T; 73] = [
        T::ZERO,
        T::from_f64(0.062_790_519_529_313_37),
        T::from_f64(0.12533323356430426),
        T::from_f64(0.1873813145857246),
        T::from_f64(0.248_689_887_164_854_8),
        T::from_f64(0.3090169943749474),
        T::from_f64(0.368_124_552_684_677_9),
        T::from_f64(0.42577929156507266),
        T::from_f64(0.481_753_674_101_715_3),
        T::from_f64(0.535_826_794_978_996_7),
        T::from_f64(0.587_785_252_292_473_1),
        T::from_f64(0.637_423_989_748_689_6),
        T::from_f64(0.684_547_105_928_688_6),
        T::from_f64(0.728_968_627_421_411_6),
        T::from_f64(0.770_513_242_775_789_3),
        T::from_f64(0.809_016_994_374_947_3),
        T::from_f64(0.844_327_925_502_015_1),
        T::from_f64(0.876_306_680_043_863_7),
        T::from_f64(0.904_827_052_466_019_6),
        T::from_f64(0.929_776_485_888_251_3),
        T::from_f64(0.951_056_516_295_153_5),
        T::from_f64(0.968_583_161_128_631_1),
        T::from_f64(0.982_287_250_728_688_6),
        T::from_f64(0.992_114_701_314_477_8),
        T::from_f64(0.998_026_728_428_271_6),
        T::ONE,
        T::from_f64(0.998_026_728_428_271_6),
        T::from_f64(0.992_114_701_314_477_9),
        T::from_f64(0.982_287_250_728_688_7),
        T::from_f64(0.968_583_161_128_631_1),
        T::from_f64(0.951_056_516_295_153_6),
        T::from_f64(0.929_776_485_888_251_5),
        T::from_f64(0.904_827_052_466_019_5),
        T::from_f64(0.876_306_680_043_863_5),
        T::from_f64(0.844_327_925_502_015),
        T::from_f64(0.809_016_994_374_947_5),
        T::from_f64(0.770_513_242_775_789_3),
        T::from_f64(0.728_968_627_421_411_4),
        T::from_f64(0.684_547_105_928_688_8),
        T::from_f64(0.637_423_989_748_689_9),
        T::from_f64(0.587_785_252_292_473_2),
        T::from_f64(0.535_826_794_978_997),
        T::from_f64(0.481_753_674_101_715_2),
        T::from_f64(0.425_779_291_565_072_9),
        T::from_f64(0.36812455268467814),
        T::from_f64(0.309_016_994_374_947_5),
        T::from_f64(0.24868988716485524),
        T::from_f64(0.18738131458572457),
        T::from_f64(0.12533323356430454),
        T::from_f64(0.062_790_519_529_313_58),
        T::ZERO,
        T::from_f64(-0.062_790_519_529_313_35),
        T::from_f64(-0.12533323356430429),
        T::from_f64(-0.18738131458572477),
        T::from_f64(-0.24868988716485457),
        T::from_f64(-0.309_016_994_374_947_3),
        T::from_f64(-0.368_124_552_684_677_9),
        T::from_f64(-0.42577929156507227),
        T::from_f64(-0.481_753_674_101_715_4),
        T::from_f64(-0.535_826_794_978_996_4),
        T::from_f64(-0.587_785_252_292_472_7),
        T::from_f64(-0.637_423_989_748_689_6),
        T::from_f64(-0.684_547_105_928_688_4),
        T::from_f64(-0.728_968_627_421_411_7),
        T::from_f64(-0.770_513_242_775_789_4),
        T::from_f64(-0.809_016_994_374_947_3),
        T::from_f64(-0.8443279255020153),
        T::from_f64(-0.876_306_680_043_863_6),
        T::from_f64(-0.9048270524660198),
        T::from_f64(-0.929_776_485_888_251_5),
        T::from_f64(-0.951_056_516_295_153_5),
        T::from_f64(-0.968_583_161_128_631_2),
        T::from_f64(-0.982_287_250_728_688_7),
    ];
    for k1 in 1..4 {
        for k2 in 1..25 {
            let k = k1 * k2;
            let c = cos_100[k];
            let s = sin_100[k];
            let idx = k1 * 25 + k2;
            let tw = t[idx];
            t[idx] = Complex::new(
                tw.re * c - sign_t * tw.im * s,
                sign_t * tw.re * s + tw.im * c,
            );
        }
    }
    for k1 in 0..4 {
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
            x[k2 * 4 + k1] = y[k2];
        }
    }
}
/// Optimized DFT of size 15.
///
/// Uses 3×5 mixed-radix decomposition.
/// sign: -1 for forward, +1 for inverse
#[inline]
pub fn notw_15<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 15);
    let sqrt3_2 = T::from_f64(0.8660254037844387);
    let half = T::from_f64(0.5);
    let sign_t = if sign < 0 { -T::ONE } else { T::ONE };
    let mut t: [Complex<T>; 15] = [Complex::zero(); 15];
    for j in 0..5 {
        let a0 = x[j];
        let a1 = x[j + 5];
        let a2 = x[j + 10];
        let sum = a0 + a1 + a2;
        let d1 = a1 - a2;
        let d2 = a0 - (a1 + a2).scale(half);
        let rot_re = -sign_t * sqrt3_2 * d1.im;
        let rot_im = -sign_t * sqrt3_2 * d1.re;
        t[j] = sum;
        t[j + 5] = Complex::new(d2.re + rot_re, d2.im - rot_im);
        t[j + 10] = Complex::new(d2.re - rot_re, d2.im + rot_im);
    }
    let c1 = T::from_f64(0.9135454576426009);
    let s1 = T::from_f64(0.4067366430758002);
    let c2 = T::from_f64(0.6691306063588582);
    let s2 = T::from_f64(0.7431448254773942);
    let c3 = T::from_f64(0.309_016_994_374_947_4);
    let s3 = T::from_f64(0.9510565162951535);
    let c4 = T::from_f64(-0.10452846326765346);
    let s4 = T::from_f64(0.9945218953682733);
    let c6 = T::from_f64(-0.8090169943749474);
    let s6 = T::from_f64(0.5877852522924731);
    let c8 = T::from_f64(-0.9781476007338057);
    let s8 = T::from_f64(-0.20791169081775934);
    let tw = t[6];
    t[6] = Complex::new(
        tw.re * c1 - sign_t * tw.im * s1,
        sign_t * tw.re * s1 + tw.im * c1,
    );
    let tw = t[7];
    t[7] = Complex::new(
        tw.re * c2 - sign_t * tw.im * s2,
        sign_t * tw.re * s2 + tw.im * c2,
    );
    let tw = t[8];
    t[8] = Complex::new(
        tw.re * c3 - sign_t * tw.im * s3,
        sign_t * tw.re * s3 + tw.im * c3,
    );
    let tw = t[9];
    t[9] = Complex::new(
        tw.re * c4 - sign_t * tw.im * s4,
        sign_t * tw.re * s4 + tw.im * c4,
    );
    let tw = t[11];
    t[11] = Complex::new(
        tw.re * c2 - sign_t * tw.im * s2,
        sign_t * tw.re * s2 + tw.im * c2,
    );
    let tw = t[12];
    t[12] = Complex::new(
        tw.re * c4 - sign_t * tw.im * s4,
        sign_t * tw.re * s4 + tw.im * c4,
    );
    let tw = t[13];
    t[13] = Complex::new(
        tw.re * c6 - sign_t * tw.im * s6,
        sign_t * tw.re * s6 + tw.im * c6,
    );
    let tw = t[14];
    t[14] = Complex::new(
        tw.re * c8 - sign_t * tw.im * s8,
        sign_t * tw.re * s8 + tw.im * c8,
    );
    let c1_5 = T::from_f64(0.309_016_994_374_947_4);
    let c2_5 = T::from_f64(-0.809_016_994_374_947_5);
    let s1_5 = T::from_f64(0.951_056_516_295_153_5);
    let s2_5 = T::from_f64(0.587_785_252_292_473_1);
    for k1 in 0..3 {
        let base = k1 * 5;
        let x0 = t[base];
        let x1 = t[base + 1];
        let x2 = t[base + 2];
        let x3 = t[base + 3];
        let x4 = t[base + 4];
        let p1 = x1 + x4;
        let p2 = x2 + x3;
        let m1 = x1 - x4;
        let m2 = x2 - x3;
        let y0 = x0 + p1 + p2;
        let t1 = x0 + p1.scale(c1_5) + p2.scale(c2_5);
        let t2 = x0 + p1.scale(c2_5) + p2.scale(c1_5);
        let r1_re = -sign_t * (m1.im * s1_5 + m2.im * s2_5);
        let r1_im = sign_t * (m1.re * s1_5 + m2.re * s2_5);
        let r2_re = -sign_t * (m1.im * s2_5 - m2.im * s1_5);
        let r2_im = sign_t * (m1.re * s2_5 - m2.re * s1_5);
        x[k1] = y0;
        x[k1 + 3] = Complex::new(t1.re + r1_re, t1.im + r1_im);
        x[k1 + 6] = Complex::new(t2.re + r2_re, t2.im + r2_im);
        x[k1 + 9] = Complex::new(t2.re - r2_re, t2.im - r2_im);
        x[k1 + 12] = Complex::new(t1.re - r1_re, t1.im - r1_im);
    }
}
/// Optimized DFT of size 18.
///
/// Uses 2×9 mixed-radix decomposition.
/// sign: -1 for forward, +1 for inverse
#[inline]
pub fn notw_18<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 18);
    let sign_t = if sign < 0 { -T::ONE } else { T::ONE };
    let mut t: [Complex<T>; 18] = [Complex::zero(); 18];
    for j in 0..9 {
        let a0 = x[j];
        let a1 = x[j + 9];
        t[j] = a0 + a1;
        t[j + 9] = a0 - a1;
    }
    let angle_18 = sign_t * T::TWO_PI / T::from_f64(18.0);
    for k2 in 1..9 {
        let angle = angle_18 * T::from_usize(k2);
        let (s, c) = Float::sin_cos(angle);
        let idx = 9 + k2;
        let tw = t[idx];
        t[idx] = Complex::new(tw.re * c - tw.im * s, tw.re * s + tw.im * c);
    }
    for k1 in 0..2 {
        let base = k1 * 9;
        let a: [Complex<T>; 9] = [
            t[base],
            t[base + 1],
            t[base + 2],
            t[base + 3],
            t[base + 4],
            t[base + 5],
            t[base + 6],
            t[base + 7],
            t[base + 8],
        ];
        let y = dft9(&a, sign_t);
        for k2 in 0..9 {
            x[k2 * 2 + k1] = y[k2];
        }
    }
}
/// Optimized DFT of size 20.
///
/// Uses 4×5 mixed-radix decomposition.
/// sign: -1 for forward, +1 for inverse
#[inline]
pub fn notw_20<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 20);
    let sign_t = if sign < 0 { -T::ONE } else { T::ONE };
    let mut t: [Complex<T>; 20] = [Complex::zero(); 20];
    for j in 0..5 {
        let a0 = x[j];
        let a1 = x[j + 5];
        let a2 = x[j + 10];
        let a3 = x[j + 15];
        let s02 = a0 + a2;
        let d02 = a0 - a2;
        let s13 = a1 + a3;
        let d13 = a1 - a3;
        let rot_d13 = Complex::new(sign_t * d13.im, -sign_t * d13.re);
        t[j] = s02 + s13;
        t[j + 5] = d02 - rot_d13;
        t[j + 10] = s02 - s13;
        t[j + 15] = d02 + rot_d13;
    }
    let angle_20 = sign_t * T::TWO_PI / T::from_f64(20.0);
    for k1 in 1..4 {
        for k2 in 1..5 {
            let angle = angle_20 * T::from_usize(k1 * k2);
            let (s, c) = Float::sin_cos(angle);
            let idx = k1 * 5 + k2;
            let tw = t[idx];
            t[idx] = Complex::new(tw.re * c - tw.im * s, tw.re * s + tw.im * c);
        }
    }
    for k1 in 0..4 {
        let base = k1 * 5;
        let a: [Complex<T>; 5] = [t[base], t[base + 1], t[base + 2], t[base + 3], t[base + 4]];
        let y = dft5(&a, sign_t);
        for k2 in 0..5 {
            x[k2 * 4 + k1] = y[k2];
        }
    }
}
/// Optimized DFT of size 30.
///
/// Uses 2×15 mixed-radix decomposition, calling notw_15 for sub-DFTs.
/// sign: -1 for forward, +1 for inverse
#[inline]
pub fn notw_30<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 30);
    let sign_t = if sign < 0 { -T::ONE } else { T::ONE };
    let mut t: [Complex<T>; 30] = [Complex::zero(); 30];
    for j in 0..15 {
        let a0 = x[j];
        let a1 = x[j + 15];
        t[j] = a0 + a1;
        t[j + 15] = a0 - a1;
    }
    #[rustfmt::skip]
    let cos_30: [T; 15] = [
        T::ONE,
        T::from_f64(0.978_147_600_733_805_7),
        T::from_f64(0.913_545_457_642_600_9),
        T::from_f64(0.809_016_994_374_947_5),
        T::from_f64(0.669_130_606_358_858_2),
        T::from_f64(0.5),
        T::from_f64(0.30901699437494745),
        T::from_f64(0.10452846326765346),
        T::from_f64(-0.10452846326765333),
        T::from_f64(-0.30901699437494734),
        T::from_f64(-0.5),
        T::from_f64(-0.6691306063588579),
        T::from_f64(-0.809_016_994_374_947_3),
        T::from_f64(-0.913_545_457_642_601),
        T::from_f64(-0.978_147_600_733_805_7),
    ];
    #[rustfmt::skip]
    let sin_30: [T; 15] = [
        T::ZERO,
        T::from_f64(0.20791169081775931),
        T::from_f64(0.40673664307580015),
        T::from_f64(0.587_785_252_292_473_1),
        T::from_f64(0.743_144_825_477_394_2),
        T::from_f64(0.8660254037844386),
        T::from_f64(0.951_056_516_295_153_5),
        T::from_f64(0.994_521_895_368_273_3),
        T::from_f64(0.9945218953682734),
        T::from_f64(0.951_056_516_295_153_6),
        T::from_f64(0.866_025_403_784_438_7),
        T::from_f64(0.743_144_825_477_394_5),
        T::from_f64(0.587_785_252_292_473_2),
        T::from_f64(0.40673664307580004),
        T::from_f64(0.20791169081775931),
    ];
    for k2 in 1..15 {
        let c = cos_30[k2];
        let s = sin_30[k2];
        let idx = 15 + k2;
        let tw = t[idx];
        t[idx] = Complex::new(
            tw.re * c - sign_t * tw.im * s,
            sign_t * tw.re * s + tw.im * c,
        );
    }
    for k1 in 0..2 {
        let base = k1 * 15;
        let mut row: [Complex<T>; 15] = [
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
        ];
        notw_15(&mut row, sign);
        for k2 in 0..15 {
            x[k2 * 2 + k1] = row[k2];
        }
    }
}
/// Optimized DFT of size 45.
///
/// Uses 9×5 mixed-radix decomposition.
/// sign: -1 for forward, +1 for inverse
#[inline]
pub fn notw_45<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 45);
    let sign_t = if sign < 0 { -T::ONE } else { T::ONE };
    let mut t: [Complex<T>; 45] = [Complex::zero(); 45];
    for j in 0..5 {
        let a: [Complex<T>; 9] = [
            x[j],
            x[j + 5],
            x[j + 10],
            x[j + 15],
            x[j + 20],
            x[j + 25],
            x[j + 30],
            x[j + 35],
            x[j + 40],
        ];
        let y = dft9(&a, sign_t);
        for k1 in 0..9 {
            t[k1 * 5 + j] = y[k1];
        }
    }
    let cos_45: [T; 33] = [
        T::ONE,
        T::from_f64(0.9902680687415704),
        T::from_f64(0.9612616959383189),
        T::from_f64(0.9135454576426009),
        T::from_f64(0.8480480961564261),
        T::from_f64(0.766044443118978),
        T::from_f64(0.6691306063588582),
        T::from_f64(0.5591929034707469),
        T::from_f64(0.4383711467890774),
        T::from_f64(0.30901699437494745),
        T::from_f64(0.17364817766693041),
        T::from_f64(0.034899496702501),
        T::from_f64(-0.10452846326765346),
        T::from_f64(-0.24192189559966773),
        T::from_f64(-0.37460659341591196),
        T::from_f64(-0.5),
        T::from_f64(-0.6156614753256583),
        T::from_f64(-0.7193398003386512),
        T::from_f64(-0.8090169943749474),
        T::from_f64(-0.8829475928589269),
        T::from_f64(-0.9396926207859084),
        T::from_f64(-0.9781476007338057),
        T::from_f64(-0.9975640502598242),
        T::from_f64(-0.9975640502598242),
        T::from_f64(-0.9781476007338057),
        T::from_f64(-0.9396926207859084),
        T::from_f64(-0.8829475928589269),
        T::from_f64(-0.8090169943749474),
        T::from_f64(-0.7193398003386512),
        T::from_f64(-0.6156614753256583),
        T::from_f64(-0.5),
        T::from_f64(-0.37460659341591196),
        T::from_f64(-0.24192189559966773),
    ];
    let sin_45: [T; 33] = [
        T::ZERO,
        T::from_f64(0.13917310096006544),
        T::from_f64(0.27563735581699916),
        T::from_f64(0.4067366430758002),
        T::from_f64(0.5299192642332049),
        T::from_f64(0.6427876096865393),
        T::from_f64(0.7431448254773942),
        T::from_f64(0.8290375725550417),
        T::from_f64(0.8987940462991669),
        T::from_f64(0.9510565162951535),
        T::from_f64(0.984807753012208),
        T::from_f64(0.9993908270190958),
        T::from_f64(0.9945218953682733),
        T::from_f64(0.9702957262759965),
        T::from_f64(0.9271838545667874),
        T::from_f64(0.8660254037844387),
        T::from_f64(0.7880107536067219),
        T::from_f64(0.6946583704589973),
        T::from_f64(0.5877852522924731),
        T::from_f64(0.46947156278589086),
        T::from_f64(0.3420201433256687),
        T::from_f64(0.20791169081775934),
        T::from_f64(0.06975647374412532),
        T::from_f64(-0.06975647374412532),
        T::from_f64(-0.20791169081775934),
        T::from_f64(-0.3420201433256687),
        T::from_f64(-0.46947156278589086),
        T::from_f64(-0.5877852522924731),
        T::from_f64(-0.6946583704589973),
        T::from_f64(-0.7880107536067219),
        T::from_f64(-0.8660254037844387),
        T::from_f64(-0.9271838545667874),
        T::from_f64(-0.9702957262759965),
    ];
    for k1 in 1..9 {
        for k2 in 1..5 {
            let k = k1 * k2;
            let c = cos_45[k];
            let s = sin_45[k];
            let idx = k1 * 5 + k2;
            let tw = t[idx];
            t[idx] = Complex::new(
                tw.re * c - sign_t * tw.im * s,
                sign_t * tw.re * s + tw.im * c,
            );
        }
    }
    for k1 in 0..9 {
        let base = k1 * 5;
        let a: [Complex<T>; 5] = [t[base], t[base + 1], t[base + 2], t[base + 3], t[base + 4]];
        let y = dft5(&a, sign_t);
        for k2 in 0..5 {
            x[k2 * 9 + k1] = y[k2];
        }
    }
}
