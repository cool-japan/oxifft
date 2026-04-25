//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#![allow(clippy::approx_constant)] // reason: twiddle constants are trigonometric values, not approximations of named constants
#![allow(clippy::unreadable_literal)] // reason: machine-generated FFT twiddle factor literals

use crate::dft::codelets::simd;
use crate::kernel::{Complex, Float};

use super::functions::{dft5, notw_12};

/// Optimized DFT of size 60.
///
/// Uses 4×15 mixed-radix decomposition.
/// sign: -1 for forward, +1 for inverse
#[inline]
pub fn notw_60<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 60);
    let sign_t = if sign < 0 { -T::ONE } else { T::ONE };
    let mut t: [Complex<T>; 60] = [Complex::zero(); 60];
    for j in 0..15 {
        let a0 = x[j];
        let a1 = x[j + 15];
        let a2 = x[j + 30];
        let a3 = x[j + 45];
        let s02 = a0 + a2;
        let d02 = a0 - a2;
        let s13 = a1 + a3;
        let d13 = a1 - a3;
        let rot_d13 = Complex::new(sign_t * d13.im, -sign_t * d13.re);
        t[j] = s02 + s13;
        t[j + 15] = d02 - rot_d13;
        t[j + 30] = s02 - s13;
        t[j + 45] = d02 + rot_d13;
    }
    let cos_60: [T; 46] = [
        T::ONE,
        T::from_f64(0.9945218953682733),
        T::from_f64(0.9781476007338057),
        T::from_f64(0.9510565162951535),
        T::from_f64(0.9135454576426009),
        T::from_f64(0.8660254037844387),
        T::from_f64(0.8090169943749474),
        T::from_f64(0.7431448254773942),
        T::from_f64(0.6691306063588582),
        T::from_f64(0.5877852522924731),
        T::from_f64(0.5),
        T::from_f64(0.4067366430758002),
        T::from_f64(0.30901699437494745),
        T::from_f64(0.20791169081775934),
        T::from_f64(0.10452846326765346),
        T::ZERO,
        T::from_f64(-0.10452846326765346),
        T::from_f64(-0.20791169081775934),
        T::from_f64(-0.30901699437494745),
        T::from_f64(-0.4067366430758002),
        T::from_f64(-0.5),
        T::from_f64(-0.5877852522924731),
        T::from_f64(-0.6691306063588582),
        T::from_f64(-0.7431448254773942),
        T::from_f64(-0.8090169943749474),
        T::from_f64(-0.8660254037844387),
        T::from_f64(-0.9135454576426009),
        T::from_f64(-0.9510565162951535),
        T::from_f64(-0.9781476007338057),
        T::from_f64(-0.9945218953682733),
        -T::ONE,
        T::from_f64(-0.9945218953682733),
        T::from_f64(-0.9781476007338057),
        T::from_f64(-0.9510565162951535),
        T::from_f64(-0.9135454576426009),
        T::from_f64(-0.8660254037844387),
        T::from_f64(-0.8090169943749474),
        T::from_f64(-0.7431448254773942),
        T::from_f64(-0.6691306063588582),
        T::from_f64(-0.5877852522924731),
        T::from_f64(-0.5),
        T::from_f64(-0.4067366430758002),
        T::from_f64(-0.30901699437494745),
        T::from_f64(-0.20791169081775934),
        T::from_f64(-0.10452846326765346),
        T::ZERO,
    ];
    let sin_60: [T; 46] = [
        T::ZERO,
        T::from_f64(0.10452846326765346),
        T::from_f64(0.20791169081775934),
        T::from_f64(0.30901699437494745),
        T::from_f64(0.4067366430758002),
        T::from_f64(0.5),
        T::from_f64(0.5877852522924731),
        T::from_f64(0.6691306063588582),
        T::from_f64(0.7431448254773942),
        T::from_f64(0.8090169943749474),
        T::from_f64(0.8660254037844387),
        T::from_f64(0.9135454576426009),
        T::from_f64(0.9510565162951535),
        T::from_f64(0.9781476007338057),
        T::from_f64(0.9945218953682733),
        T::ONE,
        T::from_f64(0.9945218953682733),
        T::from_f64(0.9781476007338057),
        T::from_f64(0.9510565162951535),
        T::from_f64(0.9135454576426009),
        T::from_f64(0.8660254037844387),
        T::from_f64(0.8090169943749474),
        T::from_f64(0.7431448254773942),
        T::from_f64(0.6691306063588582),
        T::from_f64(0.5877852522924731),
        T::from_f64(0.5),
        T::from_f64(0.4067366430758002),
        T::from_f64(0.30901699437494745),
        T::from_f64(0.20791169081775934),
        T::from_f64(0.10452846326765346),
        T::ZERO,
        T::from_f64(-0.10452846326765346),
        T::from_f64(-0.20791169081775934),
        T::from_f64(-0.30901699437494745),
        T::from_f64(-0.4067366430758002),
        T::from_f64(-0.5),
        T::from_f64(-0.5877852522924731),
        T::from_f64(-0.6691306063588582),
        T::from_f64(-0.7431448254773942),
        T::from_f64(-0.8090169943749474),
        T::from_f64(-0.8660254037844387),
        T::from_f64(-0.9135454576426009),
        T::from_f64(-0.9510565162951535),
        T::from_f64(-0.9781476007338057),
        T::from_f64(-0.9945218953682733),
        -T::ONE,
    ];
    for k1 in 1..4 {
        for k2 in 1..15 {
            let k = k1 * k2;
            let c = cos_60[k];
            let s = sin_60[k];
            let idx = k1 * 15 + k2;
            let tw = t[idx];
            t[idx] = Complex::new(
                tw.re * c - sign_t * tw.im * s,
                sign_t * tw.re * s + tw.im * c,
            );
        }
    }
    let sqrt3_2 = T::from_f64(0.8660254037844387);
    let half = T::from_f64(0.5);
    let c15_1 = T::from_f64(0.9135454576426009);
    let s15_1 = T::from_f64(0.4067366430758002);
    let c15_2 = T::from_f64(0.6691306063588582);
    let s15_2 = T::from_f64(0.7431448254773942);
    let c15_3 = T::from_f64(0.309_016_994_374_947_4);
    let s15_3 = T::from_f64(0.9510565162951535);
    let c15_4 = T::from_f64(-0.10452846326765346);
    let s15_4 = T::from_f64(0.9945218953682733);
    let c15_6 = T::from_f64(-0.8090169943749474);
    let s15_6 = T::from_f64(0.5877852522924731);
    let c15_8 = T::from_f64(-0.9781476007338057);
    let s15_8 = T::from_f64(-0.20791169081775934);
    for k1 in 0..4 {
        let base = k1 * 15;
        let mut tmp: [Complex<T>; 15] = [Complex::zero(); 15];
        for j in 0..5 {
            let a0 = t[base + j];
            let a1 = t[base + j + 5];
            let a2 = t[base + j + 10];
            let sum = a0 + a1 + a2;
            let d1 = a1 - a2;
            let d2 = a0 - (a1 + a2).scale(half);
            let rot_re = -sign_t * sqrt3_2 * d1.im;
            let rot_im = -sign_t * sqrt3_2 * d1.re;
            tmp[j] = sum;
            tmp[j + 5] = Complex::new(d2.re + rot_re, d2.im - rot_im);
            tmp[j + 10] = Complex::new(d2.re - rot_re, d2.im + rot_im);
        }
        let tw = tmp[6];
        tmp[6] = Complex::new(
            tw.re * c15_1 - sign_t * tw.im * s15_1,
            sign_t * tw.re * s15_1 + tw.im * c15_1,
        );
        let tw = tmp[7];
        tmp[7] = Complex::new(
            tw.re * c15_2 - sign_t * tw.im * s15_2,
            sign_t * tw.re * s15_2 + tw.im * c15_2,
        );
        let tw = tmp[8];
        tmp[8] = Complex::new(
            tw.re * c15_3 - sign_t * tw.im * s15_3,
            sign_t * tw.re * s15_3 + tw.im * c15_3,
        );
        let tw = tmp[9];
        tmp[9] = Complex::new(
            tw.re * c15_4 - sign_t * tw.im * s15_4,
            sign_t * tw.re * s15_4 + tw.im * c15_4,
        );
        let tw = tmp[11];
        tmp[11] = Complex::new(
            tw.re * c15_2 - sign_t * tw.im * s15_2,
            sign_t * tw.re * s15_2 + tw.im * c15_2,
        );
        let tw = tmp[12];
        tmp[12] = Complex::new(
            tw.re * c15_4 - sign_t * tw.im * s15_4,
            sign_t * tw.re * s15_4 + tw.im * c15_4,
        );
        let tw = tmp[13];
        tmp[13] = Complex::new(
            tw.re * c15_6 - sign_t * tw.im * s15_6,
            sign_t * tw.re * s15_6 + tw.im * c15_6,
        );
        let tw = tmp[14];
        tmp[14] = Complex::new(
            tw.re * c15_8 - sign_t * tw.im * s15_8,
            sign_t * tw.re * s15_8 + tw.im * c15_8,
        );
        for row in 0..3 {
            let row_base = row * 5;
            let input: [Complex<T>; 5] = [
                tmp[row_base],
                tmp[row_base + 1],
                tmp[row_base + 2],
                tmp[row_base + 3],
                tmp[row_base + 4],
            ];
            let output = dft5(&input, sign_t);
            for col in 0..5 {
                x[(row + col * 3) * 4 + k1] = output[col];
            }
        }
    }
}
/// Optimized DFT of size 72.
///
/// Uses 9×8 mixed-radix decomposition with inlined dft9.
/// sign: -1 for forward, +1 for inverse
#[inline]
pub fn notw_72<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 72);
    let sign_t = if sign < 0 { -T::ONE } else { T::ONE };
    let mut t: [Complex<T>; 72] = [Complex::zero(); 72];
    let sqrt3_2 = T::from_f64(0.8660254037844387);
    let half = T::from_f64(0.5);
    let c1_9 = T::from_f64(0.766044443118978);
    let s1_9 = T::from_f64(0.6427876096865393);
    let c2_9 = T::from_f64(0.17364817766693041);
    let s2_9 = T::from_f64(0.984807753012208);
    let c4_9 = T::from_f64(-0.9396926207859084);
    let s4_9 = T::from_f64(0.3420201433256687);
    for j in 0..8 {
        let a0 = x[j];
        let a1 = x[j + 8];
        let a2 = x[j + 16];
        let a3 = x[j + 24];
        let a4 = x[j + 32];
        let a5 = x[j + 40];
        let a6 = x[j + 48];
        let a7 = x[j + 56];
        let a8 = x[j + 64];
        let sum0 = a0 + a3 + a6;
        let d1_0 = a3 - a6;
        let d2_0 = a0 - (a3 + a6).scale(half);
        let rot_re0 = -sign_t * sqrt3_2 * d1_0.im;
        let rot_im0 = -sign_t * sqrt3_2 * d1_0.re;
        let t0 = sum0;
        let t3 = Complex::new(d2_0.re + rot_re0, d2_0.im - rot_im0);
        let t6 = Complex::new(d2_0.re - rot_re0, d2_0.im + rot_im0);
        let sum1 = a1 + a4 + a7;
        let d1_1 = a4 - a7;
        let d2_1 = a1 - (a4 + a7).scale(half);
        let rot_re1 = -sign_t * sqrt3_2 * d1_1.im;
        let rot_im1 = -sign_t * sqrt3_2 * d1_1.re;
        let t1 = sum1;
        let t4 = Complex::new(d2_1.re + rot_re1, d2_1.im - rot_im1);
        let t7 = Complex::new(d2_1.re - rot_re1, d2_1.im + rot_im1);
        let sum2 = a2 + a5 + a8;
        let d1_2 = a5 - a8;
        let d2_2 = a2 - (a5 + a8).scale(half);
        let rot_re2 = -sign_t * sqrt3_2 * d1_2.im;
        let rot_im2 = -sign_t * sqrt3_2 * d1_2.re;
        let t2 = sum2;
        let t5 = Complex::new(d2_2.re + rot_re2, d2_2.im - rot_im2);
        let t8 = Complex::new(d2_2.re - rot_re2, d2_2.im + rot_im2);
        let tw4 = Complex::new(
            t4.re * c1_9 - sign_t * t4.im * s1_9,
            sign_t * t4.re * s1_9 + t4.im * c1_9,
        );
        let tw5 = Complex::new(
            t5.re * c2_9 - sign_t * t5.im * s2_9,
            sign_t * t5.re * s2_9 + t5.im * c2_9,
        );
        let tw7 = Complex::new(
            t7.re * c2_9 - sign_t * t7.im * s2_9,
            sign_t * t7.re * s2_9 + t7.im * c2_9,
        );
        let tw8 = Complex::new(
            t8.re * c4_9 - sign_t * t8.im * s4_9,
            sign_t * t8.re * s4_9 + t8.im * c4_9,
        );
        let sum_r0 = t0 + t1 + t2;
        let d1_r0 = t1 - t2;
        let d2_r0 = t0 - (t1 + t2).scale(half);
        let rot_re_r0 = -sign_t * sqrt3_2 * d1_r0.im;
        let rot_im_r0 = -sign_t * sqrt3_2 * d1_r0.re;
        let sum_r1 = t3 + tw4 + tw5;
        let d1_r1 = tw4 - tw5;
        let d2_r1 = t3 - (tw4 + tw5).scale(half);
        let rot_re_r1 = -sign_t * sqrt3_2 * d1_r1.im;
        let rot_im_r1 = -sign_t * sqrt3_2 * d1_r1.re;
        let sum_r2 = t6 + tw7 + tw8;
        let d1_r2 = tw7 - tw8;
        let d2_r2 = t6 - (tw7 + tw8).scale(half);
        let rot_re_r2 = -sign_t * sqrt3_2 * d1_r2.im;
        let rot_im_r2 = -sign_t * sqrt3_2 * d1_r2.re;
        t[j] = sum_r0;
        t[8 + j] = sum_r1;
        t[16 + j] = sum_r2;
        t[24 + j] = Complex::new(d2_r0.re + rot_re_r0, d2_r0.im - rot_im_r0);
        t[32 + j] = Complex::new(d2_r1.re + rot_re_r1, d2_r1.im - rot_im_r1);
        t[40 + j] = Complex::new(d2_r2.re + rot_re_r2, d2_r2.im - rot_im_r2);
        t[48 + j] = Complex::new(d2_r0.re - rot_re_r0, d2_r0.im + rot_im_r0);
        t[56 + j] = Complex::new(d2_r1.re - rot_re_r1, d2_r1.im + rot_im_r1);
        t[64 + j] = Complex::new(d2_r2.re - rot_re_r2, d2_r2.im + rot_im_r2);
    }
    #[rustfmt::skip]
    let cos_72: [T; 57] = [
        T::ONE,
        T::from_f64(0.9961946980917455),
        T::from_f64(0.984807753012208),
        T::from_f64(0.9659258262890683),
        T::from_f64(0.9396926207859084),
        T::from_f64(0.9063077870366499),
        T::from_f64(0.8660254037844387),
        T::from_f64(0.8191520442889918),
        T::from_f64(0.766044443118978),
        T::from_f64(0.7071067811865476),
        T::from_f64(0.6427876096865394),
        T::from_f64(0.573_576_436_351_046),
        T::from_f64(0.5),
        T::from_f64(0.42261826174069944),
        T::from_f64(0.3420201433256687),
        T::from_f64(0.25881904510252074),
        T::from_f64(0.17364817766693041),
        T::from_f64(0.08715574274765814),
        T::ZERO,
        T::from_f64(-0.08715574274765814),
        T::from_f64(-0.17364817766693041),
        T::from_f64(-0.25881904510252074),
        T::from_f64(-0.3420201433256687),
        T::from_f64(-0.42261826174069944),
        T::from_f64(-0.5),
        T::from_f64(-0.573_576_436_351_046),
        T::from_f64(-0.6427876096865394),
        T::from_f64(-0.7071067811865476),
        T::from_f64(-0.766044443118978),
        T::from_f64(-0.8191520442889918),
        T::from_f64(-0.8660254037844387),
        T::from_f64(-0.9063077870366499),
        T::from_f64(-0.9396926207859084),
        T::from_f64(-0.9659258262890683),
        T::from_f64(-0.984807753012208),
        T::from_f64(-0.9961946980917455),
        -T::ONE,
        T::from_f64(-0.9961946980917455),
        T::from_f64(-0.984807753012208),
        T::from_f64(-0.9659258262890683),
        T::from_f64(-0.9396926207859084),
        T::from_f64(-0.9063077870366499),
        T::from_f64(-0.8660254037844387),
        T::from_f64(-0.8191520442889918),
        T::from_f64(-0.766044443118978),
        T::from_f64(-0.7071067811865476),
        T::from_f64(-0.6427876096865394),
        T::from_f64(-0.573_576_436_351_046),
        T::from_f64(-0.5),
        T::from_f64(-0.42261826174069944),
        T::from_f64(-0.3420201433256687),
        T::from_f64(-0.25881904510252074),
        T::from_f64(-0.17364817766693041),
        T::from_f64(-0.08715574274765814),
        T::ZERO,
        T::from_f64(0.08715574274765814),
        T::from_f64(0.17364817766693041),
    ];
    #[rustfmt::skip]
    let sin_72: [T; 57] = [
        T::ZERO,
        T::from_f64(0.08715574274765817),
        T::from_f64(0.17364817766693033),
        T::from_f64(0.25881904510252074),
        T::from_f64(0.3420201433256687),
        T::from_f64(0.42261826174069944),
        T::from_f64(0.5),
        T::from_f64(0.573_576_436_351_046),
        T::from_f64(0.6427876096865393),
        T::from_f64(0.7071067811865476),
        T::from_f64(0.766044443118978),
        T::from_f64(0.8191520442889918),
        T::from_f64(0.8660254037844387),
        T::from_f64(0.9063077870366499),
        T::from_f64(0.9396926207859084),
        T::from_f64(0.9659258262890683),
        T::from_f64(0.984807753012208),
        T::from_f64(0.9961946980917455),
        T::ONE,
        T::from_f64(0.9961946980917455),
        T::from_f64(0.984807753012208),
        T::from_f64(0.9659258262890683),
        T::from_f64(0.9396926207859084),
        T::from_f64(0.9063077870366499),
        T::from_f64(0.8660254037844387),
        T::from_f64(0.8191520442889918),
        T::from_f64(0.766044443118978),
        T::from_f64(0.7071067811865476),
        T::from_f64(0.6427876096865394),
        T::from_f64(0.573_576_436_351_046),
        T::from_f64(0.5),
        T::from_f64(0.42261826174069944),
        T::from_f64(0.3420201433256687),
        T::from_f64(0.25881904510252074),
        T::from_f64(0.17364817766693033),
        T::from_f64(0.08715574274765817),
        T::ZERO,
        T::from_f64(-0.08715574274765817),
        T::from_f64(-0.17364817766693033),
        T::from_f64(-0.25881904510252074),
        T::from_f64(-0.3420201433256687),
        T::from_f64(-0.42261826174069944),
        T::from_f64(-0.5),
        T::from_f64(-0.573_576_436_351_046),
        T::from_f64(-0.6427876096865394),
        T::from_f64(-0.7071067811865476),
        T::from_f64(-0.766044443118978),
        T::from_f64(-0.8191520442889918),
        T::from_f64(-0.8660254037844387),
        T::from_f64(-0.9063077870366499),
        T::from_f64(-0.9396926207859084),
        T::from_f64(-0.9659258262890683),
        T::from_f64(-0.984807753012208),
        T::from_f64(-0.9961946980917455),
        -T::ONE,
        T::from_f64(-0.9961946980917455),
        T::from_f64(-0.984807753012208),
    ];
    for k1 in 1..9 {
        for k2 in 1..8 {
            let k = k1 * k2;
            let c = cos_72[k];
            let s = sin_72[k];
            let idx = k1 * 8 + k2;
            let tw = t[idx];
            t[idx] = Complex::new(
                tw.re * c - sign_t * tw.im * s,
                sign_t * tw.re * s + tw.im * c,
            );
        }
    }
    for k1 in 0..9 {
        let base = k1 * 8;
        simd::notw_8_dispatch(&mut t[base..base + 8], sign);
    }
    for k1 in 0..9 {
        let base = k1 * 8;
        for k2 in 0..8 {
            x[k2 * 9 + k1] = t[base + k2];
        }
    }
}
/// Optimized DFT of size 96.
///
/// Uses 8×12 mixed-radix decomposition for better SIMD utilization.
/// sign: -1 for forward, +1 for inverse
#[inline]
pub fn notw_96<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 96);
    let sign_t = if sign < 0 { -T::ONE } else { T::ONE };
    let mut t: [Complex<T>; 96] = [Complex::zero(); 96];
    for j in 0..12 {
        let mut temp8: [Complex<T>; 8] = [
            x[j],
            x[j + 12],
            x[j + 24],
            x[j + 36],
            x[j + 48],
            x[j + 60],
            x[j + 72],
            x[j + 84],
        ];
        simd::notw_8_dispatch(&mut temp8, sign);
        for k1 in 0..8 {
            t[k1 * 12 + j] = temp8[k1];
        }
    }
    #[rustfmt::skip]
    let cos_96: [T; 78] = [
        T::ONE,
        T::from_f64(0.9978589232386035),
        T::from_f64(0.9914448613738104),
        T::from_f64(0.9807852804032304),
        T::from_f64(0.9659258262890683),
        T::from_f64(0.9469301294951057),
        T::from_f64(0.9238795325112867),
        T::from_f64(0.8968727415326884),
        T::from_f64(0.8660254037844387),
        T::from_f64(0.831469612302545),
        T::from_f64(0.7933533402912352),
        T::from_f64(0.7518398074789774),
        T::from_f64(0.7071067811865476),
        T::from_f64(0.6593458151000688),
        T::from_f64(0.6087614290087207),
        T::from_f64(0.5555702330196022),
        T::from_f64(0.5),
        T::from_f64(0.44228869021900125),
        T::from_f64(0.38268343236508984),
        T::from_f64(0.3214394653031617),
        T::from_f64(0.25881904510252074),
        T::from_f64(0.19509032201612828),
        T::from_f64(0.1305261922200516),
        T::from_f64(0.06540312923014327),
        T::ZERO,
        T::from_f64(-0.06540312923014327),
        T::from_f64(-0.1305261922200516),
        T::from_f64(-0.19509032201612828),
        T::from_f64(-0.25881904510252074),
        T::from_f64(-0.3214394653031617),
        T::from_f64(-0.38268343236508984),
        T::from_f64(-0.44228869021900125),
        T::from_f64(-0.5),
        T::from_f64(-0.5555702330196022),
        T::from_f64(-0.6087614290087207),
        T::from_f64(-0.6593458151000688),
        T::from_f64(-0.7071067811865476),
        T::from_f64(-0.7518398074789774),
        T::from_f64(-0.7933533402912352),
        T::from_f64(-0.831469612302545),
        T::from_f64(-0.8660254037844387),
        T::from_f64(-0.8968727415326884),
        T::from_f64(-0.9238795325112867),
        T::from_f64(-0.9469301294951057),
        T::from_f64(-0.9659258262890683),
        T::from_f64(-0.9807852804032304),
        T::from_f64(-0.9914448613738104),
        T::from_f64(-0.9978589232386035),
        -T::ONE,
        T::from_f64(-0.9978589232386035),
        T::from_f64(-0.9914448613738104),
        T::from_f64(-0.9807852804032304),
        T::from_f64(-0.9659258262890683),
        T::from_f64(-0.9469301294951057),
        T::from_f64(-0.9238795325112867),
        T::from_f64(-0.8968727415326884),
        T::from_f64(-0.8660254037844387),
        T::from_f64(-0.831469612302545),
        T::from_f64(-0.7933533402912352),
        T::from_f64(-0.7518398074789774),
        T::from_f64(-0.7071067811865476),
        T::from_f64(-0.6593458151000688),
        T::from_f64(-0.6087614290087207),
        T::from_f64(-0.5555702330196022),
        T::from_f64(-0.5),
        T::from_f64(-0.44228869021900125),
        T::from_f64(-0.38268343236508984),
        T::from_f64(-0.3214394653031617),
        T::from_f64(-0.25881904510252074),
        T::from_f64(-0.19509032201612828),
        T::from_f64(-0.1305261922200516),
        T::from_f64(-0.06540312923014327),
        T::ZERO,
        T::from_f64(0.06540312923014327),
        T::from_f64(0.1305261922200516),
        T::from_f64(0.19509032201612828),
        T::from_f64(0.25881904510252074),
        T::from_f64(0.3214394653031617),
    ];
    #[rustfmt::skip]
    let sin_96: [T; 78] = [
        T::ZERO,
        T::from_f64(0.06540312923014327),
        T::from_f64(0.1305261922200516),
        T::from_f64(0.19509032201612828),
        T::from_f64(0.25881904510252074),
        T::from_f64(0.3214394653031617),
        T::from_f64(0.38268343236508984),
        T::from_f64(0.44228869021900125),
        T::from_f64(0.5),
        T::from_f64(0.5555702330196022),
        T::from_f64(0.6087614290087207),
        T::from_f64(0.6593458151000688),
        T::from_f64(0.7071067811865476),
        T::from_f64(0.7518398074789774),
        T::from_f64(0.7933533402912352),
        T::from_f64(0.831469612302545),
        T::from_f64(0.8660254037844387),
        T::from_f64(0.8968727415326884),
        T::from_f64(0.9238795325112867),
        T::from_f64(0.9469301294951057),
        T::from_f64(0.9659258262890683),
        T::from_f64(0.9807852804032304),
        T::from_f64(0.9914448613738104),
        T::from_f64(0.9978589232386035),
        T::ONE,
        T::from_f64(0.9978589232386035),
        T::from_f64(0.9914448613738104),
        T::from_f64(0.9807852804032304),
        T::from_f64(0.9659258262890683),
        T::from_f64(0.9469301294951057),
        T::from_f64(0.9238795325112867),
        T::from_f64(0.8968727415326884),
        T::from_f64(0.8660254037844387),
        T::from_f64(0.831469612302545),
        T::from_f64(0.7933533402912352),
        T::from_f64(0.7518398074789774),
        T::from_f64(0.7071067811865476),
        T::from_f64(0.6593458151000688),
        T::from_f64(0.6087614290087207),
        T::from_f64(0.5555702330196022),
        T::from_f64(0.5),
        T::from_f64(0.44228869021900125),
        T::from_f64(0.38268343236508984),
        T::from_f64(0.3214394653031617),
        T::from_f64(0.25881904510252074),
        T::from_f64(0.19509032201612828),
        T::from_f64(0.1305261922200516),
        T::from_f64(0.06540312923014327),
        T::ZERO,
        T::from_f64(-0.06540312923014327),
        T::from_f64(-0.1305261922200516),
        T::from_f64(-0.19509032201612828),
        T::from_f64(-0.25881904510252074),
        T::from_f64(-0.3214394653031617),
        T::from_f64(-0.38268343236508984),
        T::from_f64(-0.44228869021900125),
        T::from_f64(-0.5),
        T::from_f64(-0.5555702330196022),
        T::from_f64(-0.6087614290087207),
        T::from_f64(-0.6593458151000688),
        T::from_f64(-0.7071067811865476),
        T::from_f64(-0.7518398074789774),
        T::from_f64(-0.7933533402912352),
        T::from_f64(-0.831469612302545),
        T::from_f64(-0.8660254037844387),
        T::from_f64(-0.8968727415326884),
        T::from_f64(-0.9238795325112867),
        T::from_f64(-0.9469301294951057),
        T::from_f64(-0.9659258262890683),
        T::from_f64(-0.9807852804032304),
        T::from_f64(-0.9914448613738104),
        T::from_f64(-0.9978589232386035),
        -T::ONE,
        T::from_f64(-0.9978589232386035),
        T::from_f64(-0.9914448613738104),
        T::from_f64(-0.9807852804032304),
        T::from_f64(-0.9659258262890683),
        T::from_f64(-0.9469301294951057),
    ];
    for k1 in 1..8 {
        for k2 in 1..12 {
            let k = k1 * k2;
            let c = cos_96[k];
            let s = sin_96[k];
            let idx = k1 * 12 + k2;
            let tw = t[idx];
            t[idx] = Complex::new(
                tw.re * c - sign_t * tw.im * s,
                sign_t * tw.re * s + tw.im * c,
            );
        }
    }
    for k1 in 0..8 {
        let base = k1 * 12;
        notw_12(&mut t[base..base + 12], sign);
    }
    for k1 in 0..8 {
        let base = k1 * 12;
        for k2 in 0..12 {
            x[k2 * 8 + k1] = t[base + k2];
        }
    }
}
