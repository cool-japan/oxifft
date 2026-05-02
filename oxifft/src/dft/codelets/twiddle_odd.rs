//! Runtime DIT twiddle butterflies for odd radices 3, 5, 7.
//!
//! # Twiddle Layout Convention
//!
//! For a radix-`r` stage with inner stride `stride` (= product of all *later* radices),
//! the twiddle slice is laid out as:
//!
//! ```text
//! twiddles = [W^{1,0}, W^{1,1}, ..., W^{1,stride-1},
//!             W^{2,0}, W^{2,1}, ..., W^{2,stride-1},
//!             ...
//!             W^{r-1,0}, W^{r-1,1}, ..., W^{r-1,stride-1}]
//! ```
//!
//! where `W^{j,s} = exp(±2πi · j·s / current_n)` (forward: minus sign).
//!
//! Element at inner column `s`, slot `j` (j=1..r-1): `twiddles[(j-1)*stride + s]`.
//!
//! The data layout is non-interleaved: for a stage with `stride` and `blocks`,
//! the element at block `b`, row `j`, column `s` is:
//!   `io[b * r * stride + j * stride + s]`
//!
//! # Direction convention
//!
//! `_fwd` variants: DFT with `exp(-2πi k/N)` (FFTW forward sign).
//! `_bwd` variants: inverse DFT with `exp(+2πi k/N)`.

use crate::kernel::{Complex, Float};

// ─── Radix-3 DIT butterfly ────────────────────────────────────────────────────
//
// For 3 inputs x0, x1, x2 (after twiddle multiply on x1 and x2):
//   c3_re = cos(2π/3) = -0.5
//   s3    = sin(2π/3) = √3/2 ≈ 0.8660254037844387
//
//   t1 = x1 + x2
//   t2_re = x0.re + t1_re * c3_re,  t2_im = x0.im + t1_im * c3_re
//   For forward:  t3 = (x1 - x2) * s3 * i  →  (t3.re, t3.im) = (-(x1-x2).im * s3, (x1-x2).re * s3)
//   For backward: t3 = (x2 - x1) * s3 * i  →  (t3.re, t3.im) = (-(x2-x1).im * s3, (x2-x1).re * s3)
//
//   out[0] = x0 + t1
//   out[1] = t2 - t3    = (t2.re - t3.re, t2.im - t3.im)
//   out[2] = t2 + t3    = (t2.re + t3.re, t2.im + t3.im)

/// Radix-3 DIT twiddle butterfly, forward direction (exp(-2πi k/N)).
///
/// `io` must have length >= `blocks * 3 * stride`.
/// `twiddles` must have length >= `2 * stride`.
pub fn tw3_dit_fwd<T: Float>(
    io: &mut [Complex<T>],
    twiddles: &[Complex<T>],
    stride: usize,
    blocks: usize,
) {
    let c3_re = T::from_f64(-0.5_f64);
    let s3 = T::from_f64(0.866_025_403_784_438_7_f64);

    for b in 0..blocks {
        for s in 0..stride {
            let base = b * 3 * stride + s;
            let x0 = io[base];
            let x1 = io[base + stride] * twiddles[s];
            let x2 = io[base + 2 * stride] * twiddles[stride + s];

            let t1_re = x1.re + x2.re;
            let t1_im = x1.im + x2.im;

            let t2_re = x0.re + t1_re * c3_re;
            let t2_im = x0.im + t1_im * c3_re;

            // forward: t3 = (x1 - x2) rotated by +π/2 then scaled by s3
            // +π/2 rotation: (re, im) → (-im, re)
            let diff_re = x1.re - x2.re;
            let diff_im = x1.im - x2.im;
            let t3_re = -diff_im * s3;
            let t3_im = diff_re * s3;

            io[base] = Complex::new(x0.re + t1_re, x0.im + t1_im);
            io[base + stride] = Complex::new(t2_re - t3_re, t2_im - t3_im);
            io[base + 2 * stride] = Complex::new(t2_re + t3_re, t2_im + t3_im);
        }
    }
}

/// Radix-3 DIT twiddle butterfly, backward direction (exp(+2πi k/N)).
pub fn tw3_dit_bwd<T: Float>(
    io: &mut [Complex<T>],
    twiddles: &[Complex<T>],
    stride: usize,
    blocks: usize,
) {
    let c3_re = T::from_f64(-0.5_f64);
    let s3 = T::from_f64(0.866_025_403_784_438_7_f64);

    for b in 0..blocks {
        for s in 0..stride {
            let base = b * 3 * stride + s;
            let x0 = io[base];
            let x1 = io[base + stride] * twiddles[s];
            let x2 = io[base + 2 * stride] * twiddles[stride + s];

            let t1_re = x1.re + x2.re;
            let t1_im = x1.im + x2.im;

            let t2_re = x0.re + t1_re * c3_re;
            let t2_im = x0.im + t1_im * c3_re;

            // backward: conjugate of forward s3 rotation (negate s3 effect)
            let diff_re = x1.re - x2.re;
            let diff_im = x1.im - x2.im;
            let t3_re = diff_im * s3;
            let t3_im = -diff_re * s3;

            io[base] = Complex::new(x0.re + t1_re, x0.im + t1_im);
            io[base + stride] = Complex::new(t2_re - t3_re, t2_im - t3_im);
            io[base + 2 * stride] = Complex::new(t2_re + t3_re, t2_im + t3_im);
        }
    }
}

// ─── Radix-5 DIT butterfly ────────────────────────────────────────────────────
//
// Winograd minimum-multiply radix-5 DFT.
// Constants:
//   c1 = cos(2π/5) ≈  0.3090169943749474
//   c2 = cos(4π/5) ≈ -0.8090169943749473
//   s1 = sin(2π/5) ≈  0.9510565162951535
//   s2 = sin(4π/5) ≈  0.5877852522924731
//
// For 5 inputs x0..x4 (after twiddle on x1..x4):
//   ta = x1 + x4,  tb = x2 + x3
//   tc = x1 - x4,  td = x2 - x3
//   t5 = ta + tb
//   out[0] = x0 + t5
//   r1 = x0 + c1*ta + c2*tb
//   r2 = x0 + c2*ta + c1*tb
//   i1 = -(s1*tc + s2*td)    [forward]  or  +(s1*tc + s2*td)    [backward]
//   i2 = -(s2*tc - s1*td)    [forward]  or  +(s2*tc - s1*td)    [backward]
//   out[1] = r1 + i1 * i   (r1.re - i1, r1.im + i1.re) [i = sqrt(-1)]
//
//   "multiply by i" means (re,im) → (-im, re)
//   out[1].re = r1.re - i1,   out[1].im = r1.im + 0  → Wait, i1 is a scalar here
//
// Let's think in complex:
//   After forming r1 and r2 as complex numbers (re and im parts tracked separately):
//     r1_re = x0.re + c1*ta_re + c2*tb_re
//     r1_im = x0.im + c1*ta_im + c2*tb_im
//   Then i1 and i2 are REAL scalars multiplied by the imaginary unit i:
//     i1 is the imaginary component magnitude → multiply by i means rotate 90°
//   But wait — in complex DFT, the "imaginary" direction comes from the butterfly structure.
//
// Correct formulation: let tr = complex (ta, tb etc.), and construct full complex results:
//   ta = x1 + x4  (complex)
//   tb = x2 + x3  (complex)
//   tc = x1 - x4  (complex)
//   td = x2 - x3  (complex)
//   t5 = ta + tb  (complex)
//
//   r1 = x0 + c1*ta + c2*tb  (complex)
//   r2 = x0 + c2*ta + c1*tb  (complex)
//
// For forward (exp(-2πi k/N)):
//   q1 = (s1*tc + s2*td) * i  → multiply complex (s1*tc + s2*td) by i: (re,im)→(-im,re)
//   q2 = (s2*tc - s1*td) * i  → multiply complex (s2*tc - s1*td) by i: (re,im)→(-im,re)
//   out[1] = r1 - q1
//   out[2] = r2 - q2
//   out[3] = r2 + q2
//   out[4] = r1 + q1
//
// For backward (exp(+2πi k/N)): conjugate → multiply by -i instead
//   q1 = (s1*tc + s2*td) * (-i) → (re,im)→(im,-re)
//   q2 = (s2*tc - s1*td) * (-i)

/// Radix-5 DIT twiddle butterfly, forward direction.
///
/// `twiddles` must have length >= `4 * stride`.
pub fn tw5_dit_fwd<T: Float>(
    io: &mut [Complex<T>],
    twiddles: &[Complex<T>],
    stride: usize,
    blocks: usize,
) {
    let c1 = T::from_f64(0.309_016_994_374_947_45_f64);
    let c2 = T::from_f64(-0.809_016_994_374_947_3_f64);
    let s1 = T::from_f64(0.951_056_516_295_153_5_f64);
    let s2 = T::from_f64(0.587_785_252_292_473_2_f64);

    for b in 0..blocks {
        for s in 0..stride {
            let base = b * 5 * stride + s;
            let x0 = io[base];
            let x1 = io[base + stride] * twiddles[s];
            let x2 = io[base + 2 * stride] * twiddles[stride + s];
            let x3 = io[base + 3 * stride] * twiddles[2 * stride + s];
            let x4 = io[base + 4 * stride] * twiddles[3 * stride + s];

            let ta = Complex::new(x1.re + x4.re, x1.im + x4.im);
            let tb = Complex::new(x2.re + x3.re, x2.im + x3.im);
            let tc = Complex::new(x1.re - x4.re, x1.im - x4.im);
            let td = Complex::new(x2.re - x3.re, x2.im - x3.im);

            let t5_re = ta.re + tb.re;
            let t5_im = ta.im + tb.im;

            let r1 = Complex::new(
                x0.re + c1 * ta.re + c2 * tb.re,
                x0.im + c1 * ta.im + c2 * tb.im,
            );
            let r2 = Complex::new(
                x0.re + c2 * ta.re + c1 * tb.re,
                x0.im + c2 * ta.im + c1 * tb.im,
            );

            // q1 = (s1*tc + s2*td) * i  → multiply by i: (re,im) → (-im, re)
            let u1 = Complex::new(s1 * tc.re + s2 * td.re, s1 * tc.im + s2 * td.im);
            let q1 = Complex::new(-u1.im, u1.re);

            // q2 = (s2*tc - s1*td) * i
            let u2 = Complex::new(s2 * tc.re - s1 * td.re, s2 * tc.im - s1 * td.im);
            let q2 = Complex::new(-u2.im, u2.re);

            io[base] = Complex::new(x0.re + t5_re, x0.im + t5_im);
            io[base + stride] = Complex::new(r1.re - q1.re, r1.im - q1.im);
            io[base + 2 * stride] = Complex::new(r2.re - q2.re, r2.im - q2.im);
            io[base + 3 * stride] = Complex::new(r2.re + q2.re, r2.im + q2.im);
            io[base + 4 * stride] = Complex::new(r1.re + q1.re, r1.im + q1.im);
        }
    }
}

/// Radix-5 DIT twiddle butterfly, backward direction.
pub fn tw5_dit_bwd<T: Float>(
    io: &mut [Complex<T>],
    twiddles: &[Complex<T>],
    stride: usize,
    blocks: usize,
) {
    let c1 = T::from_f64(0.309_016_994_374_947_45_f64);
    let c2 = T::from_f64(-0.809_016_994_374_947_3_f64);
    let s1 = T::from_f64(0.951_056_516_295_153_5_f64);
    let s2 = T::from_f64(0.587_785_252_292_473_2_f64);

    for b in 0..blocks {
        for s in 0..stride {
            let base = b * 5 * stride + s;
            let x0 = io[base];
            let x1 = io[base + stride] * twiddles[s];
            let x2 = io[base + 2 * stride] * twiddles[stride + s];
            let x3 = io[base + 3 * stride] * twiddles[2 * stride + s];
            let x4 = io[base + 4 * stride] * twiddles[3 * stride + s];

            let ta = Complex::new(x1.re + x4.re, x1.im + x4.im);
            let tb = Complex::new(x2.re + x3.re, x2.im + x3.im);
            let tc = Complex::new(x1.re - x4.re, x1.im - x4.im);
            let td = Complex::new(x2.re - x3.re, x2.im - x3.im);

            let t5_re = ta.re + tb.re;
            let t5_im = ta.im + tb.im;

            let r1 = Complex::new(
                x0.re + c1 * ta.re + c2 * tb.re,
                x0.im + c1 * ta.im + c2 * tb.im,
            );
            let r2 = Complex::new(
                x0.re + c2 * ta.re + c1 * tb.re,
                x0.im + c2 * ta.im + c1 * tb.im,
            );

            // backward: multiply by -i instead of +i: (re,im) → (im, -re)
            let u1 = Complex::new(s1 * tc.re + s2 * td.re, s1 * tc.im + s2 * td.im);
            let q1 = Complex::new(u1.im, -u1.re);

            let u2 = Complex::new(s2 * tc.re - s1 * td.re, s2 * tc.im - s1 * td.im);
            let q2 = Complex::new(u2.im, -u2.re);

            io[base] = Complex::new(x0.re + t5_re, x0.im + t5_im);
            io[base + stride] = Complex::new(r1.re - q1.re, r1.im - q1.im);
            io[base + 2 * stride] = Complex::new(r2.re - q2.re, r2.im - q2.im);
            io[base + 3 * stride] = Complex::new(r2.re + q2.re, r2.im + q2.im);
            io[base + 4 * stride] = Complex::new(r1.re + q1.re, r1.im + q1.im);
        }
    }
}

// ─── Radix-7 DIT butterfly ────────────────────────────────────────────────────
//
// Winograd-style radix-7 DFT using cosine/sine constants.
// c1 = cos(2π/7), c2 = cos(4π/7), c3 = cos(6π/7)
// s1 = sin(2π/7), s2 = sin(4π/7), s3 = sin(6π/7)
//
// For 7 inputs x0..x6 (after twiddle on x1..x6):
//   t1 = x1 + x6,  t2 = x2 + x5,  t3 = x3 + x4
//   t4 = x1 - x6,  t5 = x2 - x5,  t6 = x3 - x4
//   t7 = t1 + t2 + t3
//
//   out[0] = x0 + t7
//
//   r1 = x0 + c1*t1 + c2*t2 + c3*t3
//   r2 = x0 + c2*t1 + c3*t2 + c1*t3
//   r3 = x0 + c3*t1 + c1*t2 + c2*t3
//
// For forward: imaginary parts come from rotating odd-differences by i:
//   q1 = (s1*t4 + s2*t5 + s3*t6) * i  → (re,im)→(-im,re)
//   q2 = (s2*t4 - s3*t5 - s1*t6) * i
//   q3 = (s3*t4 - s1*t5 + s2*t6) * i
//
//   out[1] = r1 - q1
//   out[2] = r2 - q2
//   out[3] = r3 - q3
//   out[4] = r3 + q3
//   out[5] = r2 + q2
//   out[6] = r1 + q1
//
// For backward: multiply by -i → (re,im)→(im,-re)

/// Radix-7 DIT twiddle butterfly, forward direction.
///
/// `twiddles` must have length >= `6 * stride`.
#[allow(clippy::too_many_lines)]
pub fn tw7_dit_fwd<T: Float>(
    io: &mut [Complex<T>],
    twiddles: &[Complex<T>],
    stride: usize,
    blocks: usize,
) {
    let c1 = T::from_f64(0.623_489_801_858_733_6_f64);
    let c2 = T::from_f64(-0.222_520_933_956_314_34_f64);
    let c3 = T::from_f64(-0.900_968_867_902_419_f64);
    let s1 = T::from_f64(0.781_831_482_468_029_8_f64);
    let s2 = T::from_f64(0.974_927_912_181_823_6_f64);
    let s3 = T::from_f64(0.433_883_739_117_558_23_f64);

    for b in 0..blocks {
        for s in 0..stride {
            let base = b * 7 * stride + s;
            let x0 = io[base];
            let x1 = io[base + stride] * twiddles[s];
            let x2 = io[base + 2 * stride] * twiddles[stride + s];
            let x3 = io[base + 3 * stride] * twiddles[2 * stride + s];
            let x4 = io[base + 4 * stride] * twiddles[3 * stride + s];
            let x5 = io[base + 5 * stride] * twiddles[4 * stride + s];
            let x6 = io[base + 6 * stride] * twiddles[5 * stride + s];

            let t1 = Complex::new(x1.re + x6.re, x1.im + x6.im);
            let t2 = Complex::new(x2.re + x5.re, x2.im + x5.im);
            let t3 = Complex::new(x3.re + x4.re, x3.im + x4.im);
            let t4 = Complex::new(x1.re - x6.re, x1.im - x6.im);
            let t5 = Complex::new(x2.re - x5.re, x2.im - x5.im);
            let t6 = Complex::new(x3.re - x4.re, x3.im - x4.im);

            let t7_re = t1.re + t2.re + t3.re;
            let t7_im = t1.im + t2.im + t3.im;

            let r1 = Complex::new(
                x0.re + c1 * t1.re + c2 * t2.re + c3 * t3.re,
                x0.im + c1 * t1.im + c2 * t2.im + c3 * t3.im,
            );
            let r2 = Complex::new(
                x0.re + c2 * t1.re + c3 * t2.re + c1 * t3.re,
                x0.im + c2 * t1.im + c3 * t2.im + c1 * t3.im,
            );
            let r3 = Complex::new(
                x0.re + c3 * t1.re + c1 * t2.re + c2 * t3.re,
                x0.im + c3 * t1.im + c1 * t2.im + c2 * t3.im,
            );

            // forward: multiply by i → (re,im) → (-im, re)
            let u1 = Complex::new(
                s1 * t4.re + s2 * t5.re + s3 * t6.re,
                s1 * t4.im + s2 * t5.im + s3 * t6.im,
            );
            let q1 = Complex::new(-u1.im, u1.re);

            let u2 = Complex::new(
                s2 * t4.re - s3 * t5.re - s1 * t6.re,
                s2 * t4.im - s3 * t5.im - s1 * t6.im,
            );
            let q2 = Complex::new(-u2.im, u2.re);

            let u3 = Complex::new(
                s3 * t4.re - s1 * t5.re + s2 * t6.re,
                s3 * t4.im - s1 * t5.im + s2 * t6.im,
            );
            let q3 = Complex::new(-u3.im, u3.re);

            io[base] = Complex::new(x0.re + t7_re, x0.im + t7_im);
            io[base + stride] = Complex::new(r1.re - q1.re, r1.im - q1.im);
            io[base + 2 * stride] = Complex::new(r2.re - q2.re, r2.im - q2.im);
            io[base + 3 * stride] = Complex::new(r3.re - q3.re, r3.im - q3.im);
            io[base + 4 * stride] = Complex::new(r3.re + q3.re, r3.im + q3.im);
            io[base + 5 * stride] = Complex::new(r2.re + q2.re, r2.im + q2.im);
            io[base + 6 * stride] = Complex::new(r1.re + q1.re, r1.im + q1.im);
        }
    }
}

/// Radix-7 DIT twiddle butterfly, backward direction.
#[allow(clippy::too_many_lines)]
pub fn tw7_dit_bwd<T: Float>(
    io: &mut [Complex<T>],
    twiddles: &[Complex<T>],
    stride: usize,
    blocks: usize,
) {
    let c1 = T::from_f64(0.623_489_801_858_733_6_f64);
    let c2 = T::from_f64(-0.222_520_933_956_314_34_f64);
    let c3 = T::from_f64(-0.900_968_867_902_419_f64);
    let s1 = T::from_f64(0.781_831_482_468_029_8_f64);
    let s2 = T::from_f64(0.974_927_912_181_823_6_f64);
    let s3 = T::from_f64(0.433_883_739_117_558_23_f64);

    for b in 0..blocks {
        for s in 0..stride {
            let base = b * 7 * stride + s;
            let x0 = io[base];
            let x1 = io[base + stride] * twiddles[s];
            let x2 = io[base + 2 * stride] * twiddles[stride + s];
            let x3 = io[base + 3 * stride] * twiddles[2 * stride + s];
            let x4 = io[base + 4 * stride] * twiddles[3 * stride + s];
            let x5 = io[base + 5 * stride] * twiddles[4 * stride + s];
            let x6 = io[base + 6 * stride] * twiddles[5 * stride + s];

            let t1 = Complex::new(x1.re + x6.re, x1.im + x6.im);
            let t2 = Complex::new(x2.re + x5.re, x2.im + x5.im);
            let t3 = Complex::new(x3.re + x4.re, x3.im + x4.im);
            let t4 = Complex::new(x1.re - x6.re, x1.im - x6.im);
            let t5 = Complex::new(x2.re - x5.re, x2.im - x5.im);
            let t6 = Complex::new(x3.re - x4.re, x3.im - x4.im);

            let t7_re = t1.re + t2.re + t3.re;
            let t7_im = t1.im + t2.im + t3.im;

            let r1 = Complex::new(
                x0.re + c1 * t1.re + c2 * t2.re + c3 * t3.re,
                x0.im + c1 * t1.im + c2 * t2.im + c3 * t3.im,
            );
            let r2 = Complex::new(
                x0.re + c2 * t1.re + c3 * t2.re + c1 * t3.re,
                x0.im + c2 * t1.im + c3 * t2.im + c1 * t3.im,
            );
            let r3 = Complex::new(
                x0.re + c3 * t1.re + c1 * t2.re + c2 * t3.re,
                x0.im + c3 * t1.im + c1 * t2.im + c2 * t3.im,
            );

            // backward: multiply by -i → (re,im) → (im, -re)
            let u1 = Complex::new(
                s1 * t4.re + s2 * t5.re + s3 * t6.re,
                s1 * t4.im + s2 * t5.im + s3 * t6.im,
            );
            let q1 = Complex::new(u1.im, -u1.re);

            let u2 = Complex::new(
                s2 * t4.re - s3 * t5.re - s1 * t6.re,
                s2 * t4.im - s3 * t5.im - s1 * t6.im,
            );
            let q2 = Complex::new(u2.im, -u2.re);

            let u3 = Complex::new(
                s3 * t4.re - s1 * t5.re + s2 * t6.re,
                s3 * t4.im - s1 * t5.im + s2 * t6.im,
            );
            let q3 = Complex::new(u3.im, -u3.re);

            io[base] = Complex::new(x0.re + t7_re, x0.im + t7_im);
            io[base + stride] = Complex::new(r1.re - q1.re, r1.im - q1.im);
            io[base + 2 * stride] = Complex::new(r2.re - q2.re, r2.im - q2.im);
            io[base + 3 * stride] = Complex::new(r3.re - q3.re, r3.im - q3.im);
            io[base + 4 * stride] = Complex::new(r3.re + q3.re, r3.im + q3.im);
            io[base + 5 * stride] = Complex::new(r2.re + q2.re, r2.im + q2.im);
            io[base + 6 * stride] = Complex::new(r1.re + q1.re, r1.im + q1.im);
        }
    }
}

// ─── Radix-2/4/8/16 DIT butterfly wrappers ───────────────────────────────────
//
// These wrap the existing butterfly functions for uniform dispatch in
// the mixed-radix executor.

/// Radix-2 DIT twiddle butterfly, forward.
/// Layout: twiddles[(j-1)*stride + s] for j=1, i.e., twiddles[s].
pub fn tw2_dit_fwd<T: Float>(
    io: &mut [Complex<T>],
    twiddles: &[Complex<T>],
    stride: usize,
    blocks: usize,
) {
    for b in 0..blocks {
        for s in 0..stride {
            let base = b * 2 * stride + s;
            let x0 = io[base];
            let x1 = io[base + stride] * twiddles[s];
            io[base] = Complex::new(x0.re + x1.re, x0.im + x1.im);
            io[base + stride] = Complex::new(x0.re - x1.re, x0.im - x1.im);
        }
    }
}

/// Radix-2 DIT twiddle butterfly, backward.
pub fn tw2_dit_bwd<T: Float>(
    io: &mut [Complex<T>],
    twiddles: &[Complex<T>],
    stride: usize,
    blocks: usize,
) {
    // Same butterfly — backward twiddles are conjugates (handled by caller)
    tw2_dit_fwd(io, twiddles, stride, blocks);
}

/// Radix-4 DIT twiddle butterfly, forward.
/// Layout: twiddles[(j-1)*stride + s] for j=1,2,3.
pub fn tw4_dit_fwd<T: Float>(
    io: &mut [Complex<T>],
    twiddles: &[Complex<T>],
    stride: usize,
    blocks: usize,
) {
    for b in 0..blocks {
        for s in 0..stride {
            let base = b * 4 * stride + s;
            let x0 = io[base];
            let x1 = io[base + stride] * twiddles[s];
            let x2 = io[base + 2 * stride] * twiddles[stride + s];
            let x3 = io[base + 3 * stride] * twiddles[2 * stride + s];

            let t0 = Complex::new(x0.re + x2.re, x0.im + x2.im);
            let t1 = Complex::new(x0.re - x2.re, x0.im - x2.im);
            let t2 = Complex::new(x1.re + x3.re, x1.im + x3.im);
            let t3 = Complex::new(x1.re - x3.re, x1.im - x3.im);

            // forward: t3 rotated by -i → (re,im)→(im,-re)
            let t3r = Complex::new(t3.im, -t3.re);

            io[base] = Complex::new(t0.re + t2.re, t0.im + t2.im);
            io[base + stride] = Complex::new(t1.re + t3r.re, t1.im + t3r.im);
            io[base + 2 * stride] = Complex::new(t0.re - t2.re, t0.im - t2.im);
            io[base + 3 * stride] = Complex::new(t1.re - t3r.re, t1.im - t3r.im);
        }
    }
}

/// Radix-4 DIT twiddle butterfly, backward.
pub fn tw4_dit_bwd<T: Float>(
    io: &mut [Complex<T>],
    twiddles: &[Complex<T>],
    stride: usize,
    blocks: usize,
) {
    for b in 0..blocks {
        for s in 0..stride {
            let base = b * 4 * stride + s;
            let x0 = io[base];
            let x1 = io[base + stride] * twiddles[s];
            let x2 = io[base + 2 * stride] * twiddles[stride + s];
            let x3 = io[base + 3 * stride] * twiddles[2 * stride + s];

            let t0 = Complex::new(x0.re + x2.re, x0.im + x2.im);
            let t1 = Complex::new(x0.re - x2.re, x0.im - x2.im);
            let t2 = Complex::new(x1.re + x3.re, x1.im + x3.im);
            let t3 = Complex::new(x1.re - x3.re, x1.im - x3.im);

            // backward: t3 rotated by +i → (re,im)→(-im,re)
            let t3r = Complex::new(-t3.im, t3.re);

            io[base] = Complex::new(t0.re + t2.re, t0.im + t2.im);
            io[base + stride] = Complex::new(t1.re + t3r.re, t1.im + t3r.im);
            io[base + 2 * stride] = Complex::new(t0.re - t2.re, t0.im - t2.im);
            io[base + 3 * stride] = Complex::new(t1.re - t3r.re, t1.im - t3r.im);
        }
    }
}

/// Radix-8 DIT twiddle butterfly, forward.
/// Layout: twiddles[(j-1)*stride + s] for j=1..7.
pub fn tw8_dit_fwd<T: Float>(
    io: &mut [Complex<T>],
    twiddles: &[Complex<T>],
    stride: usize,
    blocks: usize,
) {
    let c = T::from_f64(0.707_106_781_186_547_6_f64); // 1/sqrt(2)
    for b in 0..blocks {
        for s in 0..stride {
            let base = b * 8 * stride + s;
            let x0 = io[base];
            let x1 = io[base + stride] * twiddles[s];
            let x2 = io[base + 2 * stride] * twiddles[stride + s];
            let x3 = io[base + 3 * stride] * twiddles[2 * stride + s];
            let x4 = io[base + 4 * stride] * twiddles[3 * stride + s];
            let x5 = io[base + 5 * stride] * twiddles[4 * stride + s];
            let x6 = io[base + 6 * stride] * twiddles[5 * stride + s];
            let x7 = io[base + 7 * stride] * twiddles[6 * stride + s];

            // Radix-8 DIT: decompose into two radix-4 halves
            let a0 = Complex::new(x0.re + x4.re, x0.im + x4.im);
            let a1 = Complex::new(x0.re - x4.re, x0.im - x4.im);
            let a2 = Complex::new(x2.re + x6.re, x2.im + x6.im);
            // forward: (x2-x6) * (-i)
            let d26 = Complex::new(x2.re - x6.re, x2.im - x6.im);
            let a3 = Complex::new(d26.im, -d26.re);
            let a4 = Complex::new(x1.re + x5.re, x1.im + x5.im);
            let a5 = Complex::new(x1.re - x5.re, x1.im - x5.im);
            let a6 = Complex::new(x3.re + x7.re, x3.im + x7.im);
            let d37 = Complex::new(x3.re - x7.re, x3.im - x7.im);
            let a7 = Complex::new(d37.im, -d37.re);

            let b0 = Complex::new(a0.re + a2.re, a0.im + a2.im);
            let b1 = Complex::new(a1.re + a3.re, a1.im + a3.im);
            let b2 = Complex::new(a0.re - a2.re, a0.im - a2.im);
            let b3 = Complex::new(a1.re - a3.re, a1.im - a3.im);

            let b4 = Complex::new(a4.re + a6.re, a4.im + a6.im);
            // forward: (a5 + a7) * (c - ic) = (a5+a7) * c * (1-i)
            let d5a7 = Complex::new(a5.re + a7.re, a5.im + a7.im);
            let b5 = Complex::new((d5a7.re + d5a7.im) * c, (d5a7.im - d5a7.re) * c);
            // forward: (a4-a6) * (-i)
            let d4a6 = Complex::new(a4.re - a6.re, a4.im - a6.im);
            let b6 = Complex::new(d4a6.im, -d4a6.re);
            // forward: (a5 - a7) * (-c - ic) = (a5-a7) * c * (-1-i)
            let d5s7 = Complex::new(a5.re - a7.re, a5.im - a7.im);
            let b7 = Complex::new((-d5s7.re + d5s7.im) * c, (-d5s7.im - d5s7.re) * c);

            io[base] = Complex::new(b0.re + b4.re, b0.im + b4.im);
            io[base + stride] = Complex::new(b1.re + b5.re, b1.im + b5.im);
            io[base + 2 * stride] = Complex::new(b2.re + b6.re, b2.im + b6.im);
            io[base + 3 * stride] = Complex::new(b3.re + b7.re, b3.im + b7.im);
            io[base + 4 * stride] = Complex::new(b0.re - b4.re, b0.im - b4.im);
            io[base + 5 * stride] = Complex::new(b1.re - b5.re, b1.im - b5.im);
            io[base + 6 * stride] = Complex::new(b2.re - b6.re, b2.im - b6.im);
            io[base + 7 * stride] = Complex::new(b3.re - b7.re, b3.im - b7.im);
        }
    }
}

/// Radix-8 DIT twiddle butterfly, backward.
pub fn tw8_dit_bwd<T: Float>(
    io: &mut [Complex<T>],
    twiddles: &[Complex<T>],
    stride: usize,
    blocks: usize,
) {
    let c = T::from_f64(0.707_106_781_186_547_6_f64); // 1/sqrt(2)
    for b in 0..blocks {
        for s in 0..stride {
            let base = b * 8 * stride + s;
            let x0 = io[base];
            let x1 = io[base + stride] * twiddles[s];
            let x2 = io[base + 2 * stride] * twiddles[stride + s];
            let x3 = io[base + 3 * stride] * twiddles[2 * stride + s];
            let x4 = io[base + 4 * stride] * twiddles[3 * stride + s];
            let x5 = io[base + 5 * stride] * twiddles[4 * stride + s];
            let x6 = io[base + 6 * stride] * twiddles[5 * stride + s];
            let x7 = io[base + 7 * stride] * twiddles[6 * stride + s];

            let a0 = Complex::new(x0.re + x4.re, x0.im + x4.im);
            let a1 = Complex::new(x0.re - x4.re, x0.im - x4.im);
            let a2 = Complex::new(x2.re + x6.re, x2.im + x6.im);
            // backward: (x2-x6) * (+i)
            let d26 = Complex::new(x2.re - x6.re, x2.im - x6.im);
            let a3 = Complex::new(-d26.im, d26.re);
            let a4 = Complex::new(x1.re + x5.re, x1.im + x5.im);
            let a5 = Complex::new(x1.re - x5.re, x1.im - x5.im);
            let a6 = Complex::new(x3.re + x7.re, x3.im + x7.im);
            let d37 = Complex::new(x3.re - x7.re, x3.im - x7.im);
            let a7 = Complex::new(-d37.im, d37.re);

            let b0 = Complex::new(a0.re + a2.re, a0.im + a2.im);
            let b1 = Complex::new(a1.re + a3.re, a1.im + a3.im);
            let b2 = Complex::new(a0.re - a2.re, a0.im - a2.im);
            let b3 = Complex::new(a1.re - a3.re, a1.im - a3.im);

            let b4 = Complex::new(a4.re + a6.re, a4.im + a6.im);
            // backward: (a5 + a7) * c * (1+i)
            let d5a7 = Complex::new(a5.re + a7.re, a5.im + a7.im);
            let b5 = Complex::new((d5a7.re - d5a7.im) * c, (d5a7.im + d5a7.re) * c);
            // backward: (a4-a6) * (+i)
            let d4a6 = Complex::new(a4.re - a6.re, a4.im - a6.im);
            let b6 = Complex::new(-d4a6.im, d4a6.re);
            // backward: (a5 - a7) * c * (-1+i)
            let d5s7 = Complex::new(a5.re - a7.re, a5.im - a7.im);
            let b7 = Complex::new((-d5s7.re - d5s7.im) * c, (-d5s7.im + d5s7.re) * c);

            io[base] = Complex::new(b0.re + b4.re, b0.im + b4.im);
            io[base + stride] = Complex::new(b1.re + b5.re, b1.im + b5.im);
            io[base + 2 * stride] = Complex::new(b2.re + b6.re, b2.im + b6.im);
            io[base + 3 * stride] = Complex::new(b3.re + b7.re, b3.im + b7.im);
            io[base + 4 * stride] = Complex::new(b0.re - b4.re, b0.im - b4.im);
            io[base + 5 * stride] = Complex::new(b1.re - b5.re, b1.im - b5.im);
            io[base + 6 * stride] = Complex::new(b2.re - b6.re, b2.im - b6.im);
            io[base + 7 * stride] = Complex::new(b3.re - b7.re, b3.im - b7.im);
        }
    }
}

/// Radix-16 DIT twiddle butterfly, forward.
/// Layout: twiddles[(j-1)*stride + s] for j=1..15.
#[allow(clippy::too_many_lines)]
pub fn tw16_dit_fwd<T: Float>(
    io: &mut [Complex<T>],
    twiddles: &[Complex<T>],
    stride: usize,
    blocks: usize,
) {
    // Implement as two radix-8 half-pass + combine
    // Simpler: implement as 4-stage DIT after applying twiddles
    let c2 = T::from_f64(0.707_106_781_186_547_6_f64);
    let c1 = T::from_f64(0.923_879_532_511_286_7_f64);
    let s1 = T::from_f64(0.382_683_432_365_089_8_f64);

    for b in 0..blocks {
        for s in 0..stride {
            let base = b * 16 * stride + s;

            // Apply twiddles
            let x: [Complex<T>; 16] = {
                let mut arr = [Complex::<T>::zero(); 16];
                arr[0] = io[base];
                for j in 1..16 {
                    arr[j] = io[base + j * stride] * twiddles[(j - 1) * stride + s];
                }
                arr
            };

            // Bit-reversal permutation for 16-point DIT
            let mut a = [Complex::<T>::zero(); 16];
            // bit-reversal: 0→0,1→8,2→4,3→12,4→2,5→10,6→6,7→14,
            //               8→1,9→9,10→5,11→13,12→3,13→11,14→7,15→15
            let br = [0usize, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];
            for i in 0..16 {
                a[i] = x[br[i]];
            }

            // Stage 1: span 1
            for i in (0..16).step_by(2) {
                let t = a[i + 1];
                a[i + 1] = Complex::new(a[i].re - t.re, a[i].im - t.im);
                a[i] = Complex::new(a[i].re + t.re, a[i].im + t.im);
            }

            // Stage 2: span 2, W4^0=1, W4^1=-i (fwd)
            for g in (0..16).step_by(4) {
                let t = a[g + 2];
                a[g + 2] = Complex::new(a[g].re - t.re, a[g].im - t.im);
                a[g] = Complex::new(a[g].re + t.re, a[g].im + t.im);
                let t = a[g + 3];
                let tr = Complex::new(t.im, -t.re); // *(-i)
                a[g + 3] = Complex::new(a[g + 1].re - tr.re, a[g + 1].im - tr.im);
                a[g + 1] = Complex::new(a[g + 1].re + tr.re, a[g + 1].im + tr.im);
            }

            // Stage 3: span 4, W8^k for k=0..3
            for g in (0..16).step_by(8) {
                // k=0: *1
                let t = a[g + 4];
                a[g + 4] = Complex::new(a[g].re - t.re, a[g].im - t.im);
                a[g] = Complex::new(a[g].re + t.re, a[g].im + t.im);
                // k=1: W8^1 = (1-i)*c2
                let t = a[g + 5];
                let tr = Complex::new((t.re + t.im) * c2, (t.im - t.re) * c2);
                a[g + 5] = Complex::new(a[g + 1].re - tr.re, a[g + 1].im - tr.im);
                a[g + 1] = Complex::new(a[g + 1].re + tr.re, a[g + 1].im + tr.im);
                // k=2: W8^2 = -i
                let t = a[g + 6];
                let tr = Complex::new(t.im, -t.re);
                a[g + 6] = Complex::new(a[g + 2].re - tr.re, a[g + 2].im - tr.im);
                a[g + 2] = Complex::new(a[g + 2].re + tr.re, a[g + 2].im + tr.im);
                // k=3: W8^3 = (-1-i)*c2
                let t = a[g + 7];
                let tr = Complex::new((-t.re + t.im) * c2, (-t.im - t.re) * c2);
                a[g + 7] = Complex::new(a[g + 3].re - tr.re, a[g + 3].im - tr.im);
                a[g + 3] = Complex::new(a[g + 3].re + tr.re, a[g + 3].im + tr.im);
            }

            // Stage 4: span 8, W16^k for k=0..7
            {
                // k=0
                let t = a[8];
                a[8] = Complex::new(a[0].re - t.re, a[0].im - t.im);
                a[0] = Complex::new(a[0].re + t.re, a[0].im + t.im);
                // k=1: cos(π/8)-i*sin(π/8)
                let t = a[9];
                let tr = Complex::new(t.re * c1 + t.im * s1, t.im * c1 - t.re * s1);
                a[9] = Complex::new(a[1].re - tr.re, a[1].im - tr.im);
                a[1] = Complex::new(a[1].re + tr.re, a[1].im + tr.im);
                // k=2: (1-i)*c2
                let t = a[10];
                let tr = Complex::new((t.re + t.im) * c2, (t.im - t.re) * c2);
                a[10] = Complex::new(a[2].re - tr.re, a[2].im - tr.im);
                a[2] = Complex::new(a[2].re + tr.re, a[2].im + tr.im);
                // k=3: sin(π/8)-i*cos(π/8)
                let t = a[11];
                let tr = Complex::new(t.re * s1 + t.im * c1, t.im * s1 - t.re * c1);
                a[11] = Complex::new(a[3].re - tr.re, a[3].im - tr.im);
                a[3] = Complex::new(a[3].re + tr.re, a[3].im + tr.im);
                // k=4: -i
                let t = a[12];
                let tr = Complex::new(t.im, -t.re);
                a[12] = Complex::new(a[4].re - tr.re, a[4].im - tr.im);
                a[4] = Complex::new(a[4].re + tr.re, a[4].im + tr.im);
                // k=5: -sin(π/8)-i*cos(π/8)
                let t = a[13];
                let tr = Complex::new(-t.re * s1 + t.im * c1, -t.im * s1 - t.re * c1);
                a[13] = Complex::new(a[5].re - tr.re, a[5].im - tr.im);
                a[5] = Complex::new(a[5].re + tr.re, a[5].im + tr.im);
                // k=6: (-1-i)*c2
                let t = a[14];
                let tr = Complex::new((-t.re + t.im) * c2, (-t.im - t.re) * c2);
                a[14] = Complex::new(a[6].re - tr.re, a[6].im - tr.im);
                a[6] = Complex::new(a[6].re + tr.re, a[6].im + tr.im);
                // k=7: -cos(π/8)-i*sin(π/8)
                let t = a[15];
                let tr = Complex::new(-t.re * c1 + t.im * s1, -t.im * c1 - t.re * s1);
                a[15] = Complex::new(a[7].re - tr.re, a[7].im - tr.im);
                a[7] = Complex::new(a[7].re + tr.re, a[7].im + tr.im);
            }

            for j in 0..16 {
                io[base + j * stride] = a[j];
            }
        }
    }
}

/// Radix-16 DIT twiddle butterfly, backward.
#[allow(clippy::too_many_lines)]
pub fn tw16_dit_bwd<T: Float>(
    io: &mut [Complex<T>],
    twiddles: &[Complex<T>],
    stride: usize,
    blocks: usize,
) {
    let c2 = T::from_f64(0.707_106_781_186_547_6_f64);
    let c1 = T::from_f64(0.923_879_532_511_286_7_f64);
    let s1 = T::from_f64(0.382_683_432_365_089_8_f64);

    for b in 0..blocks {
        for s in 0..stride {
            let base = b * 16 * stride + s;

            let x: [Complex<T>; 16] = {
                let mut arr = [Complex::<T>::zero(); 16];
                arr[0] = io[base];
                for j in 1..16 {
                    arr[j] = io[base + j * stride] * twiddles[(j - 1) * stride + s];
                }
                arr
            };

            let mut a = [Complex::<T>::zero(); 16];
            let br = [0usize, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];
            for i in 0..16 {
                a[i] = x[br[i]];
            }

            // Stage 1: span 1
            for i in (0..16).step_by(2) {
                let t = a[i + 1];
                a[i + 1] = Complex::new(a[i].re - t.re, a[i].im - t.im);
                a[i] = Complex::new(a[i].re + t.re, a[i].im + t.im);
            }

            // Stage 2: W4^1 = +i for backward
            for g in (0..16).step_by(4) {
                let t = a[g + 2];
                a[g + 2] = Complex::new(a[g].re - t.re, a[g].im - t.im);
                a[g] = Complex::new(a[g].re + t.re, a[g].im + t.im);
                let t = a[g + 3];
                let tr = Complex::new(-t.im, t.re); // *(+i)
                a[g + 3] = Complex::new(a[g + 1].re - tr.re, a[g + 1].im - tr.im);
                a[g + 1] = Complex::new(a[g + 1].re + tr.re, a[g + 1].im + tr.im);
            }

            // Stage 3: W8^k conjugates
            for g in (0..16).step_by(8) {
                let t = a[g + 4];
                a[g + 4] = Complex::new(a[g].re - t.re, a[g].im - t.im);
                a[g] = Complex::new(a[g].re + t.re, a[g].im + t.im);
                let t = a[g + 5];
                let tr = Complex::new((t.re - t.im) * c2, (t.im + t.re) * c2);
                a[g + 5] = Complex::new(a[g + 1].re - tr.re, a[g + 1].im - tr.im);
                a[g + 1] = Complex::new(a[g + 1].re + tr.re, a[g + 1].im + tr.im);
                let t = a[g + 6];
                let tr = Complex::new(-t.im, t.re);
                a[g + 6] = Complex::new(a[g + 2].re - tr.re, a[g + 2].im - tr.im);
                a[g + 2] = Complex::new(a[g + 2].re + tr.re, a[g + 2].im + tr.im);
                let t = a[g + 7];
                let tr = Complex::new((-t.re - t.im) * c2, (-t.im + t.re) * c2);
                a[g + 7] = Complex::new(a[g + 3].re - tr.re, a[g + 3].im - tr.im);
                a[g + 3] = Complex::new(a[g + 3].re + tr.re, a[g + 3].im + tr.im);
            }

            // Stage 4: W16^k conjugates
            {
                let t = a[8];
                a[8] = Complex::new(a[0].re - t.re, a[0].im - t.im);
                a[0] = Complex::new(a[0].re + t.re, a[0].im + t.im);
                let t = a[9];
                let tr = Complex::new(t.re * c1 - t.im * s1, t.im * c1 + t.re * s1);
                a[9] = Complex::new(a[1].re - tr.re, a[1].im - tr.im);
                a[1] = Complex::new(a[1].re + tr.re, a[1].im + tr.im);
                let t = a[10];
                let tr = Complex::new((t.re - t.im) * c2, (t.im + t.re) * c2);
                a[10] = Complex::new(a[2].re - tr.re, a[2].im - tr.im);
                a[2] = Complex::new(a[2].re + tr.re, a[2].im + tr.im);
                let t = a[11];
                let tr = Complex::new(t.re * s1 - t.im * c1, t.im * s1 + t.re * c1);
                a[11] = Complex::new(a[3].re - tr.re, a[3].im - tr.im);
                a[3] = Complex::new(a[3].re + tr.re, a[3].im + tr.im);
                let t = a[12];
                let tr = Complex::new(-t.im, t.re);
                a[12] = Complex::new(a[4].re - tr.re, a[4].im - tr.im);
                a[4] = Complex::new(a[4].re + tr.re, a[4].im + tr.im);
                let t = a[13];
                let tr = Complex::new(-t.re * s1 - t.im * c1, -t.im * s1 + t.re * c1);
                a[13] = Complex::new(a[5].re - tr.re, a[5].im - tr.im);
                a[5] = Complex::new(a[5].re + tr.re, a[5].im + tr.im);
                let t = a[14];
                let tr = Complex::new((-t.re - t.im) * c2, (-t.im + t.re) * c2);
                a[14] = Complex::new(a[6].re - tr.re, a[6].im - tr.im);
                a[6] = Complex::new(a[6].re + tr.re, a[6].im + tr.im);
                let t = a[15];
                let tr = Complex::new(-t.re * c1 - t.im * s1, -t.im * c1 + t.re * s1);
                a[15] = Complex::new(a[7].re - tr.re, a[7].im - tr.im);
                a[7] = Complex::new(a[7].re + tr.re, a[7].im + tr.im);
            }

            for j in 0..16 {
                io[base + j * stride] = a[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Complex;
    use std::f64::consts::PI;

    fn naive_dft(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let n = input.len();
        let mut out = vec![Complex::<f64>::zero(); n];
        for k in 0..n {
            for j in 0..n {
                let angle = -2.0 * PI * (j * k) as f64 / n as f64;
                let w = Complex::new(angle.cos(), angle.sin());
                let prod = input[j] * w;
                out[k] = Complex::new(out[k].re + prod.re, out[k].im + prod.im);
            }
        }
        out
    }

    fn complex_near(a: Complex<f64>, b: Complex<f64>, tol: f64) -> bool {
        (a.re - b.re).abs() < tol && (a.im - b.im).abs() < tol
    }

    fn make_unit_twiddles(n: usize) -> Vec<Complex<f64>> {
        // All twiddles = 1 → butterfly acts as a plain DFT with no rotations
        vec![Complex::new(1.0, 0.0); n]
    }

    /// Test radix-3 butterfly against naive DFT for a single block (no real twiddles needed).
    #[test]
    fn tw3_single_block_unit_twiddles() {
        // For a single block with unit twiddles, tw3_dit_fwd computes a DFT-3.
        let input: Vec<Complex<f64>> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.5),
            Complex::new(-1.0, 1.0),
        ];
        let expected = naive_dft(&input);

        let mut io = input;
        let twiddles = make_unit_twiddles(2); // 2 twiddles for r=3: W^1 and W^2 for stride 1
        tw3_dit_fwd(&mut io, &twiddles, 1, 1);

        for i in 0..3 {
            assert!(
                complex_near(io[i], expected[i], 1e-10),
                "tw3 mismatch at index {i}: got {:?}, expected {:?}",
                io[i],
                expected[i]
            );
        }
    }

    /// Test radix-5 butterfly against naive DFT for a single block.
    #[test]
    fn tw5_single_block_unit_twiddles() {
        let input: Vec<Complex<f64>> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.5, -0.5),
            Complex::new(-1.0, 0.3),
            Complex::new(0.2, 1.0),
            Complex::new(-0.3, -0.7),
        ];
        let expected = naive_dft(&input);

        let mut io = input;
        let twiddles = make_unit_twiddles(4); // 4 twiddles for r=5
        tw5_dit_fwd(&mut io, &twiddles, 1, 1);

        for i in 0..5 {
            assert!(
                complex_near(io[i], expected[i], 1e-10),
                "tw5 mismatch at index {i}: got {:?}, expected {:?}",
                io[i],
                expected[i]
            );
        }
    }

    /// Test radix-7 butterfly against naive DFT for a single block.
    #[test]
    fn tw7_single_block_unit_twiddles() {
        let input: Vec<Complex<f64>> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.5, -0.5),
            Complex::new(-1.0, 0.3),
            Complex::new(0.2, 1.0),
            Complex::new(-0.3, -0.7),
            Complex::new(0.8, 0.2),
            Complex::new(-0.6, 0.4),
        ];
        let expected = naive_dft(&input);

        let mut io = input;
        let twiddles = make_unit_twiddles(6); // 6 twiddles for r=7
        tw7_dit_fwd(&mut io, &twiddles, 1, 1);

        for i in 0..7 {
            assert!(
                complex_near(io[i], expected[i], 1e-10),
                "tw7 mismatch at index {i}: got {:?}, expected {:?}",
                io[i],
                expected[i]
            );
        }
    }

    /// Test radix-4 butterfly matches naive DFT-4 (unit twiddles).
    #[test]
    fn tw4_single_block_unit_twiddles() {
        let input: Vec<Complex<f64>> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(-1.0, 0.0),
            Complex::new(0.0, -1.0),
        ];
        let expected = naive_dft(&input);

        let mut io = input;
        let twiddles = make_unit_twiddles(3); // 3 twiddles for r=4
        tw4_dit_fwd(&mut io, &twiddles, 1, 1);

        for i in 0..4 {
            assert!(
                complex_near(io[i], expected[i], 1e-10),
                "tw4 mismatch at index {i}: got {:?}, expected {:?}",
                io[i],
                expected[i]
            );
        }
    }

    /// Discriminating test: N=6 full DIT pipeline with real twiddles.
    ///
    /// This test exercises both blocks>1 AND stride>1 simultaneously, which
    /// distinguishes the correct `twiddles[(j-1)*stride + s]` indexing from
    /// the wrong `twiddles[(j-1)*blocks + b]` indexing.
    ///
    /// Factorization: N=6 = 3 × 2, DIT order (innermost radix first): [3, 2].
    ///
    /// Stage layout:
    ///   Stage 1 (radix-3): current_n=3, stride=1, blocks=2
    ///     → twiddles are all 1 (W_3^0 = 1 for j*s with s=0..0)
    ///   Stage 2 (radix-2): current_n=6, stride=3, blocks=1
    ///     → twiddles[s] = W_6^s for s=0,1,2
    ///
    /// The index ambiguity: Stage 2 has stride=3, blocks=1.
    ///   Correct: twiddles[s] varies with s=0,1,2 → gives W_6^0, W_6^1, W_6^2
    ///   Wrong:   twiddles[b] = twiddles[0] (b=0 only) → same twiddle for all columns
    #[test]
    fn n6_full_dit_pipeline_discriminating() {
        // Input: arbitrary complex values that will expose twiddle differences
        let input: Vec<Complex<f64>> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.5, -0.3),
            Complex::new(-0.7, 0.2),
            Complex::new(0.3, 0.8),
            Complex::new(-0.4, -0.5),
            Complex::new(0.6, 0.1),
        ];
        let expected = naive_dft(&input);

        // Step 1: Apply digit-reversal permutation for factors [3, 2]
        // For N=6 = 3×2, DIT processes radix-3 first (innermost).
        // The permutation groups samples by their "stride" position:
        //   Block 0 (b=0): input[0], input[2], input[4]  (every 2nd element starting at 0)
        //   Block 1 (b=1): input[1], input[3], input[5]  (every 2nd element starting at 1)
        // So: io = [input[0], input[2], input[4], input[1], input[3], input[5]]
        // Permutation array (perm[k] = source index for io[k]):
        //   perm: 0→0, 1→2, 2→4, 3→1, 4→3, 5→5
        let perm = [0usize, 2, 4, 1, 3, 5];
        let mut io: Vec<Complex<f64>> = perm.iter().map(|&i| input[i]).collect();

        // Step 2: Stage 1 — radix-3, stride=1, blocks=2
        // Twiddles: (r-1)*stride = 2*1 = 2 entries, all W_3^0 = 1
        let tw_stage1 = vec![Complex::new(1.0, 0.0); 2];
        tw3_dit_fwd(&mut io, &tw_stage1, 1, 2);

        // Step 3: Stage 2 — radix-2, stride=3, blocks=1
        // Twiddles: (r-1)*stride = 1*3 = 3 entries
        // twiddles[s] = W_6^(1*s) = exp(-2πi*s/6) for s=0,1,2
        let tw_stage2: Vec<Complex<f64>> = (0..3)
            .map(|s| {
                let angle = -2.0 * PI * s as f64 / 6.0;
                Complex::new(angle.cos(), angle.sin())
            })
            .collect();
        tw2_dit_fwd(&mut io, &tw_stage2, 3, 1);

        // Verify against naive DFT
        for i in 0..6 {
            assert!(
                complex_near(io[i], expected[i], 1e-10),
                "N=6 DIT mismatch at index {i}: got {:?}, expected {:?}",
                io[i],
                expected[i]
            );
        }
    }

    /// Test radix-3 with blocks=2 and stride=1 (multi-block, single column).
    ///
    /// With stride=1, the stride-based and blocks-based indexing differ only
    /// if blocks > 1. This test has blocks=2 so both blocks of twiddles must
    /// be the SAME (since stride=1 has only one column index s=0).
    /// This validates multi-block dispatch correctness.
    #[test]
    fn tw3_two_blocks_unit_twiddles() {
        // Two independent DFT-3s: block 0 and block 1
        let input0 = [
            Complex::new(1.0, 0.0),
            Complex::new(0.5, 0.5),
            Complex::new(-0.5, 0.3),
        ];
        let input1 = [
            Complex::new(0.2, -0.1),
            Complex::new(-0.3, 0.7),
            Complex::new(0.8, -0.4),
        ];
        let expected0 = naive_dft(&input0);
        let expected1 = naive_dft(&input1);

        // Interleave: [b0s0, b0s1_unused? No — stride=1 so it's [b0_r0, b0_r1, b0_r2, b1_r0, b1_r1, b1_r2]]
        // With stride=1, blocks=2: io[b*3*1 + j*1 + 0] = io[b*3 + j]
        let mut io = vec![
            input0[0], input0[1], input0[2], input1[0], input1[1], input1[2],
        ];

        // twiddles: (r-1)*stride = 2*1 = 2 twiddles, all unit (W_3^0=1)
        let twiddles = make_unit_twiddles(2);
        tw3_dit_fwd(&mut io, &twiddles, 1, 2);

        for i in 0..3 {
            assert!(
                complex_near(io[i], expected0[i], 1e-10),
                "block0 mismatch at {i}: got {:?}, expected {:?}",
                io[i],
                expected0[i]
            );
        }
        for i in 0..3 {
            assert!(
                complex_near(io[3 + i], expected1[i], 1e-10),
                "block1 mismatch at {i}: got {:?}, expected {:?}",
                io[3 + i],
                expected1[i]
            );
        }
    }
}
