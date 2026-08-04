//! WebAssembly SIMD (`simd128`) codelets built on [`WasmSimdF64`] / [`WasmSimdF32`].
//!
//! These are the dispatch targets that make the WASM SIMD backend *reachable*:
//! before they existed, `WasmSimdF64`/`WasmSimdF32` were fully implemented and
//! publicly re-exported but no transform path ever constructed one, so a
//! `simd128` build performed exactly the same scalar work as a plain one.
//! [`notw_2_dispatch`](super::notw_2_dispatch),
//! [`notw_4_dispatch`](super::notw_4_dispatch) and
//! [`notw_8_dispatch`](super::notw_8_dispatch) now route into this module when
//! compiled for `wasm32` with `target_feature = "simd128"`.
//!
//! # Why this module compiles everywhere
//!
//! Every codelet here is written against the [`SimdVector`] / [`SimdComplex`]
//! traits plus the inherent `new` / `extract` / `negate` / `swap` helpers, all of
//! which `crate::wasm::simd` provides in *both* of its branches: the `v128`
//! implementation under `target_feature = "simd128"`, and the plain-array scalar
//! stand-in otherwise. So the same source is exercised on the development host
//! (against the scalar stand-in, by the tests at the bottom of this file) and on
//! `wasm32 + simd128` (against real `v128` instructions). Host tests therefore
//! validate the *butterfly algebra*; what they cannot validate is the `v128` lane
//! ordering inside the trait impls themselves, which needs a WASM runtime.
//!
//! # Layout conventions
//!
//! * [`WasmSimdF64`] holds **one** `Complex<f64>` as `[re, im]`.
//! * [`WasmSimdF32`] holds **two** `Complex<f32>` as `[re0, im0, re1, im1]`.
//!
//! [`WasmSimdF64`]: crate::wasm::WasmSimdF64
//! [`WasmSimdF32`]: crate::wasm::WasmSimdF32
//! [`SimdVector`]: crate::simd::SimdVector
//! [`SimdComplex`]: crate::simd::SimdComplex

use crate::kernel::Complex;
use crate::simd::{SimdComplex, SimdVector};
use crate::wasm::{WasmSimdF32, WasmSimdF64};

/// `1 / sqrt(2)`, the magnitude of the real and imaginary parts of `W_8^1`.
const FRAC_1_SQRT_2_F64: f64 = core::f64::consts::FRAC_1_SQRT_2;
/// `1 / sqrt(2)` in single precision.
const FRAC_1_SQRT_2_F32: f32 = core::f32::consts::FRAC_1_SQRT_2;

// ===========================================================================
// f64 (one complex per vector)
// ===========================================================================

/// Load `x[i]` into a vector.
#[inline]
fn ld64(x: &[Complex<f64>], i: usize) -> WasmSimdF64 {
    WasmSimdF64::new(x[i].re, x[i].im)
}

/// Store a vector back into `x[i]`.
#[inline]
fn st64(x: &mut [Complex<f64>], i: usize, v: WasmSimdF64) {
    x[i] = Complex {
        re: v.extract(0),
        im: v.extract(1),
    };
}

/// Multiply by `-i` (forward, `sign < 0`) or `+i` (inverse), i.e. by `W_4^1`.
///
/// `swap()` yields `[im, re]`; scaling by `[1, -1]` gives `[im, -re] = -i * z`
/// and by `[-1, 1]` gives `[-im, re] = +i * z`.
#[inline]
fn rot_quarter64(z: WasmSimdF64, sign: i32) -> WasmSimdF64 {
    let scale = if sign < 0 {
        WasmSimdF64::new(1.0, -1.0)
    } else {
        WasmSimdF64::new(-1.0, 1.0)
    };
    z.swap().mul(scale)
}

/// `W_8^1` for the requested sign: `exp(sign * 2*pi*i / 8)`.
#[inline]
fn w8_1_f64(sign: i32) -> WasmSimdF64 {
    let s = if sign < 0 {
        -FRAC_1_SQRT_2_F64
    } else {
        FRAC_1_SQRT_2_F64
    };
    WasmSimdF64::new(FRAC_1_SQRT_2_F64, s)
}

/// `W_8^3` for the requested sign: `exp(sign * 3 * 2*pi*i / 8)`.
#[inline]
fn w8_3_f64(sign: i32) -> WasmSimdF64 {
    let s = if sign < 0 {
        -FRAC_1_SQRT_2_F64
    } else {
        FRAC_1_SQRT_2_F64
    };
    WasmSimdF64::new(-FRAC_1_SQRT_2_F64, s)
}

/// Size-2 DFT for `f64` on the WASM SIMD backend: `[x0 + x1, x0 - x1]`.
///
/// # Panics
/// Panics unless `x.len() >= 2`.
#[inline]
pub fn notw_2_wasm_f64(x: &mut [Complex<f64>]) {
    assert!(x.len() >= 2, "notw_2_wasm_f64 requires at least 2 elements");
    let a = ld64(x, 0);
    let b = ld64(x, 1);
    st64(x, 0, a.add(b));
    st64(x, 1, a.sub(b));
}

/// Size-4 DFT for `f64` on the WASM SIMD backend (radix-2 decimation-in-time,
/// natural input and output order).
///
/// `sign < 0` selects the forward transform, `sign >= 0` the unnormalized
/// inverse, matching the other `notw_4_*` codelets.
///
/// # Panics
/// Panics unless `x.len() >= 4`.
#[inline]
pub fn notw_4_wasm_f64(x: &mut [Complex<f64>], sign: i32) {
    assert!(x.len() >= 4, "notw_4_wasm_f64 requires at least 4 elements");
    let x0 = ld64(x, 0);
    let x1 = ld64(x, 1);
    let x2 = ld64(x, 2);
    let x3 = ld64(x, 3);

    let t0 = x0.add(x2);
    let t1 = x0.sub(x2);
    let t2 = x1.add(x3);
    let t3 = rot_quarter64(x1.sub(x3), sign);

    st64(x, 0, t0.add(t2));
    st64(x, 1, t1.add(t3));
    st64(x, 2, t0.sub(t2));
    st64(x, 3, t1.sub(t3));
}

/// Size-8 DFT for `f64` on the WASM SIMD backend (radix-2 decimation-in-time
/// over the bit-reversed input, natural output order).
///
/// # Panics
/// Panics unless `x.len() >= 8`.
#[inline]
pub fn notw_8_wasm_f64(x: &mut [Complex<f64>], sign: i32) {
    assert!(x.len() >= 8, "notw_8_wasm_f64 requires at least 8 elements");

    // Bit-reversed load: [0, 4, 2, 6, 1, 5, 3, 7].
    let y0 = ld64(x, 0);
    let y1 = ld64(x, 4);
    let y2 = ld64(x, 2);
    let y3 = ld64(x, 6);
    let y4 = ld64(x, 1);
    let y5 = ld64(x, 5);
    let y6 = ld64(x, 3);
    let y7 = ld64(x, 7);

    // Stage 1 (m = 2): twiddle-free pairs.
    let z0 = y0.add(y1);
    let z1 = y0.sub(y1);
    let z2 = y2.add(y3);
    let z3 = y2.sub(y3);
    let z4 = y4.add(y5);
    let z5 = y4.sub(y5);
    let z6 = y6.add(y7);
    let z7 = y6.sub(y7);

    // Stage 2 (m = 4): twiddles are 1 and W_4^1 = -/+ i.
    let r3 = rot_quarter64(z3, sign);
    let r7 = rot_quarter64(z7, sign);
    let u0 = z0.add(z2);
    let u1 = z1.add(r3);
    let u2 = z0.sub(z2);
    let u3 = z1.sub(r3);
    let u4 = z4.add(z6);
    let u5 = z5.add(r7);
    let u6 = z4.sub(z6);
    let u7 = z5.sub(r7);

    // Stage 3 (m = 8): twiddles W_8^0..W_8^3.
    let t4 = u4;
    let t5 = u5.cmul(w8_1_f64(sign));
    let t6 = rot_quarter64(u6, sign);
    let t7 = u7.cmul(w8_3_f64(sign));

    st64(x, 0, u0.add(t4));
    st64(x, 1, u1.add(t5));
    st64(x, 2, u2.add(t6));
    st64(x, 3, u3.add(t7));
    st64(x, 4, u0.sub(t4));
    st64(x, 5, u1.sub(t5));
    st64(x, 6, u2.sub(t6));
    st64(x, 7, u3.sub(t7));
}

// ===========================================================================
// f32 (two complex per vector)
// ===========================================================================

/// Pack two complex values into one `f32x4` vector.
#[inline]
fn pack32(a: Complex<f32>, b: Complex<f32>) -> WasmSimdF32 {
    WasmSimdF32::new(a.re, a.im, b.re, b.im)
}

/// Unpack an `f32x4` vector into its two complex values.
#[inline]
fn unpack32(v: WasmSimdF32) -> (Complex<f32>, Complex<f32>) {
    (
        Complex {
            re: v.extract(0),
            im: v.extract(1),
        },
        Complex {
            re: v.extract(2),
            im: v.extract(3),
        },
    )
}

/// Multiply a single `Complex<f32>` by `-i` (forward) or `+i` (inverse).
#[inline]
fn rot_quarter32(z: Complex<f32>, sign: i32) -> Complex<f32> {
    if sign < 0 {
        Complex {
            re: z.im,
            im: -z.re,
        }
    } else {
        Complex {
            re: -z.im,
            im: z.re,
        }
    }
}

/// Size-2 DFT for `f32` on the WASM SIMD backend.
///
/// # Panics
/// Panics unless `x.len() >= 2`.
#[inline]
pub fn notw_2_wasm_f32(x: &mut [Complex<f32>]) {
    assert!(x.len() >= 2, "notw_2_wasm_f32 requires at least 2 elements");
    // Broadcast each operand into both complex lanes so one add and one sub
    // produce both outputs; lane 0 of the sum is X0, lane 1 of the difference X1.
    let a = pack32(x[0], x[0]);
    let b = pack32(x[1], x[1]);
    let (sum, _) = unpack32(a.add(b));
    let (_, diff) = unpack32(a.sub(b));
    x[0] = sum;
    x[1] = diff;
}

/// Size-4 DFT for `f32` on the WASM SIMD backend (radix-2 decimation-in-time,
/// natural input and output order).
///
/// # Panics
/// Panics unless `x.len() >= 4`.
#[inline]
pub fn notw_4_wasm_f32(x: &mut [Complex<f32>], sign: i32) {
    assert!(x.len() >= 4, "notw_4_wasm_f32 requires at least 4 elements");
    let lo = pack32(x[0], x[1]);
    let hi = pack32(x[2], x[3]);

    // [t0, t2] = [x0 + x2, x1 + x3];  [t1, t3] = [x0 - x2, x1 - x3]
    let (t0, t2) = unpack32(lo.add(hi));
    let (t1, t3) = unpack32(lo.sub(hi));
    let t3 = rot_quarter32(t3, sign);

    let a = pack32(t0, t1);
    let b = pack32(t2, t3);
    let (y0, y1) = unpack32(a.add(b));
    let (y2, y3) = unpack32(a.sub(b));

    x[0] = y0;
    x[1] = y1;
    x[2] = y2;
    x[3] = y3;
}

/// Size-8 DFT for `f32` on the WASM SIMD backend (radix-2 decimation-in-time
/// over the bit-reversed input, natural output order).
///
/// Each stage pairs the butterflies so both halves of an `f32x4` do useful work,
/// and the final stage applies `W_8^0..W_8^3` with two [`SimdComplex::cmul`]
/// calls instead of four scalar complex multiplies.
///
/// # Panics
/// Panics unless `x.len() >= 8`.
#[inline]
pub fn notw_8_wasm_f32(x: &mut [Complex<f32>], sign: i32) {
    assert!(x.len() >= 8, "notw_8_wasm_f32 requires at least 8 elements");

    // Bit-reversed order: [0, 4, 2, 6, 1, 5, 3, 7].
    // Stage 1 pairs (y0,y1), (y2,y3), (y4,y5), (y6,y7); packing the two members
    // of different pairs into one vector keeps both lanes busy.
    let a = pack32(x[0], x[2]);
    let b = pack32(x[4], x[6]);
    let (z0, z2) = unpack32(a.add(b));
    let (z1, z3) = unpack32(a.sub(b));

    let c = pack32(x[1], x[3]);
    let d = pack32(x[5], x[7]);
    let (z4, z6) = unpack32(c.add(d));
    let (z5, z7) = unpack32(c.sub(d));

    // Stage 2 (m = 4): twiddles 1 and W_4^1.
    let e = pack32(z0, z1);
    let f = pack32(z2, rot_quarter32(z3, sign));
    let (u0, u1) = unpack32(e.add(f));
    let (u2, u3) = unpack32(e.sub(f));

    let g = pack32(z4, z5);
    let h = pack32(z6, rot_quarter32(z7, sign));
    let (u4, u5) = unpack32(g.add(h));
    let (u6, u7) = unpack32(g.sub(h));

    // Stage 3 (m = 8): twiddles W_8^0..W_8^3, two per `cmul`.
    let s = if sign < 0 {
        -FRAC_1_SQRT_2_F32
    } else {
        FRAC_1_SQRT_2_F32
    };
    let w01 = WasmSimdF32::new(1.0, 0.0, FRAC_1_SQRT_2_F32, s);
    let w23 = WasmSimdF32::new(
        0.0,
        if sign < 0 { -1.0 } else { 1.0 },
        -FRAC_1_SQRT_2_F32,
        s,
    );

    let p = pack32(u0, u1);
    let q = pack32(u4, u5).cmul(w01);
    let (y0, y1) = unpack32(p.add(q));
    let (y4, y5) = unpack32(p.sub(q));

    let r = pack32(u2, u3);
    let t = pack32(u6, u7).cmul(w23);
    let (y2, y3) = unpack32(r.add(t));
    let (y6, y7) = unpack32(r.sub(t));

    x[0] = y0;
    x[1] = y1;
    x[2] = y2;
    x[3] = y3;
    x[4] = y4;
    x[5] = y5;
    x[6] = y6;
    x[7] = y7;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft_f64(x: &[Complex<f64>], sign: i32) -> Vec<Complex<f64>> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let mut acc = Complex { re: 0.0, im: 0.0 };
                for (j, xj) in x.iter().enumerate() {
                    let angle =
                        f64::from(sign) * core::f64::consts::TAU * (j * k) as f64 / n as f64;
                    let (s, c) = angle.sin_cos();
                    acc.re += xj.re * c - xj.im * s;
                    acc.im += xj.re * s + xj.im * c;
                }
                acc
            })
            .collect()
    }

    fn sample_f64(n: usize) -> Vec<Complex<f64>> {
        (0..n)
            .map(|i| {
                let t = i as f64;
                Complex {
                    re: (0.9 * t).sin() + 0.4 * t,
                    im: (0.31 * t + 1.0).cos() - 0.17 * t,
                }
            })
            .collect()
    }

    fn to_f32(x: &[Complex<f64>]) -> Vec<Complex<f32>> {
        x.iter()
            .map(|c| Complex {
                re: c.re as f32,
                im: c.im as f32,
            })
            .collect()
    }

    #[test]
    fn wasm_notw_2_f64_matches_naive() {
        let input = sample_f64(2);
        let expected = naive_dft_f64(&input, -1);
        let mut got = input;
        notw_2_wasm_f64(&mut got);
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(
                (g.re - e.re).abs() < 1e-12 && (g.im - e.im).abs() < 1e-12,
                "{g:?} vs {e:?}"
            );
        }
    }

    #[test]
    fn wasm_notw_4_f64_matches_naive_both_signs() {
        for sign in [-1_i32, 1] {
            let input = sample_f64(4);
            let expected = naive_dft_f64(&input, sign);
            let mut got = input;
            notw_4_wasm_f64(&mut got, sign);
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (g.re - e.re).abs() < 1e-12 && (g.im - e.im).abs() < 1e-12,
                    "sign {sign} index {i}: {g:?} vs {e:?}"
                );
            }
        }
    }

    #[test]
    fn wasm_notw_8_f64_matches_naive_both_signs() {
        for sign in [-1_i32, 1] {
            let input = sample_f64(8);
            let expected = naive_dft_f64(&input, sign);
            let mut got = input;
            notw_8_wasm_f64(&mut got, sign);
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (g.re - e.re).abs() < 1e-11 && (g.im - e.im).abs() < 1e-11,
                    "sign {sign} index {i}: {g:?} vs {e:?}"
                );
            }
        }
    }

    #[test]
    fn wasm_notw_2_f32_matches_naive() {
        let input64 = sample_f64(2);
        let expected = naive_dft_f64(&input64, -1);
        let mut got = to_f32(&input64);
        notw_2_wasm_f32(&mut got);
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(
                (f64::from(g.re) - e.re).abs() < 1e-4 && (f64::from(g.im) - e.im).abs() < 1e-4,
                "{g:?} vs {e:?}"
            );
        }
    }

    #[test]
    fn wasm_notw_4_f32_matches_naive_both_signs() {
        for sign in [-1_i32, 1] {
            let input64 = sample_f64(4);
            let expected = naive_dft_f64(&input64, sign);
            let mut got = to_f32(&input64);
            notw_4_wasm_f32(&mut got, sign);
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (f64::from(g.re) - e.re).abs() < 1e-4 && (f64::from(g.im) - e.im).abs() < 1e-4,
                    "sign {sign} index {i}: {g:?} vs {e:?}"
                );
            }
        }
    }

    #[test]
    fn wasm_notw_8_f32_matches_naive_both_signs() {
        for sign in [-1_i32, 1] {
            let input64 = sample_f64(8);
            let expected = naive_dft_f64(&input64, sign);
            let mut got = to_f32(&input64);
            notw_8_wasm_f32(&mut got, sign);
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (f64::from(g.re) - e.re).abs() < 1e-3 && (f64::from(g.im) - e.im).abs() < 1e-3,
                    "sign {sign} index {i}: {g:?} vs {e:?}"
                );
            }
        }
    }
}
