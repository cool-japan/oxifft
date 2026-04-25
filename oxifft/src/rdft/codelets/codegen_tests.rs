//! Numerical equivalence tests: generated RDFT codelets vs hand-written.
//!
//! Each test verifies that `r2hc_N_gen` (from `gen_rdft_codelet!`) produces
//! results numerically identical (within floating-point tolerance) to the
//! hand-written `r2hc_N` in this module, and likewise for `hc2r_N_gen` vs `hc2r_N`.

use super::{hc2r_2, hc2r_4, hc2r_8, r2hc_2, r2hc_4, r2hc_8};
use crate::kernel::Complex;
use oxifft_codegen::gen_rdft_codelet;

// ── Generate the codelets under test ─────────────────────────────────────────
gen_rdft_codelet!(size = 2, kind = R2hc);
gen_rdft_codelet!(size = 4, kind = R2hc);
gen_rdft_codelet!(size = 8, kind = R2hc);
gen_rdft_codelet!(size = 2, kind = Hc2r);
gen_rdft_codelet!(size = 4, kind = Hc2r);
gen_rdft_codelet!(size = 8, kind = Hc2r);

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Small deterministic LCG for reproducible test inputs.
fn lcg_f64(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let mantissa_bits = *state >> 11;
    let scale = 1.0_f64 / (1u64 << 52) as f64;
    mantissa_bits as f64 * scale - 1.0
}

fn lcg_f32(state: &mut u64) -> f32 {
    lcg_f64(state) as f32
}

fn approx_eq_f64(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn approx_eq_f32(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

// ── R2HC size 2 ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r2hc_2_gen_matches_handwritten_f64() {
        let mut state = 0x1234_5678_ABCD_EF01_u64;
        for _ in 0..100 {
            let input: Vec<f64> = (0..2).map(|_| lcg_f64(&mut state)).collect();
            let mut out_hw = vec![Complex::<f64>::zero(); 2];
            let mut out_gen = vec![Complex::<f64>::zero(); 2];
            r2hc_2(&input, &mut out_hw);
            r2hc_2_gen(&input, &mut out_gen);
            for (k, (h, g)) in out_hw.iter().zip(out_gen.iter()).enumerate() {
                assert!(
                    approx_eq_f64(h.re, g.re, 1e-12) && approx_eq_f64(h.im, g.im, 1e-12),
                    "r2hc_2[{k}]: hw={h:?}, gen={g:?}"
                );
            }
        }
    }

    #[test]
    fn r2hc_2_gen_matches_handwritten_f32() {
        let mut state = 0xDEAD_CAFE_BEEF_0002_u64;
        for _ in 0..100 {
            let input: Vec<f32> = (0..2).map(|_| lcg_f32(&mut state)).collect();
            let mut out_hw = vec![Complex::<f32>::zero(); 2];
            let mut out_gen = vec![Complex::<f32>::zero(); 2];
            r2hc_2(&input, &mut out_hw);
            r2hc_2_gen(&input, &mut out_gen);
            for (k, (h, g)) in out_hw.iter().zip(out_gen.iter()).enumerate() {
                assert!(
                    approx_eq_f32(h.re, g.re, 1e-5) && approx_eq_f32(h.im, g.im, 1e-5),
                    "r2hc_2_f32[{k}]: hw={h:?}, gen={g:?}"
                );
            }
        }
    }

    // ── R2HC size 4 ──────────────────────────────────────────────────────────

    #[test]
    fn r2hc_4_gen_matches_handwritten_f64() {
        let mut state = 0x1234_5678_ABCD_EF04_u64;
        for _ in 0..100 {
            let input: Vec<f64> = (0..4).map(|_| lcg_f64(&mut state)).collect();
            let mut out_hw = vec![Complex::<f64>::zero(); 3];
            let mut out_gen = vec![Complex::<f64>::zero(); 3];
            r2hc_4(&input, &mut out_hw);
            r2hc_4_gen(&input, &mut out_gen);
            for (k, (h, g)) in out_hw.iter().zip(out_gen.iter()).enumerate() {
                assert!(
                    approx_eq_f64(h.re, g.re, 1e-12) && approx_eq_f64(h.im, g.im, 1e-12),
                    "r2hc_4[{k}]: hw={h:?}, gen={g:?}"
                );
            }
        }
    }

    #[test]
    fn r2hc_4_gen_matches_handwritten_f32() {
        let mut state = 0xDEAD_CAFE_BEEF_0004_u64;
        for _ in 0..100 {
            let input: Vec<f32> = (0..4).map(|_| lcg_f32(&mut state)).collect();
            let mut out_hw = vec![Complex::<f32>::zero(); 3];
            let mut out_gen = vec![Complex::<f32>::zero(); 3];
            r2hc_4(&input, &mut out_hw);
            r2hc_4_gen(&input, &mut out_gen);
            for (k, (h, g)) in out_hw.iter().zip(out_gen.iter()).enumerate() {
                assert!(
                    approx_eq_f32(h.re, g.re, 1e-5) && approx_eq_f32(h.im, g.im, 1e-5),
                    "r2hc_4_f32[{k}]: hw={h:?}, gen={g:?}"
                );
            }
        }
    }

    // ── R2HC size 8 ──────────────────────────────────────────────────────────

    #[test]
    fn r2hc_8_gen_matches_handwritten_f64() {
        let mut state = 0x1234_5678_ABCD_EF08_u64;
        for _ in 0..100 {
            let input: Vec<f64> = (0..8).map(|_| lcg_f64(&mut state)).collect();
            let mut out_hw = vec![Complex::<f64>::zero(); 5];
            let mut out_gen = vec![Complex::<f64>::zero(); 5];
            r2hc_8(&input, &mut out_hw);
            r2hc_8_gen(&input, &mut out_gen);
            for (k, (h, g)) in out_hw.iter().zip(out_gen.iter()).enumerate() {
                assert!(
                    approx_eq_f64(h.re, g.re, 1e-11) && approx_eq_f64(h.im, g.im, 1e-11),
                    "r2hc_8[{k}]: hw={h:?}, gen={g:?}"
                );
            }
        }
    }

    #[test]
    fn r2hc_8_gen_matches_handwritten_f32() {
        let mut state = 0xDEAD_CAFE_BEEF_0008_u64;
        for _ in 0..100 {
            let input: Vec<f32> = (0..8).map(|_| lcg_f32(&mut state)).collect();
            let mut out_hw = vec![Complex::<f32>::zero(); 5];
            let mut out_gen = vec![Complex::<f32>::zero(); 5];
            r2hc_8(&input, &mut out_hw);
            r2hc_8_gen(&input, &mut out_gen);
            for (k, (h, g)) in out_hw.iter().zip(out_gen.iter()).enumerate() {
                assert!(
                    approx_eq_f32(h.re, g.re, 1e-5) && approx_eq_f32(h.im, g.im, 1e-5),
                    "r2hc_8_f32[{k}]: hw={h:?}, gen={g:?}"
                );
            }
        }
    }

    // ── HC2R size 2 ──────────────────────────────────────────────────────────

    #[test]
    fn hc2r_2_gen_matches_handwritten_f64() {
        let mut state = 0xABCD_1234_CAFE_0002_u64;
        for _ in 0..100 {
            // Generate a valid half-complex spectrum from a random real input
            let real_in: Vec<f64> = (0..2).map(|_| lcg_f64(&mut state)).collect();
            let mut spectrum = vec![Complex::<f64>::zero(); 2];
            r2hc_2(&real_in, &mut spectrum);

            let mut out_hw = vec![0.0_f64; 2];
            let mut out_gen = vec![0.0_f64; 2];
            hc2r_2(&spectrum, &mut out_hw);
            hc2r_2_gen(&spectrum, &mut out_gen);
            for (j, (h, g)) in out_hw.iter().zip(out_gen.iter()).enumerate() {
                assert!(approx_eq_f64(*h, *g, 1e-12), "hc2r_2[{j}]: hw={h}, gen={g}");
            }
        }
    }

    #[test]
    fn hc2r_2_gen_matches_handwritten_f32() {
        let mut state = 0xABCD_CAFE_0002_ABCD_u64;
        for _ in 0..100 {
            let real_in: Vec<f32> = (0..2).map(|_| lcg_f32(&mut state)).collect();
            let mut spectrum = vec![Complex::<f32>::zero(); 2];
            r2hc_2(&real_in, &mut spectrum);

            let mut out_hw = vec![0.0_f32; 2];
            let mut out_gen = vec![0.0_f32; 2];
            hc2r_2(&spectrum, &mut out_hw);
            hc2r_2_gen(&spectrum, &mut out_gen);
            for (j, (h, g)) in out_hw.iter().zip(out_gen.iter()).enumerate() {
                assert!(
                    approx_eq_f32(*h, *g, 1e-5),
                    "hc2r_2_f32[{j}]: hw={h}, gen={g}"
                );
            }
        }
    }

    // ── HC2R size 4 ──────────────────────────────────────────────────────────

    #[test]
    fn hc2r_4_gen_matches_handwritten_f64() {
        let mut state = 0xABCD_1234_CAFE_0004_u64;
        for _ in 0..100 {
            let real_in: Vec<f64> = (0..4).map(|_| lcg_f64(&mut state)).collect();
            let mut spectrum = vec![Complex::<f64>::zero(); 3];
            r2hc_4(&real_in, &mut spectrum);

            let mut out_hw = vec![0.0_f64; 4];
            let mut out_gen = vec![0.0_f64; 4];
            hc2r_4(&spectrum, &mut out_hw);
            hc2r_4_gen(&spectrum, &mut out_gen);
            for (j, (h, g)) in out_hw.iter().zip(out_gen.iter()).enumerate() {
                assert!(approx_eq_f64(*h, *g, 1e-12), "hc2r_4[{j}]: hw={h}, gen={g}");
            }
        }
    }

    #[test]
    fn hc2r_4_gen_matches_handwritten_f32() {
        let mut state = 0xABCD_CAFE_0004_ABCD_u64;
        for _ in 0..100 {
            let real_in: Vec<f32> = (0..4).map(|_| lcg_f32(&mut state)).collect();
            let mut spectrum = vec![Complex::<f32>::zero(); 3];
            r2hc_4(&real_in, &mut spectrum);

            let mut out_hw = vec![0.0_f32; 4];
            let mut out_gen = vec![0.0_f32; 4];
            hc2r_4(&spectrum, &mut out_hw);
            hc2r_4_gen(&spectrum, &mut out_gen);
            for (j, (h, g)) in out_hw.iter().zip(out_gen.iter()).enumerate() {
                assert!(
                    approx_eq_f32(*h, *g, 1e-4),
                    "hc2r_4_f32[{j}]: hw={h}, gen={g}"
                );
            }
        }
    }

    // ── HC2R size 8 ──────────────────────────────────────────────────────────

    #[test]
    fn hc2r_8_gen_matches_handwritten_f64() {
        let mut state = 0xABCD_1234_CAFE_0008_u64;
        for _ in 0..100 {
            let real_in: Vec<f64> = (0..8).map(|_| lcg_f64(&mut state)).collect();
            let mut spectrum = vec![Complex::<f64>::zero(); 5];
            r2hc_8(&real_in, &mut spectrum);

            let mut out_hw = vec![0.0_f64; 8];
            let mut out_gen = vec![0.0_f64; 8];
            hc2r_8(&spectrum, &mut out_hw);
            hc2r_8_gen(&spectrum, &mut out_gen);
            for (j, (h, g)) in out_hw.iter().zip(out_gen.iter()).enumerate() {
                assert!(approx_eq_f64(*h, *g, 1e-11), "hc2r_8[{j}]: hw={h}, gen={g}");
            }
        }
    }

    #[test]
    fn hc2r_8_gen_matches_handwritten_f32() {
        let mut state = 0xABCD_CAFE_0008_ABCD_u64;
        for _ in 0..100 {
            let real_in: Vec<f32> = (0..8).map(|_| lcg_f32(&mut state)).collect();
            let mut spectrum = vec![Complex::<f32>::zero(); 5];
            r2hc_8(&real_in, &mut spectrum);

            let mut out_hw = vec![0.0_f32; 8];
            let mut out_gen = vec![0.0_f32; 8];
            hc2r_8(&spectrum, &mut out_hw);
            hc2r_8_gen(&spectrum, &mut out_gen);
            for (j, (h, g)) in out_hw.iter().zip(out_gen.iter()).enumerate() {
                assert!(
                    approx_eq_f32(*h, *g, 1e-4),
                    "hc2r_8_f32[{j}]: hw={h}, gen={g}"
                );
            }
        }
    }
}
