//! Chirp sequence computation helpers for the Chirp-Z Transform.
//!
//! These functions build the three pre-computed tables used by `CztPlan`:
//! - `chirp_y`: the input modulation sequence `A^{-n} · W^{n²/2}`
//! - `chirp_filter`: the circular-convolution filter `W^{-m²/2}` in length-L order
//! - `chirp_post`: the output correction sequence `W^{k²/2}`
//!
//! All arithmetic uses complex polar exponentials computed from the float trait
//! methods, avoiding any integer overflow from squaring large indices.

use crate::kernel::{Complex, Float};

/// Compute `A^{-n} · W^{n²/2}` for `n = 0 .. N`.
///
/// For on-unit-circle (|A|=1, |W|=1) inputs `Complex::cis` is used for
/// maximum precision.  For off-unit-circle inputs we compute the polar form
/// via `r^n · exp(i·n·θ)` and combine the two terms analytically.
///
/// # Returns
///
/// A `Vec` of length `n` where entry `i` equals `A^{-i} · W^{i²/2}`.
pub fn build_chirp_y<T: Float>(n: usize, a: Complex<T>, w: Complex<T>) -> Vec<Complex<T>> {
    let mut out = Vec::with_capacity(n);
    for idx in 0..n {
        let i = T::from_usize(idx);
        // A^{-n}: polar form r_a^{-n} * exp(-i * n * arg_a)
        let a_term = complex_pow_real(a, -i);
        // W^{n^2/2}: polar form r_w^{n^2/2} * exp(i * (n^2/2) * arg_w)
        let w_exp = i * i / T::TWO;
        let w_term = complex_pow_real(w, w_exp);
        out.push(a_term * w_term);
    }
    out
}

/// Build the length-`L` circular-convolution filter `H_circ` for the CZT.
///
/// The filter is `W^{-m²/2}` evaluated at indices `m = 0..M-1` and `m = 1..N-1`
/// placed at the wrap-around tail positions `L-(N-1)..L`.
/// The region `M..L-(N-1)` is zeroed.
///
/// # Returns
///
/// A `Vec` of length `l` representing `h_circ`.
pub fn build_chirp_filter<T: Float>(
    n: usize,
    m: usize,
    l: usize,
    w: Complex<T>,
) -> Vec<Complex<T>> {
    let mut h = vec![Complex::zero(); l];

    // Head portion: m = 0..M → h[0..M] = W^{-m²/2}
    for idx in 0..m {
        let m_f = T::from_usize(idx);
        let exp = m_f * m_f / T::TWO;
        h[idx] = complex_pow_real(w, -exp);
    }

    // Tail portion (wrap-around): m = 1..N-1 → h[L-m] = W^{-m²/2}
    for idx in 1..n {
        let m_f = T::from_usize(idx);
        let exp = m_f * m_f / T::TWO;
        h[l - idx] = complex_pow_real(w, -exp);
    }

    h
}

/// Compute `W^{k²/2}` for `k = 0 .. M`.
///
/// # Returns
///
/// A `Vec` of length `m`.
pub fn build_chirp_post<T: Float>(m: usize, w: Complex<T>) -> Vec<Complex<T>> {
    let mut out = Vec::with_capacity(m);
    for idx in 0..m {
        let k = T::from_usize(idx);
        let exp = k * k / T::TWO;
        out.push(complex_pow_real(w, exp));
    }
    out
}

/// Raise a complex number `z` to a real power `p`.
///
/// Uses the polar decomposition `z^p = |z|^p · exp(i·p·arg(z))`.
///
/// For the common case where `z` lies on the unit circle (`|z|² ≈ 1`), a
/// unit-circle short-circuit is taken: the magnitude term evaluates to exactly
/// 1 and is skipped, so only the angle is computed.  This avoids the chain
/// `sqrt(re²+im²)` → `pow(r, p)` which in f32 amplifies the ~1e-7 rounding of
/// `re²+im²` to `p·1e-7` (potentially 1e-4 for p ≈ N²/2 with large N).
#[inline]
pub(crate) fn complex_pow_real<T: Float>(z: Complex<T>, p: T) -> Complex<T> {
    let r_sq = z.re * z.re + z.im * z.im;
    let theta = num_traits::Float::atan2(z.im, z.re);
    // Unit-circle check: |r² - 1| < 16·ε to avoid amplifying norm rounding error.
    let eps16 = T::from_f64(16.0) * num_traits::Float::epsilon();
    let diff = if r_sq > T::ONE {
        r_sq - T::ONE
    } else {
        T::ONE - r_sq
    };
    if diff < eps16 {
        // On the unit circle: r^p = 1 exactly; skip the sqrt/powf.
        Complex::cis(p * theta)
    } else {
        let r = num_traits::Float::sqrt(r_sq);
        let r_p = num_traits::Float::powf(r, p);
        Complex::from_polar(r_p, p * theta)
    }
}
