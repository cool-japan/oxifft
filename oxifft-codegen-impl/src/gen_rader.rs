//! Rader prime codelet generation for primes 11 and 13.
//!
//! Rader's algorithm reduces a prime-length DFT to a cyclic convolution.
//! For prime p with generator g of (ℤ/pℤ)*:
//!
//! ```text
//! X[0]      = Σ_{n=0}^{p-1} x[n]
//! X[g^b]    = x[0] + (A * B)[b]     for b = 0..p-2
//! ```
//!
//! where:
//! - `A[c] = x[g^{-c} mod p]`   (input permuted by inverse generator powers)
//! - `B[m] = e^{sign·2πi·g^m/p}` (precomputed twiddles — hardcoded at codegen time)
//! - `*` denotes cyclic convolution of length (p-1)
//!
//! The cyclic convolution is expanded as straight-line code (no sub-FFT calls),
//! consistent with the `gen_odd.rs` Winograd codelet pattern.
//!
//! # DFT Convention
//!
//! Forward DFT: `sign < 0`, W = e^{-2πi/p}
//! Inverse DFT: `sign > 0`, W = e^{+2πi/p} (unnormalized)

use proc_macro2::TokenStream;
use quote::quote;
use syn::LitInt;

// ============================================================================
// Compile-time number-theory helpers
// ============================================================================

/// Modular exponentiation: base^exp mod m.
const fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * base % m;
        }
        base = base * base % m;
        exp >>= 1;
    }
    result
}

/// Returns the set of distinct prime factors of n.
fn prime_factors(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    let mut d = 2u64;
    while d * d <= n {
        if n % d == 0 {
            factors.push(d);
            while n % d == 0 {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

/// Check whether g is a primitive root mod p (p must be prime).
///
/// g is a primitive root mod p iff g^((p-1)/q) ≢ 1 (mod p) for every prime q | (p-1).
#[must_use]
pub fn is_primitive_root(g: u64, p: u64) -> bool {
    let pm1 = p - 1;
    for q in prime_factors(pm1) {
        if mod_pow(g, pm1 / q, p) == 1 {
            return false;
        }
    }
    true
}

/// Find the smallest primitive root of prime p.
///
/// # Panics
///
/// Panics if no primitive root is found below p (impossible for actual primes —
/// every prime has a primitive root by the theory of cyclic groups).
#[must_use]
pub fn find_generator(p: u64) -> u64 {
    for g in 2..p {
        if is_primitive_root(g, p) {
            return g;
        }
    }
    panic!("find_generator: no primitive root found for prime {p}");
}

// ============================================================================
// Precomputed constant tables (codegen time)
// ============================================================================

/// Generator powers for p=11: g^k mod 11 for k = 0..9 (g=2)
const G11_POWERS: [usize; 10] = [1, 2, 4, 8, 5, 10, 9, 7, 3, 6];
/// Inverse generator powers for p=11: g^{-k} mod 11 for k = 0..9
const G11_INV_POWERS: [usize; 10] = [1, 6, 3, 7, 9, 10, 5, 8, 4, 2];

/// Forward Rader twiddle re-parts for p=11: cos(-2π·g^m/11)
#[allow(clippy::excessive_precision)]
const B11_FWD_RE: [f64; 10] = [
    0.841_253_532_831_181_2_f64,
    0.415_415_013_001_886_4_f64,
    -0.654_860_733_945_285_1_f64,
    -0.142_314_838_273_285_23_f64,
    -0.959_492_973_614_497_4_f64,
    0.841_253_532_831_181_2_f64,
    0.415_415_013_001_886_05_f64,
    -0.654_860_733_945_285_2_f64,
    -0.142_314_838_273_285_01_f64,
    -0.959_492_973_614_497_5_f64,
];
/// Forward Rader twiddle im-parts for p=11: sin(-2π·g^m/11)
const B11_FWD_IM: [f64; 10] = [
    -0.540_640_817_455_597_6_f64,
    -0.909_631_995_354_518_3_f64,
    -0.755_749_574_354_258_3_f64,
    0.989_821_441_880_932_7_f64,
    -0.281_732_556_841_429_67_f64,
    0.540_640_817_455_597_6_f64,
    0.909_631_995_354_518_6_f64,
    0.755_749_574_354_258_2_f64,
    -0.989_821_441_880_932_8_f64,
    0.281_732_556_841_429_4_f64,
];
/// Backward Rader twiddle re-parts for p=11: cos(+2π·g^m/11) — same as forward (cos is even)
const B11_BWD_RE: [f64; 10] = B11_FWD_RE;
/// Backward Rader twiddle im-parts for p=11: sin(+2π·g^m/11) — negated imaginary parts
const B11_BWD_IM: [f64; 10] = [
    0.540_640_817_455_597_6_f64,
    0.909_631_995_354_518_3_f64,
    0.755_749_574_354_258_3_f64,
    -0.989_821_441_880_932_7_f64,
    0.281_732_556_841_429_67_f64,
    -0.540_640_817_455_597_6_f64,
    -0.909_631_995_354_518_6_f64,
    -0.755_749_574_354_258_2_f64,
    0.989_821_441_880_932_8_f64,
    -0.281_732_556_841_429_4_f64,
];

/// Generator powers for p=13: g^k mod 13 for k = 0..11 (g=2)
const G13_POWERS: [usize; 12] = [1, 2, 4, 8, 3, 6, 12, 11, 9, 5, 10, 7];
/// Inverse generator powers for p=13: g^{-k} mod 13 for k = 0..11
const G13_INV_POWERS: [usize; 12] = [1, 7, 10, 5, 9, 11, 12, 6, 3, 8, 4, 2];

/// Forward Rader twiddle re-parts for p=13: cos(-2π·g^m/13)
#[allow(clippy::excessive_precision)]
const B13_FWD_RE: [f64; 12] = [
    0.885_456_025_653_209_9_f64,
    0.568_064_746_731_155_8_f64,
    -0.354_604_887_042_535_5_f64,
    -0.748_510_748_171_101_3_f64,
    0.120_536_680_255_323_01_f64,
    -0.970_941_817_426_052_f64,
    0.885_456_025_653_210_f64,
    0.568_064_746_731_154_8_f64,
    -0.354_604_887_042_535_9_f64,
    -0.748_510_748_171_101_2_f64,
    0.120_536_680_255_323_2_f64,
    -0.970_941_817_426_052_1_f64,
];
/// Forward Rader twiddle im-parts for p=13: sin(-2π·g^m/13)
#[allow(clippy::excessive_precision)]
const B13_FWD_IM: [f64; 12] = [
    -0.464_723_172_043_768_5_f64,
    -0.822_983_865_893_656_4_f64,
    -0.935_016_242_685_414_8_f64,
    0.663_122_658_240_795_f64,
    -0.992_708_874_098_054_f64,
    -0.239_315_664_287_557_68_f64,
    0.464_723_172_043_768_4_f64,
    0.822_983_865_893_657_f64,
    0.935_016_242_685_414_7_f64,
    -0.663_122_658_240_795_2_f64,
    0.992_708_874_098_054_f64,
    0.239_315_664_287_557_43_f64,
];
/// Backward Rader twiddle re-parts for p=13: cos(+2π·g^m/13) — same as forward (cos is even)
const B13_BWD_RE: [f64; 12] = B13_FWD_RE;
/// Backward Rader twiddle im-parts for p=13: sin(+2π·g^m/13) — negated imaginary parts
#[allow(clippy::excessive_precision)]
const B13_BWD_IM: [f64; 12] = [
    0.464_723_172_043_768_5_f64,
    0.822_983_865_893_656_4_f64,
    0.935_016_242_685_414_8_f64,
    -0.663_122_658_240_795_f64,
    0.992_708_874_098_054_f64,
    0.239_315_664_287_557_68_f64,
    -0.464_723_172_043_768_4_f64,
    -0.822_983_865_893_657_f64,
    -0.935_016_242_685_414_7_f64,
    0.663_122_658_240_795_2_f64,
    -0.992_708_874_098_054_f64,
    -0.239_315_664_287_557_43_f64,
];

// ============================================================================
// Public entry points
// ============================================================================

/// Parse `gen_rader_codelet!(N)` input and dispatch.
///
/// # Errors
/// Returns `syn::Error` if the input is not a valid integer literal or the
/// prime is not in {11, 13}.
pub fn generate_from_macro(input: TokenStream) -> Result<TokenStream, syn::Error> {
    let size: LitInt = syn::parse2(input)?;
    let prime: usize = size.base10_parse().map_err(|_| {
        syn::Error::new(
            size.span(),
            "gen_rader_codelet: expected an integer prime literal",
        )
    })?;

    match prime {
        11 => Ok(gen_size_11()),
        13 => Ok(gen_size_13()),
        _ => Err(syn::Error::new(
            size.span(),
            format!("gen_rader_codelet: unsupported prime {prime} (expected one of 11, 13)"),
        )),
    }
}

/// Generate a Rader-form codelet `TokenStream` for the given prime ∈ {11, 13}.
///
/// This is the non-proc-macro entry point used by benchmark/test harnesses.
///
/// # Panics
///
/// Panics if `prime` is not 11 or 13.
#[must_use]
pub fn generate_rader(prime: usize) -> TokenStream {
    match prime {
        11 => gen_size_11(),
        13 => gen_size_13(),
        _ => panic!("gen_rader: unsupported prime {prime} (expected 11 or 13)"),
    }
}

// ============================================================================
// DFT-11 codelet (Rader, straight-line cyclic convolution of length 10)
// ============================================================================

#[allow(clippy::similar_names)]
fn gen_size_11() -> TokenStream {
    // Emit g_powers and g_inv_powers as literal arrays so the quote! can use them.
    let g_pows: Vec<proc_macro2::Literal> = G11_POWERS
        .iter()
        .map(|&v| proc_macro2::Literal::usize_suffixed(v))
        .collect();
    let g_inv_pows: Vec<proc_macro2::Literal> = G11_INV_POWERS
        .iter()
        .map(|&v| proc_macro2::Literal::usize_suffixed(v))
        .collect();

    // Build forward and backward twiddle literal arrays with distinct names.
    let twd11_fwd_re: Vec<proc_macro2::Literal> = B11_FWD_RE
        .iter()
        .map(|&v| proc_macro2::Literal::f64_suffixed(v))
        .collect();
    let twd11_fwd_im: Vec<proc_macro2::Literal> = B11_FWD_IM
        .iter()
        .map(|&v| proc_macro2::Literal::f64_suffixed(v))
        .collect();
    let twd11_bwd_re: Vec<proc_macro2::Literal> = B11_BWD_RE
        .iter()
        .map(|&v| proc_macro2::Literal::f64_suffixed(v))
        .collect();
    let twd11_bwd_im: Vec<proc_macro2::Literal> = B11_BWD_IM
        .iter()
        .map(|&v| proc_macro2::Literal::f64_suffixed(v))
        .collect();

    quote! {
        /// Size-11 DFT codelet using Rader's algorithm.
        ///
        /// Reduces the prime-11 DFT to a cyclic convolution of length 10,
        /// computed as straight-line code.  Generator g = 2.
        ///
        /// `sign < 0` → forward transform (W = e^{-2πi/11});
        /// `sign > 0` → inverse (unnormalized, W = e^{+2πi/11}).
        #[inline(always)]
        #[allow(
            clippy::too_many_lines,
            clippy::approx_constant,
            clippy::suboptimal_flops,
            clippy::unreadable_literal
        )]
        pub fn codelet_notw_11<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 11);

            // ── Step 1: X[0] = sum of all inputs ──────────────────────────
            let mut sum_re = T::zero();
            let mut sum_im = T::zero();
            for i in 0..11usize {
                sum_re = sum_re + x[i].re;
                sum_im = sum_im + x[i].im;
            }

            // ── Step 2: A[c] = x[g^{-c} mod 11] ──────────────────────────
            // g_inv_powers[c] for c = 0..9
            let g_inv_powers: [usize; 10] = [#(#g_inv_pows),*];
            let mut a_re = [T::zero(); 10];
            let mut a_im = [T::zero(); 10];
            for c in 0..10usize {
                let idx = g_inv_powers[c];
                a_re[c] = x[idx].re;
                a_im[c] = x[idx].im;
            }

            // ── Step 3: Select twiddle factors based on sign ───────────────
            // Forward B[m] = e^{-2πi·g^m/11},  Inverse B[m] = e^{+2πi·g^m/11}
            let tw_re: [T; 10];
            let tw_im: [T; 10];
            if sign < 0 {
                tw_re = [#(T::from_f64(#twd11_fwd_re)),*];
                tw_im = [#(T::from_f64(#twd11_fwd_im)),*];
            } else {
                tw_re = [#(T::from_f64(#twd11_bwd_re)),*];
                tw_im = [#(T::from_f64(#twd11_bwd_im)),*];
            }

            // ── Step 4: Cyclic convolution conv[b] = Σ_c A[c]·B[(b-c)%10] ─
            let mut conv_re = [T::zero(); 10];
            let mut conv_im = [T::zero(); 10];
            for b in 0..10usize {
                let mut cr = T::zero();
                let mut ci = T::zero();
                for c in 0..10usize {
                    let bc = (10 + b - c) % 10;
                    // complex mul: A[c] * B[bc]
                    cr = cr + a_re[c] * tw_re[bc] - a_im[c] * tw_im[bc];
                    ci = ci + a_re[c] * tw_im[bc] + a_im[c] * tw_re[bc];
                }
                conv_re[b] = cr;
                conv_im[b] = ci;
            }

            // ── Step 5: Assemble output ────────────────────────────────────
            // X[0]     = sum
            // X[g^b]   = x[0] + conv[b]  for b = 0..9
            let x0_re = x[0].re;
            let x0_im = x[0].im;
            x[0] = crate::kernel::Complex::new(sum_re, sum_im);

            let g_powers: [usize; 10] = [#(#g_pows),*];
            for b in 0..10usize {
                let idx = g_powers[b];
                x[idx] = crate::kernel::Complex::new(x0_re + conv_re[b], x0_im + conv_im[b]);
            }
        }
    }
}

// ============================================================================
// DFT-13 codelet (Rader, straight-line cyclic convolution of length 12)
// ============================================================================

#[allow(clippy::similar_names)]
fn gen_size_13() -> TokenStream {
    let g_pows: Vec<proc_macro2::Literal> = G13_POWERS
        .iter()
        .map(|&v| proc_macro2::Literal::usize_suffixed(v))
        .collect();
    let g_inv_pows: Vec<proc_macro2::Literal> = G13_INV_POWERS
        .iter()
        .map(|&v| proc_macro2::Literal::usize_suffixed(v))
        .collect();

    let twd13_fwd_re: Vec<proc_macro2::Literal> = B13_FWD_RE
        .iter()
        .map(|&v| proc_macro2::Literal::f64_suffixed(v))
        .collect();
    let twd13_fwd_im: Vec<proc_macro2::Literal> = B13_FWD_IM
        .iter()
        .map(|&v| proc_macro2::Literal::f64_suffixed(v))
        .collect();
    let twd13_bwd_re: Vec<proc_macro2::Literal> = B13_BWD_RE
        .iter()
        .map(|&v| proc_macro2::Literal::f64_suffixed(v))
        .collect();
    let twd13_bwd_im: Vec<proc_macro2::Literal> = B13_BWD_IM
        .iter()
        .map(|&v| proc_macro2::Literal::f64_suffixed(v))
        .collect();

    quote! {
        /// Size-13 DFT codelet using Rader's algorithm.
        ///
        /// Reduces the prime-13 DFT to a cyclic convolution of length 12,
        /// computed as straight-line code.  Generator g = 2.
        ///
        /// `sign < 0` → forward transform (W = e^{-2πi/13});
        /// `sign > 0` → inverse (unnormalized, W = e^{+2πi/13}).
        #[inline(always)]
        #[allow(
            clippy::too_many_lines,
            clippy::approx_constant,
            clippy::suboptimal_flops,
            clippy::unreadable_literal
        )]
        pub fn codelet_notw_13<T: crate::kernel::Float>(
            x: &mut [crate::kernel::Complex<T>],
            sign: i32,
        ) {
            debug_assert!(x.len() >= 13);

            // ── Step 1: X[0] = sum of all inputs ──────────────────────────
            let mut sum_re = T::zero();
            let mut sum_im = T::zero();
            for i in 0..13usize {
                sum_re = sum_re + x[i].re;
                sum_im = sum_im + x[i].im;
            }

            // ── Step 2: A[c] = x[g^{-c} mod 13] ──────────────────────────
            let g_inv_powers: [usize; 12] = [#(#g_inv_pows),*];
            let mut a_re = [T::zero(); 12];
            let mut a_im = [T::zero(); 12];
            for c in 0..12usize {
                let idx = g_inv_powers[c];
                a_re[c] = x[idx].re;
                a_im[c] = x[idx].im;
            }

            // ── Step 3: Select twiddle factors based on sign ───────────────
            let tw_re: [T; 12];
            let tw_im: [T; 12];
            if sign < 0 {
                tw_re = [#(T::from_f64(#twd13_fwd_re)),*];
                tw_im = [#(T::from_f64(#twd13_fwd_im)),*];
            } else {
                tw_re = [#(T::from_f64(#twd13_bwd_re)),*];
                tw_im = [#(T::from_f64(#twd13_bwd_im)),*];
            }

            // ── Step 4: Cyclic convolution conv[b] = Σ_c A[c]·B[(b-c)%12] ─
            let mut conv_re = [T::zero(); 12];
            let mut conv_im = [T::zero(); 12];
            for b in 0..12usize {
                let mut cr = T::zero();
                let mut ci = T::zero();
                for c in 0..12usize {
                    let bc = (12 + b - c) % 12;
                    cr = cr + a_re[c] * tw_re[bc] - a_im[c] * tw_im[bc];
                    ci = ci + a_re[c] * tw_im[bc] + a_im[c] * tw_re[bc];
                }
                conv_re[b] = cr;
                conv_im[b] = ci;
            }

            // ── Step 5: Assemble output ────────────────────────────────────
            let x0_re = x[0].re;
            let x0_im = x[0].im;
            x[0] = crate::kernel::Complex::new(sum_re, sum_im);

            let g_powers: [usize; 12] = [#(#g_pows),*];
            for b in 0..12usize {
                let idx = g_powers[b];
                x[idx] = crate::kernel::Complex::new(x0_re + conv_re[b], x0_im + conv_im[b]);
            }
        }
    }
}

// ============================================================================
// Pure-f64 reference implementations for #[cfg(test)]
// ============================================================================

/// Naive O(N²) DFT reference (forward, sign=-1).
#[cfg(test)]
#[allow(clippy::suboptimal_flops)]
pub(crate) fn naive_dft_fwd(x_re: &[f64], x_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x_re.len();
    debug_assert_eq!(x_im.len(), n);
    let mut out_re = vec![0.0_f64; n];
    let mut out_im = vec![0.0_f64; n];
    for k in 0..n {
        for j in 0..n {
            let angle = -2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
            let (s, c) = angle.sin_cos();
            out_re[k] += x_re[j] * c - x_im[j] * s;
            out_im[k] += x_re[j] * s + x_im[j] * c;
        }
    }
    (out_re, out_im)
}

/// Naive O(N²) inverse DFT reference (sign=+1, unnormalized).
#[cfg(test)]
#[allow(clippy::suboptimal_flops)]
pub(crate) fn naive_dft_inv(x_re: &[f64], x_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x_re.len();
    debug_assert_eq!(x_im.len(), n);
    let mut out_re = vec![0.0_f64; n];
    let mut out_im = vec![0.0_f64; n];
    for k in 0..n {
        for j in 0..n {
            let angle = 2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
            let (s, c) = angle.sin_cos();
            out_re[k] += x_re[j] * c - x_im[j] * s;
            out_im[k] += x_re[j] * s + x_im[j] * c;
        }
    }
    (out_re, out_im)
}

/// Rader DFT-11 (forward) in pure f64 — mirrors the generated codelet.
#[cfg(test)]
pub(crate) fn rader_dft11_fwd(x_re: &[f64], x_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(x_re.len(), 11);
    rader_dft_generic(
        x_re,
        x_im,
        &G11_POWERS,
        &G11_INV_POWERS,
        &B11_FWD_RE,
        &B11_FWD_IM,
    )
}

/// Rader DFT-11 (inverse, unnormalized) in pure f64.
#[cfg(test)]
pub(crate) fn rader_dft11_inv(x_re: &[f64], x_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(x_re.len(), 11);
    rader_dft_generic(
        x_re,
        x_im,
        &G11_POWERS,
        &G11_INV_POWERS,
        &B11_BWD_RE,
        &B11_BWD_IM,
    )
}

/// Rader DFT-13 (forward) in pure f64.
#[cfg(test)]
pub(crate) fn rader_dft13_fwd(x_re: &[f64], x_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(x_re.len(), 13);
    rader_dft_generic(
        x_re,
        x_im,
        &G13_POWERS,
        &G13_INV_POWERS,
        &B13_FWD_RE,
        &B13_FWD_IM,
    )
}

/// Rader DFT-13 (inverse, unnormalized) in pure f64.
#[cfg(test)]
pub(crate) fn rader_dft13_inv(x_re: &[f64], x_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(x_re.len(), 13);
    rader_dft_generic(
        x_re,
        x_im,
        &G13_POWERS,
        &G13_INV_POWERS,
        &B13_BWD_RE,
        &B13_BWD_IM,
    )
}

/// Generic Rader DFT in pure f64 for testing (not compiled in production).
///
/// Computes the Rader DFT via direct straight-line cyclic convolution.
#[cfg(test)]
#[allow(clippy::suboptimal_flops)]
fn rader_dft_generic(
    x_re: &[f64],
    x_im: &[f64],
    g_powers: &[usize],
    g_inv_powers: &[usize],
    twd_re: &[f64],
    twd_im: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let p = x_re.len();
    let n = p - 1;
    debug_assert_eq!(g_powers.len(), n);
    debug_assert_eq!(g_inv_powers.len(), n);
    debug_assert_eq!(twd_re.len(), n);
    debug_assert_eq!(twd_im.len(), n);

    // Step 1: X[0] = sum of all inputs
    let sum_re: f64 = x_re.iter().sum();
    let sum_im: f64 = x_im.iter().sum();

    // Step 2: A[c] = x[g^{-c}]
    let a_re: Vec<f64> = (0..n).map(|c| x_re[g_inv_powers[c]]).collect();
    let a_im: Vec<f64> = (0..n).map(|c| x_im[g_inv_powers[c]]).collect();

    // Step 3: Cyclic convolution
    let mut conv_re = vec![0.0_f64; n];
    let mut conv_im = vec![0.0_f64; n];
    for b in 0..n {
        for c in 0..n {
            let bc = (n + b - c) % n;
            conv_re[b] += a_re[c] * twd_re[bc] - a_im[c] * twd_im[bc];
            conv_im[b] += a_re[c] * twd_im[bc] + a_im[c] * twd_re[bc];
        }
    }

    // Step 4: Assemble output
    let mut out_re = vec![0.0_f64; p];
    let mut out_im = vec![0.0_f64; p];
    out_re[0] = sum_re;
    out_im[0] = sum_im;
    for b in 0..n {
        let idx = g_powers[b];
        out_re[idx] = x_re[0] + conv_re[b];
        out_im[idx] = x_im[0] + conv_im[b];
    }

    (out_re, out_im)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn assert_close(a: &[f64], b: &[f64], label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < TOL,
                "{label}[{i}]: got {x}, expected {y}, diff = {}",
                (x - y).abs()
            );
        }
    }

    // ── Utility: generate deterministic test vectors ──────────────────────────

    fn test_vec_11() -> ([f64; 11], [f64; 11]) {
        let x_re = [
            0.493_581, -1.234_567, 0.812_345, -0.456_789, 1.123_456, -0.234_567, 0.678_901,
            -0.890_123, 0.345_678, -0.567_890, 0.901_234,
        ];
        let x_im = [
            0.234_567, 0.678_901, -0.456_789, 0.890_123, -0.123_456, 0.567_890, -0.789_012,
            0.234_567, -0.678_901, 0.456_789, -0.890_123,
        ];
        (x_re, x_im)
    }

    fn test_vec_13() -> ([f64; 13], [f64; 13]) {
        let x_re = [
            0.493_581, -1.234_567, 0.812_345, -0.456_789, 1.123_456, -0.234_567, 0.678_901,
            -0.890_123, 0.345_678, -0.567_890, 0.901_234, -0.123_456, 0.789_012,
        ];
        let x_im = [
            0.234_567, 0.678_901, -0.456_789, 0.890_123, -0.123_456, 0.567_890, -0.789_012,
            0.234_567, -0.678_901, 0.456_789, -0.890_123, 0.123_456, -0.567_890,
        ];
        (x_re, x_im)
    }

    // ── Number-theory helpers ─────────────────────────────────────────────────

    #[test]
    fn test_generator_11() {
        assert!(
            is_primitive_root(2, 11),
            "2 should be a primitive root mod 11"
        );
        assert!(
            !is_primitive_root(10, 11),
            "10 should NOT be a primitive root mod 11"
        );
        assert_eq!(find_generator(11), 2);
    }

    #[test]
    fn test_generator_13() {
        assert!(
            is_primitive_root(2, 13),
            "2 should be a primitive root mod 13"
        );
        assert_eq!(find_generator(13), 2);
    }

    #[test]
    fn test_mod_pow_basic() {
        // Fermat's little theorem: g^(p-1) ≡ 1 (mod p)
        assert_eq!(mod_pow(2, 10, 11), 1); // 2^10 ≡ 1 (mod 11)
        assert_eq!(mod_pow(2, 12, 13), 1); // 2^12 ≡ 1 (mod 13)
    }

    // ── Impulse tests (catches sign-convention bugs) ───────────────────────────

    #[test]
    fn test_dft11_forward_f64_impulse() {
        // DFT of unit impulse at index 0: all outputs should be 1+0i
        let x_re = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let x_im = [0.0; 11];
        let (got_re, got_im) = rader_dft11_fwd(&x_re, &x_im);
        assert_close(&got_re, &[1.0; 11], "dft11_impulse_re");
        assert_close(&got_im, &[0.0; 11], "dft11_impulse_im");
    }

    #[test]
    fn test_dft13_forward_f64_impulse() {
        // DFT of unit impulse at index 0: all outputs should be 1+0i
        let x_re = [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let x_im = [0.0; 13];
        let (got_re, got_im) = rader_dft13_fwd(&x_re, &x_im);
        assert_close(&got_re, &[1.0; 13], "dft13_impulse_re");
        assert_close(&got_im, &[0.0; 13], "dft13_impulse_im");
    }

    // ── Forward vs. naive DFT ─────────────────────────────────────────────────

    #[test]
    fn test_rader11_forward_vs_naive() {
        let (x_re, x_im) = test_vec_11();
        let (got_re, got_im) = rader_dft11_fwd(&x_re, &x_im);
        let (ref_re, ref_im) = naive_dft_fwd(&x_re, &x_im);
        assert_close(&got_re, &ref_re, "rader11_fwd_re");
        assert_close(&got_im, &ref_im, "rader11_fwd_im");
    }

    #[test]
    fn test_rader13_forward_vs_naive() {
        let (x_re, x_im) = test_vec_13();
        let (got_re, got_im) = rader_dft13_fwd(&x_re, &x_im);
        let (ref_re, ref_im) = naive_dft_fwd(&x_re, &x_im);
        assert_close(&got_re, &ref_re, "rader13_fwd_re");
        assert_close(&got_im, &ref_im, "rader13_fwd_im");
    }

    // ── Inverse vs. naive IDFT ────────────────────────────────────────────────

    #[test]
    fn test_rader11_inverse_vs_naive() {
        let (x_re, x_im) = test_vec_11();
        let (got_re, got_im) = rader_dft11_inv(&x_re, &x_im);
        let (ref_re, ref_im) = naive_dft_inv(&x_re, &x_im);
        assert_close(&got_re, &ref_re, "rader11_inv_re");
        assert_close(&got_im, &ref_im, "rader11_inv_im");
    }

    #[test]
    fn test_rader13_inverse_vs_naive() {
        let (x_re, x_im) = test_vec_13();
        let (got_re, got_im) = rader_dft13_inv(&x_re, &x_im);
        let (ref_re, ref_im) = naive_dft_inv(&x_re, &x_im);
        assert_close(&got_re, &ref_re, "rader13_inv_re");
        assert_close(&got_im, &ref_im, "rader13_inv_im");
    }

    // ── Round-trip: fwd → inv → scale → original ─────────────────────────────

    #[test]
    fn test_roundtrip_rader11() {
        let (x_re, x_im) = test_vec_11();
        let (fwd_re, fwd_im) = rader_dft11_fwd(&x_re, &x_im);
        let (inv_re, inv_im) = rader_dft11_inv(&fwd_re, &fwd_im);
        let n = 11.0_f64;
        let scaled_re: Vec<f64> = inv_re.iter().map(|&v| v / n).collect();
        let scaled_im: Vec<f64> = inv_im.iter().map(|&v| v / n).collect();
        assert_close(&scaled_re, &x_re, "roundtrip_rader11_re");
        assert_close(&scaled_im, &x_im, "roundtrip_rader11_im");
    }

    #[test]
    fn test_roundtrip_rader13() {
        let (x_re, x_im) = test_vec_13();
        let (fwd_re, fwd_im) = rader_dft13_fwd(&x_re, &x_im);
        let (inv_re, inv_im) = rader_dft13_inv(&fwd_re, &fwd_im);
        let n = 13.0_f64;
        let scaled_re: Vec<f64> = inv_re.iter().map(|&v| v / n).collect();
        let scaled_im: Vec<f64> = inv_im.iter().map(|&v| v / n).collect();
        assert_close(&scaled_re, &x_re, "roundtrip_rader13_re");
        assert_close(&scaled_im, &x_im, "roundtrip_rader13_im");
    }

    // ── TokenStream structural checks ─────────────────────────────────────────

    #[test]
    fn test_generate_from_macro_prime11() {
        let input: proc_macro2::TokenStream = "11".parse().expect("parse literal");
        let result = generate_from_macro(input);
        assert!(result.is_ok(), "gen_rader_codelet!(11) should succeed");
        let ts = result.expect("TokenStream for prime 11");
        let s = ts.to_string();
        assert!(
            s.contains("codelet_notw_11"),
            "should contain codelet_notw_11"
        );
        assert!(s.contains("sign"), "should contain sign parameter");
    }

    #[test]
    fn test_generate_from_macro_prime13() {
        let input: proc_macro2::TokenStream = "13".parse().expect("parse literal");
        let result = generate_from_macro(input);
        assert!(result.is_ok(), "gen_rader_codelet!(13) should succeed");
        let ts = result.expect("TokenStream for prime 13");
        let s = ts.to_string();
        assert!(
            s.contains("codelet_notw_13"),
            "should contain codelet_notw_13"
        );
    }

    #[test]
    fn test_generate_from_macro_unsupported() {
        let input: proc_macro2::TokenStream = "17".parse().expect("parse literal");
        let result = generate_from_macro(input);
        assert!(
            result.is_err(),
            "gen_rader_codelet!(17) should fail with unsupported prime"
        );
    }
}
