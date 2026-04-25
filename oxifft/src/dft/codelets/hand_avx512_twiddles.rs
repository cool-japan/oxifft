//! Precomputed twiddle factor tables for hand-tuned AVX-512 codelets.
//!
//! Stores 64-byte-aligned twiddle factors for complex DFT sizes 16, 32, 64
//! in both f64 and f32 precision.  Tables are initialised once via
//! `OnceLock` (`std::sync::OnceLock` on std, `spin::Once` on no_std).
//!
//! Layout convention for each table:
//!   - Forward (sign = −1):  `W_n^k = exp(−2πi·k/n)` stored as `[re, im]` pairs
//!   - Inverse (sign = +1):  `W_n^{−k} = exp(+2πi·k/n)` stored as `[re, im]` pairs
//!
//! All tables are 64-byte aligned for cache-line-friendly AVX-512 loads.

use crate::prelude::OnceLock;
#[cfg(not(feature = "std"))]
use crate::prelude::OnceLockExt;

// ─────────────────────────────────────────────────────────────────────────────
// Alignment wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// 64-byte-aligned f64 array for AVX-512 loads.
#[repr(C, align(64))]
pub(super) struct AlignedF64<const N: usize>(pub [f64; N]);

/// 64-byte-aligned f32 array for AVX-512 f32 loads.
#[repr(C, align(64))]
pub(super) struct AlignedF32<const N: usize>(pub [f32; N]);

// ─────────────────────────────────────────────────────────────────────────────
// Table structures
// ─────────────────────────────────────────────────────────────────────────────

/// Twiddle tables for one FFT size in f64.
///
/// `fwd.0[2*k]` = `cos(2πk/n)`, `fwd.0[2*k+1]` = `−sin(2πk/n)` (forward W_n^k).
/// `inv.0[2*k]` = `cos(2πk/n)`, `inv.0[2*k+1]` = `+sin(2πk/n)` (inverse W_n^{−k}).
pub(super) struct TwiddlesF64<const N: usize> {
    /// Forward (sign = −1) twiddle factors — interleaved `[re, im, re, im, …]`.
    pub fwd: AlignedF64<N>,
    /// Inverse (sign = +1) twiddle factors — interleaved `[re, im, re, im, …]`.
    pub inv: AlignedF64<N>,
}

/// Twiddle tables for one FFT size in f32.
pub(super) struct TwiddlesF32<const N: usize> {
    /// Forward twiddle factors.
    pub fwd: AlignedF32<N>,
    /// Inverse twiddle factors.
    pub inv: AlignedF32<N>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: fill interleaved [re, im] twiddle table
// ─────────────────────────────────────────────────────────────────────────────

/// Build an f64 twiddle table for `n`-point DFT.
///
/// `N` must equal `2 * n` (interleaved re/im for each of the `n` twiddle factors).
fn build_twiddles_f64<const N: usize>(n: usize) -> TwiddlesF64<N> {
    let pairs = N / 2; // = n
    let mut fwd = [0.0_f64; N];
    let mut inv = [0.0_f64; N];
    for k in 0..pairs {
        let theta = core::f64::consts::TAU * (k as f64) / (n as f64);
        let (s, c) = theta.sin_cos();
        fwd[2 * k] = c; // re = cos θ
        fwd[2 * k + 1] = -s; // im = −sin θ  (forward: W_n^k)
        inv[2 * k] = c; // re = cos θ
        inv[2 * k + 1] = s; // im = +sin θ  (inverse: W_n^{−k})
    }
    TwiddlesF64 {
        fwd: AlignedF64(fwd),
        inv: AlignedF64(inv),
    }
}

/// Build an f32 twiddle table for `n`-point DFT.
fn build_twiddles_f32<const N: usize>(n: usize) -> TwiddlesF32<N> {
    let pairs = N / 2;
    let mut fwd = [0.0_f32; N];
    let mut inv = [0.0_f32; N];
    for k in 0..pairs {
        let theta = core::f32::consts::TAU * (k as f32) / (n as f32);
        let (s, c) = theta.sin_cos();
        fwd[2 * k] = c;
        fwd[2 * k + 1] = -s;
        inv[2 * k] = c;
        inv[2 * k + 1] = s;
    }
    TwiddlesF32 {
        fwd: AlignedF32(fwd),
        inv: AlignedF32(inv),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Size-16 f64  (16 twiddle pairs → 32 f64 scalars)
// ─────────────────────────────────────────────────────────────────────────────

/// Return the static f64 twiddle table for the size-16 FFT.
pub(super) fn twiddles_16_f64() -> &'static TwiddlesF64<32> {
    static ONCE: OnceLock<TwiddlesF64<32>> = OnceLock::new();
    ONCE.get_or_init(|| build_twiddles_f64::<32>(16))
}

// ─────────────────────────────────────────────────────────────────────────────
// Size-32 f64  (32 twiddle pairs → 64 f64 scalars)
// ─────────────────────────────────────────────────────────────────────────────

/// Return the static f64 twiddle table for the size-32 FFT.
pub(super) fn twiddles_32_f64() -> &'static TwiddlesF64<64> {
    static ONCE: OnceLock<TwiddlesF64<64>> = OnceLock::new();
    ONCE.get_or_init(|| build_twiddles_f64::<64>(32))
}

// ─────────────────────────────────────────────────────────────────────────────
// Size-64 f64  (64 twiddle pairs → 128 f64 scalars)
// ─────────────────────────────────────────────────────────────────────────────

/// Return the static f64 twiddle table for the size-64 FFT.
pub(super) fn twiddles_64_f64() -> &'static TwiddlesF64<128> {
    static ONCE: OnceLock<TwiddlesF64<128>> = OnceLock::new();
    ONCE.get_or_init(|| build_twiddles_f64::<128>(64))
}

// ─────────────────────────────────────────────────────────────────────────────
// Size-16 f32  (16 twiddle pairs → 32 f32 scalars)
// ─────────────────────────────────────────────────────────────────────────────

/// Return the static f32 twiddle table for the size-16 FFT.
pub(super) fn twiddles_16_f32() -> &'static TwiddlesF32<32> {
    static ONCE: OnceLock<TwiddlesF32<32>> = OnceLock::new();
    ONCE.get_or_init(|| build_twiddles_f32::<32>(16))
}

// ─────────────────────────────────────────────────────────────────────────────
// Size-32 f32  (32 twiddle pairs → 64 f32 scalars)
// ─────────────────────────────────────────────────────────────────────────────

/// Return the static f32 twiddle table for the size-32 FFT.
pub(super) fn twiddles_32_f32() -> &'static TwiddlesF32<64> {
    static ONCE: OnceLock<TwiddlesF32<64>> = OnceLock::new();
    ONCE.get_or_init(|| build_twiddles_f32::<64>(32))
}

// ─────────────────────────────────────────────────────────────────────────────
// Size-64 f32  (64 twiddle pairs → 128 f32 scalars)
// ─────────────────────────────────────────────────────────────────────────────

/// Return the static f32 twiddle table for the size-64 FFT.
pub(super) fn twiddles_64_f32() -> &'static TwiddlesF32<128> {
    static ONCE: OnceLock<TwiddlesF32<128>> = OnceLock::new();
    ONCE.get_or_init(|| build_twiddles_f32::<128>(64))
}
