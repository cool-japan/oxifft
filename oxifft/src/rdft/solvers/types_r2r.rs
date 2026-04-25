//! R2R solver struct definition and cached-plan construction.
//!
//! This module defines `R2rSolver<T>` together with the size-keyed precomputed
//! twiddle tables and cached `Plan<T>` objects that drive the O(N log N) fast paths
//! introduced in v0.3.0:
//!   - DCT-II: Makhoul N-point reduction (cached `plan_fwd_n`)
//!   - DCT-III: N-point conj-FFT-conj with Hermitian reconstruction (cached `plan_fwd_n`)
//!   - DCT-IV: zero-padded 4N extraction (cached `plan_fwd_4n`; per-call planning eliminated)

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};
use crate::prelude::Vec;

use super::r2r::R2rKind;

// ─────────────────────────────────────────────────────────────────────────────
// Struct
// ─────────────────────────────────────────────────────────────────────────────

/// Real-to-Real transform solver with cached plans and twiddle tables.
///
/// Build once for a given `(kind, n)` pair, then call [`R2rSolver::execute`]
/// repeatedly — the twiddle tables and plans are precomputed at construction.
pub struct R2rSolver<T: Float> {
    /// Transform kind (DCT/DST/DHT variant).
    pub(super) kind: R2rKind,
    /// Transform size.  0 means "stateless / unknown"; fast paths fall back to
    /// direct computation when the cached-plan size doesn't match.
    pub(super) n: usize,
    /// N-point forward complex DFT plan (used by DCT-II and DCT-III).
    pub(super) plan_fwd_n: Option<Plan<T>>,
    /// 4N-point forward complex DFT plan (used by DCT-IV zero-padded extraction).
    pub(super) plan_fwd_4n: Option<Plan<T>>,
    /// exp(-j·π·k / (2N)) for k = 0 ..= N/2  (length N/2+1).
    ///
    /// Used for the DCT-II post-twiddle multiply and DCT-III pre-twiddle.
    pub(super) twiddles_dct2: Vec<Complex<T>>,
    /// exp(-j·π·(2k+1) / (4N)) for k = 0 .. N  (length N).
    ///
    /// Used for the DCT-IV post-extraction step.
    pub(super) twiddles_dct4: Vec<Complex<T>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Construction helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build the twiddle table for DCT-II / DCT-III:
///   twiddles[k] = exp(-j·π·k / (2N))  for k = 0..=N/2
fn build_twiddles_dct2<T: Float>(n: usize) -> Vec<Complex<T>> {
    let half = n / 2;
    let two_n_f = T::from_usize(2 * n);
    let mut v = Vec::with_capacity(half + 1);
    for k in 0..=half {
        let angle = -<T as Float>::PI * T::from_usize(k) / two_n_f;
        let (s, c) = Float::sin_cos(angle);
        v.push(Complex::new(c, s));
    }
    v
}

/// Build the twiddle table for DCT-IV extraction:
///   twiddles[k] = exp(-j·π·(2k+1) / (4N))  for k = 0..N
fn build_twiddles_dct4<T: Float>(n: usize) -> Vec<Complex<T>> {
    let four_n_f = T::from_usize(4 * n);
    let mut v = Vec::with_capacity(n);
    for k in 0..n {
        let angle = -<T as Float>::PI * T::from_usize(2 * k + 1) / four_n_f;
        let (s, c) = Float::sin_cos(angle);
        v.push(Complex::new(c, s));
    }
    v
}

// ─────────────────────────────────────────────────────────────────────────────
// impl R2rSolver
// ─────────────────────────────────────────────────────────────────────────────

impl<T: Float> Default for R2rSolver<T> {
    fn default() -> Self {
        Self::new(R2rKind::Redft10, 0)
    }
}

impl<T: Float> R2rSolver<T> {
    /// Create a new `R2rSolver` for the given transform kind and size.
    ///
    /// All twiddle tables and cached plans are built here; subsequent calls to
    /// [`execute`](Self::execute) perform no heap allocation.
    ///
    /// # Arguments
    /// * `kind` – which DCT/DST/DHT variant
    /// * `n`    – transform size (0 means "unknown / stateless")
    #[must_use]
    pub fn new(kind: R2rKind, n: usize) -> Self {
        if n < 2 {
            // For n < 2 or unknown size all paths fall back to direct computation;
            // we skip plan construction.
            return Self {
                kind,
                n,
                plan_fwd_n: None,
                plan_fwd_4n: None,
                twiddles_dct2: Vec::new(),
                twiddles_dct4: Vec::new(),
            };
        }

        let plan_fwd_n = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE);
        let plan_fwd_4n = Plan::dft_1d(4 * n, Direction::Forward, Flags::ESTIMATE);

        let twiddles_dct2 = build_twiddles_dct2::<T>(n);
        let twiddles_dct4 = build_twiddles_dct4::<T>(n);

        Self {
            kind,
            n,
            plan_fwd_n,
            plan_fwd_4n,
            twiddles_dct2,
            twiddles_dct4,
        }
    }

    /// Get the solver name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self.kind {
            R2rKind::Redft00 => "rdft-redft00",
            R2rKind::Redft10 => "rdft-redft10",
            R2rKind::Redft01 => "rdft-redft01",
            R2rKind::Redft11 => "rdft-redft11",
            R2rKind::Rodft00 => "rdft-rodft00",
            R2rKind::Rodft10 => "rdft-rodft10",
            R2rKind::Rodft01 => "rdft-rodft01",
            R2rKind::Rodft11 => "rdft-rodft11",
            R2rKind::Dht => "rdft-dht",
        }
    }

    /// Check if this solver is applicable for the given size.
    ///
    /// Returns `true` when `n` matches the prebuilt plan size (or always for
    /// stateless solvers with `n == 0`).
    #[must_use]
    pub fn applicable(&self, n: usize) -> bool {
        self.n == 0 || self.n == n
    }

    /// Return the transform size this solver was built for.
    #[must_use]
    pub fn size(&self) -> usize {
        self.n
    }

    /// Return the transform kind this solver was built for.
    #[must_use]
    pub fn kind(&self) -> R2rKind {
        self.kind
    }
}
