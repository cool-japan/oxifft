//! Partial FFT: compute only a subset of DFT outputs.
//!
//! Selects the most efficient algorithm based on (K, N, M) cost analysis:
//!
//! - **Goertzel**: O(K·N). Optimal when K << log₂(N) (sparse output bins).
//! - **OutputPruned**: Backed by Goertzel over a prefix range. Used when
//!   M is a power-of-two and M < log₂(N), making prefix-Goertzel cost-competitive.
//! - **FullThenSlice**: Full O(N·log N) FFT then copy a range. Fallback when
//!   K or M is large enough that full FFT is cheaper.
//!
//! # Example
//!
//! ```ignore
//! use oxifft::pruned::{PartialFft, PartialStrategy};
//! use oxifft::Complex;
//!
//! let input: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); 1024];
//!
//! // Compute bins 10, 20, 50 — automatically selects Goertzel
//! let pf = PartialFft::<f64>::new_sparse(1024, &[10, 20, 50]);
//! let mut out = vec![Complex::new(0.0_f64, 0.0); 3];
//! pf.execute(&input, &mut out);
//!
//! // Compute first 16 bins from 1024-point input — uses OutputPruned
//! let pf2 = PartialFft::<f64>::new_prefix(1024, 16);
//! let mut out2 = vec![Complex::new(0.0_f64, 0.0); 16];
//! pf2.execute(&input, &mut out2);
//! ```

use core::ops::Range;

use crate::api::{Direction, Flags, Plan};
use crate::kernel::{Complex, Float};
use crate::prelude::*;

// ─── Strategy enum ───────────────────────────────────────────────────────────

/// The algorithm that [`PartialFft`] will use when [`execute`](PartialFft::execute) is called.
///
/// The strategy is chosen automatically by [`new_sparse`](PartialFft::new_sparse) and
/// [`new_prefix`](PartialFft::new_prefix) based on a cost model.
#[derive(Debug, Clone)]
pub enum PartialStrategy {
    /// Goertzel algorithm O(K·N). Best when K << log₂(N).
    Goertzel {
        /// The bin indices that will be computed.
        bins: Vec<usize>,
    },
    /// Output-pruned computation (Goertzel over a prefix). O(M·N).
    /// Used when M is a power-of-two and M < log₂(N).
    OutputPruned {
        /// Number of prefix outputs requested.
        m: usize,
    },
    /// Full FFT then copy a contiguous range. O(N·log N). Fallback.
    FullThenSlice {
        /// The range of bins to copy from the full FFT result.
        range: Range<usize>,
    },
}

// ─── PartialFft ──────────────────────────────────────────────────────────────

/// Partial FFT that computes only a subset of DFT outputs.
///
/// Construct with [`new_sparse`](PartialFft::new_sparse) (arbitrary bins) or
/// [`new_prefix`](PartialFft::new_prefix) (first M bins), then call
/// [`execute`](PartialFft::execute).
pub struct PartialFft<T: Float> {
    /// Transform size (number of input samples).
    n: usize,
    /// The concrete algorithm to use.
    strategy: PartialStrategy,
    _phantom: core::marker::PhantomData<T>,
}

impl<T: Float> PartialFft<T> {
    /// Construct a `PartialFft` for computing arbitrary sparse bins.
    ///
    /// When `K < log₂(N)`, uses Goertzel.
    /// Otherwise falls back to a full FFT then slice (if bins form a contiguous range)
    /// or Goertzel (for non-contiguous sets where no range is possible).
    ///
    /// # Arguments
    ///
    /// * `n`    – Transform size.
    /// * `bins` – The output bins to compute. May be in any order.
    pub fn new_sparse(n: usize, bins: &[usize]) -> Self {
        let strategy = choose_strategy_sparse(n, bins);
        Self {
            n,
            strategy,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Construct a `PartialFft` for computing the first `m` output bins.
    ///
    /// Strategy selection:
    /// - If `m` is a power-of-two, `m <= n/2`, and `m < log₂(N)`: `OutputPruned`.
    /// - If `m < log₂(N)`: `Goertzel` on bins 0..m.
    /// - Otherwise: `FullThenSlice { range: 0..m }`.
    ///
    /// # Arguments
    ///
    /// * `n` – Transform size.
    /// * `m` – Number of prefix outputs needed.
    pub fn new_prefix(n: usize, m: usize) -> Self {
        let strategy = choose_strategy_prefix(n, m);
        Self {
            n,
            strategy,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Expose the chosen strategy (useful for testing and diagnostics).
    #[must_use]
    pub fn strategy(&self) -> &PartialStrategy {
        &self.strategy
    }

    /// Execute the partial FFT.
    ///
    /// # Arguments
    ///
    /// * `input`  – Input samples; length must equal `n` used at construction.
    /// * `output` – Output buffer; length must match the number of bins requested:
    ///   `K` for sparse, `M` for prefix.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != n` or `output.len()` does not match the expected count.
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        assert_eq!(
            input.len(),
            self.n,
            "PartialFft::execute: input length {} != n {}",
            input.len(),
            self.n
        );

        match &self.strategy {
            PartialStrategy::Goertzel { bins } => {
                assert_eq!(
                    output.len(),
                    bins.len(),
                    "PartialFft::execute: output.len() must equal bins.len()"
                );
                execute_goertzel(input, bins, output);
            }
            PartialStrategy::OutputPruned { m } => {
                assert_eq!(
                    output.len(),
                    *m,
                    "PartialFft::execute: output.len() must equal m"
                );
                execute_output_pruned(input, *m, output);
            }
            PartialStrategy::FullThenSlice { range } => {
                let len = range.end - range.start;
                assert_eq!(
                    output.len(),
                    len,
                    "PartialFft::execute: output.len() must equal range length"
                );
                execute_full_then_slice(input, range, output);
            }
        }
    }
}

// ─── Strategy selection ──────────────────────────────────────────────────────

/// Choose strategy for sparse (arbitrary) bins.
fn choose_strategy_sparse(n: usize, bins: &[usize]) -> PartialStrategy {
    let k = bins.len();
    let log_n = log2_ceil(n);

    // Goertzel costs K·N; full FFT costs N·log₂(N).
    // Use Goertzel when K < log₂(N).  For non-contiguous bins we always use
    // Goertzel because FullThenSlice requires a contiguous range.
    if k < log_n {
        PartialStrategy::Goertzel {
            bins: bins.to_vec(),
        }
    } else {
        // K >= log₂(N): try FullThenSlice if bins form a contiguous range,
        // otherwise stay with Goertzel (no other contiguous representation).
        if let Some(range) = bins_as_range(bins) {
            PartialStrategy::FullThenSlice { range }
        } else {
            PartialStrategy::Goertzel {
                bins: bins.to_vec(),
            }
        }
    }
}

/// Choose strategy for prefix (first m) bins.
fn choose_strategy_prefix(n: usize, m: usize) -> PartialStrategy {
    let log_n = log2_ceil(n);

    // OutputPruned: m is power-of-two, m <= n/2, and Goertzel on 0..m is
    // cheaper than full FFT (m < log₂(N)).
    if m.is_power_of_two() && m <= n / 2 && m < log_n {
        return PartialStrategy::OutputPruned { m };
    }

    // Plain Goertzel for prefix when m < log₂(N) (but not power-of-two).
    if m < log_n {
        return PartialStrategy::Goertzel {
            bins: (0..m).collect(),
        };
    }

    // Fallback: full FFT then slice.
    PartialStrategy::FullThenSlice { range: 0..m }
}

/// If `bins` is a sorted, contiguous, unit-stride slice, return it as a `Range`.
fn bins_as_range(bins: &[usize]) -> Option<Range<usize>> {
    if bins.is_empty() {
        return Some(0..0);
    }
    let start = bins[0];
    for (i, &b) in bins.iter().enumerate() {
        if b != start + i {
            return None;
        }
    }
    Some(start..start + bins.len())
}

// ─── Execution helpers ───────────────────────────────────────────────────────

/// Delegate Goertzel computation to the existing (tested) implementation.
fn execute_goertzel<T: Float>(input: &[Complex<T>], bins: &[usize], output: &mut [Complex<T>]) {
    let results = super::goertzel_multi(input, bins);
    output.copy_from_slice(&results);
}

/// Output-pruned: Goertzel on bins 0..m.
fn execute_output_pruned<T: Float>(input: &[Complex<T>], m: usize, output: &mut [Complex<T>]) {
    let bins: Vec<usize> = (0..m).collect();
    let results = super::goertzel_multi(input, &bins);
    output.copy_from_slice(&results);
}

/// Full FFT then copy the requested range into output.
fn execute_full_then_slice<T: Float>(
    input: &[Complex<T>],
    range: &Range<usize>,
    output: &mut [Complex<T>],
) {
    let n = input.len();
    let plan = match Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => {
            for o in output.iter_mut() {
                *o = Complex::<T>::zero();
            }
            return;
        }
    };

    let mut full_output = vec![Complex::<T>::zero(); n];
    plan.execute(input, &mut full_output);

    let len = range.end - range.start;
    output[..len].copy_from_slice(&full_output[range.clone()]);
}

// ─── Utility ─────────────────────────────────────────────────────────────────

/// Ceiling of log₂(n). Returns 0 for n ≤ 1.
fn log2_ceil(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let bits = usize::BITS as usize;
    bits - (n - 1).leading_zeros() as usize
}
