//! Core FFT plan types (Plan, Plan2D, Plan3D).
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#![allow(clippy::items_after_statements)] // reason: SplitRS-generated code places type defs and constants after use statements
#![allow(clippy::manual_contains)] // reason: hand-written range checks are clearer than `.contains()` for multi-variant FFT size dispatching

use crate::api::{Direction, Flags};
use crate::dft::codelets::twiddle_odd::{
    tw16_dit_bwd, tw16_dit_fwd, tw2_dit_bwd, tw2_dit_fwd, tw3_dit_bwd, tw3_dit_fwd, tw4_dit_bwd,
    tw4_dit_fwd, tw5_dit_bwd, tw5_dit_fwd, tw7_dit_bwd, tw7_dit_fwd, tw8_dit_bwd, tw8_dit_fwd,
};
use crate::dft::problem::Sign;
use crate::dft::solvers::{
    BluesteinSolver, CooleyTukeySolver, CtVariant, DirectSolver, GenericSolver, NopSolver,
    StockhamSolver,
};
use crate::kernel::{twiddles_mixed_radix, Complex, Float, TwiddleDirection};
use crate::prelude::*;

#[cfg(feature = "threading")]
use crate::threading::WorkStealingContext;

use super::types_real::RealPlan;

/// Transform kind for real FFT plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RealPlanKind {
    /// Real to Complex (forward)
    R2C,
    /// Complex to Real (backward/inverse)
    C2R,
}
/// Algorithm selection for the plan.
#[allow(dead_code)] // reason: enum variants represent all possible solver strategies; not all are constructed in every build configuration
enum Algorithm<T: Float> {
    /// No-op for size 0 or 1
    Nop,
    /// Direct O(n²) computation (only for very small sizes where overhead matters)
    Direct,
    /// Cooley-Tukey radix-2 FFT
    CooleyTukey(CtVariant),
    /// Stockham auto-sort FFT (avoids bit-reversal, good for large sizes)
    Stockham,
    /// Specialized composite codelets (12, 24, 36, 48, 60, 72, 96, 100)
    Composite(usize),
    /// Generic mixed-radix for composite sizes
    Generic(Box<GenericSolver<T>>),
    /// Bluestein's algorithm for arbitrary sizes (fallback for primes)
    Bluestein(Box<BluesteinSolver<T>>),
    /// Winograd small-prime kernel for N ∈ {3, 5, 7, 9, 11, 13}
    Winograd(usize),
    /// Prime Factor Algorithm (PFA) for coprime composites: N ∈ {15, 21, 35}
    WinogradPfa { n1: usize, n2: usize },
    /// Mixed-radix FFT for composite sizes (pre-wave stub)
    MixedRadix { factors: Vec<u16> },
}
// ============================================================================
// Mixed-radix helpers
// ============================================================================

/// Supported radices for the mixed-radix DIT engine, in preference order
/// (largest first so the greedy peel produces the fewest stages).
const MIXED_RADIX_SUPPORTED: &[u16] = &[16, 8, 7, 5, 4, 3, 2];

/// Try to factor `n` as a product of supported radices {2, 3, 4, 5, 7, 8, 16}.
///
/// Returns `Some(factors)` ordered innermost-first (the first element is the
/// innermost radix applied in the DIT pipeline), or `None` if `n` cannot be
/// expressed using only the supported radices.
///
/// The greedy strategy peels the largest supported radix from the remaining
/// value at each step; this minimises the number of stages and tends to favour
/// SIMD-friendly butterflies.
fn try_factor_mixed_radix(n: usize) -> Option<Vec<u16>> {
    if n <= 1 {
        return None; // size 0/1 handled by Nop
    }
    let mut remaining = n;
    let mut factors: Vec<u16> = Vec::new();

    'outer: while remaining > 1 {
        for &r in MIXED_RADIX_SUPPORTED {
            if remaining % r as usize == 0 {
                factors.push(r);
                remaining /= r as usize;
                continue 'outer;
            }
        }
        // No supported radix divides `remaining`
        return None;
    }

    // Return factors in innermost-first order.
    // The greedy loop produces them in outermost-first (largest peeled first),
    // so we reverse.
    factors.reverse();
    Some(factors)
}

/// Apply the mixed-radix digit-reversal (scatter) permutation in-place.
///
/// For DIT with `factors` ordered innermost-first, consecutive blocks of
/// `radix[0]` elements in the permuted array feed the first-stage butterfly.
/// The permutation maps output index `k` to input index `digit_reverse(k)`.
///
/// The scatter is computed iteratively: for each stage factor (from outermost
/// to innermost), we stride through the original indices to interleave.
fn mixed_radix_digit_rev_permute<T: Float>(data: &mut [Complex<T>], factors: &[u16]) {
    let n = data.len();
    if n <= 1 || factors.len() <= 1 {
        return;
    }

    // Build the permutation table by iteratively "unzipping" from outermost stage
    // (last in innermost-first array) down to innermost.
    let num_stages = factors.len();
    let mut perm: Vec<usize> = (0..n).collect();

    let mut stride = n;
    for stage in (0..num_stages).rev() {
        let r = factors[stage] as usize;
        stride /= r;

        // Reorder perm: interleave by stride so that consecutive groups of `r`
        // are the input samples to each butterfly in this stage.
        let mut new_perm = vec![0usize; n];
        let blocks = n / (stride * r); // number of independent groups at this stage
        for b in 0..blocks {
            for j in 0..r {
                for s in 0..stride {
                    // Output position: b*r*stride + j*stride + s
                    // Source position: b*r*stride + s*r + j
                    let dst = b * r * stride + j * stride + s;
                    let src = b * r * stride + s * r + j;
                    new_perm[dst] = perm[src];
                }
            }
        }
        perm = new_perm;
    }

    // Apply the permutation using a cycle-following algorithm (avoids scratch alloc).
    let mut visited = vec![false; n];
    for start in 0..n {
        if visited[start] || perm[start] == start {
            visited[start] = true;
            continue;
        }
        // Follow the cycle
        let first = data[start];
        let mut cur = start;
        loop {
            let next = perm[cur];
            visited[cur] = true;
            if next == start {
                data[cur] = first;
                break;
            }
            data[cur] = data[next];
            cur = next;
        }
    }
}

/// Execute the mixed-radix DIT FFT in-place on `data`.
///
/// `factors` is ordered innermost-first. After digit-reversal permutation,
/// each stage applies the corresponding twiddle butterfly.
fn execute_mixed_radix_inplace<T: Float>(
    data: &mut [Complex<T>],
    factors: &[u16],
    direction: TwiddleDirection,
) {
    let n = data.len();
    // Step 1: digit-reversal permutation
    mixed_radix_digit_rev_permute(data, factors);

    // Step 2: generate twiddle tables
    // We use f64 tables and convert, since T may be f32 or f64.
    // The per-stage tables are indexed as: table[(j-1)*stride + s]
    let tables_f64 = twiddles_mixed_radix(n, factors, direction);

    // Step 3: apply butterfly stages
    let mut current_n: usize = 1;
    for (stage, (&r_u16, table_f64)) in factors.iter().zip(tables_f64.iter()).enumerate() {
        let r = r_u16 as usize;
        current_n *= r;
        let stride = current_n / r;
        let blocks = n / current_n;

        // Convert f64 twiddles to T
        let table: Vec<Complex<T>> = table_f64
            .iter()
            .map(|&w| Complex::new(T::from_f64(w.re), T::from_f64(w.im)))
            .collect();

        let _ = stage; // suppress unused warning

        match (r, direction) {
            (2, TwiddleDirection::Forward) => tw2_dit_fwd(data, &table, stride, blocks),
            (2, TwiddleDirection::Inverse) => tw2_dit_bwd(data, &table, stride, blocks),
            (3, TwiddleDirection::Forward) => tw3_dit_fwd(data, &table, stride, blocks),
            (3, TwiddleDirection::Inverse) => tw3_dit_bwd(data, &table, stride, blocks),
            (4, TwiddleDirection::Forward) => tw4_dit_fwd(data, &table, stride, blocks),
            (4, TwiddleDirection::Inverse) => tw4_dit_bwd(data, &table, stride, blocks),
            (5, TwiddleDirection::Forward) => tw5_dit_fwd(data, &table, stride, blocks),
            (5, TwiddleDirection::Inverse) => tw5_dit_bwd(data, &table, stride, blocks),
            (7, TwiddleDirection::Forward) => tw7_dit_fwd(data, &table, stride, blocks),
            (7, TwiddleDirection::Inverse) => tw7_dit_bwd(data, &table, stride, blocks),
            (8, TwiddleDirection::Forward) => tw8_dit_fwd(data, &table, stride, blocks),
            (8, TwiddleDirection::Inverse) => tw8_dit_bwd(data, &table, stride, blocks),
            (16, TwiddleDirection::Forward) => tw16_dit_fwd(data, &table, stride, blocks),
            (16, TwiddleDirection::Inverse) => tw16_dit_bwd(data, &table, stride, blocks),
            _ => unreachable!(
                "execute_mixed_radix_inplace: unsupported radix {r} — only {{2,3,4,5,7,8,16}} supported"
            ),
        }
    }
}

/// A plan for executing FFT transforms.
///
/// Plans are created once and can be executed multiple times.
/// The planning process may measure different algorithms to find the fastest.
pub struct Plan<T: Float> {
    /// Transform size
    n: usize,
    /// Transform direction
    direction: Direction,
    /// Selected algorithm
    algorithm: Algorithm<T>,
}
// ─── Solver-name → Algorithm reconstruction ───────────────────────────────────

/// Attempt to reconstruct an [`Algorithm<T>`] from a stored solver name string.
///
/// Returns `None` when the name is unrecognised or the algorithm requires
/// state that cannot be reconstructed from the name alone (e.g. `Generic`
/// needs a `GenericSolver` which stores pre-computed twiddle tables).  The
/// caller should fall back to heuristic planning in that case.
fn algorithm_from_solver_name<T: Float>(name: &str, n: usize) -> Option<Algorithm<T>> {
    use crate::dft::solvers::CtVariant;

    match name {
        "nop" => Some(Algorithm::Nop),
        "direct" => Some(Algorithm::Direct),
        "ct-dit" => Some(Algorithm::CooleyTukey(CtVariant::Dit)),
        "ct-dif" => Some(Algorithm::CooleyTukey(CtVariant::Dif)),
        "ct-radix4" => Some(Algorithm::CooleyTukey(CtVariant::DitRadix4)),
        "ct-radix8" => Some(Algorithm::CooleyTukey(CtVariant::DitRadix8)),
        "ct-splitradix" => Some(Algorithm::CooleyTukey(CtVariant::SplitRadix)),
        "stockham" => Some(Algorithm::Stockham),
        "composite" if crate::dft::codelets::has_composite_codelet(n) => {
            Some(Algorithm::Composite(n))
        }
        "winograd" if matches!(n, 3 | 5 | 7 | 9 | 11 | 13) => Some(Algorithm::Winograd(n)),
        "winograd-pfa" if matches!(n, 15 | 21 | 35) => {
            let (n1, n2) = match n {
                15 => (3, 5),
                21 => (3, 7),
                35 => (5, 7),
                _ => return None,
            };
            Some(Algorithm::WinogradPfa { n1, n2 })
        }
        name if name.starts_with("mixed-radix-") => {
            let suffix = &name["mixed-radix-".len()..];
            let factors: Vec<u16> = suffix
                .split('-')
                .filter_map(|s| s.parse::<u16>().ok())
                .collect();
            if factors.is_empty() {
                None
            } else {
                Some(Algorithm::MixedRadix { factors })
            }
        }
        // Stateful solvers — cannot reconstruct from name alone, fall through.
        "bluestein" | "generic" | "rader" | "cache-oblivious" => None,
        _ => None,
    }
}

// ─── Baseline wisdom (compile-time binary blob from build.rs) ────────────────

/// Static cache loaded from the compile-time wisdom baseline binary.
///
/// The build script writes `$OUT_DIR/wisdom_baseline.bin` (possibly empty).
/// At runtime the bytes are decoded once and stored here.  If the file is
/// empty or malformed, this stays as an empty cache and the heuristic path
/// is used instead.
#[cfg(feature = "std")]
static BASELINE_WISDOM: std::sync::OnceLock<crate::api::wisdom::WisdomCache> =
    std::sync::OnceLock::new();

/// Return a reference to the compile-time wisdom baseline, initialising it
/// lazily on first call.
#[cfg(feature = "std")]
fn baseline_wisdom() -> &'static crate::api::wisdom::WisdomCache {
    BASELINE_WISDOM.get_or_init(|| {
        // The build script always writes this file (even 0 bytes) so that
        // `include_bytes!` does not fail at compile time.
        const BASELINE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/wisdom_baseline.bin"));
        if BASELINE.is_empty() {
            crate::api::wisdom::WisdomCache::new()
        } else {
            crate::api::wisdom::WisdomCache::from_binary(BASELINE).unwrap_or_default()
        }
    })
}

// ─────────────────────────────────────────────────────────────────────────────

impl<T: Float> Plan<T> {
    /// Create a 1D complex-to-complex DFT plan.
    ///
    /// # Arguments
    /// * `n` - Transform size
    /// * `direction` - Forward or Backward transform
    /// * `flags` - Planning flags (ESTIMATE, MEASURE, PATIENT, EXHAUSTIVE)
    ///
    /// # Returns
    /// A plan that can be executed on input/output buffers of size `n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxifft::{Complex, Direction, Flags, Plan};
    ///
    /// let plan = Plan::<f64>::dft_1d(16, Direction::Forward, Flags::ESTIMATE)
    ///     .expect("plan construction failed");
    /// let input: Vec<Complex<f64>> = (0..16)
    ///     .map(|i| Complex::new(i as f64, 0.0))
    ///     .collect();
    /// let mut output = vec![Complex::<f64>::zero(); 16];
    /// plan.execute(&input, &mut output);
    /// // DC bin is the sum of inputs: 0+1+...+15 = 120
    /// assert!((output[0].re - 120.0_f64).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn dft_1d(n: usize, direction: Direction, flags: Flags) -> Option<Self> {
        // ── 1. Check baseline wisdom (populated by build.rs when OXIFFT_TUNE=1) ──
        //
        // The baseline cache keys by raw size (`n as u64`).  When a match is
        // found the stored `solver_name` is used to reconstruct the algorithm.
        // Currently we fall through to the heuristic if reconstruction is not
        // possible; this keeps the code correct even with future unknown solver
        // names.
        #[cfg(feature = "std")]
        if let Some(entry) = baseline_wisdom().lookup(n as u64) {
            if let Some(algo) = algorithm_from_solver_name::<T>(&entry.solver_name, n) {
                return Some(Self {
                    n,
                    direction,
                    algorithm: algo,
                });
            }
        }

        // ── 2. MEASURE / PATIENT: profile the heuristic and store timing ──────
        //
        // We run the tuner (which internally always uses ESTIMATE so there is
        // no re-entry), collect the timing result, and then fall through to
        // construct the plan via the normal heuristic path.  In a future wave
        // this could pick among several candidates; for now the value is that
        // the result is written into the global wisdom cache so that
        // export_to_string() reflects real timings.
        #[cfg(feature = "std")]
        if flags.is_measure() || flags.is_patient() {
            let reps: usize = if flags.is_patient() { 200 } else { 32 };
            if let Some(result) = crate::api::plan::auto_tune::tune_size::<T>(n, direction, reps) {
                // Store the timing in the global wisdom cache for later export.
                crate::api::wisdom::store_wisdom(crate::kernel::WisdomEntry {
                    problem_hash: n as u64,
                    solver_name: result.algorithm_name,
                    cost: result.elapsed_ns as f64,
                });
            }
            // Fall through to heuristic — the tuner chose the same algorithm.
        }

        // ── 3. Heuristic selection (ESTIMATE path and fallback) ───────────────
        let algorithm = Self::select_algorithm(n, flags);
        Some(Self {
            n,
            direction,
            algorithm,
        })
    }
    /// Create a 2D complex-to-complex DFT plan.
    #[must_use]
    pub fn dft_2d(n0: usize, n1: usize, direction: Direction, flags: Flags) -> Option<Plan2D<T>> {
        Plan2D::new(n0, n1, direction, flags)
    }
    /// Create a 3D complex-to-complex DFT plan.
    #[must_use]
    pub fn dft_3d(
        n0: usize,
        n1: usize,
        n2: usize,
        direction: Direction,
        flags: Flags,
    ) -> Option<Plan3D<T>> {
        Plan3D::new(n0, n1, n2, direction, flags)
    }
    /// Create a 1D real-to-complex FFT plan.
    #[must_use]
    pub fn r2c_1d(n: usize, flags: Flags) -> Option<RealPlan<T>> {
        RealPlan::r2c_1d(n, flags)
    }
    /// Create a 1D complex-to-real FFT plan.
    #[must_use]
    pub fn c2r_1d(n: usize, flags: Flags) -> Option<RealPlan<T>> {
        RealPlan::c2r_1d(n, flags)
    }
    /// Select the best algorithm for the given size.
    fn select_algorithm(n: usize, _flags: Flags) -> Algorithm<T> {
        use crate::dft::codelets::has_composite_codelet;

        if n <= 1 {
            Algorithm::Nop
        } else if CooleyTukeySolver::<T>::applicable(n) {
            // Use DIT with SIMD-accelerated butterflies for all power-of-2 sizes
            // Note: Stockham needs optimization before it can compete with DIT+codelets
            Algorithm::CooleyTukey(CtVariant::Dit)
        } else if matches!(n, 3 | 5 | 7 | 9 | 11 | 13) {
            // Winograd symmetric-pair kernels for odd prime and prime-power sizes
            Algorithm::Winograd(n)
        } else if matches!(n, 15 | 21 | 35) {
            // Prime Factor Algorithm for coprime composites: 15=3×5, 21=3×7, 35=5×7
            let (n1, n2) = match n {
                15 => (3, 5),
                21 => (3, 7),
                35 => (5, 7),
                _ => unreachable!(),
            };
            Algorithm::WinogradPfa { n1, n2 }
        } else if has_composite_codelet(n) {
            // Use specialized composite codelets for common sizes (12, 24, 36, 48, 60, 72, 96, 100)
            Algorithm::Composite(n)
        } else if let Some(factors) = try_factor_mixed_radix(n) {
            // Mixed-radix DIT FFT for smooth-7 sizes (factors from {2,3,4,5,7,8,16}).
            // Checked before the Direct fallback so that sizes like 6=3×2, 10=5×2
            // use the correct DIT engine rather than O(n²) direct computation.
            Algorithm::MixedRadix { factors }
        } else if n <= 16 {
            // For small non-power-of-2, non-smooth sizes without codelets, use direct O(n²).
            // (Only primes like 11, 13 land here since smooth composites are handled above.)
            Algorithm::Direct
        } else if GenericSolver::<T>::applicable(n) {
            Algorithm::Generic(Box::new(GenericSolver::new(n)))
        } else {
            Algorithm::Bluestein(Box::new(BluesteinSolver::new(n)))
        }
    }
    /// Get the transform size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.n
    }
    /// Get the transform direction.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.direction
    }
    /// Return a human-readable name for the selected algorithm.
    #[must_use]
    pub fn algorithm_name(&self) -> &'static str {
        match &self.algorithm {
            Algorithm::Nop => "Nop",
            Algorithm::Direct => "Direct",
            Algorithm::CooleyTukey(v) => match v {
                CtVariant::Dit => "CooleyTukey(Dit)",
                CtVariant::Dif => "CooleyTukey(Dif)",
                CtVariant::DitRadix4 => "CooleyTukey(DitRadix4)",
                CtVariant::DitRadix8 => "CooleyTukey(DitRadix8)",
                CtVariant::SplitRadix => "CooleyTukey(SplitRadix)",
            },
            Algorithm::Stockham => "Stockham",
            Algorithm::Composite(_) => "Composite",
            Algorithm::Generic(_) => "Generic",
            Algorithm::Bluestein(_) => "Bluestein",
            Algorithm::Winograd(_) => "Winograd",
            Algorithm::WinogradPfa { .. } => "WinogradPfa",
            Algorithm::MixedRadix { .. } => "MixedRadix",
        }
    }

    /// Return the canonical wisdom-format solver name for the selected algorithm.
    ///
    /// This is the name stored in wisdom files and used for round-trip
    /// deserialization via `algorithm_from_solver_name`.  It differs from
    /// `algorithm_name` which returns a human-readable display string.
    ///
    /// For `MixedRadix` the factors (innermost-first) are embedded as
    /// `"mixed-radix-R1-R2-..."`, which is the format expected by
    /// `algorithm_from_solver_name`.
    #[must_use]
    pub fn wisdom_solver_name(&self) -> String {
        match &self.algorithm {
            Algorithm::Nop => "nop".to_string(),
            Algorithm::Direct => "direct".to_string(),
            Algorithm::CooleyTukey(v) => match v {
                CtVariant::Dit => "ct-dit".to_string(),
                CtVariant::Dif => "ct-dif".to_string(),
                CtVariant::DitRadix4 => "ct-radix4".to_string(),
                CtVariant::DitRadix8 => "ct-radix8".to_string(),
                CtVariant::SplitRadix => "ct-splitradix".to_string(),
            },
            Algorithm::Stockham => "stockham".to_string(),
            Algorithm::Composite(_) => "composite".to_string(),
            Algorithm::Generic(_) => "generic".to_string(),
            Algorithm::Bluestein(_) => "bluestein".to_string(),
            Algorithm::Winograd(_) => "winograd".to_string(),
            Algorithm::WinogradPfa { .. } => "winograd-pfa".to_string(),
            Algorithm::MixedRadix { factors } => {
                let parts: Vec<String> = factors.iter().map(|r| r.to_string()).collect();
                format!("mixed-radix-{}", parts.join("-"))
            }
        }
    }

    /// Execute the plan on the given input/output buffers.
    ///
    /// # Panics
    /// Panics if input or output buffer sizes don't match the plan size.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxifft::{Complex, Direction, Flags, Plan};
    ///
    /// let plan = Plan::<f64>::dft_1d(8, Direction::Forward, Flags::ESTIMATE).unwrap();
    /// let input: Vec<Complex<f64>> = (0..8).map(|i| Complex::new(i as f64, 0.0)).collect();
    /// let mut output = vec![Complex::<f64>::zero(); 8];
    /// plan.execute(&input, &mut output);
    /// // DC bin = sum of 0+1+...+7 = 28
    /// assert!((output[0].re - 28.0_f64).abs() < 1e-9);
    /// ```
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        use crate::dft::codelets::execute_composite_codelet;

        assert_eq!(input.len(), self.n, "Input size must match plan size");
        assert_eq!(output.len(), self.n, "Output size must match plan size");
        let sign = match self.direction {
            Direction::Forward => Sign::Forward,
            Direction::Backward => Sign::Backward,
        };
        match &self.algorithm {
            Algorithm::Nop => {
                NopSolver::new().execute(input, output);
            }
            Algorithm::Direct => {
                DirectSolver::new().execute(input, output, sign);
            }
            Algorithm::CooleyTukey(variant) => {
                CooleyTukeySolver::new(*variant).execute(input, output, sign);
            }
            Algorithm::Stockham => {
                StockhamSolver::new().execute(input, output, sign);
            }
            Algorithm::Composite(n) => {
                output.copy_from_slice(input);
                let sign_int = if sign == Sign::Forward { -1 } else { 1 };
                execute_composite_codelet(output, *n, sign_int);
            }
            Algorithm::Generic(solver) => {
                solver.execute(input, output, sign);
            }
            Algorithm::Bluestein(solver) => {
                solver.execute(input, output, sign);
            }
            Algorithm::Winograd(n) => {
                use crate::dft::codelets::winograd::{
                    winograd_11, winograd_13, winograd_3, winograd_5, winograd_7, winograd_9,
                };
                let sign_int = if sign == Sign::Forward { -1 } else { 1 };
                output.copy_from_slice(input);
                match n {
                    3 => winograd_3(output, sign_int),
                    5 => winograd_5(output, sign_int),
                    7 => winograd_7(output, sign_int),
                    9 => winograd_9(output, sign_int),
                    11 => winograd_11(output, sign_int),
                    13 => winograd_13(output, sign_int),
                    _ => unreachable!(),
                }
            }
            Algorithm::WinogradPfa { n1, n2 } => {
                use crate::dft::codelets::winograd::{winograd_3, winograd_5, winograd_7};
                use crate::dft::codelets::winograd_pfa::pfa_compose;
                let sign_int = if sign == Sign::Forward { -1 } else { 1 };
                let (k1, k2) = (*n1, *n2);
                match (k1, k2) {
                    (3, 5) => pfa_compose(input, output, 3, 5, winograd_3, winograd_5, sign_int),
                    (3, 7) => pfa_compose(input, output, 3, 7, winograd_3, winograd_7, sign_int),
                    (5, 7) => pfa_compose(input, output, 5, 7, winograd_5, winograd_7, sign_int),
                    _ => unreachable!(),
                }
            }
            Algorithm::MixedRadix { factors } => {
                let dir = match sign {
                    Sign::Forward => TwiddleDirection::Forward,
                    Sign::Backward => TwiddleDirection::Inverse,
                };
                output.copy_from_slice(input);
                execute_mixed_radix_inplace(output, factors, dir);
            }
        }
    }
    /// Execute the plan in-place.
    ///
    /// # Panics
    /// Panics if buffer size doesn't match the plan size.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxifft::{Complex, Direction, Flags, Plan};
    ///
    /// let plan = Plan::<f64>::dft_1d(8, Direction::Forward, Flags::ESTIMATE).unwrap();
    /// let mut data: Vec<Complex<f64>> = (0..8).map(|i| Complex::new(i as f64, 0.0)).collect();
    /// plan.execute_inplace(&mut data);
    /// // DC bin = sum of 0+1+...+7 = 28
    /// assert!((data[0].re - 28.0_f64).abs() < 1e-9);
    /// ```
    pub fn execute_inplace(&self, data: &mut [Complex<T>]) {
        use crate::dft::codelets::execute_composite_codelet;

        assert_eq!(data.len(), self.n, "Data size must match plan size");
        let sign = match self.direction {
            Direction::Forward => Sign::Forward,
            Direction::Backward => Sign::Backward,
        };
        match &self.algorithm {
            Algorithm::Nop => {
                NopSolver::new().execute_inplace(data);
            }
            Algorithm::Direct => {
                DirectSolver::new().execute_inplace(data, sign);
            }
            Algorithm::CooleyTukey(variant) => {
                CooleyTukeySolver::new(*variant).execute_inplace(data, sign);
            }
            Algorithm::Stockham => {
                // Stockham is out-of-place, use temp buffer and copy back
                let input = data.to_vec();
                StockhamSolver::new().execute(&input, data, sign);
            }
            Algorithm::Composite(n) => {
                let sign_int = if sign == Sign::Forward { -1 } else { 1 };
                execute_composite_codelet(data, *n, sign_int);
            }
            Algorithm::Generic(solver) => {
                solver.execute_inplace(data, sign);
            }
            Algorithm::Bluestein(solver) => {
                solver.execute_inplace(data, sign);
            }
            Algorithm::Winograd(n) => {
                use crate::dft::codelets::winograd::{
                    winograd_11, winograd_13, winograd_3, winograd_5, winograd_7, winograd_9,
                };
                let sign_int = if sign == Sign::Forward { -1 } else { 1 };
                match n {
                    3 => winograd_3(data, sign_int),
                    5 => winograd_5(data, sign_int),
                    7 => winograd_7(data, sign_int),
                    9 => winograd_9(data, sign_int),
                    11 => winograd_11(data, sign_int),
                    13 => winograd_13(data, sign_int),
                    _ => unreachable!(),
                }
            }
            Algorithm::WinogradPfa { n1, n2 } => {
                use crate::dft::codelets::winograd::{winograd_3, winograd_5, winograd_7};
                use crate::dft::codelets::winograd_pfa::pfa_compose;
                let sign_int = if sign == Sign::Forward { -1 } else { 1 };
                let (k1, k2) = (*n1, *n2);
                let input = data.to_vec();
                match (k1, k2) {
                    (3, 5) => pfa_compose(&input, data, 3, 5, winograd_3, winograd_5, sign_int),
                    (3, 7) => pfa_compose(&input, data, 3, 7, winograd_3, winograd_7, sign_int),
                    (5, 7) => pfa_compose(&input, data, 5, 7, winograd_5, winograd_7, sign_int),
                    _ => unreachable!(),
                }
            }
            Algorithm::MixedRadix { factors } => {
                let dir = match sign {
                    Sign::Forward => TwiddleDirection::Forward,
                    Sign::Backward => TwiddleDirection::Inverse,
                };
                execute_mixed_radix_inplace(data, factors, dir);
            }
        }
    }
}
/// A plan for executing 2D FFT transforms.
///
/// Implements row-column decomposition: apply 1D FFT to all rows,
/// then to all columns.  When the `threading` feature is enabled,
/// row and column passes are executed in parallel using rayon's
/// work-stealing scheduler.  Supply a custom pool via
/// [`Plan2D::with_rayon_pool`] to isolate FFT work from other rayon
/// tasks in the same process.
pub struct Plan2D<T: Float> {
    /// Number of rows
    n0: usize,
    /// Number of columns
    n1: usize,
    /// Transform direction
    direction: Direction,
    /// 1D plan for rows (size n1)
    row_plan: Plan<T>,
    /// 1D plan for columns (size n0)
    col_plan: Plan<T>,
    /// Work-stealing context for parallel row/column transforms.
    #[cfg(feature = "threading")]
    ws: WorkStealingContext,
}
impl<T: Float> Plan2D<T> {
    /// Create a 2D complex-to-complex DFT plan.
    ///
    /// # Arguments
    /// * `n0` - Number of rows
    /// * `n1` - Number of columns
    /// * `direction` - Forward or Backward transform
    /// * `flags` - Planning flags
    ///
    /// # Returns
    /// A plan that can be executed on row-major input/output buffers of size n0 × n1.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxifft::{Complex, Direction, Flags, Plan2D};
    ///
    /// // 4×4 2D forward FFT
    /// let plan = Plan2D::<f64>::new(4, 4, Direction::Forward, Flags::ESTIMATE)
    ///     .expect("plan construction failed");
    /// // All-ones input: DC bin = 16
    /// let input = vec![Complex::<f64>::new(1.0, 0.0); 16];
    /// let mut output = vec![Complex::<f64>::zero(); 16];
    /// plan.execute(&input, &mut output);
    /// // DC bin (index 0) = sum of all 16 elements
    /// assert!((output[0].re - 16.0_f64).abs() < 1e-9);
    /// // All non-DC bins should be zero for constant input
    /// assert!(output[1].re.abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn new(n0: usize, n1: usize, direction: Direction, flags: Flags) -> Option<Self> {
        let row_plan = Plan::dft_1d(n1, direction, flags)?;
        let col_plan = Plan::dft_1d(n0, direction, flags)?;
        Some(Self {
            n0,
            n1,
            direction,
            row_plan,
            col_plan,
            #[cfg(feature = "threading")]
            ws: WorkStealingContext::new(),
        })
    }

    /// Override the rayon thread pool used for parallel row/column transforms.
    ///
    /// By default `Plan2D` uses the global rayon pool.  Pass a dedicated pool
    /// here to keep FFT work isolated from other parallel tasks in the process.
    ///
    /// Requires the `threading` feature to be enabled.
    #[cfg(feature = "threading")]
    #[must_use]
    pub fn with_rayon_pool(mut self, pool: std::sync::Arc<rayon::ThreadPool>) -> Self {
        self.ws = self.ws.with_rayon_pool(pool);
        self
    }

    /// Get the number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.n0
    }
    /// Get the number of columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.n1
    }
    /// Get the total size (n0 × n1).
    #[must_use]
    pub fn size(&self) -> usize {
        self.n0 * self.n1
    }
    /// Get the transform direction.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.direction
    }
    /// Execute the 2D FFT on the given input/output buffers.
    ///
    /// Input and output are row-major: element at (i, j) is at index i*n1 + j.
    ///
    /// When the `threading` feature is enabled, row transforms are parallelised
    /// over rayon workers using work-stealing; column transforms are parallelised
    /// similarly with per-thread scratch buffers.
    ///
    /// # Panics
    /// Panics if buffer sizes don't match n0 × n1.
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        let total = self.n0 * self.n1;
        assert_eq!(input.len(), total, "Input size must match n0 × n1");
        assert_eq!(output.len(), total, "Output size must match n0 × n1");
        if total == 0 {
            return;
        }

        #[cfg(not(feature = "threading"))]
        {
            let mut temp = vec![Complex::zero(); total];
            for i in 0..self.n0 {
                let row_start = i * self.n1;
                let row_end = row_start + self.n1;
                self.row_plan
                    .execute(&input[row_start..row_end], &mut temp[row_start..row_end]);
            }
            let mut col_in = vec![Complex::zero(); self.n0];
            let mut col_out = vec![Complex::zero(); self.n0];
            for j in 0..self.n1 {
                for i in 0..self.n0 {
                    col_in[i] = temp[i * self.n1 + j];
                }
                self.col_plan.execute(&col_in, &mut col_out);
                for i in 0..self.n0 {
                    output[i * self.n1 + j] = col_out[i];
                }
            }
        }

        #[cfg(feature = "threading")]
        {
            use rayon::prelude::*;

            // Step 1: Parallel row transforms.
            // input rows → temp rows, each row is contiguous and independent.
            let mut temp = vec![Complex::<T>::zero(); total];
            // Build temp row-by-row using par_chunks_mut; each chunk is one row.
            {
                let n1 = self.n1;
                let row_plan = &self.row_plan;
                // We read from input (immutable) and write to temp (mutable, disjoint chunks).
                // Use par_chunks_mut on temp and zip with input chunks from a parallel slice.
                self.ws.install(|| {
                    temp.par_chunks_mut(n1).zip(input.par_chunks(n1)).for_each(
                        |(out_row, in_row)| {
                            row_plan.execute(in_row, out_row);
                        },
                    );
                });
            }

            // Step 2: Parallel column transforms.
            // Each column j is transformed independently using per-thread scratch buffers.
            let n0 = self.n0;
            let n1 = self.n1;
            let col_plan = &self.col_plan;
            // SAFETY: column j only reads temp[i*n1+j] and writes output[i*n1+j] for all i.
            // Each column index j is unique in the parallel iterator — no two threads access
            // the same indices, so there are no data races.
            let temp_ptr = temp.as_ptr() as usize;
            let out_ptr = output.as_mut_ptr() as usize;
            self.ws.install(|| {
                (0..n1).into_par_iter().for_each(|j| {
                    let mut col_in = vec![Complex::<T>::zero(); n0];
                    let mut col_out = vec![Complex::<T>::zero(); n0];
                    let temp_p = temp_ptr as *const Complex<T>;
                    let out_p = out_ptr as *mut Complex<T>;
                    for i in 0..n0 {
                        col_in[i] = unsafe { *temp_p.add(i * n1 + j) };
                    }
                    col_plan.execute(&col_in, &mut col_out);
                    for i in 0..n0 {
                        unsafe { *out_p.add(i * n1 + j) = col_out[i] };
                    }
                });
            });
        }
    }

    /// Execute the 2D FFT in-place.
    ///
    /// When the `threading` feature is enabled, both the row and column passes
    /// are parallelised over rayon workers.
    ///
    /// # Panics
    /// Panics if buffer size doesn't match n0 × n1.
    pub fn execute_inplace(&self, data: &mut [Complex<T>]) {
        let total = self.n0 * self.n1;
        assert_eq!(data.len(), total, "Data size must match n0 × n1");
        if total == 0 {
            return;
        }

        #[cfg(not(feature = "threading"))]
        {
            for i in 0..self.n0 {
                let row_start = i * self.n1;
                let row_end = row_start + self.n1;
                self.row_plan.execute_inplace(&mut data[row_start..row_end]);
            }
            let mut col = vec![Complex::zero(); self.n0];
            for j in 0..self.n1 {
                for i in 0..self.n0 {
                    col[i] = data[i * self.n1 + j];
                }
                self.col_plan.execute_inplace(&mut col);
                for i in 0..self.n0 {
                    data[i * self.n1 + j] = col[i];
                }
            }
        }

        #[cfg(feature = "threading")]
        {
            use rayon::prelude::*;

            // Step 1: Parallel row transforms (in-place on disjoint row chunks).
            let n1 = self.n1;
            let n0 = self.n0;
            let row_plan = &self.row_plan;
            self.ws.install(|| {
                data.par_chunks_mut(n1)
                    .for_each(|row| row_plan.execute_inplace(row));
            });

            // Step 2: Parallel column transforms with per-thread scratch buffers.
            let col_plan = &self.col_plan;
            let data_ptr = data.as_mut_ptr() as usize;
            self.ws.install(|| {
                (0..n1).into_par_iter().for_each(|j| {
                    let mut col = vec![Complex::<T>::zero(); n0];
                    // SAFETY: column j is accessed only by this thread.
                    let p = data_ptr as *mut Complex<T>;
                    for i in 0..n0 {
                        col[i] = unsafe { *p.add(i * n1 + j) };
                    }
                    col_plan.execute_inplace(&mut col);
                    for i in 0..n0 {
                        unsafe { *p.add(i * n1 + j) = col[i] };
                    }
                });
            });
        }
    }
}
/// A plan for executing 3D FFT transforms.
///
/// Implements layered decomposition: apply 2D FFT to each xy-plane,
/// then 1D FFT along z-axis.  When the `threading` feature is enabled,
/// the plane passes are executed in parallel across rayon workers using
/// work-stealing.  Supply a custom pool via [`Plan3D::with_rayon_pool`].
pub struct Plan3D<T: Float> {
    /// Dimensions (z, y, x) in row-major order
    n0: usize,
    n1: usize,
    n2: usize,
    /// Transform direction
    direction: Direction,
    /// 2D plan for each xy-plane
    plane_plan: Plan2D<T>,
    /// 1D plan for z-axis
    z_plan: Plan<T>,
    /// Work-stealing context for parallel plane transforms.
    #[cfg(feature = "threading")]
    ws: WorkStealingContext,
}
impl<T: Float> Plan3D<T> {
    /// Create a 3D complex-to-complex DFT plan.
    ///
    /// # Arguments
    /// * `n0` - Size along first axis (z/depth)
    /// * `n1` - Size along second axis (y/height)
    /// * `n2` - Size along third axis (x/width)
    /// * `direction` - Forward or Backward transform
    /// * `flags` - Planning flags
    ///
    /// # Returns
    /// A plan for row-major 3D data of size n0 × n1 × n2.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxifft::{Complex, Direction, Flags, Plan3D};
    ///
    /// // 2×2×2 3D forward FFT
    /// let plan = Plan3D::<f64>::new(2, 2, 2, Direction::Forward, Flags::ESTIMATE)
    ///     .expect("plan construction failed");
    /// // All-ones input: DC bin = 8 (total element count)
    /// let input = vec![Complex::<f64>::new(1.0, 0.0); 8];
    /// let mut output = vec![Complex::<f64>::zero(); 8];
    /// plan.execute(&input, &mut output);
    /// // DC bin = sum of all 8 elements
    /// assert!((output[0].re - 8.0_f64).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn new(
        n0: usize,
        n1: usize,
        n2: usize,
        direction: Direction,
        flags: Flags,
    ) -> Option<Self> {
        let plane_plan = Plan2D::new(n1, n2, direction, flags)?;
        let z_plan = Plan::dft_1d(n0, direction, flags)?;
        Some(Self {
            n0,
            n1,
            n2,
            direction,
            plane_plan,
            z_plan,
            #[cfg(feature = "threading")]
            ws: WorkStealingContext::new(),
        })
    }

    /// Override the rayon thread pool used for parallel plane transforms.
    ///
    /// By default `Plan3D` uses the global rayon pool.  Pass a dedicated pool
    /// here to keep FFT work isolated from other parallel tasks in the process.
    ///
    /// Requires the `threading` feature to be enabled.
    #[cfg(feature = "threading")]
    #[must_use]
    pub fn with_rayon_pool(mut self, pool: std::sync::Arc<rayon::ThreadPool>) -> Self {
        self.ws = self.ws.with_rayon_pool(pool);
        self
    }

    /// Get the total size (n0 × n1 × n2).
    #[must_use]
    pub fn size(&self) -> usize {
        self.n0 * self.n1 * self.n2
    }
    /// Get the transform direction.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.direction
    }
    /// Get first dimension (crate-internal use for Debug impl).
    #[must_use]
    pub(crate) fn dim0(&self) -> usize {
        self.n0
    }
    /// Get second dimension (crate-internal use for Debug impl).
    #[must_use]
    pub(crate) fn dim1(&self) -> usize {
        self.n1
    }
    /// Get third dimension (crate-internal use for Debug impl).
    #[must_use]
    pub(crate) fn dim2(&self) -> usize {
        self.n2
    }
    /// Execute the 3D FFT on the given input/output buffers.
    ///
    /// Data is row-major: element at (i, j, k) is at index i*n1*n2 + j*n2 + k.
    ///
    /// When the `threading` feature is enabled, 2D plane transforms are
    /// parallelised over rayon workers using work-stealing.
    ///
    /// # Panics
    /// Panics if buffer sizes don't match n0 × n1 × n2.
    pub fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        let total = self.n0 * self.n1 * self.n2;
        assert_eq!(input.len(), total, "Input size must match n0 × n1 × n2");
        assert_eq!(output.len(), total, "Output size must match n0 × n1 × n2");
        if total == 0 {
            return;
        }
        let plane_size = self.n1 * self.n2;

        #[cfg(not(feature = "threading"))]
        {
            let mut temp = vec![Complex::zero(); total];
            for i in 0..self.n0 {
                let plane_start = i * plane_size;
                let plane_end = plane_start + plane_size;
                self.plane_plan.execute(
                    &input[plane_start..plane_end],
                    &mut temp[plane_start..plane_end],
                );
            }
            let mut z_col = vec![Complex::zero(); self.n0];
            let mut z_out = vec![Complex::zero(); self.n0];
            for j in 0..self.n1 {
                for k in 0..self.n2 {
                    for i in 0..self.n0 {
                        z_col[i] = temp[i * plane_size + j * self.n2 + k];
                    }
                    self.z_plan.execute(&z_col, &mut z_out);
                    for i in 0..self.n0 {
                        output[i * plane_size + j * self.n2 + k] = z_out[i];
                    }
                }
            }
        }

        #[cfg(feature = "threading")]
        {
            use rayon::prelude::*;

            // Step 1: Parallel 2D plane transforms (each plane is independent).
            let mut temp = vec![Complex::<T>::zero(); total];
            let plane_plan = &self.plane_plan;
            self.ws.install(|| {
                temp.par_chunks_mut(plane_size)
                    .zip(input.par_chunks(plane_size))
                    .for_each(|(out_plane, in_plane)| {
                        plane_plan.execute(in_plane, out_plane);
                    });
            });

            // Step 2: Parallel z-axis transforms.
            // Each (j,k) fiber is independent.  Use raw-pointer sharing with
            // per-thread scratch buffers.
            let n0 = self.n0;
            let n1 = self.n1;
            let n2 = self.n2;
            let z_plan = &self.z_plan;
            let temp_ptr = temp.as_ptr() as usize;
            let out_ptr = output.as_mut_ptr() as usize;
            self.ws.install(|| {
                // Iterate over all (j,k) pairs in parallel.
                (0..n1 * n2).into_par_iter().for_each(|jk| {
                    let j = jk / n2;
                    let k = jk % n2;
                    let mut z_col = vec![Complex::<T>::zero(); n0];
                    let mut z_out_buf = vec![Complex::<T>::zero(); n0];
                    let temp_p = temp_ptr as *const Complex<T>;
                    let out_p = out_ptr as *mut Complex<T>;
                    for i in 0..n0 {
                        // SAFETY: each (j,k) pair is unique per thread.
                        z_col[i] = unsafe { *temp_p.add(i * plane_size + j * n2 + k) };
                    }
                    z_plan.execute(&z_col, &mut z_out_buf);
                    for i in 0..n0 {
                        unsafe { *out_p.add(i * plane_size + j * n2 + k) = z_out_buf[i] };
                    }
                });
            });
        }
    }

    /// Execute the 3D FFT in-place.
    ///
    /// When the `threading` feature is enabled, both the plane pass and the
    /// z-axis pass are parallelised over rayon workers.
    ///
    /// # Panics
    /// Panics if buffer size doesn't match n0 × n1 × n2.
    pub fn execute_inplace(&self, data: &mut [Complex<T>]) {
        let total = self.n0 * self.n1 * self.n2;
        assert_eq!(data.len(), total, "Data size must match n0 × n1 × n2");
        if total == 0 {
            return;
        }
        let plane_size = self.n1 * self.n2;

        #[cfg(not(feature = "threading"))]
        {
            for i in 0..self.n0 {
                let plane_start = i * plane_size;
                let plane_end = plane_start + plane_size;
                self.plane_plan
                    .execute_inplace(&mut data[plane_start..plane_end]);
            }
            let mut z_col = vec![Complex::zero(); self.n0];
            for j in 0..self.n1 {
                for k in 0..self.n2 {
                    for i in 0..self.n0 {
                        z_col[i] = data[i * plane_size + j * self.n2 + k];
                    }
                    self.z_plan.execute_inplace(&mut z_col);
                    for i in 0..self.n0 {
                        data[i * plane_size + j * self.n2 + k] = z_col[i];
                    }
                }
            }
        }

        #[cfg(feature = "threading")]
        {
            use rayon::prelude::*;

            // Step 1: Parallel 2D plane transforms in-place.
            let plane_plan = &self.plane_plan;
            self.ws.install(|| {
                data.par_chunks_mut(plane_size)
                    .for_each(|plane| plane_plan.execute_inplace(plane));
            });

            // Step 2: Parallel z-axis in-place transforms.
            let n0 = self.n0;
            let n1 = self.n1;
            let n2 = self.n2;
            let z_plan = &self.z_plan;
            let data_ptr = data.as_mut_ptr() as usize;
            self.ws.install(|| {
                (0..n1 * n2).into_par_iter().for_each(|jk| {
                    let j = jk / n2;
                    let k = jk % n2;
                    let mut z_col = vec![Complex::<T>::zero(); n0];
                    let p = data_ptr as *mut Complex<T>;
                    // SAFETY: each (j,k) pair is unique per thread.
                    for i in 0..n0 {
                        z_col[i] = unsafe { *p.add(i * plane_size + j * n2 + k) };
                    }
                    z_plan.execute_inplace(&mut z_col);
                    for i in 0..n0 {
                        unsafe { *p.add(i * plane_size + j * n2 + k) = z_col[i] };
                    }
                });
            });
        }
    }
}

// ============================================================================
// Tests for parallel Plan2D / Plan3D correctness
// ============================================================================

#[cfg(all(test, feature = "threading"))]
mod parallel_plan_tests {
    use super::*;

    fn make_input_2d(n0: usize, n1: usize) -> Vec<Complex<f64>> {
        let total = n0 * n1;
        (0..total)
            .map(|i| Complex::new((i as f64 * 0.017).sin(), (i as f64 * 0.031).cos()))
            .collect()
    }

    fn make_input_3d(n0: usize, n1: usize, n2: usize) -> Vec<Complex<f64>> {
        let total = n0 * n1 * n2;
        (0..total)
            .map(|i| Complex::new((i as f64 * 0.013).sin(), (i as f64 * 0.027).cos()))
            .collect()
    }

    fn complex_near(a: Complex<f64>, b: Complex<f64>, tol: f64) -> bool {
        (a.re - b.re).abs() < tol && (a.im - b.im).abs() < tol
    }

    /// Parallel vs serial correctness — 128×128 forward.
    ///
    /// Both paths must produce bit-identical output for the same input.
    #[test]
    fn test_plan2d_parallel_serial_forward_128x128() {
        let n0 = 128;
        let n1 = 128;
        let input = make_input_2d(n0, n1);
        let mut out_serial = vec![Complex::<f64>::zero(); n0 * n1];
        let mut out_parallel = vec![Complex::<f64>::zero(); n0 * n1];

        // Serial: use a 1-thread pool so rayon is effectively serialised.
        let serial_pool = std::sync::Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("serial pool"),
        );
        let plan_serial = Plan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE)
            .expect("plan_serial")
            .with_rayon_pool(serial_pool);
        plan_serial.execute(&input, &mut out_serial);

        // Parallel: global pool (default).
        let plan_parallel =
            Plan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE).expect("plan_parallel");
        plan_parallel.execute(&input, &mut out_parallel);

        for (i, (a, b)) in out_serial.iter().zip(out_parallel.iter()).enumerate() {
            assert!(
                complex_near(*a, *b, 1e-10),
                "element {i}: serial={a:?} parallel={b:?}"
            );
        }
    }

    /// Parallel vs serial correctness — 32×32 inverse (in-place).
    #[test]
    fn test_plan2d_parallel_serial_inverse_inplace_32x32() {
        let n0 = 32;
        let n1 = 32;
        let input = make_input_2d(n0, n1);

        let serial_pool = std::sync::Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("serial pool"),
        );
        let plan_serial = Plan2D::new(n0, n1, Direction::Backward, Flags::ESTIMATE)
            .expect("plan_serial")
            .with_rayon_pool(serial_pool);
        let mut out_serial = input.clone();
        plan_serial.execute_inplace(&mut out_serial);

        let plan_parallel =
            Plan2D::new(n0, n1, Direction::Backward, Flags::ESTIMATE).expect("plan_parallel");
        let mut out_parallel = input;
        plan_parallel.execute_inplace(&mut out_parallel);

        for (i, (a, b)) in out_serial.iter().zip(out_parallel.iter()).enumerate() {
            assert!(
                complex_near(*a, *b, 1e-10),
                "element {i}: serial={a:?} parallel={b:?}"
            );
        }
    }

    /// Parallel vs serial correctness — 32×32×32 forward.
    #[test]
    fn test_plan3d_parallel_serial_forward_32x32x32() {
        let n0 = 32;
        let n1 = 32;
        let n2 = 32;
        let input = make_input_3d(n0, n1, n2);
        let mut out_serial = vec![Complex::<f64>::zero(); n0 * n1 * n2];
        let mut out_parallel = vec![Complex::<f64>::zero(); n0 * n1 * n2];

        let serial_pool = std::sync::Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("serial pool"),
        );
        let plan_serial = Plan3D::new(n0, n1, n2, Direction::Forward, Flags::ESTIMATE)
            .expect("plan_serial")
            .with_rayon_pool(serial_pool);
        plan_serial.execute(&input, &mut out_serial);

        let plan_parallel =
            Plan3D::new(n0, n1, n2, Direction::Forward, Flags::ESTIMATE).expect("plan_parallel");
        plan_parallel.execute(&input, &mut out_parallel);

        for (i, (a, b)) in out_serial.iter().zip(out_parallel.iter()).enumerate() {
            assert!(
                complex_near(*a, *b, 1e-10),
                "element {i}: serial={a:?} parallel={b:?}"
            );
        }
    }

    /// Parallel vs serial correctness — 32×32×32 in-place.
    #[test]
    fn test_plan3d_parallel_serial_inplace_32x32x32() {
        let n0 = 32;
        let n1 = 32;
        let n2 = 32;
        let input = make_input_3d(n0, n1, n2);

        let serial_pool = std::sync::Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("serial pool"),
        );
        let plan_serial = Plan3D::new(n0, n1, n2, Direction::Forward, Flags::ESTIMATE)
            .expect("plan_serial")
            .with_rayon_pool(serial_pool);
        let mut out_serial = input.clone();
        plan_serial.execute_inplace(&mut out_serial);

        let plan_parallel =
            Plan3D::new(n0, n1, n2, Direction::Forward, Flags::ESTIMATE).expect("plan_parallel");
        let mut out_parallel = input;
        plan_parallel.execute_inplace(&mut out_parallel);

        for (i, (a, b)) in out_serial.iter().zip(out_parallel.iter()).enumerate() {
            assert!(
                complex_near(*a, *b, 1e-10),
                "element {i}: serial={a:?} parallel={b:?}"
            );
        }
    }

    /// Thread-pool override: 2-worker pool runs without deadlock and produces
    /// correct results on a 256×256 forward transform.
    #[test]
    fn test_plan2d_thread_pool_override_256x256() {
        let n0 = 256;
        let n1 = 256;
        let input = make_input_2d(n0, n1);
        let mut out_override = vec![Complex::<f64>::zero(); n0 * n1];
        let mut out_default = vec![Complex::<f64>::zero(); n0 * n1];

        let pool = std::sync::Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("2-thread pool"),
        );
        let plan_override = Plan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE)
            .expect("override plan")
            .with_rayon_pool(pool);
        plan_override.execute(&input, &mut out_override);

        let plan_default =
            Plan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE).expect("default plan");
        plan_default.execute(&input, &mut out_default);

        for (i, (a, b)) in out_override.iter().zip(out_default.iter()).enumerate() {
            assert!(
                complex_near(*a, *b, 1e-10),
                "element {i}: override={a:?} default={b:?}"
            );
        }
    }

    /// Smoke scaling test (marked `#[ignore]` to avoid CI flakiness).
    ///
    /// On machines with >= 4 available logical cores, asserts that 4-thread
    /// execution of a 512×512 forward transform is faster than single-thread.
    #[test]
    #[ignore = "timing-sensitive smoke test: skip on CI and low-core machines"]
    fn test_plan2d_smoke_scaling_512x512() {
        let n0 = 512;
        let n1 = 512;
        let input = make_input_2d(n0, n1);

        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        if cpus < 4 {
            return; // skip on low-core machines
        }

        // 1-thread baseline
        let pool_1 = std::sync::Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("1-thread pool"),
        );
        let plan_1 = Plan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE)
            .expect("plan 1-thread")
            .with_rayon_pool(pool_1);
        let warmup_iters = 3_usize;
        let bench_iters = 10_usize;
        for _ in 0..warmup_iters {
            let mut out = vec![Complex::<f64>::zero(); n0 * n1];
            plan_1.execute(&input, &mut out);
        }
        let t_single = {
            let start = std::time::Instant::now();
            for _ in 0..bench_iters {
                let mut out = vec![Complex::<f64>::zero(); n0 * n1];
                plan_1.execute(&input, &mut out);
            }
            start.elapsed().as_secs_f64() / bench_iters as f64
        };

        // 4-thread
        let pool_4 = std::sync::Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .expect("4-thread pool"),
        );
        let plan_4 = Plan2D::new(n0, n1, Direction::Forward, Flags::ESTIMATE)
            .expect("plan 4-thread")
            .with_rayon_pool(pool_4);
        for _ in 0..warmup_iters {
            let mut out = vec![Complex::<f64>::zero(); n0 * n1];
            plan_4.execute(&input, &mut out);
        }
        let t_four = {
            let start = std::time::Instant::now();
            for _ in 0..bench_iters {
                let mut out = vec![Complex::<f64>::zero(); n0 * n1];
                plan_4.execute(&input, &mut out);
            }
            start.elapsed().as_secs_f64() / bench_iters as f64
        };

        // 4-thread should be faster than 80% of 1-thread.
        assert!(
            t_four < t_single * 0.80,
            "4-thread ({t_four:.4}s) not significantly faster than 1-thread ({t_single:.4}s)"
        );
    }
}
