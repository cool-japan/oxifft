//! Main planner for FFT algorithm selection.
//!
//! The planner manages solver registration, wisdom caching,
//! and optimal plan selection.
//!
//! # Planning Modes
//!
//! - **ESTIMATE**: Use heuristics only, fastest planning but may not find optimal algorithm
//! - **MEASURE**: Benchmark a few candidate algorithms and select the fastest (default)
//! - **PATIENT**: Try more algorithm variants
//! - **EXHAUSTIVE**: Try all possible combinations

#[cfg(feature = "std")]
use super::Complex;
use super::{Float, PlannerFlags};
#[cfg(feature = "std")]
use crate::dft::problem::Sign;
#[cfg(feature = "std")]
use crate::dft::solvers::{
    BluesteinSolver, CacheObliviousSolver, CooleyTukeySolver, CtVariant, DirectSolver,
    GenericSolver, RaderSolver, StockhamSolver,
};
use crate::prelude::*;

/// Wisdom entry for plan caching.
#[derive(Clone, Debug)]
pub struct WisdomEntry {
    /// Problem hash.
    pub problem_hash: u64,
    /// Solver name that produced the best plan.
    pub solver_name: String,
    /// Measured cost (operation count or benchmark time).
    pub cost: f64,
}

/// Solver selection strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SolverChoice {
    /// No-op for size 0 or 1
    Nop,
    /// Direct O(n²) computation
    Direct,
    /// Cooley-Tukey radix-2 DIT
    CooleyTukeyDit,
    /// Cooley-Tukey radix-2 DIF
    CooleyTukeyDif,
    /// Cooley-Tukey radix-4
    CooleyTukeyRadix4,
    /// Cooley-Tukey radix-8
    CooleyTukeyRadix8,
    /// Cooley-Tukey split-radix
    CooleyTukeySplitRadix,
    /// Stockham auto-sort (avoids bit-reversal, optimal for large sizes)
    Stockham,
    /// Specialized composite codelets (12, 24, 36, 48, 60, 72, 96, 100)
    Composite,
    /// Generic mixed-radix for composite sizes
    Generic,
    /// Bluestein's algorithm for arbitrary sizes
    Bluestein,
    /// Rader's algorithm for prime sizes
    Rader,
    /// Cache-oblivious four-step FFT for large power-of-2 sizes
    CacheOblivious,
    /// Mixed-radix DIT FFT for smooth-7 composite sizes.
    ///
    /// `factors` is ordered innermost-first; the product equals the transform size.
    MixedRadix { factors: Vec<u16> },
}

impl SolverChoice {
    /// Short, static solver identifier (used for display and non-wisdom purposes).
    ///
    /// For `MixedRadix`, returns the generic tag `"mixed-radix"`.
    /// Use [`SolverChoice::wisdom_name`] when encoding to wisdom strings,
    /// since that method includes the factor sequence.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Nop => "nop",
            Self::Direct => "direct",
            Self::CooleyTukeyDit => "ct-dit",
            Self::CooleyTukeyDif => "ct-dif",
            Self::CooleyTukeyRadix4 => "ct-radix4",
            Self::CooleyTukeyRadix8 => "ct-radix8",
            Self::CooleyTukeySplitRadix => "ct-splitradix",
            Self::Stockham => "stockham",
            Self::Composite => "composite",
            Self::Generic => "generic",
            Self::Bluestein => "bluestein",
            Self::Rader => "rader",
            Self::CacheOblivious => "cache-oblivious",
            Self::MixedRadix { .. } => "mixed-radix",
        }
    }

    /// Wisdom-safe solver name that encodes all solver parameters.
    ///
    /// For `MixedRadix { factors: [3, 2] }` this returns `"mixed-radix-3-2"`.
    /// For all other variants, identical to [`Self::name`].
    #[must_use]
    pub fn wisdom_name(&self) -> String {
        match self {
            Self::MixedRadix { factors } => {
                let suffix: Vec<String> = factors.iter().map(|r| r.to_string()).collect();
                format!("mixed-radix-{}", suffix.join("-"))
            }
            other => other.name().to_string(),
        }
    }
}

/// Main planner for FFT transforms.
///
/// The planner:
/// 1. Maintains a registry of solvers
/// 2. Caches optimal plans in wisdom
/// 3. Measures and compares solver performance
pub struct Planner<T: Float> {
    /// Planning flags.
    pub flags: PlannerFlags,
    /// Wisdom cache (problem hash → entry).
    wisdom: HashMap<u64, WisdomEntry>,
    /// Marker for float type.
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> Default for Planner<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> Planner<T> {
    /// Create a new planner with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            flags: PlannerFlags::default(),
            wisdom: HashMap::new(),
            _marker: core::marker::PhantomData,
        }
    }

    /// Create a planner with specific flags.
    #[must_use]
    pub fn with_flags(flags: PlannerFlags) -> Self {
        Self {
            flags,
            wisdom: HashMap::new(),
            _marker: core::marker::PhantomData,
        }
    }

    /// Select the best solver for a given problem size.
    ///
    /// Uses heuristics in ESTIMATE mode, or measurements in other modes.
    #[must_use]
    pub fn select_solver(&self, n: usize) -> SolverChoice {
        // Check wisdom first (if not ESTIMATE mode)
        if !self.flags.is_estimate() {
            if let Some(entry) = self.wisdom_lookup(hash_size(n)) {
                return solver_from_name(&entry.solver_name);
            }
        }

        // Heuristic-based selection
        self.select_solver_heuristic(n)
    }

    /// Select the best solver by benchmarking (MEASURE mode).
    ///
    /// Benchmarks all applicable solvers and returns the fastest one.
    /// The result is stored in wisdom for future lookups.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn select_solver_measured(&mut self, n: usize) -> SolverChoice {
        // Check wisdom first
        if let Some(entry) = self.wisdom_lookup(hash_size(n)) {
            return solver_from_name(&entry.solver_name);
        }

        // Get candidates to benchmark
        let candidates = self.get_solver_candidates(n);

        if candidates.is_empty() {
            return SolverChoice::Nop;
        }

        if candidates.len() == 1 {
            return candidates.into_iter().next().unwrap_or(SolverChoice::Nop);
        }

        // Benchmark each candidate and find the fastest
        let (best_solver, best_time) = self.benchmark_solvers(n, &candidates);

        // Store in wisdom
        self.remember(n, best_solver.clone(), best_time);

        best_solver
    }

    /// Select the best solver with a time limit.
    ///
    /// Similar to `select_solver_measured`, but stops benchmarking if the time limit
    /// is exceeded. This is useful for planning multiple sizes within a budget.
    ///
    /// # Arguments
    /// * `n` - Transform size
    /// * `time_limit` - Maximum time to spend benchmarking
    ///
    /// # Returns
    /// A tuple of (solver, remaining_time). If time runs out, returns the best
    /// solver found so far (or heuristic if no benchmarks completed).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::time::Duration;
    ///
    /// let mut planner = Planner::with_flags(PlannerFlags::MEASURE);
    /// let time_budget = Duration::from_millis(100);
    ///
    /// // Plan multiple sizes within a time budget
    /// let mut remaining = time_budget;
    /// for size in [64, 128, 256, 512, 1024] {
    ///     let (solver, new_remaining) = planner.select_solver_timed(size, remaining);
    ///     remaining = new_remaining;
    ///     if remaining.is_zero() {
    ///         break; // Time budget exhausted
    ///     }
    /// }
    /// ```
    #[cfg(feature = "std")]
    #[must_use]
    pub fn select_solver_timed(
        &mut self,
        n: usize,
        time_limit: std::time::Duration,
    ) -> (SolverChoice, std::time::Duration) {
        use std::time::Instant;

        let start = Instant::now();

        // Check wisdom first
        if let Some(entry) = self.wisdom_lookup(hash_size(n)) {
            let elapsed = start.elapsed();
            let remaining = time_limit.saturating_sub(elapsed);
            return (solver_from_name(&entry.solver_name), remaining);
        }

        // Get candidates to benchmark
        let candidates = self.get_solver_candidates(n);

        if candidates.is_empty() {
            let elapsed = start.elapsed();
            let remaining = time_limit.saturating_sub(elapsed);
            return (SolverChoice::Nop, remaining);
        }

        if candidates.len() == 1 {
            let elapsed = start.elapsed();
            let remaining = time_limit.saturating_sub(elapsed);
            return (
                candidates.into_iter().next().unwrap_or(SolverChoice::Nop),
                remaining,
            );
        }

        // Benchmark with time limit
        let (best_solver, best_time, benchmarked_count) = self.benchmark_solvers_timed(
            n,
            &candidates,
            time_limit.saturating_sub(start.elapsed()),
        );

        // Store in wisdom if we benchmarked at least one solver
        if benchmarked_count > 0 {
            self.remember(n, best_solver.clone(), best_time);
        }

        let elapsed = start.elapsed();
        let remaining = time_limit.saturating_sub(elapsed);
        (best_solver, remaining)
    }

    /// Benchmark solvers with a time limit.
    ///
    /// Returns (best_solver, best_time, count_of_solvers_benchmarked).
    /// If time runs out before benchmarking any solver, returns the first candidate
    /// with f64::MAX time and count 0.
    #[cfg(feature = "std")]
    fn benchmark_solvers_timed(
        &self,
        n: usize,
        candidates: &[SolverChoice],
        time_limit: std::time::Duration,
    ) -> (SolverChoice, f64, usize) {
        use std::time::Instant;

        let start = Instant::now();
        let mut best_solver = candidates[0].clone();
        let mut best_time = f64::MAX;
        let mut benchmarked_count = 0;

        // Fewer iterations when time-limited
        let iterations = if n <= 64 {
            20
        } else if n <= 1024 {
            5
        } else {
            2
        };

        // Generate test data
        let input: Vec<Complex<T>> = (0..n)
            .map(|i| {
                let t = T::TWO_PI * T::from_usize(i) / T::from_usize(n);
                Complex::new(Float::cos(t), Float::sin(t))
            })
            .collect();
        let mut output = vec![Complex::<T>::zero(); n];

        for solver in candidates.iter().cloned() {
            // Check if time limit exceeded
            if start.elapsed() >= time_limit {
                break;
            }

            let time =
                self.benchmark_single_solver(n, solver.clone(), &input, &mut output, iterations);
            benchmarked_count += 1;

            if time < best_time {
                best_time = time;
                best_solver = solver;
            }
        }

        // If no solvers were benchmarked, fall back to heuristic
        if benchmarked_count == 0 {
            best_solver = self.select_solver_heuristic(n);
        }

        (best_solver, best_time, benchmarked_count)
    }

    /// Get list of solver candidates for a given size.
    #[cfg(feature = "std")]
    #[must_use]
    fn get_solver_candidates(&self, n: usize) -> Vec<SolverChoice> {
        let mut candidates = Vec::new();

        if n <= 1 {
            candidates.push(SolverChoice::Nop);
            return candidates;
        }

        let patient_or_exhaustive = self.flags.is_patient() || self.flags.is_exhaustive();
        let exhaustive = self.flags.is_exhaustive();

        // Power of 2: try CT variants and Stockham
        if n.is_power_of_two() {
            candidates.push(SolverChoice::CooleyTukeyDit);
            candidates.push(SolverChoice::CooleyTukeyDif);

            // Stockham is especially good for larger sizes (avoids bit-reversal)
            if n >= 256 {
                candidates.push(SolverChoice::Stockham);
            }

            // Cache-oblivious four-step FFT for large sizes (>= 1024)
            if n >= 1024 {
                candidates.push(SolverChoice::CacheOblivious);
            }

            // PATIENT mode: also try specialized radix variants
            if patient_or_exhaustive {
                // Radix-4 requires n to be a power of 4
                if n >= 4 && is_power_of_4(n) {
                    candidates.push(SolverChoice::CooleyTukeyRadix4);
                }
                // Radix-8 requires n to be a power of 8
                if n >= 8 && is_power_of_8(n) {
                    candidates.push(SolverChoice::CooleyTukeyRadix8);
                }
                // Split-radix for any power of 2 >= 4
                if n >= 4 {
                    candidates.push(SolverChoice::CooleyTukeySplitRadix);
                }
            }

            // EXHAUSTIVE mode: also try Bluestein and Generic
            if exhaustive {
                candidates.push(SolverChoice::Bluestein);
                // Power of 2 is also smooth, so Generic can work
                candidates.push(SolverChoice::Generic);
            }

            return candidates;
        }

        // Small sizes: try direct
        if n <= 32 || exhaustive {
            candidates.push(SolverChoice::Direct);
        }

        // Composite (smooth) sizes: try generic mixed-radix and the new MixedRadix engine.
        // For smooth-7 sizes that the MixedRadix DIT engine can handle, push both
        // candidates so MEASURE/PATIENT/EXHAUSTIVE modes actually compare them.
        if is_smooth(n, 7) {
            if let Some(factors) = try_factor_mixed_radix_heuristic(n) {
                candidates.push(SolverChoice::MixedRadix { factors });
            }
            candidates.push(SolverChoice::Generic);
        } else if patient_or_exhaustive && is_smooth(n, 13) {
            // PATIENT mode: consider slightly larger prime factors
            candidates.push(SolverChoice::Generic);
        }

        // Prime sizes
        if is_prime(n) {
            candidates.push(SolverChoice::Rader);
        } else if patient_or_exhaustive {
            // PATIENT mode: try Rader for non-primes too (may still work if n-1 factors well)
            // Actually Rader only works for primes, so skip this
        }

        // Bluestein works for any size
        candidates.push(SolverChoice::Bluestein);

        // EXHAUSTIVE mode: also try all power-of-2 solvers if n is a power of 2
        // (already handled above)

        candidates
    }

    /// Benchmark all candidate solvers and return the fastest.
    #[cfg(feature = "std")]
    fn benchmark_solvers(&self, n: usize, candidates: &[SolverChoice]) -> (SolverChoice, f64) {
        let mut best_solver = candidates[0].clone();
        let mut best_time = f64::MAX;

        // Number of benchmark iterations depends on planning mode and size
        // PATIENT and EXHAUSTIVE modes use more iterations for better accuracy
        let base_iterations = if n <= 64 {
            100
        } else if n <= 1024 {
            20
        } else {
            5
        };
        let iterations = if self.flags.is_exhaustive() {
            base_iterations * 3 // Triple iterations for EXHAUSTIVE
        } else if self.flags.is_patient() {
            base_iterations * 2 // Double iterations for PATIENT
        } else {
            base_iterations
        };

        // Generate test data
        let input: Vec<Complex<T>> = (0..n)
            .map(|i| {
                let t = T::TWO_PI * T::from_usize(i) / T::from_usize(n);
                Complex::new(Float::cos(t), Float::sin(t))
            })
            .collect();
        let mut output = vec![Complex::<T>::zero(); n];

        for solver in candidates.iter().cloned() {
            let time =
                self.benchmark_single_solver(n, solver.clone(), &input, &mut output, iterations);
            if time < best_time {
                best_time = time;
                best_solver = solver;
            }
        }

        (best_solver, best_time)
    }

    /// Benchmark a single solver.
    #[cfg(feature = "std")]
    fn benchmark_single_solver(
        &self,
        n: usize,
        solver: SolverChoice,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        iterations: usize,
    ) -> f64 {
        use std::time::Instant;

        // Warm up
        self.execute_solver(n, &solver, input, output);

        // Measure
        let start = Instant::now();
        for _ in 0..iterations {
            self.execute_solver(n, &solver, input, output);
        }
        let elapsed = start.elapsed();

        elapsed.as_secs_f64() / iterations as f64
    }

    /// Execute a specific solver on the given data.
    #[cfg(feature = "std")]
    fn execute_solver(
        &self,
        n: usize,
        solver: &SolverChoice,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
    ) {
        match solver {
            SolverChoice::Nop => {
                if n == 1 {
                    output[0] = input[0];
                }
            }
            SolverChoice::Direct => {
                let s = DirectSolver::<T>::new();
                s.execute(input, output, Sign::Forward);
            }
            SolverChoice::CooleyTukeyDit => {
                let s = CooleyTukeySolver::<T>::new(CtVariant::Dit);
                s.execute(input, output, Sign::Forward);
            }
            SolverChoice::CooleyTukeyDif => {
                let s = CooleyTukeySolver::<T>::new(CtVariant::Dif);
                s.execute(input, output, Sign::Forward);
            }
            SolverChoice::CooleyTukeyRadix4 => {
                let s = CooleyTukeySolver::<T>::new(CtVariant::DitRadix4);
                s.execute(input, output, Sign::Forward);
            }
            SolverChoice::CooleyTukeyRadix8 => {
                let s = CooleyTukeySolver::<T>::new(CtVariant::DitRadix8);
                s.execute(input, output, Sign::Forward);
            }
            SolverChoice::CooleyTukeySplitRadix => {
                let s = CooleyTukeySolver::<T>::new(CtVariant::SplitRadix);
                s.execute(input, output, Sign::Forward);
            }
            SolverChoice::Stockham => {
                let s = StockhamSolver::<T>::new();
                s.execute(input, output, Sign::Forward);
            }
            SolverChoice::Generic => {
                let s = GenericSolver::<T>::new(n);
                s.execute(input, output, Sign::Forward);
            }
            SolverChoice::Bluestein => {
                let s = BluesteinSolver::<T>::new(n);
                s.execute(input, output, Sign::Forward);
            }
            SolverChoice::Rader => {
                if let Some(s) = RaderSolver::<T>::new(n) {
                    s.execute(input, output, Sign::Forward);
                } else {
                    // Fallback to Bluestein if Rader can't be constructed
                    let s = BluesteinSolver::<T>::new(n);
                    s.execute(input, output, Sign::Forward);
                }
            }
            SolverChoice::Composite => {
                use crate::dft::codelets::execute_composite_codelet;
                output.copy_from_slice(input);
                execute_composite_codelet(output, n, -1); // Forward direction
            }
            SolverChoice::CacheOblivious => {
                let s = CacheObliviousSolver::<T>::new();
                s.execute(input, output, Sign::Forward);
            }
            SolverChoice::MixedRadix { factors } => {
                // Delegate to the types.rs executor via the public Plan API.
                // This path is used only in benchmark/measure mode, so the
                // allocation overhead is acceptable.
                use crate::api::{Direction, Flags, Plan};
                if let Some(plan) = Plan::<T>::dft_1d(n, Direction::Forward, Flags::ESTIMATE) {
                    plan.execute(input, output);
                    let _ = factors; // factors already encoded in the plan
                } else {
                    // Fallback (should never happen for valid n)
                    let s = BluesteinSolver::<T>::new(n);
                    s.execute(input, output, Sign::Forward);
                }
            }
        }
    }

    /// Select solver using heuristics only (ESTIMATE mode).
    #[must_use]
    pub fn select_solver_heuristic(&self, n: usize) -> SolverChoice {
        use crate::dft::codelets::has_composite_codelet;

        if n <= 1 {
            return SolverChoice::Nop;
        }

        // Power of 2: Use Stockham for large sizes (avoids bit-reversal), CT for smaller
        if n.is_power_of_two() {
            // Sizes >= 8192: Stockham with AVX2 radix-4 stage fusion is faster
            // (less overhead from avoiding bit-reversal at large sizes)
            if n >= 8192 {
                return SolverChoice::Stockham;
            }
            return SolverChoice::CooleyTukeyDit;
        }

        // Check if we have a specialized composite codelet
        if has_composite_codelet(n) {
            return SolverChoice::Composite;
        }

        // Small sizes: direct computation has lower overhead
        if n <= 16 {
            return SolverChoice::Direct;
        }

        // Mixed-radix DIT for smooth-7 sizes expressible with radices {2,3,4,5,7,8,16}.
        // Checked before Generic because MixedRadix has lower constant than GenericSolver.
        if let Some(factors) = try_factor_mixed_radix_heuristic(n) {
            return SolverChoice::MixedRadix { factors };
        }

        // Generic mixed-radix for remaining composite sizes (smooth-7 that require
        // prime-factoring into {2,3,5,7} but don't map to supported-radix groups).
        if is_smooth(n, 7) {
            return SolverChoice::Generic;
        }

        // Prime: use Rader for small primes, Bluestein for large
        if is_prime(n) {
            if n <= 1021 {
                return SolverChoice::Rader;
            }
            return SolverChoice::Bluestein;
        }

        // Default: Bluestein works for any size
        SolverChoice::Bluestein
    }

    /// Estimate the cost (operation count) for a solver on size n.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn estimate_cost(&self, n: usize, solver: &SolverChoice) -> f64 {
        match solver {
            SolverChoice::Nop => 0.0,
            SolverChoice::Direct => (n * n) as f64 * 4.0, // 4 ops per element pair
            SolverChoice::CooleyTukeyDit | SolverChoice::CooleyTukeyDif => {
                // O(n log n): approximately 5n log2(n) operations
                let log_n = (n as f64).log2();
                n as f64 * log_n * 5.0
            }
            SolverChoice::CooleyTukeyRadix4 => {
                // Radix-4 has fewer twiddle multiplications: ~4n log4(n) operations
                let log4_n = (n as f64).log2() / 2.0;
                n as f64 * log4_n * 4.5
            }
            SolverChoice::CooleyTukeyRadix8 => {
                // Radix-8 has even fewer twiddle multiplications: ~4n log8(n) operations
                let log8_n = (n as f64).log2() / 3.0;
                n as f64 * log8_n * 4.2
            }
            SolverChoice::CooleyTukeySplitRadix => {
                // Split-radix has optimal twiddle count: ~4n log2(n) - 6n + 8
                let log_n = (n as f64).log2();
                4.0 * n as f64 * log_n - 6.0 * n as f64 + 8.0
            }
            SolverChoice::Stockham => {
                // Stockham has similar operation count to CT but avoids bit-reversal
                // For large sizes, the avoided bit-reversal overhead makes it faster
                let log_n = (n as f64).log2();
                n as f64 * log_n * 4.5 // Slightly lower due to no bit-reversal
            }
            SolverChoice::Composite => {
                // Specialized composite codelet: very low overhead, O(n log n)
                let log_n = (n as f64).log2();
                n as f64 * log_n * 3.5 // Very efficient due to inline computation
            }
            SolverChoice::Generic => {
                // Mixed radix: slightly higher constant than pure radix-2
                let log_n = (n as f64).log2();
                n as f64 * log_n * 6.0
            }
            SolverChoice::Bluestein => {
                // Uses 3 FFTs of size 2^k >= 2n, plus O(n) setup
                let m = (2 * n - 1).next_power_of_two();
                let log_m = (m as f64).log2();
                3.0 * m as f64 * log_m * 5.0 + n as f64 * 10.0
            }
            SolverChoice::Rader => {
                // Uses 2 FFTs of size n-1
                let m = n - 1;
                let log_m = (m as f64).log2();
                2.0 * m as f64 * log_m * 5.0 + n as f64 * 8.0
            }
            SolverChoice::CacheOblivious => {
                // Four-step FFT: similar operation count to CT but with better cache behavior
                // Cost model: slightly higher constant due to transpose, but cache-friendly
                let log_n = (n as f64).log2();
                n as f64 * log_n * 4.8 + n as f64 * 2.0 // +2n for transpose
            }
            SolverChoice::MixedRadix { factors } => {
                // Mixed-radix DIT: O(n log n) with constant slightly lower than Generic,
                // because the twiddle butterflies are unrolled for each radix.
                // Cost model: ~5.5 * n * log2(n) amortized across all stages.
                if factors.is_empty() {
                    return f64::INFINITY;
                }
                let stages = factors.len() as f64;
                // n * average_ops_per_stage; each stage is O(r_i) ops per element.
                // Use geometric-mean radix as a proxy for per-stage cost.
                let geom_radix: f64 =
                    factors.iter().map(|&r| (r as f64).ln()).sum::<f64>() / stages;
                n as f64 * geom_radix.exp() * stages * 1.2
            }
        }
    }

    /// Look up wisdom for a problem hash.
    #[must_use]
    pub fn wisdom_lookup(&self, hash: u64) -> Option<&WisdomEntry> {
        self.wisdom.get(&hash)
    }

    /// Store wisdom entry.
    pub fn wisdom_store(&mut self, entry: WisdomEntry) {
        self.wisdom.insert(entry.problem_hash, entry);
    }

    /// Store wisdom for a size/solver combination.
    pub fn remember(&mut self, n: usize, solver: SolverChoice, cost: f64) {
        let entry = WisdomEntry {
            problem_hash: hash_size(n),
            solver_name: solver.wisdom_name(),
            cost,
        };
        self.wisdom_store(entry);
    }

    /// Clear all wisdom.
    pub fn wisdom_forget(&mut self) {
        self.wisdom.clear();
    }

    /// Get number of wisdom entries.
    #[must_use]
    pub fn wisdom_count(&self) -> usize {
        self.wisdom.len()
    }

    /// Recreate a plan from wisdom.
    ///
    /// Looks up the optimal solver for the given size from wisdom cache.
    /// Returns `None` if no wisdom exists for this size.
    ///
    /// This is useful for recreating plans without re-benchmarking:
    /// 1. Use MEASURE/PATIENT/EXHAUSTIVE to find optimal solvers
    /// 2. Export wisdom to file
    /// 3. In future runs, import wisdom and use `recreate_from_wisdom`
    ///
    /// # Example
    ///
    /// ```ignore
    /// // First run: measure and export
    /// let mut planner = Planner::with_flags(PlannerFlags::MEASURE);
    /// planner.select_solver_measured(1024);
    /// std::fs::write("wisdom.txt", planner.wisdom_export())?;
    ///
    /// // Subsequent runs: import and recreate
    /// let mut planner = Planner::new();
    /// planner.wisdom_import(&std::fs::read_to_string("wisdom.txt")?)?;
    /// if let Some(solver) = planner.recreate_from_wisdom(1024) {
    ///     // Use the pre-determined optimal solver
    /// }
    /// ```
    #[must_use]
    pub fn recreate_from_wisdom(&self, n: usize) -> Option<SolverChoice> {
        let hash = hash_size(n);
        self.wisdom_lookup(hash)
            .map(|entry| solver_from_name(&entry.solver_name))
    }

    /// Recreate a batch plan from wisdom.
    ///
    /// Looks up the optimal solver for the given batch configuration from wisdom.
    /// Returns `None` if no wisdom exists for this configuration.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn recreate_batch_from_wisdom(
        &self,
        n: usize,
        howmany: usize,
        istride: isize,
        idist: isize,
    ) -> Option<BatchPlan> {
        let hash = Self::hash_batch_problem(n, howmany, istride, idist);
        self.wisdom_lookup(hash).map(|entry| {
            let solver = solver_from_name(&entry.solver_name);
            BatchPlan {
                solver,
                strategy: self.select_batch_strategy(n, howmany, istride, idist),
                cost: entry.cost,
            }
        })
    }

    /// Check if wisdom exists for a given size.
    #[must_use]
    pub fn has_wisdom_for(&self, n: usize) -> bool {
        let hash = hash_size(n);
        self.wisdom.contains_key(&hash)
    }

    /// Get all sizes that have wisdom entries.
    ///
    /// Note: This returns the hashes, not the original sizes, since the hash
    /// function is one-way. Use `has_wisdom_for(n)` to check specific sizes.
    #[must_use]
    pub fn wisdom_entries(&self) -> Vec<&WisdomEntry> {
        self.wisdom.values().collect()
    }

    /// Export wisdom to string.
    #[must_use]
    pub fn wisdom_export(&self) -> String {
        let mut result = String::from("(oxifft-wisdom-1.0\n");
        for entry in self.wisdom.values() {
            result.push_str(&format!(
                "  ({} \"{}\" {})\n",
                entry.problem_hash, entry.solver_name, entry.cost
            ));
        }
        result.push(')');
        result
    }

    /// Import wisdom from string.
    ///
    /// Accepts both the legacy `(oxifft-wisdom-1.0 …)` header (format version 0)
    /// and the current `(oxifft-wisdom …)` header with an embedded
    /// `(format_version N)` line (format version ≥ 1).
    ///
    /// # Errors
    /// Returns `Err` if the string does not start with a recognised
    /// `oxifft-wisdom` header.
    pub fn wisdom_import(&mut self, s: &str) -> Result<usize, &'static str> {
        // Accept both "(oxifft-wisdom-1.0" (legacy) and "(oxifft-wisdom" (v1+).
        let s = s.trim();
        if !s.starts_with("(oxifft-wisdom") {
            return Err("Invalid wisdom format: missing header");
        }

        let mut count = 0;
        for line in s.lines().skip(1) {
            let line = line.trim();
            // Skip header lines (including "(format_version N)") and closing ")".
            if !line.starts_with('(') || !line.ends_with(')') {
                continue;
            }
            if line.starts_with("(oxifft") || line.starts_with("(format_version ") {
                continue;
            }
            // Parse: (hash "solver" cost)
            let inner = &line[1..line.len() - 1];
            let parts: Vec<&str> = inner.split_whitespace().collect();
            if parts.len() >= 3 {
                if let Ok(hash) = parts[0].parse::<u64>() {
                    let solver_name = parts[1].trim_matches('"').to_string();
                    if let Ok(cost) = parts[2].parse::<f64>() {
                        self.wisdom.insert(
                            hash,
                            WisdomEntry {
                                problem_hash: hash,
                                solver_name,
                                cost,
                            },
                        );
                        count += 1;
                    }
                }
            }
        }

        Ok(count)
    }
}

/// Simple hash for problem size.
fn hash_size(n: usize) -> u64 {
    // Simple but effective hash for single dimension (FNV-1a parameters)
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    h ^= n as u64;
    h = h.wrapping_mul(0x0100_0000_01b3);
    h
}

/// Supported radices for the mixed-radix DIT engine (largest first for greedy factoring).
const MIXED_RADIX_SUPPORTED: &[u16] = &[16, 8, 7, 5, 4, 3, 2];

/// Try to factor `n` as a product of radices in `MIXED_RADIX_SUPPORTED`.
///
/// Returns `Some(factors)` ordered innermost-first, or `None` if not possible.
fn try_factor_mixed_radix_heuristic(n: usize) -> Option<Vec<u16>> {
    if n <= 1 {
        return None;
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
        return None;
    }

    factors.reverse(); // innermost-first
    Some(factors)
}

/// Check if n only has prime factors <= max_factor.
fn is_smooth(n: usize, max_factor: usize) -> bool {
    let mut n = n;
    for p in [2, 3, 5, 7] {
        if p > max_factor {
            break;
        }
        while n.is_multiple_of(p) {
            n /= p;
        }
    }
    n == 1
}

/// Simple primality test.
fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }
    let mut i = 5;
    while i * i <= n {
        if n.is_multiple_of(i) || n.is_multiple_of(i + 2) {
            return false;
        }
        i += 6;
    }
    true
}

/// Convert solver name to choice.
fn solver_from_name(name: &str) -> SolverChoice {
    match name {
        "nop" => SolverChoice::Nop,
        "direct" => SolverChoice::Direct,
        "ct-dit" => SolverChoice::CooleyTukeyDit,
        "ct-dif" => SolverChoice::CooleyTukeyDif,
        "ct-radix4" => SolverChoice::CooleyTukeyRadix4,
        "ct-radix8" => SolverChoice::CooleyTukeyRadix8,
        "ct-splitradix" => SolverChoice::CooleyTukeySplitRadix,
        "stockham" => SolverChoice::Stockham,
        "composite" => SolverChoice::Composite,
        "generic" => SolverChoice::Generic,
        "bluestein" => SolverChoice::Bluestein,
        "rader" => SolverChoice::Rader,
        "cache-oblivious" => SolverChoice::CacheOblivious,
        _ if name.starts_with("mixed-radix-") => {
            // Parse "mixed-radix-3-2" → factors = [3, 2]
            let suffix = &name["mixed-radix-".len()..];
            let factors: Option<Vec<u16>> =
                suffix.split('-').map(|s| s.parse::<u16>().ok()).collect();
            match factors {
                Some(f) if !f.is_empty() => SolverChoice::MixedRadix { factors: f },
                _ => SolverChoice::Bluestein, // malformed wisdom entry → safe fallback
            }
        }
        // Legacy/unknown names → Bluestein as safe fallback
        _ => SolverChoice::Bluestein,
    }
}

/// Check if n is a power of 4.
#[cfg(feature = "std")]
fn is_power_of_4(n: usize) -> bool {
    n.is_power_of_two() && n.trailing_zeros().is_multiple_of(2)
}

/// Check if n is a power of 8.
#[cfg(feature = "std")]
fn is_power_of_8(n: usize) -> bool {
    n.is_power_of_two() && n.trailing_zeros().is_multiple_of(3)
}

/// Batch execution strategy.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BatchStrategy {
    /// Execute each batch independently with the optimal 1D solver
    Simple,
    /// Use buffered execution for better cache locality
    Buffered,
    /// Transpose-based strategy for better memory access patterns
    Transposed,
    /// Interleaved execution for SIMD-friendly patterns
    Interleaved,
}

#[cfg(feature = "std")]
impl BatchStrategy {
    /// Get strategy name for wisdom storage.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Simple => "batch-simple",
            Self::Buffered => "batch-buffered",
            Self::Transposed => "batch-transposed",
            Self::Interleaved => "batch-interleaved",
        }
    }
}

/// Batch planning result.
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct BatchPlan {
    /// The 1D solver to use for each transform
    pub solver: SolverChoice,
    /// The batch execution strategy
    pub strategy: BatchStrategy,
    /// Estimated cost
    pub cost: f64,
}

// Batch planning requires std for cost estimation
#[cfg(feature = "std")]
impl<T: Float> Planner<T> {
    /// Plan a batched transform.
    ///
    /// Considers the transform size, batch count, and stride pattern to
    /// select the optimal solver and execution strategy.
    ///
    /// # Arguments
    /// * `n` - Transform size
    /// * `howmany` - Number of batches
    /// * `istride` - Input stride between consecutive elements
    /// * `idist` - Input distance between batch starts
    #[must_use]
    pub fn plan_batch(&self, n: usize, howmany: usize, istride: isize, idist: isize) -> BatchPlan {
        let solver = self.select_solver(n);

        // Determine the best batch execution strategy
        let strategy = self.select_batch_strategy(n, howmany, istride, idist);

        // Estimate the total cost (borrow solver to avoid move-before-struct-literal)
        let base_cost = self.estimate_cost(n, &solver);
        let batch_cost = self.estimate_batch_cost(n, howmany, istride, idist, strategy);
        let cost = base_cost * howmany as f64 + batch_cost;

        BatchPlan {
            solver,
            strategy,
            cost,
        }
    }

    /// Select the optimal batch execution strategy.
    #[must_use]
    fn select_batch_strategy(
        &self,
        n: usize,
        howmany: usize,
        istride: isize,
        idist: isize,
    ) -> BatchStrategy {
        // Contiguous batches (stride=1, dist=n): simple is optimal
        if istride == 1 && idist == n as isize {
            return BatchStrategy::Simple;
        }

        // Column-major or strided access pattern
        if istride > 1 {
            // For large stride with many batches, transpose can help
            if howmany >= 4 && n >= 16 {
                // Transpose-based approach: pay transpose cost but get contiguous FFTs
                let transpose_cost = n * howmany; // O(n*howmany) for transpose
                let strided_cost = n * howmany * 2; // Strided access is ~2x slower

                if transpose_cost < strided_cost && howmany <= n {
                    return BatchStrategy::Transposed;
                }
            }

            // For small batches or when transpose doesn't help, use buffered
            if n >= 32 {
                return BatchStrategy::Buffered;
            }
        }

        // Small transforms with many batches: interleaved can help with SIMD
        if n <= 8 && howmany >= 8 {
            return BatchStrategy::Interleaved;
        }

        // Default to simple for most cases
        BatchStrategy::Simple
    }

    /// Estimate the overhead cost for batch execution strategy.
    #[must_use]
    fn estimate_batch_cost(
        &self,
        n: usize,
        howmany: usize,
        istride: isize,
        _idist: isize,
        strategy: BatchStrategy,
    ) -> f64 {
        match strategy {
            BatchStrategy::Simple => {
                if istride == 1 {
                    0.0 // No overhead for contiguous
                } else {
                    // Gather/scatter overhead
                    n as f64 * howmany as f64 * 0.5
                }
            }
            BatchStrategy::Buffered => {
                // Buffer copy overhead but better cache behavior
                n as f64 * howmany as f64 * 0.3
            }
            BatchStrategy::Transposed => {
                // Two transpose operations
                2.0 * n as f64 * howmany as f64 * 0.4
            }
            BatchStrategy::Interleaved => {
                // Small reordering overhead
                n as f64 * howmany as f64 * 0.2
            }
        }
    }

    /// Hash a batch problem for wisdom lookup.
    #[must_use]
    pub fn hash_batch_problem(n: usize, howmany: usize, istride: isize, idist: isize) -> u64 {
        let mut h = hash_size(n);
        h ^= (howmany as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        h ^= (istride as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
        h ^= (idist as u64).wrapping_mul(0x2bdd_0a46_c8e5_c8d7);
        h = h.wrapping_mul(0x0100_0000_01b3);
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planner_default() {
        let planner = Planner::<f64>::new();
        assert_eq!(planner.wisdom_count(), 0);
    }

    #[test]
    fn test_select_solver_power_of_2() {
        let planner = Planner::<f64>::with_flags(PlannerFlags::ESTIMATE);
        assert_eq!(planner.select_solver(1), SolverChoice::Nop);
        assert_eq!(planner.select_solver(2), SolverChoice::CooleyTukeyDit);
        assert_eq!(planner.select_solver(4), SolverChoice::CooleyTukeyDit);
        assert_eq!(planner.select_solver(8), SolverChoice::CooleyTukeyDit);
        assert_eq!(planner.select_solver(64), SolverChoice::CooleyTukeyDit);
        assert_eq!(planner.select_solver(256), SolverChoice::CooleyTukeyDit);
        assert_eq!(planner.select_solver(512), SolverChoice::CooleyTukeyDit);
        assert_eq!(planner.select_solver(1024), SolverChoice::CooleyTukeyDit);
        assert_eq!(planner.select_solver(4096), SolverChoice::CooleyTukeyDit);
        // Sizes >= 8192 use Stockham (AVX2 radix-4 stage fusion is faster for large sizes)
        assert_eq!(planner.select_solver(8192), SolverChoice::Stockham);
        assert_eq!(planner.select_solver(16384), SolverChoice::Stockham);
        assert_eq!(planner.select_solver(65536), SolverChoice::Stockham);
    }

    #[test]
    fn test_select_solver_small() {
        let planner = Planner::<f64>::with_flags(PlannerFlags::ESTIMATE);
        assert_eq!(planner.select_solver(3), SolverChoice::Direct);
        assert_eq!(planner.select_solver(5), SolverChoice::Direct);
        assert_eq!(planner.select_solver(7), SolverChoice::Direct);
        assert_eq!(planner.select_solver(9), SolverChoice::Direct); // No codelet for 9
        assert_eq!(planner.select_solver(15), SolverChoice::Composite); // Has codelet
    }

    #[test]
    fn test_select_solver_composite() {
        let planner = Planner::<f64>::with_flags(PlannerFlags::ESTIMATE);
        // Smooth numbers (only small prime factors)
        assert_eq!(planner.select_solver(18), SolverChoice::Composite); // 2*3² (has codelet)
        assert_eq!(planner.select_solver(60), SolverChoice::Composite); // 2²*3*5 (has codelet)
        assert_eq!(planner.select_solver(100), SolverChoice::Composite); // 2²*5² (has codelet)
                                                                         // Sizes without codelets: MixedRadix handles smooth-7 sizes expressible
                                                                         // with radices {2,3,4,5,7,8,16}; Generic handles the rest.
        assert_eq!(
            planner.select_solver(27),
            SolverChoice::MixedRadix {
                factors: vec![3, 3, 3]
            }
        ); // 3³ = MixedRadix [3,3,3]
    }

    #[test]
    fn test_select_solver_prime() {
        let planner = Planner::<f64>::with_flags(PlannerFlags::ESTIMATE);
        assert_eq!(planner.select_solver(17), SolverChoice::Rader);
        assert_eq!(planner.select_solver(101), SolverChoice::Rader);
        assert_eq!(planner.select_solver(997), SolverChoice::Rader);
        // Large prime uses Bluestein
        assert_eq!(planner.select_solver(1031), SolverChoice::Bluestein);
    }

    #[test]
    fn test_wisdom_store_lookup() {
        let mut planner = Planner::<f64>::new();

        planner.remember(256, SolverChoice::CooleyTukeyDit, 100.0);

        assert_eq!(planner.wisdom_count(), 1);

        let entry = planner.wisdom_lookup(hash_size(256)).unwrap();
        assert_eq!(entry.solver_name, "ct-dit");
    }

    #[test]
    fn test_wisdom_export_import() {
        let mut planner = Planner::<f64>::new();
        planner.remember(64, SolverChoice::CooleyTukeyDit, 50.0);
        planner.remember(100, SolverChoice::Generic, 80.0);

        let exported = planner.wisdom_export();
        assert!(exported.contains("oxifft-wisdom-1.0"));

        let mut planner2 = Planner::<f64>::new();
        let count = planner2.wisdom_import(&exported).unwrap();
        assert_eq!(count, 2);
        assert_eq!(planner2.wisdom_count(), 2);
    }

    #[test]
    fn test_estimate_cost() {
        let planner = Planner::<f64>::new();

        // Nop should be zero cost
        assert_eq!(planner.estimate_cost(1, &SolverChoice::Nop), 0.0);

        // Direct should be O(n²)
        let direct_cost = planner.estimate_cost(100, &SolverChoice::Direct);
        assert!(direct_cost > 10000.0); // n² = 10000

        // CT should be O(n log n), much less than O(n²)
        let ct_cost = planner.estimate_cost(1024, &SolverChoice::CooleyTukeyDit);
        let direct_large = planner.estimate_cost(1024, &SolverChoice::Direct);
        assert!(ct_cost < direct_large / 10.0); // Should be much faster
    }

    #[test]
    fn test_is_smooth() {
        assert!(is_smooth(1, 7));
        assert!(is_smooth(2, 7));
        assert!(is_smooth(4, 7));
        assert!(is_smooth(6, 7)); // 2*3
        assert!(is_smooth(12, 7)); // 2²*3
        assert!(is_smooth(60, 7)); // 2²*3*5
        assert!(is_smooth(210, 7)); // 2*3*5*7
        assert!(!is_smooth(11, 7)); // Prime > 7
        assert!(!is_smooth(22, 7)); // 2*11
    }

    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(is_prime(7));
        assert!(is_prime(11));
        assert!(is_prime(97));
        assert!(is_prime(1009));
        assert!(!is_prime(1001)); // 7*11*13
    }

    #[test]
    fn test_plan_batch_contiguous() {
        let planner = Planner::<f64>::with_flags(PlannerFlags::ESTIMATE);

        // Contiguous batches should use simple strategy
        let plan = planner.plan_batch(64, 10, 1, 64);
        assert_eq!(plan.strategy, BatchStrategy::Simple);
        assert_eq!(plan.solver, SolverChoice::CooleyTukeyDit);
    }

    #[test]
    fn test_plan_batch_strided() {
        let planner = Planner::<f64>::with_flags(PlannerFlags::ESTIMATE);

        // Strided batches with large stride and many batches
        let plan = planner.plan_batch(64, 8, 8, 1);
        assert_eq!(plan.strategy, BatchStrategy::Transposed);
    }

    #[test]
    fn test_plan_batch_small_many() {
        let planner = Planner::<f64>::with_flags(PlannerFlags::ESTIMATE);

        // Small transform with many batches but contiguous uses Simple
        // (because stride=1 and dist=n takes precedence)
        let plan = planner.plan_batch(4, 16, 1, 4);
        assert_eq!(plan.strategy, BatchStrategy::Simple);

        // Small transform with many batches and non-contiguous uses Interleaved
        let plan2 = planner.plan_batch(4, 16, 2, 8);
        assert_eq!(plan2.strategy, BatchStrategy::Interleaved);
    }

    #[test]
    fn test_batch_strategy_names() {
        assert_eq!(BatchStrategy::Simple.name(), "batch-simple");
        assert_eq!(BatchStrategy::Buffered.name(), "batch-buffered");
        assert_eq!(BatchStrategy::Transposed.name(), "batch-transposed");
        assert_eq!(BatchStrategy::Interleaved.name(), "batch-interleaved");
    }

    #[test]
    fn test_hash_batch_problem() {
        // Different parameters should produce different hashes
        let h1 = Planner::<f64>::hash_batch_problem(64, 10, 1, 64);
        let h2 = Planner::<f64>::hash_batch_problem(64, 10, 1, 65);
        let h3 = Planner::<f64>::hash_batch_problem(64, 11, 1, 64);
        let h4 = Planner::<f64>::hash_batch_problem(65, 10, 1, 64);

        assert_ne!(h1, h2);
        assert_ne!(h1, h3);
        assert_ne!(h1, h4);
    }

    #[test]
    fn test_measure_mode_power_of_2() {
        let mut planner = Planner::<f64>::with_flags(PlannerFlags::MEASURE);

        // Measure mode should benchmark and select best solver for power of 2
        let solver = planner.select_solver_measured(64);
        // Should choose CT-DIT or CT-DIF (both are valid for power of 2)
        assert!(
            solver == SolverChoice::CooleyTukeyDit || solver == SolverChoice::CooleyTukeyDif,
            "Expected CT-DIT or CT-DIF, got {solver:?}"
        );

        // Wisdom should now contain an entry
        assert_eq!(planner.wisdom_count(), 1);

        // Second call should use cached wisdom
        let solver2 = planner.select_solver_measured(64);
        assert_eq!(solver, solver2);
    }

    #[test]
    fn test_measure_mode_prime() {
        let mut planner = Planner::<f64>::with_flags(PlannerFlags::MEASURE);

        // For primes, should benchmark Rader, Bluestein, and possibly Direct (for small sizes)
        let solver = planner.select_solver_measured(17);
        // Could be Direct (fast for small sizes), Rader, or Bluestein
        assert!(
            solver == SolverChoice::Direct
                || solver == SolverChoice::Rader
                || solver == SolverChoice::Bluestein,
            "Expected Direct, Rader or Bluestein, got {solver:?}"
        );

        assert_eq!(planner.wisdom_count(), 1);
    }

    #[test]
    fn test_measure_mode_composite() {
        let mut planner = Planner::<f64>::with_flags(PlannerFlags::MEASURE);

        // For smooth composite (60 = 2²×3×5), should benchmark MixedRadix, Generic, and Bluestein.
        // MixedRadix is now in the candidate set for smooth-7 sizes and may win the benchmark.
        let solver = planner.select_solver_measured(60); // 2^2 * 3 * 5
        assert!(
            solver == SolverChoice::Generic
                || solver == SolverChoice::Bluestein
                || matches!(solver, SolverChoice::MixedRadix { .. }),
            "Expected Generic, Bluestein, or MixedRadix, got {solver:?}"
        );

        assert_eq!(planner.wisdom_count(), 1);
    }

    #[test]
    fn test_get_solver_candidates() {
        let planner = Planner::<f64>::with_flags(PlannerFlags::MEASURE);

        // Power of 2 should have CT variants
        let candidates = planner.get_solver_candidates(64);
        assert!(candidates.contains(&SolverChoice::CooleyTukeyDit));
        assert!(candidates.contains(&SolverChoice::CooleyTukeyDif));

        // Small non-power-of-2 should have Direct
        let candidates = planner.get_solver_candidates(5);
        assert!(candidates.contains(&SolverChoice::Direct));

        // Prime should have Rader and Bluestein
        let candidates = planner.get_solver_candidates(17);
        assert!(candidates.contains(&SolverChoice::Rader));
        assert!(candidates.contains(&SolverChoice::Bluestein));
    }

    #[test]
    fn test_planner_flags_measure() {
        // Default flags should be MEASURE mode
        assert!(PlannerFlags::MEASURE.is_measure());
        assert!(!PlannerFlags::MEASURE.is_estimate());
        assert!(!PlannerFlags::ESTIMATE.is_measure());
        assert!(PlannerFlags::ESTIMATE.is_estimate());
    }

    #[test]
    fn test_patient_mode_candidates() {
        let planner = Planner::<f64>::with_flags(PlannerFlags::PATIENT);

        // PATIENT mode should include radix-4, radix-8, and split-radix for power-of-2 sizes
        let candidates = planner.get_solver_candidates(64); // 64 = 4^3 = 2^6
        assert!(candidates.contains(&SolverChoice::CooleyTukeyDit));
        assert!(candidates.contains(&SolverChoice::CooleyTukeyDif));
        assert!(candidates.contains(&SolverChoice::CooleyTukeyRadix4)); // 64 is power of 4
        assert!(candidates.contains(&SolverChoice::CooleyTukeySplitRadix));

        // 512 = 8^3 is a power of 8
        let candidates = planner.get_solver_candidates(512);
        assert!(candidates.contains(&SolverChoice::CooleyTukeyRadix8)); // 512 is power of 8
    }

    #[test]
    fn test_exhaustive_mode_candidates() {
        let planner = Planner::<f64>::with_flags(PlannerFlags::EXHAUSTIVE);

        // EXHAUSTIVE mode should include even more options
        let candidates = planner.get_solver_candidates(64);
        assert!(candidates.contains(&SolverChoice::CooleyTukeyDit));
        assert!(candidates.contains(&SolverChoice::CooleyTukeyDif));
        assert!(candidates.contains(&SolverChoice::CooleyTukeyRadix4));
        assert!(candidates.contains(&SolverChoice::CooleyTukeySplitRadix));
        assert!(candidates.contains(&SolverChoice::Bluestein)); // EXHAUSTIVE includes this
        assert!(candidates.contains(&SolverChoice::Generic)); // EXHAUSTIVE includes this for power-of-2

        // EXHAUSTIVE should try Direct even for larger sizes
        let candidates = planner.get_solver_candidates(100);
        assert!(candidates.contains(&SolverChoice::Direct)); // EXHAUSTIVE always tries Direct
    }

    #[test]
    fn test_patient_mode_measurement() {
        let mut planner = Planner::<f64>::with_flags(PlannerFlags::PATIENT);

        // PATIENT mode should find a good solver for power of 4 sizes
        // 64 = 2^6 = 4^3 = 8^2, so radix-4 and radix-8 are both applicable
        let solver = planner.select_solver_measured(64);
        // Should be one of the CT variants
        assert!(
            solver == SolverChoice::CooleyTukeyDit
                || solver == SolverChoice::CooleyTukeyDif
                || solver == SolverChoice::CooleyTukeyRadix4
                || solver == SolverChoice::CooleyTukeyRadix8
                || solver == SolverChoice::CooleyTukeySplitRadix,
            "Expected a CT variant, got {solver:?}"
        );

        assert_eq!(planner.wisdom_count(), 1);
    }

    #[test]
    fn test_exhaustive_mode_measurement() {
        let mut planner = Planner::<f64>::with_flags(PlannerFlags::EXHAUSTIVE);

        // EXHAUSTIVE mode should find a reasonable solver
        let solver = planner.select_solver_measured(256);
        // Should be one of the CT variants or Generic (not Bluestein, which is slower for powers of 2)
        // Generic is valid for power-of-2 sizes since they are smooth numbers
        assert!(
            solver == SolverChoice::CooleyTukeyDit
                || solver == SolverChoice::CooleyTukeyDif
                || solver == SolverChoice::CooleyTukeyRadix4
                || solver == SolverChoice::CooleyTukeyRadix8
                || solver == SolverChoice::CooleyTukeySplitRadix
                || solver == SolverChoice::Stockham
                || solver == SolverChoice::Generic,
            "Expected a CT variant, Stockham, or Generic, got {solver:?}"
        );

        assert_eq!(planner.wisdom_count(), 1);
    }

    #[test]
    fn test_is_power_of_4() {
        assert!(is_power_of_4(1)); // 4^0
        assert!(is_power_of_4(4)); // 4^1
        assert!(is_power_of_4(16)); // 4^2
        assert!(is_power_of_4(64)); // 4^3
        assert!(is_power_of_4(256)); // 4^4
        assert!(!is_power_of_4(2)); // Not power of 4
        assert!(!is_power_of_4(8)); // Not power of 4
        assert!(!is_power_of_4(32)); // Not power of 4
    }

    #[test]
    fn test_is_power_of_8() {
        assert!(is_power_of_8(1)); // 8^0
        assert!(is_power_of_8(8)); // 8^1
        assert!(is_power_of_8(64)); // 8^2
        assert!(is_power_of_8(512)); // 8^3
        assert!(!is_power_of_8(2)); // Not power of 8
        assert!(!is_power_of_8(4)); // Not power of 8
        assert!(!is_power_of_8(16)); // Not power of 8
        assert!(!is_power_of_8(32)); // Not power of 8
    }

    #[test]
    fn test_solver_choice_names() {
        assert_eq!(SolverChoice::Nop.name(), "nop");
        assert_eq!(SolverChoice::Direct.name(), "direct");
        assert_eq!(SolverChoice::CooleyTukeyDit.name(), "ct-dit");
        assert_eq!(SolverChoice::CooleyTukeyDif.name(), "ct-dif");
        assert_eq!(SolverChoice::CooleyTukeyRadix4.name(), "ct-radix4");
        assert_eq!(SolverChoice::CooleyTukeyRadix8.name(), "ct-radix8");
        assert_eq!(SolverChoice::CooleyTukeySplitRadix.name(), "ct-splitradix");
        assert_eq!(SolverChoice::Generic.name(), "generic");
        assert_eq!(SolverChoice::Bluestein.name(), "bluestein");
        assert_eq!(SolverChoice::Rader.name(), "rader");
    }

    #[test]
    fn test_solver_from_name_roundtrip() {
        // Test that all solver choices round-trip through name conversion
        let choices = [
            SolverChoice::Nop,
            SolverChoice::Direct,
            SolverChoice::CooleyTukeyDit,
            SolverChoice::CooleyTukeyDif,
            SolverChoice::CooleyTukeyRadix4,
            SolverChoice::CooleyTukeyRadix8,
            SolverChoice::CooleyTukeySplitRadix,
            SolverChoice::Generic,
            SolverChoice::Bluestein,
            SolverChoice::Rader,
        ];

        for choice in &choices {
            let name = choice.name();
            let recovered = solver_from_name(name);
            assert_eq!(*choice, recovered, "Failed roundtrip for {choice:?}");
        }
    }

    #[test]
    fn test_mixed_radix_wisdom_name_roundtrip() {
        // MixedRadix uses wisdom_name() (not name()) for wisdom storage.
        // Verify the round-trip: wisdom_name() → solver_from_name() recovers the original.
        let cases = [
            (vec![3u16, 2], "mixed-radix-3-2"),
            (vec![3, 3, 3], "mixed-radix-3-3-3"),
            (vec![2, 5, 7], "mixed-radix-2-5-7"),
            (vec![16, 3], "mixed-radix-16-3"),
        ];

        for (factors, expected_wisdom_name) in &cases {
            let choice = SolverChoice::MixedRadix {
                factors: factors.clone(),
            };
            let wname = choice.wisdom_name();
            assert_eq!(
                wname, *expected_wisdom_name,
                "wisdom_name mismatch for factors {factors:?}"
            );
            let recovered = solver_from_name(&wname);
            assert_eq!(
                recovered, choice,
                "solver_from_name round-trip failed for {expected_wisdom_name}"
            );
        }
    }

    #[test]
    fn test_recreate_from_wisdom() {
        let mut planner = Planner::<f64>::with_flags(PlannerFlags::MEASURE);

        // Initially, no wisdom exists
        assert!(!planner.has_wisdom_for(256));
        assert!(planner.recreate_from_wisdom(256).is_none());

        // Measure to populate wisdom
        let solver = planner.select_solver_measured(256);

        // Now wisdom should exist
        assert!(planner.has_wisdom_for(256));

        // Recreate should return the same solver
        let recreated = planner.recreate_from_wisdom(256);
        assert_eq!(recreated, Some(solver));
    }

    #[test]
    fn test_recreate_from_imported_wisdom() {
        // Create first planner and measure
        let mut planner1 = Planner::<f64>::with_flags(PlannerFlags::MEASURE);
        let solver1 = planner1.select_solver_measured(512);

        // Export wisdom
        let wisdom_str = planner1.wisdom_export();

        // Create second planner and import wisdom
        let mut planner2 = Planner::<f64>::new();
        let count = planner2.wisdom_import(&wisdom_str).unwrap();
        assert_eq!(count, 1);

        // Recreate from imported wisdom
        let recreated = planner2.recreate_from_wisdom(512);
        assert_eq!(recreated, Some(solver1));
    }

    #[test]
    fn test_has_wisdom_for() {
        let mut planner = Planner::<f64>::new();

        // No wisdom initially
        assert!(!planner.has_wisdom_for(64));
        assert!(!planner.has_wisdom_for(128));

        // Add wisdom for 64
        planner.remember(64, SolverChoice::CooleyTukeyDit, 100.0);

        // Now 64 has wisdom, 128 doesn't
        assert!(planner.has_wisdom_for(64));
        assert!(!planner.has_wisdom_for(128));
    }

    #[test]
    fn test_wisdom_entries() {
        let mut planner = Planner::<f64>::new();

        assert!(planner.wisdom_entries().is_empty());

        planner.remember(64, SolverChoice::CooleyTukeyDit, 100.0);
        planner.remember(128, SolverChoice::CooleyTukeyDif, 200.0);

        let entries = planner.wisdom_entries();
        assert_eq!(entries.len(), 2);

        // Check that entries contain expected solver names
        let names: Vec<&str> = entries.iter().map(|e| e.solver_name.as_str()).collect();
        assert!(names.contains(&"ct-dit"));
        assert!(names.contains(&"ct-dif"));
    }

    #[test]
    fn test_wisdom_only_mode() {
        let mut planner = Planner::<f64>::with_flags(PlannerFlags::WISDOM_ONLY);

        // With WISDOM_ONLY, select_solver should only look at wisdom
        // Since there's no wisdom, it should fall back to heuristics
        let solver = planner.select_solver(256);

        // Should use heuristic selection (power of 2 -> CT-DIT)
        assert_eq!(solver, SolverChoice::CooleyTukeyDit);

        // Add some wisdom
        planner.remember(256, SolverChoice::CooleyTukeyDif, 50.0);

        // Now it should use wisdom
        let solver = planner.select_solver(256);
        assert_eq!(solver, SolverChoice::CooleyTukeyDif);
    }

    #[test]
    fn test_select_solver_timed_basic() {
        use std::time::Duration;

        let mut planner = Planner::<f64>::with_flags(PlannerFlags::MEASURE);

        // With generous time limit, should complete successfully
        let (solver, remaining) = planner.select_solver_timed(64, Duration::from_secs(10));

        // Should return a valid CT solver for power of 2
        assert!(
            solver == SolverChoice::CooleyTukeyDit || solver == SolverChoice::CooleyTukeyDif,
            "Expected CT solver, got {solver:?}"
        );

        // Should have time remaining
        assert!(remaining < Duration::from_secs(10));

        // Wisdom should be stored
        assert!(planner.has_wisdom_for(64));
    }

    #[test]
    fn test_select_solver_timed_uses_wisdom() {
        use std::time::Duration;

        let mut planner = Planner::<f64>::with_flags(PlannerFlags::MEASURE);

        // First call should benchmark and store wisdom
        let (solver1, _) = planner.select_solver_timed(128, Duration::from_secs(10));
        assert!(planner.has_wisdom_for(128));

        // Second call should use wisdom and be instant
        let start = std::time::Instant::now();
        let (solver2, _) = planner.select_solver_timed(128, Duration::from_millis(1));
        let elapsed = start.elapsed();

        // Should return same solver
        assert_eq!(solver1, solver2);

        // Should be very fast (wisdom lookup)
        assert!(elapsed < Duration::from_millis(10));
    }

    #[test]
    fn test_select_solver_timed_multiple_sizes() {
        use std::time::Duration;

        let mut planner = Planner::<f64>::with_flags(PlannerFlags::MEASURE);
        let budget = Duration::from_millis(500);
        let mut remaining = budget;

        let sizes = [16, 32, 64, 128, 256];
        let mut completed = 0;

        for size in sizes {
            if remaining.is_zero() {
                break;
            }
            let (solver, new_remaining) = planner.select_solver_timed(size, remaining);
            remaining = new_remaining;
            completed += 1;

            // Should return valid solver for power of 2
            assert!(
                solver == SolverChoice::CooleyTukeyDit
                    || solver == SolverChoice::CooleyTukeyDif
                    || solver == SolverChoice::Stockham
                    || solver == SolverChoice::Nop,
                "Unexpected solver {solver:?} for size {size}"
            );
        }

        // Should have completed at least some sizes
        assert!(completed >= 1, "Should have completed at least one size");
    }

    #[test]
    fn test_select_solver_timed_zero_limit() {
        use std::time::Duration;

        let mut planner = Planner::<f64>::with_flags(PlannerFlags::MEASURE);

        // With zero time limit, should fall back to heuristic
        let (solver, remaining) = planner.select_solver_timed(64, Duration::ZERO);

        // Should return heuristic choice
        assert_eq!(solver, SolverChoice::CooleyTukeyDit);

        // Remaining should be zero
        assert!(remaining.is_zero());

        // No wisdom should be stored (no benchmarking happened)
        assert!(!planner.has_wisdom_for(64));
    }

    #[test]
    fn test_select_solver_timed_size_1() {
        use std::time::Duration;

        let mut planner = Planner::<f64>::with_flags(PlannerFlags::MEASURE);

        // Size 1 should be immediate (NOP)
        let (solver, _) = planner.select_solver_timed(1, Duration::from_millis(100));
        assert_eq!(solver, SolverChoice::Nop);
    }
}
