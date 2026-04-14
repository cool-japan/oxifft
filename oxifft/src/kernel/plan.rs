//! Plan abstraction for optimized FFT execution.

use super::Problem;

/// Wake mode for plan initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WakeMode {
    /// Full initialization (compute twiddle factors, etc.)
    Full,
    /// Minimal initialization (for planning cost estimation)
    Minimal,
}

/// Wake state of a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum WakeState {
    /// Plan is sleeping (not initialized).
    #[default]
    Sleeping,
    /// Plan is awake and ready to execute.
    Awake,
}

/// Operation count for cost estimation.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpCount {
    /// Number of floating-point additions.
    pub add: usize,
    /// Number of floating-point multiplications.
    pub mul: usize,
    /// Number of fused multiply-add operations.
    pub fma: usize,
    /// Other operations.
    pub other: usize,
}

impl OpCount {
    /// Create a zero operation count.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            add: 0,
            mul: 0,
            fma: 0,
            other: 0,
        }
    }

    /// Total operation count.
    #[must_use]
    pub const fn total(&self) -> usize {
        self.add + self.mul + 2 * self.fma + self.other
    }

    /// Combine two operation counts.
    #[must_use]
    pub const fn combine(self, other: Self) -> Self {
        Self {
            add: self.add + other.add,
            mul: self.mul + other.mul,
            fma: self.fma + other.fma,
            other: self.other + other.other,
        }
    }
}

/// Base trait for all FFT plans.
///
/// A plan represents an optimized execution strategy for a specific problem.
pub trait Plan: Send + Sync {
    /// The problem type this plan solves.
    type Problem: Problem;

    /// Execute the plan.
    fn solve(&self, problem: &Self::Problem);

    /// Initialize the plan (awake from sleep).
    fn awake(&mut self, mode: WakeMode);

    /// Get operation count for cost estimation.
    fn ops(&self) -> OpCount;

    /// Get predicted cost (for planning).
    fn pcost(&self) -> f64 {
        self.ops().total() as f64
    }

    /// Get current wake state.
    fn wake_state(&self) -> WakeState;

    /// Solver name for debugging.
    fn solver_name(&self) -> &'static str;
}
