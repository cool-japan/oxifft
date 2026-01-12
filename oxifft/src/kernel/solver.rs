//! Solver trait for creating FFT plans.
//!
//! Solvers are factories that produce plans for specific problem types.

use super::{Plan, Planner, Problem, ProblemKind};

/// Solver creates plans for specific problem types.
///
/// The planner iterates through registered solvers to find
/// the best plan for a given problem.
pub trait Solver: Send + Sync {
    /// The problem type this solver handles.
    type Problem: Problem;
    /// The plan type this solver produces.
    type Plan: Plan<Problem = Self::Problem>;

    /// Problem kind this solver handles.
    fn problem_kind(&self) -> ProblemKind;

    /// Attempt to create a plan for the given problem.
    ///
    /// Returns `None` if this solver cannot handle the problem.
    fn make_plan<T>(&self, problem: &Self::Problem, planner: &mut Planner<T>) -> Option<Self::Plan>
    where
        T: super::Float;

    /// Solver name for debugging.
    fn name(&self) -> &'static str;

    /// Priority hint (higher = try first).
    fn priority(&self) -> i32 {
        0
    }
}
