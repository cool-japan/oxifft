//! DFT plan types.

use crate::kernel::{Float, OpCount, Plan, WakeMode, WakeState};

use super::problem::Sign;
use super::DftProblem;

/// DFT plan implementation.
pub struct DftPlan<T: Float> {
    /// Operation count.
    ops: OpCount,
    /// Predicted cost.
    pcost: f64,
    /// Wake state.
    state: WakeState,
    /// Solver name.
    solver_name: &'static str,
    /// Marker.
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> DftPlan<T> {
    /// Create a new DFT plan.
    #[must_use]
    pub fn new(solver_name: &'static str, ops: OpCount) -> Self {
        Self {
            ops,
            pcost: ops.total() as f64,
            state: WakeState::Sleeping,
            solver_name,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<T: Float> Plan for DftPlan<T> {
    type Problem = DftProblem<T>;

    fn solve(&self, problem: &Self::Problem) {
        use super::codelets::{execute_composite_codelet, has_composite_codelet};
        use super::solvers::{
            BluesteinSolver, CooleyTukeySolver, CtVariant, DirectSolver, GenericSolver, NopSolver,
        };

        // Bound every raw-pointer access by the element count recorded when the
        // (unsafe) constructor ran, not by a value re-derived from the size
        // tensors: that is exactly the count the caller vouched for, and unlike
        // the tensors it cannot be changed from safe code afterwards.
        let n = problem.buffer_len;
        if n == 0 || problem.input.is_null() || problem.output.is_null() {
            return;
        }

        // Safety: caller must guarantee these pointers are valid and non-overlapping
        // (or overlapping only when doing an in-place transform) for n elements.
        let input = unsafe { core::slice::from_raw_parts(problem.input as *const _, n) };
        let output = unsafe { core::slice::from_raw_parts_mut(problem.output, n) };
        let sign = problem.sign;

        if n <= 1 {
            NopSolver::new().execute(input, output);
        } else if CooleyTukeySolver::<T>::applicable(n) {
            CooleyTukeySolver::new(CtVariant::Dit).execute(input, output, sign);
        } else if has_composite_codelet(n) {
            output.copy_from_slice(input);
            let sign_int = if sign == Sign::Forward { -1 } else { 1 };
            execute_composite_codelet(output, n, sign_int);
        } else if n <= 16 {
            DirectSolver::new().execute(input, output, sign);
        } else if GenericSolver::<T>::applicable(n) {
            GenericSolver::new(n).execute(input, output, sign);
        } else {
            BluesteinSolver::new(n).execute(input, output, sign);
        }
    }

    fn awake(&mut self, _mode: WakeMode) {
        // `DftPlan` carries no precomputed per-plan state: `solve` constructs its
        // solver (and any twiddle factors it needs) on demand from `problem`, so
        // there is nothing for `WakeMode::Full` to prime that `WakeMode::Minimal`
        // does not. Both modes therefore only flip the wake state. If this plan
        // ever gains an owned twiddle cache, `WakeMode::Full` is where it would be
        // populated.
        self.state = WakeState::Awake;
    }

    fn ops(&self) -> OpCount {
        self.ops
    }

    fn pcost(&self) -> f64 {
        self.pcost
    }

    fn wake_state(&self) -> WakeState {
        self.state
    }

    fn solver_name(&self) -> &'static str {
        self.solver_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn awake_transitions_from_sleeping_to_awake_in_both_modes() {
        for mode in [WakeMode::Full, WakeMode::Minimal] {
            let mut plan = DftPlan::<f64>::new("test", OpCount::zero());
            assert_eq!(plan.wake_state(), WakeState::Sleeping);
            plan.awake(mode);
            assert_eq!(
                plan.wake_state(),
                WakeState::Awake,
                "awake({mode:?}) must wake the plan"
            );
        }
    }
}
