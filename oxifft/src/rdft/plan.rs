//! RDFT plan types.

use crate::kernel::{Float, OpCount, Plan, WakeMode, WakeState};

use super::RdftProblem;

/// RDFT plan implementation.
pub struct RdftPlan<T: Float> {
    ops: OpCount,
    state: WakeState,
    solver_name: &'static str,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> RdftPlan<T> {
    /// Create a new RDFT plan.
    #[must_use]
    pub fn new(solver_name: &'static str, ops: OpCount) -> Self {
        Self {
            ops,
            state: WakeState::Sleeping,
            solver_name,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<T: Float> Plan for RdftPlan<T> {
    type Problem = RdftProblem<T>;

    fn solve(&self, _problem: &Self::Problem) {
        // This is internal scaffolding. Actual execution goes through api::RealPlan.
        // This method exists for trait compliance but is not called in practice.
    }

    fn awake(&mut self, _mode: WakeMode) {
        self.state = WakeState::Awake;
    }

    fn ops(&self) -> OpCount {
        self.ops
    }

    fn wake_state(&self) -> WakeState {
        self.state
    }

    fn solver_name(&self) -> &'static str {
        self.solver_name
    }
}
