//! DFT plan types.

use crate::kernel::{Float, OpCount, Plan, WakeMode, WakeState};

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

    fn solve(&self, _problem: &Self::Problem) {
        // This is internal scaffolding. Actual execution goes through api::Plan.
        // This method exists for trait compliance but is not called in practice.
    }

    fn awake(&mut self, mode: WakeMode) {
        match mode {
            WakeMode::Full => {
                // Initialize twiddle factors, etc.
                self.state = WakeState::Awake;
            }
            WakeMode::Minimal => {
                self.state = WakeState::Awake;
            }
        }
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
