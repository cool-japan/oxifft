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

        let n = problem.transform_size();
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
