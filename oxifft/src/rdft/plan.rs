//! RDFT plan types.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

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

    fn solve(&self, problem: &Self::Problem) {
        use super::problem::RdftKind;
        use super::solvers::{C2rSolver, R2cSolver, R2rSolver};

        let n = problem.transform_size();
        if n == 0 {
            return;
        }

        match problem.kind {
            RdftKind::R2C | RdftKind::R2HC => {
                if problem.real_buf.is_null() || problem.complex_buf.is_null() {
                    return;
                }
                // Safety: caller guarantees validity for n real elements and n/2+1 complex elements.
                let input = unsafe { core::slice::from_raw_parts(problem.real_buf as *const _, n) };
                let complex_len = n / 2 + 1;
                let output =
                    unsafe { core::slice::from_raw_parts_mut(problem.complex_buf, complex_len) };
                R2cSolver::new(n).execute(input, output);
            }
            RdftKind::C2R | RdftKind::HC2R => {
                if problem.real_buf.is_null() || problem.complex_buf.is_null() {
                    return;
                }
                // Safety: caller guarantees validity for n/2+1 complex elements and n real elements.
                let complex_len = n / 2 + 1;
                let input = unsafe {
                    core::slice::from_raw_parts(problem.complex_buf as *const _, complex_len)
                };
                let output = unsafe { core::slice::from_raw_parts_mut(problem.real_buf, n) };
                C2rSolver::new(n).execute(input, output);
            }
            RdftKind::R2R => {
                // R2R (DCT/DST/DHT) is dispatched via the R2rSolver.
                // For this path the real_buf carries both input and output (in-place).
                if problem.real_buf.is_null() {
                    return;
                }
                // Use DCT-II as the default R2R transform (most common usage).
                use super::solvers::R2rKind;
                let data = unsafe { core::slice::from_raw_parts_mut(problem.real_buf, n) };
                // R2rSolver needs separate input/output slices; clone for in-place.
                let input: Vec<T> = data.to_vec();
                R2rSolver::new(R2rKind::Redft10, n).execute(&input, data);
            }
        }
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
