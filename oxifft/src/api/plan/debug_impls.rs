//! Manual [`core::fmt::Debug`] implementations for public plan types.
//!
//! The plan structs contain solver state (twiddle-factor tables, boxed trait
//! objects, …) whose internals are crate-private.  These hand-written
//! implementations expose only the user-visible attributes — size, direction,
//! selected algorithm name, etc. — so that `{:?}` formatting is human-readable
//! in tests, debugging sessions, and error messages.

use core::fmt;

use super::types::{Plan, Plan2D, Plan3D};
use super::types_guru::GuruPlan;
use super::types_nd::{PlanND, RealPlanND};
use super::types_r2r::R2rPlan;
use super::types_real::{RealPlan, RealPlan2D, RealPlan3D};
use super::types_split::{SplitPlan, SplitPlan2D, SplitPlan3D, SplitPlanND};
use crate::kernel::Float;

// ---------------------------------------------------------------------------
// Plan<T>
// ---------------------------------------------------------------------------

impl<T: Float> fmt::Debug for Plan<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plan")
            .field("n", &self.size())
            .field("direction", &self.direction())
            .field("algorithm", &self.algorithm_name())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Plan2D<T>
// ---------------------------------------------------------------------------

impl<T: Float> fmt::Debug for Plan2D<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plan2D")
            .field("n0", &self.rows())
            .field("n1", &self.cols())
            .field("direction", &self.direction())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Plan3D<T>
// ---------------------------------------------------------------------------

impl<T: Float> fmt::Debug for Plan3D<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plan3D")
            .field("n0", &self.dim0())
            .field("n1", &self.dim1())
            .field("n2", &self.dim2())
            .field("direction", &self.direction())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// PlanND<T>
// ---------------------------------------------------------------------------

impl<T: Float> fmt::Debug for PlanND<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlanND")
            .field("dims", &self.dims())
            .field("direction", &self.direction())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// RealPlan<T>
// ---------------------------------------------------------------------------

impl<T: Float> fmt::Debug for RealPlan<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealPlan")
            .field("n", &self.size())
            .field("kind", &self.kind())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// RealPlan2D<T>
// ---------------------------------------------------------------------------

impl<T: Float> fmt::Debug for RealPlan2D<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealPlan2D")
            .field("n0", &self.rows())
            .field("n1", &self.cols())
            .field("kind", &self.plan_kind())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// RealPlan3D<T>
// ---------------------------------------------------------------------------

impl<T: Float> fmt::Debug for RealPlan3D<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealPlan3D")
            .field("n0", &self.dim0())
            .field("n1", &self.dim1())
            .field("n2", &self.dim2())
            .field("kind", &self.plan_kind())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// RealPlanND<T>
// ---------------------------------------------------------------------------

impl<T: Float> fmt::Debug for RealPlanND<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealPlanND")
            .field("dims", &self.dims())
            .field("kind", &self.plan_kind())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// GuruPlan<T>
// ---------------------------------------------------------------------------

impl<T: Float> fmt::Debug for GuruPlan<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GuruPlan")
            .field("dims_rank", &self.dims().rank())
            .field("batch_count", &self.batch_count())
            .field("direction", &self.direction())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// SplitPlan<T>
// ---------------------------------------------------------------------------

impl<T: Float> fmt::Debug for SplitPlan<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitPlan")
            .field("n", &self.size())
            .field("direction", &self.direction())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// SplitPlan2D<T>
// ---------------------------------------------------------------------------

impl<T: Float> fmt::Debug for SplitPlan2D<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitPlan2D")
            .field("n0", &self.rows())
            .field("n1", &self.cols())
            .field("direction", &self.direction())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// SplitPlan3D<T>
// ---------------------------------------------------------------------------

impl<T: Float> fmt::Debug for SplitPlan3D<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitPlan3D")
            .field("n0", &self.dim0())
            .field("n1", &self.dim1())
            .field("n2", &self.dim2())
            .field("direction", &self.direction())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// SplitPlanND<T>
// ---------------------------------------------------------------------------

impl<T: Float> fmt::Debug for SplitPlanND<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitPlanND")
            .field("dims", &self.dims())
            .field("direction", &self.direction())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// R2rPlan<T>
// ---------------------------------------------------------------------------

impl<T: Float> fmt::Debug for R2rPlan<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("R2rPlan")
            .field("n", &self.size())
            .field("kind", &self.kind())
            .finish()
    }
}
