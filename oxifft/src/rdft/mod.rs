//! Real DFT implementations.
//!
//! Provides solvers for real-to-complex (R2C), complex-to-real (C2R),
//! and real-to-real (R2R) transforms.

pub mod codelets;
pub mod plan;
pub mod problem;
pub mod solvers;

pub use plan::RdftPlan;
pub use problem::{RdftKind, RdftProblem};
