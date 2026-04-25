//! Public user-facing API for OxiFFT.
//!
//! This module provides the high-level interface for creating and executing FFT plans.

#![allow(hidden_glob_reexports)] // reason: api module intentionally re-exports types from sub-modules; shadowing is expected

mod execute;
mod memory;
mod parallel;
mod plan;
mod types;
mod wisdom;

// execute module is reserved for future use (execution done through plan objects)
pub use memory::*;
pub use parallel::*;
pub use plan::*;
pub use types::*;
pub use wisdom::*;
