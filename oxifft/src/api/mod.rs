//! Public user-facing API for OxiFFT.
//!
//! This module provides the high-level interface for creating and executing FFT plans.

#![allow(hidden_glob_reexports)]

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
