//! Core planner and data structures.
//!
//! This module contains the fundamental types and traits that power OxiFFT:
//! - `Float` trait for generic floating-point operations
//! - `Complex<T>` for complex numbers
//! - `Tensor` and `IoDim` for N-dimensional data representation
//! - `Problem`, `Plan`, and `Solver` trait hierarchies
//! - The `Planner` that orchestrates algorithm selection

mod complex;
pub(crate) mod complex_mul;
#[cfg(feature = "f128-support")]
pub mod f128_type;
#[cfg(feature = "f16-support")]
pub mod f16;
mod flags;
mod float;
mod hash;
mod ops;
mod plan;
mod planner;
mod primes;
mod problem;
mod rader_omega;
mod solver;
mod tensor;
mod trig;
mod twiddle;

pub use complex::Complex;
#[cfg(feature = "f128-support")]
pub use f128_type::F128;
#[cfg(feature = "f16-support")]
pub use f16::F16;
pub use flags::PlannerFlags;
pub use float::Float;
pub use hash::ProblemHash;
pub use ops::OpCount;
pub use plan::{Plan, WakeMode, WakeState};
pub use planner::{Planner, SolverChoice, WisdomEntry};
pub use primes::{factor, is_prime, mod_pow, primitive_root};
pub use problem::{Problem, ProblemKind};
pub use solver::Solver;
pub use tensor::{IoDim, Tensor};
pub use trig::TrigTable;
pub use twiddle::{
    clear_twiddle_cache, get_twiddle_table_f32, get_twiddle_table_f64, get_twiddle_table_soa_f32,
    get_twiddle_table_soa_f64, twiddle_mul_scalar_f32, twiddle_mul_scalar_f64,
    twiddle_mul_simd_f32, twiddle_mul_simd_f64, twiddle_mul_soa_scalar_f32,
    twiddle_mul_soa_scalar_f64, twiddle_mul_soa_simd_f32, twiddle_mul_soa_simd_f64, TwiddleCache,
    TwiddleDirection, TwiddleKey, TwiddleTable, TwiddleTableSoA,
};
