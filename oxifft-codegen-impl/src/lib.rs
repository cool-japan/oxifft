//! Internal codelet generation logic for `OxiFFT`.
//!
//! This crate contains the codelet generation functions used by the
//! `oxifft-codegen` proc-macro crate. It is a regular library crate
//! (not `proc-macro = true`) so that its functions can be used from
//! benchmark binaries and integration tests.
//!
//! All public items in this crate are considered **semver-unstable** —
//! they may change at any time. External code should use the proc-macro
//! interface exposed by `oxifft-codegen` instead.

#![allow(clippy::cast_precision_loss)]

pub mod gen_notw;
pub mod gen_rdft;
pub mod gen_simd;
pub mod gen_twiddle;
pub mod symbolic;
