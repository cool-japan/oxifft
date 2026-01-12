//! Internal prelude module for no_std compatibility.
//!
//! This module re-exports commonly used types from either `std` or `alloc`/`core`
//! depending on the `std` feature flag.

// Allow unused imports since this is a prelude module - not all items are used everywhere
#![allow(unused_imports)]

// Re-export alloc types
#[cfg(not(feature = "std"))]
pub use alloc::{
    borrow::ToOwned,
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

#[cfg(feature = "std")]
pub use std::{
    borrow::ToOwned,
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

// HashMap - use hashbrown for both std and no_std for consistency
pub use hashbrown::HashMap;

// Sync primitives - use spin for no_std, std::sync for std
#[cfg(not(feature = "std"))]
pub use spin::{Lazy, Mutex, Once, RwLock};

#[cfg(feature = "std")]
pub use std::sync::{Mutex, OnceLock, RwLock};

#[cfg(feature = "std")]
pub use std::sync::LazyLock as Lazy;

// Atomic types from core (available in both std and no_std)
pub use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

// OnceLock compatibility
#[cfg(not(feature = "std"))]
pub type OnceLock<T> = spin::Once<T>;

// For no_std, we need a wrapper to provide OnceLock-like API
#[cfg(not(feature = "std"))]
pub trait OnceLockExt<T> {
    fn get_or_init<F>(&self, f: F) -> &T
    where
        F: FnOnce() -> T;
}

#[cfg(not(feature = "std"))]
impl<T> OnceLockExt<T> for spin::Once<T> {
    fn get_or_init<F>(&self, f: F) -> &T
    where
        F: FnOnce() -> T,
    {
        self.call_once(f)
    }
}

// PI constant
pub use core::f64::consts::PI;
