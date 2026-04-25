//! Internal prelude module for no_std compatibility.
//!
//! This module re-exports commonly used types from either `std` or `alloc`/`core`
//! depending on the `std` feature flag.

#![allow(unused_imports)] // reason: prelude glob re-exports are selectively used per feature gate (std vs no_std)

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
pub use core::sync::atomic::{AtomicUsize, Ordering};

// AtomicU64 requires 64-bit atomic support (not available on all targets like thumbv7)
#[cfg(target_has_atomic = "64")]
pub use core::sync::atomic::AtomicU64;

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

// RwLock guard helpers — abstracts std (poisonable) vs spin (infallible) RwLock.
//
// `std::sync::RwLock::read()` returns `LockResult<Guard>` which must be unwrapped.
// `spin::RwLock::read()` returns the guard directly with no poisoning concept.
// These free functions hide that difference so callers compile in both modes.

#[cfg(feature = "std")]
#[inline]
pub fn rwlock_read<T>(lock: &std::sync::RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    // reason: spin::RwLock cannot be poisoned; std::sync::RwLock poisoning is
    // propagated from a panicking writer, which we treat as fatal here.
    lock.read().expect("RwLock poisoned")
}

#[cfg(feature = "std")]
#[inline]
pub fn rwlock_write<T>(lock: &std::sync::RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    // reason: see rwlock_read above
    lock.write().expect("RwLock poisoned")
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn rwlock_read<T>(lock: &spin::RwLock<T>) -> spin::RwLockReadGuard<'_, T> {
    lock.read()
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn rwlock_write<T>(lock: &spin::RwLock<T>) -> spin::RwLockWriteGuard<'_, T> {
    lock.write()
}
