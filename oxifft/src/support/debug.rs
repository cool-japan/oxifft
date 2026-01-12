//! Debug printing utilities.
//!
//! This module is only available with the `std` feature.

use crate::kernel::{Complex, Float};

/// Print a complex array for debugging.
pub fn print_complex<T: Float + std::fmt::Display>(name: &str, data: &[Complex<T>]) {
    println!("{name}:");
    for (i, c) in data.iter().enumerate() {
        println!("  [{i}] = {} + {}i", c.re, c.im);
    }
}

/// Print a real array for debugging.
pub fn print_real<T: Float + std::fmt::Display>(name: &str, data: &[T]) {
    println!("{name}:");
    for (i, v) in data.iter().enumerate() {
        println!("  [{i}] = {v}");
    }
}
