//! Real Even DFT (Discrete Cosine Transform variants).
//!
//! REDFT00 = DCT-I
//! REDFT01 = DCT-III
//! REDFT10 = DCT-II (the standard "DCT")
//! REDFT11 = DCT-IV
//!
//! These are re-exports of the implementations in `rdft::solvers::r2r`.

use crate::kernel::Float;
use crate::rdft::solvers::{
    dct1 as r2r_dct1, dct2 as r2r_dct2, dct3 as r2r_dct3, dct4 as r2r_dct4,
};

/// DCT type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DctType {
    /// DCT-I (REDFT00)
    I,
    /// DCT-II (REDFT10) - the standard DCT
    II,
    /// DCT-III (REDFT01) - inverse of DCT-II
    III,
    /// DCT-IV (REDFT11)
    IV,
}

/// Execute DCT-II (the standard DCT).
pub fn dct_ii<T: Float>(input: &[T], output: &mut [T]) {
    r2r_dct2(input, output);
}

/// Execute DCT-III (inverse of DCT-II).
pub fn dct_iii<T: Float>(input: &[T], output: &mut [T]) {
    r2r_dct3(input, output);
}

/// Execute DCT-I.
pub fn dct_i<T: Float>(input: &[T], output: &mut [T]) {
    r2r_dct1(input, output);
}

/// Execute DCT-IV.
pub fn dct_iv<T: Float>(input: &[T], output: &mut [T]) {
    r2r_dct4(input, output);
}
