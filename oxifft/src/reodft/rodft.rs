//! Real Odd DFT (Discrete Sine Transform variants).
//!
//! RODFT00 = DST-I
//! RODFT01 = DST-III
//! RODFT10 = DST-II
//! RODFT11 = DST-IV
//!
//! These are re-exports of the implementations in `rdft::solvers::r2r`.

use crate::kernel::Float;
use crate::rdft::solvers::{
    dst1 as r2r_dst1, dst2 as r2r_dst2, dst3 as r2r_dst3, dst4 as r2r_dst4,
};

/// DST type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DstType {
    /// DST-I (RODFT00)
    I,
    /// DST-II (RODFT10)
    II,
    /// DST-III (RODFT01)
    III,
    /// DST-IV (RODFT11)
    IV,
}

/// Execute DST-I.
pub fn dst_i<T: Float>(input: &[T], output: &mut [T]) {
    r2r_dst1(input, output);
}

/// Execute DST-II.
pub fn dst_ii<T: Float>(input: &[T], output: &mut [T]) {
    r2r_dst2(input, output);
}

/// Execute DST-III.
pub fn dst_iii<T: Float>(input: &[T], output: &mut [T]) {
    r2r_dst3(input, output);
}

/// Execute DST-IV.
pub fn dst_iv<T: Float>(input: &[T], output: &mut [T]) {
    r2r_dst4(input, output);
}
