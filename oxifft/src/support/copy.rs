//! Optimized copy operations.

use crate::kernel::{Complex, Float};

/// Copy 1D array.
#[inline]
pub fn copy_1d<T: Float>(src: &[Complex<T>], dst: &mut [Complex<T>]) {
    debug_assert_eq!(src.len(), dst.len());
    dst.copy_from_slice(src);
}

/// Copy with stride (gather/scatter).
pub fn copy_strided<T: Float>(
    src: *const Complex<T>,
    src_stride: isize,
    dst: *mut Complex<T>,
    dst_stride: isize,
    count: usize,
) {
    unsafe {
        for i in 0..count {
            let src_idx = (i as isize * src_stride) as usize;
            let dst_idx = (i as isize * dst_stride) as usize;
            *dst.add(dst_idx) = *src.add(src_idx);
        }
    }
}

/// Copy 2D array (row by row).
pub fn copy_2d<T: Float>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    rows: usize,
    cols: usize,
    src_row_stride: usize,
    dst_row_stride: usize,
) {
    for row in 0..rows {
        let src_start = row * src_row_stride;
        let dst_start = row * dst_row_stride;
        dst[dst_start..dst_start + cols].copy_from_slice(&src[src_start..src_start + cols]);
    }
}
