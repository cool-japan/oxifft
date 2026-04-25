//! Matrix transpose utilities.
//!
//! Provides efficient transpose operations for 2D and higher-dimensional FFTs.
//! Includes:
//! - Simple in-place transpose for square matrices
//! - Out-of-place transpose for rectangular matrices
//! - Cache-blocked transpose for improved cache performance
//! - In-place non-square transpose using cycle-following algorithm

use crate::kernel::{Complex, Float};
use crate::prelude::*;

/// Cache block size for blocked transpose.
/// Chosen to fit L1 cache (typically 32KB).
const BLOCK_SIZE: usize = 32;

/// In-place square matrix transpose.
pub fn transpose_square<T: Float>(data: &mut [Complex<T>], n: usize) {
    debug_assert_eq!(data.len(), n * n);

    for i in 0..n {
        for j in (i + 1)..n {
            data.swap(i * n + j, j * n + i);
        }
    }
}

/// In-place square matrix transpose with cache blocking.
///
/// Uses cache-blocked algorithm for better locality on large matrices.
pub fn transpose_square_blocked<T: Float>(data: &mut [Complex<T>], n: usize) {
    debug_assert_eq!(data.len(), n * n);

    if n <= BLOCK_SIZE {
        // Use simple algorithm for small matrices
        transpose_square(data, n);
        return;
    }

    // Process blocks along the diagonal and above
    for bi in (0..n).step_by(BLOCK_SIZE) {
        let block_i_end = (bi + BLOCK_SIZE).min(n);

        // Diagonal block: transpose within the block
        for i in bi..block_i_end {
            for j in (i + 1)..block_i_end {
                data.swap(i * n + j, j * n + i);
            }
        }

        // Off-diagonal blocks: swap pairs of blocks
        for bj in ((bi + BLOCK_SIZE)..n).step_by(BLOCK_SIZE) {
            let block_j_end = (bj + BLOCK_SIZE).min(n);

            for i in bi..block_i_end {
                for j in bj..block_j_end {
                    data.swap(i * n + j, j * n + i);
                }
            }
        }
    }
}

/// Out-of-place matrix transpose.
pub fn transpose<T: Float>(src: &[Complex<T>], dst: &mut [Complex<T>], rows: usize, cols: usize) {
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert_eq!(dst.len(), rows * cols);

    for i in 0..rows {
        for j in 0..cols {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

/// Out-of-place matrix transpose with cache blocking.
///
/// Uses cache-blocked algorithm for better locality on large matrices.
/// Input is rows×cols, output is cols×rows.
#[allow(clippy::suspicious_operation_groupings)] // reason: matrix index arithmetic 'i * cols + j' and 'j * rows + i' are standard and correct, not suspicious
pub fn transpose_blocked<T: Float>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert_eq!(dst.len(), rows * cols);

    // Check if total elements is small enough for direct (non-blocked) transpose
    if rows * cols <= BLOCK_SIZE * BLOCK_SIZE {
        transpose(src, dst, rows, cols);
        return;
    }

    // Process in blocks for cache efficiency
    for bi in (0..rows).step_by(BLOCK_SIZE) {
        let block_i_end = (bi + BLOCK_SIZE).min(rows);

        for bj in (0..cols).step_by(BLOCK_SIZE) {
            let block_j_end = (bj + BLOCK_SIZE).min(cols);

            // Transpose this block
            for i in bi..block_i_end {
                for j in bj..block_j_end {
                    dst[j * rows + i] = src[i * cols + j];
                }
            }
        }
    }
}

/// In-place transpose for non-square matrices.
///
/// Uses cycle-following algorithm. This is O(n) in the number of elements
/// but may have poor cache behavior for very large matrices.
///
/// Transposes a rows×cols matrix in-place to a cols×rows matrix.
pub fn transpose_inplace<T: Float>(data: &mut [Complex<T>], rows: usize, cols: usize) {
    let n = rows * cols;
    debug_assert_eq!(data.len(), n);

    if rows == cols {
        transpose_square(data, rows);
        return;
    }

    if n <= 1 {
        return;
    }

    // Use cycle-following algorithm
    // Each element at position i needs to move to position (i*cols) mod (n-1)
    // Exception: element 0 stays at 0, element n-1 stays at n-1

    let mut visited = vec![false; n];
    visited[0] = true;
    visited[n - 1] = true;

    for start in 1..(n - 1) {
        if visited[start] {
            continue;
        }

        let temp = data[start];
        let mut current = start;

        loop {
            // Where does element at 'current' need to go?
            let next = (current * cols) % (n - 1);

            visited[current] = true;

            if next == start {
                // Cycle complete
                data[current] = temp;
                break;
            }

            // Move element from next to current
            data[current] = data[next];
            current = next;
        }
    }
}

/// Transpose a 3D array along specified axes.
///
/// Useful for optimizing 3D FFTs with better cache access patterns.
/// Swaps axes 1 and 2 (middle and inner dimensions).
pub fn transpose_3d_inner<T: Float>(
    data: &mut [Complex<T>],
    d0: usize, // outer dimension (unchanged)
    d1: usize, // middle dimension (swapped with d2)
    d2: usize, // inner dimension (swapped with d1)
) {
    debug_assert_eq!(data.len(), d0 * d1 * d2);

    let plane_size = d1 * d2;
    let mut buffer = vec![Complex::zero(); plane_size];

    for k in 0..d0 {
        let plane_start = k * plane_size;
        let plane = &mut data[plane_start..plane_start + plane_size];

        // Transpose this 2D plane from d1×d2 to d2×d1
        transpose(plane, &mut buffer, d1, d2);
        plane.copy_from_slice(&buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_square() {
        let mut data = [
            Complex::new(1.0_f64, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];

        transpose_square(&mut data, 2);

        assert_eq!(data[0].re, 1.0);
        assert_eq!(data[1].re, 3.0);
        assert_eq!(data[2].re, 2.0);
        assert_eq!(data[3].re, 4.0);
    }

    #[test]
    fn test_transpose_square_blocked() {
        let n = 64;
        let mut data: Vec<Complex<f64>> =
            (0..(n * n)).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut expected = data.clone();

        transpose_square(&mut expected, n);
        transpose_square_blocked(&mut data, n);

        for i in 0..data.len() {
            assert_eq!(data[i].re, expected[i].re);
        }
    }

    #[test]
    fn test_transpose_rectangular() {
        let src = [
            Complex::new(1.0_f64, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(6.0, 0.0),
        ];
        let mut dst = [Complex::zero(); 6];

        // 2 rows × 3 cols -> 3 rows × 2 cols
        transpose(&src, &mut dst, 2, 3);

        assert_eq!(dst[0].re, 1.0);
        assert_eq!(dst[1].re, 4.0);
        assert_eq!(dst[2].re, 2.0);
        assert_eq!(dst[3].re, 5.0);
        assert_eq!(dst[4].re, 3.0);
        assert_eq!(dst[5].re, 6.0);
    }

    #[test]
    fn test_transpose_blocked() {
        let rows = 64;
        let cols = 48;
        let src: Vec<Complex<f64>> = (0..(rows * cols))
            .map(|i| Complex::new(i as f64, 0.0))
            .collect();
        let mut dst_simple = vec![Complex::zero(); rows * cols];
        let mut dst_blocked = vec![Complex::zero(); rows * cols];

        transpose(&src, &mut dst_simple, rows, cols);
        transpose_blocked(&src, &mut dst_blocked, rows, cols);

        for i in 0..(rows * cols) {
            assert_eq!(dst_simple[i].re, dst_blocked[i].re);
        }
    }

    #[test]
    fn test_transpose_inplace() {
        let rows = 2;
        let cols = 3;
        let mut data = [
            Complex::new(1.0_f64, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(6.0, 0.0),
        ];

        transpose_inplace(&mut data, rows, cols);

        // After transpose: 3 rows × 2 cols
        assert_eq!(data[0].re, 1.0);
        assert_eq!(data[1].re, 4.0);
        assert_eq!(data[2].re, 2.0);
        assert_eq!(data[3].re, 5.0);
        assert_eq!(data[4].re, 3.0);
        assert_eq!(data[5].re, 6.0);
    }

    #[test]
    fn test_transpose_inplace_larger() {
        let rows = 3;
        let cols = 4;
        let mut data: Vec<Complex<f64>> = (0..(rows * cols))
            .map(|i| Complex::new(i as f64, 0.0))
            .collect();
        let mut expected = vec![Complex::zero(); rows * cols];
        transpose(&data.clone(), &mut expected, rows, cols);

        transpose_inplace(&mut data, rows, cols);

        for i in 0..data.len() {
            assert_eq!(data[i].re, expected[i].re);
        }
    }

    #[test]
    fn test_transpose_3d_inner() {
        // 2 planes of 3×4 matrices
        let d0 = 2;
        let d1 = 3;
        let d2 = 4;
        let mut data: Vec<Complex<f64>> = (0..(d0 * d1 * d2))
            .map(|i| Complex::new(i as f64, 0.0))
            .collect();

        transpose_3d_inner(&mut data, d0, d1, d2);

        // After transpose, each plane is now 4×3
        // First plane (originally rows 0,1,2 with 4 cols each)
        // Row 0: 0,1,2,3 -> Column 0: 0,4,8 (now row 0)
        assert_eq!(data[0].re, 0.0); // (0,0)
        assert_eq!(data[1].re, 4.0); // (0,1) was (1,0)
        assert_eq!(data[2].re, 8.0); // (0,2) was (2,0)
        assert_eq!(data[3].re, 1.0); // (1,0) was (0,1)
    }
}
