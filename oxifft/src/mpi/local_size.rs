//! Local size calculation for distributed FFT.
//!
//! These functions calculate how much local memory each process needs
//! to allocate for distributed FFT operations.

use mpi::topology::Communicator;

use super::distribution::LocalPartition;
use super::pool::MpiPool;

/// Calculate local size for a 2D distributed FFT.
///
/// Returns `(local_n0, local_0_start, alloc_local)`:
/// - `local_n0`: Number of rows owned by this process
/// - `local_0_start`: Global starting row index
/// - `alloc_local`: Total number of complex elements to allocate
///
/// This is equivalent to FFTW's `fftw_mpi_local_size_2d`.
///
/// # Arguments
/// * `n0` - Number of rows (first dimension, distributed)
/// * `n1` - Number of columns (second dimension, local)
/// * `pool` - MPI pool
pub fn local_size_2d<C: Communicator>(
    n0: usize,
    n1: usize,
    pool: &MpiPool<C>,
) -> (usize, usize, usize) {
    let partition = pool.local_partition(n0);

    // For transposed intermediate results, we may need extra space
    // In FFTW, the allocation is max(local_n0 * n1, local_n1 * n0)
    // where local_n1 would be the partition in the transposed layout
    let transposed_partition = LocalPartition::new(n1, pool.size(), pool.rank());

    let normal_alloc = partition.local_n * n1;
    let transposed_alloc = transposed_partition.local_n * n0;
    let alloc_local = normal_alloc.max(transposed_alloc);

    (partition.local_n, partition.local_start, alloc_local)
}

/// Calculate local size for a 3D distributed FFT.
///
/// Returns `(local_n0, local_0_start, alloc_local)`:
/// - `local_n0`: Number of planes owned by this process
/// - `local_0_start`: Global starting plane index
/// - `alloc_local`: Total number of complex elements to allocate
///
/// This is equivalent to FFTW's `fftw_mpi_local_size_3d`.
pub fn local_size_3d<C: Communicator>(
    n0: usize,
    n1: usize,
    n2: usize,
    pool: &MpiPool<C>,
) -> (usize, usize, usize) {
    let partition = pool.local_partition(n0);

    // For transposed intermediate, partition n1 instead
    let transposed_partition = LocalPartition::new(n1, pool.size(), pool.rank());

    let normal_alloc = partition.local_n * n1 * n2;
    let transposed_alloc = n0 * transposed_partition.local_n * n2;
    let alloc_local = normal_alloc.max(transposed_alloc);

    (partition.local_n, partition.local_start, alloc_local)
}

/// Calculate local size for an N-D distributed FFT.
///
/// Returns `(local_n0, local_0_start, alloc_local)`:
/// - `local_n0`: Number of hyperplanes owned by this process in the first dimension
/// - `local_0_start`: Global starting index
/// - `alloc_local`: Total number of complex elements to allocate
///
/// This is equivalent to FFTW's `fftw_mpi_local_size_many`.
///
/// # Arguments
/// * `dims` - Array of dimension sizes (first dimension is distributed)
/// * `pool` - MPI pool
pub fn local_size_nd<C: Communicator>(dims: &[usize], pool: &MpiPool<C>) -> (usize, usize, usize) {
    if dims.is_empty() {
        return (0, 0, 0);
    }

    if dims.len() == 1 {
        // 1D case: just partition the single dimension
        let partition = pool.local_partition(dims[0]);
        return (partition.local_n, partition.local_start, partition.local_n);
    }

    let n0 = dims[0];
    let partition = pool.local_partition(n0);

    // Product of remaining dimensions
    let remaining_product: usize = dims[1..].iter().product();

    // For transposed intermediate, partition dims[1] instead
    let n1 = dims[1];
    let transposed_partition = LocalPartition::new(n1, pool.size(), pool.rank());
    let transposed_remaining: usize = core::iter::once(n0)
        .chain(dims[2..].iter().copied())
        .product();

    let normal_alloc = partition.local_n * remaining_product;
    let transposed_alloc = transposed_partition.local_n * transposed_remaining;
    let alloc_local = normal_alloc.max(transposed_alloc);

    (partition.local_n, partition.local_start, alloc_local)
}

/// Get local partition info for both normal and transposed layouts.
///
/// Returns `(local_n0, local_0_start, local_n1, local_1_start)`.
/// Useful when working with TRANSPOSED_OUT/TRANSPOSED_IN flags.
#[allow(dead_code)]
pub fn local_size_2d_transposed<C: Communicator>(
    n0: usize,
    n1: usize,
    pool: &MpiPool<C>,
) -> (usize, usize, usize, usize) {
    let partition_0 = pool.local_partition(n0);
    let partition_1 = pool.local_partition(n1);

    (
        partition_0.local_n,
        partition_0.local_start,
        partition_1.local_n,
        partition_1.local_start,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_partition_direct() {
        // Test LocalPartition without MPI
        let p = LocalPartition::new(100, 4, 0);
        assert_eq!(p.local_n, 25);
        assert_eq!(p.local_start, 0);
    }

    #[test]
    fn test_local_partition_distribution() {
        // Verify all elements are accounted for
        let n = 100;
        let num_procs = 4;
        let total: usize = (0..num_procs)
            .map(|rank| LocalPartition::new(n, num_procs, rank).local_n)
            .sum();
        assert_eq!(total, n);
    }

    #[test]
    fn test_local_partition_uneven() {
        // 7 elements across 3 processes: 3+2+2
        let p0 = LocalPartition::new(7, 3, 0);
        let p1 = LocalPartition::new(7, 3, 1);
        let p2 = LocalPartition::new(7, 3, 2);

        assert_eq!(p0.local_n, 3);
        assert_eq!(p1.local_n, 2);
        assert_eq!(p2.local_n, 2);

        assert_eq!(p0.local_start, 0);
        assert_eq!(p1.local_start, 3);
        assert_eq!(p2.local_start, 5);
    }
}
