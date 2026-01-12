//! Distributed transpose operations.
//!
//! Implements the all-to-all transpose required for distributed FFT.

use mpi::datatype::Equivalence;
use mpi::topology::Communicator;

use crate::kernel::Complex;

use super::distribution::LocalPartition;
use super::error::MpiError;
use super::pool::{MpiFloat, MpiPool};

/// Perform a distributed transpose of a 2D array.
///
/// Transforms from row-major distribution to column-major distribution.
/// After transpose, each process owns a contiguous block of columns instead of rows.
///
/// # Arguments
/// * `pool` - MPI pool
/// * `input` - Local input data (local_n0 x n1 elements)
/// * `output` - Local output data (n0 x local_n1 elements after transpose)
/// * `n0` - Global number of rows
/// * `n1` - Global number of columns
/// * `local_n0` - Number of rows owned by this process
/// * `_local_0_start` - Global starting row for this process (unused but kept for API consistency)
///
/// # Errors
/// Returns `MpiError::SizeMismatch` if input/output buffers are too small.
pub fn distributed_transpose<T, C>(
    pool: &MpiPool<C>,
    input: &[Complex<T>],
    output: &mut [Complex<T>],
    n0: usize,
    n1: usize,
    local_n0: usize,
    _local_0_start: usize,
) -> Result<(), MpiError>
where
    T: MpiFloat,
    C: Communicator,
    Complex<T>: Equivalence,
{
    let num_procs = pool.size();
    let rank = pool.rank();

    // Verify input size
    let expected_input_size = local_n0 * n1;
    if input.len() < expected_input_size {
        return Err(MpiError::SizeMismatch {
            expected: expected_input_size,
            actual: input.len(),
        });
    }

    // Calculate local partition for transposed layout
    let transposed_partition = LocalPartition::new(n1, num_procs, rank);
    let local_n1 = transposed_partition.local_n;

    // Verify output size
    let expected_output_size = n0 * local_n1;
    if output.len() < expected_output_size {
        return Err(MpiError::SizeMismatch {
            expected: expected_output_size,
            actual: output.len(),
        });
    }

    // Calculate send counts and displacements
    let mut send_counts = Vec::with_capacity(num_procs);
    let mut send_displs = Vec::with_capacity(num_procs);
    let mut recv_counts = Vec::with_capacity(num_procs);
    let mut recv_displs = Vec::with_capacity(num_procs);

    let mut send_offset = 0;
    let mut recv_offset = 0;

    for p in 0..num_procs {
        // For process p, we send local_n0 * partition_p.local_n elements
        let partition_p = LocalPartition::new(n1, num_procs, p);

        // Send count: elements from our rows destined for process p's columns
        let send_count = local_n0 * partition_p.local_n;
        send_counts.push(send_count as i32);
        send_displs.push(send_offset as i32);
        send_offset += send_count;

        // Receive count: elements from process p's rows for our columns
        let source_partition = LocalPartition::new(n0, num_procs, p);
        let recv_count = source_partition.local_n * local_n1;
        recv_counts.push(recv_count as i32);
        recv_displs.push(recv_offset as i32);
        recv_offset += recv_count;
    }

    // Pack data for sending
    // Input is stored as: input[row * n1 + col] for row in [0, local_n0), col in [0, n1)
    // We need to pack it so elements for each destination process are contiguous
    let total_send = send_offset;
    let mut send_buffer = vec![Complex::<T>::zero(); total_send];

    let mut buf_offset = 0;
    for p in 0..num_procs {
        let partition_p = LocalPartition::new(n1, num_procs, p);
        // Pack columns [partition_p.local_start, partition_p.local_start + partition_p.local_n)
        // from all local rows
        for row in 0..local_n0 {
            for col in 0..partition_p.local_n {
                let global_col = partition_p.local_start + col;
                send_buffer[buf_offset] = input[row * n1 + global_col];
                buf_offset += 1;
            }
        }
    }

    // Receive buffer
    let total_recv = recv_offset;
    let mut recv_buffer = vec![Complex::<T>::zero(); total_recv];

    // Perform all-to-all variable
    pool.all_to_all_v_complex(
        &send_buffer,
        &send_counts,
        &send_displs,
        &mut recv_buffer,
        &recv_counts,
        &recv_displs,
    )?;

    // Unpack received data into output
    // Output should be: output[col * n0 + row] for col in [0, local_n1), row in [0, n0)
    // But we receive from each process p: source_partition.local_n rows x local_n1 cols
    let mut recv_idx = 0;
    for p in 0..num_procs {
        let source_partition = LocalPartition::new(n0, num_procs, p);
        // Data from process p: rows [source_partition.local_start, ...) x our local columns
        for src_row in 0..source_partition.local_n {
            let global_row = source_partition.local_start + src_row;
            for local_col in 0..local_n1 {
                // Output in transposed layout: [local_col][global_row]
                // Using row-major storage for the transposed result
                output[local_col * n0 + global_row] = recv_buffer[recv_idx];
                recv_idx += 1;
            }
        }
    }

    Ok(())
}

/// Perform in-place distributed transpose.
///
/// The buffer must be large enough to hold max(input_size, output_size).
#[allow(dead_code)]
pub fn distributed_transpose_inplace<T, C>(
    pool: &MpiPool<C>,
    data: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
    n0: usize,
    n1: usize,
    local_n0: usize,
    local_0_start: usize,
) -> Result<(), MpiError>
where
    T: MpiFloat,
    C: Communicator,
    Complex<T>: Equivalence,
{
    // Use scratch buffer for output, then copy back
    distributed_transpose(pool, data, scratch, n0, n1, local_n0, local_0_start)?;

    // Calculate transposed local size
    let transposed_partition = LocalPartition::new(n1, pool.size(), pool.rank());
    let local_n1 = transposed_partition.local_n;
    let output_size = n0 * local_n1;

    // Copy back to data
    data[..output_size].copy_from_slice(&scratch[..output_size]);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full transpose tests require MPI, so we test helper functions here

    #[test]
    fn test_partition_calculation() {
        // Verify partition calculations are consistent
        let n0 = 16;
        let n1 = 8;
        let num_procs = 4;

        let mut total_send = 0;
        let mut total_recv = 0;

        for rank in 0..num_procs {
            let local_partition = LocalPartition::new(n0, num_procs, rank);
            let transposed_partition = LocalPartition::new(n1, num_procs, rank);

            let local_elements = local_partition.local_n * n1;
            let transposed_elements = n0 * transposed_partition.local_n;

            // Each process should send and receive the same total
            total_send += local_elements;
            total_recv += transposed_elements;
        }

        assert_eq!(total_send, n0 * n1);
        assert_eq!(total_recv, n0 * n1);
    }
}
