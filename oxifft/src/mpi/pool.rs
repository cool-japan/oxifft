//! MPI process pool for distributed computation.

use mpi::collective::CommunicatorCollectives;
use mpi::topology::Communicator;

use crate::kernel::{Complex, Float};

use super::distribution::LocalPartition;
use super::error::MpiError;

/// Trait for float types that can be used with MPI.
pub trait MpiFloat: Float + mpi::datatype::Equivalence {}

impl MpiFloat for f32 {}
impl MpiFloat for f64 {}

/// MPI process pool for distributed FFT computation.
///
/// Wraps an MPI communicator and provides utilities for distributed operations.
pub struct MpiPool<C: Communicator> {
    /// The MPI communicator.
    comm: C,
    /// Number of processes.
    size: i32,
    /// This process's rank.
    rank: i32,
}

impl<C: Communicator> MpiPool<C> {
    /// Create a new MPI pool from a communicator.
    pub fn new(comm: C) -> Self {
        let size = comm.size();
        let rank = comm.rank();
        Self { comm, size, rank }
    }

    /// Get the number of processes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size as usize
    }

    /// Get this process's rank.
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank as usize
    }

    /// Check if this is the root process (rank 0).
    #[inline]
    pub fn is_root(&self) -> bool {
        self.rank == 0
    }

    /// Get a reference to the communicator.
    pub fn comm(&self) -> &C {
        &self.comm
    }

    /// Calculate local partition for a given dimension.
    pub fn local_partition(&self, global_n: usize) -> LocalPartition {
        LocalPartition::new(global_n, self.size(), self.rank())
    }

    /// Barrier synchronization across all processes.
    pub fn barrier(&self) {
        self.comm.barrier();
    }
}

/// Operations on MPI pool with complex data.
impl<C: Communicator> MpiPool<C> {
    /// All-to-all communication for complex data.
    ///
    /// Each process sends `count` elements to each other process.
    /// Total send/receive size is `count * num_processes`.
    ///
    /// # Errors
    /// Returns `MpiError::SizeMismatch` if buffers are too small.
    pub fn all_to_all_complex<T: MpiFloat>(
        &self,
        send_data: &[Complex<T>],
        recv_data: &mut [Complex<T>],
        count: usize,
    ) -> Result<(), MpiError>
    where
        Complex<T>: mpi::datatype::Equivalence,
    {
        let expected_len = count * self.size();
        if send_data.len() < expected_len {
            return Err(MpiError::SizeMismatch {
                expected: expected_len,
                actual: send_data.len(),
            });
        }
        if recv_data.len() < expected_len {
            return Err(MpiError::SizeMismatch {
                expected: expected_len,
                actual: recv_data.len(),
            });
        }

        self.comm
            .all_to_all_into(&send_data[..expected_len], &mut recv_data[..expected_len]);
        Ok(())
    }

    /// Variable all-to-all communication for complex data.
    ///
    /// Each process can send different amounts to different processes.
    ///
    /// # Errors
    /// Returns `MpiError` on communication failure.
    pub fn all_to_all_v_complex<T: MpiFloat>(
        &self,
        send_data: &[Complex<T>],
        send_counts: &[i32],
        send_displs: &[i32],
        recv_data: &mut [Complex<T>],
        recv_counts: &[i32],
        recv_displs: &[i32],
    ) -> Result<(), MpiError>
    where
        Complex<T>: mpi::datatype::Equivalence,
    {
        use mpi::datatype::PartitionMut;

        // Create partitions from counts and displacements
        let send_partition =
            mpi::datatype::Partition::new(send_data, send_counts.to_vec(), send_displs.to_vec());
        let mut recv_partition =
            PartitionMut::new(recv_data, recv_counts.to_vec(), recv_displs.to_vec());

        // Use all_to_all_varcount for variable-sized messages
        self.comm
            .all_to_all_varcount_into(&send_partition, &mut recv_partition);

        Ok(())
    }

    /// Broadcast data from root to all processes.
    ///
    /// # Errors
    /// Returns `MpiError` on communication failure.
    pub fn broadcast_complex<T: MpiFloat>(
        &self,
        data: &mut [Complex<T>],
        root: usize,
    ) -> Result<(), MpiError>
    where
        Complex<T>: mpi::datatype::Equivalence,
    {
        use mpi::collective::Root;

        let root_process = self.comm.process_at_rank(root as i32);
        root_process.broadcast_into(data);
        Ok(())
    }

    /// All-gather operation: gather data from all processes.
    ///
    /// # Errors
    /// Returns `MpiError::SizeMismatch` if receive buffer is too small.
    pub fn all_gather_complex<T: MpiFloat>(
        &self,
        send_data: &[Complex<T>],
        recv_data: &mut [Complex<T>],
    ) -> Result<(), MpiError>
    where
        Complex<T>: mpi::datatype::Equivalence,
    {
        let expected_recv_len = send_data.len() * self.size();
        if recv_data.len() < expected_recv_len {
            return Err(MpiError::SizeMismatch {
                expected: expected_recv_len,
                actual: recv_data.len(),
            });
        }

        self.comm.all_gather_into(send_data, recv_data);
        Ok(())
    }

    /// Variable all-gather operation.
    ///
    /// # Errors
    /// Returns `MpiError` on communication failure.
    pub fn all_gather_v_complex<T: MpiFloat>(
        &self,
        send_data: &[Complex<T>],
        recv_data: &mut [Complex<T>],
        recv_counts: &[i32],
        recv_displs: &[i32],
    ) -> Result<(), MpiError>
    where
        Complex<T>: mpi::datatype::Equivalence,
    {
        use mpi::datatype::PartitionMut;

        let mut recv_partition =
            PartitionMut::new(recv_data, recv_counts.to_vec(), recv_displs.to_vec());

        self.comm
            .all_gather_varcount_into(send_data, &mut recv_partition);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_partition() {
        // Test without MPI - just test the LocalPartition directly
        let partition = LocalPartition::new(100, 4, 1);
        assert_eq!(partition.local_n, 25);
        assert_eq!(partition.local_start, 25);
    }
}
