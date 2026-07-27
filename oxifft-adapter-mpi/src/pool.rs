//! MPI process pool for distributed computation.

use mpi::collective::CommunicatorCollectives;
use mpi::topology::Communicator;

use oxifft::kernel::{Complex, Float};

use super::distribution::LocalPartition;
use super::error::MpiError;

/// Trait for float types that can be used with MPI.
pub trait MpiFloat: Float + mpi::datatype::Equivalence {}

impl MpiFloat for f32 {}
impl MpiFloat for f64 {}

/// Reinterpret a slice of `Complex<T>` as a slice of its underlying scalars.
///
/// `Complex<T>` is `#[repr(C)] { re: T, im: T }`, so `N` complex values occupy
/// exactly `2 * N` contiguous `T` values with identical alignment. This lets the
/// distributed collectives operate on the built-in scalar MPI datatype
/// (`f32` / `f64`, which implement [`mpi::datatype::Equivalence`]) instead of on
/// `Complex<T>` directly.
///
/// This is the sound, self-contained alternative to implementing `Equivalence`
/// for `Complex<T>` — which the orphan rule forbids in this crate because both
/// `Complex` (from `oxifft`) and `Equivalence` (from `mpi`) are foreign types.
/// Treating a complex value as two contiguous reals is exactly correct for all
/// data-movement collectives used here (all-to-all, all-gather, broadcast); no
/// reductions are performed, so no complex-aware reduction datatype is required.
#[inline]
fn as_scalars<T: Float>(data: &[Complex<T>]) -> &[T] {
    // SAFETY: `Complex<T>` is `#[repr(C)]` with two `T` fields, so a slice of
    // `len` complex values is exactly `2 * len` contiguous `T` values with the
    // same alignment as `T`. The pointer is always non-null and aligned (slice
    // invariant), and `2 * len` cannot overflow because each element already
    // occupies `2 * size_of::<T>()` bytes of an existing allocation.
    unsafe { core::slice::from_raw_parts(data.as_ptr().cast::<T>(), data.len() * 2) }
}

/// Mutable counterpart of [`as_scalars`].
#[inline]
fn as_scalars_mut<T: Float>(data: &mut [Complex<T>]) -> &mut [T] {
    // SAFETY: see [`as_scalars`]; the mutable borrow is unique for the returned
    // slice's lifetime.
    unsafe { core::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<T>(), data.len() * 2) }
}

/// Double each `i32` count/displacement so it counts scalars rather than
/// complex elements, returning [`MpiError::CountOverflow`] if the doubled value
/// no longer fits in `i32`.
fn scalar_counts(counts: &[i32]) -> Result<Vec<i32>, MpiError> {
    counts
        .iter()
        .enumerate()
        .map(|(rank, &c)| {
            c.checked_mul(2).ok_or(MpiError::CountOverflow {
                count: (c as usize) * 2,
                rank,
            })
        })
        .collect()
}

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
    ) -> Result<(), MpiError> {
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

        let send = as_scalars(&send_data[..expected_len]);
        let recv = as_scalars_mut(&mut recv_data[..expected_len]);
        self.comm.all_to_all_into(send, recv);
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
    ) -> Result<(), MpiError> {
        use mpi::datatype::PartitionMut;

        // Complex counts/displacements refer to `Complex<T>` elements; the
        // underlying transfer is on scalars (2 per complex value), so double
        // every count and displacement.
        let send_counts = scalar_counts(send_counts)?;
        let send_displs = scalar_counts(send_displs)?;
        let recv_counts = scalar_counts(recv_counts)?;
        let recv_displs = scalar_counts(recv_displs)?;

        let send = as_scalars(send_data);
        let recv = as_scalars_mut(recv_data);

        // Create partitions from counts and displacements
        let send_partition = mpi::datatype::Partition::new(send, send_counts, send_displs);
        let mut recv_partition = PartitionMut::new(recv, recv_counts, recv_displs);

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
    ) -> Result<(), MpiError> {
        use mpi::collective::Root;

        let root_process = self.comm.process_at_rank(root as i32);
        root_process.broadcast_into(as_scalars_mut(data));
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
    ) -> Result<(), MpiError> {
        let expected_recv_len = send_data.len() * self.size();
        if recv_data.len() < expected_recv_len {
            return Err(MpiError::SizeMismatch {
                expected: expected_recv_len,
                actual: recv_data.len(),
            });
        }

        // `all_gather_into` requires `recv.len() == send.len() * size` exactly,
        // so slice the receive buffer to the expected length before gathering.
        let send = as_scalars(send_data);
        let recv = as_scalars_mut(&mut recv_data[..expected_recv_len]);
        self.comm.all_gather_into(send, recv);
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
    ) -> Result<(), MpiError> {
        use mpi::datatype::PartitionMut;

        let recv_counts = scalar_counts(recv_counts)?;
        let recv_displs = scalar_counts(recv_displs)?;

        let send = as_scalars(send_data);
        let recv = as_scalars_mut(recv_data);
        let mut recv_partition = PartitionMut::new(recv, recv_counts, recv_displs);

        self.comm
            .all_gather_varcount_into(send, &mut recv_partition);

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
