//! Data distribution types for distributed FFT.

/// How data is distributed across processes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Distribution {
    /// Slab decomposition (1D distribution along first dimension).
    /// Each process owns a contiguous block of rows.
    #[default]
    Slab,
    /// Pencil decomposition (2D distribution).
    /// Used for 3D transforms with more parallelism.
    Pencil,
}

/// Describes the local partition of data owned by a process.
#[derive(Debug, Clone, Copy)]
pub struct LocalPartition {
    /// Local number of elements in the distributed dimension.
    pub local_n: usize,
    /// Global starting index for this process's data.
    pub local_start: usize,
    /// Global size of the distributed dimension.
    pub global_n: usize,
    /// Total number of processes.
    pub num_procs: usize,
    /// This process's rank (0-indexed).
    pub proc_idx: usize,
}

impl LocalPartition {
    /// Calculate local partition for a given process.
    ///
    /// Uses block distribution: each process gets n / num_procs elements,
    /// with the first (n % num_procs) processes getting one extra element.
    pub fn new(global_n: usize, num_procs: usize, proc_idx: usize) -> Self {
        let base_size = global_n / num_procs;
        let remainder = global_n % num_procs;

        // First `remainder` processes get one extra element
        let (local_n, local_start) = if proc_idx < remainder {
            let local_n = base_size + 1;
            let local_start = proc_idx * (base_size + 1);
            (local_n, local_start)
        } else {
            let local_n = base_size;
            let local_start = remainder * (base_size + 1) + (proc_idx - remainder) * base_size;
            (local_n, local_start)
        };

        Self {
            local_n,
            local_start,
            global_n,
            num_procs,
            proc_idx,
        }
    }

    /// Calculate total local allocation size for a multi-dimensional array.
    ///
    /// For a distributed array where the first dimension is distributed,
    /// returns local_n * remaining_size.
    pub fn alloc_size(&self, remaining_size: usize) -> usize {
        self.local_n * remaining_size
    }

    /// Check if this process has any data.
    pub fn has_data(&self) -> bool {
        self.local_n > 0
    }

    /// Get the global index range for this process's data.
    pub fn global_range(&self) -> core::ops::Range<usize> {
        self.local_start..self.local_start + self.local_n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_partition_even() {
        // 16 elements across 4 processes
        let p0 = LocalPartition::new(16, 4, 0);
        let p1 = LocalPartition::new(16, 4, 1);
        let p2 = LocalPartition::new(16, 4, 2);
        let p3 = LocalPartition::new(16, 4, 3);

        assert_eq!(p0.local_n, 4);
        assert_eq!(p0.local_start, 0);

        assert_eq!(p1.local_n, 4);
        assert_eq!(p1.local_start, 4);

        assert_eq!(p2.local_n, 4);
        assert_eq!(p2.local_start, 8);

        assert_eq!(p3.local_n, 4);
        assert_eq!(p3.local_start, 12);
    }

    #[test]
    fn test_local_partition_uneven() {
        // 10 elements across 4 processes: 3+3+2+2
        let p0 = LocalPartition::new(10, 4, 0);
        let p1 = LocalPartition::new(10, 4, 1);
        let p2 = LocalPartition::new(10, 4, 2);
        let p3 = LocalPartition::new(10, 4, 3);

        assert_eq!(p0.local_n, 3);
        assert_eq!(p0.local_start, 0);

        assert_eq!(p1.local_n, 3);
        assert_eq!(p1.local_start, 3);

        assert_eq!(p2.local_n, 2);
        assert_eq!(p2.local_start, 6);

        assert_eq!(p3.local_n, 2);
        assert_eq!(p3.local_start, 8);

        // Total should equal global size
        assert_eq!(p0.local_n + p1.local_n + p2.local_n + p3.local_n, 10);
    }

    #[test]
    fn test_local_partition_more_procs_than_elements() {
        // 3 elements across 5 processes
        let p0 = LocalPartition::new(3, 5, 0);
        let p1 = LocalPartition::new(3, 5, 1);
        let p2 = LocalPartition::new(3, 5, 2);
        let p3 = LocalPartition::new(3, 5, 3);
        let p4 = LocalPartition::new(3, 5, 4);

        assert_eq!(p0.local_n, 1);
        assert_eq!(p1.local_n, 1);
        assert_eq!(p2.local_n, 1);
        assert_eq!(p3.local_n, 0);
        assert_eq!(p4.local_n, 0);

        assert!(p0.has_data());
        assert!(p1.has_data());
        assert!(p2.has_data());
        assert!(!p3.has_data());
        assert!(!p4.has_data());
    }

    #[test]
    fn test_global_range() {
        let p = LocalPartition::new(100, 4, 2);
        let range = p.global_range();
        assert_eq!(range, 50..75);
    }

    #[test]
    fn test_alloc_size() {
        // 2D array: 100 x 64, distributed along first dimension
        let p = LocalPartition::new(100, 4, 0);
        assert_eq!(p.alloc_size(64), 25 * 64);
    }
}
