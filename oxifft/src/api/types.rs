//! Public type definitions for the OxiFFT API.

use core::ops::BitOr;

/// Transform direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Forward transform (analysis): time domain → frequency domain
    Forward,
    /// Backward/Inverse transform (synthesis): frequency domain → time domain
    Backward,
}

impl Direction {
    /// Get the sign for the exponential: -1 for forward, +1 for backward.
    #[must_use]
    pub const fn sign(self) -> i32 {
        match self {
            Self::Forward => -1,
            Self::Backward => 1,
        }
    }
}

/// Planning flags that control algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Flags(u32);

impl Flags {
    /// Use heuristics only, don't measure (fastest planning, may not be optimal).
    pub const ESTIMATE: Self = Self(0);

    /// Measure a few algorithms and pick the best.
    pub const MEASURE: Self = Self(1 << 0);

    /// Try harder to find the optimal algorithm.
    pub const PATIENT: Self = Self(1 << 1);

    /// Try all possible algorithms exhaustively.
    pub const EXHAUSTIVE: Self = Self(1 << 2);

    /// Preserve input array (default behavior).
    pub const PRESERVE_INPUT: Self = Self(1 << 3);

    /// Allow destroying input array for potentially better performance.
    pub const DESTROY_INPUT: Self = Self(1 << 4);

    /// Plan for unaligned data.
    pub const UNALIGNED: Self = Self(1 << 5);

    /// Check if MEASURE flag is set.
    #[must_use]
    pub const fn is_measure(self) -> bool {
        self.0 & Self::MEASURE.0 != 0
    }

    /// Check if PATIENT flag is set.
    #[must_use]
    pub const fn is_patient(self) -> bool {
        self.0 & Self::PATIENT.0 != 0
    }

    /// Check if EXHAUSTIVE flag is set.
    #[must_use]
    pub const fn is_exhaustive(self) -> bool {
        self.0 & Self::EXHAUSTIVE.0 != 0
    }

    /// Check if input destruction is allowed.
    #[must_use]
    pub const fn can_destroy_input(self) -> bool {
        self.0 & Self::DESTROY_INPUT.0 != 0
    }
}

impl BitOr for Flags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// Real-to-real transform kind (DCT/DST variants).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum R2rKind {
    /// DCT-I (REDFT00)
    DctI,
    /// DCT-II (REDFT10) - "the DCT"
    DctII,
    /// DCT-III (REDFT01) - inverse of DCT-II
    DctIII,
    /// DCT-IV (REDFT11)
    DctIV,
    /// DST-I (RODFT00)
    DstI,
    /// DST-II (RODFT10)
    DstII,
    /// DST-III (RODFT01)
    DstIII,
    /// DST-IV (RODFT11)
    DstIV,
    /// Discrete Hartley Transform
    Dht,
    /// Half-complex to real (used internally)
    Hc2r,
    /// Real to half-complex (used internally)
    R2hc,
}
