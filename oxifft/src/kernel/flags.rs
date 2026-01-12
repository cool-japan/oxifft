//! Planning flags matching FFTW conventions.

use core::ops::BitOr;

/// Planning flags that control algorithm selection.
///
/// These flags match FFTW's planning modes for API compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PlannerFlags(u32);

impl PlannerFlags {
    // Planning rigor flags (mutually exclusive)
    /// Use heuristics only, fastest planning.
    pub const ESTIMATE: Self = Self(1 << 6);
    /// Measure a few algorithms.
    pub const MEASURE: Self = Self(0);
    /// More thorough search.
    pub const PATIENT: Self = Self(1 << 5);
    /// Try everything exhaustively.
    pub const EXHAUSTIVE: Self = Self(1 << 3);

    // Algorithm restriction flags
    /// Restrict to out-of-place transforms.
    pub const DESTROY_INPUT: Self = Self(1 << 0);
    /// Preserve input array.
    pub const PRESERVE_INPUT: Self = Self(1 << 4);
    /// Data is unaligned.
    pub const UNALIGNED: Self = Self(1 << 1);

    // Wisdom flags
    /// Only use wisdom, don't measure.
    pub const WISDOM_ONLY: Self = Self(1 << 21);

    /// Check if ESTIMATE mode.
    #[must_use]
    pub const fn is_estimate(self) -> bool {
        self.0 & Self::ESTIMATE.0 != 0
    }

    /// Check if PATIENT mode.
    #[must_use]
    pub const fn is_patient(self) -> bool {
        self.0 & Self::PATIENT.0 != 0
    }

    /// Check if EXHAUSTIVE mode.
    #[must_use]
    pub const fn is_exhaustive(self) -> bool {
        self.0 & Self::EXHAUSTIVE.0 != 0
    }

    /// Check if input destruction is allowed.
    #[must_use]
    pub const fn can_destroy_input(self) -> bool {
        self.0 & Self::DESTROY_INPUT.0 != 0
    }

    /// Check if wisdom-only mode.
    #[must_use]
    pub const fn is_wisdom_only(self) -> bool {
        self.0 & Self::WISDOM_ONLY.0 != 0
    }

    /// Check if MEASURE mode (default planning mode, benchmarks algorithms).
    #[must_use]
    pub const fn is_measure(self) -> bool {
        // MEASURE is the default (0), so it's active when none of ESTIMATE/PATIENT/EXHAUSTIVE are set
        !self.is_estimate() && !self.is_patient() && !self.is_exhaustive()
    }

    /// Get raw flag bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Create from raw flag bits.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }
}

impl BitOr for PlannerFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}
