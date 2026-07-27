//! Public type definitions for the OxiFFT API.

use core::fmt;
use core::ops::BitOr;

/// Transform direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Direction {
    /// Forward transform (analysis): time domain → frequency domain
    Forward,
    /// Backward/Inverse transform (synthesis): frequency domain → time domain
    Backward,
}

impl Direction {
    /// Get the sign for the exponential: -1 for forward, +1 for backward.
    ///
    /// This matches the FFTW convention where the forward (analysis) transform
    /// uses a negative exponent and the backward (synthesis) transform uses a
    /// positive exponent.
    #[must_use]
    pub const fn sign(self) -> i32 {
        match self {
            Self::Forward => -1,
            Self::Backward => 1,
        }
    }
}

impl TryFrom<i32> for Direction {
    type Error = InvalidDirection;

    /// Construct a `Direction` from its FFTW-style sign convention:
    /// `-1` means [`Direction::Forward`] (negative exponent) and
    /// `+1` means [`Direction::Backward`] (positive exponent).
    ///
    /// # Errors
    ///
    /// Returns [`InvalidDirection`] if `value` is not `-1` or `1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxifft::Direction;
    ///
    /// let fwd = Direction::try_from(-1_i32).expect("forward");
    /// assert_eq!(fwd, Direction::Forward);
    ///
    /// let bwd = Direction::try_from(1_i32).expect("backward");
    /// assert_eq!(bwd, Direction::Backward);
    ///
    /// assert!(Direction::try_from(0_i32).is_err());
    /// ```
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            -1 => Ok(Self::Forward),
            1 => Ok(Self::Backward),
            n => Err(InvalidDirection(n)),
        }
    }
}

/// Error returned when an integer cannot be converted to a [`Direction`].
///
/// Valid values are `-1` (forward, negative exponent) and `1` (backward, positive exponent),
/// following the FFTW sign convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidDirection(pub i32);

impl fmt::Display for InvalidDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid direction value {}: expected -1 (forward) or 1 (backward)",
            self.0
        )
    }
}

impl core::error::Error for InvalidDirection {}

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

    /// Only use pre-existing wisdom; do not measure.
    ///
    /// Matches FFTW's `FFTW_WISDOM_ONLY`: when set, plan construction succeeds
    /// only if a matching wisdom entry (build-time baseline or runtime cache)
    /// already exists for the transform size. If no wisdom is found, planning
    /// **fails** (returns `None`) rather than falling back to the heuristic or
    /// running a benchmark. This lets callers guarantee that every plan they
    /// build is backed by measured data.
    pub const WISDOM_ONLY: Self = Self(1 << 6);

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

    /// Check if WISDOM_ONLY flag is set.
    ///
    /// See [`Flags::WISDOM_ONLY`] for the semantics.
    #[must_use]
    pub const fn is_wisdom_only(self) -> bool {
        self.0 & Self::WISDOM_ONLY.0 != 0
    }
}

impl BitOr for Flags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

// NOTE: The public real-to-real transform kind is
// [`crate::rdft::solvers::R2rKind`] (FFTW's `REDFT00..RODFT11`/`DHT` naming),
// which is the enum every `R2rPlan*` constructor actually accepts and which the
// crate root re-exports as `oxifft::R2rKind`. A second, differently-named
// `R2rKind` (DctI/DctII/.../Hc2r/R2hc) previously lived here but was never wired
// into any API — it made `oxifft::R2rKind` a type no R2r plan could accept — so
// it was removed in 0.4.0 to leave a single, usable kind enum.
