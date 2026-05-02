// ndarray integration — opt-in via `ndarray` cargo feature
#[cfg(feature = "ndarray")]
pub mod ndarray_ext;

#[cfg(feature = "ndarray")]
pub use ndarray_ext::{FftExt, NdarrayFftError, RealFftExt};

// Test module
#[cfg(all(test, feature = "ndarray"))]
mod ndarray_tests;
