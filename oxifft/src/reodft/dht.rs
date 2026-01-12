//! Discrete Hartley Transform.
//!
//! Re-exports the implementation in `rdft::solvers::r2r`.

use crate::kernel::Float;
use crate::rdft::solvers::dht as r2r_dht;

/// Execute the Discrete Hartley Transform.
///
/// The DHT is its own inverse (up to scaling).
pub fn dht<T: Float>(input: &[T], output: &mut [T]) {
    r2r_dht(input, output);
}
