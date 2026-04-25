//! Problem hashing for wisdom lookup.

use core::hash::{Hash, Hasher};

/// Hash a problem for wisdom lookup.
pub struct ProblemHash {
    state: seahash::SeaHasher,
}

impl Default for ProblemHash {
    fn default() -> Self {
        Self::new()
    }
}

impl ProblemHash {
    /// Create a new hasher.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: seahash::SeaHasher::new(),
        }
    }

    /// Hash a hashable value.
    pub fn hash<T: Hash>(&mut self, value: &T) {
        value.hash(&mut self.state);
    }

    /// Finish and get the hash value.
    #[must_use]
    pub fn finish(self) -> u64 {
        self.state.finish()
    }
}

impl Hasher for ProblemHash {
    fn finish(&self) -> u64 {
        self.state.finish()
    }

    fn write(&mut self, bytes: &[u8]) {
        self.state.write(bytes);
    }
}

/// Hash a problem for wisdom lookup.
#[allow(dead_code)] // reason: wisdom lookup utility; not called when wisdom is disabled via feature flags
#[must_use]
pub fn hash_problem<T: Hash>(problem: &T) -> u64 {
    let mut hasher = ProblemHash::new();
    problem.hash(&mut hasher);
    hasher.finish()
}
