//! Streaming FFT implementation for real-time processing.
//!
//! This module provides Short-Time Fourier Transform (STFT) and related
//! utilities for processing continuous data streams with overlapping windows.
//!
//! # Overview
//!
//! - **Window Functions**: Hann, Hamming, Blackman, Kaiser, and custom
//! - **STFT/ISTFT**: Forward and inverse short-time transforms
//! - **Overlap-Add/Save**: Time-domain reconstruction methods
//! - **Ring Buffer**: Efficient streaming input handling
//!
//! # Example
//!
//! ```ignore
//! use oxifft::streaming::{StreamingFft, WindowFunction, stft, istft};
//!
//! // One-shot STFT
//! let signal = vec![0.0f64; 4096];
//! let spectrogram = stft(&signal, 256, 64, WindowFunction::Hann);
//!
//! // Reconstruct from spectrogram
//! let reconstructed = istft(&spectrogram, 64, WindowFunction::Hann);
//!
//! // Real-time streaming
//! let mut streaming = StreamingFft::new(256, 64, WindowFunction::Hann);
//! streaming.feed(&signal[0..128]);
//! while let Some(frame) = streaming.pop_frame() {
//!     println!("Frame: {:?}", frame);
//! }
//! ```

mod stft;
mod window;

pub use stft::{
    istft, magnitude_spectrogram, phase_spectrogram, power_spectrogram, stft, StreamingFft,
};
pub use window::{
    blackman, cola_normalization, hamming, hann, kaiser, rectangular, WindowFunction,
};

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::kernel::Float;

/// Ring buffer for efficient streaming data handling.
///
/// Provides O(1) push and efficient batch operations for streaming FFT.
#[derive(Clone, Debug)]
pub struct RingBuffer<T: Float> {
    /// Internal storage.
    data: Vec<T>,
    /// Write position (next position to write).
    write_pos: usize,
    /// Number of valid samples.
    len: usize,
    /// Capacity.
    capacity: usize,
}

impl<T: Float> RingBuffer<T> {
    /// Create a new ring buffer with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![T::ZERO; capacity],
            write_pos: 0,
            len: 0,
            capacity,
        }
    }

    /// Push a single sample.
    pub fn push(&mut self, value: T) {
        self.data[self.write_pos] = value;
        self.write_pos = (self.write_pos + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Push multiple samples.
    pub fn push_slice(&mut self, values: &[T]) {
        for &v in values {
            self.push(v);
        }
    }

    /// Get the number of valid samples.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Check if buffer is full.
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Get capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Read the last `n` samples into a slice (in order).
    ///
    /// Returns the number of samples actually read.
    pub fn read_last(&self, output: &mut [T]) -> usize {
        let n = output.len().min(self.len);
        if n == 0 {
            return 0;
        }

        // Calculate start position
        let start = if self.len == self.capacity {
            self.write_pos
        } else {
            0
        };

        // Read in order
        for i in 0..n {
            let read_idx = (start + (self.len - n) + i) % self.capacity;
            output[i] = self.data[read_idx];
        }

        n
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.len = 0;
        self.write_pos = 0;
    }

    /// Advance the buffer by `n` samples (discarding oldest).
    pub fn advance(&mut self, n: usize) {
        if n >= self.len {
            self.clear();
        } else {
            self.len -= n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let mut buf: RingBuffer<f64> = RingBuffer::new(4);

        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);

        assert_eq!(buf.len(), 3);
        assert!(!buf.is_full());

        let mut out = [0.0; 4];
        let n = buf.read_last(&mut out);
        assert_eq!(n, 3);
        assert_eq!(&out[..3], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let mut buf: RingBuffer<f64> = RingBuffer::new(4);

        buf.push_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert_eq!(buf.len(), 4);
        assert!(buf.is_full());

        let mut out = [0.0; 4];
        buf.read_last(&mut out);
        // Should contain last 4: [3, 4, 5, 6]
        assert_eq!(&out, &[3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_ring_buffer_advance() {
        let mut buf: RingBuffer<f64> = RingBuffer::new(8);
        buf.push_slice(&[1.0, 2.0, 3.0, 4.0]);

        buf.advance(2);
        assert_eq!(buf.len(), 2);
    }
}
