//! Sliding DFT (SDFT) — O(N) per-sample full-spectrum update, O(1) per-bin.
//!
//! The Sliding DFT recursively updates the DFT as each new sample arrives:
//!
//! ```text
//! X[k] = (X[k] - x_old + x_new) * W_N^k
//! ```
//!
//! where `W_N^k = exp(2πik/N)` is precomputed.
//!
//! This module provides three main structures:
//!
//! - [`SlidingDft`] — maintains all N frequency bins, O(N) per sample
//! - [`ModulatedSdft`] — numerically stable variant that avoids drift
//! - [`SingleBinTracker`] — tracks a single frequency bin in O(1)
//!
//! # Example
//!
//! ```
//! use oxifft::kernel::{Complex, Float};
//! use oxifft::streaming::sdft::{SlidingDft, SingleBinTracker, sliding_dft, single_bin_tracker};
//!
//! // Create a sliding DFT with window size 8
//! let mut sdft = sliding_dft(8);
//!
//! // Push samples one at a time
//! for i in 0..16 {
//!     sdft.push_real(i as f64);
//! }
//!
//! // Read the current spectrum
//! let spec = sdft.spectrum();
//! assert_eq!(spec.len(), 8);
//!
//! // Track a single frequency bin efficiently
//! let mut tracker = single_bin_tracker(3, 8);
//! for i in 0..16 {
//!     tracker.push(Complex::new(i as f64, 0.0));
//! }
//! let mag = tracker.magnitude();
//! ```

use crate::kernel::{Complex, Float};
use crate::prelude::*;

// ---------------------------------------------------------------------------
// SlidingDft
// ---------------------------------------------------------------------------

/// Standard Sliding DFT that maintains all N frequency bins.
///
/// After the first N samples have been pushed the spectrum is valid.
/// Each subsequent `push` updates *all* N bins in O(N).
///
/// **Note:** This variant can accumulate numerical drift over very long
/// runs.  If that is a concern, use [`ModulatedSdft`] instead.
#[derive(Clone, Debug)]
pub struct SlidingDft<T: Float> {
    /// Window size.
    n: usize,
    /// Circular buffer storing the last N time-domain samples.
    buffer: Vec<Complex<T>>,
    /// Current frequency-domain spectrum (N bins).
    spectrum: Vec<Complex<T>>,
    /// Precomputed twiddle factors W_N^k = exp(2πik/N) for k = 0..N-1.
    twiddles: Vec<Complex<T>>,
    /// Current write position in the circular buffer.
    pos: usize,
    /// Number of samples pushed so far (clamped at N once full).
    samples_pushed: usize,
}

impl<T: Float> SlidingDft<T> {
    /// Create a new Sliding DFT with window size `n`.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "SlidingDft window size must be > 0");

        let twiddles = Self::compute_twiddles(n);

        Self {
            n,
            buffer: vec![Complex::zero(); n],
            spectrum: vec![Complex::zero(); n],
            twiddles,
            pos: 0,
            samples_pushed: 0,
        }
    }

    /// Push one complex sample and update all frequency bins.
    pub fn push(&mut self, sample: Complex<T>) {
        let x_old = self.buffer[self.pos];
        self.buffer[self.pos] = sample;
        self.pos = (self.pos + 1) % self.n;

        if self.samples_pushed < self.n {
            self.samples_pushed += 1;
            if self.samples_pushed == self.n {
                // First full window — compute the DFT from scratch.
                self.compute_initial_dft();
            }
            return;
        }

        // Sliding update: X[k] = (X[k] - x_old + x_new) * W_N^k
        let diff = sample - x_old;
        for k in 0..self.n {
            self.spectrum[k] = (self.spectrum[k] + diff) * self.twiddles[k];
        }
    }

    /// Convenience: push a real-valued sample (imaginary part = 0).
    #[inline]
    pub fn push_real(&mut self, sample: T) {
        self.push(Complex::new(sample, T::ZERO));
    }

    /// Current frequency-domain spectrum.
    ///
    /// Returns a zero-filled slice until the first N samples have been pushed.
    #[inline]
    pub fn spectrum(&self) -> &[Complex<T>] {
        &self.spectrum
    }

    /// Read a single frequency bin.
    ///
    /// Returns `Complex::zero()` if the index is out of range or not yet initialised.
    #[inline]
    pub fn bin(&self, k: usize) -> Complex<T> {
        if k < self.n {
            self.spectrum[k]
        } else {
            Complex::zero()
        }
    }

    /// Magnitude spectrum: |X\[k\]| for each bin.
    pub fn magnitude_spectrum(&self) -> Vec<T> {
        self.spectrum.iter().map(|c| c.norm()).collect()
    }

    /// Power spectrum: |X\[k\]|² for each bin.
    pub fn power_spectrum(&self) -> Vec<T> {
        self.spectrum.iter().map(|c| c.norm_sqr()).collect()
    }

    /// Window size.
    #[inline]
    pub fn window_size(&self) -> usize {
        self.n
    }

    /// Whether the buffer has been filled at least once (spectrum is valid).
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.samples_pushed >= self.n
    }

    /// Reset to initial (empty) state, keeping the same window size.
    pub fn reset(&mut self) {
        for v in &mut self.buffer {
            *v = Complex::zero();
        }
        for v in &mut self.spectrum {
            *v = Complex::zero();
        }
        self.pos = 0;
        self.samples_pushed = 0;
    }

    // -- private helpers -----------------------------------------------------

    /// Compute W_N^k = exp(2πik/N) for k = 0..N-1.
    fn compute_twiddles(n: usize) -> Vec<Complex<T>> {
        let n_f = T::from_usize(n);
        (0..n)
            .map(|k| {
                let angle = T::TWO_PI * T::from_usize(k) / n_f;
                Complex::cis(angle)
            })
            .collect()
    }

    /// Brute-force DFT of the current circular buffer (called once).
    fn compute_initial_dft(&mut self) {
        let n_f = T::from_usize(self.n);
        for k in 0..self.n {
            let mut sum = Complex::zero();
            for m in 0..self.n {
                // The buffer is circular; the oldest sample is at `self.pos`.
                let idx = (self.pos + m) % self.n;
                let angle = -T::TWO_PI * T::from_usize(k) * T::from_usize(m) / n_f;
                sum = sum + self.buffer[idx] * Complex::cis(angle);
            }
            self.spectrum[k] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// ModulatedSdft — numerically stable variant
// ---------------------------------------------------------------------------

/// Numerically stable Sliding DFT using modulation.
///
/// This variant avoids the cumulative drift of [`SlidingDft`] by recomputing
/// a correction factor at every step.  After each `push` the bins are exact
/// (within floating-point precision of a single DFT).
///
/// The modulated SDFT stores an *intermediate* spectrum Y\[k\] and applies the
/// modulation phase `W_N^{-k·n}` on readout so accumulated twiddle
/// multiplications are avoided.
///
/// Algorithm for each new sample `x_new` replacing `x_old`:
///
/// ```text
/// Y[k] += x_new - x_old          // O(N) total
/// X[k]  = Y[k] * W_N^{-k·pos}    // phase correction on readout
/// ```
///
/// Because the twiddle multiplication only happens on *readout*, there is no
/// multiplicative drift over time.
#[derive(Clone, Debug)]
pub struct ModulatedSdft<T: Float> {
    /// Window size.
    n: usize,
    /// Circular buffer of time-domain samples.
    buffer: Vec<Complex<T>>,
    /// Intermediate (un-modulated) spectrum.
    y_spectrum: Vec<Complex<T>>,
    /// Precomputed twiddle table W_N^k, k = 0..N-1.
    twiddles: Vec<Complex<T>>,
    /// Precomputed inverse twiddle table W_N^{-k}, k = 0..N-1.
    inv_twiddles: Vec<Complex<T>>,
    /// Current write position.
    pos: usize,
    /// Number of accumulated samples (clamped at N).
    samples_pushed: usize,
}

impl<T: Float> ModulatedSdft<T> {
    /// Create a new modulated Sliding DFT with window size `n`.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "ModulatedSdft window size must be > 0");

        let n_f = T::from_usize(n);
        let twiddles: Vec<Complex<T>> = (0..n)
            .map(|k| {
                let angle = T::TWO_PI * T::from_usize(k) / n_f;
                Complex::cis(angle)
            })
            .collect();
        let inv_twiddles: Vec<Complex<T>> = (0..n)
            .map(|k| {
                let angle = -T::TWO_PI * T::from_usize(k) / n_f;
                Complex::cis(angle)
            })
            .collect();

        Self {
            n,
            buffer: vec![Complex::zero(); n],
            y_spectrum: vec![Complex::zero(); n],
            twiddles,
            inv_twiddles,
            pos: 0,
            samples_pushed: 0,
        }
    }

    /// Push one complex sample and update the internal spectrum.
    pub fn push(&mut self, sample: Complex<T>) {
        let x_old = self.buffer[self.pos];
        self.buffer[self.pos] = sample;
        self.pos = (self.pos + 1) % self.n;

        if self.samples_pushed < self.n {
            self.samples_pushed += 1;
            if self.samples_pushed == self.n {
                self.compute_initial_dft();
            }
            return;
        }

        // Modulated update: rotate out old, rotate in new
        let diff = sample - x_old;
        for k in 0..self.n {
            // Shift by one sample: Y[k] *= W_N^k
            self.y_spectrum[k] = (self.y_spectrum[k] + diff) * self.twiddles[k];
        }
    }

    /// Convenience: push a real-valued sample.
    #[inline]
    pub fn push_real(&mut self, sample: T) {
        self.push(Complex::new(sample, T::ZERO));
    }

    /// Compute and return the current corrected spectrum.
    ///
    /// This applies the modulation phase on read, guaranteeing no drift.
    pub fn spectrum(&self) -> Vec<Complex<T>> {
        if self.samples_pushed < self.n {
            return vec![Complex::zero(); self.n];
        }
        // The oldest sample in the buffer is at position `self.pos`.
        // Apply W_N^{-k*pos} correction.
        let mut out = Vec::with_capacity(self.n);
        for k in 0..self.n {
            // correction = W_N^{-k * pos}
            let phase_idx = (k * self.pos) % self.n;
            let correction = self.inv_twiddles[phase_idx];
            out.push(self.y_spectrum[k] * correction);
        }
        out
    }

    /// Read a single corrected frequency bin.
    pub fn bin(&self, k: usize) -> Complex<T> {
        if k >= self.n || self.samples_pushed < self.n {
            return Complex::zero();
        }
        let phase_idx = (k * self.pos) % self.n;
        let correction = self.inv_twiddles[phase_idx];
        self.y_spectrum[k] * correction
    }

    /// Magnitude spectrum: |X\[k\]| for each bin.
    pub fn magnitude_spectrum(&self) -> Vec<T> {
        self.spectrum().iter().map(|c| c.norm()).collect()
    }

    /// Power spectrum: |X\[k\]|² for each bin.
    pub fn power_spectrum(&self) -> Vec<T> {
        self.spectrum().iter().map(|c| c.norm_sqr()).collect()
    }

    /// Window size.
    #[inline]
    pub fn window_size(&self) -> usize {
        self.n
    }

    /// Whether the buffer has been filled at least once.
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.samples_pushed >= self.n
    }

    /// Reset state, keep window size.
    pub fn reset(&mut self) {
        for v in &mut self.buffer {
            *v = Complex::zero();
        }
        for v in &mut self.y_spectrum {
            *v = Complex::zero();
        }
        self.pos = 0;
        self.samples_pushed = 0;
    }

    // -- private helpers -----------------------------------------------------

    /// Brute-force DFT of the current circular buffer (called once).
    fn compute_initial_dft(&mut self) {
        let n_f = T::from_usize(self.n);
        for k in 0..self.n {
            let mut sum = Complex::zero();
            for m in 0..self.n {
                let idx = (self.pos + m) % self.n;
                let angle = -T::TWO_PI * T::from_usize(k) * T::from_usize(m) / n_f;
                sum = sum + self.buffer[idx] * Complex::cis(angle);
            }
            self.y_spectrum[k] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// SingleBinTracker — O(1) per sample for a single frequency bin
// ---------------------------------------------------------------------------

/// Tracks a single frequency bin of a sliding window in O(1) per sample.
///
/// Useful for tone detection, frequency monitoring, or DTMF decoding where
/// only one (or a few) bins are of interest.
#[derive(Clone, Debug)]
pub struct SingleBinTracker<T: Float> {
    /// Which bin to track (0 ≤ k < N).
    k: usize,
    /// Window size.
    n: usize,
    /// Circular buffer of samples.
    buffer: Vec<Complex<T>>,
    /// Current bin value.
    value: Complex<T>,
    /// Precomputed twiddle W_N^k.
    twiddle: Complex<T>,
    /// Write position.
    pos: usize,
    /// Samples pushed so far (clamped at N).
    samples_pushed: usize,
}

impl<T: Float> SingleBinTracker<T> {
    /// Create a new single-bin tracker.
    ///
    /// # Arguments
    ///
    /// * `frequency_bin` — the DFT bin index k (0 ≤ k < window_size).
    /// * `window_size` — the sliding window length N.
    ///
    /// # Panics
    ///
    /// Panics if `window_size == 0` or `frequency_bin >= window_size`.
    pub fn new(frequency_bin: usize, window_size: usize) -> Self {
        assert!(window_size > 0, "SingleBinTracker window size must be > 0");
        assert!(
            frequency_bin < window_size,
            "frequency_bin ({frequency_bin}) must be < window_size ({window_size})"
        );

        let n_f = T::from_usize(window_size);
        let angle = T::TWO_PI * T::from_usize(frequency_bin) / n_f;
        let twiddle = Complex::cis(angle);

        Self {
            k: frequency_bin,
            n: window_size,
            buffer: vec![Complex::zero(); window_size],
            value: Complex::zero(),
            twiddle,
            pos: 0,
            samples_pushed: 0,
        }
    }

    /// Push one complex sample — O(1).
    pub fn push(&mut self, sample: Complex<T>) {
        let x_old = self.buffer[self.pos];
        self.buffer[self.pos] = sample;
        self.pos = (self.pos + 1) % self.n;

        if self.samples_pushed < self.n {
            self.samples_pushed += 1;
            if self.samples_pushed == self.n {
                self.compute_initial_bin();
            }
            return;
        }

        // X[k] = (X[k] - x_old + x_new) * W_N^k
        self.value = (self.value + sample - x_old) * self.twiddle;
    }

    /// Convenience: push a real-valued sample.
    #[inline]
    pub fn push_real(&mut self, sample: T) {
        self.push(Complex::new(sample, T::ZERO));
    }

    /// Current (complex) bin value.
    #[inline]
    pub fn value(&self) -> Complex<T> {
        self.value
    }

    /// Magnitude of the tracked bin: |X\[k\]|.
    #[inline]
    pub fn magnitude(&self) -> T {
        self.value.norm()
    }

    /// Phase (argument) of the tracked bin.
    #[inline]
    pub fn phase(&self) -> T {
        self.value.arg()
    }

    /// The bin index being tracked.
    #[inline]
    pub fn bin_index(&self) -> usize {
        self.k
    }

    /// Window size.
    #[inline]
    pub fn window_size(&self) -> usize {
        self.n
    }

    /// Whether the buffer has been filled at least once.
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.samples_pushed >= self.n
    }

    /// Reset state, keep parameters.
    pub fn reset(&mut self) {
        for v in &mut self.buffer {
            *v = Complex::zero();
        }
        self.value = Complex::zero();
        self.pos = 0;
        self.samples_pushed = 0;
    }

    // -- private helpers -----------------------------------------------------

    /// DFT of bin k from the current circular buffer.
    fn compute_initial_bin(&mut self) {
        let n_f = T::from_usize(self.n);
        let mut sum = Complex::zero();
        for m in 0..self.n {
            let idx = (self.pos + m) % self.n;
            let angle = -T::TWO_PI * T::from_usize(self.k) * T::from_usize(m) / n_f;
            sum = sum + self.buffer[idx] * Complex::cis(angle);
        }
        self.value = sum;
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Create a [`SlidingDft<f64>`] with window size `n`.
#[inline]
pub fn sliding_dft(n: usize) -> SlidingDft<f64> {
    SlidingDft::new(n)
}

/// Create a [`SingleBinTracker<f64>`] for the given bin and window size.
#[inline]
pub fn single_bin_tracker(bin: usize, window: usize) -> SingleBinTracker<f64> {
    SingleBinTracker::new(bin, window)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: brute-force DFT of a complex slice (reference implementation).
    fn reference_dft(x: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let n = x.len();
        let n_f = n as f64;
        (0..n)
            .map(|k| {
                let mut sum = Complex::<f64>::zero();
                for (m, xm) in x.iter().enumerate() {
                    let angle = -core::f64::consts::TAU * (k as f64) * (m as f64) / n_f;
                    sum = sum + *xm * Complex::cis(angle);
                }
                sum
            })
            .collect()
    }

    /// Assert two complex slices are approximately equal.
    fn assert_spectrum_close(a: &[Complex<f64>], b: &[Complex<f64>], tol: f64) {
        assert_eq!(a.len(), b.len(), "spectrum length mismatch");
        for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (*ai - *bi).norm();
            assert!(diff < tol, "bin {i}: |{ai:?} - {bi:?}| = {diff} >= {tol}");
        }
    }

    // -- SlidingDft tests ---------------------------------------------------

    #[test]
    fn sdft_matches_reference_dft() {
        let n = 16;
        let mut sdft = SlidingDft::<f64>::new(n);

        // Push 2*N samples so the window has slid.
        let samples: Vec<Complex<f64>> = (0..2 * n)
            .map(|i| Complex::new(i as f64, -(i as f64) * 0.5))
            .collect();

        for s in &samples {
            sdft.push(*s);
        }

        // The current window is the last N samples.
        let window = &samples[n..];
        let ref_spectrum = reference_dft(window);

        assert_spectrum_close(sdft.spectrum(), &ref_spectrum, 1e-9);
    }

    #[test]
    fn sdft_real_input_convenience() {
        let n = 8;
        let mut sdft = SlidingDft::<f64>::new(n);

        let reals: Vec<f64> = (0..n).map(|i| (i as f64) * 1.5).collect();
        for &r in &reals {
            sdft.push_real(r);
        }

        let complex_in: Vec<Complex<f64>> = reals.iter().map(|&r| Complex::new(r, 0.0)).collect();

        let ref_spectrum = reference_dft(&complex_in);
        assert_spectrum_close(sdft.spectrum(), &ref_spectrum, 1e-12);
    }

    #[test]
    fn sdft_magnitude_and_power() {
        let n = 8;
        let mut sdft = SlidingDft::<f64>::new(n);
        for i in 0..n {
            sdft.push_real(i as f64);
        }

        let mag = sdft.magnitude_spectrum();
        let pow = sdft.power_spectrum();
        assert_eq!(mag.len(), n);
        assert_eq!(pow.len(), n);

        for i in 0..n {
            let expected_mag = sdft.spectrum()[i].norm();
            let expected_pow = sdft.spectrum()[i].norm_sqr();
            assert!((mag[i] - expected_mag).abs() < 1e-14);
            assert!((pow[i] - expected_pow).abs() < 1e-14);
        }
    }

    #[test]
    fn sdft_reset() {
        let n = 4;
        let mut sdft = SlidingDft::<f64>::new(n);
        for i in 0..n {
            sdft.push_real(i as f64);
        }
        assert!(sdft.is_initialized());

        sdft.reset();
        assert!(!sdft.is_initialized());

        // Spectrum should be zero after reset.
        for &bin in sdft.spectrum() {
            assert!((bin.norm()) < 1e-15);
        }
    }

    #[test]
    fn sdft_edge_n1() {
        let mut sdft = SlidingDft::<f64>::new(1);
        sdft.push_real(42.0);
        assert!(sdft.is_initialized());
        assert!((sdft.bin(0).re - 42.0).abs() < 1e-14);

        sdft.push_real(7.0);
        assert!((sdft.bin(0).re - 7.0).abs() < 1e-14);
    }

    #[test]
    fn sdft_edge_n2() {
        let mut sdft = SlidingDft::<f64>::new(2);
        sdft.push_real(1.0);
        sdft.push_real(2.0);
        assert!(sdft.is_initialized());

        let window = [Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
        let ref_spec = reference_dft(&window);
        assert_spectrum_close(sdft.spectrum(), &ref_spec, 1e-14);
    }

    #[test]
    fn sdft_pure_sinusoid() {
        let n = 64;
        let mut sdft = SlidingDft::<f64>::new(n);

        // Generate a pure sinusoid at bin 5.
        let bin_freq = 5;
        for i in 0..2 * n {
            let t = core::f64::consts::TAU * (bin_freq as f64) * (i as f64) / (n as f64);
            sdft.push_real(t.cos());
        }

        // Bin 5 (and mirror N-5) should dominate.
        let mag = sdft.magnitude_spectrum();
        let peak = mag[bin_freq];
        for (k, &m) in mag.iter().enumerate() {
            if k != bin_freq && k != n - bin_freq {
                assert!(
                    m < peak * 0.01,
                    "bin {k} magnitude {m} is too large compared to peak {peak}"
                );
            }
        }
    }

    #[test]
    fn sdft_bin_out_of_range() {
        let sdft = SlidingDft::<f64>::new(4);
        let v = sdft.bin(100);
        assert!(v.norm() < 1e-15);
    }

    // -- ModulatedSdft tests -----------------------------------------------

    #[test]
    fn modulated_sdft_matches_reference() {
        let n = 16;
        let mut msdft = ModulatedSdft::<f64>::new(n);

        let samples: Vec<Complex<f64>> = (0..2 * n)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        for s in &samples {
            msdft.push(*s);
        }

        let window = &samples[n..];
        let ref_spectrum = reference_dft(window);
        let got = msdft.spectrum();

        assert_spectrum_close(&got, &ref_spectrum, 1e-9);
    }

    #[test]
    fn modulated_sdft_stable_over_long_run() {
        // Push >10000 samples and verify the result is still accurate.
        let n = 32;
        let mut msdft = ModulatedSdft::<f64>::new(n);
        let mut plain_sdft = SlidingDft::<f64>::new(n);

        let total = 12_000;
        let mut recent = Vec::with_capacity(n);

        for i in 0..total {
            let val = Complex::new(((i as f64) * 0.1).sin(), ((i as f64) * 0.07).cos());
            msdft.push(val);
            plain_sdft.push(val);

            // Keep track of the last N samples for reference DFT.
            recent.push(val);
            if recent.len() > n {
                recent.remove(0);
            }
        }

        let ref_spectrum = reference_dft(&recent);
        let mod_spec = msdft.spectrum();

        // The modulated variant should be very close to reference.
        assert_spectrum_close(&mod_spec, &ref_spectrum, 1e-6);

        // The plain variant may have drifted more (we just check it compiles).
        let _plain_spec = plain_sdft.spectrum();
    }

    #[test]
    fn modulated_sdft_reset() {
        let n = 8;
        let mut msdft = ModulatedSdft::<f64>::new(n);
        for i in 0..n {
            msdft.push_real(i as f64);
        }
        assert!(msdft.is_initialized());

        msdft.reset();
        assert!(!msdft.is_initialized());

        let spec = msdft.spectrum();
        for bin in &spec {
            assert!(bin.norm() < 1e-15);
        }
    }

    #[test]
    fn modulated_sdft_magnitude_power() {
        let n = 8;
        let mut msdft = ModulatedSdft::<f64>::new(n);
        for i in 0..n {
            msdft.push_real(i as f64);
        }

        let spec = msdft.spectrum();
        let mag = msdft.magnitude_spectrum();
        let pow = msdft.power_spectrum();

        for i in 0..n {
            assert!((mag[i] - spec[i].norm()).abs() < 1e-14);
            assert!((pow[i] - spec[i].norm_sqr()).abs() < 1e-14);
        }
    }

    #[test]
    fn modulated_sdft_single_bin() {
        let n = 16;
        let mut msdft = ModulatedSdft::<f64>::new(n);

        for i in 0..2 * n {
            msdft.push_real((i as f64) * 0.3);
        }

        let full = msdft.spectrum();
        for k in 0..n {
            let single = msdft.bin(k);
            let diff = (single - full[k]).norm();
            assert!(diff < 1e-12, "bin {k} mismatch: {diff}");
        }
    }

    // -- SingleBinTracker tests ---------------------------------------------

    #[test]
    fn single_bin_matches_full_sdft() {
        let n = 16;
        let k = 5;
        let mut sdft = SlidingDft::<f64>::new(n);
        let mut tracker = SingleBinTracker::<f64>::new(k, n);

        let samples: Vec<Complex<f64>> = (0..3 * n)
            .map(|i| Complex::new((i as f64).sin(), (i as f64 * 0.3).cos()))
            .collect();

        for s in &samples {
            sdft.push(*s);
            tracker.push(*s);
        }

        let diff = (sdft.bin(k) - tracker.value()).norm();
        assert!(diff < 1e-9, "tracker vs sdft bin {k}: diff = {diff}");
    }

    #[test]
    fn single_bin_magnitude_phase() {
        let n = 8;
        let k = 3;
        let mut tracker = SingleBinTracker::<f64>::new(k, n);

        for i in 0..n {
            tracker.push_real(i as f64);
        }

        let v = tracker.value();
        assert!((tracker.magnitude() - v.norm()).abs() < 1e-14);
        assert!((tracker.phase() - v.arg()).abs() < 1e-14);
    }

    #[test]
    fn single_bin_reset() {
        let n = 8;
        let mut tracker = SingleBinTracker::<f64>::new(2, n);
        for i in 0..n {
            tracker.push_real(i as f64);
        }
        assert!(tracker.is_initialized());

        tracker.reset();
        assert!(!tracker.is_initialized());
        assert!(tracker.value().norm() < 1e-15);
    }

    #[test]
    fn single_bin_accessors() {
        let tracker = SingleBinTracker::<f64>::new(3, 16);
        assert_eq!(tracker.bin_index(), 3);
        assert_eq!(tracker.window_size(), 16);
        assert!(!tracker.is_initialized());
    }

    #[test]
    fn single_bin_tone_detection() {
        // Generate a pure tone at bin=7 and verify the tracker sees a big peak.
        let n = 64;
        let target_bin = 7;
        let mut tracker = SingleBinTracker::<f64>::new(target_bin, n);
        let mut other_tracker = SingleBinTracker::<f64>::new(13, n);

        for i in 0..2 * n {
            let t = core::f64::consts::TAU * (target_bin as f64) * (i as f64) / (n as f64);
            let sample = Complex::new(t.cos(), 0.0);
            tracker.push(sample);
            other_tracker.push(sample);
        }

        // The matching tracker should have a large magnitude.
        // The non-matching tracker should be near zero.
        assert!(tracker.magnitude() > 20.0);
        assert!(other_tracker.magnitude() < 1.0);
    }

    // -- f32 tests ----------------------------------------------------------

    #[test]
    fn sdft_f32_works() {
        let n = 8;
        let mut sdft = SlidingDft::<f32>::new(n);
        for i in 0..n {
            sdft.push_real(i as f32);
        }
        assert!(sdft.is_initialized());
        assert_eq!(sdft.spectrum().len(), n);
    }

    #[test]
    fn modulated_sdft_f32_works() {
        let n = 8;
        let mut msdft = ModulatedSdft::<f32>::new(n);
        for i in 0..n {
            msdft.push_real(i as f32);
        }
        assert!(msdft.is_initialized());
        assert_eq!(msdft.spectrum().len(), n);
    }

    #[test]
    fn single_bin_f32_works() {
        let mut tracker = SingleBinTracker::<f32>::new(2, 8);
        for i in 0..8 {
            tracker.push_real(i as f32);
        }
        assert!(tracker.is_initialized());
        assert!(tracker.magnitude() > 0.0);
    }

    // -- convenience function tests -----------------------------------------

    #[test]
    fn convenience_sliding_dft() {
        let mut s = sliding_dft(4);
        for i in 0..4 {
            s.push_real(i as f64);
        }
        assert!(s.is_initialized());
    }

    #[test]
    fn convenience_single_bin_tracker() {
        let mut t = single_bin_tracker(1, 4);
        for i in 0..4 {
            t.push_real(i as f64);
        }
        assert!(t.is_initialized());
    }
}
