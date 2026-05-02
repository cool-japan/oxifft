//! Twiddle factor computation, caching, and SIMD-accelerated complex multiplication.
//!
//! Twiddle factors are the complex exponentials W_n^k = e^(-2πik/n)
//! used in FFT butterfly operations.
//!
//! # Global Cache
//!
//! [`get_twiddle_table_f64`] and [`get_twiddle_table_f32`] provide shared,
//! precomputed twiddle tables via a process-global cache. Subsequent calls
//! with the same `(size, direction)` key return an [`Arc`]-cloned reference
//! without recomputation.
//!
//! # SIMD Twiddle Multiplication
//!
//! [`twiddle_mul_simd_f64`] and [`twiddle_mul_simd_f32`] apply twiddle factors
//! to a complex slice in-place, dispatching to the best available SIMD backend
//! at runtime (AVX2+FMA → SSE2 → NEON → scalar).

use super::{Complex, Float};
use crate::prelude::*;

use core::any::TypeId;

#[cfg(feature = "std")]
use std::sync::Arc;

#[cfg(not(feature = "std"))]
use alloc::sync::Arc;

/// Cache for twiddle factors.
///
/// Stores precomputed twiddle factors keyed by (n, k).
pub struct TwiddleCache<T: Float> {
    cache: HashMap<(usize, usize), Vec<Complex<T>>>,
}

impl<T: Float> Default for TwiddleCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> TwiddleCache<T> {
    /// Create a new empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Get or compute twiddle factors for size n and radix k.
    ///
    /// Returns W_n^0, W_n^1, ..., W_n^(k-1) for the given parameters.
    pub fn get(&mut self, n: usize, k: usize) -> &[Complex<T>] {
        self.cache
            .entry((n, k))
            .or_insert_with(|| compute_twiddles(n, k))
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

/// Compute twiddle factors W_n^j for j = 0..k.
///
/// W_n^j = e^(-2πij/n) = cos(2πj/n) - i*sin(2πj/n)
#[must_use]
pub fn compute_twiddles<T: Float>(n: usize, k: usize) -> Vec<Complex<T>> {
    let mut result = Vec::with_capacity(k);
    let theta_base = -T::TWO_PI / T::from_usize(n);

    for j in 0..k {
        let theta = theta_base * T::from_usize(j);
        result.push(Complex::cis(theta));
    }

    result
}

/// Compute a single twiddle factor W_n^k.
#[allow(dead_code)] // reason: public twiddle utility; not called in all solver paths (precomputed tables preferred)
#[inline]
#[must_use]
pub fn twiddle<T: Float>(n: usize, k: usize) -> Complex<T> {
    let theta = -T::TWO_PI * T::from_usize(k) / T::from_usize(n);
    Complex::cis(theta)
}

/// Compute twiddle factor for inverse transform: W_n^(-k).
#[allow(dead_code)] // reason: public twiddle utility; not called in all solver paths (precomputed tables preferred)
#[inline]
#[must_use]
pub fn twiddle_inverse<T: Float>(n: usize, k: usize) -> Complex<T> {
    let theta = T::TWO_PI * T::from_usize(k) / T::from_usize(n);
    Complex::cis(theta)
}

// ============================================================================
// Global twiddle table cache
// ============================================================================

/// Direction for twiddle factor computation.
///
/// Used as part of the cache key to distinguish forward and inverse twiddle tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TwiddleDirection {
    /// Forward transform: W_n^k = e^(-2πik/n)
    Forward,
    /// Inverse transform: W_n^k = e^(+2πik/n)
    Inverse,
}

/// Cache key for a global twiddle table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TwiddleKey {
    /// Transform size.
    pub size: usize,
    /// Transform direction.
    pub direction: TwiddleDirection,
    /// Type identifier (distinguishes f32 vs f64 tables).
    pub type_id: TypeId,
}

/// A precomputed table of twiddle factors.
pub struct TwiddleTable<T: Float> {
    /// W_n^0, W_n^1, ..., W_n^(size-1)
    pub factors: Vec<Complex<T>>,
}

/// Struct-of-Arrays twiddle table: separate `re` and `im` vectors.
///
/// For SIMD loads this layout eliminates the deinterleave shuffle that the
/// interleaved `TwiddleTable<T>` (AoS) requires.  The arrays are
/// capacity-padded to the next multiple of 8 elements so that SIMD code
/// can always read a full vector without a bounds check.
pub struct TwiddleTableSoA<T: Float> {
    /// Real parts: `re[k]` = cos(θ_k)
    pub re: Vec<T>,
    /// Imaginary parts: `im[k]` = sin(θ_k)
    pub im: Vec<T>,
}

impl<T: Float> TwiddleTableSoA<T> {
    /// Number of *logical* entries (i.e. the transform `size`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.re.len()
    }

    /// Returns `true` when the table is empty (size-0 transform).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.re.is_empty()
    }
}

/// Internal cache holding type-separated twiddle tables.
struct GlobalTwiddleCache {
    f32_tables: HashMap<TwiddleKey, Arc<TwiddleTable<f32>>>,
    f64_tables: HashMap<TwiddleKey, Arc<TwiddleTable<f64>>>,
    soa_f32_tables: HashMap<TwiddleKey, Arc<TwiddleTableSoA<f32>>>,
    soa_f64_tables: HashMap<TwiddleKey, Arc<TwiddleTableSoA<f64>>>,
}

impl GlobalTwiddleCache {
    fn new() -> Self {
        Self {
            f32_tables: HashMap::new(),
            f64_tables: HashMap::new(),
            soa_f32_tables: HashMap::new(),
            soa_f64_tables: HashMap::new(),
        }
    }
}

fn global_twiddle_cache() -> &'static RwLock<GlobalTwiddleCache> {
    static CACHE: OnceLock<RwLock<GlobalTwiddleCache>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(GlobalTwiddleCache::new()))
}

/// Get or compute twiddle factors for the given size and direction (f64).
///
/// Returns an `Arc` to the shared table. Subsequent calls with the same key
/// hit the cache and return a clone of the existing `Arc` without recomputation.
pub fn get_twiddle_table_f64(size: usize, direction: TwiddleDirection) -> Arc<TwiddleTable<f64>> {
    let key = TwiddleKey {
        size,
        direction,
        type_id: TypeId::of::<f64>(),
    };
    {
        let cache = rwlock_read(global_twiddle_cache());
        if let Some(t) = cache.f64_tables.get(&key) {
            return Arc::clone(t);
        }
    }
    // Compute outside the write lock to minimise lock contention.
    let table = Arc::new(compute_twiddle_table_f64(size, direction));
    {
        let mut cache = rwlock_write(global_twiddle_cache());
        // Use entry() to avoid overwriting a concurrently inserted table.
        cache
            .f64_tables
            .entry(key)
            .or_insert_with(|| Arc::clone(&table));
    }
    table
}

/// Get or compute twiddle factors for the given size and direction (f32).
///
/// Returns an `Arc` to the shared table. Subsequent calls with the same key
/// hit the cache and return a clone of the existing `Arc` without recomputation.
pub fn get_twiddle_table_f32(size: usize, direction: TwiddleDirection) -> Arc<TwiddleTable<f32>> {
    let key = TwiddleKey {
        size,
        direction,
        type_id: TypeId::of::<f32>(),
    };
    {
        let cache = rwlock_read(global_twiddle_cache());
        if let Some(t) = cache.f32_tables.get(&key) {
            return Arc::clone(t);
        }
    }
    let table = Arc::new(compute_twiddle_table_f32(size, direction));
    {
        let mut cache = rwlock_write(global_twiddle_cache());
        cache
            .f32_tables
            .entry(key)
            .or_insert_with(|| Arc::clone(&table));
    }
    table
}

/// Clear the global twiddle cache, forcing recomputation on next access.
///
/// Primarily intended for testing.
pub fn clear_twiddle_cache() {
    let mut cache = rwlock_write(global_twiddle_cache());
    *cache = GlobalTwiddleCache::new();
}

// ============================================================================
// SoA twiddle table cache — f64
// ============================================================================

/// Get or compute the SoA twiddle table for the given size and direction (f64).
///
/// Returns an `Arc` to the shared table. Subsequent calls with the same key
/// hit the cache without recomputation.
pub fn get_twiddle_table_soa_f64(
    size: usize,
    direction: TwiddleDirection,
) -> Arc<TwiddleTableSoA<f64>> {
    let key = TwiddleKey {
        size,
        direction,
        type_id: TypeId::of::<f64>(),
    };
    {
        let cache = rwlock_read(global_twiddle_cache());
        if let Some(t) = cache.soa_f64_tables.get(&key) {
            return Arc::clone(t);
        }
    }
    let table = Arc::new(compute_twiddle_table_soa_f64(size, direction));
    {
        let mut cache = rwlock_write(global_twiddle_cache());
        cache
            .soa_f64_tables
            .entry(key)
            .or_insert_with(|| Arc::clone(&table));
    }
    table
}

/// Get or compute the SoA twiddle table for the given size and direction (f32).
///
/// Returns an `Arc` to the shared table. Subsequent calls with the same key
/// hit the cache without recomputation.
pub fn get_twiddle_table_soa_f32(
    size: usize,
    direction: TwiddleDirection,
) -> Arc<TwiddleTableSoA<f32>> {
    let key = TwiddleKey {
        size,
        direction,
        type_id: TypeId::of::<f32>(),
    };
    {
        let cache = rwlock_read(global_twiddle_cache());
        if let Some(t) = cache.soa_f32_tables.get(&key) {
            return Arc::clone(t);
        }
    }
    let table = Arc::new(compute_twiddle_table_soa_f32(size, direction));
    {
        let mut cache = rwlock_write(global_twiddle_cache());
        cache
            .soa_f32_tables
            .entry(key)
            .or_insert_with(|| Arc::clone(&table));
    }
    table
}

/// Compute an aligned-capacity SoA f64 twiddle table directly from cos/sin.
///
/// Capacity is rounded up to the next multiple of 8 so that SIMD loads reading
/// a full 4-wide AVX2 double vector past the last logical element are safe.
fn compute_twiddle_table_soa_f64(size: usize, direction: TwiddleDirection) -> TwiddleTableSoA<f64> {
    let sign = match direction {
        TwiddleDirection::Forward => -1.0_f64,
        TwiddleDirection::Inverse => 1.0_f64,
    };
    // Pad capacity to next multiple of 8 for safe SIMD over-reads.
    let capacity = (size + 7) & !7;
    let mut re = Vec::with_capacity(capacity);
    let mut im = Vec::with_capacity(capacity);
    for k in 0..size {
        let angle = sign * 2.0 * core::f64::consts::PI * k as f64 / size as f64;
        re.push(angle.cos());
        im.push(angle.sin());
    }
    TwiddleTableSoA { re, im }
}

/// Compute an aligned-capacity SoA f32 twiddle table directly from cos/sin.
fn compute_twiddle_table_soa_f32(size: usize, direction: TwiddleDirection) -> TwiddleTableSoA<f32> {
    let sign = match direction {
        TwiddleDirection::Forward => -1.0_f32,
        TwiddleDirection::Inverse => 1.0_f32,
    };
    let capacity = (size + 7) & !7;
    let mut re = Vec::with_capacity(capacity);
    let mut im = Vec::with_capacity(capacity);
    for k in 0..size {
        let angle = sign * 2.0 * core::f32::consts::PI * k as f32 / size as f32;
        re.push(angle.cos());
        im.push(angle.sin());
    }
    TwiddleTableSoA { re, im }
}

fn compute_twiddle_table_f64(size: usize, direction: TwiddleDirection) -> TwiddleTable<f64> {
    let sign = match direction {
        TwiddleDirection::Forward => -1.0_f64,
        TwiddleDirection::Inverse => 1.0_f64,
    };
    let factors = (0..size)
        .map(|k| {
            let angle = sign * 2.0 * core::f64::consts::PI * k as f64 / size as f64;
            Complex::new(angle.cos(), angle.sin())
        })
        .collect();
    TwiddleTable { factors }
}

fn compute_twiddle_table_f32(size: usize, direction: TwiddleDirection) -> TwiddleTable<f32> {
    let sign = match direction {
        TwiddleDirection::Forward => -1.0_f32,
        TwiddleDirection::Inverse => 1.0_f32,
    };
    let factors = (0..size)
        .map(|k| {
            let angle = sign * 2.0 * core::f32::consts::PI * k as f32 / size as f32;
            Complex::new(angle.cos(), angle.sin())
        })
        .collect();
    TwiddleTable { factors }
}

// ============================================================================
// SIMD twiddle multiplication — f64
// ============================================================================

/// Apply twiddle factors to `data` in-place: `data[i] *= twiddles[i]`.
///
/// Dispatches to the best available SIMD backend at runtime:
/// AVX2+FMA → SSE2 → NEON → scalar.
///
/// # Panics
///
/// Panics if `data.len() != twiddles.len()`.
pub fn twiddle_mul_simd_f64(data: &mut [Complex<f64>], twiddles: &[Complex<f64>]) {
    assert_eq!(
        data.len(),
        twiddles.len(),
        "twiddle_mul_simd_f64: length mismatch"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: avx2 and fma features confirmed above
            return unsafe { twiddle_mul_avx2_f64(data, twiddles) };
        }
        if is_x86_feature_detected!("sse2") {
            // SAFETY: sse2 feature confirmed above
            return unsafe { twiddle_mul_sse2_f64(data, twiddles) };
        }
        // Fallback if somehow no x86 SIMD is available at runtime
        return twiddle_mul_scalar_f64(data, twiddles);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on aarch64 — always dispatch directly.
        // SAFETY: NEON is always present on aarch64.
        unsafe { twiddle_mul_neon_f64(data, twiddles) }
    }

    // Other architectures — plain scalar
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    twiddle_mul_scalar_f64(data, twiddles);
}

/// Scalar fallback: `data[i] *= twiddles[i]` with no SIMD.
pub fn twiddle_mul_scalar_f64(data: &mut [Complex<f64>], twiddles: &[Complex<f64>]) {
    for (d, t) in data.iter_mut().zip(twiddles.iter()) {
        *d = *d * *t;
    }
}

/// Scalar SoA fallback: `data[i] *= Complex { re: twiddle_re[i], im: twiddle_im[i] }`.
pub fn twiddle_mul_soa_scalar_f64(
    data: &mut [Complex<f64>],
    twiddle_re: &[f64],
    twiddle_im: &[f64],
) {
    debug_assert_eq!(data.len(), twiddle_re.len(), "SoA re length mismatch");
    debug_assert_eq!(data.len(), twiddle_im.len(), "SoA im length mismatch");
    for ((d, &tw_re), &tw_im) in data
        .iter_mut()
        .zip(twiddle_re.iter())
        .zip(twiddle_im.iter())
    {
        let (d_re, d_im) = (d.re, d.im);
        d.re = d_re * tw_re - d_im * tw_im;
        d.im = d_re * tw_im + d_im * tw_re;
    }
}

/// AVX2+FMA complex multiply: processes 2 complex f64 values per iteration
/// (4 f64 scalars = one `__m256d`).
///
/// Data layout: interleaved `[Re0, Im0, Re1, Im1, ...]`
///
/// Algorithm uses `_mm256_addsub_pd` which alternates sub/add:
/// `addsub(p, q) = [p0-q0, p1+q1, p2-q2, p3+q3]`
///
/// For `d = [a, b, c, d]`, `t = [e, f, g, h]` (two interleaved complex pairs):
/// - `a_re = [a, a, c, c]`, `a_im = [b, b, d, d]`
/// - `t_swap = [f, e, h, g]` (swap re/im within each pair)
/// - `prod1 = a_re * t = [a*e, a*f, c*g, c*h]`
/// - `prod2 = a_im * t_swap = [b*f, b*e, d*h, d*g]`
/// - `addsub(prod1, prod2) = [a*e-b*f, a*f+b*e, c*g-d*h, c*h+d*g]` ✓
///
/// # Safety
///
/// Caller must ensure the target CPU supports the `avx2` and `fma` features.
/// Calling this function on a CPU that lacks these features causes undefined
/// behavior (illegal instruction trap at runtime).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn twiddle_mul_avx2_f64(data: &mut [Complex<f64>], twiddles: &[Complex<f64>]) {
    use core::arch::x86_64::*;

    let chunks = data.len() / 2; // 2 complex per __m256d
    let data_ptr = data.as_mut_ptr() as *mut f64;
    let tw_ptr = twiddles.as_ptr() as *const f64;

    for i in 0..chunks {
        // Load 2 complex numbers: [re0, im0, re1, im1]
        let d = unsafe { _mm256_loadu_pd(data_ptr.add(i * 4)) };
        let t = unsafe { _mm256_loadu_pd(tw_ptr.add(i * 4)) };

        // Duplicate real parts: [re0, re0, re1, re1]
        let a_re = _mm256_permute_pd(d, 0b0000);
        // Duplicate imag parts: [im0, im0, im1, im1]
        let a_im = _mm256_permute_pd(d, 0b1111);

        // Swap re/im in twiddles: [im0, re0, im1, re1]
        let t_swap = _mm256_permute_pd(t, 0b0101);

        // prod1 = [re0*tw_re0, re0*tw_im0, re1*tw_re1, re1*tw_im1]
        let prod1 = _mm256_mul_pd(a_re, t);
        // prod2 = [im0*tw_im0, im0*tw_re0, im1*tw_im1, im1*tw_re1]
        let prod2 = _mm256_mul_pd(a_im, t_swap);

        // addsub: [prod1[0]-prod2[0], prod1[1]+prod2[1], prod1[2]-prod2[2], prod1[3]+prod2[3]]
        // = [re0*tw_re0 - im0*tw_im0, re0*tw_im0 + im0*tw_re0, ...] ✓
        let result = _mm256_addsub_pd(prod1, prod2);

        unsafe { _mm256_storeu_pd(data_ptr.add(i * 4), result) };
    }

    // Handle remaining elements with scalar
    let remainder_start = chunks * 2;
    for i in remainder_start..data.len() {
        data[i] = data[i] * twiddles[i];
    }
}

/// SSE2 complex multiply: processes 1 complex f64 value per iteration
/// (2 f64 scalars = one `__m128d`).
///
/// # Safety
///
/// Caller must ensure the target CPU supports the `sse2` feature.
/// Calling this function on a CPU that lacks `sse2` causes undefined
/// behavior (illegal instruction trap at runtime).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn twiddle_mul_sse2_f64(data: &mut [Complex<f64>], twiddles: &[Complex<f64>]) {
    use core::arch::x86_64::*;

    let data_ptr = data.as_mut_ptr() as *mut f64;
    let tw_ptr = twiddles.as_ptr() as *const f64;

    for i in 0..data.len() {
        // Load [re, im] for data and twiddle
        let d = unsafe { _mm_loadu_pd(data_ptr.add(i * 2)) };
        let t = unsafe { _mm_loadu_pd(tw_ptr.add(i * 2)) };

        // a_re = [re, re], a_im = [im, im]
        let a_re = _mm_unpacklo_pd(d, d);
        let a_im = _mm_unpackhi_pd(d, d);

        // t_swap = [tw_im, tw_re]
        let t_swap = _mm_shuffle_pd(t, t, 0b01);

        // prod1 = [re*tw_re, re*tw_im]
        let prod1 = _mm_mul_pd(a_re, t);
        // prod2 = [im*tw_im, im*tw_re]
        let prod2 = _mm_mul_pd(a_im, t_swap);

        // result = [re*tw_re - im*tw_im, re*tw_im + im*tw_re]
        // Negate prod2[0] and add: use sign flip on low element
        let sign = _mm_set_pd(0.0_f64, -0.0_f64); // negate low
        let prod2_signed = _mm_xor_pd(prod2, sign);
        let result = _mm_add_pd(prod1, prod2_signed);

        unsafe { _mm_storeu_pd(data_ptr.add(i * 2), result) };
    }
}

/// NEON complex multiply: processes 1 complex f64 value per iteration
/// (2 f64 scalars = one `float64x2_t`).
///
/// # Safety
///
/// Caller must ensure NEON is available (guaranteed on aarch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn twiddle_mul_neon_f64(data: &mut [Complex<f64>], twiddles: &[Complex<f64>]) {
    use core::arch::aarch64::*;

    let data_ptr = data.as_mut_ptr() as *mut f64;
    let tw_ptr = twiddles.as_ptr() as *const f64;

    for i in 0..data.len() {
        unsafe {
            // Load [re, im] for data and twiddle
            let d = vld1q_f64(data_ptr.add(i * 2));
            let t = vld1q_f64(tw_ptr.add(i * 2));

            // a_re = [re, re], a_im = [im, im]
            let a_re = vdupq_lane_f64(vget_low_f64(d), 0);
            let a_im = vdupq_lane_f64(vget_high_f64(d), 0);

            // t_swap = [tw_im, tw_re]
            let t_swap = vextq_f64(t, t, 1);

            // prod1 = [re*tw_re, re*tw_im]
            let prod1 = vmulq_f64(a_re, t);
            // prod2 = [im*tw_im, im*tw_re]
            let prod2 = vmulq_f64(a_im, t_swap);

            // result = [re*tw_re - im*tw_im, re*tw_im + im*tw_re]
            // Use vfmaq_f64(prod1, prod2, sign) where sign = [-1.0, 1.0]
            let sign = vld1q_f64([(-1.0_f64), 1.0_f64].as_ptr());
            let result = vfmaq_f64(prod1, prod2, sign);

            vst1q_f64(data_ptr.add(i * 2), result);
        }
    }
}

// ============================================================================
// SoA SIMD twiddle multiplication — f64
// ============================================================================

/// Apply twiddle factors from separate `twiddle_re`/`twiddle_im` SoA arrays
/// to `data` in-place.
///
/// Compared to the interleaved [`twiddle_mul_simd_f64`], this SoA version
/// avoids the deinterleave shuffle for the twiddle vector, which gives a small
/// throughput improvement on large transforms where the twiddle data is
/// already resident in L2/L3 cache.
///
/// # Panics
///
/// Panics if `data.len()`, `twiddle_re.len()`, or `twiddle_im.len()` differ.
pub fn twiddle_mul_soa_simd_f64(data: &mut [Complex<f64>], twiddle_re: &[f64], twiddle_im: &[f64]) {
    assert_eq!(
        data.len(),
        twiddle_re.len(),
        "twiddle_mul_soa_simd_f64: re length mismatch"
    );
    assert_eq!(
        data.len(),
        twiddle_im.len(),
        "twiddle_mul_soa_simd_f64: im length mismatch"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: avx2 and fma confirmed above
            return unsafe { twiddle_mul_soa_avx2_f64(data, twiddle_re, twiddle_im) };
        }
        if is_x86_feature_detected!("sse2") {
            // SAFETY: sse2 confirmed above
            return unsafe { twiddle_mul_soa_sse2_f64(data, twiddle_re, twiddle_im) };
        }
        return twiddle_mul_soa_scalar_f64(data, twiddle_re, twiddle_im);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON always present on aarch64
        unsafe { twiddle_mul_soa_neon_f64(data, twiddle_re, twiddle_im) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    twiddle_mul_soa_scalar_f64(data, twiddle_re, twiddle_im);
}

/// AVX2+FMA SoA complex multiply: 2 complex f64 values per iteration.
///
/// Data layout for `data`: interleaved `[d_re0, d_im0, d_re1, d_im1]`.
/// Twiddle factors in separate `twiddle_re`/`twiddle_im` arrays.
///
/// We build `t_re = [tw_re0, tw_re0, tw_re1, tw_re1]` and
/// `t_im = [tw_im0, tw_im0, tw_im1, tw_im1]` using `_mm256_set_pd`,
/// avoiding the AoS deinterleave permute entirely.
///
/// # Safety
///
/// Caller must ensure the target CPU supports the `avx2` and `fma` features.
/// Calling this function on a CPU that lacks these features causes undefined
/// behavior (illegal instruction trap at runtime).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn twiddle_mul_soa_avx2_f64(
    data: &mut [Complex<f64>],
    twiddle_re: &[f64],
    twiddle_im: &[f64],
) {
    use core::arch::x86_64::*;

    let chunks = data.len() / 2;
    let data_ptr = data.as_mut_ptr() as *mut f64;
    let re_ptr = twiddle_re.as_ptr();
    let im_ptr = twiddle_im.as_ptr();

    for i in 0..chunks {
        // data: [d_re0, d_im0, d_re1, d_im1]
        let d = unsafe { _mm256_loadu_pd(data_ptr.add(i * 4)) };

        // Build interleaved twiddle [tw_re0, tw_im0, tw_re1, tw_im1] from SoA scalars.
        // _mm256_set_pd fills elements [3,2,1,0] from left-to-right args.
        let tw_re0 = unsafe { *re_ptr.add(i * 2) };
        let tw_re1 = unsafe { *re_ptr.add(i * 2 + 1) };
        let tw_im0 = unsafe { *im_ptr.add(i * 2) };
        let tw_im1 = unsafe { *im_ptr.add(i * 2 + 1) };
        let t = _mm256_set_pd(tw_im1, tw_re1, tw_im0, tw_re0);
        // Swap re/im within each 128-bit lane: [tw_im0, tw_re0, tw_im1, tw_re1]
        let t_swap = _mm256_permute_pd(t, 0b0101);

        // Duplicate data re and im parts
        let d_re = _mm256_permute_pd(d, 0b0000); // [re0, re0, re1, re1]
        let d_im = _mm256_permute_pd(d, 0b1111); // [im0, im0, im1, im1]

        // Use same mul+addsub as AoS path for consistent floating-point results.
        let prod1 = _mm256_mul_pd(d_re, t); // [re0*r0, re0*i0, re1*r1, re1*i1]
        let prod2 = _mm256_mul_pd(d_im, t_swap); // [im0*i0, im0*r0, im1*i1, im1*r1]
                                                 // addsub: [prod1[k]-prod2[k] for even k, prod1[k]+prod2[k] for odd k]
        let result = _mm256_addsub_pd(prod1, prod2);

        unsafe { _mm256_storeu_pd(data_ptr.add(i * 4), result) };
    }

    let remainder_start = chunks * 2;
    for i in remainder_start..data.len() {
        let d_re = data[i].re;
        let d_im = data[i].im;
        let tw_re = twiddle_re[i];
        let tw_im = twiddle_im[i];
        data[i].re = d_re * tw_re - d_im * tw_im;
        data[i].im = d_re * tw_im + d_im * tw_re;
    }
}

/// SSE2 SoA complex multiply: 1 complex f64 per iteration.
///
/// # Safety
///
/// Caller must ensure the target CPU supports the `sse2` feature.
/// Calling this function on a CPU that lacks `sse2` causes undefined
/// behavior (illegal instruction trap at runtime).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn twiddle_mul_soa_sse2_f64(
    data: &mut [Complex<f64>],
    twiddle_re: &[f64],
    twiddle_im: &[f64],
) {
    use core::arch::x86_64::*;

    let data_ptr = data.as_mut_ptr() as *mut f64;

    for i in 0..data.len() {
        // [d_re, d_im]
        let d = unsafe { _mm_loadu_pd(data_ptr.add(i * 2)) };

        let tw_re = twiddle_re[i];
        let tw_im = twiddle_im[i];

        // [tw_re, tw_re], [tw_im, tw_im]
        let t_re = _mm_set1_pd(tw_re);
        let t_im = _mm_set1_pd(tw_im);

        // d_re = [d.re, d.re], d_im = [d.im, d.im]
        let d_re = _mm_unpacklo_pd(d, d);
        let d_im = _mm_unpackhi_pd(d, d);

        // res_re_vec = [d_re*tw_re, d_re*tw_re], res_im_vec = [d_im*tw_im, d_im*tw_im]
        let prod_re = _mm_mul_pd(d_re, t_re);
        let prod_im = _mm_mul_pd(d_im, t_im);
        let prod_cross_re = _mm_mul_pd(d_re, t_im);
        let prod_cross_im = _mm_mul_pd(d_im, t_re);

        // result_re = prod_re - prod_im, result_im = prod_cross_re + prod_cross_im
        // Extract scalars (index 0 of each)
        let res_re = _mm_sub_pd(prod_re, prod_im);
        let res_im = _mm_add_pd(prod_cross_re, prod_cross_im);

        // Pack: [res_re[0], res_im[0]]
        let result = _mm_unpacklo_pd(res_re, res_im);
        unsafe { _mm_storeu_pd(data_ptr.add(i * 2), result) };
    }
}

/// NEON SoA complex multiply: 1 complex f64 per iteration.
///
/// # Safety
///
/// Caller must ensure the target CPU supports the `neon` feature.
/// On aarch64, NEON is always available; this is enforced by the
/// `#[target_feature(enable = "neon")]` attribute.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn twiddle_mul_soa_neon_f64(
    data: &mut [Complex<f64>],
    twiddle_re: &[f64],
    twiddle_im: &[f64],
) {
    use core::arch::aarch64::*;

    let data_ptr = data.as_mut_ptr() as *mut f64;

    for i in 0..data.len() {
        unsafe {
            let d = vld1q_f64(data_ptr.add(i * 2));
            let tw_re = twiddle_re[i];
            let tw_im = twiddle_im[i];

            let d_re = vdupq_lane_f64(vget_low_f64(d), 0);
            let d_im = vdupq_lane_f64(vget_high_f64(d), 0);

            let t_re = vdupq_n_f64(tw_re);
            let t_im = vdupq_n_f64(tw_im);

            // result_re = d_re*tw_re - d_im*tw_im
            // result_im = d_re*tw_im + d_im*tw_re
            let prod_re_re = vmulq_f64(d_re, t_re);
            let prod_im_im = vmulq_f64(d_im, t_im);
            let prod_re_im = vmulq_f64(d_re, t_im);
            let prod_im_re = vmulq_f64(d_im, t_re);

            let res_re = vsubq_f64(prod_re_re, prod_im_im);
            let res_im = vaddq_f64(prod_re_im, prod_im_re);

            // Pack [res_re[0], res_im[0]]
            let result = vzip1q_f64(res_re, res_im);
            vst1q_f64(data_ptr.add(i * 2), result);
        }
    }
}

// ============================================================================
// SoA SIMD twiddle multiplication — f32
// ============================================================================

/// Scalar SoA fallback for f32.
pub fn twiddle_mul_soa_scalar_f32(
    data: &mut [Complex<f32>],
    twiddle_re: &[f32],
    twiddle_im: &[f32],
) {
    debug_assert_eq!(data.len(), twiddle_re.len(), "SoA re length mismatch (f32)");
    debug_assert_eq!(data.len(), twiddle_im.len(), "SoA im length mismatch (f32)");
    for ((d, &tw_re), &tw_im) in data
        .iter_mut()
        .zip(twiddle_re.iter())
        .zip(twiddle_im.iter())
    {
        let (d_re, d_im) = (d.re, d.im);
        d.re = d_re * tw_re - d_im * tw_im;
        d.im = d_re * tw_im + d_im * tw_re;
    }
}

/// Apply SoA twiddle factors to `data` in-place (f32).
///
/// Dispatches to the best available SIMD backend: AVX2 → SSE2 → NEON → scalar.
///
/// # Panics
///
/// Panics if the three slice lengths differ.
pub fn twiddle_mul_soa_simd_f32(data: &mut [Complex<f32>], twiddle_re: &[f32], twiddle_im: &[f32]) {
    assert_eq!(
        data.len(),
        twiddle_re.len(),
        "twiddle_mul_soa_simd_f32: re length mismatch"
    );
    assert_eq!(
        data.len(),
        twiddle_im.len(),
        "twiddle_mul_soa_simd_f32: im length mismatch"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { twiddle_mul_soa_avx2_f32(data, twiddle_re, twiddle_im) };
        }
        if is_x86_feature_detected!("sse2") {
            return unsafe { twiddle_mul_soa_sse2_f32(data, twiddle_re, twiddle_im) };
        }
        return twiddle_mul_soa_scalar_f32(data, twiddle_re, twiddle_im);
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { twiddle_mul_soa_neon_f32(data, twiddle_re, twiddle_im) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    twiddle_mul_soa_scalar_f32(data, twiddle_re, twiddle_im);
}

/// AVX2+FMA SoA complex multiply (f32): 4 complex per iteration.
///
/// Loads 4 values from `twiddle_re` and `twiddle_im`, duplicates each
/// into adjacent pairs to match the interleaved data layout, then computes
/// the complex product via FMA.
///
/// `t_re = [r0,r0, r1,r1, r2,r2, r3,r3]` built with `_mm256_set_ps`.
/// `res_re = [rr0,rr0, rr1,rr1, rr2,rr2, rr3,rr3]` after FMA.
/// Interleave with `res_im` by shuffling even elements from each half.
///
/// # Safety
///
/// Caller must ensure the target CPU supports the `avx2` and `fma` features.
/// Calling this function on a CPU that lacks these features causes undefined
/// behavior (illegal instruction trap at runtime).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn twiddle_mul_soa_avx2_f32(
    data: &mut [Complex<f32>],
    twiddle_re: &[f32],
    twiddle_im: &[f32],
) {
    use core::arch::x86_64::*;

    let chunks = data.len() / 4;
    let data_ptr = data.as_mut_ptr() as *mut f32;
    let re_ptr = twiddle_re.as_ptr();
    let im_ptr = twiddle_im.as_ptr();

    for i in 0..chunks {
        // Load interleaved data: [re0,im0, re1,im1, re2,im2, re3,im3]
        let d = unsafe { _mm256_loadu_ps(data_ptr.add(i * 8)) };

        // Build interleaved twiddle [r0,i0,r1,i1,r2,i2,r3,i3] from SoA scalars.
        // _mm256_set_ps fills elements [7,6,5,4,3,2,1,0] from left-to-right args.
        let r0 = unsafe { *re_ptr.add(i * 4) };
        let r1 = unsafe { *re_ptr.add(i * 4 + 1) };
        let r2 = unsafe { *re_ptr.add(i * 4 + 2) };
        let r3 = unsafe { *re_ptr.add(i * 4 + 3) };
        let i0 = unsafe { *im_ptr.add(i * 4) };
        let i1 = unsafe { *im_ptr.add(i * 4 + 1) };
        let i2 = unsafe { *im_ptr.add(i * 4 + 2) };
        let i3 = unsafe { *im_ptr.add(i * 4 + 3) };
        let t = _mm256_set_ps(i3, r3, i2, r2, i1, r1, i0, r0);
        // Swap re/im within each pair: [i0,r0,i1,r1,i2,r2,i3,r3]
        let t_swap = _mm256_permute_ps(t, 0b10_11_00_01);

        // d_re = [re0,re0, re1,re1, re2,re2, re3,re3]
        let d_re = _mm256_moveldup_ps(d);
        // d_im = [im0,im0, im1,im1, im2,im2, im3,im3]
        let d_im = _mm256_movehdup_ps(d);

        // Use same mul+addsub as AoS path for consistent floating-point results.
        let prod1 = _mm256_mul_ps(d_re, t); // [re*r0, re*i0, re*r1, re*i1, ...]
        let prod2 = _mm256_mul_ps(d_im, t_swap); // [im*i0, im*r0, im*i1, im*r1, ...]
                                                 // addsub: [prod1[k]-prod2[k] for even k, prod1[k]+prod2[k] for odd k]
        let result = _mm256_addsub_ps(prod1, prod2);

        unsafe { _mm256_storeu_ps(data_ptr.add(i * 8), result) };
    }

    let remainder_start = chunks * 4;
    for i in remainder_start..data.len() {
        let d_re = data[i].re;
        let d_im = data[i].im;
        let tw_re = twiddle_re[i];
        let tw_im = twiddle_im[i];
        data[i].re = d_re * tw_re - d_im * tw_im;
        data[i].im = d_re * tw_im + d_im * tw_re;
    }
}

/// SSE2 SoA complex multiply (f32): 1 complex per iteration.
///
/// # Safety
///
/// Caller must ensure the target CPU supports the `sse2` feature.
/// Calling this function on a CPU that lacks `sse2` causes undefined
/// behavior (illegal instruction trap at runtime).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn twiddle_mul_soa_sse2_f32(
    data: &mut [Complex<f32>],
    twiddle_re: &[f32],
    twiddle_im: &[f32],
) {
    for i in 0..data.len() {
        let d_re = data[i].re;
        let d_im = data[i].im;
        let tw_re = twiddle_re[i];
        let tw_im = twiddle_im[i];
        data[i].re = d_re * tw_re - d_im * tw_im;
        data[i].im = d_re * tw_im + d_im * tw_re;
    }
}

/// NEON SoA complex multiply (f32): 2 complex per iteration.
///
/// Loads interleaved data `[re0, im0, re1, im1]`, builds twiddle registers
/// `t_re = [r0, r0, r1, r1]` and `t_im = [i0, i0, i1, i1]`, computes FMA
/// result, then interleaves back via `vtrn1q_f32` (takes even lanes).
///
/// # Safety
///
/// Caller must ensure the target CPU supports the `neon` feature.
/// On aarch64, NEON is always available; this is enforced by the
/// `#[target_feature(enable = "neon")]` attribute.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn twiddle_mul_soa_neon_f32(
    data: &mut [Complex<f32>],
    twiddle_re: &[f32],
    twiddle_im: &[f32],
) {
    use core::arch::aarch64::*;

    let chunks = data.len() / 2;
    let data_ptr = data.as_mut_ptr() as *mut f32;

    for i in 0..chunks {
        unsafe {
            // [re0, im0, re1, im1]
            let d = vld1q_f32(data_ptr.add(i * 4));

            let r0 = twiddle_re[i * 2];
            let r1 = twiddle_re[i * 2 + 1];
            let im0 = twiddle_im[i * 2];
            let im1 = twiddle_im[i * 2 + 1];

            let tw_re_arr = [r0, r0, r1, r1];
            let tw_im_arr = [im0, im0, im1, im1];
            let t_re = vld1q_f32(tw_re_arr.as_ptr());
            let t_im = vld1q_f32(tw_im_arr.as_ptr());

            // d_re = [re0, re0, re1, re1]
            let d_re = vtrn1q_f32(d, d);
            // d_im = [im0, im0, im1, im1]
            let d_im = vtrn2q_f32(d, d);

            // res_re[k] = d_re[k]*t_re[k] - d_im[k]*t_im[k]
            let res_re = vmlsq_f32(vmulq_f32(d_re, t_re), d_im, t_im);
            // res_im[k] = d_re[k]*t_im[k] + d_im[k]*t_re[k]
            let res_im = vmlaq_f32(vmulq_f32(d_re, t_im), d_im, t_re);

            // res_re = [rr0, rr0, rr1, rr1], res_im = [ri0, ri0, ri1, ri1]
            // vtrn1q_f32(res_re, res_im) = [res_re[0], res_im[0], res_re[2], res_im[2]]
            //                             = [rr0, ri0, rr1, ri1]  ✓
            let result = vtrn1q_f32(res_re, res_im);
            vst1q_f32(data_ptr.add(i * 4), result);
        }
    }

    let remainder_start = chunks * 2;
    for i in remainder_start..data.len() {
        let d_re = data[i].re;
        let d_im = data[i].im;
        let tw_re = twiddle_re[i];
        let tw_im = twiddle_im[i];
        data[i].re = d_re * tw_re - d_im * tw_im;
        data[i].im = d_re * tw_im + d_im * tw_re;
    }
}

// ============================================================================
// SIMD twiddle multiplication — f32
// ============================================================================

/// Apply twiddle factors to `data` in-place: `data[i] *= twiddles[i]`.
///
/// Dispatches to the best available SIMD backend at runtime:
/// AVX2+FMA → SSE2 → NEON → scalar.
///
/// # Panics
///
/// Panics if `data.len() != twiddles.len()`.
pub fn twiddle_mul_simd_f32(data: &mut [Complex<f32>], twiddles: &[Complex<f32>]) {
    assert_eq!(
        data.len(),
        twiddles.len(),
        "twiddle_mul_simd_f32: length mismatch"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: avx2 and fma features confirmed above
            return unsafe { twiddle_mul_avx2_f32(data, twiddles) };
        }
        if is_x86_feature_detected!("sse2") {
            // SAFETY: sse2 feature confirmed above
            return unsafe { twiddle_mul_sse2_f32(data, twiddles) };
        }
        // Fallback if somehow no x86 SIMD is available at runtime
        return twiddle_mul_scalar_f32(data, twiddles);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on aarch64 — always dispatch directly.
        // SAFETY: NEON is always present on aarch64.
        unsafe { twiddle_mul_neon_f32(data, twiddles) }
    }

    // Other architectures — plain scalar
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    twiddle_mul_scalar_f32(data, twiddles);
}

/// Scalar fallback: `data[i] *= twiddles[i]` with no SIMD.
pub fn twiddle_mul_scalar_f32(data: &mut [Complex<f32>], twiddles: &[Complex<f32>]) {
    for (d, t) in data.iter_mut().zip(twiddles.iter()) {
        *d = *d * *t;
    }
}

/// AVX2+FMA complex multiply for f32: processes 4 complex f32 values per
/// iteration (8 f32 scalars = one `__m256`).
///
/// Uses `_mm256_moveldup_ps` / `_mm256_movehdup_ps` to duplicate
/// real/imag parts and `_mm256_addsub_ps` for the alternating sign combination.
///
/// # Safety
///
/// Caller must ensure the target CPU supports the `avx2` and `fma` features.
/// Calling this function on a CPU that lacks these features causes undefined
/// behavior (illegal instruction trap at runtime).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn twiddle_mul_avx2_f32(data: &mut [Complex<f32>], twiddles: &[Complex<f32>]) {
    use core::arch::x86_64::*;

    let chunks = data.len() / 4; // 4 complex per __m256
    let data_ptr = data.as_mut_ptr() as *mut f32;
    let tw_ptr = twiddles.as_ptr() as *const f32;

    for i in 0..chunks {
        // Load 4 complex numbers: [re0,im0, re1,im1, re2,im2, re3,im3]
        let d = unsafe { _mm256_loadu_ps(data_ptr.add(i * 8)) };
        let t = unsafe { _mm256_loadu_ps(tw_ptr.add(i * 8)) };

        // Duplicate real parts: [re0,re0, re1,re1, re2,re2, re3,re3]
        let a_re = _mm256_moveldup_ps(d);
        // Duplicate imag parts: [im0,im0, im1,im1, im2,im2, im3,im3]
        let a_im = _mm256_movehdup_ps(d);

        // Swap re/im in twiddles: [im0,re0, im1,re1, im2,re2, im3,re3]
        let t_swap = _mm256_permute_ps(t, 0b10_11_00_01);

        // prod1 = [re0*tw_re0, re0*tw_im0, re1*tw_re1, re1*tw_im1, ...]
        let prod1 = _mm256_mul_ps(a_re, t);
        // prod2 = [im0*tw_im0, im0*tw_re0, im1*tw_im1, im1*tw_re1, ...]
        let prod2 = _mm256_mul_ps(a_im, t_swap);

        // addsub: [prod1[0]-prod2[0], prod1[1]+prod2[1], ...] ✓
        let result = _mm256_addsub_ps(prod1, prod2);

        unsafe { _mm256_storeu_ps(data_ptr.add(i * 8), result) };
    }

    // Handle remaining elements with scalar
    let remainder_start = chunks * 4;
    for i in remainder_start..data.len() {
        data[i] = data[i] * twiddles[i];
    }
}

/// SSE2 complex multiply for f32: processes 2 complex f32 values per iteration
/// (4 f32 scalars = one `__m128`).
///
/// # Safety
///
/// Caller must ensure the target CPU supports the `sse2` feature.
/// Calling this function on a CPU that lacks `sse2` causes undefined
/// behavior (illegal instruction trap at runtime).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn twiddle_mul_sse2_f32(data: &mut [Complex<f32>], twiddles: &[Complex<f32>]) {
    use core::arch::x86_64::*;

    let chunks = data.len() / 2; // 2 complex per __m128
    let data_ptr = data.as_mut_ptr() as *mut f32;
    let tw_ptr = twiddles.as_ptr() as *const f32;

    for i in 0..chunks {
        // Load 2 complex numbers: [re0, im0, re1, im1]
        let d = unsafe { _mm_loadu_ps(data_ptr.add(i * 4)) };
        let t = unsafe { _mm_loadu_ps(tw_ptr.add(i * 4)) };

        // Duplicate real parts: [re0, re0, re1, re1]
        let a_re = unsafe { _mm_moveldup_ps(d) };
        // Duplicate imag parts: [im0, im0, im1, im1]
        let a_im = unsafe { _mm_movehdup_ps(d) };

        // Swap re/im in twiddles: [im0, re0, im1, re1]
        let t_swap = _mm_shuffle_ps(t, t, 0b10_11_00_01);

        // prod1 = [re0*tw_re0, re0*tw_im0, re1*tw_re1, re1*tw_im1]
        let prod1 = _mm_mul_ps(a_re, t);
        // prod2 = [im0*tw_im0, im0*tw_re0, im1*tw_im1, im1*tw_re1]
        let prod2 = _mm_mul_ps(a_im, t_swap);

        // addsub: [prod1[0]-prod2[0], prod1[1]+prod2[1], prod1[2]-prod2[2], prod1[3]+prod2[3]]
        let result = unsafe { _mm_addsub_ps(prod1, prod2) };

        unsafe { _mm_storeu_ps(data_ptr.add(i * 4), result) };
    }

    // Handle remaining elements with scalar
    let remainder_start = chunks * 2;
    for i in remainder_start..data.len() {
        data[i] = data[i] * twiddles[i];
    }
}

/// NEON complex multiply for f32: processes 2 complex f32 values per iteration
/// (4 f32 scalars = one `float32x4_t`).
///
/// # Safety
///
/// Caller must ensure NEON is available (guaranteed on aarch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn twiddle_mul_neon_f32(data: &mut [Complex<f32>], twiddles: &[Complex<f32>]) {
    use core::arch::aarch64::*;

    let chunks = data.len() / 2; // 2 complex per float32x4_t
    let data_ptr = data.as_mut_ptr() as *mut f32;
    let tw_ptr = twiddles.as_ptr() as *const f32;

    for i in 0..chunks {
        unsafe {
            // Load [re0, im0, re1, im1]
            let d = vld1q_f32(data_ptr.add(i * 4));
            let t = vld1q_f32(tw_ptr.add(i * 4));

            // Duplicate real parts: [re0, re0, re1, re1]
            // vtrn1q_f32(d, d) picks lane 0 from each pair: [d[0], d[0], d[2], d[2]]
            let a_re = vtrn1q_f32(d, d);
            // Duplicate imag parts: [im0, im0, im1, im1]
            // vtrn2q_f32(d, d) picks lane 1 from each pair: [d[1], d[1], d[3], d[3]]
            let a_im = vtrn2q_f32(d, d);

            // Swap re/im in twiddles: [im0, re0, im1, re1]
            let t_swap = vrev64q_f32(t);

            // prod1 = [re0*tw_re0, re0*tw_im0, re1*tw_re1, re1*tw_im1]
            let prod1 = vmulq_f32(a_re, t);
            // prod2 = [im0*tw_im0, im0*tw_re0, im1*tw_im1, im1*tw_re1]
            let prod2 = vmulq_f32(a_im, t_swap);

            // result = [re0*tw_re0 - im0*tw_im0, re0*tw_im0 + im0*tw_re0, ...]
            // NEON lacks addsub; use sign vector [-1, 1, -1, 1]
            let sign = vld1q_f32([(-1.0_f32), 1.0_f32, (-1.0_f32), 1.0_f32].as_ptr());
            let result = vfmaq_f32(prod1, prod2, sign);

            vst1q_f32(data_ptr.add(i * 4), result);
        }
    }

    // Handle remaining elements with scalar
    let remainder_start = chunks * 2;
    for i in remainder_start..data.len() {
        data[i] = data[i] * twiddles[i];
    }
}

// ============================================================================
// Mixed-radix twiddle factor generation
// ============================================================================

/// Generate per-stage twiddle factor tables for a mixed-radix DIT FFT.
///
/// # Arguments
///
/// * `n` - Total transform size.
/// * `factors` - Ordered list of radices, innermost (smallest sub-problem) first.
///   The product of all factors must equal `n`.
/// * `direction` - `Forward` produces W_n^k = exp(-2πi k/n);
///   `Inverse` produces conjugates W_n^(-k) = exp(+2πi k/n).
///
/// # Returns
///
/// A `Vec` of per-stage twiddle tables, one entry per stage.  For stage `t`
/// with radix `r_t`, current partial size `current_n = r_0 * … * r_t`, and
/// stride `stride = current_n / r_t`:
///
/// ```text
/// table.len() == (r_t - 1) * stride
/// table[(j - 1) * stride + s] = W_{current_n}^{j * s}
///   for j in 1..r_t, s in 0..stride
/// ```
///
/// Stage 0 (innermost) always has `stride = 1`, so all twiddles equal 1 and
/// the table is a `Vec` of `(r_0 - 1)` ones.
///
/// # Panics
///
/// Panics if `factors` is empty or if `factors.iter().product::<u16>() as usize != n`.
#[must_use]
pub fn twiddles_mixed_radix(
    n: usize,
    factors: &[u16],
    direction: TwiddleDirection,
) -> Vec<Vec<Complex<f64>>> {
    assert!(
        !factors.is_empty(),
        "twiddles_mixed_radix: factors must be non-empty"
    );
    let product: usize = factors.iter().map(|&r| r as usize).product();
    assert_eq!(
        product, n,
        "twiddles_mixed_radix: product of factors ({product}) must equal n ({n})"
    );

    let sign = match direction {
        TwiddleDirection::Forward => -1.0_f64,
        TwiddleDirection::Inverse => 1.0_f64,
    };

    let mut tables: Vec<Vec<Complex<f64>>> = Vec::with_capacity(factors.len());
    let mut current_n: usize = 1;

    for &r_u16 in factors {
        let r = r_u16 as usize;
        // After processing this radix, the current partial DFT size becomes r * current_n.
        // But for the twiddle generation, the base is the NEW current_n after this radix.
        current_n *= r;
        // stride = current_n / r  (elements per "column" in the current DIT stage)
        let stride = current_n / r;
        // Table size: (r - 1) * stride entries
        let table_len = (r - 1) * stride;
        let mut table = Vec::with_capacity(table_len);

        // twiddles[(j-1)*stride + s] = exp(sign * 2πi * j * s / current_n)
        //   for j in 1..r, s in 0..stride
        for j in 1..r {
            for s in 0..stride {
                let angle = sign * 2.0 * core::f64::consts::PI * (j * s) as f64 / current_n as f64;
                table.push(Complex::new(angle.cos(), angle.sin()));
            }
        }

        tables.push(table);
    }

    tables
}

/// Generate per-stage twiddle factor tables for a mixed-radix DIT FFT (f32 variant).
///
/// Identical semantics to [`twiddles_mixed_radix`] but returns `f32` tables.
///
/// # Panics
///
/// Panics if `factors` is empty or if `factors.iter().product::<u16>() as usize != n`.
#[must_use]
pub fn twiddles_mixed_radix_f32(
    n: usize,
    factors: &[u16],
    direction: TwiddleDirection,
) -> Vec<Vec<Complex<f32>>> {
    assert!(
        !factors.is_empty(),
        "twiddles_mixed_radix_f32: factors must be non-empty"
    );
    let product: usize = factors.iter().map(|&r| r as usize).product();
    assert_eq!(
        product, n,
        "twiddles_mixed_radix_f32: product of factors ({product}) must equal n ({n})"
    );

    let sign = match direction {
        TwiddleDirection::Forward => -1.0_f32,
        TwiddleDirection::Inverse => 1.0_f32,
    };

    let mut tables: Vec<Vec<Complex<f32>>> = Vec::with_capacity(factors.len());
    let mut current_n: usize = 1;

    for &r_u16 in factors {
        let r = r_u16 as usize;
        current_n *= r;
        let stride = current_n / r;
        let table_len = (r - 1) * stride;
        let mut table = Vec::with_capacity(table_len);

        for j in 1..r {
            for s in 0..stride {
                let angle = sign * 2.0 * core::f32::consts::PI * (j * s) as f32 / current_n as f32;
                table.push(Complex::new(angle.cos(), angle.sin()));
            }
        }

        tables.push(table);
    }

    tables
}

#[cfg(test)]
mod twiddle_tests;
