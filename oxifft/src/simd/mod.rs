//! SIMD abstraction layer for high-performance FFT computation.
//!
//! Provides a unified interface for SIMD operations across different architectures,
//! enabling vectorized FFT butterflies and complex arithmetic.
//!
//! # Overview
//!
//! OxiFFT's SIMD layer provides:
//! - **Automatic runtime detection** via [`detect_simd_level()`]
//! - **Unified traits** ([`SimdVector`], [`SimdComplex`]) for portable code
//! - **Architecture-specific implementations** for maximum performance
//!
//! # Available Backends
//!
//! | Backend | Architecture | Vector Width | Lanes (f64) | Lanes (f32) | Features |
//! |---------|-------------|--------------|-------------|-------------|----------|
//! | [`Scalar`] | All | 64/32-bit | 1 | 1 | Always available |
//! | `Sse2F64`/`Sse2F32` | x86_64 | 128-bit | 2 | 4 | SSE2 (baseline x86_64) |
//! | `AvxF64`/`AvxF32` | x86_64 | 256-bit | 4 | 8 | AVX |
//! | `Avx2F64`/`Avx2F32` | x86_64 | 256-bit | 4 | 8 | AVX2 + FMA3 |
//! | `Avx512F64`/`Avx512F32` | x86_64 | 512-bit | 8 | 16 | AVX-512F |
//! | `NeonF64`/`NeonF32` | aarch64 | 128-bit | 2 | 4 | NEON (mandatory) |
//! | Portable* | All | Variable | 2-8 | 4-16 | Nightly + `portable_simd` |
//!
//! *Portable SIMD requires nightly Rust and the `portable_simd` feature flag.
//!
//! # CPU Requirements
//!
//! ## x86_64
//!
//! - **SSE2**: Required for x86_64 (guaranteed on all modern CPUs since 2003)
//! - **AVX**: Intel Sandy Bridge (2011+), AMD Bulldozer (2011+)
//! - **AVX2 + FMA**: Intel Haswell (2013+), AMD Excavator (2015+)
//! - **AVX-512**: Intel Skylake-X (2017+), AMD Zen 4 (2022+), limited server CPUs
//!
//! ## aarch64 (ARM64)
//!
//! - **NEON**: Mandatory on aarch64, always available (Apple M1/M2/M3, AWS Graviton, Ampere)
//!
//! # Runtime Detection
//!
//! Use [`detect_simd_level()`] to query the highest available SIMD level at runtime:
//!
//! ```
//! use oxifft::simd::{detect_simd_level, SimdLevel};
//!
//! let level = detect_simd_level();
//! match level {
//!     SimdLevel::Avx512 => println!("Using AVX-512 (512-bit vectors)"),
//!     SimdLevel::Avx2 => println!("Using AVX2 with FMA (256-bit vectors)"),
//!     SimdLevel::Avx => println!("Using AVX (256-bit vectors)"),
//!     SimdLevel::Sse2 => println!("Using SSE2 (128-bit vectors)"),
//!     SimdLevel::Neon => println!("Using NEON (128-bit vectors)"),
//!     SimdLevel::Sve => println!("Using ARM SVE (scalable vectors)"),
//!     SimdLevel::Scalar => println!("No SIMD, using scalar fallback"),
//!     _ => println!("Unknown SIMD level"),
//! }
//! ```
//!
//! # Performance Guidelines
//!
//! ## Memory Alignment
//!
//! For optimal SIMD performance, data should be aligned:
//! - SSE2/NEON: 16-byte alignment
//! - AVX/AVX2: 32-byte alignment
//! - AVX-512: 64-byte alignment
//!
//! Use [`alloc_complex_aligned`](crate::alloc_complex_aligned) or [`AlignedBuffer`](crate::AlignedBuffer) for aligned memory.
//! Unaligned loads/stores work but may be slower on some architectures.
//!
//! ## Expected Speedups
//!
//! Typical speedups over scalar code for FFT operations:
//! - **SSE2/NEON**: 1.5-2x for f64, 2-3x for f32
//! - **AVX/AVX2**: 2-3x for f64, 3-5x for f32
//! - **AVX-512**: 3-5x for f64, 5-8x for f32
//!
//! Actual speedups depend on problem size, memory bandwidth, and cache behavior.
//!
//! ## FMA (Fused Multiply-Add)
//!
//! AVX2 and later include FMA instructions which:
//! - Compute `a * b + c` in a single operation
//! - Provide better precision (single rounding instead of two)
//! - Reduce pipeline stalls in complex arithmetic
//!
//! # Feature Flags
//!
//! - `portable_simd`: Enable experimental portable SIMD backend (requires nightly)
//!
//! # Example: Using SIMD Traits
//!
//! ```ignore
//! use oxifft::simd::{SimdVector, SimdComplex};
//!
//! fn complex_butterfly<V: SimdComplex>(a: V, b: V) -> (V, V) {
//!     V::butterfly(a, b) // Returns (a+b, a-b)
//! }
//! ```
//!
//! # Safety
//!
//! All SIMD types use `unsafe` internally but expose a safe API. The unsafe
//! operations are:
//! - `load_aligned`/`store_aligned`: Require proper alignment
//! - `load_unaligned`/`store_unaligned`: Require valid pointer for LANES elements
//!
//! OxiFFT's internal code handles alignment automatically.

mod detect;
mod scalar;
mod traits;

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod avx512;
#[cfg(target_arch = "x86_64")]
mod sse2;

#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(feature = "sve")]
mod sve;

#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "sve"))]
pub use detect::has_sve_runtime;
pub use detect::{detect_simd_level, SimdLevel};
#[cfg(target_arch = "x86_64")]
pub use detect::{has_avx, has_avx2, has_avx512};
pub use scalar::Scalar;
pub use traits::{SimdComplex, SimdVector};

#[cfg(target_arch = "x86_64")]
pub use sse2::{Sse2F32, Sse2F64};

#[cfg(target_arch = "x86_64")]
pub use avx::{AvxF32, AvxF64};

#[cfg(target_arch = "x86_64")]
pub use avx2::{has_avx2_fma, Avx2F32, Avx2F64};

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use avx512::{has_avx512f, Avx512F32, Avx512F64};

#[cfg(target_arch = "aarch64")]
pub use neon::{NeonF32, NeonF64};

#[cfg(feature = "sve")]
pub use sve::{
    has_sve, sve_f32_lanes, sve_f64_lanes, sve_vector_length_bytes, Sve256F32, Sve256F64,
    SvePredicate,
};
