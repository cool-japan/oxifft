//! # OxiFFT
//!
//! Pure Rust implementation of FFTW - the Fastest Fourier Transform in the West.
//!
//! OxiFFT provides high-performance FFT routines with:
//! - Complex DFT (forward and inverse)
//! - Real FFT (R2C/C2R)
//! - DCT/DST transforms
//! - Multi-dimensional transforms
//! - Batch processing
//! - SIMD optimization
//! - Threading support
//! - Wisdom system for plan caching
//!
//! ## Quick Start
//!
//! ```ignore
//! use oxifft::{Complex, Direction, Flags, Plan};
//!
//! // Create a plan for 256-point forward FFT
//! let plan = Plan::dft_1d(256, Direction::Forward, Flags::MEASURE)?;
//!
//! // Execute the transform
//! let mut input = vec![Complex::new(1.0, 0.0); 256];
//! let mut output = vec![Complex::zero(); 256];
//! plan.execute(&mut input, &mut output);
//! ```

// Allow pedantic/nursery warnings that are intentional in FFT code:
#![allow(clippy::similar_names)] // fwd/bwd, real/imag pairs are intentionally similar
#![allow(clippy::many_single_char_names)] // FFT math uses i, j, k, n, m by convention
#![allow(clippy::cast_precision_loss)] // FFT size computations use float for math
#![allow(clippy::cast_sign_loss)] // stride/offset calculations need signed/unsigned
#![allow(clippy::cast_possible_wrap)] // stride calculations need careful wrapping
#![allow(clippy::missing_panics_doc)] // many internal functions assert preconditions
#![allow(clippy::must_use_candidate)] // internal helpers don't need must_use
#![allow(clippy::doc_markdown)] // allow flexibility in documentation formatting
#![allow(clippy::incompatible_msrv)] // allow using newer features when available
#![allow(clippy::needless_range_loop)] // explicit loops are clearer for FFT indices
#![allow(clippy::wildcard_imports)] // use super::* in submodules is fine
#![allow(clippy::too_many_arguments)] // FFT plans legitimately need many params
#![allow(clippy::assign_op_pattern)] // a = a op b in codelet math avoids confusion
#![allow(clippy::ptr_as_ptr)] // casting raw pointers in FFT is pervasive
#![allow(clippy::suboptimal_flops)] // manual FMA control may be intentional
#![allow(clippy::imprecise_flops)] // sqrt of squares may be intentional
#![allow(clippy::not_unsafe_ptr_arg_deref)] // FFT internal ops are safe
#![allow(clippy::unnecessary_wraps)] // wrapping for API consistency
#![allow(clippy::too_many_lines)] // FFT functions can be long
#![allow(clippy::suspicious_arithmetic_impl)] // complex arithmetic is intentional
#![allow(clippy::only_used_in_recursion)] // recursive FFT is intentional
#![allow(clippy::float_cmp)] // intentional float comparison in tests
#![allow(clippy::cast_possible_truncation)] // deliberate truncation
#![allow(clippy::ptr_cast_constness)] // pointer const casting common in FFT
#![allow(clippy::significant_drop_tightening)] // locking patterns are intentional
#![allow(clippy::type_complexity)] // complex return types are needed for FFT APIs
#![allow(clippy::duplicate_mod)] // conditional compilation requires this
#![allow(clippy::suspicious_operation_groupings)] // FFT math has specific operator groupings
#![allow(clippy::missing_const_for_fn)] // many fns could be const but don't need to be
#![allow(clippy::return_self_not_must_use)] // builder patterns don't need must_use everywhere
#![allow(clippy::use_self)] // explicit type names preferred for clarity in FFT code
#![allow(clippy::option_if_let_else)] // if-let-else is clearer than map_or in some cases
#![allow(clippy::redundant_else)] // explicit else improves readability
#![allow(clippy::if_not_else)] // negated conditions are sometimes clearer
#![allow(clippy::unnested_or_patterns)] // flat patterns are easier to read
#![allow(clippy::unreadable_literal)] // FFT constants appear as hex/long literals
#![allow(clippy::unused_self)] // some trait methods require &self but don't use it
#![allow(clippy::redundant_closure_for_method_calls)] // explicit closures can be clearer
#![allow(clippy::unnecessary_cast)] // casts explicit for type documentation
#![allow(clippy::inline_always)] // FFT codelets need inline(always) for performance
#![allow(clippy::approx_constant)] // FFT constants may approximate std consts intentionally
#![allow(clippy::manual_let_else)] // let-else not always clearer in FFT context
#![allow(clippy::iter_without_into_iter)] // not all iterators need IntoIterator
#![allow(clippy::implicit_clone)] // cloning via deref is idiomatic
#![allow(clippy::cast_lossless)] // explicit casting preferred for documentation
#![allow(clippy::trivially_copy_pass_by_ref)] // API consistency for small types
#![allow(clippy::map_unwrap_or)] // map().unwrap_or() is clearer sometimes
#![allow(clippy::explicit_iter_loop)] // explicit .iter() is clear
#![allow(clippy::derive_partial_eq_without_eq)] // not all PartialEq types need Eq
#![allow(clippy::single_match_else)] // single match + else is sometimes clearer
#![allow(clippy::match_same_arms)] // identical arms for documentation/future changes
#![allow(clippy::items_after_statements)] // local items near usage is fine
#![allow(clippy::manual_assert)] // manual if + panic is sometimes clearer
#![allow(clippy::std_instead_of_core)] // std re-exports are fine
#![allow(clippy::separated_literal_suffix)] // literal suffixes may be detached
#![allow(clippy::used_underscore_binding)] // underscore bindings used intentionally
#![allow(clippy::manual_div_ceil)] // manual div_ceil for clarity
#![allow(clippy::if_then_some_else_none)] // explicit if/else preferred sometimes
#![allow(clippy::struct_field_names)] // field names may match struct name
#![allow(clippy::default_trait_access)] // Default::default() is fine
#![allow(clippy::expl_impl_clone_on_copy)] // explicit Clone for Copy types is intentional
#![allow(clippy::format_push_string)] // format! appended to String is fine
#![allow(clippy::needless_pass_by_value)] // pass by value for API ergonomics
#![allow(clippy::copy_iterator)] // iterators may impl Copy
#![allow(clippy::manual_clamp)] // manual clamp for clarity
#![allow(clippy::manual_is_variant_and)] // explicit matching is fine
#![allow(clippy::unseparated_literal_suffix)] // literal suffixes style choice
#![allow(clippy::checked_conversions)] // explicit conversion checks preferred
#![allow(clippy::semicolon_if_nothing_returned)] // statement style
#![allow(clippy::ref_as_ptr)] // ref as pointer for low-level FFT
#![allow(clippy::ptr_eq)]
// raw pointer comparison

// Compiler-enforced invariant: every unsafe fn must have a `# Safety` section.
#![warn(clippy::missing_safety_doc)]
// Compiler-enforced invariant: every fallible public fn must have a `# Errors` section.
#![warn(clippy::missing_errors_doc)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

// Internal prelude for no_std compatibility
pub(crate) mod prelude;

// Compile-time Send+Sync assertions for all public plan types.
mod assertions;

pub mod api;
#[cfg(feature = "fftw-compat")]
pub mod compat;
#[cfg(feature = "const-fft")]
pub mod const_fft;
pub mod dft;
#[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
pub mod gpu;
pub mod kernel;
#[cfg(feature = "mpi")]
pub mod mpi;
#[cfg(feature = "pruned")]
pub mod pruned;
pub mod rdft;
pub mod reodft;
#[cfg(feature = "signal")]
pub mod signal;
pub mod simd;
#[cfg(feature = "sparse")]
pub mod sparse;
#[cfg(feature = "streaming")]
pub mod streaming;
pub mod support;
pub mod threading;
#[cfg(feature = "wasm")]
pub mod wasm;

// Advanced FFT transforms
pub mod autodiff;
pub mod conv;
#[cfg(feature = "std")]
pub mod frft;
pub mod ntt;
#[cfg(feature = "std")]
pub mod nufft;

// Re-export commonly used types
pub use api::{
    fft, fft2d, fft2d_split, fft3d_split, fft_batch, fft_nd, fft_nd_split, fft_split, ifft, ifft2d,
    ifft2d_split, ifft3d_split, ifft_batch, ifft_nd, ifft_nd_split, ifft_split, irfft, irfft2d,
    irfft3d, irfft_batch, irfft_nd, rfft, rfft2d, rfft3d, rfft_batch, rfft_nd, Direction, Flags,
    GuruPlan, InvalidDirection, Plan, Plan2D, Plan3D, PlanND, R2rKind, R2rPlan, RealPlan,
    RealPlan2D, RealPlan3D, RealPlanKind, RealPlanND, SplitPlan, SplitPlan2D, SplitPlan3D,
    SplitPlanND,
};
pub use kernel::{Complex, Float, IoDim, Tensor};

// Re-export F128 when f128-support feature is enabled
#[cfg(feature = "f128-support")]
pub use kernel::F128;

// Re-export F16 when f16-support feature is enabled
#[cfg(feature = "f16-support")]
pub use kernel::F16;

// Re-export sparse FFT when sparse feature is enabled
#[cfg(feature = "sparse")]
pub use sparse::{sparse_fft, sparse_ifft, SparsePlan, SparseResult};

// Re-export pruned FFT when pruned feature is enabled
#[cfg(feature = "pruned")]
pub use pruned::{
    fft_pruned_input, fft_pruned_output, goertzel, goertzel_multi, PrunedPlan, PruningMode,
};

// Re-export WASM bindings when wasm feature is enabled
#[cfg(feature = "wasm")]
pub use wasm::{fft_f32, fft_f64, ifft_f32, ifft_f64, rfft_f64, WasmFft, WasmSimdF32, WasmSimdF64};

// Re-export streaming FFT when streaming feature is enabled
#[cfg(feature = "streaming")]
pub use streaming::{
    blackman, build_mel_filterbank, cola_normalization, hamming, hann, istft, istft_overlap_save,
    kaiser, magnitude_spectrogram, mel_spectrogram, mfcc, phase_spectrogram, power_spectrogram,
    rectangular, single_bin_tracker, sliding_dft, stft, stft_overlap_save, MelConfig,
    ModulatedSdft, RingBuffer, SingleBinTracker, SlidingDft, StreamingFft, WindowFunction,
};

// Re-export compile-time FFT when const-fft feature is enabled
#[cfg(feature = "const-fft")]
pub use const_fft::{
    const_cos, const_sin, fft_fixed, fft_fixed_inplace, ifft_fixed, ifft_fixed_inplace,
    twiddle_factor, ConstFft, ConstFftImpl,
};

// Re-export GPU FFT when gpu/cuda/metal feature is enabled
#[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
pub use gpu::{
    best_backend, is_gpu_available, query_capabilities, GpuBackend, GpuBatchFft, GpuBuffer,
    GpuCapabilities, GpuDirection, GpuError, GpuFft, GpuFftEngine, GpuPlan, GpuResult,
};

// Re-export low-level DFT functions for advanced users
pub use dft::solvers::{
    ct::{
        fft_radix2, fft_radix2_inplace, ifft_radix2, ifft_radix2_inplace, ifft_radix2_normalized,
    },
    direct::{dft_direct, idft_direct, idft_direct_normalized},
    nop::dft_nop,
};

// Re-export DCT/DST/DHT convenience functions
pub use rdft::solvers::{dct1, dct2, dct3, dct4, dht, dst1, dst2, dst3, dst4};

// Re-export memory allocation utilities
pub use api::{
    alloc_complex, alloc_complex_aligned, alloc_real, alloc_real_aligned, is_aligned,
    AlignedBuffer, DEFAULT_ALIGNMENT,
};

// Re-export NUFFT (Non-uniform FFT)
#[cfg(feature = "std")]
pub use nufft::{
    nufft2d_type1, nufft2d_type2, nufft3d_type1, nufft_type1, nufft_type2, nufft_type3, Nufft,
    NufftError, NufftOptions, NufftResult, NufftType,
};

// Re-export FrFT (Fractional Fourier Transform)
#[cfg(feature = "std")]
pub use frft::{frft, frft_checked, ifrft, ifrft_checked, Frft, FrftError, FrftResult};

// Re-export Convolution functions
pub use conv::{
    convolve, convolve_circular, convolve_complex, convolve_complex_mode, convolve_mode,
    convolve_with_mode, correlate, correlate_complex, correlate_complex_mode, correlate_mode,
    polynomial_multiply, polynomial_multiply_complex, polynomial_power, ConvMode,
};

// Re-export NTT (Number Theoretic Transform) functions
pub use ntt::{
    intt, ntt, ntt_convolve, ntt_convolve_default, NttError, NttPlan, NTT_PRIME_998244353,
    NTT_PRIME_MOD1, NTT_PRIME_MOD2,
};

// Re-export Automatic Differentiation
pub use autodiff::{
    fft_dual, fft_jacobian, grad_fft, grad_ifft, jvp_fft, vjp_fft, DiffFftPlan, Dual, DualComplex,
};

// Re-export signal processing when signal feature is enabled
#[cfg(feature = "signal")]
pub use signal::{
    coherence, complex_cepstrum, cross_spectral_density, envelope, hilbert,
    instantaneous_frequency, instantaneous_phase, minimum_phase, periodogram, real_cepstrum,
    resample, resample_to, welch, SpectralWindow, WelchConfig,
};
