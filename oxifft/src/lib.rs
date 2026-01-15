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
#![allow(clippy::missing_const_for_fn)] // const fn often not useful for runtime FFT
#![allow(clippy::doc_markdown)] // allow flexibility in documentation formatting
#![allow(clippy::return_self_not_must_use)] // builder patterns are fine
#![allow(clippy::incompatible_msrv)] // allow using newer features when available
#![allow(clippy::assign_op_pattern)] // a = a + b is fine for clarity
#![allow(clippy::needless_range_loop)] // explicit loops are clearer for FFT indices
#![allow(clippy::wildcard_imports)] // use super::* in submodules is fine
#![allow(clippy::unused_self)] // trait impls may have unused self
#![allow(clippy::too_many_arguments)] // FFT plans legitimately need many params
#![allow(clippy::redundant_closure_for_method_calls)] // allow explicit closures
#![allow(clippy::ptr_as_ptr)] // casting raw pointers is common in FFT
#![allow(clippy::fn_to_numeric_cast)] // allow function pointer casts
#![allow(clippy::suboptimal_flops)] // manual FMA control may be intentional
#![allow(clippy::manual_is_power_of_two)] // explicit bit ops may be clearer
#![allow(clippy::if_then_some_else_none)] // explicit if-else for clarity
#![allow(clippy::imprecise_flops)] // sqrt of squares may be intentional
#![allow(clippy::not_unsafe_ptr_arg_deref)] // FFT internal ops are safe
#![allow(clippy::derivable_impls)] // explicit impls may be clearer
#![allow(clippy::ptr_eq)] // allow ptr == ptr comparisons
#![allow(clippy::use_self)] // explicit type names may be clearer
#![allow(clippy::unnecessary_wraps)] // wrapping for API consistency
#![allow(clippy::too_many_lines)] // FFT functions can be long
#![allow(clippy::trivially_copy_pass_by_ref)] // consistent API style
#![allow(clippy::panic_in_result_fn)] // internal asserts for invariants
#![allow(clippy::match_same_arms)] // clarity in match arms
#![allow(clippy::suspicious_arithmetic_impl)] // complex arithmetic is intentional
#![allow(clippy::only_used_in_recursion)] // recursive FFT is intentional
#![allow(clippy::manual_div_ceil)] // explicit ceiling division
#![allow(clippy::float_cmp)] // intentional float comparison in tests
#![allow(clippy::cast_possible_truncation)] // deliberate truncation
#![allow(clippy::if_not_else)] // explicit negation for clarity
#![allow(clippy::manual_assert)] // if-panic patterns for clarity
#![allow(clippy::useless_let_if_seq)] // sequential mutation may be clearer
#![allow(clippy::ptr_cast_constness)] // pointer const casting common in FFT
#![allow(clippy::derive_partial_eq_without_eq)] // not all types are Eq
#![allow(clippy::format_push_string)] // format!() for clarity
#![allow(clippy::redundant_closure)] // explicit closures may be clearer
#![allow(clippy::significant_drop_tightening)] // locking patterns are intentional
#![allow(clippy::explicit_iter_loop)] // for i in iter.iter() for clarity
#![allow(clippy::unreadable_literal)] // FFT constants may use scientific notation
#![allow(clippy::missing_errors_doc)] // FFT error handling is self-explanatory
#![allow(clippy::manual_let_else)] // match patterns are clearer for Option handling
#![allow(clippy::type_complexity)] // complex return types are needed for FFT APIs
#![allow(clippy::needless_pass_by_value)] // API consistency for flags/options
#![allow(clippy::option_if_let_else)] // explicit if-let-else for clarity
#![allow(clippy::struct_field_names)] // field names matching type is intentional
#![allow(clippy::duplicate_mod)] // conditional compilation requires this
#![allow(clippy::suspicious_operation_groupings)] // FFT math has specific operator groupings
#![allow(clippy::verbose_bit_mask)] // explicit bit masks for clarity
#![allow(clippy::type_repetition_in_bounds)] // MPI traits need separate bounds
#![allow(clippy::no_effect_underscore_binding)] // placeholder variables for future use
#![allow(clippy::manual_clamp)] // explicit bounds checking is clearer
#![allow(clippy::used_underscore_binding)] // underscore prefix for internal use
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "portable_simd", feature(portable_simd))]

#[cfg(not(feature = "std"))]
extern crate alloc;

// Internal prelude for no_std compatibility
pub(crate) mod prelude;

pub mod api;
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
#[cfg(feature = "std")]
pub mod nufft;

// Re-export commonly used types
pub use api::{
    fft, fft2d, fft2d_split, fft3d_split, fft_batch, fft_nd, fft_nd_split, fft_split, ifft, ifft2d,
    ifft2d_split, ifft3d_split, ifft_batch, ifft_nd, ifft_nd_split, ifft_split, irfft, irfft2d,
    irfft3d, irfft_batch, irfft_nd, rfft, rfft2d, rfft3d, rfft_batch, rfft_nd, Direction, Flags,
    GuruPlan, Plan, Plan2D, Plan3D, PlanND, R2rKind, R2rPlan, RealPlan, RealPlan2D, RealPlan3D,
    RealPlanKind, RealPlanND, SplitPlan, SplitPlan2D, SplitPlan3D, SplitPlanND,
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
    blackman, cola_normalization, hamming, hann, istft, kaiser, magnitude_spectrogram,
    phase_spectrogram, power_spectrogram, rectangular, stft, RingBuffer, StreamingFft,
    WindowFunction,
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
    best_backend, is_gpu_available, query_capabilities, GpuBackend, GpuBuffer, GpuCapabilities,
    GpuDirection, GpuError, GpuFft, GpuFftEngine, GpuPlan, GpuResult,
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
    nufft_type1, nufft_type2, nufft_type3, Nufft, NufftError, NufftOptions, NufftResult, NufftType,
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

// Re-export Automatic Differentiation
pub use autodiff::{
    fft_dual, fft_jacobian, grad_fft, grad_ifft, jvp_fft, vjp_fft, DiffFftPlan, Dual, DualComplex,
};
