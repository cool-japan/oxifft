//! # OxiFFT MPI Adapter
//!
//! `oxifft-adapter-mpi` is a **Pure-Rust-v2 §5 quarantine crate**. It isolates the
//! C MPI FFI (the `mpi` → `mpi-sys` / `libffi-sys` / `clang-sys` dependency chain)
//! out of the core `oxifft` facade so that `oxifft`'s `--all-features` dependency
//! closure stays 100% Pure Rust. All distributed-FFT functionality that needs to
//! link against a system MPI library lives here instead of in `oxifft`.
//!
//! This crate provides distributed FFT computation using MPI, similar to FFTW-MPI.
//! It enables computing large-scale FFTs across multiple processes.
//!
//! # Features
//!
//! - 2D, 3D, and N-D distributed FFTs
//! - Slab decomposition (row-major distribution)
//! - Pencil decomposition (2D process grid) for 3D transforms
//! - Efficient all-to-all transpose operations
//! - Compatible with FFTW-MPI data layouts
//!
//! # Example
//!
//! ```ignore
//! use oxifft_adapter_mpi::{MpiPool, MpiPlan2D, MpiFlags};
//! use mpi::topology::Communicator;
//!
//! // Initialize MPI
//! let universe = mpi::initialize().unwrap();
//! let world = universe.world();
//!
//! // Create MPI pool
//! let pool = MpiPool::new(world.duplicate());
//!
//! // Get local allocation size
//! let (local_n0, local_0_start, alloc_local) = local_size_2d(n0, n1, &pool);
//!
//! // Create plan
//! let plan = MpiPlan2D::new(n0, n1, Direction::Forward, MpiFlags::default(), &pool)?;
//!
//! // Execute
//! let mut data = vec![Complex::zero(); alloc_local];
//! plan.execute_inplace(&mut data);
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

mod distribution;
mod error;
mod local_size;
mod plans;
mod pool;
mod transpose;

pub use distribution::{Distribution, LocalPartition};
pub use error::MpiError;
pub use local_size::{
    local_size_2d, local_size_2d_r2c, local_size_2d_transposed, local_size_3d, local_size_3d_r2c,
    local_size_nd,
};
pub use plans::{
    MpiPlan2D, MpiPlan3D, MpiPlanND, MpiRealPlan2D, MpiRealPlan3D, PencilGrid, PencilPlan3D,
};
pub use pool::{MpiFloat, MpiPool};
pub use transpose::{
    distributed_transpose, distributed_transpose_batched, distributed_transpose_inplace,
};

use oxifft::api::Flags;

/// MPI-specific planning flags.
#[derive(Debug, Clone, Copy, Default)]
pub struct MpiFlags {
    /// Base FFT planning flags.
    pub base: Flags,
    /// Output data in transposed layout (avoids final transpose).
    /// Corresponds to FFTW_MPI_TRANSPOSED_OUT.
    pub transposed_out: bool,
    /// Input data is already transposed (corresponds to FFTW_MPI_TRANSPOSED_IN).
    ///
    /// **Not yet implemented.** Setting this flag causes `MpiPlan2D::new`,
    /// `MpiPlan3D::new` and `MpiPlanND::new` to return
    /// [`MpiError::FftError`] rather than silently
    /// ignoring it and producing wrong results. It is retained in the API so the
    /// transposed-input fast path can be added without a further breaking change.
    pub transposed_in: bool,
}

impl MpiFlags {
    /// Create new MPI flags with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the base FFT planning flags.
    pub fn with_base(mut self, flags: Flags) -> Self {
        self.base = flags;
        self
    }

    /// Enable transposed output (skips final transpose).
    pub fn transposed_out(mut self) -> Self {
        self.transposed_out = true;
        self
    }

    /// Indicate that input is already transposed.
    pub fn transposed_in(mut self) -> Self {
        self.transposed_in = true;
        self
    }

    /// Convenience: create ESTIMATE flags.
    pub fn estimate() -> Self {
        Self {
            base: Flags::ESTIMATE,
            ..Default::default()
        }
    }

    /// Convenience: create MEASURE flags.
    pub fn measure() -> Self {
        Self {
            base: Flags::MEASURE,
            ..Default::default()
        }
    }
}
