//! Optimized codelets for small DFT sizes.
//!
//! Codelets are highly optimized kernels for specific transform sizes,
//! generated at compile time via procedural macros.

#[cfg(test)]
mod codegen_tests;
mod composite;
pub mod generated_simd;
#[cfg(target_arch = "x86_64")]
pub mod hand_avx512;
#[cfg(all(test, target_arch = "x86_64"))]
mod hand_avx512_tests;
#[cfg(target_arch = "x86_64")]
pub(crate) mod hand_avx512_twiddles;
mod notw;
pub mod simd;
mod twiddle;
pub(crate) mod twiddle_odd;
pub mod winograd;
pub(crate) mod winograd_constants;
pub(crate) mod winograd_pfa;

pub use composite::{
    execute_composite_codelet, has_composite_codelet, notw_100, notw_12, notw_15, notw_18, notw_20,
    notw_24, notw_30, notw_36, notw_45, notw_48, notw_50, notw_60, notw_72, notw_80, notw_96,
};
pub use generated_simd::{
    generated_simd_2_dispatch, generated_simd_4_dispatch, generated_simd_8_dispatch,
};
pub use notw::*;
pub use simd::{
    notw_1024_dispatch, notw_1024_simd_f64, notw_128_dispatch, notw_128_simd_f64, notw_16_dispatch,
    notw_16_simd_f64, notw_256_dispatch, notw_256_simd_f64, notw_2_dispatch, notw_2_simd_f64,
    notw_32_dispatch, notw_32_simd_f64, notw_4096_dispatch, notw_4096_simd_f64, notw_4_dispatch,
    notw_4_simd_f64, notw_512_dispatch, notw_512_simd_f64, notw_64_dispatch, notw_64_simd_f64,
    notw_8_dispatch, notw_8_simd_f64, simd_available,
};
pub use twiddle::*;
