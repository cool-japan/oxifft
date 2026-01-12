//! Execution utilities for FFT plans.
//!
//! Note: Direct execution of FFT transforms is done through plan objects:
//! - `Plan::execute()` / `Plan::execute_inplace()` for complex DFT
//! - `RealPlan::execute_r2c()` / `RealPlan::execute_c2r()` for real FFT
//! - Convenience functions like `fft()`, `ifft()`, `rfft()`, `irfft()` for simple cases
//!
//! This module is reserved for future execution-related utilities.
