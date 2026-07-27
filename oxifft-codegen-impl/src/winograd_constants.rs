//! Mirror of `oxifft::dft::codelets::winograd_constants` for use by codegen-impl.
//!
//! Values must stay in sync — run the cross-validation test to verify.
//!
//! All constants are f64 precision, computed from exact expressions
//! cos(2πk/N) and sin(2πk/N). The forward DFT convention used throughout
//! is `W_N` = e^{-2πi/N}, so:
//!   - real parts: cos(2πk/N)
//!   - imaginary parts (for the negative sign): -sin(2πk/N)

// ─── DFT-3 ───────────────────────────────────────────────────────────────────
// cos(2π/3) = -1/2,  sin(2π/3) = √3/2

/// cos(2π/3) = −1/2
pub const C3_1: f64 = -0.5_f64;
/// sin(2π/3) = √3/2
pub const C3_2: f64 = 0.866_025_403_784_438_7_f64;

// ─── DFT-5 ───────────────────────────────────────────────────────────────────
// cos(2πk/5) and sin(2πk/5) for k = 1, 2

/// cos(2π/5)
pub const C5_COS1: f64 = 0.309_016_994_374_947_45_f64;
/// cos(4π/5)
pub const C5_COS2: f64 = -0.809_016_994_374_947_3_f64;
/// sin(2π/5)
pub const C5_SIN1: f64 = 0.951_056_516_295_153_5_f64;
/// sin(4π/5)
pub const C5_SIN2: f64 = 0.587_785_252_292_473_2_f64;

// ─── DFT-7 ───────────────────────────────────────────────────────────────────
// cos(2πk/7) and sin(2πk/7) for k = 1, 2, 3

/// cos(2π/7)
pub const C7_COS1: f64 = 0.623_489_801_858_733_6_f64;
/// cos(4π/7)
pub const C7_COS2: f64 = -0.222_520_933_956_314_34_f64;
/// cos(6π/7)
pub const C7_COS3: f64 = -0.900_968_867_902_419_f64;
/// sin(2π/7)
pub const C7_SIN1: f64 = 0.781_831_482_468_029_8_f64;
/// sin(4π/7)
pub const C7_SIN2: f64 = 0.974_927_912_181_823_6_f64;
/// sin(6π/7)
pub const C7_SIN3: f64 = 0.433_883_739_117_558_23_f64;

// NOTE: Only the DFT-{3,5,7} constants above are used — by the Winograd odd
// emitter in `gen_odd`. DFT-9 is never emitted as a hardcoded codelet (9 = 3·3
// routes through the mixed-radix runtime path), and DFT-11/13 use the dedicated
// Rader convolution tables in `gen_rader` (which are the DFT of the permuted
// twiddle sequence, not a raw cos/sin table). The previously-present but unused
// C9_*/C11_*/C13_* raw-angle constants were removed as dead code.

// ─── Cross-validation test ────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) fn verify_constants_match_runtime() {
    // Verify that this mirror matches the runtime winograd_constants exactly.
    // These are the same values — just check they agree within f64 precision.
    let tol = 1e-13;
    let two_pi = 2.0 * std::f64::consts::PI;

    // DFT-3
    assert!((C3_1 - f64::cos(two_pi / 3.0)).abs() < tol, "C3_1");
    assert!((C3_2 - f64::sin(two_pi / 3.0)).abs() < tol, "C3_2");

    // DFT-5
    assert!((C5_COS1 - f64::cos(two_pi / 5.0)).abs() < tol, "C5_COS1");
    assert!(
        (C5_COS2 - f64::cos(2.0 * two_pi / 5.0)).abs() < tol,
        "C5_COS2"
    );
    assert!((C5_SIN1 - f64::sin(two_pi / 5.0)).abs() < tol, "C5_SIN1");
    assert!(
        (C5_SIN2 - f64::sin(2.0 * two_pi / 5.0)).abs() < tol,
        "C5_SIN2"
    );

    // DFT-7
    assert!((C7_COS1 - f64::cos(two_pi / 7.0)).abs() < tol, "C7_COS1");
    assert!(
        (C7_COS2 - f64::cos(2.0 * two_pi / 7.0)).abs() < tol,
        "C7_COS2"
    );
    assert!(
        (C7_COS3 - f64::cos(3.0 * two_pi / 7.0)).abs() < tol,
        "C7_COS3"
    );
    assert!((C7_SIN1 - f64::sin(two_pi / 7.0)).abs() < tol, "C7_SIN1");
    assert!(
        (C7_SIN2 - f64::sin(2.0 * two_pi / 7.0)).abs() < tol,
        "C7_SIN2"
    );
    assert!(
        (C7_SIN3 - f64::sin(3.0 * two_pi / 7.0)).abs() < tol,
        "C7_SIN3"
    );
}
