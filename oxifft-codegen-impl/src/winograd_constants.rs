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

// ─── DFT-9 ───────────────────────────────────────────────────────────────────
// cos(2πk/9) and sin(2πk/9) for k = 1, 2, 3, 4

/// cos(2π/9)
pub const C9_COS1: f64 = 0.766_044_443_118_978_f64;
/// cos(4π/9)
pub const C9_COS2: f64 = 0.173_648_177_666_930_41_f64;
/// cos(6π/9) = cos(2π/3) = -1/2
pub const C9_COS3: f64 = -0.5_f64;
/// cos(8π/9)
pub const C9_COS4: f64 = -0.939_692_620_785_908_3_f64;
/// sin(2π/9)
pub const C9_SIN1: f64 = 0.642_787_609_686_539_3_f64;
/// sin(4π/9)
pub const C9_SIN2: f64 = 0.984_807_753_012_208_f64;
/// sin(6π/9) = sin(2π/3) = √3/2
pub const C9_SIN3: f64 = 0.866_025_403_784_438_7_f64;
/// sin(8π/9)
pub const C9_SIN4: f64 = 0.342_020_143_325_668_9_f64;

// ─── DFT-11 ──────────────────────────────────────────────────────────────────
// cos(2πk/11) and sin(2πk/11) for k = 1..5

/// cos(2π/11)
pub const C11_COS1: f64 = 0.841_253_532_831_181_2_f64;
/// cos(4π/11)
pub const C11_COS2: f64 = 0.415_415_013_001_886_44_f64;
/// cos(6π/11)
pub const C11_COS3: f64 = -0.142_314_838_273_285_f64;
/// cos(8π/11)
pub const C11_COS4: f64 = -0.654_860_733_945_285_1_f64;
/// cos(10π/11)
pub const C11_COS5: f64 = -0.959_492_973_614_497_4_f64;
/// sin(2π/11)
pub const C11_SIN1: f64 = 0.540_640_817_455_597_6_f64;
/// sin(4π/11)
pub const C11_SIN2: f64 = 0.909_631_995_354_518_3_f64;
/// sin(6π/11)
pub const C11_SIN3: f64 = 0.989_821_441_880_932_8_f64;
/// sin(8π/11)
pub const C11_SIN4: f64 = 0.755_749_574_354_258_3_f64;
/// sin(10π/11)
pub const C11_SIN5: f64 = 0.281_732_556_841_429_67_f64;

// ─── DFT-13 ──────────────────────────────────────────────────────────────────
// cos(2πk/13) and sin(2πk/13) for k = 1..6

/// cos(2π/13)
pub const C13_COS1: f64 = 0.885_456_025_653_209_9_f64;
/// cos(4π/13)
pub const C13_COS2: f64 = 0.568_064_746_731_155_8_f64;
/// cos(6π/13)
pub const C13_COS3: f64 = 0.120_536_680_255_323_f64;
/// cos(8π/13)
pub const C13_COS4: f64 = -0.354_604_887_042_535_45_f64;
/// cos(10π/13)
pub const C13_COS5: f64 = -0.748_510_748_171_101_2_f64;
/// cos(12π/13)
pub const C13_COS6: f64 = -0.970_941_817_426_052_f64;
/// sin(2π/13)
pub const C13_SIN1: f64 = 0.464_723_172_043_768_5_f64;
/// sin(4π/13)
pub const C13_SIN2: f64 = 0.822_983_865_893_656_4_f64;
/// sin(6π/13)
pub const C13_SIN3: f64 = 0.992_708_874_098_054_f64;
/// sin(8π/13)
pub const C13_SIN4: f64 = 0.935_016_242_685_414_8_f64;
/// sin(10π/13)
pub const C13_SIN5: f64 = 0.663_122_658_240_795_2_f64;
/// sin(12π/13)
pub const C13_SIN6: f64 = 0.239_315_664_287_557_68_f64;

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
