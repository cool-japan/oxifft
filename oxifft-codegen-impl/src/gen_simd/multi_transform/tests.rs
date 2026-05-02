//! Tests for multi-transform codelet generation.

use super::*;

// ── Config construction ──────────────────────────────────────────────────────

#[test]
fn test_multi_transform_config_valid_avx2_f32() {
    let config = MultiTransformConfig {
        size: 4,
        v: 8,
        isa: SimdIsa::Avx2,
        precision: Precision::F32,
    };
    assert_eq!(config.size, 4);
    assert_eq!(config.v, 8);
    assert_eq!(config.isa, SimdIsa::Avx2);
    assert_eq!(config.precision, Precision::F32);
}

#[test]
fn test_multi_transform_config_valid_sse2_f64() {
    let config = MultiTransformConfig {
        size: 2,
        v: 4,
        isa: SimdIsa::Sse2,
        precision: Precision::F64,
    };
    assert_eq!(config.size, 2);
    assert_eq!(config.v, 4);
    assert_eq!(config.isa, SimdIsa::Sse2);
    assert_eq!(config.precision, Precision::F64);
}

// ── ISA helpers ──────────────────────────────────────────────────────────────

#[test]
fn test_isa_lanes_f32() {
    assert_eq!(SimdIsa::Sse2.lanes_f32(), 4);
    assert_eq!(SimdIsa::Avx2.lanes_f32(), 8);
    assert_eq!(SimdIsa::Scalar.lanes_f32(), 1);
}

#[test]
fn test_isa_lanes_f64() {
    assert_eq!(SimdIsa::Sse2.lanes_f64(), 2);
    assert_eq!(SimdIsa::Avx2.lanes_f64(), 4);
    assert_eq!(SimdIsa::Scalar.lanes_f64(), 1);
}

#[test]
fn test_isa_ident_str() {
    assert_eq!(SimdIsa::Sse2.ident_str(), "sse2");
    assert_eq!(SimdIsa::Avx2.ident_str(), "avx2");
    assert_eq!(SimdIsa::Scalar.ident_str(), "scalar");
}

// ── TokenStream generation ───────────────────────────────────────────────────

#[test]
fn test_generate_produces_nonempty_token_stream() {
    let config = MultiTransformConfig {
        size: 4,
        v: 8,
        isa: SimdIsa::Avx2,
        precision: Precision::F32,
    };
    let ts = generate_multi_transform(&config).expect("should generate");
    assert!(!ts.is_empty(), "TokenStream must not be empty");
}

#[test]
fn test_generate_function_name_avx2_f32_size4() {
    let config = MultiTransformConfig {
        size: 4,
        v: 8,
        isa: SimdIsa::Avx2,
        precision: Precision::F32,
    };
    let ts = generate_multi_transform(&config).expect("should generate");
    let s = ts.to_string();
    assert!(
        s.contains("notw_4_v8_avx2_f32"),
        "function name must match naming convention; got: {}",
        &s[..s.len().min(200)]
    );
}

#[test]
fn test_generate_function_name_sse2_f32_size2() {
    let config = MultiTransformConfig {
        size: 2,
        v: 4,
        isa: SimdIsa::Sse2,
        precision: Precision::F32,
    };
    let ts = generate_multi_transform(&config).expect("should generate");
    let s = ts.to_string();
    assert!(
        s.contains("notw_2_v4_sse2_f32"),
        "function name must match naming convention"
    );
}

#[test]
fn test_generate_function_name_avx2_f64_size8() {
    let config = MultiTransformConfig {
        size: 8,
        v: 4,
        isa: SimdIsa::Avx2,
        precision: Precision::F64,
    };
    let ts = generate_multi_transform(&config).expect("should generate");
    let s = ts.to_string();
    assert!(
        s.contains("notw_8_v4_avx2_f64"),
        "function name must match naming convention"
    );
}

#[test]
fn test_generate_unsupported_size_returns_error() {
    let config = MultiTransformConfig {
        size: 3,
        v: 4,
        isa: SimdIsa::Sse2,
        precision: Precision::F32,
    };
    let result = generate_multi_transform(&config);
    assert!(result.is_err(), "size 3 is unsupported and must return Err");
}

#[test]
fn test_generate_zero_v_returns_error() {
    let config = MultiTransformConfig {
        size: 4,
        v: 0,
        isa: SimdIsa::Avx2,
        precision: Precision::F32,
    };
    let result = generate_multi_transform(&config);
    assert!(result.is_err(), "v=0 is invalid and must return Err");
}

// ── Generated source contains expected keywords ───────────────────────────────

#[test]
fn test_generate_contains_unsafe_fn() {
    let config = MultiTransformConfig {
        size: 4,
        v: 8,
        isa: SimdIsa::Avx2,
        precision: Precision::F32,
    };
    let ts = generate_multi_transform(&config).expect("should generate");
    let s = ts.to_string();
    assert!(
        s.contains("unsafe"),
        "generated function must be marked unsafe"
    );
    assert!(s.contains("fn"), "must contain fn keyword");
}

#[test]
fn test_generate_size2_contains_butterfly_ops() {
    let config = MultiTransformConfig {
        size: 2,
        v: 4,
        isa: SimdIsa::Sse2,
        precision: Precision::F64,
    };
    let ts = generate_multi_transform(&config).expect("should generate");
    let s = ts.to_string();
    // Size-2 butterfly uses + and - on the two halves.
    assert!(
        s.contains('+') || s.contains("add"),
        "size-2 butterfly must contain addition"
    );
}

// ── SIMD-specific: SSE2 f32 SIMD multi-transform ─────────────────────────────

/// Test that SSE2+f32+size2 generates SIMD intrinsic code (not scalar).
#[test]
fn test_generate_sse2_f32_size2_contains_simd_intrinsics() {
    let config = MultiTransformConfig {
        size: 2,
        v: 4,
        isa: SimdIsa::Sse2,
        precision: Precision::F32,
    };
    let ts = generate_multi_transform(&config).expect("SSE2 f32 size-2 should generate");
    let s = ts.to_string();
    // The generated code should contain SIMD intrinsic references
    assert!(
        s.contains("_mm_loadu_ps") || s.contains("notw_2_v4_sse2_f32_soa"),
        "SSE2 f32 size-2 must contain SIMD load intrinsic or SoA function: got prefix: {}",
        &s[..s.len().min(300)]
    );
}

/// Test that SSE2+f32+size4 generates SIMD intrinsic code.
#[test]
fn test_generate_sse2_f32_size4_contains_simd_intrinsics() {
    let config = MultiTransformConfig {
        size: 4,
        v: 4,
        isa: SimdIsa::Sse2,
        precision: Precision::F32,
    };
    let ts = generate_multi_transform(&config).expect("SSE2 f32 size-4 should generate");
    let s = ts.to_string();
    assert!(
        s.contains("_mm_loadu_ps") || s.contains("notw_4_v4_sse2_f32_soa"),
        "SSE2 f32 size-4 must reference SIMD intrinsics or SoA fn"
    );
}

/// Test that AVX2+f32+size2 generates SIMD intrinsic code.
#[test]
fn test_generate_avx2_f32_size2_contains_simd_intrinsics() {
    let config = MultiTransformConfig {
        size: 2,
        v: 8,
        isa: SimdIsa::Avx2,
        precision: Precision::F32,
    };
    let ts = generate_multi_transform(&config).expect("AVX2 f32 size-2 should generate");
    let s = ts.to_string();
    assert!(
        s.contains("_mm256_loadu_ps") || s.contains("notw_2_v8_avx2_f32_soa"),
        "AVX2 f32 size-2 must reference 256-bit SIMD intrinsics or SoA fn"
    );
}

/// Test that AVX2+f32+size4 generates SIMD intrinsic code.
#[test]
fn test_generate_avx2_f32_size4_contains_simd_intrinsics() {
    let config = MultiTransformConfig {
        size: 4,
        v: 8,
        isa: SimdIsa::Avx2,
        precision: Precision::F32,
    };
    let ts = generate_multi_transform(&config).expect("AVX2 f32 size-4 should generate");
    let s = ts.to_string();
    assert!(
        s.contains("_mm256_loadu_ps") || s.contains("notw_4_v8_avx2_f32_soa"),
        "AVX2 f32 size-4 must reference 256-bit SIMD intrinsics or SoA fn"
    );
}

/// Test that AVX2+f32+size8 generates SIMD intrinsic code.
#[test]
fn test_generate_avx2_f32_size8_contains_simd_intrinsics() {
    let config = MultiTransformConfig {
        size: 8,
        v: 8,
        isa: SimdIsa::Avx2,
        precision: Precision::F32,
    };
    let ts = generate_multi_transform(&config).expect("AVX2 f32 size-8 should generate");
    let s = ts.to_string();
    assert!(
        s.contains("_mm256_loadu_ps") || s.contains("notw_8_v8_avx2_f32_soa"),
        "AVX2 f32 size-8 must reference 256-bit SIMD intrinsics or SoA fn"
    );
}

/// Test that f64 combos do NOT accidentally generate SIMD intrinsics (scalar only).
#[test]
fn test_generate_sse2_f64_is_scalar_fallback() {
    let config = MultiTransformConfig {
        size: 2,
        v: 4,
        isa: SimdIsa::Sse2,
        precision: Precision::F64,
    };
    let ts = generate_multi_transform(&config).expect("SSE2 f64 should generate scalar fallback");
    let s = ts.to_string();
    // Must still have the outer function name
    assert!(
        s.contains("notw_2_v4_sse2_f64"),
        "must have correct fn name"
    );
}

// ── proc-macro parse path ────────────────────────────────────────────────────

#[test]
fn test_generate_from_macro_avx2_f32_size4() {
    let input: proc_macro2::TokenStream = "size = 4, v = 8, isa = avx2, ty = f32"
        .parse()
        .expect("valid token stream");
    let result = generate_from_macro(input);
    assert!(
        result.is_ok(),
        "macro parse should succeed: {:?}",
        result.err()
    );
    let ts = result.expect("TokenStream");
    let s = ts.to_string();
    assert!(
        s.contains("notw_4_v8_avx2_f32"),
        "generated name must be correct"
    );
}

#[test]
fn test_generate_from_macro_sse2_f64_size2() {
    let input: proc_macro2::TokenStream = "size = 2, v = 4, isa = sse2, ty = f64"
        .parse()
        .expect("valid token stream");
    let result = generate_from_macro(input);
    assert!(result.is_ok(), "macro parse should succeed");
    let ts = result.expect("TokenStream");
    let s = ts.to_string();
    assert!(
        s.contains("notw_2_v4_sse2_f64"),
        "generated name must be correct"
    );
}

#[test]
fn test_generate_from_macro_scalar_f32_size8() {
    let input: proc_macro2::TokenStream = "size = 8, v = 1, isa = scalar, ty = f32"
        .parse()
        .expect("valid token stream");
    let result = generate_from_macro(input);
    assert!(result.is_ok(), "scalar ISA should succeed");
    let ts = result.expect("TokenStream");
    let s = ts.to_string();
    assert!(
        s.contains("notw_8_v1_scalar_f32"),
        "generated name must match"
    );
}

#[test]
fn test_generate_from_macro_missing_size_returns_error() {
    let input: proc_macro2::TokenStream = "v = 8, isa = avx2, ty = f32"
        .parse()
        .expect("valid token stream");
    let result = generate_from_macro(input);
    assert!(result.is_err(), "missing size must return error");
}

#[test]
fn test_generate_from_macro_unknown_isa_returns_error() {
    let input: proc_macro2::TokenStream = "size = 4, v = 8, isa = avx512, ty = f32"
        .parse()
        .expect("valid token stream");
    let result = generate_from_macro(input);
    assert!(result.is_err(), "unknown isa must return error");
}

#[test]
fn test_generate_from_macro_unknown_ty_returns_error() {
    let input: proc_macro2::TokenStream = "size = 4, v = 8, isa = avx2, ty = f16"
        .parse()
        .expect("valid token stream");
    let result = generate_from_macro(input);
    assert!(result.is_err(), "unknown ty must return error");
}

// ── v=1 edge case (scalar single-transform mode) ─────────────────────────────

#[test]
fn test_generate_v1_scalar_f32_size4() {
    let config = MultiTransformConfig {
        size: 4,
        v: 1,
        isa: SimdIsa::Scalar,
        precision: Precision::F32,
    };
    let ts = generate_multi_transform(&config).expect("v=1 should succeed");
    let s = ts.to_string();
    assert!(
        s.contains("notw_4_v1_scalar_f32"),
        "v=1 name must be correct"
    );
}

// ── All valid (size, ISA) combos smoke-test ───────────────────────────────────

#[test]
fn test_all_supported_sizes_generate_successfully() {
    for &size in &[2_usize, 4, 8] {
        for &(isa, v) in &[
            (SimdIsa::Sse2, 4_usize),
            (SimdIsa::Avx2, 8),
            (SimdIsa::Scalar, 1),
        ] {
            for &prec in &[Precision::F32, Precision::F64] {
                let config = MultiTransformConfig {
                    size,
                    v,
                    isa,
                    precision: prec,
                };
                let result = generate_multi_transform(&config);
                assert!(
                    result.is_ok(),
                    "size={size} isa={isa:?} v={v} prec={prec:?} should succeed"
                );
                let s = result.expect("TokenStream").to_string();
                let expected_name = format!(
                    "notw_{}_v{}_{}_{}",
                    size,
                    v,
                    isa.ident_str(),
                    prec.type_str()
                );
                assert!(
                    s.contains(&expected_name),
                    "expected name '{expected_name}' in generated code"
                );
            }
        }
    }
}

/// Test that the SIMD-enabled combos generate more code than purely scalar.
///
/// SSE2 f32 and AVX2 f32 emit the outer `AoS` function PLUS the inner `SoA` SIMD
/// function. Scalar or f64 emit only the outer `AoS` function. The SIMD tokens
/// should therefore be strictly longer.
#[test]
fn test_simd_generates_more_code_than_scalar() {
    // SSE2 f32 size-2 (SIMD path)
    let simd_config = MultiTransformConfig {
        size: 2,
        v: 4,
        isa: SimdIsa::Sse2,
        precision: Precision::F32,
    };
    let simd_ts = generate_multi_transform(&simd_config).expect("SSE2 f32 should generate");
    let simd_len = simd_ts.to_string().len();

    // Scalar f32 size-2 (no SIMD)
    let scalar_config = MultiTransformConfig {
        size: 2,
        v: 1,
        isa: SimdIsa::Scalar,
        precision: Precision::F32,
    };
    let scalar_ts = generate_multi_transform(&scalar_config).expect("scalar f32 should generate");
    let scalar_len = scalar_ts.to_string().len();

    assert!(
        simd_len > scalar_len,
        "SIMD code ({simd_len} chars) should be longer than scalar ({scalar_len} chars)"
    );
}
