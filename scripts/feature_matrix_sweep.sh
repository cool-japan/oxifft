#!/usr/bin/env bash
# feature_matrix_sweep.sh — Compile-only sweep over the oxifft `[features]`
# matrix, on the host target and on the documented no_std embedded target.
#
# This is deliberately hand-rolled (`cargo check`, one invocation per row)
# rather than a `cargo-hack` powerset: with 17 features a true powerset is
# 2^17 = 131072 combinations, which is not a useful signal-to-noise tradeoff
# for a workspace whose feature interactions are already modelled by hand in
# Cargo.toml (`std`-gating comments on every feature). Instead this checks:
#
#   1. `--all-features` and bare `--no-default-features` (the two extremes).
#   2. Default features (what a plain `cargo build` gets).
#   3. Every feature individually, `--no-default-features --features <feat>`
#      (each feature's own `= [...]` dependency list is what should pull in
#      whatever else it needs, e.g. `sve = ["std"]`).
#   4. The no_std-compatible features (per their Cargo.toml doc comments:
#      "no_std + alloc capable") individually on the `thumbv7em-none-eabihf`
#      embedded target, plus a bare no_std check on that target.
#
# Usage: ./scripts/feature_matrix_sweep.sh
#
# Exit code 0 = every row that is expected to compile on this host did;
# exit code 1 = at least one unexpected failure. Rows that are *known* to
# require host resources this environment may not have (a CUDA toolkit for
# `cuda`/`gpu`, a nightly toolchain for `portable_simd`) are reported but do
# not affect the exit code — see the "environment-dependent" list below.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EMBEDDED_TARGET="thumbv7em-none-eabihf"

# Every feature currently in oxifft/Cargo.toml's [features] table. Keep in
# sync with that file; a feature added there without a row here is a gap in
# this sweep, not a gap in the crate.
ALL_FEATURES=(
    std threading avx512 portable_simd f128-support f16-support sparse
    pruned sve wasm streaming const-fft cuda metal gpu signal fftw-compat
    ndarray
)

# Features documented in Cargo.toml as "no_std + alloc capable" — the only
# ones meaningful to check on the no_std embedded target individually.
NOSTD_FEATURES=(avx512 const-fft fftw-compat)

# Features whose successful compilation depends on host resources this sweep
# cannot guarantee (a CUDA toolkit, a nightly compiler): failures here are
# reported but do not fail the sweep.
ENV_DEPENDENT_FEATURES=(cuda gpu portable_simd)

PASS=0
FAIL=0
FAIL_ROWS=()
ENV_FAIL_ROWS=()

# run_row LABEL CMD...
run_row() {
    local label="$1"
    shift
    printf '  %-58s' "$label"
    if output=$("$@" 2>&1); then
        echo "OK"
        PASS=$((PASS + 1))
    else
        local is_env_dependent=0
        for f in "${ENV_DEPENDENT_FEATURES[@]}"; do
            if [[ "$label" == *"$f"* ]]; then
                is_env_dependent=1
            fi
        done
        if [[ "$is_env_dependent" -eq 1 ]]; then
            echo "FAIL (environment-dependent, not counted)"
            ENV_FAIL_ROWS+=("$label")
        else
            echo "FAIL"
            FAIL=$((FAIL + 1))
            FAIL_ROWS+=("$label")
        fi
        {
            echo "----- $label -----"
            echo "$output" | tail -40
            echo
        } >>"$SWEEP_LOG"
    fi
}

SWEEP_LOG="$(mktemp -t oxifft_feature_sweep.XXXXXX.log)"
echo "oxifft feature-matrix sweep — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Full failure output logged to: $SWEEP_LOG"
echo

echo "== Host target ($(rustc -vV | sed -n 's/host: //p')) =="
run_row "--all-features" cargo check -p oxifft --all-features
run_row "--no-default-features (bare)" cargo check -p oxifft --no-default-features
run_row "(default features)" cargo check -p oxifft
for feat in "${ALL_FEATURES[@]}"; do
    run_row "--no-default-features --features $feat" \
        cargo check -p oxifft --no-default-features --features "$feat"
done

echo
echo "== Embedded target ($EMBEDDED_TARGET, no_std) =="
if rustup target list --installed 2>/dev/null | grep -qx "$EMBEDDED_TARGET"; then
    run_row "--no-default-features (bare)" \
        cargo check -p oxifft --no-default-features --target "$EMBEDDED_TARGET"
    for feat in "${NOSTD_FEATURES[@]}"; do
        run_row "--no-default-features --features $feat" \
            cargo check -p oxifft --no-default-features --features "$feat" --target "$EMBEDDED_TARGET"
    done
    echo "  (portable_simd skipped on $EMBEDDED_TARGET: needs a nightly toolchain with"
    echo "   this target installed; not assumed present — see ENV_DEPENDENT_FEATURES)"
else
    echo "  SKIPPED: target '$EMBEDDED_TARGET' is not installed (rustup target add $EMBEDDED_TARGET)"
fi

echo
echo "== Summary =="
echo "  Passed: $PASS"
echo "  Failed (unexpected): $FAIL"
if [[ ${#FAIL_ROWS[@]} -gt 0 ]]; then
    printf '    - %s\n' "${FAIL_ROWS[@]}"
fi
if [[ ${#ENV_FAIL_ROWS[@]} -gt 0 ]]; then
    echo "  Failed (environment-dependent, not counted):"
    printf '    - %s\n' "${ENV_FAIL_ROWS[@]}"
fi

if [[ "$FAIL" -gt 0 ]]; then
    echo
    echo "See $SWEEP_LOG for full failure output."
    exit 1
fi
exit 0
