#!/usr/bin/env bash
# fftw_ratio_report.sh — Run FFTW parity gates bench and emit ratio report.
#
# Usage:
#   bash scripts/fftw_ratio_report.sh [criterion-args]
#
# Default criterion args: --save-baseline current
# Example:
#   bash scripts/fftw_ratio_report.sh "--save-baseline current --warm-up-time 3"
#
# Requirements:
#   - FFTW installed: brew install fftw  (macOS)
#                     apt install libfftw3-dev  (Linux)

set -euo pipefail

BENCH_ARGS="${1:---save-baseline current}"

echo "Running FFTW parity gates bench..."
cargo bench \
  --features fftw-compare \
  -p oxifft-bench \
  --bench fftw_parity_gates \
  -- ${BENCH_ARGS}

echo ""
echo "Computing ratios..."
cargo run \
  --features fftw-compare \
  -p oxifft-bench \
  --bin fftw_ratio_report

echo ""
echo "Done. Ratios written to benches/baselines/v0.3.0/"
