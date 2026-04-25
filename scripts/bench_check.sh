#!/usr/bin/env bash
# bench_check.sh — Run criterion benchmarks and compare against a saved baseline.
#
# Usage: ./scripts/bench_check.sh [baseline-name]
#
# Exit code 0 = no regression > 5%, exit code 1 = regression detected.
#
# Workflow:
#   1. Run the cooley_tukey_scaling bench, save results as baseline 'pr'.
#   2. Compare the 'pr' baseline against the named baseline (default: 'main').
#
# To save a 'main' baseline before opening a PR:
#   cargo bench --bench cooley_tukey_scaling -- --save-baseline main
#
# Then run this script in CI or locally:
#   ./scripts/bench_check.sh main
set -euo pipefail

BASELINE="${1:-main}"
CRATE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CRATE_DIR"

echo "==> Saving baseline 'pr'..."
cargo bench \
    -p oxifft \
    --bench cooley_tukey_scaling \
    -- \
    --save-baseline pr \
    --sample-size 20 \
    --warm-up-time 2

echo ""
echo "==> Comparing 'pr' against baseline '$BASELINE'..."
cargo bench \
    -p oxifft \
    --bench cooley_tukey_scaling \
    -- \
    --baseline "$BASELINE" \
    --sample-size 20 \
    --warm-up-time 2

echo ""
echo "==> Done. HTML report: target/criterion/report/index.html"
