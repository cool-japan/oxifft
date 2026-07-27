#!/usr/bin/env bash
#
# Build and run the oxifft-adapter-mpi distributed-FFT integration checks under
# real MPI (multiple processes).
#
# Usage:
#   scripts/run_mpi_tests.sh              # runs with -n 1 2 4
#   RANKS="1 2 3 4" scripts/run_mpi_tests.sh
#   MPIRUN=mpiexec scripts/run_mpi_tests.sh
#
# The checks live in `examples/mpi_integration.rs` (an example binary, so every
# MPI call runs on the process main thread -- a `#[test]` would deadlock because
# libtest runs test bodies on spawned threads while MPI is MPI_THREAD_SINGLE).
#
# Honours CARGO_TARGET_DIR if set. Requires an MPI launcher (mpirun / mpiexec)
# and an MPI library (OpenMPI or MPICH) on PATH.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE_DIR="$(dirname "$CRATE_DIR")"
cd "$WORKSPACE_DIR"

MPIRUN="${MPIRUN:-mpirun}"
RANKS="${RANKS:-1 2 4}"
PROFILE_DIR="debug"

if ! command -v "$MPIRUN" >/dev/null 2>&1; then
    echo "ERROR: '$MPIRUN' not found on PATH (set MPIRUN=... to override)" >&2
    exit 1
fi

# Resolve the target directory (honouring CARGO_TARGET_DIR).
TARGET_DIR="${CARGO_TARGET_DIR:-$WORKSPACE_DIR/target}"
BIN="$TARGET_DIR/$PROFILE_DIR/examples/mpi_integration"

echo "Building example 'mpi_integration'..."
cargo build -p oxifft-adapter-mpi --example mpi_integration

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: example binary not found at $BIN" >&2
    exit 1
fi
echo "Example binary: $BIN"

status=0
for n in $RANKS; do
    echo "=================================================================="
    echo "  $MPIRUN -n $n  (mpi_integration)"
    echo "=================================================================="
    if "$MPIRUN" -n "$n" "$BIN"; then
        echo "  -> PASS (n=$n)"
    else
        echo "  -> FAIL (n=$n)" >&2
        status=1
    fi
done

if [[ "$status" -eq 0 ]]; then
    echo "All MPI integration runs passed."
else
    echo "One or more MPI integration runs FAILED." >&2
fi
exit "$status"
