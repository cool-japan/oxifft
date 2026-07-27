# oxifft-adapter-mpi

Distributed (multi-process) FFT support for [OxiFFT](https://github.com/cool-japan/oxifft),
using MPI in the same style as FFTW-MPI: slab decomposition for 2D transforms,
2D-process-grid pencil decomposition for 3D transforms, general N-D
distributed complex transforms, and slab-decomposed distributed **real**
transforms (r2c / c2r).

## What this crate is

`oxifft-adapter-mpi` is a **quarantine crate**: it isolates the real, C-level
MPI FFI (the `mpi` crate → `mpi-sys` → `libffi-sys` / `clang-sys` dependency
chain) *out of* the core `oxifft` crate, so that `oxifft`'s own dependency
closure — even with `--all-features` — stays 100% Pure Rust and never links
against a system library. All distributed-FFT functionality that genuinely
needs a real, installed MPI implementation lives here instead.

Internally, each distributed plan (`MpiPlan2D`, `MpiPlan3D`, `MpiPlanND`,
`PencilPlan3D`) composes ordinary local `oxifft::api::Plan<T>` transforms with
an MPI all-to-all transpose, following the classic "four-step" FFT algorithm:

1. Local row-wise (or plane-wise) FFTs, computed with `oxifft` directly.
2. A distributed transpose (`MPI_Alltoall`/`MPI_Alltoallv`) that redistributes
   data across ranks.
3. Local column-wise (or row-wise, post-transpose) FFTs.
4. An optional distributed transpose back, unless `MpiFlags::transposed_out`
   is set (matching `FFTW_MPI_TRANSPOSED_OUT`).

## Features

- 2D and 3D slab decomposition (row-major distribution across ranks), plus
  general N-D distributed FFTs (`MpiPlanND`).
- Distributed **real** transforms in the FFTW half-complex layout
  (`MpiRealPlan2D`, `MpiRealPlan3D`): real-to-complex (r2c) and complex-to-real
  (c2r), with the last dimension `n` stored as `n/2 + 1` complex coefficients
  (`local_size_2d_r2c` / `local_size_3d_r2c` mirror `fftw_mpi_local_size_*` for
  the r2c case).
- 2D-process-grid **pencil** decomposition for 3D transforms (`PencilPlan3D`,
  `PencilGrid`), for better scaling than slab decomposition at high rank counts.
- Efficient all-to-all / all-to-all-v distributed transpose primitives
  (`distributed_transpose`, `distributed_transpose_batched`,
  `distributed_transpose_inplace`) usable directly, independent of the plan types.
- Data layouts and flag semantics compatible with FFTW-MPI
  (`local_size_2d`/`_3d`/`_nd` mirror `fftw_mpi_local_size_*`; `MpiFlags`
  mirrors `FFTW_MPI_TRANSPOSED_OUT`/`_IN`).
- Generic over `T: Float` (the same scalar types `oxifft` itself supports)
  wherever the underlying `mpi` crate has a matching wire type
  (see the `MpiFloat` trait).

## MPI requirements

This crate depends unconditionally on the [`mpi`](https://crates.io/crates/mpi)
crate ([rsmpi](https://github.com/rsmpi/rsmpi)), which requires, **at build
time**:

- A working MPI implementation with development headers/libraries — either
  [Open MPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/) —
  discoverable via `pkg-config` or the standard `mpicc`/`mpirun` wrapper
  scripts on `PATH`.
- `libclang`, because `mpi-sys` uses `bindgen` to generate FFI bindings
  against the local MPI installation's C headers.

Typical setup:

```bash
# macOS (Homebrew)
brew install open-mpi llvm   # llvm provides libclang

# Debian / Ubuntu
sudo apt-get install libopenmpi-dev libclang-dev

# Fedora / RHEL
sudo dnf install openmpi-devel clang-devel
```

Because of this, `oxifft-adapter-mpi` is **not** part of the workspace's
`default-members`: a plain `cargo build`/`cargo test` at the repository root
does not require MPI or `libclang` at all. Build this crate explicitly, or use
`cargo build --workspace`, once MPI is set up:

```bash
cargo build -p oxifft-adapter-mpi
```

See the [`mpi` crate's own documentation](https://docs.rs/mpi) for the full,
authoritative list of supported MPI implementations and platform notes.

## Quickstart

```rust,ignore
use oxifft::api::{Direction, Flags};
use oxifft::kernel::Complex;
use oxifft_adapter_mpi::{local_size_2d, MpiFlags, MpiPlan2D, MpiPool};

// Must be called exactly once, on the process's main thread, before any
// other MPI call (see "A note on threading" below).
let universe = mpi::initialize().expect("MPI_Init failed");
let world = universe.world();
let pool = MpiPool::new(world);

let (n0, n1) = (64, 32);

// How many rows this rank owns, where they start globally, and how many
// complex elements to allocate (mirrors `fftw_mpi_local_size_2d`).
let (local_n0, local_0_start, alloc_local) = local_size_2d(n0, n1, &pool);

let mut plan = MpiPlan2D::<f64, _>::new(
    n0,
    n1,
    Direction::Forward,
    MpiFlags::new().with_base(Flags::ESTIMATE),
    &pool,
)
.expect("failed to create distributed plan");

// Fill this rank's local rows (`local_n0` rows of `n1` columns each,
// starting at global row `local_0_start`) with input data, then:
let mut data = vec![Complex::new(0.0, 0.0); alloc_local];
plan.execute_inplace(&mut data).expect("distributed FFT failed");
```

Run with any MPI launcher, e.g.:

```bash
mpirun -n 4 ./your_binary
```

## Distributed real transforms (r2c / c2r)

`MpiRealPlan2D` and `MpiRealPlan3D` compute slab-decomposed real transforms in
the FFTW half-complex layout: the **last** dimension of size `n` produces
`n/2 + 1` complex coefficients (the rest are redundant by Hermitian symmetry),
while the first dimension is distributed across ranks exactly like the complex
slab plans. Odd last dimensions are supported.

```rust,ignore
use oxifft::api::Flags;
use oxifft::kernel::Complex;
use oxifft_adapter_mpi::{local_size_2d_r2c, MpiFlags, MpiRealPlan2D, MpiPool};

let universe = mpi::initialize().expect("MPI_Init failed");
let pool = MpiPool::new(universe.world());

let (n0, n1) = (64, 32); // n1 -> n1/2 + 1 = 17 complex columns
let (local_n0, local_0_start, real_alloc, complex_alloc) =
    local_size_2d_r2c(n0, n1, &pool);

// Local real input slab: `[local_n0][n1]`.
let mut input = vec![0.0_f64; real_alloc];
// ... fill `input` for this rank's rows (start at global row `local_0_start`) ...

let flags = MpiFlags::new().with_base(Flags::ESTIMATE);
let mut r2c = MpiRealPlan2D::r2c(n0, n1, flags, &pool).expect("r2c plan");
let mut spec = vec![Complex::new(0.0, 0.0); complex_alloc]; // `[local_n0][n1/2 + 1]`
r2c.execute_r2c(&input, &mut spec).expect("distributed r2c failed");

// Inverse: c2r is normalized (see below), so this recovers `input` exactly.
let mut c2r = MpiRealPlan2D::c2r(n0, n1, flags, &pool).expect("c2r plan");
let mut recovered = vec![0.0_f64; real_alloc];
c2r.execute_c2r(&spec, &mut recovered).expect("distributed c2r failed");
```

`MpiRealPlan3D` is analogous (`r2c(n0, n1, n2, ...)` / `c2r(...)`, last
dimension `n2 -> n2/2 + 1`). r2c also honours `MpiFlags::transposed_out`
(`FFTW_MPI_TRANSPOSED_OUT`), yielding the spectrum in transposed layout and
skipping the final transpose.

**Normalization.** Matching the OxiFFT 0.4.0 core convention (and unlike FFTW),
the adapter's `execute_c2r` is **normalized** by `1 / product(dims)`, so an
r2c → c2r round trip is the identity with no manual scaling. This is
intentionally consistent with the core crate's `RealPlan*::execute_c2r`.

## A note on threading

`mpi::initialize()` selects `MPI_THREAD_SINGLE`. All MPI calls — including
every method on `MpiPool` and every `MpiPlan*::execute*` call — must therefore
happen on the process's **main thread**. In particular, Rust's built-in test
harness (`#[test]`) runs test bodies on spawned threads, which deadlocks
multi-rank collectives under both OpenMPI and MPICH. For this reason, the
crate's own integration checks live in an example binary
(`examples/mpi_integration.rs`) run directly from `fn main`, driven by
`scripts/run_mpi_tests.sh`:

```bash
cargo build -p oxifft-adapter-mpi --example mpi_integration
mpirun -n 1 target/debug/examples/mpi_integration
mpirun -n 2 target/debug/examples/mpi_integration
mpirun -n 4 target/debug/examples/mpi_integration

# or, equivalently:
scripts/run_mpi_tests.sh                # runs with -n 1 2 4
RANKS="1 2 3 4" scripts/run_mpi_tests.sh
```

## License

Licensed under the Apache License, Version 2.0 — see
[`LICENSE`](../LICENSE) in the repository root.
