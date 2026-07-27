//! Real multi-process MPI integration checks for `oxifft-adapter-mpi`.
//!
//! This is an **example binary** (not a `#[test]`) on purpose: the Rust test
//! harness runs each test function on a spawned thread, but `mpi::initialize()`
//! selects `MPI_THREAD_SINGLE`, so multi-rank collectives issued from a
//! non-main thread deadlock under MPICH/OpenMPI. Running the checks from
//! `fn main()` keeps every MPI call on the process's main thread.
//!
//! It initialises MPI exactly once and runs every distributed-FFT scenario in
//! an SPMD fashion: every rank executes the same plans in the same order, so
//! the collective calls stay in lock-step.
//!
//! It is *size-adaptive* and driven by `scripts/run_mpi_tests.sh`:
//!
//! ```text
//! cargo build -p oxifft-adapter-mpi --example mpi_integration
//! mpirun -n 1 target/debug/examples/mpi_integration
//! mpirun -n 2 target/debug/examples/mpi_integration
//! mpirun -n 4 target/debug/examples/mpi_integration
//! ```
//!
//! Each scenario compares the distributed result against a serial reference
//! computed independently on every rank with the public `oxifft` API
//! (`Plan::dft_1d/2d/3d`), so no gather is required. On any mismatch the
//! offending rank panics, which makes the whole `mpirun` invocation exit
//! non-zero.

// Intentional in FFT test code (mirrors the crate's own lint policy).
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::suboptimal_flops)]

use oxifft::api::{Direction, Flags, Plan};
use oxifft::kernel::Complex;

use oxifft_adapter_mpi::{
    local_size_2d, local_size_2d_r2c, local_size_3d, local_size_3d_r2c, local_size_nd,
    Distribution, LocalPartition, MpiFlags, MpiPlan2D, MpiPlan3D, MpiPlanND, MpiPool,
    MpiRealPlan2D, MpiRealPlan3D, PencilGrid, PencilPlan3D,
};

const TOL: f64 = 1e-8;

/// Deterministic, rank-independent sample value for a given global flat index.
fn sample(idx: usize) -> Complex<f64> {
    let t = idx as f64;
    Complex {
        re: (0.1 * t).sin() + 0.3,
        im: (0.07 * t + 1.0).cos() - 0.2,
    }
}

fn max_abs_err(a: Complex<f64>, b: Complex<f64>) -> f64 {
    (a.re - b.re).hypot(a.im - b.im)
}

/// Build the full global input array `[n]` (row-major flat).
fn full_input(n: usize) -> Vec<Complex<f64>> {
    (0..n).map(sample).collect()
}

// ---------------------------------------------------------------------------
// Serial references (computed on every rank)
//
// The references are built from *1D* transforms applied along each axis in
// turn. This deliberately avoids `Plan::dft_2d`/`dft_3d`, whose parallel
// implementation lazily spins up a rayon thread-pool: leaving those worker
// threads alive has been observed to race with MPICH's `MPI_Finalize` during
// teardown. A full separable N-D DFT is independent of the order of the 1D
// passes, so this yields the same result as `dft_2d`/`dft_3d`.
// ---------------------------------------------------------------------------

/// Apply a forward 1D DFT along `axis` of a row-major array with shape `dims`.
fn fft_axis(data: &mut [Complex<f64>], dims: &[usize], axis: usize) {
    let n = dims[axis];
    let plan = Plan::<f64>::dft_1d(n, Direction::Forward, Flags::ESTIMATE).expect("dft_1d");
    let inner: usize = dims[axis + 1..].iter().product();
    let outer: usize = dims[..axis].iter().product();
    let mut buf_in = vec![Complex::<f64>::zero(); n];
    let mut buf_out = vec![Complex::<f64>::zero(); n];
    for o in 0..outer {
        for i in 0..inner {
            let base = o * n * inner + i;
            for k in 0..n {
                buf_in[k] = data[base + k * inner];
            }
            plan.execute(&buf_in, &mut buf_out);
            for k in 0..n {
                data[base + k * inner] = buf_out[k];
            }
        }
    }
}

/// Full separable N-D forward DFT of `full_input`, via per-axis 1D passes.
fn reference_nd(dims: &[usize]) -> Vec<Complex<f64>> {
    let total: usize = dims.iter().product();
    let mut data = full_input(total);
    for axis in 0..dims.len() {
        fft_axis(&mut data, dims, axis);
    }
    data
}

fn reference_1d(n: usize) -> Vec<Complex<f64>> {
    reference_nd(&[n])
}

fn reference_2d(n0: usize, n1: usize) -> Vec<Complex<f64>> {
    reference_nd(&[n0, n1])
}

fn reference_3d(n0: usize, n1: usize, n2: usize) -> Vec<Complex<f64>> {
    reference_nd(&[n0, n1, n2])
}

/// Deterministic, rank-independent **real** sample for a given global flat index.
fn rsample(idx: usize) -> f64 {
    let t = idx as f64;
    (0.1 * t).sin() + 0.3 * (0.05 * t).cos() + 0.2
}

/// Full complex forward DFT of the *real* input array (imaginary part 0) with
/// shape `dims`. The first `n_last/2 + 1` coefficients along the last axis of
/// this full spectrum are exactly the FFTW r2c half-complex output, so it doubles
/// as the r2c reference.
fn reference_real_nd(dims: &[usize]) -> Vec<Complex<f64>> {
    let total: usize = dims.iter().product();
    let mut data: Vec<Complex<f64>> = (0..total)
        .map(|i| Complex {
            re: rsample(i),
            im: 0.0,
        })
        .collect();
    for axis in 0..dims.len() {
        fft_axis(&mut data, dims, axis);
    }
    data
}

// ---------------------------------------------------------------------------
// Scenario checkers -- each returns Err(description) on mismatch.
// ---------------------------------------------------------------------------

fn check_slab_2d<C: mpi::topology::Communicator>(
    pool: &MpiPool<C>,
    n0: usize,
    n1: usize,
    transposed_out: bool,
) -> Result<(), String> {
    let size = pool.size();
    let rank = pool.rank();
    let (local_n0, local_0_start, alloc) = local_size_2d(n0, n1, pool);
    let reference = reference_2d(n0, n1);

    // Local input slab: rows [local_0_start, +local_n0), all n1 columns.
    let mut data = vec![Complex::<f64>::zero(); alloc.max(local_n0 * n1)];
    for i0 in 0..local_n0 {
        for j in 0..n1 {
            data[i0 * n1 + j] = sample((local_0_start + i0) * n1 + j);
        }
    }

    let flags = if transposed_out {
        MpiFlags::estimate().transposed_out()
    } else {
        MpiFlags::estimate()
    };
    let mut plan = MpiPlan2D::new(n0, n1, Direction::Forward, flags, pool)
        .map_err(|e| format!("[rank {rank}] MpiPlan2D::new failed: {e}"))?;
    plan.execute_inplace(&mut data)
        .map_err(|e| format!("[rank {rank}] execute_inplace failed: {e}"))?;

    if transposed_out {
        // Output layout: data[local_col * n0 + row], columns partitioned over `size`.
        let tpart = LocalPartition::new(n1, size, rank);
        for lc in 0..tpart.local_n {
            let global_col = tpart.local_start + lc;
            for row in 0..n0 {
                let got = data[lc * n0 + row];
                let exp = reference[row * n1 + global_col];
                let e = max_abs_err(got, exp);
                if e > TOL {
                    return Err(format!(
                        "[rank {rank}] slab2d T/O {n0}x{n1}: col {global_col} row {row} err {e:.2e}"
                    ));
                }
            }
        }
    } else {
        for i0 in 0..local_n0 {
            for j in 0..n1 {
                let got = data[i0 * n1 + j];
                let exp = reference[(local_0_start + i0) * n1 + j];
                let e = max_abs_err(got, exp);
                if e > TOL {
                    return Err(format!(
                        "[rank {rank}] slab2d {n0}x{n1}: row {} col {j} err {e:.2e}",
                        local_0_start + i0
                    ));
                }
            }
        }
    }
    Ok(())
}

fn check_slab_3d<C: mpi::topology::Communicator>(
    pool: &MpiPool<C>,
    n0: usize,
    n1: usize,
    n2: usize,
    transposed_out: bool,
) -> Result<(), String> {
    let size = pool.size();
    let rank = pool.rank();
    let (local_n0, local_0_start, alloc) = local_size_3d(n0, n1, n2, pool);
    let reference = reference_3d(n0, n1, n2);

    let mut data = vec![Complex::<f64>::zero(); alloc.max(local_n0 * n1 * n2)];
    for i0 in 0..local_n0 {
        for i1 in 0..n1 {
            for i2 in 0..n2 {
                data[i0 * n1 * n2 + i1 * n2 + i2] =
                    sample((local_0_start + i0) * n1 * n2 + i1 * n2 + i2);
            }
        }
    }

    let flags = if transposed_out {
        MpiFlags::estimate().transposed_out()
    } else {
        MpiFlags::estimate()
    };
    let mut plan = MpiPlan3D::new(n0, n1, n2, Direction::Forward, flags, pool)
        .map_err(|e| format!("[rank {rank}] MpiPlan3D::new failed: {e}"))?;
    plan.execute_inplace(&mut data)
        .map_err(|e| format!("[rank {rank}] execute_inplace failed: {e}"))?;

    if transposed_out {
        // Output layout: data[i1_local * n0 * n2 + i0 * n2 + i2], n1 partitioned over `size`.
        let tpart = LocalPartition::new(n1, size, rank);
        for i1l in 0..tpart.local_n {
            let g1 = tpart.local_start + i1l;
            for i0 in 0..n0 {
                for i2 in 0..n2 {
                    let got = data[i1l * n0 * n2 + i0 * n2 + i2];
                    let exp = reference[i0 * n1 * n2 + g1 * n2 + i2];
                    let e = max_abs_err(got, exp);
                    if e > TOL {
                        return Err(format!(
                            "[rank {rank}] slab3d T/O {n0}x{n1}x{n2}: ({i0},{g1},{i2}) err {e:.2e}"
                        ));
                    }
                }
            }
        }
    } else {
        for i0 in 0..local_n0 {
            let g0 = local_0_start + i0;
            for i1 in 0..n1 {
                for i2 in 0..n2 {
                    let got = data[i0 * n1 * n2 + i1 * n2 + i2];
                    let exp = reference[g0 * n1 * n2 + i1 * n2 + i2];
                    let e = max_abs_err(got, exp);
                    if e > TOL {
                        return Err(format!(
                            "[rank {rank}] slab3d {n0}x{n1}x{n2}: ({g0},{i1},{i2}) err {e:.2e}"
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn check_nd<C: mpi::topology::Communicator>(
    pool: &MpiPool<C>,
    dims: &[usize],
) -> Result<(), String> {
    let rank = pool.rank();
    let (local_n0, local_0_start, _alloc) = local_size_nd(dims, pool);
    let stride: usize = dims[1..].iter().product();

    // Reference depends on dimensionality.
    let reference = match dims.len() {
        1 => reference_1d(dims[0]),
        2 => reference_2d(dims[0], dims[1]),
        3 => reference_3d(dims[0], dims[1], dims[2]),
        _ => {
            return Err(format!(
                "[rank {rank}] check_nd: unsupported ndim {}",
                dims.len()
            ))
        }
    };

    let mut data = vec![Complex::<f64>::zero(); (local_n0 * stride).max(1)];
    for i0 in 0..local_n0 {
        for r in 0..stride {
            data[i0 * stride + r] = sample((local_0_start + i0) * stride + r);
        }
    }

    let mut plan = MpiPlanND::new(dims, Direction::Forward, MpiFlags::estimate(), pool)
        .map_err(|e| format!("[rank {rank}] MpiPlanND::new failed: {e}"))?;
    plan.execute_inplace(&mut data)
        .map_err(|e| format!("[rank {rank}] execute_inplace failed: {e}"))?;

    for i0 in 0..local_n0 {
        let g0 = local_0_start + i0;
        for r in 0..stride {
            let got = data[i0 * stride + r];
            let exp = reference[g0 * stride + r];
            let e = max_abs_err(got, exp);
            if e > TOL {
                return Err(format!(
                    "[rank {rank}] nd {dims:?}: (row {g0}, off {r}) err {e:.2e}"
                ));
            }
        }
    }
    Ok(())
}

/// Distributed 2D real transform: forward r2c matched against the serial full
/// complex FFT (its first `n1/2 + 1` last-axis bins), plus (for the natural
/// layout) an r2c -> c2r round trip that must recover the input identically.
fn check_real_2d<C: mpi::topology::Communicator>(
    pool: &MpiPool<C>,
    n0: usize,
    n1: usize,
    transposed_out: bool,
) -> Result<(), String> {
    let size = pool.size();
    let rank = pool.rank();
    let (local_n0, local_0_start, real_alloc, complex_alloc) = local_size_2d_r2c(n0, n1, pool);
    let n1c = n1 / 2 + 1;
    let reference = reference_real_nd(&[n0, n1]);

    // Local real input slab: rows [local_0_start, +local_n0), all n1 columns.
    let mut input = vec![0.0f64; real_alloc.max(1)];
    for i0 in 0..local_n0 {
        for j in 0..n1 {
            input[i0 * n1 + j] = rsample((local_0_start + i0) * n1 + j);
        }
    }

    let flags = if transposed_out {
        MpiFlags::estimate().transposed_out()
    } else {
        MpiFlags::estimate()
    };
    let mut r2c = MpiRealPlan2D::r2c(n0, n1, flags, pool)
        .map_err(|e| format!("[rank {rank}] MpiRealPlan2D::r2c failed: {e}"))?;
    let mut spec = vec![Complex::<f64>::zero(); complex_alloc.max(1)];
    r2c.execute_r2c(&input, &mut spec)
        .map_err(|e| format!("[rank {rank}] real2d execute_r2c failed: {e}"))?;

    if transposed_out {
        // Output layout: spec[lc * n0 + row], columns (n1c bins) partitioned.
        let tpart = LocalPartition::new(n1c, size, rank);
        for lc in 0..tpart.local_n {
            let gk = tpart.local_start + lc;
            for row in 0..n0 {
                let got = spec[lc * n0 + row];
                let exp = reference[row * n1 + gk];
                let e = max_abs_err(got, exp);
                if e > TOL {
                    return Err(format!(
                        "[rank {rank}] real2d r2c T/O {n0}x{n1}: bin {gk} row {row} err {e:.2e}"
                    ));
                }
            }
        }
        return Ok(());
    }

    // Natural layout: spec[i0 * n1c + k].
    for i0 in 0..local_n0 {
        let g0 = local_0_start + i0;
        for k in 0..n1c {
            let got = spec[i0 * n1c + k];
            let exp = reference[g0 * n1 + k];
            let e = max_abs_err(got, exp);
            if e > TOL {
                return Err(format!(
                    "[rank {rank}] real2d r2c {n0}x{n1}: row {g0} bin {k} err {e:.2e}"
                ));
            }
        }
    }

    // r2c -> c2r round trip must be the identity (normalized c2r).
    let mut c2r = MpiRealPlan2D::c2r(n0, n1, MpiFlags::estimate(), pool)
        .map_err(|e| format!("[rank {rank}] MpiRealPlan2D::c2r failed: {e}"))?;
    let mut recon = vec![0.0f64; real_alloc.max(1)];
    c2r.execute_c2r(&spec, &mut recon)
        .map_err(|e| format!("[rank {rank}] real2d execute_c2r failed: {e}"))?;
    for i0 in 0..local_n0 {
        let g0 = local_0_start + i0;
        for j in 0..n1 {
            let e = (recon[i0 * n1 + j] - input[i0 * n1 + j]).abs();
            if e > TOL {
                return Err(format!(
                    "[rank {rank}] real2d c2r roundtrip {n0}x{n1}: row {g0} col {j} err {e:.2e}"
                ));
            }
        }
    }
    Ok(())
}

/// Distributed 3D real transform: forward r2c matched against the serial full
/// complex FFT (its first `n2/2 + 1` last-axis bins), plus (for the natural
/// layout) an r2c -> c2r round trip that must recover the input identically.
fn check_real_3d<C: mpi::topology::Communicator>(
    pool: &MpiPool<C>,
    n0: usize,
    n1: usize,
    n2: usize,
    transposed_out: bool,
) -> Result<(), String> {
    let size = pool.size();
    let rank = pool.rank();
    let (local_n0, local_0_start, real_alloc, complex_alloc) = local_size_3d_r2c(n0, n1, n2, pool);
    let n2c = n2 / 2 + 1;
    let reference = reference_real_nd(&[n0, n1, n2]);

    let mut input = vec![0.0f64; real_alloc.max(1)];
    for i0 in 0..local_n0 {
        for i1 in 0..n1 {
            for i2 in 0..n2 {
                input[i0 * n1 * n2 + i1 * n2 + i2] =
                    rsample((local_0_start + i0) * n1 * n2 + i1 * n2 + i2);
            }
        }
    }

    let flags = if transposed_out {
        MpiFlags::estimate().transposed_out()
    } else {
        MpiFlags::estimate()
    };
    let mut r2c = MpiRealPlan3D::r2c(n0, n1, n2, flags, pool)
        .map_err(|e| format!("[rank {rank}] MpiRealPlan3D::r2c failed: {e}"))?;
    let mut spec = vec![Complex::<f64>::zero(); complex_alloc.max(1)];
    r2c.execute_r2c(&input, &mut spec)
        .map_err(|e| format!("[rank {rank}] real3d execute_r2c failed: {e}"))?;

    if transposed_out {
        // Output layout: spec[i1l * n0 * n2c + i0 * n2c + k], n1 partitioned.
        let tpart = LocalPartition::new(n1, size, rank);
        for i1l in 0..tpart.local_n {
            let g1 = tpart.local_start + i1l;
            for i0 in 0..n0 {
                for k in 0..n2c {
                    let got = spec[i1l * n0 * n2c + i0 * n2c + k];
                    let exp = reference[i0 * n1 * n2 + g1 * n2 + k];
                    let e = max_abs_err(got, exp);
                    if e > TOL {
                        return Err(format!(
                            "[rank {rank}] real3d r2c T/O {n0}x{n1}x{n2}: ({i0},{g1},{k}) err {e:.2e}"
                        ));
                    }
                }
            }
        }
        return Ok(());
    }

    // Natural layout: spec[i0 * n1 * n2c + i1 * n2c + k].
    for i0 in 0..local_n0 {
        let g0 = local_0_start + i0;
        for i1 in 0..n1 {
            for k in 0..n2c {
                let got = spec[i0 * n1 * n2c + i1 * n2c + k];
                let exp = reference[g0 * n1 * n2 + i1 * n2 + k];
                let e = max_abs_err(got, exp);
                if e > TOL {
                    return Err(format!(
                        "[rank {rank}] real3d r2c {n0}x{n1}x{n2}: ({g0},{i1},{k}) err {e:.2e}"
                    ));
                }
            }
        }
    }

    // r2c -> c2r round trip must be the identity (normalized c2r).
    let mut c2r = MpiRealPlan3D::c2r(n0, n1, n2, MpiFlags::estimate(), pool)
        .map_err(|e| format!("[rank {rank}] MpiRealPlan3D::c2r failed: {e}"))?;
    let mut recon = vec![0.0f64; real_alloc.max(1)];
    c2r.execute_c2r(&spec, &mut recon)
        .map_err(|e| format!("[rank {rank}] real3d execute_c2r failed: {e}"))?;
    for i0 in 0..local_n0 {
        let g0 = local_0_start + i0;
        for i1 in 0..n1 {
            for i2 in 0..n2 {
                let idx = i0 * n1 * n2 + i1 * n2 + i2;
                let e = (recon[idx] - input[idx]).abs();
                if e > TOL {
                    return Err(format!(
                        "[rank {rank}] real3d c2r roundtrip {n0}x{n1}x{n2}: ({g0},{i1},{i2}) err {e:.2e}"
                    ));
                }
            }
        }
    }
    Ok(())
}

fn check_pencil<C: mpi::topology::Communicator>(
    pool: &MpiPool<C>,
    grid: PencilGrid,
    n0: usize,
    n1: usize,
    n2: usize,
) -> Result<(), String> {
    let size = pool.size();
    let rank = pool.rank();
    if grid.total_procs() != size {
        return Err(format!(
            "[rank {rank}] pencil grid {}x{} != size {size}",
            grid.n_rows, grid.n_cols
        ));
    }

    let row_rank = grid.row_rank(rank);
    let col_rank = grid.col_rank(rank);
    let part0 = LocalPartition::new(n0, grid.n_rows, row_rank);
    let part1 = LocalPartition::new(n1, grid.n_cols, col_rank);
    let local_n0 = part0.local_n;
    let local_n1 = part1.local_n;

    let reference = reference_3d(n0, n1, n2);

    let mut plan = PencilPlan3D::new(n0, n1, n2, grid, Direction::Forward, Flags::ESTIMATE, pool)
        .map_err(|e| format!("[rank {rank}] PencilPlan3D::new failed: {e}"))?;
    if plan.distribution() != Distribution::Pencil {
        return Err(format!(
            "[rank {rank}] PencilPlan3D::distribution() != Pencil"
        ));
    }

    // The data buffer must be sized to `alloc_local` (the max over all layouts),
    // not merely the input footprint; the first `local_n0*local_n1*n2` elements
    // hold the input block `[local_n0][local_n1][n2]`.
    let mut data = vec![Complex::<f64>::zero(); plan.alloc_local().max(1)];
    for i0 in 0..local_n0 {
        let g0 = part0.local_start + i0;
        for i1 in 0..local_n1 {
            let g1 = part1.local_start + i1;
            for i2 in 0..n2 {
                data[i0 * local_n1 * n2 + i1 * n2 + i2] = sample(g0 * n1 * n2 + g1 * n2 + i2);
            }
        }
    }

    plan.execute_inplace(&mut data)
        .map_err(|e| format!("[rank {rank}] pencil execute_inplace failed: {e}"))?;

    if size == 1 {
        // Single-rank path keeps the natural [n0][n1][n2] layout.
        for (idx, &got) in data.iter().enumerate().take(n0 * n1 * n2) {
            let e = max_abs_err(got, reference[idx]);
            if e > TOL {
                return Err(format!(
                    "[rank {rank}] pencil P=1 {n0}x{n1}x{n2}: idx {idx} err {e:.2e}"
                ));
            }
        }
        return Ok(());
    }

    // Multi-rank output layout: data[i2c * local_n1_row * n0 + i1r * n0 + i0]
    //   g0 = i0
    //   g1 = part(n1, n_rows, row_rank).start + i1r
    //   g2 = part(n2, n_cols, col_rank).start + i2c
    let row_part_n1 = LocalPartition::new(n1, grid.n_rows, row_rank);
    let col_part_n2 = LocalPartition::new(n2, grid.n_cols, col_rank);
    let local_n1_row = row_part_n1.local_n;
    let local_n2_col = col_part_n2.local_n;

    for i2c in 0..local_n2_col {
        let g2 = col_part_n2.local_start + i2c;
        for i1r in 0..local_n1_row {
            let g1 = row_part_n1.local_start + i1r;
            for i0 in 0..n0 {
                let got = data[i2c * local_n1_row * n0 + i1r * n0 + i0];
                let exp = reference[i0 * n1 * n2 + g1 * n2 + g2];
                let e = max_abs_err(got, exp);
                if e > TOL {
                    return Err(format!(
                        "[rank {rank}] pencil {}x{} {n0}x{n1}x{n2}: g=({i0},{g1},{g2}) err {e:.2e}",
                        grid.n_rows, grid.n_cols
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Error-path and metadata regression checks. Every operation here is
/// collective-free (constructors that reject bad flags, and size checks that
/// return before any communication), so it is safe to run at any rank count.
fn check_error_paths<C: mpi::topology::Communicator>(pool: &MpiPool<C>) -> Result<(), String> {
    let rank = pool.rank();
    let ti = MpiFlags::estimate().transposed_in();

    // transposed_in must be rejected (not silently ignored) by every slab plan.
    if MpiPlan2D::<f64, _>::new(8, 4, Direction::Forward, ti, pool).is_ok() {
        return Err(format!("[rank {rank}] MpiPlan2D accepted transposed_in"));
    }
    if MpiPlan3D::<f64, _>::new(4, 4, 4, Direction::Forward, ti, pool).is_ok() {
        return Err(format!("[rank {rank}] MpiPlan3D accepted transposed_in"));
    }
    if MpiPlanND::<f64, _>::new(&[4, 4], Direction::Forward, ti, pool).is_ok() {
        return Err(format!("[rank {rank}] MpiPlanND accepted transposed_in"));
    }

    // Distribution metadata for slab plans.
    let mut p2 = MpiPlan2D::<f64, _>::new(8, 4, Direction::Forward, MpiFlags::estimate(), pool)
        .map_err(|e| format!("[rank {rank}] MpiPlan2D::new: {e}"))?;
    if p2.distribution() != Distribution::Slab {
        return Err(format!("[rank {rank}] MpiPlan2D::distribution() != Slab"));
    }

    // Undersized in-place buffer must return an error, not panic.
    let mut tiny = vec![Complex::<f64>::zero(); 1];
    if p2.execute_inplace(&mut tiny).is_ok() {
        return Err(format!(
            "[rank {rank}] execute_inplace accepted undersized buffer"
        ));
    }

    // Undersized out-of-place output must return an error, not panic.
    let input = vec![Complex::<f64>::zero(); 8 * 4];
    let mut tiny_out = vec![Complex::<f64>::zero(); 1];
    if p2.execute(&input, &mut tiny_out).is_ok() {
        return Err(format!("[rank {rank}] execute accepted undersized output"));
    }

    Ok(())
}

/// Pick a 2D process grid `(n_rows, n_cols)` whose product equals `size`.
fn pick_grid(size: usize) -> PencilGrid {
    match size {
        1 => PencilGrid::new(1, 1),
        2 => PencilGrid::new(1, 2),
        4 => PencilGrid::new(2, 2),
        6 => PencilGrid::new(2, 3),
        8 => PencilGrid::new(2, 4),
        // Fall back to a 1 x size row of pencils for any other count.
        n => PencilGrid::new(1, n),
    }
}

fn main() {
    let Some(universe) = mpi::initialize() else {
        // MPI already initialised elsewhere, or unavailable: nothing to do.
        eprintln!("mpi::initialize() returned None; skipping MPI integration run");
        return;
    };
    let world = universe.world();
    let pool = MpiPool::new(world);
    let size = pool.size();
    let rank = pool.rank();

    let mut failures: Vec<String> = Vec::new();
    let mut record = |r: Result<(), String>| {
        if let Err(e) = r {
            failures.push(e);
        }
    };

    // ---- Error paths and metadata (collective-free) ----
    record(check_error_paths(&pool));

    // ---- Slab 2D: divisible, non-divisible, size<ranks, transposed_out ----
    record(check_slab_2d(&pool, 8, 4, false));
    record(check_slab_2d(&pool, 8, 4, true));
    record(check_slab_2d(&pool, 10, 6, false)); // non-divisible for size 3/4
    record(check_slab_2d(&pool, 10, 6, true));
    record(check_slab_2d(&pool, 3, 2, false)); // size may exceed n0 (size<ranks case)
    record(check_slab_2d(&pool, 3, 5, true));

    // ---- Slab 3D: divisible, non-divisible, transposed_out ----
    record(check_slab_3d(&pool, 4, 4, 4, false));
    record(check_slab_3d(&pool, 4, 4, 4, true));
    record(check_slab_3d(&pool, 6, 5, 3, false)); // non-divisible
    record(check_slab_3d(&pool, 6, 5, 3, true));
    record(check_slab_3d(&pool, 2, 3, 3, false)); // size<ranks case for size 3/4

    // ---- N-D: 1D, 2D, 3D shaped ----
    record(check_nd(&pool, &[8]));
    record(check_nd(&pool, &[7])); // non-power-of-two 1D
    record(check_nd(&pool, &[4, 6]));
    record(check_nd(&pool, &[4, 4, 4]));
    record(check_nd(&pool, &[6, 5, 3])); // non-divisible

    // ---- Distributed real (r2c/c2r): forward-vs-reference + round trip ----
    // Even, transposed-out, odd last dim, and non-divisible n0.
    record(check_real_2d(&pool, 8, 8, false));
    record(check_real_2d(&pool, 8, 8, true)); // transposed_out
    record(check_real_2d(&pool, 8, 9, false)); // odd last dim
    record(check_real_2d(&pool, 6, 5, false)); // non-divisible n0 + odd last
    record(check_real_2d(&pool, 4, 6, true)); // transposed_out, non-divisible

    record(check_real_3d(&pool, 4, 4, 4, false));
    record(check_real_3d(&pool, 4, 4, 4, true)); // transposed_out
    record(check_real_3d(&pool, 4, 6, 5, false)); // odd last dim
    record(check_real_3d(&pool, 6, 5, 4, false)); // non-divisible n0
    record(check_real_3d(&pool, 3, 4, 6, true)); // transposed_out, non-divisible

    // ---- Pencil 3D: grid chosen from size, divisible + non-divisible ----
    let grid = pick_grid(size);
    record(check_pencil(&pool, grid, 8, 8, 8));
    record(check_pencil(&pool, grid, 6, 5, 4)); // non-divisible dims

    // All collectives are complete; synchronise so every rank agrees before
    // reporting.
    pool.barrier();

    if failures.is_empty() {
        eprintln!("[rank {rank}/{size}] MPI integration: all scenarios passed");
    } else {
        for f in &failures {
            eprintln!("FAILURE: {f}");
        }
        // Panicking makes this rank exit non-zero, so `mpirun` reports failure.
        panic!(
            "[rank {rank}/{size}] MPI integration FAILED with {} error(s)",
            failures.len()
        );
    }

    // Returning drops `universe`, which runs `MPI_Finalize`.
}
