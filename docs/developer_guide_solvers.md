# Developer Guide: Adding New Solvers

This guide explains how to add a new DFT or RDFT solver to OxiFFT, following the
existing architecture.

## Overview

OxiFFT uses a layered architecture:

```
API call
  └─ Planner   (oxifft/src/kernel/planner.rs)
       └─ Problem  — describes "what to compute"
            └─ Plan  — describes "how to compute it" (execute strategy)
                  └─ Solver  — factory that builds a Plan
                        └─ Codelet  — innermost compute kernel
```

Relevant source paths:

| Component | Path |
|-----------|------|
| `Problem` trait | `oxifft/src/kernel/problem.rs` |
| `Plan` trait | `oxifft/src/kernel/plan.rs` |
| `Solver` trait | `oxifft/src/kernel/solver.rs` |
| `Planner` | `oxifft/src/kernel/planner.rs` |
| `DftProblem` | `oxifft/src/dft/problem.rs` |
| `DftPlan` | `oxifft/src/dft/plan.rs` |
| DFT algorithm dispatch | `oxifft/src/dft/plan.rs` → `DftPlan::solve()` |
| Existing DFT solvers | `oxifft/src/dft/solvers/` |

## Core Traits

### `Problem` (`oxifft/src/kernel/problem.rs`)

```rust
pub trait Problem: Hash + Debug + Clone + Send + Sync {
    fn kind(&self) -> ProblemKind;  // Dft, Rdft, or Reodft
    fn zero(&self);
    fn total_size(&self) -> usize;
    fn is_inplace(&self) -> bool;
}
```

`DftProblem<T>` holds raw `*mut Complex<T>` pointers, a `Tensor` for dimensions/strides,
and a `Sign` (Forward = -1, Backward = +1 exponent).

### `Plan` (`oxifft/src/kernel/plan.rs`)

```rust
pub trait Plan: Send + Sync {
    type Problem: Problem;
    fn solve(&self, problem: &Self::Problem);  // the execution hot-path
    fn awake(&mut self, mode: WakeMode);
    fn ops(&self) -> OpCount;
    fn wake_state(&self) -> WakeState;
    fn solver_name(&self) -> &'static str;
}
```

`DftPlan<T>` is the concrete plan for DFT. Its `solve()` method contains the entire
algorithm-selection dispatch chain.

### `Solver` (`oxifft/src/kernel/solver.rs`)

The `Solver` trait is a *factory* for plans. Its primary method is:

```rust
fn make_plan<T>(&self, problem: &Self::Problem, planner: &mut Planner<T>) -> Option<Self::Plan>
where T: Float;
```

This is separate from the per-solver inherent `execute()` methods (e.g.,
`CooleyTukeySolver::execute()`, `BluesteinSolver::execute()`), which are called from
`DftPlan::solve()`.

### Algorithm Dispatch

There is no single `Algorithm` enum. The algorithm selection happens inline in
`DftPlan::solve()` (`oxifft/src/dft/plan.rs`):

```rust
fn solve(&self, problem: &DftProblem<T>) {
    let n = problem.transform_size();
    if n <= 1 {
        NopSolver::new().execute(input, output);
    } else if CooleyTukeySolver::<T>::applicable(n) {   // power-of-2
        CooleyTukeySolver::new(CtVariant::Dit).execute(input, output, sign);
    } else if has_composite_codelet(n) {                 // e.g. 12, 15, 24 ...
        execute_composite_codelet(output, n, sign_int);
    } else if n <= 16 {
        DirectSolver::new().execute(input, output, sign);
    } else if GenericSolver::<T>::applicable(n) {        // prime-factor
        GenericSolver::new(n).execute(input, output, sign);
    } else {
        BluesteinSolver::new(n).execute(input, output, sign); // arbitrary n
    }
}
```

`CooleyTukeySolver` also has an internal `CtVariant` enum
(`oxifft/src/dft/solvers/ct.rs`) for radix-2 DIT/DIF/Radix4/Radix8/SplitRadix variants.

## Adding a New DFT Solver

### Step 1: Create the solver file

Create `oxifft/src/dft/solvers/my_solver.rs`:

```rust
use crate::kernel::{Complex, Float};
use super::super::problem::Sign;

pub struct MySolver<T: Float> {
    n: usize,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float> MySolver<T> {
    pub fn new(n: usize) -> Self {
        Self { n, _marker: core::marker::PhantomData }
    }

    /// Returns true when this solver handles size `n`.
    pub fn applicable(n: usize) -> bool {
        // e.g. n % 3 == 0 && n > 1
        todo!()
    }

    pub fn execute(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        sign: Sign,
    ) {
        // your algorithm here
        todo!()
    }
}
```

### Step 2: Register in `solvers/mod.rs`

Add to `oxifft/src/dft/solvers/mod.rs`:

```rust
mod my_solver;
pub use my_solver::MySolver;
```

### Step 3: Wire into `DftPlan::solve()`

Edit `oxifft/src/dft/plan.rs`. Import `MySolver` at the top of `solve()` and add an
`else if` branch in the dispatch chain **before** the `BluesteinSolver` fallback:

```rust
use super::solvers::MySolver;

// inside solve():
} else if MySolver::<T>::applicable(n) {
    MySolver::new(n).execute(input, output, sign);
} else {
    BluesteinSolver::new(n).execute(input, output, sign);
}
```

Choose placement based on the sizes your solver handles. Add it before the fallback
but after any more specific solver (e.g., Cooley-Tukey for power-of-2).

### Step 4: Add tests

Add a `#[cfg(test)]` module at the bottom of `my_solver.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dft::solvers::direct::dft_direct;
    use crate::kernel::Complex;

    fn approx_eq_c(a: Complex<f64>, b: Complex<f64>) -> bool {
        (a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10
    }

    #[test]
    fn test_my_solver_matches_direct() {
        let n = /* your applicable size */;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(i as f64, 0.0))
            .collect();
        let mut output_mine = vec![Complex::new(0.0, 0.0); n];
        let mut output_ref  = vec![Complex::new(0.0, 0.0); n];

        MySolver::new(n).execute(&input, &mut output_mine, Sign::Forward);
        dft_direct(&input, &mut output_ref);

        for (a, b) in output_mine.iter().zip(output_ref.iter()) {
            assert!(approx_eq_c(*a, *b), "mismatch: {a:?} vs {b:?}");
        }
    }
}
```

`dft_direct` (in `oxifft/src/dft/solvers/direct.rs`) is the O(n²) reference.

### Step 5: Benchmark

Add a benchmark in `oxifft-bench/benches/` following the existing Criterion patterns.
Compare against `CooleyTukeySolver` for overlapping sizes:

```
cargo bench -p oxifft-bench -- my_solver
```

## Adding a New RDFT Solver (R2C / C2R / R2R)

RDFT problems live in `oxifft/src/rdft/`. The structure mirrors DFT:

| Component | Path |
|-----------|------|
| `RdftProblem<T>` | `oxifft/src/rdft/problem.rs` |
| `RdftPlan<T>` | `oxifft/src/rdft/plan.rs` |
| `R2cSolver` | `oxifft/src/rdft/solvers/r2c.rs` |
| `C2rSolver` | `oxifft/src/rdft/solvers/c2r.rs` |
| `R2rSolver` (DCT/DST/DHT) | `oxifft/src/rdft/solvers/r2r.rs` |

`RdftProblem` holds two buffers: `real_buf: *mut T` and `complex_buf: *mut Complex<T>`.
`RdftKind` selects the flavour: `R2C`, `C2R`, `R2HC`, `HC2R`, or `R2R`.

The standard R2C approach is:
1. Pre-process: pack real input into a complex half-size buffer.
2. Execute a complex DFT of size n/2.
3. Post-process: unpack using the symmetry relation
   `X[n−k] = conj(X[k])` to recover the full n/2+1 spectrum.

See `R2cSolver` for the reference. A new solver follows the same pattern—
create the struct, implement `applicable()` and `execute()`, then add the branch
in `RdftPlan::solve()`.

## Testing Conventions

- Test files live in the same module as the solver (`#[cfg(test)] mod tests { … }`).
- Reference implementation: `dft_direct` from `oxifft/src/dft/solvers/direct.rs`.
- Tolerance: use `1e-10` for `f64`, `1e-5` for `f32`.
- Always verify both forward and backward (inverse) transforms.
- Verify round-trip: `ifft(fft(x)) / n ≈ x`.

## Common Pitfalls

### Normalization

OxiFFT follows the FFTW convention: transforms are **unnormalized**. The forward DFT
computes `X[k] = Σ x[n] W^{nk}` without a `1/N` factor. The inverse computes the
conjugate sum, also without normalization. If you compute `IFFT(FFT(x))` you get `n*x`.
Your solver must match this convention exactly — do not divide by `n` inside `execute()`.

### Direction Sign Convention

- `Sign::Forward` = exponent `−1` in the twiddle: `W = e^{-2πi/N}`
- `Sign::Backward` = exponent `+1` in the twiddle: `W = e^{+2πi/N}`

This is the FFTW-compatible definition. Flipping signs is the most common source of
test failures when porting from a different library.

### Twiddle Factor Caching

For repeated transforms, use the global twiddle cache in `oxifft/src/kernel/twiddle.rs`:

```rust
use crate::kernel::{get_twiddle_table_f64, TwiddleDirection};

let table = get_twiddle_table_f64(n, TwiddleDirection::Forward);
// table.w[k] = e^{-2πik/n}
```

Do **not** compute `sin`/`cos` inside the hot loop. The cache is guarded by
`RwLock` and populated lazily; subsequent calls for the same `(n, direction)` pair
return a reference to the cached table without recomputation.
