# OxiFFT v0.x → v1.0 Migration Guide

## Overview

OxiFFT v1.0 is the first stable API release. After v1.0, all public types and
functions follow [Semantic Versioning](https://semver.org/): no breaking changes
within the v1.x series, and breaking changes only on a v2.0 bump.

The `fftw-compat` feature (available since v0.2.0) provides FFTW-style function
names (`fftw_plan_dft_1d`, `fftw_execute`, etc.) and can serve as a temporary
bridge when migrating an existing FFTW codebase.

---

## Breaking Changes from v0.x

### v0.2.0 — Return type changes on plan constructors

Four plan constructors changed their return type. The old type would have caused
a runtime panic on multi-dimensional paths; the new types make the distinction
compile-time.

| Method | v0.1.x return type | v0.2.0+ return type |
|---|---|---|
| `Plan::dft_2d(n0, n1, dir, flags)` | `Option<Plan<T>>` | `Option<Plan2D<T>>` |
| `Plan::dft_3d(n0, n1, n2, dir, flags)` | `Option<Plan<T>>` | `Option<Plan3D<T>>` |
| `Plan::r2c_1d(n, flags)` | `Option<Plan<T>>` | `Option<RealPlan<T>>` |
| `Plan::c2r_1d(n, flags)` | `Option<Plan<T>>` | `Option<RealPlan<T>>` |

**Before (v0.1.x):**
```rust
let plan: Option<Plan<f64>> = Plan::dft_2d(64, 64, Direction::Forward, Flags::ESTIMATE);
let real_plan: Option<Plan<f64>> = Plan::r2c_1d(256, Flags::ESTIMATE);
```

**After (v0.2.0+):**
```rust
let plan: Option<Plan2D<f64>> = Plan::dft_2d(64, 64, Direction::Forward, Flags::ESTIMATE);
let real_plan: Option<RealPlan<f64>> = Plan::r2c_1d(256, Flags::ESTIMATE);
```

Update any variable type annotations and any function signatures that stored or
returned the old concrete type.

### v0.2.0 — `IndirectStrategy` enum removed

`IndirectStrategy` and its `IndexArray` variant have been removed. They were
dead code that was never constructable. Remove any imports or match arms that
reference this type.

### v0.2.0 — All public enums are `#[non_exhaustive]`

Every public enum now carries `#[non_exhaustive]`. Exhaustive `match` blocks
on public enums become a compile error.

**Before:**
```rust
match direction {
    Direction::Forward => { /* … */ }
    Direction::Backward => { /* … */ }
}
```

**After — add a wildcard arm:**
```rust
match direction {
    Direction::Forward => { /* … */ }
    Direction::Backward => { /* … */ }
    _ => unreachable!(),
}
```

This applies to `Direction`, `Flags`, `GpuBackend`, `R2rKind`, and all other
public enums exported from `oxifft`.

### v0.3.0 — `GpuBackend::OpenCL` and `GpuBackend::Vulkan` variants removed

The two placeholder GPU backend variants were never backed by real code and have
been removed. Update any match arms or construction sites.

**Before:**
```rust
match backend {
    GpuBackend::Cuda => { /* … */ }
    GpuBackend::Metal => { /* … */ }
    GpuBackend::OpenCL => { /* … */ }   // compile error in v0.3.0+
    GpuBackend::Vulkan => { /* … */ }   // compile error in v0.3.0+
    _ => { /* … */ }
}
```

**After:**
```rust
match backend {
    GpuBackend::Cuda => { /* … */ }
    GpuBackend::Metal => { /* … */ }
    _ => { /* … */ }
}
```

### v0.3.0 — `libc` dependency removed; SVE detection changed

If your build script or feature flags explicitly enabled `libc` for SVE
detection via `getauxval`, that path no longer exists. SVE is now detected via
`std::arch::is_aarch64_feature_detected!("sve")`. No user action is required
unless you were directly calling the old `libc`-based helper.

---

## New Features in v1.0

- **`R2rPlan` solver caching** — twiddle tables and FFT sub-plans are built
  once at `R2rPlan` construction, not on every `execute()` call. DCT-II @ 1024
  is ~4× faster versus v0.2.0.
- **Multi-dimensional NUFFT** — `nufft/nufft2d.rs` and `nufft/nufft3d.rs`
  provide 2D and 3D non-uniform FFT.
- **Overlap-save STFT** — `streaming/stft.rs` adds an overlap-save method
  alongside the original overlap-add path.
- **Work-stealing parallel execution** — `WorkStealingContext` in
  `threading/work_stealing.rs` for Plan2D/Plan3D with user-pool override.
- **GPU batch FFT** — `GpuBatchFft<T>` trait with automatic chunking
  (`METAL_BATCH_LIMIT=1024`, `CUDA_BATCH_LIMIT=4096`).
- **AVX-512 hand-optimized codelets** for sizes 16, 32, 64.
- **Cache-oblivious 4-step FFT** (`dft/solvers/cache_oblivious.rs`).
- **WASM SIMD v128 intrinsics** via `core::arch::wasm32`.
- **FFTW parity gate benchmarks** for continuous regression tracking.

---

## Unchanged APIs

The following patterns are identical from v0.1.0 through v1.0 and need no
changes:

```rust
// 1D complex DFT — unchanged
let plan = Plan::<f64>::dft_1d(1024, Direction::Forward, Flags::ESTIMATE)
    .expect("plan failed");
plan.execute(&input, &mut output);

// Real-to-complex — constructor signature unchanged; return type changed in v0.2
let rplan = RealPlan::<f64>::r2c_1d(1024, Flags::ESTIMATE).expect("plan failed");
rplan.execute_r2c(&real_input, &mut complex_output);

// Complex-to-real — same
let cplan = RealPlan::<f64>::c2r_1d(1024, Flags::ESTIMATE).expect("plan failed");
cplan.execute_c2r(&complex_input, &mut real_output);

// R2R (DCT/DST) — unchanged
let dct = R2rPlan::<f64>::dct2(1024, Flags::ESTIMATE).expect("plan failed");
dct.execute(&input, &mut output);

// Convenience one-shot functions — unchanged
use oxifft::{fft, ifft, rfft, irfft};
let spectrum = fft(&signal);
let recovered = ifft(&spectrum);
```

---

## Migration Checklist

- [ ] Update `Cargo.toml` dependency to `oxifft = "1.0"`.
- [ ] Change `Option<Plan<T>>` return-type annotations for `dft_2d`/`dft_3d`
      to `Option<Plan2D<T>>` / `Option<Plan3D<T>>`.
- [ ] Change `Option<Plan<T>>` return-type annotations for `r2c_1d`/`c2r_1d`
      to `Option<RealPlan<T>>`.
- [ ] Add `_ => ...` wildcard arm to every `match` on a public OxiFFT enum.
- [ ] Remove any `GpuBackend::OpenCL` / `GpuBackend::Vulkan` match arms.
- [ ] Remove any `IndirectStrategy` imports or references.
- [ ] Remove explicit `libc` dependency added solely for OxiFFT SVE detection.
- [ ] Run `cargo semver-checks` to catch any remaining API surface mismatches.
- [ ] Optionally enable the `fftw-compat` feature for a smoother FFTW→OxiFFT
      migration via `oxifft::compat::fftw_plan_dft_1d` etc.
