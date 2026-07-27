# OxiFFT on `no_std`

> **Status (as of 0.4.1): the `--no-default-features` (`no_std`) core build
> compiles clean.** The `gen_simd`/`std`-only issues found by the 2026-07-22
> audit were fixed during the 0.4.0 sprint (see the "nostd-features" entry in
> [TODO.md](../TODO.md)); verified again for 0.4.1 below. A full feature-matrix
> sweep (every non-default feature individually under `no_std`) has not been
> exhaustively verified — that remains a tracked follow-up.

OxiFFT is designed to support `no_std` builds for embedded and WebAssembly
targets: the core FFT algorithms and codegen codelets require only the `alloc`
crate. Note that, as of 0.4.1, most **feature extensions are gated on `std`**
(see the matrix below) — only the core, `avx512`, `portable_simd`, `const-fft`
and `fftw-compat` features are `no_std`-shaped.

## Quick start (target state)

```toml
[dependencies]
oxifft = { version = "0.4", default-features = false, features = ["const-fft"] }
```

On embedded targets that require a global allocator:

```toml
embedded-alloc = "0.5"
```

## Feature matrix

Gating below matches `oxifft/Cargo.toml`. There is **no** `simd` feature (SIMD
is always-on via runtime detection) and **no** `mpi` feature (MPI lives in the
separate `oxifft-adapter-mpi` crate).

| Feature | Default? | `no_std` compatible? | Notes |
|---------|----------|---------------------|-------|
| Core FFT (no flag) | yes | yes (+`alloc`) | DFT, RDFT, DCT/DST, solvers |
| `std` | yes | — | File I/O wisdom, timing, `serde_json` |
| `threading` | yes | no (⇒`std`) | Rayon thread pool |
| `avx512` | no | yes (+`alloc`) | AVX-512 SIMD tier (x86_64) |
| `portable_simd` | no | yes (+`alloc`) | Nightly `#![feature(portable_simd)]` |
| `const-fft` | no | yes | Compile-time FFT via const generics |
| `fftw-compat` | no | yes (+`alloc`) | Thin FFTW-style (`fftw_*`) wrappers |
| `sparse` | no | no (⇒`std`) | FFAST O(k log n) sparse FFT |
| `pruned` | no | no (⇒`std`) | Pruned FFT / Goertzel |
| `streaming` | no | no (⇒`std`) | STFT, SlidingDFT, window functions |
| `signal` | no | no (⇒`std`) | Hilbert, Welch PSD, cepstrum |
| `f16-support` | no | no (⇒`std`) | Half-precision (16-bit) floats |
| `f128-support` | no | no (⇒`std`) | Quad-precision (128-bit) floats |
| `sve` | no | no (⇒`std`) | ARM SVE (uses `std::arch` detection) |
| `wasm` | no | no (⇒`std`) | WebAssembly bindings via `wasm-bindgen` |
| `ndarray` | no | no (⇒`std`) | ndarray integration |
| `cuda` / `metal` / `gpu` | no | no (⇒`std`) | GPU backends (OS driver APIs) |

`frft` and `nufft` are always-compiled modules but are additionally gated on
`std` at the module level, so they are unavailable in `no_std` builds.

### Notes on always-compiled modules

`conv`, `autodiff`, `ntt`, and `chirp_z` are compiled unconditionally (no feature
gate). They depend on `alloc` for their internal buffers but are `no_std+alloc`
shaped.

## Atomic caveats

`AtomicU64` is used in the wisdom cache and twiddle-factor generation, gated by
`cfg(target_has_atomic = "64")`. On 32-bit targets lacking 64-bit atomics, OxiFFT falls
back to `spin::Mutex`-protected state via the internal prelude.

## Target support

Intended `no_std` targets:

- `thumbv7em-none-eabihf` — ARMv7E-M embedded, no OS (with `embedded-alloc`)
- `wasm32-unknown-unknown` — WebAssembly, no OS (alloc via JS heap)

Verification commands (all pass as of 0.4.1; only `no_std`-shaped features
may be enabled here):

```bash
cargo check -p oxifft --no-default-features --target thumbv7em-none-eabihf
cargo check -p oxifft --no-default-features --features "const-fft,fftw-compat"
```

## Canonical `no_std` example

```rust
#![no_std]
extern crate alloc;

use alloc::vec;
use oxifft::{Complex, Direction, Flags, Plan};

fn run_fft() -> Option<()> {
    let plan = Plan::<f64>::dft_1d(16, Direction::Forward, Flags::ESTIMATE)?;
    let input = vec![Complex::new(1.0_f64, 0.0); 16];
    let mut output = vec![Complex::new(0.0_f64, 0.0); 16];
    plan.execute(&input, &mut output);
    Some(())
}
```

`Plan::dft_1d` returns `Option<Plan>` — `None` is returned for unsupported sizes. In `no_std`
builds there is no panic-on-failure; handle the `None` case explicitly.

## Known limitations

- **File I/O wisdom**: `export_to_file` / `import_from_file` / `merge_from_file` require
  `std`. In `no_std` builds, use `WisdomCache::export_string` / `import_string` /
  `merge_string` for manual wisdom management via in-memory strings.
- **Threading**: `ParallelConfig` and work-stealing require `std` (rayon thread pool).
  Multi-dimensional transforms run single-threaded in `no_std` builds. The `threading`
  feature depends on `std` and is therefore unavailable.
- **`frft` / `nufft` / `signal`**: These modules depend on `std` mathematical functions
  (`std::f64::sin`, timing APIs) and are unavailable in `no_std` builds.
- **GPU backends**: All GPU backends (`metal`, `cuda`, `gpu`) require `std` and OS APIs.
- **MPI**: Distributed FFT is provided by the separate `oxifft-adapter-mpi` crate
  (not an `oxifft` feature); it links a system MPI library and is not available on
  embedded `no_std` targets.
