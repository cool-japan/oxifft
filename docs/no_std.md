# OxiFFT on `no_std`

OxiFFT supports `no_std` builds for embedded and WebAssembly targets. The core FFT
algorithms, codegen codelets, and most feature extensions work without the Rust standard
library, requiring only the `alloc` crate.

## Quick start

```toml
[dependencies]
oxifft = { version = "0.3", default-features = false, features = ["sparse", "streaming"] }
```

On embedded targets that require a global allocator:

```toml
embedded-alloc = "0.5"
```

## Feature matrix

| Feature | Default? | `no_std` compatible? | `alloc` required? | Notes |
|---------|----------|---------------------|-------------------|-------|
| Core FFT (no flag) | yes | yes | yes | DFT, RDFT, DCT/DST, solvers |
| `std` | yes | — | — | File I/O wisdom, timing, `rayon` threading |
| `threading` | yes | no | — | Depends on `std` (rayon thread pool) |
| `simd` | no | yes | no | SIMD codelet dispatch (no alloc needed) |
| `portable_simd` | no | yes | no | Nightly `#![feature(portable_simd)]` |
| `sparse` | no | yes | yes | FFAST O(k log n) sparse FFT |
| `pruned` | no | yes | yes | Pruned FFT / Goertzel |
| `streaming` | no | yes | yes | STFT, SlidingDFT, window functions |
| `const-fft` | no | yes | no | Compile-time FFT via const generics |
| `conv` | no | yes | yes | Convolution helpers (always compiled) |
| `autodiff` | no | yes | yes | Automatic differentiation through FFT (always compiled) |
| `ntt` | no | yes | yes | Number-theoretic transform (always compiled) |
| `frft` | no | no | — | Fractional Fourier transform — requires `std` |
| `nufft` | no | no | — | Non-uniform FFT — requires `std` |
| `signal` | no | no | — | Hilbert, Welch PSD, cepstrum — requires `std` |
| `f128-support` | no | yes | yes | Quad-precision (128-bit) floats |
| `f16-support` | no | yes | yes | Half-precision (16-bit) floats |
| `wasm` | no | yes | yes | WebAssembly bindings via `wasm-bindgen` |
| `mpi` | no | no | — | Distributed FFT — requires MPI system library |
| `sve` | no | yes | no | ARM Scalable Vector Extension dispatch |
| `fftw-compat` | no | yes | yes | Thin FFTW-style API wrappers |
| `cuda` | no | no | — | NVIDIA GPU backend — requires `std` and CUDA |
| `metal` | no | no | — | Apple GPU backend — requires `std` and Metal |
| `gpu` | no | no | — | Enables both `cuda` and `metal` |

### Notes on always-compiled modules

`conv`, `autodiff`, and `ntt` are compiled unconditionally (no feature gate). They depend on
`alloc` for their internal buffers but work in `no_std+alloc` builds.

## Atomic caveats

`AtomicU64` is used in the wisdom cache and twiddle-factor generation, gated by
`cfg(target_has_atomic = "64")`. On 32-bit targets lacking 64-bit atomics, OxiFFT falls
back to `spin::Mutex`-protected state via the internal prelude.

## Verified targets

The following targets have been verified with `cargo check`:

- `thumbv7em-none-eabihf` — ARMv7E-M embedded, no OS (with `embedded-alloc`)
- `wasm32-unknown-unknown` — WebAssembly, no OS (alloc via JS heap)

Verify with:

```bash
cargo check -p oxifft --no-default-features --target thumbv7em-none-eabihf
cargo check -p oxifft --no-default-features --features "sparse,streaming"
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
- **`mpi`**: Requires a system MPI library; gated by both the `mpi` feature and the
  presence of an MPI installation. Not available in embedded `no_std` targets.
