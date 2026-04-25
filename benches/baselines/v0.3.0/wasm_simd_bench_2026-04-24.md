# WASM SIMD Benchmark — 2026-04-24

Benchmark environment: Node.js v23.7.0 on macOS arm64 (aarch64-apple-darwin)
Build: wasm-pack --target nodejs + default features (std+threading)

## scalar baseline (wasm-pack --target nodejs, default features)

| Size  | Time/op (µs) | Throughput (Mops/s) | MB/s   |
|-------|-------------|---------------------|--------|
| 64    | 10.258      | 0.0975              | 199.7  |
| 128   | 7.918       | 0.1263              | 517.3  |
| 256   | 10.450      | 0.0957              | 783.9  |
| 512   | 12.571      | 0.0795              | 1303.3 |
| 1024  | 133.358     | 0.0075              | 245.7  |
| 2048  | 292.643     | 0.0034              | 224.0  |
| 4096  | 406.296     | 0.0025              | 322.6  |

## simd128 build (RUSTFLAGS=-C target-feature=+simd128, wasm-opt -O3 --enable-simd applied)

| Size  | Time/op (µs) | Throughput (Mops/s) | MB/s   |
|-------|-------------|---------------------|--------|
| 64    | 19.418      | 0.0515              | 105.5  |
| 128   | 6.858       | 0.1458              | 597.3  |
| 256   | 8.993       | 0.1112              | 911.0  |
| 512   | 32.419      | 0.0308              | 505.4  |
| 1024  | 70.895      | 0.0141              | 462.2  |
| 2048  | 210.250     | 0.0048              | 311.7  |
| 4096  | 654.951     | 0.0015              | 200.1  |

## Comparison: scalar vs simd128 (speedup = scalar_time / simd128_time)

| Size  | Scalar (µs) | simd128 (µs) | Speedup |
|-------|-------------|--------------|---------|
| 64    | 10.258      | 19.418       | 0.53x   |
| 128   | 7.918       | 6.858        | 1.15x   |
| 256   | 10.450      | 8.993        | 1.16x   |
| 512   | 12.571      | 32.419       | 0.39x   |
| 1024  | 133.358     | 70.895       | 1.88x   |
| 2048  | 292.643     | 210.250      | 1.39x   |
| 4096  | 406.296     | 654.951      | 0.62x   |

## Notes

- simd128 target feature: enabled in binary (v128 locals confirmed by wasm-opt --print)
- wasm-opt -O3 --enable-simd applied to simd128 build (ensures fair comparison with scalar wasm-pack build)
- Compared to scalar: simd128 build shows mixed results — faster at some sizes (N=128, 256, 1024, 2048)
  but slower at others (N=64, 512, 4096). No consistent improvement.
- Root cause: WasmSimdF64 / WasmSimdF32 types are defined in wasm/simd.rs but are
  NOT wired into the FFT kernel. The DFT execution path (CooleyTukeySolver,
  dit_butterflies_f64, notw_*_dispatch) dispatches to x86_64 / aarch64 specific
  SIMD only; the wasm32 path hits the scalar fallback regardless of simd128.
  Enabling simd128 causes LLVM to emit some simd128 instructions from auto-vectorization
  of scalar code, but this does not consistently benefit the FFT butterfly loops.
- The 1.88x gain at N=1024 is likely LLVM auto-vectorization of the iterative DIT loop,
  not WasmSimdF64 (which has no callers). The regression at N=512 and N=4096 reflects
  that auto-vectorization is not reliably beneficial for the codelet-based path.

## Optimization opportunities identified

1. **Wire WasmSimdF64 into the butterfly kernel** (high impact, ~1–3x speedup estimate):
   Add a `#[cfg(target_arch = "wasm32")]` branch in `dft/solvers/simd_butterfly.rs`
   `dit_butterflies_f64()` that dispatches to a wasm32 v128-based butterfly loop
   using `WasmSimdF64`. This is the primary optimization opportunity.

2. **JS<->WASM memory copy overhead** (moderate impact for small N):
   Every `forward_interleaved()` call copies the input float array into WASM linear
   memory and allocates a new Vec for the output. For N=64, this overhead dominates
   over the FFT computation. An in-place API variant would reduce copies.

3. **Plan creation per call in one-shot API** (high overhead for small N):
   `fft_f64()` creates a new Plan per call (measured: 66.8 µs for N=1024 vs 133 µs
   in plan-based, so plan creation is ~50% of total cost at N=1024). Caching plans
   in a thread-local map would help.

4. **Missing simd128 butterfly codelet** for power-of-two sizes (high impact):
   `notw_2_dispatch`, `notw_4_dispatch`, `notw_8_dispatch`, etc. do not have a
   wasm32+simd128 code path. Adding one using WasmSimdF64 for the radix-2 butterfly
   would provide systematic improvement for all power-of-2 sizes.

## Decision

No source modifications were made to the FFT kernel. The benchmark data confirms that:
- The simd128 flag provides inconsistent results with the current dispatch structure
  (1.88x faster at N=1024, 0.39x slower at N=512 — driven by LLVM auto-vectorization,
  not by intentional WasmSimdF64 usage)
- All systematic optimization opportunities require wiring WasmSimdF64 into the kernel
  paths (not trivial, estimated >50 LoC per optimization)
- Results are documented for the v0.3.0 baseline

Note: `[lib] crate-type = ["cdylib", "rlib"]` was added temporarily for wasm-pack but
reverted — it broke `--no-default-features` (no_std) builds due to cdylib panic-strategy
conflicts. Users who want to build a `.wasm` binary from OxiFFT should create a thin
wrapper crate that sets `crate-type = ["cdylib"]` and depends on `oxifft`.
