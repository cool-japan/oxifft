# oxifft (Main Library Crate)

The core OxiFFT library providing pure Rust FFT implementations.

## Overview

This is the main library crate of the OxiFFT project. It provides:

- Complex DFT (Discrete Fourier Transform)
- Real FFT (R2C/C2R transforms)
- DCT/DST (Discrete Cosine/Sine Transforms)
- Multi-dimensional transforms
- Batch processing
- SIMD optimization
- Threading support
- Wisdom system for plan caching

## Module Structure

```
src/
├── lib.rs              # Public API exports
├── api/                # User-facing API
│   ├── plan.rs         # Plan creation (fftw_plan_* equivalents)
│   ├── execute.rs      # Execution functions
│   ├── wisdom.rs       # Wisdom import/export
│   ├── memory.rs       # Aligned allocation
│   └── types.rs        # Direction, Flags, R2rKind
│
├── kernel/             # Core infrastructure
│   ├── float.rs        # Float trait (f32/f64)
│   ├── complex.rs      # Complex<T> type
│   ├── tensor.rs       # IoDim, Tensor
│   ├── problem.rs      # Problem trait
│   ├── plan.rs         # Plan trait
│   ├── solver.rs       # Solver trait
│   ├── planner.rs      # Main planner
│   ├── twiddle.rs      # Twiddle factor cache
│   ├── primes.rs       # Prime utilities
│   └── ...
│
├── dft/                # Complex DFT
│   ├── problem.rs      # DftProblem
│   ├── plan.rs         # DftPlan
│   ├── solvers/        # Algorithm implementations
│   │   ├── ct.rs       # Cooley-Tukey
│   │   ├── rader.rs    # Rader's algorithm
│   │   ├── bluestein.rs # Bluestein/Chirp-Z
│   │   └── ...
│   └── codelets/       # Optimized kernels
│
├── rdft/               # Real DFT
│   ├── problem.rs      # RdftProblem
│   ├── plan.rs         # RdftPlan
│   └── solvers/        # R2C, C2R, etc.
│
├── reodft/             # DCT/DST
│   ├── redft.rs        # DCT variants
│   ├── rodft.rs        # DST variants
│   └── dht.rs          # Discrete Hartley
│
├── simd/               # SIMD abstraction
│   ├── traits.rs       # SimdVector trait
│   ├── detect.rs       # Runtime detection
│   ├── scalar.rs       # Fallback
│   ├── sse2.rs         # x86_64 SSE2
│   ├── avx.rs          # x86_64 AVX
│   └── neon.rs         # ARM NEON
│
├── threading/          # Parallel execution
│   ├── spawn.rs        # ThreadPool trait
│   ├── serial.rs       # Single-threaded
│   └── rayon_impl.rs   # Rayon integration
│
└── support/            # Utilities
    ├── align.rs        # Memory alignment
    ├── copy.rs         # Optimized copies
    └── transpose.rs    # Matrix transpose
```

## FFTW API Compatibility

This crate aims for high API compatibility with FFTW. Key mappings:

| FFTW Function | OxiFFT Equivalent |
|---------------|-------------------|
| `fftw_plan_dft_1d` | `Plan::dft_1d()` |
| `fftw_plan_dft_2d` | `Plan2D::new()` |
| `fftw_plan_dft_3d` | `Plan3D::new()` |
| `fftw_plan_dft_r2c` | `RealPlan::r2c_1d()` |
| `fftw_plan_dft_c2r` | `RealPlan::c2r_1d()` |
| `fftw_plan_guru_dft` | `GuruPlan::dft()` |
| `fftw_execute` | `plan.execute()` |
| `fftw_export_wisdom_to_string` | `wisdom::export_to_string()` |
| `FFTW_MEASURE` | `Flags::MEASURE` |
| `FFTW_FORWARD` | `Direction::Forward` |

## Usage

```rust
use oxifft::{Complex, Direction, Flags, Plan, Plan2D, RealPlan};

// Create a 256-point forward FFT plan
let plan = Plan::dft_1d(256, Direction::Forward, Flags::MEASURE).unwrap();

// Prepare input/output buffers
let input = vec![Complex::new(0.0, 0.0); 256];
let mut output = vec![Complex::zero(); 256];

// Execute
plan.execute(&input, &mut output);

// 2D FFT
let plan_2d = Plan2D::new(64, 64, Direction::Forward, Flags::ESTIMATE).unwrap();

// Real-to-Complex FFT
let plan_r2c = RealPlan::r2c_1d(256, Flags::MEASURE).unwrap();
let real_input = vec![0.0f64; 256];
let mut complex_output = vec![Complex::zero(); 129]; // n/2 + 1
plan_r2c.execute_r2c(&real_input, &mut complex_output);
```

## Features

- `std` (default): Enable standard library features
- `threading` (default): Enable Rayon-based parallelism
- `simd`: Enable explicit SIMD optimizations

## License

Same as the parent OxiFFT project.
