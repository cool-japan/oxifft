# OxiFFT Testing Guide

## Running Tests

```bash
# Run all tests with all features
cargo nextest run --all-features

# Run only the core library tests
cargo nextest run -p oxifft --all-features

# Run a specific test
cargo nextest run --all-features -- test_name
```

## Test Categories

### Unit Tests

Each module contains inline unit tests (`#[cfg(test)]` blocks):

- **Correctness**: Compare against direct O(n²) solver for small sizes
- **Roundtrip**: `IFFT(FFT(x)) ≈ x` within floating-point tolerance
- **Parseval**: Energy conservation across transforms
- **Linearity**: `FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)`

### Integration Tests

Located in `tests/`:

- `size_coverage.rs` — correctness for powers of 2, primes, composites, edge cases
- `fftw_comparison.rs` — compare against FFTW3 (requires `fftw` feature)

### Feature-Gated Tests

Some tests are skipped unless the relevant feature is enabled:

| Feature | Tests |
|---------|-------|
| `sparse` | Sparse FFT correctness |
| `streaming` | STFT roundtrip, mel filterbank |
| `pruned` | Goertzel, input/output pruning |
| `const-fft` | Compile-time FFT correctness |
| `signal` | Hilbert transform, Welch PSD |

## Validation Strategy

1. **Direct solver validation**: For sizes ≤ 64, compare against O(n²) DFT
2. **Cross-validation**: Compare multiple algorithms for the same size
3. **Property tests**: Parseval, linearity, inverse, symmetry
4. **SIMD validation**: Each SIMD backend validated against scalar on the CI host

## Tolerance

Floating-point comparisons use these tolerances:

| Precision | Tolerance |
|-----------|-----------|
| f32 | 1e-5 |
| f64 | 1e-10 |
| f128 | 1e-15 |

## CI Pipeline

Tests run on:
- Linux x86_64 (AVX2)
- macOS aarch64 (NEON)
- Windows x86_64 (SSE2)
