# oxifft-bench

Benchmarks and FFTW comparison tests for [OxiFFT](https://github.com/cool-japan/oxifft).

This crate is internal to the OxiFFT project and is not published to crates.io.

## Usage

```bash
# Run benchmarks
cargo bench -p oxifft-bench

# Run FFTW comparison tests (requires libfftw3)
cargo test -p oxifft-bench --features fftw-compare
```

## License

Apache-2.0 — Copyright COOLJAPAN OU (Team Kitasan)
