# Contributing to OxiFFT

Thank you for your interest in contributing to OxiFFT! This document provides guidelines and information for contributors.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Prioritize code quality and correctness
- Follow the project's coding standards

## Development Setup

### Prerequisites

- Rust 1.75 or later
- Cargo
- Git

### Optional Dependencies

- FFTW3 library (for comparison benchmarks)
- MPI implementation (for distributed computing features)
- CUDA toolkit (for GPU acceleration on NVIDIA)

### Building the Project

```bash
# Clone the repository
git clone https://github.com/oxifft/oxifft.git
cd oxifft

# Build with all features
cargo build --all-features

# Run tests
cargo test --all-features

# Run clippy
cargo clippy --all-features

# Build documentation
cargo doc --all-features --no-deps --open
```

## Project Policies

### 1. No Warnings Policy

- **Zero compiler warnings** - All code must compile without warnings
- **Zero clippy warnings** - All clippy lints must pass
- **Zero rustdoc warnings** - All documentation must be valid

### 2. No unwrap() Policy

- Production code must **never use unwrap()**
- Use `expect()` with descriptive error messages
- Test code may use `unwrap()` (this is acceptable in Rust)

### 3. Refactoring Policy

- **Maximum file size: 2000 lines**
- Split large files using the `splitrs` tool when necessary
- Keep modules focused and cohesive

### 4. Pure Rust Policy

- Default features must be **100% Pure Rust**
- No C/Fortran dependencies in default build
- Foreign dependencies must be feature-gated

### 5. Latest Crates Policy

- Always use the latest available versions on crates.io
- Update dependencies regularly
- Use workspace dependencies (*.workspace = true)

### 6. Workspace Policy

- Use workspace-level dependency management
- No version specifications in individual crate Cargo.toml files
- Keywords and categories can differ per crate

## Code Quality Standards

### Testing

All new features must include:

1. **Unit tests** - Test individual components
2. **Integration tests** - Test feature interactions
3. **Correctness tests** - Validate against known-good implementations
4. **Property-based tests** - Use proptest for mathematical properties

Example test structure:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Arrange
        let input = vec![Complex::new(1.0, 0.0); 8];

        // Act
        let result = my_function(&input);

        // Assert
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_roundtrip() {
        let input = vec![Complex::new(1.0, 0.0); 16];
        let forward = fft(&input);
        let backward = ifft(&forward);

        for (a, b) in input.iter().zip(backward.iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }
}
```

### Documentation

All public APIs must have:

1. **Summary** - One-line description
2. **Detailed description** - What it does and why
3. **Arguments** - All parameters documented
4. **Returns** - Return value documented
5. **Examples** - Working code examples (use `ignore` if not runnable in doctests)
6. **Panics** - Document panic conditions (if any)
7. **Errors** - Document error conditions (if returning Result)
8. **Safety** - Document safety invariants (for unsafe code)

Example documentation:

```rust
/// Compute the 1D complex FFT using the Cooley-Tukey algorithm.
///
/// This function performs a Decimation-in-Time FFT for power-of-two sizes.
/// For non-power-of-two sizes, consider using the generic planner.
///
/// # Arguments
///
/// * `input` - Input signal in time domain
/// * `output` - Output buffer for frequency domain (must be same length as input)
///
/// # Examples
///
/// ```ignore
/// use oxifft::{Complex, fft_radix2};
///
/// let input = vec![Complex::new(1.0, 0.0); 8];
/// let mut output = vec![Complex::zero(); 8];
/// fft_radix2(&input, &mut output);
/// ```
///
/// # Panics
///
/// Panics if input and output lengths differ, or if length is not a power of two.
pub fn fft_radix2<T: Float>(input: &[Complex<T>], output: &mut [Complex<T>]) {
    // Implementation
}
```

### Performance

- Use `#[inline]` for small hot-path functions
- Prefer `&[T]` over `&Vec<T>` for function parameters
- Use SIMD where beneficial (with fallback)
- Benchmark performance-critical changes using Criterion

### Error Handling

- Use `Result<T, Error>` for fallible operations
- Define custom error types using enums
- Implement `std::error::Error` and `Display`
- Provide context with error messages

Example error handling:

```rust
#[derive(Debug, Clone, Copy)]
pub enum FftError {
    InvalidSize(usize),
    InvalidStride,
    NullPointer,
}

impl std::fmt::Display for FftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSize(n) => write!(f, "Invalid FFT size: {}", n),
            Self::InvalidStride => write!(f, "Invalid stride for transform"),
            Self::NullPointer => write!(f, "Null pointer encountered"),
        }
    }
}

impl std::error::Error for FftError {}
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** with a descriptive name
   - Feature: `feature/add-xyz`
   - Bugfix: `fix/issue-123`
   - Documentation: `docs/improve-xyz`
3. **Make your changes**
   - Follow the coding standards
   - Add tests
   - Update documentation
4. **Run checks** locally:
   ```bash
   cargo test --all-features
   cargo clippy --all-features
   cargo doc --all-features --no-deps
   cargo fmt --check
   ```
5. **Commit** with clear messages:
   ```
   Add sparse FFT implementation

   - Implement FFAST algorithm
   - Add unit tests and benchmarks
   - Update documentation
   ```
6. **Submit PR** with:
   - Clear description of changes
   - Link to related issues
   - Test results
   - Performance impact (if applicable)

## Commit Guidelines

- **DO NOT commit** unless explicitly asked by the project maintainer
- **DO NOT push** to remote unless explicitly asked
- Use descriptive commit messages
- One logical change per commit
- End commits with attribution:
  ```
  🤖 Generated with Claude Code

  Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
  ```

## Feature Flags

When adding new features:

1. Gate behind a feature flag
2. Update Cargo.toml with the flag
3. Document the feature in README.md
4. Add to the feature table in documentation
5. Ensure default features remain Pure Rust

Example:

```toml
[features]
default = ["std", "threading"]
std = []
my-new-feature = ["dep:some-crate"]
```

## Benchmarking

When adding performance-critical features:

1. Add Criterion benchmarks
2. Compare against baseline
3. Document performance characteristics
4. Consider SIMD optimization
5. Test on multiple platforms

Example benchmark:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_fft(c: &mut Criterion) {
    let input = vec![Complex::new(1.0, 0.0); 1024];

    c.bench_function("fft_1024", |b| {
        b.iter(|| {
            fft(black_box(&input))
        });
    });
}

criterion_group!(benches, benchmark_fft);
criterion_main!(benches);
```

### Cooley-Tukey Scaling Benchmarks and Regression Check

The `cooley_tukey_scaling` bench covers forward transforms from 2^10 to 2^20
for both f64 and f32.  Run it with reduced sample count to get a quick check:

```bash
cargo bench -p oxifft --bench cooley_tukey_scaling -- --sample-size 10
```

To check for regressions against a saved baseline, use the helper script:

```bash
# Save the current main-branch performance as the reference baseline:
cargo bench -p oxifft --bench cooley_tukey_scaling -- --save-baseline main

# Later (e.g. in a PR), run the regression check:
./scripts/bench_check.sh main
```

Baseline JSON files committed at `benches/baselines/v0.3.0/` serve as
version-tagged reference points.  Full HTML reports are written to
`target/criterion/report/index.html`.

## Tools

### splitrs

Use `splitrs` for refactoring large files:

```bash
# Check files over 2000 lines
rslines 50

# Split a large module
splitrs src/large_module.rs --max-lines 2000
```

### tokei

Check code statistics:

```bash
# Project-wide statistics
tokei .

# Specific directory
tokei oxifft/src
```

### cocomo

Estimate development effort:

```bash
cocomo .
```

## Questions?

- Open an issue for bug reports or feature requests
- Use discussions for questions
- Check existing documentation and examples first

## License

By contributing to OxiFFT, you agree that your contributions will be licensed under Apache-2.0, matching the project license.
