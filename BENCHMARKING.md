# OxiFFT Benchmarking Guide

This document provides comprehensive instructions for running performance benchmarks on the OxiFFT library.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Benchmark Suites](#benchmark-suites)
3. [Running Benchmarks](#running-benchmarks)
4. [Understanding Results](#understanding-results)
5. [Comparison Testing](#comparison-testing)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

```bash
# Ensure you have Rust toolchain installed
rustc --version  # Should be 1.75 or later

# For FFTW comparison (optional)
# macOS:
brew install fftw

# Ubuntu/Debian:
sudo apt-get install libfftw3-dev

# Arch Linux:
sudo pacman -S fftw
```

### Run All Benchmarks

```bash
# Run complete benchmark suite
cargo bench --all-features

# View HTML reports
open target/criterion/report/index.html
```

## Benchmark Suites

### 1. Core DFT Benchmarks (`dft_1d`)

Location: `oxifft/benches/dft_1d.rs`

**Coverage:**
- Power-of-2 sizes: 64, 128, 256, 512, 1024, 4096
- Prime sizes: 5, 7, 11, 13, 17, 97
- Composite sizes: 12, 15, 100, 1000
- Forward transforms
- Inverse transforms
- Forward/inverse roundtrips

**Run:**
```bash
cargo bench --bench dft_1d
```

**Expected Performance (Reference):**
- 256-point FFT: ~3-5 µs
- 1024-point FFT: ~15-25 µs
- 4096-point FFT: ~70-100 µs

### 2. Advanced Features Benchmarks (`beyond_fftw`)

Location: `oxifft/benches/beyond_fftw.rs`

**Coverage:**
- Sparse FFT (k=10, n=1024/4096/16384)
- Pruned FFT (output-pruned, Goertzel)
- Streaming FFT (STFT with various configurations)
- Compile-time FFT (const-fft for sizes 8-1024)
- Window functions (Hann, Hamming, Blackman, Kaiser)

**Run:**
```bash
cargo bench --bench beyond_fftw --all-features
```

**Features Tested:**
- `sparse`: Sparse FFT algorithms
- `pruned`: Input/output pruning
- `streaming`: STFT and real-time processing
- `const-fft`: Compile-time optimized transforms

### 3. FFTW Comparison Benchmarks (`fft_comparison`)

Location: `oxifft-bench/benches/fft_comparison.rs`

**Coverage:**
- Head-to-head comparison with FFTW3
- Multiple sizes from 16 to 65536
- Power-of-2, prime, and composite sizes
- Batch transforms

**Run:**
```bash
cargo bench --package oxifft-bench --features fftw-compare
```

**Requirements:**
- FFTW3 library installed on system
- Feature flag `fftw-compare` enabled

### 4. SAR Processing Benchmarks

Location: `oxifft-bench/benches/fft_comparison.rs`

Specialized benchmarks for Synthetic Aperture Radar (SAR) image processing workloads. These benchmarks simulate real-world SAR signal processing pipelines.

**Coverage:**

| Benchmark | Description | Typical Sizes |
|-----------|-------------|---------------|
| `sar_range_compression` | 1D FFT for range line processing | 4096-32768 |
| `sar_azimuth_batch` | Batch FFTs for azimuth compression | 2048-16384 x 512-2048 |
| `sar_2d_image` | 2D FFT for image formation (Range-Doppler, ω-k) | 512x512 - 4096x4096 |
| `sar_chirp_conv` | FFT→Multiply→IFFT chirp convolution pipeline | 4096-16384 |
| `sar_roundtrip` | Forward + Inverse FFT cycle | 4096-32768 |
| `sar_real_fft` | Real-to-Complex for raw SAR data | 4096-32768 |

**Run:**
```bash
# All SAR benchmarks
cargo bench --bench fft_comparison -- "sar_"

# Specific SAR benchmarks
cargo bench --bench fft_comparison -- "sar_range"      # Range compression
cargo bench --bench fft_comparison -- "sar_azimuth"    # Azimuth batch processing
cargo bench --bench fft_comparison -- "sar_2d"         # 2D image formation
cargo bench --bench fft_comparison -- "sar_chirp"      # Chirp convolution
cargo bench --bench fft_comparison -- "sar_roundtrip"  # Forward+Inverse cycle
cargo bench --bench fft_comparison -- "sar_real"       # Real-to-Complex FFT
```

**SAR Processing Context:**

1. **Range Compression**: Processes each range line (radar echo) with 1D FFT. Typical satellite SAR uses 4096-16384 samples per range line.

2. **Azimuth Compression**: Processes columns (azimuth direction) in batch. A typical scene might have 2048-8192 azimuth samples × 1024-4096 range bins.

3. **2D Image Formation**: Algorithms like Range-Doppler and ω-k require large 2D FFTs (2048x2048 to 8192x8192).

4. **Chirp Convolution**: Core of matched filtering - Forward FFT, multiply with chirp reference, Inverse FFT.

**Expected Performance (ARM64/NEON reference):**

| Benchmark | Size | Time |
|-----------|------|------|
| Range Compression | 8192 | ~60 µs |
| Range Compression | 16384 | ~140 µs |
| 2D Image | 2048x2048 | ~95 ms |
| Chirp Convolution | 8192 | ~130 µs |

## Running Benchmarks

### Basic Commands

```bash
# Run specific benchmark suite
cargo bench --bench dft_1d

# Run benchmarks for specific size
cargo bench --bench dft_1d -- "power_of_2/1024"

# Run all benchmarks with all features
cargo bench --all-features

# Run benchmarks without generating reports (faster)
cargo bench --all-features --no-fail-fast
```

### Filtering Benchmarks

```bash
# Run only power-of-2 benchmarks
cargo bench --bench dft_1d -- "power_of_2"

# Run only prime size benchmarks
cargo bench --bench dft_1d -- "prime"

# Run only sparse FFT benchmarks
cargo bench --bench beyond_fftw --all-features -- "sparse"
```

### Saving Baselines

```bash
# Save current performance as baseline
cargo bench --bench dft_1d -- --save-baseline my-baseline

# Compare against saved baseline
cargo bench --bench dft_1d -- --baseline my-baseline
```

## Understanding Results

### Criterion Output Format

```
fft_power_of_2/256      time:   [3.7474 µs 3.8549 µs 3.9997 µs]
                        change: [-2.3456% -0.1234% +1.7890%] (p = 0.45 > 0.05)
                        No change in performance detected.
Found 8 outliers among 100 measurements (8.00%)
  3 (3.00%) high mild
  5 (5.00%) high severe
```

**Interpretation:**
- **Time**: [Lower bound, Estimate, Upper bound] at 95% confidence
- **Change**: Performance change vs previous run (if baseline exists)
- **p-value**: Statistical significance (< 0.05 indicates significant change)
- **Outliers**: Measurements that deviate from normal distribution

### Performance Metrics

#### Latency
- Time taken for a single transform
- Lower is better
- Measured in µs (microseconds) or ms (milliseconds)

#### Throughput
- Elements processed per second
- Higher is better
- Displayed as "Elements/s" in reports

### HTML Reports

Criterion generates detailed HTML reports in `target/criterion/`:

```
target/criterion/
├── report/
│   └── index.html          # Main report with all benchmarks
├── fft_power_of_2/
│   ├── 256/
│   │   ├── report/
│   │   │   └── index.html  # Detailed report for this benchmark
│   │   └── base/           # Historical data
│   └── ...
```

**View reports:**
```bash
# macOS
open target/criterion/report/index.html

# Linux
xdg-open target/criterion/report/index.html

# Windows
start target/criterion/report/index.html
```

## Comparison Testing

### Compare OxiFFT vs FFTW

```bash
# Build and run FFTW comparison
cargo bench --package oxifft-bench --features fftw-compare

# Results will show performance ratio
# Example output:
# oxifft/fftw_1024    time:   [12.345 µs]  (oxifft: 12.3 µs, FFTW: 11.8 µs, ratio: 1.04x)
```

### Compare OxiFFT vs RustFFT

RustFFT comparison is included in the test suite:

```bash
cargo test --all-features rustfft_comparison
```

### Baseline Comparison Workflow

```bash
# 1. Save baseline before optimization
cargo bench --bench dft_1d -- --save-baseline before-opt

# 2. Make code changes
# ... edit code ...

# 3. Compare against baseline
cargo bench --bench dft_1d -- --baseline before-opt

# 4. Review changes in HTML report
open target/criterion/report/index.html
```

## Advanced Usage

### Custom Sample Size

```bash
# Increase sample count for more accurate results
cargo bench --bench dft_1d -- --sample-size 200
```

### Warm-up Configuration

```bash
# Adjust warm-up time (default: 3 seconds)
cargo bench --bench dft_1d -- --warm-up-time 5
```

### Profiling Integration

```bash
# Run benchmarks under profiler
cargo bench --bench dft_1d --no-run
perf record -g target/release/deps/dft_1d-* --bench

# Generate flamegraph
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

### Benchmark-Specific Configurations

Edit `benches/dft_1d.rs` to customize:

```rust
// Adjust measurement time
group.measurement_time(Duration::from_secs(10));

// Set sample size
group.sample_size(100);

// Configure throughput measurement
group.throughput(Throughput::Elements(n as u64));
```

## Troubleshooting

### Common Issues

#### 1. Gnuplot Not Found

**Problem:**
```
Gnuplot not found, using plotters backend
```

**Solution:**
This is a warning, not an error. Criterion will use the plotters backend instead.

To install gnuplot (optional):
```bash
# macOS
brew install gnuplot

# Ubuntu/Debian
sudo apt-get install gnuplot

# Arch Linux
sudo pacman -S gnuplot
```

#### 2. FFTW Not Found

**Problem:**
```
error: linking with `cc` failed
  = note: ld: library not found for -lfftw3
```

**Solution:**
Install FFTW3 library (see Prerequisites) or run without FFTW comparison:
```bash
cargo bench --bench dft_1d
```

#### 3. Out of Memory

**Problem:**
Large transform sizes cause OOM errors.

**Solution:**
Run benchmarks for smaller sizes only:
```bash
cargo bench --bench dft_1d -- "power_of_2" --skip "4096"
```

#### 4. Inconsistent Results

**Problem:**
High variance in measurements.

**Causes & Solutions:**
- **CPU throttling**: Ensure laptop is plugged in and not in power-saving mode
- **Background processes**: Close unnecessary applications
- **Thermal throttling**: Allow CPU to cool between runs
- **Turbo boost**: Disable CPU frequency scaling for consistent results

```bash
# Linux: Disable CPU frequency scaling
sudo cpupower frequency-set --governor performance

# After benchmarking, restore
sudo cpupower frequency-set --governor powersave
```

### Best Practices

1. **Run on idle system**: Close browsers, IDEs, and other heavy applications
2. **Disable CPU turbo**: For consistent results
3. **Multiple runs**: Run benchmarks 3 times and take median
4. **Warm system**: Let CPU warm up with a quick benchmark run first
5. **Fixed power mode**: Use AC power, disable power-saving modes
6. **Same conditions**: Compare benchmarks run under identical conditions

## Performance Targets

### Expected Performance Characteristics

| Transform Size | Time (µs) | Operations | Algorithm |
|----------------|-----------|------------|-----------|
| 64             | ~1-2      | 64 log₂ 64 = 384 | Cooley-Tukey |
| 256            | ~3-5      | 256 log₂ 256 = 2,048 | Cooley-Tukey |
| 1024           | ~15-25    | 1024 log₂ 1024 = 10,240 | Cooley-Tukey |
| 4096           | ~70-100   | 4096 log₂ 4096 = 49,152 | Cooley-Tukey |
| 16384          | ~350-500  | 16384 log₂ 16384 = 229,376 | Cooley-Tukey |

### Complexity Verification

Verify O(n log n) complexity by checking that:
```
T(2n) ≈ 2 * T(n) * log(2n)/log(n)
```

Example:
```
T(1024) / T(512) ≈ 2 * (log 1024 / log 512) = 2 * (10/9) ≈ 2.22
```

## Benchmark Results Template

Use this template to document benchmark results:

```markdown
## Benchmark Results - [Date]

**System:**
- CPU: [Model, Cores, Base/Boost MHz]
- RAM: [Size, Type, Speed]
- OS: [OS Version]
- Rust: [rustc version]

**Configuration:**
- Compiler flags: [e.g., -C target-cpu=native]
- Features: [enabled features]

**Results:**

| Size | Time (µs) | Throughput (Melem/s) | vs FFTW | Notes |
|------|-----------|----------------------|---------|-------|
| 256  | 3.85      | 66.5                 | 1.05x   | -     |
| 1024 | 18.2      | 56.3                 | 1.08x   | -     |
| 4096 | 89.5      | 45.8                 | 1.12x   | -     |

**Observations:**
- [Key findings]
- [Performance characteristics]
- [Optimization opportunities]
```

## Next Steps

After running benchmarks:

1. **Review HTML reports** for detailed analysis
2. **Compare with FFTW** to validate performance
3. **Identify bottlenecks** using profiling tools
4. **Document results** for future reference
5. **Iterate optimizations** and re-benchmark

## Additional Resources

- **Criterion.rs Guide**: https://bheisler.github.io/criterion.rs/book/
- **Rust Performance Book**: https://nnethercote.github.io/perf-book/
- **FFTW Documentation**: http://www.fftw.org/fftw3_doc/
- **FFT Algorithms**: Cooley-Tukey, Bluestein, Rader (see `oxifft.md`)

---

For questions or issues with benchmarking, please open an issue at:
https://github.com/oxifft/oxifft/issues
