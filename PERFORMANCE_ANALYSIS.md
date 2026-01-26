# OxiFFT Performance Analysis Guide

This document provides comprehensive guidance for analyzing the performance characteristics of OxiFFT and understanding optimization opportunities.

## Table of Contents

1. [Performance Overview](#performance-overview)
2. [Algorithm Complexity Analysis](#algorithm-complexity-analysis)
3. [Bottleneck Identification](#bottleneck-identification)
4. [SIMD Performance](#simd-performance)
5. [Cache Effects](#cache-effects)
6. [Threading Performance](#threading-performance)
7. [Comparison Methodology](#comparison-methodology)
8. [Optimization Strategies](#optimization-strategies)
9. [Profiling Guide](#profiling-guide)

## Performance Overview

### Expected Performance Characteristics

OxiFFT implements multiple FFT algorithms, each with different performance characteristics:

| Algorithm | Complexity | Best For | Sizes |
|-----------|-----------|----------|-------|
| Cooley-Tukey (Radix-2) | O(n log n) | Powers of 2 | 2, 4, 8, ..., 2^k |
| Cooley-Tukey (Radix-4) | O(n log n) | Powers of 4 | 4, 16, 64, ..., 4^k |
| Cooley-Tukey (Radix-8) | O(n log n) | Powers of 8 | 8, 64, 512, ... |
| Split-Radix | O(n log n) | Powers of 2 | Optimized for 16+ |
| Mixed-Radix | O(n log n) | Composite | 6, 10, 12, ... |
| Rader | O(n log n) | Primes | 3, 5, 7, 11, ... |
| Bluestein | O(n log n) | Any size | Fallback for all |
| Direct | O(n²) | Very small | n ≤ 4 |

### Performance Goals

**Primary Goals:**
- Competitive with FFTW for power-of-2 sizes (within 1.2x)
- Faster than rustfft for most sizes
- O(n log n) complexity verified across all sizes
- Efficient SIMD utilization (>80% of theoretical peak)

**Secondary Goals:**
- Low memory footprint (<2x transform size)
- Good cache behavior (minimal cache misses)
- Scalable threading (near-linear speedup to 4 cores)

## Algorithm Complexity Analysis

### Theoretical Operation Counts

For a size-n FFT:
- **Multiplications**: n log₂ n
- **Additions**: n log₂ n
- **Memory accesses**: 2n log₂ n (read + write)

### Verifying O(n log n) Complexity

Run benchmarks for doubling sizes:

```bash
cargo bench --bench dft_1d -- "power_of_2"
```

Expected timing ratio for 2x size increase:
```
T(2n) / T(n) ≈ 2 * log(2n) / log(n) = 2 * (log n + 1) / log n
```

**Example calculation:**
```
n=1024: log₂(1024) = 10
n=2048: log₂(2048) = 11
Ratio = 2 * 11/10 = 2.2x
```

### Empirical Complexity Verification

Create a script to verify complexity:

```bash
# Run benchmarks and extract timings
cargo bench --bench dft_1d -- "power_of_2" > bench_results.txt

# Analyze complexity
cat > analyze_complexity.py << 'EOF'
import re
import math

results = {}
with open('bench_results.txt') as f:
    for line in f:
        if 'fft_power_of_2' in line and 'time:' in line:
            match = re.search(r'fft_power_of_2/(\d+)\s+time:\s+\[([\d.]+)\s+([µm])s', line)
            if match:
                size = int(match.group(1))
                time = float(match.group(2))
                unit = match.group(3)
                if unit == 'm':
                    time *= 1000  # Convert ms to µs
                results[size] = time

# Calculate complexity factor
print("Size\tTime (µs)\tn log n\tTime/(n log n)")
print("-" * 50)
for size in sorted(results.keys()):
    time = results[size]
    n_log_n = size * math.log2(size)
    ratio = time / n_log_n
    print(f"{size}\t{time:.2f}\t\t{n_log_n:.0f}\t{ratio:.6f}")
EOF

python3 analyze_complexity.py
```

**Expected output:**
```
Size    Time (µs)    n log n    Time/(n log n)
--------------------------------------------------
64      1.23         384        0.003203
128     2.89         896        0.003225
256     6.45         2048       0.003149
512     14.2         4608       0.003081
1024    31.5         10240      0.003076
```

If Time/(n log n) is roughly constant, O(n log n) complexity is confirmed.

## Bottleneck Identification

### CPU-Bound vs Memory-Bound

#### Test: Increase CPU Frequency

```bash
# Before
cargo bench --bench dft_1d -- "power_of_2/1024"

# Increase CPU frequency (if possible)
# Run again and compare

# If performance scales with frequency: CPU-bound
# If performance doesn't change: Memory-bound
```

#### Memory Bandwidth Test

```bash
# Use different memory access patterns
# Small sizes fit in L1 cache: CPU-bound
# Large sizes exceed cache: Memory-bound

cargo bench --bench dft_1d -- "power_of_2/64"    # L1 cache
cargo bench --bench dft_1d -- "power_of_2/4096"  # Main memory
```

### Cache Miss Analysis

#### Using perf (Linux)

```bash
# Build benchmark
cargo bench --bench dft_1d --no-run

# Profile cache misses
perf stat -e cache-references,cache-misses \
    target/release/deps/dft_1d-* --bench

# Expected output:
# 1,234,567 cache-references
#    12,345 cache-misses     # < 1% is good, > 5% is problematic
```

#### Cache Levels

| Cache | Size | Latency | Bandwidth |
|-------|------|---------|-----------|
| L1    | 32-64 KB | ~4 cycles | ~200 GB/s |
| L2    | 256-512 KB | ~12 cycles | ~100 GB/s |
| L3    | 8-32 MB | ~40 cycles | ~50 GB/s |
| RAM   | 8-64 GB | ~200 cycles | ~20 GB/s |

**Optimization target:** Keep working set in L2 cache.

### Branch Prediction

```bash
# Profile branch mispredictions
perf stat -e branches,branch-misses \
    target/release/deps/dft_1d-* --bench

# < 1% branch misses is good
# > 5% indicates prediction issues
```

## SIMD Performance

### SIMD Instruction Coverage

Check SIMD usage with compiler output:

```bash
# Enable verbose SIMD reports
RUSTFLAGS="-C target-cpu=native -C opt-level=3" \
    cargo build --release --all-features
```

### SIMD Efficiency Metrics

**Theoretical peak (AVX2):**
- 4 double-precision floats per cycle per core
- At 3 GHz: 12 GFlops/core

**Measure actual:**
```bash
# Run benchmark
cargo bench --bench dft_1d -- "power_of_2/1024"

# Calculate GFlops
# FFT operations: 5 * n * log₂(n) flops
# For n=1024: 5 * 1024 * 10 = 51,200 flops
# Time: ~15 µs
# GFlops: 51200 / 15 / 1000 ≈ 3.4 GFlops
# Efficiency: 3.4 / 12 ≈ 28%
```

**Target:** >50% of theoretical peak with SIMD

### Backend Performance Comparison

Test different SIMD backends:

```bash
# Disable SIMD (scalar fallback)
RUSTFLAGS="-C target-feature=-sse2,-avx,-avx2" \
    cargo bench --bench dft_1d -- "power_of_2/1024"

# Enable AVX2
RUSTFLAGS="-C target-cpu=native" \
    cargo bench --bench dft_1d -- "power_of_2/1024"

# Expected speedup: 2-4x with SIMD
```

## Cache Effects

### Working Set Analysis

Calculate memory footprint:

```rust
// For complex FFT of size n:
// Input: n * 16 bytes (Complex<f64>)
// Output: n * 16 bytes
// Twiddle factors: n * 16 bytes
// Total: ~48n bytes

// L1 cache (32 KB): n ≤ 682
// L2 cache (256 KB): n ≤ 5461
// L3 cache (8 MB): n ≤ 174,762
```

### Cache-Aware Algorithm Selection

Benchmark different sizes to find cache boundaries:

```bash
cargo bench --bench dft_1d
```

Plot time per element vs size:
```
Time/n (ns) ▲
            │
         L3 │        ╱
            │      ╱
         L2 │    ╱
            │  ╱
         L1 │╱─────────
            └─────────────► Size (n)
            64  1K  4K  16K
```

### Optimizing for Cache

**Strategies:**
1. **Blocking**: Process data in cache-sized chunks
2. **Transpose optimization**: Minimize cache misses during transpose
3. **Twiddle factor caching**: Reuse precomputed values

## Threading Performance

### Scalability Analysis

Run benchmarks with different thread counts:

```bash
# Serial
RAYON_NUM_THREADS=1 cargo bench --bench dft_1d -- "power_of_2/4096"

# 2 threads
RAYON_NUM_THREADS=2 cargo bench --bench dft_1d -- "power_of_2/4096"

# 4 threads
RAYON_NUM_THREADS=4 cargo bench --bench dft_1d -- "power_of_2/4096"

# 8 threads
RAYON_NUM_THREADS=8 cargo bench --bench dft_1d -- "power_of_2/4096"
```

### Speedup Calculation

```
Speedup = T(1 thread) / T(n threads)
Efficiency = Speedup / n * 100%
```

**Expected efficiency:**
- 2 threads: >90%
- 4 threads: >75%
- 8 threads: >50%

### Amdahl's Law

Theoretical maximum speedup with parallel fraction P:
```
Speedup(n) = 1 / ((1-P) + P/n)
```

For 90% parallel code (P=0.9):
```
Speedup(4) = 1 / (0.1 + 0.9/4) = 3.08x
```

## Comparison Methodology

### Fair Comparison Guidelines

**Same conditions:**
1. Same hardware (CPU, RAM)
2. Same compiler flags
3. Same optimization level
4. Same measurement methodology
5. Warm cache (run warmup iterations)

### FFTW Comparison

```bash
# Run FFTW comparison
cargo bench --package oxifft-bench --features fftw-compare
```

**Interpret results:**
```
Ratio = T(OxiFFT) / T(FFTW)
```

- Ratio < 1.0: OxiFFT faster
- Ratio ≈ 1.0: Competitive
- Ratio > 1.5: Optimization needed

### RustFFT Comparison

```bash
# Run tests with rustfft comparison
cargo test --all-features rustfft_comparison -- --nocapture
```

## Optimization Strategies

### Priority Order

1. **Algorithm selection**: Choose optimal algorithm for size
2. **SIMD**: Maximize vector instruction usage
3. **Cache optimization**: Minimize cache misses
4. **Memory layout**: Improve data locality
5. **Threading**: Parallelize large transforms
6. **Micro-optimizations**: Reduce instruction count

### Specific Optimizations

#### 1. Algorithm Selection

```rust
// Automatic selection based on size
fn select_algorithm(n: usize) -> Algorithm {
    match n {
        1 => Nop,
        2..=4 => Direct,
        _ if is_power_of_2(n) && n >= 16 => SplitRadix,
        _ if is_power_of_2(n) => CooleyTukey,
        _ if is_prime(n) => Rader,
        _ => Bluestein,
    }
}
```

#### 2. SIMD Optimization

**Codelet generation:**
- Use SIMD for butterfly operations
- Minimize register spills
- Unroll small loops

**Check generated assembly:**
```bash
cargo rustc --release -- --emit asm
# Check for vmovapd, vfmadd*, etc.
```

#### 3. Cache Blocking

For 2D FFT:
```rust
// Block rows to fit in cache
const BLOCK_SIZE: usize = 128;
for chunk in rows.chunks(BLOCK_SIZE) {
    // Process block
}
```

#### 4. Prefetching

```rust
// Explicit prefetch for next iteration
#[cfg(target_arch = "x86_64")]
unsafe {
    core::arch::x86_64::_mm_prefetch(
        next_data.as_ptr() as *const i8,
        core::arch::x86_64::_MM_HINT_T0
    );
}
```

## Profiling Guide

### CPU Profiling (Linux)

```bash
# Build with debug symbols
cargo build --release --all-features

# Profile with perf
perf record -g cargo bench --bench dft_1d -- --profile-time 10

# Generate report
perf report

# Generate flamegraph
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
```

### CPU Profiling (macOS)

```bash
# Use Instruments
instruments -t "Time Profiler" \
    target/release/deps/dft_1d-* --bench
```

### Memory Profiling

```bash
# Valgrind massif
valgrind --tool=massif \
    target/release/deps/dft_1d-* --bench

# Analyze
ms_print massif.out.*
```

### Hotspot Identification

**Look for:**
1. Functions consuming >10% of time
2. Cache miss hotspots
3. Branch misprediction clusters
4. Lock contention points

### Optimization Cycle

```
1. Profile → 2. Identify bottleneck → 3. Optimize → 4. Verify → Repeat
```

## Performance Regression Testing

### Baseline Management

```bash
# Create baseline
cargo bench --all-features -- --save-baseline v0.1.2

# After changes
cargo bench --all-features -- --baseline v0.1.2

# Check for regressions
# Look for "Performance has regressed" messages
```

### Automated Regression Detection

```bash
# In CI/CD pipeline
cargo bench --all-features -- --baseline main

# Exit with error if regression > 10%
if [ $? -ne 0 ]; then
    echo "Performance regression detected!"
    exit 1
fi
```

## Performance Metrics Summary

### Key Performance Indicators (KPIs)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Latency (1024-point) | <20 µs | Time per transform |
| Throughput | >50 Melem/s | Elements per second |
| SIMD efficiency | >50% | Actual vs theoretical peak |
| Cache miss rate | <2% | perf stat |
| Threading efficiency (4 cores) | >75% | Speedup / 4 |
| vs FFTW ratio | <1.2x | OxiFFT time / FFTW time |
| Memory footprint | <3x size | Peak RSS |

### Performance Report Template

```markdown
# Performance Report - [Date]

## System Configuration
- CPU: [Model, Cores, Frequency]
- RAM: [Size, Speed]
- Compiler: rustc [version]
- Flags: [RUSTFLAGS]

## Benchmark Results

### Latency (µs)
| Size | OxiFFT | FFTW | Ratio | Target | Status |
|------|--------|------|-------|--------|--------|
| 256  | 3.8    | 3.5  | 1.09x | <1.2x  | ✅ PASS |
| 1024 | 18.2   | 16.8 | 1.08x | <1.2x  | ✅ PASS |
| 4096 | 89.5   | 82.1 | 1.09x | <1.2x  | ✅ PASS |

### SIMD Efficiency
- Theoretical peak: 12 GFlops (AVX2 @ 3GHz)
- Actual: 6.2 GFlops
- Efficiency: 52%
- Status: ✅ PASS (>50%)

### Threading Scalability
| Threads | Time (µs) | Speedup | Efficiency | Status |
|---------|-----------|---------|------------|--------|
| 1       | 89.5      | 1.00x   | 100%       | -      |
| 2       | 47.2      | 1.90x   | 95%        | ✅ PASS |
| 4       | 28.3      | 3.16x   | 79%        | ✅ PASS |

## Bottleneck Analysis
- [Primary bottleneck: e.g., memory bandwidth]
- [Cache behavior: good/poor]
- [SIMD utilization: optimal/suboptimal]

## Optimization Opportunities
1. [Specific optimization 1]
2. [Specific optimization 2]

## Conclusion
- Overall performance: [PASS/NEEDS IMPROVEMENT]
- Production ready: [YES/NO]
```

## Continuous Performance Monitoring

### Integration with CI/CD

```yaml
# .github/workflows/benchmark.yml
name: Benchmark

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Run benchmarks
        run: cargo bench --all-features

      - name: Store benchmark result
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'cargo'
          output-file-path: target/criterion/report/index.html
```

## Conclusion

Performance analysis is an iterative process:

1. **Measure**: Run comprehensive benchmarks
2. **Analyze**: Identify bottlenecks using profiling
3. **Optimize**: Apply targeted optimizations
4. **Verify**: Confirm improvements with benchmarks
5. **Document**: Record results and insights

For specific performance issues or optimization questions, refer to:
- `BENCHMARKING.md` for running benchmarks
- `oxifft.md` for architecture details
- GitHub issues for community support

---

**Next Steps:**
1. Run initial benchmarks (see `BENCHMARKING.md`)
2. Establish baseline performance
3. Profile to identify hotspots
4. Optimize critical paths
5. Re-benchmark and verify improvements
