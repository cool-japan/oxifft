# OxiFFT Benchmark Results

**Date:** [YYYY-MM-DD]
**Version:** [e.g., v0.1.0]
**Tester:** [Name or Team]

## System Configuration

### Hardware
- **CPU:** [Model, e.g., Intel Core i7-9750H]
  - Cores: [Physical cores, e.g., 6 cores]
  - Threads: [Logical threads, e.g., 12 threads]
  - Base Clock: [e.g., 2.6 GHz]
  - Boost Clock: [e.g., 4.5 GHz]
  - Cache: [L1/L2/L3 sizes]
- **RAM:** [Size and Speed, e.g., 32 GB DDR4-2666]
- **Storage:** [Type, e.g., NVMe SSD]
- **GPU:** [If testing GPU features, e.g., NVIDIA RTX 3080]

### Software
- **OS:** [e.g., macOS 13.5, Ubuntu 22.04, Windows 11]
- **Rust Compiler:** [rustc version, e.g., 1.75.0]
- **Cargo Version:** [e.g., 1.75.0]
- **LLVM Version:** [if known]
- **RUSTFLAGS:** [e.g., `-C target-cpu=native -C opt-level=3`]

### Build Configuration
- **Profile:** [e.g., release]
- **Features Enabled:** [e.g., `all-features`, or specific list]
- **Target:** [e.g., x86_64-unknown-linux-gnu]
- **LTO:** [yes/no]
- **Codegen Units:** [e.g., 1 for maximum optimization]

## Test Conditions

- **CPU Governor:** [e.g., performance, powersave]
- **Turbo Boost:** [enabled/disabled]
- **Power Mode:** [e.g., AC power, maximum performance]
- **Background Load:** [e.g., idle system, specific processes running]
- **Thermal State:** [e.g., cold start, warmed up]
- **Number of Runs:** [e.g., 3 runs, averaged]

## Core DFT Benchmarks

### Power-of-2 Sizes (Complex FFT)

| Size  | Time (µs) | StdDev | Elements/s | vs Baseline | Notes |
|-------|-----------|--------|------------|-------------|-------|
| 64    |           |        |            |             |       |
| 128   |           |        |            |             |       |
| 256   |           |        |            |             |       |
| 512   |           |        |            |             |       |
| 1024  |           |        |            |             |       |
| 2048  |           |        |            |             |       |
| 4096  |           |        |            |             |       |
| 8192  |           |        |            |             |       |
| 16384 |           |        |            |             |       |
| 32768 |           |        |            |             |       |

### Prime Sizes (Complex FFT)

| Size | Time (µs) | StdDev | Algorithm | vs Power-of-2 | Notes |
|------|-----------|--------|-----------|---------------|-------|
| 5    |           |        | Rader     |               |       |
| 7    |           |        | Rader     |               |       |
| 11   |           |        | Rader     |               |       |
| 13   |           |        | Rader     |               |       |
| 17   |           |        | Rader     |               |       |
| 31   |           |        | Rader     |               |       |
| 97   |           |        | Rader     |               |       |
| 127  |           |        | Rader     |               |       |

### Composite Sizes (Complex FFT)

| Size | Time (µs) | StdDev | Factorization | Algorithm | Notes |
|------|-----------|--------|---------------|-----------|-------|
| 12   |           |        | 2² × 3        | Mixed     |       |
| 15   |           |        | 3 × 5         | Mixed     |       |
| 100  |           |        | 2² × 5²       | Mixed     |       |
| 120  |           |        | 2³ × 3 × 5    | Mixed     |       |
| 1000 |           |        | 2³ × 5³       | Mixed     |       |

### Real FFT (R2C/C2R)

| Size  | R2C Time (µs) | C2R Time (µs) | vs Complex FFT | Notes |
|-------|---------------|---------------|----------------|-------|
| 64    |               |               |                |       |
| 256   |               |               |                |       |
| 1024  |               |               |                |       |
| 4096  |               |               |                |       |

### 2D FFT

| Size (N×M) | Time (ms) | Throughput (Melem/s) | Notes |
|------------|-----------|----------------------|-------|
| 64×64      |           |                      |       |
| 128×128    |           |                      |       |
| 256×256    |           |                      |       |
| 512×512    |           |                      |       |

## Advanced Features Benchmarks

### Sparse FFT

| Size (n) | Sparsity (k) | Time (µs) | vs Regular FFT | Speedup | Notes |
|----------|--------------|-----------|----------------|---------|-------|
| 1024     | 10           |           |                |         |       |
| 4096     | 10           |           |                |         |       |
| 16384    | 10           |           |                |         |       |

### Pruned FFT (Goertzel)

| Size | Num Outputs | Time (µs) | vs Full FFT | Notes |
|------|-------------|-----------|-------------|-------|
| 256  | 1           |           |             |       |
| 1024 | 1           |           |             |       |
| 4096 | 1           |           |             |       |

### Streaming FFT (STFT)

| Signal Length | FFT Size | Hop Size | Time (ms) | Frames/s | Notes |
|---------------|----------|----------|-----------|----------|-------|
| 4096          | 256      | 64       |           |          |       |
| 16384         | 512      | 128      |           |          |       |
| 65536         | 1024     | 256      |           |          |       |

### Compile-Time FFT (const-fft)

| Size | Time (ns) | vs Runtime FFT | Notes |
|------|-----------|----------------|-------|
| 8    |           |                |       |
| 16   |           |                |       |
| 32   |           |                |       |
| 64   |           |                |       |
| 128  |           |                |       |
| 256  |           |                |       |

## FFTW Comparison

### Power-of-2 Sizes

| Size  | OxiFFT (µs) | FFTW (µs) | Ratio | Status | Notes |
|-------|-------------|-----------|-------|--------|-------|
| 64    |             |           |       |        |       |
| 256   |             |           |       |        |       |
| 1024  |             |           |       |        |       |
| 4096  |             |           |       |        |       |
| 16384 |             |           |       |        |       |

**Average Ratio:** [e.g., 1.08x]
**Target:** <1.2x
**Status:** [PASS/FAIL]

### Different Planning Modes

| Size | ESTIMATE (µs) | MEASURE (µs) | PATIENT (µs) | Notes |
|------|---------------|--------------|--------------|-------|
| 1024 |               |              |              |       |
| 4096 |               |              |              |       |

## Threading Performance

### Scalability (Size = 4096)

| Threads | Time (µs) | Speedup | Efficiency | Status | Notes |
|---------|-----------|---------|------------|--------|-------|
| 1       |           | 1.00x   | 100%       | -      |       |
| 2       |           |         |            |        |       |
| 4       |           |         |            |        |       |
| 8       |           |         |            |        |       |
| 16      |           |         |            |        |       |

**Amdahl's Law Analysis:**
- Serial fraction: [calculated]
- Maximum theoretical speedup: [calculated]
- Actual vs theoretical: [%]

## SIMD Performance

### Backend Comparison (Size = 1024)

| Backend    | Time (µs) | Speedup vs Scalar | Instructions | Notes |
|------------|-----------|-------------------|--------------|-------|
| Scalar     |           | 1.00x             | No SIMD      |       |
| SSE2       |           |                   | 128-bit      |       |
| AVX        |           |                   | 256-bit      |       |
| AVX2       |           |                   | 256-bit+FMA  |       |
| AVX-512    |           |                   | 512-bit      |       |
| NEON       |           |                   | ARM 128-bit  |       |

**Best Backend:** [e.g., AVX2]
**SIMD Efficiency:** [actual GFLOPS / theoretical GFLOPS]

## Cache Effects

### Performance vs Size (Identifying Cache Boundaries)

| Size  | Time (µs) | Time/Element (ns) | Cache Level | Notes |
|-------|-----------|-------------------|-------------|-------|
| 16    |           |                   | L1          |       |
| 64    |           |                   | L1          |       |
| 256   |           |                   | L1/L2       |       |
| 1024  |           |                   | L2          |       |
| 4096  |           |                   | L2/L3       |       |
| 16384 |           |                   | L3          |       |
| 65536 |           |                   | RAM         |       |

**Cache Cliff Detected at:** [size where performance degrades significantly]

## Memory Footprint

### Peak Memory Usage

| Size  | Peak RSS (MB) | Per Element (bytes) | Ratio to Input | Notes |
|-------|---------------|---------------------|----------------|-------|
| 1024  |               |                     |                |       |
| 4096  |               |                     |                |       |
| 16384 |               |                     |                |       |

**Target:** <3x input size
**Actual:** [ratio]
**Status:** [PASS/FAIL]

## Profiling Data

### Hotspots (Top 10 Functions)

| Function | Time % | Calls | Time/Call | Notes |
|----------|--------|-------|-----------|-------|
|          |        |       |           |       |
|          |        |       |           |       |

### Cache Statistics (via perf)

- **Cache References:** [count]
- **Cache Misses:** [count]
- **Miss Rate:** [%]
- **Status:** [GOOD <2%, ACCEPTABLE 2-5%, POOR >5%]

### Branch Prediction

- **Branches:** [count]
- **Branch Misses:** [count]
- **Miss Rate:** [%]
- **Status:** [GOOD <1%, ACCEPTABLE 1-5%, POOR >5%]

## Complexity Verification

### O(n log n) Confirmation

| Size | Time (µs) | n log₂ n | Time/(n log n) | Deviation |
|------|-----------|----------|----------------|-----------|
| 64   |           | 384      |                |           |
| 128  |           | 896      |                |           |
| 256  |           | 2048     |                |           |
| 512  |           | 4608     |                |           |
| 1024 |           | 10240    |                |           |

**Constant Factor:** [average of Time/(n log n)]
**Variance:** [std dev / mean]
**Status:** [CONFIRMED if variance <10%]

## Performance Summary

### Overall Assessment

**Strengths:**
- [e.g., Excellent power-of-2 performance]
- [e.g., Good SIMD utilization]
- [e.g., Competitive with FFTW]

**Weaknesses:**
- [e.g., Prime sizes slower than expected]
- [e.g., Threading efficiency drops beyond 4 cores]
- [e.g., Memory footprint higher for large sizes]

**Optimization Opportunities:**
1. [Specific optimization 1]
2. [Specific optimization 2]
3. [Specific optimization 3]

### KPI Status

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Latency (1024-pt) | <20 µs | | |
| Throughput | >50 Melem/s | | |
| SIMD Efficiency | >50% | | |
| Cache Miss Rate | <2% | | |
| Threading (4 cores) | >75% efficiency | | |
| vs FFTW | <1.2x | | |
| Memory Footprint | <3x | | |

**Overall Status:** [PASS/FAIL]
**Production Ready:** [YES/NO]

## Recommendations

### Immediate Actions
1. [Action item 1]
2. [Action item 2]

### Future Optimizations
1. [Long-term optimization 1]
2. [Long-term optimization 2]

### Next Steps
1. [Next benchmark or test]
2. [Analysis to perform]

## Appendix

### Raw Benchmark Output

```
[Paste raw criterion output here]
```

### Build Command

```bash
[Exact command used to build]
```

### Benchmark Command

```bash
[Exact command used to run benchmarks]
```

### System Info

```
[Output of system info commands like lscpu, sysctl, etc.]
```

---

**Report Generated By:** [Name]
**Review Status:** [Draft/Final]
**Approved By:** [Name, if applicable]
