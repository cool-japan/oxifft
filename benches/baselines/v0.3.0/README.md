# FFTW Parity Baseline Snapshots

Files here are machine-generated performance ratio snapshots produced by
`scripts/fftw_ratio_report.sh`. Each JSON captures oxifft/fftw timing ratios
at 7 v1.0 performance gate sizes.

## Schema

- `date`: ISO date of the run
- `git_sha`: short Git SHA at time of measurement
- `cpu`: CPU brand string
- `rustc`: compiler version
- `gates[]`: per-gate ratios with `pass: true` when below `target_ratio`
- `summary.geomean_ratio`: geometric mean across all gates

## Regenerating

```bash
# Requires: fftw installed (brew install fftw / apt install libfftw3-dev)
bash scripts/fftw_ratio_report.sh
```

## Target ratios (v1.0 goals)

| Gate | Target |
|------|--------|
| 1d_cplx_2^10 | < 2× FFTW |
| 1d_cplx_2^20 | < 2× FFTW |
| 1d_real_2^10 | < 2× FFTW |
| 2d_cplx_1024 | < 2× FFTW |
| batch_1000×256 | < 2× FFTW |
| prime_2017 | < 3× FFTW |
| dct2_1024 | < 3× FFTW |

## Snapshots

- fftw_ratios_2026-04-20-post-makhoul.json — post-Makhoul DCT re-measurement at HEAD 6a6c7a4 (2026-04-20)
