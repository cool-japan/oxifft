# Security Policy

## Supported versions

OxiFFT follows the COOLJAPAN ecosystem's rolling-release model. Only the
**latest released 0.x line** is supported with security fixes. Older 0.x
releases do not receive backported patches; please upgrade to the latest
release before reporting an issue to confirm it is still reproducible.

## Reporting a vulnerability

Please **do not** file a public GitHub issue for a suspected security
vulnerability. Instead, report it privately by emailing:

**info@kitasan.io**

Include as much detail as you can: affected crate and version, a minimal
reproduction (code, a wisdom file/string, or a `cargo fuzz` corpus entry is
ideal), enabled feature flags and target triple, the observed impact (panic,
memory unsafety, incorrect transform output, hang, excessive memory, etc.),
and — if known — a suggested fix or mitigation.

Reports are triaged privately by the maintainer. We will acknowledge receipt
as soon as practical, investigate, and coordinate a fix and disclosure
timeline with the reporter before any public disclosure.

## Threat model

OxiFFT's `unsafe` surface is concentrated in per-architecture SIMD codelets
(`src/simd/`, `src/dft/codelets/`), raw-pointer parallel-plan partitioning
(`src/api/parallel.rs`), and the low-level `DftProblem`/`DftPlan` primitives
(`src/dft/problem.rs`, `src/dft/plan.rs`). All of it is expected to be
memory-safe when reached only through the crate's public safe API; a memory-
safety bug or soundness hole reachable from 100% safe caller code — with no
`unsafe` block in the caller — is a security bug, not merely a correctness
bug, and should be reported through the process above rather than as a
regular issue.

`oxifft`'s wisdom system (`src/api/wisdom.rs`) is designed to safely consume
**untrusted, potentially adversarial** input: `WisdomCache::import_string` /
`import_from_string` / `import_from_file`, and `import_system_wisdom` (which
auto-reads `$XDG_CONFIG_HOME/oxifft/wisdom`, `$HOME/.config/oxifft/wisdom`,
and `/etc/oxifft/wisdom`) all accept a wisdom file that may have been
downloaded, shared, or otherwise supplied by someone other than the caller.
Both the S-expression text format and the binary format (`from_binary` /
`import_binary`) must never, on any input:

- panic (`unwrap()`, `expect()`, `panic!()`, `unreachable!()`, out-of-bounds
  indexing, or an integer-overflow abort in debug builds),
- cause unbounded memory allocation (a crafted small file that claims an
  enormous entry count — see `MAX_WISDOM_BINARY_ENTRIES` and the text-format
  import size ceiling),
- or — the correctness-adjacent case specific to a wisdom consumer — cause
  planning to silently reconstruct an algorithm that is **not applicable to
  the requested transform size** and produce wrong output without any error
  or panic (see `algorithm_from_solver_name` in `src/api/plan/types.rs`,
  which re-validates every solver-name arm against the size for exactly this
  reason).

Malformed wisdom input should instead be rejected with a typed `WisdomError`,
or (for stored entries specifically) silently skipped by `is_valid_entry`,
never accepted and then misapplied.

## Fuzzing

Five `cargo-fuzz` harnesses (`oxifft/fuzz/fuzz_targets/`) exercise the
untrusted-input surface and are expected to build and run cleanly (no crash,
no `should_panic`-free assertion failure) on every change to a decode or
wisdom-consuming path:

| Target | Exercises |
|---|---|
| `wisdom_parse` | S-expression text wisdom parser (`import_from_string`) |
| `wisdom_parse_binary` | Binary wisdom decoder (`WisdomCache::from_binary` / `import_binary`), including round-trip re-encoding |
| `wisdom_measure_roundtrip` | Import a fuzzed wisdom string, then plan + execute with `Flags::MEASURE`/`Flags::WISDOM_ONLY` — the path that reaches `algorithm_from_solver_name`'s applicability gating |
| `plan_create` | `Plan`/`RealPlan`/`R2rPlan` construction never panics across arbitrary sizes and R2R kinds |
| `r2c_roundtrip` | R2C/C2R round-trip numerical correctness across arbitrary sizes and input magnitudes |

Run e.g. `cargo +nightly fuzz run wisdom_parse_binary` from `oxifft/fuzz/` to
fuzz locally (requires the `nightly` toolchain and `cargo-fuzz`). A
production run is expected to run for an extended period (hours, not
seconds) against each of the wisdom-parsing targets before a release; see
`TODO.md`'s Quality Gates section for the current status.

## Scope

This policy covers the crates published from this repository
(https://github.com/cool-japan/oxifft): `oxifft`, `oxifft-codegen`,
`oxifft-codegen-impl`, and `oxifft-adapter-mpi`. `oxifft-bench` is an
unpublished (`publish = false`), development-only benchmarking/comparison
crate and is out of scope, as is any vulnerability that only reproduces with
a non-default, comparison-only feature (`fftw-compare`, `fftw-system`,
`rustfft-compare`) enabled. Vulnerabilities in upstream dependencies should
be reported to those projects directly, though we welcome a heads-up so we
can track and update our dependency pins.

## Maintainer

COOLJAPAN OU (Team Kitasan)
