# OxiFFT Wisdom File Format Specification

Current text format version: 2 (`WISDOM_FORMAT_VERSION = 2`)
Current binary format version: 2 (`BINARY_FORMAT_VERSION = 2`, backward-compatible reader for version 1)

OxiFFT has two on-disk wisdom representations:

- A **text** (S-expression) format — human-readable, used by `import_from_file`/`export_to_file`
  and the global-cache free functions. See [Text format](#text-format) below.
- A **binary** format — compact, used by `WisdomCache::to_binary`/`from_binary` and the
  build-time baseline embedded via `OXIFFT_TUNE=1`. See [Binary format](#binary-format) below.

Both formats store the same logical data (a map from problem hash to `(solver_name, cost)`),
but only the binary format is currently consulted to reconstruct an executable algorithm at
plan-construction time (`Plan::dft_1d`'s baseline-wisdom lookup); the text format is used for
export/import/merge bookkeeping.

## Text format

### Overview

OxiFFT wisdom files store plan timing measurements in an S-expression format.
When loaded, the planner skips benchmarking and reuses previously measured solver selections.

### Grammar (EBNF)

```ebnf
wisdom-file    = "(" "oxifft-wisdom" ws format-version ws entry* ws ")"
format-version = "(" "format_version" ws integer ")"
entry          = "(" ws hash ws string ws cost ws ")"
hash           = unsigned-64-bit-decimal-integer
string         = '"' <UTF-8 chars without '"'> '"'
cost           = floating-point-number  ; non-finite or negative values are rejected on import
ws             = whitespace*
```

### Entry validation rules

An entry is rejected during import (counted in `skipped_invalid`) when any of:
- `hash == 0`
- `solver_name` is an empty string
- `cost` is NaN, infinite, or negative
- `solver_name` uses the `"mixed-radix-R1-R2-..."` naming convention but the factors are not
  all supported radices (`{2,3,4,5,7,8,16}`), or their product does not equal `hash`

The last rule exists because a `mixed-radix-...` solver name is the one wisdom shape that gets
reconstructed back into an executable algorithm (see [Binary format](#binary-format)); a
degenerate factor list (e.g. a `0` factor, or an unsupported radix) would otherwise panic when
replayed, so it is rejected at import time instead.

### Legacy format (version 0)

Files written by OxiFFT v0.1.x use a legacy header:

```
(oxifft-wisdom-1.0 ...)
```

without an explicit `format_version` field. The current reader accepts both formats.

### Sample file

```
(oxifft-wisdom
  (format_version 2)
  (12345678901234567 "ct/radix4/avx2/f64/16" 1.234e-7)
  (9876543210987654 "bluestein/avx2/f64/127" 4.567e-6))
```

The hash field is a decimal `u64`. The solver-name string identifies the algorithm variant
(e.g. Cooley-Tukey radix, SIMD tier, precision, and size). The cost is a non-negative `f64`
representing the measured time in seconds.

### Hash stability

Hashes encode the problem signature (size, precision, direction, stride pattern). Stability
guarantees:

| Condition | Hash stable? |
|-----------|-------------|
| Same OxiFFT minor version, same target, same CPU features | Yes |
| Different OxiFFT minor version (e.g., 0.3.x to 0.4.x) | May change |
| Different target triple (e.g., x86_64 vs aarch64) | Always changes |
| Different SIMD feature set (e.g., AVX2 vs SSE2) | Always changes |

**Wisdom files are not portable across targets.** Generate wisdom on the target machine.

### Merge semantics

When two wisdom caches are merged (`merge_string` / `merge_from_file`), the entry with the
**lower cost** wins for each hash. This allows collecting wisdom from multiple runs and
keeping the best:

- If the hash is absent from the cache, the incoming entry is inserted.
- If the hash is present and the incoming entry has a lower cost, the existing entry is replaced.
- Otherwise the existing entry is kept unchanged.

The `WisdomMergeResult` struct reports `added`, `replaced`, and `kept_existing` counts.

### Version negotiation

| `format_version` | Reader behavior |
|------------------|-----------------|
| 0 (legacy header `oxifft-wisdom-1.0`) | Accepted; entries parsed as version 0 |
| 1 | Accepted. `MixedRadix` was stored as the literal string `"MixedRadix"` (a stub; not reconstructible). |
| 2 (current) | Accepted; full feature set. `MixedRadix` is encoded as `"mixed-radix-R1-R2-..."` and can be reconstructed. |
| > 2 (future) | Returns `WisdomError::IncompatibleVersion { found, expected: 2 }` |

If a current-format file lacks the `(format_version N)` line entirely, OxiFFT treats it
as version 1 (not an error).

### Untrusted-input ceilings

`import_string`/`merge_string` (and `import_from_file`/`merge_from_file`) reject input before
parsing it when either ceiling is exceeded, returning `WisdomError::TooLarge { kind, limit,
actual }`:

| Constant | Default | Checked against |
|----------|---------|------------------|
| `MAX_WISDOM_TEXT_BYTES` | 64 MiB | Total byte length of the input string / file |
| `MAX_WISDOM_TEXT_ENTRIES` | 1,000,000 | Number of entry lines (`(hash "name" cost)`) encountered while parsing |

`import_from_file`/`merge_from_file` check the file's size via `fs::metadata` *before* reading
it into memory, so an oversized file never gets fully buffered.

## Binary format

The binary format is a compact, non-human-readable encoding used by `WisdomCache::to_binary`/
`from_binary`. It is the format embedded at build time by `build.rs` when `OXIFFT_TUNE=1`
(read back via `Plan::dft_1d`'s baseline-wisdom lookup), and is also suitable for transmitting
wisdom between processes without the text format's parsing overhead.

### Header (16 bytes, all fields little-endian)

| Offset | Size | Field | Meaning |
|--------|------|-------|---------|
| 0 | 8 | magic | Literal ASCII bytes `OXIWISDM` |
| 8 | 2 | `format_version` | `1` (legacy) or `2` (current) |
| 10 | 2 | *(version-dependent)* | v1: `entry_count` (u16). v2: reserved, must be `0`. |
| 12 | 4 | *(version-dependent)* | v1: reserved (u32), must be `0`. v2: `entry_count` (u32). |

`from_binary` dispatches on the `format_version` field at offset 8 to select which of the two
layouts below applies to the rest of the blob.

### Version 2 (current) entries — variable length

Each entry is encoded as:

| Field | Size | Meaning |
|-------|------|---------|
| `size_key` | u64 | `= problem_hash`, which (for the baseline-wisdom path) equals the transform size `n` |
| `algo_tag` | u8 | Algorithm discriminant (see table below) |
| `factors_len` | u8 | Number of trailing `u16` mixed-radix factors (`0` unless `algo_tag == MixedRadix`) |
| `factors` | `factors_len` × u16 | Ordered, innermost-first mixed-radix factor list |
| `elapsed_ns` | u64 | `= cost` cast to `u64` (nanoseconds) |

`factors_len` is representable up to 255 (`MAX_BINARY_FACTORS`) — far more than any real
transform size needs (a size requiring 256 radix-2 stages alone would already be `2^256`
elements). `to_binary` caps at this limit defensively; `to_binary_checked` returns
`WisdomError::TooLarge` instead of capping.

### Version 1 (legacy) entries — fixed 30 bytes

Written by OxiFFT ≤ 0.3.x. Still readable for backward compatibility; **no longer written** by
this build (`to_binary`/`to_binary_checked` always emit version 2).

| Field | Size | Meaning |
|-------|------|---------|
| `size_key` | u64 | Same as above |
| `algo_tag` | u8 | Same as above |
| `factors_len` | u8 | Number of *significant* entries in the fixed `factors` array below (`0..=6`) |
| `factors` | `[u16; 6]` | Fixed-size mixed-radix factor slots — **silently truncated any name needing more than 6 factors when this format was written** (fixed in v2; see below) |
| `elapsed_ns` | u64 | Same as above |

30 bytes total (8 + 1 + 1 + 12 + 8). A real, reachable case that overflowed this layout:
`n = 2187 = 3^7` factors into seven radix-3 stages, one more than the fixed array could hold.

### Algorithm tag discriminants

| Value | Algorithm | Value | Algorithm |
|-------|-----------|-------|-----------|
| 0 | CooleyTukey | 6 | Winograd |
| 1 | SplitRadix | 7 | Direct |
| 2 | Stockham | 8 | Generic |
| 3 | Bluestein | 9 | Composite |
| 4 | Rader | 10 | Nop |
| 5 | MixedRadix | 255 | Unknown |

### Semantic validation on decode

`from_binary`/`import_binary` apply the same class of validation the text-format parser does,
for both the v1 and v2 layouts:

- `size_key == 0` → entry rejected (matches the text format's `hash == 0` rule).
- `algo_tag == 255` (Unknown) → entry rejected (no reconstructible algorithm).
- `algo_tag == MixedRadix` → **all** factors must be in `{2,3,4,5,7,8,16}` **and** their
  product must equal `size_key` exactly, or the entry is rejected. This is the check that
  keeps a corrupt/degenerate binary wisdom entry from ever reaching
  `Plan::dft_1d`'s algorithm-reconstruction step, where replaying it would otherwise panic
  (integer divide-by-zero for a zero factor, or an internal `unreachable!()` for an
  unsupported radix).

Rejected entries are silently skipped (not stored), matching the text format's
`skipped_invalid` behavior; `import_binary` reports the count, `from_binary` does not (it
mirrors `WisdomCache::new(); cache.import_binary(data)` but returns just the resulting cache,
for API-compatibility with earlier OxiFFT versions).

### Untrusted-input ceilings

| Constant | Default | Checked against |
|----------|---------|------------------|
| `MAX_WISDOM_BINARY_BYTES` | 64 MiB | Total byte length of the blob |
| `MAX_WISDOM_BINARY_ENTRIES` | 1,000,000 | `entry_count` declared in the header |

Both are checked *before* any per-entry decoding, so a small hostile blob cannot claim an
enormous `entry_count` and force a large bounds-checked scan.

### API

```rust
use oxifft::api::WisdomCache;

let mut cache = WisdomCache::new();
cache.store(oxifft::kernel::WisdomEntry {
    problem_hash: 2187,
    solver_name: "mixed-radix-3-3-3-3-3-3-3".to_string(), // 7 factors — needs v2
    cost: 123.0,
});

// Infallible: caps entries/factors beyond the representable range rather than erroring.
let bytes = cache.to_binary();

// Fallible: returns WisdomError::TooLarge instead of capping.
let bytes = cache.to_binary_checked().expect("cache fits the binary format");

// Structural/semantic errors are reported, not collapsed into `None`.
let restored = WisdomCache::from_binary(&bytes).expect("round-trip must succeed");
assert_eq!(restored.lookup(2187).unwrap().solver_name, "mixed-radix-3-3-3-3-3-3-3");

// Reports imported/skipped counts, like the text-format `import_string`.
let mut merged = WisdomCache::new();
let result = merged.import_binary(&bytes).expect("import must succeed");
assert_eq!(result.imported, 1);
```

## System paths

OxiFFT searches for wisdom in OS-appropriate locations:

| OS | User path | System path |
|----|-----------|-------------|
| Linux | `$XDG_CONFIG_HOME/oxifft/wisdom` or `~/.config/oxifft/wisdom` | `/etc/oxifft/wisdom` |
| macOS | `~/Library/Application Support/oxifft/wisdom` | (none) |
| Windows | `%APPDATA%\oxifft\wisdom` | (none) |

Use `get_user_wisdom_path()` (requires `std`) to get the OS-appropriate path at runtime.

`import_system_wisdom()` returns the *specific* error from the first existing candidate path
that fails to import (corrupt data, incompatible version, oversized file, ...) rather than a
generic "not found" — the generic `WisdomError::IoError(NotFound, ...)` is only returned when
none of the candidate paths exist at all.

## Error types

| Error variant | Description |
|---------------|-------------|
| `WisdomError::ParseError(msg)` | Malformed S-expression syntax, unrecognisable header, bad binary magic/version, or truncated binary data |
| `WisdomError::IncompatibleVersion { found, expected }` | `format_version` is newer than this build supports (text or binary) |
| `WisdomError::IoError(err)` | Filesystem read/write failure (only present with `std` feature) |
| `WisdomError::TooLarge { kind, limit, actual }` | Input exceeded a configured byte/entry/factor ceiling before parsing began |

## API

The wisdom API has two layers: a global cache (free functions) and a standalone
`WisdomCache` struct.

### Global cache (free functions, always available)

```rust
use oxifft::api::{
    export_to_string, import_from_string, merge_from_string,
};

// Export current global cache to a string (works in no_std builds)
let s = export_to_string();

// Import into global cache from a string (works in no_std builds)
let result = import_from_string(&s).expect("parse wisdom");
assert_eq!(result.format_version, 2);

// Merge into global cache — lower cost wins per hash
let merge_result = merge_from_string(&s).expect("merge wisdom");
```

### File I/O (requires `std` feature)

```rust
# #[cfg(feature = "std")] {
use oxifft::api::{import_from_file, export_to_file, merge_from_file};
use std::path::Path;

import_from_file(Path::new("/path/to/wisdom"))?;
export_to_file(Path::new("/path/to/wisdom"))?;
merge_from_file(Path::new("/extra/wisdom"))?;
# }
```

### `WisdomCache` struct (for per-instance or no_std use)

```rust
use oxifft::api::WisdomCache;

// Create an isolated cache
let mut cache = WisdomCache::new();

// Import from string (works in no_std builds)
let result = cache.import_string("(oxifft-wisdom\n  (format_version 2)\n)").unwrap();
assert_eq!(result.format_version, 2);

// Merge another cache's contents into this one
let other = WisdomCache::new();
let _ = cache.merge_string(&other.export_string());

// Export to string
let s = cache.export_string();
assert!(s.contains("oxifft-wisdom"));
```
