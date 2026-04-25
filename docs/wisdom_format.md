# OxiFFT Wisdom File Format Specification

Version: 1 (`WISDOM_FORMAT_VERSION = 1`)

## Overview

OxiFFT wisdom files store plan timing measurements in an S-expression format.
When loaded, the planner skips benchmarking and reuses previously measured solver selections.

## Grammar (EBNF)

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

### Legacy format (version 0)

Files written by OxiFFT v0.1.x use a legacy header:

```
(oxifft-wisdom-1.0 ...)
```

without an explicit `format_version` field. The current reader accepts both formats.

## Sample file

```
(oxifft-wisdom
  (format_version 1)
  (12345678901234567 "ct/radix4/avx2/f64/16" 1.234e-7)
  (9876543210987654 "bluestein/avx2/f64/127" 4.567e-6))
```

The hash field is a decimal `u64`. The solver-name string identifies the algorithm variant
(e.g. Cooley-Tukey radix, SIMD tier, precision, and size). The cost is a non-negative `f64`
representing the measured time in seconds.

## Hash stability

Hashes encode the problem signature (size, precision, direction, stride pattern). Stability
guarantees:

| Condition | Hash stable? |
|-----------|-------------|
| Same OxiFFT minor version, same target, same CPU features | Yes |
| Different OxiFFT minor version (e.g., 0.3.x to 0.4.x) | May change |
| Different target triple (e.g., x86_64 vs aarch64) | Always changes |
| Different SIMD feature set (e.g., AVX2 vs SSE2) | Always changes |

**Wisdom files are not portable across targets.** Generate wisdom on the target machine.

## Merge semantics

When two wisdom caches are merged (`merge_string` / `merge_from_file`), the entry with the
**lower cost** wins for each hash. This allows collecting wisdom from multiple runs and
keeping the best:

- If the hash is absent from the cache, the incoming entry is inserted.
- If the hash is present and the incoming entry has a lower cost, the existing entry is replaced.
- Otherwise the existing entry is kept unchanged.

The `WisdomMergeResult` struct reports `added`, `replaced`, and `kept_existing` counts.

## Version negotiation

| `format_version` | Reader behavior |
|------------------|-----------------|
| 0 (legacy header `oxifft-wisdom-1.0`) | Accepted; entries parsed as version 0 |
| 1 (current) | Accepted; full feature set |
| > 1 (future) | Returns `WisdomError::IncompatibleVersion { found, expected: 1 }` |

If a current-format file lacks the `(format_version N)` line entirely, OxiFFT treats it
as version 1 (not an error).

## System paths

OxiFFT searches for wisdom in OS-appropriate locations:

| OS | User path | System path |
|----|-----------|-------------|
| Linux | `$XDG_CONFIG_HOME/oxifft/wisdom` or `~/.config/oxifft/wisdom` | `/etc/oxifft/wisdom` |
| macOS | `~/Library/Application Support/oxifft/wisdom` | (none) |
| Windows | `%APPDATA%\oxifft\wisdom` | (none) |

Use `get_user_wisdom_path()` (requires `std`) to get the OS-appropriate path at runtime.

## Error types

| Error variant | Description |
|---------------|-------------|
| `WisdomError::ParseError(msg)` | Malformed S-expression syntax or unrecognisable header |
| `WisdomError::IncompatibleVersion { found, expected }` | `format_version` is newer than this build supports |
| `WisdomError::IoError(err)` | Filesystem read/write failure (only present with `std` feature) |

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
assert_eq!(result.format_version, 1);

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
let result = cache.import_string("(oxifft-wisdom\n  (format_version 1)\n)").unwrap();
assert_eq!(result.format_version, 1);

// Merge another cache's contents into this one
let other = WisdomCache::new();
let _ = cache.merge_string(&other.export_string());

// Export to string
let s = cache.export_string();
assert!(s.contains("oxifft-wisdom"));
```
