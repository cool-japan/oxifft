# oxifft-codegen TODO

**Current Version:** 0.2.0  
**Last Updated:** 2026-04-14

## Version Cross-Reference

| Codegen Phase | Target OxiFFT Version | Status |
|---------------|-----------------------|--------|
| Phase 1: Basic Infrastructure | v0.1.x–v0.2.0 | ✅ Complete |
| Phase 2: Non-Twiddle Codelets (16/32/64) | v0.2.x–v0.3.0 | ⚙️ In progress (16/32/64 gen done, tests pending) |
| Phase 3: Twiddle Codelets (radix-8/16/split) | v0.2.x–v0.3.0 | ⚙️ In progress (2/4/8/16 done, tests pending) |
| Phase 4: Optimization Passes | v0.3.0–v0.4.0 | ⚙️ Infrastructure ready (symbolic module), application pending |
| Phase 5: SIMD Code Generation | v0.4.0 | 📋 Infrastructure exists, generation logic pending |
| Phase 6: RDFT Code Generation | v0.4.0 | 📋 Pending |
| Phase 7: Testing & Validation | v0.3.0–v0.6.0 | 📋 Pending |
| Future Enhancements | Post-1.0 | 💡 Aspirational |

## Phase 1: Basic Infrastructure (v0.1.x–v0.2.0) ✅ COMPLETE

- [x] Set up proc-macro crate structure
- [x] Implement `gen_notw_codelet!` macro
- [x] Implement `gen_twiddle_codelet!` macro
- [x] Implement `gen_simd_codelet!` macro (infrastructure)
- [x] Implement `gen_dft_codelet!` convenience macro
- [x] Symbolic expression representation (Expr, DAG infrastructure)
- [ ] Add comprehensive error messages (defer to v0.3)

## Phase 2: Non-Twiddle Codelets (target: v0.2.x–v0.3.0)

### Size-2 ✅
- [x] Forward transform
- [x] Backward transform
- [ ] Unit tests (pending)

### Size-4 ✅
- [x] Forward transform
- [x] Backward transform
- [ ] Unit tests (pending)

### Size-8 ✅
- [x] Forward transform (radix-2 DIT with bit-reversal)
- [x] Optimized twiddle application
- [ ] Unit tests (pending)

### Size-16 ✅
- [x] Full implementation (generation infrastructure)
- [ ] Optimization passes application
- [ ] Unit tests (pending)

### Size-32 ✅
- [x] Full implementation (generation infrastructure)
- [ ] Optimization passes application
- [ ] Unit tests (pending)

### Size-64 ✅
- [x] Full implementation (generation infrastructure)
- [ ] Optimization passes application
- [ ] Unit tests (pending)

## Phase 3: Twiddle Codelets (target: v0.2.x–v0.3.0)

- [x] Radix-2 twiddle codelet
- [x] Radix-4 twiddle codelet
- [x] Radix-8 twiddle codelet (implemented)
- [x] Radix-16 twiddle codelet (implemented)
- [ ] Split-radix twiddle codelet
- [ ] Unit tests for all radixes

## Phase 4: Optimization Infrastructure (target: v0.3.0–v0.4.0)
x] Symbolic expression representation (Expr enum, structural hashing)
- [ ] Common subexpression elimination (CSE) — infrastructure exists, needs application
- [ ] Strength reduction — planned
- [ ] Constant folding — planned
- [ ] Dead code elimination — planned
- [ ] Instruction scheduling hints — planned
- [ ] Instruction scheduling hints

## Phase 5: SIMD Code Generation (target: v0.4.0)
**Infrastructure:** ✅ `gen_simd_codelet!` macro exists, placeholder implementation


### SSE2 (x86_64)
- [ ] 2-lane f64 codelets
- [ ] 4-lane f32 codelets

### AVX (x86_64)
- [ ] 4-lane f64 codelets
- [ ] 8-lane f32 codelets

### AVX2 (x86_64)
- [ ] FMA-optimized codelets
- [ ] Improved shuffles

### AVX-512 (x86_64)
- [ ] 8-lane f64 codelets
- [ ] 16-lane f32 codelets

### NEON (aarch64)
- [ ] 2-lane f64 codelets
- [ ] 4-lane f32 codelets

## Phase 6: RDFT Codelets (target: v0.4.0)

- [ ] R2HC (Real to Half-Complex) codelets
- [ ] HC2R (Half-Complex to Real) codelets3.0–v0.6.0)

**Priority:** High for v0.3.0 release

- [ ] Correctness tests vs reference implementation
- [ ] Numerical accuracy tests
- [ ] Performance benchmarks (vs hand-written equivalents)
- [ ] Compilation time benchmarks
- [ ] Code size analysis
- [ ] Integration tests with oxifft main crate

## Future Enhancements (Post-1.0)

💡 Aspirational features for mature releases

- [ ] Support for odd sizes (3, 5, 7, etc.)
- [ ] Prime-size direct codelets (Rader's algorithm codegen)
- [ ] Vectorized multi-transform codelets (batch processing)
- [ ] Custom codelet generation API for user-defined sizes
- [ ] Runtime codelet selection based on CPU features
- [ ] Mixed-radix optimization strategies

---

## Developer Notes

**Current State (v0.2.0):**
- All 4 public proc macros functional
- Generation logic for sizes 2–64 and radixes 2/4/8/16 implemented
- Symbolic optimization infrastructure exists but not fully integrated
- Test coverage is minimal — top priority for v0.3.0

**Next Milestone (v0.3.0):**
- Complete unit tests for all generated codelets
- Apply optimization passes to remaining sizes
- Implement split-radix twiddle codelet
- Comprehensive error messages

**Build Commands:**
```bash
cargo build -p oxifft-codegen          # Build proc-macro
cargo test -p oxifft-codegen           # Run tests
cargo doc -p oxifft-codegen --open     # Generate docs
```7, etc.)
- [ ] Prime-size direct codelets
- [ ] Vectorized multi-transform codelets
- [ ] Custom codelet generation API
