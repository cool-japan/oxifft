# oxifft-codegen TODO

## Phase 1: Basic Infrastructure

- [x] Set up proc-macro crate structure
- [x] Implement `gen_notw_codelet!` macro
- [x] Implement `gen_twiddle_codelet!` macro
- [x] Implement `gen_simd_codelet!` macro (placeholder)
- [ ] Add comprehensive error messages

## Phase 2: Non-Twiddle Codelets

### Size-2
- [x] Forward transform
- [x] Backward transform
- [ ] Unit tests

### Size-4
- [x] Forward transform
- [x] Backward transform
- [ ] Unit tests

### Size-8
- [x] Forward transform (basic)
- [ ] Optimized twiddle application
- [ ] Unit tests

### Size-16
- [ ] Full implementation
- [ ] Optimization passes
- [ ] Unit tests

### Size-32
- [ ] Full implementation
- [ ] Optimization passes
- [ ] Unit tests

### Size-64
- [ ] Full implementation
- [ ] Optimization passes
- [ ] Unit tests

## Phase 3: Twiddle Codelets

- [x] Radix-2 twiddle codelet
- [x] Radix-4 twiddle codelet
- [ ] Radix-8 twiddle codelet
- [ ] Radix-16 twiddle codelet
- [ ] Split-radix twiddle codelet

## Phase 4: Optimization Infrastructure

- [ ] Symbolic expression representation
- [ ] Common subexpression elimination (CSE)
- [ ] Strength reduction
- [ ] Constant folding
- [ ] Dead code elimination
- [ ] Instruction scheduling hints

## Phase 5: SIMD Code Generation

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

## Phase 6: RDFT Codelets

- [ ] R2HC (Real to Half-Complex) codelets
- [ ] HC2R (Half-Complex to Real) codelets
- [ ] Real-valued twiddle codelets

## Phase 7: Testing & Validation

- [ ] Correctness tests vs reference implementation
- [ ] Numerical accuracy tests
- [ ] Performance benchmarks
- [ ] Compilation time benchmarks
- [ ] Code size analysis

## Future Enhancements

- [ ] Support for odd sizes (3, 5, 7, etc.)
- [ ] Prime-size direct codelets
- [ ] Vectorized multi-transform codelets
- [ ] Custom codelet generation API
