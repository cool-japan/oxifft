# oxifft-codegen TODO

**Current Version:** 0.3.0  
**Last Updated:** 2026-04-14

## Version Cross-Reference

| Codegen Phase | Target OxiFFT Version | Status |
|---------------|-----------------------|--------|
| Phase 1: Basic Infrastructure | v0.1.x–v0.2.0 | ✅ Complete |
| Phase 2: Non-Twiddle Codelets (16/32/64) | v0.2.x–v0.3.0 | ✅ Generation complete (2–64 done, tests pending) |
| Phase 3: Twiddle Codelets (radix-8/16/split) | v0.2.x–v0.3.0 | ⚙️ Radix 2/4/8/16 done, split-radix & tests pending |
| Phase 4: Optimization Passes | v0.3.0–v0.4.0 | ✅ Complete (CSE, folding, DCE, strength reduction) |
| Phase 5: SIMD Code Generation | v0.4.0 | 📋 Infrastructure exists, generation logic pending |
| Phase 6: RDFT Code Generation | v0.4.0 | 📋 Pending |
| Phase 7: Testing & Validation | v0.3.0–v0.6.0 | ⚙️ Mostly complete (correctness + accuracy done, code size pending) |
| Future Enhancements | Post-1.0 | 💡 Aspirational |

## Phase 1: Basic Infrastructure (v0.1.x–v0.2.0) ✅ COMPLETE

- [x] Set up proc-macro crate structure
- [x] Implement `gen_notw_codelet!` macro
- [x] Implement `gen_twiddle_codelet!` macro
- [x] Implement `gen_simd_codelet!` macro (infrastructure)
- [x] Implement `gen_dft_codelet!` convenience macro
- [x] Symbolic expression representation (Expr, DAG infrastructure)
- [x] Add comprehensive error messages (defer to v0.3) (planned 2026-04-17)
  - **Goal:** Every proc_macro entrypoint returns `syn::Error::new(span, …).to_compile_error()` with a precise span on bad input; no `panic!`/`unwrap()` reaches compiler output.
  - **Design:** Audit all 5 macros in `lib.rs:43-85` (`gen_notw_codelet`, `gen_twiddle_codelet`, `gen_split_radix_twiddle_codelet`, `gen_simd_codelet`, `gen_dft_codelet`). Use `syn::parse_macro_input!`; convert panics to `syn::Error::new_spanned`. Add `trybuild` dev-dep for compile-fail tests. Error messages include: offending identifier + expected form + nearest valid size.
  - **Files:** `src/lib.rs`, `src/gen_notw.rs`, `src/gen_twiddle.rs`, `src/gen_simd.rs`, `Cargo.toml` (trybuild dev-dep), `tests/compile_fail/` (new), `tests/ui.rs` (new)
  - **Tests:** ≥4 trybuild cases: unknown size, non-literal size, negative size, empty input. `cargo test -p oxifft-codegen` green.
  - **Risk:** trybuild snapshots shift across toolchains; bless with `TRYBUILD=overwrite` on pinned toolchain.

## Phase 2: Non-Twiddle Codelets (target: v0.2.x–v0.3.0)

### Size-2 ✅
- [x] Forward transform
- [x] Backward transform
- [x] Unit tests (pending) [Size-2] (planned 2026-04-17)
  - **Goal:** New file `tests/notw_small_sizes.rs` proves `gen_notw_codelet!` produces numerically-correct output for sizes 2, 4, and 8 vs naïve O(n²) DFT.
  - **Design:** Invoke macro to generate `notw_2_fwd_f32/f64`, `notw_4_*`, `notw_8_*`. Inline naïve DFT reference. Tolerance: f32 ≤1e-5, f64 ≤1e-12 per element.
  - **Files:** `tests/notw_small_sizes.rs` (new), `Cargo.toml` (num-complex dev-dep if missing)
  - **Tests:** 6 tests (3 sizes × 2 precisions). `cargo test -p oxifft-codegen` green.
  - **Risk:** None — macro already exists and emits callable code.

### Size-4 ✅
- [x] Forward transform
- [x] Backward transform
- [x] Unit tests (pending) [Size-4] (planned 2026-04-17) → covered by Size-2 plan block above (all in `tests/notw_small_sizes.rs`)

### Size-8 ✅
- [x] Forward transform (radix-2 DIT with bit-reversal)
- [x] Optimized twiddle application
- [x] Unit tests (pending) [Size-8] (planned 2026-04-17) → covered by Size-2 plan block above (all in `tests/notw_small_sizes.rs`)

### Size-16 ✅
- [x] Full implementation (generation infrastructure)
- [x] Optimization passes application [Size-16] (completed 2026-04-17)
  - **Goal:** Sizes 16, 32, and 64 notw codelet generation walks CSE/strength-reduction/DCE/constant-folding with measurable op-count reduction and numerical equivalence to unoptimized emission.
  - **Design:** Expose `run_default_passes` helper in `symbolic.rs` (constant-fold → CSE → strength-reduce → DCE → constant-fold). Wire into `gen_notw.rs` size-16/32/64 paths. Add `OptimizationStats` struct gated on `OXIFFT_CODEGEN_DEBUG=1`. Test file: `tests/opt_passes_16_64.rs`.
  - **Files:** `src/gen_notw.rs`, `src/symbolic.rs`, `tests/opt_passes_16_64.rs` (new)
  - **Tests:** Numerical (16/32/64 × f32/f64 vs naïve DFT), op-count regression (size-16 strictly fewer ops), proptest for size-16 (256 cases).
  - **Risk:** CSE N log N scaling at size-64; cap proptest at 64 cases for size-64; `#[cfg(not(debug_assertions))]` gate on size-64 property test.
- [x] Unit tests (pending) [Size-16] (completed 2026-04-17) → covered by opt-passes-16 plan block above

### Size-32 ✅
- [x] Full implementation (generation infrastructure)
- [x] Optimization passes application [Size-32] (completed 2026-04-17) → covered by opt-passes-16 plan block above
- [x] Unit tests (pending) [Size-32] (completed 2026-04-17) → covered by opt-passes-16 plan block above

### Size-64 ✅
- [x] Full implementation (generation infrastructure)
- [x] Optimization passes application [Size-64] (completed 2026-04-17) → covered by opt-passes-16 plan block above
- [x] Unit tests (pending) [Size-64] (completed 2026-04-17) → covered by opt-passes-16 plan block above

## Phase 3: Twiddle Codelets (target: v0.2.x–v0.3.0)

- [x] Radix-2 twiddle codelet
- [x] Radix-4 twiddle codelet
- [x] Radix-8 twiddle codelet (implemented)
- [x] Radix-16 twiddle codelet (implemented)
- [x] Split-radix twiddle codelet (generic + specialized 8/16)
- [x] Unit tests for all radixes (112 tests across codegen + oxifft)

## Phase 4: Optimization Infrastructure (target: v0.3.0–v0.4.0)
- [x] Symbolic expression representation (Expr enum, structural hashing)
- [x] Common subexpression elimination (CSE) — fully implemented with CseOptimizer
- [x] Strength reduction — StrengthReducer implemented
- [x] Constant folding — ConstantFolder with fixpoint iteration
- [x] Dead code elimination — DeadCodeEliminator with transitive reachability
- [x] Instruction scheduling hints (planned 2026-04-19)
  - **Goal:** New `instruction_scheduling` optimizer pass in `oxifft-codegen/src/` reorders independent operations by critical-path heuristic (Sethi-Ullman-style) to increase ILP and allow out-of-order hardware to overlap independent chains.
  - **Design:** Build def-use DAG over `Vec<Stmt>`; compute critical-path depth per node via longest-path in topo order; re-emit statements placing depth-0 nodes first, then preferring nodes consumed by the longest remaining critical path; ready-queue populated from unblocked defs.
  - **Files:** `oxifft-codegen/src/` — one of `{symbolic.rs, optimizer.rs, passes.rs}` (orientation determines exact path); `oxifft/src/dft/codelets/codegen_tests.rs` (scheduling-pass snapshot tests).
  - **Tests:** Snapshot test on a constructed input IR (two independent chains) asserts output ordering places long-critical-path nodes first and interleaves independent ops; compile a generated-from-scheduled-pass codelet and run round-trip correctness.
  - **Risk:** Scheduling pass correctness is subtle → unit tests with small graphs validate topological correctness; compiler may already auto-reorder, so measure codegen assembly before/after on a representative size-8 codelet to confirm impact.

## Phase 5: SIMD Code Generation (target: v0.4.0)
**Infrastructure:** ✅ `gen_simd_codelet!` macro exists, placeholder implementation


- [x] Split `gen_simd.rs` into per-ISA submodules (simd-phase5-refactor) (completed 2026-04-17)
  - **Goal:** `oxifft-codegen/src/gen_simd.rs` (1016 lines) refactored into `gen_simd/mod.rs` + `sse2.rs` + `avx2.rs` + `neon.rs` + `scalar.rs`, each under 500 lines. No behavioral change; all existing tests green.
  - **Design:** Use splitrs or manual Read+Write split. Parent `gen_simd/mod.rs` keeps `pub fn generate()` and dispatcher. Submodules own ISA-specific codelet emitters. `gen_simd.rs` deleted (replaced by directory module). `lib.rs` needs no change (Rust resolves `gen_simd/mod.rs` automatically).
  - **Files:** `oxifft-codegen/src/gen_simd/mod.rs` (new), `sse2.rs`, `avx2.rs`, `neon.rs`, `scalar.rs` (all new), `gen_simd.rs` (deleted)
  - **Prerequisites:** none (pure structural refactor)
  - **Tests:** `cargo nextest run -p oxifft-codegen --all-features` green; `rslines 50 oxifft-codegen/src/gen_simd/` all < 500
  - **Risk:** splitrs may struggle with macro-heavy content. Mitigation: manual split fallback.
- [x] Add f32 codelets for SSE2/AVX2/NEON sizes 2/4/8 (simd-phase5-f32-expand) (completed 2026-04-17)
  - **Goal:** Each ISA submodule gains f32 codelet emission for sizes 2/4/8 (tolerance 1e-5 vs scalar). Dispatcher upgraded to route `T = f32` to SIMD paths (f32x4 for SSE2/NEON, f32x8 for AVX2).
  - **Design:** Three-arm dispatcher: `T=f64` → existing f64 SIMD; `T=f32` → new f32 SIMD; else → scalar. f32 emitters mirror f64 structure with halved lane width × doubled: SSE2 `__m128`, AVX2 `__m256`, NEON `float32x4_t`. Intrinsics: `_mm_add_pd` → `_mm_add_ps`, `vaddq_f64` → `vaddq_f32`, etc. Twiddle constants in f32 precision.
  - **Files:** `gen_simd/mod.rs` (dispatcher), `sse2.rs` (+f32), `avx2.rs` (+f32), `neon.rs` (+f32), `tests/simd_f32_parity.rs` (new)
  - **Prerequisites:** item 2a (simd-phase5-refactor) must complete first
  - **Tests:** `simd_f32_parity.rs`: 6 tests (3 sizes × fwd/inv) on NEON (Apple Silicon host); compile-only gate for x86 ISAs
  - **Risk:** AVX2 shuffle arithmetic. Mitigation: parity test at 1e-5 vs scalar catches shuffles immediately.

### SSE2 (x86_64)
- [x] 2-lane f64 codelets
- [x] 4-lane f32 codelets

### AVX (x86_64)
- [x] 4-lane f64 codelets (planned 2026-04-19)
  - **Goal:** New pure-AVX (non-AVX2, non-FMA) emitter `gen_simd/avx.rs` for sizes 2, 4, 8 on f64, using only AVX (non-FMA) instructions and gating each function with `#[target_feature(enable = "avx")]`.
  - **Design:** Mirror `gen_avx2_size_{2,4,8}` structure using only `_mm256_add_pd` / `_mm256_sub_pd` / `_mm256_mul_pd` / `_mm256_loadu_pd` / `_mm256_storeu_pd` / `_mm256_permute_pd` / `_mm256_permute2f128_pd`; complex multiply emits as `sub(mul(a,c), mul(b,d))` + `add(mul(a,d), mul(b,c))` (two-deep tree without FMA); dispatcher updated to probe `Avx512f > Avx2Fma > Avx > Sse2`.
  - **Files:** `oxifft-codegen-impl/src/gen_simd/avx.rs` (new); `oxifft-codegen-impl/src/gen_simd/mod.rs` (dispatcher arm for `SimdLevel::Avx`); `oxifft/src/dft/codelets/codegen_tests.rs` (runtime-gated AVX tests).
  - **Tests:** Runtime-gated on `is_x86_feature_detected!("avx")`; round-trip parity vs scalar reference for sizes 2/4/8 with tolerance `rel 1e-12`.
  - **Risk:** Pure-AVX without FMA has different rounding from AVX2/FMA paths → tolerance vs scalar is `rel 1e-12`, not bit-exact; confirm probe order ensures AVX path is not shadowed by AVX2 on AVX2-capable machines.
- [x] 8-lane f32 codelets

### AVX2 (x86_64)
- [x] FMA-optimized codelets (audited: no add+mul chains to fuse; existing mul-for-scale is optimal)
- [x] Improved shuffles (planned 2026-04-19)
  - **Goal:** AVX2 shuffle audit collapses redundant permute+unpack sequences into single-instruction equivalents across all `gen_avx2_size_*` functions, reducing instruction count and latency documented via Intel Intrinsics Guide references.
  - **Design:** Walk each `gen_avx2_size_*` function; collapse: `permute_pd(unpacklo_pd(a,b), 0x0)` → `unpacklo_pd(a,b)`; `permute_pd(a,0x5)` + `blend_pd(a,b,0xA)` → `unpackhi_pd(a,b)` where equivalent; `_mm256_permute2f128_pd(a,b,0x20)` replaces verbose `insertf128_pd` chains; each collapse annotated with latency comment (e.g., `permute_pd=1c, permute2f128_pd=3c on Haswell`).
  - **Files:** `oxifft-codegen-impl/src/gen_simd/avx2.rs` (shuffle audit edits); `oxifft/src/dft/codelets/codegen_tests.rs` (AVX2 shuffle-rewrite parity tests).
  - **Tests:** Pre-rewrite and post-rewrite emit bit-exact output on fixed test inputs (shuffles don't change rounding); existing AVX2 round-trip correctness tests remain green.
  - **Risk:** Incorrect shuffle substitution silently produces wrong output → parity tests catch this immediately since shuffles are deterministic and bit-exact.

### AVX-512 (x86_64)
- [x] 8-lane f64 codelets (sizes 2, 4, 8 via gen_simd/avx512.rs)
- [x] 16-lane f32 codelets (size 16 f32 via gen_avx512_size_16_f32)

### NEON (aarch64)
- [x] 2-lane f64 codelets
- [x] 4-lane f32 codelets

## Phase 6: RDFT Codelets (target: v0.4.0)

- [x] R2HC (Real to Half-Complex) codelets (completed 2026-04-17)
  - **Goal:** New `gen_rdft_codelet!(size=N, kind=R2hc|Hc2r, ty=f32|f64)` proc-macro emits code numerically equivalent (f64 ≤1e-12, f32 ≤1e-5) to hand-written `r2hc_2/4/8` and `hc2r_2/4/8` in `oxifft/src/rdft/codelets/mod.rs`.
  - **Design:** New module `src/gen_rdft.rs` (~400–600 lines). Walks Expr DAG for real-input split-complex butterfly (R2HC output layout: r₀,…,r_{N/2}, i_{N/2-1},…,i₁). Reuse `optimize_default` from opt-passes item. Pub macro in `lib.rs` with span-based errors. Scope: sizes 2/4/8. Do NOT swap R2C/C2R solver dispatch yet — parity validation first.
  - **Files:** `src/gen_rdft.rs` (new), `src/lib.rs` (+ pub fn gen_rdft_codelet), `src/symbolic.rs` (if new Op/helper needed), `tests/rdft_codelets.rs` (new), `oxifft/src/rdft/codelets/codegen_tests.rs` (new)
  - **Tests:** 12 numerical tests (3 sizes × 2 kinds × 2 precisions) vs hand-written; macro expansion compile tests.
  - **Risk:** R2HC output layout mismatch — subagent reads hand-written codelets first to derive exact layout.
- [x] HC2R (Half-Complex to Real) codelets (completed 2026-04-17) → covered by R2HC plan block above (HC2R is the second kind in gen_rdft_codelet!)

**Priority:** High for v0.3.0 release

- [x] Correctness tests vs reference implementation (DFT comparison for sizes 2-64)
- [x] Numerical accuracy tests (epsilon tolerance checks)
- [x] Performance benchmarks (vs hand-written equivalents) (completed 2026-04-17)
  - **Goal:** Criterion bench `benches/codelet_perf.rs` measures generated vs hand-written codelets side-by-side for sizes 2/4/8/16/32/64 × f32/f64.
  - **Design:** Add `[[bench]] name="codelet_perf" harness=false` to `Cargo.toml`. `criterion.workspace = true` + `oxifft.workspace = true` in dev-deps. If hand-written notw functions are `pub(crate)`, expose via a `codegen-bench` feature in `oxifft/Cargo.toml`.
  - **Files:** `benches/codelet_perf.rs` (new), `Cargo.toml` ([[bench]] entry + criterion dev-dep)
  - **Tests:** `cargo bench -p oxifft-codegen --bench codelet_perf --no-run` succeeds (compile-only gate).
  - **Risk:** hand-written codelets may not be pub; use `codegen-bench` feature-gate if needed.
- [x] Compilation time benchmarks (planned 2026-04-17)
  - **Goal:** New `oxifft-codegen/benches/codegen_time.rs` measures wall-clock cost of generating each codelet (calling `generate()` from each `gen_*.rs`). Criterion bench, compile-only gate in CI.
  - **Design:** Groups: gen_notw (sizes 2,4,8,16,32,64), gen_twiddle (radixes 2,4,8,16), gen_rdft_r2hc/hc2r (sizes 2,4,8), gen_simd (size 8 × SSE2/AVX2/NEON f64). Helper `make_size_input(n) -> TokenStream { quote!{#n} }`. Warm-up 1s, measurement 5s. Add `[[bench]] name="codegen_time" harness=false` + `criterion.workspace = true` to `Cargo.toml`.
  - **Files:** `oxifft-codegen/benches/codegen_time.rs` (new), `oxifft-codegen/Cargo.toml`
  - **Prerequisites:** none
  - **Tests:** `cargo bench -p oxifft-codegen --bench codegen_time --no-run` succeeds
  - **Risk:** TokenStream construction for bench. Mitigation: quote! helper.
- [x] Code size analysis (planned 2026-04-17)
  - **Goal:** `oxifft-codegen/examples/code_size_report.rs` binary prints per-codelet table of token count, LoC, and Op count. Not a criterion bench; `cargo run -p oxifft-codegen --example code_size_report` exits 0.
  - **Design:** Token count: `generate(input).into_iter().count()`. LoC: `prettyplease::unparse(…).lines().count()` or `tokens.to_string().lines().count()` fallback. Op count: promote `symbolic::OptimizationStats` from `pub(crate)` → `pub`. Output: stdout tab-separated `kind | size | tokens | lines | ops`.
  - **Files:** `oxifft-codegen/examples/code_size_report.rs` (new), `oxifft-codegen/Cargo.toml` (prettyplease dev-dep if workspace-pinned), `oxifft-codegen/src/symbolic.rs` (promote OptimizationStats visibility)
  - **Prerequisites:** none
  - **Tests:** `cargo run -p oxifft-codegen --example code_size_report` runs, prints, exits 0
  - **Risk:** prettyplease deps. Mitigation: coarse proxy fallback.
- [x] Integration tests with oxifft main crate (codegen_tests.rs)

## Future Enhancements (Post-1.0)

💡 Aspirational features for mature releases

- [x] Support for odd sizes (3, 5, 7, etc.)
- [x] Prime-size direct codelets (Rader's algorithm codegen)
- [x] Vectorized multi-transform codelets (batch processing)
- [x] Custom codelet generation API for user-defined sizes (planned 2026-05-01)
  - **Goal:** New gen_any_codelet!(N, ty=f32|f64, dir=Forward|Backward) proc-macro + CodeletBuilder Rust API; dispatches to best emitter for arbitrary N: notw for {2,3,4,5,7,8,11,13,16,32,64}, MixedRadix for smooth-7 composites, Rader for primes ≤ 1021, Bluestein wrapper otherwise.
  - **Design:** classify(n) → NotwSmall|WinogradOdd|RaderPrime|MixedRadix(facs)|Bluestein; CodeletBuilder::new(N).precision(F32).direction(Forward).build() → Result<TokenStream,CodegenError>; Bluestein wrapper emits chirp pre/post tables as static const arrays at codegen time.
  - **Files:** gen_any.rs (extend stub, ~500 LoC), oxifft-codegen/src/lib.rs (add gen_any_codelet!), oxifft-codegen-impl/src/lib.rs (pub use gen_any::CodeletBuilder)
  - **Prerequisites:** Mixed-radix item done (gen_mixed_radix::generate_for exists for composite N).
  - **Tests:** generate(8) → notw; generate(15) → MixedRadix [5,3]; generate(11) → Rader; generate(2003) → Bluestein; end-to-end correctness vs naive DFT for n ∈ {6,10,11,13,15,23,31,60,100,256}; CodeletBuilder::new(0).build() → Err.
  - **Risk:** n=0/n=1 edge cases → Err(InvalidSize); Bluestein chirp tables >1MB → codegen-time warning.
- [x] Runtime codelet selection based on CPU features
- [x] Mixed-radix optimization strategies (planned 2026-05-01)
  - **Goal:** True mixed-radix Cooley-Tukey for sizes factoring into {2,3,4,5,7,8,16} (smooth-7 family: 6,10,12,14,24,28,40,56,80,96,112,240,…) — get hand-tuned codelet path instead of Bluestein.
  - **Design:** Radix-3/5/7 DIT twiddle emitters in gen_twiddle.rs using Winograd min-multiply; runtime wrappers in twiddle_odd.rs; twiddles_mixed_radix<T> generator; Algorithm::MixedRadix executor with mixed-radix digit reversal + per-stage DIT butterfly; planner cost model (n·log₂(n)·per-radix-mul-ratio); wisdom format v1→v2 (backward-compat S-expr; v1 files load fine — just lack MixedRadix entries). Greedy radix peel from largest: n=240 → [16,5,3].
  - **Files:** gen_twiddle.rs (+400 LoC), gen_mixed_radix.rs (new ~250 LoC), twiddle_odd.rs (new ~350 LoC), types.rs (MixedRadix execute arm + select_algorithm branch), planner.rs (MixedRadix arm in SolverChoice + estimate_cost + wisdom S-expr encode/decode), twiddle.rs (twiddles_mixed_radix), wisdom.rs (WISDOM_FORMAT_VERSION 1→2 + compat read)
  - **Tests:** Radix-{3,5,7} butterfly DIT vs naive for blocks ∈ {1,4,16}, stride ∈ {1,r,4r}; end-to-end N ∈ {6,10,12,14,15,21,24,28,30,35,40,42,48,56,60,80,84,96,112,120,168,240} parity vs Bluestein ≤1e-10; Plan::dft_1d(40).algorithm_name()=="MixedRadix"; wisdom v1↔v2 round-trip.
  - **Risk:** Twiddle ordering subtlety (validate small sizes {6,10,14} first); planner.rs may hit ~1860 lines (split in item 4 if >1900).

---

## Developer Notes

**Current State (v0.3.0):**
- All 4 public proc macros functional
- Generation logic for sizes 2–64 and radixes 2/4/8/16 + split-radix implemented
- Full optimization pipeline: CSE, constant folding, strength reduction, dead code elimination
- 112 tests across codegen + oxifft crates, correctness verified against DFT reference
- Integration tests via codegen_tests.rs in oxifft main crate

**Next Milestone (v0.4.0):**
- SIMD code generation (SSE2, AVX, NEON)
- RDFT codelet generation
- Code size analysis and compilation time benchmarks
- Comprehensive error messages

**Build Commands:**
```bash
cargo build -p oxifft-codegen          # Build proc-macro
cargo test -p oxifft-codegen           # Run tests
cargo doc -p oxifft-codegen --open     # Generate docs
```
