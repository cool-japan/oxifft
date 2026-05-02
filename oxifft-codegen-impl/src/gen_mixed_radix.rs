// Mixed-radix codegen — item 1 (codegen-mixed-radix-strategies) landing notes
//
// The core mixed-radix DIT FFT is now implemented in the runtime executor:
//   - Butterfly functions: oxifft/src/dft/codelets/twiddle_odd.rs
//   - Twiddle generation:  oxifft/src/kernel/twiddle.rs  (twiddles_mixed_radix)
//   - Executor + permute:  oxifft/src/api/plan/types.rs   (execute_mixed_radix_inplace)
//   - Planner heuristic:   oxifft/src/kernel/planner.rs  (select_solver_heuristic)
//
// This module is reserved for future proc-macro-generated specialized codelets
// (e.g., unrolled mixed-radix-6, mixed-radix-10, mixed-radix-14 kernels with
// compile-time-known butterfly sequences).  The runtime engine is sufficient
// for all smooth-7 sizes with radices {2, 3, 4, 5, 7, 8, 16}.

#[allow(dead_code)]
pub(crate) const _IMPL_COMPLETE: () = ();
