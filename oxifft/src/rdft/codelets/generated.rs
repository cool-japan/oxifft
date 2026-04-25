//! Generated RDFT codelets for production use.
//!
//! These are the same codelets validated in `codegen_tests.rs` but exposed
//! at crate scope so the solver pipeline can dispatch to them.
use oxifft_codegen::gen_rdft_codelet;

gen_rdft_codelet!(size = 2, kind = R2hc);
gen_rdft_codelet!(size = 4, kind = R2hc);
gen_rdft_codelet!(size = 8, kind = R2hc);
gen_rdft_codelet!(size = 2, kind = Hc2r);
gen_rdft_codelet!(size = 4, kind = Hc2r);
gen_rdft_codelet!(size = 8, kind = Hc2r);
