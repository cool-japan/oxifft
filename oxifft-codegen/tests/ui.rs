/// Compile-fail tests for `oxifft-codegen` proc-macro error diagnostics.
///
/// Run `TRYBUILD=overwrite cargo test -p oxifft-codegen ui` to update .stderr snapshots.
#[test]
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
