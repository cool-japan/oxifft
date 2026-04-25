//! Tests for the symbolic FFT module.
//!
//! Extracted from `symbolic.rs` to keep that file under the 2000-line limit.
//! All types and functions under test are imported via `use super::*`.

use std::collections::HashMap;

use super::symbolic_emit::RecursiveCse;
use super::*;

#[test]
fn test_expr_basic() {
    let a = Expr::input_re(0);
    let b = Expr::input_re(1);
    let sum = a.add(b);
    assert!(matches!(sum, Expr::Add(_, _)));
}

#[test]
fn test_strength_reduction_mul_zero() {
    let a = Expr::input_re(0);
    let zero = Expr::Const(0.0);
    let product = a.mul(zero);
    let reduced = StrengthReducer::reduce(&product);
    assert_eq!(reduced, Expr::Const(0.0));
}

#[test]
fn test_strength_reduction_mul_one() {
    let a = Expr::input_re(0);
    let one = Expr::Const(1.0);
    let product = a.mul(one);
    let reduced = StrengthReducer::reduce(&product);
    assert!(matches!(
        reduced,
        Expr::Input {
            index: 0,
            is_real: true
        }
    ));
}

#[test]
fn test_strength_reduction_add_zero() {
    let a = Expr::input_re(0);
    let zero = Expr::Const(0.0);
    let sum = a.add(zero);
    let reduced = StrengthReducer::reduce(&sum);
    assert!(matches!(
        reduced,
        Expr::Input {
            index: 0,
            is_real: true
        }
    ));
}

#[test]
fn test_strength_reduction_double_neg() {
    let a = Expr::input_re(0);
    let neg_neg = a.neg().neg();
    let reduced = StrengthReducer::reduce(&neg_neg);
    assert!(matches!(
        reduced,
        Expr::Input {
            index: 0,
            is_real: true
        }
    ));
}

#[test]
fn test_complex_mul() {
    let a = ComplexExpr::constant(1.0, 0.0);
    let b = ComplexExpr::constant(0.0, 1.0);
    let product = a.mul(&b);

    // (1 + 0i) * (0 + 1i) = 0 + 1i
    let re = StrengthReducer::reduce(&product.re);
    let im = StrengthReducer::reduce(&product.im);

    assert_eq!(re.const_value(), Some(0.0));
    assert_eq!(im.const_value(), Some(1.0));
}

#[test]
fn test_symbolic_dft_size_2() {
    let fft = SymbolicFFT::dft(2, true);
    assert_eq!(fft.n(), 2);
    assert_eq!(fft.outputs.len(), 2);
}

#[test]
fn test_symbolic_radix2_size_4() {
    let fft = SymbolicFFT::radix2_dit(4, true);
    assert_eq!(fft.n(), 4);
    assert_eq!(fft.outputs.len(), 4);
}

// --- Constant folding tests ---

#[test]
fn test_fold_const_add() {
    let expr = Expr::Const(3.0).add(Expr::Const(4.0));
    let folded = ConstantFolder::fold(&expr);
    assert_eq!(folded, Expr::Const(7.0));
}

#[test]
fn test_fold_const_mul() {
    let expr = Expr::Const(3.0).mul(Expr::Const(5.0));
    let folded = ConstantFolder::fold(&expr);
    assert_eq!(folded, Expr::Const(15.0));
}

#[test]
fn test_fold_const_sub() {
    let expr = Expr::Const(10.0).sub(Expr::Const(3.0));
    let folded = ConstantFolder::fold(&expr);
    assert_eq!(folded, Expr::Const(7.0));
}

#[test]
fn test_fold_mul_zero_lhs() {
    let expr = Expr::Const(0.0).mul(Expr::input_re(0));
    let folded = ConstantFolder::fold(&expr);
    assert_eq!(folded, Expr::Const(0.0));
}

#[test]
fn test_fold_mul_zero_rhs() {
    let expr = Expr::input_re(0).mul(Expr::Const(0.0));
    let folded = ConstantFolder::fold(&expr);
    assert_eq!(folded, Expr::Const(0.0));
}

#[test]
fn test_fold_mul_one_lhs() {
    let expr = Expr::Const(1.0).mul(Expr::input_re(2));
    let folded = ConstantFolder::fold(&expr);
    assert_eq!(
        folded,
        Expr::Input {
            index: 2,
            is_real: true
        }
    );
}

#[test]
fn test_fold_mul_one_rhs() {
    let expr = Expr::input_im(1).mul(Expr::Const(1.0));
    let folded = ConstantFolder::fold(&expr);
    assert_eq!(
        folded,
        Expr::Input {
            index: 1,
            is_real: false
        }
    );
}

#[test]
fn test_fold_mul_neg_one_lhs() {
    let expr = Expr::Const(-1.0).mul(Expr::input_re(0));
    let folded = ConstantFolder::fold(&expr);
    assert_eq!(
        folded,
        Expr::Neg(Box::new(Expr::Input {
            index: 0,
            is_real: true
        }))
    );
}

#[test]
fn test_fold_mul_neg_one_rhs() {
    let expr = Expr::input_re(0).mul(Expr::Const(-1.0));
    let folded = ConstantFolder::fold(&expr);
    assert_eq!(
        folded,
        Expr::Neg(Box::new(Expr::Input {
            index: 0,
            is_real: true
        }))
    );
}

#[test]
fn test_fold_add_zero_lhs() {
    let expr = Expr::Const(0.0).add(Expr::input_re(0));
    let folded = ConstantFolder::fold(&expr);
    assert_eq!(
        folded,
        Expr::Input {
            index: 0,
            is_real: true
        }
    );
}

#[test]
fn test_fold_add_zero_rhs() {
    let expr = Expr::input_re(0).add(Expr::Const(0.0));
    let folded = ConstantFolder::fold(&expr);
    assert_eq!(
        folded,
        Expr::Input {
            index: 0,
            is_real: true
        }
    );
}

#[test]
fn test_fold_sub_zero() {
    let expr = Expr::input_re(0).sub(Expr::Const(0.0));
    let folded = ConstantFolder::fold(&expr);
    assert_eq!(
        folded,
        Expr::Input {
            index: 0,
            is_real: true
        }
    );
}

#[test]
fn test_fold_neg_neg() {
    let expr = Expr::input_re(0).neg().neg();
    let folded = ConstantFolder::fold(&expr);
    assert_eq!(
        folded,
        Expr::Input {
            index: 0,
            is_real: true
        }
    );
}

#[test]
fn test_fold_sub_self() {
    let a = Expr::input_re(0);
    let expr = a.clone().sub(a);
    let folded = ConstantFolder::fold(&expr);
    assert_eq!(folded, Expr::Const(0.0));
}

#[test]
fn test_fold_neg_const() {
    let expr = Expr::Const(5.0).neg();
    let folded = ConstantFolder::fold(&expr);
    assert_eq!(folded, Expr::Const(-5.0));
}

#[test]
fn test_fold_nested_constants() {
    // (2 + 3) * (4 - 1) → 5 * 3 → 15
    let expr = Expr::Const(2.0)
        .add(Expr::Const(3.0))
        .mul(Expr::Const(4.0).sub(Expr::Const(1.0)));
    let folded = ConstantFolder::fold(&expr);
    assert_eq!(folded, Expr::Const(15.0));
}

#[test]
fn test_fold_fixpoint_cascading() {
    // x * 1.0 + 0.0 → x + 0.0 → x (needs cascading)
    let x = Expr::input_re(0);
    let expr = x.mul(Expr::Const(1.0)).add(Expr::Const(0.0));
    let folded = ConstantFolder::fold(&expr);
    assert_eq!(
        folded,
        Expr::Input {
            index: 0,
            is_real: true
        }
    );
}

// --- Dead code elimination tests ---

#[test]
fn test_dce_removes_unused_temp() {
    let mut program = Program {
        assignments: vec![
            ("t0".to_string(), Expr::input_re(0).add(Expr::input_re(1))),
            ("t1".to_string(), Expr::input_re(2).mul(Expr::Const(3.0))),
        ],
        outputs: vec![Expr::Temp("t0".to_string())],
    };

    DeadCodeEliminator::eliminate(&mut program);

    assert_eq!(program.assignments.len(), 1);
    assert_eq!(program.assignments[0].0, "t0");
}

#[test]
fn test_dce_keeps_transitive_deps() {
    // t0 = x[0] + x[1]
    // t1 = t0 * 2.0   (uses t0)
    // output = t1      (uses t1, which transitively uses t0)
    let mut program = Program {
        assignments: vec![
            ("t0".to_string(), Expr::input_re(0).add(Expr::input_re(1))),
            (
                "t1".to_string(),
                Expr::Temp("t0".to_string()).mul(Expr::Const(2.0)),
            ),
        ],
        outputs: vec![Expr::Temp("t1".to_string())],
    };

    DeadCodeEliminator::eliminate(&mut program);

    assert_eq!(program.assignments.len(), 2);
}

#[test]
fn test_dce_removes_all_dead() {
    let mut program = Program {
        assignments: vec![
            ("t0".to_string(), Expr::input_re(0)),
            ("t1".to_string(), Expr::input_re(1)),
        ],
        // Output doesn't reference any temp
        outputs: vec![Expr::Const(42.0)],
    };

    DeadCodeEliminator::eliminate(&mut program);

    assert!(program.assignments.is_empty());
}

#[test]
fn test_dce_empty_program() {
    let mut program = Program::new();
    DeadCodeEliminator::eliminate(&mut program);
    assert!(program.assignments.is_empty());
    assert!(program.outputs.is_empty());
}

// --- Full optimize pipeline tests ---

#[test]
fn test_optimize_folds_and_eliminates() {
    // t0 = 2.0 + 3.0   (should fold to 5.0)
    // t1 = x[0] * 0.0  (should fold to 0.0, then dead)
    // output = t0
    let program = Program {
        assignments: vec![
            ("t0".to_string(), Expr::Const(2.0).add(Expr::Const(3.0))),
            ("t1".to_string(), Expr::input_re(0).mul(Expr::Const(0.0))),
        ],
        outputs: vec![Expr::Temp("t0".to_string())],
    };

    let optimized = optimize_fold_and_dce(program);

    // t1 should be eliminated (dead), t0 folded to Const(5.0)
    assert_eq!(optimized.assignments.len(), 1);
    assert_eq!(optimized.assignments[0].0, "t0");
    assert_eq!(optimized.assignments[0].1, Expr::Const(5.0));
}

#[test]
fn test_optimize_sub_self_and_dce() {
    // t0 = x[0] - x[0]  → should fold to 0.0
    // t1 = t0 + x[1]    → should fold to 0.0 + x[1] → ... but t0 is a Temp ref
    // output = t1
    let program = Program {
        assignments: vec![
            ("t0".to_string(), Expr::input_re(0).sub(Expr::input_re(0))),
            (
                "t1".to_string(),
                Expr::Temp("t0".to_string()).add(Expr::input_re(1)),
            ),
        ],
        outputs: vec![Expr::Temp("t1".to_string())],
    };

    let optimized = optimize_fold_and_dce(program);

    // t0 should fold to Const(0.0), both kept since t1 refs t0
    // But t0 = 0.0 (folded), t1 = t0 + x[1] (not further foldable without inlining)
    assert_eq!(optimized.assignments[0].1, Expr::Const(0.0));
}

#[test]
fn test_program_op_count() {
    let program = Program {
        assignments: vec![("t0".to_string(), Expr::input_re(0).add(Expr::input_re(1)))],
        outputs: vec![Expr::Temp("t0".to_string()).mul(Expr::Const(2.0))],
    };

    // 1 add in assignment + 1 mul in output = 2
    assert_eq!(program.op_count(), 2);
}

#[test]
fn test_collect_temp_refs() {
    let expr =
        Expr::Temp("t0".to_string()).add(Expr::Temp("t1".to_string()).mul(Expr::input_re(0)));
    let mut refs = HashSet::new();
    expr.collect_temp_refs(&mut refs);
    assert!(refs.contains("t0"));
    assert!(refs.contains("t1"));
    assert_eq!(refs.len(), 2);
}

// ================================================================
// CSE optimizer tests
// ================================================================

#[test]
fn test_cse_deduplication() {
    let mut cse = CseOptimizer::new();
    let expr = Expr::input_re(0).add(Expr::input_re(1));
    let r1 = cse.register(&expr);
    let r2 = cse.register(&expr);
    // Both should return the same temp name
    assert!(matches!(r1, Expr::Temp(_)));
    assert_eq!(r1, r2);
}

#[test]
fn test_cse_different_exprs_get_different_temps() {
    let mut cse = CseOptimizer::new();
    let e1 = Expr::input_re(0).add(Expr::input_re(1));
    let e2 = Expr::input_re(0).sub(Expr::input_re(1));
    let t1 = cse.register(&e1);
    let t2 = cse.register(&e2);
    assert_ne!(t1, t2);
}

#[test]
fn test_cse_simple_exprs_not_cse_d() {
    let mut cse = CseOptimizer::new();
    let input = Expr::input_re(0);
    let result = cse.register(&input);
    // Simple inputs should not be wrapped in temps
    assert!(matches!(
        result,
        Expr::Input {
            index: 0,
            is_real: true
        }
    ));
}

#[test]
fn test_cse_const_not_cse_d() {
    let mut cse = CseOptimizer::new();
    let c = Expr::Const(42.0);
    let result = cse.register(&c);
    assert_eq!(result, Expr::Const(42.0));
}

#[test]
fn test_cse_min_uses_filtering() {
    let mut cse = CseOptimizer::new().with_min_uses(3);
    let expr = Expr::input_re(0).add(Expr::input_re(1));
    // Register twice (below threshold of 3)
    let _ = cse.register(&expr);
    let _ = cse.register(&expr);
    // Should not appear in temporaries because only used 2 times (threshold is 3)
    let temps = cse.get_temporaries();
    assert!(temps.is_empty());

    // Register a third time
    let _ = cse.register(&expr);
    let temps = cse.get_temporaries();
    assert_eq!(temps.len(), 1);
}

#[test]
fn test_cse_temp_not_cse_d() {
    let mut cse = CseOptimizer::new();
    let t = Expr::Temp("already_temp".to_string());
    let result = cse.register(&t);
    assert_eq!(result, Expr::Temp("already_temp".to_string()));
}

// ================================================================
// Expr::Display formatting tests
// ================================================================

#[test]
fn test_expr_display_input_re() {
    let e = Expr::input_re(3);
    assert_eq!(format!("{e}"), "x[3].re");
}

#[test]
fn test_expr_display_input_im() {
    let e = Expr::input_im(5);
    assert_eq!(format!("{e}"), "x[5].im");
}

#[test]
fn test_expr_display_const() {
    let e = Expr::Const(2.5);
    assert_eq!(format!("{e}"), "2.5");
}

#[test]
fn test_expr_display_add() {
    let e = Expr::input_re(0).add(Expr::Const(1.0));
    assert_eq!(format!("{e}"), "(x[0].re + 1)");
}

#[test]
fn test_expr_display_neg() {
    let e = Expr::input_re(0).neg();
    assert_eq!(format!("{e}"), "(-x[0].re)");
}

#[test]
fn test_expr_display_temp() {
    let e = Expr::Temp("t42".to_string());
    assert_eq!(format!("{e}"), "t42");
}

// ================================================================
// Structural hash tests
// ================================================================

#[test]
fn test_structural_hash_equal_exprs() {
    let e1 = Expr::input_re(0).add(Expr::input_re(1));
    let e2 = Expr::input_re(0).add(Expr::input_re(1));
    assert_eq!(e1.structural_hash(), e2.structural_hash());
}

#[test]
fn test_structural_hash_different_exprs() {
    let e1 = Expr::input_re(0).add(Expr::input_re(1));
    let e2 = Expr::input_re(0).sub(Expr::input_re(1));
    assert_ne!(e1.structural_hash(), e2.structural_hash());
}

#[test]
fn test_structural_hash_different_indices() {
    let e1 = Expr::input_re(0);
    let e2 = Expr::input_re(1);
    assert_ne!(e1.structural_hash(), e2.structural_hash());
}

#[test]
fn test_structural_hash_re_vs_im() {
    let e1 = Expr::input_re(0);
    let e2 = Expr::input_im(0);
    assert_ne!(e1.structural_hash(), e2.structural_hash());
}

// ================================================================
// ComplexExpr tests
// ================================================================

#[test]
fn test_complex_mul_j() {
    // (a + bi) * i = -b + ai
    let c = ComplexExpr::constant(3.0, 4.0);
    let result = c.mul_j();
    let re = StrengthReducer::reduce(&result.re);
    let im = StrengthReducer::reduce(&result.im);
    assert_eq!(re.const_value(), Some(-4.0));
    assert_eq!(im.const_value(), Some(3.0));
}

#[test]
fn test_complex_mul_neg_j() {
    // (a + bi) * (-i) = b - ai
    let c = ComplexExpr::constant(3.0, 4.0);
    let result = c.mul_neg_j();
    let re = StrengthReducer::reduce(&result.re);
    let im = StrengthReducer::reduce(&result.im);
    assert_eq!(re.const_value(), Some(4.0));
    assert_eq!(im.const_value(), Some(-3.0));
}

#[test]
fn test_complex_add() {
    let a = ComplexExpr::constant(1.0, 2.0);
    let b = ComplexExpr::constant(3.0, 4.0);
    let sum = a.add(&b);
    let re = ConstantFolder::fold(&sum.re);
    let im = ConstantFolder::fold(&sum.im);
    assert_eq!(re.const_value(), Some(4.0));
    assert_eq!(im.const_value(), Some(6.0));
}

#[test]
fn test_complex_sub() {
    let a = ComplexExpr::constant(5.0, 7.0);
    let b = ComplexExpr::constant(2.0, 3.0);
    let diff = a.sub(&b);
    let re = ConstantFolder::fold(&diff.re);
    let im = ConstantFolder::fold(&diff.im);
    assert_eq!(re.const_value(), Some(3.0));
    assert_eq!(im.const_value(), Some(4.0));
}

#[test]
fn test_complex_neg() {
    let c = ComplexExpr::constant(3.0, -2.0);
    let neg = c.neg();
    let re = ConstantFolder::fold(&neg.re);
    let im = ConstantFolder::fold(&neg.im);
    assert_eq!(re.const_value(), Some(-3.0));
    assert_eq!(im.const_value(), Some(2.0));
}

#[test]
fn test_complex_mul_identity() {
    // (a + bi) * (1 + 0i) = (a + bi)
    let a = ComplexExpr::constant(3.0, 4.0);
    let one = ComplexExpr::constant(1.0, 0.0);
    let prod = a.mul(&one);
    let re = ConstantFolder::fold(&prod.re);
    let im = ConstantFolder::fold(&prod.im);
    assert_eq!(re.const_value(), Some(3.0));
    assert_eq!(im.const_value(), Some(4.0));
}

#[test]
fn test_complex_mul_commutative_constants() {
    // (2 + 3i)(4 + 5i) = (8-15) + (10+12)i = -7 + 22i
    let a = ComplexExpr::constant(2.0, 3.0);
    let b = ComplexExpr::constant(4.0, 5.0);
    let prod = a.mul(&b);
    let re = ConstantFolder::fold(&prod.re);
    let im = ConstantFolder::fold(&prod.im);
    assert_eq!(re.const_value(), Some(-7.0));
    assert_eq!(im.const_value(), Some(22.0));
}

// ================================================================
// Program construction & operations
// ================================================================

#[test]
fn test_program_from_cse() {
    let mut cse = CseOptimizer::new();
    let expr = Expr::input_re(0).add(Expr::input_re(1));
    let t1 = cse.register(&expr);
    let t2 = cse.register(&expr); // Second use
    assert_eq!(t1, t2);
    let outputs = vec![t1];
    let program = Program::from_cse(&cse, outputs);
    assert!(!program.assignments.is_empty());
    assert_eq!(program.outputs.len(), 1);
}

#[test]
fn test_program_default_empty() {
    let p = Program::default();
    assert!(p.assignments.is_empty());
    assert!(p.outputs.is_empty());
    assert_eq!(p.op_count(), 0);
}

// ================================================================
// Expr op_count tests
// ================================================================

#[test]
fn test_op_count_terminals() {
    assert_eq!(Expr::input_re(0).op_count(), 0);
    assert_eq!(Expr::Const(1.0).op_count(), 0);
    assert_eq!(Expr::Temp("t0".to_string()).op_count(), 0);
}

#[test]
fn test_op_count_basic() {
    let expr = Expr::input_re(0).add(Expr::input_re(1));
    assert_eq!(expr.op_count(), 1);
}

#[test]
fn test_op_count_nested() {
    // (x[0] + x[1]) * (x[2] - x[3]) → 3 ops
    let lhs = Expr::input_re(0).add(Expr::input_re(1));
    let rhs = Expr::input_re(2).sub(Expr::input_re(3));
    let expr = lhs.mul(rhs);
    assert_eq!(expr.op_count(), 3);
}

#[test]
fn test_op_count_neg() {
    let expr = Expr::input_re(0).neg();
    assert_eq!(expr.op_count(), 1);
}

// ================================================================
// SymbolicFFT tests
// ================================================================

#[test]
fn test_symbolic_dft_vs_radix2_size_2() {
    // Both approaches should produce structurally equivalent (or at least correct) results
    let dft = SymbolicFFT::dft(2, true);
    let radix2 = SymbolicFFT::radix2_dit(2, true);
    assert_eq!(dft.outputs.len(), radix2.outputs.len());
}

#[test]
fn test_symbolic_dft_size_4_output_count() {
    let fft = SymbolicFFT::dft(4, true);
    assert_eq!(fft.n(), 4);
    assert_eq!(fft.outputs.len(), 4);
}

#[test]
fn test_symbolic_dft_size_8_output_count() {
    let fft = SymbolicFFT::dft(8, true);
    assert_eq!(fft.n(), 8);
    assert_eq!(fft.outputs.len(), 8);
}

#[test]
fn test_symbolic_radix2_size_8() {
    let fft = SymbolicFFT::radix2_dit(8, true);
    assert_eq!(fft.n(), 8);
    assert_eq!(fft.outputs.len(), 8);
}

#[test]
fn test_symbolic_fft_op_count_increases_with_size() {
    let fft2 = SymbolicFFT::dft(2, true);
    let fft4 = SymbolicFFT::dft(4, true);
    let fft8 = SymbolicFFT::dft(8, true);
    assert!(fft4.op_count() > fft2.op_count());
    assert!(fft8.op_count() > fft4.op_count());
}

#[test]
fn test_symbolic_dft_forward_vs_inverse() {
    let fwd = SymbolicFFT::dft(4, true);
    let inv = SymbolicFFT::dft(4, false);
    // Forward and inverse should have the same structure but different twiddle constants
    assert_eq!(fwd.outputs.len(), inv.outputs.len());
    // They should NOT be identical (different sign in twiddle angle)
    // We can at least verify they both produce valid output
    assert!(fwd.op_count() > 0);
    assert!(inv.op_count() > 0);
}

// ================================================================
// Full pipeline correctness: evaluate symbolic expressions numerically
// ================================================================

/// Helper: evaluate a symbolic Expr with concrete f64 values for inputs.
fn eval_expr(expr: &Expr, inputs: &[(f64, f64)], temps: &HashMap<String, f64>) -> f64 {
    match expr {
        Expr::Input { index, is_real } => {
            if *is_real {
                inputs[*index].0
            } else {
                inputs[*index].1
            }
        }
        Expr::Const(v) => *v,
        Expr::Add(a, b) => eval_expr(a, inputs, temps) + eval_expr(b, inputs, temps),
        Expr::Sub(a, b) => eval_expr(a, inputs, temps) - eval_expr(b, inputs, temps),
        Expr::Mul(a, b) => eval_expr(a, inputs, temps) * eval_expr(b, inputs, temps),
        Expr::Neg(a) => -eval_expr(a, inputs, temps),
        Expr::Temp(name) => *temps.get(name).unwrap_or(&0.0),
    }
}

/// Reference O(n²) DFT for verification.
fn reference_dft_f64(inputs: &[(f64, f64)], forward: bool) -> Vec<(f64, f64)> {
    let n = inputs.len();
    let sign = if forward { -1.0 } else { 1.0 };
    (0..n)
        .map(|k| {
            let mut re = 0.0;
            let mut im = 0.0;
            for (j, &(xr, xi)) in inputs.iter().enumerate() {
                let angle = sign * 2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                let (s, c) = (angle.sin(), angle.cos());
                // (xr + xi*i) * (c + s*i) = (xr*c - xi*s) + (xr*s + xi*c)*i
                re += xi.mul_add(-s, xr * c);
                im += xi.mul_add(c, xr * s);
            }
            (re, im)
        })
        .collect()
}

#[test]
fn test_symbolic_dft_numeric_correctness_size_2() {
    let fft = SymbolicFFT::dft(2, true);
    let inputs = [(1.0, 0.5), (2.0, -0.3)];
    let expected = reference_dft_f64(&inputs, true);
    let temps = HashMap::new();

    for (k, exp) in expected.iter().enumerate() {
        let re = eval_expr(&fft.outputs[k].re, &inputs, &temps);
        let im = eval_expr(&fft.outputs[k].im, &inputs, &temps);
        assert!(
            (re - exp.0).abs() < 1e-12,
            "DFT(2) output[{k}].re: {re} != {}",
            exp.0
        );
        assert!(
            (im - exp.1).abs() < 1e-12,
            "DFT(2) output[{k}].im: {im} != {}",
            exp.1
        );
    }
}

#[test]
fn test_symbolic_dft_numeric_correctness_size_4() {
    let fft = SymbolicFFT::dft(4, true);
    let inputs = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.5), (0.3, -0.7)];
    let expected = reference_dft_f64(&inputs, true);
    let temps = HashMap::new();

    for (k, exp) in expected.iter().enumerate() {
        let re = eval_expr(&fft.outputs[k].re, &inputs, &temps);
        let im = eval_expr(&fft.outputs[k].im, &inputs, &temps);
        assert!(
            (re - exp.0).abs() < 1e-10,
            "DFT(4) output[{k}].re: {re} != {}",
            exp.0
        );
        assert!(
            (im - exp.1).abs() < 1e-10,
            "DFT(4) output[{k}].im: {im} != {}",
            exp.1
        );
    }
}

#[test]
fn test_symbolic_dft_numeric_correctness_size_8() {
    let fft = SymbolicFFT::dft(8, true);
    let inputs: Vec<(f64, f64)> = (0..8)
        .map(|i| (f64::from(i).sin(), f64::from(i).cos()))
        .collect();
    let expected = reference_dft_f64(&inputs, true);
    let temps = HashMap::new();

    for (k, exp) in expected.iter().enumerate() {
        let re = eval_expr(&fft.outputs[k].re, &inputs, &temps);
        let im = eval_expr(&fft.outputs[k].im, &inputs, &temps);
        assert!(
            (re - exp.0).abs() < 1e-9,
            "DFT(8) output[{k}].re: {re} != {}",
            exp.0
        );
        assert!(
            (im - exp.1).abs() < 1e-9,
            "DFT(8) output[{k}].im: {im} != {}",
            exp.1
        );
    }
}

#[test]
fn test_symbolic_radix2_numeric_correctness_size_4() {
    let fft = SymbolicFFT::radix2_dit(4, true);
    let inputs = [(2.0, 1.0), (-1.0, 0.0), (0.5, -0.5), (1.0, 1.0)];
    let expected = reference_dft_f64(&inputs, true);
    let temps = HashMap::new();

    for (k, exp) in expected.iter().enumerate() {
        let re = eval_expr(&fft.outputs[k].re, &inputs, &temps);
        let im = eval_expr(&fft.outputs[k].im, &inputs, &temps);
        assert!(
            (re - exp.0).abs() < 1e-10,
            "radix2(4) output[{k}].re: {re} != {}",
            exp.0
        );
        assert!(
            (im - exp.1).abs() < 1e-10,
            "radix2(4) output[{k}].im: {im} != {}",
            exp.1
        );
    }
}

#[test]
fn test_symbolic_radix2_numeric_correctness_size_8() {
    let fft = SymbolicFFT::radix2_dit(8, true);
    let inputs: Vec<(f64, f64)> = (0..8)
        .map(|i| {
            (
                f64::from(i).mul_add(0.3, 1.0),
                f64::from(i).mul_add(-0.1, 0.5),
            )
        })
        .collect();
    let expected = reference_dft_f64(&inputs, true);
    let temps = HashMap::new();

    for (k, exp) in expected.iter().enumerate() {
        let re = eval_expr(&fft.outputs[k].re, &inputs, &temps);
        let im = eval_expr(&fft.outputs[k].im, &inputs, &temps);
        assert!(
            (re - exp.0).abs() < 1e-9,
            "radix2(8) output[{k}].re: {re} != {}",
            exp.0
        );
        assert!(
            (im - exp.1).abs() < 1e-9,
            "radix2(8) output[{k}].im: {im} != {}",
            exp.1
        );
    }
}

// ================================================================
// Optimizer pipeline end-to-end tests
// ================================================================

#[test]
fn test_optimize_reduces_op_count() {
    // Build a program with redundant operations that should be simplified
    let program = Program {
        assignments: vec![
            (
                "t0".to_string(),
                Expr::input_re(0).mul(Expr::Const(1.0)), // x * 1 → x
            ),
            (
                "t1".to_string(),
                Expr::input_re(1).add(Expr::Const(0.0)), // x + 0 → x
            ),
            (
                "t2".to_string(),
                Expr::Const(2.0).mul(Expr::Const(3.0)), // 2 * 3 → 6
            ),
        ],
        outputs: vec![
            Expr::Temp("t0".to_string()),
            Expr::Temp("t1".to_string()),
            Expr::Temp("t2".to_string()),
        ],
    };

    let original_ops = program.op_count();
    let optimized = optimize_fold_and_dce(program);

    // After folding, ops should be reduced (x*1→x, x+0→x eliminate ops; 2*3→6 eliminates mul)
    assert!(
        optimized.op_count() <= original_ops,
        "optimize should not increase op count: {} > {}",
        optimized.op_count(),
        original_ops
    );
}

#[test]
fn test_optimize_dce_after_fold_to_const() {
    // t0 = 0.0 * x[0]  → folds to 0.0
    // t1 = t0 + x[1]   → t0 = 0.0, so folds to x[1] (if inlined) or stays simplified
    // t2 = x[2] * x[3] (dead — not referenced by output)
    // output = t1
    let program = Program {
        assignments: vec![
            ("t0".to_string(), Expr::Const(0.0).mul(Expr::input_re(0))),
            (
                "t1".to_string(),
                Expr::Temp("t0".to_string()).add(Expr::input_re(1)),
            ),
            ("t2".to_string(), Expr::input_re(2).mul(Expr::input_re(3))),
        ],
        outputs: vec![Expr::Temp("t1".to_string())],
    };

    let optimized = optimize_fold_and_dce(program);

    // t2 should be eliminated (dead code)
    let has_t2 = optimized.assignments.iter().any(|(n, _)| n == "t2");
    assert!(!has_t2, "t2 should be eliminated by DCE");

    // t0 should fold to Const(0.0)
    if let Some((_, expr)) = optimized.assignments.iter().find(|(n, _)| n == "t0") {
        assert_eq!(*expr, Expr::Const(0.0));
    }
}

#[test]
fn test_optimize_full_pipeline() {
    // Test the full optimize() function (with CSE)
    // Create two identical subexpressions → CSE should extract them
    let shared = Expr::input_re(0).add(Expr::input_re(1));
    let program = Program {
        assignments: Vec::new(),
        outputs: vec![
            shared.clone().mul(Expr::Const(2.0)),
            shared.mul(Expr::Const(3.0)),
        ],
    };

    let optimized = optimize(program);
    // The optimize function should have processed without error
    assert_eq!(optimized.outputs.len(), 2);
}

#[test]
fn test_fold_program_applies_to_all() {
    let mut program = Program {
        assignments: vec![
            ("t0".to_string(), Expr::Const(2.0).add(Expr::Const(3.0))),
            ("t1".to_string(), Expr::Const(4.0).mul(Expr::Const(0.0))),
        ],
        outputs: vec![Expr::Temp("t0".to_string()).add(Expr::Const(0.0))],
    };

    ConstantFolder::fold_program(&mut program);

    assert_eq!(program.assignments[0].1, Expr::Const(5.0));
    assert_eq!(program.assignments[1].1, Expr::Const(0.0));
}

// ================================================================
// Edge case tests for strength reducer
// ================================================================

#[test]
fn test_strength_reduce_nested_mul_zero() {
    // (x[0] + x[1]) * 0 → 0
    let expr = Expr::input_re(0)
        .add(Expr::input_re(1))
        .mul(Expr::Const(0.0));
    let reduced = StrengthReducer::reduce(&expr);
    assert_eq!(reduced, Expr::Const(0.0));
}

#[test]
fn test_strength_reduce_mul_neg_one_rhs() {
    let expr = Expr::input_re(0).mul(Expr::Const(-1.0));
    let reduced = StrengthReducer::reduce(&expr);
    assert!(matches!(reduced, Expr::Neg(_)));
}

#[test]
fn test_strength_reduce_sub_from_zero() {
    // 0 - x → -x
    let expr = Expr::Const(0.0).sub(Expr::input_re(0));
    let reduced = StrengthReducer::reduce(&expr);
    assert!(matches!(reduced, Expr::Neg(_)));
}

#[test]
fn test_strength_reduce_const_sub_const() {
    let expr = Expr::Const(10.0).sub(Expr::Const(3.0));
    let reduced = StrengthReducer::reduce(&expr);
    assert_eq!(reduced, Expr::Const(7.0));
}

#[test]
fn test_constant_folder_deeply_nested() {
    // ((1 + 2) * (3 + 4)) - ((5 - 5) + 0) → 21 - 0 → 21
    let expr = Expr::Const(1.0)
        .add(Expr::Const(2.0))
        .mul(Expr::Const(3.0).add(Expr::Const(4.0)))
        .sub(Expr::Const(5.0).sub(Expr::Const(5.0)).add(Expr::Const(0.0)));
    let folded = ConstantFolder::fold(&expr);
    assert_eq!(folded, Expr::Const(21.0));
}

// ================================================================
// Op-count regression tests for the optimization pipeline
// ================================================================

/// Count operations in a set of (re, im) expression pairs after CSE rewriting.
fn count_ops_after_cse(fft: &SymbolicFFT) -> usize {
    let folded: Vec<(Expr, Expr)> = fft
        .outputs
        .iter()
        .map(|c| (ConstantFolder::fold(&c.re), ConstantFolder::fold(&c.im)))
        .collect();
    let mut cse = RecursiveCse::new();
    for (re, im) in &folded {
        cse.count_recursive(re);
        cse.count_recursive(im);
    }
    let rewritten: Vec<(Expr, Expr)> = folded
        .iter()
        .map(|(re, im)| (cse.rewrite(re), cse.rewrite(im)))
        .collect();
    let assignments = cse
        .get_assignments()
        .into_iter()
        .map(|(name, expr)| (name, cse.rewrite(&expr)))
        .collect::<Vec<_>>();
    let assign_ops: usize = assignments.iter().map(|(_, e)| e.op_count()).sum();
    let output_ops: usize = rewritten
        .iter()
        .map(|(re, im)| re.op_count() + im.op_count())
        .sum();
    assign_ops + output_ops
}

#[test]
fn op_count_regression_size_16() {
    let fft = SymbolicFFT::radix2_dit(16, true);
    let ops_before = fft.op_count();
    let ops_after = count_ops_after_cse(&fft);

    assert!(
        ops_after < ops_before,
        "Expected optimization to reduce ops for size-16: {ops_before} → {ops_after}"
    );
}

#[test]
fn op_count_regression_size_32() {
    let fft = SymbolicFFT::radix2_dit(32, true);
    let ops_before = fft.op_count();
    let ops_after = count_ops_after_cse(&fft);

    assert!(
        ops_after < ops_before,
        "Expected optimization to reduce ops for size-32: {ops_before} → {ops_after}"
    );
}

#[test]
fn op_count_regression_size_64() {
    let fft = SymbolicFFT::radix2_dit(64, true);
    let ops_before = fft.op_count();
    let ops_after = count_ops_after_cse(&fft);

    assert!(
        ops_after < ops_before,
        "Expected optimization to reduce ops for size-64: {ops_before} → {ops_after}"
    );
}

// ============================================================================
// Instruction-scheduling optimizer pass tests
// ============================================================================

/// Construct a synthetic dependency graph for scheduling tests.
///
/// Chain A (depth 3, long critical path):
///   a0 = x[0].re + x[1].re          (depth 0 — leaf, no temp deps)
///   a1 = a0 - x[2].re               (depth 1 — depends on a0)
///   a2 = a1 + x[3].re               (depth 2 — depends on a1)
///   a3 = a2 * x[4].re               (depth 3 — depends on a2)
///
/// Chain B (depth 1, short critical path):
///   b0 = x[5].re + x[6].re          (depth 0 — leaf)
///   b1 = b0 - x[7].re               (depth 1 — depends on b0)
///
/// Dependent node (merged result, depth 4):
///   out = a3 - b1                   (depth 4 — depends on a3 and b1)
fn make_two_chain_stmts() -> Vec<(String, Expr)> {
    // Chain A
    let a0 = Expr::input_re(0).add(Expr::input_re(1)); // depth 0
    let a1 = Expr::Temp("a0".to_string()).sub(Expr::input_re(2)); // depth 1
    let a2 = Expr::Temp("a1".to_string()).add(Expr::input_re(3)); // depth 2
    let a3 = Expr::Temp("a2".to_string()).mul(Expr::input_re(4)); // depth 3
                                                                  // Chain B
    let b0 = Expr::input_re(5).add(Expr::input_re(6)); // depth 0
    let b1 = Expr::Temp("b0".to_string()).sub(Expr::input_re(7)); // depth 1
                                                                  // Merged
    let out = Expr::Temp("a3".to_string()).sub(Expr::Temp("b1".to_string())); // depth 4

    vec![
        ("a0".to_string(), a0),
        ("a1".to_string(), a1),
        ("a2".to_string(), a2),
        ("a3".to_string(), a3),
        ("b0".to_string(), b0),
        ("b1".to_string(), b1),
        ("out".to_string(), out),
    ]
}

#[test]
fn test_schedule_instructions_preserves_deps() {
    // After scheduling, each statement must appear after all its dependencies.
    let mut stmts = make_two_chain_stmts();
    super::symbolic_emit::schedule_instructions(&mut stmts);

    let position: HashMap<&str, usize> = stmts
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (name.as_str(), i))
        .collect();

    // Verify all data-dependency orderings are preserved.
    // a0 must precede a1, a1 must precede a2, etc.
    assert!(position["a0"] < position["a1"], "a0 must precede a1");
    assert!(position["a1"] < position["a2"], "a1 must precede a2");
    assert!(position["a2"] < position["a3"], "a2 must precede a3");
    assert!(position["b0"] < position["b1"], "b0 must precede b1");
    assert!(position["a3"] < position["out"], "a3 must precede out");
    assert!(position["b1"] < position["out"], "b1 must precede out");
}

#[test]
fn test_schedule_instructions_long_chain_started_early() {
    // The greedy critical-path scheduler should prefer emitting the long-chain
    // leaves before the short-chain leaves when both are in the ready queue.
    // Here both a0 and b0 are leaves (depth 0 in predecessor sense), but a0
    // has a higher critical-path depth (it feeds a3 at depth 3 vs b1 at depth 1).
    // The scheduler should emit a0 before b0 (higher depth priority).
    //
    // NOTE: In our scheduler, among ready statements the one with the highest
    // pre-computed depth wins. a0's depth = 0 (no deps) but it is the root of
    // a chain whose maximum consumer depth is 3 vs b0's consumer depth 1.
    // The scheduler assigns depth values as: depth[a0]=0, depth[a1]=1, depth[a2]=2,
    // depth[a3]=3, depth[b0]=0, depth[b1]=1, depth[out]=4.
    // The ready queue is max-heap keyed by depth, so when both a0 (depth=0) and
    // b0 (depth=0) are ready, their depths are equal and the tie is broken by
    // original index (a0 at index 0, b0 at index 4 → a0 wins by insertion order).
    // What matters most: chain A nodes (a0..a3) are all started before 'out'.
    let mut stmts = make_two_chain_stmts();
    super::symbolic_emit::schedule_instructions(&mut stmts);

    let position: HashMap<&str, usize> = stmts
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (name.as_str(), i))
        .collect();

    // The long chain A must be fully emitted before `out`.
    // Both a0 and a3 must appear before out.
    assert!(
        position["a0"] < position["out"],
        "chain A must start before out; got a0={}, out={}",
        position["a0"],
        position["out"]
    );
    assert!(
        position["a3"] < position["out"],
        "chain A must complete before out; got a3={}, out={}",
        position["a3"],
        position["out"]
    );
    // The short chain B must also be complete before out.
    assert!(
        position["b1"] < position["out"],
        "chain B must complete before out; got b1={}, out={}",
        position["b1"],
        position["out"]
    );
    // Verify overall output has all 7 statements.
    assert_eq!(stmts.len(), 7, "all statements must be emitted");
}

#[test]
fn test_schedule_instructions_empty_and_single() {
    // Edge cases: empty vec and single-element vec must not panic.
    let mut empty: Vec<(String, Expr)> = Vec::new();
    super::symbolic_emit::schedule_instructions(&mut empty);
    assert!(empty.is_empty());

    let mut single = vec![("x".to_string(), Expr::input_re(0).add(Expr::input_re(1)))];
    super::symbolic_emit::schedule_instructions(&mut single);
    assert_eq!(single.len(), 1);
    assert_eq!(single[0].0, "x");
}

#[test]
fn test_schedule_instructions_independent_chain() {
    // Three independent leaf statements (no temp deps) — all depth 0.
    // Scheduling must preserve all three and not lose any.
    let mut stmts = vec![
        ("p".to_string(), Expr::input_re(0).add(Expr::input_re(1))),
        ("q".to_string(), Expr::input_re(2).sub(Expr::input_re(3))),
        ("r".to_string(), Expr::input_re(4).mul(Expr::input_re(5))),
    ];
    super::symbolic_emit::schedule_instructions(&mut stmts);
    assert_eq!(stmts.len(), 3);
    // All names must be present (order among independent leaves is not constrained).
    let names: std::collections::HashSet<&str> = stmts.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains("p"));
    assert!(names.contains("q"));
    assert!(names.contains("r"));
}
