//! Symbolic FFT operation representation.
//!
//! This module provides a symbolic representation of FFT operations that can be
//! optimized through common subexpression elimination (CSE) and strength reduction
//! before code generation.
//!
//! Note: These types are infrastructure for the codegen proc-macros and are tested
//! but not directly exported (proc-macro crates can only export proc-macro functions).

#![allow(clippy::cast_precision_loss)] // FFT sizes fit comfortably in f64 mantissa

#[cfg(test)]
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;

/// A symbolic expression representing FFT operations.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// Input variable: `x[index].re` or `x[index].im`
    Input { index: usize, is_real: bool },
    /// Constant value
    Const(f64),
    /// Addition
    Add(Box<Self>, Box<Self>),
    /// Subtraction
    Sub(Box<Self>, Box<Self>),
    /// Multiplication
    Mul(Box<Self>, Box<Self>),
    /// Negation
    Neg(Box<Self>),
    /// Named temporary (result of CSE)
    Temp(String),
}

impl Expr {
    /// Create a real input reference.
    #[must_use]
    pub const fn input_re(index: usize) -> Self {
        Self::Input {
            index,
            is_real: true,
        }
    }

    /// Create an imaginary input reference.
    #[must_use]
    pub const fn input_im(index: usize) -> Self {
        Self::Input {
            index,
            is_real: false,
        }
    }

    /// Create a constant.
    #[must_use]
    pub const fn constant(value: f64) -> Self {
        Self::Const(value)
    }

    /// Create addition expression.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        Self::Add(Box::new(self), Box::new(other))
    }

    /// Create subtraction expression.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Self) -> Self {
        Self::Sub(Box::new(self), Box::new(other))
    }

    /// Create multiplication expression.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        Self::Mul(Box::new(self), Box::new(other))
    }

    /// Get constant value if this is a constant.
    #[must_use]
    pub const fn const_value(&self) -> Option<f64> {
        match self {
            Self::Const(v) => Some(*v),
            _ => None,
        }
    }

    /// Hash the expression for CSE.
    #[must_use]
    pub fn structural_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        self.hash_recursive(&mut hasher);
        hasher.finish()
    }

    fn hash_recursive<H: std::hash::Hasher>(&self, hasher: &mut H) {
        use std::hash::Hash;
        match self {
            Self::Input { index, is_real } => {
                0u8.hash(hasher);
                index.hash(hasher);
                is_real.hash(hasher);
            }
            Self::Const(v) => {
                1u8.hash(hasher);
                v.to_bits().hash(hasher);
            }
            Self::Add(a, b) => {
                2u8.hash(hasher);
                a.hash_recursive(hasher);
                b.hash_recursive(hasher);
            }
            Self::Sub(a, b) => {
                3u8.hash(hasher);
                a.hash_recursive(hasher);
                b.hash_recursive(hasher);
            }
            Self::Mul(a, b) => {
                4u8.hash(hasher);
                a.hash_recursive(hasher);
                b.hash_recursive(hasher);
            }
            Self::Neg(a) => {
                5u8.hash(hasher);
                a.hash_recursive(hasher);
            }
            Self::Temp(name) => {
                6u8.hash(hasher);
                name.hash(hasher);
            }
        }
    }

    /// Collect all `Temp` variable names referenced in this expression.
    pub fn collect_temp_refs(&self, refs: &mut HashSet<String>) {
        match self {
            Self::Temp(name) => {
                refs.insert(name.clone());
            }
            Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b) => {
                a.collect_temp_refs(refs);
                b.collect_temp_refs(refs);
            }
            Self::Neg(a) => a.collect_temp_refs(refs),
            Self::Input { .. } | Self::Const(_) => {}
        }
    }

    /// Count operations in this expression.
    #[must_use]
    pub fn op_count(&self) -> usize {
        match self {
            Self::Input { .. } | Self::Const(_) | Self::Temp(_) => 0,
            Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b) => 1 + a.op_count() + b.op_count(),
            Self::Neg(a) => 1 + a.op_count(),
        }
    }
}

#[cfg(test)]
impl Expr {
    /// Create negation. (test helper)
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        Self::Neg(Box::new(self))
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input { index, is_real } => {
                write!(f, "x[{}].{}", index, if *is_real { "re" } else { "im" })
            }
            Self::Const(v) => write!(f, "{v}"),
            Self::Add(a, b) => write!(f, "({a} + {b})"),
            Self::Sub(a, b) => write!(f, "({a} - {b})"),
            Self::Mul(a, b) => write!(f, "({a} * {b})"),
            Self::Neg(a) => write!(f, "(-{a})"),
            Self::Temp(name) => write!(f, "{name}"),
        }
    }
}

/// A complex symbolic expression (real, imaginary pair).
#[derive(Clone, Debug)]
pub struct ComplexExpr {
    pub re: Expr,
    pub im: Expr,
}

impl ComplexExpr {
    /// Create from input index.
    #[must_use]
    pub const fn input(index: usize) -> Self {
        Self {
            re: Expr::input_re(index),
            im: Expr::input_im(index),
        }
    }

    /// Create from constant.
    #[must_use]
    pub const fn constant(re: f64, im: f64) -> Self {
        Self {
            re: Expr::constant(re),
            im: Expr::constant(im),
        }
    }

    /// Complex addition.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            re: self.re.clone().add(other.re.clone()),
            im: self.im.clone().add(other.im.clone()),
        }
    }

    /// Complex subtraction.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            re: self.re.clone().sub(other.re.clone()),
            im: self.im.clone().sub(other.im.clone()),
        }
    }

    /// Complex multiplication.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn mul(&self, other: &Self) -> Self {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        Self {
            re: self
                .re
                .clone()
                .mul(other.re.clone())
                .sub(self.im.clone().mul(other.im.clone())),
            im: self
                .re
                .clone()
                .mul(other.im.clone())
                .add(self.im.clone().mul(other.re.clone())),
        }
    }
}

#[cfg(test)]
impl ComplexExpr {
    /// Multiply by j = sqrt(-1). (test helper)
    #[must_use]
    pub fn mul_j(&self) -> Self {
        // (a + bi) * i = -b + ai
        Self {
            re: self.im.clone().neg(),
            im: self.re.clone(),
        }
    }

    /// Multiply by -j = -sqrt(-1). (test helper)
    #[must_use]
    pub fn mul_neg_j(&self) -> Self {
        // (a + bi) * (-i) = b - ai
        Self {
            re: self.im.clone(),
            im: self.re.clone().neg(),
        }
    }

    /// Negation. (test helper)
    #[must_use]
    pub fn neg(&self) -> Self {
        Self {
            re: self.re.clone().neg(),
            im: self.im.clone().neg(),
        }
    }
}

/// Common Subexpression Elimination optimizer. (used only in tests)
#[cfg(test)]
pub struct CseOptimizer {
    /// Map from expression hash to (expression, temp name, use count).
    expr_cache: HashMap<u64, (Expr, String, usize)>,
    /// Counter for generating temp names.
    temp_counter: usize,
    /// Threshold for CSE (min uses to create temp).
    min_uses: usize,
}

#[cfg(test)]
impl CseOptimizer {
    /// Create a new CSE optimizer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            expr_cache: HashMap::new(),
            temp_counter: 0,
            min_uses: 2,
        }
    }

    /// Set minimum uses threshold for CSE.
    #[must_use]
    pub const fn with_min_uses(mut self, min_uses: usize) -> Self {
        self.min_uses = min_uses;
        self
    }

    /// Register an expression and return the optimized version.
    #[must_use]
    pub fn register(&mut self, expr: &Expr) -> Expr {
        // Don't CSE simple expressions
        if matches!(expr, Expr::Input { .. } | Expr::Const(_) | Expr::Temp(_)) {
            return expr.clone();
        }

        let hash = expr.structural_hash();

        if let Some((_, name, count)) = self.expr_cache.get_mut(&hash) {
            *count += 1;
            return Expr::Temp(name.clone());
        }

        let name = format!("t{}", self.temp_counter);
        self.temp_counter += 1;
        self.expr_cache
            .insert(hash, (expr.clone(), name.clone(), 1));
        Expr::Temp(name)
    }

    /// Get all temporaries that should be generated.
    #[must_use]
    pub fn get_temporaries(&self) -> Vec<(String, Expr)> {
        let mut temps: Vec<_> = self
            .expr_cache
            .values()
            .filter(|(_, _, count)| *count >= self.min_uses)
            .map(|(expr, name, _)| (name.clone(), expr.clone()))
            .collect();
        temps.sort_by(|a, b| a.0.cmp(&b.0));
        temps
    }
}

#[cfg(test)]
impl Default for CseOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Strength reduction optimizer.
pub struct StrengthReducer;

impl StrengthReducer {
    /// Apply strength reduction to an expression.
    /// Reduces recursively from bottom up.
    #[must_use]
    pub fn reduce(expr: &Expr) -> Expr {
        match expr {
            // Mul: reduce children first, then simplify
            Expr::Mul(a, b) => {
                let ra = Self::reduce(a);
                let rb = Self::reduce(b);

                // Mul by 0 -> 0
                if ra.const_value() == Some(0.0) || rb.const_value() == Some(0.0) {
                    return Expr::Const(0.0);
                }
                // Mul by 1 -> identity
                if ra.const_value() == Some(1.0) {
                    return rb;
                }
                if rb.const_value() == Some(1.0) {
                    return ra;
                }
                // Mul by -1 -> negation
                if ra.const_value() == Some(-1.0) {
                    return Expr::Neg(Box::new(rb));
                }
                if rb.const_value() == Some(-1.0) {
                    return Expr::Neg(Box::new(ra));
                }
                // Const * Const -> Const
                if let (Some(va), Some(vb)) = (ra.const_value(), rb.const_value()) {
                    return Expr::Const(va * vb);
                }
                Expr::Mul(Box::new(ra), Box::new(rb))
            }

            // Add: reduce children first, then simplify
            Expr::Add(a, b) => {
                let ra = Self::reduce(a);
                let rb = Self::reduce(b);

                // Add with 0 -> identity
                if ra.const_value() == Some(0.0) {
                    return rb;
                }
                if rb.const_value() == Some(0.0) {
                    return ra;
                }
                // Const + Const -> Const
                if let (Some(va), Some(vb)) = (ra.const_value(), rb.const_value()) {
                    return Expr::Const(va + vb);
                }
                Expr::Add(Box::new(ra), Box::new(rb))
            }

            // Sub: reduce children first, then simplify
            Expr::Sub(a, b) => {
                let ra = Self::reduce(a);
                let rb = Self::reduce(b);

                // x - x -> 0 (structural equality)
                if ra == rb {
                    return Expr::Const(0.0);
                }
                // Sub with 0 -> identity/negation
                if rb.const_value() == Some(0.0) {
                    return ra;
                }
                if ra.const_value() == Some(0.0) {
                    return Expr::Neg(Box::new(rb));
                }
                // Const - Const -> Const
                if let (Some(va), Some(vb)) = (ra.const_value(), rb.const_value()) {
                    return Expr::Const(va - vb);
                }
                Expr::Sub(Box::new(ra), Box::new(rb))
            }

            // Neg: reduce child first, then simplify
            Expr::Neg(a) => {
                let ra = Self::reduce(a);

                // Neg of Neg -> identity
                if let Expr::Neg(inner) = &ra {
                    return *inner.clone();
                }
                // Neg of Const -> Const
                if let Some(v) = ra.const_value() {
                    return Expr::Const(-v);
                }
                Expr::Neg(Box::new(ra))
            }

            // Terminals
            Expr::Input { .. } | Expr::Const(_) | Expr::Temp(_) => expr.clone(),
        }
    }
}

/// Constant folder that applies algebraic simplifications to fixpoint.
///
/// This wraps [`StrengthReducer`] and applies it repeatedly until the expression
/// no longer changes, ensuring all nested constant folding opportunities are caught.
pub struct ConstantFolder;

impl ConstantFolder {
    /// Apply constant folding to an expression until fixpoint.
    ///
    /// This applies strength reduction (which includes constant folding rules)
    /// repeatedly until the expression stabilizes.
    #[must_use]
    pub fn fold(expr: &Expr) -> Expr {
        let mut current = expr.clone();
        loop {
            let folded = StrengthReducer::reduce(&current);
            if folded == current {
                return current;
            }
            current = folded;
        }
    }
}

#[cfg(test)]
impl ConstantFolder {
    /// Apply constant folding to all expressions in a program. (test helper)
    pub fn fold_program(program: &mut Program) {
        for (_name, expr) in &mut program.assignments {
            *expr = Self::fold(expr);
        }
        for expr in &mut program.outputs {
            *expr = Self::fold(expr);
        }
    }
}

/// Dead code eliminator for symbolic programs. (used only in tests)
#[cfg(test)]
pub struct DeadCodeEliminator;

#[cfg(test)]
impl DeadCodeEliminator {
    /// Eliminate dead temporary assignments from a program.
    ///
    /// Performs a reachability analysis starting from output expressions,
    /// transitively marking all referenced temporaries as live, then
    /// removes any assignments not in the live set.
    pub fn eliminate(program: &mut Program) {
        // Collect all temp refs from output expressions
        let mut live: HashSet<String> = HashSet::new();
        for expr in &program.outputs {
            expr.collect_temp_refs(&mut live);
        }

        // Build a map from temp name to its expression for transitive lookup
        let assign_map: HashMap<String, &Expr> = program
            .assignments
            .iter()
            .map(|(name, expr)| (name.clone(), expr))
            .collect();

        // Transitive closure: keep discovering new live temps
        let mut worklist: Vec<String> = live.iter().cloned().collect();
        while let Some(name) = worklist.pop() {
            if let Some(expr) = assign_map.get(&name) {
                let mut new_refs = HashSet::new();
                expr.collect_temp_refs(&mut new_refs);
                for r in new_refs {
                    if live.insert(r.clone()) {
                        worklist.push(r);
                    }
                }
            }
        }

        // Retain only live assignments
        program.assignments.retain(|(name, _)| live.contains(name));
    }
}

/// A symbolic program: a sequence of temporary assignments plus output expressions.
///
/// This type is used in tests for the optimization pipeline infrastructure.
/// For code generation, `emit_body_from_symbolic` uses `RecursiveCse` directly.
#[cfg(test)]
#[derive(Clone, Debug)]
pub struct Program {
    /// Temporary variable assignments in order: `(name, expression)`.
    pub assignments: Vec<(String, Expr)>,
    /// Output expressions (may reference temps from assignments).
    pub outputs: Vec<Expr>,
}

#[cfg(test)]
impl Program {
    /// Create a new empty program.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            assignments: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Create a program from CSE optimizer results and output expressions.
    #[must_use]
    pub fn from_cse(cse: &CseOptimizer, outputs: Vec<Expr>) -> Self {
        Self {
            assignments: cse.get_temporaries(),
            outputs,
        }
    }

    /// Total operation count across all assignments and outputs.
    #[must_use]
    pub fn op_count(&self) -> usize {
        let assign_ops: usize = self.assignments.iter().map(|(_, e)| e.op_count()).sum();
        let output_ops: usize = self.outputs.iter().map(Expr::op_count).sum();
        assign_ops + output_ops
    }
}

#[cfg(test)]
impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply all optimization passes to a program.
///
/// The optimization pipeline is:
/// 1. **Constant folding** — simplify constant expressions and algebraic identities
/// 2. **CSE** — extract common subexpressions into temporaries
/// 3. **Dead code elimination** — remove unused temporaries
///
/// Returns the optimized program.
#[cfg(test)]
#[must_use]
pub fn optimize(mut program: Program) -> Program {
    // Pass 1: Constant folding
    ConstantFolder::fold_program(&mut program);

    // Pass 2: CSE on the folded expressions
    let mut cse = CseOptimizer::new();
    let new_outputs: Vec<Expr> = program
        .outputs
        .iter()
        .map(|expr| cse.register(expr))
        .collect();

    // Also register assignment RHS through CSE
    let new_assignments: Vec<(String, Expr)> = program
        .assignments
        .iter()
        .map(|(name, expr)| (name.clone(), cse.register(expr)))
        .collect();

    // Merge CSE-generated temporaries with existing ones
    let mut all_assignments = cse.get_temporaries();
    for (name, expr) in new_assignments {
        if !all_assignments.iter().any(|(n, _)| n == &name) {
            all_assignments.push((name, expr));
        }
    }

    program.assignments = all_assignments;
    program.outputs = new_outputs;

    // Pass 3: Dead code elimination
    DeadCodeEliminator::eliminate(&mut program);

    program
}

/// Apply constant folding and DCE without CSE (for cases where CSE is handled separately).
#[cfg(test)]
#[must_use]
pub fn optimize_fold_and_dce(mut program: Program) -> Program {
    ConstantFolder::fold_program(&mut program);
    DeadCodeEliminator::eliminate(&mut program);
    program
}

/// FFT symbolic computation.
pub struct SymbolicFFT {
    /// Output expressions (real, imag pairs). Length equals the transform size.
    pub outputs: Vec<ComplexExpr>,
}

impl SymbolicFFT {
    /// Generate radix-2 Cooley-Tukey FFT symbolically.
    ///
    /// # Panics
    /// Panics if `n` is not a power of two.
    #[must_use]
    pub fn radix2_dit(n: usize, forward: bool) -> Self {
        assert!(n.is_power_of_two(), "n must be power of 2");

        let sign = if forward { -1.0 } else { 1.0 };

        // Start with inputs
        let mut data: Vec<ComplexExpr> = (0..n).map(ComplexExpr::input).collect();

        // Bit-reversal permutation
        let mut j = 0;
        for i in 0..n {
            if i < j {
                data.swap(i, j);
            }
            let mut m = n >> 1;
            while m >= 1 && j >= m {
                j -= m;
                m >>= 1;
            }
            j += m;
        }

        // Cooley-Tukey stages
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let angle_step = sign * 2.0 * std::f64::consts::PI / len as f64;

            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let angle = angle_step * k as f64;
                    let twiddle = ComplexExpr::constant(angle.cos(), angle.sin());

                    let u = data[start + k].clone();
                    let t = data[start + k + half].mul(&twiddle);

                    data[start + k] = u.add(&t);
                    data[start + k + half] = u.sub(&t);
                }
            }

            len *= 2;
        }

        // Apply strength reduction to all outputs
        let outputs: Vec<ComplexExpr> = data
            .into_iter()
            .map(|c| ComplexExpr {
                re: StrengthReducer::reduce(&c.re),
                im: StrengthReducer::reduce(&c.im),
            })
            .collect();

        Self { outputs }
    }

    /// Total operation count.
    #[must_use]
    pub fn op_count(&self) -> usize {
        self.outputs
            .iter()
            .map(|c| c.re.op_count() + c.im.op_count())
            .sum()
    }
}

#[cfg(test)]
impl SymbolicFFT {
    /// Size of the FFT (derived from output count).
    #[must_use]
    pub fn n(&self) -> usize {
        self.outputs.len()
    }

    /// Generate naive O(n²) DFT symbolically. (test helper)
    #[must_use]
    pub fn dft(n: usize, forward: bool) -> Self {
        let sign = if forward { -1.0 } else { 1.0 };
        let mut outputs = Vec::with_capacity(n);

        for k in 0..n {
            let mut re = Expr::Const(0.0);
            let mut im = Expr::Const(0.0);

            for j in 0..n {
                let angle = sign * 2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                let tw_re = angle.cos();
                let tw_im = angle.sin();

                let input = ComplexExpr::input(j);
                let twiddle = ComplexExpr::constant(tw_re, tw_im);
                let product = input.mul(&twiddle);

                re = re.add(product.re);
                im = im.add(product.im);
            }

            outputs.push(ComplexExpr {
                re: StrengthReducer::reduce(&re),
                im: StrengthReducer::reduce(&im),
            });
        }

        Self { outputs }
    }
}

// ============================================================================
// Code emission: symbolic FFT → proc_macro2::TokenStream
// (implementation lives in symbolic_emit.rs to keep this file under 2000 lines)
// ============================================================================

#[path = "symbolic_emit.rs"]
mod symbolic_emit;
pub use symbolic_emit::{emit_body_from_symbolic, schedule_instructions};

// ============================================================================
// Tests
// (implementation lives in symbolic_tests.rs to keep this file under 2000 lines)
// ============================================================================

#[cfg(test)]
#[path = "symbolic_tests.rs"]
mod tests;
