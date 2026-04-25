//! Code emission: symbolic FFT expressions → `proc_macro2::TokenStream`.
//!
//! This module contains the recursive CSE optimizer used for code generation
//! and the emission functions that convert symbolic FFT expressions into Rust
//! token streams.

use std::collections::{BinaryHeap, HashMap, HashSet};

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use super::{ConstantFolder, Expr, SymbolicFFT};

// ============================================================================
// Recursive CSE for code generation
// ============================================================================

/// Recursive CSE optimizer for code emission.
///
/// Unlike `CseOptimizer::register()` which only hashes the top-level expression,
/// this walker traverses the entire expression tree, extracts shared
/// subexpressions into named temporaries, and replaces their occurrences with
/// `Expr::Temp` references.  Expressions used only once are left inline.
pub(super) struct RecursiveCse {
    /// Map from structural hash → (original expr, temp name, use count).
    cache: HashMap<u64, (Expr, String, usize)>,
    counter: usize,
}

impl RecursiveCse {
    pub(super) fn new() -> Self {
        Self {
            cache: HashMap::new(),
            counter: 0,
        }
    }

    /// Count usages of each subexpression across all outputs (bottom-up).
    pub(super) fn count_recursive(&mut self, expr: &Expr) {
        match expr {
            Expr::Input { .. } | Expr::Const(_) | Expr::Temp(_) => {}
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
                self.count_recursive(a);
                self.count_recursive(b);
                let hash = expr.structural_hash();
                let entry = self.cache.entry(hash).or_insert_with(|| {
                    let name = format!("t{}", self.counter);
                    self.counter += 1;
                    (expr.clone(), name, 0)
                });
                entry.2 += 1;
            }
            Expr::Neg(a) => {
                self.count_recursive(a);
                let hash = expr.structural_hash();
                let entry = self.cache.entry(hash).or_insert_with(|| {
                    let name = format!("t{}", self.counter);
                    self.counter += 1;
                    (expr.clone(), name, 0)
                });
                entry.2 += 1;
            }
        }
    }

    /// Rewrite an expression replacing shared subexpressions with `Temp` refs.
    /// Only extracts subexpressions used >= 2 times.
    ///
    /// Set `top_level_name` to `Some(name)` when rewriting the RHS of an
    /// assignment — this prevents the expression from replacing itself with
    /// its own temp name (self-reference).
    fn rewrite_inner(&self, expr: &Expr, exclude_hash: Option<u64>) -> Expr {
        match expr {
            Expr::Input { .. } | Expr::Const(_) | Expr::Temp(_) => expr.clone(),
            Expr::Add(a, b) => {
                let hash = expr.structural_hash();
                if exclude_hash != Some(hash) {
                    if let Some((_, name, count)) = self.cache.get(&hash) {
                        if *count >= 2 {
                            return Expr::Temp(name.clone());
                        }
                    }
                }
                Expr::Add(
                    Box::new(self.rewrite_inner(a, None)),
                    Box::new(self.rewrite_inner(b, None)),
                )
            }
            Expr::Sub(a, b) => {
                let hash = expr.structural_hash();
                if exclude_hash != Some(hash) {
                    if let Some((_, name, count)) = self.cache.get(&hash) {
                        if *count >= 2 {
                            return Expr::Temp(name.clone());
                        }
                    }
                }
                Expr::Sub(
                    Box::new(self.rewrite_inner(a, None)),
                    Box::new(self.rewrite_inner(b, None)),
                )
            }
            Expr::Mul(a, b) => {
                let hash = expr.structural_hash();
                if exclude_hash != Some(hash) {
                    if let Some((_, name, count)) = self.cache.get(&hash) {
                        if *count >= 2 {
                            return Expr::Temp(name.clone());
                        }
                    }
                }
                Expr::Mul(
                    Box::new(self.rewrite_inner(a, None)),
                    Box::new(self.rewrite_inner(b, None)),
                )
            }
            Expr::Neg(a) => {
                let hash = expr.structural_hash();
                if exclude_hash != Some(hash) {
                    if let Some((_, name, count)) = self.cache.get(&hash) {
                        if *count >= 2 {
                            return Expr::Temp(name.clone());
                        }
                    }
                }
                Expr::Neg(Box::new(self.rewrite_inner(a, None)))
            }
        }
    }

    /// Rewrite an output expression (replacing shared subexpressions with Temp refs).
    pub(super) fn rewrite(&self, expr: &Expr) -> Expr {
        self.rewrite_inner(expr, None)
    }

    /// Rewrite the RHS of an assignment, excluding the assignment itself from
    /// self-reference replacement.
    pub(super) fn rewrite_assignment_rhs(&self, name: &str, expr: &Expr) -> Expr {
        // Find the hash for this assignment's expression
        let hash = self
            .cache
            .iter()
            .find(|(_, (_, n, _))| n == name)
            .map(|(h, _)| *h);
        self.rewrite_inner(expr, hash)
    }

    /// Return sorted assignments for temps used >= 2 times.
    pub(super) fn get_assignments(&self) -> Vec<(String, Expr)> {
        let mut result: Vec<(String, Expr)> = self
            .cache
            .values()
            .filter(|(_, _, count)| *count >= 2)
            .map(|(expr, name, _)| (name.clone(), expr.clone()))
            .collect();
        // Sort by the numeric suffix for deterministic output.
        // Names are "t0", "t1", ..., "t99", "t100", etc.
        result.sort_by(|a, b| {
            let na: usize = a.0[1..].parse().unwrap_or(0);
            let nb: usize = b.0[1..].parse().unwrap_or(0);
            na.cmp(&nb)
        });
        result
    }
}

// ============================================================================
// Code emission: symbolic FFT → proc_macro2::TokenStream
// ============================================================================

/// Build the body `TokenStream` for one direction of an n-point FFT from
/// symbolic computation and optimization passes.
///
/// This function:
/// 1. Builds the symbolic DAG via `SymbolicFFT::radix2_dit(n, forward)`.
/// 2. Applies constant folding to each output expression.
/// 3. Performs recursive CSE to extract shared subexpressions into named temps.
/// 4. Emits: input extractions, CSE temporaries, output assignments.
#[must_use]
pub fn emit_body_from_symbolic(n: usize, forward: bool) -> TokenStream {
    let fft = SymbolicFFT::radix2_dit(n, forward);

    // Step 1: constant-fold each output
    let folded_outputs: Vec<(Expr, Expr)> = fft
        .outputs
        .iter()
        .map(|c| (ConstantFolder::fold(&c.re), ConstantFolder::fold(&c.im)))
        .collect();

    let ops_before = fft.op_count();

    // Step 2: recursive CSE across all folded expressions
    let mut cse = RecursiveCse::new();
    for (re, im) in &folded_outputs {
        cse.count_recursive(re);
        cse.count_recursive(im);
    }

    // Step 3: rewrite outputs to replace shared subexpressions with Temp refs
    let rewritten_outputs: Vec<(Expr, Expr)> = folded_outputs
        .iter()
        .map(|(re, im)| (cse.rewrite(re), cse.rewrite(im)))
        .collect();

    // Also rewrite the CSE assignment RHS (their children may be shared too).
    // Use rewrite_assignment_rhs to avoid self-reference replacement.
    let mut assignments: Vec<(String, Expr)> = cse
        .get_assignments()
        .into_iter()
        .map(|(name, expr)| {
            let rewritten = cse.rewrite_assignment_rhs(&name, &expr);
            (name, rewritten)
        })
        .collect();

    // Topological sort: assignments that reference other assignments must come later
    assignments = topological_sort_assignments(assignments);

    if std::env::var("OXIFFT_CODEGEN_DEBUG").is_ok() {
        let ops_after: usize = assignments.iter().map(|(_, e)| e.op_count()).sum::<usize>()
            + rewritten_outputs
                .iter()
                .map(|(re, im)| re.op_count() + im.op_count())
                .sum::<usize>();
        let pct = if ops_before > 0 {
            (ops_after as f64 - ops_before as f64) / ops_before as f64 * 100.0
        } else {
            0.0
        };
        eprintln!(
            "[oxifft-codegen] n={n} forward={forward}: {ops_before} ops → {ops_after} ops ({pct:+.1}%)",
        );
    }

    // Step 4: instruction-scheduling optimizer pass
    // Re-orders assignments by critical-path priority (Sethi-Ullman heuristic):
    // leaves (depth=0) are emitted first; among equal depths, prefer statements
    // whose results are consumed by the longest remaining critical path.
    schedule_instructions(&mut assignments);

    emit_folded_body(n, &assignments, &rewritten_outputs)
}

// ============================================================================
// Instruction-scheduling optimizer pass
// ============================================================================

/// Schedule assignment statements to maximise instruction-level parallelism (ILP).
///
/// Algorithm (Sethi-Ullman critical-path heuristic):
/// 1. Build a def-use dependency graph: for each statement `(name, expr)`, record
///    all prior statements whose results `expr` references (via `Expr::Temp`).
/// 2. Compute critical-path depth per statement via longest-path from leaves:
///    - Statements with no temp-ref dependencies → depth 0.
///    - Each dependent statement → `1 + max(deps' depths)`.
/// 3. Topological re-ordering: maintain a ready-queue of statements whose all
///    dependencies have already been emitted.  Among ready candidates, prefer
///    those with the **largest** critical-path depth (i.e., the ones that unblock
///    the longest remaining work) — this is the "greedy critical-path first" rule.
/// 4. Guaranteed correctness: no statement is emitted before all its deps are done.
///
/// The pass operates in-place on the assignment vector. It will not reorder
/// statements that were placed in a topologically invalid order beforehand —
/// call `topological_sort_assignments` first if needed.  In practice,
/// `emit_body_from_symbolic` calls both in sequence.
pub fn schedule_instructions(stmts: &mut Vec<(String, Expr)>) {
    let n = stmts.len();
    if n <= 1 {
        return;
    }

    // Build name → index map for O(1) predecessor lookup.
    let index_of: std::collections::HashMap<String, usize> = stmts
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (name.clone(), i))
        .collect();

    // For each statement, collect its direct predecessor indices (statements it depends on).
    let predecessors: Vec<Vec<usize>> = stmts
        .iter()
        .map(|(_, expr)| {
            let mut refs = HashSet::new();
            expr.collect_temp_refs(&mut refs);
            refs.iter()
                .filter_map(|r| index_of.get(r).copied())
                .collect()
        })
        .collect();

    // Compute critical-path depth per statement (longest path from a leaf).
    // Leaves (no deps) have depth 0.  We process in topological order (guaranteed
    // by the caller's prior topological sort).
    let mut depth = vec![0usize; n];
    for (i, preds) in predecessors.iter().enumerate() {
        for &pred in preds {
            let candidate = depth[pred] + 1;
            if candidate > depth[i] {
                depth[i] = candidate;
            }
        }
    }

    // Build successor sets: for each statement i, which statements directly use it?
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, preds) in predecessors.iter().enumerate() {
        for &pred in preds {
            successors[pred].push(i);
        }
    }

    // Greedy critical-path scheduler.
    // ready_queue: statements all of whose predecessors have been emitted, stored
    // as (depth, original_index) — highest depth first (max-heap via BinaryHeap).
    let mut in_degree: Vec<usize> = predecessors.iter().map(Vec::len).collect();
    let mut emitted = vec![false; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);

    // Seed ready queue with all depth-0 (no-predecessor) statements.
    let mut ready: BinaryHeap<(usize, usize)> = BinaryHeap::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            ready.push((depth[i], i));
        }
    }

    while let Some((_, idx)) = ready.pop() {
        if emitted[idx] {
            continue; // guard against duplicate insertions
        }
        emitted[idx] = true;
        order.push(idx);
        // Decrement in-degree for each successor; push newly ready ones.
        for &succ in &successors[idx] {
            if in_degree[succ] > 0 {
                in_degree[succ] -= 1;
            }
            if in_degree[succ] == 0 && !emitted[succ] {
                ready.push((depth[succ], succ));
            }
        }
    }

    // If scheduling was incomplete (cycle or bug), preserve original order for stragglers.
    if order.len() < n {
        for (i, &already_emitted) in emitted.iter().enumerate() {
            if !already_emitted {
                order.push(i);
            }
        }
    }

    // Reorder stmts according to the computed schedule.
    // We need to physically rearrange the Vec without cloning Expr trees.
    // Build a temporary vec of (old_index, new_position) pairs, then permute.
    let mut positioned: Vec<Option<(String, Expr)>> = stmts.drain(..).map(Some).collect();
    let reordered: Vec<(String, Expr)> = order
        .into_iter()
        .filter_map(|i| positioned[i].take())
        .collect();
    *stmts = reordered;
}

/// Topologically sort assignments so that each temp is defined before use.
fn topological_sort_assignments(assignments: Vec<(String, Expr)>) -> Vec<(String, Expr)> {
    let mut defined: HashSet<String> = HashSet::new();
    let mut result: Vec<(String, Expr)> = Vec::with_capacity(assignments.len());
    let mut remaining = assignments;

    // Iterative pass: on each iteration, move all assignments whose dependencies
    // are fully satisfied into `result`.  Repeat until stable.
    loop {
        let before_len = result.len();
        let mut next_remaining = Vec::new();
        for (name, expr) in remaining {
            let mut refs: HashSet<String> = HashSet::new();
            expr.collect_temp_refs(&mut refs);
            if refs.iter().all(|r| defined.contains(r)) {
                defined.insert(name.clone());
                result.push((name, expr));
            } else {
                next_remaining.push((name, expr));
            }
        }
        remaining = next_remaining;
        if remaining.is_empty() || result.len() == before_len {
            // Either done, or there's a cycle (shouldn't happen in acyclic DAG)
            result.extend(remaining);
            break;
        }
    }
    result
}

/// Emit the inner body statements from constant-folded and CSE-optimized outputs.
///
/// Emits:
/// - input extraction: `let x{i}_re = x[{i}].re; let x{i}_im = x[{i}].im;`
/// - CSE temporaries: `let {name} = {expr};`
/// - output assignments: `x[{k}] = crate::kernel::Complex::new({re}, {im});`
fn emit_folded_body(
    n: usize,
    assignments: &[(String, Expr)],
    outputs: &[(Expr, Expr)],
) -> TokenStream {
    assert_eq!(
        outputs.len(),
        n,
        "expected n outputs for n-point complex FFT, got {}",
        outputs.len()
    );

    let mut body = TokenStream::new();

    // Extract inputs
    for i in 0..n {
        let re_name = format_ident!("x{i}_re");
        let im_name = format_ident!("x{i}_im");
        body.extend(quote! {
            let #re_name = x[#i].re;
            let #im_name = x[#i].im;
        });
    }

    // Emit CSE temporaries
    for (name, expr) in assignments {
        let id = format_ident!("{name}");
        let tok = emit_scalar_expr(expr);
        body.extend(quote! { let #id = #tok; });
    }

    // Emit outputs
    for (k, (re_expr, im_expr)) in outputs.iter().enumerate() {
        let re_tok = emit_scalar_expr(re_expr);
        let im_tok = emit_scalar_expr(im_expr);
        body.extend(quote! {
            x[#k] = crate::kernel::Complex::new(#re_tok, #im_tok);
        });
    }

    body
}

/// Emit a single scalar `Expr` as a `TokenStream`.
fn emit_scalar_expr(expr: &Expr) -> TokenStream {
    match expr {
        Expr::Input { index, is_real } => {
            let name = if *is_real {
                format_ident!("x{index}_re")
            } else {
                format_ident!("x{index}_im")
            };
            quote! { #name }
        }
        Expr::Const(v) => {
            if (*v - 0.0_f64).abs() < f64::EPSILON {
                quote! { T::ZERO }
            } else if (*v - 1.0_f64).abs() < f64::EPSILON {
                quote! { T::ONE }
            } else if (*v - (-1.0_f64)).abs() < f64::EPSILON {
                quote! { (-T::ONE) }
            } else {
                let v = *v;
                quote! { T::from_f64(#v) }
            }
        }
        Expr::Add(a, b) => {
            let a = emit_scalar_expr(a);
            let b = emit_scalar_expr(b);
            quote! { (#a + #b) }
        }
        Expr::Sub(a, b) => {
            let a = emit_scalar_expr(a);
            let b = emit_scalar_expr(b);
            quote! { (#a - #b) }
        }
        Expr::Mul(a, b) => {
            let a = emit_scalar_expr(a);
            let b = emit_scalar_expr(b);
            quote! { (#a * #b) }
        }
        Expr::Neg(a) => {
            let a = emit_scalar_expr(a);
            quote! { (-#a) }
        }
        Expr::Temp(name) => {
            let id = format_ident!("{name}");
            quote! { #id }
        }
    }
}
