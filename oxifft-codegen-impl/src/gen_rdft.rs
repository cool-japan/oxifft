//! RDFT codelet generation — R2HC and HC2R.
//!
//! Generates optimized codelets matching the hand-written signatures in
//! `oxifft/src/rdft/codelets/mod.rs`:
//!
//! - `r2hc_N<T: Float>(x: &[T], y: &mut [Complex<T>])` — real to half-complex
//! - `hc2r_N<T: Float>(y: &[Complex<T>], x: &mut [T])` — half-complex to real (unnormalized)
//!
//! R2HC output stores N/2+1 complex bins: Y\[0\]…Y\[N/2\].
//! Y\[0\].im and Y\[N/2\].im are always zero for real inputs.
//! HC2R is the exact inverse butterfly; caller divides by N for true inverse.

#![allow(clippy::cast_precision_loss)] // small FFT sizes (≤8) fit safely in f64 mantissa

use std::collections::HashMap;

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{parse::ParseStream, Ident, LitInt, Token};

use crate::symbolic::{ConstantFolder, Expr, StrengthReducer};

// ============================================================================
// Input parsing
// ============================================================================

/// Parsed arguments for `gen_rdft_codelet!(size = N, kind = R2hc | Hc2r)`.
pub struct RdftInput {
    pub size: usize,
    pub kind: RdftKind,
}

/// Which codelet direction to generate.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RdftKind {
    R2hc,
    Hc2r,
}

impl syn::parse::Parse for RdftInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        // Parse: size = <int>, kind = <Ident>
        let kw_size: Ident = input.parse()?;
        if kw_size != "size" {
            return Err(syn::Error::new(
                kw_size.span(),
                "expected `size = N, kind = R2hc | Hc2r`",
            ));
        }
        let _eq: Token![=] = input.parse()?;
        let size_lit: LitInt = input.parse()?;
        let size: usize = size_lit
            .base10_parse()
            .map_err(|_| syn::Error::new(size_lit.span(), "expected integer size literal"))?;

        let _comma: Token![,] = input.parse()?;

        let kw_kind: Ident = input.parse()?;
        if kw_kind != "kind" {
            return Err(syn::Error::new(
                kw_kind.span(),
                "expected `kind = R2hc | Hc2r`",
            ));
        }
        let _eq2: Token![=] = input.parse()?;
        let kind_ident: Ident = input.parse()?;

        let kind = match kind_ident.to_string().as_str() {
            "R2hc" => RdftKind::R2hc,
            "Hc2r" => RdftKind::Hc2r,
            other => {
                return Err(syn::Error::new(
                    kind_ident.span(),
                    format!("unknown RDFT kind `{other}`, expected `R2hc` or `Hc2r`"),
                ))
            }
        };

        Ok(Self { size, kind })
    }
}

// ============================================================================
// Public entry point
// ============================================================================

/// Generate a `gen_rdft_codelet!(size = N, kind = R2hc|Hc2r)` codelet.
///
/// # Errors
/// Returns `syn::Error` if the input fails to parse or the size is unsupported.
pub fn generate(input: TokenStream) -> Result<TokenStream, syn::Error> {
    let parsed: RdftInput = syn::parse2(input)?;
    match parsed.kind {
        RdftKind::R2hc => gen_r2hc(parsed.size),
        RdftKind::Hc2r => gen_hc2r(parsed.size),
    }
}

// ============================================================================
// R2HC generation
// ============================================================================

fn gen_r2hc(n: usize) -> Result<TokenStream, syn::Error> {
    match n {
        2 | 4 | 8 => Ok(emit_r2hc_codelet(n)),
        _ => Err(syn::Error::new(
            Span::call_site(),
            format!("gen_rdft_codelet: unsupported size {n} for R2hc (expected 2, 4, or 8)"),
        )),
    }
}

/// Build symbolic R2HC expressions for all k in 0..=n/2.
///
/// Y[k].re = Σ_{j=0..n-1} x[j] · cos(-2π·j·k/N)  = Σ x[j]·cos(2π·j·k/N)
/// Y[k].im = Σ_{j=0..n-1} x[j] · sin(-2π·j·k/N)  = -Σ x[j]·sin(2π·j·k/N)
///
/// (Uses the DFT kernel e^{-2πi·j·k/N} = cos - i·sin, so im component accumulates
///  sin of the negative angle.)
fn symbolic_r2hc(n: usize) -> Vec<(Expr, Expr)> {
    let half = n / 2;
    let mut outputs = Vec::with_capacity(half + 1);

    for k in 0..=half {
        let mut re_acc = Expr::Const(0.0);
        let mut im_acc = Expr::Const(0.0);
        for j in 0..n {
            // angle for e^{-2πi·j·k/N}
            let angle = -2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
            let cos_val = angle.cos();
            let sin_val = angle.sin(); // = -sin(2π·j·k/N)
            let xj = Expr::input_re(j);
            re_acc = re_acc.add(xj.clone().mul(Expr::Const(cos_val)));
            im_acc = im_acc.add(xj.mul(Expr::Const(sin_val)));
        }
        let re_red = ConstantFolder::fold(&StrengthReducer::reduce(&re_acc));
        let im_red = ConstantFolder::fold(&StrengthReducer::reduce(&im_acc));
        outputs.push((re_red, im_red));
    }
    outputs
}

/// Emit the full R2HC codelet function as a `TokenStream`.
fn emit_r2hc_codelet(n: usize) -> TokenStream {
    let outputs = symbolic_r2hc(n); // len = n/2 + 1
    let half = n / 2;
    let min_out = half + 1;
    let fn_name = format_ident!("r2hc_{n}_gen");
    let body = emit_r2hc_body(n, &outputs);

    quote! {
        /// Generated R2HC (real to half-complex) codelet.
        ///
        /// Input: `x` — N real samples.
        /// Output: `y` — N/2+1 complex bins Y[0]..Y[N/2].
        #[inline(always)]
        #[allow(clippy::too_many_lines, clippy::approx_constant, clippy::suboptimal_flops)]
        pub fn #fn_name<T: crate::kernel::Float>(x: &[T], y: &mut [crate::kernel::Complex<T>]) {
            debug_assert_eq!(x.len(), #n);
            debug_assert!(y.len() >= #min_out);
            #body
        }
    }
}

/// Build the body for R2HC.
fn emit_r2hc_body(n: usize, outputs: &[(Expr, Expr)]) -> TokenStream {
    // Collect all expressions for CSE
    let all_exprs: Vec<&Expr> = outputs.iter().flat_map(|(re, im)| [re, im]).collect();

    let mut cse = LocalCse::new();
    for expr in &all_exprs {
        cse.count_recursive(expr);
    }

    let mut body = TokenStream::new();

    // Input extraction: `let x0 = x[0];` ...
    for i in 0..n {
        let var = format_ident!("x{i}");
        body.extend(quote! { let #var = x[#i]; });
    }

    // CSE temporaries: emit original expr (no sub-CSE of assignment bodies).
    // Assignments are topologically sorted by Temp-ref dependencies.
    // Since the original exprs contain no Temp nodes (they are raw symbolic trees),
    // topological_sort leaves them in name order (t0, t1, ...) which is correct
    // because count_recursive assigns names in traversal order.
    let assignments = cse.get_assignments();
    for (name, expr) in &assignments {
        let id = format_ident!("{name}");
        // Emit the original expr without further CSE rewriting to avoid
        // self-referential or forward Temp dependencies.
        let tok = emit_real_scalar(expr);
        body.extend(quote! { let #id = #tok; });
    }

    // Output assignments: `y[k] = Complex::new(re, im);`
    for (k, (re_expr, im_expr)) in outputs.iter().enumerate() {
        let re_tok = emit_real_scalar(&cse.rewrite(re_expr));
        let im_tok = emit_real_scalar(&cse.rewrite(im_expr));
        body.extend(quote! {
            y[#k] = crate::kernel::Complex::new(#re_tok, #im_tok);
        });
    }

    body
}

// ============================================================================
// HC2R generation
// ============================================================================

fn gen_hc2r(n: usize) -> Result<TokenStream, syn::Error> {
    match n {
        2 | 4 | 8 => Ok(emit_hc2r_codelet(n)),
        _ => Err(syn::Error::new(
            Span::call_site(),
            format!("gen_rdft_codelet: unsupported size {n} for Hc2r (expected 2, 4, or 8)"),
        )),
    }
}

/// Build symbolic HC2R expressions for all j in 0..n.
///
/// Unnormalized inverse (no 1/N factor):
///   x[j] = Y[0].re
///         + Y[N/2].re · cos(π·j)                              (Nyquist)
///         + 2 · Σ_{k=1..N/2-1} (Y[k].re·cos(2π·j·k/N) - Y[k].im·sin(2π·j·k/N))
///
/// Input indices: `Expr::input_re(k)` = Y[k].re, `Expr::input_im(k)` = Y[k].im
/// (only k in 0..=N/2 are provided as inputs).
fn symbolic_hc2r(n: usize) -> Vec<Expr> {
    let half = n / 2;
    let mut outputs = Vec::with_capacity(n);

    for j in 0..n {
        // DC component: Y[0].re (always 1× contribution)
        let mut acc = Expr::input_re(0);

        // Interior bins k=1..N/2-1: factor-of-2 from conjugate symmetry
        for k in 1..half {
            let angle = 2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let yk_re = Expr::input_re(k);
            let yk_im = Expr::input_im(k);

            let term_re = yk_re.mul(Expr::Const(cos_val));
            let term_im = yk_im.mul(Expr::Const(sin_val));
            let term = term_re.sub(term_im);
            // Multiply by 2 for the conjugate symmetric pair
            let term2 = term.mul(Expr::Const(2.0));
            acc = acc.add(term2);
        }

        // Nyquist bin: Y[N/2].re · cos(π·j) = Y[N/2].re · (-1)^j
        let nyquist_angle = std::f64::consts::PI * j as f64;
        let nyquist_cos = nyquist_angle.cos(); // exactly +1.0 or -1.0
        let nyquist_term = Expr::input_re(half).mul(Expr::Const(nyquist_cos));
        acc = acc.add(nyquist_term);

        let reduced = ConstantFolder::fold(&StrengthReducer::reduce(&acc));
        outputs.push(reduced);
    }
    outputs
}

/// Emit the full HC2R codelet function as a `TokenStream`.
fn emit_hc2r_codelet(n: usize) -> TokenStream {
    let outputs = symbolic_hc2r(n);
    let half = n / 2;
    let min_in = half + 1;
    let fn_name = format_ident!("hc2r_{n}_gen");
    let body = emit_hc2r_body(n, &outputs, half);

    quote! {
        /// Generated HC2R (half-complex to real) codelet.
        ///
        /// Input: `y` — N/2+1 complex bins Y[0]..Y[N/2].
        /// Output: `x` — N real samples (unnormalized; caller divides by N).
        #[inline(always)]
        #[allow(clippy::too_many_lines, clippy::approx_constant, clippy::suboptimal_flops)]
        pub fn #fn_name<T: crate::kernel::Float>(y: &[crate::kernel::Complex<T>], x: &mut [T]) {
            debug_assert!(y.len() >= #min_in);
            debug_assert_eq!(x.len(), #n);
            #body
        }
    }
}

/// Build the body for HC2R.
fn emit_hc2r_body(_n: usize, outputs: &[Expr], half: usize) -> TokenStream {
    let mut cse = LocalCse::new();
    for expr in outputs {
        cse.count_recursive(expr);
    }

    let mut body = TokenStream::new();

    // Input extraction: `let y0_re = y[0].re; let y0_im = y[0].im;` ...
    for k in 0..=half {
        let re_var = format_ident!("y{k}_re");
        let im_var = format_ident!("y{k}_im");
        body.extend(quote! {
            let #re_var = y[#k].re;
            let #im_var = y[#k].im;
        });
    }

    // CSE temporaries: emit original exprs directly (no sub-CSE of bodies).
    let assignments = cse.get_assignments();
    for (name, expr) in &assignments {
        let id = format_ident!("{name}");
        let tok = emit_hc2r_scalar(expr);
        body.extend(quote! { let #id = #tok; });
    }

    // Output assignments: `x[j] = <expr>;`
    for (j, expr) in outputs.iter().enumerate() {
        let val_tok = emit_hc2r_scalar(&cse.rewrite(expr));
        body.extend(quote! { x[#j] = #val_tok; });
    }

    body
}

// ============================================================================
// Local CSE (recursive, mirrors symbolic_emit::RecursiveCse)
// ============================================================================

/// Local recursive CSE for RDFT expressions.
///
/// Counts subexpression usages across all outputs, then rewrites
/// shared subexpressions (used ≥ 2 times) as `Temp` references.
struct LocalCse {
    /// `structural_hash` → (original expr, temp name, use count)
    cache: HashMap<u64, (Expr, String, usize)>,
    counter: usize,
}

impl LocalCse {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
            counter: 0,
        }
    }

    /// Count subexpression usages (bottom-up traversal).
    fn count_recursive(&mut self, expr: &Expr) {
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

    /// Rewrite an expression, replacing shared subexpressions with `Temp` refs.
    fn rewrite(&self, expr: &Expr) -> Expr {
        self.rewrite_inner(expr, None)
    }

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

    /// Return sorted assignments for temps used ≥ 2 times.
    fn get_assignments(&self) -> Vec<(String, Expr)> {
        let mut result: Vec<(String, Expr)> = self
            .cache
            .values()
            .filter(|(_, _, count)| *count >= 2)
            .map(|(expr, name, _)| (name.clone(), expr.clone()))
            .collect();
        result.sort_by(|a, b| {
            let na: usize = a.0[1..].parse().unwrap_or(0);
            let nb: usize = b.0[1..].parse().unwrap_or(0);
            na.cmp(&nb)
        });
        result
    }
}

// ============================================================================
// Scalar emission
// ============================================================================

/// Emit a scalar `Expr` for R2HC.
///
/// `Input { index, is_real: true }` → `x{index}` (real input)
/// `Input { index, is_real: false }` → should not occur in R2HC; emits
///   `y{index}_im` as a fallback (occurs in shared emitter paths only).
fn emit_real_scalar(expr: &Expr) -> TokenStream {
    match expr {
        Expr::Input { index, is_real } => {
            if *is_real {
                let name = format_ident!("x{index}");
                quote! { #name }
            } else {
                // Not expected in R2HC but safe fallback
                let name = format_ident!("y{index}_im");
                quote! { #name }
            }
        }
        Expr::Const(v) => emit_const(*v),
        Expr::Add(a, b) => {
            let a = emit_real_scalar(a);
            let b = emit_real_scalar(b);
            quote! { (#a + #b) }
        }
        Expr::Sub(a, b) => {
            let a = emit_real_scalar(a);
            let b = emit_real_scalar(b);
            quote! { (#a - #b) }
        }
        Expr::Mul(a, b) => {
            let a = emit_real_scalar(a);
            let b = emit_real_scalar(b);
            quote! { (#a * #b) }
        }
        Expr::Neg(a) => {
            let a = emit_real_scalar(a);
            quote! { (-#a) }
        }
        Expr::Temp(name) => {
            let id = format_ident!("{name}");
            quote! { #id }
        }
    }
}

/// Emit a scalar `Expr` for HC2R (complex inputs `y{k}_re` / `y{k}_im`).
fn emit_hc2r_scalar(expr: &Expr) -> TokenStream {
    match expr {
        Expr::Input { index, is_real } => {
            let name = if *is_real {
                format_ident!("y{index}_re")
            } else {
                format_ident!("y{index}_im")
            };
            quote! { #name }
        }
        Expr::Const(v) => emit_const(*v),
        Expr::Add(a, b) => {
            let a = emit_hc2r_scalar(a);
            let b = emit_hc2r_scalar(b);
            quote! { (#a + #b) }
        }
        Expr::Sub(a, b) => {
            let a = emit_hc2r_scalar(a);
            let b = emit_hc2r_scalar(b);
            quote! { (#a - #b) }
        }
        Expr::Mul(a, b) => {
            let a = emit_hc2r_scalar(a);
            let b = emit_hc2r_scalar(b);
            quote! { (#a * #b) }
        }
        Expr::Neg(a) => {
            let a = emit_hc2r_scalar(a);
            quote! { (-#a) }
        }
        Expr::Temp(name) => {
            let id = format_ident!("{name}");
            quote! { #id }
        }
    }
}

/// Emit a constant value as `T::ZERO`, `T::ONE`, `(-T::ONE)`, or `T::from_f64(v)`.
fn emit_const(v: f64) -> TokenStream {
    if (v - 0.0_f64).abs() < f64::EPSILON {
        quote! { T::ZERO }
    } else if (v - 1.0_f64).abs() < f64::EPSILON {
        quote! { T::ONE }
    } else if (v - (-1.0_f64)).abs() < f64::EPSILON {
        quote! { (-T::ONE) }
    } else {
        quote! { T::from_f64(#v) }
    }
}
