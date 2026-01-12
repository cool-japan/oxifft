//! Symbolic FFT operation representation.
//!
//! This module provides a symbolic representation of FFT operations that can be
//! optimized through common subexpression elimination (CSE) and strength reduction
//! before code generation.
//!
//! Note: These types are infrastructure for the codegen proc-macros and are tested
//! but not directly exported (proc-macro crates can only export proc-macro functions).

#![allow(dead_code)] // Infrastructure types used by tests and codegen
#![allow(clippy::cast_precision_loss)] // FFT sizes fit comfortably in f64 mantissa

use std::collections::HashMap;
use std::fmt;

/// A symbolic expression representing FFT operations.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// Input variable: x[index].re or x[index].im
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
    pub const fn input_re(index: usize) -> Self {
        Self::Input {
            index,
            is_real: true,
        }
    }

    /// Create an imaginary input reference.
    pub const fn input_im(index: usize) -> Self {
        Self::Input {
            index,
            is_real: false,
        }
    }

    /// Create a constant.
    pub const fn constant(value: f64) -> Self {
        Self::Const(value)
    }

    /// Create addition.
    pub fn add(self, other: Self) -> Self {
        Self::Add(Box::new(self), Box::new(other))
    }

    /// Create subtraction.
    pub fn sub(self, other: Self) -> Self {
        Self::Sub(Box::new(self), Box::new(other))
    }

    /// Create multiplication.
    pub fn mul(self, other: Self) -> Self {
        Self::Mul(Box::new(self), Box::new(other))
    }

    /// Create negation.
    pub fn neg(self) -> Self {
        Self::Neg(Box::new(self))
    }

    /// Check if this is a constant.
    pub const fn is_const(&self) -> bool {
        matches!(self, Self::Const(_))
    }

    /// Get constant value if this is a constant.
    pub const fn const_value(&self) -> Option<f64> {
        match self {
            Self::Const(v) => Some(*v),
            _ => None,
        }
    }

    /// Hash the expression for CSE.
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

    /// Count operations in this expression.
    pub fn op_count(&self) -> usize {
        match self {
            Self::Input { .. } | Self::Const(_) | Self::Temp(_) => 0,
            Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b) => 1 + a.op_count() + b.op_count(),
            Self::Neg(a) => 1 + a.op_count(),
        }
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
    pub const fn input(index: usize) -> Self {
        Self {
            re: Expr::input_re(index),
            im: Expr::input_im(index),
        }
    }

    /// Create from constant.
    pub const fn constant(re: f64, im: f64) -> Self {
        Self {
            re: Expr::constant(re),
            im: Expr::constant(im),
        }
    }

    /// Complex addition.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            re: self.re.clone().add(other.re.clone()),
            im: self.im.clone().add(other.im.clone()),
        }
    }

    /// Complex subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            re: self.re.clone().sub(other.re.clone()),
            im: self.im.clone().sub(other.im.clone()),
        }
    }

    /// Complex multiplication.
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

    /// Multiply by j = sqrt(-1).
    pub fn mul_j(&self) -> Self {
        // (a + bi) * i = -b + ai
        Self {
            re: self.im.clone().neg(),
            im: self.re.clone(),
        }
    }

    /// Multiply by -j = -sqrt(-1).
    pub fn mul_neg_j(&self) -> Self {
        // (a + bi) * (-i) = b - ai
        Self {
            re: self.im.clone(),
            im: self.re.clone().neg(),
        }
    }

    /// Negation.
    pub fn neg(&self) -> Self {
        Self {
            re: self.re.clone().neg(),
            im: self.im.clone().neg(),
        }
    }
}

/// Common Subexpression Elimination optimizer.
pub struct CseOptimizer {
    /// Map from expression hash to (expression, temp name, use count).
    expr_cache: HashMap<u64, (Expr, String, usize)>,
    /// Counter for generating temp names.
    temp_counter: usize,
    /// Threshold for CSE (min uses to create temp).
    min_uses: usize,
}

impl CseOptimizer {
    /// Create a new CSE optimizer.
    pub fn new() -> Self {
        Self {
            expr_cache: HashMap::new(),
            temp_counter: 0,
            min_uses: 2,
        }
    }

    /// Set minimum uses threshold for CSE.
    pub const fn with_min_uses(mut self, min_uses: usize) -> Self {
        self.min_uses = min_uses;
        self
    }

    /// Register an expression and return the optimized version.
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

/// FFT symbolic computation.
pub struct SymbolicFFT {
    /// Size of the FFT.
    pub n: usize,
    /// Output expressions (real, imag pairs).
    pub outputs: Vec<ComplexExpr>,
}

impl SymbolicFFT {
    /// Generate symbolic DFT for size n.
    pub fn dft(n: usize, forward: bool) -> Self {
        let sign = if forward { -1.0 } else { 1.0 };
        let mut outputs = Vec::with_capacity(n);

        // Generate DFT symbolically
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

        Self { n, outputs }
    }

    /// Generate radix-2 Cooley-Tukey FFT symbolically.
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

        Self { n, outputs }
    }

    /// Total operation count.
    pub fn op_count(&self) -> usize {
        self.outputs
            .iter()
            .map(|c| c.re.op_count() + c.im.op_count())
            .sum()
    }
}

#[cfg(test)]
mod tests {
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
        assert_eq!(fft.n, 2);
        assert_eq!(fft.outputs.len(), 2);
    }

    #[test]
    fn test_symbolic_radix2_size_4() {
        let fft = SymbolicFFT::radix2_dit(4, true);
        assert_eq!(fft.n, 4);
        assert_eq!(fft.outputs.len(), 4);
    }
}
