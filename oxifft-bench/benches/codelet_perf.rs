//! Codelet performance benchmarks.
//!
//! Benchmarks the symbolic-optimized notw codelets for sizes 2/4/8/16/32/64
//! in both f32 and f64 precision.
//!
//! Cross-version comparison (generated vs previous hard-coded templates) is
//! done by running these benchmarks on the `master` and `0.3.0` branches
//! separately and comparing criterion output.
//!
//! To run: `cargo bench -p oxifft-bench --bench codelet_perf`

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::approx_constant)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::similar_names)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxifft_codegen::gen_notw_codelet;
use std::hint::black_box;

// ============================================================================
// Minimal kernel types needed by generated codelets.
//
// The generated code uses `crate::kernel::Float` and `crate::kernel::Complex<T>`.
// We define a `kernel` module here so the generated code can resolve those paths.
// ============================================================================

pub mod kernel {
    use core::fmt;
    use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

    pub trait Float:
        Copy
        + Clone
        + Default
        + fmt::Debug
        + Send
        + Sync
        + PartialOrd
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
        + Neg<Output = Self>
        + num_traits::NumAssign
        + num_traits::Float
        + num_traits::FloatConst
        + 'static
    {
        const ZERO: Self;
        const ONE: Self;
        const TWO: Self;
        const PI: Self;
        const TWO_PI: Self;

        #[must_use]
        fn sin(self) -> Self;
        #[must_use]
        fn cos(self) -> Self;
        #[must_use]
        fn sin_cos(self) -> (Self, Self);
        #[must_use]
        fn sqrt(self) -> Self;
        #[must_use]
        fn abs(self) -> Self;
        #[must_use]
        fn from_usize(n: usize) -> Self;
        #[must_use]
        fn from_isize(n: isize) -> Self;
        #[must_use]
        fn from_f64(n: f64) -> Self;
    }

    impl Float for f32 {
        const ZERO: Self = 0.0;
        const ONE: Self = 1.0;
        const TWO: Self = 2.0;
        const PI: Self = core::f32::consts::PI;
        const TWO_PI: Self = core::f32::consts::TAU;

        fn sin(self) -> Self {
            num_traits::Float::sin(self)
        }
        fn cos(self) -> Self {
            num_traits::Float::cos(self)
        }
        fn sin_cos(self) -> (Self, Self) {
            num_traits::Float::sin_cos(self)
        }
        fn sqrt(self) -> Self {
            num_traits::Float::sqrt(self)
        }
        fn abs(self) -> Self {
            num_traits::Float::abs(self)
        }
        fn from_usize(n: usize) -> Self {
            n as Self
        }
        fn from_isize(n: isize) -> Self {
            n as Self
        }
        fn from_f64(n: f64) -> Self {
            n as Self
        }
    }

    impl Float for f64 {
        const ZERO: Self = 0.0;
        const ONE: Self = 1.0;
        const TWO: Self = 2.0;
        const PI: Self = core::f64::consts::PI;
        const TWO_PI: Self = core::f64::consts::TAU;

        fn sin(self) -> Self {
            num_traits::Float::sin(self)
        }
        fn cos(self) -> Self {
            num_traits::Float::cos(self)
        }
        fn sin_cos(self) -> (Self, Self) {
            num_traits::Float::sin_cos(self)
        }
        fn sqrt(self) -> Self {
            num_traits::Float::sqrt(self)
        }
        fn abs(self) -> Self {
            num_traits::Float::abs(self)
        }
        fn from_usize(n: usize) -> Self {
            n as Self
        }
        fn from_isize(n: isize) -> Self {
            n as Self
        }
        fn from_f64(n: f64) -> Self {
            n
        }
    }

    #[derive(Copy, Clone, Default, PartialEq)]
    #[repr(C)]
    pub struct Complex<T: Float> {
        pub re: T,
        pub im: T,
    }

    impl<T: Float> Complex<T> {
        #[inline]
        pub const fn new(re: T, im: T) -> Self {
            Self { re, im }
        }

        #[inline]
        pub fn zero() -> Self {
            Self::new(T::ZERO, T::ZERO)
        }
    }

    impl<T: Float> fmt::Debug for Complex<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{:?}+{:?}i", self.re, self.im)
        }
    }

    impl<T: Float> Add for Complex<T> {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self::new(self.re + rhs.re, self.im + rhs.im)
        }
    }
    impl<T: Float> AddAssign for Complex<T> {
        fn add_assign(&mut self, rhs: Self) {
            self.re += rhs.re;
            self.im += rhs.im;
        }
    }
    impl<T: Float> Sub for Complex<T> {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self::new(self.re - rhs.re, self.im - rhs.im)
        }
    }
    impl<T: Float> SubAssign for Complex<T> {
        fn sub_assign(&mut self, rhs: Self) {
            self.re -= rhs.re;
            self.im -= rhs.im;
        }
    }
    impl<T: Float> Mul for Complex<T> {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self::new(
                self.re * rhs.re - self.im * rhs.im,
                self.re * rhs.im + self.im * rhs.re,
            )
        }
    }
    impl<T: Float> MulAssign for Complex<T> {
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }
    impl<T: Float> Div for Complex<T> {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            let norm_sq = rhs.re * rhs.re + rhs.im * rhs.im;
            Self::new(
                (self.re * rhs.re + self.im * rhs.im) / norm_sq,
                (self.im * rhs.re - self.re * rhs.im) / norm_sq,
            )
        }
    }
    impl<T: Float> DivAssign for Complex<T> {
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }
    impl<T: Float> Neg for Complex<T> {
        type Output = Self;
        fn neg(self) -> Self {
            Self::new(-self.re, -self.im)
        }
    }
    impl<T: Float> Mul<T> for Complex<T> {
        type Output = Self;
        fn mul(self, rhs: T) -> Self {
            Self::new(self.re * rhs, self.im * rhs)
        }
    }
}

// ============================================================================
// Generate all codelets via the symbolic optimization pipeline.
// ============================================================================

gen_notw_codelet!(2);
gen_notw_codelet!(4);
gen_notw_codelet!(8);
gen_notw_codelet!(16);
gen_notw_codelet!(32);
gen_notw_codelet!(64);

// ============================================================================
// Benchmark helpers
// ============================================================================

fn make_data_f64(n: usize) -> Vec<kernel::Complex<f64>> {
    (0..n)
        .map(|i| kernel::Complex::new((i as f64).sin(), (i as f64).cos()))
        .collect()
}

fn make_data_f32(n: usize) -> Vec<kernel::Complex<f32>> {
    (0..n)
        .map(|i| kernel::Complex::new((i as f32).sin(), (i as f32).cos()))
        .collect()
}

// ============================================================================
// Benchmarks — f64, forward
// ============================================================================

fn bench_notw_f64_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("codelet_notw_f64_forward");

    let mut data2 = make_data_f64(2);
    group.throughput(Throughput::Elements(2));
    group.bench_function(BenchmarkId::new("notw", 2), |b| {
        b.iter(|| codelet_notw_2(black_box(&mut data2), -1));
    });

    let mut data4 = make_data_f64(4);
    group.throughput(Throughput::Elements(4));
    group.bench_function(BenchmarkId::new("notw", 4), |b| {
        b.iter(|| codelet_notw_4(black_box(&mut data4), -1));
    });

    let mut data8 = make_data_f64(8);
    group.throughput(Throughput::Elements(8));
    group.bench_function(BenchmarkId::new("notw", 8), |b| {
        b.iter(|| codelet_notw_8(black_box(&mut data8), -1));
    });

    let mut data16 = make_data_f64(16);
    group.throughput(Throughput::Elements(16));
    group.bench_function(BenchmarkId::new("notw", 16), |b| {
        b.iter(|| codelet_notw_16(black_box(&mut data16), -1));
    });

    let mut data32 = make_data_f64(32);
    group.throughput(Throughput::Elements(32));
    group.bench_function(BenchmarkId::new("notw", 32), |b| {
        b.iter(|| codelet_notw_32(black_box(&mut data32), -1));
    });

    let mut data64 = make_data_f64(64);
    group.throughput(Throughput::Elements(64));
    group.bench_function(BenchmarkId::new("notw", 64), |b| {
        b.iter(|| codelet_notw_64(black_box(&mut data64), -1));
    });

    group.finish();
}

// ============================================================================
// Benchmarks — f32, forward
// ============================================================================

fn bench_notw_f32_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("codelet_notw_f32_forward");

    let mut data2 = make_data_f32(2);
    group.throughput(Throughput::Elements(2));
    group.bench_function(BenchmarkId::new("notw", 2), |b| {
        b.iter(|| codelet_notw_2(black_box(&mut data2), -1));
    });

    let mut data4 = make_data_f32(4);
    group.throughput(Throughput::Elements(4));
    group.bench_function(BenchmarkId::new("notw", 4), |b| {
        b.iter(|| codelet_notw_4(black_box(&mut data4), -1));
    });

    let mut data8 = make_data_f32(8);
    group.throughput(Throughput::Elements(8));
    group.bench_function(BenchmarkId::new("notw", 8), |b| {
        b.iter(|| codelet_notw_8(black_box(&mut data8), -1));
    });

    let mut data16 = make_data_f32(16);
    group.throughput(Throughput::Elements(16));
    group.bench_function(BenchmarkId::new("notw", 16), |b| {
        b.iter(|| codelet_notw_16(black_box(&mut data16), -1));
    });

    let mut data32 = make_data_f32(32);
    group.throughput(Throughput::Elements(32));
    group.bench_function(BenchmarkId::new("notw", 32), |b| {
        b.iter(|| codelet_notw_32(black_box(&mut data32), -1));
    });

    let mut data64 = make_data_f32(64);
    group.throughput(Throughput::Elements(64));
    group.bench_function(BenchmarkId::new("notw", 64), |b| {
        b.iter(|| codelet_notw_64(black_box(&mut data64), -1));
    });

    group.finish();
}

criterion_group!(benches, bench_notw_f64_forward, bench_notw_f32_forward);
criterion_main!(benches);
