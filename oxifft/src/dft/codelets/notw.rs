//! Non-twiddle (base case) codelets.
//!
//! These are the leaf-level DFT kernels that don't require twiddle factors.

use crate::kernel::{Complex, Float};

/// Size-2 DFT (butterfly).
#[inline]
pub fn notw_2<T: Float>(x: &mut [Complex<T>]) {
    debug_assert!(x.len() >= 2);
    let a = x[0];
    let b = x[1];
    x[0] = a + b;
    x[1] = a - b;
}

/// Size-3 DFT.
///
/// Uses Winograd's algorithm for minimal multiplications.
/// Twiddle factors: W_3 = e^{-2πi/3} = -1/2 - i*(√3/2) for forward.
#[inline]
pub fn notw_3<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 3);

    // Constants for size-3 DFT
    // cos(2π/3) = -1/2
    // sin(2π/3) = √3/2 ≈ 0.8660254037844386
    let c = T::from_f64(-0.5);
    let s = T::from_f64(0.866_025_403_784_438_6);

    let x0 = x[0];
    let x1 = x[1];
    let x2 = x[2];

    // Sum of all inputs
    let t0 = x1 + x2;
    let t1 = x0 + t0 * c; // x0 + (x1+x2)*cos(2π/3)

    // Difference term for imaginary rotation
    let t2 = x1 - x2;

    // Apply ±i*sin(2π/3)*(x1-x2)
    // For forward (sign < 0): -i*sin means (im, -re)
    // For inverse (sign > 0): +i*sin means (-im, re)
    let t2_rot = if sign < 0 {
        Complex::new(t2.im * s, -t2.re * s)
    } else {
        Complex::new(-t2.im * s, t2.re * s)
    };

    // Output
    x[0] = x0 + t0; // DC component
    x[1] = t1 + t2_rot;
    x[2] = t1 - t2_rot;
}

/// Size-5 DFT.
///
/// Uses Rader/Winograd-style algorithm for reduced operation count.
/// Twiddle factors: W_5^k = e^{-2πik/5} for forward.
#[inline]
pub fn notw_5<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 5);

    // Constants for size-5 DFT
    // cos(2π/5) ≈ 0.309016994374947
    // cos(4π/5) ≈ -0.809016994374947
    // sin(2π/5) ≈ 0.951056516295154
    // sin(4π/5) ≈ 0.587785252292473
    let c1 = T::from_f64(0.309_016_994_374_947_4); // cos(2π/5)
    let c2 = T::from_f64(-0.809_016_994_374_947_4); // cos(4π/5)
    let s1 = T::from_f64(0.951_056_516_295_153_5); // sin(2π/5)
    let s2 = T::from_f64(0.587_785_252_292_473_1); // sin(4π/5)

    let x0 = x[0];
    let x1 = x[1];
    let x2 = x[2];
    let x3 = x[3];
    let x4 = x[4];

    // Symmetric combinations
    let a1 = x1 + x4; // x[1] + x[4]
    let a2 = x2 + x3; // x[2] + x[3]
    let b1 = x1 - x4; // x[1] - x[4]
    let b2 = x2 - x3; // x[2] - x[3]

    // Real parts of outputs (before adding x0)
    let r1 = a1 * c1 + a2 * c2; // cos(2π/5)*(x1+x4) + cos(4π/5)*(x2+x3)
    let r2 = a1 * c2 + a2 * c1; // cos(4π/5)*(x1+x4) + cos(2π/5)*(x2+x3)

    // Imaginary rotation terms
    // For forward: multiply by -i*sin
    // For inverse: multiply by +i*sin
    let t1 = b1 * s1 + b2 * s2;
    let t2 = b1 * s2 - b2 * s1;

    let (i1, i2) = if sign < 0 {
        // Forward: -i * (sin(2π/5)*b1 + sin(4π/5)*b2)
        (Complex::new(t1.im, -t1.re), Complex::new(t2.im, -t2.re))
    } else {
        // Inverse: +i * (sin(2π/5)*b1 + sin(4π/5)*b2)
        (Complex::new(-t1.im, t1.re), Complex::new(-t2.im, t2.re))
    };

    // DC component
    x[0] = x0 + a1 + a2;

    // Other outputs
    x[1] = x0 + r1 + i1;
    x[4] = x0 + r1 - i1;
    x[2] = x0 + r2 + i2;
    x[3] = x0 + r2 - i2;
}

/// Size-7 DFT.
///
/// Uses Rader/Winograd-style algorithm for reduced operation count.
/// Twiddle factors: W_7^k = e^{-2πik/7} for forward.
#[inline]
pub fn notw_7<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 7);

    // Constants for size-7 DFT
    // cos(2π/7), cos(4π/7), cos(6π/7)
    // sin(2π/7), sin(4π/7), sin(6π/7)
    let c1 = T::from_f64(0.623_489_801_858_734); // cos(2π/7)
    let c2 = T::from_f64(-0.222_520_933_956_314); // cos(4π/7)
    let c3 = T::from_f64(-0.900_968_867_902_419); // cos(6π/7)
    let s1 = T::from_f64(0.781_831_482_468_03); // sin(2π/7)
    let s2 = T::from_f64(0.974_927_912_181_824); // sin(4π/7)
    let s3 = T::from_f64(0.433_883_739_117_558); // sin(6π/7)

    let x0 = x[0];
    let x1 = x[1];
    let x2 = x[2];
    let x3 = x[3];
    let x4 = x[4];
    let x5 = x[5];
    let x6 = x[6];

    // Symmetric combinations
    let a1 = x1 + x6;
    let a2 = x2 + x5;
    let a3 = x3 + x4;
    let b1 = x1 - x6;
    let b2 = x2 - x5;
    let b3 = x3 - x4;

    // Real parts
    let r1 = a1 * c1 + a2 * c2 + a3 * c3;
    let r2 = a1 * c2 + a2 * c3 + a3 * c1;
    let r3 = a1 * c3 + a2 * c1 + a3 * c2;

    // Imaginary rotation terms
    let t1 = b1 * s1 + b2 * s2 + b3 * s3;
    let t2 = b1 * s2 - b2 * s3 - b3 * s1;
    let t3 = b1 * s3 - b2 * s1 + b3 * s2;

    let (i1, i2, i3) = if sign < 0 {
        (
            Complex::new(t1.im, -t1.re),
            Complex::new(t2.im, -t2.re),
            Complex::new(t3.im, -t3.re),
        )
    } else {
        (
            Complex::new(-t1.im, t1.re),
            Complex::new(-t2.im, t2.re),
            Complex::new(-t3.im, t3.re),
        )
    };

    // DC component
    x[0] = x0 + a1 + a2 + a3;

    // Other outputs
    x[1] = x0 + r1 + i1;
    x[6] = x0 + r1 - i1;
    x[2] = x0 + r2 + i2;
    x[5] = x0 + r2 - i2;
    x[3] = x0 + r3 + i3;
    x[4] = x0 + r3 - i3;
}

/// Size-4 DFT.
#[inline]
pub fn notw_4<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 4);

    let x0 = x[0];
    let x1 = x[1];
    let x2 = x[2];
    let x3 = x[3];

    // First stage
    let t0 = x0 + x2;
    let t1 = x0 - x2;
    let t2 = x1 + x3;
    let t3 = x1 - x3;

    // Apply -i or +i rotation based on sign
    let t3_rot = if sign < 0 {
        Complex::new(t3.im, -t3.re)
    } else {
        Complex::new(-t3.im, t3.re)
    };

    // Second stage
    x[0] = t0 + t2;
    x[2] = t0 - t2;
    x[1] = t1 + t3_rot;
    x[3] = t1 - t3_rot;
}

/// Size-8 DFT.
///
/// Uses a radix-2 decimation-in-frequency approach with manually unrolled butterflies.
/// Twiddle factors are W8^k = e^(-2πik/8) for forward, e^(+2πik/8) for inverse.
#[inline]
pub fn notw_8<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 8);

    // Constants: cos(π/4) = sin(π/4) = 1/sqrt(2) ≈ 0.7071067811865476
    let sqrt2_2 = T::from_f64(core::f64::consts::FRAC_1_SQRT_2);

    // Load all inputs
    let x0 = x[0];
    let x1 = x[1];
    let x2 = x[2];
    let x3 = x[3];
    let x4 = x[4];
    let x5 = x[5];
    let x6 = x[6];
    let x7 = x[7];

    // Stage 1: 4 radix-2 butterflies (no twiddles)
    let t0 = x0 + x4;
    let t1 = x0 - x4;
    let t2 = x2 + x6;
    let t3 = x2 - x6;
    let t4 = x1 + x5;
    let t5 = x1 - x5;
    let t6 = x3 + x7;
    let t7 = x3 - x7;

    // Stage 2: Apply twiddles and 4 more butterflies
    // W8^0 = 1
    // W8^1 = (1-i)/sqrt(2) for forward, (1+i)/sqrt(2) for inverse
    // W8^2 = -i for forward, +i for inverse
    // W8^3 = (-1-i)/sqrt(2) for forward, (-1+i)/sqrt(2) for inverse

    // t3 *= W8^2 = -i (forward) or +i (inverse)
    let t3_rot = if sign < 0 {
        Complex::new(t3.im, -t3.re) // multiply by -i
    } else {
        Complex::new(-t3.im, t3.re) // multiply by +i
    };

    // t5 *= W8^1
    let t5_rot = if sign < 0 {
        // W8^1 = e^(-iπ/4) = (1-i)/sqrt(2)
        Complex::new((t5.re + t5.im) * sqrt2_2, (-t5.re + t5.im) * sqrt2_2)
    } else {
        // W8^(-1) = e^(+iπ/4) = (1+i)/sqrt(2)
        Complex::new((t5.re - t5.im) * sqrt2_2, (t5.re + t5.im) * sqrt2_2)
    };

    // t7 *= W8^3
    let t7_rot = if sign < 0 {
        // W8^3 = e^(-3iπ/4) = (-1-i)/sqrt(2)
        Complex::new((-t7.re + t7.im) * sqrt2_2, (-t7.re - t7.im) * sqrt2_2)
    } else {
        // W8^(-3) = e^(+3iπ/4) = (-1+i)/sqrt(2)
        Complex::new((-t7.re - t7.im) * sqrt2_2, (t7.re - t7.im) * sqrt2_2)
    };

    // 4 more butterflies
    let u0 = t0 + t2;
    let u1 = t0 - t2;
    let u2 = t4 + t6;
    let u3 = t4 - t6;
    let u4 = t1 + t3_rot;
    let u5 = t1 - t3_rot;
    let u6 = t5_rot + t7_rot;
    let u7 = t5_rot - t7_rot;

    // Stage 3: Final butterflies with twiddle W4^k for second half
    // u3 *= W4^1 = -i (forward) or +i (inverse)
    let u3_rot = if sign < 0 {
        Complex::new(u3.im, -u3.re)
    } else {
        Complex::new(-u3.im, u3.re)
    };

    // u7 *= W4^1 = -i (forward) or +i (inverse)
    let u7_rot = if sign < 0 {
        Complex::new(u7.im, -u7.re)
    } else {
        Complex::new(-u7.im, u7.re)
    };

    // Final outputs in bit-reversed order, then reorder
    // DIF produces bit-reversed output, so we need to reorder to natural order
    let y0 = u0 + u2; // X[0]
    let y4 = u0 - u2; // X[4]
    let y2 = u1 + u3_rot; // X[2]
    let y6 = u1 - u3_rot; // X[6]
    let y1 = u4 + u6; // X[1]
    let y5 = u4 - u6; // X[5]
    let y3 = u5 + u7_rot; // X[3]
    let y7 = u5 - u7_rot; // X[7]

    x[0] = y0;
    x[1] = y1;
    x[2] = y2;
    x[3] = y3;
    x[4] = y4;
    x[5] = y5;
    x[6] = y6;
    x[7] = y7;
}

/// Size-16 DFT.
///
/// Uses a standard Cooley-Tukey radix-2 DIT algorithm with explicit stages.
/// This delegates to two size-8 sub-transforms for correctness.
#[inline]
pub fn notw_16<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 16);

    // Twiddle factors W16^k = e^(-2πik/16) for forward (sign < 0)
    // or W16^(-k) = e^(+2πik/16) for inverse (sign > 0)
    let c1 = T::from_f64(0.923_879_532_511_286_7); // cos(π/8)
    let s1 = T::from_f64(0.382_683_432_365_089_8); // sin(π/8)
    let c2 = T::from_f64(core::f64::consts::FRAC_1_SQRT_2); // cos(π/4) = sin(π/4)
    let c3 = s1; // cos(3π/8) = sin(π/8)
    let s3 = c1; // sin(3π/8) = cos(π/8)

    // Step 1: Bit-reversal permutation of input
    // For DIT, we need input in bit-reversed order
    // Bit-reverse mapping for 4 bits: swap elements at positions that are bit-reverses
    let mut a = [Complex::<T>::zero(); 16];
    a[0] = x[0];
    a[1] = x[8];
    a[2] = x[4];
    a[3] = x[12];
    a[4] = x[2];
    a[5] = x[10];
    a[6] = x[6];
    a[7] = x[14];
    a[8] = x[1];
    a[9] = x[9];
    a[10] = x[5];
    a[11] = x[13];
    a[12] = x[3];
    a[13] = x[11];
    a[14] = x[7];
    a[15] = x[15];

    // Stage 1: 8 butterflies with span 1 (no twiddles needed, W2^0 = 1)
    for i in (0..16).step_by(2) {
        let t = a[i + 1];
        a[i + 1] = a[i] - t;
        a[i] = a[i] + t;
    }

    // Stage 2: 4 groups of 2 butterflies with span 2
    // Twiddle factors: W4^0 = 1, W4^1 = -i (forward) or +i (inverse)
    for group in (0..16).step_by(4) {
        // k=0: W4^0 = 1
        let t = a[group + 2];
        a[group + 2] = a[group] - t;
        a[group] = a[group] + t;

        // k=1: W4^1 = -i (forward) or +i (inverse)
        let t = a[group + 3];
        let t_tw = if sign < 0 {
            Complex::new(t.im, -t.re) // multiply by -i
        } else {
            Complex::new(-t.im, t.re) // multiply by +i
        };
        a[group + 3] = a[group + 1] - t_tw;
        a[group + 1] = a[group + 1] + t_tw;
    }

    // Stage 3: 2 groups of 4 butterflies with span 4
    // Twiddle factors: W8^0, W8^1, W8^2, W8^3
    for group in (0..16).step_by(8) {
        // k=0: W8^0 = 1
        let t = a[group + 4];
        a[group + 4] = a[group] - t;
        a[group] = a[group] + t;

        // k=1: W8^1 = (1-i)/sqrt(2) (forward) or (1+i)/sqrt(2) (inverse)
        let t = a[group + 5];
        let t_tw = if sign < 0 {
            Complex::new((t.re + t.im) * c2, (t.im - t.re) * c2)
        } else {
            Complex::new((t.re - t.im) * c2, (t.im + t.re) * c2)
        };
        a[group + 5] = a[group + 1] - t_tw;
        a[group + 1] = a[group + 1] + t_tw;

        // k=2: W8^2 = -i (forward) or +i (inverse)
        let t = a[group + 6];
        let t_tw = if sign < 0 {
            Complex::new(t.im, -t.re)
        } else {
            Complex::new(-t.im, t.re)
        };
        a[group + 6] = a[group + 2] - t_tw;
        a[group + 2] = a[group + 2] + t_tw;

        // k=3: W8^3 = (-1-i)/sqrt(2) (forward) or (-1+i)/sqrt(2) (inverse)
        let t = a[group + 7];
        let t_tw = if sign < 0 {
            Complex::new((-t.re + t.im) * c2, (-t.im - t.re) * c2)
        } else {
            Complex::new((-t.re - t.im) * c2, (-t.im + t.re) * c2)
        };
        a[group + 7] = a[group + 3] - t_tw;
        a[group + 3] = a[group + 3] + t_tw;
    }

    // Stage 4: 1 group of 8 butterflies with span 8
    // Twiddle factors: W16^0..W16^7
    // k=0: W16^0 = 1
    let t = a[8];
    a[8] = a[0] - t;
    a[0] = a[0] + t;

    // k=1: W16^1 = cos(π/8) - i*sin(π/8) (forward)
    let t = a[9];
    let t_tw = if sign < 0 {
        Complex::new(t.re * c1 + t.im * s1, t.im * c1 - t.re * s1)
    } else {
        Complex::new(t.re * c1 - t.im * s1, t.im * c1 + t.re * s1)
    };
    a[9] = a[1] - t_tw;
    a[1] = a[1] + t_tw;

    // k=2: W16^2 = (1-i)/sqrt(2) (forward)
    let t = a[10];
    let t_tw = if sign < 0 {
        Complex::new((t.re + t.im) * c2, (t.im - t.re) * c2)
    } else {
        Complex::new((t.re - t.im) * c2, (t.im + t.re) * c2)
    };
    a[10] = a[2] - t_tw;
    a[2] = a[2] + t_tw;

    // k=3: W16^3 = cos(3π/8) - i*sin(3π/8) (forward)
    let t = a[11];
    let t_tw = if sign < 0 {
        Complex::new(t.re * c3 + t.im * s3, t.im * c3 - t.re * s3)
    } else {
        Complex::new(t.re * c3 - t.im * s3, t.im * c3 + t.re * s3)
    };
    a[11] = a[3] - t_tw;
    a[3] = a[3] + t_tw;

    // k=4: W16^4 = -i (forward) or +i (inverse)
    let t = a[12];
    let t_tw = if sign < 0 {
        Complex::new(t.im, -t.re)
    } else {
        Complex::new(-t.im, t.re)
    };
    a[12] = a[4] - t_tw;
    a[4] = a[4] + t_tw;

    // k=5: W16^5 = sin(3π/8) - i*cos(3π/8) = cos(π/8) * (-sin(π/8)/cos(π/8) - i)
    // Actually: W16^5 = cos(5π/8) - i*sin(5π/8) = -sin(π/8) - i*cos(π/8) (forward)
    let t = a[13];
    let t_tw = if sign < 0 {
        Complex::new(-t.re * s1 + t.im * c1, -t.im * s1 - t.re * c1)
    } else {
        Complex::new(-t.re * s1 - t.im * c1, -t.im * s1 + t.re * c1)
    };
    a[13] = a[5] - t_tw;
    a[5] = a[5] + t_tw;

    // k=6: W16^6 = (-1-i)/sqrt(2) (forward)
    let t = a[14];
    let t_tw = if sign < 0 {
        Complex::new((-t.re + t.im) * c2, (-t.im - t.re) * c2)
    } else {
        Complex::new((-t.re - t.im) * c2, (-t.im + t.re) * c2)
    };
    a[14] = a[6] - t_tw;
    a[6] = a[6] + t_tw;

    // k=7: W16^7 = cos(7π/8) - i*sin(7π/8) = -cos(π/8) - i*sin(π/8) (forward)
    let t = a[15];
    let t_tw = if sign < 0 {
        Complex::new(-t.re * c1 + t.im * s1, -t.im * c1 - t.re * s1)
    } else {
        Complex::new(-t.re * c1 - t.im * s1, -t.im * c1 + t.re * s1)
    };
    a[15] = a[7] - t_tw;
    a[7] = a[7] + t_tw;

    // Output is in natural order after DIT
    x[0] = a[0];
    x[1] = a[1];
    x[2] = a[2];
    x[3] = a[3];
    x[4] = a[4];
    x[5] = a[5];
    x[6] = a[6];
    x[7] = a[7];
    x[8] = a[8];
    x[9] = a[9];
    x[10] = a[10];
    x[11] = a[11];
    x[12] = a[12];
    x[13] = a[13];
    x[14] = a[14];
    x[15] = a[15];
}

/// Size-32 DFT.
///
/// Uses a radix-2 DIT decomposition: two size-16 sub-transforms combined with twiddles.
#[inline]
pub fn notw_32<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 32);

    // Split into even and odd indices for Cooley-Tukey decomposition
    let mut even = [Complex::<T>::zero(); 16];
    let mut odd = [Complex::<T>::zero(); 16];

    for i in 0..16 {
        even[i] = x[2 * i];
        odd[i] = x[2 * i + 1];
    }

    // Apply size-16 DFT to each half
    notw_16(&mut even, sign);
    notw_16(&mut odd, sign);

    // Combine using twiddles: X[k] = E[k] + W32^k * O[k], X[k+16] = E[k] - W32^k * O[k]
    // W32^k = e^(-2πik/32) for forward, e^(+2πik/32) for inverse

    // Apply twiddles using recurrence (faster than sin_cos in loop)
    // For forward (sign < 0), angle_step = -PI/16
    // For inverse (sign >= 0), angle_step = +PI/16
    let angle_step = if sign < 0 {
        -<T as Float>::PI / T::from_usize(16)
    } else {
        <T as Float>::PI / T::from_usize(16)
    };
    let w_step = Complex::cis(angle_step);
    let mut w = Complex::new(T::one(), T::zero());

    for k in 0..16 {
        // Complex multiply: t = odd[k] * w
        let t = Complex::new(
            odd[k].re * w.re - odd[k].im * w.im,
            odd[k].im * w.re + odd[k].re * w.im,
        );

        x[k] = even[k] + t;
        x[k + 16] = even[k] - t;

        w = w * w_step;
    }
}

/// Size-64 DFT.
///
/// Uses a radix-2 DIT decomposition: two size-32 sub-transforms combined with twiddles.
#[inline]
pub fn notw_64<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 64);

    // Split into even and odd indices for Cooley-Tukey decomposition
    let mut even = [Complex::<T>::zero(); 32];
    let mut odd = [Complex::<T>::zero(); 32];

    for i in 0..32 {
        even[i] = x[2 * i];
        odd[i] = x[2 * i + 1];
    }

    // Apply size-32 DFT to each half
    notw_32(&mut even, sign);
    notw_32(&mut odd, sign);

    // Combine using twiddles: X[k] = E[k] + W64^k * O[k], X[k+32] = E[k] - W64^k * O[k]
    // W64^k = e^(-2πik/64) for forward, e^(+2πik/64) for inverse

    // Apply twiddles using recurrence (faster than sin_cos in loop)
    let angle_step = if sign < 0 {
        -<T as Float>::PI / T::from_usize(32)
    } else {
        <T as Float>::PI / T::from_usize(32)
    };
    let w_step = Complex::cis(angle_step);
    let mut w = Complex::new(T::one(), T::zero());

    for k in 0..32 {
        // Complex multiply: t = odd[k] * w
        let t = Complex::new(
            odd[k].re * w.re - odd[k].im * w.im,
            odd[k].im * w.re + odd[k].re * w.im,
        );

        x[k] = even[k] + t;
        x[k + 32] = even[k] - t;

        w = w * w_step;
    }
}

/// Size-128 DFT.
///
/// Uses a radix-2 DIT decomposition: two size-64 sub-transforms combined with twiddles.
#[inline]
pub fn notw_128<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 128);

    // Split into even and odd indices for Cooley-Tukey decomposition
    let mut even = [Complex::<T>::zero(); 64];
    let mut odd = [Complex::<T>::zero(); 64];

    for i in 0..64 {
        even[i] = x[2 * i];
        odd[i] = x[2 * i + 1];
    }

    // Apply size-64 DFT to each half
    notw_64(&mut even, sign);
    notw_64(&mut odd, sign);

    // Combine using twiddles: X[k] = E[k] + W128^k * O[k], X[k+64] = E[k] - W128^k * O[k]
    // W128^k = e^(-2πik/128) for forward, e^(+2πik/128) for inverse

    // Apply twiddles using recurrence (faster than sin_cos in loop)
    let angle_step = if sign < 0 {
        -<T as Float>::PI / T::from_usize(64)
    } else {
        <T as Float>::PI / T::from_usize(64)
    };
    let w_step = Complex::cis(angle_step);
    let mut w = Complex::new(T::one(), T::zero());

    for k in 0..64 {
        // Complex multiply: t = odd[k] * w
        let t = Complex::new(
            odd[k].re * w.re - odd[k].im * w.im,
            odd[k].im * w.re + odd[k].re * w.im,
        );

        x[k] = even[k] + t;
        x[k + 64] = even[k] - t;

        w = w * w_step;
    }
}

/// Size-256 DFT.
///
/// Uses a radix-2 DIT decomposition: two size-128 sub-transforms combined with twiddles.
#[inline]
pub fn notw_256<T: Float>(x: &mut [Complex<T>], sign: i32) {
    debug_assert!(x.len() >= 256);

    // Split into even and odd indices for Cooley-Tukey decomposition
    let mut even = [Complex::<T>::zero(); 128];
    let mut odd = [Complex::<T>::zero(); 128];

    for i in 0..128 {
        even[i] = x[2 * i];
        odd[i] = x[2 * i + 1];
    }

    // Apply size-128 DFT to each half
    notw_128(&mut even, sign);
    notw_128(&mut odd, sign);

    // Combine using twiddles: X[k] = E[k] + W256^k * O[k], X[k+128] = E[k] - W256^k * O[k]
    // W256^k = e^(-2πik/256) for forward, e^(+2πik/256) for inverse

    // Apply twiddles using recurrence (faster than sin_cos in loop)
    let angle_step = if sign < 0 {
        -<T as Float>::PI / T::from_usize(128)
    } else {
        <T as Float>::PI / T::from_usize(128)
    };
    let w_step = Complex::cis(angle_step);
    let mut w = Complex::new(T::one(), T::zero());

    for k in 0..128 {
        // Complex multiply: t = odd[k] * w
        let t = Complex::new(
            odd[k].re * w.re - odd[k].im * w.im,
            odd[k].im * w.re + odd[k].re * w.im,
        );

        x[k] = even[k] + t;
        x[k + 128] = even[k] - t;

        w = w * w_step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::fft;

    fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
        (a.re - b.re).abs() < eps && (a.im - b.im).abs() < eps
    }

    #[test]
    fn test_notw_2() {
        let mut x = [Complex::new(1.0_f64, 0.0), Complex::new(2.0, 0.0)];
        notw_2(&mut x);
        assert!((x[0].re - 3.0).abs() < 1e-10);
        assert!((x[1].re - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_notw_3() {
        let input: Vec<Complex<f64>> = (0..3).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        // Reference result from full FFT
        let expected = fft(&input);

        // Test codelet
        let mut actual = input;
        notw_3(&mut actual, -1);

        // Compare
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *e, 1e-10),
                "Mismatch at index {i}: got {a:?}, expected {e:?}"
            );
        }
    }

    #[test]
    fn test_notw_3_roundtrip() {
        let original: Vec<Complex<f64>> = (0..3)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();

        // Forward transform
        let mut transformed = original.clone();
        notw_3(&mut transformed, -1);

        // Inverse transform
        let mut recovered = transformed.clone();
        notw_3(&mut recovered, 1);

        // Normalize (divide by N=3)
        for x in &mut recovered {
            *x = *x / 3.0;
        }

        // Compare
        for (i, (a, e)) in recovered.iter().zip(original.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *e, 1e-10),
                "Roundtrip mismatch at index {i}: got {a:?}, expected {e:?}"
            );
        }
    }

    #[test]
    fn test_notw_5() {
        let input: Vec<Complex<f64>> = (0..5).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        // Reference result from full FFT
        let expected = fft(&input);

        // Test codelet
        let mut actual = input;
        notw_5(&mut actual, -1);

        // Compare
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *e, 1e-10),
                "Mismatch at index {i}: got {a:?}, expected {e:?}"
            );
        }
    }

    #[test]
    fn test_notw_5_roundtrip() {
        let original: Vec<Complex<f64>> = (0..5)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();

        // Forward transform
        let mut transformed = original.clone();
        notw_5(&mut transformed, -1);

        // Inverse transform
        let mut recovered = transformed.clone();
        notw_5(&mut recovered, 1);

        // Normalize (divide by N=5)
        for x in &mut recovered {
            *x = *x / 5.0;
        }

        // Compare
        for (i, (a, e)) in recovered.iter().zip(original.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *e, 1e-10),
                "Roundtrip mismatch at index {i}: got {a:?}, expected {e:?}"
            );
        }
    }

    #[test]
    fn test_notw_7() {
        let input: Vec<Complex<f64>> = (0..7).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        // Reference result from full FFT
        let expected = fft(&input);

        // Test codelet
        let mut actual = input;
        notw_7(&mut actual, -1);

        // Compare
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *e, 1e-10),
                "Mismatch at index {i}: got {a:?}, expected {e:?}"
            );
        }
    }

    #[test]
    fn test_notw_7_roundtrip() {
        let original: Vec<Complex<f64>> = (0..7)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();

        // Forward transform
        let mut transformed = original.clone();
        notw_7(&mut transformed, -1);

        // Inverse transform
        let mut recovered = transformed.clone();
        notw_7(&mut recovered, 1);

        // Normalize (divide by N=7)
        for x in &mut recovered {
            *x = *x / 7.0;
        }

        // Compare
        for (i, (a, e)) in recovered.iter().zip(original.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *e, 1e-10),
                "Roundtrip mismatch at index {i}: got {a:?}, expected {e:?}"
            );
        }
    }

    #[test]
    fn test_notw_4() {
        let mut x = [
            Complex::new(1.0_f64, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        notw_4(&mut x, -1);

        // X[0] = sum = 10
        assert!((x[0].re - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_notw_8() {
        let input: Vec<Complex<f64>> = (0..8).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        // Reference result from full FFT
        let expected = fft(&input);

        // Test codelet
        let mut actual = input;
        notw_8(&mut actual, -1);

        // Compare
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *e, 1e-10),
                "Mismatch at index {i}: got {a:?}, expected {e:?}"
            );
        }
    }

    #[test]
    fn test_notw_8_roundtrip() {
        let original: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();

        // Forward transform
        let mut transformed = original.clone();
        notw_8(&mut transformed, -1);

        // Inverse transform
        let mut recovered = transformed.clone();
        notw_8(&mut recovered, 1);

        // Normalize (divide by N=8)
        for x in &mut recovered {
            *x = *x / 8.0;
        }

        // Compare
        for (i, (a, e)) in recovered.iter().zip(original.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *e, 1e-10),
                "Roundtrip mismatch at index {i}: got {a:?}, expected {e:?}"
            );
        }
    }

    #[test]
    fn test_notw_8_dc_component() {
        // DC component should be sum of all inputs
        let input: Vec<Complex<f64>> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(7.0, 0.0),
            Complex::new(8.0, 0.0),
        ];

        let mut result = input;
        notw_8(&mut result, -1);

        // X[0] = sum = 1+2+3+4+5+6+7+8 = 36
        assert!(
            (result[0].re - 36.0).abs() < 1e-10,
            "DC component should be 36, got {}",
            result[0].re
        );
        assert!(
            result[0].im.abs() < 1e-10,
            "DC component should have no imaginary part"
        );
    }

    #[test]
    fn test_notw_16() {
        let input: Vec<Complex<f64>> = (0..16).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        // Reference result from full FFT
        let expected = fft(&input);

        // Test codelet
        let mut actual = input;
        notw_16(&mut actual, -1);

        // Compare
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *e, 1e-9),
                "Mismatch at index {i}: got {a:?}, expected {e:?}"
            );
        }
    }

    #[test]
    fn test_notw_16_roundtrip() {
        let original: Vec<Complex<f64>> = (0..16)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();

        // Forward transform
        let mut transformed = original.clone();
        notw_16(&mut transformed, -1);

        // Inverse transform
        let mut recovered = transformed.clone();
        notw_16(&mut recovered, 1);

        // Normalize (divide by N=16)
        for x in &mut recovered {
            *x = *x / 16.0;
        }

        // Compare
        for (i, (a, e)) in recovered.iter().zip(original.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *e, 1e-9),
                "Roundtrip mismatch at index {i}: got {a:?}, expected {e:?}"
            );
        }
    }

    #[test]
    fn test_notw_16_dc_component() {
        // DC component should be sum of all inputs
        let input: Vec<Complex<f64>> = (1..=16).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        let mut result = input;
        notw_16(&mut result, -1);

        // X[0] = sum = 1+2+...+16 = 136
        assert!(
            (result[0].re - 136.0).abs() < 1e-9,
            "DC component should be 136, got {}",
            result[0].re
        );
        assert!(
            result[0].im.abs() < 1e-9,
            "DC component should have no imaginary part"
        );
    }

    #[test]
    fn test_notw_32() {
        let input: Vec<Complex<f64>> = (0..32).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        // Reference result from full FFT
        let expected = fft(&input);

        // Test codelet
        let mut actual = input;
        notw_32(&mut actual, -1);

        // Compare
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *e, 1e-8),
                "Mismatch at index {i}: got {a:?}, expected {e:?}"
            );
        }
    }

    #[test]
    fn test_notw_32_roundtrip() {
        let original: Vec<Complex<f64>> = (0..32)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();

        // Forward transform
        let mut transformed = original.clone();
        notw_32(&mut transformed, -1);

        // Inverse transform
        let mut recovered = transformed.clone();
        notw_32(&mut recovered, 1);

        // Normalize (divide by N=32)
        for x in &mut recovered {
            *x = *x / 32.0;
        }

        // Compare
        for (i, (a, e)) in recovered.iter().zip(original.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *e, 1e-9),
                "Roundtrip mismatch at index {i}: got {a:?}, expected {e:?}"
            );
        }
    }

    #[test]
    fn test_notw_32_dc_component() {
        // DC component should be sum of all inputs
        let input: Vec<Complex<f64>> = (1..=32).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        let mut result = input;
        notw_32(&mut result, -1);

        // X[0] = sum = 1+2+...+32 = 32*33/2 = 528
        assert!(
            (result[0].re - 528.0).abs() < 1e-8,
            "DC component should be 528, got {}",
            result[0].re
        );
        assert!(
            result[0].im.abs() < 1e-8,
            "DC component should have no imaginary part"
        );
    }

    #[test]
    fn test_notw_64() {
        let input: Vec<Complex<f64>> = (0..64).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        // Reference result from full FFT
        let expected = fft(&input);

        // Test codelet
        let mut actual = input;
        notw_64(&mut actual, -1);

        // Compare
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *e, 1e-7),
                "Mismatch at index {i}: got {a:?}, expected {e:?}"
            );
        }
    }

    #[test]
    fn test_notw_64_roundtrip() {
        let original: Vec<Complex<f64>> = (0..64)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();

        // Forward transform
        let mut transformed = original.clone();
        notw_64(&mut transformed, -1);

        // Inverse transform
        let mut recovered = transformed.clone();
        notw_64(&mut recovered, 1);

        // Normalize (divide by N=64)
        for x in &mut recovered {
            *x = *x / 64.0;
        }

        // Compare
        for (i, (a, e)) in recovered.iter().zip(original.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *e, 1e-9),
                "Roundtrip mismatch at index {i}: got {a:?}, expected {e:?}"
            );
        }
    }

    #[test]
    fn test_notw_64_dc_component() {
        // DC component should be sum of all inputs
        let input: Vec<Complex<f64>> = (1..=64).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        let mut result = input;
        notw_64(&mut result, -1);

        // X[0] = sum = 1+2+...+64 = 64*65/2 = 2080
        assert!(
            (result[0].re - 2080.0).abs() < 1e-7,
            "DC component should be 2080, got {}",
            result[0].re
        );
        assert!(
            result[0].im.abs() < 1e-7,
            "DC component should have no imaginary part"
        );
    }

    #[test]
    fn test_notw_128() {
        let input: Vec<Complex<f64>> = (0..128).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        // Reference result from full FFT
        let expected = fft(&input);

        // Test codelet
        let mut actual = input;
        notw_128(&mut actual, -1);

        // Compare
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *e, 1e-6),
                "Mismatch at index {i}: got {a:?}, expected {e:?}"
            );
        }
    }

    #[test]
    fn test_notw_128_roundtrip() {
        let original: Vec<Complex<f64>> = (0..128)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();

        // Forward transform
        let mut transformed = original.clone();
        notw_128(&mut transformed, -1);

        // Inverse transform
        let mut recovered = transformed.clone();
        notw_128(&mut recovered, 1);

        // Normalize (divide by N=128)
        for x in &mut recovered {
            *x = *x / 128.0;
        }

        // Compare
        for (i, (a, e)) in recovered.iter().zip(original.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *e, 1e-8),
                "Roundtrip mismatch at index {i}: got {a:?}, expected {e:?}"
            );
        }
    }

    #[test]
    fn test_notw_128_dc_component() {
        // DC component should be sum of all inputs
        let input: Vec<Complex<f64>> = (1..=128).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        let mut result = input;
        notw_128(&mut result, -1);

        // X[0] = sum = 1+2+...+128 = 128*129/2 = 8256
        assert!(
            (result[0].re - 8256.0).abs() < 1e-5,
            "DC component should be 8256, got {}",
            result[0].re
        );
        assert!(
            result[0].im.abs() < 1e-5,
            "DC component should have no imaginary part"
        );
    }

    #[test]
    fn test_notw_256() {
        let input: Vec<Complex<f64>> = (0..256).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        // Reference result from full FFT
        let expected = fft(&input);

        // Test codelet
        let mut actual = input;
        notw_256(&mut actual, -1);

        // Compare
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *e, 1e-5),
                "Mismatch at index {i}: got {a:?}, expected {e:?}"
            );
        }
    }

    #[test]
    fn test_notw_256_roundtrip() {
        let original: Vec<Complex<f64>> = (0..256)
            .map(|i| Complex::new(f64::from(i).sin(), f64::from(i).cos()))
            .collect();

        // Forward transform
        let mut transformed = original.clone();
        notw_256(&mut transformed, -1);

        // Inverse transform
        let mut recovered = transformed.clone();
        notw_256(&mut recovered, 1);

        // Normalize (divide by N=256)
        for x in &mut recovered {
            *x = *x / 256.0;
        }

        // Compare
        for (i, (a, e)) in recovered.iter().zip(original.iter()).enumerate() {
            assert!(
                complex_approx_eq(*a, *e, 1e-8),
                "Roundtrip mismatch at index {i}: got {a:?}, expected {e:?}"
            );
        }
    }

    #[test]
    fn test_notw_256_dc_component() {
        // DC component should be sum of all inputs
        let input: Vec<Complex<f64>> = (1..=256).map(|i| Complex::new(f64::from(i), 0.0)).collect();

        let mut result = input;
        notw_256(&mut result, -1);

        // X[0] = sum = 1+2+...+256 = 256*257/2 = 32896
        assert!(
            (result[0].re - 32896.0).abs() < 1e-4,
            "DC component should be 32896, got {}",
            result[0].re
        );
        assert!(
            result[0].im.abs() < 1e-4,
            "DC component should have no imaginary part"
        );
    }
}
