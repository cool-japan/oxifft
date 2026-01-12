//! Tests for the `GuruPlan` interface.
//!
//! Tests arbitrary strides, batching, and multi-dimensional transforms
//! using the FFTW-style guru interface.

#![allow(clippy::cast_precision_loss)] // FFT size computations use float for math
#![allow(clippy::cast_sign_loss)] // stride calculations need signed/unsigned
#![allow(clippy::cast_possible_wrap)] // stride calculations need careful wrapping

use oxifft::{Complex, Direction, Flags, GuruPlan, IoDim, Plan, Tensor};

/// Generate test input data.
fn generate_input(n: usize) -> Vec<Complex<f64>> {
    (0..n)
        .map(|i| {
            let t = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            Complex::new(t.cos(), t.sin() * 0.5)
        })
        .collect()
}

/// Check if two complex vectors are approximately equal.
fn approx_eq(a: &[Complex<f64>], b: &[Complex<f64>], tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x.re - y.re).hypot(x.im - y.im);
        let mag = y.re.hypot(y.im).max(1.0);
        if diff / mag > tol {
            eprintln!(
                "Mismatch at index {}: got ({}, {}), expected ({}, {}), rel_diff = {}",
                i,
                x.re,
                x.im,
                y.re,
                y.im,
                diff / mag
            );
            return false;
        }
    }

    true
}

// ============================================================================
// Basic creation tests
// ============================================================================

#[test]
fn test_guru_plan_creation_1d() {
    let dims = Tensor::new(vec![IoDim::new(16, 1, 1)]);
    let howmany = Tensor::empty();

    let plan = GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE);
    assert!(plan.is_some());

    let plan = plan.unwrap();
    assert_eq!(plan.transform_size(), 16);
    assert_eq!(plan.batch_count(), 1);
}

#[test]
fn test_guru_plan_creation_2d() {
    let dims = Tensor::new(vec![IoDim::new(8, 1, 1), IoDim::new(8, 8, 8)]);
    let howmany = Tensor::empty();

    let plan = GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE);
    assert!(plan.is_some());

    let plan = plan.unwrap();
    assert_eq!(plan.transform_size(), 64);
    assert_eq!(plan.batch_count(), 1);
}

#[test]
fn test_guru_plan_creation_batched() {
    let dims = Tensor::new(vec![IoDim::new(16, 1, 1)]);
    let howmany = Tensor::new(vec![IoDim::new(10, 16, 16)]);

    let plan = GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE);
    assert!(plan.is_some());

    let plan = plan.unwrap();
    assert_eq!(plan.transform_size(), 16);
    assert_eq!(plan.batch_count(), 10);
}

#[test]
fn test_guru_plan_rejects_empty_dims() {
    let dims = Tensor::empty();
    let howmany = Tensor::empty();

    let plan = GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE);
    assert!(plan.is_none());
}

#[test]
fn test_guru_plan_rejects_zero_size() {
    let dims = Tensor::new(vec![IoDim::new(0, 1, 1)]);
    let howmany = Tensor::empty();

    let plan = GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE);
    assert!(plan.is_none());
}

// ============================================================================
// 1D execution tests with contiguous strides
// ============================================================================

#[test]
fn test_guru_1d_contiguous_forward() {
    let n = 16;
    let dims = Tensor::new(vec![IoDim::new(n, 1, 1)]);
    let howmany = Tensor::empty();

    let guru_plan =
        GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE).unwrap();
    let simple_plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();

    let input = generate_input(n);
    let mut guru_output = vec![Complex::new(0.0, 0.0); n];
    let mut simple_output = vec![Complex::new(0.0, 0.0); n];

    guru_plan.execute(&input, &mut guru_output);
    simple_plan.execute(&input, &mut simple_output);

    assert!(
        approx_eq(&guru_output, &simple_output, 1e-12),
        "GuruPlan 1D contiguous should match Plan::dft_1d"
    );
}

#[test]
fn test_guru_1d_contiguous_roundtrip() {
    let n = 32;
    let dims = Tensor::new(vec![IoDim::new(n, 1, 1)]);
    let howmany = Tensor::empty();

    let forward =
        GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE).unwrap();
    let backward =
        GuruPlan::<f64>::dft(&dims, &howmany, Direction::Backward, Flags::ESTIMATE).unwrap();

    let input = generate_input(n);
    let mut freq = vec![Complex::new(0.0, 0.0); n];
    let mut recovered = vec![Complex::new(0.0, 0.0); n];

    forward.execute(&input, &mut freq);
    backward.execute(&freq, &mut recovered);

    // Normalize
    let scale = 1.0 / (n as f64);
    for c in &mut recovered {
        c.re *= scale;
        c.im *= scale;
    }

    assert!(
        approx_eq(&input, &recovered, 1e-12),
        "GuruPlan 1D roundtrip failed"
    );
}

// ============================================================================
// 1D execution tests with non-contiguous strides
// ============================================================================

#[test]
fn test_guru_1d_strided_input() {
    let n = 8;
    // Input has stride 2 (every other element)
    let dims = Tensor::new(vec![IoDim::new(n, 2, 1)]);
    let howmany = Tensor::empty();

    let guru_plan =
        GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE).unwrap();
    let simple_plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();

    // Create input with stride 2
    let base_input = generate_input(n);
    let mut strided_input = vec![Complex::new(0.0, 0.0); n * 2];
    for (i, &val) in base_input.iter().enumerate() {
        strided_input[i * 2] = val;
    }

    let mut guru_output = vec![Complex::new(0.0, 0.0); n];
    let mut simple_output = vec![Complex::new(0.0, 0.0); n];

    guru_plan.execute(&strided_input, &mut guru_output);
    simple_plan.execute(&base_input, &mut simple_output);

    assert!(
        approx_eq(&guru_output, &simple_output, 1e-12),
        "GuruPlan with strided input should match contiguous Plan"
    );
}

#[test]
fn test_guru_1d_strided_output() {
    let n = 8;
    // Output has stride 2
    let dims = Tensor::new(vec![IoDim::new(n, 1, 2)]);
    let howmany = Tensor::empty();

    let guru_plan =
        GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE).unwrap();
    let simple_plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();

    let input = generate_input(n);
    let mut strided_output = vec![Complex::new(0.0, 0.0); n * 2];
    let mut simple_output = vec![Complex::new(0.0, 0.0); n];

    guru_plan.execute(&input, &mut strided_output);
    simple_plan.execute(&input, &mut simple_output);

    // Extract strided output
    let guru_extracted: Vec<Complex<f64>> = (0..n).map(|i| strided_output[i * 2]).collect();

    assert!(
        approx_eq(&guru_extracted, &simple_output, 1e-12),
        "GuruPlan with strided output should match contiguous Plan"
    );
}

#[test]
fn test_guru_1d_both_strided() {
    let n = 8;
    // Both input and output strided
    let dims = Tensor::new(vec![IoDim::new(n, 2, 3)]);
    let howmany = Tensor::empty();

    let guru_plan =
        GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE).unwrap();
    let simple_plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();

    let base_input = generate_input(n);

    // Create strided input
    let mut strided_input = vec![Complex::new(0.0, 0.0); n * 2];
    for (i, &val) in base_input.iter().enumerate() {
        strided_input[i * 2] = val;
    }

    let mut strided_output = vec![Complex::new(0.0, 0.0); n * 3];
    let mut simple_output = vec![Complex::new(0.0, 0.0); n];

    guru_plan.execute(&strided_input, &mut strided_output);
    simple_plan.execute(&base_input, &mut simple_output);

    // Extract strided output
    let guru_extracted: Vec<Complex<f64>> = (0..n).map(|i| strided_output[i * 3]).collect();

    assert!(
        approx_eq(&guru_extracted, &simple_output, 1e-12),
        "GuruPlan with both strided should match contiguous Plan"
    );
}

// ============================================================================
// Batch tests
// ============================================================================

#[test]
fn test_guru_batch_simple() {
    let n = 16;
    let batch = 4;

    // Contiguous batches: each batch is size n, consecutively stored
    let dims = Tensor::new(vec![IoDim::new(n, 1, 1)]);
    let howmany = Tensor::new(vec![IoDim::new(batch, n as isize, n as isize)]);

    let guru_plan =
        GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE).unwrap();
    let simple_plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();

    // Create batched input
    let total = n * batch;
    let input: Vec<Complex<f64>> = (0..total)
        .map(|i| {
            let batch_idx = i / n;
            let elem_idx = i % n;
            let t = (elem_idx as f64).mul_add(0.1, (batch_idx as f64) * 0.5);
            Complex::new(t.sin(), t.cos())
        })
        .collect();

    let mut guru_output = vec![Complex::new(0.0, 0.0); total];

    guru_plan.execute(&input, &mut guru_output);

    // Verify against individual transforms
    for b in 0..batch {
        let offset = b * n;
        let batch_input: Vec<Complex<f64>> = input[offset..offset + n].to_vec();
        let mut batch_output = vec![Complex::new(0.0, 0.0); n];
        simple_plan.execute(&batch_input, &mut batch_output);

        let guru_slice = &guru_output[offset..offset + n];
        assert!(
            approx_eq(guru_slice, &batch_output, 1e-12),
            "Batch {b} mismatch"
        );
    }
}

#[test]
fn test_guru_batch_with_stride() {
    let n = 8;
    let batch = 3;
    let batch_stride = n as isize + 4; // Some padding between batches

    let dims = Tensor::new(vec![IoDim::new(n, 1, 1)]);
    let howmany = Tensor::new(vec![IoDim::new(batch, batch_stride, batch_stride)]);

    let guru_plan =
        GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE).unwrap();
    let simple_plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();

    // Create input with padding
    let total_elements = (batch - 1) * batch_stride as usize + n;
    let mut input = vec![Complex::new(0.0, 0.0); total_elements];
    for b in 0..batch {
        let offset = b * batch_stride as usize;
        for i in 0..n {
            let t = (i as f64).mul_add(0.1, (b as f64) * 0.5);
            input[offset + i] = Complex::new(t.sin(), t.cos());
        }
    }

    let mut guru_output = vec![Complex::new(0.0, 0.0); total_elements];
    guru_plan.execute(&input, &mut guru_output);

    // Verify each batch
    for b in 0..batch {
        let offset = b * batch_stride as usize;
        let batch_input: Vec<Complex<f64>> = input[offset..offset + n].to_vec();
        let mut batch_output = vec![Complex::new(0.0, 0.0); n];
        simple_plan.execute(&batch_input, &mut batch_output);

        let guru_slice = &guru_output[offset..offset + n];
        assert!(
            approx_eq(guru_slice, &batch_output, 1e-12),
            "Strided batch {b} mismatch"
        );
    }
}

#[test]
fn test_guru_batch_roundtrip() {
    let n = 16;
    let batch = 5;

    let dims = Tensor::new(vec![IoDim::new(n, 1, 1)]);
    let howmany = Tensor::new(vec![IoDim::new(batch, n as isize, n as isize)]);

    let forward =
        GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE).unwrap();
    let backward =
        GuruPlan::<f64>::dft(&dims, &howmany, Direction::Backward, Flags::ESTIMATE).unwrap();

    let total = n * batch;
    let input: Vec<Complex<f64>> = (0..total)
        .map(|i| {
            let t = (i as f64) * 0.123;
            Complex::new(t.sin(), t.cos())
        })
        .collect();

    let mut freq = vec![Complex::new(0.0, 0.0); total];
    let mut recovered = vec![Complex::new(0.0, 0.0); total];

    forward.execute(&input, &mut freq);
    backward.execute(&freq, &mut recovered);

    // Normalize
    let scale = 1.0 / (n as f64);
    for c in &mut recovered {
        c.re *= scale;
        c.im *= scale;
    }

    assert!(
        approx_eq(&input, &recovered, 1e-12),
        "Batched roundtrip failed"
    );
}

// ============================================================================
// 2D tests
// ============================================================================

#[test]
fn test_guru_2d_contiguous() {
    use oxifft::Plan2D;

    let nx = 8;
    let ny = 8;
    let n = nx * ny;

    // Row-major layout: fast dimension has stride 1, slow dimension has stride nx
    let dims = Tensor::new(vec![
        IoDim::new(nx, 1, 1),                     // Fast dimension (columns)
        IoDim::new(ny, nx as isize, nx as isize), // Slow dimension (rows)
    ]);
    let howmany = Tensor::empty();

    let guru_plan =
        GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE).unwrap();
    let plan_2d = Plan2D::new(nx, ny, Direction::Forward, Flags::ESTIMATE).unwrap();

    let input: Vec<Complex<f64>> = (0..n)
        .map(|i| {
            let t = (i as f64) * 0.1;
            Complex::new(t.sin(), t.cos() * 0.5)
        })
        .collect();

    let mut guru_output = vec![Complex::new(0.0, 0.0); n];
    let mut plan2d_output = vec![Complex::new(0.0, 0.0); n];

    guru_plan.execute(&input, &mut guru_output);
    plan_2d.execute(&input, &mut plan2d_output);

    assert!(
        approx_eq(&guru_output, &plan2d_output, 1e-10),
        "GuruPlan 2D should match Plan2D"
    );
}

#[test]
fn test_guru_2d_roundtrip() {
    let nx = 8;
    let ny = 8;
    let n = nx * ny;

    let dims = Tensor::new(vec![
        IoDim::new(nx, 1, 1),
        IoDim::new(ny, nx as isize, nx as isize),
    ]);
    let howmany = Tensor::empty();

    let forward =
        GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE).unwrap();
    let backward =
        GuruPlan::<f64>::dft(&dims, &howmany, Direction::Backward, Flags::ESTIMATE).unwrap();

    let input: Vec<Complex<f64>> = (0..n)
        .map(|i| {
            let t = (i as f64) * 0.123;
            Complex::new(t.sin(), t.cos())
        })
        .collect();

    let mut freq = vec![Complex::new(0.0, 0.0); n];
    let mut recovered = vec![Complex::new(0.0, 0.0); n];

    forward.execute(&input, &mut freq);
    backward.execute(&freq, &mut recovered);

    // Normalize
    let scale = 1.0 / (n as f64);
    for c in &mut recovered {
        c.re *= scale;
        c.im *= scale;
    }

    assert!(
        approx_eq(&input, &recovered, 1e-10),
        "GuruPlan 2D roundtrip failed"
    );
}

// ============================================================================
// In-place tests
// ============================================================================

#[test]
fn test_guru_1d_inplace() {
    let n = 16;
    let dims = Tensor::new(vec![IoDim::new(n, 1, 1)]);
    let howmany = Tensor::empty();

    let plan = GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE).unwrap();

    let input = generate_input(n);

    // Out-of-place reference
    let mut out_of_place = vec![Complex::new(0.0, 0.0); n];
    plan.execute(&input, &mut out_of_place);

    // In-place
    let mut in_place = input;
    plan.execute_inplace(&mut in_place);

    assert!(
        approx_eq(&in_place, &out_of_place, 1e-12),
        "In-place should match out-of-place"
    );
}

#[test]
fn test_guru_batch_inplace() {
    let n = 8;
    let batch = 4;
    let total = n * batch;

    let dims = Tensor::new(vec![IoDim::new(n, 1, 1)]);
    let howmany = Tensor::new(vec![IoDim::new(batch, n as isize, n as isize)]);

    let plan = GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE).unwrap();

    let input: Vec<Complex<f64>> = (0..total)
        .map(|i| {
            let t = (i as f64) * 0.1;
            Complex::new(t.sin(), t.cos())
        })
        .collect();

    // Out-of-place reference
    let mut out_of_place = vec![Complex::new(0.0, 0.0); total];
    plan.execute(&input, &mut out_of_place);

    // In-place
    let mut in_place = input;
    plan.execute_inplace(&mut in_place);

    assert!(
        approx_eq(&in_place, &out_of_place, 1e-12),
        "Batched in-place should match out-of-place"
    );
}

// ============================================================================
// Various sizes
// ============================================================================

#[test]
fn test_guru_various_sizes() {
    let sizes = [4, 7, 8, 13, 16, 17, 32, 64, 100, 128];

    for &n in &sizes {
        let dims = Tensor::new(vec![IoDim::new(n, 1, 1)]);
        let howmany = Tensor::empty();

        let guru =
            GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE).unwrap();
        let simple = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).unwrap();

        let input = generate_input(n);
        let mut guru_out = vec![Complex::new(0.0, 0.0); n];
        let mut simple_out = vec![Complex::new(0.0, 0.0); n];

        guru.execute(&input, &mut guru_out);
        simple.execute(&input, &mut simple_out);

        let tol = 1e-10 * (n as f64).log2().max(1.0);
        assert!(
            approx_eq(&guru_out, &simple_out, tol),
            "GuruPlan size {n} mismatch"
        );
    }
}

#[test]
fn test_guru_2d_various_sizes() {
    use oxifft::Plan2D;

    // (n_cols, n_rows) - fast dimension first, then slow
    let sizes = [(4, 4), (4, 8), (8, 4), (8, 8), (16, 8), (7, 11)];

    for &(n_cols, n_rows) in &sizes {
        let n = n_cols * n_rows;

        // Row-major layout: n_cols elements per row (fast dim), n_rows rows (slow dim)
        let dims = Tensor::new(vec![
            IoDim::new(n_cols, 1, 1), // Fast dimension (columns within row)
            IoDim::new(n_rows, n_cols as isize, n_cols as isize), // Slow dimension (rows)
        ]);
        let howmany = Tensor::empty();

        let guru =
            GuruPlan::<f64>::dft(&dims, &howmany, Direction::Forward, Flags::ESTIMATE).unwrap();
        // Plan2D::new(n0=rows, n1=cols)
        let plan2d = Plan2D::new(n_rows, n_cols, Direction::Forward, Flags::ESTIMATE).unwrap();

        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let t = (i as f64) * 0.1;
                Complex::new(t.sin(), t.cos())
            })
            .collect();

        let mut guru_out = vec![Complex::new(0.0, 0.0); n];
        let mut plan2d_out = vec![Complex::new(0.0, 0.0); n];

        guru.execute(&input, &mut guru_out);
        plan2d.execute(&input, &mut plan2d_out);

        let tol = 1e-9 * (n as f64).log2().max(1.0);
        assert!(
            approx_eq(&guru_out, &plan2d_out, tol),
            "GuruPlan 2D {n_cols}x{n_rows} mismatch"
        );
    }
}
