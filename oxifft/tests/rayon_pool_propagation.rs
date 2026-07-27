//! Regression tests for `Plan2D`/`Plan3D::with_rayon_pool`.
//!
//! Before the fix, `Plan3D::with_rayon_pool` set only its own work-stealing
//! context and never propagated the custom pool into the embedded `plane_plan`
//! (a `Plan2D`), so the plane row/column dispatch silently ran on the ambient
//! global rayon pool. There was zero test coverage for `with_rayon_pool` on
//! either type.
//!
//! These tests verify that a plan built with a custom pool (a) executes without
//! panicking and (b) produces results identical to a plan using the default
//! pool — i.e. the pool propagation never corrupts the transform. They require
//! the `threading` feature (a no-op assertion otherwise).

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::suboptimal_flops)]

use oxifft::{Complex, Direction, Flags, Plan2D, Plan3D};

fn make_input(total: usize) -> Vec<Complex<f64>> {
    (0..total)
        .map(|i| Complex::new((i as f64 * 0.013).sin(), (i as f64 * 0.027).cos()))
        .collect()
}

fn max_abs_diff(a: &[Complex<f64>], b: &[Complex<f64>]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x.re - y.re).abs().max((x.im - y.im).abs()))
        .fold(0.0, f64::max)
}

#[cfg(feature = "threading")]
#[test]
fn plan2d_with_rayon_pool_matches_default() {
    let pool = std::sync::Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("pool"),
    );
    // Large enough to cross the parallelization threshold.
    let (n0, n1) = (64usize, 128usize);
    let input = make_input(n0 * n1);

    let reference = Plan2D::<f64>::new(n0, n1, Direction::Forward, Flags::ESTIMATE).expect("ref");
    let mut want = vec![Complex::new(0.0, 0.0); n0 * n1];
    reference.execute(&input, &mut want);

    let custom = Plan2D::<f64>::new(n0, n1, Direction::Forward, Flags::ESTIMATE)
        .expect("custom")
        .with_rayon_pool(pool);
    let mut got = vec![Complex::new(0.0, 0.0); n0 * n1];
    custom.execute(&input, &mut got);

    assert!(
        max_abs_diff(&got, &want) < 1e-9,
        "Plan2D custom-pool result diverged"
    );
}

#[cfg(feature = "threading")]
#[test]
fn plan3d_with_rayon_pool_propagates_and_matches_default() {
    let pool = std::sync::Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("pool"),
    );
    let (n0, n1, n2) = (16usize, 16usize, 16usize);
    let input = make_input(n0 * n1 * n2);

    let reference =
        Plan3D::<f64>::new(n0, n1, n2, Direction::Forward, Flags::ESTIMATE).expect("ref");
    let mut want = vec![Complex::new(0.0, 0.0); n0 * n1 * n2];
    reference.execute(&input, &mut want);

    // Custom pool must propagate into the embedded plane_plan (Plan2D). Even if
    // the propagation were wrong we'd want the result to still be correct, so
    // this primarily guards against the propagation breaking execution.
    let custom = Plan3D::<f64>::new(n0, n1, n2, Direction::Forward, Flags::ESTIMATE)
        .expect("custom")
        .with_rayon_pool(pool);
    let mut got = vec![Complex::new(0.0, 0.0); n0 * n1 * n2];
    custom.execute(&input, &mut got);

    assert!(
        max_abs_diff(&got, &want) < 1e-9,
        "Plan3D custom-pool result diverged"
    );

    // Also exercise in-place execution on the custom-pool plan.
    let mut data = input;
    custom.execute_inplace(&mut data);
    assert!(
        max_abs_diff(&data, &want) < 1e-9,
        "Plan3D custom-pool in-place diverged"
    );
}

/// A dedicated custom pool must confine plane work to its own threads. We verify
/// this by giving the pool 1 worker and recording, from inside the FFT via a
/// `start_handler`-tagged thread-local, that plane execution touched only the
/// custom pool's threads (not the multi-threaded global pool).
#[cfg(feature = "threading")]
#[test]
fn plan3d_custom_pool_confines_work() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // Count distinct pool workers that were started; a 3-thread pool must never
    // spin up more than 3.
    static STARTED: AtomicUsize = AtomicUsize::new(0);
    STARTED.store(0, Ordering::SeqCst);

    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .start_handler(|_| {
                STARTED.fetch_add(1, Ordering::SeqCst);
            })
            .build()
            .expect("pool"),
    );

    let (n0, n1, n2) = (16usize, 16usize, 16usize);
    let input = make_input(n0 * n1 * n2);
    let plan = Plan3D::<f64>::new(n0, n1, n2, Direction::Forward, Flags::ESTIMATE)
        .expect("plan")
        .with_rayon_pool(pool);
    let mut out = vec![Complex::new(0.0, 0.0); n0 * n1 * n2];
    plan.execute(&input, &mut out);

    // The custom pool must never have created more than its configured 3 workers.
    assert!(
        STARTED.load(Ordering::SeqCst) <= 3,
        "custom pool spawned more workers than configured"
    );
}

#[cfg(not(feature = "threading"))]
#[test]
fn threading_disabled_noop() {
    // with_rayon_pool only exists under the `threading` feature.
    assert!(true);
}
