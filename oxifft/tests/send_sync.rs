//! Compile-time `Send + Sync` assertions for all public plan types.
//!
//! `OxiFFT` plan types must be `Send + Sync` so that they can be shared across
//! threads (e.g., stored in `Arc<Plan>`) and used safely from thread pools.
//! This file verifies the property at compile time — if any plan type loses
//! `Send` or `Sync`, these tests will fail to compile.

use oxifft::{
    Direction, Flags, GuruPlan, Plan, Plan2D, Plan3D, PlanND, RealPlan, RealPlan2D, RealPlan3D,
    RealPlanND, SplitPlan,
};

/// Helper: assert that a type is both Send and Sync.
const fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn plan_types_are_send_sync_f64() {
    assert_send_sync::<Plan<f64>>();
    assert_send_sync::<Plan2D<f64>>();
    assert_send_sync::<Plan3D<f64>>();
    assert_send_sync::<PlanND<f64>>();
    assert_send_sync::<RealPlan<f64>>();
    assert_send_sync::<RealPlan2D<f64>>();
    assert_send_sync::<RealPlan3D<f64>>();
    assert_send_sync::<RealPlanND<f64>>();
    assert_send_sync::<SplitPlan<f64>>();
    assert_send_sync::<GuruPlan<f64>>();
}

#[test]
fn plan_types_are_send_sync_f32() {
    assert_send_sync::<Plan<f32>>();
    assert_send_sync::<Plan2D<f32>>();
    assert_send_sync::<Plan3D<f32>>();
    assert_send_sync::<PlanND<f32>>();
    assert_send_sync::<RealPlan<f32>>();
    assert_send_sync::<RealPlan2D<f32>>();
    assert_send_sync::<RealPlan3D<f32>>();
    assert_send_sync::<RealPlanND<f32>>();
    assert_send_sync::<SplitPlan<f32>>();
    assert_send_sync::<GuruPlan<f32>>();
}

/// Verify plans can be shared across threads via Arc.
#[test]
fn plan_usable_from_multiple_threads() {
    use std::sync::Arc;

    let plan = Arc::new(
        Plan::<f64>::dft_1d(64, Direction::Forward, Flags::ESTIMATE)
            .expect("plan creation should succeed"),
    );

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let p = Arc::clone(&plan);
            std::thread::spawn(move || {
                let input = vec![oxifft::Complex::<f64>::new(1.0, 0.0); 64];
                let mut output = vec![oxifft::Complex::<f64>::new(0.0, 0.0); 64];
                p.execute(&input, &mut output);
                output[0].re.abs() // return something to avoid dead-code elimination
            })
        })
        .collect();

    for h in handles {
        let result = h.join().expect("thread should not panic");
        // DC component for all-ones input of size 64 is 64.0
        assert!(
            (result - 64.0).abs() < 1e-10,
            "unexpected DC value: {result}"
        );
    }
}

/// Verify Plan can be sent to a different thread.
#[test]
fn plan_sendable_across_threads() {
    let plan = Plan::<f64>::dft_1d(32, Direction::Forward, Flags::ESTIMATE)
        .expect("plan creation should succeed");

    let handle = std::thread::spawn(move || {
        // plan moved into this thread
        let input = vec![oxifft::Complex::<f64>::new(1.0, 0.0); 32];
        let mut output = vec![oxifft::Complex::<f64>::new(0.0, 0.0); 32];
        plan.execute(&input, &mut output);
        output[0].re
    });

    let dc = handle.join().expect("thread should not panic");
    assert!((dc - 32.0).abs() < 1e-10, "unexpected DC value: {dc}");
}
