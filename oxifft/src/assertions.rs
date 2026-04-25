//! Compile-time assertions that all public plan types implement `Send + Sync`.
//!
//! This module contains a single never-called function whose body consists
//! of `assert_send_sync::<T>()` calls.  If any type fails to satisfy
//! `Send + Sync + 'static`, the compiler will report an error at the
//! call-site, making type-safety regressions visible at compile time.
//!
//! The function is private and annotated `#[allow(unused)]` so that no
//! unused-item warning is emitted.

/// Helper that triggers a compile error if `T: Send + Sync + 'static` is not satisfied.
fn assert_send_sync<T: Send + Sync + 'static>() {}

/// Compile-time check: all public plan / transform types must be `Send + Sync`.
///
/// This function is intentionally never called at runtime.  Its sole purpose
/// is to produce a compile error when a public plan type stops being
/// `Send + Sync` (e.g. because a non-thread-safe field was introduced).
#[allow(unused)] // reason: compile-time check function is intentionally never called at runtime
fn check_all_public_types() {
    // ── Core 1-D / 2-D / 3-D / ND plans ───────────────────────────────────
    assert_send_sync::<crate::Plan<f32>>();
    assert_send_sync::<crate::Plan<f64>>();
    assert_send_sync::<crate::Plan2D<f32>>();
    assert_send_sync::<crate::Plan2D<f64>>();
    assert_send_sync::<crate::Plan3D<f32>>();
    assert_send_sync::<crate::Plan3D<f64>>();
    assert_send_sync::<crate::PlanND<f32>>();
    assert_send_sync::<crate::PlanND<f64>>();

    // ── Real / Split plans ─────────────────────────────────────────────────
    assert_send_sync::<crate::RealPlan<f32>>();
    assert_send_sync::<crate::RealPlan<f64>>();
    assert_send_sync::<crate::RealPlan2D<f32>>();
    assert_send_sync::<crate::RealPlan2D<f64>>();
    assert_send_sync::<crate::RealPlan3D<f32>>();
    assert_send_sync::<crate::RealPlan3D<f64>>();
    assert_send_sync::<crate::RealPlanND<f32>>();
    assert_send_sync::<crate::RealPlanND<f64>>();
    assert_send_sync::<crate::SplitPlan<f32>>();
    assert_send_sync::<crate::SplitPlan<f64>>();
    assert_send_sync::<crate::SplitPlan2D<f32>>();
    assert_send_sync::<crate::SplitPlan2D<f64>>();
    assert_send_sync::<crate::SplitPlan3D<f32>>();
    assert_send_sync::<crate::SplitPlan3D<f64>>();
    assert_send_sync::<crate::SplitPlanND<f32>>();
    assert_send_sync::<crate::SplitPlanND<f64>>();

    // ── R2R / Guru plans ───────────────────────────────────────────────────
    assert_send_sync::<crate::R2rPlan<f32>>();
    assert_send_sync::<crate::R2rPlan<f64>>();
    assert_send_sync::<crate::GuruPlan<f32>>();
    assert_send_sync::<crate::GuruPlan<f64>>();

    // ── NTT plan ───────────────────────────────────────────────────────────
    assert_send_sync::<crate::NttPlan>();

    // ── Automatic-differentiation plan ────────────────────────────────────
    assert_send_sync::<crate::DiffFftPlan<f32>>();
    assert_send_sync::<crate::DiffFftPlan<f64>>();

    // ── Feature-gated plans ────────────────────────────────────────────────
    // These are compiled only when the corresponding feature is enabled.

    #[cfg(feature = "std")]
    {
        assert_send_sync::<crate::Nufft<f32>>();
        assert_send_sync::<crate::Nufft<f64>>();
        assert_send_sync::<crate::Frft<f32>>();
        assert_send_sync::<crate::Frft<f64>>();
    }

    #[cfg(feature = "sparse")]
    {
        assert_send_sync::<crate::SparsePlan<f32>>();
        assert_send_sync::<crate::SparsePlan<f64>>();
    }

    #[cfg(feature = "pruned")]
    {
        assert_send_sync::<crate::PrunedPlan<f32>>();
        assert_send_sync::<crate::PrunedPlan<f64>>();
    }

    #[cfg(feature = "streaming")]
    {
        assert_send_sync::<crate::StreamingFft<f32>>();
        assert_send_sync::<crate::StreamingFft<f64>>();
    }

    #[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
    {
        assert_send_sync::<crate::GpuPlan<f32>>();
        assert_send_sync::<crate::GpuPlan<f64>>();
    }
}
