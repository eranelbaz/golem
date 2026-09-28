//! std/loom cfg-swap primitives for the pool module (T6): every concurrent
//! type in `pool` is imported from here, so loom explores the same code paths
//! that run in production. `model(f)` runs one pass in std mode and the full
//! loom state space under `--cfg loom`.
//
// `cfg(loom)` is set via RUSTFLAGS by the T6 loom leg (the `loom` feature only
// links the crate), so declare the cfg name known here.
#![allow(unexpected_cfgs)]

#[cfg(loom)]
pub use loom::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
#[cfg(loom)]
pub use loom::sync::{Arc, Mutex};
/// Thread handle from [`spawn`] (modeled under loom).
#[cfg(loom)]
pub type JoinHandle<T> = loom::thread::JoinHandle<T>;
#[cfg(loom)]
pub fn model<F>(f: F)
where
    F: Fn() + Send + Sync + 'static,
{
    loom::model(f);
}
/// Spawns a modeled thread (loom) or a real thread (std).
#[cfg(loom)]
pub fn spawn<F, T>(f: F) -> loom::thread::JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    loom::thread::spawn(f)
}
/// Yields the current thread (modeled under loom).
#[cfg(loom)]
pub fn yield_now() {
    loom::thread::yield_now();
}

#[cfg(not(loom))]
pub use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
#[cfg(not(loom))]
pub use std::sync::{Arc, Mutex};
#[cfg(not(loom))]
/// Thread handle from [`spawn`] (std).
pub type JoinHandle<T> = std::thread::JoinHandle<T>;
/// Runs `f` once (std mode; loom mode explores the full state space).
#[cfg(not(loom))]
pub fn model<F>(f: F)
where
    F: Fn(),
{
    f();
}
/// Spawns a real thread.
#[cfg(not(loom))]
pub fn spawn<F, T>(f: F) -> std::thread::JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    std::thread::spawn(f)
}
/// Yields the current thread.
#[cfg(not(loom))]
pub fn yield_now() {
    std::thread::yield_now();
}

/// Coerces a concrete `Arc<T>` to a trait-object `Arc` (used for
/// `Arc<dyn ExpertSource>`). loom's `Arc` is a newtype, so the compiler's
/// implicit unsizing coercion does not apply; both modes end at the same
/// typed surface.
#[cfg(loom)]
pub fn arc_dyn<T: ?Sized>(value: std::sync::Arc<T>) -> Arc<T> {
    loom::sync::Arc::from_std(value)
}
/// Pass-through in std mode (the built-in unsizing coercion already applies).
#[cfg(not(loom))]
pub fn arc_dyn<T: ?Sized>(value: Arc<T>) -> Arc<T> {
    value
}
