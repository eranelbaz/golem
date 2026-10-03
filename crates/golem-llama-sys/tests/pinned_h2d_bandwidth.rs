//! §3.5 measurement harness: pinned (hipHostRegister'd) vs pageable host→device
//! copy bandwidth over the HIP runtime. Sizes follow the real GLM-5.3-Flash
//! expert planes (~4.7 MB up/gate, ~5.8 MB down) plus a 64 MiB bulk size.
//! Timing harness only — asserts nothing about the absolute numbers.
//!
//! Run (HIP gate box):
//! `GOLEM_LLAMA_SYS_BACKEND=hip cargo nextest run -p golem-llama-sys pinned_h2d --success-output=immediate`

#![cfg(golem_backend_hip)]
#![allow(unsafe_code)]

use golem_llama_sys::pinned::hip_host_register;
use std::time::Instant;

unsafe extern "C" {
    fn hipMalloc(dev: *mut *mut std::ffi::c_void, size: usize) -> std::ffi::c_int;
    fn hipFree(dev: *mut std::ffi::c_void) -> std::ffi::c_int;
    fn hipMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        size: usize,
        kind: std::ffi::c_int,
    ) -> std::ffi::c_int;
    fn hipDeviceSynchronize() -> std::ffi::c_int;
}

// hipMemcpyHostToHost=0, HostToDevice=1 (hip_runtime_api.h).
const HIP_MEMCPY_HOST_TO_DEVICE: std::ffi::c_int = 1;
const PAGE: usize = 4096;

/// Unregisters on drop, while the backing allocation is still alive.
struct Unreg(*const u8);
impl Drop for Unreg {
    fn drop(&mut self) {
        // SAFETY: exact pointer from the registration, still alive.
        unsafe { golem_llama_sys::pinned::hip_host_unregister(self.0.cast_mut()) }
            .expect("hipHostUnregister failed");
    }
}

/// Measures `iters` synchronous H2D copies of `size` bytes from `src`;
/// returns (avg, worst) GB/s.
fn h2d_gbps(dst: *mut std::ffi::c_void, src: *const u8, size: usize, iters: usize) -> (f64, f64) {
    // One warmup copy (allocator/runtime one-time costs) — excluded.
    // SAFETY: dst holds `size` device bytes; src is readable for `size` bytes.
    let r = unsafe { hipMemcpy(dst, src.cast(), size, HIP_MEMCPY_HOST_TO_DEVICE) };
    assert_eq!(r, 0, "hipMemcpy warmup failed: {r}");

    let mut total = 0.0f64;
    let mut worst = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        // SAFETY: same buffer bounds as the warmup copy.
        let r = unsafe { hipMemcpy(dst, src.cast(), size, HIP_MEMCPY_HOST_TO_DEVICE) };
        assert_eq!(r, 0, "hipMemcpy failed: {r}");
        // SAFETY: plain synchronize, no invariants.
        let s = unsafe { hipDeviceSynchronize() };
        assert_eq!(s, 0, "hipDeviceSynchronize failed: {s}");
        let gbps = size as f64 / t.elapsed().as_secs_f64() / 1e9;
        total += gbps;
        worst = worst.min(gbps);
    }
    (total / iters as f64, worst)
}

fn bench_size(size: usize, iters: usize) {
    // Device destination.
    let mut dev: *mut std::ffi::c_void = std::ptr::null_mut();
    // SAFETY: plain device allocation, freed on all paths below.
    let r = unsafe { hipMalloc(&mut dev, size) };
    assert_eq!(r, 0, "hipMalloc({size}) failed: {r}");

    // Pageable source: heap Vec (allocator pages, not registered).
    let pageable = vec![0xA5u8; size];

    // Pinned source: page-aligned interior of a heap allocation, registered
    // with the HIP runtime (§3.5 arena does the same to its mlock'd range).
    let backing = vec![0xA5u8; size + PAGE];
    let base = backing.as_ptr() as usize;
    let off = (PAGE - base % PAGE) % PAGE;
    // SAFETY: off < PAGE and backing holds size + PAGE bytes, so the result
    // stays in bounds with `size` bytes after it.
    let pinned: *const u8 = unsafe { backing.as_ptr().add(off) };
    assert_eq!(pinned as usize % PAGE, 0);
    // SAFETY: page-aligned live heap allocation of exactly `size` bytes,
    // not registered before, unregistered while `backing` is still alive.
    unsafe { hip_host_register(pinned.cast_mut(), size) }.expect("hipHostRegister failed");
    let _unreg = Unreg(pinned);

    let (p_avg, p_worst) = h2d_gbps(dev, pageable.as_ptr(), size, iters);
    let (pin_avg, pin_worst) = h2d_gbps(dev, pinned, size, iters);

    println!(
        "pinned_h2d: size={size} ({:.1} MiB) iters={iters} | pinned avg={pin_avg:.2} GB/s worst={pin_worst:.2} | pageable avg={p_avg:.2} GB/s worst={p_worst:.2} | speedup={:.2}x",
        size as f64 / (1 << 20) as f64,
        pin_avg / p_avg,
    );

    // SAFETY: allocation owned by this scope, no kernels in flight.
    let r = unsafe { hipFree(dev) };
    assert_eq!(r, 0, "hipFree failed: {r}");
}

#[test]
#[ignore = "benchmark"]
fn pinned_h2d_bandwidth_report() {
    bench_size(4_700_000, 20); // gate/up plane ≈ 4.7 MB
    bench_size(5_800_000, 20); // down plane ≈ 5.8 MB
    bench_size(64 << 20, 20); // 64 MiB bulk
    // No ordering assert: measured on RX 7900 XTX / ROCm 7.2, pageable
    // hipMemcpy reaches pinned bandwidth at large sizes (driver staging),
    // so a pinned>pageable assert is not universally true — report only.
}
