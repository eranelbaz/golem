//! P2/P4 acceptance: policy_query is consulted per routed expert tensor with
//! a valid permutation, notify_access fires per routed expert, and the
//! persistent device cache skips re-routed experts on an identical second
//! turn (P4 counters expose the traffic through `Session::device_cache_stats`).
//! The permutation seam test registers its own recording hooks over the
//! pool-off offload config (C engages P2 on hook presence alone — Task 1's
//! call-site); the counters test drives the engine-registered hooks (pool
//! on). Runs only where a GPU split exists (feature `hip-tests`, HIP CI
//! job); CPU builds never enter the split-scheduler expert-copy path.
#![cfg(feature = "hip-tests")]
#![allow(unsafe_code)]
mod common;

use golem_engine::session::{ChatTurn, Session};

fn chat(session: &mut Session, prompt: &str) -> String {
    let mut out = String::new();
    session
        .chat_turn(
            &[ChatTurn {
                role: "user",
                content: prompt,
            }],
            &mut |piece| out.push_str(piece),
        )
        .unwrap();
    out
}

/// FFI seam: C consults policy_query per routed expert tensor and consumes a
/// valid permutation (invariant 4: any permutation is legal — identity is
/// the cheapest legal answer); notify_access reports each routed expert.
#[test]
fn policy_query_is_consulted_with_valid_permutations() {
    use std::sync::Mutex;

    static QUERIES: Mutex<Vec<Vec<i32>>> = Mutex::new(Vec::new());
    static NOTIFIES: Mutex<Vec<i64>> = Mutex::new(Vec::new());

    extern "C" fn on_query(
        _ctx: *mut libc::c_void,
        _tensor: *const libc::c_void,
        n: i32,
    ) -> *const i32 {
        let v: Vec<i32> = (0..n).collect(); // any permutation is legal (invariant 4)
        QUERIES.lock().unwrap().push(v.clone());
        // contract: valid until next call — leak-per-test is fine (test process)
        Box::into_raw(v.into_boxed_slice()) as *const i32
    }
    extern "C" fn on_notify(_ctx: *mut libc::c_void, expert_id: i64) {
        NOTIFIES.lock().unwrap().push(expert_id);
    }
    extern "C" fn on_ensure(
        _ctx: *mut libc::c_void,
        _tensor: *const libc::c_void,
        _first: i32,
        _last: i32,
        _ptrs: *mut *const libc::c_void,
    ) {
        // ABI v2 (C-2): the substrate copies from the returned pointers; these
        // hook-level tests don't page from a pool, so serve one static buffer.
        const SRC: [u8; 65536] = [0u8; 65536];
        let src = SRC.as_ptr() as *const libc::c_void;
        for i in 0..((_last - _first + 1) as usize) {
            unsafe { *_ptrs.add(i) = src };
        }
    }

    // SAFETY: valid fn pointers; process-lifetime registration (no unregis-
    // ter exists) is fine for a test process — nextest isolates per-test
    // processes, so these hooks only ever see this test's decode.
    unsafe {
        golem_llama_sys::raw::golem_register_hooks(
            Some(on_ensure),
            Some(on_query),
            Some(on_notify),
            None,
            std::ptr::null_mut(),
        );
    }

    // Pool OFF (engine registers nothing): these hooks stand in for the
    // engine's, isolating the C call-site seam. P2 engages because BOTH
    // policy_query and ensure_host_resident are non-null (Task 1 contract).
    let cfg = common::offload::offload_config();
    let mut session = Session::new(&cfg).unwrap();
    let out = chat(&mut session, "Hello");
    assert!(!out.is_empty());

    let queries = QUERIES.lock().unwrap().clone();
    assert!(
        !queries.is_empty(),
        "policy_query never fired during offload decode"
    );
    for order in &queries {
        assert_eq!(order.len(), 8, "tiny-moe expert count");
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            (0..order.len() as i32).collect::<Vec<i32>>(),
            "returned order must be a permutation of 0..n"
        );
    }

    let notifies = NOTIFIES.lock().unwrap().clone();
    assert!(!notifies.is_empty(), "notify_access never fired");
    assert!(
        notifies.iter().all(|&id| (0..8).contains(&id)),
        "routed expert ids must be within 0..8"
    );
}

/// Engine-registered hooks (pool on): the persistent device cache skips the
/// identical re-route of a second turn, and P4 counters surface the traffic.
/// Skips with a reason when the 1 GiB RAM pool cannot mlock (RLIMIT_MEMLOCK
/// below 1 GiB — local dev boxes); the HIP CI job runs it for real.
#[test]
fn device_cache_skips_on_second_identical_turn() {
    let mut cfg = common::offload::offload_config();
    cfg.ram_experts_gib = 1; // pool on ⇒ engine registers all three hooks
    let mut session = match Session::new(&cfg) {
        Ok(s) => s,
        Err(e) if e.to_string().contains("mlock") => {
            // A failed precondition FAILS (review H-7): an early return reads
            // as a pass to the harness. Opt-in skip only via explicit env.
            assert!(
                std::env::var("GOLEM_ALLOW_SKIP").as_deref() == Ok("1"),
                "precondition failed: RLIMIT_MEMLOCK below the 1 GiB pool                  (ram_experts_gib=1); raise memlock or set GOLEM_ALLOW_SKIP=1                  to opt out: {e}"
            );
            eprintln!(
                "SKIPPED: RLIMIT_MEMLOCK below the 1 GiB pool (ram_experts_gib=1); \
                 run on a machine with unlimited memlock (HIP CI job): {e}"
            );
            return;
        }
        Err(e) => panic!("session with pooled offload failed: {e}"),
    };

    let out1 = chat(&mut session, "Hello");
    assert!(!out1.is_empty());
    let first = session
        .device_cache_stats()
        .expect("pooled GPU split ⇒ device-cache stats present");
    assert!(
        first.bytes_copied > 0,
        "cold turn copies routed experts into the device cache: {first:?}"
    );

    let out2 = chat(&mut session, "Hello");
    assert_eq!(
        out1, out2,
        "device cache must not change decode output (§7.4)"
    );
    let second = session.device_cache_stats().unwrap();
    assert!(
        second.bytes_skipped > 0,
        "identical re-route must hit the device cache"
    );
    assert!(
        second.skip_ratio > first.skip_ratio,
        "skip ratio must climb on an all-hit turn (first {first:?}, second {second:?})"
    );
}
