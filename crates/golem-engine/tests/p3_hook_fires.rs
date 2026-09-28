//! P3 acceptance seam: ensure_host_resident is invoked by the split scheduler
//! before expert host bytes are read, with the weights tensor + routed range.
//! Runs only where a GPU split exists (feature `hip-tests`, HIP CI job);
//! CPU builds never enter the split-scheduler expert-copy path.
#![cfg(feature = "hip-tests")]
mod common;

use golem_engine::session::{ChatTurn, Session};

#[test]
fn ensure_host_resident_fires_on_offload_decode() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CALLS: AtomicUsize = AtomicUsize::new(0);
    static LAST_RANGE: AtomicUsize = AtomicUsize::new(0); // (first<<16)|last

    extern "C" fn on_ensure(
        _ctx: *mut libc::c_void,
        _tensor: *const libc::c_void,
        first: i32,
        last: i32,
    ) {
        CALLS.fetch_add(1, Ordering::SeqCst);
        LAST_RANGE.store(((first as usize) << 16) | last as usize, Ordering::SeqCst);
    }
    // SAFETY: valid fn pointer; process-lifetime registration (no unregister
    // exists in this test binary — hooks staying set for the process is fine,
    // the callback only touches its own atomics).
    unsafe {
        golem_llama_sys::raw::golem_register_hooks(
            Some(on_ensure),
            None,
            None,
            std::ptr::null_mut(),
        );
    }

    let cfg = common::offload::offload_config();
    let mut session = Session::new(&cfg).unwrap();
    let turns = [ChatTurn {
        role: "user",
        content: "Hello",
    }];
    let mut out = String::new();
    session
        .chat_turn(&turns, &mut |piece| out.push_str(piece))
        .unwrap(); // helper spelling per chat_tiny_moe.rs
    assert!(!out.is_empty());

    let calls = CALLS.load(Ordering::SeqCst);
    assert!(
        calls > 0,
        "ensure_host_resident never fired during offload decode"
    );
    let packed = LAST_RANGE.load(Ordering::SeqCst);
    let (first, last) = ((packed >> 16) as i32, (packed & 0xffff) as i32);
    assert!(first <= last, "inverted range ({first}, {last})");
    assert!(
        last < 8,
        "range exceeds tiny-moe expert count 8: ({first}, {last})"
    );
}
