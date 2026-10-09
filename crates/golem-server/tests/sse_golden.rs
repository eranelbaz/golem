//! Streaming (`stream: true`) over a real tiny-moe worker: golden frame
//! sequence, `reasoning_content` routing, and client-disconnect cancel.
use std::time::Duration;

use golem_server::test_support;

#[tokio::test(flavor = "multi_thread")]
async fn stream_with_usage_matches_golden() {
    let frames = test_support::sse_frames(r#"{"model":"tiny-moe","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}"#).await;
    let norm = test_support::normalize(&frames);
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/stream_usage.sse");
    if std::env::var_os("GOLEM_BLESS").is_some() {
        std::fs::write(path, &norm).unwrap();
    }
    assert_eq!(
        norm,
        std::fs::read_to_string(path).expect("golden missing: run once with GOLEM_BLESS=1")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn client_disconnect_cancels_slot() {
    let logs = test_support::capture_logs();
    test_support::drop_after_first_frame(
        r#"{"model":"tiny-moe","messages":[{"role":"user","content":"hi"}],"max_tokens":4096,"stream":true}"#,
    )
    .await;
    assert!(logs.wait_for("cancel: sequence removed", Duration::from_secs(10)));
    // slot free again: next request succeeds
    test_support::complete_once().await.unwrap();
}
