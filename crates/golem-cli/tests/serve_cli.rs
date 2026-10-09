//! `golem serve` as a process: SIGTERM during an open request exits 0.
mod common;

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[test]
fn sigterm_during_stream_exits_cleanly() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_golem"))
        .arg("serve")
        .arg(common::tiny_moe_path())
        .args(["--port", "0", "--ctx", "4096"])
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut err = BufReader::new(child.stderr.take().unwrap());
    let mut log = String::new();
    let addr = loop {
        let mut line = String::new();
        assert!(err.read_line(&mut line).unwrap() > 0, "serve exited: {log}");
        log.push_str(&line);
        if let Some(a) = line.trim().strip_prefix("listening on ") {
            break a.to_owned();
        }
    };
    let body =
        r#"{"model":"tiny-moe","messages":[{"role":"user","content":"hi"}],"max_tokens":4096}"#;
    let mut s = TcpStream::connect(&addr).unwrap();
    write!(
        s,
        "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\n\
         content-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    std::thread::sleep(Duration::from_millis(100));
    let kill = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(kill.success());
    let deadline = Instant::now() + Duration::from_secs(70);
    let status = loop {
        if let Some(st) = child.try_wait().unwrap() {
            break st;
        }
        assert!(Instant::now() < deadline, "serve did not exit in 70 s");
        std::thread::sleep(Duration::from_millis(50));
    };
    err.read_to_string(&mut log).unwrap();
    assert_eq!(status.code(), Some(0), "{log}");
    let mut resp = String::new();
    s.read_to_string(&mut resp).unwrap();
    // Either the turn finished before the signal, or it was drained.
    assert!(
        resp.starts_with("HTTP/1.1 200") || resp.contains("server draining"),
        "{resp}"
    );
}
