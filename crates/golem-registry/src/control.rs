//! Instance control socket (§9.2): `<instances>/<id>.sock`, one
//! line-delimited JSON request, one JSON reply (`{"ok":true,...}` or an
//! `ErrorEnvelope`).

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::{Registry, RegistryError};

/// How `stop` ends the instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StopMode {
    /// Refuse new requests, let the in-flight one finish (30 s bound).
    Graceful,
    /// Cancel the in-flight request and exit.
    Now,
}

/// One control request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum ControlRequest {
    /// Shut the instance down.
    Stop {
        /// Drain mode.
        mode: StopMode,
    },
    /// Health and tier counters.
    Status,
}

/// Successful reply: `ok` plus op-specific fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ControlResponse {
    /// Always `true` on this type; failures arrive as an `ErrorEnvelope`.
    pub ok: bool,
    /// Op-specific payload (`status`: health, turns, cache counters).
    #[serde(flatten)]
    pub body: serde_json::Map<String, serde_json::Value>,
}

/// Sends `req` to instance `id`'s socket, found in the registry dirs.
pub fn send(id: &str, req: &ControlRequest) -> Result<ControlResponse, RegistryError> {
    let reg = Registry::open()?;
    let sock = reg
        .read_dirs()
        .map(|d| d.join(format!("{id}.sock")))
        .find(|p| p.exists())
        .ok_or_else(|| RegistryError::InstanceNotFound(id.to_owned()))?;
    let mut s = UnixStream::connect(sock)?;
    // Stop replies only after the drain; 30 s drain + slack.
    s.set_read_timeout(Some(Duration::from_secs(40)))?;
    let mut line = serde_json::to_string(req)?;
    line.push('\n');
    s.write_all(line.as_bytes())?;
    let mut reply = String::new();
    BufReader::new(s).read_line(&mut reply)?;
    let v: serde_json::Value = serde_json::from_str(&reply)?;
    if v.get("ok").and_then(serde_json::Value::as_bool) == Some(true) {
        Ok(serde_json::from_value(v)?)
    } else {
        Err(RegistryError::Control(serde_json::from_value(v)?))
    }
}
