//! Instance process: HTTP server, slot scheduler, decode loop (tech design sections 2 and 7).
//!
//! Scaffold crate (task B1); implementation lands with its board task in
//! `docs/worklog.md` — see the tech design for the authoritative behavior.
#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert_eq!(env!("CARGO_PKG_VERSION"), "0.1.0");
    }
}
