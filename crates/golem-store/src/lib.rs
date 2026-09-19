//! Multi-disk mirrored storage, striped reads, checksums (tech design section 5).
//!
//! Scaffold crate (task B1); implementation lands with its board task in
//! `docs/worklog.md` — see the tech design for the authoritative behavior.
#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert_eq!(env!("CARGO_PKG_NAME"), "golem-store");
    }
}
