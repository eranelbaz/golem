//! Offline VRAM/RAM/disk estimation shared by CLI and runtime (tech design section 10).
//!
//! Scaffold crate (task B1); implementation lands with its board task in
//! `docs/worklog.md` — see the tech design for the authoritative behavior.
#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert_eq!(env!("CARGO_PKG_NAME"), "golem-plan");
    }
}
