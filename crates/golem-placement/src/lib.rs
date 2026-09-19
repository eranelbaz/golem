//! Three-tier expert placement policy: disk, RAM, VRAM (tech design section 6).
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
