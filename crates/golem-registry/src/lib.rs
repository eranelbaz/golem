//! Instance registry, lifecycle control, heartbeats (tech design section 9.2).
//!
//! Scaffold crate (task B1); implementation lands with its board task in
//! board task — see the tech design for the authoritative behavior.
#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert_eq!(env!("CARGO_PKG_NAME"), "golem-registry");
    }
}
