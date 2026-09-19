//! `golem` CLI entry point (tech design §9): run, serve, list, stop, plan, pull.
//!
//! Scaffold (task B1): subcommands arrive with tasks M4.5/M4.6/M3.2.
fn main() {
    println!("golem {}", env!("CARGO_PKG_VERSION"));
}
