//! Workspace automation: `gen-tiny-moe` (M1.2), parity runner (M1.4),
//! bench matrix (M5.3). Scaffold (task B1).
fn main() {
    match std::env::args().nth(1).as_deref() {
        None | Some("hello") => println!("xtask: workspace automation entry point"),
        Some(other) => {
            eprintln!("unknown xtask command: {other}");
            std::process::exit(2);
        }
    }
}
