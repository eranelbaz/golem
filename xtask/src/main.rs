//! Binary entry point for workspace xtask commands.
#![allow(missing_docs)]
use std::io::Write;

use xtask::{parity, tiny_moe};

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        None | Some("hello") => println!("xtask: workspace automation entry point"),
        Some("gen-tiny-moe") => gen_tiny_moe(args.collect()),
        Some("parity") => parity::run(args.collect()),
        Some(other) => {
            eprintln!("unknown xtask command: {other}");
            std::process::exit(2);
        }
    }
}

fn die(msg: &str) -> ! {
    eprintln!("gen-tiny-moe: {msg}");
    std::process::exit(2);
}

fn parse_seed(s: &str) -> u64 {
    s.parse::<u64>()
        .ok()
        .or_else(|| {
            s.strip_prefix("0x")
                .and_then(|hex| u64::from_str_radix(hex, 16).ok())
        })
        .unwrap_or_else(|| die(&format!("--seed must be u64 decimal or 0x hex, got {s:?}")))
}

fn gen_tiny_moe(args: Vec<String>) {
    let mut out = tiny_moe::DEFAULT_OUT.to_string();
    let mut seed = tiny_moe::DEFAULT_SEED;
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out = it.next().unwrap_or_else(|| die("--out requires a value")),
            "--seed" => {
                let v = it.next().unwrap_or_else(|| die("--seed requires a value"));
                seed = parse_seed(&v);
            }
            other => {
                eprintln!("unknown gen-tiny-moe flag: {other}");
                std::process::exit(2);
            }
        }
    }
    let path = std::path::Path::new(&out);
    if let Some(dir) = path.parent()
        && !dir.as_os_str().is_empty()
        && let Err(e) = std::fs::create_dir_all(dir)
    {
        die(&format!("create output dir {}: {e}", dir.display()));
    }
    let tmp = path.with_extension("gguf.tmp");
    if let Err(e) = write_model(&tmp, seed) {
        let _ = std::fs::remove_file(&tmp);
        die(&format!("write {}: {e}", tmp.display()));
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        die(&format!("persist {out}: {e}"));
    }
    println!("wrote {out} (seed {seed})");
}

fn write_model(tmp: &std::path::Path, seed: u64) -> std::io::Result<()> {
    let f = std::fs::File::create(tmp)?;
    let mut w = std::io::BufWriter::new(f);
    tiny_moe::generate(&mut w, seed)?;
    w.flush()
}
