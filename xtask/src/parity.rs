//! `xtask parity` — T2 harness scaffold: validates the model parses as GGUF v3
//! with a `general.architecture` KV and emits the scaffold report line (full
//! parity execution is M1.4).

use std::path::PathBuf;

use crate::gguf;

/// Runs `xtask parity --model <PATH>`; exits 2 on bad usage or an unparseable
/// model, matching the existing xtask exit-code contract.
pub fn run(args: Vec<String>) {
    let mut model: Option<PathBuf> = None;
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => {
                model = Some(PathBuf::from(
                    it.next().unwrap_or_else(|| die("--model requires a value")),
                ));
            }
            other => die(&format!("unknown parity flag: {other}")),
        }
    }
    let model = model.unwrap_or_else(|| die("--model <PATH> is required"));
    let arch = match gguf::read_arch(&model) {
        Ok(arch) => arch,
        Err(e) => die(&e),
    };
    println!("{}", scaffold_json(&model.display().to_string(), &arch));
}

fn scaffold_json(model: &str, arch: &str) -> String {
    format!(
        "{{\"harness\":\"t2-parity\",\"status\":\"scaffold\",\"model\":\"{}\",\"arch\":\"{}\",\"note\":\"native leg arrives with M1.4 (token-ID comparison with logit-gap epsilon, M0.5 session 10)\"}}",
        json_escape(model),
        json_escape(arch),
    )
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn die(msg: &str) -> ! {
    eprintln!("parity: {msg}");
    std::process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_json_matches_pinned_line() {
        let v: serde_json::Value = serde_json::from_str(&scaffold_json("/tmp/m.gguf", "glm4moe"))
            .expect("scaffold line must be valid JSON");
        assert_eq!(v["harness"], "t2-parity");
        assert_eq!(v["status"], "scaffold");
        assert_eq!(v["model"], "/tmp/m.gguf");
        assert_eq!(v["arch"], "glm4moe");
        assert_eq!(
            v["note"],
            "native leg arrives with M1.4 (token-ID comparison with logit-gap epsilon, M0.5 session 10)"
        );
    }

    #[test]
    fn json_escape_quotes_and_backslashes() {
        assert_eq!(json_escape("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    }
}
