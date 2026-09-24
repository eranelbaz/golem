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
    // serde_json escapes the interpolated values; the fixed key order keeps
    // the pinned report line stable.
    format!(
        "{{\"harness\":\"t2-parity\",\"status\":\"scaffold\",\"model\":{},\"arch\":{},\"note\":\"native leg arrives with M1.4 (token-ID comparison with logit-gap epsilon, M0.5 session 10)\"}}",
        serde_json::to_string(model).unwrap(),
        serde_json::to_string(arch).unwrap(),
    )
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
    fn scaffold_json_escapes_hostile_values() {
        let v: serde_json::Value = serde_json::from_str(&scaffold_json("a\"b\\c\nd", "glm4moe"))
            .expect("escaped line must stay valid JSON");
        assert_eq!(v["model"], "a\"b\\c\nd");
    }
}
