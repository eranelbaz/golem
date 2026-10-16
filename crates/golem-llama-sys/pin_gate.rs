//! Port of the canonical `cs()` recipe in scripts/build-substrate.sh; never run tooling in-tree —
//! the hash covers untracked + gitignored files.
use sha2::Digest;
use std::fs;
use std::io::{self, BufRead};
use std::path::Path;

fn hex_of(digest: &[u8; 32]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(64);
    for byte in digest {
        let _ = write!(s, "{byte:02x}");
    }
    s
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = sha2::Sha256::new();
    h.update(data);
    hex_of(&h.finalize().into())
}

pub fn hash_stream(root: &Path) -> io::Result<Vec<u8>> {
    let mut relatives = Vec::new();
    walk(root, &empty_os(), true, &mut relatives)?;
    sort_bytewise(&mut relatives);

    let mut stream = Vec::new();
    for rel in &relatives {
        let path = root.join(rel);
        let mut h = sha2::Sha256::new();
        let file = fs::File::open(&path)?;
        let mut reader = io::BufReader::with_capacity(1 << 20, file);
        loop {
            let chunk = reader.fill_buf()?;
            if chunk.is_empty() {
                break;
            }
            let len = chunk.len();
            h.update(chunk);
            reader.consume(len);
        }
        push_sha256sum_line(&hex_of(&h.finalize().into()), rel, &mut stream);
    }
    Ok(stream)
}

pub fn content_hash(root: &Path) -> io::Result<String> {
    let stream = hash_stream(root)?;
    let mut h = sha2::Sha256::new();
    h.update(&stream);
    Ok(hex_of(&h.finalize().into()))
}

#[derive(Debug, thiserror::Error)]
pub enum PinError {
    #[error("I/O error: {0}")]
    Io(io::Error),
    #[error("content_sha256 key missing/empty/malformed in {pin_file}")]
    MissingKey { pin_file: String },
    #[error("vendor tree drifted (actual {actual} != expected {expected}, pin {pin_file})")]
    Mismatch {
        expected: String,
        actual: String,
        pin_file: String,
    },
}

impl From<io::Error> for PinError {
    fn from(e: io::Error) -> Self {
        PinError::Io(e)
    }
}

pub fn verify_pin(vendor_root: &Path, vendor_sha_path: &Path) -> Result<(), PinError> {
    let text = fs::read_to_string(vendor_sha_path)?;
    let expected = read_kv_value(&text, "content_sha256")
        .map(|v| v.to_ascii_lowercase())
        .ok_or_else(|| PinError::MissingKey {
            pin_file: vendor_sha_path.display().to_string(),
        })?;
    if !is_sha256_hex(&expected) {
        return Err(PinError::MissingKey {
            pin_file: vendor_sha_path.display().to_string(),
        });
    }
    let actual = content_hash(vendor_root)?;
    if actual == expected {
        Ok(())
    } else {
        Err(PinError::Mismatch {
            expected,
            actual,
            pin_file: vendor_sha_path.display().to_string(),
        })
    }
}

pub fn read_kv_value(text: &str, key: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        let Some(after_key) = trimmed.strip_prefix(key) else {
            continue;
        };
        let after_key = after_key.trim_start();
        let Some(after_eq) = after_key.strip_prefix('=') else {
            continue;
        };
        let value = match after_eq.split_once('#') {
            Some((v, _)) => v,
            None => after_eq,
        };
        let value = value.trim();
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn empty_os() -> std::ffi::OsString {
    std::ffi::OsString::new()
}

#[cfg(unix)]
fn os_from_bytes(bytes: &[u8]) -> std::ffi::OsString {
    use std::os::unix::ffi::OsStringExt;
    std::ffi::OsString::from_vec(bytes.to_vec())
}

#[cfg(unix)]
fn os_bytes(s: &std::ffi::OsStr) -> &[u8] {
    use std::os::unix::ffi::OsStrExt;
    s.as_bytes()
}

#[cfg(not(unix))]
fn os_from_bytes(bytes: &[u8]) -> std::ffi::OsString {
    std::ffi::OsString::from(String::from_utf8_lossy(bytes).into_owned())
}

#[cfg(not(unix))]
fn os_bytes(s: &std::ffi::OsStr) -> &[u8] {
    s.to_str().unwrap_or_default().as_bytes()
}

fn join_rel(parent: &std::ffi::OsStr, name: &std::ffi::OsStr) -> std::ffi::OsString {
    if parent.is_empty() {
        return name.to_os_string();
    }
    let mut bytes = os_bytes(parent).to_vec();
    bytes.push(b'/');
    bytes.extend_from_slice(os_bytes(name));
    os_from_bytes(&bytes)
}

fn sort_bytewise(v: &mut [std::ffi::OsString]) {
    v.sort_by(|a, b| os_bytes(a).cmp(os_bytes(b)));
}

fn walk(
    dir: &Path,
    rel: &std::ffi::OsStr,
    is_root: bool,
    out: &mut Vec<std::ffi::OsString>,
) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        if is_root && (name == *".git" || name == *"build") {
            continue;
        }
        let child_rel = join_rel(rel, &name);
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            if name == *"__pycache__" {
                continue;
            }
            walk(&entry.path(), &child_rel, false, out)?;
        } else if file_type.is_file() {
            out.push(child_rel);
        }
    }
    Ok(())
}

fn push_sha256sum_line(hex: &str, rel: &std::ffi::OsStr, stream: &mut Vec<u8>) {
    let mut name = vec![b'.', b'/'];
    name.extend_from_slice(os_bytes(rel));
    let escaped = name.contains(&b'\\') || name.contains(&b'\n');
    if escaped {
        stream.push(b'\\');
    }
    stream.extend_from_slice(hex.as_bytes());
    stream.push(b' ');
    stream.push(b' ');
    if escaped {
        for &b in &name {
            match b {
                b'\\' => stream.extend_from_slice(b"\\\\"),
                b'\n' => stream.extend_from_slice(b"\\n"),
                _ => stream.push(b),
            }
        }
    } else {
        stream.extend_from_slice(&name);
    }
    stream.push(b'\n');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let msg = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(
            sha256_hex(msg.as_bytes()),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        let long = "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno\
ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
        assert_eq!(
            sha256_hex(long.as_bytes()),
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
        );
    }

    mod parser {
        use super::*;

        #[test]
        fn reads_value_and_ignores_comments_and_prefix_confusion() {
            let text = "# comment\ncontent_sha256 = abc123 # inline tail\n\nother_key = x\n";
            assert_eq!(read_kv_value(text, "content_sha256"), Some("abc123".into()));
            assert_eq!(read_kv_value(text, "other_key"), Some("x".into()));
            let tricky = "xcontent_sha256 = deadbeef\n";
            assert_eq!(read_kv_value(tricky, "content_sha256"), None);
        }

        #[test]
        fn tolerates_crlf_and_spaces() {
            let text = "content_sha256 =  f00d \r\n";
            assert_eq!(read_kv_value(text, "content_sha256"), Some("f00d".into()));
        }

        #[test]
        fn empty_or_missing_values_are_none() {
            assert_eq!(read_kv_value("content_sha256 =\n", "content_sha256"), None);
            assert_eq!(read_kv_value("", "content_sha256"), None);
        }
    }

    mod verify_pin_tests {
        use super::*;
        use std::path::PathBuf;

        fn tempdir(tag: &str) -> PathBuf {
            let dir = std::env::temp_dir()
                .join(format!("golem-pin-gate-unit-{}-{tag}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("tempdir");
            dir
        }

        struct Guard(PathBuf);
        impl Drop for Guard {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }

        /// The pin must sit outside the hashed tree, or the digest covers itself.
        fn setup(tag: &str) -> (PathBuf, PathBuf) {
            let root = tempdir(tag);
            fs::create_dir_all(root.join("tree")).unwrap();
            fs::write(root.join("tree/f.txt"), b"payload").unwrap();
            let pin = root.join("VENDOR_SHA");
            (root.join("tree"), pin)
        }

        #[test]
        fn ok_on_pristine_tree() {
            let (tree, pin) = setup("ok");
            let _g = Guard(tree.parent().unwrap().to_path_buf());
            let digest = content_hash(&tree).unwrap();
            fs::write(&pin, format!("content_sha256 = {digest}\n")).unwrap();
            assert!(verify_pin(&tree, &pin).is_ok());
        }

        #[test]
        fn pycache_is_ignored() {
            let (tree, pin) = setup("pycache");
            let _g = Guard(tree.parent().unwrap().to_path_buf());
            let digest = content_hash(&tree).unwrap();
            fs::write(&pin, format!("content_sha256 = {digest}\n")).unwrap();
            fs::create_dir_all(tree.join("a/__pycache__")).unwrap();
            fs::write(tree.join("a/__pycache__/x.pyc"), b"junk").unwrap();
            assert!(verify_pin(&tree, &pin).is_ok());
        }

        #[test]
        fn uppercase_pin_is_accepted() {
            let (tree, pin) = setup("upper");
            let _g = Guard(tree.parent().unwrap().to_path_buf());
            let digest = content_hash(&tree).unwrap().to_uppercase();
            fs::write(&pin, format!("content_sha256 = {digest}\n")).unwrap();
            assert!(verify_pin(&tree, &pin).is_ok());
        }

        #[test]
        fn drifted_tree_is_mismatch_with_paths() {
            let (tree, pin) = setup("drift");
            let _g = Guard(tree.parent().unwrap().to_path_buf());
            let digest = content_hash(&tree).unwrap();
            fs::write(&pin, format!("content_sha256 = {digest}\n")).unwrap();
            fs::write(tree.join("f.txt"), b"TAMPERED").unwrap();
            let err = verify_pin(&tree, &pin).unwrap_err();
            let PinError::Mismatch { pin_file, .. } = &err else {
                panic!("expected Mismatch, got {err:?}");
            };
            assert!(pin_file.ends_with("VENDOR_SHA"));
            assert!(err.to_string().contains("drifted"));
        }

        #[test]
        fn malformed_and_missing_keys_are_missing_key() {
            let (tree, pin) = setup("malformed");
            let _g = Guard(tree.parent().unwrap().to_path_buf());
            for body in ["", "content_sha256 = nothex", "commit = abc\n"] {
                fs::write(&pin, body).unwrap();
                let err = verify_pin(&tree, &pin).unwrap_err();
                assert!(matches!(err, PinError::MissingKey { .. }), "got {err:?}");
            }
        }
    }
}
