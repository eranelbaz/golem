//! Port of the canonical `cs()` recipe in scripts/build-substrate.sh; never run tooling in-tree —
//! the hash covers untracked + gitignored files.
use std::fs;
use std::io::{self, BufRead};
use std::path::Path;

const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

const H0: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

pub struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64,
}

impl Sha256 {
    pub fn new() -> Self {
        Sha256 {
            state: H0,
            buf: [0; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
            if data.is_empty() {
                return; // fully buffered; falling through would clobber buf_len
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            self.compress(&block);
            data = &data[64..];
        }
        self.buf[..data.len()].copy_from_slice(data);
        self.buf_len = data.len();
    }

    pub fn finish(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8); // captured before padding
        self.update(&[0x80]);
        while self.buf_len != 56 {
            self.update(&[0]);
        }
        let tail = bit_len.to_be_bytes();
        self.update(&tail);
        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[4 * i],
                block[4 * i + 1],
                block[4 * i + 2],
                block[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (s, v) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

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
    let mut h = Sha256::new();
    h.update(data);
    hex_of(&h.finish())
}

pub fn hash_stream(root: &Path) -> io::Result<Vec<u8>> {
    let mut relatives = Vec::new();
    walk(root, &empty_os(), true, &mut relatives)?;
    sort_bytewise(&mut relatives);

    let mut stream = Vec::new();
    for rel in &relatives {
        let path = root.join(rel);
        let mut h = Sha256::new();
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
        push_sha256sum_line(&hex_of(&h.finish()), rel, &mut stream);
    }
    Ok(stream)
}

pub fn content_hash(root: &Path) -> io::Result<String> {
    let stream = hash_stream(root)?;
    let mut h = Sha256::new();
    h.update(&stream);
    Ok(hex_of(&h.finish()))
}

#[derive(Debug)]
pub enum PinError {
    Io(io::Error),
    MissingKey {
        pin_file: String,
    },
    Mismatch {
        expected: String,
        actual: String,
        pin_file: String,
    },
}

impl std::fmt::Display for PinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PinError::Io(e) => write!(f, "I/O error: {e}"),
            PinError::MissingKey { pin_file } => write!(
                f,
                "content_sha256 key missing/empty/malformed in {pin_file}"
            ),
            PinError::Mismatch {
                expected,
                actual,
                pin_file,
            } => write!(
                f,
                "vendor tree drifted (actual {actual} != expected {expected}, pin {pin_file})"
            ),
        }
    }
}

impl std::error::Error for PinError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PinError::Io(e) => Some(e),
            _ => None,
        }
    }
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
