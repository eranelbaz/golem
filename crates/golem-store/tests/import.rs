//! `golem_store::import::scan_source` against the shape-realistic fixture:
//! shard-set discovery, arch gate, live-only planes, and pinned geometry.
use golem_store::import::{ImportError, hash_shards, scan_source, verify_replica};
use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    xtask::shape_moe::cached_fixture()[0]
        .parent()
        .unwrap()
        .to_path_buf()
}

fn tempdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("golem-import-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn copy_fixture_to(dst: &Path) {
    for s in xtask::shape_moe::cached_fixture() {
        std::fs::copy(&s, dst.join(s.file_name().unwrap())).unwrap();
    }
}

#[test]
fn scan_fixture_lists_live_layers_only() {
    let scan = scan_source(&fixture_dir()).unwrap();
    assert_eq!(scan.shards.len(), 2);
    assert_eq!(scan.arch, "glm5next");
    let layers: Vec<u32> = scan
        .planes
        .iter()
        .map(|p| p.layer)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    assert_eq!(
        layers,
        xtask::shape_moe::dims::LIVE.to_vec(),
        "dense blk.0 and MTP blk.4 excluded"
    );
    assert_eq!(scan.planes.len(), 3 * xtask::shape_moe::dims::LIVE.len());
    assert!(scan.planes.iter().all(
        |p| p.tensor.ends_with("_exps.weight") && p.experts == xtask::shape_moe::dims::N_EXPERT
    ));
}

#[test]
fn scan_geometry_pins_expert_bytes() {
    let scan = scan_source(&fixture_dir()).unwrap();
    for p in &scan.planes {
        let bytes = std::fs::read(&scan.shards[p.shard]).unwrap();
        let kind = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
            .iter()
            .position(|k| p.tensor.contains(k))
            .unwrap();
        for e in [0, 1, p.experts - 1] {
            let a = (p.base + u64::from(e) * p.stride) as usize;
            let want: Vec<u8> = (0..4)
                .map(|i| xtask::shape_moe::fill_byte(p.layer, kind, e, i))
                .collect();
            assert_eq!(&bytes[a..a + 4], &want[..], "{} expert {e}", p.tensor);
        }
    }
}

#[test]
fn scan_rejects_missing_shard() {
    let tmp = tempdir("missing");
    let src = &xtask::shape_moe::cached_fixture()[0];
    std::fs::copy(src, tmp.join(src.file_name().unwrap())).unwrap();
    assert!(matches!(scan_source(&tmp), Err(ImportError::Source { .. })));
}

#[test]
fn scan_rejects_mmproj_and_foreign_stem() {
    let tmp = tempdir("mmproj");
    copy_fixture_to(&tmp);
    std::fs::write(tmp.join("mmproj-x.gguf"), b"GGUF").unwrap();
    assert!(scan_source(&tmp).is_err());

    let tmp = tempdir("stem");
    copy_fixture_to(&tmp);
    let s2 = xtask::shape_moe::cached_fixture()[1]
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    std::fs::rename(
        tmp.join(&s2),
        tmp.join(s2.replacen("shape-moe", "other", 1)),
    )
    .unwrap();
    assert!(scan_source(&tmp).is_err());
}

#[test]
fn scan_rejects_truncated_shard() {
    let tmp = tempdir("trunc");
    copy_fixture_to(&tmp);
    let s2 = tmp.join(xtask::shape_moe::cached_fixture()[1].file_name().unwrap());
    let f = std::fs::OpenOptions::new().write(true).open(&s2).unwrap();
    f.set_len(f.metadata().unwrap().len() - 1).unwrap();
    let e = scan_source(&tmp).unwrap_err().to_string();
    assert!(e.contains("past end"), "{e}");
}

#[test]
fn scan_requires_chat_template_kv() {
    // Fixture shard 1 carries `tokenizer.chat_template` in its KV metadata;
    // patching one key byte in place (same length, so no data offsets move)
    // leaves a shard that is otherwise valid but lacks the KV.
    let tmp = tempdir("tmpl");
    copy_fixture_to(&tmp);
    let s1 = tmp.join(xtask::shape_moe::cached_fixture()[0].file_name().unwrap());
    let mut bytes = std::fs::read(&s1).unwrap();
    let pos = bytes
        .windows(23)
        .position(|w| w == b"tokenizer.chat_template")
        .expect("fixture shard 1 must contain the chat_template KV key");
    bytes[pos] = b'x';
    std::fs::write(&s1, bytes).unwrap();
    let err = scan_source(&tmp).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("missing tokenizer.chat_template"), "{msg}");
    assert!(matches!(err, ImportError::Source { .. }));
}

fn no_tmp(dir: &Path) -> bool {
    std::fs::read_dir(dir).unwrap().all(|e| {
        !e.unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".golem-import")
    })
}

// The fixture shards are ≈4 MB — smaller than the default 8 MiB chunk — so a
// fixed chunk size could never exercise an expert spanning a chunk boundary;
// the chunk-size parameter exists for exactly this test.
#[test]
fn digests_and_crcs_are_chunk_size_invariant() {
    use sha2::{Digest, Sha256};
    let scan = scan_source(&fixture_dir()).unwrap();
    let mut prev: Option<Vec<Vec<u32>>> = None;
    for chunk in [4096 + 7, 64 * 1024, 1 << 20, 8 << 20] {
        let out = hash_shards(&scan, &[], chunk).unwrap();
        for (i, p) in scan.shards.iter().enumerate() {
            let bytes = std::fs::read(p).unwrap();
            assert_eq!(out.sha256[i], format!("{:x}", Sha256::digest(&bytes)));
            assert_eq!(out.bytes[i], bytes.len() as u64);
        }
        for (pi, plane) in scan.planes.iter().enumerate() {
            let bytes = std::fs::read(&scan.shards[plane.shard]).unwrap();
            for e in 0..plane.experts as usize {
                let a = plane.base as usize + e * plane.stride as usize;
                assert_eq!(
                    out.crc[pi][e],
                    crc32c::crc32c(&bytes[a..a + plane.stride as usize]),
                    "chunk {chunk}"
                );
            }
        }
        if let Some(p) = &prev {
            assert_eq!(p, &out.crc);
        }
        prev = Some(out.crc);
    }
}

#[test]
fn replica_clone_is_byte_identical_and_clean() {
    let scan = scan_source(&fixture_dir()).unwrap();
    let r = tempdir("replica");
    hash_shards(&scan, std::slice::from_ref(&r), 64 * 1024).unwrap();
    for p in &scan.shards {
        assert_eq!(
            std::fs::read(p).unwrap(),
            std::fs::read(r.join(p.file_name().unwrap())).unwrap()
        );
    }
    assert!(no_tmp(&r), "tmp and lock removed");
}

#[test]
fn same_size_wrong_bytes_replica_is_recloned() {
    let scan = scan_source(&fixture_dir()).unwrap();
    let r = tempdir("stale");
    copy_fixture_to(&r);
    let victim = r.join(scan.shards[1].file_name().unwrap());
    let mut b = std::fs::read(&victim).unwrap();
    let mid = b.len() / 2;
    b[mid] ^= 0xff;
    std::fs::write(&victim, &b).unwrap();
    hash_shards(&scan, std::slice::from_ref(&r), 64 * 1024).unwrap();
    assert_eq!(
        std::fs::read(&scan.shards[1]).unwrap(),
        std::fs::read(&victim).unwrap()
    );
}

#[test]
fn verify_replica_reports_mismatch() {
    let scan = scan_source(&fixture_dir()).unwrap();
    let r = tempdir("verify");
    let f = r.join("x.gguf");
    std::fs::write(&f, b"not the shard").unwrap();
    let want = hash_shards(&scan, &[], 1 << 20).unwrap().sha256[1].clone();
    assert!(matches!(
        verify_replica(&f, &want, 1 << 20),
        Err(ImportError::ReplicaMismatch { .. })
    ));
}

#[test]
fn failed_replica_leaves_no_tmp_in_any_replica() {
    let scan = scan_source(&fixture_dir()).unwrap();
    let good = tempdir("good");
    let bad = tempdir("bad").join("missing-subdir");
    assert!(hash_shards(&scan, &[good.clone(), bad], 64 * 1024).is_err());
    assert!(no_tmp(&good));
}

#[test]
fn stale_tmp_is_swept_and_live_lock_refuses() {
    let scan = scan_source(&fixture_dir()).unwrap();
    let r = tempdir("sweep");
    std::fs::write(r.join("old.gguf.golem-import.123.0.tmp"), b"x").unwrap();
    hash_shards(&scan, std::slice::from_ref(&r), 64 * 1024).unwrap();
    assert!(no_tmp(&r));
    std::fs::write(r.join(".golem-import.lock"), b"not-a-pid").unwrap();
    hash_shards(&scan, std::slice::from_ref(&r), 64 * 1024).unwrap();
    assert!(no_tmp(&r), "stale lock taken over and released");
    std::fs::write(r.join(".golem-import.lock"), std::process::id().to_string()).unwrap();
    let e = hash_shards(&scan, std::slice::from_ref(&r), 64 * 1024)
        .unwrap_err()
        .to_string();
    assert!(e.contains(".golem-import.lock"), "{e}");
}
