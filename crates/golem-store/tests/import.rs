//! `golem_store::import::scan_source` against the shape-realistic fixture:
//! shard-set discovery, arch gate, live-only planes, and pinned geometry.
use golem_store::import::{
    ImportError, ImportRequest, hash_shards, import, scan_source, verify_replica,
};
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

fn req(source: PathBuf, replicas: Vec<PathBuf>) -> ImportRequest {
    ImportRequest {
        source,
        replicas,
        name: None,
        quant: None,
        revision_label: None,
    }
}

#[test]
fn import_builds_valid_manifest_with_crc_and_mirror() {
    let src = fixture_dir();
    let r = tempdir("mirror");
    let out = import(&ImportRequest {
        name: Some("shape".into()),
        quant: Some("Q4_K".into()),
        revision_label: Some("v1".into()),
        ..req(src.clone(), vec![r.clone()])
    })
    .unwrap();
    let m = out.manifest;
    assert_eq!(
        (
            m.name.as_str(),
            m.revision.as_deref(),
            m.revision_source.as_deref()
        ),
        ("shape", Some("v1"), Some("user"))
    );
    assert_eq!(m.arch.as_deref(), Some("glm5next"));
    let idx = m.expert_index.as_ref().unwrap();
    assert_eq!(idx.len(), 3 * xtask::shape_moe::dims::LIVE.len());
    assert!(
        idx.iter()
            .all(|e| e.crc32c.as_ref().unwrap().len() == e.experts as usize && e.layer != 4)
    );
    let want = vec![
        std::fs::canonicalize(&src).unwrap(),
        std::fs::canonicalize(&r).unwrap(),
    ];
    assert!(m.shards.iter().all(|s| s.replicas == want));
    for s in &m.shards {
        let p = src.join(&s.file);
        assert_eq!(
            s.sampled_sha256.as_deref(),
            Some(&golem_store::import::sampled_sha256(&p).unwrap()[..])
        );
        assert_eq!(s.sampled_sha256.as_ref().unwrap().len(), 16);
        let mt = |p: &Path| {
            std::fs::metadata(p)
                .unwrap()
                .modified()
                .unwrap()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        };
        assert_eq!(s.mtime, Some(mt(&p)));
        assert_eq!(mt(&r.join(&s.file)), mt(&p), "replica mtime = source");
    }
}

#[test]
fn import_rejects_aliased_replicas() {
    let src = fixture_dir();
    assert!(
        import(&req(src.clone(), vec![src.join(".")])).is_err(),
        "replica == source"
    );
    let r = tempdir("dup");
    assert!(
        import(&req(
            src.clone(),
            vec![r.clone(), r.join("..").join(r.file_name().unwrap())]
        ))
        .is_err(),
        "dup replica"
    );
    let link = tempdir("link").join("l");
    std::os::unix::fs::symlink(&src, &link).unwrap();
    assert!(import(&req(src, vec![link])).is_err(), "symlink to source");
}

#[test]
fn import_rejects_bad_name_and_empty_label() {
    let src = fixture_dir();
    for bad in ["", "..", "a/b"] {
        assert!(
            import(&ImportRequest {
                name: Some(bad.into()),
                ..req(src.clone(), vec![])
            })
            .is_err(),
            "{bad:?}"
        );
    }
    assert!(
        import(&ImportRequest {
            revision_label: Some(String::new()),
            ..req(src, vec![])
        })
        .is_err()
    );
}

#[test]
fn revision_from_hf_cache_metadata_and_rejects_non_hex() {
    for (line1, want) in [
        ("abc1234def", Some("abc1234def")),
        ("not a sha", None),
        // Hash lines are matched without trimming: surrounding whitespace
        // disqualifies the line, so the revision is `unknown`.
        ("  abc1234def  ", None),
    ] {
        let root = tempdir(&format!("hf-{}", want.is_some()));
        let sub = root.join("Q");
        let meta = root.join(".cache/huggingface/download/Q");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::create_dir_all(&meta).unwrap();
        copy_fixture_to(&sub);
        let n = xtask::shape_moe::cached_fixture()[0]
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        std::fs::write(
            meta.join(format!("{n}.metadata")),
            format!("{line1}\nffff\n1.0\n"),
        )
        .unwrap();
        let m = import(&req(sub, vec![])).unwrap().manifest;
        assert_eq!(m.revision.as_deref(), want);
        assert_eq!(
            m.revision_source.as_deref(),
            Some(if want.is_some() {
                "hf-cache"
            } else {
                "unknown"
            })
        );
    }
}

#[test]
fn revision_walks_all_ancestors_and_splits_absent_from_unknown() {
    let root = tempdir("hf-nested");
    let sha = "beefcafe1234567";
    let snap = Path::new("hub/models--org--name/snapshots").join(sha);
    let sub = root.join(&snap);
    std::fs::create_dir_all(&sub).unwrap();
    copy_fixture_to(&sub);
    let n = xtask::shape_moe::cached_fixture()[0]
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let meta = root
        .join(".cache/huggingface/download")
        .join(&snap)
        .join(format!("{n}.metadata"));

    // Metadata absent at every ancestor → no revision and no source line.
    let m = import(&req(sub.clone(), vec![])).unwrap().manifest;
    assert_eq!(
        (m.revision.as_deref(), m.revision_source.as_deref()),
        (None, None)
    );

    // The first *existing* metadata decides even when bad, so a bad one at
    // the found ancestor yields `unknown`, never a walk past it.
    std::fs::create_dir_all(meta.parent().unwrap()).unwrap();
    std::fs::write(&meta, "garbage line\n").unwrap();
    let m = import(&req(sub.clone(), vec![])).unwrap().manifest;
    assert_eq!(
        (m.revision.as_deref(), m.revision_source.as_deref()),
        (None, Some("unknown"))
    );

    // Commit hash on line 1 of that same file, reached three ancestors up.
    std::fs::write(&meta, format!("{sha}\netag\n1.0\n")).unwrap();
    let m = import(&req(sub, vec![])).unwrap().manifest;
    assert_eq!(m.revision.as_deref(), Some(sha));
    assert_eq!(m.revision_source.as_deref(), Some("hf-cache"));
}

// Independent full-plane CRC check against the real GLM-5.3-Flash import:
// locates the tensor via `golem_gguf::tensor_entries` (not `PlaneSpan`) and
// verifies every expert's bytes against the manifest CRC32C. Needs ~1.6 GB of
// shard reads, so it is opt-in; without GOLEM_IMPORT_MANIFEST it fails (AGENTS
// evidence rule), it never silently passes.
#[test]
#[ignore = "reads ~1.6 GB of real model shards; set GOLEM_IMPORT_MANIFEST and run with --run-ignored only"]
fn real_plane_crc() {
    use std::io::{Read, Seek};

    let path = std::env::var("GOLEM_IMPORT_MANIFEST")
        .expect("GOLEM_IMPORT_MANIFEST must point at the imported manifest");
    let m = golem_store::load_manifest(Path::new(&path)).unwrap();
    let entry = m
        .expert_index
        .as_ref()
        .expect("manifest must carry an expert index")
        .iter()
        .find(|e| e.layer == 23 && e.tensor.contains("ffn_down_exps"))
        .expect("middle-layer ffn_down_exps entry")
        .clone();
    let crcs = entry.crc32c.as_ref().expect("manifest CRC32C per expert");
    assert_eq!(crcs.len(), entry.experts as usize);

    // Locate the shard holding the tensor by walking each shard's tensor
    // table (header-only reads); replicas[0] is the source copy.
    let mut loc = None;
    for s in &m.shards {
        let mut f = std::fs::File::open(s.replicas[0].join(&s.file)).unwrap();
        let entries = golem_gguf::tensor_entries(&mut f).unwrap();
        if let Some(t) = entries.iter().find(|t| t.name == entry.tensor) {
            let dso = golem_gguf::data_section_offset(&mut f).unwrap();
            loc = Some((f, dso + t.offset));
            break;
        }
    }
    let (mut f, base) = loc.expect("tensor found in some shard");

    let row = &mut vec![0u8; entry.bytes_per_expert as usize];
    for (i, off) in entry.offsets_bytes.iter().enumerate() {
        f.seek(std::io::SeekFrom::Start(base + off)).unwrap();
        f.read_exact(row).unwrap();
        assert_eq!(crc32c::crc32c(row), crcs[i], "{} expert {i}", entry.tensor);
    }
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
    // Empty/unparseable lock is treated as live: the holder may not have
    // written its pid yet, so taking over could kill a concurrent import.
    std::fs::write(r.join(".golem-import.lock"), b"not-a-pid").unwrap();
    assert!(hash_shards(&scan, std::slice::from_ref(&r), 64 * 1024).is_err());
    assert!(std::fs::read(r.join(".golem-import.lock")).unwrap() == b"not-a-pid");
    // Dead-pid lock is taken over and released.
    let dead = ((1000..100000u32)
        .find(|&p| !Path::new("/proc").join(p.to_string()).exists())
        .expect("no free pid"))
    .to_string();
    std::fs::write(r.join(".golem-import.lock"), &dead).unwrap();
    hash_shards(&scan, std::slice::from_ref(&r), 64 * 1024).unwrap();
    assert!(no_tmp(&r), "stale lock taken over and released");
    std::fs::write(r.join(".golem-import.lock"), std::process::id().to_string()).unwrap();
    let e = hash_shards(&scan, std::slice::from_ref(&r), 64 * 1024)
        .unwrap_err()
        .to_string();
    assert!(e.contains(".golem-import.lock"), "{e}");
}
