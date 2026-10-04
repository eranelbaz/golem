//! `golem_store::import::scan_source` against the shape-realistic fixture:
//! shard-set discovery, arch gate, live-only planes, and pinned geometry.
use golem_store::import::{ImportError, scan_source};
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
