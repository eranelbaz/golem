//! Atomic file write shared by every harness leg that emits a dump.
//! Durability: fsync the temp file before rename, fsync the parent directory after — a crash leaves the old
//! or the new file, never a truncated one or an unlinked rename.

use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Writes `contents` to `path` atomically: temp file in the target's
/// directory (same filesystem, so rename cannot degrade to a copy), then
/// rename over the target — readers see the old or the new file, never a
/// partial one. The temp name is unique per process and per call, so
/// concurrent writers of the same target cannot race on the temp file.
pub fn write_atomic(path: &Path, contents: &str) -> std::io::Result<()> {
    let name = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "dump path has no file name",
        )
    })?;
    let tmp = path.with_file_name(format!(
        "{}.tmp.{}.{}",
        name.to_string_lossy(),
        std::process::id(),
        TMP_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    if let Err(e) = std::fs::File::create(&tmp).and_then(|mut f| {
        f.write_all(contents.as_bytes())?;
        // Contents must reach the platter before the rename that publishes
        // them, else a crash can publish empty/partial bytes.
        f.sync_all()
    }) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    // A failed rename must not orphan the temp file either. If the rename
    // succeeded but the parent-dir fsync failed, the error is still
    // returned while the file is already published: `path` exists and its
    // bytes are durable (the file itself was fsynced above) — only the
    // directory entry's persistence is unconfirmed, so callers must treat
    // this Err as "state unknown", not "target unchanged". The tmp removal
    // below is then a harmless no-op.
    let publish = std::fs::rename(&tmp, path).and_then(|()| {
        let dir = std::fs::File::open(path.parent().unwrap_or(Path::new(".")));
        dir.and_then(|d| d.sync_all())
    });
    if let Err(e) = publish {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "golem-engine-atomic-{}-{}-{tag}",
            std::process::id(),
            std::thread::current().name().unwrap_or("t")
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn write_then_read_back_and_no_tmp_leftover() {
        let dir = scratch_dir("roundtrip");
        let target = dir.join("dump.jsonl");
        write_atomic(&target, "line1\nline2\n").unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "line1\nline2\n");
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "leftover tmp files: {leftovers:?}");
    }

    #[test]
    fn unique_tmp_names_across_calls_do_not_collide() {
        let dir = scratch_dir("unique");
        let target = dir.join("dump.jsonl");
        write_atomic(&target, "a").unwrap();
        write_atomic(&target, "b").unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "b");
    }

    #[test]
    fn rejects_path_without_file_name() {
        let dir = scratch_dir("noname");
        assert!(write_atomic(&dir, "x").is_err());
    }
}
