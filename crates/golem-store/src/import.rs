//! `golem import` (§5.5): scan a local shard set, hash + CRC it in one pass,
//! clone verified full mirrors, build the §4.2 manifest.

use sha2::Digest;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use golem_gguf::{
    blk_of, data_section_offset, ggml_row_bytes, kv_int, kv_str, live_layers, read_kvs, shard_path,
    tensor_entries,
};

/// Import failure.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    /// Bad source layout or header content.
    #[error("{path}: {msg}")]
    Source {
        /// Offending path.
        path: PathBuf,
        /// What was wrong.
        msg: String,
    },
    /// GGUF parse failure.
    #[error("{path}: {source}")]
    Gguf {
        /// Shard path.
        path: PathBuf,
        /// Parse error.
        #[source]
        source: golem_gguf::Error,
    },
    /// I/O failure.
    #[error("{path}: {source}")]
    Io {
        /// File or dir path.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },
    /// A replica's re-hash disagrees with the source.
    #[error("replica {path}: sha256 {got} != source {want}")]
    ReplicaMismatch {
        /// Replica file.
        path: PathBuf,
        /// Replica digest.
        got: String,
        /// Source digest.
        want: String,
    },
}

fn source(path: impl Into<PathBuf>, msg: impl Into<String>) -> ImportError {
    ImportError::Source {
        path: path.into(),
        msg: msg.into(),
    }
}

/// One expert plane: absolute offset of expert 0 and per-expert stride.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaneSpan {
    /// Full tensor name, e.g. `blk.1.ffn_gate_exps.weight`.
    pub tensor: String,
    /// Block the plane belongs to (live blocks only; MTP never lands here).
    pub layer: u32,
    /// Index into [`SourceScan::shards`].
    pub shard: usize,
    /// Absolute file offset of expert 0's first byte.
    pub base: u64,
    /// Bytes per expert slice (`ggml_row_bytes(ne0) · ne1`).
    pub stride: u64,
    /// Expert count (`dims[2]`).
    pub experts: u32,
}

/// Result of walking the source headers.
#[derive(Debug, Clone)]
pub struct SourceScan {
    /// Directory the shards were scanned from.
    pub dir: PathBuf,
    /// Shard paths in split order (a single file when unsplit).
    pub shards: Vec<PathBuf>,
    /// `(len, mtime)` per shard at scan time; re-checked after hashing.
    pub stamps: Vec<(u64, SystemTime)>,
    /// `general.architecture` of shard 1.
    pub arch: String,
    /// sha256 hex of `tokenizer.chat_template` on shard 1; required (scan
    /// errors with `ImportError::Source` when the KV is absent).
    pub chat_template_sha: String,
    /// Live expert planes, sorted by `(layer, tensor)`.
    pub planes: Vec<PlaneSpan>,
}

fn open_kvs(path: &Path) -> Result<(std::fs::File, Vec<(String, golem_gguf::KvVal)>), ImportError> {
    let mut f = std::fs::File::open(path).map_err(|e| ImportError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let kvs = read_kvs(&mut f).map_err(|e| ImportError::Gguf {
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok((f, kvs))
}

/// Walks `dir`'s GGUF shard set: discovers and validates the split layout,
/// gates on `general.architecture`, then records the absolute geometry of
/// every expert plane on a live layer. Every expert tensor (MTP included) is
/// bounds-checked against the file length *before* the live-layer filter, so
/// truncation inside an excluded block is still rejected.
///
/// # Errors
/// [`ImportError::Source`] for a bad layout, [`ImportError::Gguf`] for
/// unparseable headers, [`ImportError::Io`] for filesystem failures.
pub fn scan_source(dir: &Path) -> Result<SourceScan, ImportError> {
    let mut names: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| ImportError::Io {
        path: dir.to_path_buf(),
        source: e,
    })? {
        let entry = entry.map_err(|e| ImportError::Io {
            path: dir.to_path_buf(),
            source: e,
        })?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".gguf") {
            names.push(name);
        }
    }
    names.sort();
    if let Some(n) = names.iter().find(|n| n.starts_with("mmproj")) {
        return Err(source(
            dir.join(n),
            "mmproj projector shards are not an importable model",
        ));
    }
    let Some(first_name) = names.first().cloned() else {
        return Err(source(dir, "no .gguf files found"));
    };
    let first_path = dir.join(&first_name);
    let (_, first_kvs) = open_kvs(&first_path)?;

    // Shard-set shape: single file, or the exact -NNNNN-of-MMMMM family.
    let count = kv_int(&first_kvs, "split.count");
    let shards: Vec<PathBuf> = match count {
        None => {
            if names.len() != 1 {
                return Err(source(
                    &first_path,
                    format!("no split.count but {} .gguf files present", names.len()),
                ));
            }
            vec![first_path.clone()]
        }
        Some(count) => {
            if kv_int(&first_kvs, "split.no") != Some(0) {
                return Err(source(
                    &first_path,
                    format!(
                        "split.no {:?} != 0 (first sorted file must be shard 1)",
                        kv_int(&first_kvs, "split.no")
                    ),
                ));
            }
            let expected: Vec<PathBuf> = (0..count)
                .map(|i| shard_path(&first_path, i, count))
                .collect::<std::io::Result<_>>()
                .map_err(|e| ImportError::Io {
                    path: first_path.clone(),
                    source: e,
                })?;
            let listed: std::collections::HashSet<&str> =
                names.iter().map(String::as_str).collect();
            for p in &expected {
                let n = p.file_name().unwrap().to_string_lossy();
                if !listed.contains(n.as_ref()) {
                    return Err(source(&first_path, format!("missing shard {n}")));
                }
            }
            for n in &names {
                if !expected
                    .iter()
                    .any(|p| p.file_name().unwrap().to_string_lossy() == *n)
                {
                    return Err(source(dir.join(n), format!("unexpected file {n}")));
                }
            }
            expected
        }
    };

    // Shard-1 gates: arch, chat template, live range, tensor total.
    let arch = match kv_str(&first_kvs, "general.architecture") {
        Some(a) if a == "glm5next" => a.to_owned(),
        other => {
            return Err(source(
                &first_path,
                format!("general.architecture {other:?} != \"glm5next\""),
            ));
        }
    };
    let chat_template_sha = match kv_str(&first_kvs, "tokenizer.chat_template") {
        Some(t) => sha256_hex(t.as_bytes()),
        None => {
            return Err(source(&first_path, "missing tokenizer.chat_template"));
        }
    };
    let live = live_layers(&first_kvs).map_err(|e| ImportError::Gguf {
        path: first_path.clone(),
        source: e,
    })?;
    let want_tensors = match count {
        None => None,
        Some(_) => Some(
            kv_int(&first_kvs, "split.tensors.count")
                .ok_or_else(|| source(&first_path, "split model missing split.tensors.count"))?,
        ),
    };

    // Per-shard walk: split identity, section offset, expert geometry.
    let mut stamps = Vec::with_capacity(shards.len());
    let mut planes: Vec<PlaneSpan> = Vec::new();
    let mut total_tensors = 0u64;
    for (i, path) in shards.iter().enumerate() {
        let (mut f, kvs) = open_kvs(path)?;
        if let Some(c) = kv_int(&kvs, "split.count") {
            if Some(c) != count {
                return Err(source(
                    path,
                    format!("split.count {c} != first shard's {count:?}"),
                ));
            }
            let no = kv_int(&kvs, "split.no")
                .ok_or_else(|| source(path, "split.count present but split.no missing"))?;
            if no != i as u64 {
                return Err(source(path, format!("split.no {no} != {i}")));
            }
        }
        let data_start = data_section_offset(&mut f).map_err(|e| ImportError::Gguf {
            path: path.clone(),
            source: e,
        })?;
        let entries = tensor_entries(&mut f).map_err(|e| ImportError::Gguf {
            path: path.clone(),
            source: e,
        })?;
        total_tensors += entries.len() as u64;
        let md = f.metadata().map_err(|e| ImportError::Io {
            path: path.clone(),
            source: e,
        })?;
        let mtime = md.modified().map_err(|e| ImportError::Io {
            path: path.clone(),
            source: e,
        })?;
        stamps.push((md.len(), mtime));

        for e in &entries {
            if e.expert_kind().is_none() {
                continue;
            }
            let blk = blk_of(&e.name).ok_or_else(|| {
                source(
                    path,
                    format!("expert tensor {} lacks a blk.<N> prefix", e.name),
                )
            })?;
            let stride = ggml_row_bytes(e.ggml_type, e.dims[0])
                .and_then(|row| row.checked_mul(e.dims[1]))
                .ok_or_else(|| {
                    source(
                        path,
                        format!(
                            "plane {}: no whole-block rows (type {}, ne0 {})",
                            e.name, e.ggml_type, e.dims[0]
                        ),
                    )
                })?;
            let experts = u32::try_from(e.dims[2]).map_err(|_| {
                source(
                    path,
                    format!("plane {}: expert count {} out of range", e.name, e.dims[2]),
                )
            })?;
            let base = data_start.checked_add(e.offset).ok_or_else(|| {
                source(path, format!("plane {}: data offset overflows u64", e.name))
            })?;
            let end =
                base.checked_add(u64::from(experts).checked_mul(stride).ok_or_else(|| {
                    source(path, format!("plane {}: size overflows u64", e.name))
                })?)
                .ok_or_else(|| source(path, format!("plane {}: end overflows u64", e.name)))?;
            if end > md.len() {
                return Err(source(
                    path,
                    format!("plane {} runs past end of shard", e.name),
                ));
            }
            // Bounds before filter: an excluded (MTP/dense) plane must be
            // whole too, so the truncation test above fires on it.
            if !live.contains(&blk) {
                continue;
            }
            planes.push(PlaneSpan {
                tensor: e.name.clone(),
                layer: blk,
                shard: i,
                base,
                stride,
                experts,
            });
        }
    }

    if let Some(want) = want_tensors {
        if total_tensors != want {
            return Err(source(
                &first_path,
                format!("split.tensors.count {want} != {total_tensors} tensors across shards"),
            ));
        }
    }
    planes.sort_by(|a, b| (&a.layer, &a.tensor).cmp(&(&b.layer, &b.tensor)));
    if let Some(dup) = planes.windows(2).find(|w| w[0].tensor == w[1].tensor) {
        return Err(source(
            &first_path,
            format!("duplicate tensor {}", dup[0].tensor),
        ));
    }
    if planes.len() != 3 * live.len() as usize {
        return Err(source(
            &first_path,
            format!(
                "expected 3 expert planes (gate/up/down) per live layer, got {} planes across {} live layers",
                planes.len(),
                live.len()
            ),
        ));
    }
    Ok(SourceScan {
        dir: dir.to_path_buf(),
        shards,
        stamps,
        arch,
        chat_template_sha,
        planes,
    })
}

/// Lowercase hex.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Lowercase sha256 hex.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&sha2::Sha256::digest(bytes))
}

/// Default read/write chunk: big enough for NVMe sequential throughput.
pub const CHUNK_BYTES: usize = 8 << 20;

/// Per-shard digests and per-plane CRCs (index-aligned with
/// [`SourceScan::planes`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardDigests {
    /// Lowercase sha256 hex, one per shard in [`SourceScan::shards`] order.
    pub sha256: Vec<String>,
    /// Bytes hashed per shard.
    pub bytes: Vec<u64>,
    /// Running crc32c per expert slice, per plane.
    pub crc: Vec<Vec<u32>>,
}

/// Warn once per process when a `POSIX_FADV_DONTNEED` advise fails (advisory:
/// a failure costs cache pressure, never correctness).
static DONTNEED_WARN: Once = Once::new();

fn warn_dontneed(path: &Path) {
    DONTNEED_WARN.call_once(|| {
        tracing::warn!(path = %path.display(), "posix_fadvise DONTNEED failed; \
            expert bytes may linger in the page cache");
    });
}

/// Reads one span into `buf`, looping until the buffer is full or EOF.
/// Returns the number of bytes read (0 only at EOF into an empty buffer).
fn read_span(f: &mut std::fs::File, path: &Path, buf: &mut [u8]) -> Result<usize, ImportError> {
    let mut done = 0;
    while done < buf.len() {
        match f.read(&mut buf[done..]).map_err(|e| ImportError::Io {
            path: path.to_path_buf(),
            source: e,
        })? {
            0 => break,
            n => done += n,
        }
    }
    Ok(done)
}

/// Rejects replicas that resolve to the source directory itself, so a
/// replica commit can never rename over a source shard.
fn reject_aliases(dir: &Path, replicas: &[PathBuf]) -> Result<(), ImportError> {
    use std::os::unix::fs::MetadataExt;
    let dev_ino = |p: &Path| -> Result<(u64, u64), ImportError> {
        let md = std::fs::metadata(p).map_err(|e| ImportError::Io {
            path: p.to_path_buf(),
            source: e,
        })?;
        Ok((md.dev(), md.ino()))
    };
    let src = dir.canonicalize().map_err(|e| ImportError::Io {
        path: dir.to_path_buf(),
        source: e,
    })?;
    let src_di = dev_ino(&src)?;
    for r in replicas {
        let rp = r.canonicalize().map_err(|e| ImportError::Io {
            path: r.clone(),
            source: e,
        })?;
        if rp == src || dev_ino(&rp)? == src_di {
            return Err(source(r, "replica aliases the source directory"));
        }
    }
    Ok(())
}

/// Deletes every leftover `*.golem-import.*.tmp` under `replica`. Only
/// called while the replica lock is held, so no live writer's tmp exists.
fn sweep_tmps(replica: &Path) {
    let Ok(rd) = std::fs::read_dir(replica) else {
        return;
    };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.ends_with(".tmp") && name.contains(".golem-import.") {
            let _ = std::fs::remove_file(replica.join(e.file_name()));
        }
    }
}

/// Exclusive per-replica import lock: `create_new` on `<replica>/.golem-import.lock`,
/// staking our pid inside. A live holder (pid in `/proc`) refuses; a stale
/// lock (dead pid or unparseable) is taken over. `Drop` removes the lock.
struct ReplicaLock(PathBuf);

impl ReplicaLock {
    fn acquire(replica: &Path) -> Result<Self, ImportError> {
        let lock = replica.join(".golem-import.lock");
        for _ in 0..2 {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock)
            {
                Ok(mut f) => {
                    f.write_all(std::process::id().to_string().as_bytes())
                        .map_err(|e| ImportError::Io {
                            path: lock.clone(),
                            source: e,
                        })?;
                    sweep_tmps(replica);
                    return Ok(Self(lock));
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let pid = std::fs::read_to_string(&lock)
                        .ok()
                        .and_then(|s| s.trim().parse::<u32>().ok());
                    let live = pid.is_some_and(|p| Path::new("/proc").join(p.to_string()).exists());
                    if live {
                        return Err(source(
                            &lock,
                            format!("import pid {} holds {}", pid.unwrap(), lock.display()),
                        ));
                    }
                    let _ = std::fs::remove_file(&lock);
                }
                Err(e) => {
                    return Err(ImportError::Io {
                        path: lock,
                        source: e,
                    });
                }
            }
        }
        Err(source(&lock, "lost the stale-lock takeover race"))
    }
}

impl Drop for ReplicaLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Scratch write target: a temp file next to its final destination, removed
/// on drop unless committed. `commit` is the only durable publication path
/// (fsync file, rename, fsync parent dir).
struct TmpFile {
    /// Final destination of [`Self::commit`].
    dst: PathBuf,
    /// Current temp path; removed by `Drop` unless committed.
    path: PathBuf,
    file: std::fs::File,
    committed: bool,
    /// Bytes written since the last `sync_data`.
    dirty: u64,
    /// File offset up to which bytes are synced.
    synced: u64,
}

/// Per-tmp dirty-page budget between `sync_data` calls: bounds the page
/// cache held by clones at ≤ 1 GiB per tmp against 93 GiB RAM.
const SYNC_EVERY: u64 = 1 << 30;

impl TmpFile {
    fn create(dst: &Path) -> Result<Self, ImportError> {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut name = dst.file_name().unwrap_or_default().to_os_string();
        name.push(format!(".golem-import.{}.{}.tmp", std::process::id(), seq));
        let path = dst
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf()
            .join(name);
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| ImportError::Io {
                path: path.clone(),
                source: e,
            })?;
        Ok(Self {
            dst: dst.to_path_buf(),
            path,
            file,
            committed: false,
            dirty: 0,
            synced: 0,
        })
    }

    /// Accounts `n` newly written bytes, syncing and evicting at the
    /// per-tmp dirty budget.
    fn pump(&mut self, n: u64) -> Result<(), ImportError> {
        self.dirty += n;
        if self.dirty >= SYNC_EVERY {
            self.settle()?;
        }
        Ok(())
    }

    /// Syncs everything written so far and drops those pages from cache.
    fn settle(&mut self) -> Result<(), ImportError> {
        if self.dirty > 0 {
            self.file.sync_data().map_err(|e| ImportError::Io {
                path: self.path.clone(),
                source: e,
            })?;
            if !crate::sys::drop_page_cache(&self.file, self.synced, self.dirty as usize) {
                warn_dontneed(&self.path);
            }
            self.synced += self.dirty;
            self.dirty = 0;
        }
        Ok(())
    }

    /// Durably publishes the temp file to its destination: fsync the file,
    /// rename over `dst`, fsync the parent directory.
    fn commit(mut self) -> Result<(), ImportError> {
        self.file.sync_all().map_err(|e| ImportError::Io {
            path: self.path.clone(),
            source: e,
        })?;
        std::fs::rename(&self.path, &self.dst).map_err(|e| ImportError::Io {
            path: self.dst.clone(),
            source: e,
        })?;
        let parent = self.dst.parent().unwrap_or(Path::new("."));
        let dir = std::fs::File::open(parent).map_err(|e| ImportError::Io {
            path: parent.to_path_buf(),
            source: e,
        })?;
        dir.sync_all().map_err(|e| ImportError::Io {
            path: parent.to_path_buf(),
            source: e,
        })?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for TmpFile {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// One source shard into each unverified tmp: straight read → write with
/// DONTNEED behind both cursors, no bounce buffer beyond the shared chunk.
fn clone_shard(src_path: &Path, tmp: &mut TmpFile, buf: &mut [u8]) -> Result<u64, ImportError> {
    let mut src = std::fs::File::open(src_path).map_err(|e| ImportError::Io {
        path: src_path.to_path_buf(),
        source: e,
    })?;
    let mut off = 0u64;
    loop {
        let n = read_span(&mut src, src_path, buf)?;
        if n == 0 {
            break;
        }
        tmp.file.write_all(&buf[..n]).map_err(|e| ImportError::Io {
            path: tmp.path.clone(),
            source: e,
        })?;
        tmp.pump(n as u64)?;
        if !crate::sys::drop_page_cache(&src, off, n) {
            warn_dontneed(src_path);
        }
        off += n as u64;
    }
    Ok(off)
}

/// One pass over the source shard set: computes per-shard sha256 + byte
/// counts and per-plane expert crc32cs, while cloning every replica that is
/// not already a byte-identical keep candidate. Replicas are verified
/// *before* being kept; a same-size-but-wrong candidate is re-cloned.
/// The source is re-stamped after each shard; any change mid-import aborts.
///
/// `chunk` is the I/O buffer size in bytes (see [`CHUNK_BYTES`]); digests
/// and CRCs are chunk-size invariant. Callers running the full import use
/// the default; tests shrink it to exercise expert planes spanning chunk
/// boundaries.
///
/// # Errors
/// [`ImportError::Source`] on aliasing or a source that changed mid-read,
/// [`ImportError::Io`] for filesystem failures, [`ImportError::ReplicaMismatch`]
/// when a re-cloned replica still fails verification.
pub fn hash_shards(
    scan: &SourceScan,
    replicas: &[PathBuf],
    chunk: usize,
) -> Result<ShardDigests, ImportError> {
    // Alias check first: a replica must never be able to rename over the
    // source directory.
    reject_aliases(&scan.dir, replicas)?;
    let mut locks = Vec::with_capacity(replicas.len());
    for r in replicas {
        locks.push(ReplicaLock::acquire(r)?);
    }

    let mut sha256 = Vec::with_capacity(scan.shards.len());
    let mut bytes = Vec::with_capacity(scan.shards.len());
    let mut crc: Vec<Vec<u32>> = scan
        .planes
        .iter()
        .map(|p| vec![0u32; p.experts as usize])
        .collect();
    let mut buf = vec![0u8; chunk.max(1)];

    for (s, src_path) in scan.shards.iter().enumerate() {
        let name = src_path.file_name().unwrap();
        let want_len = scan.stamps[s].0;
        let mut tmps: Vec<TmpFile> = Vec::new();
        let mut keeps: Vec<PathBuf> = Vec::new();
        for r in replicas {
            let cand = r.join(name);
            match std::fs::metadata(&cand) {
                Ok(md) if md.is_file() && md.len() == want_len => keeps.push(cand),
                _ => tmps.push(TmpFile::create(&cand)?),
            }
        }

        // Planes of this shard, sorted by base so the CRC feed below walks
        // them in file order.
        let mut order: Vec<usize> = (0..scan.planes.len())
            .filter(|&pi| scan.planes[pi].shard == s)
            .collect();
        order.sort_by_key(|&pi| scan.planes[pi].base);

        let mut src = std::fs::File::open(src_path).map_err(|e| ImportError::Io {
            path: src_path.to_path_buf(),
            source: e,
        })?;
        let mut hasher = sha2::Sha256::new();
        let mut off = 0u64;
        loop {
            let n = read_span(&mut src, src_path, &mut buf)?;
            if n == 0 {
                break;
            }
            let b = &buf[..n];
            hasher.update(b);
            let chunk_end = off + n as u64;
            for t in &mut tmps {
                t.file.write_all(b).map_err(|e| ImportError::Io {
                    path: t.path.clone(),
                    source: e,
                })?;
                t.pump(n as u64)?;
            }
            for &pi in &order {
                let p = &scan.planes[pi];
                let base = p.base;
                let stride = p.stride;
                if stride == 0 {
                    continue;
                }
                let end = base + u64::from(p.experts) * stride;
                if off >= end || chunk_end <= base {
                    continue;
                }
                // Experts e0..e1 of the plane intersect [off, chunk_end).
                let e0 = ((off.max(base) - base) / stride) as usize;
                let e1 = ((end.min(chunk_end) - base).div_ceil(stride)) as usize;
                for (e, slot) in crc[pi].iter_mut().enumerate().take(e1).skip(e0) {
                    let lo = (base + e as u64 * stride).max(off);
                    let hi = (base + (e as u64 + 1) * stride).min(chunk_end);
                    *slot =
                        crc32c::crc32c_append(*slot, &b[(lo - off) as usize..(hi - off) as usize]);
                }
            }
            if !crate::sys::drop_page_cache(&src, off, n) {
                warn_dontneed(src_path);
            }
            off = chunk_end;
        }

        // The source must not have moved under us.
        let md = src.metadata().map_err(|e| ImportError::Io {
            path: src_path.to_path_buf(),
            source: e,
        })?;
        let mtime = md.modified().map_err(|e| ImportError::Io {
            path: src_path.to_path_buf(),
            source: e,
        })?;
        if (md.len(), mtime) != scan.stamps[s] {
            return Err(source(src_path, "changed during import"));
        }
        drop(src);

        let digest = hex(&hasher.finalize());
        bytes.push(md.len());
        sha256.push(digest.clone());

        // Verify-then-commit every fresh clone; verify-then-keep every
        // candidate, re-cloning the ones whose bytes went stale.
        let mut stale: Vec<PathBuf> = Vec::new();
        for t in &mut tmps {
            t.settle()?;
            verify_replica(&t.path, &digest, chunk)?;
        }
        for cand in &keeps {
            match verify_replica(cand, &digest, chunk) {
                Ok(()) => {}
                Err(ImportError::ReplicaMismatch { .. }) => stale.push(cand.clone()),
                Err(e) => return Err(e),
            }
        }
        for cand in stale {
            let mut t = TmpFile::create(&cand)?;
            clone_shard(src_path, &mut t, &mut buf)?;
            t.settle()?;
            verify_replica(&t.path, &digest, chunk)?;
            t.commit()?;
        }
        for t in tmps {
            t.commit()?;
        }
    }

    drop(locks);
    Ok(ShardDigests { sha256, bytes, crc })
}

/// Re-hashes `path` in `chunk`-sized reads and compares against `want_sha`,
/// dropping pages behind the cursor (verification reads each replica file
/// exactly once).
///
/// # Errors
/// [`ImportError::Io`] on filesystem failures, [`ImportError::ReplicaMismatch`]
/// when the digest disagrees.
pub fn verify_replica(path: &Path, want_sha: &str, chunk: usize) -> Result<(), ImportError> {
    let mut f = std::fs::File::open(path).map_err(|e| ImportError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = vec![0u8; chunk.max(1)];
    let mut off = 0u64;
    loop {
        let n = read_span(&mut f, path, &mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        if !crate::sys::drop_page_cache(&f, off, n) {
            warn_dontneed(path);
        }
        off += n as u64;
    }
    let got = hex(&hasher.finalize());
    if got == want_sha {
        Ok(())
    } else {
        Err(ImportError::ReplicaMismatch {
            path: path.to_path_buf(),
            got,
            want: want_sha.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::Write;

    #[test]
    fn tmpfile_drop_removes_and_commit_keeps() {
        let d = std::env::temp_dir().join(format!("golem-import-unit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let dst = d.join("shard.gguf");
        {
            let mut t = TmpFile::create(&dst).unwrap();
            t.file.write_all(b"abc").unwrap();
        } // dropped uncommitted
        assert!(!dst.exists());
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0);
        let mut t = TmpFile::create(&dst).unwrap();
        t.file.write_all(b"abc").unwrap();
        t.commit().unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), b"abc");
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 1, "tmp gone");
        let _ = std::fs::remove_dir_all(&d);
    }
}
