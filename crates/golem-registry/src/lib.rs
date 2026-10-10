//! Instance registry, lifecycle control, heartbeats (tech design section 9.2).
//!
//! One JSON file per instance (`<id>.json`, written tmp + rename) next to its
//! control socket (`<id>.sock`). Admission and reaping run under one shared
//! `flock` (`$XDG_STATE_HOME/golem/admit.lock`), so the check-then-register
//! of two concurrent starts cannot both pass.

pub mod control;
mod sys;

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use golem_api::{ErrorCode, ErrorEnvelope};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use control::{ControlRequest, ControlResponse, StopMode, send};

const GIB: u64 = 1 << 30;
/// RAM kept free for the OS and page cache (§10, O5).
const RAM_HEADROOM: u64 = 4 * GIB;
const HEARTBEAT: Duration = Duration::from_secs(1);

/// One registered instance (`<id>.json`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Instance {
    /// Short unique id.
    pub id: String,
    /// Owning process.
    pub pid: u32,
    /// `/proc/<pid>/stat` start time (clock ticks); detects pid reuse.
    pub pid_start_time: u64,
    /// `serve`, `run` or `optimize`.
    pub kind: String,
    /// `http://host:port` for `serve`, empty otherwise.
    pub endpoint: String,
    /// Served model name.
    pub model: String,
    /// sha256 of the canonical [`EffectiveConfig`] JSON (§9.4 fast path).
    pub config_hash: String,
    /// Unix seconds.
    pub started_at: u64,
    /// Planned device bytes (§10).
    pub planned_vram: u64,
    /// Planned total host bytes: pool, snapshots, staging ring, host overhead.
    pub planned_ram: u64,
    /// `loading`, `healthy`, `degraded`, `draining`, `fatal`.
    pub health: String,
    /// Unix seconds of the last heartbeat.
    pub heartbeat_at: u64,
    /// Helper processes owned by this instance.
    pub children: Vec<u32>,
    /// The requested configuration, for the §9.4 [`compatible`] check
    /// beyond the `config_hash` fast path.
    #[serde(default)]
    pub config: Option<EffectiveConfig>,
}

/// The requested configuration an instance was started with; its canonical
/// JSON hashes to [`Instance::config_hash`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveConfig {
    /// Canonical model path.
    pub model_path: PathBuf,
    /// Model file size.
    pub model_size: u64,
    /// Model mtime, Unix seconds.
    pub model_mtime: u64,
    /// Context length per sequence.
    pub ctx: u32,
    /// Decode slots.
    pub parallel: u32,
    /// Requested RAM expert pool, GiB.
    pub ram_experts_gib: u64,
    /// Requested snapshot arena; `None` = planner default.
    pub snapshot_ram_bytes: Option<u64>,
}

impl EffectiveConfig {
    /// sha256 hex of the canonical (field-ordered) JSON.
    pub fn hash(&self) -> String {
        let json = serde_json::to_vec(self).unwrap_or_default();
        Sha256::digest(&json)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

/// §9.4: `want` can attach to `running` when the model identity matches and
/// every requested budget fits inside the running one's.
pub fn compatible(running: &EffectiveConfig, want: &EffectiveConfig) -> Result<(), String> {
    if (&running.model_path, running.model_size, running.model_mtime)
        != (&want.model_path, want.model_size, want.model_mtime)
    {
        return Err(format!(
            "model {} differs from the running {}",
            want.model_path.display(),
            running.model_path.display()
        ));
    }
    let mut conflicts = Vec::new();
    if want.ctx > running.ctx {
        conflicts.push(format!("--ctx {} > running {}", want.ctx, running.ctx));
    }
    if want.parallel > running.parallel {
        conflicts.push(format!(
            "parallel {} > running {}",
            want.parallel, running.parallel
        ));
    }
    if want.ram_experts_gib > running.ram_experts_gib {
        conflicts.push(format!(
            "--ram-experts-gib {} > running {}",
            want.ram_experts_gib, running.ram_experts_gib
        ));
    }
    if want.snapshot_ram_bytes.unwrap_or(0) > running.snapshot_ram_bytes.unwrap_or(u64::MAX) {
        conflicts.push(format!(
            "snapshot arena {:?} B > running {:?} B",
            want.snapshot_ram_bytes, running.snapshot_ram_bytes
        ));
    }
    if conflicts.is_empty() {
        Ok(())
    } else {
        Err(conflicts.join(", "))
    }
}

/// Registry failures.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// Filesystem or socket failure.
    #[error("registry io: {0}")]
    Io(#[from] io::Error),
    /// Malformed entry or control message.
    #[error("registry json: {0}")]
    Json(#[from] serde_json::Error),
    /// No live instance with that id or name.
    #[error("no running instance `{0}`")]
    InstanceNotFound(String),
    /// More than one live instance serves that model name.
    #[error("model `{0}` is served by several instances ({1}); pass an id")]
    Ambiguous(String, String),
    /// Neither `XDG_RUNTIME_DIR`, `XDG_STATE_HOME` nor `HOME` is set.
    #[error("no registry dir: set XDG_RUNTIME_DIR or HOME")]
    NoDir,
    /// The instance answered with an error envelope.
    #[error("{0}")]
    Control(ErrorEnvelope),
}

impl From<RegistryError> for ErrorEnvelope {
    fn from(e: RegistryError) -> Self {
        match e {
            RegistryError::Control(env) => env,
            RegistryError::InstanceNotFound(_) => {
                ErrorEnvelope::new(ErrorCode::InstanceNotFound, e.to_string())
            }
            _ => ErrorEnvelope::new(ErrorCode::InvalidRequest, e.to_string()),
        }
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Start time (clock ticks since boot, `/proc/<pid>/stat` field 22) of a
/// live, non-zombie process.
pub fn pid_start_time(pid: u32) -> Result<u64, RegistryError> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"))?;
    // comm may contain spaces and parens: fields resume after the last ')'.
    let bad = || io::Error::new(io::ErrorKind::InvalidData, "malformed /proc stat");
    let mut f = stat.rsplit_once(')').ok_or_else(bad)?.1.split_whitespace();
    if f.next() == Some("Z") {
        return Err(io::Error::new(io::ErrorKind::NotFound, "zombie").into());
    }
    // `f` now starts at field 4; start time is field 22.
    f.nth(18)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| bad().into())
}

/// The process is alive and is the one the entry was written by.
fn owner_alive(i: &Instance) -> bool {
    sys::pid_exists(i.pid) && pid_start_time(i.pid).is_ok_and(|t| t == i.pid_start_time)
}

/// Sends `sig` to the instance's process; `InstanceNotFound` once the owner
/// exited or its pid was reused, so a stale entry never kills a stranger.
pub fn signal(i: &Instance, sig: libc::c_int) -> Result<(), RegistryError> {
    if !owner_alive(i) {
        return Err(RegistryError::InstanceNotFound(i.id.clone()));
    }
    Ok(sys::signal(i.pid, sig)?)
}

/// `true` once `i`'s process is gone (or its pid was reused).
pub fn exited(i: &Instance) -> bool {
    !owner_alive(i)
}

/// O5 admission: planned VRAM against the device total, planned total RAM
/// against `min(ram_total − Σ running planned_ram, ram_free − 4 GiB)`.
/// Target: one GLM instance plans pool 63 + snapshots 4 + staging 3.7 + host
/// 2 = 72.7 GiB; a second sees min(93 − 72.7, free − 4) ≈ 16 GiB → refused.
pub fn admit(
    running: &[Instance],
    new_vram: u64,
    new_ram: u64,
    mach: &golem_plan::Machine,
    allow_overcommit: bool,
) -> Result<(), ErrorEnvelope> {
    if allow_overcommit {
        return Ok(());
    }
    let used_vram: u64 = running.iter().map(|i| i.planned_vram).sum();
    let used_ram: u64 = running.iter().map(|i| i.planned_ram).sum();
    let ram_room = mach
        .ram_total
        .saturating_sub(used_ram)
        .min(mach.ram_free.saturating_sub(RAM_HEADROOM));
    let gib = |b: u64| b as f64 / GIB as f64;
    let mut over = Vec::new();
    if used_vram.saturating_add(new_vram) > mach.vram_total {
        over.push(format!(
            "VRAM {:.1} GiB + {:.1} GiB running > {:.1} GiB",
            gib(new_vram),
            gib(used_vram),
            gib(mach.vram_total)
        ));
    }
    if new_ram > ram_room {
        over.push(format!(
            "RAM {:.1} GiB > {:.1} GiB available (min of total − running, free − 4 GiB)",
            gib(new_ram),
            gib(ram_room)
        ));
    }
    if over.is_empty() {
        return Ok(());
    }
    let shares: Vec<String> = running
        .iter()
        .map(|i| {
            format!(
                "instance `{}` ({}, {}) holds {:.1} GiB VRAM / {:.1} GiB RAM",
                i.id,
                i.kind,
                i.health,
                gib(i.planned_vram),
                gib(i.planned_ram)
            )
        })
        .collect();
    Err(ErrorEnvelope::new(
        ErrorCode::PlanInfeasible,
        format!(
            "{}; {}; stop one (`golem stop <id>`) or pass --allow-overcommit",
            over.join(", "),
            shares.join("; ")
        ),
    ))
}

/// The instance registry: one writable dir plus every dir clients read.
#[derive(Debug, Clone)]
pub struct Registry {
    dir: PathBuf,
    lock_path: PathBuf,
    also_read: Vec<PathBuf>,
}

fn state_home() -> Result<PathBuf, RegistryError> {
    std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state")))
        .map(|d| d.join("golem"))
        .ok_or(RegistryError::NoDir)
}

impl Registry {
    /// `$XDG_RUNTIME_DIR/golem/instances` (state dir when unset), also
    /// reading the linger fallback `$XDG_STATE_HOME/golem/instances`.
    pub fn open() -> Result<Self, RegistryError> {
        let state = state_home()?;
        let fallback = state.join("instances");
        let dir = std::env::var_os("XDG_RUNTIME_DIR").map_or_else(
            || fallback.clone(),
            |d| PathBuf::from(d).join("golem/instances"),
        );
        Ok(Self::at(dir, state.join("admit.lock")).with_read_dir(fallback))
    }

    /// [`Registry::open`] writing to the linger fallback (state) dir, which
    /// logind does not remove at logout.
    pub fn open_fallback() -> Result<Self, RegistryError> {
        let r = Self::open()?;
        let dir = r.also_read[0].clone();
        Ok(Self::at(dir, r.lock_path).with_read_dir(r.dir))
    }

    /// Explicit dirs, so tests never touch the real XDG dirs.
    pub fn at(dir: PathBuf, lock_path: PathBuf) -> Self {
        Self {
            dir,
            lock_path,
            also_read: Vec::new(),
        }
    }

    /// Also reads (and reaps) entries in `dir`.
    #[must_use]
    pub fn with_read_dir(mut self, dir: PathBuf) -> Self {
        if dir != self.dir && !self.also_read.contains(&dir) {
            self.also_read.push(dir);
        }
        self
    }

    /// The dir new entries are written to.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Every dir clients read, writable dir first.
    pub fn read_dirs(&self) -> impl Iterator<Item = &PathBuf> {
        std::iter::once(&self.dir).chain(&self.also_read)
    }

    fn lock(&self) -> Result<File, RegistryError> {
        if let Some(p) = self.lock_path.parent() {
            create_private_dir(p)?;
        }
        let f = OpenOptions::new()
            .mode(0o600)
            .create(true)
            .truncate(false)
            .write(true)
            .open(&self.lock_path)?;
        sys::flock_exclusive(&f)?;
        Ok(f)
    }

    /// Writes `i` and starts its heartbeat; dropping the guard removes it.
    pub fn register(&self, i: &Instance) -> Result<Guard, RegistryError> {
        create_private_dir(&self.dir)?;
        let mut i = i.clone();
        i.heartbeat_at = now();
        let path = self.dir.join(format!("{}.json", i.id));
        write_entry(&path, &i)?;
        let inst = Arc::new(Mutex::new(i));
        let (stop, rx) = channel::<()>();
        let (p, w) = (path.clone(), inst.clone());
        let beat = std::thread::Builder::new()
            .name("golem-heartbeat".into())
            .spawn(move || {
                while let Err(RecvTimeoutError::Timeout) = rx.recv_timeout(HEARTBEAT) {
                    let mut g = w.lock().unwrap_or_else(|p| p.into_inner());
                    g.heartbeat_at = now();
                    let _ = write_entry(&p, &g);
                }
            })?;
        Ok(Guard {
            path,
            inst,
            stop: Some(stop),
            beat: Some(beat),
        })
    }

    /// Live entries across every read dir; stale ones are removed.
    pub fn list_live(&self) -> Result<Vec<Instance>, RegistryError> {
        let _lock = self.lock()?;
        self.live_locked()
    }

    /// Caller holds the admission lock.
    fn live_locked(&self) -> Result<Vec<Instance>, RegistryError> {
        let mut live = Vec::new();
        for dir in self.read_dirs() {
            let rd = match std::fs::read_dir(dir) {
                Ok(rd) => rd,
                Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e.into()),
            };
            for e in rd {
                let path = e?.path();
                if path.extension().is_none_or(|x| x != "json") {
                    continue;
                }
                // Unparseable entries are left alone: tmp + rename means a
                // reader never sees a torn write, so this is foreign data.
                let Ok(i) = std::fs::read(&path)
                    .map_err(RegistryError::from)
                    .and_then(|b| Ok(serde_json::from_slice::<Instance>(&b)?))
                else {
                    continue;
                };
                if stale(&i) {
                    let _ = std::fs::remove_file(&path);
                    let _ = std::fs::remove_file(dir.join(format!("{}.sock", i.id)));
                } else {
                    live.push(i);
                }
            }
        }
        live.sort_by(|a, b| (a.started_at, &a.id).cmp(&(b.started_at, &b.id)));
        Ok(live)
    }

    /// Live instance by id, else by served model name.
    pub fn find(&self, id_or_name: &str) -> Result<Instance, RegistryError> {
        let mut live = self.list_live()?;
        if let Some(i) = live.iter().position(|i| i.id == id_or_name) {
            return Ok(live.swap_remove(i));
        }
        live.retain(|i| i.model == id_or_name);
        match live.len() {
            0 => Err(RegistryError::InstanceNotFound(id_or_name.to_owned())),
            1 => Ok(live.swap_remove(0)),
            _ => Err(RegistryError::Ambiguous(
                id_or_name.to_owned(),
                live.iter()
                    .map(|i| i.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
            )),
        }
    }

    /// A serving instance (`healthy`/`degraded`) started with `config_hash`.
    pub fn find_compatible(&self, config_hash: &str) -> Result<Option<Instance>, RegistryError> {
        Ok(self.list_live()?.into_iter().find(|i| {
            i.kind == "serve"
                && i.config_hash == config_hash
                && matches!(i.health.as_str(), "healthy" | "degraded")
        }))
    }

    /// Atomic `list_live → admit → register` under the admission lock; the
    /// new entry (normally `loading`) counts against everyone after it.
    pub fn admit_and_register(
        &self,
        i: &Instance,
        mach: &golem_plan::Machine,
        allow_overcommit: bool,
    ) -> Result<Guard, ErrorEnvelope> {
        let _lock = self.lock()?;
        let running = self.live_locked()?;
        exclusive_optimize(&running, &i.kind)?;
        admit(
            &running,
            i.planned_vram,
            i.planned_ram,
            mach,
            allow_overcommit,
        )?;
        Ok(self.register(i)?)
    }
}

/// #81: an `optimize` sweep measures the whole machine, so it starts only
/// with nothing live, and nothing starts while it runs. Never bypassed by
/// `allow_overcommit`.
fn exclusive_optimize(running: &[Instance], kind: &str) -> Result<(), ErrorEnvelope> {
    let refuse = |msg: String| Err(ErrorEnvelope::new(ErrorCode::Busy, msg));
    if let Some(o) = running.iter().find(|i| i.kind == "optimize") {
        return refuse(format!(
            "optimize sweep `{}` is running; wait for it or `golem stop {}`",
            o.id, o.id
        ));
    }
    if kind == "optimize"
        && let Some(i) = running.first()
    {
        return refuse(format!(
            "optimize needs the machine to itself; instance `{}` ({}) is live — `golem stop {}`",
            i.id, i.kind, i.id
        ));
    }
    Ok(())
}

/// Stale per §9.2: owner dead or pid reused. A live owner is never reaped
/// (an old heartbeat or a refusing socket only means it is busy or hung),
/// so its socket is never unlinked under it.
fn stale(i: &Instance) -> bool {
    !owner_alive(i)
}

/// Owner-only (0700) registry dirs: entries and sockets control instances.
fn create_private_dir(p: &Path) -> io::Result<()> {
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(p)
}

fn write_entry(path: &Path, i: &Instance) -> io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    let mut f = OpenOptions::new()
        .mode(0o600)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp)?;
    f.write_all(&serde_json::to_vec(i)?)?;
    std::fs::rename(&tmp, path)
}

/// A registered entry: owns its heartbeat thread and is the only writer of
/// the file. Drop stops the heartbeat and removes `<id>.json`/`<id>.sock`.
#[derive(Debug)]
pub struct Guard {
    path: PathBuf,
    inst: Arc<Mutex<Instance>>,
    stop: Option<Sender<()>>,
    beat: Option<JoinHandle<()>>,
}

impl Guard {
    /// Mutates the entry and rewrites it (serialized with the heartbeat).
    /// A failed write is retried by the next heartbeat.
    pub fn update(&self, f: impl FnOnce(&mut Instance)) {
        let mut g = self.inst.lock().unwrap_or_else(|p| p.into_inner());
        f(&mut g);
        let _ = write_entry(&self.path, &g);
    }

    /// Current entry contents.
    pub fn instance(&self) -> Instance {
        self.inst.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// The control socket path next to the entry.
    pub fn sock_path(&self) -> PathBuf {
        self.path.with_extension("sock")
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(t) = self.beat.take() {
            let _ = t.join();
        }
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_file(self.sock_path());
    }
}
