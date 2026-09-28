//! Placement journal (§11): a fixed-capacity ring buffer of per-step
//! records with a durable text serialization, used for warm start
//! (`seed_from_journal`, M2.5 Task 3).

use crate::{PlacementConfig, Tier};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Ring capacity the journaling loop is expected to use (§11).
pub const JOURNAL_CAP: usize = 4096;

const MAGIC: &str = "golem-journal v1";

/// One journaled step: accesses observed and transitions applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalRecord {
    /// Step the record belongs to.
    pub step: u64,
    /// Flat expert keys accessed this step.
    pub accesses: Vec<usize>,
    /// `(key, tier, present)` transitions applied this step.
    pub transitions: Vec<(usize, Tier, bool)>,
}

/// Config fingerprint written alongside the records; [`read_journal`]
/// surfaces it so the warm-start consumer can refuse a journal recorded
/// under different dimensions or half-life semantics (§11).
#[derive(Debug, Clone, PartialEq)]
pub struct JournalHeader {
    /// Number of expert layers at record time.
    pub layers: u32,
    /// Experts per layer at record time.
    pub experts_per_layer: u32,
    /// Score half-life in steps at record time (§6.2).
    pub half_life_steps: f64,
}

/// Fixed-capacity ring buffer of [`JournalRecord`]s, oldest → newest.
#[derive(Debug)]
pub struct Journal {
    recs: Vec<JournalRecord>,
    /// Index of the oldest record; wraps once the buffer is full.
    start: usize,
    cap: usize,
}

impl Journal {
    /// New empty ring holding at most `capacity` records.
    ///
    /// # Panics
    /// Panics if `capacity == 0` (a typo'd zero capacity must not silently
    /// disable warm start forever).
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "journal capacity must be > 0");
        Self {
            recs: Vec::with_capacity(capacity),
            start: 0,
            cap: capacity,
        }
    }

    /// Append a record, evicting the oldest when at capacity.
    pub fn push(&mut self, rec: JournalRecord) {
        if self.recs.len() < self.cap {
            self.recs.push(rec);
        } else {
            self.recs[self.start] = rec;
            self.start = (self.start + 1) % self.cap;
        }
    }

    /// Number of records held.
    pub fn len(&self) -> usize {
        self.recs.len()
    }

    /// True when no records are held.
    pub fn is_empty(&self) -> bool {
        self.recs.is_empty()
    }

    /// Records oldest → newest.
    pub fn iter(&self) -> impl Iterator<Item = &JournalRecord> {
        let (start, cap) = (self.start, self.cap);
        (0..self.recs.len()).map(move |i| &self.recs[(start + i) % cap])
    }

    /// Step of the newest record, if any.
    pub fn last_step(&self) -> Option<u64> {
        self.iter().last().map(|r| r.step)
    }

    /// Drop all records.
    pub fn clear(&mut self) {
        self.recs.clear();
        self.start = 0;
    }
}

fn invalid_data(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn parse_tier(s: &str) -> io::Result<Tier> {
    match s {
        "disk" => Ok(Tier::Disk),
        "ram" => Ok(Tier::Ram),
        "vram" => Ok(Tier::Vram),
        _ => Err(invalid_data(format!("unknown tier `{s}`"))),
    }
}

fn parse_header(line: &str) -> io::Result<JournalHeader> {
    let bad = || invalid_data(format!("malformed header: {line:?}"));
    let rest = line
        .strip_prefix(MAGIC)
        .filter(|r| r.starts_with(' '))
        .ok_or_else(|| {
            invalid_data(format!(
                "expected header `{MAGIC} <layers> <experts_per_layer> <half_life_steps>`: {line:?}"
            ))
        })?;
    let mut f = rest[1..].split(' ');
    let (Some(l), Some(e), Some(h), None) = (f.next(), f.next(), f.next(), f.next()) else {
        return Err(bad());
    };
    Ok(JournalHeader {
        layers: l.parse().map_err(|_| bad())?,
        experts_per_layer: e.parse().map_err(|_| bad())?,
        half_life_steps: h.parse().map_err(|_| bad())?,
    })
}

fn parse_record_line(line: &str) -> io::Result<JournalRecord> {
    let bad = || invalid_data(format!("malformed record line: {line:?}"));
    let rest = line
        .strip_prefix("S ")
        .ok_or_else(|| invalid_data(format!("record line must start with `S `: {line:?}")))?;
    let mut f = rest.split(' ');
    let mut next = move || f.next().ok_or_else(bad);
    let step: u64 = next()?.parse().map_err(|_| bad())?;
    let n_acc: usize = next()?.parse().map_err(|_| bad())?;
    let mut accesses = Vec::with_capacity(n_acc);
    for _ in 0..n_acc {
        accesses.push(next()?.parse().map_err(|_| bad())?);
    }
    let n_tr: usize = next()?.parse().map_err(|_| bad())?;
    let mut transitions = Vec::with_capacity(n_tr);
    for _ in 0..n_tr {
        let key: usize = next()?.parse().map_err(|_| bad())?;
        let tier = parse_tier(next()?)?;
        let present = match next()? {
            "0" => false,
            "1" => true,
            _ => return Err(bad()),
        };
        transitions.push((key, tier, present));
    }
    if next().is_ok() {
        return Err(invalid_data(format!(
            "extra fields in record line: {line:?}"
        )));
    }
    Ok(JournalRecord {
        step,
        accesses,
        transitions,
    })
}

/// Write the journal durably (§11): tmp file + `sync_all` + rename + parent
/// `sync_all`, so the 24 h soak survives both process death and power loss.
pub fn write_journal(journal: &Journal, cfg: &PlacementConfig, path: &Path) -> io::Result<()> {
    let mut text = format!(
        "{MAGIC} {} {} {}\n",
        cfg.layers, cfg.experts_per_layer, cfg.half_life_steps
    );
    for rec in journal.iter() {
        text.push_str(&format!("S {} {}", rec.step, rec.accesses.len()));
        for k in &rec.accesses {
            text.push(' ');
            text.push_str(&k.to_string());
        }
        text.push(' ');
        text.push_str(&rec.transitions.len().to_string());
        for (k, tier, present) in &rec.transitions {
            text.push_str(&format!(" {k} {tier} {}", u8::from(*present)));
        }
        text.push('\n');
    }
    let tmp = path.with_extension("tmp");
    let mut file = std::fs::File::create(&tmp)?;
    file.write_all(text.as_bytes())?;
    file.sync_all()?;
    std::fs::rename(&tmp, path)?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::File::open(parent)?.sync_all()?;
    }
    Ok(())
}

/// Read a journal written by [`write_journal`]. Strict except at EOF: a
/// malformed non-terminal line or wrong field count is `InvalidData`; a
/// truncated final (unterminated) line is dropped as an incomplete crash tail.
pub fn read_journal(path: &Path) -> io::Result<(Journal, JournalHeader)> {
    let text = std::fs::read_to_string(path)?;
    // Only a final line lacking its '\n' is a truncation; drop it.
    let complete = if text.ends_with('\n') {
        &text[..text.len() - 1]
    } else {
        &text[..text.rfind('\n').unwrap_or(0)]
    };
    let mut lines = complete.split('\n');
    let header = parse_header(lines.next().unwrap_or_default())?;
    let mut recs = Vec::new();
    for line in lines {
        recs.push(parse_record_line(line)?);
    }
    let mut journal = Journal::new(recs.len().max(1));
    for rec in recs {
        journal.push(rec);
    }
    Ok((journal, header))
}

/// Journal directory for `instance` under an explicit data dir; `None` when
/// no data dir is configured.
pub fn journal_dir_from(data_dir: Option<&str>, instance: Option<&str>) -> Option<PathBuf> {
    Some(
        Path::new(data_dir?)
            .join("journals")
            .join(instance.unwrap_or("default")),
    )
}

/// [`journal_dir_from`] over the process environment (`GOLEM_DATA_DIR`,
/// `GOLEM_INSTANCE`). Thin wrapper only — no env mutation anywhere
/// (workspace denies `unsafe`; `set_var` is unsafe in edition 2024).
pub fn journal_dir() -> Option<PathBuf> {
    journal_dir_from(
        std::env::var("GOLEM_DATA_DIR").ok().as_deref(),
        std::env::var("GOLEM_INSTANCE").ok().as_deref(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("golem-journal-test-{}-{name}", std::process::id()))
    }

    fn test_cfg() -> PlacementConfig {
        PlacementConfig {
            layers: 2,
            experts_per_layer: 4,
            half_life_steps: 512.0,
            ..Default::default()
        }
    }

    /// Empty record, accesses-only record, and a mixed record touching all
    /// three tiers with both present values.
    fn three_records() -> Vec<JournalRecord> {
        vec![
            JournalRecord {
                step: 0,
                accesses: vec![],
                transitions: vec![],
            },
            JournalRecord {
                step: 7,
                accesses: vec![3, 9, 1],
                transitions: vec![],
            },
            JournalRecord {
                step: 42,
                accesses: vec![5],
                transitions: vec![
                    (0, Tier::Disk, false),
                    (1, Tier::Ram, true),
                    (7, Tier::Vram, true),
                    (2, Tier::Ram, false),
                    (3, Tier::Vram, false),
                ],
            },
        ]
    }

    #[test]
    fn round_trip_preserves_records_and_full_header() {
        let path = tmp_path("round-trip");
        let mut j = Journal::new(16);
        for rec in three_records() {
            j.push(rec);
        }
        let cfg = test_cfg();
        write_journal(&j, &cfg, &path).unwrap();
        let (back, header) = read_journal(&path).unwrap();
        assert_eq!(
            back.iter().collect::<Vec<_>>(),
            j.iter().collect::<Vec<_>>()
        );
        assert_eq!(header.layers, cfg.layers);
        assert_eq!(header.experts_per_layer, cfg.experts_per_layer);
        assert_eq!(header.half_life_steps, cfg.half_life_steps);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn ring_evicts_oldest_at_4096() {
        let mut j = Journal::new(JOURNAL_CAP);
        for step in 0..(JOURNAL_CAP as u64 + 10) {
            j.push(JournalRecord {
                step,
                accesses: vec![step as usize],
                transitions: vec![],
            });
        }
        assert_eq!(j.len(), JOURNAL_CAP);
        assert!(!j.is_empty());
        assert_eq!(j.iter().next().unwrap().step, 10, "oldest 10 evicted");
        assert_eq!(j.iter().last().unwrap().step, JOURNAL_CAP as u64 + 9);
        assert_eq!(j.last_step(), Some(JOURNAL_CAP as u64 + 9));
    }

    #[test]
    fn ring_works_at_capacity_one() {
        let mut j = Journal::new(1);
        for step in 0..11_u64 {
            j.push(JournalRecord {
                step,
                accesses: vec![],
                transitions: vec![],
            });
        }
        assert_eq!(j.len(), 1);
        assert_eq!(j.iter().next().unwrap().step, 10);
        j.clear();
        assert!(j.is_empty());
        assert_eq!(j.last_step(), None);
    }

    #[test]
    #[should_panic]
    fn zero_capacity_panics() {
        let _ = Journal::new(0);
    }

    #[test]
    fn parses_documented_record_format() {
        let path = tmp_path("parse-literal");
        std::fs::write(&path, "golem-journal v1 2 4 512\nS 7 2 3 9 1 5 vram 1\n").unwrap();
        let (j, header) = read_journal(&path).unwrap();
        assert_eq!((header.layers, header.experts_per_layer), (2, 4));
        assert_eq!(header.half_life_steps, 512.0);
        let recs: Vec<_> = j.iter().collect();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].step, 7);
        assert_eq!(recs[0].accesses, vec![3, 9]);
        assert_eq!(recs[0].transitions, vec![(5, Tier::Vram, true)]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn malformed_mid_file_line_is_invalid_data() {
        let path = tmp_path("malformed");
        std::fs::write(
            &path,
            "golem-journal v1 2 4 512\nS 1 0 0\nnot a record\nS 2 0 0\n",
        )
        .unwrap();
        assert_eq!(
            read_journal(&path).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn truncated_final_line_is_dropped() {
        let path = tmp_path("truncated");
        std::fs::write(
            &path,
            "golem-journal v1 2 4 512\nS 1 1 3 0\nS 2 1 9 0 2 ram 0",
        )
        .unwrap();
        let (j, _) = read_journal(&path).unwrap();
        let recs: Vec<_> = j.iter().collect();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].step, 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn wrong_magic_is_invalid_data() {
        let path = tmp_path("wrong-magic");
        std::fs::write(&path, "other-journal v1 2 4 512\n").unwrap();
        assert_eq!(
            read_journal(&path).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn wrong_version_is_invalid_data() {
        let path = tmp_path("wrong-version");
        std::fs::write(&path, "golem-journal v2 2 4 512\n").unwrap();
        assert_eq!(
            read_journal(&path).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn journal_dir_from_table() {
        let from = |d, i| journal_dir_from(d, i).map(|p| p.to_string_lossy().into_owned());
        assert_eq!(
            from(Some("/d"), None).as_deref(),
            Some("/d/journals/default")
        );
        assert_eq!(
            from(Some("/d"), Some("i")).as_deref(),
            Some("/d/journals/i")
        );
        assert_eq!(from(None, Some("i")), None);
        assert_eq!(from(None, None), None);
    }
}
