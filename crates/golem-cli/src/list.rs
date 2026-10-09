//! `golem list` (§9.2): live instances, or `--models` stored manifests.

use std::error::Error;

use golem_registry::{ControlRequest, Registry};

/// Arguments of the `list` subcommand.
#[derive(Debug, clap::Args)]
pub struct ListArgs {
    /// List the model manifests under `$GOLEM_DATA_DIR` instead.
    #[arg(long)]
    pub models: bool,
}

/// Prints the table.
pub fn list(args: &ListArgs) -> Result<(), Box<dyn Error>> {
    if args.models {
        return models();
    }
    let live = Registry::open()?.list_live()?;
    println!(
        "{:<10} {:<24} {:<8} {:<9} {:>6}  FLAGS",
        "ID", "MODEL", "KIND", "HEALTH", "TIER%"
    );
    for i in &live {
        // Only `serve` has a control socket; the entry's health is current
        // for the others.
        let status = (i.kind == "serve")
            .then(|| golem_registry::send(&i.id, &ControlRequest::Status).ok())
            .flatten();
        let body = status.as_ref().map(|s| &s.body);
        let health = body
            .and_then(|b| b.get("health")?.as_str())
            .unwrap_or(&i.health);
        let fill = body
            .and_then(|b| b.get("device_cache_skip_ratio")?.as_f64())
            .map_or("-".to_owned(), |r| format!("{:.0}", r * 100.0));
        let flags = if health == "degraded" {
            "degraded"
        } else {
            "-"
        };
        println!(
            "{:<10} {:<24} {:<8} {:<9} {:>6}  {flags}",
            i.id, i.model, i.kind, health, fill
        );
    }
    for pid in unregistered(&live) {
        println!(
            "{:<10} {:<24} {:<8} {:<9} {:>6}  unregistered",
            pid, "?", "?", "?", "-"
        );
    }
    Ok(())
}

/// `golem` processes (other than this one) with no registry entry.
fn unregistered(live: &[golem_registry::Instance]) -> Vec<u32> {
    let me = std::process::id();
    let Ok(rd) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    rd.filter_map(|e| e.ok()?.file_name().to_str()?.parse::<u32>().ok())
        .filter(|&pid| pid != me && !live.iter().any(|i| i.pid == pid))
        .filter(|pid| {
            std::fs::read_to_string(format!("/proc/{pid}/comm")).is_ok_and(|c| c.trim() == "golem")
        })
        .collect()
}

/// Manifests at `$GOLEM_DATA_DIR/**/manifest.json`: name, quant, size,
/// replica count.
fn models() -> Result<(), Box<dyn Error>> {
    let root = golem_store::data_dir().ok_or("GOLEM_DATA_DIR is not set")?;
    println!(
        "{:<32} {:<12} {:>10}  REPLICAS",
        "NAME", "QUANT", "SIZE GiB"
    );
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "manifest.json") {
                let m = golem_store::load_manifest(&p)?;
                let size: u64 = m.shards.iter().map(|s| s.bytes).sum();
                let replicas = m.shards.iter().map(|s| s.replicas.len()).min().unwrap_or(0);
                println!(
                    "{:<32} {:<12} {:>10.2}  {replicas}",
                    m.name,
                    m.quant.as_deref().unwrap_or("-"),
                    size as f64 / (1u64 << 30) as f64
                );
            }
        }
    }
    Ok(())
}
