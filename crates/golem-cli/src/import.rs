//! `golem import` (§5.5): scan a local shard dir, hash + CRC + mirror it,
//! and write the manifest under `$GOLEM_DATA_DIR/models/<name>/manifest.json`.

use std::error::Error;
use std::path::PathBuf;

use golem_engine::atomic::write_atomic;
use golem_store::import::{ImportRequest, import, resolve_name};

/// `golem import` arguments (§5.5).
#[derive(Debug, clap::Args)]
pub struct ImportArgs {
    /// Directory holding the model's shard files (source of truth).
    pub source: PathBuf,
    /// Full-mirror target directories, one per disk.
    pub replicas: Vec<PathBuf>,
    /// Manifest name (default: source dir name).
    #[arg(long)]
    pub name: Option<String>,
    /// Quantization label recorded in the manifest.
    #[arg(long)]
    pub quant: Option<String>,
    /// Revision label; overrides hf-cache detection.
    #[arg(long)]
    pub revision_label: Option<String>,
    /// Overwrite an existing manifest of the same name.
    #[arg(long)]
    pub force: bool,
}

/// Runs one import to completion: fail fast on an existing manifest, hash +
/// mirror, print same-disk warnings, then publish the manifest atomically.
/// The overwrite guard is re-checked after the (possibly long) hash, but
/// concurrent imports may still race on the final manifest write — last
/// writer wins, and both outputs are valid manifests.
///
/// # Errors
/// Any failure from argument resolution, the import itself, manifest
/// serialization, or the atomic write.
pub fn run(args: &ImportArgs) -> Result<(), Box<dyn Error>> {
    let data = golem_store::data_dir().ok_or("GOLEM_DATA_DIR is not set")?;
    let name = resolve_name(&args.source, args.name.as_deref())?;
    let dest = data.join("models").join(&name).join("manifest.json");
    let exists = dest.exists();
    if exists && !args.force {
        return Err(format!("{dest:?} already exists; pass --force to overwrite").into());
    }
    let out = import(&ImportRequest {
        source: args.source.clone(),
        replicas: args.replicas.clone(),
        name: args.name.clone(),
        quant: args.quant.clone(),
        revision_label: args.revision_label.clone(),
    })?;
    for w in &out.warnings {
        eprintln!("warning: {w}");
    }
    if dest.exists() && !args.force {
        return Err(
            format!("{dest:?} appeared during the import; pass --force to overwrite").into(),
        );
    }
    std::fs::create_dir_all(dest.parent().unwrap_or(&data))?;
    write_atomic(&dest, &serde_json::to_string_pretty(&out.manifest)?)?;
    let shards = out.manifest.shards.len();
    let bytes: u64 = out.manifest.shards.iter().map(|s| s.bytes).sum();
    println!(
        "imported {name}: {shards} shards, {bytes} B -> {}",
        dest.display()
    );
    Ok(())
}
