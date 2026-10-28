//! `golem import` (§5.5): verified mirror of a local shard dir into every
//! replica, then `manifest.json` into the source dir and every replica so any
//! of them is self-describing. `--sync` runs the same clone from `golem.toml`.

use std::collections::BTreeSet;
use std::error::Error;
use std::path::{Path, PathBuf};

use golem_store::ModelManifest;
use golem_store::import::{ImportOutcome, ImportRequest, import, publish};

/// Arguments of the `import` subcommand (§5.5).
#[derive(Debug, clap::Args)]
pub struct ImportArgs {
    /// Directory holding the model's shard files (source of truth).
    #[arg(value_name = "SRC")]
    pub source: Option<PathBuf>,
    /// Full-mirror target directories, one per disk.
    #[arg(value_name = "REPLICA")]
    pub replicas: Vec<PathBuf>,
    /// Manifest name (default: source dir name, else the existing manifest's).
    #[arg(long)]
    pub name: Option<String>,
    /// Quantization label recorded in the manifest.
    #[arg(long)]
    pub quant: Option<String>,
    /// Revision label; overrides hf-cache detection.
    #[arg(long)]
    pub revision_label: Option<String>,
    /// Take the model and replicas from `golem.toml` instead of the paths:
    /// bare `--sync` syncs the top-level `model`, `--sync <NAME>` syncs
    /// `[models.<NAME>]`. The same clone `run`/`serve` run at startup.
    #[arg(long, value_name = "NAME", num_args = 0..=1)]
    pub sync: Option<Option<String>>,
}

/// Prints the §5.1 same-device warnings on stderr.
fn report(out: &ImportOutcome) {
    for w in &out.warnings {
        eprintln!("warning: {w}");
    }
}

/// `golem import`: one verified mirror, then the manifest everywhere.
///
/// # Errors
/// Argument resolution, the import itself, or an unwritable manifest dir.
pub fn run(args: &ImportArgs) -> Result<(), Box<dyn Error>> {
    match &args.sync {
        Some(name) => {
            if args.source.is_some()
                || !args.replicas.is_empty()
                || args.name.is_some()
                || args.quant.is_some()
                || args.revision_label.is_some()
            {
                return Err(
                    "`--sync` keeps the model's existing manifest metadata; drop the paths, \
                     --name, --quant and --revision-label"
                        .into(),
                );
            }
            let (model, replicas) = sync_target(name.as_deref())?;
            let dir = model
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            match sync_model(&model, &replicas)? {
                None => println!(
                    "up to date: {} + {} replica(s)",
                    dir.display(),
                    replicas.len()
                ),
                Some(out) => {
                    report(&out);
                    println!(
                        "synced {} replica shard(s) into {} replica(s) of {}",
                        out.manifest.shards.len() * replicas.len() - out.reused_shards,
                        replicas.len(),
                        dir.display()
                    );
                }
            }
            Ok(())
        }
        None => import_once(args),
    }
}

/// The `<src> [<replica>...]` form.
fn import_once(args: &ImportArgs) -> Result<(), Box<dyn Error>> {
    let source = args
        .source
        .as_deref()
        .ok_or("golem import: pass the source dir, or --sync <model>")?;
    let out = import(&ImportRequest {
        source: source.to_path_buf(),
        replicas: args.replicas.clone(),
        name: args.name.clone(),
        quant: args.quant.clone(),
        revision_label: args.revision_label.clone(),
    })?;
    report(&out);
    let mut dirs = Vec::with_capacity(1 + args.replicas.len());
    dirs.push(source.to_path_buf());
    dirs.extend(args.replicas.iter().cloned());
    let written = publish(&out.manifest, &dirs)?;
    let bytes: u64 = out.manifest.shards.iter().map(|s| s.bytes).sum();
    println!(
        "imported {}: {} shards, {bytes} B, {} replica shard(s) kept",
        out.manifest.name,
        out.manifest.shards.len(),
        out.reused_shards
    );
    for p in written {
        println!("manifest {}", p.display());
    }
    Ok(())
}

/// `--sync` target: `(model path, replicas)` from `golem.toml`.
fn sync_target(name: Option<&str>) -> Result<(PathBuf, Vec<PathBuf>), Box<dyn Error>> {
    let layers = crate::config::Layers::from_process()?;
    let (model, replicas) = match name {
        None => (
            layers.model(None)?,
            crate::config::replicas_of(&layers.file, "replicas", layers.env)?,
        ),
        Some(name) => {
            let key = format!("models.{name}");
            let t = crate::config::models(&layers.file)?
                .and_then(|m| m.get(name))
                .and_then(toml::Value::as_table)
                .ok_or_else(|| format!("[{key}] is not a table in golem.toml"))?;
            let path = t
                .get("path")
                .and_then(toml::Value::as_str)
                .ok_or_else(|| format!("[{key}] has no `path`"))?;
            (
                crate::config::expand_tilde(PathBuf::from(path), layers.env),
                crate::config::replicas_of(t, &format!("{key}.replicas"), layers.env)?,
            )
        }
    };
    if replicas.is_empty() {
        return Err(format!("{}: no `replicas` in golem.toml", model.display()).into());
    }
    Ok((model, replicas))
}

/// Brings `model`'s replicas up to date through the one verified clone, and
/// publishes the manifest into the source dir and every replica. Returns
/// `None` when everything already matches the manifest: size, mtime and the
/// R-8 windows per shard (`golem_engine::verify::verify_shards`), and the
/// declared replica set equals the configured one.
///
/// # Errors
/// An unwritable replica root, the import, or the publish.
pub fn sync_model(
    model: &Path,
    replicas: &[PathBuf],
) -> Result<Option<ImportOutcome>, Box<dyn Error>> {
    if replicas.is_empty() {
        return Ok(None);
    }
    let source = model.parent().filter(|p| !p.as_os_str().is_empty());
    let source = source.unwrap_or(Path::new("."));
    let source = source
        .canonicalize()
        .unwrap_or_else(|_| source.to_path_buf());
    // Roots are created before canonicalizing so the identity comparison below
    // sees a real path; `import` re-creates them idempotently.
    let mut roots = Vec::with_capacity(replicas.len());
    for r in replicas {
        std::fs::create_dir_all(r)?;
        roots.push(r.canonicalize()?);
    }
    // A junk manifest is replaced by this sync, the same tolerance
    // `infer_defaults` applies to an unusable one.
    let old = existing_manifest(&source.join("manifest.json"));
    if let Some(old) = &old
        && replica_set_matches(&source, &roots, old)
        && std::iter::once(&source)
            .chain(roots.iter())
            .all(|r| golem_engine::verify::verify_shards(r, old, false).is_ok())
    {
        return Ok(None);
    }
    let revision_label = old.as_ref().and_then(|m| {
        (m.revision_source.as_deref() == Some("user"))
            .then(|| m.revision.clone())
            .flatten()
    });
    let out = import(&ImportRequest {
        source: source.clone(),
        replicas: roots.clone(),
        name: old.as_ref().map(|m| m.name.clone()),
        quant: old.as_ref().and_then(|m| m.quant.clone()),
        revision_label,
    })?;
    let mut manifest = out.manifest;
    if let Some(old) = &old {
        // A sync is about bytes, never metadata: keep what the user imported.
        manifest.quant.clone_from(&old.quant);
        manifest.repo.clone_from(&old.repo);
        manifest.revision.clone_from(&old.revision);
        manifest.revision_source.clone_from(&old.revision_source);
    }
    let mut dirs = Vec::with_capacity(1 + roots.len());
    dirs.push(source);
    dirs.extend(roots);
    publish(&manifest, &dirs)?;
    Ok(Some(ImportOutcome {
        manifest,
        warnings: out.warnings,
        reused_shards: out.reused_shards,
    }))
}

/// The manifest at `path` when one parses.
fn existing_manifest(path: &Path) -> Option<ModelManifest> {
    path.is_file()
        .then(|| golem_store::load_manifest(path).ok())
        .flatten()
}

/// The manifest's replica set for every shard is exactly the source plus the
/// configured roots (a removed replica must be dropped from the manifest).
fn replica_set_matches(source: &Path, roots: &[PathBuf], m: &ModelManifest) -> bool {
    let mut have: BTreeSet<PathBuf> = roots.iter().cloned().collect();
    have.insert(source.to_path_buf());
    m.shards
        .iter()
        .all(|s| s.replicas.len() == have.len() && s.replicas.iter().all(|r| have.contains(r)))
}

/// Refuses a manifest whose replica shards are gone (§5.1): never silently
/// fall back to one disk. Callers load tolerantly, so a junk manifest keeps
/// the existing single-file fallback.
///
/// # Errors
/// The missing replica path and the shard it serves.
pub fn require_replicas(m: &ModelManifest) -> Result<(), Box<dyn Error>> {
    for s in &m.shards {
        for root in &s.replicas {
            let p = root.join(&s.file);
            if !p.is_file() {
                return Err(format!(
                    "manifest lists replica {} for shard {} but it is missing; \
                     re-import or run `golem import --sync` to restore it",
                    p.display(),
                    s.file
                )
                .into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{Args as _, FromArgMatches as _};

    fn parse(argv: &[&str]) -> ImportArgs {
        let matches = ImportArgs::augment_args(clap::Command::new("import"))
            .try_get_matches_from(argv)
            .expect("test argv must parse");
        ImportArgs::from_arg_matches(&matches).expect("matches fit ImportArgs")
    }

    #[test]
    fn sync_flag_takes_an_optional_model_name() {
        assert_eq!(parse(&["import", "src"]).sync, None);
        assert_eq!(parse(&["import", "--sync"]).sync, Some(None));
        assert_eq!(
            parse(&["import", "--sync", "glm"]).sync,
            Some(Some("glm".into()))
        );
        assert!(parse(&["import", "src", "r1"]).replicas.len() == 1);
    }

    #[test]
    fn replica_set_must_match_exactly() {
        let mut m = ModelManifest {
            name: "t".into(),
            ..Default::default()
        };
        m.shards.push(golem_store::ShardEntry {
            file: "a.gguf".into(),
            replicas: vec![PathBuf::from("/s"), PathBuf::from("/r1")],
            ..Default::default()
        });
        let source = Path::new("/s");
        assert!(replica_set_matches(source, &[PathBuf::from("/r1")], &m));
        assert!(!replica_set_matches(source, &[], &m), "missing replica");
        assert!(
            !replica_set_matches(source, &[PathBuf::from("/r1"), PathBuf::from("/r2")], &m),
            "new replica must trigger a sync"
        );
    }
    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("golem-import-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn fixture_copy(dst: &Path) {
        for s in xtask::shape_moe::cached_fixture() {
            std::fs::copy(&s, dst.join(s.file_name().unwrap())).unwrap();
        }
    }

    fn shard_names() -> Vec<String> {
        xtask::shape_moe::cached_fixture()
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn sync_model_clones_then_reports_up_to_date_and_tracks_the_set() {
        let d = scratch("sync");
        let src = d.join("src");
        let (r1, r2) = (d.join("r1"), d.join("r2"));
        for p in [&src, &r1, &r2] {
            std::fs::create_dir_all(p).unwrap();
        }
        fixture_copy(&src);
        let model = src.join(&shard_names()[0]);
        let roots = vec![r1.clone(), r2.clone()];

        let out = sync_model(&model, &roots)
            .unwrap()
            .expect("first sync clones");
        assert_eq!(out.manifest.name, "src");
        assert!(
            src.join("manifest.json").is_file(),
            "source is self-describing"
        );
        assert!(r1.join("manifest.json").is_file());
        assert!(r2.join("manifest.json").is_file());
        for s in shard_names() {
            assert_eq!(
                std::fs::read(src.join(&s)).unwrap(),
                std::fs::read(r1.join(&s)).unwrap()
            );
        }
        assert!(sync_model(&model, &roots).unwrap().is_none(), "up to date");

        // A removed replica drops out of the manifest.
        let out = sync_model(&model, std::slice::from_ref(&r1))
            .unwrap()
            .expect("set changed");
        assert_eq!(out.manifest.shards[0].replicas.len(), 2, "source + r1");

        // A deleted shard is cloned back.
        std::fs::remove_file(r1.join(&shard_names()[1])).unwrap();
        assert!(
            sync_model(&model, std::slice::from_ref(&r1))
                .unwrap()
                .is_some()
        );
        assert_eq!(
            std::fs::read(r1.join(&shard_names()[1])).unwrap(),
            std::fs::read(src.join(&shard_names()[1])).unwrap()
        );

        // A touched source invalidates the manifest even when replicas are
        // intact: sync re-reads the source and republishes.
        let mtime = std::fs::metadata(&model).unwrap().modified().unwrap();
        std::fs::File::options()
            .write(true)
            .open(&model)
            .unwrap()
            .set_modified(mtime + std::time::Duration::from_secs(2))
            .unwrap();
        assert!(
            sync_model(&model, std::slice::from_ref(&r1))
                .unwrap()
                .is_some(),
            "a changed source re-syncs"
        );
        assert_eq!(
            std::fs::metadata(r1.join(&shard_names()[0]))
                .unwrap()
                .modified()
                .unwrap(),
            std::fs::metadata(&model).unwrap().modified().unwrap(),
            "replica mtime follows the new source stamp"
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn require_replicas_names_the_missing_path() {
        let d = scratch("require");
        let src = d.join("src");
        let r = d.join("r1");
        std::fs::create_dir_all(&src).unwrap();
        fixture_copy(&src);
        let model = src.join(&shard_names()[0]);
        sync_model(&model, std::slice::from_ref(&r))
            .unwrap()
            .expect("clones");
        let m = golem_store::load_manifest(&src.join("manifest.json")).unwrap();
        require_replicas(&m).expect("all replicas present");
        let gone = r.join(&shard_names()[1]);
        std::fs::remove_file(&gone).unwrap();
        let e = require_replicas(&m).unwrap_err().to_string();
        assert!(
            e.contains(gone.to_str().unwrap()) && e.contains("--sync"),
            "{e}"
        );
        std::fs::remove_dir_all(&d).unwrap();
    }
}
