//! §9.4 config precedence: flag > env (`GOLEM_<FLAG>`) > file
//! (`golem.toml` in [`config_dir`]) > profile > built-in default. A present
//! but unparsable layer is an error naming source, key and value — never a
//! fall-through to a lower layer.

use std::str::FromStr;

/// Why a config value could not be resolved.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// A layer holds a value that does not parse as the flag's type.
    #[error("{layer} {key} = {value:?}: {msg}")]
    Invalid {
        /// `env`, `file` or `profile`.
        layer: &'static str,
        /// Env var or file key.
        key: String,
        /// The offending raw value.
        value: String,
        /// Parser message.
        msg: String,
    },
    /// The config file exists but cannot be read or parsed.
    #[error("config file {path}: {msg}")]
    File {
        /// File path.
        path: std::path::PathBuf,
        /// What failed.
        msg: String,
    },
}

/// No layer supplies the model path.
#[derive(Debug, thiserror::Error)]
#[error(
    "no model: pass it as an argument, set GOLEM_MODEL, or set `model` in {}",
    .0.as_deref().map_or("golem.toml".into(), |p| p.display().to_string())
)]
pub struct NoModel(pub Option<std::path::PathBuf>);

/// `$XDG_CONFIG_HOME/golem`, else `$HOME/.config/golem`.
pub fn config_home(env: &dyn Fn(&str) -> Option<String>) -> Option<std::path::PathBuf> {
    env("XDG_CONFIG_HOME")
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| env("HOME").map(|h| std::path::Path::new(&h).join(".config")))
        .map(|d| d.join("golem"))
}

/// Directory holding `golem.toml`: `$GOLEM_CONFIG_DIR`, else [`config_home`].
pub fn config_dir(env: &dyn Fn(&str) -> Option<String>) -> Option<std::path::PathBuf> {
    env("GOLEM_CONFIG_DIR")
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| config_home(env))
}

/// `~` or `~/…` against `$HOME`; anything else unchanged.
pub(crate) fn expand_tilde(
    p: std::path::PathBuf,
    env: &dyn Fn(&str) -> Option<String>,
) -> std::path::PathBuf {
    let Some(home) = env("HOME") else {
        return p;
    };
    match p.strip_prefix("~") {
        Ok(rest) => std::path::Path::new(&home).join(rest),
        Err(_) => p,
    }
}

/// Raw string of a TOML value as the flag would have spelled it.
fn toml_raw(v: &toml::Value) -> String {
    match v {
        toml::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// The `[models.*]` table of a file layer, the one reader of that key;
/// `None` when absent.
///
/// # Errors
/// [`ConfigError::Invalid`] when `models` is present but not a table.
pub fn models(file: &toml::Table) -> Result<Option<&toml::Table>, ConfigError> {
    let Some(v) = file.get("models") else {
        return Ok(None);
    };
    v.as_table().map(Some).ok_or_else(|| ConfigError::Invalid {
        layer: "file",
        key: "models".into(),
        value: v.to_string(),
        msg: "must be a table of tables".into(),
    })
}

/// `replicas` of a model-layout table (the file layer or one `[models.*]`
/// entry): an array of paths, `~` expanded. A layout key, not an engine flag,
/// so it has no env/flag layer (§9.4 leaves it file-only).
///
/// # Errors
/// [`ConfigError::Invalid`] naming `key` when it is not an array of strings.
pub fn replicas_of(
    table: &toml::Table,
    key: &str,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<Vec<std::path::PathBuf>, ConfigError> {
    let Some(v) = table.get("replicas") else {
        return Ok(Vec::new());
    };
    let invalid = |value: String, msg: &str| ConfigError::Invalid {
        layer: "file",
        key: key.to_owned(),
        value,
        msg: msg.into(),
    };
    let arr = v
        .as_array()
        .ok_or_else(|| invalid(v.to_string(), "must be an array of paths"))?;
    arr.iter()
        .map(|e| match e.as_str() {
            Some(s) => Ok(expand_tilde(std::path::PathBuf::from(s), env)),
            None => Err(invalid(e.to_string(), "must be a string path")),
        })
        .collect()
}

fn parse<T: FromStr>(layer: &'static str, key: &str, raw: String) -> Result<T, ConfigError>
where
    T::Err: std::fmt::Display,
{
    raw.parse().map_err(|e: T::Err| ConfigError::Invalid {
        layer,
        key: key.to_owned(),
        value: raw,
        msg: e.to_string(),
    })
}

/// [`resolve`] for flags without a built-in default: `None` when no layer
/// sets the key.
pub fn resolve_opt<T: FromStr>(
    flag: Option<T>,
    env_key: &str,
    file: &toml::Table,
    file_key: &str,
    profile: &toml::Table,
    profile_key: &str,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<Option<T>, ConfigError>
where
    T::Err: std::fmt::Display,
{
    if flag.is_some() {
        return Ok(flag);
    }
    if let Some(v) = env(env_key) {
        return parse("env", env_key, v).map(Some);
    }
    if let Some(v) = file.get(file_key) {
        return parse("file", file_key, toml_raw(v)).map(Some);
    }
    if !profile_key.is_empty()
        && let Some(v) = profile.get(profile_key)
    {
        return parse("profile", profile_key, toml_raw(v)).map(Some);
    }
    Ok(None)
}

/// One flag through every layer, highest first.
#[allow(clippy::too_many_arguments)] // the §9.4 layer list, fixed by design
pub fn resolve<T: FromStr>(
    flag: Option<T>,
    env_key: &str,
    file: &toml::Table,
    file_key: &str,
    profile: &toml::Table,
    profile_key: &str,
    default: T,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<T, ConfigError>
where
    T::Err: std::fmt::Display,
{
    Ok(resolve_opt(flag, env_key, file, file_key, profile, profile_key, env)?.unwrap_or(default))
}

/// The profile layer for `knobs` (`None`: no profile, or
/// `--no-optimize-profile`). Zero/false knobs mean "engine default" and are
/// left out so lower layers decide.
pub fn profile_table(knobs: Option<&crate::profile::Knobs>) -> toml::Table {
    let mut t = toml::Table::new();
    let Some(k) = knobs else {
        return t;
    };
    let mut put = |key: &str, on: bool, v: toml::Value| {
        if on {
            t.insert(key.into(), v);
        }
    };
    put("threads", k.threads > 0, i64::from(k.threads).into());
    put("numa_interleave", k.numa_interleave, true.into());
    put("o_direct", k.o_direct, true.into());
    put("io_threads", k.io_threads > 0, (k.io_threads as i64).into());
    put("io_depth", k.io_depth > 0, i64::from(k.io_depth).into());
    put("async_h2d", k.async_h2d, true.into());
    put(
        "vram_experts_frac",
        k.vram_experts_frac > 0.0,
        f64::from(k.vram_experts_frac).into(),
    );
    put("cpu_expert_math", k.cpu_expert_math, true.into());
    put(
        "cpu_expert_frac",
        k.cpu_expert_frac > 0.0,
        f64::from(k.cpu_expert_frac).into(),
    );
    t
}

/// The `--cache-type-k`/`--cache-type-v` flags, as the layers resolve them.
pub type CacheTypeFlags = (
    Option<golem_plan::KvCacheType>,
    Option<golem_plan::KvCacheType>,
);

/// The non-flag layers for one invocation.
pub struct Layers<'a> {
    /// `golem.toml` contents (empty when absent).
    pub file: toml::Table,
    /// `golem optimize` profile knobs ([`profile_table`]).
    pub profile: toml::Table,
    /// Where `golem.toml` was looked for.
    pub path: Option<std::path::PathBuf>,
    /// Env lookup (`std::env::var` in production).
    pub env: &'a dyn Fn(&str) -> Option<String>,
}

impl Layers<'_> {
    /// Process env + `golem.toml` in [`config_dir`].
    pub fn from_process() -> Result<Layers<'static>, ConfigError> {
        Layers::from_env(&|k| std::env::var(k).ok())
    }

    /// `env` + `golem.toml` in [`config_dir`] (no file layer when absent).
    pub fn from_env(env: &dyn Fn(&str) -> Option<String>) -> Result<Layers<'_>, ConfigError> {
        let path = config_dir(env).map(|d| d.join("golem.toml"));
        let file = match &path {
            Some(path) => match std::fs::read_to_string(path) {
                Ok(s) => {
                    let t = toml::from_str(&s).map_err(|e| ConfigError::File {
                        path: path.clone(),
                        msg: e.to_string(),
                    })?;
                    tracing::info!("golem: config file {} loaded", path.display());
                    t
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    tracing::info!("golem: no config file at {}", path.display());
                    toml::Table::new()
                }
                Err(e) => {
                    return Err(ConfigError::File {
                        path: path.clone(),
                        msg: e.to_string(),
                    });
                }
            },
            None => {
                tracing::info!(
                    "golem: no config file: GOLEM_CONFIG_DIR, XDG_CONFIG_HOME and HOME unset"
                );
                toml::Table::new()
            }
        };
        Ok(Layers {
            file,
            profile: toml::Table::new(),
            path,
            env,
        })
    }

    /// The model path: `flag` > `GOLEM_MODEL` > file `model`, `~` expanded.
    pub fn model(
        &self,
        flag: Option<std::path::PathBuf>,
    ) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
        match self.opt(flag, "model")? {
            Some(m) => Ok(expand_tilde(m, self.env)),
            None => Err(NoModel(self.path.clone()).into()),
        }
    }

    /// This layer's `replicas` (§5.1/§5.5): the file layer's array, `~`
    /// expanded. Empty when unset.
    ///
    /// # Errors
    /// [`ConfigError::Invalid`] when the key is not an array of strings.
    pub fn replicas(&self) -> Result<Vec<std::path::PathBuf>, ConfigError> {
        replicas_of(&self.file, "replicas", self.env)
    }

    /// Adds the saved `golem optimize` profile for `model` as the profile
    /// layer, unless `disabled` (`--no-optimize-profile`), or the profile does
    /// not apply to this run: measured under other KV cache types
    /// (`cache_types` flags) or with more free VRAM/RAM than is free now. Both
    /// skips are logged with their reason.
    pub fn with_profile(
        mut self,
        model: &std::path::Path,
        disabled: bool,
        cache_types: CacheTypeFlags,
    ) -> Result<Self, ConfigError> {
        if disabled {
            return Ok(self);
        }
        let (k, v) = self.cache_types(cache_types)?;
        let types = golem_plan::kv_cache_types(k, v);
        let Some(p) = crate::profile::fingerprint(model)
            .ok()
            .and_then(|fp| crate::profile::load_profile(&fp, &types))
        else {
            return Ok(self);
        };
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let vram = crate::profile::sysfs_vram().map(|(f, _)| f);
        match crate::profile::applicable(&p, &types, vram, sys.available_memory()) {
            Ok(()) => {
                tracing::info!(
                    "golem: optimize profile {} applied (KV cache {types})",
                    &p.fingerprint[..12]
                );
                self.profile = profile_table(p.knobs.as_ref());
            }
            Err(why) => tracing::info!(
                "golem: optimize profile {} ignored: {why}",
                &p.fingerprint[..12]
            ),
        }
        Ok(self)
    }

    /// No env, file or profile: flags and defaults only.
    #[cfg(test)]
    pub fn none() -> Layers<'static> {
        Layers {
            file: toml::Table::new(),
            profile: toml::Table::new(),
            path: None,
            env: &|_| None,
        }
    }

    /// `flag` (spelled `snake_case`) through env `GOLEM_<FLAG>`, file and
    /// profile key `flag`, then `default`.
    pub fn get<T: FromStr>(&self, flag: Option<T>, key: &str, default: T) -> Result<T, ConfigError>
    where
        T::Err: std::fmt::Display,
    {
        let env_key = format!("GOLEM_{}", key.to_uppercase());
        resolve(
            flag,
            &env_key,
            &self.file,
            key,
            &self.profile,
            key,
            default,
            self.env,
        )
    }

    /// [`Self::get`] without a default.
    pub fn opt<T: FromStr>(&self, flag: Option<T>, key: &str) -> Result<Option<T>, ConfigError>
    where
        T::Err: std::fmt::Display,
    {
        let env_key = format!("GOLEM_{}", key.to_uppercase());
        resolve_opt(
            flag,
            &env_key,
            &self.file,
            key,
            &self.profile,
            key,
            self.env,
        )
    }

    /// The run's KV cache element types (flag > env > file > engine default).
    /// No profile layer holds them, so this resolves the same before and after
    /// [`Self::with_profile`].
    pub fn cache_types(
        &self,
        flag: CacheTypeFlags,
    ) -> Result<(golem_plan::KvCacheType, golem_plan::KvCacheType), ConfigError> {
        let d = golem_engine::config::EngineConfig::default();
        Ok((
            self.get(flag.0, "cache_type_k", d.cache_type_k)?,
            self.get(flag.1, "cache_type_v", d.cache_type_v)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_beats_env_beats_file_beats_default() {
        let file: toml::Table = toml::from_str("ctx = 300").unwrap();
        let set = |k: &str| (k == "GOLEM_CTX").then(|| "200".to_owned());
        let unset = |_: &str| None;
        assert_eq!(
            resolve(
                Some(100u32),
                "GOLEM_CTX",
                &file,
                "ctx",
                &toml::Table::new(),
                "",
                400,
                &set
            )
            .unwrap(),
            100
        );
        assert_eq!(
            resolve(
                None,
                "GOLEM_CTX",
                &file,
                "ctx",
                &toml::Table::new(),
                "",
                400,
                &set
            )
            .unwrap(),
            200
        );
        assert_eq!(
            resolve(
                None,
                "GOLEM_CTX",
                &file,
                "ctx",
                &toml::Table::new(),
                "",
                400,
                &unset
            )
            .unwrap(),
            300
        );
        assert_eq!(
            resolve(
                None,
                "GOLEM_CTX",
                &toml::Table::new(),
                "ctx",
                &toml::Table::new(),
                "",
                400,
                &unset
            )
            .unwrap(),
            400
        );
    }

    #[test]
    fn unparsable_layer_is_an_error_not_a_default() {
        let bad_env = |k: &str| (k == "GOLEM_CTX").then(|| "2O0".to_owned());
        assert!(
            resolve::<u32>(
                None,
                "GOLEM_CTX",
                &toml::Table::new(),
                "ctx",
                &toml::Table::new(),
                "",
                400,
                &bad_env
            )
            .is_err()
        );
        let bad_file: toml::Table = toml::from_str(r#"ctx = "lots""#).unwrap();
        assert!(
            resolve::<u32>(
                None,
                "GOLEM_CTX",
                &bad_file,
                "ctx",
                &toml::Table::new(),
                "",
                400,
                &|_| None
            )
            .is_err()
        );
    }

    #[test]
    fn timing_csv_follows_precedence() {
        let file: toml::Table = toml::from_str(r#"timing_csv = "/f.csv""#).unwrap();
        let env = |k: &str| (k == "GOLEM_TIMING_CSV").then(|| "/e.csv".to_owned());
        let none: Option<std::path::PathBuf> = None;
        assert_eq!(
            resolve_opt(
                Some(std::path::PathBuf::from("/x.csv")),
                "GOLEM_TIMING_CSV",
                &file,
                "timing_csv",
                &toml::Table::new(),
                "",
                &env
            )
            .unwrap(),
            Some("/x.csv".into())
        );
        assert_eq!(
            resolve_opt(
                none.clone(),
                "GOLEM_TIMING_CSV",
                &file,
                "timing_csv",
                &toml::Table::new(),
                "",
                &env
            )
            .unwrap(),
            Some("/e.csv".into())
        );
        assert_eq!(
            resolve_opt(
                none,
                "GOLEM_TIMING_CSV",
                &file,
                "timing_csv",
                &toml::Table::new(),
                "",
                &|_| None
            )
            .unwrap(),
            Some("/f.csv".into())
        );
    }

    #[test]
    fn error_names_source_key_and_value() {
        let env = |k: &str| (k == "GOLEM_CTX").then(|| "2O0".to_owned());
        let e = resolve::<u32>(
            None,
            "GOLEM_CTX",
            &toml::Table::new(),
            "ctx",
            &toml::Table::new(),
            "",
            1,
            &env,
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("env") && e.contains("GOLEM_CTX") && e.contains("2O0"),
            "{e}"
        );
    }

    #[test]
    fn layers_map_flag_names_to_env_and_file_keys() {
        let l = Layers {
            file: toml::from_str("prefetch = \"read\"\nflash_attn = true").unwrap(),
            profile: toml::Table::new(),
            path: None,
            env: &|k| (k == "GOLEM_N_PREDICT").then(|| "7".to_owned()),
        };
        assert_eq!(l.get(None, "n_predict", 64u32).unwrap(), 7);
        assert_eq!(
            l.get(None, "prefetch", golem_engine::config::PrefetchMode::Off)
                .unwrap(),
            golem_engine::config::PrefetchMode::Read
        );
        assert!(l.get(None, "flash_attn", false).unwrap());
    }

    #[test]
    fn config_dir_prefers_golem_then_xdg_then_home() {
        let env = |vars: &'static [(&'static str, &'static str)]| {
            move |k: &str| {
                vars.iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| (*v).to_owned())
            }
        };
        let all = env(&[
            ("GOLEM_CONFIG_DIR", "/g"),
            ("XDG_CONFIG_HOME", "/x"),
            ("HOME", "/h"),
        ]);
        let xdg = env(&[("XDG_CONFIG_HOME", "/x"), ("HOME", "/h")]);
        let home = env(&[("HOME", "/h")]);
        assert_eq!(config_dir(&all), Some("/g".into()));
        assert_eq!(config_dir(&xdg), Some("/x/golem".into()));
        assert_eq!(config_dir(&home), Some("/h/.config/golem".into()));
        assert_eq!(config_dir(&|_| None), None);
        assert_eq!(config_home(&all), Some("/x/golem".into()));
    }

    #[test]
    fn model_flag_beats_env_beats_file_and_expands_tilde() {
        let layers = |env: &'static dyn Fn(&str) -> Option<String>| Layers {
            file: toml::from_str(r#"model = "~/m/file.gguf""#).unwrap(),
            profile: toml::Table::new(),
            path: None,
            env,
        };
        let home: &'static dyn Fn(&str) -> Option<String> =
            &|k| (k == "HOME").then(|| "/h".to_owned());
        let with_env: &'static dyn Fn(&str) -> Option<String> = &|k| match k {
            "HOME" => Some("/h".to_owned()),
            "GOLEM_MODEL" => Some("/e.gguf".to_owned()),
            _ => None,
        };
        assert_eq!(
            layers(with_env).model(Some("/f.gguf".into())).unwrap(),
            std::path::PathBuf::from("/f.gguf")
        );
        assert_eq!(
            layers(with_env).model(None).unwrap(),
            std::path::PathBuf::from("/e.gguf")
        );
        assert_eq!(
            layers(home).model(None).unwrap(),
            std::path::PathBuf::from("/h/m/file.gguf")
        );
    }

    #[test]
    fn missing_model_names_all_three_ways() {
        let l = Layers {
            path: Some("/c/golem.toml".into()),
            ..Layers::none()
        };
        let e = l.model(None).unwrap_err();
        assert!(e.is::<NoModel>());
        let e = e.to_string();
        assert!(
            e.contains("argument") && e.contains("GOLEM_MODEL") && e.contains("/c/golem.toml"),
            "{e}"
        );
    }

    #[test]
    fn from_env_reads_default_dir_file_and_errors_on_bad_toml() {
        let tmp = std::env::temp_dir().join(format!("golem-cfg-{}-from-env", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let dir = tmp.join(".config/golem");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("golem.toml"), "model = \"/m.gguf\"\nctx = 512").unwrap();
        let home = tmp.to_str().unwrap().to_owned();
        let env = move |k: &str| (k == "HOME").then(|| home.clone());
        let l = Layers::from_env(&env).unwrap();
        assert_eq!(l.model(None).unwrap(), std::path::PathBuf::from("/m.gguf"));
        assert_eq!(l.get(None, "ctx", 1u32).unwrap(), 512);
        let other = tmp.join("other");
        std::fs::create_dir_all(&other).unwrap();
        let other_s = other.to_str().unwrap().to_owned();
        let golem_dir = move |k: &str| (k == "GOLEM_CONFIG_DIR").then(|| other_s.clone());
        assert!(Layers::from_env(&golem_dir).unwrap().file.is_empty());
        std::fs::write(dir.join("golem.toml"), "ctx = = 1").unwrap();
        let e = Layers::from_env(&env).err().unwrap().to_string();
        assert!(e.contains("golem.toml"), "{e}");
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn cache_types_resolve_through_layers() {
        use golem_plan::KvCacheType::{F16, Q4_0, Q8_0};
        let l = Layers {
            file: toml::from_str("cache_type_k = \"q8_0\"\ncache_type_v = \"q8\"").unwrap(),
            profile: toml::Table::new(),
            path: None,
            env: &|k| (k == "GOLEM_CACHE_TYPE_K").then(|| "q4_0".to_owned()),
        };
        assert_eq!(l.get(None, "cache_type_k", F16).unwrap(), Q4_0);
        assert_eq!(l.get(Some(Q8_0), "cache_type_k", F16).unwrap(), Q8_0);
        let e = l.get(None, "cache_type_v", F16).unwrap_err().to_string();
        assert!(e.contains("q8_0") && e.contains("\"q8\""), "{e}");
    }

    /// #161: one resolver, so the types a profile is compared against are the
    /// types the engine loads (the file layer, the env var and the flag), and
    /// the default is the engine's own.
    #[test]
    fn cache_types_resolver_matches_engine_defaults() {
        use golem_plan::KvCacheType::{F16, Q8_0};
        let d = golem_engine::config::EngineConfig::default();
        let none = Layers::none();
        assert_eq!(
            none.cache_types((None, None)).unwrap(),
            (d.cache_type_k, d.cache_type_v)
        );
        let l = Layers {
            file: toml::from_str("cache_type_k = \"q8_0\"").unwrap(),
            profile: toml::Table::new(),
            path: None,
            env: &|k| (k == "GOLEM_CACHE_TYPE_V").then(|| "q8_0".to_owned()),
        };
        assert_eq!(l.cache_types((None, None)).unwrap(), (Q8_0, Q8_0));
        assert_eq!(l.cache_types((Some(F16), Some(F16))).unwrap(), (F16, F16));
    }
    #[test]
    fn replicas_reads_file_array_and_expands_tilde() {
        let env = |k: &str| (k == "HOME").then(|| "/h".to_owned());
        let file: toml::Table = toml::from_str(r#"replicas = ["~/r1", "/r2"]"#).unwrap();
        assert_eq!(
            replicas_of(&file, "replicas", &env).unwrap(),
            vec![
                std::path::PathBuf::from("/h/r1"),
                std::path::PathBuf::from("/r2")
            ]
        );
        let l = Layers {
            file,
            profile: toml::Table::new(),
            path: None,
            env: &|k| (k == "HOME").then(|| "/h".to_owned()),
        };
        assert_eq!(l.replicas().unwrap().len(), 2);
        assert!(Layers::none().replicas().unwrap().is_empty(), "absent");
    }

    #[test]
    fn replicas_rejects_non_array_and_non_string() {
        let env = |_: &str| None;
        for (body, want) in [
            (r#"replicas = "/r1""#, "array"),
            (r#"replicas = [1, 2]"#, "string path"),
        ] {
            let file: toml::Table = toml::from_str(body).unwrap();
            let e = replicas_of(&file, "replicas", &env)
                .unwrap_err()
                .to_string();
            assert!(e.contains("replicas") && e.contains(want), "{e}");
        }
    }

    #[test]
    fn models_absent_is_none_and_non_table_errors() {
        assert!(models(&toml::Table::new()).unwrap().is_none());
        let file: toml::Table = toml::from_str(r#"models = 1"#).unwrap();
        let e = models(&file).unwrap_err().to_string();
        assert!(e.contains("models") && e.contains("table"), "{e}");
        let file: toml::Table = toml::from_str("[models.a]\npath = \"/m\"").unwrap();
        assert!(models(&file).unwrap().unwrap().contains_key("a"));
    }
}
