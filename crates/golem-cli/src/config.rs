//! §9.4 config precedence: flag > env (`GOLEM_<FLAG>`) > file
//! (`$GOLEM_CONFIG_DIR/golem.toml`) > profile > built-in default. A present
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

/// Raw string of a TOML value as the flag would have spelled it.
fn toml_raw(v: &toml::Value) -> String {
    match v {
        toml::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
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

/// The non-flag layers for one invocation.
pub struct Layers<'a> {
    /// `golem.toml` contents (empty when absent).
    pub file: toml::Table,
    /// Profile defaults (empty until profiles land).
    pub profile: toml::Table,
    /// Env lookup (`std::env::var` in production).
    pub env: &'a dyn Fn(&str) -> Option<String>,
}

impl Layers<'_> {
    /// Process env + `$GOLEM_CONFIG_DIR/golem.toml` (none when unset).
    pub fn from_process() -> Result<Layers<'static>, ConfigError> {
        let file = match std::env::var_os("GOLEM_CONFIG_DIR") {
            Some(dir) => {
                let path = std::path::Path::new(&dir).join("golem.toml");
                match std::fs::read_to_string(&path) {
                    Ok(s) => toml::from_str(&s).map_err(|e| ConfigError::File {
                        path,
                        msg: e.to_string(),
                    })?,
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => toml::Table::new(),
                    Err(e) => {
                        return Err(ConfigError::File {
                            path,
                            msg: e.to_string(),
                        });
                    }
                }
            }
            None => toml::Table::new(),
        };
        Ok(Layers {
            file,
            profile: toml::Table::new(),
            env: &|k| std::env::var(k).ok(),
        })
    }

    /// No env, file or profile: flags and defaults only.
    #[cfg(test)]
    pub fn none() -> Layers<'static> {
        Layers {
            file: toml::Table::new(),
            profile: toml::Table::new(),
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
}
