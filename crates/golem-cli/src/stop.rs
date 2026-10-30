//! `golem stop` (§9.2): graceful drain over the control socket, `--now`
//! cancels in-flight work, `--force` SIGKILLs.

use std::error::Error;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use golem_registry::{ControlRequest, Registry, RegistryError, StopMode};

/// Arguments of the `stop` subcommand.
#[derive(Debug, clap::Args)]
pub struct StopArgs {
    /// Instance id or served model name.
    pub id: String,
    /// SIGKILL without draining.
    #[arg(long, conflicts_with = "now")]
    pub force: bool,
    /// Cancel the in-flight request instead of draining it.
    #[arg(long)]
    pub now: bool,
}

/// Grace between SIGTERM and SIGKILL for instances without a socket.
const TERM_GRACE: Duration = Duration::from_secs(5);
/// `run` grace after SIGTERM: teardown (munlock + unregister of the pool,
/// snapshot arena, staging ring) measured ≤ 0.42 s on the real model target
/// with a 52 GiB pool, but a SIGTERM during model load is honoured only
/// after it (13 s warm, longer cold), so the bound covers the load, not
/// the teardown.
const TEARDOWN: Duration = Duration::from_secs(60);
/// Exit wait after a socket stop: 30 s drain + 60 s worker join + teardown
/// + 30 s margin.
const EXIT_WAIT: Duration = Duration::from_secs(30 + 60 + TEARDOWN.as_secs() + 30);

/// Heartbeat age (10 × the 1 s beat) and sample gap after which an instance
/// whose control socket is dead counts as hung: wide enough to ride out an
/// mlock/I-O stall of the heartbeat thread while the ≈ 63 GiB pool locks.
const HUNG_AFTER: Duration = Duration::from_secs(10);
/// Heartbeat re-read interval while deciding [`HUNG_AFTER`].
const HUNG_POLL: Duration = Duration::from_secs(1);

/// `golem stop` gave up waiting for the process to exit.
#[derive(Debug, thiserror::Error)]
#[error("instance {id} did not exit within {within:?}; retry with --force")]
pub struct StopTimeout {
    id: String,
    within: Duration,
}

#[cfg(test)]
impl StopTimeout {
    pub fn for_test(id: &str) -> Self {
        Self {
            id: id.into(),
            within: EXIT_WAIT,
        }
    }
}

/// Stops the instance and waits for its process to exit.
pub fn stop(args: &StopArgs) -> Result<(), Box<dyn Error>> {
    let i = Registry::open()?.find(&args.id)?;
    if args.force {
        golem_registry::signal(&i, libc::SIGKILL)?;
        return wait_exit(&i, EXIT_WAIT);
    }
    if i.kind == "serve" {
        let mode = if args.now {
            StopMode::Now
        } else {
            StopMode::Graceful
        };
        return match golem_registry::send(&i.id, &ControlRequest::Stop { mode }) {
            Ok(_) => follow_drain(&i, args.now),
            Err(e) => on_send_err(&i, e, || {
                hung(&i, HUNG_AFTER, HUNG_POLL, |id| Registry::open()?.find(id))
            }),
        };
    }
    golem_registry::signal(&i, libc::SIGTERM)?;
    // `run` tears its session down on SIGTERM (§3.4); others just exit.
    let grace = if i.kind == "run" {
        TEARDOWN
    } else {
        TERM_GRACE
    };
    if wait_exit(&i, grace).is_err() {
        // `signal` re-checks the owner: exiting during the grace is success.
        match golem_registry::signal(&i, libc::SIGKILL) {
            Err(golem_registry::RegistryError::InstanceNotFound(_)) => return Ok(()),
            r => r?,
        }
        wait_exit(&i, EXIT_WAIT)?;
    }
    Ok(())
}

/// A failed control-socket stop: an instance error is the answer; an
/// exited process is stopped; a hung one (`is_hung`) is SIGKILLed.
fn on_send_err(
    i: &golem_registry::Instance,
    e: RegistryError,
    is_hung: impl FnOnce() -> Result<bool, Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    if matches!(e, RegistryError::Control(_)) {
        return Err(e.into());
    }
    if golem_registry::exited(i) {
        return Ok(());
    }
    if !is_hung()? {
        return Err(e.into());
    }
    // A SIGSTOPped process never acts on SIGTERM.
    eprintln!("golem: {} is hung; killing", i.id);
    match golem_registry::signal(i, libc::SIGKILL) {
        // Exited while the heartbeat was being sampled.
        Err(RegistryError::InstanceNotFound(_)) => Ok(()),
        r => {
            r?;
            wait_exit(i, EXIT_WAIT)
        }
    }
}

/// Polls the registry entry until it is gone or the pid exits, printing
/// each phase (`draining → cancelling → releasing`) as the instance's
/// published health moves through it.
fn follow_drain(i: &golem_registry::Instance, now: bool) -> Result<(), Box<dyn Error>> {
    let t = Instant::now();
    let mut printed: Vec<&str> = Vec::new();
    let mut say = |p: &'static str| {
        if !printed.contains(&p) {
            printed.push(p);
            eprintln!("golem: {} {p}", i.id);
        }
    };
    if !now {
        say("draining");
    }
    loop {
        if golem_registry::exited(i) {
            return Ok(());
        }
        let health = match Registry::open()?.find(&i.id) {
            Ok(e) => e.health,
            Err(golem_registry::RegistryError::InstanceNotFound(_)) => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        if health == "releasing" || now || t.elapsed() > golem_server::DRAIN_GRACE {
            say("cancelling");
        }
        if health == "releasing" {
            say("releasing");
        }
        if t.elapsed() > EXIT_WAIT {
            return Err(StopTimeout {
                id: i.id.clone(),
                within: EXIT_WAIT,
            }
            .into());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn wait_exit(i: &golem_registry::Instance, within: Duration) -> Result<(), Box<dyn Error>> {
    let t = Instant::now();
    while !golem_registry::exited(i) {
        if t.elapsed() > within {
            return Err(StopTimeout {
                id: i.id.clone(),
                within,
            }
            .into());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    Ok(())
}

/// The heartbeat is at least `after` old on two samples `after` apart and
/// never advanced in between (re-read via `find` every `poll`). Heartbeat
/// staleness never reaps an entry (§9.2).
fn hung(
    i: &golem_registry::Instance,
    after: Duration,
    poll: Duration,
    find: impl Fn(&str) -> Result<golem_registry::Instance, RegistryError>,
) -> Result<bool, Box<dyn Error>> {
    let stale = |i: &golem_registry::Instance| {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        now.saturating_sub(i.heartbeat_at) >= after.as_secs()
    };
    if !stale(i) {
        return Ok(false);
    }
    eprintln!(
        "golem: {} not responding; checking heartbeat ({} s)",
        i.id,
        after.as_secs()
    );
    let t = Instant::now();
    loop {
        std::thread::sleep(poll);
        let e = match find(&i.id) {
            Ok(e) => e,
            Err(RegistryError::InstanceNotFound(_)) => return Ok(false),
            Err(e) => return Err(e.into()),
        };
        if e.heartbeat_at > i.heartbeat_at {
            return Ok(false);
        }
        if t.elapsed() >= after {
            return Ok(stale(&e));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An instance whose owner pid has exited.
    fn dead(heartbeat_at: u64) -> golem_registry::Instance {
        let mut p = std::process::Command::new("true").spawn().unwrap();
        let pid = p.id();
        p.wait().unwrap();
        golem_registry::Instance {
            id: "x".into(),
            pid,
            pid_start_time: 1,
            kind: "serve".into(),
            endpoint: String::new(),
            model: "m".into(),
            config_hash: "h".into(),
            started_at: 0,
            planned_vram: 0,
            planned_ram: 0,
            health: "healthy".into(),
            heartbeat_at,
            children: vec![],
            config: None,
        }
    }

    const MS: Duration = Duration::from_millis(1);

    #[test]
    fn hung_false_when_entry_vanishes_or_heartbeat_advances() {
        let i = dead(0);
        let gone = |id: &str| Err(RegistryError::InstanceNotFound(id.into()));
        assert!(!hung(&i, Duration::from_secs(1), MS, gone).unwrap());
        let beat = |_: &str| Ok(dead(1));
        assert!(!hung(&i, Duration::from_secs(1), MS, beat).unwrap());
        let frozen = |_: &str| Ok(dead(0));
        assert!(hung(&i, Duration::ZERO, MS, frozen).unwrap());
    }

    #[test]
    fn hung_propagates_registry_errors() {
        let broken = |_: &str| Err(RegistryError::NoDir);
        assert!(hung(&dead(0), Duration::from_secs(1), MS, broken).is_err());
    }

    #[test]
    fn send_failure_on_exited_instance_is_success() {
        let i = dead(0);
        let e = RegistryError::Io(std::io::ErrorKind::ConnectionRefused.into());
        on_send_err(&i, e, || panic!("exited instance sampled for hang")).unwrap();
    }

    #[test]
    fn send_failure_with_control_error_propagates() {
        let env = golem_api::ErrorEnvelope::new(golem_api::ErrorCode::InvalidRequest, "no");
        let r = on_send_err(&dead(0), RegistryError::Control(env), || panic!("sampled"));
        assert!(r.unwrap_err().to_string().contains("no"));
    }
}
