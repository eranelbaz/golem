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
/// Teardown allowance (munlock + unregister of the ≈ 63 GiB pool, snapshot
/// arena, staging ring). Unmeasured on the target until the
/// `m4.7-start-verify.md` HW leg records it; 60 s is a placeholder bound.
const TEARDOWN: Duration = Duration::from_secs(60);
/// Exit wait after a socket stop: 30 s drain + 60 s worker join + teardown
/// + 30 s margin.
const EXIT_WAIT: Duration = Duration::from_secs(30 + 60 + TEARDOWN.as_secs() + 30);

/// Heartbeat age (10 × the 1 s beat) and sample gap after which an instance
/// whose control socket is dead counts as hung: wide enough to ride out an
/// mlock/I-O stall of the heartbeat thread while the ≈ 63 GiB pool locks.
const HUNG_AFTER: Duration = Duration::from_secs(10);

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
        match golem_registry::send(&i.id, &ControlRequest::Stop { mode }) {
            Ok(_) => return follow_drain(&i, args.now),
            Err(e @ RegistryError::Control(_)) => return Err(e.into()),
            Err(_) if golem_registry::exited(&i) => return Ok(()),
            Err(e) if !hung(&i)? => return Err(e.into()),
            Err(_) => {}
        }
        // A SIGSTOPped process never acts on SIGTERM.
        eprintln!("golem: {} is hung; killing", i.id);
        golem_registry::signal(&i, libc::SIGKILL)?;
        return wait_exit(&i, EXIT_WAIT);
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

/// The heartbeat is at least [`HUNG_AFTER`] old on two samples
/// [`HUNG_AFTER`] apart. Heartbeat staleness never reaps an entry (§9.2).
fn hung(i: &golem_registry::Instance) -> Result<bool, Box<dyn Error>> {
    let stale = |i: &golem_registry::Instance| {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        now.saturating_sub(i.heartbeat_at) >= HUNG_AFTER.as_secs()
    };
    if !stale(i) {
        return Ok(false);
    }
    eprintln!(
        "golem: {} not responding; checking heartbeat ({} s)",
        i.id,
        HUNG_AFTER.as_secs()
    );
    std::thread::sleep(HUNG_AFTER);
    match Registry::open()?.find(&i.id) {
        Ok(e) => Ok(stale(&e)),
        Err(RegistryError::InstanceNotFound(_)) => Ok(false),
        Err(e) => Err(e.into()),
    }
}
