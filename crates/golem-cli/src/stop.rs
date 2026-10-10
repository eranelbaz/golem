//! `golem stop` (§9.2): graceful drain over the control socket, `--now`
//! cancels in-flight work, `--force` SIGKILLs.

use std::error::Error;
use std::time::{Duration, Instant};

use golem_registry::{ControlRequest, Registry, StopMode};

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
        golem_registry::send(&i.id, &ControlRequest::Stop { mode })?;
        return follow_drain(&i, args.now);
    }
    golem_registry::signal(&i, libc::SIGTERM)?;
    if wait_exit(&i, TERM_GRACE).is_err() {
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
            return Err(format!("instance {} did not exit within {EXIT_WAIT:?}", i.id).into());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn wait_exit(i: &golem_registry::Instance, within: Duration) -> Result<(), Box<dyn Error>> {
    let t = Instant::now();
    while !golem_registry::exited(i) {
        if t.elapsed() > within {
            return Err(format!("instance {} did not exit within {within:?}", i.id).into());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    Ok(())
}
