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
/// Wait for the process to exit after a socket stop (30 s drain + teardown).
const EXIT_WAIT: Duration = Duration::from_secs(90);

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
        return wait_exit(&i, EXIT_WAIT);
    }
    golem_registry::signal(&i, libc::SIGTERM)?;
    if wait_exit(&i, TERM_GRACE).is_err() {
        golem_registry::signal(&i, libc::SIGKILL)?;
        wait_exit(&i, EXIT_WAIT)?;
    }
    Ok(())
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
