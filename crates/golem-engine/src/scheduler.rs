//! Continuous batching over the session's persistent slots (§7, D-13).
//!
//! Arithmetic (target, `--parallel 4`): 4 slots route ≈ 30.7 unique experts
//! per layer per step vs 32 → aggregate tok/s ≈ flat, each slot ≈ ¼, so the
//! default stays `parallel = 1`. Per-slot memory: recurrent state 145.56 MiB
//! × 4 = 582 MiB VRAM plus KV 19,712 B × n_ctx × 4 (Task 7 plan terms). The
//! fairness interval between streamed joiner chunks is
//! [`crate::slot::STREAM_EVERY`] (cited there against #49's measured cost).

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::chat::Rendered;
use crate::error::EngineError;
use crate::session::{Core, Session};
use crate::slot::{self, SamplingParams, TurnOutcome};

/// Index of a slot (= its llama sequence id).
pub type SlotId = usize;

/// One admitted request: rendered + tokenized on the worker before `admit`.
pub struct SlotJob {
    /// Tokenized prompt (see [`Scheduler::tokenize`]).
    pub tokens: Vec<i32>,
    /// Sampler chain parameters.
    pub sampling: SamplingParams,
    /// Generated-token cap; `None` = `EngineConfig::n_predict`.
    pub max_tokens: Option<u32>,
    /// Stop strings.
    pub stop: Vec<String>,
    /// Set by the client (disconnect) or drain.
    pub cancel: Arc<AtomicBool>,
    /// The render the prompt came from: its tool-call grammar joins the
    /// slot's sampler chain (Task 6); `None` for raw prompts.
    pub rendered: Option<Rendered>,
}

/// What one slot produced during a [`Scheduler::step`].
#[derive(Debug)]
pub enum SlotEvent {
    /// Generated text piece (reasoning/tool parsing stays with the caller,
    /// which owns the request's parser).
    Text(String),
    /// The turn finished; the slot is free again.
    Done(TurnOutcome),
    /// The turn failed; the slot is free again. Maps to 500
    /// `golem_backend_unavailable` on the server.
    Failed(EngineError),
}

/// Drives every slot of one [`Session`] through the shared `slot::step` —
/// the same loop `Session::turn` uses for slot 0.
pub struct Scheduler {
    session: Session,
    /// Each slot's last render, kept for its parser until the slot is
    /// re-admitted (the final parse runs after `Done`).
    rendered: Vec<Option<Rendered>>,
}

impl Scheduler {
    /// `parallel` comes from the session's config only.
    pub fn new(mut session: Session) -> Result<Self, EngineError> {
        let n = session.parts().1.len();
        Ok(Self {
            session,
            rendered: (0..n).map(|_| None).collect(),
        })
    }

    /// The parser of slot `id`'s current (or just-finished) job.
    pub fn parser(&self, id: SlotId) -> Option<&crate::chat::OutputParser> {
        self.rendered.get(id)?.as_ref().map(|r| &r.parser)
    }

    /// The owned session (render, stats, reasoning splitter).
    pub fn session(&mut self) -> &mut Session {
        &mut self.session
    }

    /// Tokenizes a rendered prompt exactly as `Session::turn` does.
    pub fn tokenize(&mut self, text: &str) -> Result<Vec<i32>, EngineError> {
        self.session.parts().0.tokenize(text)
    }

    /// Slots without a turn.
    pub fn free_slots(&mut self) -> usize {
        self.session
            .parts()
            .1
            .iter()
            .filter(|s| s.is_free())
            .count()
    }

    /// Any slot prefilling or generating.
    pub fn any_active(&mut self) -> bool {
        self.session.parts().1.iter().any(|s| s.is_active())
    }

    /// Arms the first free slot; hands the job back when all are busy. An
    /// arming error (context overflow, grammar) surfaces as that slot's
    /// `Failed` on the next `step`. The job is returned by value (the brief's
    /// interface); it is moved once per admission, never per step.
    #[allow(clippy::result_large_err)]
    pub fn admit(&mut self, job: SlotJob) -> Result<SlotId, SlotJob> {
        let (inner, slots) = self.session.parts();
        let Some(id) = slots.iter().position(|s| s.is_free()) else {
            return Err(job);
        };
        let slot = &mut slots[id];
        match inner.arm(
            slot,
            job.tokens,
            &job.sampling,
            job.rendered.as_ref(),
            &job.stop,
            job.max_tokens,
            Some(job.cancel),
        ) {
            Ok(guard) => guard.disarm(),
            Err(e) => slot.fail(e),
        }
        self.rendered[id] = job.rendered;
        Ok(id)
    }

    /// K of a device-cache degradation hit by a batched step since the last
    /// call (greedy slots replay it, so no `Failed` may carry it).
    pub fn take_degraded(&mut self) -> Option<u32> {
        self.session.parts().0.degraded.take()
    }

    /// One `llama_decode` across the active slots (see `slot::step`), then
    /// reports finished slots. `Ok(true)` while any slot is still active.
    pub fn step(&mut self, out: &mut dyn FnMut(SlotId, SlotEvent)) -> Result<bool, EngineError> {
        let (inner, slots) = self.session.parts();
        if slots.iter().any(|s| s.is_active()) {
            slot::step(
                Core {
                    session: inner,
                    slots,
                },
                &mut |i, t| out(i, SlotEvent::Text(t.to_owned())),
                true,
            )?;
        }
        let mut active = false;
        for (i, s) in slots.iter_mut().enumerate() {
            let ev = match s.take_result() {
                Some(Ok(o)) => SlotEvent::Done(o),
                Some(Err(e)) => SlotEvent::Failed(e),
                None => {
                    active |= s.is_active();
                    continue;
                }
            };
            out(i, ev);
        }
        Ok(active)
    }
}
