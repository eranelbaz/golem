//! Per-request decode slots: sampling params, stop strings, `max_tokens`,
//! cancel, and the single decode loop (`step`) shared by `Session::turn` and
//! the multi-slot scheduler. No `unsafe` here: every FFI call goes through the
//! safe wrappers on `SessionInner` (session.rs).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::EngineError;
use crate::parity::ParityStep;
use crate::session::{ChatTurn, Core, Sampler, TurnFfi, emit_valid_prefix, flush_utf8_tail};

/// Per-request sampling knobs (OpenAI semantics). `temperature <= 0` is greedy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    /// Softmax temperature; `<= 0` selects greedy.
    pub temperature: f32,
    /// Nucleus mass kept (1.0 = off).
    pub top_p: f32,
    /// Top-k cut (0 = off).
    pub top_k: u32,
    /// `dist` sampler seed.
    pub seed: u32,
    /// OpenAI frequency penalty (0 = off).
    pub frequency_penalty: f32,
    /// OpenAI presence penalty (0 = off).
    pub presence_penalty: f32,
}

impl Default for SamplingParams {
    // Hand-written: a derived `top_p: 0.0` would silently mean greedy.
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

/// One turn's request; build with [`TurnRequest::new`] and the setters.
pub struct TurnRequest<'a> {
    /// Transcript to render with the session template; ignored when
    /// `messages_json` is set.
    pub messages: &'a [ChatTurn<'a>],
    /// Raw OpenAI `messages` array (keeps assistant `tool_calls` and `tool`
    /// messages, which `ChatTurn` cannot carry).
    pub messages_json: Option<&'a str>,
    /// OpenAI `tools` array JSON.
    pub tools_json: Option<&'a str>,
    /// OpenAI `tool_choice` JSON (`"auto"`, `"none"`, `"required"` or a
    /// named-function object).
    pub tool_choice: Option<&'a str>,
    /// Pre-rendered prompt from [`crate::Session::render`]; overrides the
    /// three fields above.
    pub rendered: Option<&'a crate::chat::Rendered>,
    /// Sampler chain parameters.
    pub sampling: SamplingParams,
    /// Generated-token cap; `None` = `EngineConfig::n_predict`.
    pub max_tokens: Option<u32>,
    /// Stop strings; the match and everything after it are never emitted.
    pub stop: &'a [String],
    /// Checked before every `llama_decode`; set → `FinishReason::Cancelled`.
    pub cancel: Option<Arc<AtomicBool>>,
}

impl<'a> TurnRequest<'a> {
    /// Request with default (greedy) sampling and no limits.
    pub fn new(messages: &'a [ChatTurn<'a>]) -> Self {
        Self {
            messages,
            messages_json: None,
            tools_json: None,
            tool_choice: None,
            rendered: None,
            sampling: SamplingParams::default(),
            max_tokens: None,
            stop: &[],
            cancel: None,
        }
    }

    /// Sets [`TurnRequest::messages_json`].
    pub fn messages_json(mut self, json: &'a str) -> Self {
        self.messages_json = Some(json);
        self
    }

    /// Sets [`TurnRequest::tools_json`] and [`TurnRequest::tool_choice`].
    pub fn tools(mut self, tools_json: Option<&'a str>, tool_choice: Option<&'a str>) -> Self {
        self.tools_json = tools_json;
        self.tool_choice = tool_choice;
        self
    }

    /// Sets [`TurnRequest::rendered`].
    pub fn rendered(mut self, rendered: &'a crate::chat::Rendered) -> Self {
        self.rendered = Some(rendered);
        self
    }

    /// Sets [`TurnRequest::sampling`].
    pub fn sampling(mut self, sampling: SamplingParams) -> Self {
        self.sampling = sampling;
        self
    }

    /// Sets [`TurnRequest::max_tokens`].
    pub fn max_tokens(mut self, max_tokens: Option<u32>) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Sets [`TurnRequest::stop`].
    pub fn stop(mut self, stop: &'a [String]) -> Self {
        self.stop = stop;
        self
    }

    /// Sets [`TurnRequest::cancel`].
    pub fn cancel(mut self, cancel: Option<Arc<AtomicBool>>) -> Self {
        self.cancel = cancel;
        self
    }
}

/// Why a turn ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// EOG token or stop string.
    Stop,
    /// `max_tokens` or the context window reached.
    Length,
    /// The request's cancel flag was set.
    Cancelled,
    /// Set only by tool-call parsing.
    ToolCalls,
}

/// Usage and finish reason of one turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnOutcome {
    /// Tokenized prompt length.
    pub prompt_tokens: u32,
    /// Tokens generated (EOG excluded).
    pub completion_tokens: u32,
    /// Why the turn ended.
    pub finish: FinishReason,
}

/// Holds back up to `max_stop_len - 1` bytes so a stop string split across
/// pieces is still caught; the tail buffer is reused across turns.
#[derive(Default)]
pub(crate) struct StopMatcher {
    stops: Vec<String>,
    max_len: usize,
    held: String,
}

impl StopMatcher {
    fn reset(&mut self, stops: &[String]) {
        self.stops.clear();
        self.stops
            .extend(stops.iter().filter(|s| !s.is_empty()).cloned());
        self.max_len = self.stops.iter().map(String::len).max().unwrap_or(0);
        self.held.clear();
    }

    /// Emits text that can no longer start a stop; `true` once a stop matched
    /// (text before it emitted, the stop and anything after dropped).
    fn push(&mut self, s: &str, emit: &mut dyn FnMut(&str)) -> bool {
        if self.stops.is_empty() {
            if !s.is_empty() {
                emit(s);
            }
            return false;
        }
        self.held.push_str(s);
        if let Some(i) = self.stops.iter().filter_map(|st| self.held.find(st)).min() {
            if i > 0 {
                emit(&self.held[..i]);
            }
            self.held.clear();
            return true;
        }
        let keep = self.max_len - 1;
        if self.held.len() > keep {
            let cut = self.held.floor_char_boundary(self.held.len() - keep);
            if cut > 0 {
                emit(&self.held[..cut]);
                self.held.drain(..cut);
            }
        }
        false
    }

    fn flush(&mut self, emit: &mut dyn FnMut(&str)) {
        if !self.held.is_empty() {
            emit(&self.held);
            self.held.clear();
        }
    }
}

/// Per-turn sampler chain and FFI state, owned by the active phase so
/// `Prefill`/`Decode` cannot exist without them.
pub(crate) struct Armed {
    pub(crate) sampler: Sampler,
    pub(crate) ffi: TurnFfi,
}

enum Phase {
    Idle,
    Prefill(Armed),
    Decode(Armed),
    Done(FinishReason),
}

/// One decode slot, allocated once at `Session::new` and reused per turn.
pub struct SlotState {
    pub(crate) seq: i32,
    phase: Phase,
    stop: StopMatcher,
    pending: Vec<u8>,
    scratch: String,
    tok: i32,
    pos: usize,
    pub(crate) generated: Vec<i32>,
    limit: u32,
    pub(crate) prompt_tokens: u32,
    cancel: Option<Arc<AtomicBool>>,
    pub(crate) parity: Option<Vec<ParityStep>>,
}

impl SlotState {
    pub(crate) fn new(seq: i32, n_predict: u32) -> Self {
        Self {
            seq,
            phase: Phase::Idle,
            stop: StopMatcher::default(),
            pending: Vec::new(),
            scratch: String::new(),
            tok: 0,
            pos: 0,
            generated: Vec::with_capacity(n_predict as usize),
            limit: 0,
            prompt_tokens: 0,
            cancel: None,
            parity: None,
        }
    }

    /// Arms the slot for a turn whose prompt (`prompt_tokens`) is already
    /// tokenized and whose sampler is built; `limit` is the clamped token cap.
    pub(crate) fn start(
        &mut self,
        req: &TurnRequest<'_>,
        prompt_tokens: u32,
        limit: u32,
        armed: Armed,
    ) {
        self.phase = Phase::Prefill(armed);
        self.stop.reset(req.stop);
        self.pending.clear();
        self.scratch.clear();
        self.pos = 0;
        self.generated.clear();
        // No growth during decode when max_tokens > n_predict.
        self.generated.reserve(limit as usize);
        self.limit = limit;
        self.prompt_tokens = prompt_tokens;
        self.cancel = req.cancel.clone();
    }

    pub(crate) fn finished(&self) -> Option<FinishReason> {
        match &self.phase {
            Phase::Done(f) => Some(*f),
            _ => None,
        }
    }

    fn cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .is_some_and(|c| c.load(Ordering::Relaxed))
    }

    fn done(&mut self, f: FinishReason, emit: &mut dyn FnMut(&str)) {
        self.flush_text(emit);
        self.phase = Phase::Done(f);
    }

    /// Drops any active phase (sampler, batch) after an `Err` exit.
    pub(crate) fn abort(&mut self) {
        self.phase = Phase::Idle;
    }

    /// Turn-end text flush: UTF-8 residue through the stop matcher, then the
    /// held tail. Runs on every finish and on `Err`.
    pub(crate) fn flush_text(&mut self, emit: &mut dyn FnMut(&str)) {
        flush_utf8_tail(&mut self.pending, &mut self.scratch);
        let hit = self.stop.push(&self.scratch, emit);
        self.scratch.clear();
        if !hit {
            self.stop.flush(emit);
        }
    }
}

/// Advances every active slot by one unit of work: a slot in `Prefill` runs its
/// chunked prefill and samples the bootstrap token; a slot in `Decode` emits
/// its pending token and decodes it. Cancel is checked before every
/// `llama_decode`.
pub(crate) fn step(core: Core<'_>, emit: &mut dyn FnMut(&str)) -> Result<(), EngineError> {
    let Core { session, slots } = core;
    for slot in slots.iter_mut() {
        // Move the phase out so the armed state is owned here; every arm
        // writes the next phase back (no allocation: a stack move).
        match std::mem::replace(&mut slot.phase, Phase::Idle) {
            p @ (Phase::Idle | Phase::Done(_)) => slot.phase = p,
            Phase::Prefill(mut a) => {
                let cancel = slot.cancel.clone();
                match session.prefill(&mut a.ffi, slot.seq, cancel.as_deref())? {
                    None => slot.done(FinishReason::Cancelled, emit),
                    Some(_) if slot.limit == 0 => slot.done(FinishReason::Length, emit),
                    Some(last) => {
                        session.snapshot_prompt(&a.ffi, slot.seq);
                        slot.pos = slot.prompt_tokens as usize;
                        session.start_decode()?;
                        slot.tok = session.sample(
                            &a.sampler,
                            &a.ffi,
                            last as i32 - 1,
                            slot.parity.as_mut(),
                            0,
                        );
                        slot.phase = Phase::Decode(a);
                    }
                }
            }
            Phase::Decode(mut a) => {
                let t0 = std::time::Instant::now();
                if session.is_eog(slot.tok) {
                    slot.done(FinishReason::Stop, emit);
                    continue;
                }
                let piece = session.piece(slot.tok);
                emit_valid_prefix(&mut slot.pending, &mut slot.scratch, &piece);
                let hit = slot.stop.push(&slot.scratch, emit);
                slot.scratch.clear();
                slot.generated.push(slot.tok);
                if hit {
                    slot.pending.clear();
                    slot.phase = Phase::Done(FinishReason::Stop);
                    continue;
                }
                // One KV slot stays free: never decode at pos >= n_ctx_seq - 1.
                let n = slot.generated.len() as u32;
                if n >= slot.limit || slot.pos + 1 >= session.n_ctx_seq() as usize {
                    slot.done(FinishReason::Length, emit);
                    continue;
                }
                if slot.cancelled() {
                    slot.done(FinishReason::Cancelled, emit);
                    continue;
                }
                if !session.decode(&mut a.ffi, slot.seq, slot.tok, slot.pos, n, t0)? {
                    slot.done(FinishReason::Cancelled, emit);
                    continue;
                }
                slot.pos += 1;
                slot.tok = session.sample(&a.sampler, &a.ffi, 0, slot.parity.as_mut(), n);
                slot.phase = Phase::Decode(a);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(stops: &[&str], pieces: &[&str]) -> (String, bool) {
        let stops: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        let mut m = StopMatcher::default();
        m.reset(&stops);
        let mut out = String::new();
        let mut hit = false;
        for p in pieces {
            if m.push(p, &mut |s| out.push_str(s)) {
                hit = true;
                break;
            }
        }
        if !hit {
            m.flush(&mut |s| out.push_str(s));
        }
        (out, hit)
    }

    #[test]
    fn stop_holdback_is_char_boundary_safe_and_flushed_on_length() {
        let text = "ab😀";
        let bounds: Vec<usize> = text
            .char_indices()
            .map(|(i, _)| i)
            .chain([text.len()])
            .collect();
        for &a in &bounds {
            for &b in bounds.iter().filter(|&&b| b >= a) {
                let pieces = [&text[..a], &text[a..b], &text[b..]];
                // No match: Length flush returns the text intact.
                for stops in [&["😀x"][..], &["xy"], &["x😀"], &["zzzzzzzzz"]] {
                    assert_eq!(
                        feed(stops, &pieces),
                        (text.to_owned(), false),
                        "{stops:?} {a} {b}"
                    );
                }
                // Match: truncated before the stop.
                assert_eq!(feed(&["😀"], &pieces), ("ab".to_owned(), true), "{a} {b}");
                assert_eq!(feed(&["b😀"], &pieces), ("a".to_owned(), true), "{a} {b}");
            }
        }
        assert_eq!(feed(&[], &["ab", "😀"]), (text.to_owned(), false));
    }

    #[test]
    fn default_sampling_is_greedy_with_full_top_p() {
        let p = SamplingParams::default();
        assert_eq!((p.temperature, p.top_p), (0.0, 1.0));
    }
}
