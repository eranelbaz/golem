//! Per-request decode slots: sampling params, stop strings, `max_tokens`,
//! cancel, and the single decode loop (`step`) shared by `Session::turn` and
//! the multi-slot scheduler. No `unsafe` here: every FFI call goes through the
//! safe wrappers on `SessionInner` (session.rs).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::EngineError;
use crate::parity::ParityStep;
use crate::session::{
    ChatTurn, Core, Sampler, SessionInner, TurnFfi, emit_valid_prefix, flush_utf8_tail,
};

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
    /// Parse reasoning and its markers into `content` (`--reasoning inline`).
    pub reasoning_inline: bool,
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
            reasoning_inline: false,
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

    /// Sets [`TurnRequest::reasoning_inline`].
    pub fn reasoning_inline(mut self, inline: bool) -> Self {
        self.reasoning_inline = inline;
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
    Failed(EngineError),
}

/// Decode steps a generating slot runs between two streamed prefill chunks
/// of a joiner (§7.6). A streamed chunk stalls every generating slot for its
/// whole duration: measured ≈ 91 s cold for one 512-token chunk on the
/// target (`docs/baselines/m3.6-prefill-stream.md`, on512 mean 104 s wall
/// incl. ≈ 10 s load → ≈ 5.6 prompt tok/s). 32 steps bounds that stall to
/// at most 1 in 33 inter-token gaps; the `parallel ∈ {1,2,4}` HW leg
/// (`docs/baselines/m4.4-batching.md`) re-derives it from measured ITL p99.
pub(crate) const STREAM_EVERY: u32 = 32;

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
    /// Greedy without penalties: a failed batch can be replayed exactly. A
    /// grammar does not matter: greedy + grammar is deterministic too.
    replayable: bool,
    /// Failed batches survived this turn (a second one fails the slot).
    failures: u8,
    /// Participates in the step being built: its logits row (`-1` = no
    /// logits) and its prefill chunk length (0 = a generation token).
    in_batch: bool,
    row: i32,
    chunk: usize,
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
            replayable: true,
            failures: 0,
            in_batch: false,
            row: -1,
            chunk: 0,
            parity: None,
        }
    }

    /// Arms the slot for a turn whose prompt (`prompt_tokens`) is already
    /// tokenized and whose sampler is built; `limit` is the clamped token cap.
    pub(crate) fn start(
        &mut self,
        stop: &[String],
        cancel: Option<Arc<AtomicBool>>,
        sampling: &SamplingParams,
        prompt_tokens: u32,
        limit: u32,
        armed: Armed,
    ) {
        self.phase = Phase::Prefill(armed);
        self.stop.reset(stop);
        self.pending.clear();
        self.scratch.clear();
        self.pos = 0;
        self.generated.clear();
        // No growth during decode when max_tokens > n_predict.
        self.generated.reserve(limit as usize);
        self.limit = limit;
        self.prompt_tokens = prompt_tokens;
        self.cancel = cancel;
        self.replayable = sampling.temperature <= 0.0
            && sampling.frequency_penalty == 0.0
            && sampling.presence_penalty == 0.0;
        self.failures = 0;
        self.in_batch = false;
    }

    pub(crate) fn finished(&self) -> Option<FinishReason> {
        match &self.phase {
            Phase::Done(f) => Some(*f),
            _ => None,
        }
    }

    /// No turn armed: the scheduler may admit a job here.
    pub(crate) fn is_free(&self) -> bool {
        matches!(self.phase, Phase::Idle)
    }

    pub(crate) fn is_active(&self) -> bool {
        matches!(self.phase, Phase::Prefill(_) | Phase::Decode(_))
    }

    /// Takes a finished slot's result and frees it.
    pub(crate) fn take_result(&mut self) -> Option<Result<TurnOutcome, EngineError>> {
        let r = match std::mem::replace(&mut self.phase, Phase::Idle) {
            Phase::Done(finish) => Ok(TurnOutcome {
                prompt_tokens: self.prompt_tokens,
                completion_tokens: self.generated.len() as u32,
                finish,
            }),
            Phase::Failed(e) => Err(e),
            p => {
                self.phase = p;
                return None;
            }
        };
        Some(r)
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

    /// Ends an armed-but-failed admission with `e` (reported on next step).
    pub(crate) fn fail(&mut self, e: EngineError) {
        self.phase = Phase::Failed(e);
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

    /// Emits the sampled-but-undecoded token and runs the finish checks;
    /// `true` when it still has to be decoded (the slot generates on).
    fn emit_pending(&mut self, session: &SessionInner, emit: &mut dyn FnMut(&str)) -> bool {
        if session.is_eog(self.tok) {
            self.done(FinishReason::Stop, emit);
            return false;
        }
        let piece = session.piece(self.tok);
        emit_valid_prefix(&mut self.pending, &mut self.scratch, &piece);
        let hit = self.stop.push(&self.scratch, emit);
        self.scratch.clear();
        self.generated.push(self.tok);
        if hit {
            self.pending.clear();
            self.phase = Phase::Done(FinishReason::Stop);
            return false;
        }
        // One KV slot stays free: never decode at pos >= n_ctx_seq - 1.
        let n = self.generated.len() as u32;
        if n >= self.limit || self.pos + 1 >= session.n_ctx_seq() as usize {
            self.done(FinishReason::Length, emit);
            return false;
        }
        if self.cancelled() {
            self.done(FinishReason::Cancelled, emit);
            return false;
        }
        true
    }
}

/// Advances the active slots by exactly one `llama_decode` (the batching
/// invariant): either (a) every generating slot's next token plus at most one
/// joiner's prefill chunk below `--prefill-stream-min` (routed-only fill), or
/// (b) one streamed joiner chunk alone — never a streamed ubatch carrying
/// decode tokens. (b) runs only when no slot is generating or after
/// [`STREAM_EVERY`] decode steps. Joiners rotate round-robin. Cancel is
/// checked before every decode; a cancelled slot's sequence is dropped.
///
/// `recover`: a failed batch (other than abort) resets every participant's
/// sequence; greedy slots re-prefill `prompt ++ generated` and continue
/// exactly, sampled ones end `Failed`. Without it the error is returned
/// (single-turn callers own their recovery).
pub(crate) fn step(
    core: Core<'_>,
    emit: &mut dyn FnMut(usize, &str),
    recover: bool,
) -> Result<(), EngineError> {
    let Core { session, slots } = core;
    let n = slots.len();
    for s in slots.iter_mut() {
        s.in_batch = false;
    }

    // Cancelled prefills end before any decode.
    for (i, s) in slots.iter_mut().enumerate() {
        if matches!(s.phase, Phase::Prefill(_)) && s.cancelled() {
            cancel_seq(session, s.seq);
            s.done(FinishReason::Cancelled, &mut |t| emit(i, t));
        }
    }

    // The joiner: first prefilling slot from the round-robin cursor; if it
    // needs a streamed chunk that is deferred, the next routed-only one.
    let n_dec = slots
        .iter()
        .filter(|s| matches!(s.phase, Phase::Decode(_)))
        .count();
    let may_stream = n_dec == 0 || session.decode_since_stream >= STREAM_EVERY;
    let streams = |i: usize| match &slots[i].phase {
        Phase::Prefill(a) => Some(session.next_chunk(&a.ffi, 0).1),
        _ => None,
    };
    let joiner = (0..n)
        .map(|k| (session.rr + k) % n)
        .find(|&i| streams(i).is_some_and(|st| may_stream || !st));
    let stream_mode = joiner.is_some_and(|j| streams(j) == Some(true));

    session.batch_clear();
    let mut n_gen = 0u32;
    if !stream_mode {
        for (i, s) in slots.iter_mut().enumerate() {
            if !matches!(s.phase, Phase::Decode(_)) {
                continue;
            }
            if !s.emit_pending(session, &mut |t| emit(i, t)) {
                if s.finished() == Some(FinishReason::Cancelled) {
                    cancel_seq(session, s.seq);
                }
                continue;
            }
            s.row = session.batch_push(s.tok, s.pos, s.seq, true)?;
            s.chunk = 0;
            s.in_batch = true;
            n_gen += 1;
        }
    }
    let mut stream: Option<Option<Arc<AtomicBool>>> = None;
    if let Some(j) = joiner {
        let s = &mut slots[j];
        if let Phase::Prefill(a) = &mut s.phase {
            let (len, streams) = session.next_chunk(&a.ffi, n_gen as usize);
            if len > 0 {
                if a.ffi.chunks == 0 {
                    session.prefill_begin(&mut a.ffi);
                }
                s.row = session.push_chunk(&a.ffi, s.seq, len)?;
                s.chunk = len;
                s.in_batch = true;
                session.rr = (j + 1) % n;
                if streams {
                    stream = Some(s.cancel.clone());
                }
            }
        }
    }
    if !slots.iter().any(|s| s.in_batch) {
        return Ok(());
    }

    let t0 = std::time::Instant::now();
    let step_no = slots
        .iter()
        .filter(|s| s.in_batch && s.chunk == 0)
        .map(|s| s.generated.len() as u64)
        .max()
        .unwrap_or(1)
        .saturating_sub(1);
    session.abort.arm(
        slots
            .iter()
            .filter(|s| s.in_batch)
            .map(|s| s.cancel.as_ref()),
    );
    let res = session.run_batch(stream.as_ref().map(Option::as_ref), n_gen, step_no, t0);
    session.abort.clear();
    if stream_mode {
        session.decode_since_stream = 0;
    } else if n_gen > 0 {
        session.decode_since_stream = session.decode_since_stream.saturating_add(1);
    }

    match res {
        Ok(true) => {}
        Ok(false) => {
            // Aborted: every participant was cancelled.
            for (i, s) in slots.iter_mut().enumerate().filter(|(_, s)| s.in_batch) {
                cancel_seq(session, s.seq);
                s.done(FinishReason::Cancelled, &mut |t| emit(i, t));
            }
            return Ok(());
        }
        // A failed page-in (ABI v13) on a dead shard: nothing is retryable.
        Err(_) if session.storage_lost() => return Err(EngineError::StorageUnavailable),
        Err(e) if !recover => return Err(e),
        Err(e) => {
            let e = session.recover_slab(e);
            if let EngineError::DeviceCacheDegraded(k) = e {
                session.degraded = Some(k);
            }
            tracing::warn!(error = %e, "batch decode failed: resetting its sequences");
            for (i, s) in slots.iter_mut().enumerate().filter(|(_, s)| s.in_batch) {
                recover_slot(session, s, &e, &mut |t| emit(i, t));
            }
            return Ok(());
        }
    }

    for s in slots.iter_mut().filter(|s| s.in_batch) {
        let (Phase::Prefill(a) | Phase::Decode(a)) = &mut s.phase else {
            continue;
        };
        if s.chunk == 0 {
            s.pos += 1;
            let n = s.generated.len() as u32;
            s.tok = session.sample(&a.sampler, &a.ffi, s.row, s.parity.as_mut(), n);
            session.verify_decode(&a.ffi)?;
            continue;
        }
        a.ffi.n_past += s.chunk;
        a.ffi.chunks += 1;
        if a.ffi.chunks == session.test_cancel_after_chunks
            && let Some(c) = &s.cancel
        {
            c.store(true, Ordering::Relaxed);
        }
        if a.ffi.n_past < a.ffi.prompt.len() {
            continue;
        }
        session.prefill_end(&a.ffi)?;
        if !a.ffi.replay {
            session.snapshot_prompt(&a.ffi, s.seq);
        }
        if s.limit == 0 {
            s.phase = Phase::Done(FinishReason::Length);
            continue;
        }
        s.pos = a.ffi.prompt.len();
        if n_dec == 0 && n_gen == 0 {
            session.start_decode()?;
        }
        let n = s.generated.len() as u32;
        s.tok = session.sample(&a.sampler, &a.ffi, s.row, s.parity.as_mut(), n);
        let Phase::Prefill(a) = std::mem::replace(&mut s.phase, Phase::Idle) else {
            unreachable!("matched Prefill above");
        };
        s.phase = Phase::Decode(a);
    }
    Ok(())
}

fn cancel_seq(session: &SessionInner, seq: i32) {
    session.seq_rm(seq);
    tracing::info!(slot = seq, "cancel: sequence removed");
}

/// One participant of a failed batch: its sequence is gone. A prefilling
/// slot restarts its prefill; a greedy generating slot re-prefills
/// `prompt ++ generated` (already-sent text is never re-emitted); a sampled
/// one cannot replay its RNG/penalty state and fails.
fn recover_slot(
    session: &mut SessionInner,
    s: &mut SlotState,
    e: &EngineError,
    emit: &mut dyn FnMut(&str),
) {
    session.seq_rm(s.seq);
    s.failures += 1;
    let fail = |s: &mut SlotState, emit: &mut dyn FnMut(&str), msg: String| {
        s.flush_text(emit);
        s.phase = Phase::Failed(match e {
            EngineError::DeviceCacheDegraded(k) => EngineError::DeviceCacheDegraded(*k),
            _ => EngineError::BackendUnavailable(msg),
        });
    };
    if s.failures > 1 {
        fail(s, emit, format!("decode failed twice: {e}"));
        return;
    }
    match std::mem::replace(&mut s.phase, Phase::Idle) {
        Phase::Prefill(mut a) => {
            a.ffi.n_past = session.restore_into(s.seq, &a.ffi.prompt);
            a.ffi.chunks = 0;
            s.phase = Phase::Prefill(a);
        }
        Phase::Decode(mut a) if s.replayable => {
            a.ffi.prompt.extend_from_slice(&s.generated);
            a.ffi.n_past = session.restore_into(s.seq, &a.ffi.prompt);
            a.ffi.chunks = 0;
            a.ffi.replay = true;
            s.phase = Phase::Prefill(a);
        }
        Phase::Decode(_) => fail(
            s,
            emit,
            "decode failed, sampled output cannot be resumed".into(),
        ),
        p => s.phase = p,
    }
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
