//! `stream: true` responses (design §8.1): `chat.completion.chunk` frames
//! over SSE — role chunk, content/reasoning/tool-call deltas, the finish chunk, an
//! optional usage chunk, then `[DONE]`.

use std::convert::Infallible;

use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use golem_api::chat::{ChatCompletionChunk, ChunkChoice, Delta, FinishReason};
use golem_api::{ErrorEnvelope, Usage};
use tokio::sync::mpsc::{Receiver, Sender, channel};
use tokio_stream::Stream;
use tokio_stream::wrappers::ReceiverStream;

use crate::app::{CancelOnDrop, backend_gone_envelope, finish_reason};
use crate::worker::{EVENT_CAPACITY, Event};

/// Chunk identity shared by every frame of one response.
pub(crate) struct ChunkMeta {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub include_usage: bool,
    pub health: std::sync::Arc<crate::worker::HealthState>,
}

impl ChunkMeta {
    fn chunk(&self, choices: Vec<ChunkChoice>, usage: Option<Usage>) -> SseEvent {
        let c = ChatCompletionChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk".into(),
            created: self.created,
            model: self.model.clone(),
            choices,
            usage,
        };
        SseEvent::default().data(serde_json::to_string(&c).unwrap_or_default())
    }

    fn delta(&self, delta: Delta, finish: Option<FinishReason>) -> SseEvent {
        self.chunk(
            vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason: finish,
            }],
            None,
        )
    }
}

/// SSE body over the job's event channel. A forwarding task owns `guard`:
/// when the client disconnects the body (and the frame receiver) drops,
/// `closed()` fires even mid-prefill, the task returns and the guard sets
/// the job's cancel flag.
pub(crate) fn response(
    meta: ChunkMeta,
    rx: Receiver<Event>,
    guard: CancelOnDrop,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let role = Delta {
        role: Some("assistant".into()),
        ..Delta::default()
    };
    let first = vec![meta.delta(role, None)];
    stream(first, rx, guard, KeepAlive::default(), move |ev| {
        openai_frames(&meta, ev)
    })
}

/// Shared SSE plumbing: `first` frames, then `map` per worker event until a
/// terminal one (`Done`, `Failed` or a closed channel).
pub(crate) fn stream(
    first: Vec<SseEvent>,
    rx: Receiver<Event>,
    guard: CancelOnDrop,
    keep_alive: KeepAlive,
    map: impl FnMut(Option<Event>) -> Vec<SseEvent> + Send + 'static,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let (tx, frames) = channel(EVENT_CAPACITY);
    tokio::spawn(forward(first, rx, tx, guard, map));
    Sse::new(tokio_stream::StreamExt::map(
        ReceiverStream::new(frames),
        Ok,
    ))
    .keep_alive(keep_alive)
}

async fn forward(
    first: Vec<SseEvent>,
    mut rx: Receiver<Event>,
    tx: Sender<SseEvent>,
    _guard: CancelOnDrop,
    mut map: impl FnMut(Option<Event>) -> Vec<SseEvent>,
) {
    for f in first {
        if tx.send(f).await.is_err() {
            return;
        }
    }
    loop {
        let ev = tokio::select! {
            _ = tx.closed() => return,
            ev = rx.recv() => ev,
        };
        let terminal = !matches!(
            ev,
            Some(Event::Text(_) | Event::Reasoning(_) | Event::ToolCall(_))
        );
        for f in map(ev) {
            if tx.send(f).await.is_err() {
                return;
            }
        }
        if terminal {
            return;
        }
    }
}

fn openai_frames(meta: &ChunkMeta, ev: Option<Event>) -> Vec<SseEvent> {
    match ev {
        Some(Event::Text(t)) => vec![meta.delta(
            Delta {
                content: Some(t),
                ..Delta::default()
            },
            None,
        )],
        Some(Event::Reasoning(t)) => vec![meta.delta(
            Delta {
                reasoning_content: Some(t),
                ..Delta::default()
            },
            None,
        )],
        Some(Event::ToolCall(c)) => vec![meta.delta(
            Delta {
                tool_calls: Some(vec![c]),
                ..Delta::default()
            },
            None,
        )],
        Some(Event::Done(o)) => {
            let mut v = vec![meta.delta(Delta::default(), Some(finish_reason(o.finish)))];
            if meta.include_usage {
                v.push(meta.chunk(
                    Vec::new(),
                    Some(Usage {
                        prompt_tokens: o.prompt_tokens,
                        completion_tokens: o.completion_tokens,
                        total_tokens: o.prompt_tokens + o.completion_tokens,
                    }),
                ));
            }
            v.push(SseEvent::default().data("[DONE]"));
            v
        }
        Some(Event::Failed(e)) => {
            meta.health.count_error(e.error.code);
            vec![error_frame(&e), SseEvent::default().data("[DONE]")]
        }
        // Worker died mid-stream: the error frame, then `[DONE]` (§8.3).
        None => vec![
            error_frame(&backend_gone_envelope()),
            SseEvent::default().data("[DONE]"),
        ],
    }
}

fn error_frame(e: &ErrorEnvelope) -> SseEvent {
    SseEvent::default().data(e.to_string())
}
