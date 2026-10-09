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
    let (tx, frames) = channel(EVENT_CAPACITY);
    tokio::spawn(forward(meta, rx, tx, guard));
    Sse::new(tokio_stream::StreamExt::map(
        ReceiverStream::new(frames),
        Ok,
    ))
    .keep_alive(KeepAlive::default())
}

async fn forward(
    meta: ChunkMeta,
    mut rx: Receiver<Event>,
    tx: Sender<SseEvent>,
    _guard: CancelOnDrop,
) {
    let role = Delta {
        role: Some("assistant".into()),
        ..Delta::default()
    };
    if tx.send(meta.delta(role, None)).await.is_err() {
        return;
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
        let frames = match ev {
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
            Some(Event::Failed(e)) => vec![error_frame(&e)],
            // Worker died mid-stream: an error frame, never a clean `[DONE]`.
            None => vec![error_frame(&backend_gone_envelope())],
        };
        for f in frames {
            if tx.send(f).await.is_err() {
                return;
            }
        }
        if terminal {
            return;
        }
    }
}

fn error_frame(e: &ErrorEnvelope) -> SseEvent {
    SseEvent::default().data(e.to_string())
}
