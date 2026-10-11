//! OpenAI-compatible chat-completion wire types (tech design §8.1–§8.3).

use serde::{Deserialize, Serialize};

/// `POST /v1/chat/completions` body after §8.2 validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Must equal the instance's loaded model.
    pub model: String,
    /// Conversation so far.
    pub messages: Vec<ChatMessage>,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling mass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Frequency penalty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Presence penalty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Completion token cap; `validate` folds `max_completion_tokens` into it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Newer OpenAI alias of `max_tokens`; `None` after `validate`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// Stop sequences.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Stop>,
    /// RNG seed; `validate` bounds it below `u32::MAX`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Choice count; only 1 is accepted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Server-sent-events streaming.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Streaming options.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// Tool definitions, passed through to the chat template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    /// Tool selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// Tolerated, no effect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    /// Tolerated, no effect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Tolerated, no effect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Golem extension: top-k sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Passed to the chat template as `reasoning_effort`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

/// `stop` is a string or a list of strings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    /// Single sequence.
    One(String),
    /// Several sequences.
    Many(Vec<String>),
}

/// `stream_options` object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamOptions {
    /// Emit a final usage chunk.
    pub include_usage: bool,
}

/// One message in a request or response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// `system`, `user`, `assistant` or `tool`.
    pub role: String,
    /// Text or content parts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    /// Model reasoning text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Assistant tool calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Id of the call a `tool` message answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional participant name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Message content: plain text or typed parts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text.
    Text(String),
    /// Typed parts; only `text` is accepted by `validate`.
    Parts(Vec<ContentPart>),
}

/// One typed content part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentPart {
    /// Part type (`text`, `image_url`, ...).
    #[serde(rename = "type")]
    pub kind: String,
    /// Text of a `text` part.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Remaining fields of non-text parts.
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// Assistant tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Position within a streamed delta.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    /// Call id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Always `function`.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Called function.
    pub function: FunctionCall,
}

/// Function name and JSON-encoded arguments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Function name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Arguments as a JSON string (possibly partial when streamed).
    #[serde(default)]
    pub arguments: String,
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// EOS or stop sequence.
    Stop,
    /// Token cap reached.
    Length,
    /// Model emitted tool calls.
    ToolCalls,
}

/// Token accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt tokens.
    pub prompt_tokens: u32,
    /// Generated tokens.
    pub completion_tokens: u32,
    /// Sum of both.
    pub total_tokens: u32,
}

/// Non-streaming response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// Completion id.
    pub id: String,
    /// Always `chat.completion`.
    pub object: String,
    /// Unix seconds.
    pub created: u64,
    /// Served model.
    pub model: String,
    /// Exactly one choice.
    pub choices: Vec<Choice>,
    /// Token accounting.
    pub usage: Usage,
}

/// Non-streaming choice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    /// Choice index.
    pub index: u32,
    /// Assistant message.
    pub message: ChatMessage,
    /// Stop reason.
    pub finish_reason: FinishReason,
}

/// One SSE chunk of a streaming response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    /// Completion id, shared by every chunk.
    pub id: String,
    /// Always `chat.completion.chunk`.
    pub object: String,
    /// Unix seconds.
    pub created: u64,
    /// Served model.
    pub model: String,
    /// Empty on the final usage chunk.
    pub choices: Vec<ChunkChoice>,
    /// Present only on the final chunk with `include_usage`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// Streaming choice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkChoice {
    /// Choice index.
    pub index: u32,
    /// Incremental message.
    pub delta: Delta,
    /// Set on the last content chunk.
    pub finish_reason: Option<FinishReason>,
}

/// Incremental message fields.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Delta {
    /// Set on the first chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Text piece.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Reasoning piece.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Tool-call pieces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// `GET /v1/models` response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelList {
    /// Always `list`.
    pub object: String,
    /// Loaded models.
    pub data: Vec<ModelCard>,
}

/// One model entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCard {
    /// Model id.
    pub id: String,
    /// Always `model`.
    pub object: String,
    /// Unix seconds.
    pub created: u64,
    /// Always `golem`.
    pub owned_by: String,
}
