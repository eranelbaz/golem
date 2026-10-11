//! Model loading, decode loop, and llama.h glue (tech design section 7).

pub mod atomic;
pub mod chat;
pub mod config;
pub mod error;
pub mod gguf;
pub mod p3_glue;
pub mod parity;
pub mod reasoning;
pub mod sanity;
pub mod scheduler;
pub mod session;
pub mod slot;
pub mod snapshot;
pub mod stream;
pub mod timing;
pub mod verify;

pub use atomic::write_atomic;
pub use chat::{ChatTemplates, OutputParser, Rendered};
pub use error::EngineError;
pub use golem_placement::pool::P3Stats;
pub use parity::ParityStep;
pub use reasoning::{Piece, ReasoningSplitter};
pub use scheduler::{Scheduler, SlotEvent, SlotId, SlotJob};
pub use session::{ChatTurn, Session};
pub use slot::{FinishReason, SamplingParams, TurnOutcome, TurnRequest};
