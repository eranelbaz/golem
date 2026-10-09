//! Model loading, decode loop, and llama.h glue (tech design section 7).

pub mod atomic;
pub mod config;
pub mod error;
pub mod gguf;
pub mod p3_glue;
pub mod parity;
pub mod sanity;
pub mod session;
pub mod slot;
pub mod snapshot;
pub mod stream;
pub mod timing;

pub use atomic::write_atomic;
pub use error::EngineError;
pub use golem_placement::pool::P3Stats;
pub use parity::ParityStep;
pub use session::{ChatTurn, Session};
pub use slot::{FinishReason, SamplingParams, TurnOutcome, TurnRequest};
