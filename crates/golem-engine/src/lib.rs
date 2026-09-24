//! Model loading, decode loop, and llama.h glue (tech design section 7).

pub mod config;
pub mod error;
pub mod gguf;
pub mod parity;
pub mod sanity;
pub mod session;

pub use parity::ParityStep;
