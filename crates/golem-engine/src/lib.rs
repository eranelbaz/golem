//! Model loading, decode loop, and llama.h glue (tech design section 7).

pub mod config;
pub mod error;
pub mod gguf;
pub mod parity;
pub mod parity_op;
pub mod sanity;
pub mod session;

pub use parity::ParityStep;
pub use parity_op::{Quant, run_op_leg};
