#![allow(unsafe_code, missing_docs)]
include!(concat!(env!("OUT_DIR"), "/backend_info.rs"));

#[path = "../pin_gate.rs"]
pub mod pin_gate;

#[path = "../backend_probe.rs"]
pub mod backend_probe;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Hip,
}

impl Backend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Hip => "hip",
        }
    }
}

pub fn substrate_backend() -> Backend {
    BACKEND
}

pub fn backend_env_input() -> &'static str {
    BACKEND_INPUT
}

pub fn pin_gate_skipped() -> bool {
    option_env!("GOLEM_LLAMA_PIN_GATE_SKIPPED") == Some("1")
}

pub mod raw {
    // bindgen bitfield accessors trip `unnecessary_transmutes`
    #![allow(
        non_camel_case_types,
        non_snake_case,
        non_upper_case_globals,
        improper_ctypes,
        unnecessary_transmutes,
        clippy::all
    )]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
