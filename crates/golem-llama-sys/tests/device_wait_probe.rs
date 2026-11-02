//! ABI v20: the device-side landed wait blocks the compute stream until
//! the host stores the landed word, in both the stream wait-value mode and the
//! wait-kernel fallback. HIP substrate with the C test seams.
#![cfg(golem_backend_hip)]
#![allow(unsafe_code)]

use golem_llama_sys::raw;

fn probe(mode: i32) -> i32 {
    // SAFETY: registry query, then a self-contained probe on that device: it
    // creates and frees its own backend, mapped words and tensor.
    unsafe {
        let dev =
            raw::ggml_backend_dev_by_type(raw::ggml_backend_dev_type::GGML_BACKEND_DEVICE_TYPE_GPU);
        assert!(!dev.is_null(), "no GPU device in the registry");
        raw::golem_device_wait_probe(dev, mode)
    }
}

#[test]
fn stream_wait_value_blocks_until_the_word_is_landed() {
    assert_eq!(
        probe(1),
        0,
        "hipStreamWaitValue32 must hold the stream until the host lands the word \
         (gfx1100 reports hipDeviceAttributeCanUseStreamWaitValue = 1)"
    );
}

#[test]
fn wait_kernel_fallback_blocks_until_the_word_is_landed() {
    assert_eq!(
        probe(2),
        0,
        "the wait kernel must hold the stream until the host lands the word"
    );
}
