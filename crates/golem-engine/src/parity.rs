//! Parity dump support (M1.4 T2 harness leg): the per-generation-step record
//! type and the top-2 logits fetch backing `Session::chat_turn_parity`.
//! Raw llama.cpp FFI, mirroring session.rs's unsafe-allowed module style.
#![allow(unsafe_code)]

use golem_llama_sys::raw;

/// One generation step of a parity dump: the sampler's token choice plus the
/// two largest logits of the row it was sampled from.
pub struct ParityStep {
    /// 0-based generation-step index (not `llama_pos`).
    pub position: u32,
    /// Token id returned by `llama_sampler_sample` (greedy chain — never a
    /// recomputed argmax; tie-breaks belong to llama.cpp).
    pub token: i32,
    /// Largest logit of the sampled logits row.
    pub top1: f32,
    /// Second-largest logit (`0.0` when the row has fewer than two entries).
    pub top2: f32,
}

/// The two largest values of `logits` as a `(top1, top2)` pair with
/// `top1 >= top2`. Ties count twice (`[5, 5, 3]` → `(5.0, 5.0)`); fewer than
/// two entries yield `0.0` in the vacant slots; NaN entries never rank.
pub(crate) fn top2_of(logits: &[f32]) -> (f32, f32) {
    let mut top = [f32::NEG_INFINITY; 2];
    for &v in logits {
        if v > top[0] {
            top[1] = top[0];
            top[0] = v;
        } else if v > top[1] {
            top[1] = v;
        }
    }
    match logits.len() {
        0 => (0.0, 0.0),
        1 => (top[0], 0.0),
        _ => (top[0], top[1]),
    }
}

/// Fetches the logits row for `ith` and returns its top-2 values. A pure read:
/// `llama_sampler_sample` has already consumed the row by the time this runs,
/// and the accessor mutates no context state, so decode behavior is identical
/// with or without recording.
///
/// # Safety
/// `ctx` must be a live `llama_context` whose most recent decode succeeded and
/// emitted logits for slot `ith`, and `n_vocab` must be that row's length.
pub(crate) unsafe fn top2_logits_ith(
    ctx: *mut raw::llama_context,
    ith: i32,
    n_vocab: usize,
) -> (f32, f32) {
    // SAFETY: ctx is live and its last decode emitted logits for slot ith
    // (caller contract, llama.h); a null return would be a broken FFI
    // invariant, not a model rejection — same precedent as the session.rs
    // decode self-check.
    let ptr = unsafe { raw::llama_get_logits_ith(ctx, ith) };
    if ptr.is_null() {
        panic!("llama_get_logits_ith returned null during parity recording");
    }
    // SAFETY: ptr is non-null and valid for n_vocab f32 values (llama.h contract).
    let logits = unsafe { std::slice::from_raw_parts(ptr, n_vocab) };
    top2_of(logits)
}

#[cfg(test)]
mod tests {
    use super::top2_of;

    #[test]
    fn two_largest_in_order() {
        assert_eq!(top2_of(&[1.0, 7.0, 3.0]), (7.0, 3.0));
        assert_eq!(top2_of(&[-1.0, -5.0]), (-1.0, -5.0));
    }

    #[test]
    fn ties_count_twice() {
        assert_eq!(top2_of(&[5.0, 5.0, 3.0]), (5.0, 5.0));
    }

    #[test]
    fn fewer_than_two_entries_zero_fill() {
        assert_eq!(top2_of(&[4.25]), (4.25, 0.0));
        assert_eq!(top2_of(&[]), (0.0, 0.0));
    }

    #[test]
    fn nan_never_ranks() {
        assert_eq!(top2_of(&[f32::NAN, 2.0, 9.0, f32::NAN]), (9.0, 2.0));
    }
}
