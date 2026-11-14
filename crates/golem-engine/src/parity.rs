//! Parity dump support (T2 harness leg): the per-generation-step record
//! type and the top-2 logits fetch backing `Session::chat_turn_parity`.
//! The logits row itself is read in session.rs (FFI glue); this module is safe.

use golem_llama_sys::raw;
use serde::Serialize;

/// One generation step of a parity dump: the sampler's token choice plus the
/// two largest logits of the row it was sampled from. Serialized directly by
/// the `golem run` dump writer (frozen field order = declaration order).
#[derive(Serialize)]
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
/// two entries yield `0.0` in the vacant slots; NaN entries and the `skip`
/// token ids (the sampler's suppressed EOG tokens, so the record matches
/// what it can emit) never rank.
pub(crate) fn top2_of(logits: &[f32], skip: &[raw::llama_logit_bias]) -> (f32, f32) {
    let mut top = [f32::NEG_INFINITY; 2];
    let ranked = logits
        .iter()
        .enumerate()
        .filter(|(i, _)| !skip.iter().any(|b| b.token as usize == *i));
    for (_, &v) in ranked {
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

#[cfg(test)]
mod tests {
    use super::{raw, top2_of};

    #[test]
    fn two_largest_in_order() {
        assert_eq!(top2_of(&[1.0, 7.0, 3.0], &[]), (7.0, 3.0));
        assert_eq!(top2_of(&[-1.0, -5.0], &[]), (-1.0, -5.0));
    }

    #[test]
    fn ties_count_twice() {
        assert_eq!(top2_of(&[5.0, 5.0, 3.0], &[]), (5.0, 5.0));
    }

    #[test]
    fn fewer_than_two_entries_zero_fill() {
        assert_eq!(top2_of(&[4.25], &[]), (4.25, 0.0));
        assert_eq!(top2_of(&[], &[]), (0.0, 0.0));
    }

    #[test]
    fn nan_never_ranks() {
        assert_eq!(top2_of(&[f32::NAN, 2.0, 9.0, f32::NAN], &[]), (9.0, 2.0));
    }

    #[test]
    fn skipped_tokens_never_rank() {
        let eog = [raw::llama_logit_bias {
            token: 1,
            bias: f32::NEG_INFINITY,
        }];
        assert_eq!(top2_of(&[1.0, 9.0, 3.0], &eog), (3.0, 1.0));
    }
}
