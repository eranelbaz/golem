//! Splits generated text into reasoning and answer content at the
//! chat-template reasoning markers (`<think>`/`</think>` for GLM-5.3, design
//! §8.1 `reasoning_content`). Markers may straddle piece boundaries; the
//! splitter holds back at most one marker's worth of text.

/// Classified text piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Piece<'a> {
    /// Answer text.
    Content(&'a str),
    /// Text between the open and close markers.
    Reasoning(&'a str),
}

/// Streaming open/close-marker splitter; markers themselves are never emitted.
#[derive(Debug, Clone)]
pub struct ReasoningSplitter {
    open: String,
    close: String,
    in_reasoning: bool,
    held: String,
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

impl ReasoningSplitter {
    /// Splitter for the given markers.
    pub fn new(open: &str, close: &str) -> Self {
        Self {
            open: open.to_owned(),
            close: close.to_owned(),
            in_reasoning: false,
            held: String::new(),
        }
    }

    /// Splitter for a chat template that uses `<think>`/`</think>`; `None`
    /// when the template has no reasoning markers (everything is content).
    pub fn from_template(tpl: &str) -> Option<Self> {
        (tpl.contains(THINK_OPEN) && tpl.contains(THINK_CLOSE))
            .then(|| Self::new(THINK_OPEN, THINK_CLOSE))
    }

    /// Feeds one generated piece.
    pub fn push(&mut self, text: &str, out: &mut dyn FnMut(Piece)) {
        let Self {
            open,
            close,
            in_reasoning,
            held,
        } = self;
        held.push_str(text);
        loop {
            let marker = if *in_reasoning { &*close } else { &*open };
            if let Some(i) = held.find(marker) {
                emit(*in_reasoning, &held[..i], out);
                held.drain(..i + marker.len());
                *in_reasoning = !*in_reasoning;
                continue;
            }
            // Hold back the longest tail that could still start `marker`.
            let keep = (1..marker.len().min(held.len() + 1))
                .rev()
                .find(|&k| {
                    held.is_char_boundary(held.len() - k)
                        && marker.starts_with(&held[held.len() - k..])
                })
                .unwrap_or(0);
            let cut = held.len() - keep;
            emit(*in_reasoning, &held[..cut], out);
            held.drain(..cut);
            return;
        }
    }

    /// Turn end: flushes the held tail and resets for the next turn.
    pub fn finish(&mut self, out: &mut dyn FnMut(Piece)) {
        emit(self.in_reasoning, &self.held, out);
        self.held.clear();
        self.in_reasoning = false;
    }
}

fn emit(reasoning: bool, t: &str, out: &mut dyn FnMut(Piece)) {
    if !t.is_empty() {
        out(if reasoning {
            Piece::Reasoning(t)
        } else {
            Piece::Content(t)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(s: &mut ReasoningSplitter, pieces: &[&str]) -> (String, String) {
        let (mut r, mut c) = (String::new(), String::new());
        let mut out = |p: Piece| match p {
            Piece::Reasoning(t) => r.push_str(t),
            Piece::Content(t) => c.push_str(t),
        };
        for p in pieces {
            s.push(p, &mut out);
        }
        s.finish(&mut out);
        (r, c)
    }

    #[test]
    fn splits_across_token_boundaries() {
        for cut in 1..12 {
            let text = "<think>plan it</think>answer";
            let mut s = ReasoningSplitter::new("<think>", "</think>");
            let (mut r, mut c) = (String::new(), String::new());
            let mut out = |p: Piece| match p {
                Piece::Reasoning(t) => r.push_str(t),
                Piece::Content(t) => c.push_str(t),
            };
            for ch in text.as_bytes().chunks(cut) {
                s.push(std::str::from_utf8(ch).unwrap(), &mut out);
            }
            s.finish(&mut out);
            assert_eq!((r.as_str(), c.as_str()), ("plan it", "answer"), "cut {cut}");
        }
    }

    #[test]
    fn false_marker_prefix_and_multibyte_tail_are_content() {
        let mut s = ReasoningSplitter::new("<think>", "</think>");
        assert_eq!(
            split(&mut s, &["a<th", "ing ", "é<", "x"]),
            (String::new(), "a<thing é<x".into())
        );
        // Unclosed reasoning flushes as reasoning; state resets per turn.
        assert_eq!(
            split(&mut s, &["<think>half"]),
            ("half".into(), String::new())
        );
        assert_eq!(split(&mut s, &["plain"]), (String::new(), "plain".into()));
    }

    #[test]
    fn from_template_needs_both_markers() {
        assert!(ReasoningSplitter::from_template("{{'<think>'}}x{{'</think>'}}").is_some());
        assert!(ReasoningSplitter::from_template("<|im_start|>").is_none());
        assert!(ReasoningSplitter::from_template("<think> only").is_none());
    }
}
