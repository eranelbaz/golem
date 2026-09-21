"""Output parsing and text comparison primitives (fail-closed extraction)."""
from __future__ import annotations

from .config import BUILD_RE, FA_WARN_RE


def extract_text(raw: str) -> str:
    return raw.rstrip()


def extract_generation(raw: str, prompt: str) -> str:
    """Parse llama-cli stdout down to the generated text.

    stdout layout (with --simple-io --no-display-prompt): banner, 'Loading model...',
    a line starting '> ' (echoing the prompt), the generation, then the
    '[ Prompt: ... | Generation: ... t/s ]' stats line and 'Exiting...'.
    Returns '' when the layout is not recognized — callers treat that as a
    harness error, never as an (empty) generation.
    """
    lines = raw.splitlines()
    end = None
    for marker in ("[ Prompt:", "Exiting..."):
        last = max((k for k, ln in enumerate(lines) if ln.startswith(marker)), default=None)
        if last is not None:
            end = last if end is None else min(end, last)
    if end is None:
        return ""
    start = next((i + 1 for i in range(end) if lines[i].startswith("> ")), None)
    if start is None:
        return ""
    body = lines[start:end]
    if body and body[0].startswith(prompt):
        body[0] = body[0][len(prompt):]
    return "\n".join(body).strip("\n")


def prefix_words(a: str, b: str) -> int | None:
    """Common whitespace-word prefix length; None when texts are byte-equal."""
    if a == b:
        return None
    n = 0
    for x, y in zip(a.split(), b.split()):
        if x != y:
            break
        n += 1
    return n


def match_count(texts_a: list[str], texts_b: list[str]) -> int:
    """Count of byte-equal prompt pairs."""
    return sum(1 for a, b in zip(texts_a, texts_b) if a == b)


def parse_run_output(combined: str, stderr: str) -> tuple[str | None, list[str]]:
    m = BUILD_RE.search(combined)
    build_id = m.group(1) if m else None
    return build_id, FA_WARN_RE.findall(stderr)
