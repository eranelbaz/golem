"""Test functions for the parity harness — runnable via pytest AND via
`parity-floor.py --selftest` (offline ritual on the measurement box)."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

from .compare import verify_texts
from .config import GEN_RE, BUILD_RE, FA_WARN_RE, PROMPTS, leg_args
from .extract import extract_generation, extract_text, match_count, prefix_words
from .files import atomic_write, prompt_paths, write_leg_json
from .runner import cached_result


def test_extract_text():
    assert extract_text("hello\n\n") == "hello"
    assert extract_text("") == ""


# Fixture: real stdout layout captured from llama-cli (stories15M smoke, this machine);
# banner bytes differ per model but the marker lines are the contract.
REAL = (
    "\n\nLoading model... \n"
    "\nbuild      : b53-d773b57\nmodel      : m.gguf\n\n"
    "> Once upon a time\n"
    "Aweak in the air. Aweak of ice cream is a\n"
    "\n[ Prompt: 18912.5 t/s | Generation: 5114.2 t/s ]\n"
    "\n\nExiting...\n"
)


def test_extract_generation_real_layout():
    assert extract_generation(REAL, "Once upon a time") == \
        "Aweak in the air. Aweak of ice cream is a"


def test_extract_generation_thinking_block():
    glm = "> p\n[Start thinking]\nThe user wants\n\n[ Prompt: 1.0 t/s | Generation: 1.0 t/s ]\nExiting...\n"
    assert extract_generation(glm, "p") == "[Start thinking]\nThe user wants"


def test_extract_generation_marker_in_generation_not_truncated():
    # end anchor = LAST marker occurrence, so a generation containing literal
    # marker text must not truncate the extraction
    tricky = "> p\nabout Exiting... and [ Prompt: fake lines\nreal content\n\n[ Prompt: 1.0 t/s | Generation: 2.0 t/s ]\nExiting...\n"
    assert "real content" in extract_generation(tricky, "p")


def test_extract_generation_fail_closed():
    assert "t/s" not in extract_generation(REAL, "Once upon a time")
    assert extract_generation("no marker here", "p") == ""
    assert extract_generation("> p\nalpha\n", "p") == ""


def test_gen_re():
    assert GEN_RE.search(extract_generation(REAL, "Once upon a time") + "\n" + REAL).group(1) == "5114.2"
    assert GEN_RE.search("[ Prompt: 1.0 t/s | Generation: 12.5 t/s ]").group(1) == "12.5"
    assert GEN_RE.search("Generation: 12.5 t/s trailing junk") is None


def test_build_re():
    assert BUILD_RE.search(REAL).group(1) == "b53-d773b57"


def test_prefix_words():
    assert prefix_words("a b c", "a b d") == 2
    assert prefix_words("x", "x") is None
    assert prefix_words("", "a") == 0
    assert prefix_words("a b", "a b c") == 2


def test_match_count():
    assert match_count(["x", "y"], ["x", "z"]) == 1


def test_prompt_set():
    assert len(PROMPTS) == 8
    assert PROMPTS[0].startswith("Write a short story about a robot")


def test_leg_args():
    hip = leg_args("BIN", "M", "P", True, None, 96)
    assert "-ncmoe" in hip and "45" in hip and hip[hip.index("-t") + 1] == "16"
    assert hip[hip.index("-n") + 1] == "96"
    cpu = leg_args("BIN", "M", "P", False, 32, 96)
    assert "-ncmoe" not in cpu and cpu[cpu.index("-t") + 1] == "32"
    for flag_set in (hip, cpu):
        assert "--temp" in flag_set and "0" in flag_set and "--simple-io" in flag_set


def test_fa_warn_re():
    assert FA_WARN_RE.search("ggml_flash_attn_ext: flash attention not supported, using fallback")
    assert not FA_WARN_RE.search("all good, nothing suspicious here")


def test_cached_result():
    # config match, pin drift, text-hash tamper, absent hash, corrupt meta
    with tempfile.TemporaryDirectory() as td:
        txt = "generated words"
        good = {"rc": 0, "args_sha": "AAA", "pin_sha": "PIN", "gen_s": 1.0,
                "build_id": "b", "text_sha": hashlib.sha256(txt.encode()).hexdigest(),
                "fa_warnings": []}
        base = os.path.join(td, "leg")
        os.makedirs(base)
        p = prompt_paths(base, 0)
        atomic_write(p["txt"], txt)
        atomic_write(p["meta"], json.dumps(good))
        hit = cached_result(base, 0, "AAA", "PIN")
        assert hit is not None and hit["text"] == txt and hit["wall_s"] is None
        assert cached_result(base, 0, "BBB", "PIN") is None          # config mismatch
        assert cached_result(base, 0, "AAA", "OTHER") is None        # pin drift
        tampered = dict(good, text_sha="0" * 64)
        atomic_write(p["meta"], json.dumps(tampered))
        assert cached_result(base, 0, "AAA", "PIN") is None          # hash mismatch
        missing = {k: v for k, v in good.items() if k != "text_sha"}
        atomic_write(p["meta"], json.dumps(missing))
        assert cached_result(base, 0, "AAA", "PIN") is None          # hash absent
        atomic_write(p["meta"], "{broken json")
        assert cached_result(base, 0, "AAA", "PIN") is None          # corrupt meta


def test_write_leg_json_computes_ok():
    with tempfile.TemporaryDirectory() as td:
        write_leg_json(td, {"leg": "x"}, [{"rc": 0}, {"rc": -9}])
        with open(os.path.join(td, "leg.json")) as f:
            assert json.load(f)["ok"] == 1


def test_verify_texts_catches_tamper():
    with tempfile.TemporaryDirectory() as td:
        legdir = os.path.join(td, "legA")
        os.makedirs(legdir)
        prompts = [{"idx": i, "rc": 0, "text": f"t{i}"} for i in range(8)]
        meta = {"leg": "legA", "n_tokens": 96, "model": "M",
                "harness_version": 4, "hip": True, "threads": 16, "prompts": prompts}
        write_leg_json(legdir, meta, prompts)
        for r in prompts:
            p = prompt_paths(legdir, r["idx"])
            atomic_write(p["meta"], json.dumps(
                {"rc": 0, "text_sha": hashlib.sha256(r["text"].encode()).hexdigest()}))
        verify_texts(td, "legA", meta)  # clean passes
        meta["prompts"][3]["text"] = "tampered"
        try:
            verify_texts(td, "legA", meta)
            raise AssertionError("tamper not caught")
        except SystemExit as e:
            assert "text_sha mismatch" in str(e)
