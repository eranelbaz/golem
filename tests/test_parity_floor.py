"""Parity-floor harness tests.

Single source of truth: runnable via pytest AND via
`scripts/parity-floor.py --selftest` (offline ritual on the measurement box).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

from parity_floor.compare import verify_texts
from parity_floor.config import BUILD_RE, FA_WARN_RE, GEN_RE, PROMPTS, leg_args
from parity_floor.extract import extract_generation, extract_text, match_count, prefix_words
from parity_floor.files import atomic_write, prompt_paths, write_leg_json
from parity_floor.runner import cached_result


def test_extract_text():
    """Trailing whitespace is stripped."""
    assert extract_text("hello\n\n") == "hello"
    assert extract_text("") == ""


# Real stdout layout captured from llama-cli (stories15M smoke, this machine);
# banner bytes differ per model but the marker lines are the contract.
REAL_STDOUT = (
    "\n\nLoading model... \n"
    "\nbuild      : b53-d773b57\nmodel      : m.gguf\n\n"
    "> Once upon a time\n"
    "Aweak in the air. Aweak of ice cream is a\n"
    "\n[ Prompt: 18912.5 t/s | Generation: 5114.2 t/s ]\n"
    "\n\nExiting...\n"
)


def test_extract_generation_real_layout():
    assert extract_generation(REAL_STDOUT, "Once upon a time") == \
        "Aweak in the air. Aweak of ice cream is a"


def test_extract_generation_thinking_block():
    glm = "> p\n[Start thinking]\nThe user wants\n\n[ Prompt: 1.0 t/s | Generation: 1.0 t/s ]\nExiting...\n"
    assert extract_generation(glm, "p") == "[Start thinking]\nThe user wants"


def test_extract_generation_survives_markers_in_text():
    """The end anchor is the LAST marker line, so generations containing
    literal '[ Prompt:' / 'Exiting...' must not truncate the extraction."""
    tricky = "> p\nabout Exiting... and [ Prompt: fake lines\nreal content\n\n[ Prompt: 1.0 t/s | Generation: 2.0 t/s ]\nExiting...\n"
    assert "real content" in extract_generation(tricky, "p")


def test_extract_generation_never_leaks_stats_lines():
    assert "t/s" not in extract_generation(REAL_STDOUT, "Once upon a time")


def test_extract_generation_fails_closed_on_unknown_layout():
    assert extract_generation("no marker here", "p") == ""
    assert extract_generation("> p\nalpha\n", "p") == ""


def test_generation_speed_regex():
    assert GEN_RE.search("[ Prompt: 1.0 t/s | Generation: 12.5 t/s ]").group(1) == "12.5"
    assert GEN_RE.search("Generation: 12.5 t/s trailing junk") is None


def test_build_regex():
    assert BUILD_RE.search(REAL_STDOUT).group(1) == "b53-d773b57"


def test_prefix_words():
    assert prefix_words("a b c", "a b d") == 2
    assert prefix_words("x", "x") is None
    assert prefix_words("", "a") == 0
    assert prefix_words("a b", "a b c") == 2


def test_match_count():
    assert match_count(["x", "y"], ["x", "z"]) == 1


def test_prompt_set_is_frozen():
    """Changing PROMPTS invalidates baselines — this test makes it loud."""
    assert len(PROMPTS) == 8
    assert PROMPTS[0].startswith("Write a short story about a robot")


def test_leg_args_flag_contract():
    hip = leg_args("BIN", "M", "P", True, None, 96)
    assert "-ncmoe" in hip and "45" in hip and hip[hip.index("-t") + 1] == "16"
    assert hip[hip.index("-n") + 1] == "96"
    cpu = leg_args("BIN", "M", "P", False, 32, 96)
    assert "-ncmoe" not in cpu and cpu[cpu.index("-t") + 1] == "32"
    for flag_set in (hip, cpu):
        assert "--temp" in flag_set and "0" in flag_set and "--simple-io" in flag_set


def test_fa_warning_regex():
    assert FA_WARN_RE.search("ggml_flash_attn_ext: flash attention not supported, using fallback")
    assert not FA_WARN_RE.search("all good, nothing suspicious here")


def _write_cache_fixture(base_dir: str, sidecar: dict, text: str = "generated words") -> dict:
    paths = prompt_paths(base_dir, 0)
    atomic_write(paths["txt"], text)
    atomic_write(paths["meta"], json.dumps(sidecar))
    return paths


def test_cached_result_accepts_valid_sidecar():
    text = "generated words"
    good = {"rc": 0, "args_sha": "AAA", "pin_sha": "PIN", "gen_s": 1.0, "build_id": "b",
            "text_sha": hashlib.sha256(text.encode()).hexdigest(), "fa_warnings": []}
    with tempfile.TemporaryDirectory() as tmp:
        _write_cache_fixture(tmp, good)
        hit = cached_result(tmp, 0, "AAA", "PIN")
        assert hit is not None and hit["text"] == text and hit["wall_s"] is None


def test_cached_result_rejects_config_mismatch():
    text = "generated words"
    good = {"rc": 0, "args_sha": "AAA", "pin_sha": "PIN", "gen_s": 1.0, "build_id": "b",
            "text_sha": hashlib.sha256(text.encode()).hexdigest(), "fa_warnings": []}
    with tempfile.TemporaryDirectory() as tmp:
        _write_cache_fixture(tmp, good)
        assert cached_result(tmp, 0, "BBB", "PIN") is None
        assert cached_result(tmp, 0, "AAA", "OTHER") is None  # pin drift


def test_cached_result_rejects_bad_text_hash():
    """Missing text_sha must also be rejected — no fail-open path."""
    text = "generated words"
    good = {"rc": 0, "args_sha": "AAA", "pin_sha": "PIN", "gen_s": 1.0, "build_id": "b",
            "text_sha": hashlib.sha256(text.encode()).hexdigest(), "fa_warnings": []}
    with tempfile.TemporaryDirectory() as tmp:
        _write_cache_fixture(tmp, dict(good, text_sha="0" * 64))
        assert cached_result(tmp, 0, "AAA", "PIN") is None
        _write_cache_fixture(tmp, {k: v for k, v in good.items() if k != "text_sha"})
        assert cached_result(tmp, 0, "AAA", "PIN") is None


def test_cached_result_rejects_corrupt_meta():
    text = "generated words"
    good = {"rc": 0, "args_sha": "AAA", "pin_sha": "PIN", "gen_s": 1.0, "build_id": "b",
            "text_sha": hashlib.sha256(text.encode()).hexdigest(), "fa_warnings": []}
    with tempfile.TemporaryDirectory() as tmp:
        paths = _write_cache_fixture(tmp, good)
        atomic_write(paths["meta"], "{broken json")
        assert cached_result(tmp, 0, "AAA", "PIN") is None


def test_write_leg_json_computes_ok_from_return_codes():
    with tempfile.TemporaryDirectory() as tmp:
        write_leg_json(tmp, {"leg": "x"}, [{"rc": 0}, {"rc": -9}])
        with open(os.path.join(tmp, "leg.json")) as leg_file:
            assert json.load(leg_file)["ok"] == 1


def test_verify_texts_accepts_clean_leg():
    with tempfile.TemporaryDirectory() as tmp:
        leg_dir = os.path.join(tmp, "legA")
        os.makedirs(leg_dir)
        prompts = [{"idx": i, "rc": 0, "text": f"t{i}"} for i in range(8)]
        leg_record = {"leg": "legA", "n_tokens": 96, "model": "M", "harness_version": 4,
                      "hip": True, "threads": 16, "prompts": prompts}
        write_leg_json(leg_dir, leg_record, prompts)
        for record in prompts:
            paths = prompt_paths(leg_dir, record["idx"])
            atomic_write(paths["meta"], json.dumps(
                {"rc": 0, "text_sha": hashlib.sha256(record["text"].encode()).hexdigest()}))
        verify_texts(tmp, "legA", leg_record)


def test_verify_texts_catches_leg_json_tamper():
    with tempfile.TemporaryDirectory() as tmp:
        leg_dir = os.path.join(tmp, "legA")
        os.makedirs(leg_dir)
        prompts = [{"idx": i, "rc": 0, "text": f"t{i}"} for i in range(8)]
        leg_record = {"leg": "legA", "n_tokens": 96, "model": "M", "harness_version": 4,
                      "hip": True, "threads": 16, "prompts": prompts}
        write_leg_json(leg_dir, leg_record, prompts)
        for record in prompts:
            paths = prompt_paths(leg_dir, record["idx"])
            atomic_write(paths["meta"], json.dumps(
                {"rc": 0, "text_sha": hashlib.sha256(record["text"].encode()).hexdigest()}))
        leg_record["prompts"][3]["text"] = "tampered"
        try:
            verify_texts(tmp, "legA", leg_record)
            raise AssertionError("tamper not caught")
        except SystemExit as exit_call:
            assert "text_sha mismatch" in str(exit_call)
