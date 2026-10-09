import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pytest
from pathlib import Path


@pytest.fixture
def tiny_moe_path():
    p = Path(__file__).resolve().parents[1] / ".test-cache" / "tiny-moe-320.gguf"
    if not p.exists():
        pytest.fail(f"{p} missing: run `cargo nextest run -p golem-engine` once to generate it")
    return p
