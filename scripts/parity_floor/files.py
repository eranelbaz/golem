"""Leg-directory file layout, atomic writes, and durable leg records."""
from __future__ import annotations

import json
import os


def atomic_write(path: str, content: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"  # pid-suffixed: compare runs outside the leg lock
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


def prompt_paths(legdir: str, idx: int) -> dict[str, str]:
    base = os.path.join(legdir, f"prompt-{idx:02d}")
    return {"txt": base + ".txt", "log": base + ".log",
            "meta": base + ".meta.json", "stdout_bad": base + ".stdout.bad"}


def write_leg_json(legdir: str, meta: dict, results: list) -> None:
    # rewritten after every prompt: intentional crash-resume durability, cheap at n=8
    ok = sum(1 for r in results if r["rc"] == 0)
    atomic_write(os.path.join(legdir, "leg.json"),
                 json.dumps({**meta, "ok": ok, "prompts": results}, indent=2))
