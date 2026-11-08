#!/usr/bin/env python3
"""Prints shell assignments for a [models.<name>] entry in golem.toml
($GOLEM_CONFIG_DIR, else ~/.config/golem), default its model_name:
GOLEM_MODEL, GOLEM_AB_GIB and GOLEM_MODEL_FLAGS (the model's VRAM knobs as
`golem run` flags).
usage: eval "$(scripts/golem-toml-model.py [name])" """
import os
import shlex
import sys
import tomllib
from pathlib import Path

FLAGS = ("device_cache_experts", "vram_reserve_gib", "vram_experts_frac", "prefill_ubatch")


def main() -> None:
    cfg = Path(os.environ.get("GOLEM_CONFIG_DIR", Path.home() / ".config/golem")) / "golem.toml"
    t = tomllib.loads(cfg.read_text())
    name = (sys.argv[1:] or [t.get("model_name")])[0] or sys.exit(f"{cfg}: no model_name")
    m = {**{k: v for k, v in t.items() if not isinstance(v, dict)}, **t["models"][name]}
    flags = [f"--{k.replace('_', '-')} {m[k]}" for k in FLAGS if k in m]
    if m.get("flash_attn"):
        flags.append("--flash-attn")
    print(f"GOLEM_MODEL={shlex.quote(os.path.expanduser(m['path']))}")
    print(f"GOLEM_AB_GIB={m['ram_experts_gib']}")
    print(f"GOLEM_MODEL_FLAGS={shlex.quote(' '.join(flags))}")


if __name__ == "__main__":
    main()
