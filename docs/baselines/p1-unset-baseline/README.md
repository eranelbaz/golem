# P1 pristine unset-hooks parity-dump baseline

Byte-identity reference for the M2.1 vendor-patch parity work (Tasks 4–6 compare
against this dump). Frozen from an **unpatched** vendor tree.

- **Model:** tiny-moe GGUF (`xtask gen-tiny-moe`), sha256
  `7841ad0a71541113eb9d7c58237fd20871273fe59194321e64ed79d1a871cfe3`, 179040 bytes.
- **Generator revision:** `a0641e1` (branch `19-taskm21-patch-p1`, vendor/ clean).
- **Command:**

  ```bash
  GOLEM_LLAMA_SYS_BACKEND=cpu cargo build -p golem-cli
  cargo run -p xtask -- gen-tiny-moe --out /tmp/tiny-moe.gguf
  GOLEM_LLAMA_SYS_BACKEND=cpu target/debug/golem run --greedy /tmp/tiny-moe.gguf \
    --prompt "Hello" --n-predict 16 \
    --parity-dump docs/baselines/p1-unset-baseline/dump.jsonl
  ```

- **dump.jsonl sha256:** `af41e8d89457957e888c5f8b5bd79ada74c8db575900fb0d89d2df05c20bb68c`
- **Determinism:** re-run into `/tmp/dump-2.jsonl` is byte-identical (`cmp` clean).
  Greedy decode, CPU substrate, threads=1; header carries model sha256 + size, no paths.
