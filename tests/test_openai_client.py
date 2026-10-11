import subprocess, re, time, openai
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

GOLEM = next((p for p in (REPO / "target/release/golem", REPO / "target/debug/golem") if p.exists()), None)

def test_openai_client_non_streaming(tiny_moe_path, tmp_path):
    assert GOLEM, "build golem first (cargo build -p golem-cli)"
    log = tmp_path / "serve.log"
    p = subprocess.Popen([str(GOLEM),"serve",str(tiny_moe_path),"--port","0","--allow-overcommit"], stderr=log.open("w"))
    try:
        deadline = time.monotonic() + 120
        while not (m := re.search(r"listening on 127\.0\.0\.1:(\d+)", log.read_text())):
            assert p.poll() is None, log.read_text()
            assert time.monotonic() < deadline, "server did not start in 120 s"
            time.sleep(0.2)
        port = int(m.group(1))
        c = openai.OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="x")
        r = c.chat.completions.create(model="tiny-moe", messages=[{"role":"user","content":"hi"}], max_tokens=4, temperature=0)
        # random weights: an EOG token may end the completion early; temperature pinned so output is deterministic
        assert r.choices[0].message.content is not None and 0 < r.usage.completion_tokens <= 4
    finally:
        p.terminate()
        assert p.wait(70) == 0, log.read_text()
