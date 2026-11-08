"""scripts/ab-report.py: windowed tok/s, the timing mean columns and the
prefill trailer, over a synthetic ab-bench leg."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ab-report.py"
HEADER = (
    "step,tokens,wall_ms,p3_wait_ms,disk_wait_ms,disk_worker_ms,disk_bytes,h2d_ms,h2d_bytes,"
    "policy_ms,compute_ms,cpu_expert_math_experts,cpu_expert_math_bytes,cpu_expert_math_ms,"
    "cpu_expert_join_wait_ms,ring_d2h_ms,ring_d2h_bytes,ring_units,ring_full,"
    "ids_readbacks,ids_sync_ms,ids_gap_ms,ids_gap_max_ms"
)


def load():
    spec = importlib.util.spec_from_file_location("ab_report", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def work(tmp_path):
    """One leg `a`: a 256-token run (255 rows, steps 0..254), wall 100 ms and
    1 token per row; `disk_wait_ms` measured on even steps only, `h2d_ms`
    never measured."""
    leg = tmp_path / "a"
    leg.mkdir()
    rows = []
    for s in range(255):
        disk_wait = "2.0" if s % 2 == 0 else "unmeasured"
        rows.append(
            f"{s},1,100.0,10.0,{disk_wait},1.0,64,unmeasured,unmeasured,unmeasured,90.0,"
            "unmeasured,unmeasured,unmeasured,unmeasured,unmeasured,unmeasured,unmeasured,"
            "unmeasured,unmeasured,unmeasured,unmeasured,unmeasured"
        )
    (leg / "timing.csv").write_text(
        "\n".join(["# engine_version=0 fork_commit=x", HEADER, *rows,
                   "# prefill_ms=250.0 prompt_tokens=500"]) + "\n"
    )
    (leg / "meta.txt").write_text("leg=a\nrc=0\n")
    return tmp_path


def report(capsys, *argv):
    load().main(list(argv))
    out = capsys.readouterr().out.splitlines()
    head = [c.strip() for c in out[0].strip("|").split("|")]
    row = [c.strip() for c in out[2].strip("|").split("|")]
    return dict(zip(head, row))


def test_window_selects_steps_and_sums_tokens(work, capsys):
    r = report(capsys, "--window", "129-256", str(work), "a")
    assert r["steps"] == "128"
    assert r["tok/s"] == "10.000"


def test_window_past_last_row_exits_nonzero(work):
    with pytest.raises(SystemExit) as e:
        load().main(["--window", "2-300", str(work), "a"])
    assert e.value.code not in (0, None)


def test_window_first_below_two_is_rejected(work):
    with pytest.raises(SystemExit) as e:
        load().main(["--window", "1-10", str(work), "a"])
    assert e.value.code not in (0, None)


def test_prefill_trailer_columns(work, capsys):
    r = report(capsys, str(work), "a")
    assert r["prefill ms"] == "250.0"
    assert r["prompt tokens"] == "500"
    assert r["prefill tok/s"] == "2000.0"


def test_mean_columns_mark_partial_and_unmeasured(work, capsys):
    r = report(capsys, str(work), "a")
    assert r["p3_wait_ms"] == "10.000"
    assert r["compute_ms"] == "90.000"
    assert r["disk_wait_ms"] == "2.000 (n=128)"
    assert r["h2d_ms"] == "-"


def test_missing_trailer_renders_dashes(work, capsys):
    csv = work / "a" / "timing.csv"
    csv.write_text("\n".join(l for l in csv.read_text().splitlines()
                             if not l.startswith("# prefill")) + "\n")
    r = report(capsys, str(work), "a")
    assert (r["prefill ms"], r["prompt tokens"], r["prefill tok/s"]) == ("-", "-", "-")
