"""The command line end to end: a real Chromium and the mock application, driven through the same
commands a person would type. Discovery uses a scripted model (a live one is the evidence run)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, cast

import pytest
from targets.mockbank.server import make_server
from tests.discovery_scripts import discover as discover_capability
from tests.discovery_scripts import lookup_script
from tests.mockbank_support import FakeClock, MockHandle
from typer.testing import CliRunner

from cua import cli
from cua.artifact.store import save_capability
from cua.llm import FakeLLM
from cua.surface import PlaywrightSurface, SurfaceConfig

pytestmark = pytest.mark.browser

runner = CliRunner()
CREDENTIALS = {"MOCK_USER": "teller01", "MOCK_PASS": "demo-only"}


@pytest.fixture(scope="module")
def capabilities(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A capability discovered once for the whole module, against a private mock server."""
    folder = tmp_path_factory.mktemp("capabilities")
    server = make_server(port=0, clock=FakeClock())
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    host, port = cast(tuple[str, int], server.server_address[:2])
    surface = PlaywrightSurface(SurfaceConfig(headless=True), secrets=CREDENTIALS)
    surface.open()
    try:
        capability = discover_capability(surface, f"http://{host}:{port}", folder / "scratch")
    finally:
        surface.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    save_capability(capability, folder)
    return folder


@pytest.fixture(autouse=True)
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in CREDENTIALS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("NO_COLOR", "1")


def replay(
    capabilities: Path,
    mock: MockHandle,
    evidence: Path,
    *extra: str,
    member: str = "12345",
) -> Any:
    return runner.invoke(
        cli.app,
        [
            "replay",
            str(capabilities / "member_lookup.json"),
            "--param",
            f"member_id={member}",
            "--target",
            mock.base,
            "--evidence",
            str(evidence),
            *extra,
        ],
    )


def arm(mock: MockHandle, **fault: object) -> None:
    assert mock.client().request("POST", "/_admin/faults", json_body=fault).status == 200


def test_replay_succeeds_and_reads_like_a_run_log(
    capabilities: Path, mock: MockHandle, tmp_path: Path
) -> None:
    out = replay(capabilities, mock, tmp_path / "ev", member="12346")
    assert out.exit_code == 0, out.output
    lines = out.output.splitlines()
    assert lines[0].startswith("[--] member_lookup@1.0.0 inputs: {'member_id': '***46'}")
    assert "[--] replaying member_lookup@1.0.0 (no model is involved)" in lines
    assert sum(line.startswith("[ok] s") for line in lines) == 8
    assert "[ok] RESULT  success  (exit 0)" in lines
    assert any("savings_balance = $100.00" in line for line in lines)
    assert "\x1b" not in out.output  # NO_COLOR is honoured
    kept = {p.name for p in (tmp_path / "ev").iterdir()}
    assert {"run.jsonl", "result.json", "capability.json"} <= kept


def test_json_mode_prints_only_the_result_document(
    capabilities: Path, mock: MockHandle, tmp_path: Path
) -> None:
    out = replay(capabilities, mock, tmp_path / "ev", "--json", member="12346")
    assert out.exit_code == 0, out.output
    document = json.loads(out.output)
    assert (document["status"], document["outputs"]) == (
        "success",
        {"savings_balance": "$100.00"},
    )


def test_a_business_outcome_exits_10_and_says_what_it_means(
    capabilities: Path, mock: MockHandle, tmp_path: Path
) -> None:
    arm(mock, mode="member_not_found")
    out = replay(capabilities, mock, tmp_path / "ev")
    assert out.exit_code == 10, out.output
    assert "[!!] RESULT  business_outcome  (exit 10)" in out.output
    assert "member_not_found" in out.output


def test_a_hard_failure_exits_30_with_step_expectation_and_evidence(
    capabilities: Path, mock: MockHandle, tmp_path: Path
) -> None:
    arm(mock, mode="app_error", step="results")
    out = replay(capabilities, mock, tmp_path / "ev")
    assert out.exit_code == 30, out.output
    assert "[xx] RESULT  hard_failure  (exit 30)" in out.output
    assert "app_error" in out.output
    assert (tmp_path / "ev" / "failure.png").read_bytes()[:4] == b"\x89PNG"


def test_a_session_that_expires_with_nobody_on_call_exits_20(
    capabilities: Path, mock: MockHandle, tmp_path: Path
) -> None:
    arm(mock, mode="session_timeout", step="results")
    out = replay(capabilities, mock, tmp_path / "ev")
    assert out.exit_code == 20, out.output
    assert "[!!] RESULT  escalated  (exit 20)" in out.output
    assert "session_expired" in out.output


def test_a_missing_secret_is_reported_by_name_and_never_by_value(
    capabilities: Path, mock: MockHandle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MOCK_PASS")
    out = replay(capabilities, mock, tmp_path / "ev")
    assert out.exit_code == 30, out.output
    assert "missing_secret" in out.output
    assert "MOCK_PASS" in out.output
    assert "demo-only" not in out.output


def test_nothing_the_command_writes_contains_a_credential(
    capabilities: Path, mock: MockHandle, tmp_path: Path
) -> None:
    out = replay(capabilities, mock, tmp_path / "ev", member="12346")
    written = "".join(p.read_text(errors="ignore") for p in (tmp_path / "ev").glob("*.json*"))
    assert "demo-only" not in out.output + written


# --- discovery ----------------------------------------------------------------------------------


def test_run_discovers_and_saves_a_capability_with_its_evidence(
    mock: MockHandle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script: list[Any] = lookup_script()
    monkeypatch.setattr(cli, "make_llm", lambda provider, model, log: FakeLLM(script))
    out = runner.invoke(
        cli.app,
        [
            "run",
            "--task",
            "member_lookup",
            "--target",
            mock.base,
            "--evidence",
            str(tmp_path / "ev"),
            "--capabilities",
            str(tmp_path / "caps"),
        ],
    )
    assert out.exit_code == 0, out.output
    assert "[ok] finished:" in out.output
    assert (tmp_path / "caps" / "member_lookup.json").is_file()
    kept = {p.name for p in (tmp_path / "ev").iterdir()}
    assert {"run.jsonl", "summary.json", "review.md", "capability.json", "00-start.png"} <= kept
    summary = json.loads((tmp_path / "ev" / "summary.json").read_text())
    assert (summary["outcome"], summary["model"]) == ("finished", "fake")
    everything = "".join(
        p.read_text(errors="ignore") for p in (tmp_path / "ev").iterdir() if p.suffix != ".png"
    )
    assert "demo-only" not in everything + out.output


def test_run_without_an_api_key_says_where_to_put_it(
    mock: MockHandle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    policy = Path(__file__).resolve().parent.parent / "policies" / "memberserv.json"
    out = runner.invoke(
        cli.app,
        [
            "run",
            "--policy",
            str(policy),
            "--task",
            "member_lookup",
            "--target",
            mock.base,
            "--evidence",
            str(tmp_path / "ev"),
        ],
    )
    assert out.exit_code == 2, out.output
    assert "ANTHROPIC_API_KEY" in out.output
    assert ".env" in out.output


def test_a_run_that_does_not_finish_exits_30_and_saves_nothing(
    mock: MockHandle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cua.llm import say

    monkeypatch.setattr(cli, "make_llm", lambda provider, model, log: FakeLLM([say("hmm")] * 5))
    out = runner.invoke(
        cli.app,
        [
            "run",
            "--task",
            "member_lookup",
            "--target",
            mock.base,
            "--evidence",
            str(tmp_path / "ev"),
            "--capabilities",
            str(tmp_path / "caps"),
        ],
    )
    assert out.exit_code == 30, out.output
    assert "[xx] discovery dead_end" in out.output
    assert not (tmp_path / "caps").exists()


# --- an agent-style call ------------------------------------------------------------------------


def call(capabilities: Path, mock: MockHandle, evidence: Path, args: str) -> Any:
    return runner.invoke(
        cli.app,
        [
            "capabilities",
            "call",
            "member_lookup",
            "--args",
            args,
            "--dir",
            str(capabilities),
            "--target",
            mock.base,
            "--evidence",
            str(evidence),
        ],
    )


def test_an_agent_style_call_returns_a_structured_result_with_no_model(
    capabilities: Path, mock: MockHandle, tmp_path: Path
) -> None:
    out = call(capabilities, mock, tmp_path / "ev", '{"member_id": "12346"}')
    assert out.exit_code == 0, out.output
    reply = json.loads(out.output)
    assert reply["isError"] is False
    assert reply["structuredContent"]["status"] == "success"
    assert reply["structuredContent"]["outputs"] == {"savings_balance": "$100.00"}
    assert json.loads(reply["content"][0]["text"]) == reply["structuredContent"]


def test_an_agent_told_no_such_member_gets_an_answer_not_an_error(
    capabilities: Path, mock: MockHandle, tmp_path: Path
) -> None:
    arm(mock, mode="member_not_found")
    out = call(capabilities, mock, tmp_path / "ev", '{"member_id": "12345"}')
    assert out.exit_code == 10, out.output
    reply = json.loads(out.output)
    assert reply["isError"] is False
    assert reply["structuredContent"]["outcome_code"] == "member_not_found"
