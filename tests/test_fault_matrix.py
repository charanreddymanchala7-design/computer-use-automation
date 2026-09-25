"""The fault matrix: every runtime condition the mock can produce, replayed through the real
engine in real Chromium, asserting what the caller is told and what the server actually did.

The server's own log is the ground truth: a write must happen exactly once (or not at all),
never be silently retried, and nothing outside the allowlist may ever be requested.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from targets.mockbank.server import make_server
from tests.discovery_scripts import discover, policy
from tests.mockbank_support import MockHandle

from cua.artifact import Capability
from cua.evlog import EventLog
from cua.gateway import ActionGateway
from cua.redact import Redactor
from cua.replay import ReplayEngine
from cua.result import ReplayResult, Status
from cua.surface import PlaywrightSurface

pytestmark = pytest.mark.browser

SECRETS = {"MOCK_USER": "teller01", "MOCK_PASS": "demo-only"}


@pytest.fixture(scope="module")
def capabilities(
    browser_surface: PlaywrightSurface, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[dict[str, Capability]]:
    """Each capability is discovered once (against its own server) and replayed many times."""
    server = make_server(port=0)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    found: dict[str, Capability] = {}
    tmp = tmp_path_factory.mktemp("discovery")
    for which in ("lookup", "open"):
        browser_surface.reset()
        server.state.reset()
        found[which] = discover(browser_surface, base, tmp, which)
    yield found
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@dataclass(frozen=True)
class Case:
    name: str
    capability: str
    inputs: dict[str, str]
    status: Status
    code: str
    exit_code: int
    faults: tuple[dict[str, Any], ...] = ()
    recoveries: tuple[str, ...] = ()
    failed_step: str | None = None
    effects: tuple[str, ...] = ()  # what the server says changed
    submits: int = 0  # POSTs of the write step the server received
    outputs: dict[str, str] | None = None
    evidence: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


LOOKUP = {"member_id": "12345"}
OPEN = {"member_id": "12345", "deposit": "25.00"}

CASES = [
    Case(
        "lookup: happy path",
        "lookup",
        LOOKUP,
        Status.SUCCESS,
        "completed",
        0,
        outputs={"savings_balance": "$2,480.15"},
    ),
    Case(
        "lookup: a different member",
        "lookup",
        {"member_id": "12347"},
        Status.SUCCESS,
        "completed",
        0,
        outputs={"savings_balance": "$50.00"},
    ),
    Case(
        "open: happy path",
        "open",
        OPEN,
        Status.SUCCESS,
        "completed",
        0,
        effects=("subaccount_created",),
        submits=1,
        outputs={"confirmation_ref": "CNF-000001"},
    ),
    # business outcomes: legitimate answers, never crashes
    Case(
        "member_not_found (natural)",
        "lookup",
        {"member_id": "99999"},
        Status.BUSINESS_OUTCOME,
        "member_not_found",
        10,
    ),
    Case(
        "member_not_found (fault)",
        "lookup",
        LOOKUP,
        Status.BUSINESS_OUTCOME,
        "member_not_found",
        10,
        faults=({"mode": "member_not_found"},),
    ),
    Case(
        "validation_error (natural: deposit below minimum)",
        "open",
        {**OPEN, "deposit": "1.00"},
        Status.BUSINESS_OUTCOME,
        "validation_rejected",
        10,
        submits=1,
    ),
    Case(
        "validation_error (fault)",
        "open",
        OPEN,
        Status.BUSINESS_OUTCOME,
        "validation_rejected",
        10,
        faults=({"mode": "validation_error"},),
        submits=1,
    ),
    # recoverable: handled, reported, and the run still succeeds
    Case(
        "interstitial_known",
        "lookup",
        LOOKUP,
        Status.SUCCESS,
        "completed",
        0,
        faults=({"mode": "interstitial_known"},),
        recoveries=("eod_notice",),
        outputs={"savings_balance": "$2,480.15"},
    ),
    Case(
        "slow_load",
        "lookup",
        LOOKUP,
        Status.SUCCESS,
        "completed",
        0,
        faults=({"mode": "slow_load", "params": {"delay_ms": 1800}},),
        recoveries=("slow_response",),
        outputs={"savings_balance": "$2,480.15"},
    ),
    # hard failures and escalations: stop, say why, keep evidence
    Case(
        "session_timeout",
        "lookup",
        LOOKUP,
        Status.ESCALATED,
        "session_expired",
        20,
        faults=({"mode": "session_timeout", "step": "member"},),
        evidence=True,
    ),
    Case(
        "app_error (read)",
        "lookup",
        LOOKUP,
        Status.HARD_FAILURE,
        "app_error",
        30,
        faults=({"mode": "app_error", "step": "results"},),
        failed_step="s7",
        evidence=True,
    ),
    Case(
        "app_error after the write was committed",
        "open",
        OPEN,
        Status.HARD_FAILURE,
        "app_error",
        30,
        faults=({"mode": "app_error", "params": {"commit": True}},),
        failed_step="s10",
        effects=("subaccount_created",),
        submits=1,
        evidence=True,
    ),
    Case(
        "app_error before the write was committed",
        "open",
        OPEN,
        Status.HARD_FAILURE,
        "app_error",
        30,
        faults=({"mode": "app_error"},),
        failed_step="s10",
        submits=1,
        evidence=True,
    ),
]


def run_case(
    case: Case,
    capabilities: dict[str, Capability],
    surface: PlaywrightSurface,
    mock: MockHandle,
    tmp_path: Path,
) -> tuple[ReplayResult, dict[str, Any]]:
    for fault in case.faults:
        assert mock.client().request("POST", "/_admin/faults", json_body=fault).status == 200
    log = EventLog(tmp_path / "run.jsonl", run_id="run_m", redactor=Redactor(secrets=["demo-only"]))
    engine = ReplayEngine(
        surface,
        ActionGateway(surface, policy(), log),
        log,
        base_url=mock.base,
        secrets=SECRETS,
        evidence_dir=tmp_path / "evidence",
        run_id="run_m",
    )
    result = engine.run(capabilities[case.capability], case.inputs)
    truth: dict[str, Any] = json.loads(mock.client().get("/_admin/log").text)
    return result, truth


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_every_condition_is_told_to_the_caller_and_the_server_agrees(
    case: Case,
    capabilities: dict[str, Capability],
    surface: PlaywrightSurface,
    mock: MockHandle,
    tmp_path: Path,
) -> None:
    result, truth = run_case(case, capabilities, surface, mock, tmp_path)

    # what the caller is told
    described = (result.status, result.outcome_code, result.failed_step, result.observed)
    assert (result.status, result.outcome_code, result.exit_code) == (
        case.status,
        case.code,
        case.exit_code,
    ), described
    assert [r.rule_id for r in result.recoveries] == list(case.recoveries)
    if case.outputs is not None:
        assert result.outputs == case.outputs
    else:
        assert result.outputs is None
    if case.status is Status.HARD_FAILURE:
        assert result.failed_step == case.failed_step
        assert result.expected
        assert result.observed
    if case.status is Status.ESCALATED:
        assert result.escalation is not None
        assert result.escalation.request_id.startswith("ir_")
    evidence = tmp_path / "evidence" / "run_m"
    assert (evidence / "failure.png").exists() is case.evidence
    assert (evidence / "page.json").exists() is case.evidence

    # what the server says happened: the ground truth
    effects = [e["kind"] for e in truth["effects"]]
    assert effects == list(case.effects)
    submits = [
        r for r in truth["requests"] if r["step"] == "newsub_submit" and r["method"] == "POST"
    ]
    assert len(submits) == case.submits  # a write is never silently retried
    paths = [r["path"] for r in truth["requests"]]
    assert "/msv/admin.cgi" not in paths
    assert "/msv/close.cgi" not in paths


def test_a_committed_write_that_errored_says_so_to_the_caller(
    capabilities: dict[str, Capability],
    surface: PlaywrightSurface,
    mock: MockHandle,
    tmp_path: Path,
) -> None:
    case = next(c for c in CASES if c.name == "app_error after the write was committed")
    result, _ = run_case(case, capabilities, surface, mock, tmp_path)
    assert result.message is not None
    assert "check whether the change was applied" in result.message
    assert "APPLICATION ERROR" in (result.observed or "")


def test_a_fault_that_fires_once_does_not_poison_the_next_run(
    capabilities: dict[str, Capability],
    surface: PlaywrightSurface,
    mock: MockHandle,
    tmp_path: Path,
) -> None:
    first = next(c for c in CASES if c.name == "member_not_found (fault)")
    result, _ = run_case(first, capabilities, surface, mock, tmp_path / "one")
    assert result.status is Status.BUSINESS_OUTCOME
    mock.client().post("/_admin/reset", {})
    surface.reset()
    again = next(c for c in CASES if c.name == "lookup: happy path")
    result, _ = run_case(again, capabilities, surface, mock, tmp_path / "two")
    assert result.status is Status.SUCCESS
