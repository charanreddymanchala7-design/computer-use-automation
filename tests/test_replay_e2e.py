"""The thesis, end to end: the model discovers once, the artifact becomes a capability, and replay
runs it for other inputs with no model involved. Real Chromium, the real mock, the real gateway."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.discovery_scripts import discover as discover_capability
from tests.mockbank_support import MockHandle

from cua.artifact import Capability
from cua.artifact.store import load_capability, save_capability
from cua.evlog import EventLog, read_events
from cua.gateway import ActionGateway
from cua.policy import Policy, UrlRule
from cua.redact import Redactor
from cua.replay import ReplayEngine
from cua.result import ReplayResult, Status
from cua.surface import PlaywrightSurface

pytestmark = pytest.mark.browser

SECRETS = {"MOCK_USER": "teller01", "MOCK_PASS": "demo-only"}


def discover(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> tuple[Capability, int]:
    """One discovery run with a scripted model; returns the capability and how many model calls."""
    capability = discover_capability(surface, mock.base, tmp_path)
    # a clean slate for replay: signed out, empty server log
    mock.client().post("/_admin/reset", {})
    surface.reset()
    return capability, 1


def replay(
    surface: PlaywrightSurface,
    mock: MockHandle,
    tmp_path: Path,
    capability: Capability,
    inputs: dict[str, str],
    name: str = "replay",
) -> tuple[ReplayResult, EventLog]:
    policy = Policy(
        allow=(UrlRule(host="127.0.0.1", path_prefix="/msv/"),),
        deny=(UrlRule(host="127.0.0.1", path_prefix="/msv/admin.cgi"),),
    )
    log = EventLog(
        tmp_path / f"{name}.jsonl", run_id=name, redactor=Redactor(secrets=["demo-only"])
    )
    gateway = ActionGateway(surface, policy, log)
    engine = ReplayEngine(
        surface,
        gateway,
        log,
        base_url=mock.base,
        secrets=SECRETS,
        evidence_dir=tmp_path / "evidence",
        run_id=name,
    )
    return engine.run(capability, inputs), log


def test_a_capability_discovered_for_one_member_replays_for_another_with_no_model(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    capability, model_calls = discover(surface, mock, tmp_path)
    assert model_calls > 0  # discovery needed the model; nothing below does

    same, _ = replay(surface, mock, tmp_path, capability, {"member_id": "12345"}, "same")
    assert same.status is Status.SUCCESS, (same.outcome_code, same.failed_step, same.observed)
    assert same.outputs == {"savings_balance": "$2,480.15"}

    mock.client().post("/_admin/reset", {})
    surface.reset()
    other, log = replay(surface, mock, tmp_path, capability, {"member_id": "12347"}, "other")
    assert other.status is Status.SUCCESS, (other.outcome_code, other.failed_step, other.observed)
    assert other.outputs == {"savings_balance": "$50.00"}  # a different member's own balance
    assert other.degraded == []  # every element was found by its first-choice strategy

    steps = [e for e in read_events(log.path) if e["event"] == "step_ok"]
    assert [e["step"] for e in steps] == [f"s{i}" for i in range(1, 9)]


def test_replay_is_repeatable_and_leaves_the_server_untouched(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    capability, _ = discover(surface, mock, tmp_path)
    results = []
    for n in range(2):
        mock.client().post("/_admin/reset", {})
        surface.reset()
        result, _ = replay(surface, mock, tmp_path, capability, {"member_id": "12345"}, f"r{n}")
        results.append(result)
    assert [r.outputs for r in results] == [{"savings_balance": "$2,480.15"}] * 2
    state = mock.server.state
    assert state.confirmations == 0
    assert state.closed_accounts == set()
    paths = [r["path"] for r in state.requests]
    assert "/msv/admin.cgi" not in paths
    assert "/msv/close.cgi" not in paths


def test_a_capability_saved_to_disk_and_loaded_back_replays_identically(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    capability, _ = discover(surface, mock, tmp_path)
    path = save_capability(capability, tmp_path / "capabilities")
    loaded = load_capability(path)
    assert loaded == capability
    result, _ = replay(surface, mock, tmp_path, loaded, {"member_id": "12346"})
    assert result.status is Status.SUCCESS, (
        result.outcome_code,
        result.failed_step,
        result.observed,
    )
    assert result.outputs == {"savings_balance": "$100.00"}


def test_replay_never_leaks_a_secret(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    capability, _ = discover(surface, mock, tmp_path)
    result, log = replay(surface, mock, tmp_path, capability, {"member_id": "12345"})
    everything = result.model_dump_json() + log.path.read_text() + capability.model_dump_json()
    assert "demo-only" not in everything
    assert "MOCK_PASS" in capability.required_secrets()  # names only


def test_a_step_that_no_longer_exists_fails_with_the_step_and_keeps_evidence(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    capability, _ = discover(surface, mock, tmp_path)
    data = capability.model_dump(mode="json")
    data["steps"][4]["locator"]["strategies"] = [
        {
            "kind": "text",
            "text": "A control this app has never had",
            "exact": True,
            "rationale": "simulates the vendor renaming the control",
        }
    ]
    broken = Capability.model_validate(data)
    result, _ = replay(surface, mock, tmp_path, broken, {"member_id": "12345"}, "broken")
    assert (result.status, result.outcome_code, result.failed_step) == (
        Status.HARD_FAILURE,
        "locator_not_found",
        "s5",
    )
    assert (tmp_path / "evidence" / "broken" / "failure.png").read_bytes().startswith(b"\x89PNG")
