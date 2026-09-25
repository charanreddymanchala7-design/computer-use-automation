"""The replay engine: the production path. No model is involved; everything is a locator bundle,
a typed input, an expectation and a checkpoint, and every failure says what step, what was
expected and what was seen. Logic tests against a scripted page with a controllable clock."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from tests.agent_support import BASE
from tests.replay_support import Clock, FakeReplaySurface, make_gateway

from cua.artifact import Capability
from cua.evlog import EventLog, read_events
from cua.replay import ReplayEngine, ReplayLimits
from cua.result import DegradedLocator, ReplayResult, Status
from cua.surface import Action

INPUTS = {"member_id": "12345"}


@dataclass
class Ran:
    result: ReplayResult
    surface: FakeReplaySurface
    log: EventLog
    clock: Clock
    tmp: Path

    def events(self, name: str) -> list[dict[str, Any]]:
        return [e for e in read_events(self.log.path) if e["event"] == name]


def go(
    tmp_path: Path,
    capability_dict: dict[str, Any],
    *,
    inputs: dict[str, Any] | None = None,
    secrets: dict[str, str] | None = None,
    limits: ReplayLimits | None = None,
    evidence: bool = False,
    configure: Callable[[FakeReplaySurface], None] | None = None,
    mutate: Callable[[dict[str, Any]], None] | None = None,
    base_url: str = BASE,
) -> Ran:
    if mutate:
        mutate(capability_dict)
    capability = Capability.model_validate(capability_dict)
    clock = Clock()
    surface = FakeReplaySurface(clock)
    surface.texts = {"Search results", "Savings"}
    surface.url = f"{BASE}/members/12345"
    surface.reads = {"savings balance": "$2,480.15"}
    if configure:
        configure(surface)
    gateway, log = make_gateway(surface, tmp_path)
    engine = ReplayEngine(
        surface,
        gateway,
        log,
        base_url=base_url,
        secrets=secrets,
        evidence_dir=tmp_path / "evidence" if evidence else None,
        limits=limits or ReplayLimits(step_timeout_s=2.0, run_timeout_s=60.0),
        clock=clock,
        run_id="run_r",
    )
    result = engine.run(capability, INPUTS if inputs is None else inputs)
    return Ran(result, surface, log, clock, tmp_path)


# --- the happy path ----------------------------------------------------------------------------


def test_a_capability_replays_to_its_outputs_without_a_model(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    ran = go(tmp_path, capability_dict)
    assert ran.result.status is Status.SUCCESS
    assert ran.result.outcome_code == "completed"
    assert ran.result.outputs == {"savings_balance": "$2,480.15"}
    assert ran.result.exit_code == 0
    assert ran.result.evidence.screenshot is None  # nothing went wrong, nothing to keep
    assert [a.kind for a in ran.surface.acts] == ["navigate", "fill", "click"]
    assert ran.surface.acts[1].text == "12345"  # the caller's value, typed into the field


def test_the_same_inputs_give_the_same_result_every_time(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    first = go(tmp_path / "a", json.loads(json.dumps(capability_dict)))
    second = go(tmp_path / "b", json.loads(json.dumps(capability_dict)))
    drop = {"duration_ms"}
    assert first.result.model_dump(exclude=drop) == second.result.model_dump(exclude=drop)


def test_locators_are_resolved_with_the_callers_values_for_their_placeholders(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    ran = go(tmp_path, capability_dict)
    assert all(values == INPUTS for values in ran.surface.resolved_values)


# --- inputs and secrets are checked before anything is touched ---------------------------------


@pytest.mark.parametrize(
    ("inputs", "observed"),
    [
        ({}, "missing member_id"),
        ({"member_id": "abc"}, "member_id: failed pattern"),
        ({"member_id": 12345}, "member_id: failed type"),
        ({"member_id": "12345", "extra": 1}, "unexpected property extra"),
    ],
)
def test_bad_inputs_are_refused_before_the_app_is_touched_and_never_echoed(
    tmp_path: Path, capability_dict: dict[str, Any], inputs: dict[str, Any], observed: str
) -> None:
    ran = go(tmp_path, capability_dict, inputs=inputs)
    result = ran.result
    assert (result.status, result.outcome_code, result.failed_step) == (
        Status.HARD_FAILURE,
        "invalid_input",
        "inputs",
    )
    assert result.observed == observed
    assert ran.surface.acts == []
    assert "abc" not in result.model_dump_json()  # the offending value is not repeated


def secret_capability(capability_dict: dict[str, Any]) -> None:
    step = next(s for s in capability_dict["steps"] if s["id"] == "s2")
    step["value"] = {"source": "secret", "name": "BANK_PASSWORD"}
    step["sensitive"] = True


def test_a_missing_secret_is_named_and_stops_the_run_at_the_door(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    ran = go(tmp_path, capability_dict, mutate=secret_capability)
    assert ran.result.outcome_code == "missing_secret"
    assert ran.result.observed == "missing: BANK_PASSWORD"
    assert ran.surface.acts == []


def test_a_secret_is_typed_by_name_and_never_reaches_the_log(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    ran = go(
        tmp_path, capability_dict, mutate=secret_capability, secrets={"BANK_PASSWORD": "s3cr3t-pw"}
    )
    assert ran.result.status is Status.SUCCESS
    assert ran.surface.secrets == {"BANK_PASSWORD": "s3cr3t-pw"}  # made available to the surface
    assert ran.surface.acts[1].secret == "BANK_PASSWORD"
    assert ran.surface.acts[1].text is None
    assert "s3cr3t-pw" not in ran.log.path.read_text()


def test_a_sensitive_input_is_typed_as_a_secret_and_never_logged(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def sensitive(cap: dict[str, Any]) -> None:
        cap["inputs"]["properties"]["member_id"]["x-sensitive"] = True

    ran = go(tmp_path, capability_dict, mutate=sensitive)
    assert ran.surface.secrets == {"param:member_id": "12345"}
    assert ran.surface.acts[1].secret == "param:member_id"
    assert "12345" not in ran.log.path.read_text()
    assert ran.result.status is Status.SUCCESS


# --- navigation --------------------------------------------------------------------------------


@pytest.mark.parametrize("base", [BASE, BASE + "/"])
def test_a_portable_path_is_joined_to_the_tenants_base_url(
    tmp_path: Path, capability_dict: dict[str, Any], base: str
) -> None:
    def portable(cap: dict[str, Any]) -> None:
        cap["steps"][0]["url_template"] = "/msv/login.cgi?mid={member_id}"

    ran = go(tmp_path, capability_dict, mutate=portable, base_url=base)
    assert ran.surface.acts[0] == Action.navigate(f"{BASE}/msv/login.cgi?mid=12345")


def test_values_are_url_encoded_where_they_are_substituted_into_a_url(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def free_text(cap: dict[str, Any]) -> None:
        cap["inputs"]["properties"]["member_id"] = {"type": "string"}
        cap["steps"][0]["url_template"] = "/msv/find.cgi?q={member_id}"

    ran = go(tmp_path, capability_dict, mutate=free_text, inputs={"member_id": "a b/c&d"})
    assert ran.surface.acts[0] == Action.navigate(f"{BASE}/msv/find.cgi?q=a%20b%2Fc%26d")


def test_navigating_off_the_allowlist_is_a_policy_failure(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def elsewhere(cap: dict[str, Any]) -> None:
        cap["steps"][0]["url_template"] = "http://evil.example/x"

    ran = go(tmp_path, capability_dict, mutate=elsewhere)
    assert ran.result.outcome_code == "blocked_by_policy"
    assert ran.result.failed_step == "s1"
    assert "host_not_allowed" in (ran.result.observed or "")
    assert ran.surface.acts == []


# --- finding elements --------------------------------------------------------------------------


def test_an_element_that_arrives_late_is_waited_for_not_slept_for(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def late(surface: FakeReplaySurface) -> None:
        surface.missing = {"member number field": 2}

    ran = go(tmp_path, capability_dict, configure=late)
    assert ran.result.status is Status.SUCCESS
    assert ran.surface.pauses == [100, 100]  # polled twice, then it was there


def test_an_element_that_never_arrives_is_a_clear_hard_failure(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def gone(surface: FakeReplaySurface) -> None:
        surface.never = {"member number field"}

    ran = go(tmp_path, capability_dict, configure=gone)
    result = ran.result
    assert (result.status, result.outcome_code, result.failed_step) == (
        Status.HARD_FAILURE,
        "locator_not_found",
        "s2",
    )
    assert "member number field" in (result.expected or "")
    assert "label: no_match" in (result.observed or "")
    assert ran.clock.now >= 2.0  # it waited out the step timeout before giving up
    assert [a.kind for a in ran.surface.acts] == ["navigate"]


def test_a_fallback_strategy_matching_is_reported_as_drift_but_the_run_succeeds(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def fallback(surface: FakeReplaySurface) -> None:
        surface.strategy = {"member number field": 1}

    ran = go(tmp_path, capability_dict, configure=fallback)
    assert ran.result.status is Status.SUCCESS
    assert ran.result.degraded == [
        DegradedLocator(step_id="s2", strategy_index=1, strategy_kind="attribute_fingerprint")
    ]
    assert ran.events("step_ok")[1]["strategy"] == "attribute_fingerprint"


def test_drift_is_still_reported_when_the_run_later_fails(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def fallback_then_missing_text(surface: FakeReplaySurface) -> None:
        surface.strategy = {"member number field": 1}
        surface.texts = set()

    ran = go(tmp_path, capability_dict, configure=fallback_then_missing_text)
    assert ran.result.status is Status.HARD_FAILURE
    assert [d.step_id for d in ran.result.degraded] == ["s2"]


def test_a_coordinate_strategy_clicks_the_point(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def blind(surface: FakeReplaySurface) -> None:
        surface.coordinates = {"search button": (12.0, 34.0)}

    ran = go(tmp_path, capability_dict, configure=blind)
    assert ran.surface.acts[2] == Action.click_at(12.0, 34.0)


# --- actions -----------------------------------------------------------------------------------


def test_press_and_select_steps_are_replayed_with_their_declared_values(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def more(cap: dict[str, Any]) -> None:
        bundle = cap["steps"][1]["locator"]
        cap["steps"] += [
            {
                "id": "s5",
                "action": "select",
                "description": "pick",
                "locator": bundle,
                "value": {"source": "literal", "value": "Holiday Club"},
                "risk_class": "read",
            },
            {
                "id": "s6",
                "action": "press",
                "description": "submit",
                "risk_class": "reversible_write",
                "value": {"source": "literal", "value": "Enter"},
                "expect": {"text_present": ["Search results"]},
            },
        ]

    ran = go(tmp_path, capability_dict, mutate=more)
    assert ran.result.status is Status.SUCCESS
    assert [a.kind for a in ran.surface.acts][-2:] == ["select", "press"]
    assert ran.surface.acts[-2].option == "Holiday Club"
    assert ran.surface.acts[-1].key == "Enter"


def test_a_failed_action_is_a_hard_failure_with_the_reason(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def broken(surface: FakeReplaySurface) -> None:
        surface.ok = False
        surface.error = "Timeout 8000ms exceeded"

    ran = go(tmp_path, capability_dict, configure=broken)
    assert (ran.result.outcome_code, ran.result.failed_step) == ("action_failed", "s1")
    assert ran.result.observed == "Timeout 8000ms exceeded"


def test_an_irreversible_step_escalates_instead_of_running_unattended(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def close(cap: dict[str, Any]) -> None:
        step = next(s for s in cap["steps"] if s["id"] == "s3")
        step["locator"]["description"] = "link 'Close'"
        step["risk_class"] = "irreversible_write"

    ran = go(tmp_path, capability_dict, mutate=close)
    result = ran.result
    assert (result.status, result.outcome_code, result.exit_code) == (
        Status.ESCALATED,
        "confirmation_required",
        20,
    )
    assert result.escalation is not None
    assert result.escalation.request_id.startswith("cf_")
    assert result.escalation.step_id == "s3"
    assert "click" not in [a.kind for a in ran.surface.acts]  # it was never clicked


# --- dialogs -----------------------------------------------------------------------------------


def declares_confirm(cap: dict[str, Any]) -> None:
    step = next(s for s in cap["steps"] if s["id"] == "s3")
    step["dialogs"] = [{"kind": "confirm", "message": "Search member {member_id}?"}]


def test_a_declared_dialog_is_accepted_with_its_placeholders_filled(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def raises(surface: FakeReplaySurface) -> None:
        surface.provoke = {"click": [("confirm", "Search member 12345?")]}

    ran = go(tmp_path, capability_dict, mutate=declares_confirm, configure=raises)
    assert ran.result.status is Status.SUCCESS


def test_a_dialog_nobody_declared_is_dismissed_and_stops_the_run(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def raises(surface: FakeReplaySurface) -> None:
        surface.provoke = {"click": [("confirm", "Branch closing early. Continue?")]}

    ran = go(tmp_path, capability_dict, mutate=declares_confirm, configure=raises)
    result = ran.result
    assert (result.outcome_code, result.failed_step) == ("unexpected_dialog", "s3")
    assert "Branch closing early" in (result.observed or "")
    assert ran.surface.policy is not None
    assert ran.surface.policy("confirm", "Branch closing early. Continue?") is False


def test_a_dialog_declared_as_dismissed_is_dismissed(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def dismiss(cap: dict[str, Any]) -> None:
        step = next(s for s in cap["steps"] if s["id"] == "s3")
        step["dialogs"] = [{"kind": "confirm", "message": "Discard?", "action": "dismiss"}]

    def raises(surface: FakeReplaySurface) -> None:
        surface.provoke = {"click": [("confirm", "Discard?")]}

    ran = go(tmp_path, capability_dict, mutate=dismiss, configure=raises)
    assert ran.result.status is Status.SUCCESS
    assert ran.surface.acts[2].kind == "click"


# --- expectations, waiting and the checkpoint --------------------------------------------------


def test_an_expectation_that_arrives_after_a_few_polls_is_auto_waited_for(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def slow(surface: FakeReplaySurface) -> None:
        surface.texts = {"Savings"}
        surface.appear_after = {"Search results": 3}

    ran = go(tmp_path, capability_dict, configure=slow)
    assert ran.result.status is Status.SUCCESS
    assert len(ran.surface.pauses) == 3


def test_a_missing_expectation_names_the_step_what_was_expected_and_what_was_seen(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def wrong_page(surface: FakeReplaySurface) -> None:
        surface.texts = {"APPLICATION ERROR", "Savings"}

    ran = go(tmp_path, capability_dict, configure=wrong_page)
    result = ran.result
    assert (result.outcome_code, result.failed_step) == ("expectation_failed", "s3")
    assert "Search results" in (result.expected or "")
    assert "APPLICATION ERROR" in (result.observed or "")


def test_text_that_must_be_absent_and_a_url_pattern_are_checked_too(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def strict(cap: dict[str, Any]) -> None:
        cap["steps"][2]["expect"] = {"text_absent": ["ERROR"], "url_pattern": "/members/:id"}

    ok = go(tmp_path / "ok", json.loads(json.dumps(capability_dict)), mutate=strict)
    assert ok.result.status is Status.SUCCESS

    def error_shown(surface: FakeReplaySurface) -> None:
        surface.texts = {"ERROR", "Savings"}

    bad = go(
        tmp_path / "bad",
        json.loads(json.dumps(capability_dict)),
        mutate=strict,
        configure=error_shown,
    )
    assert bad.result.outcome_code == "expectation_failed"

    def elsewhere(surface: FakeReplaySurface) -> None:
        surface.url = f"{BASE}/login"

    off = go(
        tmp_path / "off",
        json.loads(json.dumps(capability_dict)),
        mutate=strict,
        configure=elsewhere,
    )
    assert off.result.outcome_code == "expectation_failed"
    assert "/members/:id" in (off.result.expected or "")


def test_wait_for_steps_wait_for_their_text(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def add_wait(cap: dict[str, Any]) -> None:
        cap["steps"].insert(
            3,
            {
                "id": "s3b",
                "action": "wait_for",
                "description": "the result",
                "risk_class": "read",
                "expect": {"text_present": ["Loaded"], "timeout_ms": 5000},
            },
        )

    def slow(surface: FakeReplaySurface) -> None:
        surface.appear_after = {"Loaded": 2}

    ran = go(tmp_path, capability_dict, mutate=add_wait, configure=slow)
    assert ran.result.status is Status.SUCCESS
    assert len(ran.surface.pauses) == 2


def test_the_checkpoint_must_hold_for_the_run_to_count_as_a_success(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def missing(surface: FakeReplaySurface) -> None:
        surface.texts = {"Search results"}  # no "Savings"

    ran = go(tmp_path, capability_dict, configure=missing)
    result = ran.result
    assert (result.outcome_code, result.failed_step) == ("checkpoint_failed", "checkpoint")
    assert "Savings" in (result.expected or "")
    assert result.outputs is None  # values read on the way do not count without the checkpoint


def test_the_checkpoint_url_pattern_is_checked(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def elsewhere(surface: FakeReplaySurface) -> None:
        surface.url = f"{BASE}/somewhere/else"

    ran = go(tmp_path, capability_dict, configure=elsewhere)
    assert ran.result.outcome_code == "checkpoint_failed"
    assert "/members/:id" in (ran.result.expected or "")


# --- outputs -----------------------------------------------------------------------------------


def test_an_output_that_breaks_its_schema_is_refused_without_repeating_it(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def constrained(cap: dict[str, Any]) -> None:
        cap["outputs"]["properties"]["savings_balance"]["pattern"] = r"^\$[0-9.,]+$"

    def odd(surface: FakeReplaySurface) -> None:
        surface.reads = {"savings balance": "call the branch"}

    ran = go(tmp_path, capability_dict, mutate=constrained, configure=odd)
    assert ran.result.outcome_code == "output_invalid"
    assert ran.result.observed == "savings_balance: failed pattern"
    assert "call the branch" not in ran.result.model_dump_json()


# --- limits ------------------------------------------------------------------------------------


def test_a_run_that_takes_too_long_is_stopped(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def slow(surface: FakeReplaySurface) -> None:
        surface.on_act = lambda action: surface.clock.advance(100)

    ran = go(tmp_path, capability_dict, configure=slow, limits=ReplayLimits(run_timeout_s=30.0))
    assert (ran.result.outcome_code, ran.result.failed_step) == ("timeout", "s2")


# --- evidence ----------------------------------------------------------------------------------


def test_a_failure_keeps_a_screenshot_and_a_redacted_page_snapshot(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def wrong_page(surface: FakeReplaySurface) -> None:
        surface.texts = {"APPLICATION ERROR", "password=hunter2 was rejected"}

    ran = go(tmp_path, capability_dict, evidence=True, configure=wrong_page)
    evidence = ran.result.evidence
    assert (evidence.screenshot, evidence.aria_snapshot, evidence.log) == (
        "failure.png",
        "page.json",
        "run.jsonl",
    )
    folder = tmp_path / "evidence" / "run_r"
    assert (folder / "failure.png").read_bytes().startswith(b"\x89PNG")
    snapshot = (folder / "page.json").read_text()
    assert "APPLICATION ERROR" in snapshot
    assert "hunter2" not in snapshot


def test_a_success_writes_no_evidence_folder(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    go(tmp_path, capability_dict, evidence=True)
    assert not (tmp_path / "evidence").exists()


def test_what_a_failure_reports_as_observed_is_redacted(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def wrong_page(surface: FakeReplaySurface) -> None:
        surface.texts = {"ssn 123-45-6789 on file"}

    ran = go(tmp_path, capability_dict, configure=wrong_page)
    assert "123-45-6789" not in (ran.result.observed or "")


# --- the audit trail ---------------------------------------------------------------------------


def test_every_step_is_logged_with_the_strategy_that_found_it(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    ran = go(tmp_path, capability_dict)
    assert [e["step"] for e in ran.events("step_start")] == ["s1", "s2", "s3", "s4"]
    oks = ran.events("step_ok")
    assert [e["step"] for e in oks] == ["s1", "s2", "s3", "s4"]
    assert oks[1]["strategy"] == "label"
    assert ran.events("replay_start")[0]["capability"] == "member_lookup@1.0.0"
    assert ran.events("replay_end")[0]["status"] == "success"
