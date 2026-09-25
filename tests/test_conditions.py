"""Runtime conditions: the failures that are not layout drift. A replay must recognise what the
application is telling it and respond deliberately: a business outcome is an answer, a known
interstitial is dismissed within a budget, a slow page is waited for and reported, and anything
else stops with a clear, evidenced failure or brings in a human."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from tests.agent_support import BASE
from tests.replay_support import FakeReplaySurface
from tests.test_replay import Ran, go

from cua.result import RecoveryRecord, Status

# The shared capability declares three rules: no_such_member (business outcome: "No member
# found"), welcome_interstitial (recoverable: dismiss "Notice to all staff" with a Continue
# button, two attempts) and app_crashed (hard failure app_error: "Internal Server Error").


def shows(*texts: str) -> Callable[[FakeReplaySurface], None]:
    def configure(surface: FakeReplaySurface) -> None:
        surface.texts = set(texts)

    return configure


def clone(data: dict[str, Any]) -> dict[str, Any]:
    copy: dict[str, Any] = json.loads(json.dumps(data))
    return copy


# --- a business outcome is an answer, not a crash ----------------------------------------------


def test_no_such_member_is_a_business_outcome_the_caller_can_branch_on(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    ran = go(tmp_path, capability_dict, configure=shows("No member found"), evidence=True)
    result = ran.result
    assert (result.status, result.outcome_code, result.exit_code) == (
        Status.BUSINESS_OUTCOME,
        "member_not_found",
        10,
    )
    assert result.message == "No member matches the supplied number"
    assert result.outputs is None
    assert not (tmp_path / "evidence").exists()  # nothing went wrong, so nothing to debug


def test_a_business_outcome_is_recognised_when_the_element_never_shows_up(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def no_rows(surface: FakeReplaySurface) -> None:
        surface.texts = {"No member found", "Savings"}
        surface.never = {"savings balance"}

    ran = go(tmp_path, capability_dict, configure=no_rows)
    assert ran.result.outcome_code == "member_not_found"


def test_a_business_outcome_is_recognised_even_at_the_checkpoint(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def no_savings(surface: FakeReplaySurface) -> None:
        surface.texts = {"Search results", "No member found"}

    ran = go(tmp_path, capability_dict, configure=no_savings)
    assert ran.result.status is Status.BUSINESS_OUTCOME


# --- hard failures -----------------------------------------------------------------------------


def test_an_application_error_is_a_hard_failure_with_step_expectation_and_evidence(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    ran = go(tmp_path, capability_dict, configure=shows("Internal Server Error"), evidence=True)
    result = ran.result
    assert (result.status, result.outcome_code, result.exit_code) == (
        Status.HARD_FAILURE,
        "app_error",
        30,
    )
    assert result.failed_step == "s3"
    assert "Search results" in (result.expected or "")
    assert "condition 'app_crashed' detected" in (result.observed or "")
    assert "Internal Server Error" in (result.observed or "")
    assert (tmp_path / "evidence" / "run_r" / "failure.png").exists()
    assert (tmp_path / "evidence" / "run_r" / "page.json").exists()


def test_a_hard_failure_that_asks_for_a_human_becomes_an_escalation(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def expires(cap: dict[str, Any]) -> None:
        rule = cap["error_map"][2]
        rule.update({"id": "session_expired", "code": "session_expired", "escalate": True})
        rule["detect"] = {"text_present": ["User ID"]}

    ran = go(tmp_path, capability_dict, mutate=expires, configure=shows("User ID"), evidence=True)
    result = ran.result
    assert (result.status, result.outcome_code, result.exit_code) == (
        Status.ESCALATED,
        "session_expired",
        20,
    )
    assert result.escalation is not None
    assert result.escalation.step_id == "s3"
    assert result.escalation.request_id.startswith("ir_")
    assert (tmp_path / "evidence" / "run_r" / "failure.png").exists()  # the human gets context


def test_the_first_matching_rule_wins(tmp_path: Path, capability_dict: dict[str, Any]) -> None:
    ran = go(
        tmp_path,
        capability_dict,
        configure=shows("No member found", "Internal Server Error"),
    )
    assert ran.result.outcome_code == "member_not_found"  # declared first


def test_a_rule_can_be_detected_by_url_or_by_an_element(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def by_url(cap: dict[str, Any]) -> None:
        cap["error_map"][2]["detect"] = {"url_pattern": "/login"}

    def at_login(surface: FakeReplaySurface) -> None:
        surface.texts = {"whatever"}
        surface.url = f"{BASE}/login"

    ran = go(tmp_path / "url", clone(capability_dict), mutate=by_url, configure=at_login)
    assert ran.result.outcome_code == "app_error"

    def by_element(cap: dict[str, Any]) -> None:
        bundle = cap["steps"][2]["locator"]
        cap["error_map"][2]["detect"] = {"element": bundle}

    elem = go(
        tmp_path / "element",
        clone(capability_dict),
        mutate=by_element,
        configure=lambda s: setattr(s, "texts", {"whatever"}),
    )
    assert elem.result.outcome_code == "app_error"


def test_rules_are_only_consulted_when_something_is_missing(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    ran = go(
        tmp_path,
        capability_dict,
        configure=shows(
            "Search results", "Savings", "Notice to all staff", "Internal Server Error"
        ),
    )
    assert ran.result.status is Status.SUCCESS  # the noise was there, but nothing was blocked
    assert ran.result.recoveries == []
    assert [a.kind for a in ran.surface.acts] == ["navigate", "fill", "click"]


# --- recoverable conditions --------------------------------------------------------------------


def interstitial(surface: FakeReplaySurface, *, persists: bool = False) -> None:
    """The notice covers the page until Continue is clicked."""
    surface.texts = {"Notice to all staff"}

    def dismiss(s: FakeReplaySurface) -> None:
        if not persists:
            s.texts = {"Search results", "Savings"}

    surface.on_click = {"button 'Continue'": dismiss}


def test_a_known_interstitial_is_dismissed_and_the_run_carries_on(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def notice(cap: dict[str, Any]) -> None:
        cap["error_map"][1]["recovery"]["locator"]["description"] = "button 'Continue'"

    ran = go(tmp_path, capability_dict, mutate=notice, configure=interstitial)
    assert ran.result.status is Status.SUCCESS
    assert ran.result.recoveries == [
        RecoveryRecord(rule_id="welcome_interstitial", step_id="s3", kind="dismiss", attempts=1)
    ]
    assert ran.result.outputs == {"savings_balance": "$2,480.15"}
    assert [a.kind for a in ran.surface.acts] == ["navigate", "fill", "click", "click"]


def test_a_recovery_that_does_not_clear_the_condition_is_bounded_then_a_hard_failure(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def notice(cap: dict[str, Any]) -> None:
        cap["error_map"][1]["recovery"]["locator"]["description"] = "button 'Continue'"

    ran = go(
        tmp_path,
        capability_dict,
        mutate=notice,
        configure=lambda s: interstitial(s, persists=True),
        evidence=True,
    )
    result = ran.result
    assert (result.status, result.outcome_code, result.failed_step) == (
        Status.HARD_FAILURE,
        "recovery_exhausted",
        "s3",
    )
    dismissals = [a for a in ran.surface.acts if a.kind == "click"][1:]  # after the search click
    assert len(dismissals) == 2  # max_attempts, not one more
    assert "welcome_interstitial" in (result.observed or "")


def test_a_recovery_whose_button_cannot_be_found_is_a_hard_failure_not_a_hang(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def notice(cap: dict[str, Any]) -> None:
        cap["error_map"][1]["recovery"]["locator"]["description"] = "button 'Continue'"

    def no_button(surface: FakeReplaySurface) -> None:
        surface.texts = {"Notice to all staff"}
        surface.never = {"button 'Continue'"}

    ran = go(tmp_path, capability_dict, mutate=notice, configure=no_button)
    assert ran.result.outcome_code == "recovery_exhausted"


def wait_retry(cap: dict[str, Any]) -> None:
    rule = cap["error_map"][1]
    rule["detect"] = {"text_present": ["Service busy"]}
    rule["recovery"] = {"kind": "wait_retry", "max_attempts": 3, "backoff_ms": 200}


def test_a_transient_condition_is_waited_out_with_growing_backoff_and_reported(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def busy_then_ready(surface: FakeReplaySurface) -> None:
        surface.texts = {"Service busy"}
        surface.appear_after = {"Search results": 2, "Savings": 2}
        original = surface.pause

        def pause(ms: int) -> None:
            original(ms)
            if len(surface.pauses) >= 2:
                surface.texts.discard("Service busy")

        surface.pause = pause  # type: ignore[method-assign]

    ran = go(tmp_path, capability_dict, mutate=wait_retry, configure=busy_then_ready)
    assert ran.result.status is Status.SUCCESS
    backoffs = [ms for ms in ran.surface.pauses if ms >= 200]
    assert backoffs == [200, 400]  # doubles each attempt
    (recovery,) = [r for r in ran.result.recoveries if r.rule_id == "welcome_interstitial"]
    assert (recovery.kind, recovery.step_id) == ("wait_retry", "s3")
    assert recovery.attempts == 2


def test_a_transient_condition_that_never_clears_is_bounded(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    ran = go(
        tmp_path,
        capability_dict,
        mutate=wait_retry,
        configure=shows("Service busy"),
    )
    assert ran.result.outcome_code == "recovery_exhausted"
    assert len([ms for ms in ran.surface.pauses if ms >= 200]) == 3  # max_attempts backoffs


def test_each_step_gets_its_own_recovery_budget(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def notice_twice(cap: dict[str, Any]) -> None:
        cap["error_map"][1]["recovery"]["locator"]["description"] = "button 'Continue'"
        cap["error_map"][1]["recovery"]["max_attempts"] = 1

    def covers(surface: FakeReplaySurface) -> None:
        surface.texts = {"Notice to all staff"}
        surface.missing = {"member number field": 1}  # so the notice is noticed at s2 as well
        state = {"clicks": 0}

        def dismiss(s: FakeReplaySurface) -> None:
            state["clicks"] += 1
            if state["clicks"] >= 2:  # the notice comes back once, at the next step
                s.texts = {"Search results", "Savings"}

        surface.on_click = {"button 'Continue'": dismiss}

    ran = go(tmp_path, capability_dict, mutate=notice_twice, configure=covers)
    assert ran.result.status is Status.SUCCESS
    assert [(r.step_id, r.attempts) for r in ran.result.recoveries] == [("s2", 1), ("s3", 1)]


# --- slow pages --------------------------------------------------------------------------------


def test_a_slow_page_is_waited_for_and_reported_without_failing_the_run(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def slow(surface: FakeReplaySurface) -> None:
        surface.missing = {"member number field": 15}  # 1.5 s of polling at 100 ms

    ran = go(tmp_path, capability_dict, configure=slow)
    assert ran.result.status is Status.SUCCESS
    (record,) = ran.result.recoveries
    assert (record.rule_id, record.step_id, record.kind) == ("slow_response", "s2", "wait_retry")
    assert record.attempts == 15


def test_an_action_that_itself_takes_long_is_reported_as_a_slow_response(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def slow_click(surface: FakeReplaySurface) -> None:
        # the browser absorbed a slow page while settling: no element was ever "missing"
        surface.on_act = lambda a: surface.clock.advance(1.8) if a.kind == "click" else None

    ran = go(tmp_path, capability_dict, configure=slow_click)
    assert ran.result.status is Status.SUCCESS
    (record,) = ran.result.recoveries
    assert (record.rule_id, record.step_id, record.kind, record.attempts) == (
        "slow_response",
        "s3",
        "wait_retry",
        1,
    )


def test_a_step_is_reported_slow_at_most_once(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def slow_everything(surface: FakeReplaySurface) -> None:
        surface.on_act = lambda a: surface.clock.advance(1.8) if a.kind == "click" else None
        surface.appear_after = {}
        surface.texts = set()
        surface.missing = {"search button": 20}  # also waited for, before the slow click

    ran = go(tmp_path, capability_dict, configure=slow_everything)
    assert [r.step_id for r in ran.result.recoveries].count("s3") <= 1


def test_waiting_for_text_on_purpose_is_not_reported_as_slow(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def add_wait(cap: dict[str, Any]) -> None:
        cap["steps"].insert(
            3,
            {
                "id": "s3b",
                "action": "wait_for",
                "description": "processing page",
                "risk_class": "read",
                "expect": {"text_present": ["Loaded"], "timeout_ms": 5000},
            },
        )

    def two_seconds(surface: FakeReplaySurface) -> None:
        surface.appear_after = {"Loaded": 20}  # 2 s of polling: that is what wait_for is for

    ran = go(tmp_path, capability_dict, mutate=add_wait, configure=two_seconds)
    assert ran.result.status is Status.SUCCESS
    assert ran.result.recoveries == []


def test_a_page_that_is_only_briefly_late_is_not_worth_reporting(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def brisk(surface: FakeReplaySurface) -> None:
        surface.missing = {"member number field": 3}

    assert go(tmp_path, capability_dict, configure=brisk).result.recoveries == []


def test_a_page_slower_than_the_budget_is_a_hard_failure(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def hopeless(surface: FakeReplaySurface) -> None:
        surface.missing = {"member number field": 1000}

    ran = go(tmp_path, capability_dict, configure=hopeless)
    assert (ran.result.outcome_code, ran.result.failed_step) == ("locator_not_found", "s2")


# --- the audit trail ---------------------------------------------------------------------------


def test_recoveries_and_the_condition_that_ended_the_run_are_logged(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def notice(cap: dict[str, Any]) -> None:
        cap["error_map"][1]["recovery"]["locator"]["description"] = "button 'Continue'"

    ran: Ran = go(tmp_path, capability_dict, mutate=notice, configure=interstitial)
    (event,) = ran.events("recovery")
    assert (event["reason"], event["step"], event["outcome"]) == (
        "welcome_interstitial",
        "s3",
        "dismiss",
    )
    clicks = [e for e in ran.events("action") if e["action"] == "click"]
    assert (
        clicks[-1]["step"] == "s3"
    )  # the dismissal went through the gateway, attributed to the step

    crashed = go(tmp_path / "b", clone(capability_dict), configure=shows("Internal Server Error"))
    (detected,) = crashed.events("condition")
    assert (detected["reason"], detected["step"]) == ("app_crashed", "s3")


@pytest.mark.parametrize("text", ["ssn 123-45-6789 shown", "password=hunter2 rejected"])
def test_what_the_page_showed_is_redacted_in_the_failure(
    tmp_path: Path, capability_dict: dict[str, Any], text: str
) -> None:
    ran = go(tmp_path, capability_dict, configure=shows("Internal Server Error", text))
    assert "123-45-6789" not in (ran.result.observed or "")
    assert "hunter2" not in (ran.result.observed or "")
