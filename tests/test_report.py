"""What a run looks like on a terminal. Markers carry the meaning, so colour is optional."""

from __future__ import annotations

from typing import Any

import pytest

from cua.artifact import Capability
from cua.report import mark_up, marker, mask_inputs, progress_lines, render_result
from cua.result import (
    DegradedLocator,
    EscalationInfo,
    EvidenceRefs,
    InterventionRecord,
    RecoveryRecord,
    ReplayResult,
    Status,
)


def result(**over: Any) -> ReplayResult:
    base: dict[str, Any] = {
        "status": Status.SUCCESS,
        "outcome_code": "completed",
        "capability_id": "member_lookup",
        "capability_version": "1.0.0",
        "run_id": "r1",
        "outputs": {"savings_balance": "$2,480.15"},
        "duration_ms": 4200,
    }
    base.update(over)
    return ReplayResult(**base)


@pytest.mark.parametrize(
    ("kind", "text"), [("ok", "[ok]"), ("warn", "[!!]"), ("fail", "[xx]"), ("info", "[--]")]
)
def test_every_marker_is_plain_text_without_colour(kind: str, text: str) -> None:
    assert marker(kind, color=False) == text


def test_colour_is_added_around_the_marker_never_instead_of_it() -> None:
    coloured = marker("fail", color=True)
    assert "[xx]" in coloured
    assert coloured.startswith("\x1b[")


def test_a_success_reads_top_down_with_its_outputs() -> None:
    text = render_result(result())
    assert text.splitlines()[0] == "[ok] RESULT  success  (exit 0)"
    assert "savings_balance = $2,480.15" in text
    assert "duration  4.2 s" in text
    assert "\x1b" not in text


def test_a_business_outcome_is_not_dressed_up_as_a_failure() -> None:
    text = render_result(
        result(
            status=Status.BUSINESS_OUTCOME,
            outcome_code="member_not_found",
            outputs=None,
            message="No member matches the supplied number",
        )
    )
    assert text.splitlines()[0] == "[!!] RESULT  business_outcome  (exit 10)"
    assert "No member matches the supplied number" in text


def test_a_hard_failure_says_which_step_and_what_was_expected_and_seen() -> None:
    text = render_result(
        result(
            status=Status.HARD_FAILURE,
            outcome_code="locator_not_found",
            outputs=None,
            failed_step="s5",
            expected="element search button",
            observed="no strategy matched",
            evidence=EvidenceRefs(screenshot="failure.png", aria_snapshot="page.json"),
        )
    )
    assert text.splitlines()[0] == "[xx] RESULT  hard_failure  (exit 30)"
    for needle in ("failed step  s5", "expected", "search button", "no strategy matched"):
        assert needle in text
    assert "failure.png" in text


def test_an_escalation_and_the_handoff_are_shown() -> None:
    text = render_result(
        result(
            status=Status.ESCALATED,
            outcome_code="session_expired",
            outputs=None,
            escalation=EscalationInfo(request_id="ir_1", reason="session_expired", step_id="s7"),
            interventions=[
                InterventionRecord(
                    request_id="ir_1",
                    step_id="s7",
                    reason_code="session_expired",
                    outcome="aborted",
                    taken_by="ops",
                    duration_ms=12300,
                )
            ],
        )
    )
    assert text.splitlines()[0] == "[!!] RESULT  escalated  (exit 20)"
    assert "session_expired at s7" in text
    assert "ir_1 at s7: aborted by ops (12.3 s)" in text


def test_recoveries_and_drift_are_reported_on_a_success() -> None:
    text = render_result(
        result(
            recoveries=[
                RecoveryRecord(rule_id="eod_notice", step_id="s7", kind="dismiss", attempts=1)
            ],
            degraded=[DegradedLocator(step_id="s3", strategy_index=1, strategy_kind="text")],
        )
    )
    assert "eod_notice at s7 (dismiss x1)" in text
    assert "s3 found by fallback strategy 1 (text)" in text


def test_identifiers_are_masked_and_sensitive_inputs_are_withheld() -> None:
    shown = mask_inputs({"member_id": "12345", "pin": "9999"}, sensitive={"pin"})
    assert shown == {"member_id": "***45", "pin": "<withheld>"}


def test_progress_is_read_from_the_runs_own_log(capability_dict: dict[str, Any]) -> None:
    capability = Capability.model_validate(capability_dict)
    events = [
        {"event": "replay_start", "capability": "member_lookup@1.0.0"},
        {"event": "step_ok", "step": "s1"},
        {"event": "recovery", "step": "s2", "reason": "welcome_interstitial", "outcome": "dismiss"},
        {"event": "intervention_raised", "step": "s3", "reason": "session_expired"},
        {"event": "control_taken", "actor": "ops"},
        {"event": "control_resumed"},
        {"event": "intervention_timed_out", "step": "s3"},
        {"event": "action_blocked", "step": "s4", "reason": "url_not_allowed"},
        {"event": "irrelevant"},
    ]
    lines = progress_lines(events, capability, color=False)
    assert lines[0] == "[--] replaying member_lookup@1.0.0 (no model is involved)"
    assert lines[1] == "[ok] s1 Open the member search page"
    assert lines[2].startswith("[!!] s2 recovered from 'welcome_interstitial'")
    assert lines[3].startswith("[!!] s3 stuck (session_expired)")
    assert "ops" in lines[4]
    assert lines[5].startswith("[ok] control handed back")
    assert lines[6] == "[xx] the person did not finish the handoff: timed out"
    assert lines[7].startswith("[xx] s4 blocked by policy")
    assert len(lines) == 8


def test_mark_up_prefixes_a_line_with_its_marker() -> None:
    assert mark_up("fail", "it broke", color=False) == "[xx] it broke"
