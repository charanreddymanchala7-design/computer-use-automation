"""Why a run is stuck, in words a person can act on, and the request that carries it."""

from __future__ import annotations

import pytest

from cua.control import (
    InterventionRequest,
    StuckReason,
    build_request,
    stuck_reason,
)
from cua.redact import Redactor


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("confirmation_required", StuckReason.RISKY_STEP),
        ("session_expired", StuckReason.SESSION_EXPIRED),
        ("unexpected_dialog", StuckReason.UNEXPECTED_DIALOG),
        ("locator_not_found", StuckReason.UNRECOGNIZED_STATE),
        ("expectation_failed", StuckReason.UNRECOGNIZED_STATE),
        ("checkpoint_failed", StuckReason.UNRECOGNIZED_STATE),
        ("recovery_exhausted", StuckReason.REPEATED_FAILURE),
    ],
)
def test_an_outcome_code_maps_to_the_reason_a_person_needs(code: str, reason: StuckReason) -> None:
    assert stuck_reason(code) is reason


@pytest.mark.parametrize("code", ["completed", "member_not_found", "app_error", "invalid_input"])
def test_ordinary_outcomes_are_not_a_reason_to_bring_in_a_human(code: str) -> None:
    assert stuck_reason(code) is None


def test_a_discovery_dead_end_is_its_own_reason() -> None:
    assert stuck_reason("dead_end", discovery=True) is StuckReason.DISCOVERY_DEAD_END
    assert stuck_reason("dead_end") is None  # in replay, a dead end is not a thing


def build(**over: object) -> InterventionRequest:
    args: dict[str, object] = {
        "request_id": "ir_run_1",
        "capability_id": "member_lookup",
        "goal": "Look up a member",
        "step_id": "s3",
        "outcome_code": "expectation_failed",
        "expected": "text 'Search results' present",
        "observed": "LEGAL HOLD NOTICE | select a reason",
        "url": "http://127.0.0.1:4310/msv/frameset.cgi?tok=abc",
        "screenshot": "failure.png",
        "redactor": Redactor(secrets=["hunter2"]),
        "now": 42.0,
    }
    args.update(over)
    return build_request(**args)  # type: ignore[arg-type]


def test_a_request_says_where_why_and_what_the_operator_should_look_at() -> None:
    req = build()
    assert (req.id, req.capability_id, req.step_id) == ("ir_run_1", "member_lookup", "s3")
    assert req.reason_code == "unrecognized_state"
    assert "s3" in req.reason
    assert "Search results" in req.reason  # what was expected
    assert "LEGAL HOLD NOTICE" in req.reason  # what was seen
    assert req.url == "/msv/frameset.cgi"  # a path, never a query that may carry tokens
    assert (req.screenshot, req.status, req.created_at) == ("failure.png", "pending", 42.0)


def test_a_request_never_carries_a_secret_or_pii() -> None:
    req = build(observed="password=hunter2 rejected for ssn 123-45-6789", goal="use hunter2")
    public = str(req.to_public())
    assert "hunter2" not in public
    assert "123-45-6789" not in public


def test_the_public_view_is_what_the_operator_page_shows() -> None:
    public = build().to_public()
    assert public["id"] == "ir_run_1"
    assert public["status"] == "pending"
    assert public["reason_code"] == "unrecognized_state"
    assert public["headline"] == "The page is not in a state this capability recognizes"
    assert public["screenshot"] == "failure.png"


@pytest.mark.parametrize(
    ("code", "headline"),
    [
        ("confirmation_required", "A step that cannot be undone needs a person's decision"),
        ("session_expired", "The session expired and needs a person to sign in"),
        ("unexpected_dialog", "The application raised a dialog nobody expected"),
        ("recovery_exhausted", "A known problem did not clear after the allowed retries"),
    ],
)
def test_each_reason_has_a_plain_language_headline(code: str, headline: str) -> None:
    assert build(outcome_code=code).to_public()["headline"] == headline


def test_a_request_from_discovery_says_the_model_is_stuck() -> None:
    req = build(
        outcome_code="dead_end", discovery=True, observed="repeated the same action 3 times"
    )
    assert req.reason_code == "discovery_dead_end"
    assert req.to_public()["headline"] == "The agent could not find a way forward"


def test_an_unclassified_outcome_still_makes_a_usable_request() -> None:
    req = build(outcome_code="something_new")
    assert req.reason_code == "unrecognized_state"
