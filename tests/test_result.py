"""The replay result is what a calling agent branches on, so its contract must be unambiguous."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from pydantic import ValidationError

from cua.result import (
    EXIT_CODES,
    BusinessOutcome,
    CuaError,
    EscalationRequired,
    EvidenceRefs,
    HardFailure,
    RecoverableCondition,
    RecoveryRecord,
    ReplayResult,
    Status,
    replay_result_json_schema,
)

ROOT = Path(__file__).resolve().parent.parent

META: dict[str, Any] = {
    "capability_id": "member_lookup",
    "capability_version": "1.0.0",
    "run_id": "run_001",
    "duration_ms": 1234,
}


def success(**over: Any) -> dict[str, Any]:
    return {
        **META,
        "status": "success",
        "outcome_code": "completed",
        "outputs": {"savings_balance": "$1,250.00"},
        **over,
    }


def failure(**over: Any) -> dict[str, Any]:
    return {
        **META,
        "status": "hard_failure",
        "outcome_code": "app_error",
        "failed_step": "s3",
        "expected": "Search results",
        "observed": "Internal Server Error",
        **over,
    }


# --- exit codes -------------------------------------------------------------------------------


def test_each_status_has_its_own_exit_code() -> None:
    assert EXIT_CODES == {
        Status.SUCCESS: 0,
        Status.BUSINESS_OUTCOME: 10,
        Status.ESCALATED: 20,
        Status.HARD_FAILURE: 30,
    }
    assert len(set(EXIT_CODES.values())) == len(Status)


def test_result_exposes_its_exit_code() -> None:
    assert ReplayResult.model_validate(success()).exit_code == 0
    assert ReplayResult.model_validate(failure()).exit_code == 30


# --- consistency rules ------------------------------------------------------------------------


def test_success_returns_outputs_and_round_trips() -> None:
    result = ReplayResult.model_validate(success())
    assert result.outputs == {"savings_balance": "$1,250.00"}
    assert ReplayResult.model_validate_json(result.model_dump_json()) == result


def test_success_may_have_empty_outputs_but_not_missing_ones() -> None:
    assert ReplayResult.model_validate(success(outputs={})).outputs == {}
    with pytest.raises(ValidationError, match="success must return outputs"):
        ReplayResult.model_validate(success(outputs=None))


def test_success_cannot_carry_failure_detail() -> None:
    with pytest.raises(ValidationError, match="success must not carry failed_step"):
        ReplayResult.model_validate(success(failed_step="s1"))


def test_business_outcome_is_a_legitimate_answer_not_a_crash() -> None:
    result = ReplayResult.model_validate(
        {
            **META,
            "status": "business_outcome",
            "outcome_code": "member_not_found",
            "message": "No member matches the supplied number",
        }
    )
    assert result.exit_code == 10
    assert result.outputs is None


def test_business_outcome_cannot_return_outputs_or_failure_detail() -> None:
    base: dict[str, Any] = {
        **META,
        "status": "business_outcome",
        "outcome_code": "member_not_found",
    }
    with pytest.raises(ValidationError, match="business_outcome must not return outputs"):
        ReplayResult.model_validate({**base, "outputs": {}})
    with pytest.raises(ValidationError, match="business_outcome must not carry failed_step"):
        ReplayResult.model_validate({**base, "failed_step": "s2"})


@pytest.mark.parametrize("missing", ["failed_step", "expected", "observed"])
def test_hard_failure_says_what_step_what_was_expected_and_what_was_seen(missing: str) -> None:
    data = failure()
    del data[missing]
    with pytest.raises(ValidationError, match=f"hard_failure requires {missing}"):
        ReplayResult.model_validate(data)


def test_hard_failure_cannot_return_outputs() -> None:
    with pytest.raises(ValidationError, match="hard_failure must not return outputs"):
        ReplayResult.model_validate(failure(outputs={"x": "1"}))


def test_escalated_points_at_the_intervention_request() -> None:
    data = {
        **META,
        "status": "escalated",
        "outcome_code": "stuck_unrecognized_state",
        "escalation": {"request_id": "ir_1", "reason": "Unknown dialog", "step_id": "s3"},
    }
    result = ReplayResult.model_validate(data)
    assert result.exit_code == 20
    with pytest.raises(ValidationError, match="escalated requires escalation"):
        ReplayResult.model_validate({k: v for k, v in data.items() if k != "escalation"})


def test_only_escalated_results_carry_escalation() -> None:
    with pytest.raises(ValidationError, match="success must not carry escalation"):
        ReplayResult.model_validate(
            success(escalation={"request_id": "ir_1", "reason": "x", "step_id": None})
        )


@pytest.mark.parametrize("bad", ["", "Not A Code", "ok!", "x" * 65])
def test_every_result_has_a_machine_readable_outcome_code(bad: str) -> None:
    with pytest.raises(ValidationError, match="outcome_code"):
        ReplayResult.model_validate(success(outcome_code=bad))


def test_recoveries_are_metadata_on_the_final_status() -> None:
    result = ReplayResult.model_validate(
        success(
            recoveries=[
                {
                    "rule_id": "welcome_interstitial",
                    "step_id": "s2",
                    "kind": "dismiss",
                    "attempts": 1,
                }
            ]
        )
    )
    assert result.status is Status.SUCCESS
    assert result.recoveries == [
        RecoveryRecord(rule_id="welcome_interstitial", step_id="s2", kind="dismiss", attempts=1)
    ]


def test_duration_cannot_be_negative() -> None:
    with pytest.raises(ValidationError, match="duration_ms"):
        ReplayResult.model_validate(success(duration_ms=-1))


# --- exceptions become results ----------------------------------------------------------------


def build(exc: BaseException, **kw: Any) -> ReplayResult:
    return ReplayResult.from_error(
        exc,
        capability_id="member_lookup",
        capability_version="1.0.0",
        run_id="run_001",
        duration_ms=50,
        **kw,
    )


def test_business_outcome_exception_becomes_a_business_outcome_result() -> None:
    result = build(BusinessOutcome("member_not_found", message="No such member"))
    assert (result.status, result.outcome_code) == (Status.BUSINESS_OUTCOME, "member_not_found")
    assert result.message == "No such member"


def test_hard_failure_exception_keeps_step_expected_and_observed() -> None:
    result = build(HardFailure("app_error", step_id="s3", expected="results", observed="500 page"))
    assert result.status is Status.HARD_FAILURE
    assert (result.failed_step, result.expected, result.observed) == ("s3", "results", "500 page")


def test_escalation_exception_becomes_an_escalated_result() -> None:
    result = build(EscalationRequired("stuck_unrecognized_state", request_id="ir_7", step_id="s2"))
    assert result.status is Status.ESCALATED
    assert result.escalation is not None
    assert (result.escalation.request_id, result.escalation.step_id) == ("ir_7", "s2")


def test_a_recoverable_condition_that_escapes_means_recovery_ran_out() -> None:
    result = build(RecoverableCondition("welcome_interstitial", step_id="s2", kind="dismiss"))
    assert result.status is Status.HARD_FAILURE
    assert result.outcome_code == "recovery_exhausted"
    assert result.failed_step == "s2"
    assert "welcome_interstitial" in (result.observed or "")


def test_unexpected_exceptions_never_leak_their_message() -> None:
    result = build(RuntimeError("password=hunter2 rejected"))
    assert result.status is Status.HARD_FAILURE
    assert result.outcome_code == "unexpected_error"
    assert result.observed == "RuntimeError"
    assert "hunter2" not in result.model_dump_json()


def test_recoveries_and_evidence_are_carried_onto_failures() -> None:
    rec = RecoveryRecord(rule_id="slow", step_id="s1", kind="wait_retry", attempts=2)
    result = build(
        BusinessOutcome("member_not_found"),
        recoveries=[rec],
        evidence=EvidenceRefs(screenshot="failure.png"),
    )
    assert result.recoveries == [rec]
    assert result.evidence.screenshot == "failure.png"


def test_all_expected_errors_share_one_base_class() -> None:
    for exc in (
        BusinessOutcome("x_code"),
        HardFailure("x_code", step_id="s1", expected="a", observed="b"),
        EscalationRequired("x_reason", request_id="ir_1"),
        RecoverableCondition("rule", step_id="s1", kind="dismiss"),
    ):
        assert isinstance(exc, CuaError)


# --- exported JSON Schema ---------------------------------------------------------------------


def test_exported_json_schema_is_valid_and_accepts_a_result() -> None:
    schema = replay_result_json_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    dumped = json.loads(ReplayResult.model_validate(success()).model_dump_json())
    jsonschema.validate(dumped, schema)


def test_committed_schema_file_is_current() -> None:
    committed = json.loads((ROOT / "schemas" / "replay_result.schema.json").read_text())
    assert committed == replay_result_json_schema(), "run: uv run python scripts/export_schemas.py"
