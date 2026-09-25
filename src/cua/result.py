"""The replay result contract and the error taxonomy behind it.

A caller (a person, or an AI agent invoking a capability) branches on exactly one of four
statuses, each with its own process exit code:

* ``success``           the checkpoint held; declared outputs are returned
* ``business_outcome``  a legitimate answer such as "no such member"; not a crash
* ``escalated``         the system could not safely continue and a human was brought in
* ``hard_failure``      stop and debug: which step, what was expected, what was observed

Recoverable conditions (a known interstitial, a slow load) are not a status: they happen
mid-run, the run continues, and what was handled is recorded in ``recoveries``. A recoverable
condition that could not be recovered ends the run as a ``hard_failure``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from cua.common import CODE_PATTERN, StrictModel


class Status(StrEnum):
    SUCCESS = "success"
    BUSINESS_OUTCOME = "business_outcome"
    ESCALATED = "escalated"
    HARD_FAILURE = "hard_failure"


EXIT_CODES: dict[Status, int] = {
    Status.SUCCESS: 0,
    Status.BUSINESS_OUTCOME: 10,
    Status.ESCALATED: 20,
    Status.HARD_FAILURE: 30,
}


# --- errors raised inside a run ---------------------------------------------------------------


class CuaError(Exception):
    """Base class of every expected runtime condition."""


class BusinessOutcome(CuaError):
    def __init__(self, outcome_code: str, *, message: str | None = None) -> None:
        super().__init__(outcome_code)
        self.outcome_code = outcome_code
        self.message = message


class RecoverableCondition(CuaError):
    """Handled inside the run. Escaping the run means its retry budget was spent."""

    def __init__(self, rule_id: str, *, step_id: str, kind: str) -> None:
        super().__init__(rule_id)
        self.rule_id = rule_id
        self.step_id = step_id
        self.kind = kind


class HardFailure(CuaError):
    def __init__(
        self,
        code: str,
        *,
        step_id: str,
        expected: str,
        observed: str,
        message: str | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.step_id = step_id
        self.expected = expected
        self.observed = observed
        self.message = message


class EscalationRequired(CuaError):
    def __init__(
        self,
        reason: str,
        *,
        request_id: str,
        step_id: str | None = None,
        expected: str | None = None,
        observed: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.request_id = request_id
        self.step_id = step_id
        self.expected = expected  # what the run was waiting for, for the person who is asked
        self.observed = observed


# --- the result -------------------------------------------------------------------------------


class RecoveryRecord(StrictModel):
    """A known condition that was handled without ending the run."""

    rule_id: str
    step_id: str
    kind: Literal["dismiss", "wait_retry"]
    attempts: int = Field(ge=1)


class InterventionRecord(StrictModel):
    """A person was brought in during the run, and how that ended."""

    request_id: str
    step_id: str
    reason_code: str
    outcome: Literal["handed_back", "aborted", "timed_out"]
    taken_by: str | None = None
    duration_ms: int = Field(ge=0)


class DegradedLocator(StrictModel):
    """A step that was found, but not by its first-choice strategy: an early sign of UI drift."""

    step_id: str
    strategy_index: int = Field(ge=1)
    strategy_kind: str


class EvidenceRefs(StrictModel):
    """Paths, relative to the run directory, of the richer signals kept for debugging."""

    screenshot: str | None = None
    aria_snapshot: str | None = None
    trace: str | None = None
    log: str | None = None


class EscalationInfo(StrictModel):
    request_id: str
    reason: str
    step_id: str | None = None


_REQUIRED: dict[Status, tuple[str, ...]] = {
    Status.SUCCESS: ("outputs",),
    Status.BUSINESS_OUTCOME: (),
    Status.ESCALATED: ("escalation",),
    Status.HARD_FAILURE: ("failed_step", "expected", "observed"),
}
_FORBIDDEN: dict[Status, tuple[str, ...]] = {
    Status.SUCCESS: ("failed_step", "expected", "observed", "escalation"),
    Status.BUSINESS_OUTCOME: ("outputs", "failed_step", "expected", "observed", "escalation"),
    Status.ESCALATED: ("outputs", "failed_step", "expected", "observed"),
    Status.HARD_FAILURE: ("outputs", "escalation"),
}


class ReplayResult(StrictModel):
    status: Status
    outcome_code: str = Field(pattern=CODE_PATTERN, description="Machine-readable, always set")
    capability_id: str
    capability_version: str
    run_id: str
    outputs: dict[str, Any] | None = None
    message: str | None = None
    failed_step: str | None = Field(default=None, min_length=1)
    expected: str | None = Field(default=None, min_length=1)
    observed: str | None = Field(default=None, min_length=1)
    escalation: EscalationInfo | None = None
    recoveries: list[RecoveryRecord] = Field(default_factory=list)
    degraded: list[DegradedLocator] = Field(default_factory=list)
    interventions: list[InterventionRecord] = Field(default_factory=list)
    evidence: EvidenceRefs = Field(default_factory=EvidenceRefs)
    duration_ms: int = Field(ge=0)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]

    @model_validator(mode="after")
    def _fields_match_status(self) -> Self:
        status = self.status.value
        for name in _REQUIRED[self.status]:
            if getattr(self, name) is None:
                raise ValueError(
                    f"{status} must return outputs"
                    if name == "outputs"
                    else f"{status} requires {name}"
                )
        for name in _FORBIDDEN[self.status]:
            if getattr(self, name) is not None:
                verb = "return" if name == "outputs" else "carry"
                raise ValueError(f"{status} must not {verb} {name}")
        return self

    @classmethod
    def from_error(
        cls,
        exc: BaseException,
        *,
        capability_id: str,
        capability_version: str,
        run_id: str,
        duration_ms: int,
        recoveries: list[RecoveryRecord] | None = None,
        degraded: list[DegradedLocator] | None = None,
        interventions: list[InterventionRecord] | None = None,
        evidence: EvidenceRefs | None = None,
    ) -> Self:
        """Turn whatever ended a run into the one result a caller understands."""
        common: dict[str, Any] = {
            "capability_id": capability_id,
            "capability_version": capability_version,
            "run_id": run_id,
            "duration_ms": duration_ms,
            "recoveries": recoveries or [],
            "degraded": degraded or [],
            "interventions": interventions or [],
            "evidence": evidence or EvidenceRefs(),
        }
        return cls(**common, **_error_fields(exc))


def _error_fields(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, BusinessOutcome):
        return {
            "status": Status.BUSINESS_OUTCOME,
            "outcome_code": exc.outcome_code,
            "message": exc.message,
        }
    if isinstance(exc, EscalationRequired):
        info = EscalationInfo(request_id=exc.request_id, reason=exc.reason, step_id=exc.step_id)
        return {"status": Status.ESCALATED, "outcome_code": exc.reason, "escalation": info}
    if isinstance(exc, HardFailure):
        return {
            "status": Status.HARD_FAILURE,
            "outcome_code": exc.code,
            "message": exc.message,
            "failed_step": exc.step_id,
            "expected": exc.expected,
            "observed": exc.observed,
        }
    if isinstance(exc, RecoverableCondition):
        return {
            "status": Status.HARD_FAILURE,
            "outcome_code": "recovery_exhausted",
            "failed_step": exc.step_id,
            "expected": f"condition '{exc.rule_id}' cleared by its {exc.kind} recovery",
            "observed": f"condition '{exc.rule_id}' still present after the retry budget",
        }
    # Anything else is a bug or an environment fault. Only the type is reported: the message
    # could carry page content or credentials.
    return {
        "status": Status.HARD_FAILURE,
        "outcome_code": "unexpected_error",
        "failed_step": "unknown",
        "expected": "the step completes without an unhandled error",
        "observed": type(exc).__name__,
    }


def replay_result_json_schema() -> dict[str, Any]:
    """The JSON Schema of a replay result, for callers that parse it."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **ReplayResult.model_json_schema(),
    }
