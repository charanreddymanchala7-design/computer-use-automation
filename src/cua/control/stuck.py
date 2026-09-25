"""Why a run needs a person, and the request that says so.

An ``InterventionRequest`` is what an operator sees when the system stops: which capability and
step, why it stopped, what it expected against what it found, and a screenshot. It is built from
already-redacted evidence and is safe to show on a shared page.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from cua.redact import Redactor

_SEEN_LIMIT = 300
_PAGE_TEXT_LIMIT = 2000


class StuckReason(StrEnum):
    RISKY_STEP = "risky_step"
    SESSION_EXPIRED = "session_expired"
    UNEXPECTED_DIALOG = "unexpected_dialog"
    UNRECOGNIZED_STATE = "unrecognized_state"
    REPEATED_FAILURE = "repeated_failure"
    DISCOVERY_DEAD_END = "discovery_dead_end"


HEADLINES: dict[StuckReason, str] = {
    StuckReason.RISKY_STEP: "A step that cannot be undone needs a person's decision",
    StuckReason.SESSION_EXPIRED: "The session expired and needs a person to sign in",
    StuckReason.UNEXPECTED_DIALOG: "The application raised a dialog nobody expected",
    StuckReason.UNRECOGNIZED_STATE: "The page is not in a state this capability recognizes",
    StuckReason.REPEATED_FAILURE: "A known problem did not clear after the allowed retries",
    StuckReason.DISCOVERY_DEAD_END: "The agent could not find a way forward",
}

_BY_OUTCOME: dict[str, StuckReason] = {
    "confirmation_required": StuckReason.RISKY_STEP,
    "session_expired": StuckReason.SESSION_EXPIRED,
    "unexpected_dialog": StuckReason.UNEXPECTED_DIALOG,
    "locator_not_found": StuckReason.UNRECOGNIZED_STATE,
    "expectation_failed": StuckReason.UNRECOGNIZED_STATE,
    "checkpoint_failed": StuckReason.UNRECOGNIZED_STATE,
    "recovery_exhausted": StuckReason.REPEATED_FAILURE,
}


def stuck_reason(outcome_code: str, *, discovery: bool = False) -> StuckReason | None:
    """The reason a person is needed, or ``None`` for an ordinary outcome that needs no one."""
    if discovery and outcome_code == "dead_end":
        return StuckReason.DISCOVERY_DEAD_END
    return _BY_OUTCOME.get(outcome_code)


@dataclass(frozen=True)
class InterventionRequest:
    id: str
    capability_id: str
    goal: str
    step_id: str
    reason_code: str
    reason: str
    outcome_code: str
    url: str
    created_at: float
    screenshot: str | None = None
    page_text: str = ""
    status: str = "pending"  # pending -> taken -> handed_back | aborted
    taken_by: str | None = None

    @property
    def headline(self) -> str:
        return HEADLINES[StuckReason(self.reason_code)]

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "capability_id": self.capability_id,
            "goal": self.goal,
            "step_id": self.step_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "headline": self.headline,
            "reason": self.reason,
            "outcome_code": self.outcome_code,
            "url": self.url,
            "screenshot": self.screenshot,
            "page_text": self.page_text,
            "created_at": self.created_at,
            "taken_by": self.taken_by,
        }


def build_request(
    *,
    request_id: str,
    capability_id: str,
    goal: str,
    step_id: str,
    outcome_code: str,
    expected: str,
    observed: str,
    url: str,
    screenshot: str | None,
    redactor: Redactor,
    now: float,
    discovery: bool = False,
) -> InterventionRequest:
    reason = stuck_reason(outcome_code, discovery=discovery) or StuckReason.UNRECOGNIZED_STATE
    seen = redactor.redact_text(observed)
    return InterventionRequest(
        id=request_id,
        capability_id=capability_id,
        goal=redactor.redact_text(goal),
        step_id=step_id,
        reason_code=reason.value,
        reason=redactor.redact_text(f"{step_id}: expected {expected}; saw {seen[:_SEEN_LIMIT]}"),
        outcome_code=outcome_code,
        # a path only: the query string is where session tokens live
        url=urlsplit(url).path,
        created_at=now,
        screenshot=screenshot,
        page_text=seen[:_PAGE_TEXT_LIMIT],
    )
