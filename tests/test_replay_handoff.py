"""Replay with a person on call: when the run is stuck it hands the live session over, and after the
person hands it back it continues, without ever repeating an action that already ran."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.agent_support import BASE
from tests.replay_support import Clock, FakeReplaySurface, open_policy
from tests.test_handoff import RecordingOperator

from cua.artifact import Capability
from cua.control import ControlLease, Phase
from cua.control.handoff import Handoff
from cua.evlog import EventLog, read_events
from cua.gateway import ActionGateway
from cua.redact import Redactor
from cua.replay import ReplayEngine, ReplayLimits
from cua.result import ReplayResult, Status

SESSION_RULE = {
    "id": "session_expired",
    "detect": {"text_present": ["Sign in to continue"]},
    "classification": "hard_failure",
    "code": "session_expired",
    "escalate": True,
    "message": "The session expired",
}


@dataclass
class Ran:
    result: ReplayResult
    surface: FakeReplaySurface
    lease: ControlLease
    operator: RecordingOperator
    log: EventLog
    tmp: Path

    def kinds(self) -> list[str]:
        return [a.kind for a in self.surface.acts]

    def events(self, name: str) -> list[dict[str, Any]]:
        return [e for e in read_events(self.log.path) if e["event"] == name]


def run(
    tmp_path: Path,
    capability_dict: dict[str, Any],
    *,
    configure: Callable[[FakeReplaySurface], None] | None = None,
    person: Callable[[FakeReplaySurface, ControlLease], list[Callable[[], None]]] | None = None,
    session_rule: bool = True,
    max_interventions: int = 2,
    claim_timeout_s: float = 60.0,
    with_handoff: bool = True,
) -> Ran:
    data = json.loads(json.dumps(capability_dict))
    if session_rule:
        data["error_map"].append(SESSION_RULE)
    clock = Clock()
    surface = FakeReplaySurface(clock)
    surface.texts = {"Search results", "Savings"}
    surface.url = f"{BASE}/members/12345"
    surface.reads = {"savings balance": "$2,480.15"}
    if configure:
        configure(surface)
    log = EventLog(tmp_path / "run.jsonl", run_id="run_r", redactor=Redactor(secrets=["s3cr3t-pw"]))
    lease = ControlLease()
    gateway = ActionGateway(surface, open_policy(), log, lease=lease if with_handoff else None)
    operator = RecordingOperator(lease)
    handoff = (
        Handoff(
            lease,
            gateway,
            surface,
            log,
            operator,
            clock=clock,
            claim_timeout_s=claim_timeout_s,
            control_timeout_s=60,
        )
        if with_handoff
        else None
    )
    if person:
        queue = person(surface, lease)

        def on_pause(_: FakeReplaySurface) -> None:
            # a person acts only once they have been asked; the engine's own polling pauses
            # before that must not use up their script
            if queue and lease.state.phase in (Phase.WAITING_FOR_HUMAN, Phase.HUMAN_IN_CONTROL):
                queue.pop(0)()

        surface.on_pause = on_pause
    engine = ReplayEngine(
        surface,
        gateway,
        log,
        base_url=BASE,
        evidence_dir=tmp_path / "evidence",
        limits=ReplayLimits(
            step_timeout_s=2.0, run_timeout_s=600.0, max_interventions=max_interventions
        ),
        clock=clock,
        run_id="run_r",
        handoff=handoff,
    )
    result = engine.run(Capability.model_validate(data), {"member_id": "12345"})
    return Ran(result, surface, lease, operator, log, tmp_path)


def take(lease: ControlLease, who: str = "ops@example.test") -> Callable[[], None]:
    def step() -> None:
        request = lease.state.request
        assert request is not None
        lease.take_control(request.id, human=who)

    return step


def give_back(lease: ControlLease) -> Callable[[], None]:
    def step() -> None:
        lease.hand_back(lease.state.epoch)

    return step


def abort(lease: ControlLease) -> Callable[[], None]:
    def step() -> None:
        request = lease.state.request
        assert request is not None
        lease.abort(request.id, who="ops@example.test")

    return step


def session_expired(surface: FakeReplaySurface) -> None:
    surface.texts = {"Sign in to continue"}
    surface.never = {"member number field"}


def signs_back_in(surface: FakeReplaySurface) -> Callable[[], None]:
    def step() -> None:
        surface.texts = {"Search results", "Savings"}
        surface.never.discard("member number field")

    return step


# --- a stuck run is finished by a person --------------------------------------------------------


def test_an_expired_session_is_fixed_by_a_person_and_the_run_completes(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    ran = run(
        tmp_path,
        capability_dict,
        configure=session_expired,
        person=lambda s, lease: [take(lease), signs_back_in(s), give_back(lease)],
    )
    result = ran.result
    assert result.status is Status.SUCCESS, (result.outcome_code, result.observed)
    assert result.outputs == {"savings_balance": "$2,480.15"}
    assert [(i.step_id, i.reason_code, i.outcome, i.taken_by) for i in result.interventions] == [
        ("s2", "session_expired", "handed_back", "ops@example.test")
    ]
    assert ran.lease.state.phase is Phase.IDLE  # the run released the lease


def test_the_step_that_was_stuck_is_run_again_but_nothing_before_it_is(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    ran = run(
        tmp_path,
        capability_dict,
        configure=session_expired,
        person=lambda s, lease: [take(lease), signs_back_in(s), give_back(lease)],
    )
    assert ran.kinds() == ["navigate", "fill", "click"]  # each exactly once


def test_an_action_that_already_ran_is_never_repeated_after_a_handoff(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    # the click on Search happened, but the results never appeared: only the *check* may be
    # retried after a person looks, because clicking again could submit twice
    def no_results(surface: FakeReplaySurface) -> None:
        surface.texts = {"Savings"}

    def results_appear(surface: FakeReplaySurface) -> Callable[[], None]:
        return lambda: surface.texts.add("Search results")

    ran = run(
        tmp_path,
        capability_dict,
        configure=no_results,
        person=lambda s, lease: [take(lease), results_appear(s), give_back(lease)],
    )
    assert ran.result.status is Status.SUCCESS, (ran.result.outcome_code, ran.result.observed)
    assert ran.kinds().count("click") == 1
    assert [(i.step_id, i.reason_code) for i in ran.result.interventions] == [
        ("s3", "unrecognized_state")
    ]


def test_a_step_that_cannot_find_its_element_is_a_reason_to_ask_for_help(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def moved(surface: FakeReplaySurface) -> None:
        surface.never = {"search button"}

    def fixed(surface: FakeReplaySurface) -> Callable[[], None]:
        return lambda: surface.never.discard("search button")

    ran = run(
        tmp_path,
        capability_dict,
        configure=moved,
        session_rule=False,
        person=lambda s, lease: [take(lease), fixed(s), give_back(lease)],
    )
    assert ran.result.status is Status.SUCCESS
    assert ran.result.interventions[0].reason_code == "unrecognized_state"


# --- what the person is shown -------------------------------------------------------------------


def test_the_operator_is_told_where_why_and_what_the_page_showed(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    ran = run(
        tmp_path,
        capability_dict,
        configure=session_expired,
        person=lambda s, lease: [take(lease), signs_back_in(s), give_back(lease)],
    )
    (request,) = ran.operator.requests
    assert (request.capability_id, request.step_id) == ("member_lookup", "s2")
    assert request.reason_code == "session_expired"
    assert request.id == "ir_run_r_1"
    assert "Sign in to continue" in request.page_text
    assert request.screenshot == "run_r/intervention-1.png"
    assert (tmp_path / "evidence" / request.screenshot).read_bytes().startswith(b"\x89PNG")


def test_what_the_operator_is_shown_never_contains_a_secret(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def leaky(surface: FakeReplaySurface) -> None:
        session_expired(surface)
        surface.texts.add("your password s3cr3t-pw expired")

    ran = run(
        tmp_path,
        capability_dict,
        configure=leaky,
        person=lambda s, lease: [abort(lease)],
    )
    assert "s3cr3t-pw" not in str(ran.operator.requests[0].to_public())
    assert "s3cr3t-pw" not in ran.result.model_dump_json()
    assert "s3cr3t-pw" not in ran.log.path.read_text()


# --- a person does not finish -------------------------------------------------------------------


def test_a_person_aborting_ends_the_run_as_escalated_with_the_original_reason(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    ran = run(
        tmp_path,
        capability_dict,
        configure=session_expired,
        person=lambda s, lease: [take(lease), abort(lease)],
    )
    result = ran.result
    assert (result.status, result.outcome_code, result.exit_code) == (
        Status.ESCALATED,
        "session_expired",
        20,
    )
    assert result.escalation is not None
    assert result.escalation.request_id == "ir_run_r_1"
    assert [i.outcome for i in result.interventions] == ["aborted"]
    assert ran.lease.state.phase is Phase.IDLE


def test_nobody_answering_ends_the_run_as_escalated_too(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    ran = run(tmp_path, capability_dict, configure=session_expired, claim_timeout_s=5)
    assert (ran.result.status, ran.result.outcome_code) == (Status.ESCALATED, "session_expired")
    assert [i.outcome for i in ran.result.interventions] == ["timed_out"]


def test_a_stuck_step_with_no_known_rule_escalates_under_its_own_failure_code(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def moved(surface: FakeReplaySurface) -> None:
        surface.never = {"search button"}

    ran = run(
        tmp_path,
        capability_dict,
        configure=moved,
        session_rule=False,
        person=lambda s, lease: [abort(lease)],
    )
    assert (ran.result.status, ran.result.outcome_code) == (Status.ESCALATED, "locator_not_found")
    assert ran.result.escalation is not None
    assert ran.result.escalation.step_id == "s3"


def test_a_person_who_hands_back_without_fixing_it_gets_asked_at_most_twice(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def moved(surface: FakeReplaySurface) -> None:
        surface.never = {"search button"}

    def person(_: FakeReplaySurface, lease: ControlLease) -> list[Callable[[], None]]:
        return [take(lease), give_back(lease), take(lease), give_back(lease)]

    ran = run(tmp_path, capability_dict, configure=moved, session_rule=False, person=person)
    assert (ran.result.status, ran.result.outcome_code) == (
        Status.HARD_FAILURE,  # help was given twice; the run is broken, not waiting
        "locator_not_found",
    )
    assert [i.outcome for i in ran.result.interventions] == ["handed_back", "handed_back"]
    assert len(ran.operator.requests) == 2


# --- what never needs a person ------------------------------------------------------------------


def test_an_answer_or_a_known_crash_never_brings_a_person_in(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    def crashed(surface: FakeReplaySurface) -> None:
        surface.texts = {"Internal Server Error"}
        surface.never = {"search button"}

    crash = run(tmp_path / "a", capability_dict, configure=crashed)
    assert (crash.result.status, crash.result.outcome_code) == (Status.HARD_FAILURE, "app_error")

    def none_found(surface: FakeReplaySurface) -> None:
        surface.texts = {"No member found"}
        surface.never = {"search button"}

    answer = run(tmp_path / "b", capability_dict, configure=none_found)
    assert answer.result.status is Status.BUSINESS_OUTCOME
    assert crash.operator.requests == [] == answer.operator.requests


def test_an_action_that_needs_a_confirmation_is_not_handed_to_the_operator_page(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    data = json.loads(json.dumps(capability_dict))
    step = next(s for s in data["steps"] if s["id"] == "s3")
    step["locator"]["description"] = "link 'Close'"
    step["risk_class"] = "irreversible_write"
    ran = run(tmp_path, data, person=lambda s, lease: [abort(lease)])
    assert (ran.result.status, ran.result.outcome_code) == (
        Status.ESCALATED,
        "confirmation_required",
    )
    assert ran.operator.requests == []  # approval goes through the gateway's confirmation instead
    assert ran.result.interventions == []


def test_without_a_handoff_a_stuck_run_ends_exactly_as_before(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    ran = run(tmp_path, capability_dict, configure=session_expired, with_handoff=False)
    assert (ran.result.status, ran.result.outcome_code) == (Status.ESCALATED, "session_expired")
    assert ran.result.interventions == []
    assert ran.operator.requests == []


def test_the_result_carries_the_interventions_and_still_validates(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    ran = run(
        tmp_path,
        capability_dict,
        configure=session_expired,
        person=lambda s, lease: [take(lease), signs_back_in(s), give_back(lease)],
    )
    again = ReplayResult.model_validate_json(ran.result.model_dump_json())
    assert again == ran.result
    assert again.interventions[0].duration_ms >= 0
