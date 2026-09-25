"""The handoff: the agent stops, a person takes the same live session, and control comes back.

The "person" here is a script that acts on the lease whenever the agent pumps the page, one step
per pump, so every ordering is deterministic and needs no threads. A shared fake clock makes the
timeouts instant."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from tests.agent_support import BASE
from tests.replay_support import Clock, FakeReplaySurface, open_policy
from tests.test_lease import request as make_request

from cua.control import (
    ControlLease,
    Handoff,
    InterventionRequest,
    Operator,
    Phase,
    StaleEpoch,
)
from cua.evlog import EventLog, read_events
from cua.gateway import ActionGateway
from cua.redact import Redactor
from cua.surface import Action, HumanAction


class RecordingOperator:
    def __init__(self, lease: ControlLease) -> None:
        self.lease = lease
        self.requests: list[InterventionRequest] = []
        self.phase_when_told: list[Phase] = []

    def notify(self, request: InterventionRequest) -> None:
        self.requests.append(request)
        self.phase_when_told.append(self.lease.state.phase)


@dataclass
class World:
    clock: Clock
    surface: FakeReplaySurface
    log: EventLog
    lease: ControlLease
    gateway: ActionGateway
    handoff: Handoff
    operator: RecordingOperator
    queue: list[Callable[[], None]] = field(default_factory=list)

    def script(self, *steps: Callable[[], None]) -> None:
        """What the person does, one step each time the agent pumps the page."""
        self.queue = list(steps)

        def on_pause(_: FakeReplaySurface) -> None:
            if self.queue:
                self.queue.pop(0)()

        self.surface.on_pause = on_pause

    def take(self, who: str = "ops@example.test") -> Callable[[], None]:
        def step() -> None:
            request = self.lease.state.request
            assert request is not None
            self.lease.take_control(request.id, human=who)

        return step

    def give_back(self) -> None:
        self.lease.hand_back(self.lease.state.epoch)

    def abort(self) -> None:
        request = self.lease.state.request
        assert request is not None
        self.lease.abort(request.id, who="ops@example.test")

    def events(self, name: str) -> list[dict[str, Any]]:
        return [e for e in read_events(self.log.path) if e["event"] == name]


def world(
    tmp_path: Path,
    *,
    claim_timeout_s: float = 60.0,
    control_timeout_s: float = 60.0,
    operator: Operator | None = None,
    keep: Callable[[str, bytes], None] | None = None,
) -> World:
    clock = Clock()
    surface = FakeReplaySurface(clock)
    log = EventLog(tmp_path / "run.jsonl", run_id="run_r", redactor=Redactor(secrets=["s3cr3t-pw"]))
    lease = ControlLease()
    gateway = ActionGateway(surface, open_policy(), log, lease=lease)
    recording = RecordingOperator(lease)
    handoff = Handoff(
        lease,
        gateway,
        surface,
        log,
        operator or recording,
        clock=clock,
        poll_ms=250,
        claim_timeout_s=claim_timeout_s,
        control_timeout_s=control_timeout_s,
        keep_screenshot=keep,
    )
    handoff.begin()
    return World(clock, surface, log, lease, gateway, handoff, recording)


# --- the whole handoff --------------------------------------------------------------------------


def test_a_person_takes_the_session_and_hands_it_back(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.script(w.take(), w.give_back)
    result = w.handoff.escalate(make_request())
    assert (result.outcome, result.taken_by) == ("handed_back", "ops@example.test")
    assert w.lease.state.phase is Phase.RUNNING  # the agent holds the session again


def test_the_agent_is_already_stopped_when_the_operator_is_told(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.script(w.take(), w.give_back)
    w.handoff.escalate(make_request())
    assert w.operator.phase_when_told == [Phase.WAITING_FOR_HUMAN]
    assert [r.id for r in w.operator.requests] == ["ir_1"]


def test_nothing_the_agent_does_reaches_the_page_while_a_person_is_in_control(
    tmp_path: Path,
) -> None:
    w = world(tmp_path)
    refused: list[str] = []

    def agent_tries_anyway() -> None:
        try:
            w.gateway.act(Action.navigate(f"{BASE}/msv/x"))
        except StaleEpoch as exc:
            refused.append(str(exc))

    w.script(agent_tries_anyway, w.take(), agent_tries_anyway, w.give_back)
    w.handoff.escalate(make_request())
    assert len(refused) == 2  # while waiting, and while the person held the session
    assert w.surface.acts == []


def test_after_handing_back_the_agent_acts_again_under_a_fresh_epoch(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.script(w.take(), w.give_back)
    w.handoff.escalate(make_request())
    assert w.gateway.act(Action.navigate(f"{BASE}/msv/x")).executed


def test_the_handoff_is_recorded_step_by_step_and_names_who_took_control(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.script(w.take("ops@example.test"), w.give_back)
    w.handoff.escalate(make_request())
    names = [
        e["event"]
        for e in read_events(w.log.path)
        if e["event"]
        in ("intervention_raised", "control_taken", "control_returned", "control_resumed")
    ]
    assert names == [
        "intervention_raised",
        "control_taken",
        "control_returned",
        "control_resumed",
    ]
    assert w.events("control_taken")[0]["actor"] == "ops@example.test"
    assert w.events("intervention_raised")[0]["reason"] == "unrecognized_state"


def test_what_the_person_left_behind_is_observed_before_the_agent_resumes(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.surface.url = f"{BASE}/msv/results.cgi?tok=abc"

    def person_signs_in() -> None:
        w.surface.texts = {"Signed in", "password s3cr3t-pw"}

    w.script(w.take(), person_signs_in, w.give_back)
    w.handoff.escalate(make_request())
    seen = w.events("handback_observed")[0]
    assert seen["url"] == "/msv/results.cgi"  # the path, never the query
    assert "Signed in" in seen["text"]
    assert "s3cr3t-pw" not in w.log.path.read_text()


def test_dialogs_are_not_accepted_on_a_persons_behalf_while_they_hold_the_session(
    tmp_path: Path,
) -> None:
    w = world(tmp_path)
    decisions: list[bool] = []

    def a_dialog_appears() -> None:
        assert w.surface.policy is not None
        decisions.append(w.surface.policy("confirm", "Delete this account?"))

    w.script(w.take(), a_dialog_appears, w.give_back)
    w.handoff.escalate(make_request())
    assert decisions == [False]  # cancel is the safe answer to an irreversible confirm


# --- a person does not finish -------------------------------------------------------------------


def test_a_person_can_abort_instead_of_handing_back(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.script(w.take(), w.abort)
    result = w.handoff.escalate(make_request())
    assert (result.outcome, result.taken_by) == ("aborted", "ops@example.test")
    assert w.lease.state.phase is Phase.ABORTED
    with pytest.raises(StaleEpoch):
        w.gateway.act(Action.navigate(f"{BASE}/msv/x"))  # an aborted run cannot act


def test_a_person_can_abort_before_taking_control(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.script(w.abort)
    result = w.handoff.escalate(make_request())
    assert (result.outcome, result.taken_by) == ("aborted", None)


def test_nobody_claiming_the_request_ends_the_wait_and_says_so(tmp_path: Path) -> None:
    w = world(tmp_path, claim_timeout_s=5)
    result = w.handoff.escalate(make_request())
    assert (result.outcome, result.taken_by) == ("timed_out", None)
    assert w.lease.state.phase is Phase.ABORTED
    assert w.clock.now >= 5
    assert w.events("intervention_timed_out")[0]["detail"] == "nobody took control"


def test_a_person_who_takes_control_and_never_returns_it_times_out_too(tmp_path: Path) -> None:
    w = world(tmp_path, control_timeout_s=5)
    w.script(w.take())
    result = w.handoff.escalate(make_request())
    assert (result.outcome, result.taken_by) == ("timed_out", "ops@example.test")
    assert w.events("intervention_timed_out")[0]["detail"] == "control was not handed back"
    assert w.lease.state.phase is Phase.ABORTED


def test_the_duration_covers_the_whole_wait(tmp_path: Path) -> None:
    w = world(tmp_path, claim_timeout_s=5)
    result = w.handoff.escalate(make_request())
    assert result.duration_ms >= 5000


# --- the operator notice is best effort ---------------------------------------------------------


def test_an_operator_that_cannot_be_notified_does_not_stop_the_handoff(tmp_path: Path) -> None:
    class Broken:
        def notify(self, request: InterventionRequest) -> None:
            raise RuntimeError("smtp down")

    w = world(tmp_path, operator=Broken())
    w.script(w.take(), w.give_back)
    assert w.handoff.escalate(make_request()).outcome == "handed_back"
    assert w.events("operator_unreachable")[0]["detail"] == "RuntimeError"


# --- the run around it --------------------------------------------------------------------------


def test_ending_a_run_releases_the_lease_so_the_next_run_can_start(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.script(w.take(), w.abort)
    w.handoff.escalate(make_request())
    w.handoff.end()
    assert w.lease.state.phase is Phase.IDLE
    w.handoff.begin()
    assert w.lease.state.phase.value == "running"
    assert w.gateway.act(Action.navigate(f"{BASE}/msv/x")).executed


def test_ending_a_run_that_never_began_is_harmless(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.handoff.end()
    w.handoff.end()
    assert w.lease.state.phase is Phase.IDLE


# --- recording what the person did --------------------------------------------------------------


def does(w: World, *actions: HumanAction) -> Callable[[], None]:
    def step() -> None:
        assert w.surface.capture_sink is not None, (
            "capture should be on while a person is in control"
        )
        for action in actions:
            w.surface.capture_sink(action)

    return step


CLICK_SEARCH = HumanAction("click", "button 'Search'", "main")
TYPED_ID = HumanAction("typed", "text field 'Member No:'", "main", chars=5)


def test_what_the_person_did_is_recorded_in_order_and_described_without_content(
    tmp_path: Path,
) -> None:
    w = world(tmp_path)
    w.script(w.take(), does(w, TYPED_ID, CLICK_SEARCH), w.give_back)
    result = w.handoff.escalate(make_request())
    assert result.actions == (
        "typed 5 characters into text field 'Member No:'",
        "click button 'Search'",
    )
    logged = w.events("human_action")
    assert [e["what"] for e in logged] == list(result.actions)
    assert [e["n"] for e in logged] == [1, 2]
    assert [e["frame"] for e in logged] == ["main", "main"]
    assert w.events("control_returned")[0]["actions"] == 2


def test_nothing_is_recorded_before_a_person_takes_control_or_after_they_hand_back(
    tmp_path: Path,
) -> None:
    w = world(tmp_path)
    seen: list[object] = []
    w.script(
        lambda: seen.append(w.surface.capture_sink),  # asked, nobody in control yet
        w.take(),
        w.give_back,
    )
    w.handoff.escalate(make_request())
    assert seen == [None]
    assert w.surface.capture_sink is None  # and it is switched off again


def test_the_last_action_before_handing_back_is_not_lost(tmp_path: Path) -> None:
    w = world(tmp_path)
    # the hand-back arrives first and the final click is only delivered on the next pump
    w.script(w.take(), w.give_back, does(w, CLICK_SEARCH))
    assert w.handoff.escalate(make_request()).actions == ("click button 'Search'",)


@pytest.mark.parametrize("how", ["abort", "timeout"])
def test_capture_is_switched_off_however_the_handoff_ends(tmp_path: Path, how: str) -> None:
    w = world(tmp_path, control_timeout_s=5)
    w.script(w.take(), does(w, CLICK_SEARCH), *([w.abort] if how == "abort" else []))
    result = w.handoff.escalate(make_request())
    assert result.outcome == ("aborted" if how == "abort" else "timed_out")
    assert result.actions == ("click button 'Search'",)  # what was done is kept either way
    assert w.surface.capture_sink is None


def test_a_flood_of_actions_is_capped_and_says_so(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.script(w.take(), does(w, *[CLICK_SEARCH] * 75), w.give_back)
    result = w.handoff.escalate(make_request())
    assert len(result.actions) == 51
    assert result.actions[-1] == "(+25 more, see the log)"
    assert len(w.events("human_action")) == 75  # the log keeps every one


def test_a_surface_that_cannot_capture_still_hands_over(tmp_path: Path) -> None:
    class Bare:
        """A surface without the capture methods, as a desktop stub might be."""

        def __init__(self, inner: FakeReplaySurface) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> object:
            if name in ("start_capture", "stop_capture"):
                raise AttributeError(name)
            return getattr(self._inner, name)

    w = world(tmp_path)
    w.handoff._surface = Bare(w.surface)  # type: ignore[assignment]
    w.script(w.take(), w.give_back)
    result = w.handoff.escalate(make_request())
    assert (result.outcome, result.actions) == ("handed_back", ())


def test_a_screenshot_of_what_the_person_left_behind_can_be_kept(tmp_path: Path) -> None:
    kept: list[tuple[str, bytes]] = []
    w = world(tmp_path, keep=lambda name, png: kept.append((name, png)))
    w.script(w.take(), w.give_back)
    w.handoff.escalate(make_request())
    assert [(name, png[:4]) for name, png in kept] == [("handback-ir_1", b"\x89PNG")]
