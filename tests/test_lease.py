"""The control lease: who may act on the live session, and when that changes.

A single writer is enforced by our own gate, because the browser protocol itself lets several
clients drive one page. Every transition bumps an epoch, so an action issued under an old epoch
(a slow agent thread that did not notice it lost control) is refused rather than racing a human.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from tests.test_gateway import BASE, FakeSurface, mock_policy

from cua.control import (
    ControlLease,
    Controller,
    InterventionRequest,
    LeaseError,
    Phase,
    StaleEpoch,
)
from cua.evlog import EventLog
from cua.gateway import ActionGateway
from cua.redact import Redactor
from cua.surface import Action


def request(rid: str = "ir_1", step: str = "s3") -> InterventionRequest:
    return InterventionRequest(
        id=rid,
        capability_id="member_lookup",
        goal="Look up a member",
        step_id=step,
        reason_code="unrecognized_state",
        reason="s3: expected the results page; saw an unknown notice",
        outcome_code="expectation_failed",
        url="/msv/frameset.cgi",
        created_at=1.0,
        screenshot="failure.png",
        page_text="LEGAL HOLD NOTICE",
    )


def lease() -> ControlLease:
    return ControlLease()


# --- the state machine --------------------------------------------------------------------------


def test_a_lease_starts_idle_with_nobody_in_control() -> None:
    state = lease().state
    assert (state.phase, state.controller, state.epoch) == (Phase.IDLE, Controller.NONE, 0)


def test_the_whole_handoff_walks_the_states_in_order() -> None:
    lz = lease()
    epochs = [lz.state.epoch]
    seen = [lz.state.phase]

    def step(epoch: int) -> None:
        epochs.append(epoch)
        seen.append(lz.state.phase)

    step(lz.start())
    step(lz.raise_intervention(lz.state.epoch, request()))
    step(lz.take_control("ir_1", human="ops@example.test"))
    step(lz.hand_back(lz.state.epoch))
    step(lz.resume())
    lz.release()
    seen.append(lz.state.phase)
    assert seen == [
        Phase.IDLE,
        Phase.RUNNING,
        Phase.WAITING_FOR_HUMAN,
        Phase.HUMAN_IN_CONTROL,
        Phase.HANDED_BACK,
        Phase.RUNNING,
        Phase.IDLE,
    ]
    assert epochs == sorted(set(epochs))  # every transition moves the epoch forward, never back


@pytest.mark.parametrize(
    ("phase_after", "controller"),
    [
        (Phase.RUNNING, Controller.AGENT),
        (Phase.WAITING_FOR_HUMAN, Controller.NONE),
        (Phase.HUMAN_IN_CONTROL, Controller.HUMAN),
        (Phase.HANDED_BACK, Controller.NONE),
    ],
)
def test_exactly_one_party_is_ever_in_control(phase_after: Phase, controller: Controller) -> None:
    lz = lease()
    lz.start()
    if phase_after is not Phase.RUNNING:
        lz.raise_intervention(lz.state.epoch, request())
    if phase_after in (Phase.HUMAN_IN_CONTROL, Phase.HANDED_BACK):
        lz.take_control("ir_1", human="ops")
    if phase_after is Phase.HANDED_BACK:
        lz.hand_back(lz.state.epoch)
    assert (lz.state.phase, lz.state.controller) == (phase_after, controller)


def test_the_agent_gives_up_control_the_moment_it_raises_an_intervention() -> None:
    lz = lease()
    agent_epoch = lz.start()
    new_epoch = lz.raise_intervention(agent_epoch, request())
    assert new_epoch > agent_epoch
    with pytest.raises(StaleEpoch):
        lz.require(Controller.AGENT, agent_epoch)  # a late agent action is refused


def test_only_the_current_holder_can_act() -> None:
    lz = lease()
    epoch = lz.start()
    lz.require(Controller.AGENT, epoch)  # no error
    with pytest.raises(StaleEpoch, match="human"):
        lz.require(Controller.HUMAN, epoch)


def test_a_stale_epoch_cannot_raise_an_intervention_or_hand_back() -> None:
    lz = lease()
    epoch = lz.start()
    lz.raise_intervention(epoch, request())
    with pytest.raises(StaleEpoch):
        lz.raise_intervention(epoch, request("ir_2"))
    lz.take_control("ir_1", human="ops")
    with pytest.raises(StaleEpoch):
        lz.hand_back(epoch)


@pytest.mark.parametrize(
    ("setup", "attempt", "message"),
    [
        (lambda lz: None, lambda lz: lz.raise_intervention(0, request()), "cannot raise"),
        (lambda lz: None, lambda lz: lz.take_control("ir_1", human="h"), "cannot take control"),
        (lambda lz: None, lambda lz: lz.hand_back(0), "cannot hand back"),
        (lambda lz: None, lambda lz: lz.resume(), "cannot resume"),
        (lambda lz: lz.start(), lambda lz: lz.start(), "cannot start"),
        (
            lambda lz: (lz.start(), lz.raise_intervention(lz.state.epoch, request())),
            lambda lz: lz.resume(),
            "cannot resume",
        ),
    ],
)
def test_a_transition_out_of_order_is_refused_with_a_reason(
    setup: object, attempt: object, message: str
) -> None:
    lz = lease()
    setup(lz)  # type: ignore[operator]
    with pytest.raises(LeaseError, match=message):
        attempt(lz)  # type: ignore[operator]


def test_a_human_can_only_take_control_of_a_request_that_exists() -> None:
    lz = lease()
    lz.raise_intervention(lz.start(), request("ir_1"))
    with pytest.raises(LeaseError, match="no pending request 'ir_nope'"):
        lz.take_control("ir_nope", human="ops")
    assert lz.state.phase is Phase.WAITING_FOR_HUMAN  # a wrong id changed nothing


def test_the_request_records_who_took_control_and_how_it_ended() -> None:
    lz = lease()
    lz.raise_intervention(lz.start(), request())
    assert lz.state.request is not None
    assert lz.state.request.status == "pending"
    lz.take_control("ir_1", human="ops@example.test")
    assert (lz.state.request.status, lz.state.request.taken_by) == ("taken", "ops@example.test")
    lz.hand_back(lz.state.epoch)
    assert lz.state.request.status == "handed_back"


def test_a_human_can_abort_the_run_instead_of_handing_back() -> None:
    lz = lease()
    lz.raise_intervention(lz.start(), request())
    lz.abort("ir_1", who="ops")
    assert lz.state.phase is Phase.ABORTED
    assert lz.state.request is not None
    assert lz.state.request.status == "aborted"
    with pytest.raises(LeaseError, match="cannot resume"):
        lz.resume()


@pytest.mark.parametrize("stage", ["waiting", "human", "handed_back"])
def test_abort_works_from_every_stage_of_a_handoff(stage: str) -> None:
    lz = lease()
    lz.raise_intervention(lz.start(), request())
    if stage != "waiting":
        lz.take_control("ir_1", human="ops")
    if stage == "handed_back":
        lz.hand_back(lz.state.epoch)
    lz.abort("ir_1", who="ops")
    assert lz.state.phase is Phase.ABORTED


def test_release_ends_the_run_from_any_state_and_the_lease_can_be_reused() -> None:
    lz = lease()
    lz.raise_intervention(lz.start(), request())
    lz.release()
    assert (lz.state.phase, lz.state.request) == (Phase.IDLE, None)
    assert lz.start() > 0  # a new run


# --- waiting across threads ---------------------------------------------------------------------


def test_the_agent_waits_for_a_human_who_takes_control_on_another_thread() -> None:
    lz = lease()
    lz.raise_intervention(lz.start(), request())

    def operator() -> None:
        time.sleep(0.03)
        lz.take_control("ir_1", human="ops")
        time.sleep(0.03)
        lz.hand_back(lz.state.epoch)

    thread = threading.Thread(target=operator)
    thread.start()
    assert lz.wait_for({Phase.HUMAN_IN_CONTROL}, timeout_s=2) is Phase.HUMAN_IN_CONTROL
    assert lz.wait_for({Phase.HANDED_BACK, Phase.ABORTED}, timeout_s=2) is Phase.HANDED_BACK
    thread.join()


def test_waiting_gives_up_after_the_timeout_and_says_so() -> None:
    lz = lease()
    lz.raise_intervention(lz.start(), request())
    started = time.monotonic()
    assert lz.wait_for({Phase.HUMAN_IN_CONTROL}, timeout_s=0.05) is None
    assert time.monotonic() - started < 1


def test_waiting_returns_at_once_if_the_phase_is_already_reached() -> None:
    lz = lease()
    lz.start()
    assert lz.wait_for({Phase.RUNNING}, timeout_s=0) is Phase.RUNNING


def test_an_abort_wakes_a_waiting_agent() -> None:
    lz = lease()
    lz.raise_intervention(lz.start(), request())
    threading.Timer(0.03, lambda: lz.abort("ir_1", who="ops")).start()
    assert lz.wait_for({Phase.HUMAN_IN_CONTROL, Phase.ABORTED}, timeout_s=2) is Phase.ABORTED


def test_concurrent_operators_cannot_both_take_control() -> None:
    lz = lease()
    lz.raise_intervention(lz.start(), request())
    outcomes: list[str] = []
    lock = threading.Lock()

    def try_take(name: str) -> None:
        try:
            lz.take_control("ir_1", human=name)
            result = "took"
        except LeaseError:
            result = "refused"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=try_take, args=(f"op{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["refused"] * 7 + ["took"]


# --- observers ----------------------------------------------------------------------------------


def test_every_transition_is_announced_with_its_name() -> None:
    seen: list[tuple[str, Phase, int]] = []
    lz = ControlLease(
        on_change=lambda state, action: seen.append((action, state.phase, state.epoch))
    )
    lz.raise_intervention(lz.start(), request())
    lz.take_control("ir_1", human="ops")
    lz.hand_back(lz.state.epoch)
    lz.resume()
    lz.release()
    assert [name for name, _, _ in seen] == [
        "start",
        "raise_intervention",
        "take_control",
        "hand_back",
        "resume",
        "release",
    ]
    assert [e for _, _, e in seen] == sorted(e for _, _, e in seen)


def test_a_broken_observer_cannot_break_the_lease() -> None:
    def explode(state: object, action: str) -> None:
        raise RuntimeError("logging failed")

    lz = ControlLease(on_change=explode)
    assert lz.start() == 1
    assert lz.state.phase is Phase.RUNNING


# --- the gateway enforces the lease -------------------------------------------------------------


def gated(tmp_path: Path) -> tuple[ActionGateway, FakeSurface, ControlLease]:
    surface = FakeSurface()
    log = EventLog(tmp_path / "run.jsonl", run_id="r", redactor=Redactor())
    lz = lease()
    gateway = ActionGateway(surface, mock_policy(), log, lease=lz)
    gateway.attach(lz.start())
    return gateway, surface, lz


def test_actions_under_the_current_epoch_run(tmp_path: Path) -> None:
    gateway, surface, _ = gated(tmp_path)
    assert gateway.act(Action.navigate(f"{BASE}/msv/x")).executed
    assert len(surface.acts) == 1


def test_an_action_after_control_was_given_up_is_refused_and_never_reaches_the_surface(
    tmp_path: Path,
) -> None:
    gateway, surface, lz = gated(tmp_path)
    lz.raise_intervention(lz.state.epoch, request())  # the agent is quiesced
    with pytest.raises(StaleEpoch):
        gateway.act(Action.navigate(f"{BASE}/msv/x"))
    lz.take_control("ir_1", human="ops")
    with pytest.raises(StaleEpoch):  # and a human holding the lease is not the agent
        gateway.act(Action.navigate(f"{BASE}/msv/x"))
    assert surface.acts == []


def test_the_agent_may_act_again_only_after_resuming_and_re_attaching(tmp_path: Path) -> None:
    gateway, surface, lz = gated(tmp_path)
    lz.raise_intervention(lz.state.epoch, request())
    lz.take_control("ir_1", human="ops")
    lz.hand_back(lz.state.epoch)
    with pytest.raises(StaleEpoch):
        gateway.act(Action.navigate(f"{BASE}/msv/x"))  # handed back is not the same as resumed
    gateway.attach(lz.resume())
    assert gateway.act(Action.navigate(f"{BASE}/msv/x")).executed
    assert len(surface.acts) == 1


def test_a_gateway_with_no_lease_behaves_as_before(tmp_path: Path) -> None:
    surface = FakeSurface()
    log = EventLog(tmp_path / "run.jsonl", run_id="r", redactor=Redactor())
    assert ActionGateway(surface, mock_policy(), log).act(Action.navigate(f"{BASE}/msv/x")).executed


def test_a_leased_gateway_that_was_never_attached_refuses_to_act(tmp_path: Path) -> None:
    surface = FakeSurface()
    log = EventLog(tmp_path / "run.jsonl", run_id="r", redactor=Redactor())
    gateway = ActionGateway(surface, mock_policy(), log, lease=lease())
    with pytest.raises(StaleEpoch, match="attach"):
        gateway.act(Action.navigate(f"{BASE}/msv/x"))


def test_abort_is_refused_when_no_handoff_is_open_or_the_request_id_is_wrong() -> None:
    lz = lease()
    with pytest.raises(LeaseError, match="cannot abort from idle"):
        lz.abort("ir_1", who="ops")
    lz.raise_intervention(lz.start(), request("ir_1"))
    with pytest.raises(LeaseError, match="no open request 'ir_other'"):
        lz.abort("ir_other", who="ops")
    assert lz.state.phase is Phase.WAITING_FOR_HUMAN  # a wrong id changed nothing
