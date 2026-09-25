"""What an operator can do, and the terminal channel for doing it.

The service is the same whatever the channel (terminal, web page, chat): it only ever moves the
control lease, so an operator surface can never touch the browser itself."""

from __future__ import annotations

import io
import threading
from collections.abc import Callable, Iterator

import pytest
from tests.test_lease import request as make_request

from cua.control import ControlLease, LeaseError, Phase, StaleEpoch
from cua.control.operator import Broadcast, OperatorService, TerminalOperator


def waiting() -> tuple[ControlLease, OperatorService]:
    lease = ControlLease()
    lease.raise_intervention(lease.start(), make_request())
    return lease, OperatorService(lease)


def test_the_snapshot_says_who_holds_the_session_and_what_is_being_asked() -> None:
    _, service = waiting()
    snap = service.snapshot()
    assert (snap["phase"], snap["controller"]) == ("waiting_for_human", "none")
    assert snap["request"]["id"] == "ir_1"
    assert snap["request"]["headline"] == "The page is not in a state this capability recognizes"


def test_an_idle_snapshot_has_no_request() -> None:
    assert OperatorService(ControlLease()).snapshot()["request"] is None


def test_taking_control_and_handing_back_move_the_lease() -> None:
    lease, service = waiting()
    snap = service.take_control("ir_1", human="ops@example.test")
    assert (snap["phase"], snap["controller"]) == ("human_in_control", "human")
    assert snap["request"]["taken_by"] == "ops@example.test"
    assert service.hand_back()["phase"] == "handed_back"
    assert lease.state.phase is Phase.HANDED_BACK


def test_handing_back_with_a_stale_epoch_is_refused() -> None:
    _, service = waiting()
    epoch = service.take_control("ir_1", human="ops")["epoch"]
    with pytest.raises(StaleEpoch):
        service.hand_back(epoch=epoch - 1)


def test_handing_back_before_taking_control_is_refused() -> None:
    _, service = waiting()
    with pytest.raises(LeaseError, match="cannot hand back"):
        service.hand_back()


def test_abort_ends_the_run_from_the_operator_side() -> None:
    lease, service = waiting()
    assert service.abort("ir_1", who="ops")["phase"] == "aborted"
    assert lease.state.phase is Phase.ABORTED


def test_a_human_name_is_trimmed_and_bounded_and_never_empty() -> None:
    lease, service = waiting()
    snap = service.take_control("ir_1", human="  " + "x" * 200 + "  ")
    assert snap["request"]["taken_by"] == "x" * 64
    lease2, service2 = waiting()
    assert service2.take_control("ir_1", human="   ")["request"]["taken_by"] == "operator"
    del lease, lease2


# --- the terminal ------------------------------------------------------------------------------


def lines(*commands: str) -> Callable[[], str]:
    it: Iterator[str] = iter(commands)

    def read() -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return read


def run_terminal(*commands: str) -> tuple[ControlLease, str]:
    lease, _ = waiting()
    out = io.StringIO()
    terminal = TerminalOperator(lease, input_fn=lines(*commands), out=out, human="tester")
    terminal.notify(lease.state.request)  # type: ignore[arg-type]
    terminal.join(timeout=5)
    return lease, out.getvalue()


def test_the_terminal_shows_what_is_wrong_and_how_to_respond() -> None:
    _, shown = run_terminal("abort")
    assert "[!!]" in shown
    assert "The page is not in a state this capability recognizes" in shown
    assert "member_lookup" in shown
    assert "s3" in shown
    assert "take" in shown
    assert "abort" in shown
    assert "failure.png" in shown  # where to look


def test_typing_take_then_done_hands_the_session_back() -> None:
    lease, shown = run_terminal("take", "done")
    assert lease.state.phase is Phase.HANDED_BACK
    assert lease.state.request is not None
    assert lease.state.request.taken_by == "tester"
    assert "[ok]" in shown


def test_typing_abort_ends_the_run() -> None:
    lease, _ = run_terminal("abort")
    assert lease.state.phase is Phase.ABORTED


def test_an_unknown_command_is_explained_and_the_terminal_keeps_listening() -> None:
    lease, shown = run_terminal("dance", "take", "done")
    assert "unknown command 'dance'" in shown
    assert lease.state.phase is Phase.HANDED_BACK


def test_a_command_out_of_order_is_reported_not_raised() -> None:
    lease, shown = run_terminal("done", "take", "done")
    assert "[xx]" in shown  # 'done' before 'take' is refused, and the person can carry on
    assert lease.state.phase is Phase.HANDED_BACK


def test_a_closed_input_stops_the_terminal_quietly() -> None:
    lease, _ = run_terminal()
    assert (
        lease.state.phase is Phase.WAITING_FOR_HUMAN
    )  # nobody answered; the handoff will time out


def test_the_terminal_stops_listening_when_the_run_ends_elsewhere() -> None:
    lease, _ = waiting()
    gate = threading.Event()

    def blocking_input() -> str:
        gate.wait(timeout=5)
        return "take"

    terminal = TerminalOperator(lease, input_fn=blocking_input, out=io.StringIO(), human="t")
    terminal.notify(lease.state.request)  # type: ignore[arg-type]
    lease.abort("ir_1", who="web operator")  # answered on the web page instead
    gate.set()
    terminal.join(timeout=5)
    assert lease.state.phase is Phase.ABORTED  # the late 'take' changed nothing


def test_a_broadcast_reaches_every_channel_even_when_one_fails() -> None:
    heard: list[str] = []

    class Loud:
        def notify(self, request: object) -> None:
            heard.append("loud")

    class Broken:
        def notify(self, request: object) -> None:
            raise RuntimeError("down")

    Broadcast(Broken(), Loud()).notify(make_request())
    assert heard == ["loud"]
