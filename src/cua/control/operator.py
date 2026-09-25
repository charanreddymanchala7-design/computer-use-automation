"""The operator side of a handoff: what a person can do, and the terminal way of doing it.

``OperatorService`` is the only thing an operator channel (terminal, web page, chat bot) talks
to, and it only ever moves the control lease. A channel therefore cannot touch the browser: the
person works in the browser window itself, and the channel is just how they say "I have it" and
"it is yours again".
"""

from __future__ import annotations

import contextlib
import getpass
import sys
import threading
from collections.abc import Callable
from typing import Any, TextIO

from cua.control.handoff import Operator
from cua.control.lease import ControlLease, LeaseError, Phase
from cua.control.stuck import InterventionRequest

_NAME_LIMIT = 64
_OPEN = {Phase.WAITING_FOR_HUMAN, Phase.HUMAN_IN_CONTROL}


def _clean_name(name: str) -> str:
    return name.strip()[:_NAME_LIMIT] or "operator"


class OperatorService:
    def __init__(self, lease: ControlLease) -> None:
        self._lease = lease
        self._epoch: int | None = None  # the epoch the person was granted when they took control

    def snapshot(self) -> dict[str, Any]:
        state = self._lease.state
        return {
            "phase": state.phase.value,
            "controller": state.controller.value,
            "epoch": state.epoch,
            "request": state.request.to_public() if state.request else None,
        }

    def take_control(self, request_id: str, *, human: str) -> dict[str, Any]:
        self._epoch = self._lease.take_control(request_id, human=_clean_name(human))
        return self.snapshot()

    def hand_back(self, epoch: int | None = None) -> dict[str, Any]:
        if epoch is None:
            # whoever took control, on this or another channel; a wrong phase is reported as such
            epoch = self._epoch if self._epoch is not None else self._lease.state.epoch
        self._lease.hand_back(epoch)
        return self.snapshot()

    def abort(self, request_id: str, *, who: str) -> dict[str, Any]:
        self._lease.abort(request_id, who=_clean_name(who))
        return self.snapshot()


class TerminalOperator:
    """Prompts on a terminal. Output goes to stderr so stdout stays machine-readable."""

    def __init__(
        self,
        lease: ControlLease,
        *,
        input_fn: Callable[[], str] = input,
        out: TextIO | None = None,
        human: str | None = None,
    ) -> None:
        self._lease = lease
        self._service = OperatorService(lease)
        self._input = input_fn
        self._out = out if out is not None else sys.stderr
        self._human = human or _safe_user()
        self._thread: threading.Thread | None = None

    def notify(self, request: InterventionRequest) -> None:
        self._say("[!!]", f"A person is needed: {request.headline}")
        self._say(
            "    ",
            f"capability {request.capability_id}, step {request.step_id}, request {request.id}",
        )
        self._say("    ", request.reason)
        if request.screenshot:
            self._say("    ", f"screenshot: {request.screenshot}")
        self._say(
            "    ",
            "type 'take' to take over the browser window, or 'abort' to stop the run",
        )
        self._thread = threading.Thread(
            target=self._listen, args=(request.id,), name="cua-terminal-operator", daemon=True
        )
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # --- internals -------------------------------------------------------------------------------

    def _listen(self, request_id: str) -> None:
        while self._lease.state.phase in _OPEN:
            try:
                command = self._input().strip().lower()
            except EOFError:
                return  # no terminal to talk to; the handoff will time out on its own
            if self._lease.state.phase not in _OPEN:
                return  # answered elsewhere while this was waiting for a line
            self._handle(command, request_id)

    def _handle(self, command: str, request_id: str) -> None:
        try:
            if command == "take":
                self._service.take_control(request_id, human=self._human)
                self._say(
                    "[ok]",
                    "you have control: fix the problem in the browser window, then type 'done'",
                )
            elif command == "done":
                self._service.hand_back()
                self._say("[ok]", "control handed back; the run will re-check the page and go on")
            elif command == "abort":
                self._service.abort(request_id, who=self._human)
                self._say("[ok]", "run aborted")
            else:
                self._say("[xx]", f"unknown command {command!r}: use take, done or abort")
        except LeaseError as exc:
            self._say("[xx]", str(exc))

    def _say(self, marker: str, text: str) -> None:
        print(f"{marker} {text}", file=self._out, flush=True)


class Broadcast:
    """Tell every channel at once (terminal and web page); one failing does not silence another."""

    def __init__(self, *channels: Operator) -> None:
        self._channels = channels

    def notify(self, request: InterventionRequest) -> None:
        for channel in self._channels:
            with contextlib.suppress(Exception):
                channel.notify(request)


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "operator"
