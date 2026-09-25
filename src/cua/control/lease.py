"""The control lease: exactly one party acts on the live session at a time.

The browser protocol lets any number of clients drive one page, so single-writer is something we
enforce ourselves. The lease is a small state machine with an epoch that advances on every
transition. The agent's actions carry the epoch they were issued under (see ``ActionGateway``);
an action from an older epoch, such as a slow thread that never noticed it had been asked to
stop, is refused instead of racing the human who now holds the page.

    IDLE -> RUNNING -> WAITING_FOR_HUMAN -> HUMAN_IN_CONTROL -> HANDED_BACK -> RUNNING
                              \\-------------------\\----------------\\------> ABORTED

Handing back is not resuming: the caller re-observes the page and re-verifies the checkpoint
between the two, so a human's changes are never assumed to have left the session where the
capability expects it.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from cua.control.stuck import InterventionRequest
from cua.result import CuaError


class Phase(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING_FOR_HUMAN = "waiting_for_human"
    HUMAN_IN_CONTROL = "human_in_control"
    HANDED_BACK = "handed_back"
    ABORTED = "aborted"


class Controller(StrEnum):
    AGENT = "agent"
    HUMAN = "human"
    NONE = "none"


_CONTROLLER_OF = {
    Phase.RUNNING: Controller.AGENT,
    Phase.HUMAN_IN_CONTROL: Controller.HUMAN,
}


class LeaseError(CuaError):
    """A transition that is not allowed from the current state."""


class StaleEpoch(LeaseError):
    """The caller acted under control it no longer holds."""


@dataclass(frozen=True)
class LeaseState:
    phase: Phase
    epoch: int
    request: InterventionRequest | None = None

    @property
    def controller(self) -> Controller:
        return _CONTROLLER_OF.get(self.phase, Controller.NONE)


Observer = Callable[[LeaseState, str], None]
_KEEP: Any = object()  # "leave the request as it is", distinct from None ("no request")


class ControlLease:
    def __init__(self, on_change: Observer | None = None) -> None:
        self._cond = threading.Condition()
        self._state = LeaseState(Phase.IDLE, 0)
        self._on_change = on_change

    @property
    def state(self) -> LeaseState:
        with self._cond:
            return self._state

    # --- transitions -----------------------------------------------------------------------------

    def start(self) -> int:
        """A run begins; the agent holds the lease. Returns the agent's epoch."""
        with self._cond:
            self._need(Phase.IDLE, "start")
            return self._move("start", Phase.RUNNING)

    def raise_intervention(self, epoch: int, request: InterventionRequest) -> int:
        """The agent stops and asks for a person. It holds nothing from here on."""
        with self._cond:
            self._fresh(epoch)
            self._need(Phase.RUNNING, "raise an intervention")
            return self._move("raise_intervention", Phase.WAITING_FOR_HUMAN, request)

    def take_control(self, request_id: str, *, human: str) -> int:
        """A person claims the session. Only one can: the first caller wins. Returns their epoch."""
        with self._cond:
            self._need(Phase.WAITING_FOR_HUMAN, "take control")
            request = self._state.request
            if request is None or request.id != request_id:
                raise LeaseError(f"no pending request {request_id!r}")
            return self._move(
                "take_control",
                Phase.HUMAN_IN_CONTROL,
                replace(request, status="taken", taken_by=human),
            )

    def hand_back(self, epoch: int) -> int:
        """The person is done. The agent may not act until it has resumed."""
        with self._cond:
            self._fresh(epoch)
            self._need(Phase.HUMAN_IN_CONTROL, "hand back")
            assert self._state.request is not None
            return self._move(
                "hand_back", Phase.HANDED_BACK, replace(self._state.request, status="handed_back")
            )

    def resume(self) -> int:
        """The agent takes the lease back after re-verifying the page. Returns its new epoch."""
        with self._cond:
            self._need(Phase.HANDED_BACK, "resume")
            return self._move("resume", Phase.RUNNING)

    def abort(self, request_id: str, *, who: str) -> int:
        """A person ends the run instead of continuing it."""
        with self._cond:
            request = self._state.request
            if self._state.phase not in (
                Phase.WAITING_FOR_HUMAN,
                Phase.HUMAN_IN_CONTROL,
                Phase.HANDED_BACK,
            ):
                raise LeaseError(f"cannot abort from {self._state.phase.value}")
            assert request is not None
            if request.id != request_id:
                raise LeaseError(f"no open request {request_id!r}")
            return self._move("abort", Phase.ABORTED, replace(request, status="aborted"))

    def release(self) -> int:
        """The run is over, however it ended. The lease can be used again."""
        with self._cond:
            return self._move("release", Phase.IDLE, None)

    # --- checking and waiting --------------------------------------------------------------------

    def require(self, controller: Controller, epoch: int) -> None:
        """Raise unless ``controller`` holds the lease under exactly this epoch."""
        with self._cond:
            state = self._state
            if state.controller is not controller or state.epoch != epoch:
                raise StaleEpoch(
                    f"{controller.value} cannot act: the lease is held by "
                    f"{state.controller.value} (epoch {state.epoch}, caller's epoch {epoch})"
                )

    def wait_for(self, phases: set[Phase], timeout_s: float) -> Phase | None:
        """Block until the lease reaches one of ``phases``; ``None`` if it does not in time."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._state.phase not in phases:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._state.phase

    # --- internals (callers hold the condition) ---------------------------------------------------

    def _need(self, phase: Phase, doing: str) -> None:
        if self._state.phase is not phase:
            raise LeaseError(f"cannot {doing} from {self._state.phase.value}")

    def _fresh(self, epoch: int) -> None:
        if epoch != self._state.epoch:
            raise StaleEpoch(f"epoch {epoch} is stale; the lease is at {self._state.epoch}")

    def _move(self, action: str, phase: Phase, request: Any = _KEEP) -> int:
        kept = self._state.request if request is _KEEP else request
        self._state = LeaseState(phase, self._state.epoch + 1, kept)
        self._cond.notify_all()
        if self._on_change is not None:
            # an observer (logging, a UI) must never be able to break control handoff
            with contextlib.suppress(Exception):
                self._on_change(self._state, action)
        return self._state.epoch
