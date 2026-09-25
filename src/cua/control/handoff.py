"""Hand the live session to a person and take it back.

``Handoff.escalate`` is what the replay engine calls when it is stuck. It quiesces the agent
(the lease moves, so the gateway refuses further agent actions), tells the operator, and then
waits, *pumping the browser's event loop* rather than sleeping, because the synchronous browser
API only services dialogs and page events while its own thread is in a browser call. The person
works in the very same browser window and session; nothing is torn down or restarted.

When the person hands back, control does not pass straight to the agent. The page is observed
first, so what the person left behind is on the record, and only then does the agent resume under
a fresh epoch. Whether the run can *continue* from that page is the caller's decision: the replay
engine re-verifies the step it stopped at, and never repeats an action that already ran.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol
from urllib.parse import urlsplit

from cua.control.lease import ControlLease, LeaseError, Phase
from cua.control.stuck import InterventionRequest
from cua.evlog import EventLog
from cua.surface import Surface, SurfaceError

if TYPE_CHECKING:  # the gateway imports the lease, so this import must not run
    from cua.gateway import ActionGateway

_OBSERVED_LIMIT = 500


class Operator(Protocol):
    """Whatever tells a person that they are needed: a terminal prompt, a web page, a chat."""

    def notify(self, request: InterventionRequest) -> None: ...


Outcome = Literal["handed_back", "aborted", "timed_out"]


@dataclass(frozen=True)
class HandoffResult:
    outcome: Outcome
    taken_by: str | None
    duration_ms: int


def _cancel_dialogs(kind: str, message: str) -> bool:
    """Cancel is the safe answer to a confirm the person did not see coming."""
    return False


class Handoff:
    def __init__(
        self,
        lease: ControlLease,
        gateway: ActionGateway,
        surface: Surface,
        log: EventLog,
        operator: Operator,
        *,
        clock: Callable[[], float] = time.monotonic,
        poll_ms: int = 250,
        claim_timeout_s: float = 900.0,
        control_timeout_s: float = 1800.0,
    ) -> None:
        self._lease = lease
        self._gateway = gateway
        self._surface = surface
        self._log = log
        self._operator = operator
        self._clock = clock
        self._poll_ms = poll_ms
        self._claim_timeout_s = claim_timeout_s
        self._control_timeout_s = control_timeout_s
        self._epoch: int | None = None

    # --- the run around it -------------------------------------------------------------------

    def begin(self) -> None:
        """A run starts: the agent takes the lease and its gateway is bound to that epoch."""
        self._epoch = self._lease.start()
        self._gateway.attach(self._epoch)

    def end(self) -> None:
        """The run is over, however it ended, so the next one can start."""
        if self._lease.state.phase is not Phase.IDLE:
            self._lease.release()
        self._epoch = None

    # --- the handoff ---------------------------------------------------------------------------

    def escalate(self, request: InterventionRequest) -> HandoffResult:
        started = self._clock()
        assert self._epoch is not None, "begin() first"
        self._lease.raise_intervention(self._epoch, request)
        self._epoch = None  # the agent holds nothing until it resumes
        self._log.emit(
            "intervention_raised",
            step=request.step_id,
            reason=request.reason_code,
            request_id=request.id,
            outcome_code=request.outcome_code,
        )
        self._tell(request)

        phase = self._wait({Phase.HUMAN_IN_CONTROL, Phase.ABORTED}, self._claim_timeout_s)
        if phase is None:
            return self._timed_out(request, "nobody took control", None, started)
        if phase is Phase.ABORTED:
            return self._aborted(request, None, started)

        taken_by = self._taken_by()
        self._surface.set_dialog_policy(_cancel_dialogs)
        self._log.emit("control_taken", request_id=request.id, actor=taken_by)

        phase = self._wait({Phase.HANDED_BACK, Phase.ABORTED}, self._control_timeout_s)
        if phase is None:
            return self._timed_out(request, "control was not handed back", taken_by, started)
        if phase is Phase.ABORTED:
            return self._aborted(request, taken_by, started)

        self._log.emit("control_returned", request_id=request.id, actor=taken_by)
        self._observe_what_was_left()
        self._epoch = self._lease.resume()
        self._gateway.attach(self._epoch)
        self._log.emit("control_resumed", request_id=request.id)
        return HandoffResult("handed_back", taken_by, self._elapsed_ms(started))

    # --- internals -------------------------------------------------------------------------------

    def _tell(self, request: InterventionRequest) -> None:
        try:
            self._operator.notify(request)
        except Exception as exc:
            self._log.emit("operator_unreachable", request_id=request.id, detail=type(exc).__name__)

    def _wait(self, phases: set[Phase], timeout_s: float) -> Phase | None:
        """Wait for the lease to reach a phase, keeping the browser's event loop turning."""
        deadline = self._clock() + timeout_s
        while True:
            reached = self._lease.wait_for(phases, timeout_s=0)
            if reached is not None:
                return reached
            if self._clock() >= deadline:
                return None
            self._surface.pause(self._poll_ms)

    def _taken_by(self) -> str | None:
        request = self._lease.state.request
        return request.taken_by if request is not None else None

    def _observe_what_was_left(self) -> None:
        try:
            obs = self._surface.observe()
        except SurfaceError:
            self._log.emit("handback_observed", detail="the page could not be read")
            return
        text = " | ".join(f.text for f in obs.frames if f.text)[:_OBSERVED_LIMIT]
        self._log.emit(
            "handback_observed",
            url=urlsplit(obs.url).path,
            text=text,
            dialogs=[f"{d.kind}: {d.message}" for d in obs.dialogs],
        )

    def _timed_out(
        self, request: InterventionRequest, detail: str, taken_by: str | None, started: float
    ) -> HandoffResult:
        with contextlib.suppress(LeaseError):
            self._lease.abort(request.id, who="system")
        self._log.emit("intervention_timed_out", request_id=request.id, detail=detail)
        return HandoffResult("timed_out", taken_by, self._elapsed_ms(started))

    def _aborted(
        self, request: InterventionRequest, taken_by: str | None, started: float
    ) -> HandoffResult:
        self._log.emit("intervention_aborted", request_id=request.id, actor=taken_by)
        return HandoffResult("aborted", taken_by, self._elapsed_ms(started))

    def _elapsed_ms(self, started: float) -> int:
        return max(0, int((self._clock() - started) * 1000))
