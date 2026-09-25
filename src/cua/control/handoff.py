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
from cua.surface import Capturable, HumanAction, Surface, SurfaceError

if TYPE_CHECKING:  # the gateway imports the lease, so this import must not run
    from cua.gateway import ActionGateway

_OBSERVED_LIMIT = 500
_ACTION_LIMIT = 50  # actions summarised in the result; the log keeps every one
_FLUSH_MS = 100  # after a hand-back, let the person's last events reach us before capture stops


class Operator(Protocol):
    """Whatever tells a person that they are needed: a terminal prompt, a web page, a chat."""

    def notify(self, request: InterventionRequest) -> None: ...


Outcome = Literal["handed_back", "aborted", "timed_out"]


@dataclass(frozen=True)
class HandoffResult:
    outcome: Outcome
    taken_by: str | None
    duration_ms: int
    actions: tuple[str, ...] = ()  # what the person did, described without what they typed


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
        keep_screenshot: Callable[[str, bytes], None] | None = None,
    ) -> None:
        self._keep_screenshot = keep_screenshot
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

        recorded = self._start_recording(request)
        phase = self._wait({Phase.HANDED_BACK, Phase.ABORTED}, self._control_timeout_s)
        self._surface.pause(_FLUSH_MS)
        self._stop_recording()
        actions = self._summarise(recorded)
        if phase is None:
            return self._timed_out(
                request, "control was not handed back", taken_by, started, actions
            )
        if phase is Phase.ABORTED:
            return self._aborted(request, taken_by, started, actions)

        self._log.emit(
            "control_returned", request_id=request.id, actor=taken_by, actions=len(recorded)
        )
        self._observe_what_was_left(request)
        self._epoch = self._lease.resume()
        self._gateway.attach(self._epoch)
        self._log.emit("control_resumed", request_id=request.id)
        return HandoffResult("handed_back", taken_by, self._elapsed_ms(started), actions)

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

    def _observe_what_was_left(self, request: InterventionRequest) -> None:
        try:
            obs = self._surface.observe()
        except SurfaceError:
            self._log.emit("handback_observed", detail="the page could not be read")
            return
        if self._keep_screenshot is not None:
            self._keep_screenshot(f"handback-{request.id}", obs.screenshot)
        text = " | ".join(f.text for f in obs.frames if f.text)[:_OBSERVED_LIMIT]
        self._log.emit(
            "handback_observed",
            url=urlsplit(obs.url).path,
            text=text,
            dialogs=[f"{d.kind}: {d.message}" for d in obs.dialogs],
        )

    def _start_recording(self, request: InterventionRequest) -> list[HumanAction]:
        recorded: list[HumanAction] = []
        if not isinstance(self._surface, Capturable):
            return recorded

        def sink(action: HumanAction) -> None:
            recorded.append(action)
            self._log.emit(
                "human_action",
                request_id=request.id,
                n=len(recorded),
                what=action.describe(),
                frame=action.frame,
            )

        self._surface.start_capture(sink)
        return recorded

    def _stop_recording(self) -> None:
        if isinstance(self._surface, Capturable):
            self._surface.stop_capture()

    @staticmethod
    def _summarise(recorded: list[HumanAction]) -> tuple[str, ...]:
        lines = [action.describe() for action in recorded[:_ACTION_LIMIT]]
        if len(recorded) > _ACTION_LIMIT:
            lines.append(f"(+{len(recorded) - _ACTION_LIMIT} more, see the log)")
        return tuple(lines)

    def _timed_out(
        self,
        request: InterventionRequest,
        detail: str,
        taken_by: str | None,
        started: float,
        actions: tuple[str, ...] = (),
    ) -> HandoffResult:
        with contextlib.suppress(LeaseError):
            self._lease.abort(request.id, who="system")
        self._log.emit("intervention_timed_out", request_id=request.id, detail=detail)
        return HandoffResult("timed_out", taken_by, self._elapsed_ms(started), actions)

    def _aborted(
        self,
        request: InterventionRequest,
        taken_by: str | None,
        started: float,
        actions: tuple[str, ...] = (),
    ) -> HandoffResult:
        self._log.emit("intervention_aborted", request_id=request.id, actor=taken_by)
        return HandoffResult("aborted", taken_by, self._elapsed_ms(started), actions)

    def _elapsed_ms(self, started: float) -> int:
        return max(0, int((self._clock() - started) * 1000))
