"""Fault bookkeeping for MemberServ: which runtime condition to inject, where, and how often.

Faults are armed out-of-band (the admin API or the MOCK_FAULTS environment variable at boot),
never through a URL the agent can see, so a fault name can never leak into an observation or a
recorded capability. What a fault *does* to a response lives in server.py, next to the pages.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_APP_STEPS = frozenset({"search", "results", "member", "newsub_form", "newsub_submit"})
# what "any" matches: real app pages, never images, the header/navigation frames or the frameset
ANY_STEPS = _APP_STEPS | {"processing", "done"}
STEPS = _APP_STEPS | {"login", "any"}


class FaultError(ValueError):
    """An arming request that cannot be honoured; the message says why."""


@dataclass(frozen=True)
class ModeSpec:
    default_step: str
    steps: frozenset[str]  # where this mode makes sense
    get_only: bool = False  # a GET-only notice must never replace a POST


MODES: dict[str, ModeSpec] = {
    "member_not_found": ModeSpec("results", frozenset({"results", "member"})),
    "validation_error": ModeSpec("newsub_submit", frozenset({"newsub_submit"})),
    "session_timeout": ModeSpec("newsub_form", _APP_STEPS | {"any"}),
    "slow_load": ModeSpec("search", _APP_STEPS | {"login", "any"}),
    "interstitial_known": ModeSpec(
        "member", frozenset({"search", "member", "newsub_form", "any"}), get_only=True
    ),
    "app_error": ModeSpec("newsub_submit", _APP_STEPS | {"any"}),
}


@dataclass
class ArmedFault:
    mode: str
    step: str
    remaining: int | None  # None means always
    params: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "step": self.step,
            "times_remaining": "always" if self.remaining is None else self.remaining,
            "params": self.params,
        }


def _is_delay(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value >= 0


def _is_status(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 200 <= value <= 599


def _is_flag(value: object) -> bool:
    return isinstance(value, bool)


_ALLOWED_PARAMS: dict[str, dict[str, Callable[[object], bool]]] = {
    "slow_load": {"delay_ms": _is_delay},
    "app_error": {"commit": _is_flag, "status": _is_status},
}


def _check_params(mode: str, params: dict[str, Any]) -> None:
    allowed = _ALLOWED_PARAMS.get(mode, {})
    for key, value in params.items():
        check = allowed.get(key)
        if check is None:
            raise FaultError(f"unknown param {key!r} for mode {mode!r}")
        if not check(value):
            raise FaultError(f"params.{key} has an invalid value")


def _parse_times(value: object) -> int | None:
    if value == "always":
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    raise FaultError("times must be a positive integer or 'always'")


class FaultEngine:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._armed: list[ArmedFault] = []

    def arm(
        self,
        mode: str,
        *,
        step: str | None = None,
        times: object = 1,
        params: object = None,
    ) -> ArmedFault:
        spec = MODES.get(mode)
        if spec is None:
            raise FaultError(f"unknown mode {mode!r}; known modes: {', '.join(MODES)}")
        chosen = step or spec.default_step
        if chosen not in STEPS:
            raise FaultError(f"unknown step {chosen!r}; known steps: {', '.join(sorted(STEPS))}")
        if chosen not in spec.steps:
            raise FaultError(f"mode {mode!r} does not apply to step {chosen!r}")
        remaining = _parse_times(times)
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise FaultError("params must be an object")
        _check_params(mode, params)
        fault = ArmedFault(mode, chosen, remaining, dict(params))
        with self._lock:
            self._armed.append(fault)
        return fault

    def arm_from_spec(self, spec: str) -> None:
        """Arm from `mode[@step][:times]` items separated by commas (the MOCK_FAULTS format)."""
        for raw in spec.split(","):
            item = raw.strip()
            if not item:
                continue
            head, has_times, times_text = item.partition(":")
            mode, _, step = head.partition("@")
            if not has_times:
                times: object = 1
            elif times_text == "always":
                times = "always"
            else:
                try:
                    times = int(times_text)
                except ValueError:
                    raise FaultError("times must be a positive integer or 'always'") from None
            self.arm(mode.strip(), step=step.strip() or None, times=times)

    def disarm(self) -> None:
        with self._lock:
            self._armed.clear()

    def armed(self) -> list[dict[str, Any]]:
        with self._lock:
            return [fault.describe() for fault in self._armed]

    def take(self, step: str, method: str) -> ArmedFault | None:
        """The first armed fault that applies here, consuming one use of it."""
        with self._lock:
            for fault in self._armed:
                if not self._applies(fault, step, method):
                    continue
                if fault.remaining is not None:
                    fault.remaining -= 1
                    if fault.remaining == 0:
                        self._armed.remove(fault)
                return fault
            return None

    @staticmethod
    def _applies(fault: ArmedFault, step: str, method: str) -> bool:
        if MODES[fault.mode].get_only and method != "GET":
            return False
        if fault.step == "any":
            return step in ANY_STEPS
        return fault.step == step
