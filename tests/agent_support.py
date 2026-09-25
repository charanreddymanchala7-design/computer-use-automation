"""A fake surface that can be scripted page by page, so the agent loop's logic is tested fast and
deterministically, with the real gateway and policy in front of it."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from cua.artifact import LocatorBundle, TextLocator
from cua.evlog import EventLog
from cua.gateway import ActionGateway
from cua.policy import Policy, UrlRule
from cua.redact import Redactor
from cua.surface import (
    Action,
    ActionResult,
    DialogEvent,
    ElementInfo,
    FrameObservation,
    HarvestError,
    Observation,
    RequestInfo,
    Resolved,
    Surface,
    UnknownRefError,
)

BASE = "http://127.0.0.1:4310"


def element(
    ref: str,
    tag: str = "a",
    text: str = "",
    role: str | None = None,
    frame: int = 0,
    label_hint: str | None = None,
    **attrs: str,
) -> ElementInfo:
    return ElementInfo(
        ref=ref,
        frame=frame,
        tag=tag,
        role=role,
        text=text,
        attrs=attrs,
        label_hint=label_hint,
        label_after=None,
        box=None,
        checked=None,
        disabled=False,
        options=[],
    )


def frame(
    index: int = 0,
    name: str | None = None,
    text: str = "",
    elements: Sequence[ElementInfo] = (),
    url: str = f"{BASE}/msv/x.cgi",
) -> FrameObservation:
    return FrameObservation(
        index=index, path=(), name=name, url=url, text=text, aria="", elements=list(elements)
    )


def page(*frames: FrameObservation, url: str = f"{BASE}/msv/x.cgi") -> Observation:
    return Observation(
        url=url,
        title="Test page",
        viewport=(1280, 800),
        frames=list(frames or [frame()]),
        screenshot=b"\x89PNG-fake",
    )


class FakeRecordingSurface:
    """Serves the scripted observations in order (the last one repeats) and records every call."""

    def __init__(self, observations: Sequence[Observation]) -> None:
        self.observations = list(observations)
        self._index = -1
        self.calls: list[str] = []
        self.acts: list[Action] = []
        self.ok = True
        self.act_error: str | None = None
        self.dialogs: list[DialogEvent] = []
        self.wait_ok = True
        self.harvest_calls: list[tuple[str, dict[str, str], bool]] = []
        self.harvest_errors: dict[str, str] = {}
        self.known_values: set[str] = set()
        self.read_back: str | None = None
        self._last_value = ""
        self.guard: Callable[[RequestInfo], bool] | None = None
        self.requests_on_click: list[str] = []  # what a click makes the page request

    @property
    def current(self) -> Observation:
        return self.observations[max(self._index, 0)]

    def observe(self) -> Observation:
        self.calls.append("observe")
        self._index = min(self._index + 1, len(self.observations) - 1)
        return self.observations[self._index]

    def act(self, action: Action) -> ActionResult:
        self.calls.append("act")
        self.acts.append(action)
        if action.kind == "click" and self.guard is not None:
            for url in self.requests_on_click:
                self.guard(RequestInfo(url, "GET", "document", True))
        return ActionResult(
            self.ok, f"did {action.kind}", f"{BASE}/msv/x.cgi", list(self.dialogs), self.act_error
        )

    def element_info(self, ref: str) -> ElementInfo:
        for el in self.current.elements():
            if el.ref == ref:
                return el
        raise UnknownRefError(ref)

    def wait_for_text(self, text: str, *, timeout_ms: int = 5000) -> bool:
        self.calls.append("wait_for_text")
        return self.wait_ok

    def set_request_guard(self, guard: Callable[[RequestInfo], bool] | None) -> None:
        self.guard = guard

    def current_url(self) -> str:
        return f"{BASE}/msv/x.cgi"

    def pause(self, ms: int) -> None:
        return None

    def add_secrets(self, secrets: Mapping[str, str]) -> None:
        return None

    def set_dialog_policy(self, policy: Callable[[str, str], bool] | None) -> None:
        return None

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None

    def harvest(
        self, ref: str, *, params: Mapping[str, str] | None = None, for_click: bool = False
    ) -> LocatorBundle:
        self.calls.append("harvest")
        self.harvest_calls.append((ref, dict(params or {}), for_click))
        if ref in self.harvest_errors:
            raise HarvestError(self.harvest_errors[ref])
        return LocatorBundle(
            description=f"element {ref}",
            strategies=[TextLocator(text=f"text of {ref}", exact=True, rationale="scripted fake")],
        )

    def harvest_value(
        self, value: str, *, anchor_text: str | None = None, params: Mapping[str, str] | None = None
    ) -> LocatorBundle:
        self.calls.append("harvest_value")
        if value not in self.known_values:
            raise HarvestError(f"{value!r} not found on the page: quote it exactly")
        self._last_value = value
        return LocatorBundle(
            description=f"value near {anchor_text!r}",
            strategies=[TextLocator(text=value, exact=True, rationale="scripted fake")],
        )

    def resolve(self, bundle: LocatorBundle, values: Mapping[str, str] | None = None) -> Resolved:
        self.calls.append("resolve")
        return Resolved("r1", element("r1", text=self._last_value), 0, "text", ())

    def read_text(self, ref: str) -> str:
        return self.read_back if self.read_back is not None else self._last_value


def policy() -> Policy:
    return Policy(
        allow=(UrlRule(host="127.0.0.1", path_prefix="/msv/"),),
        deny=(UrlRule(host="127.0.0.1", path_prefix="/msv/admin.cgi"),),
    )


def make_gateway(surface: Surface, tmp_path: Path) -> tuple[ActionGateway, EventLog]:
    log = EventLog(tmp_path / "run.jsonl", run_id="run_t", redactor=Redactor())
    return ActionGateway(surface, policy(), log), log
