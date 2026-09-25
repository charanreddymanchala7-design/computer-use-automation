"""A scripted page for replay logic: locators that are found late, never, or by a fallback;
text that appears after some pauses; dialogs an action provokes. A shared fake clock advances
whenever the engine waits, so timeouts are tested instantly and deterministically."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from tests.agent_support import BASE, element, frame, page

from cua.artifact import LocatorBundle
from cua.evlog import EventLog
from cua.gateway import ActionGateway
from cua.policy import Policy, UrlRule
from cua.redact import Redactor
from cua.surface import (
    Action,
    ActionResult,
    DialogEvent,
    ElementInfo,
    HarvestError,
    LocatorAttempt,
    LocatorNotFound,
    Observation,
    RequestInfo,
    Resolved,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeReplaySurface:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.acts: list[Action] = []
        self.ok = True
        self.error: str | None = None
        self.texts: set[str] = set()
        self.appear_after: dict[str, int] = {}  # text -> number of pauses before it appears
        self.url = f"{BASE}/msv/x.cgi"
        self.reads: dict[str, str] = {}  # bundle description -> text read
        self.strategy: dict[str, int] = {}  # bundle description -> strategy index that matches
        self.missing: dict[str, int] = {}  # bundle description -> resolves that fail first
        self.never: set[str] = set()
        self.coordinates: dict[str, tuple[float, float]] = {}
        self.provoke: dict[str, list[tuple[str, str]]] = {}  # action kind -> dialogs raised
        self.on_act: Callable[[Action], None] | None = None
        self.on_click: dict[str, Callable[[FakeReplaySurface], None]] = {}  # by bundle description
        self.pauses: list[int] = []
        self.secrets: dict[str, str] = {}
        self.policy: Callable[[str, str], bool] | None = None
        self.resolved_values: list[dict[str, str]] = []
        self.guard: Callable[[RequestInfo], bool] | None = None
        self._refs: dict[str, str] = {}
        self._count = 0

    # --- the Surface surface --------------------------------------------------------------------

    def observe(self) -> Observation:
        return page(frame(0, "main", " | ".join(sorted(self.texts))), url=self.url)

    def act(self, action: Action) -> ActionResult:
        self.acts.append(action)
        if self.on_act is not None:
            self.on_act(action)
        hook = self.on_click.get(self._refs.get(action.ref or "", ""))
        if action.kind == "click" and hook is not None:
            hook(self)
        events = []
        for kind, message in self.provoke.get(action.kind, []):
            accepted = self.policy(kind, message) if self.policy else True
            events.append(DialogEvent(kind, message, accepted))
        return ActionResult(self.ok, f"did {action.kind}", self.url, events, self.error)

    def element_info(self, ref: str) -> ElementInfo:
        return element(ref, text=self._refs.get(ref, ""))

    def wait_for_text(self, text: str, *, timeout_ms: int = 5000) -> bool:
        if text in self.texts:
            return True
        if timeout_ms > 0:
            self.clock.advance(timeout_ms / 1000)
        return False

    def current_url(self) -> str:
        return self.url

    def pause(self, ms: int) -> None:
        self.pauses.append(ms)
        self.clock.advance(ms / 1000)
        for text, after in self.appear_after.items():
            if len(self.pauses) >= after:
                self.texts.add(text)

    def add_secrets(self, secrets: Mapping[str, str]) -> None:
        self.secrets.update(secrets)

    def set_dialog_policy(self, policy: Callable[[str, str], bool] | None) -> None:
        self.policy = policy

    def set_request_guard(self, guard: Callable[[RequestInfo], bool] | None) -> None:
        self.guard = guard

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None

    # --- locating -------------------------------------------------------------------------------

    def resolve(self, bundle: LocatorBundle, values: Mapping[str, str] | None = None) -> Resolved:
        self.resolved_values.append(dict(values or {}))
        description = bundle.description
        attempt = LocatorAttempt(bundle.strategies[0].kind, "no_match")
        if description in self.never:
            raise LocatorNotFound(description, (attempt,))
        left = self.missing.get(description, 0)
        if left > 0:
            self.missing[description] = left - 1
            raise LocatorNotFound(description, (attempt,))
        if description in self.coordinates:
            return Resolved(None, None, 0, "coordinates", (), self.coordinates[description])
        self._count += 1
        ref = f"r{self._count}"
        self._refs[ref] = description
        index = min(self.strategy.get(description, 0), len(bundle.strategies) - 1)
        return Resolved(
            ref, element(ref, text=description), index, bundle.strategies[index].kind, ()
        )

    def read_text(self, ref: str) -> str:
        return self.reads.get(self._refs.get(ref, ""), "")

    def harvest(
        self, ref: str, *, params: Mapping[str, str] | None = None, for_click: bool = False
    ) -> LocatorBundle:
        raise HarvestError("replay never harvests")

    def harvest_value(
        self, value: str, *, anchor_text: str | None = None, params: Mapping[str, str] | None = None
    ) -> LocatorBundle:
        raise HarvestError("replay never harvests")


def open_policy() -> Policy:
    return Policy(
        allow=(UrlRule(host="127.0.0.1"),),
        deny=(UrlRule(host="127.0.0.1", path_prefix="/msv/admin.cgi"),),
    )


def make_gateway(surface: FakeReplaySurface, tmp_path: Path) -> tuple[ActionGateway, EventLog]:
    log = EventLog(tmp_path / "run.jsonl", run_id="run_r", redactor=Redactor(secrets=["s3cr3t-pw"]))
    return ActionGateway(surface, open_policy(), log), log
