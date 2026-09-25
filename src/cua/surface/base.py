"""The surface seam: how the system perceives and acts on an application.

Everything above this line (the agent loop, the recorded artifact, replay) talks to a
``Surface``. Only the implementation below it knows whether the app is a web page, a legacy
frameset or, later, a desktop window. A web surface observes frames and DOM facts; a desktop
surface would observe an accessibility tree and pixels; both hand back the same ``Observation``
and accept the same ``Action`` vocabulary, targeting elements by ref or by screen coordinates.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, Self

from cua.artifact import LocatorBundle

# --- errors -----------------------------------------------------------------------------------


class SurfaceError(Exception):
    """Something the surface could not do; the message says what to do next."""


class SurfaceUnavailable(SurfaceError):
    """The underlying driver (e.g. the browser binary) is not available on this machine."""


class UnknownRefError(SurfaceError):
    """The ref was never issued, or was issued before the last reset."""


class StaleRefError(SurfaceError):
    """The ref existed but its element is gone (the page changed); observe again."""


class HarvestError(SurfaceError):
    """No reliable locator could be built; the message says what to change."""


@dataclass(frozen=True)
class LocatorAttempt:
    """One strategy tried while resolving a bundle: matched, no_match or ambiguous."""

    kind: str
    outcome: str


class LocatorNotFound(SurfaceError):
    """No strategy in the bundle found exactly one element."""

    def __init__(
        self, description: str, attempts: tuple[LocatorAttempt, ...], why: str = ""
    ) -> None:
        tried = ", ".join(f"{a.kind}: {a.outcome}" for a in attempts) or "none tried"
        super().__init__(f"could not find {description!r} ({why or tried})")
        self.description = description
        self.attempts = attempts


class UnknownSecretError(SurfaceError):
    """A fill asked for a secret the caller did not provide."""


# --- what the surface reports ------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """Pixel box in top-level viewport coordinates (frame offsets already applied)."""

    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass(frozen=True)
class FrameHop:
    """One step into a frame. Legacy frames may lack names, so index and URL are kept too."""

    name: str | None
    index: int
    url: str


@dataclass
class ElementInfo:
    """Facts about one visible, interactive element. ``ref`` is valid until the next observe."""

    ref: str
    frame: int  # index into Observation.frames
    tag: str
    role: str | None  # None means the element has no semantics: only text, attrs and position
    text: str
    attrs: dict[str, str]
    label_hint: str | None  # text of the neighbouring cell / preceding text: the legacy "label"
    label_after: str | None  # text right after the element (radios and checkboxes)
    box: Box | None
    checked: bool | None
    disabled: bool
    options: list[str]  # option labels for a select


@dataclass
class FrameObservation:
    index: int
    path: tuple[FrameHop, ...]  # from the top window; empty for the top document
    name: str | None
    url: str
    text: str
    aria: str  # accessibility snapshot of this frame only (a parent's never includes a child's)
    elements: list[ElementInfo]
    unreadable: bool = False  # the frame changed under us while it was being read


@dataclass(frozen=True)
class DialogEvent:
    kind: str
    message: str
    accepted: bool


@dataclass
class Observation:
    url: str
    title: str
    viewport: tuple[int, int]
    frames: list[FrameObservation]
    screenshot: bytes
    dialogs: list[DialogEvent] = field(default_factory=list)  # handled since the last observe

    def elements(self) -> Iterator[ElementInfo]:
        for frame in self.frames:
            yield from frame.elements

    def element(self, ref: str) -> ElementInfo:
        for element in self.elements():
            if element.ref == ref:
                return element
        raise UnknownRefError(f"no element {ref!r} in this observation")


# --- what the surface can be asked to do ------------------------------------------------------


@dataclass(frozen=True)
class RequestInfo:
    """A network request the page is about to make, for the request guard to allow or block."""

    url: str
    method: str
    resource_type: str
    is_navigation: bool


RequestGuard = Callable[[RequestInfo], bool]

ActionKind = Literal["navigate", "click", "fill", "select", "press", "wait"]
MAX_WAIT_MS = 30_000


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    ref: str | None = None
    x: float | None = None
    y: float | None = None
    url: str | None = None
    text: str | None = None
    secret: str | None = None
    option: str | None = None
    key: str | None = None
    ms: int | None = None

    @classmethod
    def navigate(cls, url: str) -> Self:
        return cls("navigate", url=url)

    @classmethod
    def click(cls, ref: str) -> Self:
        return cls("click", ref=ref)

    @classmethod
    def click_at(cls, x: float, y: float) -> Self:
        return cls("click", x=x, y=y)

    @classmethod
    def fill(cls, ref: str, text: str | None = None, secret: str | None = None) -> Self:
        if (text is None) == (secret is None):
            raise ValueError("fill needs exactly one of text or secret")
        return cls("fill", ref=ref, text=text, secret=secret)

    @classmethod
    def select(cls, ref: str, option: str) -> Self:
        return cls("select", ref=ref, option=option)

    @classmethod
    def press(cls, key: str) -> Self:
        return cls("press", key=key)

    @classmethod
    def wait(cls, ms: int) -> Self:
        return cls("wait", ms=ms)


@dataclass
class ActionResult:
    ok: bool
    detail: str  # short, safe to show a model: never contains a secret value
    url: str
    dialogs: list[DialogEvent] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class Resolved:
    """A locator bundle resolved on the live page.

    ``ref`` names the element like an observed one (so it can be acted on and risk-classified);
    it is None when the winning strategy was a screen coordinate.
    """

    ref: str | None
    info: ElementInfo | None
    strategy_index: int
    strategy_kind: str
    attempts: tuple[LocatorAttempt, ...]
    coordinates: tuple[float, float] | None = None


class Surface(Protocol):
    """What the agent loop and replay engine need from any application surface."""

    def observe(self) -> Observation: ...

    def act(self, action: Action) -> ActionResult: ...

    def element_info(self, ref: str) -> ElementInfo:
        """The facts about a ref from the latest observation (raises UnknownRefError)."""
        ...

    def wait_for_text(self, text: str, *, timeout_ms: int = 5000) -> bool: ...

    def set_request_guard(self, guard: RequestGuard | None) -> None:
        """Route every request the surface makes through ``guard``; blocked ones never leave."""
        ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


class LocatingSurface(Surface, Protocol):
    """A surface that can turn what it sees into locators, and locators back into elements."""

    def harvest(
        self, ref: str, *, params: Mapping[str, str] | None = None, for_click: bool = False
    ) -> LocatorBundle:
        """A ranked bundle for an observed element; every strategy was verified to find it."""
        ...

    def harvest_value(
        self, value: str, *, anchor_text: str | None = None, params: Mapping[str, str] | None = None
    ) -> LocatorBundle:
        """A bundle for text shown on the page, located through the row that labels it."""
        ...

    def resolve(self, bundle: LocatorBundle, values: Mapping[str, str] | None = None) -> Resolved:
        """The element a bundle finds now: strategies in order, the first unique match wins."""
        ...

    def read_text(self, ref: str) -> str:
        """The text (or value, for a field) of a ref, with secrets masked."""
        ...
