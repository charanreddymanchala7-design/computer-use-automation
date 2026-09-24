"""The surface seam: how the system perceives and acts on an application.

Everything above this line (the agent loop, the recorded artifact, replay) talks to a
``Surface``. Only the implementation below it knows whether the app is a web page, a legacy
frameset or, later, a desktop window. A web surface observes frames and DOM facts; a desktop
surface would observe an accessibility tree and pixels; both hand back the same ``Observation``
and accept the same ``Action`` vocabulary, targeting elements by ref or by screen coordinates.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal, Protocol, Self

# --- errors -----------------------------------------------------------------------------------


class SurfaceError(Exception):
    """Something the surface could not do; the message says what to do next."""


class SurfaceUnavailable(SurfaceError):
    """The underlying driver (e.g. the browser binary) is not available on this machine."""


class UnknownRefError(SurfaceError):
    """The ref was never issued, or was issued before the last reset."""


class StaleRefError(SurfaceError):
    """The ref existed but its element is gone (the page changed); observe again."""


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


class Surface(Protocol):
    """What the agent loop and replay engine need from any application surface."""

    def observe(self) -> Observation: ...

    def act(self, action: Action) -> ActionResult: ...

    def wait_for_text(self, text: str, *, timeout_ms: int = 5000) -> bool: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...
