"""How the system perceives and acts on an application surface."""

from cua.surface.base import (
    Action,
    ActionResult,
    Box,
    DialogEvent,
    ElementInfo,
    FrameHop,
    FrameObservation,
    Observation,
    StaleRefError,
    Surface,
    SurfaceError,
    SurfaceUnavailable,
    UnknownRefError,
    UnknownSecretError,
)
from cua.surface.playwright_surface import PlaywrightSurface, SurfaceConfig

__all__ = [
    "Action",
    "ActionResult",
    "Box",
    "DialogEvent",
    "ElementInfo",
    "FrameHop",
    "FrameObservation",
    "Observation",
    "PlaywrightSurface",
    "StaleRefError",
    "Surface",
    "SurfaceConfig",
    "SurfaceError",
    "SurfaceUnavailable",
    "UnknownRefError",
    "UnknownSecretError",
]
