"""The recorded run: what discovery did, decoupled from the raw model transcript.

A run is steps (each with a verified locator bundle, a typed value reference and the page that
resulted), the outputs that were read, and how it ended. It is the input to artifact synthesis.
It holds no secret values (only their names) and no conversation.
"""

from __future__ import annotations

import hashlib
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field

from cua.artifact import LocatorBundle, RiskClass, ValueRef
from cua.common import StrictModel
from cua.surface import Observation

StepKind = Literal["navigate", "click", "fill", "select", "press", "wait_for", "extract"]
Outcome = Literal["finished", "failed", "max_steps", "timeout", "dead_end"]
_DIGEST_TEXT = 500


class FrameDigest(StrictModel):
    name: str | None
    path: list[str]
    url: str  # path and query only
    text: str
    elements: int


class ObservationDigest(StrictModel):
    """A page reduced to what a checkpoint or expectation could be built from."""

    url: str
    frames: list[FrameDigest]
    dialogs: list[str] = Field(default_factory=list)

    @classmethod
    def from_observation(cls, obs: Observation) -> ObservationDigest:
        def path_of(raw: str) -> str:
            parts = urlsplit(raw)
            return parts.path + (f"?{parts.query}" if parts.query else "")

        return cls(
            url=path_of(obs.url),
            frames=[
                FrameDigest(
                    name=frame.name,
                    path=[hop.name or f"#{hop.index}" for hop in frame.path],
                    url=path_of(frame.url),
                    text=frame.text[:_DIGEST_TEXT],
                    elements=len(frame.elements),
                )
                for frame in obs.frames
            ],
            dialogs=[f"{d.kind}: {d.message}" for d in obs.dialogs],
        )


def state_key(obs: Observation) -> str:
    """A fingerprint of everything on the page that a person could see change, typed values
    included, so a fill counts as progress and a click that did nothing does not."""
    digest = hashlib.sha256(obs.url.encode())
    for frame in obs.frames:
        digest.update(frame.url.encode())
        digest.update(frame.text.encode())
        for element in frame.elements:
            digest.update(element.attrs.get("value", "").encode())
            digest.update(str(element.checked).encode())
            digest.update(element.text.encode())
    return digest.hexdigest()


class RecordedDialog(StrictModel):
    kind: str
    message: str
    accepted: bool


class RecordedStep(StrictModel):
    id: str
    kind: StepKind
    description: str  # the model's stated reason: reviewable intent
    url: str | None = None
    locator: LocatorBundle | None = None
    value: ValueRef | None = None
    output: str | None = None
    expect_text: str | None = None
    risk: RiskClass
    dialogs: list[RecordedDialog] = Field(default_factory=list)
    after: ObservationDigest | None = None


class RecordedRun(StrictModel):
    run_id: str
    goal: str
    start_url: str
    model: str
    params: dict[str, str]
    secrets_used: list[str]
    steps: list[RecordedStep]
    outputs: dict[str, str]
    missing_outputs: list[str]
    outcome: Outcome
    reason: str
    summary: str = ""
    llm_steps: int
    tokens: int
    cost_usd: float | None
    duration_s: float
    final: ObservationDigest | None = None
