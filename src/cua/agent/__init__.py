"""The discovery agent: an LLM observes and acts through the gateway, and the run is recorded."""

from cua.agent.loop import DiscoveryLoop, compact
from cua.agent.prompts import NUDGE, SYSTEM_PROMPT, first_message
from cua.agent.recording import (
    FrameDigest,
    ObservationDigest,
    RecordedDialog,
    RecordedRun,
    RecordedStep,
    state_key,
)
from cua.agent.render import render_observation
from cua.agent.tools import (
    ACT_KINDS,
    TOOLS,
    DiscoveryTask,
    Limits,
    OutputSpec,
    ParamSpec,
)

__all__ = [
    "ACT_KINDS",
    "NUDGE",
    "SYSTEM_PROMPT",
    "TOOLS",
    "DiscoveryLoop",
    "DiscoveryTask",
    "FrameDigest",
    "Limits",
    "ObservationDigest",
    "OutputSpec",
    "ParamSpec",
    "RecordedDialog",
    "RecordedRun",
    "RecordedStep",
    "compact",
    "first_message",
    "render_observation",
    "state_key",
]
