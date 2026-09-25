"""The tools the model can call, and the task it is given."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from cua.llm import ToolSpec

ACT_KINDS = ("navigate", "click", "fill", "select", "press", "wait", "wait_for")


@dataclass(frozen=True)
class ParamSpec:
    """A caller-supplied input. `value` is the example used while discovering; a sensitive one
    is typed by the harness but never shown to the model or kept in the recorded run."""

    value: str
    description: str
    pattern: str | None = None  # a regular expression the value must match, for the input schema
    sensitive: bool = False


@dataclass(frozen=True)
class OutputSpec:
    description: str
    sensitive: bool = False


@dataclass(frozen=True)
class DiscoveryTask:
    goal: str
    start_url: str
    params: Mapping[str, ParamSpec] = field(default_factory=dict)
    outputs: Mapping[str, OutputSpec] = field(default_factory=dict)
    secrets: Sequence[str] = ()


@dataclass(frozen=True)
class Limits:
    max_steps: int = 40  # model calls
    timeout_s: float = 300.0
    max_repeats: int = 3  # the same action, back to back
    max_errors: int = 4  # errors and refusals in a row
    max_stalled: int = 6  # successful actions that changed nothing on the page
    max_nudges: int = 2  # replies with no tool call


_ACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": list(ACT_KINDS)},
        "reason": {"type": "string", "description": "Why, in one short sentence."},
        "ref": {"type": "string", "description": "Element ref from the latest observation."},
        "x": {"type": "number", "description": "Screen x, only if the element has no ref."},
        "y": {"type": "number", "description": "Screen y, only if the element has no ref."},
        "url": {"type": "string", "description": "For navigate."},
        "text": {"type": "string", "description": "For fill: a constant to type."},
        "secret": {"type": "string", "description": "For fill: the name of a secret to type."},
        "param": {"type": "string", "description": "For fill: the name of a parameter to type."},
        "option": {"type": "string", "description": "For select: option label or value."},
        "key": {"type": "string", "description": "For press, for example Enter."},
        "ms": {"type": "integer", "description": "For wait: milliseconds."},
    },
    "required": ["kind", "reason"],
}

TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "observe",
        "Look at the page again: every frame's text and controls (with refs) and a screenshot.",
        {"type": "object", "properties": {}},
    ),
    ToolSpec(
        "act",
        "Do one thing on the page: navigate, click, fill, select, press, wait, or wait_for text. "
        "Returns the resulting page. Fill takes exactly one of text, secret or param.",
        _ACT_SCHEMA,
    ),
    ToolSpec(
        "extract",
        "Record an output you can see on the page. Give the exact displayed text as value and, "
        "if it appears more than once, anchor_text: the text of the row that labels it.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "One of the requested output names."},
                "value": {"type": "string", "description": "The value exactly as displayed."},
                "anchor_text": {"type": "string", "description": "Text of the labelling row."},
                "reason": {"type": "string"},
            },
            "required": ["name", "value"],
        },
    ),
    ToolSpec(
        "finish",
        "End the run. success is true only when the goal is achieved and every requested output "
        "was extracted.",
        {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "summary": {"type": "string", "description": "One sentence."},
            },
            "required": ["success", "summary"],
        },
    ),
)
