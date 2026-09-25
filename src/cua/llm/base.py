"""The model boundary: one decision step in, one response out, with usage counted.

The agent loop talks to ``LLM`` and nothing else, so it is provider-agnostic and every test can
run against a scripted fake with no key and no network.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Self


class LLMError(Exception):
    """A model call failed. Never carries request bodies, headers or keys."""


class LLMConfigError(LLMError):
    """The model cannot be used as configured (for example, no API key)."""


# --- conversation -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    png: bytes


@dataclass(frozen=True)
class ToolUse:
    """The model asking for a tool to be run."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """What running the tool produced, linked back to the call by id."""

    tool_use_id: str
    content: tuple[TextPart | ImagePart, ...]
    is_error: bool = False


Part = TextPart | ImagePart | ToolUse | ToolResult


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    parts: tuple[Part, ...]

    @classmethod
    def user(cls, *parts: Part) -> Self:
        return cls("user", parts)

    @classmethod
    def assistant(cls, *parts: Part) -> Self:
        return cls("assistant", parts)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


# --- usage and cost ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


# USD per million (input, output) tokens, as reported by research on 2026-09-25. These are
# estimates for the run summary, not billing: check current pricing before relying on them.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-fable-5-1": (10.0, 50.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}
CACHE_READ_FACTOR = 0.1  # a cached prompt token is billed at a tenth of the input price
CACHE_WRITE_FACTOR = 1.25  # writing to the cache costs a quarter more than a plain input token


def estimate_cost(model: str, usage: Usage) -> float | None:
    """Estimated USD for this usage, or None for a model whose price is not known."""
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    price_in, price_out = prices
    cost = (
        usage.input_tokens * price_in
        + usage.output_tokens * price_out
        + usage.cache_read_tokens * price_in * CACHE_READ_FACTOR
        + usage.cache_write_tokens * price_in * CACHE_WRITE_FACTOR
    )
    return cost / 1_000_000


class Meter:
    """Running totals for one run: steps, tokens and an estimated cost."""

    def __init__(self) -> None:
        self.steps = 0
        self.usage = Usage()
        self._cost = 0.0
        self._cost_known = True

    def record(self, model: str, usage: Usage) -> None:
        self.steps += 1
        self.usage = self.usage + usage
        cost = estimate_cost(model, usage)
        if cost is None:
            self._cost_known = False
        else:
            self._cost += cost

    @property
    def cost_usd(self) -> float | None:
        return self._cost if self._cost_known else None


# --- the call ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMRequest:
    system: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tool_calls: tuple[ToolUse, ...] = ()
    usage: Usage = Usage()
    stop_reason: str = "end_turn"

    def as_message(self) -> Message:
        """The assistant turn to append to the history before answering its tool calls."""
        parts: list[Part] = [TextPart(self.text)] if self.text else []
        parts.extend(self.tool_calls)
        return Message.assistant(*parts)


class LLM(Protocol):
    model: str
    meter: Meter

    def step(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse: ...
