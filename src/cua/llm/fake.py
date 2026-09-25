"""A scripted model for tests and for running the system with no key and no network.

A script is a list of responses, or of functions that build a response from the request, so a
test can drive the real agent loop step by step (or read the page it is shown and answer it).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

from cua.llm.base import LLMError, LLMRequest, LLMResponse, Message, Meter, ToolSpec, ToolUse

ScriptStep = LLMResponse | Callable[[LLMRequest], LLMResponse]

_ids = itertools.count(1)


def say(text: str) -> LLMResponse:
    """A plain text answer with no tool call: the model saying it is done."""
    return LLMResponse(text=text, stop_reason="end_turn")


def tool_call(name: str, /, **arguments: Any) -> LLMResponse:
    """The model asking for one tool to be run."""
    call = ToolUse(id=f"tu_fake_{next(_ids)}", name=name, input=dict(arguments))
    return LLMResponse(text="", tool_calls=(call,), stop_reason="tool_use")


class FakeLLM:
    model = "fake"

    def __init__(self, script: Sequence[ScriptStep]) -> None:
        self._script = list(script)
        self.calls: list[LLMRequest] = []
        self.meter = Meter()

    def step(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        request = LLMRequest(system, tuple(messages), tuple(tools))
        self.calls.append(request)
        if not self._script:
            raise LLMError("FakeLLM script exhausted: the loop asked for more steps than scripted")
        step = self._script.pop(0)
        response = step(request) if callable(step) else step
        self.meter.record(self.model, response.usage)
        return response
