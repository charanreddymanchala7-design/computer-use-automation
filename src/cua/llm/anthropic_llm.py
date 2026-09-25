"""The Anthropic Messages API behind the ``LLM`` protocol.

* The key is read from the environment mapping it is given, never from a file in the repo, and it
  is never logged, echoed in an error or included in a repr.
* Only metadata is logged (model, token counts, stop reason, tool names). Prompts and replies can
  contain page content, so they never reach a log from here.
* No sampling parameters are sent: the newest models reject non-default temperature/top_p/top_k,
  so reproducibility comes from deterministic replay, not from the model.
* The system prompt and the tool list carry cache breakpoints, so the long stable prefix of every
  step is billed at the cached rate.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Mapping, Sequence
from typing import Any

import anthropic
from anthropic.types import CacheControlEphemeralParam

from cua.evlog import EventLog
from cua.llm.base import (
    ImagePart,
    LLMConfigError,
    LLMError,
    LLMResponse,
    Message,
    Meter,
    Part,
    TextPart,
    ToolSpec,
    ToolUse,
    Usage,
    estimate_cost,
)

_EPHEMERAL: CacheControlEphemeralParam = {"type": "ephemeral"}


def _content(part: TextPart | ImagePart) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(part.png).decode(),
        },
    }


def _part(part: Part) -> dict[str, Any]:
    if isinstance(part, TextPart | ImagePart):
        return _content(part)
    if isinstance(part, ToolUse):
        return {"type": "tool_use", "id": part.id, "name": part.name, "input": part.input}
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": part.tool_use_id,
        "content": [_content(p) for p in part.content],
    }
    if part.is_error:
        block["is_error"] = True
    return block


def to_api_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    return [{"role": m.role, "content": [_part(p) for p in m.parts]} for m in messages]


def to_api_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    payload = [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]
    if payload:
        payload[-1]["cache_control"] = _EPHEMERAL  # caches the system prompt and every tool
    return payload


class AnthropicLLM:
    def __init__(
        self,
        model: str,
        *,
        client: anthropic.Anthropic | None = None,
        env: Mapping[str, str] | None = None,
        max_tokens: int = 1024,
        timeout_s: float = 60.0,
        log: EventLog | None = None,
    ) -> None:
        self.model = model
        self.meter = Meter()
        self._max_tokens = max_tokens
        self._log = log
        if client is None:
            key = (env if env is not None else os.environ).get("ANTHROPIC_API_KEY", "").strip()
            if not key:
                raise LLMConfigError(
                    "ANTHROPIC_API_KEY is not set: put it in the gitignored .env file "
                    "(see .env.example) or export it in your shell"
                )
            client = anthropic.Anthropic(api_key=key, max_retries=3, timeout=timeout_s)
        self.client = client

    def __repr__(self) -> str:
        return f"AnthropicLLM(model={self.model!r})"  # deliberately says nothing about the key

    def step(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        try:
            reply = self.client.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                system=[{"type": "text", "text": system, "cache_control": _EPHEMERAL}],
                messages=to_api_messages(messages),  # type: ignore[arg-type]
                tools=to_api_tools(tools),  # type: ignore[arg-type]
            )
        except anthropic.APIError as exc:
            status = getattr(exc, "status_code", None)
            raise LLMError(
                f"{type(exc).__name__}: the model call failed"
                + (f" (HTTP {status})" if status else "")
            ) from exc

        blocks = list(reply.content)
        text = "\n".join(b.text for b in blocks if b.type == "text")
        calls = tuple(ToolUse(b.id, b.name, dict(b.input)) for b in blocks if b.type == "tool_use")
        raw = reply.usage
        usage = Usage(
            input_tokens=raw.input_tokens or 0,
            output_tokens=raw.output_tokens or 0,
            cache_read_tokens=getattr(raw, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(raw, "cache_creation_input_tokens", 0) or 0,
        )
        response = LLMResponse(text, calls, usage, reply.stop_reason or "")
        self.meter.record(self.model, usage)
        self._record(response)
        return response

    def _record(self, response: LLMResponse) -> None:
        if self._log is None:
            return
        fields: dict[str, Any] = {
            "model": self.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_tokens": response.usage.cache_read_tokens,
            "cache_write_tokens": response.usage.cache_write_tokens,
            "stop_reason": response.stop_reason,
            "tool_calls": [call.name for call in response.tool_calls],
        }
        cost = estimate_cost(self.model, response.usage)
        if cost is not None:
            fields["cost_usd"] = round(cost, 6)
        self._log.emit("llm_step", **fields)
