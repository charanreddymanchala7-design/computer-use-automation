"""A local model served by Ollama, behind the same ``LLM`` protocol.

Why this exists: discovery has to be a real model run, and a local open model needs no API key,
no account and sends no page content off the machine. Trade-offs, stated plainly:

* The model here is text-only, so screenshots are left out. The page description the harness
  renders (frames, roles, labels, element refs) is what it reads, and it is the same text the
  hosted model gets alongside the image.
* An 8B model is weaker than a hosted one. Discovery has hard limits and dead-end detection
  because of that, and the run summary records which model produced the capability.
* ``temperature`` is 0 and a seed is fixed, since a local model, unlike the newest hosted ones,
  accepts them. Reproducibility of *replay* still comes from the artifact, not the model.

Only metadata is logged, as for the hosted model. Standard library only: no new dependency.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from cua.evlog import EventLog
from cua.llm.base import (
    ImagePart,
    LLMConfigError,
    LLMError,
    LLMResponse,
    Message,
    Meter,
    TextPart,
    ToolResult,
    ToolSpec,
    ToolUse,
    Usage,
)

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_HOST = "http://127.0.0.1:11434"


def _text(parts: Sequence[TextPart | ImagePart]) -> str:
    return "\n".join(p.text for p in parts if isinstance(p, TextPart))


def to_ollama_messages(system: str, messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Ollama's chat shape: tool results are their own ``tool`` messages, images are dropped."""
    sent: list[dict[str, Any]] = [{"role": "system", "content": system}]
    names: dict[str, str] = {}  # tool call id -> tool name, so a result can say which tool ran
    for message in messages:
        if message.role == "assistant":
            calls = [p for p in message.parts if isinstance(p, ToolUse)]
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": _text([p for p in message.parts if isinstance(p, TextPart)]),
            }
            if calls:
                entry["tool_calls"] = [
                    {"function": {"name": c.name, "arguments": c.input}} for c in calls
                ]
                names.update({c.id: c.name for c in calls})
            sent.append(entry)
            continue
        for part in message.parts:
            if isinstance(part, ToolResult):
                body = _text(part.content)
                sent.append(
                    {
                        "role": "tool",
                        "content": f"ERROR: {body}" if part.is_error else body,
                        "tool_name": names.get(part.tool_use_id, ""),
                    }
                )
        rest = _text([p for p in message.parts if isinstance(p, TextPart)])
        if rest:
            sent.append({"role": "user", "content": rest})
    return sent


def to_ollama_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


class OllamaLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        host: str | None = None,
        env: Mapping[str, str] | None = None,
        num_ctx: int = 16384,
        max_tokens: int = 1024,
        timeout_s: float = 300.0,
        log: EventLog | None = None,
    ) -> None:
        self.model = model
        self.meter = Meter()
        chosen = host or (env if env is not None else os.environ).get("OLLAMA_HOST") or DEFAULT_HOST
        self.host = chosen if "://" in chosen else f"http://{chosen}"
        self._num_ctx = num_ctx
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._log = log
        self._calls = 0

    def __repr__(self) -> str:
        return f"OllamaLLM(model={self.model!r}, host={self.host!r})"

    def step(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": to_ollama_messages(system, messages),
            "tools": to_ollama_tools(tools),
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": 7,
                "num_ctx": self._num_ctx,
                "num_predict": self._max_tokens,
            },
        }
        reply = self._post("/api/chat", payload)
        message = reply.get("message") or {}
        calls: list[ToolUse] = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            self._calls += 1
            calls.append(
                ToolUse(
                    f"call_{self._calls}",
                    str(function.get("name", "")),
                    _arguments(function.get("arguments")),
                )
            )
        usage = Usage(
            input_tokens=int(reply.get("prompt_eval_count") or 0),
            output_tokens=int(reply.get("eval_count") or 0),
        )
        stop = "tool_use" if calls else str(reply.get("done_reason") or "stop")
        response = LLMResponse(str(message.get("content") or ""), tuple(calls), usage, stop)
        self.meter.record(self.model, usage)
        self._record(response)
        return response

    def installed_models(self) -> list[str]:
        """The models this Ollama has pulled."""
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=10) as response:
                data = json.loads(response.read())
        except (urllib.error.URLError, OSError, ValueError):
            raise LLMConfigError(
                f"Ollama is not reachable at {self.host}: start it with `ollama serve`"
            ) from None
        return sorted(str(m.get("name", "")) for m in data.get("models") or [])

    # --- internals -------------------------------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise LLMConfigError(
                    f"the model {self.model!r} is not installed: run `ollama pull {self.model}`"
                ) from None
            raise LLMError(f"Ollama call failed (HTTP {exc.code})") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, socket.timeout | TimeoutError):
                raise LLMError("Ollama did not answer in time") from None
            raise LLMConfigError(
                f"Ollama is not reachable at {self.host}: start it with `ollama serve`"
            ) from None
        except (TimeoutError, OSError):
            raise LLMError("Ollama did not answer in time") from None
        try:
            data = json.loads(raw)
        except ValueError:
            raise LLMError("Ollama replied with something that is not valid JSON") from None
        if not isinstance(data, dict):
            raise LLMError("Ollama replied with something that is not valid JSON")
        return data

    def _record(self, response: LLMResponse) -> None:
        if self._log is None:
            return
        self._log.emit(
            "llm_step",
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=0,
            cache_write_tokens=0,
            stop_reason=response.stop_reason,
            tool_calls=[call.name for call in response.tool_calls],
        )
