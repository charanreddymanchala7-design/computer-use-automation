"""Google's Gemini API behind the ``LLM`` protocol, for anyone without a hosted-Claude key.

Google AI Studio issues a free key without a card, which is why this exists: discovery must be a
real model run and the take-home should not depend on a paid account. Standard library only.

* The key comes from the environment mapping it is given (``GEMINI_API_KEY`` or
  ``GOOGLE_API_KEY``), goes out in a header rather than the URL, and is never logged, echoed in an
  error or included in a repr.
* Only metadata is logged, as for the other models.
* Free-tier limits are per minute, so a 429 is waited out and retried a few times, backing off.
* Thought signatures that Gemini attaches to a function call are handed back with it, because
  newer models require that on the next turn.
"""

from __future__ import annotations

import base64
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
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

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_HOST = "https://generativelanguage.googleapis.com"
_RETRY_WAITS = (5.0, 15.0, 30.0, 60.0)  # after a 429 or a 5xx: at most four retries
_ALLOWED_SCHEMA_KEYS = frozenset(
    {"type", "description", "properties", "required", "enum", "items", "format", "nullable"}
    | {"minimum", "maximum"}
)


def clean_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """The subset of JSON Schema that Gemini's function declarations accept."""
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _ALLOWED_SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, Mapping):
            cleaned[key] = {name: clean_schema(sub) for name, sub in value.items()}
        elif key == "items" and isinstance(value, Mapping):
            cleaned[key] = clean_schema(value)
        else:
            cleaned[key] = value
    return cleaned


def _image(png: bytes) -> dict[str, Any]:
    return {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(png).decode()}}


def _declarations(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    declared: list[dict[str, Any]] = []
    for tool in tools:
        entry: dict[str, Any] = {"name": tool.name, "description": tool.description}
        if tool.input_schema.get("properties"):  # an empty object schema is refused by the API
            entry["parameters"] = clean_schema(tool.input_schema)
        declared.append(entry)
    return [{"functionDeclarations": declared}] if declared else []


class GeminiLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        env: Mapping[str, str] | None = None,
        host: str | None = None,
        max_tokens: int = 2048,
        timeout_s: float = 120.0,
        log: EventLog | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        source = env if env is not None else os.environ
        key = (source.get("GEMINI_API_KEY") or source.get("GOOGLE_API_KEY") or "").strip()
        if not key:
            raise LLMConfigError(
                "GEMINI_API_KEY is not set: get a free key at https://aistudio.google.com/apikey "
                "and put it in the gitignored .env file (see .env.example)"
            )
        self.model = model
        self.meter = Meter()
        self._key = key
        self._host = (host or DEFAULT_HOST).rstrip("/")
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._log = log
        self._sleep = sleep
        self._calls = 0
        self._names: dict[str, str] = {}  # tool call id -> tool name
        self._signatures: dict[str, str] = {}  # tool call id -> Gemini's thought signature

    def __repr__(self) -> str:
        return f"GeminiLLM(model={self.model!r})"  # deliberately says nothing about the key

    def step(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [self._content(m) for m in messages],
            "generationConfig": {"maxOutputTokens": self._max_tokens},
        }
        declared = _declarations(tools)
        if declared:
            payload["tools"] = declared
            payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
        reply = self._post(payload)
        response = self._parse(reply)
        self.meter.record(self.model, response.usage)
        self._record(response)
        return response

    # --- conversion ------------------------------------------------------------------------------

    def _content(self, message: Message) -> dict[str, Any]:
        if message.role == "assistant":
            parts: list[dict[str, Any]] = []
            for part in message.parts:
                if isinstance(part, TextPart):
                    parts.append({"text": part.text})
                elif isinstance(part, ToolUse):
                    call: dict[str, Any] = {"functionCall": {"name": part.name, "args": part.input}}
                    if part.id in self._signatures:
                        call["thoughtSignature"] = self._signatures[part.id]
                    parts.append(call)
            return {"role": "model", "parts": parts}
        responses: list[dict[str, Any]] = []
        result_images: list[dict[str, Any]] = []
        own: list[dict[str, Any]] = []
        for part in message.parts:
            if isinstance(part, ToolResult):
                text = "\n".join(p.text for p in part.content if isinstance(p, TextPart))
                responses.append(
                    {
                        "functionResponse": {
                            "name": self._names.get(part.tool_use_id, "unknown"),
                            "response": {"error" if part.is_error else "output": text},
                        }
                    }
                )
                result_images += [_image(p.png) for p in part.content if isinstance(p, ImagePart)]
            elif isinstance(part, TextPart):
                own.append({"text": part.text})
            elif isinstance(part, ImagePart):
                own.append(_image(part.png))
        # responses must directly follow the call they answer; pictures and words come after
        return {"role": "user", "parts": [*responses, *result_images, *own]}

    def _parse(self, reply: dict[str, Any]) -> LLMResponse:
        candidates = reply.get("candidates") or []
        if not candidates:
            reason = (reply.get("promptFeedback") or {}).get("blockReason")
            raise LLMError(
                f"the prompt was blocked ({reason})" if reason else "the model returned no answer"
            )
        candidate = candidates[0]
        texts: list[str] = []
        calls: list[ToolUse] = []
        for part in (candidate.get("content") or {}).get("parts") or []:
            if part.get("thought"):
                continue  # a summary of its reasoning, not the answer
            if "functionCall" in part:
                function = part["functionCall"]
                self._calls += 1
                call = ToolUse(
                    f"call_{self._calls}",
                    str(function.get("name", "")),
                    dict(function.get("args") or {}),
                )
                calls.append(call)
                self._names[call.id] = call.name
                if part.get("thoughtSignature"):
                    self._signatures[call.id] = str(part["thoughtSignature"])
            elif "text" in part:
                texts.append(str(part["text"]))
        usage_raw = reply.get("usageMetadata") or {}
        usage = Usage(
            input_tokens=int(usage_raw.get("promptTokenCount") or 0),
            # reasoning tokens are billed and counted as output
            output_tokens=int(usage_raw.get("candidatesTokenCount") or 0)
            + int(usage_raw.get("thoughtsTokenCount") or 0),
        )
        stop = "tool_use" if calls else str(candidate.get("finishReason") or "STOP").lower()
        return LLMResponse("\n".join(texts), tuple(calls), usage, stop)

    # --- transport -------------------------------------------------------------------------------

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._host}/v1beta/models/{self.model}:generateContent"
        body = json.dumps(payload).encode()
        for attempt in range(len(_RETRY_WAITS) + 1):
            request = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json", "x-goog-api-key": self._key},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                    raw = response.read()
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise LLMConfigError(
                        f"the API rejected the key (HTTP {exc.code}): check GEMINI_API_KEY"
                    ) from None
                if exc.code == 404:
                    raise LLMConfigError(
                        f"Gemini does not know the model {self.model!r} (HTTP 404): run "
                        "`cua models --provider gemini` to see the ones this key can use, then "
                        "pass one with --model"
                    ) from None
                retryable = exc.code == 429 or exc.code >= 500
                if retryable and attempt < len(_RETRY_WAITS):
                    self._sleep(_RETRY_WAITS[attempt])
                    continue
                if exc.code == 429:
                    raise LLMError(
                        "Gemini rate limit (HTTP 429): the free tier allows only a few requests "
                        "a minute; try again shortly"
                    ) from None
                raise LLMError(f"Gemini call failed (HTTP {exc.code})") from None
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, socket.timeout | TimeoutError):
                    raise LLMError("Gemini did not answer in time") from None
                raise LLMError("could not reach the Gemini API") from None
            except (TimeoutError, OSError):
                raise LLMError("Gemini did not answer in time") from None
            try:
                data = json.loads(raw)
            except ValueError:
                raise LLMError("Gemini replied with something that is not valid JSON") from None
            if not isinstance(data, dict):
                raise LLMError("Gemini replied with something that is not valid JSON")
            return data
        raise LLMError("Gemini call failed")  # unreachable: the loop returns or raises

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


def _version(name: str) -> float:
    match = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(match.group(1)) if match else 0.0


def list_models(*, env: Mapping[str, str] | None = None, host: str | None = None) -> list[str]:
    """The Gemini models this key may call ``generateContent`` on, newest version first."""
    source = env if env is not None else os.environ
    key = (source.get("GEMINI_API_KEY") or source.get("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise LLMConfigError(
            "GEMINI_API_KEY is not set: get a free key at https://aistudio.google.com/apikey "
            "and put it in the gitignored .env file (see .env.example)"
        )
    base = (host or DEFAULT_HOST).rstrip("/")
    names: list[str] = []
    token = ""
    while True:
        url = f"{base}/v1beta/models?pageSize=100" + (f"&pageToken={token}" if token else "")
        request = urllib.request.Request(url, headers={"x-goog-api-key": key}, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 401, 403):
                raise LLMConfigError(
                    f"the API rejected the key (HTTP {exc.code}): check GEMINI_API_KEY"
                ) from None
            raise LLMError(f"listing models failed (HTTP {exc.code})") from None
        except (urllib.error.URLError, OSError, ValueError):
            raise LLMError("could not list the Gemini models") from None
        for model in data.get("models") or []:
            if "generateContent" in (model.get("supportedGenerationMethods") or []):
                names.append(str(model.get("name", "")).removeprefix("models/"))
        token = str(data.get("nextPageToken") or "")
        if not token:
            break
    return sorted({n for n in names if n.startswith("gemini")}, key=lambda n: (-_version(n), n))
