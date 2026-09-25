"""The Gemini adapter: what is sent, what comes back, how failures read, and that the key never
leaks. A stdlib HTTP server stands in for the API; nothing here needs a key or a network."""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from cua.evlog import EventLog, read_events
from cua.llm import (
    ImagePart,
    LLMConfigError,
    LLMError,
    Message,
    TextPart,
    ToolResult,
    ToolSpec,
    ToolUse,
)
from cua.llm.gemini_llm import GeminiLLM, clean_schema, list_models
from cua.redact import Redactor

KEY = "AIzaSy-test-key-1234567890abcdef"
PNG = b"\x89PNG\r\n\x1a\nfake-image-bytes"
TOOLS = (
    ToolSpec("observe", "Look at the page", {"type": "object", "properties": {}}),
    ToolSpec(
        "act",
        "Do one thing",
        {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["click", "fill"]},
                "ms": {"type": "integer", "description": "milliseconds"},
            },
            "required": ["kind"],
            "additionalProperties": False,
        },
    ),
)


class Fake:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.paths: list[str] = []
        self.replies: list[tuple[int, Any]] = []
        self.default: Any = {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 5},
        }


@pytest.fixture
def gemini() -> Iterator[tuple[Fake, str]]:
    fake = Fake()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return None

        def do_GET(self) -> None:
            fake.paths.append(self.path)
            fake.headers.append({k.lower(): v for k, v in self.headers.items()})
            status, reply = fake.replies.pop(0) if fake.replies else (200, fake.default)
            body = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            fake.paths.append(self.path)
            fake.headers.append({k.lower(): v for k, v in self.headers.items()})
            fake.requests.append(json.loads(self.rfile.read(length)))
            status, reply = fake.replies.pop(0) if fake.replies else (200, fake.default)
            body = reply if isinstance(reply, bytes) else json.dumps(reply).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    try:
        yield fake, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def make(host: str, **kwargs: Any) -> GeminiLLM:
    kwargs.setdefault("sleep", lambda _: None)
    return GeminiLLM(host=host, env={"GEMINI_API_KEY": KEY}, **kwargs)


def step(llm: GeminiLLM, *messages: Message) -> Any:
    return llm.step(
        system="be careful", messages=messages or (Message.user(TextPart("go")),), tools=TOOLS
    )


# --- the request --------------------------------------------------------------------------------


def test_the_key_travels_in_a_header_never_in_the_url(gemini: tuple[Fake, str]) -> None:
    fake, host = gemini
    step(make(host))
    assert fake.headers[0]["x-goog-api-key"] == KEY
    assert KEY not in fake.paths[0]
    assert fake.paths == ["/v1beta/models/gemini-3.5-flash:generateContent"]


def test_the_request_carries_the_system_prompt_tools_and_limits(gemini: tuple[Fake, str]) -> None:
    fake, host = gemini
    step(make(host, max_tokens=777))
    sent = fake.requests[0]
    assert sent["systemInstruction"] == {"parts": [{"text": "be careful"}]}
    assert sent["generationConfig"]["maxOutputTokens"] == 777
    assert sent["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}
    declarations = sent["tools"][0]["functionDeclarations"]
    assert [d["name"] for d in declarations] == ["observe", "act"]
    assert "parameters" not in declarations[0]  # an empty object schema is refused by the API
    assert declarations[1]["parameters"]["properties"]["kind"]["enum"] == ["click", "fill"]


def test_json_schema_is_cut_down_to_what_the_api_accepts() -> None:
    cleaned = clean_schema(
        {
            "$schema": "x",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "a": {"type": "string", "default": "z", "description": "d"},
                "b": {"type": "array", "items": {"type": "integer", "minimum": 0}},
            },
            "required": ["a"],
        }
    )
    assert cleaned == {
        "type": "object",
        "properties": {
            "a": {"type": "string", "description": "d"},
            "b": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        },
        "required": ["a"],
    }


def test_images_are_sent_inline(gemini: tuple[Fake, str]) -> None:
    fake, host = gemini
    step(make(host), Message.user(TextPart("page"), ImagePart(PNG)))
    parts = fake.requests[0]["contents"][0]["parts"]
    assert parts[0] == {"text": "page"}
    assert parts[1] == {
        "inlineData": {"mimeType": "image/png", "data": base64.b64encode(PNG).decode()}
    }


def test_a_tool_round_trip_uses_geminis_call_and_response_parts(gemini: tuple[Fake, str]) -> None:
    fake, host = gemini
    history = (
        Message.user(TextPart("go")),
        Message.assistant(TextPart("clicking"), ToolUse("call_1", "act", {"kind": "click"})),
        Message.user(
            ToolResult("call_1", (TextPart("clicked"), ImagePart(PNG))), TextPart("what next?")
        ),
    )
    llm = make(host)
    llm._names["call_1"] = "act"
    step(llm, *history)
    contents = fake.requests[0]["contents"]
    assert contents[1] == {
        "role": "model",
        "parts": [
            {"text": "clicking"},
            {"functionCall": {"name": "act", "args": {"kind": "click"}}},
        ],
    }
    reply = contents[2]
    assert reply["role"] == "user"
    assert reply["parts"][0] == {
        "functionResponse": {"name": "act", "response": {"output": "clicked"}}
    }
    assert reply["parts"][1]["inlineData"]["mimeType"] == "image/png"  # after the responses
    assert reply["parts"][2] == {"text": "what next?"}


def test_a_failed_tool_result_is_reported_as_an_error(gemini: tuple[Fake, str]) -> None:
    fake, host = gemini
    llm = make(host)
    llm._names["call_1"] = "act"
    step(
        llm,
        Message.user(TextPart("go")),
        Message.assistant(ToolUse("call_1", "act", {})),
        Message.user(ToolResult("call_1", (TextPart("no such ref"),), is_error=True)),
    )
    part = fake.requests[0]["contents"][2]["parts"][0]
    assert part["functionResponse"]["response"] == {"error": "no such ref"}


# --- the reply ----------------------------------------------------------------------------------


def test_calls_text_and_usage_come_back_and_are_counted(gemini: tuple[Fake, str]) -> None:
    fake, host = gemini
    fake.default = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "clicking"},
                        {
                            "functionCall": {"name": "act", "args": {"kind": "click"}},
                            "thoughtSignature": "sig-abc",
                        },
                        {"functionCall": {"name": "observe", "args": {}}},
                    ],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 20,
            "thoughtsTokenCount": 40,
        },
    }
    llm = make(host)
    reply = step(llm)
    assert reply.text == "clicking"
    assert [(c.name, c.input) for c in reply.tool_calls] == [
        ("act", {"kind": "click"}),
        ("observe", {}),
    ]
    assert len({c.id for c in reply.tool_calls}) == 2
    assert reply.stop_reason == "tool_use"
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (100, 60)  # thinking is billed
    assert (llm.meter.steps, llm.meter.usage.total_tokens) == (1, 160)


def test_a_thought_signature_is_handed_back_with_its_call(gemini: tuple[Fake, str]) -> None:
    fake, host = gemini
    fake.default = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "act", "args": {}}, "thoughtSignature": "sig-abc"}
                    ],
                }
            }
        ]
    }
    llm = make(host)
    first = step(llm)
    step(llm, Message.user(TextPart("go")), first.as_message())
    model_turn = fake.requests[1]["contents"][1]
    assert model_turn["parts"][0]["thoughtSignature"] == "sig-abc"


def test_thought_summaries_are_not_mistaken_for_the_answer(gemini: tuple[Fake, str]) -> None:
    fake, host = gemini
    fake.default["candidates"][0]["content"]["parts"] = [
        {"thought": True, "text": "private reasoning"},
        {"text": "the answer"},
    ]
    assert step(make(host)).text == "the answer"


def test_a_prompt_the_api_refused_is_an_error_that_says_why(gemini: tuple[Fake, str]) -> None:
    fake, host = gemini
    fake.default = {"promptFeedback": {"blockReason": "SAFETY"}}
    with pytest.raises(LLMError, match="SAFETY"):
        step(make(host))


def test_only_metadata_is_logged_never_prompts_or_replies(
    gemini: tuple[Fake, str], tmp_path: Path
) -> None:
    fake, host = gemini
    fake.default["candidates"][0]["content"]["parts"] = [{"text": "the balance is $2,480.15"}]
    log = EventLog(tmp_path / "log.jsonl", run_id="r", redactor=Redactor())
    step(make(host, log=log), Message.user(TextPart("member 12345 card 4111111111111111")))
    text = (tmp_path / "log.jsonl").read_text()
    event = read_events(log.path)[0]
    assert (event["event"], event["model"]) == ("llm_step", "gemini-3.5-flash")
    assert "2,480.15" not in text
    assert "4111" not in text
    assert KEY not in text


# --- failures and the key -----------------------------------------------------------------------


def test_a_missing_key_says_where_to_get_and_put_one() -> None:
    with pytest.raises(LLMConfigError) as caught:
        GeminiLLM(env={})
    message = str(caught.value)
    assert "GEMINI_API_KEY" in message
    assert ".env" in message
    assert "aistudio.google.com" in message


def test_google_api_key_is_accepted_too() -> None:
    assert GeminiLLM(env={"GOOGLE_API_KEY": KEY}).model == "gemini-3.5-flash"


def test_the_key_is_never_in_a_repr_or_an_error(gemini: tuple[Fake, str]) -> None:
    fake, host = gemini
    llm = make(host)
    assert KEY not in repr(llm)
    fake.replies = [(500, {"error": {"message": f"boom {KEY}"}})] * 5
    with pytest.raises(LLMError) as caught:
        step(llm)
    assert KEY not in str(caught.value)
    assert "boom" not in str(caught.value)


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_is_a_config_error(gemini: tuple[Fake, str], status: int) -> None:
    fake, host = gemini
    fake.replies = [(status, {"error": {"message": "API key not valid"}})]
    with pytest.raises(LLMConfigError, match="rejected"):
        step(make(host))


def test_a_rate_limit_is_waited_out_and_retried(gemini: tuple[Fake, str]) -> None:
    fake, host = gemini
    fake.replies = [(429, {"error": {"message": "quota"}}), (429, {"error": {}})]
    waits: list[float] = []
    reply = step(make(host, sleep=waits.append))
    assert reply.text == "ok"
    assert len(fake.requests) == 3
    assert len(waits) == 2
    assert 0 < waits[0] < waits[1]  # backing off


def test_a_rate_limit_that_never_clears_gives_up_and_says_so(gemini: tuple[Fake, str]) -> None:
    fake, host = gemini
    fake.replies = [(429, {"error": {}})] * 10
    with pytest.raises(LLMError, match="rate limit"):
        step(make(host))
    assert len(fake.requests) == 5  # the first try and four retries, no more


def test_a_server_error_is_retried_then_reported_without_the_body(gemini: tuple[Fake, str]) -> None:
    fake, host = gemini
    fake.replies = [(503, {"error": {"message": "overloaded 12345"}})] * 10
    with pytest.raises(LLMError) as caught:
        step(make(host))
    assert "HTTP 503" in str(caught.value)
    assert "12345" not in str(caught.value)


def test_a_bad_request_is_not_retried(gemini: tuple[Fake, str]) -> None:
    fake, host = gemini
    fake.replies = [(400, {"error": {"message": "bad schema"}})]
    with pytest.raises(LLMError, match="HTTP 400"):
        step(make(host))
    assert len(fake.requests) == 1


def test_a_reply_that_is_not_json_is_an_llm_error(gemini: tuple[Fake, str]) -> None:
    fake, host = gemini
    fake.replies = [(200, b"<html>proxy</html>")]
    with pytest.raises(LLMError, match="not valid JSON"):
        step(make(host))


def test_an_unreachable_server_is_an_llm_error() -> None:
    with pytest.raises(LLMError, match="could not reach"):
        step(make("http://127.0.0.1:1", timeout_s=2))


# --- a model name the API does not know ---------------------------------------------------------


def test_an_unknown_model_says_how_to_find_a_real_one(gemini: tuple[Fake, str]) -> None:
    fake, host = gemini
    fake.replies = [(404, {"error": {"message": "models/gemini-9 is not found"}})]
    with pytest.raises(LLMConfigError) as caught:
        step(make(host))
    assert "gemini-3.5-flash" in str(caught.value)
    assert "retired" in str(caught.value)
    assert "cua models --provider gemini" in str(caught.value)


def test_the_models_a_key_can_use_are_listed_newest_first_and_only_those_that_generate(
    gemini: tuple[Fake, str],
) -> None:
    fake, host = gemini
    fake.replies = [
        (
            200,
            {
                "models": [
                    {
                        "name": "models/gemini-2.0-flash",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/text-embedding-004",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                    {
                        "name": "models/gemini-3-flash",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                ],
                "nextPageToken": "page2",
            },
        ),
        (
            200,
            {
                "models": [
                    {
                        "name": "models/gemini-2.5-flash",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {"name": "models/imagen-4", "supportedGenerationMethods": ["predict"]},
                ]
            },
        ),
    ]
    names = list_models(env={"GEMINI_API_KEY": KEY}, host=host)
    assert names == ["gemini-3-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
    assert fake.headers[0]["x-goog-api-key"] == KEY
    assert KEY not in fake.paths[0]
    assert "pageToken=page2" in fake.paths[1]


def test_listing_models_needs_a_key_and_reports_a_rejected_one() -> None:
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        list_models(env={})


def test_listing_models_reports_a_rejected_key(gemini: tuple[Fake, str]) -> None:
    fake, host = gemini
    fake.replies = [(403, {"error": {"message": "nope"}})]
    with pytest.raises(LLMConfigError, match="rejected"):
        list_models(env={"GEMINI_API_KEY": KEY}, host=host)


# --- a flaky connection -------------------------------------------------------------------------


def test_a_dropped_connection_is_retried_and_the_run_carries_on(
    gemini: tuple[Fake, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import urllib.error
    import urllib.request

    _, host = gemini
    real = urllib.request.urlopen
    attempts: list[int] = []

    def flaky(request: Any, timeout: float = 0) -> Any:
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.URLError(ConnectionResetError("reset by peer"))
        return real(request, timeout=timeout)

    monkeypatch.setattr("cua.llm.gemini_llm.urllib.request.urlopen", flaky)
    waits: list[float] = []
    assert step(make(host, sleep=waits.append)).text == "ok"
    assert len(attempts) == 3
    assert len(waits) == 2


def test_the_kind_of_network_failure_is_named_but_nothing_else() -> None:
    with pytest.raises(LLMError) as caught:
        step(make("http://127.0.0.1:1", timeout_s=2))
    assert "ConnectionRefusedError" in str(caught.value)
    assert KEY not in str(caught.value)
