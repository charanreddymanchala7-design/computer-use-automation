"""The local-model adapter: what is sent to Ollama, what comes back, and how failures read.

A stdlib HTTP server stands in for Ollama and records exactly what it was sent, so nothing here
needs a model or a network."""

from __future__ import annotations

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
from cua.llm.ollama_llm import OllamaLLM
from cua.redact import Redactor

TOOLS = (
    ToolSpec(
        "act",
        "Do one thing",
        {"type": "object", "properties": {"kind": {"type": "string"}}, "required": ["kind"]},
    ),
)


class Fake:
    """A stand-in Ollama: canned replies, and a record of every request."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self.status = 200
        self.reply: Any = {
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 11,
            "eval_count": 5,
        }


@pytest.fixture
def ollama() -> Iterator[tuple[Fake, str]]:
    fake = Fake()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return None

        def do_GET(self) -> None:
            body = json.dumps(
                {"models": [{"name": "qwen2.5:14b"}, {"name": "llama3.1:8b"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            fake.paths.append(self.path)
            fake.requests.append(json.loads(self.rfile.read(length)))
            body = fake.reply if isinstance(fake.reply, bytes) else json.dumps(fake.reply).encode()
            self.send_response(fake.status)
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


def step(llm: OllamaLLM, *messages: Message) -> Any:
    return llm.step(
        system="be careful", messages=messages or (Message.user(TextPart("go")),), tools=TOOLS
    )


def test_the_request_names_the_model_the_tools_and_a_context_big_enough_for_a_page(
    ollama: tuple[Fake, str],
) -> None:
    fake, host = ollama
    step(OllamaLLM("llama3.1:8b", host=host, num_ctx=16384))
    assert fake.paths == ["/api/chat"]
    sent = fake.requests[0]
    assert sent["model"] == "llama3.1:8b"
    assert sent["stream"] is False
    # Ollama's default context is small enough to silently cut a page description off
    assert sent["options"]["num_ctx"] == 16384
    assert sent["options"]["temperature"] == 0
    assert sent["messages"][0] == {"role": "system", "content": "be careful"}
    assert sent["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "act",
                "description": "Do one thing",
                "parameters": TOOLS[0].input_schema,
            },
        }
    ]


def test_screenshots_are_left_out_because_the_local_model_reads_text(
    ollama: tuple[Fake, str],
) -> None:
    fake, host = ollama
    step(
        OllamaLLM(host=host),
        Message.user(TextPart("page: Member No"), ImagePart(b"\x89PNG-fake")),
    )
    user = fake.requests[0]["messages"][1]
    assert user == {"role": "user", "content": "page: Member No"}
    assert "images" not in json.dumps(fake.requests[0])


def test_a_tool_round_trip_is_sent_in_ollamas_shape(ollama: tuple[Fake, str]) -> None:
    fake, host = ollama
    history = (
        Message.user(TextPart("go")),
        Message.assistant(TextPart("clicking"), ToolUse("call_1", "act", {"kind": "click"})),
        Message.user(
            ToolResult("call_1", (TextPart("clicked"), ImagePart(b"png"))),
            TextPart("now what?"),
        ),
    )
    step(OllamaLLM(host=host), *history)
    sent = fake.requests[0]["messages"]
    assert sent[2] == {
        "role": "assistant",
        "content": "clicking",
        "tool_calls": [{"function": {"name": "act", "arguments": {"kind": "click"}}}],
    }
    assert sent[3] == {"role": "tool", "content": "clicked", "tool_name": "act"}
    assert sent[4] == {"role": "user", "content": "now what?"}  # the rest of that turn follows


def test_a_failed_tool_result_says_so_in_words(ollama: tuple[Fake, str]) -> None:
    fake, host = ollama
    history = (
        Message.user(TextPart("go")),
        Message.assistant(ToolUse("call_1", "act", {})),
        Message.user(ToolResult("call_1", (TextPart("no such ref"),), is_error=True)),
    )
    step(OllamaLLM(host=host), *history)
    assert fake.requests[0]["messages"][-1]["content"] == "ERROR: no such ref"


def test_tool_calls_text_and_usage_come_back_and_are_counted(ollama: tuple[Fake, str]) -> None:
    fake, host = ollama
    fake.reply = {
        "message": {
            "role": "assistant",
            "content": "let me click",
            "tool_calls": [
                {"function": {"name": "act", "arguments": {"kind": "click", "ref": "e1"}}},
                {"function": {"name": "act", "arguments": '{"kind": "fill"}'}},  # a JSON string
            ],
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 120,
        "eval_count": 30,
    }
    llm = OllamaLLM(host=host)
    reply = step(llm)
    assert reply.text == "let me click"
    assert [(c.name, c.input) for c in reply.tool_calls] == [
        ("act", {"kind": "click", "ref": "e1"}),
        ("act", {"kind": "fill"}),
    ]
    assert len({c.id for c in reply.tool_calls}) == 2  # Ollama gives none, so they are made up
    assert reply.stop_reason == "tool_use"
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (120, 30)
    assert (llm.meter.steps, llm.meter.usage.total_tokens) == (1, 150)


def test_ids_stay_unique_across_steps(ollama: tuple[Fake, str]) -> None:
    fake, host = ollama
    fake.reply = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "act", "arguments": {}}}],
        },
        "done": True,
    }
    llm = OllamaLLM(host=host)
    first, second = step(llm), step(llm)
    assert first.tool_calls[0].id != second.tool_calls[0].id


def test_arguments_that_are_not_an_object_become_empty_so_the_loop_can_say_so(
    ollama: tuple[Fake, str],
) -> None:
    fake, host = ollama
    fake.reply = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "act", "arguments": "not json at all"}}],
        },
        "done": True,
    }
    assert step(OllamaLLM(host=host)).tool_calls[0].input == {}


def test_a_plain_answer_has_no_tool_calls(ollama: tuple[Fake, str]) -> None:
    _, host = ollama
    reply = step(OllamaLLM(host=host))
    assert (reply.text, reply.tool_calls, reply.stop_reason) == ("ok", (), "stop")


def test_only_metadata_is_logged_never_prompts_or_replies(
    ollama: tuple[Fake, str], tmp_path: Path
) -> None:
    fake, host = ollama
    fake.reply["message"]["content"] = "the balance is $2,480.15"
    log = EventLog(tmp_path / "log.jsonl", run_id="r", redactor=Redactor())
    step(OllamaLLM(host=host, log=log), Message.user(TextPart("member 12345 has 4111111111111111")))
    text = (tmp_path / "log.jsonl").read_text()
    event = read_events(log.path)[0]
    assert event["event"] == "llm_step"
    assert (event["model"], event["input_tokens"], event["output_tokens"]) == ("llama3.1:8b", 11, 5)
    assert "2,480.15" not in text
    assert "4111" not in text


# --- when it cannot be used ----------------------------------------------------------------------


def test_ollama_not_running_says_how_to_start_it() -> None:
    llm = OllamaLLM(host="http://127.0.0.1:1", timeout_s=2)  # nothing listens on port 1
    with pytest.raises(LLMConfigError, match="ollama serve"):
        step(llm)


def test_a_model_that_is_not_installed_says_how_to_get_it(ollama: tuple[Fake, str]) -> None:
    fake, host = ollama
    fake.status, fake.reply = 404, {"error": "model 'nope' not found"}
    with pytest.raises(LLMConfigError, match="ollama pull nope"):
        step(OllamaLLM("nope", host=host))


def test_a_server_error_is_an_llm_error_without_the_body(ollama: tuple[Fake, str]) -> None:
    fake, host = ollama
    fake.status, fake.reply = 500, {"error": "boom with member 12345"}
    with pytest.raises(LLMError) as caught:
        step(OllamaLLM(host=host))
    assert "HTTP 500" in str(caught.value)
    assert "12345" not in str(caught.value)


def test_a_reply_that_is_not_json_is_an_llm_error(ollama: tuple[Fake, str]) -> None:
    fake, host = ollama
    fake.reply = b"<html>proxy error</html>"
    with pytest.raises(LLMError, match="not valid JSON"):
        step(OllamaLLM(host=host))


def test_a_host_without_a_scheme_is_accepted_and_the_environment_is_read(
    ollama: tuple[Fake, str],
) -> None:
    fake, host = ollama
    bare = host.removeprefix("http://")
    step(OllamaLLM(env={"OLLAMA_HOST": bare}))
    assert fake.paths == ["/api/chat"]


def test_the_default_host_is_the_local_one() -> None:
    assert OllamaLLM(env={}).host == "http://127.0.0.1:11434"


def test_the_installed_models_are_listed_sorted(ollama: tuple[Fake, str]) -> None:
    _, host = ollama
    assert OllamaLLM(host=host).installed_models() == ["llama3.1:8b", "qwen2.5:14b"]


def test_listing_models_when_ollama_is_down_says_how_to_start_it() -> None:
    with pytest.raises(LLMConfigError, match="ollama serve"):
        OllamaLLM(host="http://127.0.0.1:1").installed_models()
