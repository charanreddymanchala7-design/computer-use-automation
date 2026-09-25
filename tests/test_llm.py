"""The model boundary: what is sent, what comes back, what is counted, and what never leaks.

The real API is never called in tests. The Anthropic implementation runs against a stub client
that records exactly what would have been sent.
"""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest

from cua.evlog import EventLog, read_events
from cua.llm import (
    AnthropicLLM,
    FakeLLM,
    ImagePart,
    LLMConfigError,
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    TextPart,
    ToolResult,
    ToolSpec,
    ToolUse,
    Usage,
    estimate_cost,
    say,
    tool_call,
)
from cua.redact import Redactor

TOOLS = (
    ToolSpec("observe", "Look at the page", {"type": "object", "properties": {}}),
    ToolSpec(
        "act",
        "Do one thing",
        {"type": "object", "properties": {"kind": {"type": "string"}}, "required": ["kind"]},
    ),
)
PNG = b"\x89PNG\r\n\x1a\nfake-image-bytes"


class StubMessages:
    def __init__(self, reply: object | Exception) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class StubClient:
    def __init__(self, reply: object | Exception) -> None:
        self.messages = StubMessages(reply)


def api_reply(
    *blocks: object,
    stop_reason: str = "end_turn",
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_read: int = 0,
    cache_write: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ),
    )


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_block(id_: str, name: str, **input_: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def llm_with(reply: object | Exception, **kwargs: Any) -> tuple[AnthropicLLM, StubClient]:
    client = StubClient(reply)
    return AnthropicLLM(model="claude-sonnet-5", client=client, **kwargs), client  # type: ignore[arg-type]


def step(llm: AnthropicLLM, *messages: Message) -> LLMResponse:
    return llm.step(system="You drive a browser.", messages=list(messages), tools=TOOLS)


# --- what is sent ------------------------------------------------------------------------------


def test_the_request_carries_system_tools_and_a_cache_breakpoint_on_each() -> None:
    llm, client = llm_with(api_reply(text_block("ok")))
    step(llm, Message.user(TextPart("goal: look up member 12345")))
    (sent,) = client.messages.calls
    assert sent["model"] == "claude-sonnet-5"
    assert sent["system"] == [
        {"type": "text", "text": "You drive a browser.", "cache_control": {"type": "ephemeral"}}
    ]
    assert [t["name"] for t in sent["tools"]] == ["observe", "act"]
    assert sent["tools"][1]["input_schema"]["required"] == ["kind"]
    assert "cache_control" not in sent["tools"][0]
    assert sent["tools"][-1]["cache_control"] == {"type": "ephemeral"}  # caches system + tools


def test_no_sampling_parameters_are_sent_because_the_newest_models_reject_them() -> None:
    llm, client = llm_with(api_reply(text_block("ok")))
    step(llm, Message.user(TextPart("hi")))
    (sent,) = client.messages.calls
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(sent)
    assert sent["max_tokens"] == 1024


def test_images_are_sent_as_base64_png_and_tool_results_link_back_to_their_call() -> None:
    llm, client = llm_with(api_reply(text_block("ok")))
    history = [
        Message.user(TextPart("goal")),
        Message.assistant(TextPart("looking"), ToolUse("tu_1", "observe", {})),
        Message.user(ToolResult("tu_1", (TextPart("frames: 4"), ImagePart(PNG)))),
    ]
    step(llm, *history)
    (sent,) = client.messages.calls
    assert [m["role"] for m in sent["messages"]] == ["user", "assistant", "user"]
    assert sent["messages"][1]["content"] == [
        {"type": "text", "text": "looking"},
        {"type": "tool_use", "id": "tu_1", "name": "observe", "input": {}},
    ]
    result = sent["messages"][2]["content"][0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "tu_1"
    assert "is_error" not in result
    assert result["content"][0] == {"type": "text", "text": "frames: 4"}
    assert result["content"][1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(PNG).decode(),
        },
    }


def test_an_error_result_is_flagged_so_the_model_can_recover() -> None:
    llm, client = llm_with(api_reply(text_block("ok")))
    step(
        llm,
        Message.user(TextPart("goal")),
        Message.assistant(ToolUse("tu_1", "act", {"kind": "click"})),
        Message.user(ToolResult("tu_1", (TextPart("stale ref; observe again"),), is_error=True)),
    )
    (sent,) = client.messages.calls
    assert sent["messages"][2]["content"][0]["is_error"] is True


# --- what comes back ---------------------------------------------------------------------------


def test_text_and_tool_calls_are_parsed_out_of_the_response() -> None:
    reply = api_reply(
        text_block("I will click Go."),
        tool_block("tu_9", "act", kind="click", ref="e4"),
        stop_reason="tool_use",
    )
    llm, _ = llm_with(reply)
    response = step(llm, Message.user(TextPart("goal")))
    assert response.text == "I will click Go."
    assert response.tool_calls == (ToolUse("tu_9", "act", {"kind": "click", "ref": "e4"}),)
    assert response.stop_reason == "tool_use"
    assert response.as_message() == Message.assistant(
        TextPart("I will click Go."), ToolUse("tu_9", "act", {"kind": "click", "ref": "e4"})
    )


def test_unknown_block_types_are_ignored_not_fatal() -> None:
    llm, _ = llm_with(api_reply(SimpleNamespace(type="thinking", thinking="hmm"), text_block("hi")))
    assert step(llm, Message.user(TextPart("goal"))).text == "hi"


def test_usage_is_reported_including_cache_tokens() -> None:
    reply = api_reply(
        text_block("ok"), input_tokens=500, output_tokens=80, cache_read=9000, cache_write=1200
    )
    llm, _ = llm_with(reply)
    usage = step(llm, Message.user(TextPart("goal"))).usage
    assert usage == Usage(
        input_tokens=500, output_tokens=80, cache_read_tokens=9000, cache_write_tokens=1200
    )
    assert usage.total_tokens == 500 + 80 + 9000 + 1200


def test_cost_is_estimated_and_cached_reads_are_cheap() -> None:
    plain = Usage(input_tokens=1_000_000, output_tokens=0)
    cached = Usage(input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000)
    price = estimate_cost("claude-sonnet-5", plain)
    assert price is not None
    assert price > 0
    assert estimate_cost("claude-sonnet-5", cached) == pytest.approx(price * 0.1)
    assert estimate_cost("some-future-model", plain) is None  # unknown: say so, do not guess


def test_the_meter_accumulates_across_steps() -> None:
    llm, _ = llm_with(api_reply(text_block("ok"), input_tokens=100, output_tokens=10))
    step(llm, Message.user(TextPart("a")))
    step(llm, Message.user(TextPart("b")))
    assert llm.meter.steps == 2
    assert llm.meter.usage.input_tokens == 200
    assert llm.meter.usage.output_tokens == 20
    assert llm.meter.cost_usd is not None
    assert llm.meter.cost_usd > 0


# --- keys and errors ---------------------------------------------------------------------------


def test_the_key_comes_from_the_environment_mapping_it_is_given() -> None:
    llm = AnthropicLLM(
        model="claude-sonnet-5", env={"ANTHROPIC_API_KEY": "sk-fake-envkey0123456789"}
    )
    assert isinstance(llm.client, anthropic.Anthropic)


def test_a_missing_key_is_a_clear_error_that_says_where_to_put_it() -> None:
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY") as info:
        AnthropicLLM(model="claude-sonnet-5", env={})
    assert ".env" in str(info.value)


def test_the_key_never_appears_in_a_repr_or_an_error(tmp_path: Path) -> None:
    key = "sk-fake-envkey0123456789"
    llm = AnthropicLLM(model="claude-sonnet-5", env={"ANTHROPIC_API_KEY": key})
    assert key not in repr(llm)
    assert key not in str(llm.__dict__.get("model"))
    boom = anthropic.APIConnectionError(request=SimpleNamespace(url="https://x"))  # type: ignore[arg-type]
    failing, _ = llm_with(boom)
    with pytest.raises(LLMError) as info:
        step(failing, Message.user(TextPart("goal")))
    assert key not in str(info.value)


def test_api_failures_become_one_error_type_the_loop_can_handle() -> None:
    limit = anthropic.RateLimitError(
        "slow down",
        response=SimpleNamespace(status_code=429, headers={}, request=SimpleNamespace()),  # type: ignore[arg-type]
        body=None,
    )
    llm, _ = llm_with(limit)
    with pytest.raises(LLMError, match="RateLimitError"):
        step(llm, Message.user(TextPart("goal")))


# --- logging discipline ------------------------------------------------------------------------


def test_only_metadata_is_logged_never_prompts_or_replies(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "run.jsonl", run_id="run_1", redactor=Redactor(secrets=["hunter2"]))
    reply = api_reply(
        text_block("the password is hunter2"),
        tool_block("tu_1", "act", kind="fill", text="hunter2"),
        stop_reason="tool_use",
        input_tokens=700,
        output_tokens=40,
        cache_read=5000,
    )
    llm, _ = llm_with(reply, log=log)
    step(llm, Message.user(TextPart("ssn 123-45-6789 and hunter2")))
    text = log.path.read_text()
    assert "hunter2" not in text
    assert "123-45-6789" not in text
    (event,) = read_events(log.path)
    assert event["event"] == "llm_step"
    assert event["model"] == "claude-sonnet-5"
    assert (event["input_tokens"], event["output_tokens"], event["cache_read_tokens"]) == (
        700,
        40,
        5000,
    )
    assert event["tool_calls"] == ["act"]
    assert event["stop_reason"] == "tool_use"
    assert event["cost_usd"] > 0


# --- the scripted fake -------------------------------------------------------------------------


def test_the_fake_replays_its_script_in_order_and_records_every_request() -> None:
    fake = FakeLLM([tool_call("observe"), tool_call("act", kind="click", ref="e1"), say("done")])
    first = fake.step(system="s", messages=[Message.user(TextPart("goal"))], tools=TOOLS)
    second = fake.step(system="s", messages=[Message.user(TextPart("again"))], tools=TOOLS)
    third = fake.step(system="s", messages=[], tools=())
    assert first.tool_calls[0].name == "observe"
    assert second.tool_calls[0].input == {"kind": "click", "ref": "e1"}
    assert third.text == "done"
    assert third.tool_calls == ()
    assert [len(c.messages) for c in fake.calls] == [1, 1, 0]
    assert fake.calls[0].tools == TOOLS
    assert fake.meter.steps == 3


def test_the_fake_can_decide_from_the_request_so_scripts_can_read_the_page() -> None:
    def pick(request: LLMRequest) -> LLMResponse:
        goal = request.messages[0].parts[0]
        assert isinstance(goal, TextPart)
        return say(f"heard: {goal.text}")

    fake = FakeLLM([pick])
    assert (
        fake.step(system="", messages=[Message.user(TextPart("hi"))], tools=()).text == "heard: hi"
    )


def test_a_script_that_runs_out_is_an_error_not_a_silent_loop() -> None:
    fake = FakeLLM([say("only one")])
    fake.step(system="", messages=[], tools=())
    with pytest.raises(LLMError, match="script exhausted"):
        fake.step(system="", messages=[], tools=())


def test_scripted_tool_calls_get_distinct_ids() -> None:
    fake = FakeLLM([tool_call("observe"), tool_call("observe")])
    ids = [fake.step(system="", messages=[], tools=()).tool_calls[0].id for _ in range(2)]
    assert len(set(ids)) == 2
