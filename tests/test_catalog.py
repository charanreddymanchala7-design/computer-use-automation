"""The agent-facing catalog: capabilities as tool definitions an AI agent can discover and call.

The shape follows the Model Context Protocol's tool listing (name, description, inputSchema,
outputSchema, annotations) so an MCP server could serve it as is; invoking a tool is exactly a
replay, so the model is never involved in a call."""

from __future__ import annotations

import json
from typing import Any

import jsonschema
import pytest

from cua.artifact import Capability
from cua.catalog import ToolError, build_catalog, call_tool, tool_definition
from cua.result import ReplayResult, Status


def cap(data: dict[str, Any]) -> Capability:
    return Capability.model_validate(data)


def writes(data: dict[str, Any], risk: str) -> dict[str, Any]:
    step = next(s for s in data["steps"] if s["id"] == "s3")
    step["risk_class"] = risk
    return data


def test_a_tool_definition_carries_what_an_agent_needs_to_choose_and_call_it(
    capability_dict: dict[str, Any],
) -> None:
    tool = tool_definition(cap(capability_dict))
    assert tool["name"] == "member_lookup"
    assert tool["description"].startswith("Search for a member by number")
    assert tool["inputSchema"]["required"] == ["member_id"]
    assert tool["outputSchema"]["title"] == "ReplayResult"
    assert tool["_meta"]["cua/outputs"]["properties"] == {"savings_balance": {"type": "string"}}
    assert tool["_meta"]["cua/version"] == "1.0.0"
    assert tool["_meta"]["cua/digest"].startswith("sha256:")


def test_a_read_only_capability_says_so(capability_dict: dict[str, Any]) -> None:
    hints = tool_definition(cap(capability_dict))["annotations"]
    assert (hints["readOnlyHint"], hints["destructiveHint"]) == (True, False)
    assert hints["openWorldHint"] is False  # it only ever touches the one application
    assert hints["title"] == "Look up a member and read their savings balance"


@pytest.mark.parametrize(
    ("risk", "read_only", "destructive"),
    [("reversible_write", False, False), ("irreversible_write", False, True)],
)
def test_a_capability_that_changes_state_says_how_far(
    capability_dict: dict[str, Any], risk: str, read_only: bool, destructive: bool
) -> None:
    hints = tool_definition(cap(writes(capability_dict, risk)))["annotations"]
    assert (hints["readOnlyHint"], hints["destructiveHint"]) == (read_only, destructive)


def test_secrets_are_named_never_valued(capability_dict: dict[str, Any]) -> None:
    step = next(s for s in capability_dict["steps"] if s["id"] == "s2")
    step["value"] = {"source": "secret", "name": "BANK_PASSWORD"}
    tool = tool_definition(cap(capability_dict))
    assert tool["_meta"]["cua/requires_secrets"] == ["BANK_PASSWORD"]


def test_every_input_schema_is_itself_a_valid_json_schema(capability_dict: dict[str, Any]) -> None:
    tool = tool_definition(cap(capability_dict))
    jsonschema.Draft202012Validator.check_schema(tool["inputSchema"])
    jsonschema.Draft202012Validator.check_schema(tool["outputSchema"])


def test_the_catalog_is_sorted_and_byte_for_byte_repeatable(
    capability_dict: dict[str, Any],
) -> None:
    other = json.loads(json.dumps(capability_dict))
    other["id"] = "another_capability"
    first = build_catalog([cap(capability_dict), cap(other)])
    again = build_catalog([cap(other), cap(capability_dict)])
    assert first == again  # order of discovery does not matter
    assert [t["name"] for t in first["tools"]] == ["another_capability", "member_lookup"]
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)
    assert first["version"] == 1


def test_the_catalog_holds_no_host_or_credential(capability_dict: dict[str, Any]) -> None:
    text = json.dumps(build_catalog([cap(capability_dict)]))
    assert "127.0.0.1" not in text
    assert "demo-only" not in text


# --- calling a tool -----------------------------------------------------------------------------


def result(status: Status = Status.SUCCESS, **over: Any) -> ReplayResult:
    fields: dict[str, Any] = {
        "status": status,
        "outcome_code": "completed",
        "capability_id": "member_lookup",
        "capability_version": "1.0.0",
        "run_id": "r1",
        "outputs": {"savings_balance": "$2,480.15"},
        "duration_ms": 10,
    }
    fields.update(over)
    return ReplayResult(**fields)


def test_a_call_is_a_replay_and_its_answer_is_structured(capability_dict: dict[str, Any]) -> None:
    seen: list[dict[str, Any]] = []

    def run(capability: Capability, arguments: dict[str, Any]) -> ReplayResult:
        seen.append(arguments)
        return result()

    reply = call_tool([cap(capability_dict)], "member_lookup", {"member_id": "12345"}, run)
    assert seen == [{"member_id": "12345"}]
    assert reply["isError"] is False
    assert reply["structuredContent"]["outputs"] == {"savings_balance": "$2,480.15"}
    assert json.loads(reply["content"][0]["text"]) == reply["structuredContent"]


def test_an_answer_that_is_not_an_error_is_not_flagged_as_one(
    capability_dict: dict[str, Any],
) -> None:
    def run(capability: Capability, arguments: dict[str, Any]) -> ReplayResult:
        return result(Status.BUSINESS_OUTCOME, outcome_code="member_not_found", outputs=None)

    assert (
        call_tool([cap(capability_dict)], "member_lookup", {"member_id": "12345"}, run)["isError"]
        is False
    )


@pytest.mark.parametrize("status", [Status.HARD_FAILURE, Status.ESCALATED])
def test_a_run_that_did_not_complete_is_flagged_as_an_error(
    capability_dict: dict[str, Any], status: Status
) -> None:
    extra: dict[str, Any] = (
        {"failed_step": "s2", "expected": "x", "observed": "y"}
        if status is Status.HARD_FAILURE
        else {"escalation": {"request_id": "ir_1", "reason": "session_expired"}}
    )

    def run(capability: Capability, arguments: dict[str, Any]) -> ReplayResult:
        return result(status, outcome_code="locator_not_found", outputs=None, **extra)

    reply = call_tool([cap(capability_dict)], "member_lookup", {"member_id": "12345"}, run)
    assert reply["isError"] is True


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "member_id"),  # missing
        ({"member_id": "abc"}, "member_id"),  # wrong shape
        ({"member_id": "12345", "extra": 1}, "extra"),  # not declared
    ],
)
def test_arguments_are_validated_before_anything_runs(
    capability_dict: dict[str, Any], arguments: dict[str, Any], message: str
) -> None:
    def run(capability: Capability, arguments: dict[str, Any]) -> ReplayResult:
        raise AssertionError("must not run")

    with pytest.raises(ToolError, match=message):
        call_tool([cap(capability_dict)], "member_lookup", arguments, run)


def test_a_bad_argument_error_never_repeats_the_value(capability_dict: dict[str, Any]) -> None:
    with pytest.raises(ToolError) as caught:
        call_tool(
            [cap(capability_dict)],
            "member_lookup",
            {"member_id": "secret-ish"},
            lambda c, a: result(),
        )
    assert "secret-ish" not in str(caught.value)


def test_an_unknown_tool_lists_the_ones_that_exist(capability_dict: dict[str, Any]) -> None:
    with pytest.raises(ToolError, match=r"unknown tool 'nope'.*member_lookup"):
        call_tool([cap(capability_dict)], "nope", {}, lambda c, a: result())
