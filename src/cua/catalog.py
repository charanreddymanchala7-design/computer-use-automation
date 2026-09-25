"""Capabilities as tools an AI agent can discover and call.

Each saved capability becomes one tool definition shaped like the Model Context Protocol's tool
listing: a name, a description, the JSON Schema of its input, and annotations that say whether it
only reads or can change state. A call is a replay, so no model is involved in serving it, and
the arguments are validated against the schema before anything touches the application.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from cua.artifact import Capability, RiskClass
from cua.replay.engine import describe_schema_error, first_schema_error
from cua.result import ReplayResult, Status, replay_result_json_schema

CATALOG_VERSION = 1

Runner = Callable[[Capability, dict[str, Any]], ReplayResult]


class ToolError(ValueError):
    """A call that could not be made: an unknown tool or arguments that do not fit."""


def tool_definition(capability: Capability) -> dict[str, Any]:
    risks = {step.risk_class for step in capability.steps}
    return {
        "name": capability.id,
        "description": capability.description,
        "inputSchema": capability.inputs,
        # a call returns the whole replay result; its `outputs` are described under _meta
        "outputSchema": replay_result_json_schema(),
        "annotations": {
            "title": capability.title,
            "readOnlyHint": risks <= {RiskClass.READ},
            "destructiveHint": RiskClass.IRREVERSIBLE_WRITE in risks,
            "idempotentHint": risks <= {RiskClass.READ},
            "openWorldHint": False,  # it only ever touches the one application it was made for
        },
        "_meta": {
            "cua/version": capability.capability_version,
            "cua/digest": capability.content_digest(),
            "cua/outputs": capability.outputs,
            "cua/requires_secrets": capability.required_secrets(),
        },
    }


def build_catalog(capabilities: Iterable[Capability]) -> dict[str, Any]:
    """The whole catalog, deterministic: sorted by name and free of times, hosts and secrets."""
    tools = [tool_definition(c) for c in sorted(capabilities, key=lambda c: c.id)]
    return {"version": CATALOG_VERSION, "tools": tools}


def call_tool(
    capabilities: Iterable[Capability],
    name: str,
    arguments: Mapping[str, Any],
    run: Runner,
) -> dict[str, Any]:
    """Invoke a tool by name. The reply is MCP-shaped: text content, the structured result, and
    ``isError`` set when the run did not complete (a business outcome is an answer, not a
    failure)."""
    by_name = {c.id: c for c in capabilities}
    capability = by_name.get(name)
    if capability is None:
        raise ToolError(f"unknown tool {name!r}; available: {', '.join(sorted(by_name)) or 'none'}")
    error = first_schema_error(capability.inputs, dict(arguments))
    if error is not None:
        raise ToolError(
            f"invalid arguments for {name}: {describe_schema_error(error, capability.inputs)}"
        )
    result = run(capability, dict(arguments))
    document = result.model_dump(mode="json")
    return {
        "content": [{"type": "text", "text": result.model_dump_json()}],
        "structuredContent": document,
        "isError": result.status in (Status.HARD_FAILURE, Status.ESCALATED),
    }
