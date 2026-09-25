"""Scripted discoveries against the MemberServ mock.

The scripted model reads the rendered page to find its refs, exactly as a real model would, so
these exercise the observation format as well as the loop. They are shared by the end-to-end,
replay and fault-matrix tests.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from cua.agent import DiscoveryLoop, DiscoveryTask, Limits
from cua.artifact import Capability
from cua.artifact.synthesize import CapabilitySpec, synthesize
from cua.demo.memberserv import (
    lookup_spec,
    lookup_task,
    open_subaccount_spec,
    open_subaccount_task,
)
from cua.evlog import EventLog
from cua.gateway import ActionGateway
from cua.llm import FakeLLM, LLMRequest, LLMResponse, TextPart, ToolResult, tool_call
from cua.policy import Policy, UrlRule
from cua.redact import Redactor
from cua.surface import PlaywrightSurface


def latest_page(request: LLMRequest) -> str:
    """The most recent page description the model was shown."""
    for message in reversed(request.messages):
        for part in message.parts:
            texts: list[str] = []
            if isinstance(part, TextPart):
                texts = [part.text]
            elif isinstance(part, ToolResult):
                texts = [p.text for p in part.content if isinstance(p, TextPart)]
            for text in texts:
                if "URL:" in text and "[frame" in text:
                    return text
    raise AssertionError("the model was never shown a page")


def ref_of(request: LLMRequest, *needles: str) -> str:
    """The ref of the element whose description contains every needle, found by reading."""
    page = latest_page(request)
    for line in page.splitlines():
        stripped = line.strip()
        if re.match(r"e\d+ ", stripped) and all(n in stripped for n in needles):
            return stripped.split()[0]
    raise AssertionError(f"no element with {needles} on:\n{page}")


def do(kind: str, *needles: str, **args: object) -> object:
    def step(request: LLMRequest) -> LLMResponse:
        return tool_call("act", kind=kind, ref=ref_of(request, *needles), reason=kind, **args)

    return step


def sign_in_steps() -> list[object]:
    return [
        do("fill", "name=u", secret="MOCK_USER"),
        do("fill", "name=p", secret="MOCK_PASS"),
        do("click", "type=image"),
    ]


def lookup_script() -> list[object]:
    return [
        *sign_in_steps(),
        do("fill", "name=F1", param="member_id"),
        do("click", "alt=Go"),
        do("click", "tr", "12345", "TESTERSON"),
        tool_call(
            "extract",
            name="savings_balance",
            value="$2,480.15",
            anchor_text="SHARE SAVINGS",
            reason="the balance",
        ),
        tool_call("finish", success=True, summary="read the savings balance"),
    ]


def open_script() -> list[object]:
    return [
        *sign_in_steps(),
        do("fill", "name=F1", param="member_id"),
        do("click", "alt=Go"),
        do("click", "tr", "12345", "TESTERSON"),
        do("click", "td", "Open Sub-Account"),
        do("fill", "name=F8", param="deposit"),
        do("click", 'value="Submit"'),
        tool_call("act", kind="wait_for", text="SUB-ACCOUNT OPENED", reason="the confirmation"),
        tool_call(
            "extract",
            name="confirmation_ref",
            value="CNF-000001",
            anchor_text="Reference No:",
            reason="the reference",
        ),
        tool_call("finish", success=True, summary="opened a sub-account"),
    ]


def policy() -> Policy:
    return Policy(
        allow=(UrlRule(host="127.0.0.1", path_prefix="/msv/"),),
        deny=(UrlRule(host="127.0.0.1", path_prefix="/msv/admin.cgi"),),
    )


def loop_for(
    surface: PlaywrightSurface,
    tmp_path: Path,
    script: list[object],
    limits: Limits | None = None,
) -> tuple[DiscoveryLoop, FakeLLM, ActionGateway, EventLog]:
    log = EventLog(
        tmp_path / "run.jsonl", run_id="run_e2e", redactor=Redactor(secrets=["demo-only"])
    )
    gateway = ActionGateway(surface, policy(), log)
    llm = FakeLLM(script)  # type: ignore[arg-type]
    return (
        DiscoveryLoop(surface, gateway, llm, log, limits=limits, run_id="run_e2e"),
        llm,
        gateway,
        log,
    )


def task_for(base_url: str) -> DiscoveryTask:
    return lookup_task(base_url)


def discover(
    surface: PlaywrightSurface, base_url: str, tmp_path: Path, which: str = "lookup"
) -> Capability:
    """One scripted discovery, synthesized into a capability (portable: no host in it)."""
    task, spec, script = (
        (lookup_task(base_url), lookup_spec(), lookup_script())
        if which == "lookup"
        else (open_subaccount_task(base_url), open_subaccount_spec(), open_script())
    )
    loop, _, _, _ = loop_for(surface, tmp_path / f"discover_{which}", script)
    run = loop.run(task)
    assert run.outcome == "finished", run.reason
    return synthesize(run, task, spec, now=datetime(2026, 9, 25, tzinfo=UTC)).capability


__all__ = [
    "CapabilitySpec",
    "discover",
    "do",
    "latest_page",
    "lookup_script",
    "loop_for",
    "open_script",
    "policy",
    "ref_of",
    "task_for",
]
