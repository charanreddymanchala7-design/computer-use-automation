"""The discovery loop end to end: real Chromium, the real mock, the real gateway.

Only the model is scripted, and the script has to read the rendered page to find its refs, so
the observation format is exercised exactly as a real model would use it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from tests.mockbank_support import MockHandle

from cua.agent import DiscoveryLoop, DiscoveryTask, Limits, OutputSpec, ParamSpec, RecordedRun
from cua.artifact import (
    AncestorAnchorLocator,
    ParamRef,
    SecretRef,
    TextLocator,
)
from cua.evlog import EventLog, read_events
from cua.gateway import ActionGateway
from cua.llm import FakeLLM, LLMRequest, LLMResponse, TextPart, ToolResult, tool_call
from cua.policy import Policy, UrlRule
from cua.redact import Redactor
from cua.surface import PlaywrightSurface

pytestmark = pytest.mark.browser


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


def task_for(mock: MockHandle) -> DiscoveryTask:
    return DiscoveryTask(
        goal="Sign in and read the current savings balance of a member",
        start_url=f"{mock.base}/msv/login.cgi",
        params={"member_id": ParamSpec("12345", "Member number")},
        outputs={"savings_balance": OutputSpec("Current share savings balance")},
        secrets=("MOCK_USER", "MOCK_PASS"),
    )


def loop_for(
    surface: PlaywrightSurface,
    tmp_path: Path,
    script: list[object],
    limits: Limits | None = None,
) -> tuple[DiscoveryLoop, FakeLLM, ActionGateway, EventLog]:
    policy = Policy(
        allow=(UrlRule(host="127.0.0.1", path_prefix="/msv/"),),
        deny=(UrlRule(host="127.0.0.1", path_prefix="/msv/admin.cgi"),),
    )
    log = EventLog(
        tmp_path / "run.jsonl", run_id="run_e2e", redactor=Redactor(secrets=["demo-only"])
    )
    gateway = ActionGateway(surface, policy, log)
    llm = FakeLLM(script)  # type: ignore[arg-type]
    return (
        DiscoveryLoop(surface, gateway, llm, log, limits=limits, run_id="run_e2e"),
        llm,
        gateway,
        log,
    )


def lookup_script() -> list[object]:
    return [
        do("fill", "name=u", secret="MOCK_USER"),
        do("fill", "name=p", secret="MOCK_PASS"),
        do("click", "type=image"),
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


def test_a_member_lookup_is_discovered_end_to_end_and_recorded(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    loop, llm, _, log = loop_for(surface, tmp_path, lookup_script())
    run = loop.run(task_for(mock))

    assert run.outcome == "finished", run.reason
    assert run.outputs == {"savings_balance": "$2,480.15"}
    assert run.missing_outputs == []
    assert [s.kind for s in run.steps] == [
        "navigate",
        "fill",
        "fill",
        "click",
        "fill",
        "click",
        "click",
        "extract",
    ]
    assert [s.value for s in run.steps[1:3]] == [
        SecretRef(name="MOCK_USER"),
        SecretRef(name="MOCK_PASS"),
    ]
    assert run.steps[4].value == ParamRef(name="member_id")
    assert run.secrets_used == ["MOCK_USER", "MOCK_PASS"]

    number_field = run.steps[4].locator
    assert number_field is not None
    assert [hop.name for hop in number_field.frame_path] == ["main"]
    anchor = number_field.strategies[0]
    assert isinstance(anchor, AncestorAnchorLocator)
    assert anchor.anchor_text == "Member No:"  # anchored on its label, not its volatile id

    row = run.steps[6].locator
    assert row is not None
    assert any(isinstance(s, TextLocator) and s.text == "{member_id}" for s in row.strategies)

    balance = run.steps[7].locator
    assert balance is not None
    assert [hop.name for hop in balance.frame_path] == ["main", "acct"]

    # the model saw the login, then the frameset, then the results, then the member page
    pages = [latest_page(r) for r in llm.calls]
    assert "User" in pages[0]
    assert '"acct"' in pages[-1]
    assert log.path.exists()


def test_no_secret_reaches_the_model_the_recording_or_the_log(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    loop, llm, _, log = loop_for(surface, tmp_path, lookup_script())
    run = loop.run(task_for(mock))
    everything = run.model_dump_json() + log.path.read_text() + repr(llm.calls)
    assert "demo-only" not in everything
    assert "teller01" not in everything.replace("TELLER01", "")  # the header shows the user name
    events = read_events(log.path)
    assert {e["secret_name"] for e in events if "secret_name" in e} == {"MOCK_USER", "MOCK_PASS"}


def test_a_read_only_lookup_changes_nothing_on_the_server(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    loop, _, _, _ = loop_for(surface, tmp_path, lookup_script())
    loop.run(task_for(mock))
    state = mock.server.state
    assert state.confirmations == 0
    assert state.closed_accounts == set()
    paths = [r["path"] for r in state.requests]
    assert "/msv/admin.cgi" not in paths
    assert "/msv/close.cgi" not in paths


def test_a_model_that_wanders_into_the_admin_page_is_stopped_and_the_click_is_not_recorded(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    script: list[object] = [
        do("fill", "name=u", secret="MOCK_USER"),
        do("fill", "name=p", secret="MOCK_PASS"),
        do("click", "type=image"),
        do("click", "link", '"Admin"'),
        tool_call("finish", success=False, summary="the goal needs the admin page"),
    ]
    loop, llm, gateway, _ = loop_for(surface, tmp_path, script)
    run = loop.run(task_for(mock))
    assert run.outcome == "failed"
    assert gateway.blocked_navigations == [f"{mock.base}/msv/admin.cgi"]
    assert "/msv/admin.cgi" not in [r["path"] for r in mock.server.state.requests]
    assert [s.kind for s in run.steps] == [
        "navigate",
        "fill",
        "fill",
        "click",
    ]  # not the Admin click
    refusal = llm.calls[-1].messages[-1].parts[0]
    assert isinstance(refusal, ToolResult)
    assert refusal.is_error


def test_an_account_closing_click_is_refused_and_the_server_never_sees_it(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    script: list[object] = [
        do("fill", "name=u", secret="MOCK_USER"),
        do("fill", "name=p", secret="MOCK_PASS"),
        do("click", "type=image"),
        do("fill", "name=F1", param="member_id"),
        do("click", "alt=Go"),
        do("click", "tr", "12345", "TESTERSON"),
        do("click", "link", '"Close"', "SYN-12345-S01"),
        tool_call("finish", success=False, summary="closing an account needs a human"),
    ]
    loop, llm, _, _ = loop_for(surface, tmp_path, script)
    run = loop.run(task_for(mock))
    assert run.outcome == "failed"
    assert mock.server.state.closed_accounts == set()
    assert "/msv/close.cgi" not in [r["path"] for r in mock.server.state.requests]
    refusal = llm.calls[-1].messages[-1].parts[0]
    assert isinstance(refusal, ToolResult)
    assert "needs a human's confirmation" in "".join(
        p.text for p in refusal.content if isinstance(p, TextPart)
    )


def test_opening_a_sub_account_is_recorded_with_its_write_risk_dialog_and_wait(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    script: list[object] = [
        do("fill", "name=u", secret="MOCK_USER"),
        do("fill", "name=p", secret="MOCK_PASS"),
        do("click", "type=image"),
        do("fill", "name=F1", param="member_id"),
        do("click", "alt=Go"),
        do("click", "tr", "12345", "TESTERSON"),
        do("click", "td", "Open Sub-Account"),
        do("fill", "name=F8", text="25.00"),
        do("click", 'value="Submit"'),
        tool_call("act", kind="wait_for", text="SUB-ACCOUNT OPENED", reason="the confirmation"),
        tool_call("finish", success=True, summary="opened a sub-account"),
    ]
    loop, _, _, _ = loop_for(surface, tmp_path, script)
    task = DiscoveryTask(
        goal="Open a sub-account for a member",
        start_url=f"{mock.base}/msv/login.cgi",
        params={"member_id": ParamSpec("12345", "Member number")},
        secrets=("MOCK_USER", "MOCK_PASS"),
    )
    run: RecordedRun = loop.run(task)
    assert run.outcome == "finished", run.reason
    assert mock.server.state.confirmations == 1
    submit = next(s for s in run.steps if s.kind == "click" and s.risk.value == "reversible_write")
    assert [(d.kind, d.accepted) for d in submit.dialogs] == [("confirm", True)]
    assert run.steps[-1].kind == "wait_for"
    assert run.steps[-1].expect_text == "SUB-ACCOUNT OPENED"


def test_the_limits_stop_a_model_that_never_finishes(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    loop, llm, _, _ = loop_for(
        surface, tmp_path, [tool_call("observe")] * 10, limits=Limits(max_steps=4)
    )
    run = loop.run(task_for(mock))
    assert run.outcome == "max_steps"
    assert len(llm.calls) == 4
