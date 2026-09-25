"""The discovery loop end to end: real Chromium, the real mock, the real gateway.

Only the model is scripted, and the script has to read the rendered page to find its refs, so
the observation format is exercised exactly as a real model would use it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.discovery_scripts import do, latest_page, lookup_script, loop_for
from tests.mockbank_support import MockHandle

from cua.agent import DiscoveryTask, Limits, ParamSpec, RecordedRun
from cua.artifact import AncestorAnchorLocator, ParamRef, SecretRef, TextLocator
from cua.demo.memberserv import lookup_task
from cua.evlog import read_events
from cua.llm import TextPart, ToolResult, tool_call
from cua.surface import PlaywrightSurface

pytestmark = pytest.mark.browser


def test_a_member_lookup_is_discovered_end_to_end_and_recorded(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    loop, llm, _, log = loop_for(surface, tmp_path, lookup_script())
    run = loop.run(lookup_task(mock.base))

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
    run = loop.run(lookup_task(mock.base))
    everything = run.model_dump_json() + log.path.read_text() + repr(llm.calls)
    assert "demo-only" not in everything
    assert "teller01" not in everything.replace("TELLER01", "")  # the header shows the user name
    events = read_events(log.path)
    assert {e["secret_name"] for e in events if "secret_name" in e} == {"MOCK_USER", "MOCK_PASS"}


def test_a_read_only_lookup_changes_nothing_on_the_server(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    loop, _, _, _ = loop_for(surface, tmp_path, lookup_script())
    loop.run(lookup_task(mock.base))
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
    run = loop.run(lookup_task(mock.base))
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
    run = loop.run(lookup_task(mock.base))
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
    run = loop.run(lookup_task(mock.base))
    assert run.outcome == "max_steps"
    assert len(llm.calls) == 4
