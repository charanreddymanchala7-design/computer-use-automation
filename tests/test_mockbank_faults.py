"""The fault table is the test matrix: every implemented fault, what it does on the wire, and the
guarantees around arming it (out-of-band, bounded, never visible to the agent)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from targets.mockbank.faults import MODES, FaultEngine, FaultError
from tests.mockbank_support import MockClient, MockHandle, Resp, token_of

ROOT = Path(__file__).resolve().parent.parent
ADMIN = "/_admin/faults"


def arm(mock: MockHandle, mode: str, **kwargs: Any) -> Resp:
    return mock.client().request("POST", ADMIN, json_body={"mode": mode, **kwargs})


def armed(mock: MockHandle) -> list[dict[str, Any]]:
    payload = json.loads(mock.client().get(ADMIN).text)
    listed: list[dict[str, Any]] = payload["armed"]
    return listed


def admin_log(mock: MockHandle) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(mock.client().get("/_admin/log").text)
    return payload


def submit(client: MockClient, deposit: str = "25.00") -> Resp:
    form = client.get("/msv/newsub.cgi?mid=12345")
    return client.post(
        "/msv/newsub.cgi?mid=12345",
        {"F7": "01", "F8": deposit, "F9": "", "tok": token_of(form.text)},
    )


def search(client: MockClient, number: str = "12345") -> str:
    page = client.get("/msv/search.cgi")
    resp = client.post(
        "/msv/results.cgi", {"SB": "1", "F1": number, "F2": "", "tok": token_of(page.text)}
    )
    return resp.text


# --- the six faults ---------------------------------------------------------------------------


def test_member_not_found_hides_a_member_that_exists_and_then_clears(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    assert arm(mock, "member_not_found").status == 200
    assert "NO RECORDS FOUND FOR CRITERIA" in search(client)
    assert "TESTERSON, ADA" in search(client)  # times defaults to 1


def test_validation_error_rejects_a_valid_deposit_and_creates_nothing(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    arm(mock, "validation_error")
    resp = submit(client)
    assert resp.status == 200
    assert "ERR 1042: OPENING DEPOSIT BELOW MINIMUM ($5.00)" in resp.text
    assert mock.server.state.confirmations == 0
    assert "SELECT NAME=F7" in resp.text.upper()  # the form is back so a caller could retry


def test_session_timeout_renders_the_login_form_in_the_frame_and_kills_the_session(
    mock: MockHandle,
) -> None:
    client = mock.logged_in_client()
    arm(mock, "session_timeout")
    resp = client.get("/msv/newsub.cgi?mid=12345")
    assert resp.status == 200
    assert "User&nbsp;ID" in resp.text
    assert "SELECT NAME=F7" not in resp.text.upper()
    # the session really is gone: the next page is the login form too, and the header lost the user
    assert "User&nbsp;ID" in client.get("/msv/search.cgi").text
    assert "TELLER01" not in client.get("/msv/hdr.cgi").text


def test_slow_load_delays_the_response_and_then_serves_the_real_page(mock: MockHandle) -> None:
    slept: list[float] = []
    mock.server.state.sleep = slept.append
    client = mock.logged_in_client()
    arm(mock, "slow_load", params={"delay_ms": 250})
    resp = client.get("/msv/search.cgi")
    assert slept == [0.25]
    assert "Member&nbsp;No" in resp.text  # it is only slow, not broken


def test_slow_load_defaults_to_three_seconds(mock: MockHandle) -> None:
    slept: list[float] = []
    mock.server.state.sleep = slept.append
    client = mock.logged_in_client()
    arm(mock, "slow_load")
    client.get("/msv/search.cgi")
    assert slept == [3.0]


def test_a_known_interstitial_can_be_acknowledged_to_reach_the_real_page(
    mock: MockHandle,
) -> None:
    client = mock.logged_in_client()
    arm(mock, "interstitial_known")
    notice = client.get("/msv/member.cgi?mid=12345")
    assert "SYSTEM NOTICE: END-OF-DAY BATCH AT 17:00" in notice.text
    ack = re.search(r"CLASS=btn ONCLICK=\"location='([^']+)'\">Acknowledge", notice.text)
    assert ack, "the notice needs an Acknowledge cell"
    assert ack.group(1) == "/msv/member.cgi?mid=12345"
    assert "TESTERSON, ADA" in client.get(ack.group(1)).text


def test_app_error_shows_an_error_page_and_creates_nothing_by_default(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    arm(mock, "app_error")
    resp = submit(client)
    assert resp.status == 200
    assert "APPLICATION ERROR 0x8004 - CONTACT YOUR SYSTEM ADMINISTRATOR" in resp.text
    assert mock.server.state.confirmations == 0


def test_app_error_with_commit_creates_the_account_before_failing(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    arm(mock, "app_error", params={"commit": True})
    resp = submit(client)
    assert "APPLICATION ERROR 0x8004" in resp.text
    assert mock.server.state.confirmations == 1  # an uncertain outcome: it did happen
    effects = admin_log(mock)["effects"]
    assert [e["kind"] for e in effects] == ["subaccount_created"]


def test_app_error_can_be_a_real_500(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    arm(mock, "app_error", step="search", params={"status": 500})
    page = client.get("/msv/search.cgi")
    assert page.status == 500
    assert "APPLICATION ERROR" in page.text


# --- arming: bounded, out-of-band, validated --------------------------------------------------


def test_a_fault_fires_the_requested_number_of_times_then_stops(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    arm(mock, "member_not_found", times=2)
    assert armed(mock)[0]["times_remaining"] == 2
    outcomes = ["NO RECORDS" in search(client) for _ in range(3)]
    assert outcomes == [True, True, False]
    assert armed(mock) == []


def test_always_keeps_a_fault_armed_until_it_is_disarmed(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    arm(mock, "member_not_found", times="always")
    assert all("NO RECORDS" in search(client) for _ in range(3))
    assert armed(mock)[0]["times_remaining"] == "always"
    assert mock.client().request("DELETE", ADMIN).status == 200
    assert armed(mock) == []
    assert "TESTERSON" in search(client)


def test_the_default_step_comes_from_the_mode(mock: MockHandle) -> None:
    arm(mock, "session_timeout")
    arm(mock, "interstitial_known")
    assert {(f["mode"], f["step"]) for f in armed(mock)} == {
        ("session_timeout", "newsub_form"),
        ("interstitial_known", "member"),
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"mode": "gremlins"}, "unknown mode"),
        ({"mode": "app_error", "step": "nowhere"}, "unknown step"),
        ({"mode": "validation_error", "step": "search"}, "does not apply"),
        ({"mode": "app_error", "times": 0}, "times"),
        ({"mode": "app_error", "times": "sometimes"}, "times"),
        ({"mode": "app_error", "params": "not-an-object"}, "params"),
        ({"mode": "slow_load", "params": {"delay_ms": "soon"}}, "params.delay_ms"),
        ({"mode": "slow_load", "params": {"delay_ms": -5}}, "params.delay_ms"),
        ({"mode": "app_error", "params": {"status": 99}}, "params.status"),
        ({"mode": "app_error", "params": {"commit": "yes"}}, "params.commit"),
        ({"mode": "member_not_found", "params": {"delay_ms": 5}}, "unknown param"),
        ({}, "mode"),
    ],
)
def test_bad_arming_requests_are_rejected_with_a_reason(
    mock: MockHandle, payload: dict[str, Any], message: str
) -> None:
    resp = mock.client().request("POST", ADMIN, json_body=payload)
    assert resp.status == 400
    assert message in json.loads(resp.text)["error"]
    assert armed(mock) == []


def test_a_body_that_is_not_json_is_rejected(mock: MockHandle) -> None:
    resp = mock.client().request("POST", ADMIN, data={"mode": "app_error"})
    assert resp.status == 400


def test_the_fault_name_never_appears_in_anything_the_agent_sees(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    bodies: list[str] = []
    for mode in MODES:
        step = MODES[mode].default_step
        params = {"params": {"delay_ms": 1}} if mode == "slow_load" else {}
        arm(mock, mode, step=step, **params)
    mock.server.state.sleep = lambda _s: None
    bodies.append(client.get("/msv/member.cgi?mid=12345").text)  # interstitial
    bodies.append(search(client))  # slow_load, then member_not_found
    bodies.append(client.get("/msv/newsub.cgi?mid=12345").text)  # session_timeout
    for body in bodies:
        assert not any(mode in body for mode in MODES), body


def test_reset_disarms_faults_and_clears_the_log(mock: MockHandle) -> None:
    arm(mock, "app_error", times="always")
    mock.client().post("/_admin/reset", {})
    assert armed(mock) == []
    assert admin_log(mock)["requests"] == []


# --- ground truth -----------------------------------------------------------------------------


def test_the_log_says_which_fault_was_applied_to_which_request(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    arm(mock, "member_not_found")
    search(client)
    search(client)
    posts = [r for r in admin_log(mock)["requests"] if r["step"] == "results"]
    assert [p["fault_applied"] for p in posts] == ["member_not_found", None]


def test_side_effects_are_recorded_so_a_double_submit_is_provable(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    submit(client)
    client.get("/msv/close.cgi?acct=SYN-12345-S01")
    kinds = [e["kind"] for e in admin_log(mock)["effects"]]
    assert kinds == ["subaccount_created", "account_closed"]


def test_any_matches_pages_but_not_images_or_the_header_frame(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    arm(mock, "app_error", step="any")
    assert client.get("/msv/go.gif").status == 200
    assert "TELLER01" in client.get("/msv/hdr.cgi").text
    assert "Member&nbsp;Inquiry" in client.get("/msv/nav.cgi").text
    assert "APPLICATION ERROR" in client.get("/msv/search.cgi").text


def test_an_interstitial_only_ever_interrupts_a_get(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    form = client.get("/msv/newsub.cgi?mid=12345")
    arm(mock, "interstitial_known", step="any", times="always")
    data = {"F7": "01", "F8": "25.00", "F9": "", "tok": token_of(form.text)}
    resp = client.post("/msv/newsub.cgi?mid=12345", data)
    assert resp.status == 302  # a POST is never replaced by a GET-only notice
    assert "SYSTEM NOTICE" in client.get("/msv/member.cgi?mid=12345").text
    assert armed(mock)[0]["times_remaining"] == "always"


def test_public_pages_are_never_faulted(mock: MockHandle) -> None:
    slept: list[float] = []
    mock.server.state.sleep = slept.append
    arm(mock, "slow_load", step="login", times="always")
    assert "User&nbsp;ID" in mock.client().get("/msv/login.cgi").text
    assert mock.client().login().status == 302
    assert slept == []
    assert armed(mock)[0]["times_remaining"] == "always"  # never consumed


# --- the engine on its own --------------------------------------------------------------------


def test_the_boot_spec_arms_faults() -> None:
    engine = FaultEngine()
    engine.arm_from_spec("session_timeout@newsub_form:1, slow_load@search:always ,app_error")
    described = [(f["mode"], f["step"], f["times_remaining"]) for f in engine.armed()]
    assert described == [
        ("session_timeout", "newsub_form", 1),
        ("slow_load", "search", "always"),
        ("app_error", "newsub_submit", 1),
    ]


@pytest.mark.parametrize("spec", ["gremlins", "app_error@nowhere", "app_error:0", "app_error:x"])
def test_a_bad_boot_spec_fails_loudly(spec: str) -> None:
    with pytest.raises(FaultError):
        FaultEngine().arm_from_spec(spec)


def test_an_empty_boot_spec_arms_nothing() -> None:
    engine = FaultEngine()
    engine.arm_from_spec("")
    engine.arm_from_spec(" , ")
    assert engine.armed() == []


def test_take_returns_nothing_when_no_fault_matches() -> None:
    engine = FaultEngine()
    engine.arm("member_not_found")
    assert engine.take("search", "GET") is None
    fault = engine.take("results", "POST")
    assert fault is not None
    assert fault.mode == "member_not_found"


# --- documentation stays honest ---------------------------------------------------------------


def test_every_implemented_fault_is_documented_in_the_fault_matrix() -> None:
    doc = (ROOT / "docs" / "fault-matrix.md").read_text()
    for mode in MODES:
        assert f"`{mode}`" in doc, f"{mode} is missing from docs/fault-matrix.md"
