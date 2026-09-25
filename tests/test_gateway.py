"""The action gateway: the one choke point every action passes through, whoever issued it.

Fast tests against a fake surface. What matters here is the policy: what is allowed, what is
risky, what needs a human's yes, and that every decision leaves a redacted trace.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from cua.artifact import RiskClass
from cua.evlog import EventLog, read_events
from cua.gateway import ActionGateway, GatewayError, PolicyViolation
from cua.policy import Policy, UrlRule
from cua.redact import Redactor
from cua.surface import (
    Action,
    ActionResult,
    ElementInfo,
    Observation,
    RequestInfo,
    UnknownRefError,
)

BASE = "http://127.0.0.1:4310"


def element(
    ref: str = "e1",
    tag: str = "a",
    text: str = "",
    role: str | None = None,
    **attrs: str,
) -> ElementInfo:
    return ElementInfo(
        ref=ref,
        frame=0,
        tag=tag,
        role=role,
        text=text,
        attrs=attrs,
        label_hint=None,
        label_after=None,
        box=None,
        checked=None,
        disabled=False,
        options=[],
    )


class FakeSurface:
    """Records what reached it, so a test can prove a blocked action never did."""

    def __init__(self, *elements: ElementInfo) -> None:
        self.acts: list[Action] = []
        self.elements = {e.ref: e for e in elements}
        self.ok = True
        self.guard: Callable[[RequestInfo], bool] | None = None

    def observe(self) -> Observation:
        raise NotImplementedError

    def act(self, action: Action) -> ActionResult:
        self.acts.append(action)
        return ActionResult(self.ok, f"did {action.kind}", f"{BASE}/msv/x")

    def element_info(self, ref: str) -> ElementInfo:
        if ref not in self.elements:
            raise UnknownRefError(ref)
        return self.elements[ref]

    def wait_for_text(self, text: str, *, timeout_ms: int = 5000) -> bool:
        return True

    def set_request_guard(self, guard: Callable[[RequestInfo], bool] | None) -> None:
        self.guard = guard

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None


def mock_policy(**overrides: object) -> Policy:
    base: dict[str, object] = {
        "allow": (UrlRule(host="127.0.0.1", path_prefix="/msv/"),),
        "deny": (UrlRule(host="127.0.0.1", path_prefix="/msv/admin.cgi"),),
    }
    return Policy(**{**base, **overrides})


def build(
    tmp_path: Path, *elements: ElementInfo, policy: Policy | None = None
) -> tuple[ActionGateway, FakeSurface, EventLog]:
    surface = FakeSurface(*elements)
    log = EventLog(tmp_path / "run.jsonl", run_id="run_001", redactor=Redactor(secrets=["hunter2"]))
    return ActionGateway(surface, policy or mock_policy(), log), surface, log


# --- the URL allowlist -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        f"{BASE}/msv/login.cgi",
        f"{BASE}/msv/member.cgi?mid=12345",
        "http://127.0.0.1/msv/x",
        "HTTP://127.0.0.1:9/msv/x",
        "http://127.0.0.1./msv/x",
    ],
)
def test_navigation_inside_the_allowlist_is_allowed(url: str) -> None:
    assert mock_policy().url_decision(url).allowed


@pytest.mark.parametrize(
    ("url", "code"),
    [
        ("http://evil.example/msv/login.cgi", "host_not_allowed"),
        ("http://127.0.0.1.evil.example/msv/", "host_not_allowed"),
        ("http://127.0.0.1@evil.example/msv/", "userinfo_not_allowed"),
        ("http://user:pw@127.0.0.1/msv/", "userinfo_not_allowed"),
        (f"{BASE}/_admin/state", "path_not_allowed"),
        (f"{BASE}/msv/../_admin/state", "path_not_allowed"),
        (f"{BASE}/msv/%2e%2e/_admin/state", "path_not_allowed"),
        (f"{BASE}/msv/..%5c_admin/state", "path_not_allowed"),
        (f"{BASE}//_admin/state", "path_not_allowed"),
        (f"{BASE}/msv/admin.cgi", "denied_by_rule"),
        (f"{BASE}/msv/./admin.cgi?x=1", "denied_by_rule"),
        ("file:///etc/passwd", "scheme_not_allowed"),
        ("javascript:alert(1)", "scheme_not_allowed"),
        ("data:text/html,hi", "scheme_not_allowed"),
        ("ftp://127.0.0.1/msv/", "scheme_not_allowed"),
        ("http://127.0.0.1:99999/msv/", "invalid_url"),
        ("http://[::1/msv/", "invalid_url"),
        ("", "scheme_not_allowed"),
    ],
)
def test_navigation_outside_the_allowlist_is_denied_with_a_reason(url: str, code: str) -> None:
    decision = mock_policy().url_decision(url)
    assert not decision.allowed
    assert decision.code == code


def test_a_deny_rule_beats_an_allow_rule() -> None:
    policy = mock_policy()
    assert policy.url_decision(f"{BASE}/msv/search.cgi").allowed
    assert not policy.url_decision(f"{BASE}/msv/admin.cgi").allowed


def test_a_rule_can_pin_a_port_and_a_scheme() -> None:
    policy = Policy(allow=(UrlRule(host="bank.test", port=8443, schemes=("https",)),))
    assert policy.url_decision("https://bank.test:8443/x").allowed
    assert policy.url_decision("https://bank.test/x").code == "host_not_allowed"  # 443 != 8443
    assert policy.url_decision("http://bank.test:8443/x").code == "scheme_not_allowed"


def test_default_ports_count_when_a_rule_pins_one() -> None:
    policy = Policy(allow=(UrlRule(host="bank.test", port=443),))
    assert policy.url_decision("https://bank.test/x").allowed
    assert not policy.url_decision("http://bank.test/x").allowed


def test_ipv6_loopback_can_be_allowed() -> None:
    policy = Policy(allow=(UrlRule(host="::1"),))
    assert policy.url_decision("http://[::1]:4310/msv/").allowed


def test_a_path_prefix_matches_on_segment_boundaries() -> None:
    policy = Policy(allow=(UrlRule(host="127.0.0.1", path_prefix="/msv"),))
    assert policy.url_decision(f"{BASE}/msv").allowed
    assert policy.url_decision(f"{BASE}/msv/x").allowed
    assert not policy.url_decision(f"{BASE}/msvx/x").allowed


@pytest.mark.parametrize(
    "bad",
    [
        {"host": ""},
        {"host": "a/b"},
        {"host": "user@host"},
        {"host": "h", "port": 0},
        {"host": "h", "port": 70000},
        {"host": "h", "path_prefix": "no-slash"},
        {"host": "h", "schemes": ["gopher"]},
    ],
)
def test_a_malformed_rule_is_rejected_when_the_policy_is_loaded(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        UrlRule.model_validate(bad)


def test_a_policy_loads_from_json() -> None:
    policy = Policy.model_validate_json(
        '{"allow": [{"host": "127.0.0.1", "path_prefix": "/msv/"}],'
        ' "verbs": ["navigate", "click"], "irreversible": "block"}'
    )
    assert policy.irreversible == "block"
    assert policy.verbs == frozenset({"navigate", "click"})
    with pytest.raises(ValidationError):
        Policy.model_validate_json('{"allow": [], "surprise": 1}')


# --- verbs -------------------------------------------------------------------------------------


def test_a_verb_outside_the_allowlist_never_reaches_the_surface(tmp_path: Path) -> None:
    policy = mock_policy(verbs=frozenset({"click", "navigate"}))
    gateway, surface, log = build(tmp_path, policy=policy)
    outcome = gateway.act(Action.press("Enter"), step="s2")
    assert not outcome.executed
    assert outcome.decision.reason == "verb_not_allowed"
    assert surface.acts == []
    (event,) = read_events(log.path)
    assert (event["event"], event["outcome"], event["reason"]) == (
        "action_blocked",
        "blocked",
        "verb_not_allowed",
    )
    assert event["step"] == "s2"
    assert event["action"] == "press"


def test_every_default_verb_is_allowed_by_default(tmp_path: Path) -> None:
    gateway, surface, _ = build(tmp_path, element("e1", "input", name="F1"))
    for action in (
        Action.navigate(f"{BASE}/msv/search.cgi"),
        Action.fill("e1", text="12345"),
        Action.select("e1", "01"),
        Action.wait(10),
    ):
        assert gateway.act(action).executed
    assert len(surface.acts) == 4


# --- navigation through the gateway ------------------------------------------------------------


def test_navigating_off_the_allowlist_is_blocked_and_logged(tmp_path: Path) -> None:
    gateway, surface, log = build(tmp_path)
    outcome = gateway.act(Action.navigate("http://evil.example/steal"))
    assert not outcome.executed
    assert outcome.decision.reason == "url_not_allowed"
    assert surface.acts == []
    (event,) = read_events(log.path)
    assert event["event"] == "action_blocked"
    assert event["detail"] == "host_not_allowed"


def test_navigating_inside_the_allowlist_runs(tmp_path: Path) -> None:
    gateway, surface, log = build(tmp_path)
    outcome = gateway.act(Action.navigate(f"{BASE}/msv/search.cgi"))
    assert outcome.executed
    assert outcome.result is not None
    assert outcome.result.ok
    assert [a.kind for a in surface.acts] == ["navigate"]
    (event,) = read_events(log.path)
    assert (event["event"], event["outcome"], event["risk"]) == ("action", "ok", "read")


# --- risk classification -----------------------------------------------------------------------


def classify(policy: Policy, action: Action, info: ElementInfo | None = None) -> RiskClass:
    return policy.classify(action, info)


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        (
            element(text="Close", href="javascript:closeAcct('SYN-12345-S01')"),
            RiskClass.IRREVERSIBLE_WRITE,
        ),
        (
            element("e2", "input", value="Delete member", type="button"),
            RiskClass.IRREVERSIBLE_WRITE,
        ),
        (element("e3", "td", text="Void transaction"), RiskClass.IRREVERSIBLE_WRITE),
        (
            element("e4", "img", alt="Freeze account", onclick="freeze()"),
            RiskClass.IRREVERSIBLE_WRITE,
        ),
        (element("e5", "input", value="Submit", type="button"), RiskClass.REVERSIBLE_WRITE),
        (element("e6", "a", text="Save changes"), RiskClass.REVERSIBLE_WRITE),
        (element("e7", "img", alt="Go", onclick="doSearch()"), RiskClass.READ),
        (element("e8", "tr", text="12345 TESTERSON, ADA ACTIVE"), RiskClass.READ),
        (element("e9", "td", text="Open Sub-Account..."), RiskClass.READ),
        (element("e10", "a", text="Disclosed fees"), RiskClass.READ),  # 'close' inside a word
    ],
)
def test_a_click_is_classified_from_what_the_element_says(
    info: ElementInfo, expected: RiskClass
) -> None:
    assert classify(mock_policy(), Action.click(info.ref), info) is expected


def test_typing_selecting_navigating_and_waiting_are_reads() -> None:
    policy = mock_policy()
    field = element("e1", "input", name="F1")
    assert classify(policy, Action.fill("e1", text="Close"), field) is RiskClass.READ
    assert classify(policy, Action.select("e1", "01"), field) is RiskClass.READ
    assert classify(policy, Action.navigate(f"{BASE}/msv/x")) is RiskClass.READ
    assert classify(policy, Action.wait(5)) is RiskClass.READ


def test_pressing_enter_can_submit_a_form_so_it_counts_as_a_write() -> None:
    policy = mock_policy()
    assert classify(policy, Action.press("Enter")) is RiskClass.REVERSIBLE_WRITE
    assert classify(policy, Action.press("Tab")) is RiskClass.READ


def test_a_click_by_coordinates_cannot_be_vouched_for_so_it_counts_as_a_write() -> None:
    assert classify(mock_policy(), Action.click_at(10, 10)) is RiskClass.REVERSIBLE_WRITE


def test_the_terms_are_configurable() -> None:
    policy = mock_policy(irreversible_terms=("launch",), write_terms=())
    assert classify(policy, Action.click("e1"), element(text="Launch missile")) is (
        RiskClass.IRREVERSIBLE_WRITE
    )
    assert classify(policy, Action.click("e1"), element(text="Close")) is RiskClass.READ


def test_a_declared_risk_can_raise_the_classification_but_never_lower_it(tmp_path: Path) -> None:
    close = element("e1", text="Close", href="javascript:closeAcct('x')")
    gateway, surface, _ = build(tmp_path, close)
    outcome = gateway.act(Action.click("e1"), declared_risk=RiskClass.READ)
    assert outcome.decision.risk is RiskClass.IRREVERSIBLE_WRITE  # the page's own words win
    assert not outcome.executed
    assert surface.acts == []

    plain = element("e2", "img", alt="Go")
    gateway2, _, _ = build(tmp_path / "b", plain)
    raised = gateway2.act(Action.click("e2"), declared_risk=RiskClass.IRREVERSIBLE_WRITE)
    assert raised.decision.risk is RiskClass.IRREVERSIBLE_WRITE  # a recording knew better


# --- irreversible actions and the confirmation gate --------------------------------------------


CLOSE = element("e1", text="Close", href="javascript:closeAcct('SYN-12345-S01')")


def test_an_irreversible_action_is_held_until_someone_confirms_it(tmp_path: Path) -> None:
    gateway, surface, log = build(tmp_path, CLOSE)
    held = gateway.act(Action.click("e1"))
    assert not held.executed
    assert held.decision.reason == "confirmation_required"
    assert held.decision.confirmation_id is not None
    assert surface.acts == []

    gateway.confirm(held.decision.confirmation_id, approver="human:supervisor")
    released = gateway.act(Action.click("e1"))
    assert released.executed
    assert released.decision.reason == "confirmed"
    assert len(surface.acts) == 1

    events = read_events(log.path)
    assert [e["event"] for e in events] == ["action_blocked", "confirmation_granted", "action"]
    assert events[1]["approver"] == "human:supervisor"
    assert events[1]["confirmation_id"] == held.decision.confirmation_id


def test_a_confirmation_is_single_use(tmp_path: Path) -> None:
    gateway, surface, _ = build(tmp_path, CLOSE)
    first = gateway.act(Action.click("e1"))
    assert first.decision.confirmation_id
    gateway.confirm(first.decision.confirmation_id, approver="human")
    assert gateway.act(Action.click("e1")).executed
    again = gateway.act(Action.click("e1"))
    assert not again.executed
    assert again.decision.reason == "confirmation_required"
    assert again.decision.confirmation_id != first.decision.confirmation_id
    assert len(surface.acts) == 1


def test_a_confirmation_only_unlocks_the_action_it_was_issued_for(tmp_path: Path) -> None:
    other = element("e2", text="Delete", href="javascript:del()")
    gateway, surface, _ = build(tmp_path, CLOSE, other)
    held = gateway.act(Action.click("e1"))
    assert held.decision.confirmation_id
    gateway.confirm(held.decision.confirmation_id, approver="human")
    elsewhere = gateway.act(Action.click("e2"))
    assert not elsewhere.executed
    assert surface.acts == []


def test_an_unconfirmed_id_or_an_unknown_id_cannot_be_used(tmp_path: Path) -> None:
    gateway, surface, _ = build(tmp_path, CLOSE)
    held = gateway.act(Action.click("e1"))
    assert held.decision.confirmation_id
    unlock_attempt = gateway.act(Action.click("e1"), confirmation_id=held.decision.confirmation_id)
    assert not unlock_attempt.executed  # naming an id is not the same as a human confirming it
    with pytest.raises(GatewayError, match="no pending confirmation"):
        gateway.confirm("cf_nope", approver="human")
    assert surface.acts == []


def test_block_mode_never_releases_an_irreversible_action(tmp_path: Path) -> None:
    gateway, surface, log = build(tmp_path, CLOSE, policy=mock_policy(irreversible="block"))
    outcome = gateway.act(Action.click("e1"))
    assert outcome.decision.reason == "irreversible_blocked"
    assert outcome.decision.confirmation_id is None
    assert surface.acts == []
    assert read_events(log.path)[0]["risk"] == "irreversible_write"


def test_a_reversible_write_runs_and_is_flagged_in_the_log(tmp_path: Path) -> None:
    submit = element("e1", "input", value="Submit", type="button")
    gateway, surface, log = build(tmp_path, submit)
    assert gateway.act(Action.click("e1")).executed
    assert len(surface.acts) == 1
    assert read_events(log.path)[0]["risk"] == "reversible_write"


def test_a_click_on_a_ref_the_surface_does_not_know_is_a_gateway_error(tmp_path: Path) -> None:
    gateway, surface, _ = build(tmp_path)
    with pytest.raises(UnknownRefError):
        gateway.act(Action.click("e404"))
    assert surface.acts == []


# --- results and violations --------------------------------------------------------------------


def test_a_failed_action_is_still_a_decision_that_was_executed(tmp_path: Path) -> None:
    gateway, surface, log = build(tmp_path)
    surface.ok = False
    outcome = gateway.act(Action.navigate(f"{BASE}/msv/x"))
    assert outcome.executed
    assert outcome.result is not None
    assert not outcome.result.ok
    assert read_events(log.path)[0]["outcome"] == "failed"


def test_require_returns_the_result_or_raises_the_decision(tmp_path: Path) -> None:
    gateway, _, _ = build(tmp_path)
    assert gateway.act(Action.navigate(f"{BASE}/msv/x")).require().ok
    blocked = gateway.act(Action.navigate("http://evil.example/"))
    with pytest.raises(PolicyViolation) as info:
        blocked.require()
    assert info.value.decision.reason == "url_not_allowed"


# --- logging discipline ------------------------------------------------------------------------


def test_a_secret_fill_logs_the_secret_name_and_never_a_value(tmp_path: Path) -> None:
    field = element("e1", "input", name="p", type="password")
    gateway, _, log = build(tmp_path, field)
    gateway.act(Action.fill("e1", secret="MOCK_PASS"))
    text = log.path.read_text()
    (event,) = read_events(log.path)
    assert event["secret_name"] == "MOCK_PASS"
    assert "typed" not in event
    assert "hunter2" not in text


def test_typed_text_is_redacted_by_shape_and_by_registered_secret(tmp_path: Path) -> None:
    field = element("e1", "input", name="F1")
    gateway, _, log = build(tmp_path, field)
    gateway.act(Action.fill("e1", text="ssn 123-45-6789 and hunter2"))
    text = log.path.read_text()
    assert "123-45-6789" not in text
    assert "hunter2" not in text


def test_every_decision_is_logged_with_step_action_target_and_reason(tmp_path: Path) -> None:
    gateway, _, log = build(tmp_path, CLOSE)
    gateway.act(Action.click("e1"), step="s7")
    (event,) = read_events(log.path)
    assert event["step"] == "s7"
    assert event["action"] == "click"
    assert "Close" in event["target"]
    assert event["reason"] == "confirmation_required"


# --- the network-level guard -------------------------------------------------------------------


def test_the_gateway_installs_a_request_guard_on_the_surface(tmp_path: Path) -> None:
    _, surface, _ = build(tmp_path)
    assert surface.guard is not None


def test_the_guard_allows_and_blocks_requests_and_logs_the_blocks(tmp_path: Path) -> None:
    _, surface, log = build(tmp_path)
    assert surface.guard is not None
    ok = RequestInfo(f"{BASE}/msv/go.gif", "GET", "image", False)
    bad = RequestInfo(f"{BASE}/msv/admin.cgi?u=x", "GET", "document", True)
    assert surface.guard(ok) is True
    assert surface.guard(bad) is False
    (event,) = read_events(log.path)
    assert event["event"] == "url_blocked"
    assert event["detail"] == "denied_by_rule"
    assert event["method"] == "GET"
    assert event["resource_type"] == "document"
    assert event["navigation"] is True
    assert "/msv/admin.cgi" in event["url"]


def test_the_guard_can_be_left_off(tmp_path: Path) -> None:
    surface = FakeSurface()
    log = EventLog(tmp_path / "run.jsonl", run_id="r", redactor=Redactor())
    ActionGateway(surface, mock_policy(), log, guard_requests=False)
    assert surface.guard is None
