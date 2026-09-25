"""Shared fixtures. `capability_dict` is a realistic, valid member-lookup capability."""

from __future__ import annotations

import copy
import socket
import threading
from collections.abc import Iterator
from typing import Any, cast

import pytest
from targets.mockbank.server import make_server
from tests.mockbank_support import FakeClock, MockHandle

from cua.surface import PlaywrightSurface, SurfaceConfig, SurfaceUnavailable


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def browser_surface() -> Iterator[PlaywrightSurface]:
    """One real headless Chromium per test module; each test resets it (see `surface`)."""
    surface = PlaywrightSurface(
        SurfaceConfig(headless=True, debug_port=_free_port()),
        secrets={"MOCK_USER": "teller01", "MOCK_PASS": "demo-only"},
    )
    try:
        surface.open()
    except SurfaceUnavailable as exc:
        pytest.skip(str(exc))
    yield surface
    surface.close()


@pytest.fixture
def surface(browser_surface: PlaywrightSurface, mock: MockHandle) -> PlaywrightSurface:
    browser_surface.reset()
    return browser_surface


@pytest.fixture
def mock() -> Iterator[MockHandle]:
    clock = FakeClock()
    server = make_server(port=0, clock=clock)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    host, port = cast(tuple[str, int], server.server_address[:2])
    yield MockHandle(f"http://{host}:{port}", host, port, server, clock)
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _bundle(
    *strategies: dict[str, Any], frame: str | None = "main", description: str = "element"
) -> dict[str, Any]:
    return {
        "description": description,
        "frame_path": [{"name": frame}] if frame else [],
        "strategies": list(strategies),
    }


@pytest.fixture
def capability_dict() -> dict[str, Any]:
    """A fresh deep copy per test, so tests can mutate it freely."""
    data: dict[str, Any] = {
        "schema_version": "1",
        "id": "member_lookup",
        "capability_version": "1.0.0",
        "title": "Look up a member and read their savings balance",
        "description": "Search for a member by number and read the current savings balance.",
        "status": "draft",
        "target": {
            "vendor": "MemberServ",
            "product": "MemberServ",
            "version_range": ">=3.1,<4",
            "tenant_profile": "default",
            "surface": "web",
        },
        "inputs": {
            "type": "object",
            "properties": {
                "member_id": {"type": "string", "pattern": "^[0-9]{5}$"},
            },
            "required": ["member_id"],
            "additionalProperties": False,
        },
        "outputs": {
            "type": "object",
            "properties": {"savings_balance": {"type": "string"}},
            "required": ["savings_balance"],
            "additionalProperties": False,
        },
        "steps": [
            {
                "id": "s1",
                "action": "navigate",
                "description": "Open the member search page",
                "url_template": "http://127.0.0.1:4310/members/search",
                "risk_class": "read",
            },
            {
                "id": "s2",
                "action": "fill",
                "description": "Type the member number",
                "locator": _bundle(
                    {
                        "kind": "label",
                        "text": "Member #",
                        "rationale": "Visible label survives markup changes",
                    },
                    {
                        "kind": "attribute_fingerprint",
                        "tag": "input",
                        "attributes": {"name": "mno"},
                        "rationale": "name attribute is stable even without a label",
                    },
                    description="member number field",
                ),
                "value": {"source": "param", "name": "member_id"},
                "risk_class": "read",
            },
            {
                "id": "s3",
                "action": "click",
                "description": "Press Search",
                "locator": _bundle(
                    {
                        "kind": "role_name",
                        "role": "button",
                        "name": "Search",
                        "rationale": "Accessible name of the button",
                    },
                    {
                        "kind": "text",
                        "text": "Search",
                        "rationale": "Fallback on visible text",
                    },
                    description="search button",
                ),
                "risk_class": "read",
                "expect": {"text_present": ["Search results"]},
            },
            {
                "id": "s4",
                "action": "extract",
                "description": "Read the savings balance",
                "locator": _bundle(
                    {
                        "kind": "ancestor_anchor",
                        "anchor_text": "Savings",
                        "container": "row",
                        "target_role": "cell",
                        "rationale": "Row labelled Savings holds the balance",
                    },
                    frame="accounts",
                    description="savings balance",
                ),
                "output": "savings_balance",
                "risk_class": "read",
            },
        ],
        "checkpoint": {"url_pattern": "/members/:id", "text_present": ["Savings"]},
        "error_map": [
            {
                "id": "no_such_member",
                "detect": {"text_present": ["No member found"]},
                "classification": "business_outcome",
                "outcome_code": "member_not_found",
                "message": "No member matches the supplied number",
            },
            {
                "id": "welcome_interstitial",
                "detect": {"text_present": ["Notice to all staff"]},
                "classification": "recoverable",
                "recovery": {
                    "kind": "dismiss",
                    "locator": _bundle(
                        {
                            "kind": "role_name",
                            "role": "button",
                            "name": "Continue",
                            "rationale": "Known interstitial has a Continue button",
                        }
                    ),
                    "max_attempts": 2,
                },
            },
            {
                "id": "app_crashed",
                "detect": {"text_present": ["Internal Server Error"]},
                "classification": "hard_failure",
                "code": "app_error",
            },
        ],
    }
    return copy.deepcopy(data)
