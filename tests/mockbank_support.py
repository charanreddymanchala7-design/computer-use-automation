"""Test support for the MemberServ mock: a fake clock and a tiny cookie-keeping HTTP client."""

from __future__ import annotations

import http.client
import json
import re
from dataclasses import dataclass, field
from http.cookies import SimpleCookie
from typing import Any, NamedTuple
from urllib.parse import urlencode

from targets.mockbank.server import MockBankServer


class FakeClock:
    """Injected into the mock server so session expiry can be tested without sleeping."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Resp(NamedTuple):
    status: int
    headers: http.client.HTTPMessage
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("iso-8859-1")

    @property
    def location(self) -> str | None:
        return self.headers.get("Location")


@dataclass
class MockClient:
    """A tiny browser stand-in: keeps the session cookie, never follows redirects."""

    host: str
    port: int
    cookies: dict[str, str] = field(default_factory=dict)

    def request(
        self,
        method: str,
        path: str,
        data: dict[str, str] | None = None,
        json_body: Any = None,
    ) -> Resp:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
        headers = {}
        if self.cookies:
            headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in self.cookies.items())
        body = None
        if data is not None:
            body = urlencode(data)
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        if json_body is not None:
            body = json.dumps(json_body)
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=body, headers=headers)
        raw = conn.getresponse()
        payload = raw.read()
        for value in raw.headers.get_all("Set-Cookie") or []:
            jar: SimpleCookie = SimpleCookie(value)
            for name, morsel in jar.items():
                self.cookies[name] = morsel.value
        resp = Resp(raw.status, raw.headers, payload)
        conn.close()
        return resp

    def get(self, path: str) -> Resp:
        return self.request("GET", path)

    def post(self, path: str, data: dict[str, str]) -> Resp:
        return self.request("POST", path, data)

    def login(self, user: str = "teller01", password: str = "demo-only") -> Resp:
        page = self.get("/msv/login.cgi")
        return self.post("/msv/login.cgi", {"u": user, "p": password, "tok": token_of(page.text)})


def token_of(html: str) -> str:
    """The hidden one-time token of the (first) form on a page."""
    found = re.search(r"NAME=tok VALUE=(\w+)", html)
    assert found, "page has no one-time token"
    return found.group(1)


@dataclass
class MockHandle:
    base: str
    host: str
    port: int
    server: MockBankServer
    clock: FakeClock

    def client(self) -> MockClient:
        return MockClient(self.host, self.port)

    def logged_in_client(self) -> MockClient:
        client = self.client()
        assert client.login().status == 302
        return client
