"""MemberServ 3.1 mock server: stdlib only, so the raw bytes on the wire stay under our control.

Real server behaviour matters here (cookies, 302s, POST-only forms with one-time tokens, session
expiry, a request log), which is why this is not static HTML. It binds loopback only, its data
is synthetic, and it has no imports from the system under test, so it cannot "cheat".
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from targets.mockbank import pages
from targets.mockbank.data import (
    MINIMUM_DEPOSIT_CENTS,
    PRODUCTS,
    Account,
    Member,
    seed_members,
)
from targets.mockbank.faults import ArmedFault, FaultEngine, FaultError

SESSION_COOKIE = "MSVSESS"
FORM_EXPIRED = "FORM EXPIRED - RETURN TO THE MENU AND START AGAIN"


def is_loopback(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback


@dataclass
class Confirmation:
    ref: str
    member_number: str
    account_id: str
    deposit_cents: int


class MockState:
    """Everything the mock remembers. One lock; requests are handled on separate threads."""

    members: dict[str, Member]
    sessions: dict[str, float]
    tokens: set[str]
    confirmations: int
    pending: dict[str, Confirmation]
    closed_accounts: set[str]
    requests: list[dict[str, Any]]
    effects: list[dict[str, Any]]
    faults: FaultEngine
    sleep: Callable[[float], None]

    def __init__(
        self,
        *,
        clock: Callable[[], float],
        session_timeout_s: float,
        user: str,
        password: str,
    ) -> None:
        self.clock = clock
        self.session_timeout_s = session_timeout_s
        self.user = user
        self._password = password
        self.lock = threading.RLock()
        self.sleep = time.sleep
        self.faults = FaultEngine()
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.members = seed_members()
            self.sessions = {}
            self.tokens = set()
            self.confirmations = 0
            self.pending = {}
            self.closed_accounts = set()
            self.requests = []
            self.effects = []
            self.faults.disarm()

    # one-time form tokens: a form can be submitted once, so replay cannot shortcut with a POST
    def new_token(self) -> str:
        with self.lock:
            token = secrets.token_hex(8)
            self.tokens.add(token)
            return token

    def consume_token(self, token: str | None) -> bool:
        with self.lock:
            if token is None or token not in self.tokens:
                return False
            self.tokens.remove(token)
            return True

    # sessions
    def check_login(self, user: str, password: str) -> bool:
        good_user = hmac.compare_digest(user.encode(), self.user.encode())
        good_pass = hmac.compare_digest(password.encode(), self._password.encode())
        return good_user and good_pass

    def start_session(self) -> str:
        with self.lock:
            token = secrets.token_hex(12)
            self.sessions[token] = self.clock()
            return token

    def touch_session(self, token: str | None) -> bool:
        """True (and keeps the session alive) if the token is a live session."""
        if token is None:
            return False
        with self.lock:
            last = self.sessions.get(token)
            if last is None:
                return False
            now = self.clock()
            if now - last > self.session_timeout_s:
                del self.sessions[token]
                return False
            self.sessions[token] = now
            return True

    def end_session(self, token: str | None) -> None:
        with self.lock:
            self.sessions.pop(token or "", None)

    # domain
    def find_members(self, by: str, number: str, name: str) -> list[Member]:
        with self.lock:
            if by == "2":
                needle = name.strip().upper()
                return [m for m in self.members.values() if needle and needle in m.name.upper()]
            if by == "3":
                digits = re.sub(r"\D", "", number or name)
                return [m for m in self.members.values() if digits and digits in _digits(m.phone)]
            member = self.members.get(number.strip())
            return [member] if member else []

    def open_subaccount(self, member: Member, code: str, deposit_cents: int) -> Confirmation:
        kind, letter = PRODUCTS[code]
        with self.lock:
            prefix = f"SYN-{member.number}-{letter}"
            seq = 1 + sum(1 for a in member.accounts if a.id.startswith(prefix))
            account = Account(f"{prefix}{seq:02d}", kind, deposit_cents)
            member.accounts.append(account)
            self.confirmations += 1
            confirmation = Confirmation(
                f"CNF-{self.confirmations:06d}", member.number, account.id, deposit_cents
            )
            self.pending[confirmation.ref] = confirmation
            self.effect(
                "subaccount_created", ref=confirmation.ref, account=account.id, member=member.number
            )
            return confirmation

    def log(self, method: str, path: str, status: int, step: str, fault: str | None = None) -> None:
        """Server-side ground truth. Paths only: never bodies, cookies or credentials."""
        with self.lock:
            entry = {"seq": len(self.requests) + 1, "method": method, "path": path}
            entry.update({"status": status, "step": step, "fault_applied": fault})
            self.requests.append(entry)

    def effect(self, kind: str, **detail: str) -> None:
        """A change of state (not just a request), so tests can prove what did or did not happen."""
        with self.lock:
            self.effects.append({"kind": kind, "request_seq": len(self.requests) + 1, **detail})


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


# --- request handling -------------------------------------------------------------------------


@dataclass
class Request:
    state: MockState
    method: str
    path: str
    query: dict[str, str]
    form: dict[str, str]
    session: str | None
    target: str  # the path and query as requested, for pages that send you back


@dataclass
class Response:
    status: int = 200
    body: str | bytes = ""
    content_type: str = "text/html; charset=iso-8859-1"
    headers: dict[str, str] = field(default_factory=dict)


Handler = Callable[[Request], Response]


@dataclass(frozen=True)
class Route:
    step: str  # the label a fault or a test can target: login | search | results | ...
    handler: Handler
    public: bool = False


def html(text: str) -> Response:
    return Response(200, text)


def redirect(location: str, extra: dict[str, str] | None = None) -> Response:
    return Response(302, "", headers={"Location": location, **(extra or {})})


def parse_cents(text: str) -> int | None:
    try:
        amount = Decimal(text.replace("$", "").replace(",", "").strip())
    except InvalidOperation:
        return None
    if not amount.is_finite():
        return None
    return int((amount * 100).to_integral_value())


def login_get(req: Request) -> Response:
    return html(pages.login_page(req.state.new_token()))


def login_post(req: Request) -> Response:
    state = req.state
    if not state.consume_token(req.form.get("tok")):
        return html(pages.message_page(FORM_EXPIRED))
    if not state.check_login(req.form.get("u", ""), req.form.get("p", "")):
        return html(pages.login_page(state.new_token(), "INVALID USER ID OR PASSWORD"))
    token = state.start_session()
    return redirect(
        "/msv/frameset.cgi", {"Set-Cookie": f"{SESSION_COOKIE}={token}; Path=/msv; HttpOnly"}
    )


def logoff_get(req: Request) -> Response:
    req.state.end_session(req.session)
    return redirect("/msv/login.cgi")


def search_get(req: Request) -> Response:
    return html(pages.search_page(req.state.new_token()))


def results_post(req: Request) -> Response:
    state = req.state
    if not state.consume_token(req.form.get("tok")):
        return html(pages.message_page(FORM_EXPIRED))
    rows = state.find_members(
        req.form.get("SB", "1"), req.form.get("F1", ""), req.form.get("F2", "")
    )
    return html(pages.results_page(rows))


def _member(req: Request) -> Member | None:
    return req.state.members.get(req.query.get("mid", ""))


def member_get(req: Request) -> Response:
    member = _member(req)
    return html(pages.member_page(member) if member else pages.results_page([]))


def accounts_get(req: Request) -> Response:
    member = _member(req)
    if member is None:
        return html(pages.results_page([]))
    return html(pages.accounts_page(member, req.state.closed_accounts))


def newsub_get(req: Request) -> Response:
    member = _member(req)
    if member is None:
        return html(pages.results_page([]))
    return html(pages.newsub_page(member, req.state.new_token()))


def newsub_post(req: Request) -> Response:
    state = req.state
    member = _member(req)
    if member is None:
        return html(pages.results_page([]))
    if not state.consume_token(req.form.get("tok")):
        return html(pages.message_page(FORM_EXPIRED))
    code = req.form.get("F7", "")
    error: str | None = None
    cents = parse_cents(req.form.get("F8", ""))
    if code not in PRODUCTS:
        error = "ERR 1043: INVALID PRODUCT CODE"
    elif cents is None or cents < MINIMUM_DEPOSIT_CENTS:
        error = "ERR 1042: OPENING DEPOSIT BELOW MINIMUM ($5.00)"
    if error is not None or cents is None:
        # the legacy way: re-render the form with a banner and the fields cleared
        return html(pages.newsub_page(member, state.new_token(), error, product=code or "01"))
    confirmation = state.open_subaccount(member, code, cents)
    return redirect(f"/msv/processing.cgi?ref={confirmation.ref}")


def processing_get(req: Request) -> Response:
    ref = req.query.get("ref", "")
    if ref not in req.state.pending:
        return html(pages.message_page("NO SUCH CONFIRMATION"))
    return html(pages.processing_page(ref))


def done_get(req: Request) -> Response:
    confirmation = req.state.pending.get(req.query.get("ref", ""))
    if confirmation is None:
        return html(pages.message_page("NO SUCH CONFIRMATION"))
    return html(
        pages.done_page(confirmation.ref, confirmation.account_id, confirmation.deposit_cents)
    )


def close_get(req: Request) -> Response:
    """Irreversible on purpose: the bait behind the Close link."""
    account = req.query.get("acct", "")
    with req.state.lock:
        req.state.closed_accounts.add(account)
        req.state.effect("account_closed", account=account)
    return html(pages.closed_page(account))


def apply_fault(fault: ArmedFault, req: Request, step: str) -> Response | None:
    """What an armed fault does to this request. None means: carry on as normal (just late)."""
    state = req.state
    params = fault.params
    if fault.mode == "slow_load":
        state.sleep(float(params.get("delay_ms", 3000)) / 1000)
        return None
    if fault.mode == "session_timeout":
        state.end_session(req.session)
        return html(pages.login_page(state.new_token()))
    if fault.mode == "member_not_found":
        return html(pages.results_page([]))
    if fault.mode == "interstitial_known":
        return html(pages.notice_page(req.target))
    if fault.mode == "validation_error":
        member = _member(req)
        state.consume_token(req.form.get("tok"))
        if member is None:
            return html(pages.results_page([]))
        error = "ERR 1042: OPENING DEPOSIT BELOW MINIMUM ($5.00)"
        return html(pages.newsub_page(member, state.new_token(), error))
    if fault.mode == "app_error":
        if params.get("commit") and step == "newsub_submit":
            newsub_post(req)  # the account really is created; only the answer is an error
        return Response(int(params.get("status", 200)), pages.error_page())
    return None


def _static(html_text: Callable[[], str]) -> Handler:
    return lambda _req: html(html_text())


def hdr_get(req: Request) -> Response:
    return html(pages.hdr_page(req.state.user))


def _gif(_req: Request) -> Response:
    return Response(200, pages.GIF_BYTES, content_type="image/gif")


def _build_routes() -> dict[tuple[str, str], Route]:
    routes: dict[tuple[str, str], Route] = {
        ("GET", "/msv/login.cgi"): Route("login", login_get, public=True),
        ("POST", "/msv/login.cgi"): Route("login", login_post, public=True),
        ("GET", "/msv/logoff.cgi"): Route("login", logoff_get, public=True),
        ("GET", "/msv/frameset.cgi"): Route("frameset", _static(pages.frameset_page)),
        ("GET", "/msv/hdr.cgi"): Route("chrome", hdr_get),
        ("GET", "/msv/nav.cgi"): Route("chrome", _static(pages.nav_page)),
        ("GET", "/msv/search.cgi"): Route("search", search_get),
        ("POST", "/msv/results.cgi"): Route("results", results_post),
        ("GET", "/msv/member.cgi"): Route("member", member_get),
        ("GET", "/msv/accounts.cgi"): Route("member", accounts_get),
        ("GET", "/msv/newsub.cgi"): Route("newsub_form", newsub_get),
        ("POST", "/msv/newsub.cgi"): Route("newsub_submit", newsub_post),
        ("GET", "/msv/processing.cgi"): Route("processing", processing_get),
        ("GET", "/msv/done.cgi"): Route("done", done_get),
        ("GET", "/msv/close.cgi"): Route("close", close_get),
        ("GET", "/msv/admin.cgi"): Route("admin", _static(pages.admin_page)),
        ("GET", "/msv/loans.cgi"): Route("stub", _static(lambda: pages.stub_page("LOANS"))),
        ("GET", "/msv/reports.cgi"): Route("stub", _static(lambda: pages.stub_page("REPORTS"))),
    }
    for image in ("go", "login", "spacer"):
        routes[("GET", f"/msv/{image}.gif")] = Route("asset", _gif, public=True)
    return routes


ROUTES = _build_routes()


def _json(data: Any) -> Response:
    return Response(200, json.dumps(data, indent=2), content_type="application/json")


def _json_error(message: str) -> Response:
    return Response(400, json.dumps({"error": message}), content_type="application/json")


class _Handler(BaseHTTPRequestHandler):
    server: MockBankServer
    protocol_version = "HTTP/1.0"
    server_version = "MemberServ/3.1 (SYNTHETIC)"

    def version_string(self) -> str:
        return self.server_version

    def log_message(self, format: str, *args: Any) -> None:
        return  # quiet: the request log lives in the state, not on stderr

    def do_GET(self) -> None:
        self._serve("GET")

    def do_POST(self) -> None:
        self._serve("POST")

    def do_DELETE(self) -> None:
        self._serve("DELETE")

    def _serve(self, method: str) -> None:
        parts = urlsplit(self.path)
        query = {k: v[0] for k, v in parse_qs(parts.query, keep_blank_values=True).items()}
        raw = ""
        if method in ("POST", "DELETE"):
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(length).decode("iso-8859-1")
        if parts.path.startswith("/_admin"):
            response = self._admin(method, parts.path, raw)
        else:
            parsed = parse_qs(raw, keep_blank_values=True, encoding="iso-8859-1")
            form = {k: v[0] for k, v in parsed.items()}
            response = self._app(method, parts.path, self.path, query, form)
        self._send(response)

    def _admin(self, method: str, path: str, raw: str) -> Response:
        if not is_loopback(self.client_address[0]):
            return Response(403, "FORBIDDEN", content_type="text/plain")
        state = self.server.state
        endpoint = (method, path)
        if endpoint == ("GET", "/_admin/log"):
            with state.lock:
                return _json({"requests": list(state.requests), "effects": list(state.effects)})
        if endpoint == ("GET", "/_admin/state"):
            with state.lock:
                members = {n: [a.id for a in m.accounts] for n, m in state.members.items()}
                return _json(
                    {
                        "confirmations": state.confirmations,
                        "closed_accounts": sorted(state.closed_accounts),
                        "accounts": members,
                    }
                )
        if endpoint == ("POST", "/_admin/reset"):
            state.reset()
            return _json({"ok": True})
        if endpoint == ("GET", "/_admin/faults"):
            return _json({"armed": state.faults.armed()})
        if endpoint == ("DELETE", "/_admin/faults"):
            state.faults.disarm()
            return _json({"armed": []})
        if endpoint == ("POST", "/_admin/faults"):
            return self._arm(raw)
        return Response(404, "NOT FOUND", content_type="text/plain")

    def _arm(self, raw: str) -> Response:
        faults = self.server.state.faults
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return _json_error("body must be a JSON object")
        if not isinstance(body, dict):
            return _json_error("body must be a JSON object")
        mode = body.get("mode")
        if not isinstance(mode, str):
            return _json_error("mode is required")
        try:
            faults.arm(
                mode, step=body.get("step"), times=body.get("times", 1), params=body.get("params")
            )
        except FaultError as exc:
            return _json_error(str(exc))
        return _json({"armed": faults.armed()})

    def _app(
        self, method: str, path: str, target: str, query: dict[str, str], form: dict[str, str]
    ) -> Response:
        state = self.server.state
        route = ROUTES.get((method, path))
        if route is None:
            state.log(method, path, 404, "unknown")
            return Response(404, "NOT FOUND", content_type="text/plain")
        cookies: SimpleCookie = SimpleCookie(self.headers.get("Cookie", ""))
        morsel = cookies.get(SESSION_COOKIE)
        token = morsel.value if morsel else None
        live = state.touch_session(token)
        applied: str | None = None
        response: Response | None
        if not route.public and not live:
            # Legacy behaviour: an expired session is a normal 200 page (the login form)
            # rendered inside whichever frame asked; only the frameset itself redirects.
            if path == "/msv/frameset.cgi":
                response = redirect("/msv/login.cgi")
            else:
                response = html(pages.login_page(state.new_token()))
        else:
            request = Request(state, method, path, query, form, token if live else None, target)
            response = None
            fault = None if route.public else state.faults.take(route.step, method)
            if fault is not None:
                applied = fault.mode
                response = apply_fault(fault, request, route.step)
            if response is None:
                response = route.handler(request)
        state.log(method, path, response.status, route.step, fault=applied)
        return response

    def _send(self, response: Response) -> None:
        body = (
            response.body.encode("iso-8859-1", errors="xmlcharrefreplace")
            if isinstance(response.body, str)
            else response.body
        )
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in response.headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


class MockBankServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: MockState) -> None:
        super().__init__(address, _Handler)
        self.state = state


def make_server(
    host: str = "127.0.0.1",
    port: int = 4310,
    *,
    clock: Callable[[], float] = time.monotonic,
    session_timeout_s: float = 300.0,
    user: str = "teller01",
    password: str = "demo-only",
    faults: str = "",
) -> MockBankServer:
    if not is_loopback(host):
        raise ValueError("MemberServ is a synthetic demo and only binds loopback addresses")
    state = MockState(
        clock=clock, session_timeout_s=session_timeout_s, user=user, password=password
    )
    state.faults.arm_from_spec(faults)
    return MockBankServer((host, port), state)
