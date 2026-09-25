"""A small operator page and JSON API, served on loopback by the standard library.

It is the web channel for the same ``OperatorService`` the terminal uses, so it can only move the
control lease. Three things keep a page open in another browser tab from steering it:

* it listens on 127.0.0.1 only, and a request addressed to any other Host name is refused (DNS
  rebinding);
* every state-changing call must carry a random per-server token that only this page receives;
* the page sets all content as text (never markup) under a nonce-based content policy.
"""

from __future__ import annotations

import hmac
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from cua.control.lease import ControlLease, LeaseError
from cua.control.operator import OperatorService
from cua.control.stuck import InterventionRequest

_MAX_BODY = 4096

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="cua-token" content="__TOKEN__">
<title>Operator - computer-use automation</title>
<style nonce="__NONCE__">
  :root { color-scheme: light; }
  body { margin: 0; font: 16px/1.5 system-ui, sans-serif; color: #1a1a1a; background: #ffffff; }
  main { max-width: 52rem; margin: 0 auto; padding: 1.5rem 1rem 3rem; }
  h1 { font-size: 1.25rem; margin: 0 0 1rem; }
  h2 { font-size: 1.15rem; margin: 1.5rem 0 .5rem; }
  .status { padding: .75rem 1rem; border: 2px solid #1a1a1a; border-radius: .25rem; }
  dl { display: grid; grid-template-columns: max-content 1fr; gap: .25rem 1rem; margin: 0; }
  dt { font-weight: 600; }
  dd { margin: 0; overflow-wrap: anywhere; }
  pre { background: #f3f3f3; border: 1px solid #767676; padding: .75rem; white-space: pre-wrap;
        overflow-wrap: anywhere; max-height: 14rem; overflow: auto; }
  img { max-width: 100%; border: 1px solid #767676; }
  label { display: block; font-weight: 600; margin-top: 1rem; }
  input { font: inherit; padding: .4rem .5rem; border: 2px solid #4d4d4d; border-radius: .25rem;
          width: 100%; max-width: 20rem; box-sizing: border-box; }
  .actions { display: flex; flex-wrap: wrap; gap: .75rem; margin-top: 1rem; }
  button { font: inherit; font-weight: 600; padding: .6rem 1rem; border-radius: .25rem;
           border: 2px solid #0b5cad; background: #0b5cad; color: #ffffff; cursor: pointer; }
  button.quiet { background: #ffffff; color: #0b5cad; }
  button.danger { border-color: #a12622; background: #ffffff; color: #a12622; }
  button:disabled { border-color: #767676; background: #ffffff; color: #595959;
                    cursor: not-allowed; }
  :focus-visible { outline: 3px solid #1a1a1a; outline-offset: 2px; }
  [hidden] { display: none; }
</style>
</head>
<body>
<main>
  <h1>Operator</h1>
  <p id="status" class="status" role="status" aria-live="polite">Loading...</p>
  <section id="request" hidden>
    <h2 id="headline"></h2>
    <dl>
      <dt>Capability</dt><dd id="capability"></dd>
      <dt>Goal</dt><dd id="goal"></dd>
      <dt>Step</dt><dd id="step"></dd>
      <dt>Why it stopped</dt><dd id="reason"></dd>
      <dt>Page</dt><dd id="url"></dd>
      <dt>Taken by</dt><dd id="taken"></dd>
    </dl>
    <h2>What the page showed</h2>
    <pre id="page"></pre>
    <img id="shot" alt="Screenshot of the page where the run stopped" hidden>
    <label for="name">Your name</label>
    <input id="name" autocomplete="name" value="operator">
    <div class="actions">
      <button id="take" type="button">Take control</button>
      <button id="back" type="button" class="quiet">Hand back to the agent</button>
      <button id="abort" type="button" class="danger">Abort the run</button>
    </div>
    <p>Take control, fix the problem in the browser window the agent opened, then hand back.
       The agent re-checks the page before it carries on.</p>
  </section>
</main>
<script nonce="__NONCE__">
const token = document.querySelector('meta[name="cua-token"]').content;
const $ = (id) => document.getElementById(id);
let state = null;
function say(text) { $('status').textContent = text; }
function render() {
  const req = state.request;
  $('request').hidden = !req;
  if (!req) {
    say('Nothing needs a person right now (' + state.phase.replaceAll('_', ' ') + ').');
    return;
  }
  say(state.phase === 'waiting_for_human' ? 'A person is needed.'
    : state.phase === 'human_in_control' ? 'You have control of the browser.'
    : 'Request ' + req.status.replaceAll('_', ' ') + '.');
  $('headline').textContent = req.headline;
  $('capability').textContent = req.capability_id;
  $('goal').textContent = req.goal;
  $('step').textContent = req.step_id;
  $('reason').textContent = req.reason;
  $('url').textContent = req.url;
  $('taken').textContent = req.taken_by || 'nobody yet';
  $('page').textContent = req.page_text;
  const shot = $('shot');
  shot.hidden = !req.screenshot;
  if (req.screenshot) shot.src = '/evidence/' + req.screenshot;
  $('take').disabled = state.phase !== 'waiting_for_human';
  $('back').disabled = state.phase !== 'human_in_control';
  $('abort').disabled = !['waiting_for_human', 'human_in_control'].includes(state.phase);
}
async function refresh() {
  try { state = await (await fetch('/api/state')).json(); render(); }
  catch (e) { say('Cannot reach the run. It may have finished.'); }
  setTimeout(refresh, 1000);
}
async function act(path, body) {
  const res = await fetch(path, { method: 'POST', body: JSON.stringify(body),
    headers: { 'Content-Type': 'application/json', 'X-CUA-Token': token } });
  const data = await res.json();
  if (res.ok) { state = data; render(); } else { say(data.error); }
}
$('take').addEventListener('click', () =>
  act('/api/take-control', { request_id: state.request.id, human: $('name').value }));
$('back').addEventListener('click', () => act('/api/hand-back', { epoch: state.epoch }));
$('abort').addEventListener('click', () =>
  act('/api/abort', { request_id: state.request.id, who: $('name').value }));
refresh();
</script>
</body>
</html>
"""


class OperatorServer:
    def __init__(
        self, lease: ControlLease, *, evidence_dir: Path | None = None, port: int = 0
    ) -> None:
        self._service = OperatorService(lease)
        self._evidence = evidence_dir.resolve() if evidence_dir is not None else None
        self.token = secrets.token_urlsafe(24)
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self._thread: threading.Thread | None = None
        self._stopped = False

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def notify(self, request: InterventionRequest) -> None:
        """The page polls the lease, so there is nothing to push."""

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.02},
            name="cua-operator-web",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # --- request handling ------------------------------------------------------------------------

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return None  # the run's own log is the audit trail

            def do_GET(self) -> None:
                server._get(self)

            def do_POST(self) -> None:
                server._post(self)

        return Handler

    def _host_ok(self, handler: BaseHTTPRequestHandler) -> bool:
        port = self._httpd.server_address[1]
        return handler.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _get(self, handler: BaseHTTPRequestHandler) -> None:
        if not self._host_ok(handler):
            return self._send_json(handler, 403, {"error": "unexpected Host header"})
        path = urlsplit(handler.path).path
        if path == "/":
            return self._send_page(handler)
        if path == "/api/state":
            return self._send_json(handler, 200, self._service.snapshot())
        if path.startswith("/evidence/"):
            return self._send_evidence(handler, unquote(path.removeprefix("/evidence/")))
        self._send_json(handler, 404, {"error": "not found"})

    def _post(self, handler: BaseHTTPRequestHandler) -> None:
        if not self._host_ok(handler):
            return self._send_json(handler, 403, {"error": "unexpected Host header"})
        supplied = handler.headers.get("X-CUA-Token") or ""
        if not hmac.compare_digest(supplied.encode(), self.token.encode()):
            return self._send_json(handler, 403, {"error": "missing or wrong page token"})
        body = self._read_body(handler)
        if body is None:
            return self._send_json(handler, 400, {"error": "the body must be a JSON object"})
        path = urlsplit(handler.path).path
        try:
            if path == "/api/take-control":
                done = self._service.take_control(
                    str(body.get("request_id", "")), human=str(body.get("human", ""))
                )
            elif path == "/api/hand-back":
                epoch = body.get("epoch")
                done = self._service.hand_back(epoch if isinstance(epoch, int) else None)
            elif path == "/api/abort":
                done = self._service.abort(
                    str(body.get("request_id", "")), who=str(body.get("who", ""))
                )
            else:
                return self._send_json(handler, 404, {"error": "not found"})
        except LeaseError as exc:
            return self._send_json(handler, 409, {"error": str(exc)})
        self._send_json(handler, 200, done)

    @staticmethod
    def _read_body(handler: BaseHTTPRequestHandler) -> dict[str, Any] | None:
        try:
            length = int(handler.headers.get("Content-Length") or 0)
            if not 0 <= length <= _MAX_BODY:
                return None
            data = json.loads(handler.rfile.read(length) or b"{}")
        except (ValueError, OSError):
            return None
        return data if isinstance(data, dict) else None

    # --- responses -------------------------------------------------------------------------------

    @staticmethod
    def _headers(handler: BaseHTTPRequestHandler, status: int, kind: str, length: int) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", kind)
        handler.send_header("Content-Length", str(length))
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("X-Content-Type-Options", "nosniff")

    def _send_json(self, handler: BaseHTTPRequestHandler, status: int, data: Any) -> None:
        raw = json.dumps(data).encode()
        self._headers(handler, status, "application/json; charset=utf-8", len(raw))
        handler.end_headers()
        handler.wfile.write(raw)

    def _send_page(self, handler: BaseHTTPRequestHandler) -> None:
        nonce = secrets.token_urlsafe(16)
        raw = _PAGE.replace("__TOKEN__", self.token).replace("__NONCE__", nonce).encode()
        self._headers(handler, 200, "text/html; charset=utf-8", len(raw))
        handler.send_header(
            "Content-Security-Policy",
            "default-src 'none'; img-src 'self'; connect-src 'self'; "
            f"script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
            "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
        )
        handler.end_headers()
        handler.wfile.write(raw)

    def _send_evidence(self, handler: BaseHTTPRequestHandler, relative: str) -> None:
        root = self._evidence
        target = (root / relative).resolve() if root is not None else None
        if (
            root is None
            or target is None
            or not target.is_relative_to(root)
            or target.suffix.lower() != ".png"
            or not target.is_file()
        ):
            return self._send_json(handler, 404, {"error": "not found"})
        raw = target.read_bytes()
        self._headers(handler, 200, "image/png", len(raw))
        handler.end_headers()
        handler.wfile.write(raw)
