"""Stand in for a person, for demos and tests.

A real operator opens the browser window the run is using, fixes the problem by hand, and clicks
"Hand back". This script does the same through the same two doors, so the handoff is exercised
for real rather than mocked:

* the operator API, to take control and to hand back (over HTTP, like the web page does);
* a *second client* attached to the very same live browser over its loopback debug port, to do the
  clicking (a separate process, exactly as a second client has to be).

It types the sign-in credentials it is given because a person would: the system never stores them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from typing import Any

from playwright.sync_api import Locator, Page, sync_playwright


def call(base: str, path: str, body: dict[str, Any] | None = None, token: str = "") -> Any:
    request = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-CUA-Token": token},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        raw = response.read()
    return raw if path == "/" else json.loads(raw)


def wait_until_asked(base: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            state = call(base, "/api/state")
        except OSError:  # the run has not opened its operator page yet
            time.sleep(0.1)
            continue
        if state["phase"] == "waiting_for_human":
            return state  # type: ignore[no-any-return]
        time.sleep(0.1)
    raise SystemExit("nobody asked for a person within the time allowed")


def in_any_frame(page: Page, selector: str, timeout_s: float = 10.0) -> Locator:
    """The first match in any frame of the page: legacy apps put their forms in frames."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for frame in page.frames:
            matches = frame.locator(selector)
            if matches.count():
                return matches.first
        page.wait_for_timeout(100)
    raise SystemExit(f"never found {selector!r} on the page")


def maybe_sign_in(page: Page, user: str, password: str) -> bool:
    if not any(f.locator("input[name=u]").count() for f in page.frames):
        return False
    in_any_frame(page, "input[name=u]").fill(user)
    in_any_frame(page, "input[name=p]").fill(password)
    in_any_frame(page, "input[type=image]").click()
    return True


def search(page: Page, member: str) -> None:
    in_any_frame(page, "input[name=F1]").fill(member)
    in_any_frame(page, "[alt=Go]").click()
    in_any_frame(page, f"tr:has-text('{member}')")  # the results are on screen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--operator", required=True, help="the operator page URL")
    parser.add_argument("--cdp", required=True, help="the browser's loopback debug endpoint")
    parser.add_argument("--name", default="simulated operator")
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--member", required=True, help="the member to search for after sign-in")
    parser.add_argument("--wait", type=float, default=60.0)
    args = parser.parse_args()

    asked = wait_until_asked(args.operator, args.wait)
    match = re.search(rb'name="cua-token" content="([^"]+)"', call(args.operator, "/"))
    if match is None:
        raise SystemExit("the operator page carries no token")
    token = match.group(1).decode()
    taken = call(
        args.operator,
        "/api/take-control",
        {"request_id": asked["request"]["id"], "human": args.name},
        token,
    )
    with sync_playwright() as pw:
        browser = pw.chromium.connect_over_cdp(args.cdp)
        page = browser.contexts[0].pages[0]
        signed_in = maybe_sign_in(page, args.user, args.password)
        search(page, args.member)
        browser.close()  # detaches this client; the run's browser stays open
    call(args.operator, "/api/hand-back", {"epoch": taken["epoch"]}, token)
    print(json.dumps({"request": asked["request"]["id"], "signed_in": signed_in}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
