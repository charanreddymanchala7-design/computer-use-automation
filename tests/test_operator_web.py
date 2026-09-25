"""The operator web page and its JSON API: a stdlib server on loopback, over the same lease."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from tests.test_lease import request as make_request

from cua.control import ControlLease, Phase
from cua.control.web import OperatorServer


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[OperatorServer, ControlLease]]:
    evidence = tmp_path / "evidence"
    (evidence / "run_r").mkdir(parents=True)
    (evidence / "run_r" / "failure.png").write_bytes(b"\x89PNG-fake")
    (tmp_path / "secret.txt").write_text("not for the web")
    lease = ControlLease()
    lease.raise_intervention(lease.start(), make_request())
    server = OperatorServer(lease, evidence_dir=evidence)
    server.start()
    try:
        yield server, lease
    finally:
        server.stop()


def call(
    server: OperatorServer,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    token: str | None = None,
    host: str | None = None,
) -> tuple[int, dict[str, str], bytes]:
    headers: dict[str, str] = {}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token is not None:
        headers["X-CUA-Token"] = token
    if host is not None:
        headers["Host"] = host
    req = urllib.request.Request(
        server.url + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers=headers,
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read()


def as_json(raw: bytes) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(raw)
    return data


def test_it_only_listens_on_loopback(served: tuple[OperatorServer, ControlLease]) -> None:
    server, _ = served
    assert server.url.startswith("http://127.0.0.1:")


def test_the_state_endpoint_reports_the_open_request(
    served: tuple[OperatorServer, ControlLease],
) -> None:
    server, _ = served
    status, headers, raw = call(server, "/api/state")
    data = as_json(raw)
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert (data["phase"], data["request"]["id"]) == ("waiting_for_human", "ir_1")
    assert data["request"]["screenshot"] == "failure.png"


def test_the_whole_handoff_can_be_done_through_the_api(
    served: tuple[OperatorServer, ControlLease],
) -> None:
    server, lease = served
    status, _, raw = call(
        server,
        "/api/take-control",
        body={"request_id": "ir_1", "human": "ops@example.test"},
        token=server.token,
    )
    taken = as_json(raw)
    assert status == 200
    assert (taken["phase"], taken["request"]["taken_by"]) == (
        "human_in_control",
        "ops@example.test",
    )
    status, _, raw = call(
        server, "/api/hand-back", body={"epoch": taken["epoch"]}, token=server.token
    )
    assert status == 200
    assert lease.state.phase is Phase.HANDED_BACK


def test_abort_through_the_api(served: tuple[OperatorServer, ControlLease]) -> None:
    server, lease = served
    status, _, _ = call(server, "/api/abort", body={"request_id": "ir_1"}, token=server.token)
    assert status == 200
    assert lease.state.phase is Phase.ABORTED


def test_an_out_of_order_action_is_a_409_with_a_reason(
    served: tuple[OperatorServer, ControlLease],
) -> None:
    server, lease = served
    status, _, raw = call(server, "/api/hand-back", body={}, token=server.token)
    assert status == 409
    assert "cannot hand back" in as_json(raw)["error"]
    assert lease.state.phase is Phase.WAITING_FOR_HUMAN


def test_a_stale_epoch_is_a_409(served: tuple[OperatorServer, ControlLease]) -> None:
    server, _ = served
    call(server, "/api/take-control", body={"request_id": "ir_1", "human": "a"}, token=server.token)
    status, _, raw = call(server, "/api/hand-back", body={"epoch": 1}, token=server.token)
    assert status == 409
    assert "stale" in as_json(raw)["error"]


@pytest.mark.parametrize("token", [None, "", "wrong"])
def test_a_change_without_the_page_token_is_refused(
    served: tuple[OperatorServer, ControlLease], token: str | None
) -> None:
    server, lease = served
    status, _, _ = call(server, "/api/abort", body={"request_id": "ir_1"}, token=token)
    assert status == 403
    assert lease.state.phase is Phase.WAITING_FOR_HUMAN  # a page in another tab cannot do this


def test_a_request_addressed_to_another_host_name_is_refused(
    served: tuple[OperatorServer, ControlLease],
) -> None:
    server, _ = served
    assert call(server, "/api/state", host="evil.example")[0] == 403  # DNS rebinding
    assert call(server, "/api/state", host="localhost:1")[0] == 403  # wrong port


def test_a_body_that_is_not_json_is_a_400(served: tuple[OperatorServer, ControlLease]) -> None:
    server, _ = served
    req = urllib.request.Request(
        server.url + "/api/abort",
        data=b"not json",
        headers={"X-CUA-Token": server.token, "Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(req, timeout=5)
    assert caught.value.code == 400


def test_unknown_paths_are_404(served: tuple[OperatorServer, ControlLease]) -> None:
    server, _ = served
    assert call(server, "/nope")[0] == 404
    assert call(server, "/api/nope", body={}, token=server.token)[0] == 404


# --- the screenshot -------------------------------------------------------------------------------


def test_the_screenshot_is_served_from_the_evidence_folder(
    served: tuple[OperatorServer, ControlLease],
) -> None:
    server, _ = served
    status, headers, raw = call(server, "/evidence/run_r/failure.png")
    assert (status, headers["Content-Type"], raw) == (200, "image/png", b"\x89PNG-fake")


@pytest.mark.parametrize(
    "path",
    [
        "/evidence/../secret.txt",
        "/evidence/%2e%2e/secret.txt",
        "/evidence/run_r/../../secret.txt",
        "/evidence//etc/passwd",
        "/evidence/run_r/missing.png",
        "/evidence/run_r",
    ],
)
def test_nothing_outside_the_evidence_folder_can_be_read(
    served: tuple[OperatorServer, ControlLease], path: str
) -> None:
    server, _ = served
    assert call(server, path)[0] == 404


def test_only_images_are_served_from_the_evidence_folder(
    served: tuple[OperatorServer, ControlLease], tmp_path: Path
) -> None:
    server, _ = served
    (tmp_path / "evidence" / "run_r" / "run.jsonl").write_text("{}")
    assert call(server, "/evidence/run_r/run.jsonl")[0] == 404


# --- the page -------------------------------------------------------------------------------------


def test_the_page_is_accessible_and_carries_the_token_and_a_strict_policy(
    served: tuple[OperatorServer, ControlLease],
) -> None:
    server, _ = served
    status, headers, raw = call(server, "/")
    page = raw.decode()
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert '<html lang="en">' in page
    assert 'aria-live="polite"' in page
    assert "<main" in page
    assert f'content="{server.token}"' in page
    assert "innerHTML" not in page  # everything is set as text, so page content cannot inject
    policy = headers["Content-Security-Policy"]
    assert "default-src 'none'" in policy
    assert "script-src 'nonce-" in policy
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"


def test_the_buttons_are_real_buttons_with_visible_names(
    served: tuple[OperatorServer, ControlLease],
) -> None:
    server, _ = served
    page = call(server, "/")[2].decode()
    for label in ("Take control", "Hand back to the agent", "Abort the run"):
        assert f">{label}</button>" in page
    assert '<label for="name">' in page


def test_the_server_can_be_stopped_and_stopping_twice_is_harmless(
    served: tuple[OperatorServer, ControlLease],
) -> None:
    server, _ = served
    server.stop()
    server.stop()
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(server.url + "/api/state", timeout=2)
