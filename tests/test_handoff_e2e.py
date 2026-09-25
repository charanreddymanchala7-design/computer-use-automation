"""The handoff for real: Chromium, the mock application, the web operator API, and a second client
attached to the same live browser acting as the person.

The application drops the session mid-run. The run stops and asks for a person; the "person" (a
separate process, as a second client must be) takes control over HTTP, signs in and searches again
in the same browser window, and hands back. The run re-checks the page and finishes, and the
server's own log shows exactly which requests happened."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from tests.discovery_scripts import discover as discover_capability
from tests.discovery_scripts import policy
from tests.mockbank_support import MockHandle

from cua.control import ControlLease
from cua.control.handoff import Handoff
from cua.control.operator import Broadcast
from cua.control.web import OperatorServer
from cua.evlog import EventLog, read_events
from cua.gateway import ActionGateway
from cua.redact import Redactor
from cua.replay import ReplayEngine
from cua.result import Status
from cua.surface import PlaywrightSurface

pytestmark = pytest.mark.browser

SIMULATOR = Path(__file__).resolve().parent.parent / "scripts" / "simulate_operator.py"
SECRETS = {"MOCK_USER": "teller01", "MOCK_PASS": "demo-only"}


def test_a_session_that_expires_mid_run_is_finished_by_a_person_in_the_same_browser(
    surface: PlaywrightSurface, mock: MockHandle, tmp_path: Path
) -> None:
    capability = discover_capability(surface, mock.base, tmp_path)
    mock.client().post("/_admin/reset", {})
    surface.reset()
    armed = mock.client().request(
        "POST", "/_admin/faults", json_body={"mode": "session_timeout", "step": "results"}
    )
    assert armed.status == 200

    log = EventLog(tmp_path / "run.jsonl", run_id="run_h", redactor=Redactor(secrets=["demo-only"]))
    lease = ControlLease()
    gateway = ActionGateway(surface, policy(), log, lease=lease)
    web = OperatorServer(lease, evidence_dir=tmp_path / "evidence")
    web.start()
    handoff = Handoff(lease, gateway, surface, log, Broadcast(web), claim_timeout_s=60)
    assert surface.debug_endpoint is not None
    person = subprocess.Popen(
        [
            sys.executable,
            str(SIMULATOR),
            "--operator",
            web.url,
            "--cdp",
            surface.debug_endpoint,
            "--user",
            SECRETS["MOCK_USER"],
            "--password",
            SECRETS["MOCK_PASS"],
            "--member",
            "12345",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        engine = ReplayEngine(
            surface,
            gateway,
            log,
            base_url=mock.base,
            secrets=SECRETS,
            evidence_dir=tmp_path / "evidence",
            run_id="run_h",
            handoff=handoff,
        )
        result = engine.run(capability, {"member_id": "12345"})
        out, err = person.communicate(timeout=60)
    finally:
        if person.poll() is None:
            person.kill()
        web.stop()

    assert person.returncode == 0, err
    assert result.status is Status.SUCCESS, (
        result.outcome_code,
        result.failed_step,
        result.observed,
    )
    assert result.outputs == {"savings_balance": "$2,480.15"}
    assert [(i.reason_code, i.outcome, i.taken_by) for i in result.interventions] == [
        ("session_expired", "handed_back", "simulated operator")
    ]

    truth = json.loads(mock.client().get("/_admin/log").text)
    searches = [r for r in truth["requests"] if r["method"] == "POST" and r["step"] == "results"]
    assert len(searches) == 2  # the run's search (which lost the session) and the person's own
    assert truth["effects"] == []  # a read-only capability changed nothing

    names = [e["event"] for e in read_events(log.path)]
    assert names.index("intervention_raised") < names.index("control_taken")
    assert names.index("control_taken") < names.index("control_returned")
    assert names.index("control_returned") < names.index("control_resumed")
    assert "demo-only" not in log.path.read_text() + result.model_dump_json() + out + err
    assert (tmp_path / "evidence" / "run_h" / "intervention-1.png").read_bytes()[:4] == b"\x89PNG"
