"""The composition root: wire a surface, gateway, log and engine into one discovery or replay.

The CLI, the capability catalog and the tests all come through here, so a run is assembled the same
way wherever it starts: the surface is opened, every action goes through the gateway, all output is
redacted, and the browser is closed whatever happens.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cua.agent import DiscoveryLoop, Limits, RecordedRun
from cua.artifact import Capability
from cua.artifact.render import render_review
from cua.artifact.store import save_capability
from cua.artifact.synthesize import synthesize
from cua.control import ControlLease
from cua.control.handoff import Handoff, Operator
from cua.control.operator import Broadcast, TerminalOperator
from cua.control.web import OperatorServer
from cua.demo.memberserv import TASKS
from cua.evlog import EventLog, read_events, write_redacted_json
from cua.gateway import ActionGateway
from cua.llm import LLM
from cua.policy import Policy
from cua.redact import Redactor
from cua.replay import ReplayEngine
from cua.result import ReplayResult
from cua.surface import PlaywrightSurface, SurfaceConfig

OPERATORS = ("none", "terminal", "web", "both")


def load_policy(path: Path) -> Policy:
    return Policy.model_validate_json(path.read_text(encoding="utf-8"))


def _redactor(secrets: Mapping[str, str]) -> Redactor:
    return Redactor(secrets=secrets.values())


# --- replay ------------------------------------------------------------------------------------


@dataclass
class ReplayRun:
    result: ReplayResult
    evidence_dir: Path
    events: list[dict[str, Any]]


def _operators(
    kind: str,
    lease: ControlLease,
    evidence_root: Path,
    announce: Callable[[str], None],
    port: int,
) -> tuple[Operator | None, OperatorServer | None]:
    if kind == "none":
        return None, None
    channels: list[Operator] = []
    web: OperatorServer | None = None
    if kind in ("terminal", "both"):
        channels.append(TerminalOperator(lease))
    if kind in ("web", "both"):
        web = OperatorServer(lease, evidence_dir=evidence_root, port=port)
        web.start()
        channels.append(web)
        announce(f"operator page: {web.url}")
    return Broadcast(*channels), web


def replay_capability(
    capability: Capability,
    inputs: Mapping[str, Any],
    *,
    base_url: str,
    policy: Policy,
    evidence_dir: Path,
    secrets: Mapping[str, str],
    headed: bool = False,
    debug_port: int | None = None,
    operator: str = "none",
    operator_port: int = 0,
    claim_timeout_s: float = 900.0,
    announce: Callable[[str], None] = lambda _: None,
) -> ReplayRun:
    """Run a capability with no model in the loop. ``evidence_dir`` ends up holding the redacted
    log, the result, the capability used and, on trouble, a screenshot and page snapshot."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    run_id, evidence_root = evidence_dir.name, evidence_dir.parent
    log = EventLog(evidence_dir / "run.jsonl", run_id=run_id, redactor=_redactor(secrets))
    surface = PlaywrightSurface(
        SurfaceConfig(headless=not headed, debug_port=debug_port), secrets=secrets
    )
    web: OperatorServer | None = None
    try:
        surface.open()
        lease = ControlLease()

        def keep_png(name: str, png: bytes) -> None:
            (evidence_dir / f"{name}.png").write_bytes(png)

        gateway = ActionGateway(surface, policy, log, lease=lease if operator != "none" else None)
        channels, web = _operators(operator, lease, evidence_root, announce, operator_port)
        handoff = (
            Handoff(
                lease,
                gateway,
                surface,
                log,
                channels,
                claim_timeout_s=claim_timeout_s,
                keep_screenshot=keep_png,
            )
            if channels is not None
            else None
        )
        engine = ReplayEngine(
            surface,
            gateway,
            log,
            base_url=base_url,
            secrets=secrets,
            evidence_dir=evidence_root,
            run_id=run_id,
            handoff=handoff,
        )
        result = engine.run(capability, inputs)
    finally:
        if web is not None:
            web.stop()
        surface.close()
    write_redacted_json(evidence_dir / "result.json", result.model_dump(mode="json"), log.redactor)
    (evidence_dir / "capability.json").write_text(capability.canonical_json(), encoding="utf-8")
    return ReplayRun(result, evidence_dir, list(read_events(log.path)))


# --- discovery ---------------------------------------------------------------------------------


@dataclass
class DiscoveryRun:
    recorded: RecordedRun
    capability: Capability | None
    capability_path: Path | None
    evidence_dir: Path
    warnings: list[str]
    error: str | None = None


def discover_task(
    task_name: str,
    llm_for: Callable[[EventLog], LLM],
    *,
    base_url: str,
    policy: Policy,
    evidence_dir: Path,
    capabilities_dir: Path,
    secrets: Mapping[str, str],
    headed: bool = False,
    limits: Limits | None = None,
) -> DiscoveryRun:
    """One model-driven discovery, then the capability it becomes. The model is used here and
    nowhere in replay."""
    make_task, make_spec = TASKS[task_name]
    task, spec = make_task(base_url), make_spec()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    run_id = evidence_dir.name
    log = EventLog(evidence_dir / "run.jsonl", run_id=run_id, redactor=_redactor(secrets))

    def keep(name: str, png: bytes) -> None:
        (evidence_dir / f"{name}.png").write_bytes(png)

    surface = PlaywrightSurface(SurfaceConfig(headless=not headed), secrets=secrets)
    try:
        surface.open()
        gateway = ActionGateway(surface, policy, log)
        loop = DiscoveryLoop(
            surface, gateway, llm_for(log), log, limits=limits, run_id=run_id, on_screenshot=keep
        )
        recorded = loop.run(task)
    finally:
        surface.close()

    capability: Capability | None = None
    path: Path | None = None
    warnings: list[str] = []
    error: str | None = None
    try:
        synthesis = synthesize(recorded, task, spec, now=datetime.now(UTC))
    except ValueError as exc:  # SynthesisError: the run cannot become a trustworthy capability
        error = str(exc)
    else:
        capability, warnings = synthesis.capability, synthesis.warnings
        capabilities_dir.mkdir(parents=True, exist_ok=True)
        path = save_capability(capability, capabilities_dir, log.redactor)
        (evidence_dir / "capability.json").write_text(capability.canonical_json(), encoding="utf-8")
        (evidence_dir / "review.md").write_text(render_review(capability) + "\n", encoding="utf-8")
    summary = {
        "outcome": recorded.outcome,
        "reason": recorded.reason,
        "model": recorded.model,
        "steps_recorded": len(recorded.steps),
        "llm_steps": recorded.llm_steps,
        "tokens": recorded.tokens,
        "cost_usd": recorded.cost_usd,
        "duration_s": round(recorded.duration_s, 1),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "warnings": warnings,
        "error": error,
    }
    (evidence_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    return DiscoveryRun(recorded, capability, path, evidence_dir, warnings, error)
